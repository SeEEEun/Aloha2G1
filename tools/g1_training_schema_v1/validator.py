from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .causal_alignment import action_chunk_indices, validate_episode_timestamps
from .constants import (
    ACTION_DIM,
    ACTION_KEY,
    CANONICAL_JOINT_NAMES,
    CHUNK_SIZE,
    FPS,
    IMAGE_KEY,
    JOINT_SPECS,
    STATE_DIM,
    STATE_KEY,
    TASK_TEXT,
)
from .source_audit import SourceDatasetIndex, load_json, sha256_file


def validate_joint_order(names: list[str] | tuple[str, ...]) -> None:
    if len(names) != STATE_DIM or len(set(names)) != STATE_DIM:
        raise ValueError(f"joint order must contain {STATE_DIM} unique names")
    if tuple(names) != CANONICAL_JOINT_NAMES:
        raise ValueError("joint order is not the canonical G1/Dex3 policy order")


def validate_feature_schema(features: dict[str, Any]) -> None:
    required = {
        IMAGE_KEY,
        STATE_KEY,
        ACTION_KEY,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    if set(features) != required:
        raise ValueError(f"feature key mismatch: expected={sorted(required)}, actual={sorted(features)}")
    if features[STATE_KEY]["shape"] != [STATE_DIM] or features[ACTION_KEY]["shape"] != [ACTION_DIM]:
        raise ValueError("G1 state/action dimensions do not match the 28D contract")
    validate_joint_order(features[STATE_KEY]["names"])
    validate_joint_order(features[ACTION_KEY]["names"])
    if features[STATE_KEY]["names"] != features[ACTION_KEY]["names"]:
        raise ValueError("state and action joint order mismatch")
    prohibited = {"method", "retargeting_method", "dataset_method", "method_id"}
    if prohibited & set(features):
        raise ValueError("method identity is present in policy features")


def validate_ab_schema_equality(info_a: dict[str, Any], info_b: dict[str, Any]) -> None:
    validate_feature_schema(info_a["features"])
    validate_feature_schema(info_b["features"])
    keys = ("codebase_version", "robot_type", "fps", "data_path", "video_path", "features")
    mismatches = [key for key in keys if info_a.get(key) != info_b.get(key)]
    if mismatches:
        raise ValueError(f"A/B common schema mismatch: {mismatches}")


def validate_target_episode(
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    source_frame_count: int,
) -> None:
    state = np.asarray(state)
    action = np.asarray(action)
    if state.shape != (source_frame_count, STATE_DIM):
        raise ValueError(f"state shape {state.shape} != ({source_frame_count},{STATE_DIM})")
    if action.shape != (source_frame_count, ACTION_DIM):
        raise ValueError(f"action shape {action.shape} != ({source_frame_count},{ACTION_DIM})")
    if state.dtype != np.float32 or action.dtype != np.float32:
        raise ValueError("state/action must be float32")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state/action contain non-finite values")
    lower = np.asarray([joint.minimum for joint in JOINT_SPECS], dtype=np.float32)
    upper = np.asarray([joint.maximum for joint in JOINT_SPECS], dtype=np.float32)
    if np.any(state < lower - 1e-5) or np.any(state > upper + 1e-5):
        raise ValueError("target state/action violates the verified controlled-joint limits")
    if not np.array_equal(state, action):
        raise ValueError("v1 retargeted_target_state and same-row absolute target action must match")
    validate_episode_timestamps(timestamps, source_frame_count)


def validate_source_not_copied_as_target(source_state: np.ndarray, target_state: np.ndarray) -> None:
    if np.asarray(source_state).shape[-1] != 14:
        raise ValueError("the authoritative source state is expected to be 14D ALOHA")
    if np.asarray(target_state).shape[-1] != STATE_DIM:
        raise ValueError("target state must be 28D G1")
    if np.asarray(source_state).shape == np.asarray(target_state).shape and np.array_equal(source_state, target_state):
        raise ValueError("ALOHA observation.state was copied as target G1 state")


def validate_separate_output_roots(output_a: str | Path, output_b: str | Path) -> None:
    if Path(output_a).resolve() == Path(output_b).resolve():
        raise ValueError("Dataset A and Dataset B output roots must be separate")


def _read_all_data(root: Path) -> dict[str, np.ndarray]:
    paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError("packaged dataset has no data parquet")
    tables = [pq.read_table(path) for path in paths]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    return {
        STATE_KEY: np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32),
        ACTION_KEY: np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32),
        **{key: np.asarray(table[key].to_numpy()) for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]},
    }


def validate_packaged_dataset(root: str | Path, source_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    info = load_json(root / "meta/info.json")
    validate_feature_schema(info["features"])
    if info["codebase_version"] != "v3.0" or float(info["fps"]) != FPS:
        raise ValueError("packaged LeRobot version/fps mismatch")
    arrays = _read_all_data(root)
    if len(arrays[STATE_KEY]) != int(info["total_frames"]):
        raise ValueError("packaged total_frames mismatch")
    if not np.array_equal(arrays["index"], np.arange(len(arrays["index"]))):
        raise ValueError("global index is not contiguous")
    if not np.isfinite(arrays[STATE_KEY]).all() or not np.isfinite(arrays[ACTION_KEY]).all():
        raise ValueError("packaged state/action contains non-finite values")
    if not np.array_equal(arrays[STATE_KEY], arrays[ACTION_KEY]):
        raise ValueError("packaged same-row state/action contract violated")
    episode_paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    episode_rows: list[dict[str, Any]] = []
    for path in episode_paths:
        episode_rows.extend(pq.read_table(path).to_pylist())
    if len(episode_rows) != int(info["total_episodes"]):
        raise ValueError("packaged total_episodes mismatch")
    if sorted(int(row["episode_index"]) for row in episode_rows) != list(range(len(episode_rows))):
        raise ValueError("packaged episode_index is not contiguous")
    for row in sorted(episode_rows, key=lambda item: int(item["episode_index"])):
        ep = int(row["episode_index"])
        mask = arrays["episode_index"] == ep
        length = int(row["length"])
        if int(mask.sum()) != length:
            raise ValueError(f"episode {ep} frame count mismatch")
        if not np.array_equal(arrays["frame_index"][mask], np.arange(length)):
            raise ValueError(f"episode {ep} frame_index mismatch")
        validate_episode_timestamps(arrays["timestamp"][mask], length)
        indices, padding = action_chunk_indices(length - 1, length, CHUNK_SIZE)
        if indices[0] != length - 1 or not padding[1:].all() or padding[0]:
            raise ValueError(f"episode {ep} last-frame action chunk convention mismatch")
        if row["tasks"] != [TASK_TEXT]:
            raise ValueError(f"episode {ep} task text changed")
        video_rel = info["video_path"].format(
            video_key=IMAGE_KEY,
            chunk_index=int(row[f"videos/{IMAGE_KEY}/chunk_index"]),
            file_index=int(row[f"videos/{IMAGE_KEY}/file_index"]),
        )
        if not (root / video_rel).is_file():
            raise FileNotFoundError(f"packaged RGB reference missing: {video_rel}")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    if len(tasks) != 1 or tasks[0].get("__index_level_0__") != TASK_TEXT:
        raise ValueError("task/language metadata changed")
    if source_root is not None:
        source = SourceDatasetIndex(source_root)
        manifest = load_json(root / "meta/g1_packaging_manifest.json")
        for asset in manifest["video_assets"]:
            src = source.root / asset["relative_path"]
            dst = root / asset["relative_path"]
            if sha256_file(src) != sha256_file(dst) or asset["sha256"] != sha256_file(dst):
                raise ValueError(f"RGB asset hash mismatch: {asset['relative_path']}")
    return {
        "status": "PASS",
        "root": str(root),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "state_dimension": STATE_DIM,
        "action_dimension": ACTION_DIM,
        "rgb_key": IMAGE_KEY,
        "task_preserved": True,
        "episode_boundaries_valid": True,
        "last_frame_masking_valid": True,
    }


def deterministic_tree_hash(root: str | Path) -> str:
    import hashlib

    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()
