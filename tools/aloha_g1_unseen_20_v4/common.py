"""Frozen dependency and unseen LeRobot source contracts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from aloha_g1_dataset_v1.core import SourceEpisode, fixed_list_numpy, scalar_numpy

from .constants import (
    CAMERA_KEY,
    EXPECTED_FROZEN_SHA256,
    FROZEN_PATHS,
    ORIGINAL_DATASET_ROOT,
    OUTPUT_ROOT,
    RAW_RECORDING_NAMES,
    SOURCE_DATASET_ROOT,
    TASK,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def combined_sha256(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def freeze_dependencies(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    destination = output_root / "dependencies"
    destination.mkdir(parents=True, exist_ok=True)
    names = {
        "common_arm_v2": "frozen_common_arm_v2_config.json",
        "feasibility_v3": "frozen_feasibility_v3_config.json",
        "proposed_hand_v2_1": "frozen_proposed_hand_v2_1.json",
        "collision_v4": "frozen_global_collision_v4_config.json",
    }
    entries: dict[str, Any] = {}
    for key, source in FROZEN_PATHS.items():
        if not source.is_file():
            raise RuntimeError(f"FROZEN_V4_DEPENDENCY_MISMATCH: missing {source}")
        actual = sha256_file(source)
        expected = EXPECTED_FROZEN_SHA256[key]
        if actual != expected:
            raise RuntimeError(
                f"FROZEN_V4_DEPENDENCY_MISMATCH: {key} expected {expected}, got {actual}"
            )
        target = destination / names[key]
        if target.exists() and sha256_file(target) != expected:
            raise RuntimeError(f"FROZEN_V4_DEPENDENCY_MISMATCH: existing copy {target}")
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".incomplete")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        entries[key] = {
            "authoritative_path": str(source.resolve()),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "copy_path": str(target.resolve()),
            "copy_sha256": sha256_file(target),
            "verified": actual == expected == sha256_file(target),
        }
    arm = load_json(FROZEN_PATHS["common_arm_v2"])
    feasibility = load_json(FROZEN_PATHS["feasibility_v3"])
    hand = load_json(FROZEN_PATHS["proposed_hand_v2_1"])
    collision = load_json(FROZEN_PATHS["collision_v4"])
    semantic_checks = {
        "arm_immutable": arm.get("status") == "IMMUTABLE_AFTER_CALIBRATION_SELECTION",
        "feasibility_immutable": feasibility.get("status")
        == "IMMUTABLE_AFTER_CALIBRATION_SELECTION",
        "collision_immutable": collision.get("status")
        == "IMMUTABLE_AFTER_40_EPISODE_CALIBRATION",
        "hand_schema": hand.get("schema_version") == "proposed_hand_v2_1_candidate",
        "hand_method_scope": hand.get("method_scope") == "Proposed/Dataset B only",
        "one_shared_feasibility_solver": feasibility["solver_parameters"].get(
            "method_specific_parameters_allowed"
        )
        is False,
        "one_shared_collision_solver": collision.get("method_specific_parameters") is False,
        "global_mapping_change_forbidden": collision.get("global_mapping_changed") is False,
        "hand_change_forbidden": collision.get("hand_v2_1_changed") is False,
        "acceptance_gate_change_forbidden": collision.get("acceptance_gates_changed") is False,
    }
    if not all(semantic_checks.values()):
        raise RuntimeError(f"FROZEN_V4_DEPENDENCY_MISMATCH: {semantic_checks}")
    manifest = {
        "schema_version": "frozen_v4_unseen_20_dependency_manifest_v1",
        "status": "ALL_FROZEN_V4_HASHES_VERIFIED",
        "verification_timing": "before any unseen retargeting",
        "dependencies": entries,
        "combined_dependency_sha256": combined_sha256(
            {key: row["actual_sha256"] for key, row in entries.items()}
        ),
        "semantic_checks": semantic_checks,
        "new_20_used_for_parameter_selection": False,
        "candidate_search_executed": False,
        "calibration_executed": False,
        "retuning_allowed": False,
    }
    atomic_json(destination / "dependency_checksums.json", manifest)
    return {
        "manifest": manifest,
        "arm": arm,
        "feasibility": feasibility,
        "hand": hand,
        "collision": collision,
    }


def verify_dependencies_unchanged(manifest: dict[str, Any]) -> dict[str, Any]:
    after = {key: sha256_file(path) for key, path in FROZEN_PATHS.items()}
    checks = {
        key: after[key] == EXPECTED_FROZEN_SHA256[key]
        for key in EXPECTED_FROZEN_SHA256
    }
    copy_checks = {
        key: sha256_file(Path(row["copy_path"])) == EXPECTED_FROZEN_SHA256[key]
        for key, row in manifest["dependencies"].items()
    }
    return {
        "authoritative_after_sha256": after,
        "authoritative_unchanged": checks,
        "immutable_copies_unchanged": copy_checks,
        "all_unchanged": all(checks.values()) and all(copy_checks.values()),
    }


class UnseenSourceDataset:
    """Read-only SourceDataset-compatible view of the separately built 20 set."""

    def __init__(self, root: str | Path = SOURCE_DATASET_ROOT):
        self.root = Path(root).resolve()
        self.info = load_json(self.root / "meta/info.json")
        if self.info.get("codebase_version") != "v3.0":
            raise RuntimeError("unseen integrated source is not LeRobot v3.0")
        if int(self.info.get("total_episodes", -1)) != 20:
            raise RuntimeError("unseen integrated source does not contain exactly 20 episodes")
        if float(self.info.get("fps", 0.0)) != 30.0:
            raise RuntimeError("unseen integrated source FPS is not 30")
        features = self.info["features"]
        if features["observation.state"]["shape"] != [14] or features["action"]["shape"] != [14]:
            raise RuntimeError("unseen state/action shape mismatch")
        if CAMERA_KEY not in features or features[CAMERA_KEY]["dtype"] != "video":
            raise RuntimeError("unseen cam_high feature mismatch")

        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            raise RuntimeError("unseen source has no integrated data shards")
        tables = [
            pq.read_table(
                path,
                columns=[
                    "observation.state",
                    "action",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ],
            )
            for path in data_files
        ]
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        self.state = fixed_list_numpy(table["observation.state"], 14).astype(np.float64)
        self.action = fixed_list_numpy(table["action"], 14).astype(np.float64)
        self.timestamp = scalar_numpy(table["timestamp"], np.float64)
        self.frame_index = scalar_numpy(table["frame_index"], np.int64)
        self.episode_index = scalar_numpy(table["episode_index"], np.int64)
        self.global_index = scalar_numpy(table["index"], np.int64)
        self.task_index = scalar_numpy(table["task_index"], np.int64)
        if self.action.shape != (int(self.info["total_frames"]), 14):
            raise RuntimeError("unseen integrated source action shape mismatch")
        if not all(np.isfinite(value).all() for value in (self.state, self.action, self.timestamp)):
            raise RuntimeError("unseen integrated source contains NaN/Inf")
        if not np.array_equal(np.unique(self.episode_index), np.arange(20)):
            raise RuntimeError("unseen local episode IDs are not contiguous 0..19")

        task_table = pq.read_table(self.root / "meta/tasks.parquet")
        self.tasks = {
            int(row["task_index"]): str(row["__index_level_0__"])
            for row in task_table.to_pylist()
        }
        if self.tasks != {0: TASK}:
            raise RuntimeError("unseen integrated source task metadata mismatch")
        episode_files = sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet"))
        episode_rows: list[dict[str, Any]] = []
        for path in episode_files:
            episode_rows.extend(pq.read_table(path).to_pylist())
        self.episode_meta = {int(row["episode_index"]): row for row in episode_rows}
        raw_manifest = load_json(OUTPUT_ROOT / "source/new_20_raw_manifest.json")
        self.source_manifest = {
            int(row["episode_id"]): row for row in raw_manifest["recordings"]
        }
        self.data_files = data_files
        self._validate()

    def _validate(self) -> None:
        for episode_id in self.episode_ids():
            indices = np.flatnonzero(self.episode_index == episode_id)
            if len(indices) == 0 or not np.array_equal(
                indices, np.arange(indices[0], indices[-1] + 1)
            ):
                raise RuntimeError(f"new episode {episode_id} absent/noncontiguous")
            frame = self.frame_index[indices]
            timestamp = self.timestamp[indices]
            length = int(self.episode_meta[episode_id]["length"])
            if len(indices) != length or not np.array_equal(frame, np.arange(length)):
                raise RuntimeError(f"new episode {episode_id} boundary mismatch")
            if len(timestamp) > 1 and not np.allclose(
                np.diff(timestamp), 1.0 / 30.0, atol=1e-5, rtol=0.0
            ):
                raise RuntimeError(f"new episode {episode_id} timestamp mismatch")
            if self.source_manifest[episode_id]["raw_directory_name"] != RAW_RECORDING_NAMES[episode_id]:
                raise RuntimeError(f"new episode {episode_id} raw identity mismatch")

    def episode_ids(self) -> list[int]:
        return list(range(20))

    def episode(self, episode_id: int, max_frames: int | None = None) -> SourceEpisode:
        if episode_id not in self.episode_meta:
            raise KeyError(episode_id)
        indices = np.flatnonzero(self.episode_index == episode_id)
        if max_frames is not None:
            indices = indices[:max_frames]
        source = self.source_manifest[episode_id]
        meta = self.episode_meta[episode_id]
        task_id = int(self.task_index[indices[0]])
        reference = {
            "key": CAMERA_KEY,
            "dataset_root": str(self.root),
            "video_path_template": self.info["video_path"],
            "chunk_index": int(meta[f"videos/{CAMERA_KEY}/chunk_index"]),
            "file_index": int(meta[f"videos/{CAMERA_KEY}/file_index"]),
            "from_timestamp": float(meta[f"videos/{CAMERA_KEY}/from_timestamp"]),
            "to_timestamp": float(meta[f"videos/{CAMERA_KEY}/to_timestamp"]),
            "assets_duplicated": False,
        }
        return SourceEpisode(
            episode_id=episode_id,
            action=self.action[indices].copy(),
            state=self.state[indices].copy(),
            timestamps=self.timestamp[indices].copy(),
            frame_index=self.frame_index[indices].copy(),
            task_index=self.task_index[indices].copy(),
            task=self.tasks[task_id],
            nominal_fps=30.0,
            source_folder=source["absolute_path"],
            source_parquet=str(Path(source["absolute_path"]) / "data/chunk-000/episode_000000.parquet"),
            image_reference=reference,
        )

    def integrity(self) -> dict[str, Any]:
        build = load_json(OUTPUT_ROOT / "source/source_lerobot_build_report.json")
        actual_info = sha256_file(self.root / "meta/info.json")
        actual_data = {
            str(path.relative_to(self.root)): sha256_file(path) for path in self.data_files
        }
        return {
            "root": str(self.root),
            "info_sha256": actual_info,
            "data_shards_sha256": actual_data,
            "matches_build_report": (
                actual_info == build["info_sha256"]
                and actual_data == build["data_shards_sha256"]
            ),
            "episode_count": len(self.episode_ids()),
            "frame_count": len(self.action),
            "fps": 30.0,
        }


def original_source_integrity(runtime_config: dict[str, Any]) -> dict[str, Any]:
    expected = runtime_config["source_dataset"]
    info = ORIGINAL_DATASET_ROOT / "meta/info.json"
    data = ORIGINAL_DATASET_ROOT / "data/chunk-000/file-000.parquet"
    actual = {
        "info_sha256": sha256_file(info),
        "data_sha256": sha256_file(data),
    }
    wanted = {
        "info_sha256": expected["info_sha256"],
        "data_sha256": expected["data_sha256"],
    }
    return {
        "root": str(ORIGINAL_DATASET_ROOT),
        "expected": wanted,
        "actual": actual,
        "unchanged": wanted == actual,
    }


__all__ = [
    "UnseenSourceDataset",
    "atomic_csv",
    "atomic_json",
    "atomic_npz",
    "freeze_dependencies",
    "load_json",
    "original_source_integrity",
    "sha256_file",
    "verify_dependencies_unchanged",
]
