from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.g1_training_schema_v1.constants import (
    ACTION_KEY,
    CANONICAL_JOINT_NAMES,
    FPS,
    IMAGE_KEY,
    SCHEMA_VERSION,
    STATE_KEY,
    TASK_TEXT,
    policy_features,
)
from tools.g1_training_schema_v1.normalization import feature_stats, fit_accepted_normalization
from tools.g1_training_schema_v1.source_audit import SourceDatasetIndex, load_json, sha256_file
from tools.g1_training_schema_v1.state_adapter import adapt_target_qpos
from tools.g1_training_schema_v1.target_contract import load_retargeted_trajectory
from tools.g1_training_schema_v1.validator import (
    validate_ab_schema_equality,
    validate_packaged_dataset,
    validate_source_not_copied_as_target,
    validate_target_episode,
)

MATCHED_COUNT = 51
STABLE_ID_RE = re.compile(r"^(old50|new20):(\d{3})$")
SOURCE_NAMESPACES = {
    "old50": "lerobot_magsafe_50_cam_high_v3",
    "new20": "lerobot_magsafe_20_cam_high_v3_unseen_20260813",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixed_list(values: np.ndarray, dimension: int) -> pa.FixedSizeListArray:
    flattened = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flattened, dimension)


@dataclass(frozen=True)
class MatchedEpisode:
    packaged_episode_index: int
    stable_source_id: str
    namespace: str
    local_episode_id: int
    raw_recording_name: str
    source: SourceDatasetIndex
    source_episode: Any
    source_timestamps: np.ndarray
    source_video_path: Path
    source_video_sha256: str
    q_a: np.ndarray
    q_b: np.ndarray
    trajectory_a_path: Path
    trajectory_b_path: Path
    trajectory_a_sha256: str
    trajectory_b_sha256: str
    phase_a_path: Path
    phase_b_path: Path

    @property
    def frame_count(self) -> int:
        return int(self.q_a.shape[0])


@dataclass(frozen=True)
class MatchedPool:
    manifest_path: Path
    manifest_sha256: str
    entries: tuple[MatchedEpisode, ...]
    sources: dict[str, SourceDatasetIndex]

    @property
    def total_frames(self) -> int:
        return sum(entry.frame_count for entry in self.entries)

    @property
    def stable_source_ids(self) -> list[str]:
        return [entry.stable_source_id for entry in self.entries]


def _require_pass_episode(root: Path) -> None:
    validation_path = root / "validation.json"
    validation = load_json(validation_path)
    if validation.get("status") != "PASS" or validation.get("pass") is not True:
        raise ValueError(f"matched manifest points to a non-PASS trajectory: {root}")


def load_matched_pool(manifest_path: str | Path, project_root: str | Path) -> MatchedPool:
    project_root = Path(project_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = load_json(manifest_path)
    entries_raw = manifest.get("entries", [])
    if manifest.get("entry_count") != MATCHED_COUNT or len(entries_raw) != MATCHED_COUNT:
        raise ValueError(f"matched identity count must be {MATCHED_COUNT}")

    sources = {
        namespace: SourceDatasetIndex(project_root / relative)
        for namespace, relative in SOURCE_NAMESPACES.items()
    }
    source_camera_features = [source.info["features"][IMAGE_KEY] for source in sources.values()]
    if source_camera_features[0] != source_camera_features[1]:
        raise ValueError("old50/new20 camera feature schema differs")
    task_hashes = {sha256_file(source.root / "meta/tasks.parquet") for source in sources.values()}
    if len(task_hashes) != 1:
        raise ValueError("old50/new20 task metadata differs")

    stable_ids: set[str] = set()
    matched: list[MatchedEpisode] = []
    for packaged_index, record in enumerate(entries_raw):
        stable_id = str(record.get("stable_source_id"))
        match = STABLE_ID_RE.fullmatch(stable_id)
        if match is None:
            raise ValueError(f"invalid stable source identity: {stable_id}")
        namespace, encoded_local_id = match.groups()
        local_id = int(record.get("local_episode_id"))
        if int(encoded_local_id) != local_id or record.get("namespace") != namespace:
            raise ValueError(f"stable/local identity mismatch: {stable_id}")
        if stable_id in stable_ids:
            raise ValueError(f"duplicate stable source identity: {stable_id}")
        stable_ids.add(stable_id)
        if record.get("dual_pass") is not True:
            raise ValueError(f"matched record is not dual-PASS: {stable_id}")
        source = sources[namespace]
        if local_id not in source.episodes:
            raise ValueError(f"source episode missing: {stable_id}")
        source_episode = source.episodes[local_id]
        source_arrays = source.read_episode_arrays(local_id)
        if source_episode.tasks != (TASK_TEXT,):
            raise ValueError(f"task changed for {stable_id}")

        method_data: dict[str, tuple[np.ndarray, Path, str]] = {}
        for method in ("dataset_a", "dataset_b"):
            method_record = record.get(method, {})
            if method_record.get("stable_source_id") != stable_id or method_record.get("status") != "PASS":
                raise ValueError(f"{method} identity/PASS mismatch: {stable_id}")
            if Path(method_record.get("source_dataset_root", "")).resolve() != source.root:
                raise ValueError(f"{method} source dataset mismatch: {stable_id}")
            episode_root = Path(method_record.get("retargeted_episode_root", "")).resolve()
            _require_pass_episode(episode_root)
            trajectory = load_retargeted_trajectory(episode_root)
            adapted = adapt_target_qpos(trajectory)
            validate_target_episode(
                adapted.observation_state,
                adapted.action,
                adapted.timestamps,
                source_episode.length,
            )
            validate_source_not_copied_as_target(source_arrays[STATE_KEY], adapted.observation_state)
            expected_hash = str(method_record.get("g1_full_action_sha256"))
            if trajectory.trajectory_sha256 != expected_hash:
                raise ValueError(f"{method} trajectory hash mismatch: {stable_id}")
            if not np.allclose(adapted.timestamps, source_arrays["timestamp"], rtol=0.0, atol=1e-5):
                raise ValueError(f"{method} timestamp mismatch: {stable_id}")
            method_data[method] = (adapted.action, trajectory.trajectory_path, expected_hash)

        q_a, path_a, hash_a = method_data["dataset_a"]
        q_b, path_b, hash_b = method_data["dataset_b"]
        if q_a.shape != q_b.shape:
            raise ValueError(f"A/B frame shape mismatch: {stable_id}")
        video_path = source.validate_rgb_reference(local_id)
        matched.append(
            MatchedEpisode(
                packaged_episode_index=packaged_index,
                stable_source_id=stable_id,
                namespace=namespace,
                local_episode_id=local_id,
                raw_recording_name=str(record.get("raw_recording_name")),
                source=source,
                source_episode=source_episode,
                source_timestamps=np.asarray(source_arrays["timestamp"], dtype=np.float32),
                source_video_path=video_path,
                source_video_sha256=sha256_file(video_path),
                q_a=np.ascontiguousarray(q_a, dtype=np.float32),
                q_b=np.ascontiguousarray(q_b, dtype=np.float32),
                trajectory_a_path=path_a,
                trajectory_b_path=path_b,
                trajectory_a_sha256=hash_a,
                trajectory_b_sha256=hash_b,
                phase_a_path=path_a.parent / "g1_hand_action.npz",
                phase_b_path=path_b.parent / "g1_hand_action.npz",
            )
        )
    return MatchedPool(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        entries=tuple(matched),
        sources=sources,
    )


def _video_map(pool: MatchedPool) -> dict[Path, int]:
    unique = sorted(
        {entry.source_video_path for entry in pool.entries},
        key=lambda path: (0 if SOURCE_NAMESPACES["old50"] in str(path) else 1, str(path)),
    )
    return {path: index for index, path in enumerate(unique)}


def pairing_rows(pool: MatchedPool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in pool.entries:
        rows.append(
            {
                "packaged_episode_index": entry.packaged_episode_index,
                "stable_source_id": entry.stable_source_id,
                "raw_recording_name": entry.raw_recording_name,
                "source_dataset": str(entry.source.root),
                "source_episode_index": entry.local_episode_id,
                "dataset_a_source_trajectory_path": str(entry.trajectory_a_path),
                "dataset_b_source_trajectory_path": str(entry.trajectory_b_path),
                "source_rgb_reference": str(entry.source_video_path),
                "source_rgb_sha256": entry.source_video_sha256,
                "source_rgb_from_timestamp": entry.source_episode.video_from_timestamp,
                "source_rgb_to_timestamp": entry.source_episode.video_to_timestamp,
                "task_index": 0,
                "frame_count": entry.frame_count,
            }
        )
    return rows


def write_pairing_csv(pool: MatchedPool, path: str | Path) -> None:
    path = Path(path)
    rows = pairing_rows(pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _episode_table(pool: MatchedPool, method: str, video_map: dict[Path, int]) -> pa.Table:
    rows: list[dict[str, Any]] = []
    offset = 0
    for entry in pool.entries:
        q = entry.q_a if method == "dataset_a" else entry.q_b
        prefix = f"videos/{IMAGE_KEY}"
        row: dict[str, Any] = {
            "episode_index": entry.packaged_episode_index,
            "tasks": [TASK_TEXT],
            "length": entry.frame_count,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": offset,
            "dataset_to_index": offset + entry.frame_count,
            f"{prefix}/chunk_index": 0,
            f"{prefix}/file_index": video_map[entry.source_video_path],
            f"{prefix}/from_timestamp": entry.source_episode.video_from_timestamp,
            f"{prefix}/to_timestamp": entry.source_episode.video_to_timestamp,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feature in (STATE_KEY, ACTION_KEY):
            for stat, value in feature_stats(q).items():
                row[f"stats/{feature}/{stat}"] = value
        rows.append(row)
        offset += entry.frame_count
    return pa.Table.from_pylist(rows)


def _data_table(pool: MatchedPool, method: str) -> pa.Table:
    q_values = [entry.q_a if method == "dataset_a" else entry.q_b for entry in pool.entries]
    state = np.concatenate(q_values, axis=0)
    timestamps = np.concatenate([entry.source_timestamps for entry in pool.entries]).astype(np.float32)
    frame_indices = np.concatenate([np.arange(entry.frame_count, dtype=np.int64) for entry in pool.entries])
    episode_indices = np.concatenate(
        [np.full(entry.frame_count, entry.packaged_episode_index, dtype=np.int64) for entry in pool.entries]
    )
    total_frames = len(state)
    return pa.Table.from_arrays(
        [
            _fixed_list(state, 28),
            _fixed_list(np.array(state, copy=True), 28),
            pa.array(timestamps, type=pa.float32()),
            pa.array(frame_indices, type=pa.int64()),
            pa.array(episode_indices, type=pa.int64()),
            pa.array(np.arange(total_frames, dtype=np.int64), type=pa.int64()),
            pa.array(np.zeros(total_frames, dtype=np.int64), type=pa.int64()),
        ],
        names=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "index", "task_index"],
    )


def _package_one(pool: MatchedPool, method: str, temp_root: Path, video_storage: str) -> None:
    video_map = _video_map(pool)
    data_path = temp_root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_data_table(pool, method), data_path, compression="snappy", use_dictionary=True)
    episodes_path = temp_root / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_episode_table(pool, method, video_map), episodes_path, compression="snappy", use_dictionary=True)

    first_source = pool.sources["old50"]
    shutil.copy2(first_source.root / "meta/tasks.parquet", temp_root / "meta/tasks.parquet")
    video_assets: list[dict[str, Any]] = []
    for source_path, output_file_index in video_map.items():
        relative = Path(f"videos/{IMAGE_KEY}/chunk-000/file-{output_file_index:03d}.mp4")
        destination = temp_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if video_storage == "hardlink":
            os.link(source_path, destination)
        elif video_storage == "copy":
            shutil.copy2(source_path, destination)
        else:
            raise ValueError(f"unknown video storage mode: {video_storage}")
        video_assets.append(
            {
                "output_relative_path": str(relative),
                "source_path": str(source_path),
                "sha256": sha256_file(source_path),
                "storage": video_storage,
                "same_inode_as_source": os.path.samefile(source_path, destination),
            }
        )

    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1_fixed_base_dex3_retargeted",
        "total_episodes": len(pool.entries),
        "total_frames": pool.total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": int(FPS),
        "splits": {"train": f"0:{len(pool.entries)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": policy_features(first_source.info["features"][IMAGE_KEY]),
    }
    _write_json(temp_root / "meta/info.json", info)

    q_records = [
        {
            "accepted": True,
            STATE_KEY: entry.q_a if method == "dataset_a" else entry.q_b,
            ACTION_KEY: entry.q_a if method == "dataset_a" else entry.q_b,
        }
        for entry in pool.entries
    ]
    _write_json(temp_root / "meta/stats.json", fit_accepted_normalization(q_records))
    common_contract = {
        "schema_version": SCHEMA_VERSION,
        "observation_state_semantic": "retargeted_target_state q_target[t]; not measured real-G1 actual state",
        "future_real_g1_observation_state": "actual measured G1 qpos in the identical 28D joint order",
        "action_semantic": "absolute q_target[t]",
        "causal_relation": "observation.state[t] -> action[t]",
        "row_offset": 0,
        "action_chunk_start": "action[t]",
        "action_chunk_size": 50,
        "source_visual_embodiment": "ALOHA",
        "target_action_embodiment": "Unitree_G1_Dex3",
        "joint_order": list(CANONICAL_JOINT_NAMES),
        "method_identity_is_policy_feature": False,
    }
    _write_json(temp_root / "meta/g1_training_contract.json", common_contract)

    trajectories = []
    mapping = []
    for entry in pool.entries:
        path = entry.trajectory_a_path if method == "dataset_a" else entry.trajectory_b_path
        digest = entry.trajectory_a_sha256 if method == "dataset_a" else entry.trajectory_b_sha256
        trajectories.append(
            {
                "stable_source_id": entry.stable_source_id,
                "path": str(path),
                "sha256": digest,
            }
        )
        mapping.append(
            {
                "output_episode_id": entry.packaged_episode_index,
                "stable_source_id": entry.stable_source_id,
                "namespace": entry.namespace,
                "source_episode_id": entry.local_episode_id,
                "raw_recording_name": entry.raw_recording_name,
            }
        )
    manifest = {
        "schema_version": "g1_matched51_policy_dataset_v1",
        "training_schema_version": SCHEMA_VERSION,
        "method_metadata_only": method,
        "method_is_policy_input": False,
        "method_definition": (
            "trajectory-centric wrist-level baseline with binary OPEN/CLOSE"
            if method == "dataset_a"
            else "interaction-aware task/pinch-frame and bimanual retargeting with frozen Hand-v2.1"
        ),
        "matched_manifest": str(pool.manifest_path),
        "matched_manifest_sha256": pool.manifest_sha256,
        "matched_identity_count": len(pool.entries),
        "stable_source_ids": pool.stable_source_ids,
        "output_episode_mapping": mapping,
        "trajectory_files": trajectories,
        "video_assets": video_assets,
        "video_storage_strategy": video_storage,
        "normalization": "per-dataset matched-51 accepted episodes, MEAN_STD population statistics",
        "source_visual_embodiment": "ALOHA",
        "target_action_embodiment": "Unitree_G1_Dex3",
        "retargeted_target_state_is_measured_real_g1": False,
        "images_reencoded": False,
        "images_duplicated": video_storage == "copy",
        "failed_or_unmatched_episodes_included": False,
        "training_executed": False,
    }
    _write_json(temp_root / "meta/g1_packaging_manifest.json", manifest)
    validation = validate_packaged_dataset(temp_root)
    # Keep the dataset relocatable and make independently repeated packaging
    # byte-deterministic; the temporary staging path is intentionally omitted.
    validation["root"] = "."
    validation.update(
        {
            "stable_source_ids": pool.stable_source_ids,
            "trajectory_values_exactly_preserved": True,
            "source_video_assets_hash_verified": True,
        }
    )
    _write_json(temp_root / "meta/g1_validation.json", validation)


def _read_q(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    return (
        np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32),
        np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32),
        np.asarray(table["episode_index"].to_numpy(), dtype=np.int64),
    )


def validate_matched_pair(pool: MatchedPool, root_a: str | Path, root_b: str | Path) -> dict[str, Any]:
    root_a, root_b = Path(root_a).resolve(), Path(root_b).resolve()
    result_a = validate_packaged_dataset(root_a)
    result_b = validate_packaged_dataset(root_b)
    info_a, info_b = load_json(root_a / "meta/info.json"), load_json(root_b / "meta/info.json")
    validate_ab_schema_equality(info_a, info_b)
    if info_a != info_b:
        raise ValueError("A/B meta/info.json is not byte-semantic identical")
    manifest_a = load_json(root_a / "meta/g1_packaging_manifest.json")
    manifest_b = load_json(root_b / "meta/g1_packaging_manifest.json")
    if manifest_a["stable_source_ids"] != pool.stable_source_ids or manifest_b["stable_source_ids"] != pool.stable_source_ids:
        raise ValueError("packaged stable source identities changed")
    if manifest_a["stable_source_ids"] != manifest_b["stable_source_ids"]:
        raise ValueError("A/B packaged identities differ")

    state_a, action_a, ep_a = _read_q(root_a)
    state_b, action_b, ep_b = _read_q(root_b)
    if state_a.shape != state_b.shape or not np.array_equal(ep_a, ep_b):
        raise ValueError("A/B packaged frame alignment differs")
    if not np.array_equal(state_a, action_a) or not np.array_equal(state_b, action_b):
        raise ValueError("same-row state/action contract changed")
    offset = 0
    for entry in pool.entries:
        end = offset + entry.frame_count
        if not np.array_equal(state_a[offset:end], entry.q_a):
            raise ValueError(f"Dataset A values changed: {entry.stable_source_id}")
        if not np.array_equal(state_b[offset:end], entry.q_b):
            raise ValueError(f"Dataset B values changed: {entry.stable_source_id}")
        offset = end

    assets_a = manifest_a["video_assets"]
    assets_b = manifest_b["video_assets"]
    if len(assets_a) != len(assets_b):
        raise ValueError("A/B RGB asset counts differ")
    for asset_a, asset_b in zip(assets_a, assets_b, strict=True):
        if asset_a["sha256"] != asset_b["sha256"] or asset_a["output_relative_path"] != asset_b["output_relative_path"]:
            raise ValueError("A/B RGB assets differ")
        path_a, path_b = root_a / asset_a["output_relative_path"], root_b / asset_b["output_relative_path"]
        if sha256_file(path_a) != asset_a["sha256"] or sha256_file(path_b) != asset_b["sha256"]:
            raise ValueError("packaged RGB hash mismatch")
        if asset_a["storage"] == "hardlink" and not os.path.samefile(path_a, path_b):
            raise ValueError("A/B hardlinked RGB assets do not share an inode")
    if sha256_file(root_a / "meta/tasks.parquet") != sha256_file(root_b / "meta/tasks.parquet"):
        raise ValueError("A/B task metadata differs")
    if (root_a / "meta/g1_training_contract.json").read_bytes() != (root_b / "meta/g1_training_contract.json").read_bytes():
        raise ValueError("A/B training contracts differ")
    return {
        "status": "PASS",
        "dataset_a": result_a,
        "dataset_b": result_b,
        "matched_episode_count": len(pool.entries),
        "total_frames": pool.total_frames,
        "corresponding_frame_counts_identical": True,
        "schema_identical": True,
        "rgb_assets_byte_identical": True,
        "task_metadata_identical": True,
        "trajectory_values_exactly_preserved": True,
    }


def package_matched_pair(
    pool: MatchedPool,
    output_a: str | Path,
    output_b: str | Path,
    video_storage: str = "hardlink",
) -> dict[str, Any]:
    output_a, output_b = Path(output_a).resolve(), Path(output_b).resolve()
    if output_a == output_b:
        raise ValueError("Dataset A/B output roots must be separate")
    if output_a.exists() or output_b.exists():
        raise FileExistsError("refusing to overwrite an existing packaged dataset root")
    if output_a.parent != output_b.parent:
        raise ValueError("atomic pair packaging requires one common parent")
    output_a.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".matched51-pair.tmp-", dir=output_a.parent))
    temp_a, temp_b = staging / output_a.name, staging / output_b.name
    try:
        _package_one(pool, "dataset_a", temp_a, video_storage)
        _package_one(pool, "dataset_b", temp_b, video_storage)
        validation = validate_matched_pair(pool, temp_a, temp_b)
        os.replace(temp_a, output_a)
        os.replace(temp_b, output_b)
        staging.rmdir()
        return validation
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if output_a.exists() and not output_b.exists():
            shutil.rmtree(output_a)
        raise
