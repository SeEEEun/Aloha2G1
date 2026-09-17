from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .constants import ACTION_KEY, FPS, IMAGE_KEY, SCHEMA_VERSION, STATE_KEY, TASK_TEXT, policy_features
from .episode_filter import audit_episode_root, select_episode_ids
from .normalization import feature_stats, fit_accepted_normalization
from .source_audit import SourceDatasetIndex, sha256_file
from .state_adapter import adapt_target_qpos
from .target_contract import load_retargeted_trajectory
from .validator import validate_packaged_dataset, validate_source_not_copied_as_target, validate_target_episode


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unknown video storage mode: {mode}")


def _fixed_list(values: np.ndarray, dimension: int) -> pa.FixedSizeListArray:
    flattened = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flattened, dimension)


def inspect_packaging_inputs(
    method: str,
    input_root: str | Path,
    source_dataset: str | Path,
    episode_mode: str = "native",
    matched_with: str | Path | None = None,
    failure_policy: str = "reject_failed",
) -> dict[str, Any]:
    if failure_policy != "reject_failed":
        raise NotImplementedError(
            f"failure policy {failure_policy!r} is intentionally unsupported in schema v1; use reject_failed"
        )
    source = SourceDatasetIndex(source_dataset)
    directories, decisions = audit_episode_root(input_root)
    other_decisions = None
    if episode_mode == "matched":
        if matched_with is None:
            raise ValueError("matched episode mode requires --matched-with")
        _, other_decisions = audit_episode_root(matched_with)
    selected = select_episode_ids(decisions, episode_mode, other_decisions)
    missing_source = sorted(set(directories) - set(source.episodes))
    if missing_source:
        raise ValueError(f"retarget episodes absent from source dataset: {missing_source}")
    validated_inputs: list[dict[str, Any]] = []
    for episode_id in selected:
        trajectory = load_retargeted_trajectory(directories[episode_id])
        adapted = adapt_target_qpos(trajectory)
        source_arrays = source.read_episode_arrays(episode_id)
        validate_target_episode(
            adapted.observation_state, adapted.action, adapted.timestamps, len(source_arrays["timestamp"])
        )
        validate_source_not_copied_as_target(source_arrays[STATE_KEY], adapted.observation_state)
        if not np.allclose(adapted.timestamps, source_arrays["timestamp"], rtol=0.0, atol=1e-5):
            raise ValueError(f"episode {episode_id} target/source timestamp mismatch")
        rgb_path = source.validate_rgb_reference(episode_id)
        validated_inputs.append(
            {
                "episode_id": episode_id,
                "frame_count": len(adapted.timestamps),
                "state_dimension": adapted.observation_state.shape[1],
                "action_dimension": adapted.action.shape[1],
                "trajectory_sha256": trajectory.trajectory_sha256,
                "canonical_reorder_indices": list(trajectory.reorder_indices),
                "rgb_reference": str(rgb_path),
                "rgb_sha256": sha256_file(rgb_path),
                "status": "PASS_INPUT_VALIDATION",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "method_metadata_only": method,
        "input_root": str(Path(input_root).resolve()),
        "source_dataset": str(source.root),
        "episode_set_mode": episode_mode,
        "failure_policy": failure_policy,
        "discovered_episode_ids": sorted(directories),
        "accepted_episode_ids": sorted(ep for ep, decision in decisions.items() if decision.accepted),
        "selected_episode_ids": selected,
        "validated_selected_inputs": validated_inputs,
        "rejected_episodes": [
            {**decisions[ep].to_json(), "method": method}
            for ep in sorted(decisions)
            if not decisions[ep].accepted
        ],
        "counts": {
            "discovered": len(directories),
            "accepted_native": sum(decision.accepted for decision in decisions.values()),
            "selected": len(selected),
            "rejected": sum(not decision.accepted for decision in decisions.values()),
        },
    }


def package_dataset(
    method: str,
    input_root: str | Path,
    source_dataset: str | Path,
    output_root: str | Path,
    episode_mode: str = "native",
    matched_with: str | Path | None = None,
    video_storage: str = "hardlink",
    failure_policy: str = "reject_failed",
) -> dict[str, Any]:
    if method not in {"dataset_a", "dataset_b"}:
        raise ValueError("method must be dataset_a or dataset_b")
    if failure_policy != "reject_failed":
        raise NotImplementedError(
            f"failure policy {failure_policy!r} is intentionally unsupported in schema v1; use reject_failed"
        )
    source = SourceDatasetIndex(source_dataset)
    directories, decisions = audit_episode_root(input_root)
    other_decisions = None
    if episode_mode == "matched":
        if matched_with is None:
            raise ValueError("matched mode requires matched_with")
        _, other_decisions = audit_episode_root(matched_with)
    selected_ids = select_episode_ids(decisions, episode_mode, other_decisions)
    if not selected_ids:
        raise ValueError("no accepted episodes selected; no dataset was written")
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if output_root == Path(input_root).resolve() or output_root == source.root:
        raise ValueError("output root must differ from input and source roots")

    records: list[dict[str, Any]] = []
    video_sources: dict[str, Path] = {}
    for source_episode_id in selected_ids:
        if source_episode_id not in source.episodes or source_episode_id not in directories:
            raise ValueError(f"episode {source_episode_id} missing from source or retarget root")
        if not decisions[source_episode_id].accepted:
            raise ValueError(f"episode {source_episode_id} was not accepted")
        trajectory = load_retargeted_trajectory(directories[source_episode_id])
        adapted = adapt_target_qpos(trajectory)
        source_arrays = source.read_episode_arrays(source_episode_id)
        source_episode = source.episodes[source_episode_id]
        validate_target_episode(
            adapted.observation_state, adapted.action, adapted.timestamps, source_episode.length
        )
        validate_source_not_copied_as_target(source_arrays[STATE_KEY], adapted.observation_state)
        if not np.allclose(adapted.timestamps, source_arrays["timestamp"], rtol=0.0, atol=1e-5):
            raise ValueError(f"episode {source_episode_id} target/source timestamp mismatch")
        video_path = source.validate_rgb_reference(source_episode_id)
        video_rel = str(video_path.relative_to(source.root))
        video_sources[video_rel] = video_path
        records.append(
            {
                "accepted": True,
                "source_episode_id": source_episode_id,
                "source_episode": source_episode,
                STATE_KEY: adapted.observation_state,
                ACTION_KEY: adapted.action,
                "timestamp": source_arrays["timestamp"].astype(np.float32),
                "trajectory": trajectory,
            }
        )

    normalization = fit_accepted_normalization(records)
    total_frames = sum(len(record[STATE_KEY]) for record in records)
    temp_parent = output_root.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=temp_parent))
    try:
        state = np.concatenate([record[STATE_KEY] for record in records], axis=0)
        action = np.concatenate([record[ACTION_KEY] for record in records], axis=0)
        timestamps = np.concatenate([record["timestamp"] for record in records]).astype(np.float32)
        frame_indices = np.concatenate(
            [np.arange(len(record[STATE_KEY]), dtype=np.int64) for record in records]
        )
        episode_indices = np.concatenate(
            [np.full(len(record[STATE_KEY]), output_ep, dtype=np.int64) for output_ep, record in enumerate(records)]
        )
        table = pa.Table.from_arrays(
            [
                _fixed_list(state, state.shape[1]),
                _fixed_list(action, action.shape[1]),
                pa.array(timestamps, type=pa.float32()),
                pa.array(frame_indices, type=pa.int64()),
                pa.array(episode_indices, type=pa.int64()),
                pa.array(np.arange(total_frames, dtype=np.int64), type=pa.int64()),
                pa.array(np.zeros(total_frames, dtype=np.int64), type=pa.int64()),
            ],
            names=[STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "index", "task_index"],
        )
        data_path = temp_root / "data/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path, compression="snappy", use_dictionary=True)

        episode_rows: list[dict[str, Any]] = []
        offset = 0
        for output_ep, record in enumerate(records):
            source_episode = record["source_episode"]
            length = len(record[STATE_KEY])
            prefix = f"videos/{IMAGE_KEY}"
            row: dict[str, Any] = {
                "episode_index": output_ep,
                "tasks": [TASK_TEXT],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": offset,
                "dataset_to_index": offset + length,
                f"{prefix}/chunk_index": source_episode.video_chunk_index,
                f"{prefix}/file_index": source_episode.video_file_index,
                f"{prefix}/from_timestamp": source_episode.video_from_timestamp,
                f"{prefix}/to_timestamp": source_episode.video_to_timestamp,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            for feature in (STATE_KEY, ACTION_KEY):
                for stat, value in feature_stats(record[feature]).items():
                    row[f"stats/{feature}/{stat}"] = value
            episode_rows.append(row)
            offset += length
        episodes_path = temp_root / "meta/episodes/chunk-000/file-000.parquet"
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(episode_rows), episodes_path, compression="snappy", use_dictionary=True)

        # Language metadata is tiny and copied so --video-storage copy also works across filesystems.
        _link_or_copy(source.root / "meta/tasks.parquet", temp_root / "meta/tasks.parquet", "copy")
        video_assets: list[dict[str, Any]] = []
        for relative_path, source_path in sorted(video_sources.items()):
            _link_or_copy(source_path, temp_root / relative_path, video_storage)
            video_assets.append(
                {
                    "relative_path": relative_path,
                    "sha256": sha256_file(source_path),
                    "storage": video_storage,
                    "source": str(source_path),
                }
            )

        info = {
            "codebase_version": "v3.0",
            "robot_type": "unitree_g1_fixed_base_dex3_retargeted",
            "total_episodes": len(records),
            "total_frames": total_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500,
            "fps": int(FPS),
            "splits": {"train": f"0:{len(records)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": policy_features(source.info["features"][IMAGE_KEY]),
        }
        _write_json(temp_root / "meta/info.json", info)
        _write_json(temp_root / "meta/stats.json", normalization)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "method_metadata_only": method,
            "method_is_policy_input": False,
            "source_dataset": str(source.root),
            "source_info_sha256": sha256_file(source.info_path),
            "retarget_input_root": str(Path(input_root).resolve()),
            "episode_set_mode": episode_mode,
            "failure_policy": failure_policy,
            "source_episode_ids": selected_ids,
            "rejected_episodes": [
                {**decisions[ep].to_json(), "method": method}
                for ep in sorted(decisions)
                if not decisions[ep].accepted
            ],
            "output_episode_mapping": [
                {"output_episode_id": idx, "source_episode_id": record["source_episode_id"]}
                for idx, record in enumerate(records)
            ],
            "trajectory_files": [
                {
                    "source_episode_id": record["source_episode_id"],
                    "path": str(record["trajectory"].trajectory_path),
                    "sha256": record["trajectory"].trajectory_sha256,
                    "input_joint_names": list(record["trajectory"].input_joint_names),
                    "canonical_reorder_indices": list(record["trajectory"].reorder_indices),
                }
                for record in records
            ],
            "video_assets": video_assets,
            "normalization": "per-dataset accepted training episodes, MEAN_STD",
            "images_duplicated": video_storage == "copy",
            "structural_dry_run": False,
            "training_executed": False,
        }
        _write_json(temp_root / "meta/g1_packaging_manifest.json", manifest)
        validation = validate_packaged_dataset(temp_root, source.root)
        validation["root"] = "."
        _write_json(temp_root / "meta/g1_validation.json", validation)
        os.replace(temp_root, output_root)
    except Exception:
        if temp_root.is_dir() and temp_root.parent == temp_parent:
            shutil.rmtree(temp_root)
        raise
    return validate_packaged_dataset(output_root, source.root)
