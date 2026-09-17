#!/usr/bin/env python3
"""Finalize the approved Doll-Handoff Proposed-B dataset without mutating it.

This driver is intentionally outside the frozen retargeting and feasibility
packages.  Every stage fails closed on the hashes captured by the 2026-08-21
freeze manifests.  The initial ``audit-new`` stage is source-data-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    atomic_csv,
    atomic_json,
    implementation_fingerprint,
    load_json,
    sha256_file,
)
from tools.doll_handoff_feasibility.solver import (  # noqa: E402
    resolver_implementation_hash,
)


RAW = REPOSITORY / "raw_recordings"
FINAL_ROOT = REPOSITORY / "outputs/doll_handoff_dataset_b_final"
DATASET_ROOT = REPOSITORY / "datasets/doll_handoff_proposed_b_50"
TRAINING_ROOT = FINAL_ROOT / "training"
SMOKE_ROOT = FINAL_ROOT / "training_smoke"
SMOKE_OUTPUT = SMOKE_ROOT / "NOT_A_RESEARCH_POLICY"
REAL_TRAINING_OUTPUT = REPOSITORY / "outputs/policy_b_doll_handoff_proposed_b_50"
LEROBOT_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
LEROBOT_TRAIN = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train")
LEROBOT_SOURCE = Path("/home/jbnu/lerobot-smolvla/src/lerobot")
BASE_MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
BASE_MODEL = (
    Path("/home/jbnu/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots")
    / BASE_MODEL_REVISION
)
REFERENCE_G1_POLICY_CONFIG = (
    REPOSITORY
    / "outputs/g1_policy_dataset_packaging_v1/training_configs/policy_b_config.json"
)
FULL_TRAINING_CONFIG = TRAINING_ROOT / "policy_b_full_config.json"
SMOKE_TRAINING_CONFIG = TRAINING_ROOT / "policy_b_smoke_config.json"
MODEL_PREFLIGHT = TRAINING_ROOT / "smolvla_preflight.json"
MODEL_COMPATIBILITY = TRAINING_ROOT / "model_compatibility.json"
SMOKE_STEPS = 10
FULL_TRAINING_STEPS = 20_000
REVIEW_ROOT = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
)
RESOLVER_ROOT = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/g1_feasibility_resolver_2026-08-21"
)
APPROVAL_FREEZE = REVIEW_ROOT / "frozen_approval/freeze_manifest.json"
MOTION_FREEZE = (
    REVIEW_ROOT
    / "review/dataset_b_gate/motion_freeze/motion_freeze_manifest.json"
)
RESOLVER_FREEZE = RESOLVER_ROOT / "freeze_manifest.json"
NATURAL_ARM_FREEZE = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/natural_arm_audit/"
    "frozen_common_natural_arm/freeze_manifest.json"
)
NEW_SOURCE_NAMES = (
    "GoPark_20260823_135848",
    "GoPark_20260823_140035",
)
NEW_STABLE_IDS = {
    "GoPark_20260823_135848": "doll_handoff_20260823_135848",
    "GoPark_20260823_140035": "doll_handoff_20260823_140035",
}
TASK_INSTRUCTION = (
    "Pick up the doll with the left hand, handoff it to the right hand, "
    "and place it in the trash bin."
)
REFERENCE_SOURCE = "GoPark_20260820_154342"
EXPECTED_CHANNEL_NAMES = [f"left_joint_{index}" for index in range(7)] + [
    f"right_joint_{index}" for index in range(7)
]
EXPECTED_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
EXPECTED_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_low",
    "observation.images.cam_right_wrist",
)


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_array_sha256(value: np.ndarray) -> str:
    """Hash an ndarray with the convention used by the resolver freeze."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def canonical_equal(first: Any, second: Any) -> bool:
    return stable_json_sha256(jsonable(first)) == stable_json_sha256(jsonable(second))


def verify_frozen_provenance() -> dict[str, Any]:
    approval = load_json(APPROVAL_FREEZE)
    motion = load_json(MOTION_FREEZE)
    resolver = load_json(RESOLVER_FREEZE)
    natural = load_json(NATURAL_ARM_FREEZE)
    current_implementation, current_files = implementation_fingerprint()
    current_resolver = resolver_implementation_hash()
    checks = {
        "approval_status": approval.get("status")
        == "PROPOSED_B_APPROVED_FOR_50_EPISODE_BATCH",
        "motion_status": motion.get("status")
        == "PROPOSED_B_MOTION_FROZEN_FOR_DATASET_AUDIT",
        "resolver_status": resolver.get("status")
        == "GENERIC_G1_FEASIBILITY_REVIEW_FROZEN",
        "natural_arm_status": natural.get("status")
        == "COMMON_NATURAL_ARM_SOLVER_FROZEN",
        "proposed_b_implementation": current_implementation
        == approval.get("implementation_sha256")
        == motion.get("implementation_sha256"),
        "proposed_b_file_hashes": current_files
        == approval.get("implementation_files"),
        "resolver_implementation": current_resolver["implementation_sha256"]
        == resolver.get("resolver", {}).get("implementation_sha256"),
        "resolver_file_hashes": current_resolver["files"]
        == resolver.get("resolver", {}).get("files"),
        "resolver_config": sha256_file(
            REPOSITORY / "configs/doll_handoff_g1_feasibility_resolver.json"
        )
        == resolver.get("resolver", {}).get("config_sha256"),
        "natural_arm_config": natural.get("config_sha256")
        == resolver.get("frozen_before", {}).get("common_natural_arm_solver_sha256"),
        "cartesian_targets": motion.get("cartesian_target_array_set_sha256")
        == resolver.get("frozen_before", {}).get("cartesian_target_array_set_sha256"),
        "handoff_cartesian_residual_zero": float(
            motion.get("handoff_cartesian_residual_m", float("nan"))
        )
        == 0.0,
    }
    result = {
        "checks": checks,
        "pass": all(checks.values()),
        "authoritative": {
            "proposed_b_implementation_sha256": current_implementation,
            "cartesian_target_array_set_sha256": motion.get(
                "cartesian_target_array_set_sha256"
            ),
            "common_natural_arm_solver_sha256": natural.get("config_sha256"),
            "generic_feasibility_resolver_sha256": current_resolver[
                "implementation_sha256"
            ],
            "generic_feasibility_resolver_config_sha256": sha256_file(
                REPOSITORY / "configs/doll_handoff_g1_feasibility_resolver.json"
            ),
            "scene_layout_sha256": approval.get(
                "approved_resolved_file_sha256", {}
            ).get("scene_layout"),
        },
    }
    if not result["pass"]:
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"frozen Proposed-B provenance mismatch: {failures}")
    return result


def fixed_list_numpy(column: pa.ChunkedArray) -> np.ndarray:
    values = column.combine_chunks()
    if not pa.types.is_fixed_size_list(values.type):
        raise TypeError(f"not a fixed-size list: {values.type}")
    return np.asarray(values.values.to_numpy(zero_copy_only=False)).reshape(
        len(values), values.type.list_size
    )


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    return {
        "readable": completed.returncode == 0,
        "returncode": completed.returncode,
        "probe": json.loads(completed.stdout or "{}")
        if completed.returncode == 0
        else None,
        "stderr": completed.stderr.strip(),
    }


def inspect_source(root: Path, reference_schema: pa.Schema) -> dict[str, Any]:
    problems: list[str] = []
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if len(parquet_files) != 1:
        problems.append(f"expected one parquet; found {len(parquet_files)}")
        return {
            "source_name": root.name,
            "source_root": str(root.resolve()),
            "valid": False,
            "problems": problems,
        }
    parquet = parquet_files[0]
    try:
        table = pq.read_table(parquet)
    except Exception as error:
        problems.append(f"parquet unreadable: {type(error).__name__}: {error}")
        return {
            "source_name": root.name,
            "source_root": str(root.resolve()),
            "parquet_path": str(parquet.resolve()),
            "valid": False,
            "problems": problems,
        }
    schema = table.schema.remove_metadata()
    if schema != reference_schema:
        problems.append("Arrow schema differs from authoritative 2026-08-20 schema")
    if tuple(table.column_names) != EXPECTED_COLUMNS:
        problems.append(
            f"column order differs: expected {EXPECTED_COLUMNS}, got {tuple(table.column_names)}"
        )
    frame_count = int(table.num_rows)
    arrays: dict[str, np.ndarray] = {}
    for key in ("action", "observation.state"):
        try:
            arrays[key] = fixed_list_numpy(table[key]).astype(np.float64)
        except Exception as error:
            problems.append(f"{key} invalid: {type(error).__name__}: {error}")
            arrays[key] = np.empty((0, 0), dtype=np.float64)
        if arrays[key].shape != (frame_count, 14):
            problems.append(f"{key} shape is {arrays[key].shape}, expected {(frame_count, 14)}")
    scalars: dict[str, np.ndarray] = {}
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        try:
            scalars[key] = np.asarray(
                table[key].combine_chunks().to_numpy(zero_copy_only=False)
            )
        except Exception as error:
            problems.append(f"{key} invalid: {type(error).__name__}: {error}")
            scalars[key] = np.empty(0)
    numeric = [*arrays.values(), *scalars.values()]
    nan_count = int(
        sum(np.count_nonzero(np.isnan(value)) for value in numeric if value.dtype.kind == "f")
    )
    inf_count = int(
        sum(np.count_nonzero(np.isinf(value)) for value in numeric if value.dtype.kind == "f")
    )
    if nan_count or inf_count:
        problems.append(f"numeric data contain NaN={nan_count}, Inf={inf_count}")
    if not np.array_equal(scalars["frame_index"], np.arange(frame_count)):
        problems.append("frame_index is not contiguous from zero")
    if not np.array_equal(scalars["index"], np.arange(frame_count)):
        problems.append("index is not contiguous from zero")
    if not np.array_equal(scalars["episode_index"], np.zeros(frame_count, dtype=np.int64)):
        problems.append("episode_index is not uniformly zero")
    if not np.array_equal(scalars["task_index"], np.zeros(frame_count, dtype=np.int64)):
        problems.append("task_index is not uniformly zero")

    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.jsonl"
    info: dict[str, Any] = {}
    task_rows: list[dict[str, Any]] = []
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as error:
        problems.append(f"meta/info.json invalid: {error}")
    try:
        task_rows = [
            json.loads(line)
            for line in tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as error:
        problems.append(f"meta/tasks.jsonl invalid: {error}")
    fps = float(info.get("fps", 0.0) or 0.0)
    timestamp = scalars["timestamp"].astype(np.float64)
    timestamp_monotonic = bool(
        len(timestamp) <= 1 or np.all(np.diff(timestamp) > 0.0)
    )
    cadence_matches = bool(
        len(timestamp) <= 1
        or (
            fps > 0.0
            and np.allclose(np.diff(timestamp), 1.0 / fps, atol=2e-6, rtol=0.0)
        )
    )
    if not timestamp_monotonic:
        problems.append("timestamps are not strictly monotonic")
    if not cadence_matches:
        problems.append("timestamp cadence differs from metadata FPS")
    if fps != 30.0:
        problems.append(f"FPS is {fps}, expected 30.0")
    feature_shapes = {
        key: info.get("features", {}).get(key, {}).get("shape")
        for key in ("action", "observation.state")
    }
    feature_names = {
        key: info.get("features", {}).get(key, {}).get("names")
        for key in ("action", "observation.state")
    }
    if any(value != [14] for value in feature_shapes.values()):
        problems.append(f"metadata action/state shapes differ from [14]: {feature_shapes}")
    if any(value != EXPECTED_CHANNEL_NAMES for value in feature_names.values()):
        problems.append("metadata action/state joint names or order differ")
    camera_keys = tuple(
        sorted(
            key
            for key, value in info.get("features", {}).items()
            if isinstance(value, Mapping) and value.get("dtype") == "video"
        )
    )
    if camera_keys != EXPECTED_CAMERAS:
        problems.append(f"camera keys differ: {camera_keys}")
    cameras: dict[str, Any] = {}
    for key in EXPECTED_CAMERAS:
        directory = root / "images" / key / "episode_000000"
        images = sorted(directory.glob("frame_*.png"))
        names_contiguous = [path.name for path in images] == [
            f"frame_{index:06d}.png" for index in range(frame_count)
        ]
        sample = cv2.imread(str(images[0]), cv2.IMREAD_COLOR) if images else None
        sample_shape = list(sample.shape) if sample is not None else None
        expected_shape = info.get("features", {}).get(key, {}).get("shape")
        if len(images) != frame_count:
            problems.append(f"{key} has {len(images)} images, expected {frame_count}")
        if not names_contiguous:
            problems.append(f"{key} frame filenames are not contiguous")
        if sample_shape != expected_shape:
            problems.append(
                f"{key} sample shape {sample_shape} differs from metadata {expected_shape}"
            )
        cameras[key] = {
            "image_directory": str(directory.resolve()),
            "image_count": len(images),
            "filenames_contiguous": names_contiguous,
            "sample_shape": sample_shape,
            "metadata_shape": expected_shape,
            "first_image_sha256": sha256_file(images[0]) if images else None,
            "last_image_sha256": sha256_file(images[-1]) if images else None,
        }
    video_files = sorted((root / "videos").rglob("*.mp4"))
    videos = {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            **ffprobe_video(path),
        }
        for path in video_files
    }
    if any(not value["readable"] for value in videos.values()):
        problems.append("one or more packaged source videos are unreadable")
    task_valid = bool(
        len(task_rows) == 1
        and task_rows[0].get("task_index") == 0
        and isinstance(task_rows[0].get("task"), str)
        and task_rows[0]["task"].strip()
    )
    if not task_valid:
        problems.append("task/language metadata are missing or malformed")
    raw_meta_counts = {
        key: info.get(key)
        for key in (
            "total_episodes",
            "total_frames",
            "total_tasks",
            "total_videos",
            "total_chunks",
        )
    }
    return {
        "source_name": root.name,
        "source_root": str(root.resolve()),
        "original_date": root.name.split("_")[1][:8],
        "parquet_path": str(parquet.resolve()),
        "parquet_sha256": sha256_file(parquet),
        "parquet_count": len(parquet_files),
        "actual_episode_count": 1,
        "frame_count": frame_count,
        "fps": fps,
        "timestamp_dtype": str(scalars["timestamp"].dtype),
        "timestamp_start_s": float(timestamp[0]) if len(timestamp) else None,
        "timestamp_end_s": float(timestamp[-1]) if len(timestamp) else None,
        "duration_s": float(timestamp[-1] - timestamp[0]) if len(timestamp) else 0.0,
        "timestamp_monotonic": timestamp_monotonic,
        "timestamp_cadence_matches_fps": cadence_matches,
        "observation_state_key": "observation.state",
        "observation_state_shape": list(arrays["observation.state"].shape),
        "action_key": "action",
        "action_shape": list(arrays["action"].shape),
        "joint_channel_names": EXPECTED_CHANNEL_NAMES,
        "left_arm_channels": list(range(0, 6)),
        "left_gripper_channel": 6,
        "right_arm_channels": list(range(7, 13)),
        "right_gripper_channel": 13,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "arrow_schema": str(schema),
        "arrow_schema_sha256": stable_json_sha256(str(schema)),
        "schema_matches_20260820": schema == reference_schema,
        "camera_assets": cameras,
        "video_assets": {
            "count": len(video_files),
            "files": videos,
            "interpretation": (
                "Standalone raw recordings retain frame-aligned PNG camera assets; "
                "their videos directory is empty, matching the authoritative 2026-08-20 recordings."
            ),
        },
        "task_metadata": task_rows,
        "task_metadata_valid": task_valid,
        "raw_info_counters": raw_meta_counts,
        "raw_info_counter_interpretation": (
            "Standalone acquisition metadata counters are zero in both historical "
            "and new recordings; actual counts are audited from parquet and images."
        ),
        "valid": not problems,
        "problems": problems,
    }


def audit_new_sources() -> dict[str, Any]:
    provenance = verify_frozen_provenance()
    reference_path = next((RAW / REFERENCE_SOURCE / "data").rglob("*.parquet"))
    reference_table = pq.read_table(reference_path)
    reference_schema = reference_table.schema.remove_metadata()
    reference_info = load_json(RAW / REFERENCE_SOURCE / "meta/info.json")
    records = [inspect_source(RAW / name, reference_schema) for name in NEW_SOURCE_NAMES]
    output = FINAL_ROOT / "new_source_audit"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "doll_handoff_dataset_b_new_source_manifest_v1",
        "status": "PASS" if all(record["valid"] for record in records) else "FAIL",
        "source_count": len(records),
        "authoritative_reference_source": REFERENCE_SOURCE,
        "authoritative_reference_parquet_sha256": sha256_file(reference_path),
        "authoritative_reference_arrow_schema": str(reference_schema),
        "authoritative_reference_metadata": {
            "codebase_version": reference_info.get("codebase_version"),
            "robot_type": reference_info.get("robot_type"),
            "fps": reference_info.get("fps"),
            "task": json.loads(
                (RAW / REFERENCE_SOURCE / "meta/tasks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            ),
        },
        "frozen_provenance": provenance["authoritative"],
        "episodes": records,
    }
    validation = {
        "schema_version": "doll_handoff_dataset_b_new_source_validation_v1",
        "status": manifest["status"],
        "checks": {
            "frozen_provenance": provenance["pass"],
            "source_count_is_two": len(records) == 2,
            "both_parquets_readable": all(record.get("parquet_count") == 1 for record in records),
            "both_single_episode": all(record.get("actual_episode_count") == 1 for record in records),
            "schema_matches_authoritative": all(record.get("schema_matches_20260820") for record in records),
            "state_shape_14": all(record.get("observation_state_shape", [None])[-1] == 14 for record in records),
            "action_shape_14": all(record.get("action_shape", [None])[-1] == 14 for record in records),
            "timestamps_valid": all(
                record.get("timestamp_monotonic")
                and record.get("timestamp_cadence_matches_fps")
                for record in records
            ),
            "camera_assets_complete": all(
                all(
                    camera["image_count"] == record["frame_count"]
                    and camera["filenames_contiguous"]
                    and camera["sample_shape"] == camera["metadata_shape"]
                    for camera in record["camera_assets"].values()
                )
                for record in records
            ),
            "task_metadata_present": all(record.get("task_metadata_valid") for record in records),
            "nan_zero": all(record.get("nan_count") == 0 for record in records),
            "inf_zero": all(record.get("inf_count") == 0 for record in records),
            "both_valid": all(record["valid"] for record in records),
        },
        "episodes": {
            record["source_name"]: {
                "valid": record["valid"],
                "problems": record["problems"],
            }
            for record in records
        },
    }
    validation["pass"] = all(validation["checks"].values())
    if not validation["pass"]:
        validation["status"] = "FAIL"
        manifest["status"] = "FAIL"
    atomic_json(output / "new_episode_manifest.json", manifest)
    atomic_json(output / "validation.json", validation)
    lines = [
        "# New 2026-08-23 source schema comparison",
        "",
        f"Overall status: **{validation['status']}**",
        "",
        f"Authoritative comparator: `{REFERENCE_SOURCE}` from the frozen 2026-08-20 batch.",
        "",
        "| source | episodes | frames | FPS | state | action | cameras | videos | NaN / Inf | schema |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| `{record['source_name']}` | {record.get('actual_episode_count')} | "
            f"{record.get('frame_count')} | {record.get('fps')} | "
            f"`{record.get('observation_state_shape')}` | `{record.get('action_shape')}` | "
            f"{len(record.get('camera_assets', {}))} PNG streams | "
            f"{record.get('video_assets', {}).get('count')} | "
            f"{record.get('nan_count')} / {record.get('inf_count')} | "
            f"{'MATCH' if record.get('schema_matches_20260820') else 'DIFFERENT'} |"
        )
    lines.extend(
        [
            "",
            "Both recordings use the same 14-channel fixed-size-list Arrow schema, "
            "joint order, 30 Hz timestamp convention, and four frame-aligned 640×480 "
            "PNG camera streams as the accepted 2026-08-20 sources.",
            "",
            "The standalone acquisition `meta/info.json` counters remain zero and the "
            "`videos/` trees contain no MP4 files. This is the same layout as the "
            "authoritative recordings: actual episode/frame counts come from parquet "
            "and raw PNGs, which are complete. No source data were repaired or synthesized.",
            "",
            "The raw task string is `A dummy task`, also matching the historical source "
            "metadata. Dataset packaging must use the project-level authoritative "
            "Doll-Handoff instruction consistently rather than treat this placeholder as semantic truth.",
            "",
        ]
    )
    (output / "schema_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    if not validation["pass"]:
        raise RuntimeError("one or both new recordings failed source validation")
    return validation


def replacement_provenance() -> dict[str, Any]:
    source_manifest = load_json(
        REVIEW_ROOT / "frozen_approval/source/source_manifest.json"
    )
    resolver_freeze = load_json(RESOLVER_FREEZE)
    with (RESOLVER_ROOT / "full50/per_episode.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = {int(row["episode_index"]): row for row in csv.DictReader(stream)}
    freeze_episodes = {
        int(row["episode_index"]): row
        for row in resolver_freeze["integrity"]["episodes"]
    }
    expected_names = {13: "GoPark_20260820_154342", 36: "GoPark_20260820_161731"}
    excluded: list[dict[str, Any]] = []
    for episode in (13, 36):
        source = source_manifest["records"][episode]
        row = rows[episode]
        frozen = freeze_episodes[episode]
        if int(source["episode_index"]) != episode:
            raise RuntimeError(f"frozen source record ordering drift at ep{episode:03d}")
        if source["source_name"] != expected_names[episode]:
            raise RuntimeError(
                f"unexpected frozen mapping ep{episode:03d}: {source['source_name']}"
            )
        if row["after_classification"] != "HARD_FAIL":
            raise RuntimeError(f"ep{episode:03d} is no longer a frozen HARD_FAIL")
        after_path = (
            RESOLVER_ROOT
            / "after/trajectories"
            / f"doll_handoff_20260820_ep{episode:03d}.npz"
        )
        if sha256_file(after_path) != frozen["after_trajectory_sha256"]:
            raise RuntimeError(f"resolved trajectory hash mismatch at ep{episode:03d}")
        excluded.append(
            {
                "original_episode_index": episode,
                "stable_episode_id": f"doll_handoff_20260820_ep{episode:03d}",
                "raw_directory": source["source_name"],
                "raw_directory_path": source["source_root"],
                "raw_parquet_sha256": source["parquet_sha256"],
                "source_action_sha256": source["action_sha256"],
                "source_state_trajectory_sha256": source["state_sha256"],
                "frozen_proposed_b_trajectory_sha256": frozen[
                    "before_trajectory_sha256"
                ],
                "resolved_proposed_b_trajectory_sha256": frozen[
                    "after_trajectory_sha256"
                ],
                "resolved_action_array_set_sha256": frozen[
                    "after_action_array_set_sha256"
                ],
                "classification": row["after_classification"],
                "hard_fail_reasons": json.loads(row["after_classification_reasons"]),
                "persistent_hard_collision": {
                    "arm_torso_frames": int(row["after_arm_torso_hard_frames"]),
                    "cross_arm_frames": 0,
                    "distal_frames": int(row["after_distal_hard_frames"]),
                    "maximum_penetration_m": float(
                        row["after_maximum_penetration_m"]
                    ),
                    "segments": json.loads(row["after_collision_segments"]),
                },
                "exclusion_reason": (
                    "persistent target-realization HARD_FAIL under the frozen "
                    "generic resolver"
                ),
                "raw_recording_deleted": False,
            }
        )
    result = {
        "schema_version": "doll_handoff_dataset_b_replacement_provenance_v1",
        "status": "VERIFIED",
        "mapping_rule": source_manifest["stable_id_rule"],
        "authoritative_source_manifest": str(
            (REVIEW_ROOT / "frozen_approval/source/source_manifest.json").resolve()
        ),
        "authoritative_source_manifest_sha256": sha256_file(
            REVIEW_ROOT / "frozen_approval/source/source_manifest.json"
        ),
        "excluded_original_episodes": excluded,
        "replacement_episodes": list(NEW_SOURCE_NAMES),
        "raw_recordings_preserved": True,
    }
    atomic_json(FINAL_ROOT / "replaced_hard_failures.json", result)
    return result


def source_image_object_estimate(source_name: str) -> dict[str, Any]:
    """Run the already-established common cam_high planar diagnostic."""
    from tools import calibrate_doll_handoff_scene_from_cam_high as calibration

    scene_calibration = load_json(
        REPOSITORY / "outputs/doll_handoff_retargeting/scene_recalibration.json"
    )
    detector = scene_calibration["detectors"]
    expected_detector = {
        "doll": {
            "hsv_lower_opencv": calibration.DOLL_HSV_LOWER.tolist(),
            "hsv_upper_opencv": calibration.DOLL_HSV_UPPER.tolist(),
            "roi_xyxy_px": list(calibration.DOLL_ROI_XYXY_PX),
        },
        "bin": {
            "reference_episode": calibration.BIN_REFERENCE_EPISODE,
            "reference_frame": 0,
            "reference_opening_polygon_px": calibration.BIN_REFERENCE_OPENING_POLYGON_PX.tolist(),
            "template_bounds_xyxy_px": list(calibration.BIN_TEMPLATE_BOUNDS_PX),
            "search_bounds_xyxy_px": list(calibration.BIN_SEARCH_BOUNDS_PX),
            "acceptance_max_sqdiff_normalized": 0.03,
        },
    }
    actual_detector = {
        "doll": {key: detector["doll"][key] for key in expected_detector["doll"]},
        "bin": {key: detector["bin"][key] for key in expected_detector["bin"]},
    }
    if not canonical_equal(expected_detector, actual_detector):
        raise RuntimeError("established source-image object detector definition drifted")
    homography = np.asarray(
        scene_calibration["metric_reference"]["image_to_task_homography"],
        dtype=np.float64,
    )
    reference_path = (
        RAW
        / calibration.BIN_REFERENCE_EPISODE
        / calibration.IMAGE_RELATIVE_TEMPLATE.format(frame=0)
    )
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"cannot read bin template source {reference_path}")
    tx0, ty0, tx1, ty1 = calibration.BIN_TEMPLATE_BOUNDS_PX
    template = reference[ty0:ty1, tx0:tx1]
    template_mask = np.zeros(template.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(
        template_mask,
        np.rint(
            calibration.BIN_REFERENCE_OPENING_POLYGON_PX - [tx0, ty0]
        ).astype(np.int32),
        255,
    )
    observations: list[dict[str, Any]] = []
    doll_xy: list[np.ndarray] = []
    bin_xy: list[np.ndarray] = []
    for frame in calibration.FRAME_INDICES:
        path = RAW / source_name / calibration.IMAGE_RELATIVE_TEMPLATE.format(frame=frame)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read source-image calibration frame {path}")
        doll_px, doll_pixels = calibration._doll_center_px(image)
        bin_px, bin_polygon, score = calibration._bin_match(
            image, template, template_mask
        )
        bin_method = "historical_masked_translation_template"
        fallback: dict[str, Any] | None = None
        if score > float(detector["bin"]["acceptance_max_sqdiff_normalized"]):
            # The replacement collection uses a larger white bin, so the frozen
            # translation-only appearance template is correctly rejected.  The
            # fallback measures the visible opening directly and is diagnostic
            # only: one common white-wall/open-hole rule is used for both sources,
            # and its result is never supplied to retargeting or the resolver.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            wall = cv2.inRange(
                hsv,
                np.asarray([0, 0, 210], dtype=np.uint8),
                np.asarray([179, 90, 255], dtype=np.uint8),
            )
            sx0, sy0, sx1, sy1 = calibration.BIN_SEARCH_BOUNDS_PX
            keep = np.zeros_like(wall)
            keep[sy0:sy1, sx0:sx1] = 255
            wall = cv2.bitwise_and(wall, keep)
            wall = cv2.morphologyEx(
                wall,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8),
            )
            contours, hierarchy = cv2.findContours(
                wall, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            candidates: list[tuple[float, int]] = []
            if hierarchy is not None:
                for index, contour in enumerate(contours):
                    if int(hierarchy[0, index, 3]) >= 0:
                        candidates.append((float(cv2.contourArea(contour)), index))
            if not candidates:
                raise RuntimeError(
                    f"no visible white-bin opening contour for {source_name} frame {frame}; "
                    f"historical template score={score}"
                )
            area, index = max(candidates)
            contour = contours[index]
            if area < 1000.0:
                raise RuntimeError(
                    f"white-bin opening contour too small for {source_name} frame {frame}: {area} px2"
                )
            moments = cv2.moments(contour)
            if moments["m00"] <= 0.0:
                raise RuntimeError("degenerate white-bin opening contour")
            bin_px = np.asarray(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )
            rectangle = cv2.minAreaRect(contour)
            bin_polygon = cv2.boxPoints(rectangle).astype(np.float64)
            bin_method = "common_white_wall_enclosed_opening_contour"
            fallback = {
                "reason": "historical_translation_only_template_rejected_larger_replacement_bin",
                "historical_template_score": score,
                "common_hsv_lower": [0, 0, 210],
                "common_hsv_upper": [179, 90, 255],
                "common_search_bounds_xyxy_px": list(
                    calibration.BIN_SEARCH_BOUNDS_PX
                ),
                "common_morphology": "5x5 close",
                "selected_enclosed_opening_area_px2": area,
                "selection": "largest enclosed contour; identical rule for both new sources",
                "trajectory_feedback": False,
                "scene_mutation": False,
            }
        current_doll = calibration._transform(doll_px, homography)[0]
        current_bin = calibration._transform(bin_px, homography)[0]
        doll_xy.append(current_doll)
        bin_xy.append(current_bin)
        observations.append(
            {
                "frame": frame,
                "image": str(path.resolve()),
                "image_sha256": sha256_file(path),
                "doll_center_px": doll_px.tolist(),
                "doll_mask_pixels": doll_pixels,
                "doll_center_task_xy_m": current_doll.tolist(),
                "bin_opening_center_px": bin_px.tolist(),
                "bin_opening_polygon_px": bin_polygon.tolist(),
                "bin_template_sqdiff_normalized": score,
                "bin_estimate_method": bin_method,
                "bin_estimate_fallback": fallback,
                "bin_center_task_xy_m": current_bin.tolist(),
            }
        )
    return {
        "method": "historical_template_then_common_visible_opening_diagnostic_v1",
        "bin_estimate_note": (
            "The historical translation-only appearance template is attempted first. "
            "If it rejects the visibly larger replacement bin, a single common "
            "white-wall/enclosed-opening contour measurement is used for both new "
            "sources. This diagnostic never changes scene or trajectory values."
        ),
        "used_to_modify_trajectory": False,
        "used_as_hard_dataset_gate": False,
        "detector_definition": actual_detector,
        "reference_image": str(reference_path.resolve()),
        "reference_image_sha256": sha256_file(reference_path),
        "doll_initial_center_task_xy_m": np.median(
            np.asarray(doll_xy), axis=0
        ).tolist(),
        "bin_center_task_xy_m": np.median(np.asarray(bin_xy), axis=0).tolist(),
        "observations": observations,
    }


def new_source_episode(record: Mapping[str, Any], episode_index: int):
    from tools.doll_handoff_retargeting.source import (
        SourceEpisode,
        SourceRecord,
        fixed_list_numpy as source_fixed_list_numpy,
        scalar_numpy,
    )

    root = Path(record["source_root"])
    parquet = Path(record["parquet_path"])
    source_record = SourceRecord(
        episode_index=episode_index,
        stable_episode_id=NEW_STABLE_IDS[record["source_name"]],
        source_name=record["source_name"],
        root=root,
        parquet=parquet,
        frame_count=int(record["frame_count"]),
        fps=float(record["fps"]),
        duration_sec=float(record["duration_s"]),
        camera_keys=tuple(record["camera_assets"]),
        image_directories={
            key: Path(value["image_directory"])
            for key, value in record["camera_assets"].items()
        },
        valid=True,
        problems=(),
    )
    table = pq.read_table(
        parquet,
        columns=[
            "action",
            "observation.state",
            "timestamp",
            "frame_index",
            "task_index",
        ],
    )
    episode = SourceEpisode(
        record=source_record,
        action=source_fixed_list_numpy(table["action"], 14).astype(np.float64),
        state=source_fixed_list_numpy(table["observation.state"], 14).astype(np.float64),
        timestamps=scalar_numpy(table["timestamp"], np.float64),
        frame_index=scalar_numpy(table["frame_index"], np.int64),
        task_index=scalar_numpy(table["task_index"], np.int64),
    )
    return source_record, episode


def runtime_frozen_checks(pipeline: Any) -> dict[str, bool]:
    frozen_common = load_json(REVIEW_ROOT / "frozen_approval/config/common_config.json")
    frozen_proposed = load_json(REVIEW_ROOT / "frozen_approval/config/proposed_config.json")
    frozen_tool = load_json(REVIEW_ROOT / "frozen_approval/config/tool_frame_report.json")
    frozen_events = load_json(
        REVIEW_ROOT / "frozen_approval/config/event_detector_config.json"
    )
    checks = {
        "implementation": pipeline.implementation_sha256
        == load_json(APPROVAL_FREEZE)["implementation_sha256"],
        "event_detector": canonical_equal(
            pipeline.event_auditor.detector_config, frozen_events
        ),
        "source_to_target_axis_alignment": canonical_equal(
            pipeline.alignment, frozen_tool["source_to_target_axis_alignment"]
        ),
        "dex3_joint_names": canonical_equal(
            pipeline.primitives["joint_names"], frozen_tool["g1"]["joint_names"]
        ),
        "dex3_states": canonical_equal(
            pipeline.primitives["states"], frozen_tool["g1"]["canonical_states"]
        ),
        "wrist_to_grasp_frame": canonical_equal(
            pipeline.primitives["wrist_to_grasp_frame"],
            {
                side: frozen_tool["g1"][f"{side}_wrist_to_grasp_frame"]
                for side in ("left", "right")
            },
        ),
        "proposed_semantics": canonical_equal(
            pipeline.representation.proposed_semantics_report(),
            frozen_proposed["resolved"]["interaction_and_ownership_semantics"],
        ),
        "natural_arm_resolver": canonical_equal(
            pipeline.solver.natural_reference_report,
            frozen_common["resolved"]["natural_arm_redundancy"],
        ),
        "scene": sha256_file(Path(pipeline.common["scene_config"]))
        == load_json(APPROVAL_FREEZE)["approved_resolved_file_sha256"]["scene_layout"],
        "handoff_residual": float(
            pipeline.proposed_template["handoff_cartesian_residual"]["active_offset_m"]
        )
        == 0.0,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"fresh frozen-pipeline realization mismatch: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    return checks


def event_payload(event: Any) -> dict[str, Any]:
    return {
        "episode_index": event.episode_index,
        "source_name": event.source_name,
        "frames": event.frames,
        "transitions": event.transitions,
        "handoff_window": list(event.handoff_window),
        "ownership_sequence": list(dict.fromkeys(event.ownership_labels.tolist())),
        "left_phase_sequence": list(
            dict.fromkeys(event.semantic_labels["left"].tolist())
        ),
        "right_phase_sequence": list(
            dict.fromkeys(event.semantic_labels["right"].tolist())
        ),
        "anomalies": list(event.anomalies),
        "source_semantic_valid": event.source_semantic_valid,
        "minimum_inter_hand_distance_m": float(np.min(event.inter_hand_distance_m)),
    }


def initial_classification(
    resolver: Any, values: Mapping[str, np.ndarray], metric: Mapping[str, Any]
) -> str:
    timestamp = np.asarray(values["timestamp"], dtype=np.float64)
    fps = float(1.0 / np.median(np.diff(timestamp)))
    source, _ = resolver._source_targets(values)
    achieved_model, _ = resolver._pose_arrays(values["g1_arm_qpos"].astype(np.float64))
    achieved = {
        side: resolver.g1.model_to_world_position(achieved_model[side])
        for side in ("left", "right")
    }
    error = np.maximum(
        *[
            np.linalg.norm(achieved[side] - source[side], axis=1)
            for side in ("left", "right")
        ]
    )
    physical_frames = np.flatnonzero(error > resolver.physical_tolerance)
    segments: list[tuple[int, int]] = []
    if len(physical_frames):
        start = previous = int(physical_frames[0])
        for raw in physical_frames[1:]:
            frame = int(raw)
            if frame != previous + 1:
                segments.append((start, previous))
                start = frame
            previous = frame
        segments.append((start, previous))
    hard_ik = bool(
        np.max(error, initial=0.0) >= 2.0 * resolver.physical_tolerance
        or max(((end - start + 1) / fps for start, end in segments), default=0.0)
        >= 0.25
    )
    collision = resolver._collision_metrics(
        values["g1_arm_qpos"].astype(np.float64),
        values["left_dex3_qpos"].astype(np.float64),
        values["right_dex3_qpos"].astype(np.float64),
        fps,
    )
    hard = bool(
        not metric["finite"]
        or int(metric["nan_inf_count"])
        or int(metric["joint_limit_violation_count"])
        or int(metric["branch_discontinuity_count"])
        or hard_ik
        or int(collision["hard_collision_frame_count"])
        or not metric["semantics"]["right_grasp_before_left_release"]
        or not metric["semantics"]["ownership_transition_validity"]
    )
    if hard:
        return "HARD_FAIL"
    warning = bool(
        np.any(error > resolver.strict_tolerance)
        or int(collision["contact_frame_count"])
        or float(metric["maximum_joint_acceleration_rad_s2"])
        > float(resolver.acceptance["maximum_acceleration_rad_s2"])
    )
    return "USABLE_WITH_WARNING" if warning else "CLEAN_PASS"


def render_new_episode(
    resolver: Any,
    source_name: str,
    episode_index: int,
    stable_id: str,
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    result: Any,
    row: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.doll_handoff_feasibility.render_review import (
        BeforeAfterRenderer,
        _collision_arrays,
        _overlay,
    )
    from tools.doll_handoff_retargeting.render import (
        _decoded_video,
        _handoff_label,
        _writer,
    )

    renderer = BeforeAfterRenderer(resolver)
    output: dict[str, Any] = {}
    try:
        timestamp = before["timestamp"].astype(np.float64)
        fps = float(1.0 / np.median(np.diff(timestamp)))
        before_model, _ = resolver._pose_arrays(before["g1_arm_qpos"].astype(np.float64))
        before_world = {
            side: resolver.g1.model_to_world_position(before_model[side])
            for side in ("left", "right")
        }
        before_error = np.maximum(
            *[
                np.linalg.norm(before_world[side] - result.source_position_world[side], axis=1)
                for side in ("left", "right")
            ]
        )
        after_error = np.maximum(
            *[
                np.linalg.norm(
                    result.achieved_position_world[side]
                    - result.realized_position_world[side],
                    axis=1,
                )
                for side in ("left", "right")
            ]
        )
        before_collision = resolver._collision_metrics(
            result.q_before, result.left_hand, result.right_hand, fps
        )
        after_collision = resolver._collision_metrics(
            result.q_after, result.left_hand, result.right_hand, fps
        )
        before_state, before_depth = _collision_arrays(before_collision, len(timestamp))
        after_state, after_depth = _collision_arrays(after_collision, len(timestamp))
        images = sorted(
            (
                RAW
                / source_name
                / "images/observation.images.cam_high/episode_000000"
            ).glob("frame_*.png")
        )
        if len(images) != len(timestamp):
            raise RuntimeError(f"source images incomplete during render: {source_name}")
        sampled = list(range(0, len(timestamp), renderer.stride))
        if sampled[-1] != len(timestamp) - 1:
            sampled.append(len(timestamp) - 1)
        render_root = FINAL_ROOT / "new_episode_conversion/renders"
        for camera in ("overview", "top", "side"):
            path = render_root / f"{stable_id}_{camera}.mp4"
            writer = _writer(path, renderer.fps, (3 * renderer.width, renderer.height))
            try:
                for frame in sampled:
                    source = cv2.imread(str(images[frame]), cv2.IMREAD_COLOR)
                    source = cv2.resize(source, (renderer.width, renderer.height))
                    label = _handoff_label(event, frame, renderer.stride) or "SOURCE MOTION"
                    source = _overlay(
                        source,
                        "SOURCE ALOHA / CAM_HIGH",
                        episode_index,
                        frame,
                        float(timestamp[frame]),
                        [label, source_name, "30 Hz synchronized"],
                        "PASS",
                    )
                    phases = (
                        f"L={before['left_hand_phase'][frame]} "
                        f"R={before['right_hand_phase'][frame]}"
                    )
                    ownership = str(before["ownership_state"][frame])
                    before_panel = renderer.robot_panel(
                        before,
                        before_world,
                        episode_index,
                        frame,
                        float(timestamp[frame]),
                        camera,
                        "FROZEN PROPOSED B / BEFORE",
                        str(row["before_classification"]),
                        [
                            f"{phases} | owner={ownership}",
                            f"source residual={before_error[frame]*1000:.2f} mm",
                            f"collision={before_state[frame]} depth={before_depth[frame]*1000:.2f} mm",
                            "handoff Cartesian residual=0.0 mm",
                        ],
                    )
                    after_panel = renderer.robot_panel(
                        after,
                        result.achieved_position_world,
                        episode_index,
                        frame,
                        float(timestamp[frame]),
                        camera,
                        "FROZEN B + GENERIC FEASIBILITY / AFTER",
                        str(row["after_classification"]),
                        [
                            f"{phases} | owner={ownership}",
                            f"realized residual={after_error[frame]*1000:.2f} mm",
                            f"collision={after_state[frame]} depth={after_depth[frame]*1000:.2f} mm",
                            f"projection={np.max(result.projection_translation_m[frame])*1000:.2f} mm",
                        ],
                    )
                    writer.write(np.hstack((source, before_panel, after_panel)))
            finally:
                writer.release()
            decoded = _decoded_video(path)
            if decoded[0] != len(sampled) or abs(decoded[1] - renderer.fps) > 0.1:
                raise RuntimeError(f"render readback failed: {path}: {decoded}")
            output[camera] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "decoded_frames": decoded[0],
                "fps": decoded[1],
                "width": decoded[2],
                "height": decoded[3],
            }
    finally:
        renderer.close()
    return output


def convert_new_sources() -> dict[str, Any]:
    provenance = verify_frozen_provenance()
    audit_path = FINAL_ROOT / "new_source_audit/validation.json"
    if not audit_path.is_file() or load_json(audit_path).get("status") != "PASS":
        raise RuntimeError("new-source audit has not passed")
    replacements = replacement_provenance()
    from tools.doll_handoff_retargeting.pipeline import DollHandoffPipeline

    runtime_root = FINAL_ROOT / "new_episode_conversion/frozen_proposed_b_runtime"
    pipeline = DollHandoffPipeline(output_root=runtime_root)
    runtime_checks = runtime_frozen_checks(pipeline)
    pipeline.config_paths["common"] = (
        REVIEW_ROOT / "frozen_approval/config/common_config.json"
    )
    pipeline.config_paths["proposed"] = (
        REVIEW_ROOT / "frozen_approval/config/proposed_config.json"
    )
    manifest = load_json(FINAL_ROOT / "new_source_audit/new_episode_manifest.json")
    source_records = {row["source_name"]: row for row in manifest["episodes"]}
    before_paths: dict[int, Path] = {}
    before_metrics: dict[int, dict[str, Any]] = {}
    event_records: dict[int, dict[str, Any]] = {}
    object_estimates: dict[int, dict[str, Any]] = {}
    conversion_records: list[dict[str, Any]] = []
    for offset, source_name in enumerate(NEW_SOURCE_NAMES):
        episode_index = 50 + offset
        record, episode = new_source_episode(source_records[source_name], episode_index)
        if len(pipeline.sources.records) != episode_index:
            raise RuntimeError("new episode insertion index is not append-only")
        pipeline.sources.records.append(record)
        pipeline.sources._cache[episode_index] = episode
        fk = pipeline.aloha.fk(episode.state)
        event = pipeline.event_auditor.detect_episode(episode, fk)
        pipeline.event_auditor.fk_cache[episode_index] = fk
        pipeline.event_auditor.events[episode_index] = event
        result = pipeline.convert("proposed", episode_index)
        paths = pipeline.export(result)
        before_path = paths["trajectory"]
        before_paths[episode_index] = before_path
        before_metrics[episode_index] = result.metrics
        event_records[episode_index] = event_payload(event)
        object_estimates[episode_index] = source_image_object_estimate(source_name)
        conversion_records.append(
            {
                "episode_index": episode_index,
                "stable_episode_id": record.stable_episode_id,
                "source_name": source_name,
                "frame_count": record.frame_count,
                "fps": record.fps,
                "source_semantic_valid": event.source_semantic_valid,
                "source_anomalies": list(event.anomalies),
                "initial_converter_status": result.validation["status"],
                "initial_trajectory_path": str(before_path.resolve()),
                "initial_trajectory_sha256": sha256_file(before_path),
                "initial_metrics_path": str(paths["metrics"].resolve()),
                "initial_metrics_sha256": sha256_file(paths["metrics"]),
                "initial_validation_path": str(paths["validation"].resolve()),
                "initial_validation_sha256": sha256_file(paths["validation"]),
                "cartesian_target_sha256": result.targets["cartesian_target_sha256"],
            }
        )
        print(
            f"[FROZEN PROPOSED-B] {source_name} frames={record.frame_count} "
            f"status={result.validation['status']}",
            flush=True,
        )
    atomic_json(
        FINAL_ROOT / "new_episode_conversion/events.json",
        {str(key): value for key, value in event_records.items()},
    )
    atomic_json(
        FINAL_ROOT / "new_episode_conversion/source_image_object_estimates.json",
        {str(key): value for key, value in object_estimates.items()},
    )

    from tools.doll_handoff_feasibility.solver import GenericG1FeasibilityResolver
    import tools.doll_handoff_feasibility.solver as solver_module

    feasibility_root = FINAL_ROOT / "new_episode_conversion/generic_feasibility"
    resolver = GenericG1FeasibilityResolver(output_root=feasibility_root)
    before_values = {
        episode: {
            name: np.asarray(payload[name])
            for name in payload.files
        }
        for episode, path in before_paths.items()
        for payload in [np.load(path, allow_pickle=False)]
    }
    stable_ids = {
        50 + offset: NEW_STABLE_IDS[source_name]
        for offset, source_name in enumerate(NEW_SOURCE_NAMES)
    }
    original_load = solver_module.load_trajectory
    original_stable = solver_module.stable_episode_id
    results: dict[int, Any] = {}
    try:
        solver_module.load_trajectory = lambda episode: before_values[int(episode)]
        solver_module.stable_episode_id = lambda episode: stable_ids[int(episode)]
        for episode in sorted(before_values):
            print(f"[GENERIC FEASIBILITY] solving {stable_ids[episode]}", flush=True)
            results[episode] = resolver.solve_episode(episode, export=True)
    finally:
        solver_module.load_trajectory = original_load
        solver_module.stable_episode_id = original_stable

    gate_rows: list[dict[str, Any]] = []
    for episode in sorted(results):
        source_name = NEW_SOURCE_NAMES[episode - 50]
        stable_id = stable_ids[episode]
        after_path = feasibility_root / "after/trajectories" / f"{stable_id}.npz"
        with np.load(after_path, allow_pickle=False) as payload:
            after_values = {name: np.asarray(payload[name]) for name in payload.files}
        event = event_records[episode]
        release_frame = event["frames"].get("RIGHT_FINAL_RELEASE")
        bin_xy = np.asarray(
            object_estimates[episode]["bin_center_task_xy_m"], dtype=np.float64
        )
        release = (
            after_values["achieved_right_physical_grasp_frame_position_world"][
                int(release_frame)
            ].astype(np.float64)
            if release_frame is not None
            else np.full(3, np.nan)
        )
        opening = np.asarray(resolver.scene["bin"]["opening_dimensions_xy_m"])
        source_bin_inside = bool(
            np.isfinite(release).all()
            and np.all(np.abs(release[:2] - bin_xy) <= 0.5 * opening)
        )
        gate_rows.append(
            {
                "episode_index": episode,
                "classification": initial_classification(
                    resolver, before_values[episode], before_metrics[episode]
                ),
                "handoff_order_valid": bool(
                    before_metrics[episode]["semantics"][
                        "right_grasp_before_left_release"
                    ]
                ),
                "ownership_transition_valid": bool(
                    before_metrics[episode]["semantics"][
                        "ownership_transition_validity"
                    ]
                ),
                "release_event_present": release_frame is not None,
                "per_episode_source_bin_release_xy_inside": source_bin_inside,
                "source_image_bin_center_task_xy_m": bin_xy.tolist(),
                "right_release_grasp_frame_world_m": release.tolist(),
                "right_release_horizontal_distance_to_source_bin_m": float(
                    np.linalg.norm(release[:2] - bin_xy)
                ),
            }
        )
    gate_csv = FINAL_ROOT / "new_episode_conversion/new_source_gate_inputs.csv"
    atomic_csv(gate_csv, gate_rows)

    import tools.doll_handoff_feasibility.evaluate as evaluate_module

    eval_load = evaluate_module.load_trajectory
    eval_stable = evaluate_module.stable_episode_id
    eval_gate = evaluate_module.CLASSIFICATION_CSV
    evaluated: list[dict[str, Any]] = []
    try:
        evaluate_module.load_trajectory = lambda episode: before_values[int(episode)]
        evaluate_module.stable_episode_id = lambda episode: stable_ids[int(episode)]
        evaluate_module.CLASSIFICATION_CSV = gate_csv
        for episode in sorted(results):
            evaluated.append(evaluate_module.evaluate_episode(resolver, results[episode]))
    finally:
        evaluate_module.load_trajectory = eval_load
        evaluate_module.stable_episode_id = eval_stable
        evaluate_module.CLASSIFICATION_CSV = eval_gate
    gate_by_episode = {int(row["episode_index"]): row for row in gate_rows}
    for row in evaluated:
        row["source_name"] = NEW_SOURCE_NAMES[int(row["episode_index"]) - 50]
        row["source_image_bin_diagnostic"] = gate_by_episode[int(row["episode_index"])]
        row["finite_trajectory"] = bool(
            all(
                np.isfinite(value).all()
                for value in (
                    results[int(row["episode_index"])].q_after,
                    results[int(row["episode_index"])].left_hand,
                    results[int(row["episode_index"])].right_hand,
                )
            )
        )
        row["usable_realized_ik_success_rate"] = row[
            "after_physical_realized_ik_success_rate"
        ]
        row["whole_hand_grasp_frame_error_m"] = {
            "mean_realized_target": row["after_mean_realized_position_residual_m"],
            "max_realized_target": row["after_max_realized_position_residual_m"],
            "mean_source_target": row["after_mean_source_position_residual_m"],
            "max_source_target": row["after_max_source_position_residual_m"],
        }
        row["bimanual_relation_error_m"] = {
            "mean": row["mean_bimanual_relation_change_m"],
            "max": row["max_bimanual_relation_change_m"],
            "dual_contact_mean": row[
                "mean_dual_contact_bimanual_relation_change_m"
            ],
            "dual_contact_max": row[
                "max_dual_contact_bimanual_relation_change_m"
            ],
        }
    atomic_json(
        FINAL_ROOT / "new_episode_conversion/new_episode_classification.json",
        evaluated,
    )
    atomic_csv(
        FINAL_ROOT / "new_episode_conversion/new_episode_classification.csv",
        [
            {
                key: json.dumps(value, separators=(",", ":"), sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in evaluated
        ],
    )
    renders: dict[str, Any] = {}
    for row in evaluated:
        episode = int(row["episode_index"])
        stable_id = stable_ids[episode]
        after_path = feasibility_root / "after/trajectories" / f"{stable_id}.npz"
        with np.load(after_path, allow_pickle=False) as payload:
            after = {name: np.asarray(payload[name]) for name in payload.files}
        renders[stable_id] = render_new_episode(
            resolver,
            NEW_SOURCE_NAMES[episode - 50],
            episode,
            stable_id,
            before_values[episode],
            after,
            results[episode],
            row,
            event_records[episode],
        )
    atomic_json(FINAL_ROOT / "new_episode_conversion/render_manifest.json", renders)
    for record, row in zip(conversion_records, evaluated, strict=True):
        episode = int(row["episode_index"])
        stable_id = stable_ids[episode]
        after_path = feasibility_root / "after/trajectories" / f"{stable_id}.npz"
        record.update(
            {
                "resolved_trajectory_path": str(after_path.resolve()),
                "resolved_trajectory_sha256": sha256_file(after_path),
                "solver_metrics_path": str(
                    (
                        feasibility_root
                        / "after/metrics"
                        / f"{stable_id}.solver.json"
                    ).resolve()
                ),
                "classification": row["after_classification"],
                "classification_reasons": row["after_classification_reasons"],
                "render_views": renders[stable_id],
            }
        )
    summary = {
        "schema_version": "doll_handoff_dataset_b_new_conversion_v1",
        "status": (
            "PASS"
            if all(row["after_classification"] != "HARD_FAIL" for row in evaluated)
            else "HARD_FAIL_STOP"
        ),
        "frozen_provenance": provenance["authoritative"],
        "runtime_frozen_checks": runtime_checks,
        "replacement_provenance": str(
            (FINAL_ROOT / "replaced_hard_failures.json").resolve()
        ),
        "replacement_provenance_sha256": sha256_file(
            FINAL_ROOT / "replaced_hard_failures.json"
        ),
        "episodes": conversion_records,
        "classification_rows": evaluated,
        "replacements": replacements["replacement_episodes"],
        "converter_tuning_performed": False,
        "parameter_search_performed": False,
        "episode_specific_parameters": 0,
        "handoff_cartesian_residual_m": 0.0,
    }
    atomic_json(FINAL_ROOT / "new_episode_conversion/summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if summary["status"] != "PASS":
        raise RuntimeError("one or both new episodes are HARD_FAIL; stop before packaging")
    return summary


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _parsed_json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return []
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return value


def _as_float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float("nan")
    return float(value)


def _as_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return int(value)
    return int(float(value))


def _as_bool(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _distribution(rows: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([_as_float(row, key) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite aggregate metric {key}")
    return {
        "per_episode": values.tolist(),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def _trajectory_policy_arrays(path: Path) -> dict[str, Any]:
    from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES

    with np.load(path, allow_pickle=False) as payload:
        required = {
            "timestamp",
            "source_frame_index",
            "g1_arm_joint_names",
            "g1_arm_qpos",
            "left_dex3_joint_names",
            "left_dex3_qpos",
            "right_dex3_joint_names",
            "right_dex3_qpos",
            "replay_joint_names",
            "replay_named_joint_qpos",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"{path}: missing trajectory arrays {missing}")
        timestamps = np.asarray(payload["timestamp"], dtype=np.float64)
        source_frames = np.asarray(payload["source_frame_index"], dtype=np.int64)
        arm = np.asarray(payload["g1_arm_qpos"], dtype=np.float32)
        left = np.asarray(payload["left_dex3_qpos"], dtype=np.float32)
        right = np.asarray(payload["right_dex3_qpos"], dtype=np.float32)
        names = tuple(
            str(value)
            for key in (
                "g1_arm_joint_names",
                "left_dex3_joint_names",
                "right_dex3_joint_names",
            )
            for value in np.asarray(payload[key]).tolist()
        )
        replay_names = tuple(
            str(value) for value in np.asarray(payload["replay_joint_names"]).tolist()
        )
        replay = np.asarray(payload["replay_named_joint_qpos"], dtype=np.float32)
        component_arrays = {
            key: np.asarray(payload[key])
            for key in (
                "g1_arm_qpos",
                "left_dex3_qpos",
                "right_dex3_qpos",
            )
        }
        target_arrays = {
            key: np.asarray(payload[key])
            for key in (
                "target_left_wrist_position_model",
                "target_right_wrist_position_model",
                "target_left_wrist_rotation_model",
                "target_right_wrist_rotation_model",
                "target_left_interaction_frame_position_world",
                "target_right_interaction_frame_position_world",
                "target_left_interaction_frame_position_task",
                "target_right_interaction_frame_position_task",
            )
        }
    direct = np.concatenate((arm, left, right), axis=1)
    if names != replay_names or not np.array_equal(direct, replay):
        raise RuntimeError(f"{path}: named replay array is not the component action")
    if len(names) != 28 or len(set(names)) != 28:
        raise RuntimeError(f"{path}: trajectory does not contain 28 unique controlled joints")
    canonical_names = tuple(CANONICAL_JOINT_NAMES)
    if set(names) != set(canonical_names):
        raise RuntimeError(f"{path}: controlled joint-name set differs from G1 policy contract")
    reorder = tuple(names.index(name) for name in canonical_names)
    action = np.ascontiguousarray(direct[:, reorder], dtype=np.float32)
    frame_count = action.shape[0]
    if timestamps.shape != (frame_count,) or source_frames.shape != (frame_count,):
        raise RuntimeError(f"{path}: timestamp/source frame length mismatch")
    if not np.array_equal(source_frames, np.arange(frame_count, dtype=np.int64)):
        raise RuntimeError(f"{path}: source frame indices are not contiguous from zero")
    if frame_count > 1:
        fps = float(1.0 / np.median(np.diff(timestamps)))
    else:
        fps = 30.0
    if not np.isfinite(action).all() or not np.isfinite(timestamps).all():
        raise RuntimeError(f"{path}: non-finite policy action/timestamp")
    if not np.all(np.diff(timestamps) > 0.0):
        raise RuntimeError(f"{path}: timestamps are not strictly monotonic")
    if not np.allclose(timestamps, np.arange(frame_count) / 30.0, atol=2e-6, rtol=0.0):
        raise RuntimeError(f"{path}: timestamp convention differs from frame_index/30")
    return {
        "action": action,
        "timestamps": timestamps,
        "source_frames": source_frames,
        "input_joint_names": names,
        "canonical_joint_names": canonical_names,
        "canonical_reorder_indices": reorder,
        "fps": fps,
        "component_action_sha256": stable_json_sha256(
            {key: frozen_array_sha256(value) for key, value in component_arrays.items()}
        ),
        "canonical_policy_action_sha256": frozen_array_sha256(action),
        "timestamp_sha256": frozen_array_sha256(timestamps),
        "cartesian_target_array_set_sha256": stable_json_sha256(
            {key: frozen_array_sha256(value) for key, value in target_arrays.items()}
        ),
    }


def assemble_final_actions() -> dict[str, Any]:
    """Freeze the final 48 historical + 2 replacement trajectory set."""
    from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES, JOINT_SPECS

    provenance = verify_frozen_provenance()
    conversion_path = FINAL_ROOT / "new_episode_conversion/summary.json"
    conversion = load_json(conversion_path)
    if conversion.get("status") != "PASS":
        raise RuntimeError("new-episode frozen conversion gate has not passed")
    if any(
        row.get("after_classification") == "HARD_FAIL"
        for row in conversion["classification_rows"]
    ):
        raise RuntimeError("new conversion summary contains a HARD_FAIL")
    replacements = replacement_provenance()
    source_freeze_path = REVIEW_ROOT / "frozen_approval/source/source_manifest.json"
    original_sources = load_json(source_freeze_path)
    resolver_freeze = load_json(RESOLVER_FREEZE)
    frozen_by_episode = {
        int(row["episode_index"]): row
        for row in resolver_freeze["integrity"]["episodes"]
    }
    old_csv_path = RESOLVER_ROOT / "full50/per_episode.csv"
    with old_csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        old_fields = list(reader.fieldnames or [])
        old_rows = {int(row["episode_index"]): row for row in reader}
    new_audit = load_json(FINAL_ROOT / "new_source_audit/new_episode_manifest.json")
    new_sources = {row["source_name"]: row for row in new_audit["episodes"]}
    new_conversion = {row["source_name"]: row for row in conversion["episodes"]}
    new_metrics = {
        row["source_name"]: row for row in conversion["classification_rows"]
    }
    replacement_at = {
        13: "GoPark_20260823_135848",
        36: "GoPark_20260823_140035",
    }
    actions_root = FINAL_ROOT / "retargeted_actions"
    trajectories_root = actions_root / "trajectories"
    trajectories_root.mkdir(parents=True, exist_ok=True)
    final_records: list[dict[str, Any]] = []
    final_metric_rows: list[dict[str, Any]] = []
    file_set: dict[str, str] = {}
    action_set: dict[str, str] = {}
    component_set: dict[str, str] = {}
    target_set: dict[str, str] = {}
    timestamp_set: dict[str, str] = {}
    for final_index in range(50):
        if final_index in replacement_at:
            source_name = replacement_at[final_index]
            source = new_sources[source_name]
            converted = new_conversion[source_name]
            metric = dict(new_metrics[source_name])
            source_kind = "2026-08-23 replacement"
            original_episode_index = None
            stable_id = str(converted["stable_episode_id"])
            source_trajectory = Path(converted["resolved_trajectory_path"])
            expected_trajectory_sha = str(converted["resolved_trajectory_sha256"])
        else:
            source = original_sources["records"][final_index]
            source_name = str(source["source_name"])
            metric = dict(old_rows[final_index])
            source_kind = "frozen 2026-08-20 accepted source"
            original_episode_index = final_index
            stable_id = str(frozen_by_episode[final_index]["stable_episode_id"])
            source_trajectory = (
                RESOLVER_ROOT / "after/trajectories" / f"{stable_id}.npz"
            )
            expected_trajectory_sha = str(
                frozen_by_episode[final_index]["after_trajectory_sha256"]
            )
            if metric["after_classification"] == "HARD_FAIL":
                raise RuntimeError(
                    f"unreplaced historical HARD_FAIL at final episode {final_index}"
                )
        if not source_trajectory.is_file():
            raise FileNotFoundError(source_trajectory)
        actual_sha = sha256_file(source_trajectory)
        if actual_sha != expected_trajectory_sha:
            raise RuntimeError(f"trajectory provenance mismatch: {source_trajectory}")
        arrays = _trajectory_policy_arrays(source_trajectory)
        if int(source["frame_count"]) != int(arrays["action"].shape[0]):
            raise RuntimeError(f"source/trajectory frame mismatch for {source_name}")
        if abs(float(source["fps"]) - 30.0) > 1e-12:
            raise RuntimeError(f"source FPS mismatch for {source_name}")
        if final_index not in replacement_at:
            expected_action_hash = frozen_by_episode[final_index][
                "after_action_array_set_sha256"
            ]
            if arrays["component_action_sha256"] != expected_action_hash:
                raise RuntimeError(f"frozen action-array hash mismatch at ep{final_index:03d}")
            if arrays["cartesian_target_array_set_sha256"] != frozen_by_episode[
                final_index
            ]["target_array_set_sha256"]:
                raise RuntimeError(f"frozen target-array hash mismatch at ep{final_index:03d}")
        destination = trajectories_root / f"episode_{final_index:06d}.npz"
        if destination.exists():
            if sha256_file(destination) != actual_sha:
                raise RuntimeError(f"refusing to replace differing frozen output {destination}")
        else:
            shutil.copy2(source_trajectory, destination)
        if sha256_file(destination) != actual_sha:
            raise RuntimeError(f"copied trajectory hash mismatch: {destination}")
        reasons = _parsed_json_cell(metric.get("after_classification_reasons"))
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        warning_reasons = (
            reasons if metric["after_classification"] == "USABLE_WITH_WARNING" else []
        )
        key = f"episode_{final_index:06d}"
        file_set[key] = actual_sha
        action_set[key] = arrays["canonical_policy_action_sha256"]
        component_set[key] = arrays["component_action_sha256"]
        target_set[key] = arrays["cartesian_target_array_set_sha256"]
        timestamp_set[key] = arrays["timestamp_sha256"]
        final_records.append(
            {
                "final_dataset_index": final_index,
                "source_kind": source_kind,
                "raw_directory": source_name,
                "raw_directory_path": str(Path(source["source_root"]).resolve()),
                "original_date": source_name.split("_")[1][:8],
                "original_episode_index": original_episode_index,
                "stable_episode_id": stable_id,
                "source_frame_count": int(source["frame_count"]),
                "source_fps": float(source["fps"]),
                "source_parquet_path": str(Path(source["parquet_path"]).resolve()),
                "source_parquet_sha256": str(source["parquet_sha256"]),
                "retargeted_trajectory_path": str(destination.resolve()),
                "retargeted_trajectory_sha256": actual_sha,
                "canonical_policy_action_sha256": arrays[
                    "canonical_policy_action_sha256"
                ],
                "classification": str(metric["after_classification"]),
                "warning_reasons": warning_reasons,
            }
        )
        selected_metric = {
            field: metric.get(field, "") for field in old_fields
        }
        selected_metric.update(
            {
                "episode_index": final_index,
                "stable_episode_id": stable_id,
                "frame_count": int(source["frame_count"]),
                "final_dataset_index": final_index,
                "raw_directory": source_name,
                "source_kind": source_kind,
                "original_episode_index": original_episode_index,
            }
        )
        final_metric_rows.append(selected_metric)
    if len(final_records) != 50 or len({row["raw_directory"] for row in final_records}) != 50:
        raise RuntimeError("final source set is not 50 unique recordings")
    status_counts = {
        status: sum(row["classification"] == status for row in final_records)
        for status in ("CLEAN_PASS", "USABLE_WITH_WARNING", "HARD_FAIL")
    }
    if status_counts["HARD_FAIL"] != 0:
        raise RuntimeError(f"final 50 still contains HARD_FAIL: {status_counts}")
    final_source_manifest = {
        "schema_version": "doll_handoff_dataset_b_final_source_manifest_v1",
        "status": "FINAL_50_FROZEN_NO_HARD_FAILURES",
        "task": "DOLL-HANDOFF-TO-BIN",
        "retargeting_method": "interaction_centric_proposed_b",
        "ordering_rule": (
            "Preserve original frozen episode indices 0..49; replace original ep013 "
            "and ep036 in-place in collection order with the two 2026-08-23 recordings."
        ),
        "source_count": 50,
        "classification_counts": status_counts,
        "original_frozen_source_manifest": str(source_freeze_path.resolve()),
        "original_frozen_source_manifest_sha256": sha256_file(source_freeze_path),
        "excluded_source_episodes": replacements["excluded_original_episodes"],
        "exclusion_reason": (
            "persistent target-realization HARD_FAIL under the frozen generic resolver"
        ),
        "replacement_episodes": [
            {
                "final_dataset_index": index,
                "replaces_original_episode_index": index,
                "raw_directory": replacement_at[index],
            }
            for index in sorted(replacement_at)
        ],
        "raw_failed_recordings_deleted": False,
        "episodes": final_records,
    }
    final_source_path = FINAL_ROOT / "final_source_manifest.json"
    atomic_json(final_source_path, final_source_manifest)
    source_manifest_sha = sha256_file(final_source_path)

    csv_fields = [
        "final_dataset_index",
        "raw_directory",
        "source_kind",
        "original_episode_index",
        *old_fields,
    ]
    # episode_index/stable_episode_id/frame_count occur in old_fields and are not duplicated.
    csv_fields = list(dict.fromkeys(csv_fields))
    atomic_csv(
        FINAL_ROOT / "final50/per_episode.csv",
        [
            {field: _json_cell(row.get(field, "")) for field in csv_fields}
            for row in final_metric_rows
        ],
    )
    numeric_sums = {
        key: int(sum(_as_int(row, key) for row in final_metric_rows))
        for key in (
            "joint_limit_violations_after",
            "branch_discontinuities_after",
            "after_hard_collision_frames",
            "after_arm_torso_hard_frames",
            "after_distal_hard_frames",
            "ownership_timing_change",
            "source_interaction_target_change",
            "hands_changed",
        )
    }
    aggregate = {
        "schema_version": "doll_handoff_dataset_b_final50_metrics_v1",
        "status": "PASS",
        "episode_count": 50,
        "total_frames": int(sum(row["source_frame_count"] for row in final_records)),
        "classification_counts": status_counts,
        "classification_warning_reason_counts": {
            reason: sum(reason in row["warning_reasons"] for row in final_records)
            for reason in sorted(
                {reason for row in final_records for reason in row["warning_reasons"]}
            )
        },
        "strict_source_ik_success_rate": _distribution(
            final_metric_rows, "after_strict_source_ik_success_rate"
        ),
        "strict_realized_ik_success_rate": _distribution(
            final_metric_rows, "after_strict_realized_ik_success_rate"
        ),
        "usable_source_ik_success_rate": _distribution(
            final_metric_rows, "after_physical_source_ik_success_rate"
        ),
        "usable_realized_ik_success_rate": _distribution(
            final_metric_rows, "after_physical_realized_ik_success_rate"
        ),
        "realized_target_position_residual_m": {
            "per_episode_mean": _distribution(
                final_metric_rows, "after_mean_realized_position_residual_m"
            ),
            "per_episode_max": _distribution(
                final_metric_rows, "after_max_realized_position_residual_m"
            ),
        },
        "joint_limit_branch_collision_totals": numeric_sums,
        "maximum_velocity_rad_s": _distribution(
            final_metric_rows, "maximum_velocity_after_rad_s"
        ),
        "maximum_acceleration_rad_s2": _distribution(
            final_metric_rows, "maximum_acceleration_after_rad_s2"
        ),
        "bimanual_relation_error_m": {
            "mean": _distribution(final_metric_rows, "mean_bimanual_relation_change_m"),
            "max": _distribution(final_metric_rows, "max_bimanual_relation_change_m"),
            "dual_contact_mean": _distribution(
                final_metric_rows, "mean_dual_contact_bimanual_relation_change_m"
            ),
            "dual_contact_max": _distribution(
                final_metric_rows, "max_dual_contact_bimanual_relation_change_m"
            ),
        },
        "semantic_checks": {
            "handoff_order_all_valid": all(
                _as_bool(row, "handoff_order_valid") for row in final_metric_rows
            ),
            "ownership_transition_all_valid": all(
                _as_bool(row, "ownership_transition_valid") for row in final_metric_rows
            ),
            "release_event_all_present": all(
                _as_bool(row, "release_event_present") for row in final_metric_rows
            ),
            "source_image_bin_release_all_inside": all(
                _as_bool(row, "source_bin_release_inside") for row in final_metric_rows
            ),
            "source_arrays_all_preserved": all(
                _as_bool(row, "source_array_checks_pass") for row in final_metric_rows
            ),
        },
        "final_condition_hard_fail_zero": status_counts["HARD_FAIL"] == 0,
    }
    if numeric_sums["joint_limit_violations_after"] != 0:
        raise RuntimeError("final 50 contains joint-limit violations")
    if numeric_sums["branch_discontinuities_after"] != 0:
        raise RuntimeError("final 50 contains branch discontinuities")
    if numeric_sums["after_hard_collision_frames"] != 0:
        raise RuntimeError("final 50 contains hard self-collision frames")
    if not all(aggregate["semantic_checks"].values()):
        raise RuntimeError(f"final 50 semantic checks failed: {aggregate['semantic_checks']}")
    aggregate_path = FINAL_ROOT / "final50/aggregate_metrics.json"
    atomic_json(aggregate_path, aggregate)
    action_freeze = {
        "schema_version": "doll_handoff_dataset_b_retargeted_action_freeze_v1",
        "status": "FINAL_PROPOSED_B_ACTION_LABELS_FROZEN",
        "source_episode_manifest": str(final_source_path.resolve()),
        "source_episode_manifest_sha256": source_manifest_sha,
        "converter_implementation_sha256": provenance["authoritative"][
            "proposed_b_implementation_sha256"
        ],
        "historical_frozen_cartesian_target_set_sha256": provenance[
            "authoritative"
        ]["cartesian_target_array_set_sha256"],
        "final_cartesian_target_array_set_sha256": stable_json_sha256(target_set),
        "common_natural_arm_solver_sha256": provenance["authoritative"][
            "common_natural_arm_solver_sha256"
        ],
        "generic_feasibility_resolver_sha256": provenance["authoritative"][
            "generic_feasibility_resolver_sha256"
        ],
        "generic_feasibility_resolver_config_sha256": provenance[
            "authoritative"
        ]["generic_feasibility_resolver_config_sha256"],
        "scene_layout_sha256": provenance["authoritative"]["scene_layout_sha256"],
        "frozen_config_sha256": {
            "common": sha256_file(
                REVIEW_ROOT / "frozen_approval/config/common_config.json"
            ),
            "proposed_b": sha256_file(
                REVIEW_ROOT / "frozen_approval/config/proposed_config.json"
            ),
            "tool_frame": sha256_file(
                REVIEW_ROOT / "frozen_approval/config/tool_frame_report.json"
            ),
            "event_detector": sha256_file(
                REVIEW_ROOT / "frozen_approval/config/event_detector_config.json"
            ),
            "generic_feasibility": sha256_file(
                REPOSITORY / "configs/doll_handoff_g1_feasibility_resolver.json"
            ),
        },
        "handoff_cartesian_residual_m": 0.0,
        "episode_count": 50,
        "total_frames": aggregate["total_frames"],
        "fps": 30.0,
        "timestamp_convention": "episode-local source frame_index / 30 Hz; no dropped frames",
        "joint_names": list(CANONICAL_JOINT_NAMES),
        "joint_order": "left_arm_7 + right_arm_7 + left_Dex3_DDS_7 + right_Dex3_DDS_7",
        "joint_specs": [jsonable(spec.__dict__) for spec in JOINT_SPECS],
        "action_dimension": 28,
        "trajectory_directory": str(trajectories_root.resolve()),
        "individual_trajectory_sha256": file_set,
        "trajectory_set_sha256": stable_json_sha256(file_set),
        "individual_component_action_array_set_sha256": component_set,
        "component_action_array_set_sha256": stable_json_sha256(component_set),
        "individual_canonical_policy_action_sha256": action_set,
        "canonical_policy_action_set_sha256": stable_json_sha256(action_set),
        "individual_timestamp_sha256": timestamp_set,
        "timestamp_set_sha256": stable_json_sha256(timestamp_set),
        "name_based_reordering_required": True,
        "full_model_qpos_dimension": 50,
        "full_model_qpos_is_policy_action": False,
        "per_episode_metrics": str(
            (FINAL_ROOT / "final50/per_episode.csv").resolve()
        ),
        "aggregate_metrics": str(aggregate_path.resolve()),
    }
    freeze_path = actions_root / "freeze_manifest.json"
    atomic_json(freeze_path, action_freeze)
    result = {
        "status": "PASS",
        "source_manifest": str(final_source_path.resolve()),
        "source_manifest_sha256": source_manifest_sha,
        "action_freeze_manifest": str(freeze_path.resolve()),
        "action_freeze_manifest_sha256": sha256_file(freeze_path),
        "trajectory_set_sha256": action_freeze["trajectory_set_sha256"],
        "canonical_policy_action_set_sha256": action_freeze[
            "canonical_policy_action_set_sha256"
        ],
        "classification_counts": status_counts,
        "total_frames": aggregate["total_frames"],
    }
    atomic_json(FINAL_ROOT / "final50/assembly_validation.json", result)
    return result


def build_training_schema() -> dict[str, Any]:
    """Resolve and validate the target G1 state/action contract."""
    import mujoco

    from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES, JOINT_SPECS

    assembly = load_json(FINAL_ROOT / "final50/assembly_validation.json")
    if assembly.get("status") != "PASS":
        raise RuntimeError("final trajectory assembly has not passed")
    source_manifest_path = FINAL_ROOT / "final_source_manifest.json"
    source_manifest = load_json(source_manifest_path)
    action_freeze_path = FINAL_ROOT / "retargeted_actions/freeze_manifest.json"
    action_freeze = load_json(action_freeze_path)
    if action_freeze.get("status") != "FINAL_PROPOSED_B_ACTION_LABELS_FROZEN":
        raise RuntimeError("action labels are not frozen")
    model_path = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model_joint_rows: list[dict[str, Any]] = []
    for spec in JOINT_SPECS:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name
        )
        if joint_id < 0:
            raise RuntimeError(f"controlled joint absent from G1 model: {spec.joint_name}")
        model_range = model.jnt_range[joint_id].astype(np.float64)
        contract_range = np.asarray([spec.minimum, spec.maximum], dtype=np.float64)
        if not np.array_equal(model_range, contract_range):
            raise RuntimeError(f"joint-limit contract mismatch: {spec.joint_name}")
        model_joint_rows.append(
            {
                "policy_index": spec.index,
                "joint_name": spec.joint_name,
                "mujoco_joint_id": int(joint_id),
                "mujoco_qpos_address": int(model.jnt_qposadr[joint_id]),
                "limit_rad": model_range.tolist(),
                "command_channel": spec.command_channel,
            }
        )
    controlled_min = np.full(28, np.inf, dtype=np.float64)
    controlled_max = np.full(28, -np.inf, dtype=np.float64)
    reorder_records: list[dict[str, Any]] = []
    for episode in source_manifest["episodes"]:
        arrays = _trajectory_policy_arrays(Path(episode["retargeted_trajectory_path"]))
        action = arrays["action"].astype(np.float64)
        controlled_min = np.minimum(controlled_min, np.min(action, axis=0))
        controlled_max = np.maximum(controlled_max, np.max(action, axis=0))
        reorder_records.append(
            {
                "final_dataset_index": int(episode["final_dataset_index"]),
                "input_joint_names": list(arrays["input_joint_names"]),
                "canonical_reorder_indices": list(
                    arrays["canonical_reorder_indices"]
                ),
                "canonical_action_sha256": arrays[
                    "canonical_policy_action_sha256"
                ],
            }
        )
    lower = np.asarray([spec.minimum for spec in JOINT_SPECS], dtype=np.float64)
    upper = np.asarray([spec.maximum for spec in JOINT_SPECS], dtype=np.float64)
    if np.any(controlled_min < lower - 1e-6) or np.any(controlled_max > upper + 1e-6):
        raise RuntimeError("final controlled trajectory exceeds G1 model joint limits")
    evidence_paths = [
        model_path,
        Path(
            "/home/jbnu/jaeyoung/unitree/unitree_sdk2_python/example/g1/"
            "high_level/g1_arm7_sdk_dds_example.py"
        ),
        Path(
            "/home/jbnu/jaeyoung/unitree/unitree_sdk2/example/g1/dex3/"
            "g1_dex3_example.cpp"
        ),
        REPOSITORY / "tools/record_g1_behavior_readonly.py",
        REPOSITORY / "tools/g1_behavior_schema.py",
        Path(
            "/home/jbnu/.lerobot_trossen_ai_data_collection_ui/lerobot/"
            "common/robot_devices/robots/manipulator.py"
        ),
        Path(
            "/home/jbnu/.lerobot_trossen_ai_data_collection_ui/lerobot/"
            "common/robot_devices/control_utils.py"
        ),
        Path(
            "/home/jbnu/lerobot-smolvla/src/lerobot/policies/smolvla/"
            "configuration_smolvla.py"
        ),
        Path("/home/jbnu/lerobot-smolvla/src/lerobot/datasets/factory.py"),
        Path("/home/jbnu/lerobot-smolvla/src/lerobot/datasets/dataset_reader.py"),
    ]
    for path in evidence_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen_source = load_json(
        REVIEW_ROOT / "frozen_approval/source/source_manifest.json"
    )
    source_alignment = {
        "recorded_source_row_semantics": (
            "teleop_step reads leader targets, sends the absolute follower goal, then "
            "reads follower present positions and cam_high; control_loop merges that "
            "observation and action into one row. Thus source action[t] is the command "
            "associated with row t, not q[t+1]."
        ),
        "frozen_converter_motion_source": frozen_source["motion_source_key"],
        "frozen_converter_command_source": frozen_source["command_source_key"],
        "frozen_converter_motion_selection": frozen_source[
            "motion_source_selection"
        ],
        "pooled_arm_command_to_measured_lag_frames": frozen_source[
            "pooled_action_to_measured_state_arm_lag_audit"
        ]["best_state_lag_frames"],
        "pooled_gripper_command_to_measured_lag_frames": frozen_source[
            "pooled_action_to_measured_state_gripper_lag_audit"
        ]["best_state_lag_frames"],
        "lag_interpretation": (
            "The pooled eight-frame value diagnoses source plant response and was part "
            "of the already-frozen event/motion treatment. It does not redefine LeRobot "
            "row semantics and is not applied as a new label shift."
        ),
    }
    policy_names = list(CANONICAL_JOINT_NAMES)
    schema = {
        "schema_version": "doll_handoff_g1_training_schema_v1",
        "status": "PASS_UNAMBIGUOUS_TARGET_SCHEMA",
        "task": "DOLL-HANDOFF-TO-BIN",
        "task_instruction": TASK_INSTRUCTION,
        "fps": 30.0,
        "rgb": {
            "key": "observation.images.cam_high",
            "source": "unaltered source ALOHA cam_high RGB frame at row t",
            "shape_hwc": [480, 640, 3],
            "synthesized": False,
            "g1_render_substituted": False,
        },
        "observation_state": {
            "key": "observation.state",
            "dimension": 28,
            "joint_names": policy_names,
            "joint_order": (
                "left_arm_7 + right_arm_7 + left_Dex3_DDS_7 + right_Dex3_DDS_7"
            ),
            "unit": "radian",
            "dtype": "float32",
            "definition": (
                "current controlled-joint G1/Dex3 retargeted target state q_target[t] "
                "at the source RGB row t"
            ),
            "internal_semantic_name": "retargeted_target_state",
            "measured_real_g1_state": False,
            "deployment_adapter": (
                "At deployment, assemble measured G1 arm and Dex3 qpos by exact joint "
                "name in this same order. Never supply floating-base/full-model qpos."
            ),
        },
        "action": {
            "key": "action",
            "dimension": 28,
            "joint_names": policy_names,
            "joint_order": (
                "left_arm_7 + right_arm_7 + left_Dex3_DDS_7 + right_Dex3_DDS_7"
            ),
            "unit": "radian",
            "dtype": "float32",
            "definition": "same-row absolute controlled-joint position target q_target[t]",
            "delta_action": False,
            "next_frame_state": False,
        },
        "state_action_joint_set_relation": (
            "observation.state and action use exactly the same 28 named controlled joints "
            "in exactly the same order"
        ),
        "state_action_value_relation_offline": (
            "For these retargeted demonstrations observation.state[t] and action[t] are "
            "distinct float32 arrays with equal values q_target[t]. This target-state "
            "surrogate is explicit and is not claimed to be measured real-G1 state."
        ),
        "temporal_convention": {
            "state_row": "q_target[t] at source frame t",
            "action_row": "absolute q_target[t] at source frame t",
            "row_offset": 0,
            "timestamp": "episode-local source timestamp t = frame_index / 30 Hz",
            "input_frame_indices": "0..T-1",
            "target_frame_indices": "0..T-1",
            "dropped_boundary_frames": 0,
            "last_frame": "retained; same-row action is valid",
            "smolvla_observation_delta_indices": [0],
            "smolvla_action_delta_indices": list(range(50)),
            "smolvla_chunk_boundary": (
                "future indices are clamped to the episode last row and marked action_is_pad"
            ),
        },
        "channels": {
            "left_arm": {"indices": list(range(0, 7)), "dimension": 7},
            "right_arm": {"indices": list(range(7, 14)), "dimension": 7},
            "left_Dex3": {"indices": list(range(14, 21)), "dimension": 7},
            "right_Dex3": {"indices": list(range(21, 28)), "dimension": 7},
        },
        "controller_contract": {
            "left_arm_motor_ids": list(range(15, 22)),
            "right_arm_motor_ids": list(range(22, 29)),
            "left_Dex3_motor_ids": list(range(7)),
            "right_Dex3_motor_ids": list(range(7)),
            "arm_transport": "absolute G1 arm motor q targets",
            "hand_transport": "absolute Dex3 q targets",
            "full_model_qpos_dimension": int(model.nq),
            "full_model_qpos_is_policy_action": False,
            "excluded": [
                "floating base",
                "legs",
                "waist",
                "walking/base velocity/body-height commands",
            ],
        },
        "model_joint_validation": {
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "model_nq": int(model.nq),
            "model_njnt": int(model.njnt),
            "all_28_names_present": len(model_joint_rows) == 28,
            "all_limits_exact": True,
            "joints": model_joint_rows,
            "dataset_min_rad": controlled_min.tolist(),
            "dataset_max_rad": controlled_max.tolist(),
            "dataset_within_model_limits": True,
        },
        "name_remapping_validation": {
            "all_50_trajectory_name_sets_exact": True,
            "positional_guessing_used": False,
            "episodes": reorder_records,
        },
        "source_temporal_audit": source_alignment,
        "cross_embodiment_construction": {
            "description": (
                "Source ALOHA cam_high RGB and language/timestamps are paired with "
                "frozen Proposed-B G1/Dex3 target state/action labels."
            ),
            "source_observation_morphology": "ALOHA",
            "state_action_embodiment": "Unitree G1 fixed-base arms + Dex3",
            "domain_gap_acknowledged": True,
            "source_ALOHA_14D_state_used_as_target_state": False,
            "replacement_images_synthesized": False,
        },
        "provenance": {
            "final_source_manifest": str(source_manifest_path.resolve()),
            "final_source_manifest_sha256": sha256_file(source_manifest_path),
            "retargeted_action_freeze": str(action_freeze_path.resolve()),
            "retargeted_action_freeze_sha256": sha256_file(action_freeze_path),
            "evidence_files": {
                str(path): sha256_file(path) for path in evidence_paths
            },
        },
    }
    schema_root = FINAL_ROOT / "training_schema"
    schema_path = schema_root / "g1_training_schema.json"
    atomic_json(schema_path, schema)
    md = f"""# G1 training state/action schema

Status: **PASS — unambiguous target-compatible schema**

Dataset B uses one fixed 28D controlled-joint vector for both fields. The order is
`left arm 7 + right arm 7 + left Dex3 DDS 7 + right Dex3 DDS 7`; every column is
selected by joint name. The 50D MuJoCo qpos includes floating-base and lower-body
coordinates and is explicitly not a policy action.

`observation.state[t]` is the current retargeted G1/Dex3 target state
`q_target[t]`. `action[t]` is an absolute controlled-joint position target with
the same value, names, order, units, and source-row timestamp. This is a
retargeted target-state surrogate, not measured real-G1 feedback. At deployment,
the state adapter must provide measured arm/Dex3 qpos in this exact named order.

There is no row shift: input indices and target indices are both `0..T-1`, all
boundary rows are retained, and timestamps remain `frame_index / 30 Hz`. The
source collector stores the same-cycle command and measured observation in one
row; its pooled eight-frame plant-response lag is diagnostic and is not a label
shift. SmolVLA consumes state offset `[0]` and action chunk offsets `[0..49]`.

The four channel groups are left arm indices 0–6, right arm 7–13, left Dex3
14–20, and right Dex3 21–27. All values are radians. Every named joint and limit
matches `{model_path}`, and all final Dataset-B values are within those limits.

The visual/state-action pairing is intentionally cross-embodiment: unmodified
ALOHA `cam_high` RGB, language, and timestamps are preserved, while state/action
are G1/Dex3 labels. No G1 render or synthesized image replaces source RGB.

Task instruction: `{TASK_INSTRUCTION}`
"""
    schema_root.mkdir(parents=True, exist_ok=True)
    (schema_root / "g1_training_schema.md").write_text(md, encoding="utf-8")
    result = {
        "status": "PASS",
        "schema": str(schema_path.resolve()),
        "schema_sha256": sha256_file(schema_path),
        "report": str((schema_root / "g1_training_schema.md").resolve()),
        "report_sha256": sha256_file(schema_root / "g1_training_schema.md"),
        "state_dimension": 28,
        "action_dimension": 28,
        "timestamp_offset_frames": 0,
    }
    atomic_json(schema_root / "validation.json", result)
    return result


def _fixed_list_arrow(values: np.ndarray, dimension: int) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    flattened = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flattened, dimension)


def _feature_statistics(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError("statistics require a finite non-empty [N,D] array")
    result: dict[str, list[Any]] = {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=0).tolist(),
        "count": [int(len(values))],
    }
    for quantile in (0.01, 0.10, 0.50, 0.90, 0.99):
        result[f"q{int(quantile * 100):02d}"] = np.quantile(
            values, quantile, axis=0
        ).tolist()
    return result


def _validated_video_probe(path: Path, frame_count: int) -> dict[str, Any]:
    probe = ffprobe_video(path)
    streams = (probe.get("probe") or {}).get("streams", [])
    if not probe["readable"] or len(streams) != 1:
        raise RuntimeError(f"video is not readable: {path}: {probe}")
    stream = streams[0]
    numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/")
    fps = float(numerator) / float(denominator)
    if (
        int(stream.get("width", -1)) != 640
        or int(stream.get("height", -1)) != 480
        or int(stream.get("nb_frames", -1)) != frame_count
        or abs(fps - 30.0) > 1e-6
    ):
        raise RuntimeError(f"encoded video stream mismatch: {path}: {stream}")
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": int(stream["nb_frames"]),
    }


def _encode_source_video(
    source_images: Path, destination: Path, frame_count: int
) -> dict[str, Any]:
    if destination.is_file():
        stream = _validated_video_probe(destination, frame_count)
        return {"reused_validated_output": True, **stream}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    if temporary.exists():
        raise RuntimeError(f"stale video encoder temporary exists: {temporary}")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-framerate",
        "30",
        "-start_number",
        "0",
        "-i",
        str(source_images / "frame_%06d.png"),
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-threads",
        "2",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {source_images}: {completed.stderr.strip()}"
        )
    stream = _validated_video_probe(temporary, frame_count)
    os.replace(temporary, destination)
    return {"reused_validated_output": False, **stream}


def package_lerobot_dataset() -> dict[str, Any]:
    """Package final labels plus unmodified source cam_high observations as LeRobot v3."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES

    schema_path = FINAL_ROOT / "training_schema/g1_training_schema.json"
    schema = load_json(schema_path)
    if schema.get("status") != "PASS_UNAMBIGUOUS_TARGET_SCHEMA":
        raise RuntimeError("G1 training schema has not passed")
    source_manifest_path = FINAL_ROOT / "final_source_manifest.json"
    source_manifest = load_json(source_manifest_path)
    if source_manifest.get("classification_counts", {}).get("HARD_FAIL") != 0:
        raise RuntimeError("cannot package a final source set containing HARD_FAIL")
    if DATASET_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {DATASET_ROOT}")
    staging = DATASET_ROOT.parent / ".doll_handoff_proposed_b_50.build-v1"
    staging.mkdir(parents=True, exist_ok=True)
    episode_records: list[dict[str, Any]] = []
    for episode in source_manifest["episodes"]:
        final_index = int(episode["final_dataset_index"])
        trajectory = _trajectory_policy_arrays(
            Path(episode["retargeted_trajectory_path"])
        )
        raw_root = Path(episode["raw_directory_path"])
        raw_parquet = Path(episode["source_parquet_path"])
        raw = pq.read_table(
            raw_parquet,
            columns=[
                "timestamp",
                "frame_index",
                "episode_index",
                "task_index",
                "observation.state",
                "action",
            ],
        )
        frame_count = int(episode["source_frame_count"])
        if raw.num_rows != frame_count or trajectory["action"].shape != (
            frame_count,
            28,
        ):
            raise RuntimeError(f"frame count mismatch before packaging ep{final_index:03d}")
        raw_timestamp = np.asarray(
            raw["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float32,
        )
        raw_frame = np.asarray(
            raw["frame_index"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        if not np.array_equal(raw_frame, np.arange(frame_count, dtype=np.int64)):
            raise RuntimeError(f"raw frame index mismatch ep{final_index:03d}")
        if not np.allclose(
            raw_timestamp,
            trajectory["timestamps"],
            atol=2e-6,
            rtol=0.0,
        ):
            raise RuntimeError(f"raw/target timestamp mismatch ep{final_index:03d}")
        image_root = (
            raw_root
            / "images/observation.images.cam_high/episode_000000"
        )
        images = sorted(image_root.glob("frame_*.png"))
        expected_names = [f"frame_{index:06d}.png" for index in range(frame_count)]
        if len(images) != frame_count or [path.name for path in images] != expected_names:
            raise RuntimeError(f"source cam_high frames incomplete ep{final_index:03d}")
        first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        last = cv2.imread(str(images[-1]), cv2.IMREAD_COLOR)
        if first is None or last is None or first.shape != (480, 640, 3) or last.shape != (
            480,
            640,
            3,
        ):
            raise RuntimeError(f"source cam_high image shape/read failure ep{final_index:03d}")
        episode_records.append(
            {
                "final_index": final_index,
                "source": episode,
                "action": trajectory["action"],
                "state": np.array(trajectory["action"], copy=True),
                "timestamp": raw_timestamp,
                "image_root": image_root,
                "first_image_sha256": sha256_file(images[0]),
                "last_image_sha256": sha256_file(images[-1]),
            }
        )
    if [record["final_index"] for record in episode_records] != list(range(50)):
        raise RuntimeError("packaging inputs are not contiguous final episodes 0..49")
    print("[PACKAGE] encoding 50 cam_high source episodes as H.264 LeRobot videos", flush=True)
    futures: dict[Any, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for record in episode_records:
            destination = (
                staging
                / "videos/observation.images.cam_high/chunk-000"
                / f"file-{record['final_index']:03d}.mp4"
            )
            record["video_path"] = destination
            future = executor.submit(
                _encode_source_video,
                record["image_root"],
                destination,
                len(record["timestamp"]),
            )
            futures[future] = record
        completed_count = 0
        for future in as_completed(futures):
            record = futures[future]
            record["video_probe"] = future.result()
            completed_count += 1
            print(
                f"[PACKAGE] video {completed_count:02d}/50 "
                f"ep{record['final_index']:03d}",
                flush=True,
            )
    all_state = np.concatenate([record["state"] for record in episode_records])
    all_action = np.concatenate([record["action"] for record in episode_records])
    all_timestamp = np.concatenate(
        [record["timestamp"] for record in episode_records]
    ).astype(np.float32)
    all_frame = np.concatenate(
        [np.arange(len(record["state"]), dtype=np.int64) for record in episode_records]
    )
    all_episode = np.concatenate(
        [
            np.full(len(record["state"]), record["final_index"], dtype=np.int64)
            for record in episode_records
        ]
    )
    total_frames = len(all_state)
    aggregate = load_json(FINAL_ROOT / "final50/aggregate_metrics.json")
    if total_frames != int(aggregate["total_frames"]):
        raise RuntimeError("packaged total frame count differs from final-50 freeze")
    if not np.array_equal(all_state, all_action):
        raise RuntimeError("state/action target-state contract changed during packaging")
    table = pa.Table.from_arrays(
        [
            _fixed_list_arrow(all_state, 28),
            _fixed_list_arrow(all_action, 28),
            pa.array(all_timestamp, type=pa.float32()),
            pa.array(all_frame, type=pa.int64()),
            pa.array(all_episode, type=pa.int64()),
            pa.array(np.arange(total_frames, dtype=np.int64), type=pa.int64()),
            pa.array(np.zeros(total_frames, dtype=np.int64), type=pa.int64()),
        ],
        names=[
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    )
    data_path = staging / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        data_path,
        compression="snappy",
        use_dictionary=True,
        row_group_size=8192,
    )
    episode_rows: list[dict[str, Any]] = []
    offset = 0
    for record in episode_records:
        length = len(record["state"])
        row: dict[str, Any] = {
            "episode_index": record["final_index"],
            "tasks": [TASK_INSTRUCTION],
            "length": length,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": offset,
            "dataset_to_index": offset + length,
            "videos/observation.images.cam_high/chunk_index": 0,
            "videos/observation.images.cam_high/file_index": record["final_index"],
            "videos/observation.images.cam_high/from_timestamp": 0.0,
            "videos/observation.images.cam_high/to_timestamp": length / 30.0,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for feature in ("observation.state", "action"):
            values = record["state"] if feature == "observation.state" else record["action"]
            for statistic, value in _feature_statistics(values).items():
                row[f"stats/{feature}/{statistic}"] = value
        episode_rows.append(row)
        offset += length
    episodes_path = staging / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(episode_rows),
        episodes_path,
        compression="snappy",
        use_dictionary=True,
    )
    task_reference = pq.read_table(
        REPOSITORY / "lerobot_magsafe_50_cam_high_v3/meta/tasks.parquet"
    )
    task_table = pa.Table.from_arrays(
        [
            pa.array([0], type=pa.int64()),
            pa.array([TASK_INSTRUCTION], type=pa.string()),
        ],
        names=["task_index", "__index_level_0__"],
    ).replace_schema_metadata(task_reference.schema.metadata)
    pq.write_table(task_table, staging / "meta/tasks.parquet")
    feature = {
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": 30.0,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [28],
            "names": list(CANONICAL_JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [28],
            "names": list(CANONICAL_JOINT_NAMES),
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1_fixed_base_dex3_retargeted",
        "total_episodes": 50,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": 30,
        "splits": {"train": "0:50"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": feature,
    }
    atomic_json(staging / "meta/info.json", info)
    normalization = {
        "observation.state": _feature_statistics(all_state),
        "action": _feature_statistics(all_action),
    }
    atomic_json(staging / "meta/stats.json", normalization)
    alignment_rows = []
    video_rows = []
    for record in episode_records:
        source = record["source"]
        length = len(record["state"])
        alignment_rows.append(
            {
                "final_dataset_episode_index": record["final_index"],
                "source_raw_episode": source["raw_directory"],
                "source_timestamp_start_s": float(record["timestamp"][0]),
                "source_timestamp_end_s": float(record["timestamp"][-1]),
                "source_frame_count": length,
                "rgb_frame_count": length,
                "state_frame_count": length,
                "action_frame_count": length,
                "input_frame_indices": [0, length - 1],
                "target_frame_indices": [0, length - 1],
                "row_offset": 0,
                "dropped_boundary_frames": 0,
                "classification": source["classification"],
                "warning_reasons": source["warning_reasons"],
            }
        )
        video_rows.append(
            {
                "final_dataset_episode_index": record["final_index"],
                "source_raw_episode": source["raw_directory"],
                "source_png_directory": str(record["image_root"]),
                "source_png_count": length,
                "source_first_png_sha256": record["first_image_sha256"],
                "source_last_png_sha256": record["last_image_sha256"],
                "source_processing": "sequential encode only; no crop, resize, synthesis, or G1 render",
                "dataset_video": str(record["video_path"].relative_to(staging)),
                "dataset_video_sha256": sha256_file(record["video_path"]),
                "probe": record["video_probe"],
            }
        )
    packaging_manifest = {
        "schema_version": "doll_handoff_proposed_b_lerobot_package_v1",
        "status": "PACKAGED_PENDING_FULL_READER_VALIDATION",
        "retargeting_method": "interaction_centric_proposed_b",
        "method_is_policy_input": False,
        "source_visual_embodiment": "ALOHA",
        "target_state_action_embodiment": "Unitree G1 fixed-base arms + Dex3",
        "cross_embodiment_pairing_explicit": True,
        "source_ALOHA_state_packaged": False,
        "images_synthesized": False,
        "images_replaced_by_G1_render": False,
        "source_image_encoding": "H.264 CRF 18 yuv420p, 30 Hz, no crop or resize",
        "task_instruction": TASK_INSTRUCTION,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "action_freeze_manifest": str(
            (FINAL_ROOT / "retargeted_actions/freeze_manifest.json").resolve()
        ),
        "action_freeze_manifest_sha256": sha256_file(
            FINAL_ROOT / "retargeted_actions/freeze_manifest.json"
        ),
        "training_schema": str(schema_path.resolve()),
        "training_schema_sha256": sha256_file(schema_path),
        "episode_count": 50,
        "total_frames": total_frames,
        "state_dimension": 28,
        "action_dimension": 28,
        "joint_names": list(CANONICAL_JOINT_NAMES),
        "temporal_convention": "same row t, zero offset, no dropped frames",
        "episode_alignment": alignment_rows,
        "video_assets": video_rows,
        "external_per_episode_metadata": (
            "source IDs and classifications remain in this manifest rather than adding "
            "nonstandard columns to the LeRobot data parquet"
        ),
        "normalization": (
            "Dataset-B final 50 only; per-channel MEAN_STD population statistics"
        ),
        "training_executed": False,
    }
    atomic_json(staging / "meta/g1_packaging_manifest.json", packaging_manifest)
    atomic_json(
        staging / "meta/g1_training_contract.json",
        {
            "schema_version": schema["schema_version"],
            "observation_state_semantic": schema["observation_state"]["definition"],
            "observation_state_is_measured_real_g1": False,
            "future_real_g1_observation_state": schema["observation_state"][
                "deployment_adapter"
            ],
            "action_semantic": schema["action"]["definition"],
            "causal_relation": "observation.state[t] -> action[t]",
            "row_offset": 0,
            "action_chunk_start": "action[t]",
            "action_chunk_size": 50,
            "source_visual_embodiment": "ALOHA",
            "target_action_embodiment": "Unitree_G1_Dex3",
            "joint_order": list(CANONICAL_JOINT_NAMES),
        },
    )
    normalization_root = FINAL_ROOT / "normalization"
    atomic_json(
        normalization_root / "dataset_b_state_action_normalization.json",
        {
            "schema_version": "doll_handoff_dataset_b_normalization_v1",
            "status": "PASS",
            "algorithm": "MEAN_STD",
            "formula": "(x - mean) / (population_std + 1e-8)",
            "population_std_ddof": 0,
            "fit_episode_count": 50,
            "fit_frame_count": total_frames,
            "failed_episode_used": False,
            "source_ALOHA_statistics_reused": False,
            "statistics": normalization,
        },
    )
    # Publish only after all inputs, data, videos, and metadata are complete.
    os.replace(staging, DATASET_ROOT)
    result = {
        "status": "PASS_PACKAGED_PENDING_FULL_AUDIT",
        "dataset_path": str(DATASET_ROOT.resolve()),
        "episodes": 50,
        "frames": total_frames,
        "videos": 50,
        "state_dimension": 28,
        "action_dimension": 28,
        "normalization": str(
            (normalization_root / "dataset_b_state_action_normalization.json").resolve()
        ),
    }
    atomic_json(FINAL_ROOT / "dataset_package_result.json", result)
    return result


def _dataset_tree_sha256(root: Path) -> tuple[str, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return stable_json_sha256(entries), entries


def _motion_diagnostic(
    episode_values: list[np.ndarray], joint_names: list[str], fps: float
) -> dict[str, Any]:
    steps: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    step_locations: list[tuple[int, int]] = []
    acceleration_locations: list[tuple[int, int]] = []
    for episode, values in enumerate(episode_values):
        delta = np.diff(values.astype(np.float64), axis=0)
        second = np.diff(values.astype(np.float64), n=2, axis=0)
        steps.append(delta)
        accelerations.append(second)
        step_locations.extend((episode, frame) for frame in range(1, len(values)))
        acceleration_locations.extend(
            (episode, frame) for frame in range(2, len(values))
        )
    step = np.concatenate(steps, axis=0)
    acceleration_step = np.concatenate(accelerations, axis=0)
    absolute_step = np.abs(step)
    absolute_velocity = absolute_step * fps
    absolute_acceleration = np.abs(acceleration_step) * fps * fps

    def maximum_record(values: np.ndarray, locations: list[tuple[int, int]]) -> dict[str, Any]:
        flat = int(np.argmax(values))
        row, joint = np.unravel_index(flat, values.shape)
        episode, frame = locations[int(row)]
        return {
            "value": float(values[row, joint]),
            "episode_index": int(episode),
            "frame_index": int(frame),
            "joint_index": int(joint),
            "joint_name": joint_names[int(joint)],
        }

    return {
        "maximum_absolute_step_rad": maximum_record(absolute_step, step_locations),
        "maximum_absolute_velocity_rad_s": maximum_record(
            absolute_velocity, step_locations
        ),
        "maximum_absolute_acceleration_rad_s2": maximum_record(
            absolute_acceleration, acceleration_locations
        ),
        "per_joint_maximum_absolute_step_rad": np.max(
            absolute_step, axis=0
        ).tolist(),
        "per_joint_maximum_absolute_velocity_rad_s": np.max(
            absolute_velocity, axis=0
        ).tolist(),
        "per_joint_maximum_absolute_acceleration_rad_s2": np.max(
            absolute_acceleration, axis=0
        ).tolist(),
        "mean_absolute_step_rad": float(np.mean(absolute_step)),
        "mean_absolute_velocity_rad_s": float(np.mean(absolute_velocity)),
        "mean_absolute_acceleration_rad_s2": float(
            np.mean(absolute_acceleration)
        ),
    }


def validate_lerobot_dataset() -> dict[str, Any]:
    """Run structural, numerical, video, alignment, and real-reader validation."""
    from tools.g1_training_schema_v1.constants import CANONICAL_JOINT_NAMES, JOINT_SPECS

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(DATASET_ROOT)
    validation_root = FINAL_ROOT / "dataset_validation"
    info = load_json(DATASET_ROOT / "meta/info.json")
    packaging = load_json(DATASET_ROOT / "meta/g1_packaging_manifest.json")
    source_manifest = load_json(FINAL_ROOT / "final_source_manifest.json")
    if info.get("total_episodes") != 50 or info.get("total_frames") != 34478:
        raise RuntimeError("dataset info episode/frame count mismatch")
    expected_names = list(CANONICAL_JOINT_NAMES)
    checks: dict[str, bool] = {
        "episode_count_50": int(info["total_episodes"]) == 50,
        "fps_30": float(info["fps"]) == 30.0,
        "state_dimension_28": info["features"]["observation.state"]["shape"] == [28],
        "action_dimension_28": info["features"]["action"]["shape"] == [28],
        "state_joint_order_constant": info["features"]["observation.state"][
            "names"
        ]
        == expected_names,
        "action_joint_order_constant": info["features"]["action"]["names"]
        == expected_names,
        "rgb_key_present": "observation.images.cam_high" in info["features"],
        "task_count_one": int(info["total_tasks"]) == 1,
        "no_hard_failure_in_source_manifest": source_manifest[
            "classification_counts"
        ]["HARD_FAIL"]
        == 0,
    }
    data_paths = sorted((DATASET_ROOT / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise RuntimeError("dataset contains no data Parquet")
    data_tables = [pq.read_table(path) for path in data_paths]
    table = pa.concat_tables(data_tables)
    state = fixed_list_numpy(table["observation.state"]).astype(np.float32)
    action = fixed_list_numpy(table["action"]).astype(np.float32)
    scalar_arrays = {
        key: np.asarray(
            table[key].combine_chunks().to_numpy(zero_copy_only=False)
        )
        for key in (
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        )
    }
    total_frames = len(state)
    checks.update(
        {
            "all_data_parquet_readable": all(path.is_file() for path in data_paths),
            "total_frames_matches": total_frames == int(info["total_frames"]),
            "state_shape_constant": state.shape == (total_frames, 28),
            "action_shape_constant": action.shape == (total_frames, 28),
            "nan_zero": int(np.count_nonzero(np.isnan(state)))
            + int(np.count_nonzero(np.isnan(action)))
            == 0,
            "inf_zero": int(np.count_nonzero(np.isinf(state)))
            + int(np.count_nonzero(np.isinf(action)))
            == 0,
            "same_row_state_action_equal": np.array_equal(state, action),
            "global_index_contiguous": np.array_equal(
                scalar_arrays["index"], np.arange(total_frames, dtype=np.int64)
            ),
            "task_index_zero": np.array_equal(
                scalar_arrays["task_index"], np.zeros(total_frames, dtype=np.int64)
            ),
        }
    )
    episode_paths = sorted(
        (DATASET_ROOT / "meta/episodes").glob("chunk-*/*.parquet")
    )
    episode_rows: list[dict[str, Any]] = []
    for path in episode_paths:
        episode_rows.extend(pq.read_table(path).to_pylist())
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    checks["all_episode_parquet_readable"] = bool(episode_paths)
    checks["episode_rows_50"] = len(episode_rows) == 50
    checks["episode_indices_contiguous"] = [
        int(row["episode_index"]) for row in episode_rows
    ] == list(range(50))
    task_table = pq.read_table(DATASET_ROOT / "meta/tasks.parquet")
    task_rows = task_table.to_pylist()
    checks["task_metadata_present"] = (
        len(task_rows) == 1
        and task_rows[0].get("task_index") == 0
        and task_rows[0].get("__index_level_0__") == TASK_INSTRUCTION
        and all(row.get("tasks") == [TASK_INSTRUCTION] for row in episode_rows)
    )
    lower = np.asarray([spec.minimum for spec in JOINT_SPECS], dtype=np.float32)
    upper = np.asarray([spec.maximum for spec in JOINT_SPECS], dtype=np.float32)
    checks["joint_range_valid"] = bool(
        np.all(state >= lower - 1e-6) and np.all(state <= upper + 1e-6)
    )
    per_episode_state: list[np.ndarray] = []
    per_episode_action: list[np.ndarray] = []
    alignment_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    for episode, (source, metadata) in enumerate(
        zip(source_manifest["episodes"], episode_rows, strict=True)
    ):
        mask = scalar_arrays["episode_index"] == episode
        indices = np.flatnonzero(mask)
        length = int(metadata["length"])
        if len(indices) != length:
            raise RuntimeError(f"episode {episode} data length mismatch")
        current_state = state[indices]
        current_action = action[indices]
        per_episode_state.append(current_state)
        per_episode_action.append(current_action)
        raw_table = pq.read_table(
            Path(source["source_parquet_path"]), columns=["timestamp", "frame_index"]
        )
        raw_timestamp = np.asarray(
            raw_table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.float32,
        )
        timestamps = scalar_arrays["timestamp"][indices].astype(np.float32)
        trajectory = _trajectory_policy_arrays(
            Path(source["retargeted_trajectory_path"])
        )
        source_images = sorted(
            (
                Path(source["raw_directory_path"])
                / "images/observation.images.cam_high/episode_000000"
            ).glob("frame_*.png")
        )
        episode_alignment = {
            "final_dataset_index": episode,
            "source_raw_episode": source["raw_directory"],
            "length": length,
            "rgb_frames": len(source_images),
            "state_frames": len(current_state),
            "action_frames": len(current_action),
            "source_timestamps_monotonic": bool(np.all(np.diff(raw_timestamp) > 0.0)),
            "dataset_timestamps_monotonic": bool(np.all(np.diff(timestamps) > 0.0)),
            "source_dataset_timestamps_exact": bool(
                np.array_equal(raw_timestamp, timestamps)
            ),
            "timestamp_frame_index_over_fps": bool(
                np.allclose(
                    timestamps,
                    np.arange(length) / 30.0,
                    atol=2e-6,
                    rtol=0.0,
                )
            ),
            "frame_index_contiguous": bool(
                np.array_equal(
                    scalar_arrays["frame_index"][indices],
                    np.arange(length, dtype=np.int64),
                )
            ),
            "state_matches_frozen_trajectory": bool(
                np.array_equal(current_state, trajectory["action"])
            ),
            "action_matches_frozen_trajectory": bool(
                np.array_equal(current_action, trajectory["action"])
            ),
            "row_offset": 0,
            "dropped_boundary_frames": 0,
        }
        if not all(
            value
            for key, value in episode_alignment.items()
            if key
            not in {
                "final_dataset_index",
                "source_raw_episode",
                "length",
                "rgb_frames",
                "state_frames",
                "action_frames",
                "row_offset",
                "dropped_boundary_frames",
            }
        ) or not (
            len(source_images)
            == len(current_state)
            == len(current_action)
            == length
        ):
            raise RuntimeError(
                f"episode {episode} frame/time/trajectory alignment failed: {episode_alignment}"
            )
        alignment_rows.append(episode_alignment)
        video_path = (
            DATASET_ROOT
            / info["video_path"].format(
                video_key="observation.images.cam_high",
                chunk_index=int(
                    metadata[
                        "videos/observation.images.cam_high/chunk_index"
                    ]
                ),
                file_index=int(
                    metadata["videos/observation.images.cam_high/file_index"]
                ),
            )
        )
        stream = _validated_video_probe(video_path, length)
        video_rows.append(
            {
                "episode_index": episode,
                "path": str(video_path.relative_to(DATASET_ROOT)),
                "sha256": sha256_file(video_path),
                **stream,
            }
        )
    checks.update(
        {
            "all_episodes_loadable_structurally": len(alignment_rows) == 50,
            "all_source_timestamps_monotonic": all(
                row["source_timestamps_monotonic"] for row in alignment_rows
            ),
            "all_state_action_timestamps_consistent": all(
                row["source_dataset_timestamps_exact"]
                and row["timestamp_frame_index_over_fps"]
                for row in alignment_rows
            ),
            "no_unexplained_frame_loss": all(
                row["rgb_frames"]
                == row["state_frames"]
                == row["action_frames"]
                == row["length"]
                and row["dropped_boundary_frames"] == 0
                for row in alignment_rows
            ),
            "all_trajectory_labels_hash_equivalent": all(
                row["state_matches_frozen_trajectory"]
                and row["action_matches_frozen_trajectory"]
                for row in alignment_rows
            ),
            "all_50_videos_ffprobe_readable": len(video_rows) == 50,
        }
    )
    persisted_stats = load_json(DATASET_ROOT / "meta/stats.json")
    recomputed_stats = {
        "observation.state": _feature_statistics(state),
        "action": _feature_statistics(action),
    }
    checks["normalization_recomputed_exact"] = canonical_equal(
        persisted_stats, recomputed_stats
    )
    if not checks["normalization_recomputed_exact"]:
        raise RuntimeError("persisted normalization differs from packaged data")
    lengths = np.asarray([int(row["length"]) for row in episode_rows], dtype=np.int64)
    statistics = {
        "schema_version": "doll_handoff_dataset_b_statistics_v1",
        "status": "PASS",
        "episodes": 50,
        "total_frames": total_frames,
        "episode_length_frames": {
            "per_episode": lengths.tolist(),
            "minimum": int(np.min(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "standard_deviation": float(np.std(lengths, ddof=0)),
            "maximum": int(np.max(lengths)),
            "q10": float(np.quantile(lengths, 0.10)),
            "q90": float(np.quantile(lengths, 0.90)),
        },
        "fps_distribution": {"30.0": 50},
        "joint_names": expected_names,
        "observation.state": recomputed_stats["observation.state"],
        "action": recomputed_stats["action"],
        "state_motion": _motion_diagnostic(
            per_episode_state, expected_names, 30.0
        ),
        "action_motion": _motion_diagnostic(
            per_episode_action, expected_names, 30.0
        ),
        "nan_count": int(np.count_nonzero(np.isnan(state)))
        + int(np.count_nonzero(np.isnan(action))),
        "inf_count": int(np.count_nonzero(np.isinf(state)))
        + int(np.count_nonzero(np.isinf(action))),
    }
    statistics_path = validation_root / "dataset_statistics.json"
    atomic_json(statistics_path, statistics)

    readback_path = validation_root / "lerobot_readback.json"
    command = [
        "/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python",
        str(REPOSITORY / "tools/validate_doll_handoff_dataset_b_lerobot.py"),
        "--dataset",
        str(DATASET_ROOT),
        "--output",
        str(readback_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "actual LeRobotDataset readback failed:\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    readback = load_json(readback_path)
    checks["actual_lerobot_dataset_readback"] = readback.get("status") == "PASS"
    checks["all_episodes_actual_video_decode"] = (
        readback.get("every_episode_middle_frame_read_count") == 50
    )
    checks["first_middle_last_random_reads"] = len(
        readback.get("first_middle_last_and_random_reads", [])
    ) >= 15
    checks["action_chunk_and_padding_read"] = len(
        readback.get("chunk_and_padding_reads", [])
    ) == 3
    if not all(checks.values()):
        failures = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Dataset-B full audit failed: {failures}")
    packaging["status"] = "PACKAGED_AND_FULLY_VALIDATED"
    packaging["full_validation"] = {
        "status": "PASS",
        "actual_lerobot_dataset_readback": True,
        "every_episode_video_decoded": True,
        "state_action_normalization_recomputed": True,
        "validation_completed_before_training": True,
    }
    atomic_json(DATASET_ROOT / "meta/g1_packaging_manifest.json", packaging)
    in_dataset_validation = {
        "schema_version": "doll_handoff_dataset_b_in_tree_validation_v1",
        "status": "PASS",
        "checks": checks,
        "lerobot_readback_sha256": sha256_file(readback_path),
        "dataset_statistics_sha256": sha256_file(statistics_path),
        "training_smoke_executed": False,
    }
    atomic_json(DATASET_ROOT / "meta/g1_validation.json", in_dataset_validation)
    tree_sha, tree_entries = _dataset_tree_sha256(DATASET_ROOT)
    audit = {
        "schema_version": "doll_handoff_dataset_b_full_audit_v1",
        "status": "PASS",
        "dataset_path": str(DATASET_ROOT.resolve()),
        "checks": checks,
        "episodes": 50,
        "total_frames": total_frames,
        "state_dimension": 28,
        "action_dimension": 28,
        "joint_order": expected_names,
        "task_instruction": TASK_INSTRUCTION,
        "parquet_files": [
            {
                "path": str(path.relative_to(DATASET_ROOT)),
                "sha256": sha256_file(path),
                "readable": True,
            }
            for path in (*data_paths, *episode_paths, DATASET_ROOT / "meta/tasks.parquet")
        ],
        "video_files": video_rows,
        "episode_alignment": alignment_rows,
        "normalization_statistics": persisted_stats,
        "statistics_path": str(statistics_path.resolve()),
        "statistics_sha256": sha256_file(statistics_path),
        "lerobot_readback_path": str(readback_path.resolve()),
        "lerobot_readback_sha256": sha256_file(readback_path),
        "dataset_tree_sha256": tree_sha,
        "dataset_tree_file_count": len(tree_entries),
        "dataset_tree_size_bytes": int(
            sum(entry["size_bytes"] for entry in tree_entries)
        ),
        "dataset_tree_entries": tree_entries,
        "source_observations_preserved": (
            "All source cam_high sequences were encoded in order without crop, resize, "
            "synthesis, or G1-render substitution."
        ),
        "no_target_hard_failure_episode_included": True,
    }
    audit_path = validation_root / "full_dataset_audit.json"
    atomic_json(audit_path, audit)
    result = {
        "status": "PASS",
        "dataset_path": str(DATASET_ROOT.resolve()),
        "episodes": 50,
        "frames": total_frames,
        "dataset_tree_sha256": tree_sha,
        "audit": str(audit_path.resolve()),
        "audit_sha256": sha256_file(audit_path),
        "lerobot_readback": str(readback_path.resolve()),
        "lerobot_readback_sha256": sha256_file(readback_path),
    }
    atomic_json(validation_root / "validation.json", result)
    return result


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _training_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(REPOSITORY)
            + (
                ":" + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            ),
            "MPLCONFIGDIR": "/tmp/doll_handoff_dataset_b_training_mpl",
        }
    )
    return environment


def _real_training_command() -> str:
    return (
        "CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
        "HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "
        f"{LEROBOT_TRAIN} --config_path {FULL_TRAINING_CONFIG}"
    )


def _model_compatibility_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Dataset B / SmolVLA compatibility",
            "",
            f"Status: **{report['status']}**",
            "",
            "## Interfaces",
            "",
            "- Pretrained logical source interface: state 6D, action 6D, three visual keys.",
            "- Dataset B interface: state 28D, action 28D, `observation.images.cam_high`.",
            "- Checkpoint projection architecture: state/action maximum dimension 32.",
            "- Dataset B is not reshaped to ALOHA 14D.",
            "",
            "## Adaptation",
            "",
            "LeRobot is configured with the authoritative 28D named G1 interface. "
            "SmolVLA pads those logical vectors to its unchanged 32D checkpoint projections, "
            "then slices training loss and predictions back to 28D. The pretrained weights "
            "therefore load without resizing a learned projection.",
            "",
            "Only source `cam_high` is a dataset observation. Two configured missing-camera "
            "keys are generated internally as fully masked empty tensors so the three-view "
            "pretrained visual convention remains explicit; they are not stored or synthesized "
            "camera observations.",
            "",
            "## Normalization",
            "",
            "State and action use Dataset-B `MEAN_STD` statistics from `meta/stats.json`. "
            "Visual input uses `IDENTITY`. The ALOHA/base-model action statistics are not reused.",
            "",
            "## Frozen full-run settings",
            "",
            f"- Dataset: `{report['dataset']['path']}`",
            f"- Pretrained snapshot: `{report['pretrained_model']['path']}`",
            f"- Output: `{report['full_training']['output_directory']}`",
            f"- Batch size: {report['full_training']['batch_size']}",
            f"- Learning rate: {report['full_training']['learning_rate']}",
            f"- Steps: {report['full_training']['steps']}",
            f"- Device/precision: {report['full_training']['device']} / AMP={report['full_training']['use_amp']}",
            "",
            "The real full-training command is prepared but was not executed.",
            "",
        ]
    )


def prepare_model_compatibility() -> dict[str, Any]:
    """Freeze the actual local LeRobot/SmolVLA interface and run a no-train preflight."""
    dataset_validation = load_json(FINAL_ROOT / "dataset_validation/validation.json")
    if dataset_validation.get("status") != "PASS":
        raise RuntimeError("Dataset B must pass full validation before model preflight")
    if not REFERENCE_G1_POLICY_CONFIG.is_file():
        raise FileNotFoundError(REFERENCE_G1_POLICY_CONFIG)
    if not BASE_MODEL.is_dir() or not LEROBOT_TRAIN.is_file():
        raise FileNotFoundError("pinned local SmolVLA snapshot or lerobot-train missing")
    source = load_json(REFERENCE_G1_POLICY_CONFIG)
    full = copy.deepcopy(source)
    full["dataset"]["repo_id"] = "local/doll_handoff_proposed_b_50"
    full["dataset"]["root"] = str(DATASET_ROOT.resolve())
    full["dataset"]["episodes"] = None
    full["dataset"]["video_backend"] = "torchcodec"
    full["output_dir"] = str(REAL_TRAINING_OUTPUT.resolve())
    full["job_name"] = "policy_b_doll_handoff_proposed_b_50_smolvla"
    full["batch_size"] = 16
    full["num_workers"] = 4
    full["steps"] = FULL_TRAINING_STEPS
    full["log_freq"] = 100
    full["save_freq"] = 5000
    full["seed"] = 1000
    full["optimizer"]["lr"] = 1e-4
    full["scheduler"]["peak_lr"] = 1e-4
    full["policy"]["optimizer_lr"] = 1e-4
    full["policy"]["pretrained_path"] = str(BASE_MODEL.resolve())
    full["policy"]["type"] = "smolvla"
    full["policy"]["device"] = "cuda"
    full["policy"]["use_amp"] = True
    full["policy"]["max_state_dim"] = 32
    full["policy"]["max_action_dim"] = 32
    full["policy"]["adapt_to_pi_aloha"] = False
    full["policy"]["use_delta_joint_actions_aloha"] = False
    full["policy"]["empty_cameras"] = 2
    full["policy"]["normalization_mapping"] = {
        "VISUAL": "IDENTITY",
        "STATE": "MEAN_STD",
        "ACTION": "MEAN_STD",
    }
    full["policy"]["input_features"] = {
        "observation.images.cam_high": {
            "shape": [3, 480, 640],
            "type": "VISUAL",
        },
        "observation.images.empty_camera_0": {
            "shape": [3, 480, 640],
            "type": "VISUAL",
        },
        "observation.images.empty_camera_1": {
            "shape": [3, 480, 640],
            "type": "VISUAL",
        },
        "observation.state": {"shape": [28], "type": "STATE"},
    }
    full["policy"]["output_features"] = {
        "action": {"shape": [28], "type": "ACTION"}
    }
    full["eval_steps"] = 0
    full["max_eval_samples"] = 0
    full["wandb"]["enable"] = False
    full["policy"]["push_to_hub"] = False
    atomic_json(FULL_TRAINING_CONFIG, full)
    smoke = copy.deepcopy(full)
    smoke["output_dir"] = str(SMOKE_OUTPUT.resolve())
    smoke["job_name"] = f"{full['job_name']}_SMOKE10_NOT_A_RESEARCH_POLICY"
    smoke["steps"] = SMOKE_STEPS
    smoke["log_freq"] = 1
    smoke["save_freq"] = SMOKE_STEPS
    atomic_json(SMOKE_TRAINING_CONFIG, smoke)

    help_result = subprocess.run(
        [str(LEROBOT_TRAIN), "--help"],
        cwd=REPOSITORY,
        env=_training_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    if help_result.returncode != 0:
        raise RuntimeError(f"lerobot-train --help failed: {help_result.stderr}")
    help_path = TRAINING_ROOT / "lerobot_train_help.txt"
    atomic_text(help_path, help_result.stdout + help_result.stderr)
    command_path = TRAINING_ROOT / "REAL_POLICY_B_TRAINING_COMMAND.txt"
    atomic_text(
        command_path,
        "# PREPARED ONLY. DO NOT RUN AS PART OF DATASET FINALIZATION.\n"
        + _real_training_command()
        + "\n",
    )
    preflight_command = [
        str(LEROBOT_PYTHON),
        str(REPOSITORY / "tools/audit_doll_handoff_dataset_b_smolvla.py"),
        "preflight",
        "--dataset",
        str(DATASET_ROOT),
        "--base-model",
        str(BASE_MODEL),
        "--config",
        str(FULL_TRAINING_CONFIG),
        "--config",
        str(SMOKE_TRAINING_CONFIG),
        "--output",
        str(MODEL_PREFLIGHT),
    ]
    completed = subprocess.run(
        preflight_command,
        cwd=REPOSITORY,
        env=_training_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_text(
        TRAINING_ROOT / "smolvla_preflight_stdout.log",
        completed.stdout + completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "SmolVLA preflight failed:\n" + completed.stdout + completed.stderr
        )
    preflight = load_json(MODEL_PREFLIGHT)
    schema = load_json(FINAL_ROOT / "training_schema/g1_training_schema.json")
    info = load_json(DATASET_ROOT / "meta/info.json")
    checks = {
        "dataset_validation_pass": dataset_validation["status"] == "PASS",
        "smolvla_preflight_pass": preflight["status"] == "PASS",
        "dataset_state_schema_28": info["features"]["observation.state"][
            "shape"
        ]
        == [28],
        "dataset_action_schema_28": info["features"]["action"]["shape"]
        == [28],
        "schema_unambiguous": schema["status"]
        == "PASS_UNAMBIGUOUS_TARGET_SCHEMA",
        "checkpoint_max_dims_32": preflight["checks"]["base_max_state_32"]
        and preflight["checks"]["base_max_action_32"],
        "no_aloha_14d_reshape": not preflight["adaptation"][
            "aloha_14d_reshape"
        ],
        "dataset_specific_normalization": preflight["normalization"][
            "dataset_specific_override_in_training"
        ],
        "real_training_not_started": not REAL_TRAINING_OUTPUT.exists(),
    }
    report = {
        "schema_version": "doll_handoff_dataset_b_model_compatibility_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pretrained_model": {
            "name": "lerobot/smolvla_base",
            "revision": BASE_MODEL_REVISION,
            "path": str(BASE_MODEL.resolve()),
            "model_sha256": sha256_file(BASE_MODEL / "model.safetensors"),
            "config_sha256": sha256_file(BASE_MODEL / "config.json"),
            "logical_source_observation_keys": preflight[
                "pretrained_logical_source_interface"
            ]["observation_keys"],
            "logical_source_state_dimension": 6,
            "logical_source_action_dimension": 6,
            "architecture_max_state_dimension": 32,
            "architecture_max_action_dimension": 32,
        },
        "dataset": {
            "path": str(DATASET_ROOT.resolve()),
            "observation_keys": preflight["dataset_interface"][
                "observation_keys"
            ],
            "state_dimension": 28,
            "action_dimension": 28,
            "joint_order": schema["action"]["joint_names"],
        },
        "required_config_changes": {
            "logical_input_features": full["policy"]["input_features"],
            "logical_output_features": full["policy"]["output_features"],
            "empty_cameras": 2,
            "max_state_dim_unchanged": 32,
            "max_action_dim_unchanged": 32,
            "adapt_to_pi_aloha": False,
            "use_delta_joint_actions_aloha": False,
            "normalization_mapping": full["policy"]["normalization_mapping"],
            "dataset_statistics_path": str(
                (DATASET_ROOT / "meta/stats.json").resolve()
            ),
        },
        "adaptation_mechanism": preflight["adaptation"],
        "full_training": {
            "policy_type": "smolvla",
            "pretrained_path": str(BASE_MODEL.resolve()),
            "dataset_path": str(DATASET_ROOT.resolve()),
            "output_directory": str(REAL_TRAINING_OUTPUT.resolve()),
            "batch_size": 16,
            "learning_rate": 1e-4,
            "steps": FULL_TRAINING_STEPS,
            "device": "cuda",
            "use_amp": True,
            "normalization": "Dataset-B state/action MEAN_STD; visual IDENTITY",
            "config_path": str(FULL_TRAINING_CONFIG.resolve()),
            "exact_command": _real_training_command(),
            "executed": False,
        },
        "provenance": {
            "reference_g1_config": str(REFERENCE_G1_POLICY_CONFIG.resolve()),
            "reference_g1_config_sha256": sha256_file(
                REFERENCE_G1_POLICY_CONFIG
            ),
            "full_config_sha256": sha256_file(FULL_TRAINING_CONFIG),
            "smoke_config_sha256": sha256_file(SMOKE_TRAINING_CONFIG),
            "lerobot_train_help_sha256": sha256_file(help_path),
            "lerobot_modeling_source": str(
                LEROBOT_SOURCE / "policies/smolvla/modeling_smolvla.py"
            ),
            "lerobot_modeling_source_sha256": sha256_file(
                LEROBOT_SOURCE / "policies/smolvla/modeling_smolvla.py"
            ),
            "lerobot_training_source_sha256": sha256_file(
                LEROBOT_SOURCE / "scripts/lerobot_train.py"
            ),
            "preflight_path": str(MODEL_PREFLIGHT.resolve()),
            "preflight_sha256": sha256_file(MODEL_PREFLIGHT),
        },
    }
    if report["status"] != "PASS":
        raise RuntimeError(f"model compatibility failed: {checks}")
    atomic_json(MODEL_COMPATIBILITY, report)
    atomic_text(
        TRAINING_ROOT / "model_compatibility.md",
        _model_compatibility_markdown(report),
    )
    return {
        "status": report["status"],
        "model_compatibility": str(MODEL_COMPATIBILITY.resolve()),
        "model_compatibility_sha256": sha256_file(MODEL_COMPATIBILITY),
        "full_training_config": str(FULL_TRAINING_CONFIG.resolve()),
        "full_training_config_sha256": sha256_file(FULL_TRAINING_CONFIG),
        "smoke_training_config": str(SMOKE_TRAINING_CONFIG.resolve()),
        "smoke_training_config_sha256": sha256_file(SMOKE_TRAINING_CONFIG),
        "real_training_command": _real_training_command(),
        "real_training_executed": False,
    }


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*:\s*"
    r"(?P<value>(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan)[KMB]?)",
    re.IGNORECASE,
)
STEP_RE = re.compile(
    r"^(?P<value>\d+(?:\.\d+)?)(?P<suffix>[KMB]?)$", re.IGNORECASE
)


def _parse_smoke_metrics(log_path: Path) -> list[dict[str, Any]]:
    raw = ANSI_RE.sub("", log_path.read_text(encoding="utf-8", errors="replace"))
    aliases = {
        "loss": "loss",
        "lr": "learning_rate",
        "grdn": "gradient_norm",
        "grad_norm": "gradient_norm",
        "gpu_mem": "gpu_memory_gb",
        "gpu_mem_gb": "gpu_memory_gb",
        "update_s": "update_seconds",
        "data_s": "data_seconds",
    }
    rows: list[dict[str, Any]] = []
    for segment in re.split(r"[\r\n]+", raw):
        pairs = {
            match.group("key").lower(): match.group("value")
            for match in METRIC_RE.finditer(segment)
        }
        if "step" not in pairs or "loss" not in pairs:
            continue
        match = STEP_RE.fullmatch(pairs["step"])
        if match is None:
            continue
        multipliers = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
        step = int(
            float(match.group("value"))
            * multipliers[match.group("suffix").upper()]
        )
        row: dict[str, Any] = {"step": step}
        for source, target in aliases.items():
            if source in pairs:
                row[target] = float(pairs[source])
        rows.append(row)
    if len(rows) == SMOKE_STEPS:
        for index, row in enumerate(rows, start=1):
            row["step"] = index
    return rows


def _gpu_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "gpu_query_returncode": gpu.returncode,
        "gpu": gpu.stdout.strip(),
        "compute_query_returncode": processes.returncode,
        "compute_processes": [
            line.strip() for line in processes.stdout.splitlines() if line.strip()
        ],
    }


def run_short_training_smoke() -> dict[str, Any]:
    """Run exactly ten optimization steps; never launch the real Policy-B run."""
    compatibility = load_json(MODEL_COMPATIBILITY)
    if compatibility.get("status") != "PASS":
        raise RuntimeError("SmolVLA compatibility must be PASS before smoke")
    if REAL_TRAINING_OUTPUT.exists():
        raise RuntimeError(
            f"real Policy-B output already exists; refusing to launch anything: {REAL_TRAINING_OUTPUT}"
        )
    if SMOKE_OUTPUT.exists():
        raise RuntimeError(
            f"smoke output already exists; refusing duplicate/overwrite: {SMOKE_OUTPUT}"
        )
    process_scan = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    conflicting_jobs = [
        line.strip()
        for line in process_scan
        if (
            "lerobot.scripts.lerobot_train" in line
            or "/lerobot-train" in line
        )
        and "ps -eo" not in line
    ]
    if conflicting_jobs:
        raise RuntimeError(f"active LeRobot training job detected: {conflicting_jobs}")
    gpu_before = _gpu_snapshot()
    if gpu_before["compute_processes"]:
        raise RuntimeError(
            "active GPU compute process detected before smoke: "
            f"{gpu_before['compute_processes']}"
        )
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = SMOKE_ROOT / "training.log"
    command = [str(LEROBOT_TRAIN), "--config_path", str(SMOKE_TRAINING_CONFIG)]
    started_at = now_iso()
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            env=_training_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    runtime = {
        "command": command,
        "config_path": str(SMOKE_TRAINING_CONFIG.resolve()),
        "start_timestamp": started_at,
        "end_timestamp": now_iso(),
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
        "gpu_before": gpu_before,
        "gpu_after": _gpu_snapshot(),
        "real_training_output_absent_before_and_after": not REAL_TRAINING_OUTPUT.exists(),
    }
    atomic_json(SMOKE_ROOT / "runtime.json", runtime)
    if returncode != 0:
        raise RuntimeError(
            f"10-step Dataset-B training smoke failed; inspect {log_path}"
        )
    checkpoint = SMOKE_OUTPUT / "checkpoints/000010/pretrained_model"
    if not checkpoint.is_dir():
        raise RuntimeError(f"smoke checkpoint missing: {checkpoint}")
    checkpoint_audit_path = SMOKE_ROOT / "checkpoint_audit.json"
    audit_command = [
        str(LEROBOT_PYTHON),
        str(REPOSITORY / "tools/audit_doll_handoff_dataset_b_smolvla.py"),
        "checkpoint",
        "--dataset",
        str(DATASET_ROOT),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(checkpoint_audit_path),
    ]
    audited = subprocess.run(
        audit_command,
        cwd=REPOSITORY,
        env=_training_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    atomic_text(
        SMOKE_ROOT / "checkpoint_audit_stdout.log",
        audited.stdout + audited.stderr,
    )
    if audited.returncode != 0:
        raise RuntimeError(
            "smoke checkpoint reload failed:\n" + audited.stdout + audited.stderr
        )
    checkpoint_audit = load_json(checkpoint_audit_path)
    metrics = _parse_smoke_metrics(log_path)
    losses = [float(row["loss"]) for row in metrics if "loss" in row]
    gradients = [
        float(row["gradient_norm"])
        for row in metrics
        if "gradient_norm" in row
    ]
    log_lower = log_path.read_text(encoding="utf-8", errors="replace").lower()
    checks = {
        "exactly_10_optimization_steps": len(metrics) == SMOKE_STEPS
        and [row["step"] for row in metrics] == list(range(1, SMOKE_STEPS + 1)),
        "all_losses_finite": len(losses) == SMOKE_STEPS
        and all(math.isfinite(value) for value in losses),
        "backward_gradient_finite": len(gradients) == SMOKE_STEPS
        and all(math.isfinite(value) for value in gradients),
        "dataloader_no_error": "dataloader" not in log_lower
        or "error" not in log_lower,
        "video_decode_no_error": "video" not in log_lower
        or "decode error" not in log_lower,
        "no_cuda_oom": "out of memory" not in log_lower
        and "cuda oom" not in log_lower,
        "no_tensor_shape_error": "shape mismatch" not in log_lower
        and "size mismatch" not in log_lower,
        "checkpoint_saved": checkpoint.is_dir(),
        "checkpoint_reload_forward_prediction_pass": checkpoint_audit["status"]
        == "PASS",
        "dataset_specific_normalization_in_checkpoint": checkpoint_audit[
            "checks"
        ]["dataset_specific_normalization_loaded"],
        "real_training_not_started": not REAL_TRAINING_OUTPUT.exists(),
    }
    summary = {
        "schema_version": "doll_handoff_dataset_b_training_smoke_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "research_policy": False,
        "label": "NOT_A_RESEARCH_POLICY",
        "checks": checks,
        "steps": SMOKE_STEPS,
        "batch_size": 16,
        "seed": 1000,
        "loss": {
            "per_logged_step": losses,
            "initial": losses[0] if losses else None,
            "final": losses[-1] if losses else None,
            "minimum": min(losses) if losses else None,
            "maximum": max(losses) if losses else None,
            "mean": float(np.mean(losses)) if losses else None,
        },
        "gradient_norm": {
            "per_logged_step": gradients,
            "maximum": max(gradients) if gradients else None,
        },
        "runtime": runtime,
        "training_log": str(log_path.resolve()),
        "training_log_sha256": sha256_file(log_path),
        "config": str(SMOKE_TRAINING_CONFIG.resolve()),
        "config_sha256": sha256_file(SMOKE_TRAINING_CONFIG),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "checkpoint_audit": checkpoint_audit,
        "checkpoint_audit_sha256": sha256_file(checkpoint_audit_path),
        "full_policy_b_training": "NOT_STARTED_BY_DESIGN",
    }
    atomic_json(SMOKE_ROOT / "smoke_result.json", summary)
    atomic_text(
        SMOKE_OUTPUT / "README_NOT_A_RESEARCH_POLICY.md",
        "# NOT A RESEARCH POLICY\n\n"
        "This checkpoint is the output of the authorized ten-step Dataset-B "
        "dataloader/forward/backward/save/reload smoke test only. It must not be "
        "reported, evaluated, or deployed as a trained research policy.\n",
    )
    if summary["status"] != "PASS":
        raise RuntimeError(f"Dataset-B smoke gate failed: {checks}")
    return summary


def _final_report_text(manifest: dict[str, Any]) -> str:
    new = manifest["new_source_episodes"]
    excluded = manifest["excluded_original_episodes"]
    schema = manifest["training_schema"]
    dataset = manifest["lerobot_dataset"]
    smoke = manifest["training_smoke"]
    counts = manifest["final_source_set"]["classification_counts"]
    joint_order = ", ".join(schema["action_joint_order"])
    return "\n".join(
        [
            "NEW SOURCE EPISODES",
            "",
            "GoPark_20260823_135848:",
            f"source audit: {new['GoPark_20260823_135848']['source_audit']}",
            f"B classification: {new['GoPark_20260823_135848']['classification']}",
            "",
            "GoPark_20260823_140035:",
            f"source audit: {new['GoPark_20260823_140035']['source_audit']}",
            f"B classification: {new['GoPark_20260823_140035']['classification']}",
            "",
            "",
            "REPLACED HARD FAILURES",
            "",
            "original ep013:",
            f"verified raw directory: {excluded['ep013']['raw_directory']}",
            f"reason: {excluded['ep013']['reason']}",
            "",
            "original ep036:",
            f"verified raw directory: {excluded['ep036']['raw_directory']}",
            f"reason: {excluded['ep036']['reason']}",
            "",
            "",
            "FINAL B SOURCE SET",
            "",
            f"episodes: {manifest['final_source_set']['episodes']}",
            f"CLEAN_PASS: {counts['CLEAN_PASS']}",
            f"USABLE_WITH_WARNING: {counts['USABLE_WITH_WARNING']}",
            f"HARD_FAIL: {counts['HARD_FAIL']}",
            "",
            f"manifest: {manifest['artifacts']['final_source_manifest']['path']}",
            f"SHA256: {manifest['artifacts']['final_source_manifest']['sha256']}",
            "",
            "",
            "PROPOSED-B ACTION LABELS",
            "",
            f"episodes: {manifest['action_labels']['episodes']}",
            f"frames: {manifest['action_labels']['frames']}",
            f"action dimension: {manifest['action_labels']['action_dimension']}",
            f"joint order: {joint_order}",
            f"trajectory-set SHA256: {manifest['action_labels']['trajectory_set_sha256']}",
            "",
            "",
            "G1 TRAINING SCHEMA",
            "",
            f"state dimension: {schema['state_dimension']}",
            f"state joint order: {joint_order}",
            "",
            f"action dimension: {schema['action_dimension']}",
            f"action joint order: {joint_order}",
            "",
            f"temporal convention: {schema['temporal_convention']}",
            "",
            "",
            "LEROBOT DATASET B",
            "",
            f"path: {dataset['path']}",
            f"episodes: {dataset['episodes']}",
            f"frames: {dataset['frames']}",
            f"RGB key: {dataset['rgb_key']}",
            f"state key: {dataset['state_key']}",
            f"action key: {dataset['action_key']}",
            f"task key: {dataset['task_key']}",
            "",
            f"validation: {dataset['validation']}",
            "",
            "",
            "MODEL COMPATIBILITY",
            "",
            f"pretrained model: {manifest['model_compatibility']['pretrained_model']}",
            f"state adaptation: {manifest['model_compatibility']['state_adaptation']}",
            f"action adaptation: {manifest['model_compatibility']['action_adaptation']}",
            f"normalization: {manifest['model_compatibility']['normalization']}",
            "",
            "",
            "TRAINING SMOKE",
            "",
            f"steps: {smoke['steps']}",
            f"loss: initial={smoke['initial_loss']}, final={smoke['final_loss']}",
            f"status: {smoke['status']}",
            f"output: {smoke['output']}",
            "",
            "",
            "FULL POLICY-B TRAINING",
            "",
            "NOT_STARTED_BY_DESIGN",
            "",
            "Exact command:",
            manifest["real_policy_b_training"]["exact_command"],
            "",
            "",
            "BASELINE A",
            "",
            "NOT_STARTED_BY_DESIGN",
            "",
            "DATASET_B_READY_FOR_POLICY_TRAINING",
            "",
        ]
    )


def freeze_final_dataset_b() -> dict[str, Any]:
    smoke = load_json(SMOKE_ROOT / "smoke_result.json")
    compatibility = load_json(MODEL_COMPATIBILITY)
    dataset_validation_path = FINAL_ROOT / "dataset_validation/validation.json"
    dataset_validation = load_json(dataset_validation_path)
    if not (
        smoke.get("status") == "PASS"
        and compatibility.get("status") == "PASS"
        and dataset_validation.get("status") == "PASS"
    ):
        raise RuntimeError("final freeze requires dataset, compatibility, and smoke PASS")
    if REAL_TRAINING_OUTPUT.exists():
        raise RuntimeError("real Policy-B training output exists; finalization must stop before it")
    packaging_path = DATASET_ROOT / "meta/g1_packaging_manifest.json"
    packaging = load_json(packaging_path)
    packaging["status"] = "DATASET_B_READY_TRAINING_SMOKE_PASSED"
    packaging["training_smoke"] = {
        "status": "PASS",
        "steps": SMOKE_STEPS,
        "output": str(SMOKE_OUTPUT.resolve()),
        "research_policy": False,
        "smoke_result_sha256": sha256_file(SMOKE_ROOT / "smoke_result.json"),
        "real_policy_training_started": False,
    }
    atomic_json(packaging_path, packaging)
    in_tree_validation_path = DATASET_ROOT / "meta/g1_validation.json"
    in_tree_validation = load_json(in_tree_validation_path)
    in_tree_validation["training_smoke_executed"] = True
    in_tree_validation["training_smoke_status"] = "PASS"
    in_tree_validation["training_smoke_steps"] = SMOKE_STEPS
    in_tree_validation["training_smoke_research_policy"] = False
    in_tree_validation["training_smoke_result_sha256"] = sha256_file(
        SMOKE_ROOT / "smoke_result.json"
    )
    in_tree_validation["real_policy_b_training_started"] = False
    atomic_json(in_tree_validation_path, in_tree_validation)
    tree_sha, tree_entries = _dataset_tree_sha256(DATASET_ROOT)
    audit_path = FINAL_ROOT / "dataset_validation/full_dataset_audit.json"
    audit = load_json(audit_path)
    audit["dataset_tree_sha256"] = tree_sha
    audit["dataset_tree_file_count"] = len(tree_entries)
    audit["dataset_tree_size_bytes"] = int(
        sum(entry["size_bytes"] for entry in tree_entries)
    )
    audit["dataset_tree_entries"] = tree_entries
    audit["training_smoke"] = {
        "status": "PASS",
        "steps": SMOKE_STEPS,
        "smoke_result_sha256": sha256_file(SMOKE_ROOT / "smoke_result.json"),
        "not_a_research_policy": True,
    }
    atomic_json(audit_path, audit)
    dataset_validation["dataset_tree_sha256"] = tree_sha
    dataset_validation["audit_sha256"] = sha256_file(audit_path)
    dataset_validation["training_smoke_status"] = "PASS"
    dataset_validation["training_smoke_steps"] = SMOKE_STEPS
    dataset_validation["real_policy_b_training_started"] = False
    atomic_json(dataset_validation_path, dataset_validation)

    source_path = FINAL_ROOT / "final_source_manifest.json"
    actions_path = FINAL_ROOT / "retargeted_actions/freeze_manifest.json"
    schema_path = FINAL_ROOT / "training_schema/g1_training_schema.json"
    normalization_path = (
        FINAL_ROOT / "normalization/dataset_b_state_action_normalization.json"
    )
    source = load_json(source_path)
    actions = load_json(actions_path)
    schema = load_json(schema_path)
    new_summary = load_json(FINAL_ROOT / "new_episode_conversion/summary.json")
    replaced = load_json(FINAL_ROOT / "replaced_hard_failures.json")
    new_audit = load_json(FINAL_ROOT / "new_source_audit/validation.json")
    individual_new = {
        row["source_name"]: row
        for row in new_summary["episodes"]
    }
    excluded_by_index = {
        f"ep{int(row['original_episode_index']):03d}": row
        for row in replaced["excluded_original_episodes"]
    }
    artifact_paths = {
        "final_source_manifest": source_path,
        "retargeted_action_freeze": actions_path,
        "training_schema_json": schema_path,
        "training_schema_markdown": FINAL_ROOT
        / "training_schema/g1_training_schema.md",
        "dataset_info": DATASET_ROOT / "meta/info.json",
        "dataset_tasks": DATASET_ROOT / "meta/tasks.parquet",
        "dataset_episodes": DATASET_ROOT
        / "meta/episodes/chunk-000/file-000.parquet",
        "dataset_statistics": DATASET_ROOT / "meta/stats.json",
        "normalization_statistics": normalization_path,
        "dataset_full_audit": audit_path,
        "lerobot_readback": FINAL_ROOT
        / "dataset_validation/lerobot_readback.json",
        "model_compatibility": MODEL_COMPATIBILITY,
        "smolvla_preflight": MODEL_PREFLIGHT,
        "full_training_config": FULL_TRAINING_CONFIG,
        "smoke_training_config": SMOKE_TRAINING_CONFIG,
        "smoke_result": SMOKE_ROOT / "smoke_result.json",
        "smoke_checkpoint_model": Path(smoke["checkpoint"])
        / "model.safetensors",
        "real_training_command": TRAINING_ROOT
        / "REAL_POLICY_B_TRAINING_COMMAND.txt",
    }
    artifacts = {
        key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for key, path in artifact_paths.items()
    }
    manifest = {
        "schema_version": "doll_handoff_final_dataset_b_manifest_v1",
        "status": "DATASET_B_READY_FOR_POLICY_TRAINING",
        "created_at": now_iso(),
        "task": "DOLL-HANDOFF-TO-BIN",
        "retargeting_method": "interaction_centric_proposed_b",
        "new_source_episodes": {
            name: {
                "source_audit": "PASS",
                "classification": individual_new[name]["classification"],
                "warning_reasons": individual_new[name][
                    "classification_reasons"
                ],
                "source_frames": individual_new[name]["frame_count"],
                "trajectory_sha256": individual_new[name][
                    "resolved_trajectory_sha256"
                ],
            }
            for name in NEW_SOURCE_NAMES
        },
        "new_source_audit": {
            "status": new_audit["status"],
            "manifest_sha256": sha256_file(
                FINAL_ROOT / "new_source_audit/new_episode_manifest.json"
            ),
            "validation_sha256": sha256_file(
                FINAL_ROOT / "new_source_audit/validation.json"
            ),
        },
        "excluded_original_episodes": {
            key: {
                "original_episode_index": row["original_episode_index"],
                "raw_directory": row["raw_directory"],
                "original_frozen_trajectory_sha256": row[
                    "frozen_proposed_b_trajectory_sha256"
                ],
                "original_resolved_trajectory_sha256": row[
                    "resolved_proposed_b_trajectory_sha256"
                ],
                "reason": "ARM_TORSO_HARD_COLLISION",
                "classification_reasons": row["hard_fail_reasons"],
                "exclusion_reason": row["exclusion_reason"],
                "raw_recording_deleted": False,
            }
            for key, row in excluded_by_index.items()
        },
        "replacement_episodes": list(NEW_SOURCE_NAMES),
        "final_source_set": {
            "episodes": int(source["source_count"]),
            "source_identities": [
                row["raw_directory"] for row in source["episodes"]
            ],
            "classification_counts": source["classification_counts"],
            "hard_fail_required": 0,
            "hard_fail_actual": source["classification_counts"]["HARD_FAIL"],
        },
        "action_labels": {
            "episodes": int(actions["episode_count"]),
            "frames": int(actions["total_frames"]),
            "action_dimension": int(actions["action_dimension"]),
            "joint_order": actions["joint_names"],
            "fps": actions["fps"],
            "trajectory_set_sha256": actions["trajectory_set_sha256"],
            "canonical_policy_action_set_sha256": actions[
                "canonical_policy_action_set_sha256"
            ],
        },
        "training_schema": {
            "state_dimension": schema["observation_state"]["dimension"],
            "state_joint_order": schema["observation_state"]["joint_names"],
            "action_dimension": schema["action"]["dimension"],
            "action_joint_order": schema["action"]["joint_names"],
            "same_controlled_joint_set": schema[
                "state_action_joint_set_relation"
            ],
            "temporal_convention": (
                "same-row absolute q_target[t] state/action at source frame t; "
                "row offset 0; no dropped boundary frames; 30 Hz"
            ),
            "state_is_measured_real_g1": False,
            "deployment_state_contract": schema["observation_state"][
                "deployment_adapter"
            ],
        },
        "lerobot_dataset": {
            "path": str(DATASET_ROOT.resolve()),
            "episodes": 50,
            "frames": 34478,
            "rgb_key": "observation.images.cam_high",
            "state_key": "observation.state",
            "action_key": "action",
            "task_key": "task",
            "task_instruction": TASK_INSTRUCTION,
            "validation": "PASS",
            "actual_lerobot_reader": "PASS",
            "dataset_tree_sha256": tree_sha,
            "dataset_tree_file_count": len(tree_entries),
            "dataset_tree_size_bytes": int(
                sum(entry["size_bytes"] for entry in tree_entries)
            ),
            "dataset_tree_entries": tree_entries,
        },
        "converter_provenance": {
            "converter_implementation_sha256": actions[
                "converter_implementation_sha256"
            ],
            "historical_cartesian_target_set_sha256": actions[
                "historical_frozen_cartesian_target_set_sha256"
            ],
            "common_natural_arm_solver_sha256": actions[
                "common_natural_arm_solver_sha256"
            ],
            "generic_feasibility_resolver_sha256": actions[
                "generic_feasibility_resolver_sha256"
            ],
            "generic_feasibility_resolver_config_sha256": actions[
                "generic_feasibility_resolver_config_sha256"
            ],
            "scene_layout_sha256": actions["scene_layout_sha256"],
            "handoff_cartesian_residual_m": actions[
                "handoff_cartesian_residual_m"
            ],
        },
        "model_compatibility": {
            "status": compatibility["status"],
            "pretrained_model": (
                "lerobot/smolvla_base@" + BASE_MODEL_REVISION
            ),
            "state_adaptation": (
                "logical G1 28D -> zero-pad to unchanged checkpoint max_state_dim=32"
            ),
            "action_adaptation": (
                "logical G1 28D -> checkpoint max_action_dim=32 internally; "
                "loss/prediction sliced to named 28D; no ALOHA 14D reshape"
            ),
            "normalization": (
                "Dataset-B state/action MEAN_STD from Dataset B; visual IDENTITY; "
                "ALOHA statistics not reused"
            ),
            "report_sha256": sha256_file(MODEL_COMPATIBILITY),
        },
        "normalization": {
            "state": "Dataset-B MEAN_STD",
            "action": "Dataset-B MEAN_STD",
            "finite": True,
            "statistics_path": str(normalization_path.resolve()),
            "statistics_sha256": sha256_file(normalization_path),
        },
        "training_smoke": {
            "steps": smoke["steps"],
            "losses": smoke["loss"]["per_logged_step"],
            "initial_loss": smoke["loss"]["initial"],
            "final_loss": smoke["loss"]["final"],
            "status": smoke["status"],
            "output": str(SMOKE_OUTPUT.resolve()),
            "checkpoint": smoke["checkpoint"],
            "label": "NOT_A_RESEARCH_POLICY",
            "research_policy": False,
        },
        "real_policy_b_training": {
            "status": "NOT_STARTED_BY_DESIGN",
            "output_directory_absent": not REAL_TRAINING_OUTPUT.exists(),
            "config": str(FULL_TRAINING_CONFIG.resolve()),
            "exact_command": _real_training_command(),
        },
        "baseline_a": {
            "conversion": "NOT_STARTED_BY_DESIGN",
            "dataset_packaging": "NOT_STARTED_BY_DESIGN",
            "policy_training": "NOT_STARTED_BY_DESIGN",
        },
        "artifacts": artifacts,
    }
    if not (
        manifest["final_source_set"]["episodes"] == 50
        and manifest["final_source_set"]["hard_fail_actual"] == 0
        and manifest["action_labels"]["episodes"] == 50
        and manifest["lerobot_dataset"]["frames"] == 34478
        and manifest["training_schema"]["state_dimension"] == 28
        and manifest["training_schema"]["action_dimension"] == 28
        and manifest["training_smoke"]["status"] == "PASS"
        and manifest["real_policy_b_training"]["output_directory_absent"]
    ):
        raise RuntimeError("final Dataset-B manifest invariant failure")
    final_path = FINAL_ROOT / "FINAL_DATASET_B_MANIFEST.json"
    atomic_json(final_path, manifest)
    final_sha = sha256_file(final_path)
    atomic_text(
        FINAL_ROOT / "FINAL_DATASET_B_MANIFEST.sha256",
        f"{final_sha}  FINAL_DATASET_B_MANIFEST.json\n",
    )
    report_path = FINAL_ROOT / "FINAL_REPORT.txt"
    atomic_text(report_path, _final_report_text(manifest))
    result = {
        "status": "DATASET_B_READY_FOR_POLICY_TRAINING",
        "manifest": str(final_path.resolve()),
        "manifest_sha256": final_sha,
        "report": str(report_path.resolve()),
        "dataset_tree_sha256": tree_sha,
        "training_smoke": "PASS",
        "real_policy_b_training": "NOT_STARTED_BY_DESIGN",
        "baseline_a": "NOT_STARTED_BY_DESIGN",
    }
    atomic_json(FINAL_ROOT / "final_freeze_validation.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "audit-new",
            "convert-new",
            "assemble-final",
            "schema",
            "package",
            "validate",
            "model-compatibility",
            "smoke",
            "final-freeze",
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "audit-new":
        result = audit_new_sources()
    elif args.stage == "convert-new":
        result = convert_new_sources()
    elif args.stage == "assemble-final":
        result = assemble_final_actions()
    elif args.stage == "schema":
        result = build_training_schema()
    elif args.stage == "package":
        result = package_lerobot_dataset()
    elif args.stage == "validate":
        result = validate_lerobot_dataset()
    elif args.stage == "model-compatibility":
        result = prepare_model_compatibility()
    elif args.stage == "smoke":
        result = run_short_training_smoke()
    elif args.stage == "final-freeze":
        result = freeze_final_dataset_b()
    else:  # pragma: no cover
        raise ValueError(args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
