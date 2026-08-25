#!/usr/bin/env python3
"""Build and structurally validate final-common-50 Fair Baseline Dataset A."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import atomic_json, sha256_file  # noqa: E402
from tools.doll_handoff_retargeting.pipeline import DollHandoffPipeline  # noqa: E402


FINAL_SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
EXPECTED_FINAL_SOURCE_MANIFEST_SHA256 = "8fd073d78cfe2e27075954cd43910b1cac43ccf7f373ea528866a82d3cd04f76"
COMMON_TEMPLATE = ROOT / "configs/doll_handoff_retargeting/common_config.template.json"
BASELINE_TEMPLATE = ROOT / "configs/doll_handoff_retargeting/baseline_config.template.json"
NATURAL_ARM_FREEZE = ROOT / "outputs/doll_handoff_retargeting/natural_arm_audit/frozen_common_natural_arm/freeze_manifest.json"
COMMON_SAFETY_FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
DATASET_B = ROOT / "datasets/doll_handoff_proposed_b_50"
RETARGET_ROOT = ROOT / "outputs/dataset_a_final50_retargeting"
DESTINATION = ROOT / "datasets/doll_handoff_trajectory_a_50"
REPORT_ROOT = ROOT / "outputs/pre_mount_readiness/dataset_a"
LEROBOT_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
TASK = "Pick up the doll with the left hand, handoff it to the right hand, and place it in the trash bin."


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def stats(value: np.ndarray) -> dict[str, Any]:
    value64 = np.asarray(value, dtype=np.float64)
    result = {
        "count": [int(len(value64))],
        "max": np.max(value64, axis=0).tolist(),
        "mean": np.mean(value64, axis=0).tolist(),
        "min": np.min(value64, axis=0).tolist(),
    }
    for percentile, key in ((0.01, "q01"), (0.10, "q10"), (0.50, "q50"), (0.90, "q90"), (0.99, "q99")):
        result[key] = np.quantile(value64, percentile, axis=0).tolist()
    result["std"] = np.std(value64, axis=0).tolist()
    return result


def fixed_list(value: np.ndarray) -> pa.FixedSizeListArray:
    value = np.ascontiguousarray(value, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1), type=pa.float32()), 28)


def tree_hash(root: Path, excluded: set[str] | None = None) -> tuple[str, list[dict[str, Any]]]:
    excluded = excluded or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), rows


def prepare_common_config() -> Path:
    if sha256_file(FINAL_SOURCE_MANIFEST) != EXPECTED_FINAL_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("the exact final common 50-source manifest hash changed")
    config = read_json(COMMON_TEMPLATE)
    config["source_selection_manifest"] = str(FINAL_SOURCE_MANIFEST)
    config["source_selection_manifest_sha256"] = EXPECTED_FINAL_SOURCE_MANIFEST_SHA256
    config["expected_source_count"] = 50
    config["status"] = "FAIR_BASELINE_A_FINAL_COMMON_50_FROZEN_INPUT"
    config["dataset_a_contract"] = {
        "method": "baseline",
        "same_common_temporal_ik_backend_as_ab_audit": True,
        "same_common_natural_arm_solver": str(NATURAL_ARM_FREEZE),
        "interaction_centric_b_information_allowed": False,
        "episode_specific_correction_allowed": False,
    }
    path = RETARGET_ROOT / "input_contract/common_config.final_common_50.json"
    atomic_json(path, config)
    return path


def run_retarget() -> dict[str, Any]:
    common_path = prepare_common_config()
    pipeline = DollHandoffPipeline(common_path=common_path, output_root=RETARGET_ROOT)
    source = pipeline.source_manifest
    if source["enumerated_count"] != 50 or source["valid_count"] != 50:
        raise RuntimeError(f"final-common source audit failed: {source['valid_count']}/50 valid")
    result = pipeline.run("baseline", range(50))
    return {"common_config": str(common_path), **result}


def collect_actions() -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    source = read_json(FINAL_SOURCE_MANIFEST)
    b_info = read_json(DATASET_B / "meta/info.json")
    canonical_names = list(b_info["features"]["action"]["names"])
    episode_rows = pq.read_table(DATASET_B / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    actions = []
    records = []
    for episode, (source_row, metadata) in enumerate(zip(source["episodes"], episode_rows, strict=True)):
        stable = str(source_row["stable_episode_id"])
        trajectory_path = RETARGET_ROOT / "baseline/trajectories" / f"{stable}.npz"
        validation_path = RETARGET_ROOT / "baseline/metrics" / f"{stable}.validation.json"
        manifest_path = RETARGET_ROOT / "baseline/metrics" / f"{stable}.manifest.json"
        validation = read_json(validation_path)
        manifest = read_json(manifest_path)
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            names = trajectory["replay_joint_names"].astype(str).tolist()
            replay = np.asarray(trajectory["replay_named_joint_qpos"], dtype=np.float32)
        if set(names) != set(canonical_names) or len(names) != 28:
            raise RuntimeError(f"episode {episode}: baseline trajectory has wrong named 28D schema")
        replay = replay[:, [names.index(name) for name in canonical_names]]
        expected_length = int(metadata["length"])
        if replay.shape != (expected_length, 28) or expected_length != int(source_row["source_frame_count"]):
            raise RuntimeError(f"episode {episode}: baseline/source/dataset frame identity failed")
        if not np.isfinite(replay).all():
            raise RuntimeError(f"episode {episode}: nonfinite baseline action")
        records.append(
            {
                "episode_index": episode,
                "source_raw_episode": source_row["raw_directory"],
                "stable_episode_id": stable,
                "frames": expected_length,
                "trajectory": str(trajectory_path),
                "trajectory_sha256": sha256_file(trajectory_path),
                "validation": str(validation_path),
                "validation_status": validation["status"],
                "kinematic_pass": bool(validation.get("pass")),
                "converter_manifest": str(manifest_path),
                "episode_specific_correction": False,
            }
        )
        actions.append(replay)
    return np.concatenate(actions, axis=0), records, canonical_names


def package_dataset() -> dict[str, Any]:
    if DESTINATION.exists() or DESTINATION.with_name(DESTINATION.name + ".incomplete").exists():
        raise FileExistsError(f"refusing to overwrite Dataset A: {DESTINATION}")
    actions, records, names = collect_actions()
    hard_fails = [row for row in records if not row["kinematic_pass"]]
    safety = read_json(COMMON_SAFETY_FREEZE)
    specs = safety["joint_specs"]
    if [row["joint_name"] for row in specs] != names:
        raise RuntimeError("common hard-limit definition does not match canonical 28D order")
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float32)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float32)
    hard_limit_excess = np.maximum(lower - actions, actions - upper)
    hard_limit_count = int(np.count_nonzero(hard_limit_excess > 1e-6))
    if hard_limit_count:
        hard_fails.append(
            {
                "type": "COMMON_HARD_LIMIT_VIOLATION",
                "scalar_count": hard_limit_count,
                "maximum_excess_rad": float(np.max(hard_limit_excess)),
            }
        )
    if hard_fails:
        categories: dict[str, int] = {}
        for row in hard_fails:
            key = str(row.get("validation_status", row.get("type", "UNKNOWN")))
            categories[key] = categories.get(key, 0) + 1
        report = {
            "schema_version": "doll_handoff_dataset_a_validation_v1",
            "status": "HARD_FAIL_STOPPED_BEFORE_PACKAGING",
            "episodes_attempted": 50,
            "source_episode_identity": "PASS_EXACT_FINAL_COMMON_50_ORDER",
            "structural_classification_counts": {
                "PASS": 50 - len(hard_fails),
                "WARNING": 0,
                "HARD_FAIL": len(hard_fails),
            },
            "hard_fail_categories": categories,
            "hard_fails": hard_fails,
            "silent_exclusions": 0,
            "dataset_created": False,
            "dataset_path": str(DESTINATION),
            "aggregate_converter_report": str(
                RETARGET_ROOT / "baseline/metrics/aggregate_summary.json"
            ),
            "next_gate": "POLICY_A_SOURCE_TRAINING_NOT_ALLOWED",
        }
        atomic_json(REPORT_ROOT / "dataset_a_validation.json", report)
        raise RuntimeError(f"Dataset A has {len(hard_fails)} true hard failures; packaging stopped")

    states = np.empty_like(actions)
    cursor = 0
    for row in records:
        length = int(row["frames"])
        episode_actions = actions[cursor : cursor + length]
        states[cursor] = episode_actions[0]
        states[cursor + 1 : cursor + length] = episode_actions[:-1]
        cursor += length
    if cursor != len(actions) or len(actions) != 34478:
        raise RuntimeError(f"unexpected total Dataset A frame count: {len(actions)}")

    source_table_path = DATASET_B / "data/chunk-000/file-000.parquet"
    table = pq.read_table(source_table_path)
    table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", fixed_list(states))
    table = table.set_column(table.schema.get_field_index("action"), "action", fixed_list(actions))
    staging = DESTINATION.with_name(DESTINATION.name + ".incomplete")
    (staging / "data/chunk-000").mkdir(parents=True)
    pq.write_table(table, staging / "data/chunk-000/file-000.parquet", compression="zstd")
    (staging / "meta/episodes/chunk-000").mkdir(parents=True)
    shutil.copy2(DATASET_B / "meta/episodes/chunk-000/file-000.parquet", staging / "meta/episodes/chunk-000/file-000.parquet")
    shutil.copy2(DATASET_B / "meta/tasks.parquet", staging / "meta/tasks.parquet")
    info = read_json(DATASET_B / "meta/info.json")
    info["robot_type"] = "unitree_g1_fixed_base_dex3_fair_baseline_a_retargeted"
    atomic_json(staging / "meta/info.json", info)
    atomic_json(staging / "meta/stats.json", {"action": stats(actions), "observation.state": stats(states)})

    source_videos = DATASET_B / "videos"
    hardlink_count = 0
    copy_count = 0
    for source_video in sorted(source_videos.rglob("*.mp4")):
        destination_video = staging / "videos" / source_video.relative_to(source_videos)
        destination_video.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_video, destination_video)
            hardlink_count += 1
        except OSError:
            shutil.copy2(source_video, destination_video)
            copy_count += 1

    contract = {
        "schema_version": "doll_handoff_dataset_a_training_contract_v1",
        "status": "FAIR_BASELINE_A_FROZEN",
        "method": "Trajectory-Centric Wrist-Level Retargeting",
        "common_source_manifest": str(FINAL_SOURCE_MANIFEST),
        "common_source_manifest_sha256": EXPECTED_FINAL_SOURCE_MANIFEST_SHA256,
        "observation_state_semantic": "RETARGETED_G1_STATE_SURROGATE: state[0]=action[0], state[t]=action[t-1] within each episode",
        "observation_state_is_measured_real_g1": False,
        "action_semantic": "absolute 28D G1 arm + Dex3 Fair Baseline A target",
        "action_chunk_size": 50,
        "joint_order": names,
        "source_visual_embodiment": "ALOHA",
        "interaction_centric_b_information_used": False,
        "episode_specific_correction_used": False,
        "common_natural_arm_solver": str(NATURAL_ARM_FREEZE),
        "common_natural_arm_solver_sha256": sha256_file(NATURAL_ARM_FREEZE),
        "common_hard_limit_definition": str(COMMON_SAFETY_FREEZE),
        "common_hard_limit_definition_sha256": sha256_file(COMMON_SAFETY_FREEZE),
    }
    atomic_json(staging / "meta/g1_training_contract.json", contract)
    manifest = {
        "schema_version": "doll_handoff_dataset_a_50_manifest_v1",
        "status": "PACKAGED_PENDING_LEROBOT_READBACK",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DESTINATION),
        "source_dataset_used_only_for_common_video_and_row_metadata": str(DATASET_B),
        "episodes": 50,
        "frames": len(actions),
        "fps": 30,
        "state_dimension": 28,
        "action_dimension": 28,
        "state_rule": "lag-1 within episode with frame-0 duplication",
        "action_sha256": array_sha256(actions),
        "state_sha256": array_sha256(states),
        "source_manifest": str(FINAL_SOURCE_MANIFEST),
        "source_manifest_sha256": EXPECTED_FINAL_SOURCE_MANIFEST_SHA256,
        "fair_baseline_config": str(BASELINE_TEMPLATE),
        "fair_baseline_config_sha256": sha256_file(BASELINE_TEMPLATE),
        "common_config": str(RETARGET_ROOT / "config/common_config.json"),
        "common_config_sha256": sha256_file(RETARGET_ROOT / "config/common_config.json"),
        "video_materialization": {"hardlinks": hardlink_count, "copies": copy_count},
        "hard_fails": [],
        "warnings": [
            "Dataset state is a retargeted lag-1 surrogate, not measured real-G1 state.",
            "Task success is not a Dataset-A acceptance criterion; Fair Baseline A was not altered to improve success.",
        ],
        "silent_exclusions": 0,
        "episodes_detail": records,
    }
    atomic_json(staging / "meta/dataset_a_manifest.json", manifest)
    os.rename(staging, DESTINATION)
    return manifest


def lerobot_readback() -> dict[str, Any]:
    output = REPORT_ROOT / "lerobot_readback.json"
    command = [
        str(LEROBOT_PYTHON),
        str(ROOT / "tools/validate_doll_handoff_dataset_b_lerobot.py"),
        "--dataset",
        str(DESTINATION),
        "--output",
        str(output),
        "--repo-id",
        "local/doll_handoff_trajectory_a_50",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return read_json(output)


def finalize_report(manifest: dict[str, Any], readback: dict[str, Any]) -> dict[str, Any]:
    source = read_json(FINAL_SOURCE_MANIFEST)
    dataset_manifest = read_json(DESTINATION / "meta/dataset_a_manifest.json")
    tree, entries = tree_hash(DESTINATION, {"meta/dataset_a_validation.json"})
    checks = {
        "episodes_50": manifest["episodes"] == 50 and readback["episode_count"] == 50,
        "same_source_episode_identities_as_b": [row["source_raw_episode"] for row in manifest["episodes_detail"]]
        == [row["raw_directory"] for row in source["episodes"]],
        "state_dimension_28": manifest["state_dimension"] == 28,
        "action_dimension_28": manifest["action_dimension"] == 28,
        "finite": True,
        "lag1_state_rule": manifest["state_rule"].startswith("lag-1"),
        "videos_load_all_50": readback["every_episode_middle_frame_read_count"] == 50,
        "lerobot_dataset_reads": readback["status"] == "PASS",
        "task_reads": readback["task_loading"] == "PASS",
        "hard_fails_zero": len(manifest["hard_fails"]) == 0,
        "silent_exclusions_zero": manifest["silent_exclusions"] == 0,
        "final_common_manifest_hash": manifest["source_manifest_sha256"] == EXPECTED_FINAL_SOURCE_MANIFEST_SHA256,
    }
    status = "PASS" if all(checks.values()) else "HARD_FAIL"
    report = {
        "schema_version": "doll_handoff_dataset_a_validation_v1",
        "status": status,
        "episodes": 50,
        "frames": manifest["frames"],
        "hard_fails": [] if status == "PASS" else [key for key, value in checks.items() if not value],
        "warnings": manifest["warnings"],
        "checks": checks,
        "dataset": str(DESTINATION),
        "dataset_manifest": str(DESTINATION / "meta/dataset_a_manifest.json"),
        "lerobot_readback": str(REPORT_ROOT / "lerobot_readback.json"),
        "content_tree_sha256_excluding_validation": tree,
        "content_tree_file_count_excluding_validation": len(entries),
        "dataset_manifest_identity": dataset_manifest["action_sha256"],
    }
    atomic_json(REPORT_ROOT / "dataset_a_validation.json", report)
    atomic_json(DESTINATION / "meta/dataset_a_validation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--retarget-only", action="store_true")
    args = parser.parse_args()
    if args.package_only and args.retarget_only:
        parser.error("choose at most one of --package-only and --retarget-only")
    retarget = None
    if not args.package_only:
        retarget = run_retarget()
    if args.retarget_only:
        print(json.dumps(retarget, indent=2))
        return 0
    manifest = package_dataset()
    readback = lerobot_readback()
    report = finalize_report(manifest, readback)
    print(json.dumps({"status": report["status"], "episodes": 50, "hard_fails": report["hard_fails"], "path": str(DESTINATION)}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
