#!/usr/bin/env python3
"""Validate and summarize the frozen Proposed-B-only 50-episode review batch."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


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


ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
FREEZE = ROOT / "frozen_approval"
REVIEW = ROOT / "review"
EXPECTED_EPISODES = tuple(range(50))
SMOKE_EPISODES = (0, 24, 49)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pair_mean(section: dict[str, Any], left: str, right: str) -> float:
    return 0.5 * (float(section[left]["mean"]) + float(section[right]["mean"]))


def finite_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "median", "std", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def normalized_common(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for key in ("status", "runtime_fingerprint", "smoke_execution", "smoke_execution_integrity_pass"):
        result.pop(key, None)
    source = result.get("resolved", {}).get("source_audit", {})
    if "manifest_path" in source:
        source["manifest_path"] = "<OUTPUT_ROOT>/source_audit/source_manifest.json"
    return result


def normalized_proposed(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for key in ("status", "runtime_fingerprint", "smoke_execution", "smoke_execution_integrity_pass"):
        result.pop(key, None)
    result.get("resolved", {}).pop("common_runtime_fingerprint", None)
    return result


def failure_reasons(metric: dict[str, Any], validation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    thresholds = validation["thresholds"]
    collision = metric["collisions"]
    semantics = metric["semantics"]
    if not bool(metric["finite"]):
        reasons.append(f"NONFINITE_VALUES:{metric['nan_inf_count']}")
    if int(metric["joint_limit_violation_count"]):
        reasons.append(f"JOINT_LIMIT_VIOLATIONS:{metric['joint_limit_violation_count']}")
    if float(metric["ik_success_rate"]) < float(thresholds["required_ik_success_rate"]):
        reasons.append(
            "IK_SUCCESS_BELOW_REQUIRED:"
            f"{float(metric['ik_success_rate']):.9f}<"
            f"{float(thresholds['required_ik_success_rate']):.9f}"
        )
    if int(metric["branch_discontinuity_count"]):
        reasons.append(f"BRANCH_DISCONTINUITIES:{metric['branch_discontinuity_count']}")
    if float(metric["maximum_joint_step_rad"]) > float(thresholds["maximum_joint_step_rad"]):
        reasons.append(
            f"MAX_JOINT_STEP:{metric['maximum_joint_step_rad']:.9f}>"
            f"{thresholds['maximum_joint_step_rad']:.9f}"
        )
    if float(metric["maximum_joint_velocity_rad_s"]) > float(
        thresholds["maximum_velocity_rad_s"]
    ):
        reasons.append(
            f"MAX_JOINT_VELOCITY:{metric['maximum_joint_velocity_rad_s']:.9f}>"
            f"{thresholds['maximum_velocity_rad_s']:.9f}"
        )
    if float(metric["maximum_joint_acceleration_rad_s2"]) > float(
        thresholds["maximum_acceleration_rad_s2"]
    ):
        reasons.append(
            f"MAX_JOINT_ACCELERATION:{metric['maximum_joint_acceleration_rad_s2']:.9f}>"
            f"{thresholds['maximum_acceleration_rad_s2']:.9f}"
        )
    invalid = int(collision["invalid_self_body_collision_frames"])
    if invalid:
        categories = [
            f"{name}={count}"
            for name, count in collision["frame_counts"].items()
            if name != "DISTAL_HAND_HAND" and int(count)
        ]
        reasons.append(f"INVALID_SELF_COLLISION_FRAMES:{invalid}:" + ",".join(categories))
    if not bool(semantics["source_semantic_valid"]):
        reasons.append("SOURCE_SEMANTIC_ANOMALY:" + ",".join(semantics["source_anomalies"]))
    if not bool(semantics["right_grasp_before_left_release"]):
        reasons.append("RIGHT_ACQUIRE_NOT_BEFORE_LEFT_RELEASE")
    if not bool(semantics["ownership_transition_validity"]):
        reasons.append("OWNERSHIP_TRANSITION_INVALID")
    return reasons or ["PASS_ALL_FROZEN_GATES"]


def episode_row(episode: int) -> tuple[dict[str, Any], dict[str, Any]]:
    stable = f"doll_handoff_20260820_ep{episode:03d}"
    metric_path = ROOT / "proposed/metrics" / f"{stable}.json"
    validation_path = ROOT / "proposed/metrics" / f"{stable}.validation.json"
    manifest_path = ROOT / "proposed/metrics" / f"{stable}.manifest.json"
    trajectory_path = ROOT / "proposed/trajectories" / f"{stable}.npz"
    metric = load_json(metric_path)
    validation = load_json(validation_path)
    manifest = load_json(manifest_path)
    task = metric["task_space"]
    bimanual = metric["bimanual"]
    collision = metric["collisions"]
    semantics = metric["semantics"]
    scene = metric["scene_diagnostics"]
    natural = metric["natural_arm"]
    records = collision.get("records", [])
    penetrations = [float(record["penetration_m"]) for record in records]
    reasons = failure_reasons(metric, validation)
    failed_checks = [name for name, passed in validation["checks"].items() if not passed]
    row = {
        "episode_index": episode,
        "stable_episode_id": stable,
        "source_name": metric["source_name"],
        "frame_count": int(metric["frame_count"]),
        "status": metric["status"],
        "conversion_attempted": bool(metric["conversion_attempted"]),
        "explicit_result": "PASS" if metric["status"] == "PASS" else "FAIL",
        "failure_reason_codes": ";".join(reasons),
        "failed_validation_checks": ";".join(failed_checks),
        "strict_ik_success_rate": float(metric["ik_success_rate"]),
        "physically_usable_ik_success_rate": float(
            metric["physically_usable_ik_success_rate"]
        ),
        "mean_ik_task_error_m": float(metric["mean_ik_task_error_m"]),
        "max_ik_task_error_m": float(metric["max_ik_task_error_m"]),
        "nan_inf_count": int(metric["nan_inf_count"]),
        "joint_limit_violation_count": int(metric["joint_limit_violation_count"]),
        "branch_discontinuity_count": int(metric["branch_discontinuity_count"]),
        "maximum_joint_step_rad": float(metric["maximum_joint_step_rad"]),
        "maximum_joint_velocity_rad_s": float(metric["maximum_joint_velocity_rad_s"]),
        "maximum_joint_acceleration_rad_s2": float(
            metric["maximum_joint_acceleration_rad_s2"]
        ),
        "maximum_joint_jerk_rad_s3": float(metric["maximum_joint_jerk_rad_s3"]),
        "invalid_self_collision_frames": int(
            collision["invalid_self_body_collision_frames"]
        ),
        "distal_hand_contact_review_frames": int(
            collision["distal_hand_contact_review_frames"]
        ),
        "all_self_penetration_frames": int(collision["all_self_penetration_frames"]),
        "arm_torso_collision_frames": int(collision["frame_counts"]["ARM_TORSO"]),
        "cross_arm_collision_frames": int(collision["frame_counts"]["CROSS_ARM"]),
        "wrist_palm_torso_collision_frames": int(
            collision["frame_counts"]["WRIST_OR_PALM_TORSO"]
        ),
        "maximum_penetration_m": max(penetrations, default=0.0),
        "mean_wrist_target_error_m": pair_mean(
            task, "left_wrist_target_error_m", "right_wrist_target_error_m"
        ),
        "mean_wrist_orientation_error_rad": pair_mean(
            task,
            "left_wrist_orientation_error_rad",
            "right_wrist_orientation_error_rad",
        ),
        "mean_physical_grasp_frame_error_m": pair_mean(
            task,
            "left_physical_grasp_frame_target_error_m",
            "right_physical_grasp_frame_target_error_m",
        ),
        "mean_static_grasp_objective_error_m": pair_mean(
            task,
            "left_static_canonical_grasp_frame_target_error_m",
            "right_static_canonical_grasp_frame_target_error_m",
        ),
        "bimanual_midpoint_error_m": float(bimanual["midpoint_error_m"]["mean"]),
        "bimanual_relative_vector_error_m": float(
            bimanual["relative_vector_error_m"]["mean"]
        ),
        "bimanual_relation_error_m": float(
            bimanual["inter_hand_relation_error_m"]["mean"]
        ),
        "minimum_grasp_frame_separation_m": float(
            bimanual["inter_grasp_frame_distance_m"]["min"]
        ),
        "handoff_order_valid": bool(semantics["right_grasp_before_left_release"]),
        "ownership_transition_valid": bool(
            semantics["ownership_transition_validity"]
        ),
        "source_semantic_valid": bool(semantics["source_semantic_valid"]),
        "dual_hold_frames": int(semantics["dual_hold_frames"]),
        "dual_hold_sec": float(semantics["dual_hold_sec"]),
        "release_xy_inside_bin": bool(
            scene["right_final_release_xy_inside_bin_opening"]
        ),
        "release_horizontal_error_m": float(
            scene["right_final_release_horizontal_distance_to_opening_center_m"]
        ),
        "release_height_relative_to_bin_opening_m": float(
            scene["right_final_release_height_relative_to_bin_opening_m"]
        ),
        "left_grasp_frame_to_doll_m": float(
            scene["target_left_grasp_frame_to_doll_at_left_grasp_m"]
        ),
        "handoff_grasp_frame_distance_m": float(
            scene["inter_grasp_frame_distance_at_handoff_m"]
        ),
        "maximum_sew_jump_rad": float(
            natural["maximum_frame_to_frame_sew_change_rad"]
        ),
        "minimum_torso_clearance_m": float(natural["torso_clearance_m"]["min"]),
        "minimum_manipulability": min(
            float(natural["left_manipulability_min_singular_value"]["min"]),
            float(natural["right_manipulability_min_singular_value"]["min"]),
        ),
        "cartesian_target_sha256": metric["cartesian_target_sha256"],
        "trajectory_sha256": sha256_file(trajectory_path),
        "metrics_sha256": sha256_file(metric_path),
        "implementation_sha256": manifest["implementation_sha256"],
        "scene_config_sha256": manifest["scene_config_sha256"],
        "dataset_packaging": bool(manifest["dataset_packaging"]),
        "policy_training": bool(manifest["policy_training"]),
    }
    integrity = {
        "episode_index": episode,
        "manifest_episode_index_matches": int(manifest["episode_index"]) == episode,
        "stable_id_matches": manifest["stable_episode_id"] == stable,
        "conversion_attempted": bool(manifest["conversion_attempted"]),
        "trajectory_exists": trajectory_path.is_file(),
        "trajectory_sha_matches_manifest": sha256_file(trajectory_path)
        == manifest["trajectory_sha256"],
        "metrics_sha_matches_manifest": sha256_file(metric_path)
        == manifest["metrics_sha256"],
        "status_matches_validation": metric["status"] == validation["status"],
        "status_matches_manifest": metric["status"] == manifest["status"],
    }
    return row, integrity


def smoke_array_integrity() -> dict[str, Any]:
    approved = REPOSITORY / "outputs/doll_handoff_retargeting/ab_smoke_final_candidate"
    keys = (
        "g1_arm_qpos",
        "left_dex3_qpos",
        "right_dex3_qpos",
        "target_left_wrist_position_model",
        "target_right_wrist_position_model",
        "target_left_interaction_frame_position_world",
        "target_right_interaction_frame_position_world",
        "achieved_left_physical_grasp_frame_position_world",
        "achieved_right_physical_grasp_frame_position_world",
        "ownership_state",
    )
    result: dict[str, Any] = {}
    for episode in SMOKE_EPISODES:
        stable = f"doll_handoff_20260820_ep{episode:03d}.npz"
        with np.load(approved / "proposed/trajectories" / stable, allow_pickle=False) as old, np.load(
            ROOT / "proposed/trajectories" / stable, allow_pickle=False
        ) as new:
            key_match = {key: bool(np.array_equal(old[key], new[key])) for key in keys}
        result[f"ep{episode:03d}"] = {
            "all_protected_arrays_identical": all(key_match.values()),
            "per_array": key_match,
        }
    return result


def artifact_row_pass(item: dict[str, Any]) -> bool:
    return all(bool(value) for key, value in item.items() if key != "episode_index")


def main() -> int:
    freeze = load_json(FREEZE / "freeze_manifest.json")
    current_implementation, _ = implementation_fingerprint()
    rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    for episode in EXPECTED_EPISODES:
        row, integrity = episode_row(episode)
        rows.append(row)
        integrity_rows.append(integrity)

    approved_common = load_json(FREEZE / "config/common_config.json")
    approved_proposed = load_json(FREEZE / "config/proposed_config.json")
    batch_common = load_json(ROOT / "config/common_config.json")
    batch_proposed = load_json(ROOT / "config/proposed_config.json")
    protected = {
        "normalized_common_config_identical": canonical_hash(
            normalized_common(approved_common)
        )
        == canonical_hash(normalized_common(batch_common)),
        "normalized_proposed_config_identical": canonical_hash(
            normalized_proposed(approved_proposed)
        )
        == canonical_hash(normalized_proposed(batch_proposed)),
        "task_frame_report_byte_identical": sha256_file(
            FREEZE / "config/task_frame_report.json"
        )
        == sha256_file(ROOT / "config/task_frame_report.json"),
        "tool_frame_report_byte_identical": sha256_file(
            FREEZE / "config/tool_frame_report.json"
        )
        == sha256_file(ROOT / "config/tool_frame_report.json"),
        "event_detector_config_byte_identical": sha256_file(
            FREEZE / "config/event_detector_config.json"
        )
        == sha256_file(ROOT / "event_audit/detector_config.json"),
        "scene_layout_still_identical": sha256_file(FREEZE / "scene/scene_layout.json")
        == sha256_file(REPOSITORY / "isaaclab_doll_handoff_scene/scene_layout.json"),
        "dex3_whole_hand_geometry_still_identical": sha256_file(
            FREEZE / "config/dex3_whole_hand.sim.json"
        )
        == sha256_file(
            REPOSITORY / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json"
        ),
        "common_natural_arm_solver_still_identical": sha256_file(
            FREEZE / "config/common_natural_arm_solver.json"
        )
        == sha256_file(
            REPOSITORY
            / "outputs/doll_handoff_retargeting/natural_arm_audit/frozen_common_natural_arm/common_natural_arm_solver.json"
        ),
        "implementation_identical": current_implementation
        == freeze["implementation_sha256"],
        "all_episode_implementation_hashes_identical": all(
            row["implementation_sha256"] == freeze["implementation_sha256"] for row in rows
        ),
        "all_episode_scene_hashes_identical": all(
            row["scene_config_sha256"]
            == freeze["approved_resolved_file_sha256"]["scene_layout"]
            for row in rows
        ),
        "handoff_cartesian_residual_zero": float(
            batch_proposed["handoff_cartesian_residual"]["active_offset_m"]
        )
        == 0.0,
        "episode_specific_parameters_disabled": not bool(
            batch_proposed["resolved"]["interaction_and_ownership_semantics"][
                "episode_specific_parameters"
            ]
        ),
        "baseline_directory_absent": not (ROOT / "baseline").exists(),
        "dataset_packaging_not_run": all(not row["dataset_packaging"] for row in rows),
        "policy_training_not_run": all(not row["policy_training"] for row in rows),
    }
    smoke_integrity = smoke_array_integrity()
    status_counts = Counter(row["status"] for row in rows)
    failure_code_counts: Counter[str] = Counter()
    for row in rows:
        if row["status"] != "PASS":
            for reason in row["failure_reason_codes"].split(";"):
                failure_code_counts[reason.split(":", 1)[0]] += 1
    metric_names = (
        "strict_ik_success_rate",
        "physically_usable_ik_success_rate",
        "mean_ik_task_error_m",
        "max_ik_task_error_m",
        "invalid_self_collision_frames",
        "distal_hand_contact_review_frames",
        "maximum_penetration_m",
        "mean_wrist_target_error_m",
        "mean_wrist_orientation_error_rad",
        "mean_physical_grasp_frame_error_m",
        "mean_static_grasp_objective_error_m",
        "bimanual_midpoint_error_m",
        "bimanual_relative_vector_error_m",
        "bimanual_relation_error_m",
        "minimum_grasp_frame_separation_m",
        "release_horizontal_error_m",
        "release_height_relative_to_bin_opening_m",
        "left_grasp_frame_to_doll_m",
        "handoff_grasp_frame_distance_m",
        "maximum_joint_velocity_rad_s",
        "maximum_joint_acceleration_rad_s2",
        "maximum_joint_jerk_rad_s3",
        "maximum_sew_jump_rad",
        "minimum_torso_clearance_m",
        "minimum_manipulability",
    )
    integrity_pass = (
        len(rows) == 50
        and all(row["conversion_attempted"] for row in rows)
        and all(artifact_row_pass(item) for item in integrity_rows)
        and all(protected.values())
        and all(
            item["all_protected_arrays_identical"] for item in smoke_integrity.values()
        )
    )
    summary = {
        "schema_version": "interaction_centric_proposed_b_50_review_v1",
        "status": "COMPLETE" if integrity_pass else "FAIL_INTEGRITY",
        "scope": "PROPOSED_B_ONLY",
        "attempted_episode_count": len(rows),
        "authoritative_source_count": 50,
        "status_counts": dict(sorted(status_counts.items())),
        "pass_count": status_counts["PASS"],
        "fail_count": len(rows) - status_counts["PASS"],
        "failure_episode_indices": [
            row["episode_index"] for row in rows if row["status"] != "PASS"
        ],
        "failure_reason_code_counts": dict(sorted(failure_code_counts.items())),
        "aggregate_metrics": {
            name: finite_stats(float(row[name]) for row in rows) for name in metric_names
        },
        "counts": {
            "finite_episodes": sum(row["nan_inf_count"] == 0 for row in rows),
            "joint_limit_clean_episodes": sum(
                row["joint_limit_violation_count"] == 0 for row in rows
            ),
            "branch_clean_episodes": sum(
                row["branch_discontinuity_count"] == 0 for row in rows
            ),
            "invalid_collision_clean_episodes": sum(
                row["invalid_self_collision_frames"] == 0 for row in rows
            ),
            "strict_ik_at_least_95_percent": sum(
                row["strict_ik_success_rate"] >= 0.95 for row in rows
            ),
            "physically_usable_ik_at_least_95_percent": sum(
                row["physically_usable_ik_success_rate"] >= 0.95 for row in rows
            ),
            "handoff_order_valid": sum(row["handoff_order_valid"] for row in rows),
            "ownership_transition_valid": sum(
                row["ownership_transition_valid"] for row in rows
            ),
            "source_semantic_valid": sum(row["source_semantic_valid"] for row in rows),
            "release_xy_inside_bin": sum(row["release_xy_inside_bin"] for row in rows),
            "total_invalid_self_collision_frames": sum(
                row["invalid_self_collision_frames"] for row in rows
            ),
            "total_distal_hand_contact_review_frames": sum(
                row["distal_hand_contact_review_frames"] for row in rows
            ),
            "total_frames": sum(row["frame_count"] for row in rows),
        },
        "protected_configuration_integrity": protected,
        "protected_configuration_integrity_pass": all(protected.values()),
        "smoke_reproduction": smoke_integrity,
        "artifact_integrity_pass": all(
            artifact_row_pass(item) for item in integrity_rows
        ),
        "overall_integrity_pass": integrity_pass,
        "baseline_a": "NOT_RUN_BY_DESIGN",
        "dataset_packaging": "NOT_STARTED_BY_DESIGN",
        "policy_training": "NOT_STARTED_BY_DESIGN",
        "g1_xr_data": "NOT_USED",
    }
    failures = [
        {
            "episode_index": row["episode_index"],
            "stable_episode_id": row["stable_episode_id"],
            "source_name": row["source_name"],
            "status": row["status"],
            "failure_reason_codes": row["failure_reason_codes"].split(";"),
            "failed_validation_checks": row["failed_validation_checks"].split(";")
            if row["failed_validation_checks"]
            else [],
            "strict_ik_success_rate": row["strict_ik_success_rate"],
            "invalid_self_collision_frames": row["invalid_self_collision_frames"],
            "distal_hand_contact_review_frames": row[
                "distal_hand_contact_review_frames"
            ],
            "release_xy_inside_bin": row["release_xy_inside_bin"],
        }
        for row in rows
        if row["status"] != "PASS"
    ]
    REVIEW.mkdir(parents=True, exist_ok=True)
    atomic_csv(REVIEW / "proposed_b_per_episode.csv", rows)
    atomic_json(REVIEW / "proposed_b_aggregate_summary.json", summary)
    atomic_json(REVIEW / "proposed_b_failures.json", failures)
    atomic_json(REVIEW / "artifact_integrity.json", integrity_rows)
    lines = [
        "# Interaction-Centric Proposed B — 50-episode review",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"Attempted: **{len(rows)} / 50**",
        f"PASS: **{status_counts['PASS']}**",
        f"FAIL_IK: **{status_counts['FAIL_IK']}**",
        f"FAIL_COLLISION: **{status_counts['FAIL_COLLISION']}**",
        "",
        "## Aggregate",
        "",
        f"- Mean strict IK: `{summary['aggregate_metrics']['strict_ik_success_rate']['mean']:.4%}`",
        f"- Median strict IK: `{summary['aggregate_metrics']['strict_ik_success_rate']['median']:.4%}`",
        f"- Mean physical grasp-frame diagnostic error: `{1000*summary['aggregate_metrics']['mean_physical_grasp_frame_error_m']['mean']:.3f} mm`",
        f"- Mean static grasp objective error: `{1000*summary['aggregate_metrics']['mean_static_grasp_objective_error_m']['mean']:.3f} mm`",
        f"- Mean bimanual relation error: `{1000*summary['aggregate_metrics']['bimanual_relation_error_m']['mean']:.3f} mm`",
        f"- Handoff order valid: `{summary['counts']['handoff_order_valid']} / 50`",
        f"- Ownership transition valid: `{summary['counts']['ownership_transition_valid']} / 50`",
        f"- Release XY inside bin: `{summary['counts']['release_xy_inside_bin']} / 50`",
        f"- Invalid body-collision-clean episodes: `{summary['counts']['invalid_collision_clean_episodes']} / 50`",
        f"- Total invalid body-collision frames: `{summary['counts']['total_invalid_self_collision_frames']}`",
        f"- Total distal hand-contact review frames: `{summary['counts']['total_distal_hand_contact_review_frames']}`",
        "",
        "## Explicit failures",
        "",
        "| episode | status | strict IK | invalid collision frames | reasons |",
        "|---:|---|---:|---:|---|",
    ]
    for item in failures:
        lines.append(
            f"| {item['episode_index']:03d} | {item['status']} | "
            f"{item['strict_ik_success_rate']:.2%} | "
            f"{item['invalid_self_collision_frames']} | "
            f"{'<br>'.join(item['failure_reason_codes'])} |"
        )
    lines.extend(
        [
            "",
            "## Freeze integrity",
            "",
            f"Protected configuration integrity: **{'PASS' if all(protected.values()) else 'FAIL'}**",
            f"All artifact checksums/manifests: **{'PASS' if summary['artifact_integrity_pass'] else 'FAIL'}**",
            "Smoke ep000/024/049 protected arrays: **"
            + ("IDENTICAL" if all(item["all_protected_arrays_identical"] for item in smoke_integrity.values()) else "DIFFER")
            + "**",
            "",
            "Baseline A, dataset packaging, and policy training: **NOT RUN**.",
            "",
        ]
    )
    (REVIEW / "proposed_b_50_episode_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if integrity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
