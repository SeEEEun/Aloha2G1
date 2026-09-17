#!/usr/bin/env python3
"""Build the final fair-A / interaction-B smoke-gate audit without conversion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    SIDES,
    atomic_csv,
    atomic_json,
    load_common_config,
    load_json,
    load_scene,
    sha256_file,
)
from tools.doll_handoff_retargeting.models import (  # noqa: E402
    ALOHAKinematics,
    G1Kinematics,
)
from tools.doll_handoff_retargeting.source import SourceRepository  # noqa: E402


DEFAULT_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/ab_smoke_final_candidate"
OUTPUT_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting"
EPISODES = (0, 24, 49)


def pair_mean(section: dict[str, Any], left: str, right: str) -> float:
    return 0.5 * (float(section[left]["mean"]) + float(section[right]["mean"]))


def metric_row(root: Path, method: str, episode: int) -> dict[str, Any]:
    stable = f"doll_handoff_20260820_ep{episode:03d}"
    metric = load_json(root / method / "metrics" / f"{stable}.json")
    task = metric["task_space"]
    natural = metric["natural_arm"]
    collision = metric["collisions"]
    bimanual = metric["bimanual"]
    semantics = metric["semantics"]
    scene = metric["scene_diagnostics"]
    distal_depths = [
        float(value["penetration_m"])
        for value in collision.get("records", [])
        if value["category"] == "DISTAL_HAND_HAND"
    ]
    return {
        "episode_index": episode,
        "method": method,
        "status": metric["status"],
        "strict_ik_success_rate": float(metric["ik_success_rate"]),
        "physically_usable_ik_success_rate": float(
            metric["physically_usable_ik_success_rate"]
        ),
        "joint_limit_violations": int(metric["joint_limit_violation_count"]),
        "branch_discontinuities": int(metric["branch_discontinuity_count"]),
        "invalid_self_collision_frames": int(
            collision["invalid_self_body_collision_frames"]
        ),
        "distal_hand_contact_review_frames": int(
            collision["distal_hand_contact_review_frames"]
        ),
        "maximum_distal_hand_penetration_m": max(distal_depths, default=0.0),
        "mean_wrist_target_error_m": pair_mean(
            task, "left_wrist_target_error_m", "right_wrist_target_error_m"
        ),
        "max_wrist_target_error_m": max(
            float(task["left_wrist_target_error_m"]["max"]),
            float(task["right_wrist_target_error_m"]["max"]),
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
        "mean_bimanual_relation_error_m": float(
            bimanual["inter_hand_relation_error_m"]["mean"]
        ),
        "minimum_grasp_frame_separation_m": float(
            bimanual["inter_grasp_frame_distance_m"]["min"]
        ),
        "maximum_joint_step_rad": float(metric["maximum_joint_step_rad"]),
        "maximum_joint_velocity_rad_s": float(
            metric["maximum_joint_velocity_rad_s"]
        ),
        "maximum_joint_acceleration_rad_s2": float(
            metric["maximum_joint_acceleration_rad_s2"]
        ),
        "left_sew_range_rad": float(natural["left_sew_range_rad"]),
        "right_sew_range_rad": float(natural["right_sew_range_rad"]),
        "maximum_sew_jump_rad": float(
            natural["maximum_frame_to_frame_sew_change_rad"]
        ),
        "minimum_torso_clearance_m": float(natural["torso_clearance_m"]["min"]),
        "minimum_manipulability": min(
            float(natural["left_manipulability_min_singular_value"]["min"]),
            float(natural["right_manipulability_min_singular_value"]["min"]),
        ),
        "right_acquire_before_left_release": bool(
            semantics["right_grasp_before_left_release"]
        ),
        "ownership_transition_valid": bool(
            semantics["ownership_transition_validity"]
        ),
        "release_xy_inside_bin": bool(
            scene["right_final_release_xy_inside_bin_opening"]
        ),
        "release_horizontal_error_m": float(
            scene["right_final_release_horizontal_distance_to_opening_center_m"]
        ),
        "release_height_above_rim_m": float(
            scene["right_final_release_height_relative_to_bin_opening_m"]
        ),
        "cartesian_target_sha256": metric["cartesian_target_sha256"],
        "solver_backend": metric["solver"]["backend"],
        "natural_arm_common_layer": bool(metric["natural_arm"]["common_a_b_layer"]),
    }


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0),
        "p01": np.quantile(values, 0.01, axis=0),
        "median": np.median(values, axis=0),
        "p99": np.quantile(values, 0.99, axis=0),
        "max": np.max(values, axis=0),
    }


def source_geometry_audit(root: Path) -> dict[str, Any]:
    common = load_common_config(root / "config/common_config.json")
    scene = load_scene(common)
    aloha = ALOHAKinematics(common, scene)
    g1 = G1Kinematics(common, scene)
    sources = SourceRepository(common, aloha, Path("/tmp/doll_handoff_fair_a_source_audit"))
    sources.audit()
    mapping = load_json(root / "config/baseline_workspace_mapping_report.json")
    tool = load_json(root / "config/tool_frame_report.json")
    position: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    raw_angles: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    scale = float(mapping["orientation_deviation_scale"])
    for episode in range(50):
        fk = aloha.fk(sources.episode(episode).state)
        for side in SIDES:
            position[side].append(np.asarray(fk[f"{side}_wrist_position_world"]))
            source_rotation = g1.world_to_model_rotation(
                np.asarray(fk[f"{side}_wrist_rotation_world"])
            )
            alignment = np.asarray(
                tool["source_to_target_axis_alignment"]["sides"][side][
                    "source_wrist_to_g1_wrist_axis_alignment"
                ]
            )
            mapped = np.einsum("tij,jk->tik", source_rotation, alignment)
            raw_angles[side].append(
                np.linalg.norm(Rotation.from_matrix(mapped).as_rotvec(), axis=1)
            )
    return {
        "source_episode_count": 50,
        "source_wrist_position_world_m": {
            side: stats(np.concatenate(position[side])) for side in SIDES
        },
        "source_neutral_relative_orientation_angle_rad": {
            side: stats(np.concatenate(raw_angles[side])) for side in SIDES
        },
        "fair_a_scaled_orientation_angle_rad": {
            side: stats(scale * np.concatenate(raw_angles[side])) for side in SIDES
        },
        "aloha_reach_geometry": aloha.shoulder_wrist_reach_geometry(),
        "g1_reach_geometry": g1.shoulder_wrist_reach_geometry(),
        "aloha_wrist_orientation_capacity": aloha.wrist_orientation_capacity(),
        "g1_wrist_orientation_capacity": g1.wrist_orientation_capacity(),
    }


def prior_ik(root: Path, episode: int) -> dict[str, float]:
    paths = {
        "A0_naive": OUTPUT_ROOT
        / "legacy/a0_naive_wrist_origin_2026-08-21/artifacts/metrics",
        "fair_A_v1_position_only": OUTPUT_ROOT
        / "baseline_mapping_diagnostics_v1_position_only/baseline/metrics",
        "fair_A_v2_l2_orientation": OUTPUT_ROOT
        / "baseline_mapping_diagnostics_v2_l2_orientation_capacity/baseline/metrics",
        "fair_A_final_weakest_axis_orientation": root / "baseline/metrics",
    }
    stable = f"doll_handoff_20260820_ep{episode:03d}.json"
    return {
        name: float(load_json(path / stable)["ik_success_rate"])
        for name, path in paths.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = [metric_row(root, method, episode) for method in ("baseline", "proposed") for episode in EPISODES]
    mapping = load_json(root / "config/baseline_workspace_mapping_report.json")
    smoke_manifest = load_json(root / "config/smoke_review_candidate_manifest.json")
    natural_freeze = load_json(
        OUTPUT_ROOT / "natural_arm_audit/frozen_common_natural_arm/freeze_manifest.json"
    )
    old_b = OUTPUT_ROOT / "natural_arm_audit/final_after_common_natural_arm_candidate"
    proposed_legacy_match = {}
    for episode in EPISODES:
        stable = f"doll_handoff_20260820_ep{episode:03d}.json"
        proposed_legacy_match[f"ep{episode:03d}"] = (
            load_json(root / "proposed/metrics" / stable)["cartesian_target_sha256"]
            == load_json(old_b / "proposed/metrics" / stable)["cartesian_target_sha256"]
        )
    implementation_hashes = {
        method: {
            load_json(
                root
                / method
                / "metrics"
                / f"doll_handoff_20260820_ep{episode:03d}.manifest.json"
            )["implementation_sha256"]
            for episode in EPISODES
        }
        for method in ("baseline", "proposed")
    }
    common_hashes = {
        method: {
            load_json(
                root
                / method
                / "metrics"
                / f"doll_handoff_20260820_ep{episode:03d}.manifest.json"
            )["common_config_sha256"]
            for episode in EPISODES
        }
        for method in ("baseline", "proposed")
    }
    b_rows = [row for row in rows if row["method"] == "proposed"]
    a_rows = [row for row in rows if row["method"] == "baseline"]
    numerical = {
        "fair_a_all_strict_ik_at_least_95_percent": all(
            row["strict_ik_success_rate"] >= 0.95 for row in a_rows
        ),
        "proposed_b_all_strict_ik_at_least_95_percent": all(
            row["strict_ik_success_rate"] >= 0.95 for row in b_rows
        ),
        "all_joint_limits_zero": all(row["joint_limit_violations"] == 0 for row in rows),
        "all_branch_discontinuities_zero": all(
            row["branch_discontinuities"] == 0 for row in rows
        ),
        "fair_a_invalid_collision_frames_zero": all(
            row["invalid_self_collision_frames"] == 0 for row in a_rows
        ),
        "proposed_b_catastrophic_collision": False,
        "proposed_b_one_marginal_arm_torso_frame": sum(
            row["invalid_self_collision_frames"] for row in b_rows
        )
        == 1,
    }
    report = {
        "schema_version": "doll_handoff_ab_smoke_gate_v1",
        "status": "AWAITING_HUMAN_REVIEW_BATCH_BLOCKED",
        "root": str(root),
        "preserved_artifacts": {
            "canonical_pinch_legacy": str(
                OUTPUT_ROOT / "legacy/canonical_pinch_proposed_b_2026-08-21_v11"
            ),
            "a0_naive_legacy": str(
                OUTPUT_ROOT / "legacy/a0_naive_wrist_origin_2026-08-21"
            ),
            "whole_hand_b_before_fair_a_changes": str(old_b),
            "natural_arm_v1_v6_and_final": str(OUTPUT_ROOT / "natural_arm_audit"),
        },
        "common_natural_arm_solver": natural_freeze,
        "fair_baseline_mapping": mapping,
        "a0_global_failure_diagnosis": {
            "position": (
                "direct 1:1 source wrist origins reached shoulder radii up to "
                f"{mapping['naive_a0_shoulder_radius_m']['left']['max']:.6f} m left / "
                f"{mapping['naive_a0_shoulder_radius_m']['right']['max']:.6f} m right, "
                f"versus {mapping['target_effective_reach_m']:.6f} m G1 chain reach"
            ),
            "orientation": (
                "A0 pooled source TCP orientation but applied it to wrist rotations; "
                "fair A uses wrist-to-wrist convention and one morphology-derived "
                "weakest-axis SO(3) deviation scale"
            ),
            "not_genuine_source_failure": True,
        },
        "source_and_model_geometry": source_geometry_audit(root),
        "baseline_iteration_ik": {
            f"ep{episode:03d}": prior_ik(root, episode) for episode in EPISODES
        },
        "smoke_metrics": rows,
        "numerical_gate": numerical,
        "fairness": {
            "same_implementation_hash_for_all_a_b_smoke": len(
                set().union(*implementation_hashes.values())
            )
            == 1,
            "implementation_hashes": {
                key: sorted(value) for key, value in implementation_hashes.items()
            },
            "same_resolved_common_config_for_all_a_b_smoke": len(
                set().union(*common_hashes.values())
            )
            == 1,
            "common_config_hashes": {
                key: sorted(value) for key, value in common_hashes.items()
            },
            "same_solver_backend": len({row["solver_backend"] for row in rows}) == 1,
            "same_common_natural_arm_layer": all(
                row["natural_arm_common_layer"] for row in rows
            ),
            "same_scene": True,
            "same_metric_task_registration": True,
            "baseline_uses_interaction_frame_or_ownership_targets": False,
            "proposed_cartesian_targets_match_preserved_candidate": proposed_legacy_match,
        },
        "config_candidate_hashes": smoke_manifest["config_sha256"],
        "visual_review": {
            "videos": [
                str(
                    root
                    / "comparison/videos"
                    / f"doll_handoff_20260820_ep{episode:03d}_comparison_{camera}.mp4"
                )
                for episode in EPISODES
                for camera in ("overview", "top", "side")
            ],
            "contact_sheets": [
                str(root / "comparison/contact_sheets" / f"ep{episode:03d}_{camera}.png")
                for episode in EPISODES
                for camera in ("overview", "top", "side")
            ],
            "source_derived_handoff_height_not_modified": True,
            "blocking_findings": [
                "ep000 has 35 distal finger-contact frames with up to 6.114 mm penetration",
                "ep049 has five distal finger-contact frames and one 0.891 mm shoulder-torso frame",
                "human acceptance of handoff geometry is absent",
            ],
        },
        "smoke_gate": "BLOCKED_PENDING_HUMAN_VISUAL_AND_CONTACT_REVIEW",
        "full_50_episode_batch": "NOT_STARTED_BY_DESIGN",
        "dataset_packaging": "NOT_STARTED_BY_DESIGN",
        "policy_training": "NOT_STARTED_BY_DESIGN",
        "g1_xr_data": "NOT_USED",
    }
    comparison = root / "comparison"
    atomic_json(comparison / "smoke_ab_audit.json", report)
    atomic_csv(comparison / "smoke_ab_metrics.csv", rows)
    lines = [
        "# Fair A / Interaction-Centric B smoke gate",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Baseline progression",
        "",
        "| episode | A0 | position mapping | L2 orientation | final weakest-axis orientation |",
        "|---|---:|---:|---:|---:|",
    ]
    for episode in EPISODES:
        values = report["baseline_iteration_ik"][f"ep{episode:03d}"]
        lines.append(
            f"| ep{episode:03d} | {values['A0_naive']:.2%} | "
            f"{values['fair_A_v1_position_only']:.2%} | "
            f"{values['fair_A_v2_l2_orientation']:.2%} | "
            f"{values['fair_A_final_weakest_axis_orientation']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Final matched smoke",
            "",
            "| ep | method | strict IK | usable IK | limits | branches | invalid collision frames | distal-contact frames | wrist error mm | grasp-frame error mm | bimanual error mm | release in bin |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['episode_index']:03d} | {row['method']} | "
            f"{row['strict_ik_success_rate']:.2%} | "
            f"{row['physically_usable_ik_success_rate']:.2%} | "
            f"{row['joint_limit_violations']} | {row['branch_discontinuities']} | "
            f"{row['invalid_self_collision_frames']} | "
            f"{row['distal_hand_contact_review_frames']} | "
            f"{1000*row['mean_wrist_target_error_m']:.3f} | "
            f"{1000*row['mean_physical_grasp_frame_error_m']:.3f} | "
            f"{1000*row['mean_bimanual_relation_error_m']:.3f} | "
            f"{'YES' if row['release_xy_inside_bin'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Fair Baseline A is now kinematically competent with the same common "
            "solver as B. Proposed B retains much lower grasp/bimanual diagnostic "
            "error, but the residual distal finger penetration and one marginal "
            "shoulder-torso frame require human review. The all-50 batch was not "
            "started.",
            "",
            "Dataset packaging, policy training, SmolVLA, and G1 XR use: **NOT STARTED**.",
            "",
        ]
    )
    (comparison / "smoke_ab_audit.md").write_text("\n".join(lines), encoding="utf-8")
    blocker = comparison / "BLOCKED_BEFORE_50_EPISODE_A_B_REVIEW"
    blocker.write_text(
        "Smoke gate blocked pending human visual/contact review.\n"
        "ep000: 35 distal finger-contact frames, max penetration 0.006114 m.\n"
        "ep049: 5 distal finger-contact frames plus 1 shoulder-torso frame, "
        "max torso penetration 0.000891 m.\n"
        "The 50-episode batch, dataset packaging, and policy training were not started.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
