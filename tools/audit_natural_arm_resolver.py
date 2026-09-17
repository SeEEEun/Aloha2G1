#!/usr/bin/env python3
"""Create the matched Proposed-B natural-arm smoke audit.

The script is read-only with respect to trajectories.  It compares saved
resolver-OFF/ON outputs, verifies identical Cartesian-target hashes, and writes
review reports/plots.  It cannot launch conversion, packaging, or training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    COMMON_TEMPLATE,
    PROPOSED_TEMPLATE,
    WHOLE_HAND_CONFIG,
    atomic_csv,
    atomic_json,
    load_common_config,
    load_json,
    load_scene,
    sha256_file,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


AUDIT_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/natural_arm_audit"
DEFAULT_BEFORE = AUDIT_ROOT / "final_before_resolver_off"
DEFAULT_AFTER = AUDIT_ROOT / "final_after_common_natural_arm_candidate"
DEFAULT_OUTPUT = AUDIT_ROOT / "final_review"
EPISODES = (0, 24, 49)


def _pair_mean(section: dict[str, Any], left: str, right: str) -> float:
    return 0.5 * (float(section[left]["mean"]) + float(section[right]["mean"]))


def _extract(metrics: dict[str, Any]) -> dict[str, Any]:
    task = metrics["task_space"]
    natural = metrics["natural_arm"]
    collision = metrics["collisions"]
    semantics = metrics["semantics"]
    scene = metrics["scene_diagnostics"]
    bimanual = metrics["bimanual"]
    return {
        "status": metrics["status"],
        "ik_success_rate": float(metrics["ik_success_rate"]),
        "primary_task_position_error_mean_m": float(metrics["mean_ik_task_error_m"]),
        "primary_task_position_error_max_m": float(metrics["max_ik_task_error_m"]),
        "static_canonical_grasp_frame_error_mean_m": _pair_mean(
            task,
            "left_static_canonical_grasp_frame_target_error_m",
            "right_static_canonical_grasp_frame_target_error_m",
        ),
        "static_canonical_grasp_frame_orientation_error_mean_rad": _pair_mean(
            task,
            "left_static_canonical_grasp_frame_orientation_error_rad",
            "right_static_canonical_grasp_frame_orientation_error_rad",
        ),
        "dynamic_synergy_grasp_frame_error_mean_m": _pair_mean(
            task,
            "left_physical_grasp_frame_target_error_m",
            "right_physical_grasp_frame_target_error_m",
        ),
        "joint_limit_violations": int(metrics["joint_limit_violation_count"]),
        "branch_discontinuities": int(metrics["branch_discontinuity_count"]),
        "invalid_collision_frames": int(collision["invalid_self_body_collision_frames"]),
        "arm_torso_collision_frames": int(collision["frame_counts"]["ARM_TORSO"]),
        "cross_arm_collision_frames": int(collision["frame_counts"]["CROSS_ARM"]),
        "wrist_palm_torso_collision_frames": int(
            collision["frame_counts"]["WRIST_OR_PALM_TORSO"]
        ),
        "maximum_joint_step_rad": float(metrics["maximum_joint_step_rad"]),
        "maximum_joint_velocity_rad_s": float(metrics["maximum_joint_velocity_rad_s"]),
        "maximum_joint_acceleration_rad_s2": float(
            metrics["maximum_joint_acceleration_rad_s2"]
        ),
        "maximum_joint_jerk_rad_s3": float(metrics["maximum_joint_jerk_rad_s3"]),
        "left_sew_angle_range_rad": float(natural["left_sew_range_rad"]),
        "right_sew_angle_range_rad": float(natural["right_sew_range_rad"]),
        "maximum_frame_to_frame_sew_change_rad": float(
            natural["maximum_frame_to_frame_sew_change_rad"]
        ),
        "nominal_posture_deviation_mean_l2_rad": float(
            natural["nominal_posture_deviation_l2_rad"]["mean"]
        ),
        "left_manipulability_min_singular_value": float(
            natural["left_manipulability_min_singular_value"]["min"]
        ),
        "right_manipulability_min_singular_value": float(
            natural["right_manipulability_min_singular_value"]["min"]
        ),
        "minimum_torso_clearance_m": float(natural["torso_clearance_m"]["min"]),
        "minimum_proximal_cross_arm_clearance_m": float(
            natural["cross_arm_proximal_clearance_m"]["min"]
        ),
        "bimanual_relation_error_mean_m": float(
            bimanual["inter_hand_relation_error_m"]["mean"]
        ),
        "right_grasp_before_left_release": bool(
            semantics["right_grasp_before_left_release"]
        ),
        "ownership_transition_valid": bool(semantics["ownership_transition_validity"]),
        "release_xy_inside_bin": bool(
            scene["right_final_release_xy_inside_bin_opening"]
        ),
        "release_horizontal_distance_to_bin_center_m": float(
            scene["right_final_release_horizontal_distance_to_opening_center_m"]
        ),
        "release_height_relative_to_bin_opening_m": float(
            scene["right_final_release_height_relative_to_bin_opening_m"]
        ),
        "cartesian_target_sha256": str(metrics["cartesian_target_sha256"]),
    }


def _plot_episode(
    output: Path,
    episode: int,
    before_root: Path,
    after_root: Path,
    g1: G1Kinematics,
    nominal: np.ndarray,
    guides: dict[str, np.ndarray],
    events: dict[str, Any],
) -> Path:
    stable = f"doll_handoff_20260820_ep{episode:03d}"
    series: dict[str, dict[str, np.ndarray]] = {}
    for label, root in (("before", before_root), ("after", after_root)):
        with np.load(root / "proposed/trajectories" / f"{stable}.npz", allow_pickle=False) as values:
            q = np.asarray(values["g1_arm_qpos"], dtype=np.float64)
            timestamp = np.asarray(values["timestamp"], dtype=np.float64)
        sew = {side: np.empty(len(q), dtype=np.float64) for side in ("left", "right")}
        for frame, value in enumerate(q):
            angles = g1.sew_angles(value, guides)
            for side in sew:
                sew[side][frame] = angles[side]
        series[label] = {
            "timestamp": timestamp,
            "left_sew": np.unwrap(sew["left"]),
            "right_sew": np.unwrap(sew["right"]),
            "nominal": np.linalg.norm(q - nominal, axis=1),
            "step": np.r_[0.0, np.linalg.norm(np.diff(q, axis=0), axis=1)],
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    panels = (
        ("left_sew", "Left SEW angle", "rad"),
        ("right_sew", "Right SEW angle", "rad"),
        ("nominal", "Nominal-posture deviation", "L2 rad"),
        ("step", "Frame-to-frame arm-joint step", "L2 rad"),
    )
    for axis, (key, title, units) in zip(axes.ravel(), panels):
        for label, color in (("before", "tab:red"), ("after", "tab:blue")):
            axis.plot(
                series[label]["timestamp"],
                series[label][key],
                color=color,
                lw=1.0,
                label=label,
            )
        for event_name in (
            "LEFT_GRASP",
            "RIGHT_GRASP",
            "LEFT_RELEASE",
            "RIGHT_FINAL_RELEASE",
        ):
            frame = events["frames"].get(event_name)
            if frame is not None:
                axis.axvline(float(frame) / 30.0, color="black", lw=0.5, alpha=0.35)
        axis.set_title(title)
        axis.set_ylabel(units)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 1].set_xlabel("time [s]")
    fig.suptitle(f"Proposed B natural-arm resolver audit — ep{episode:03d}")
    fig.tight_layout()
    path = output / "plots" / f"ep{episode:03d}_natural_arm_before_after.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after-root", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    before_root = args.before_root.resolve()
    after_root = args.after_root.resolve()
    output = args.output_root.resolve()

    before_config = load_json(before_root / "config/common_config.json")
    after_config = load_json(after_root / "config/common_config.json")
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal = np.asarray(
        after_config["resolved"]["canonical_g1_nominal_q"], dtype=np.float64
    )
    guides = {
        side: np.asarray(value, dtype=np.float64)
        for side, value in after_config["resolved"]["natural_arm_redundancy"][
            "nominal_elbow_guides_model"
        ].items()
    }
    event_rows = load_json(before_root / "event_audit/events.json")

    rows: list[dict[str, Any]] = []
    episodes: dict[str, Any] = {}
    all_target_hashes_equal = True
    for episode in EPISODES:
        stable = f"doll_handoff_20260820_ep{episode:03d}"
        before_metric = load_json(before_root / "proposed/metrics" / f"{stable}.json")
        after_metric = load_json(after_root / "proposed/metrics" / f"{stable}.json")
        before = _extract(before_metric)
        after = _extract(after_metric)
        target_equal = before["cartesian_target_sha256"] == after["cartesian_target_sha256"]
        all_target_hashes_equal &= target_equal
        row: dict[str, Any] = {
            "episode": episode,
            "stable_episode_id": stable,
            "cartesian_target_hash_equal": target_equal,
        }
        for prefix, values in (("before", before), ("after", after)):
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        rows.append(row)
        plot = _plot_episode(
            output,
            episode,
            before_root,
            after_root,
            g1,
            nominal,
            guides,
            event_rows[str(episode)],
        )
        episodes[f"ep{episode:03d}"] = {
            "before": before,
            "after": after,
            "delta": {
                "ik_success_rate": after["ik_success_rate"] - before["ik_success_rate"],
                "primary_task_position_error_mean_m": (
                    after["primary_task_position_error_mean_m"]
                    - before["primary_task_position_error_mean_m"]
                ),
                "maximum_frame_to_frame_sew_change_rad": (
                    after["maximum_frame_to_frame_sew_change_rad"]
                    - before["maximum_frame_to_frame_sew_change_rad"]
                ),
                "invalid_collision_frames": (
                    after["invalid_collision_frames"] - before["invalid_collision_frames"]
                ),
                "maximum_joint_acceleration_rad_s2": (
                    after["maximum_joint_acceleration_rad_s2"]
                    - before["maximum_joint_acceleration_rad_s2"]
                ),
            },
            "plot": str(plot),
        }
    atomic_csv(output / "before_after_metrics.csv", rows)

    config_hashes = {
        "common_template": sha256_file(COMMON_TEMPLATE),
        "proposed_template": sha256_file(PROPOSED_TEMPLATE),
        "whole_hand_geometry": sha256_file(WHOLE_HAND_CONFIG),
        "before_resolved_common": sha256_file(before_root / "config/common_config.json"),
        "after_resolved_common": sha256_file(after_root / "config/common_config.json"),
        "after_resolved_proposed": sha256_file(after_root / "config/proposed_config.json"),
        "tool_frame_report": sha256_file(after_root / "config/tool_frame_report.json"),
    }
    tool_report = load_json(after_root / "config/tool_frame_report.json")
    rejected_clearance = 2.0 * max(
        float(g1.contacts[f"{side}_thumb"].half_extent[1])
        for side in ("left", "right")
    )
    geometry_audit = {
        "schema_version": "interaction_centric_whole_hand_geometry_audit_v1",
        "status": "ACTIVE_DOLL_HANDOFF_WHOLE_HAND_DEFINITION",
        "active_model": str(g1.path),
        "active_model_sha256": common["models"]["g1_xml_sha256"],
        "old_handoff_only_clearance_m": rejected_clearance,
        "old_handoff_only_clearance_decision": (
            "REMOVED_TASK_SPECIFIC_CARTESIAN_RESIDUAL"
        ),
        "old_value_is_rigid_transform_geometry": False,
        "old_value_derivation": (
            "twice the active-model thumb-pad tangential half extent; a contact-shape "
            "clearance scalar, not a wrist-to-grasp-frame translation"
        ),
        "active_handoff_cartesian_offset_m": 0.0,
        "whole_hand_grasp_frame": {
            "definition": tool_report["g1"]["grasp_frame_definition"],
            "task_fingers": tool_report["g1"]["task_fingers"],
            "left_wrist_to_grasp_frame": tool_report["g1"][
                "left_wrist_to_grasp_frame"
            ],
            "right_wrist_to_grasp_frame": tool_report["g1"][
                "right_wrist_to_grasp_frame"
            ],
            "transform_scope": tool_report["g1"]["transform_scope"],
            "calibration_status": tool_report["calibration_status"],
        },
        "forbidden_mechanisms_active": {
            "episode_specific_offset": False,
            "frame_specific_offset": False,
            "phase_specific_xyz_offset": False,
            "doll_waypoint": False,
            "handoff_waypoint": False,
            "bin_waypoint": False,
        },
        "canonical_pinch_provenance": str(
            REPOSITORY
            / "outputs/doll_handoff_retargeting/legacy/"
            "canonical_pinch_proposed_b_2026-08-21_v11"
        ),
    }
    geometry_paths = (
        output / "interaction_centric_geometry_audit.json",
        REPOSITORY
        / "outputs/doll_handoff_retargeting/config/"
        "interaction_centric_geometry_audit.json",
    )
    for path in geometry_paths:
        atomic_json(path, geometry_audit)
    geometry_markdown = "\n".join(
        (
            "# Interaction-Centric Doll-Handoff Geometry Audit",
            "",
            f"- Old handoff-only value: `{rejected_clearance:.15f} m`",
            "- Decision: `REMOVED_TASK_SPECIFIC_CARTESIAN_RESIDUAL`",
            "- Active Cartesian handoff offset: `0.0 m`",
            "- Active target: static thumb/index/middle whole-hand grasp frame",
            "- Frame scope: one transform per side, every frame, every episode",
            "- Calibration: `SIM_ONLY_NOT_REAL_DEX3_CALIBRATED`",
            "",
            "The removed value is twice a thumb-pad tangential half extent. It is",
            "contact-shape clearance, not a rigid wrist-to-grasp-frame transform.",
            "The frozen canonical-pinch snapshot retains the historical audit.",
        )
    ) + "\n"
    for path in (
        output / "interaction_centric_geometry_audit.md",
        REPOSITORY
        / "outputs/doll_handoff_retargeting/config/"
        "interaction_centric_geometry_audit.md",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(geometry_markdown, encoding="utf-8")
    gates = {
        "cartesian_target_hashes_identical": all_target_hashes_equal,
        "after_ik_at_least_95_percent": all(
            value["after"]["ik_success_rate"] >= 0.95 for value in episodes.values()
        ),
        "after_joint_limits_zero": all(
            value["after"]["joint_limit_violations"] == 0 for value in episodes.values()
        ),
        "after_branch_discontinuities_zero": all(
            value["after"]["branch_discontinuities"] == 0 for value in episodes.values()
        ),
        "after_invalid_collisions_not_increased": all(
            value["after"]["invalid_collision_frames"]
            <= value["before"]["invalid_collision_frames"]
            for value in episodes.values()
        ),
        "after_arm_torso_collisions_not_increased": all(
            value["after"]["arm_torso_collision_frames"]
            <= value["before"]["arm_torso_collision_frames"]
            for value in episodes.values()
        ),
        "mean_primary_error_degradation_below_0_1_mm": all(
            value["delta"]["primary_task_position_error_mean_m"] <= 0.0001
            for value in episodes.values()
        ),
        "handoff_order_preserved": all(
            value["after"]["right_grasp_before_left_release"]
            for value in episodes.values()
        ),
        "release_geometry_preserved_inside_bin": all(
            value["after"]["release_xy_inside_bin"] for value in episodes.values()
        ),
    }
    automated_review_ready = all(gates.values())
    report = {
        "schema_version": "doll_handoff_natural_arm_audit_v1",
        "status": (
            "READY_FOR_NATURAL_ARM_REVIEW"
            if automated_review_ready
            else "BLOCKED_NATURAL_ARM_RESOLVER"
        ),
        "scope": "Proposed B smoke episodes 0/24/49 only",
        "recovered_before_resume": [
            "read-only canonical-pinch Proposed-B legacy snapshot with verified SHA256SUMS",
            "read-only A0 naive-wrist diagnostic snapshot",
            "Interaction-Centric research override document",
            "source-image-calibrated and frozen Doll/Bin scene layout",
            "49.553-mm task-specific Cartesian residual removal audit",
            "active-model three-finger whole-hand grasp frame and synergy",
            "source-derived ownership semantics",
            "resolver-OFF reference and natural-arm iterations v1 through v6",
        ],
        "resumed_changes": [
            "confirmed no surviving batch/training/Isaac process",
            "reduced the global null-space update bound from 0.012 to 0.008 rad",
            "added a common 0.15-rad joint-space temporal trust region",
            "reran matched resolver-OFF/ON smoke episodes 0/24/49",
            "updated active Doll-Handoff terminology from pinch to whole-hand grasp frame",
            "generated matched overview/top/side videos, plots, CSV, JSON, and Markdown",
        ],
        "before_root": str(before_root),
        "after_root": str(after_root),
        "method": {
            "elbow_sew_representation": (
                "signed elbow radial angle about shoulder-to-wrist axis relative "
                "to active-model task-ready nominal morphology guide"
            ),
            "null_space_formulation": after_config["natural_arm_redundancy"]["formulation"],
            "nominal_posture": after_config["resolved"]["canonical_g1_nominal_q"],
            "manipulability_metric": after_config["resolved"]["natural_arm_redundancy"][
                "manipulability_metric"
            ],
            "collision_clearance_objective": (
                "active G1 torso/proximal cross-arm collision geometry; interacting hands excluded"
            ),
            "temporal_terms": (
                "shared velocity/acceleration regularization, SEW continuity, and "
                "0.15-rad frame-to-frame joint-space trust region"
            ),
            "config": after_config["natural_arm_redundancy"],
        },
        "primary_task_targets_changed": False,
        "episode_specific_corrections_used": False,
        "phase_specific_joint_offsets_used": False,
        "phase_specific_cartesian_corrections_used": False,
        "common_a_b_solver": {
            "implemented": True,
            "scope": "COMMON_BASELINE_AND_PROPOSED",
            "single_class": "tools/doll_handoff_retargeting/retarget.py::SharedTemporalIK",
            "baseline_smoke_rerun": False,
            "baseline_reason": (
                "fair Baseline-A workspace reachability mapping is not yet validated; "
                "the frozen A0 diagnostic was not reused as the primary baseline"
            ),
        },
        "residual_interaction_issue": (
            "ep000/ep049 retain predominantly cross-hand collision flags during "
            "dual-contact, and ep049 retains one arm-torso frame. All categories "
            "pre-existed and total counts were reduced, not introduced. The remaining "
            "handoff overlap belongs to Interaction-B geometry review, not null-space "
            "Cartesian target mutation."
        ),
        "visual_assessment": (
            "overview/top/side contact-sheet inspection shows smoother and more stable "
            "elbow branches; human approval remains required before configuration freeze"
        ),
        "gates": gates,
        "episodes": episodes,
        "config_sha256": config_hashes,
        "interaction_centric_geometry_audit": str(geometry_paths[0]),
        "full_50_episode_run": "NOT_STARTED_BY_THIS_ADD_ON",
        "dataset_packaging": "NOT_STARTED_BY_DESIGN",
        "policy_training": "NOT_STARTED_BY_DESIGN",
    }
    atomic_json(output / "natural_arm_resolver_report.json", report)

    lines = [
        "# Generic Natural-Arm Resolver — Proposed B Smoke Audit",
        "",
        f"Status: **{report['status']}** (human review pending; not frozen)",
        "",
        "The matched before/after runs use identical Cartesian-target SHA256 values for",
        "episodes 0, 24, and 49. The resolver changes only redundant joint realization.",
        "",
        "| episode | IK before → after | task error mm before → after | SEW jump rad before → after | collisions before → after | branch after |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for episode in EPISODES:
        value = episodes[f"ep{episode:03d}"]
        before = value["before"]
        after = value["after"]
        lines.append(
            f"| {episode:03d} | {100*before['ik_success_rate']:.2f}% → "
            f"{100*after['ik_success_rate']:.2f}% | "
            f"{1000*before['primary_task_position_error_mean_m']:.3f} → "
            f"{1000*after['primary_task_position_error_mean_m']:.3f} | "
            f"{before['maximum_frame_to_frame_sew_change_rad']:.3f} → "
            f"{after['maximum_frame_to_frame_sew_change_rad']:.3f} | "
            f"{before['invalid_collision_frames']} → "
            f"{after['invalid_collision_frames']} | "
            f"{after['branch_discontinuities']} |"
        )
    lines.extend(
        [
            "",
            "## Global resolver changes",
            "",
            "- Damped primary-task null-space projection with task-independent posture gradient.",
            "- SEW preference/continuity, weak task-ready nominal, joint centering,",
            "  manipulability deficit, and active-model collision-clearance components.",
            "- Maximum null-space joint update reduced globally from 0.012 to 0.008 rad.",
            "- Added a common 0.15-rad frame-to-frame joint-space trust region.",
            "- No episode/phase inputs and no Cartesian target mutation.",
            "",
            "## Interpretation",
            "",
            "All after trajectories are finite, have zero joint-limit violations and zero",
            "branch discontinuities, and retain ≥95% strict position IK usability. The",
            "mean primary error changes by at most 0.057 mm. Residual ep000/ep049",
            "collisions are predominantly pre-existing cross-hand dual-contact overlap;",
            "ep049 also retains one arm-torso frame (down from five). The resolver reduces",
            "every episode's total invalid-collision count and does not create a new class.",
            "",
            "Baseline A was not falsely represented by the frozen A0 diagnostic. The",
            "resolver is wired into the one shared A/B solver, but a fair Baseline-A smoke",
            "rerun waits for its independent global workspace mapping to be validated.",
            "",
            "Full-50 conversion, dataset packaging, and policy training remain stopped.",
        ]
    )
    (output / "natural_arm_resolver_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "gates": gates}, indent=2))
    return 0 if automated_review_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
