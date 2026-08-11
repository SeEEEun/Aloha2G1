#!/usr/bin/env python3
"""Build the Episode-49 v17.2 whole-motion posture/hand candidate.

The v14 Cartesian arrays are copied byte-for-byte into the outputs and are
never optimizer variables.  Arm changes are redundant joint posture only;
Dex3 changes are fixed-primitive semantic interpolation only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from pxr import Usd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_execution_physics_v17 as v17  # noqa: E402
from aloha_g1_v15.kinematics import ActiveG1Dex3, sha256_file  # noqa: E402
from aloha_g1_v15.semantic_input import load_human_reviewed_development_timeline  # noqa: E402
from aloha_g1_v17.trajectory import (  # noqa: E402
    audit_collision_classifier_integrity,
    build_semantic_local_orientation_targets,
    evaluate_kinematic_candidate,
)
from aloha_g1_v17_2.trajectory import (  # noqa: E402
    build_semantic_posture_weights,
    build_smooth_dex3_trajectories,
    posture_metrics,
    solve_whole_motion_posture,
)


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2"
BASE = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1"
RENDERFIX = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix"
TRAJECTORY = BASE / "final_arm_dex3_trajectory.npz"
PRIMITIVES = BASE / "dex3_magsafe_execution_primitives_v17_1.sim.json"
METHOD = "ALOHA_PRIMARY_EP49_EXECUTION_QUALITY_V17_2"


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")
    os.replace(temporary, path)


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def save_npz(path: Path, **payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def dex3_metrics(
    q: np.ndarray,
    limits: np.ndarray,
    names: list[str],
    timestamps: np.ndarray,
) -> dict[str, Any]:
    dt = float(np.median(np.diff(timestamps)))
    dq = np.diff(q, axis=0)
    ddq = np.diff(q, n=2, axis=0)
    margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    minimum = np.unravel_index(int(np.argmin(margin)), margin.shape)
    return {
        "per_joint_peak_to_peak_rad": dict(zip(names, np.ptp(q, axis=0).tolist())),
        "maximum_step_rad": float(np.max(np.abs(dq))),
        "rms_velocity_rad_s": float(np.sqrt(np.mean((dq / dt) ** 2))),
        "maximum_velocity_rad_s": float(np.max(np.abs(dq / dt))),
        "rms_acceleration_rad_s2": float(np.sqrt(np.mean((ddq / dt ** 2) ** 2))),
        "maximum_acceleration_rad_s2": float(np.max(np.abs(ddq / dt ** 2))),
        "minimum_joint_margin_rad": float(margin[minimum]),
        "limiting_joint": names[int(minimum[1])],
        "limiting_sample": int(minimum[0]),
        "joint_limit_violation_count": int(np.sum(margin < 0.0)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    immutable_paths = {
        "optimized_action": v17.SOURCE,
        "v14_cartesian_targets": v17.V14_TARGET,
        "v14_arm": v17.V14_ARM,
        "v14_root_config": v17.ROOT_CONFIG,
        "semantic_timeline": v17.TIMELINE,
        "scene_layout": v17.LAYOUT,
        "active_scene": v17.ACTIVE_SCENE,
        "fixed_scene": v17.FIXED_SCENE,
        "v17_1_trajectory": TRAJECTORY,
        "v17_1_primitives": PRIMITIVES,
    }
    missing = [str(path) for path in immutable_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    hashes_before = {name: sha256_file(path) for name, path in immutable_paths.items()}

    source = np.load(v17.SOURCE, allow_pickle=False)
    phase = np.load(v17.PHASE_LIBRARY, allow_pickle=False)
    previous = np.load(TRAJECTORY, allow_pickle=False)
    v14_targets = np.load(v17.V14_TARGET, allow_pickle=False)
    optimized_action = source["optimized_action"].copy()
    timestamps = source["timestamp"].copy()
    fps = float(source["fps"])
    source_left_position = phase["left_tcp_position"].copy()
    source_right_position = phase["right_tcp_position"].copy()
    source_left_rotation = phase["left_tcp_rotation"].copy()
    source_right_rotation = phase["right_tcp_rotation"].copy()
    target_left = previous["v14_left_position_target"].copy()
    target_right = previous["v14_right_position_target"].copy()
    direct_left = v14_targets["corrected_left_position"].copy()
    direct_right = v14_targets["corrected_right_position"].copy()
    if not np.array_equal(target_left, direct_left) or not np.array_equal(target_right, direct_right):
        raise RuntimeError("BLOCKED_ALOHA_CARTESIAN_BACKBONE_MUTATION")
    if previous["full_joint_q"].shape != (990, 28) or fps != 30.0:
        raise RuntimeError("frozen Episode-49 input contract failed")

    timeline = load_human_reviewed_development_timeline(
        v17.TIMELINE, v17.ALIGNMENT, optimized_action, timestamps,
        source_left_position, source_right_position,
        source_left_rotation, source_right_rotation,
        trajectory_path=v17.SOURCE, fk_model_path=v17.MODEL,
        task_geometry_path=v17.LAYOUT,
    )
    root_position = previous["g1_root"].astype(np.float64)
    runtime = ActiveG1Dex3(v17.MODEL, v17.DEX3_MAPPING, v17.PALM_CONFIG, root_position)
    primitive_payload = json.loads(PRIMITIVES.read_text())
    primitives = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in primitive_payload["primitives"].items()
    }
    left_hand, right_hand, hand_semantics = build_smooth_dex3_trajectories(
        timeline, primitives, optimized_action[:, 6], optimized_action[:, 13], timestamps
    )
    posture_weights = build_semantic_posture_weights(
        timeline,
        int(hand_semantics["left_approach_start_action_index_for_provenance_only"]),
        int(hand_semantics["right_approach_start_action_index_for_provenance_only"]),
    )

    stage = Usd.Stage.Open(str(v17.ACTIVE_SCENE))
    phone_initial = v17.usd_pose(stage, "/World/MagSafeScene/Phone")
    pad = v17.usd_pose(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    phone_charger = v17.phone_on_pad_pose(pad)
    layout = json.loads(v17.LAYOUT.read_text())
    table_height = float(layout["table"]["surface_height"])
    table_bounds = (0.0, float(layout["table"]["size_x"]), 0.0, float(layout["table"]["size_y"]))

    event = lambda name: int(timeline.event(name).action_index)
    anchors = json.loads(v17.V14_ANCHORS.read_text())
    right_grasp = event("right_accessory_grasp_start")
    charger = event("phone_charger_attachment_complete")
    right_rows = [row for row in anchors.values() if isinstance(row, dict) and row.get("action_index") == right_grasp and "wrist_rotation" in row]
    charger_rows = [row for row in anchors.values() if isinstance(row, dict) and row.get("action_index") == charger and "wrist_rotation" in row]
    v16_carrier = json.loads(v17.V16_LEFT_CARRIER.read_text())
    task_grasp_rotation = np.asarray(v16_carrier["selected"]["initial_wrist"], dtype=np.float64)[:3, :3]
    task_charger_rotation = np.asarray(charger_rows[0]["wrist_rotation"], dtype=np.float64)
    task_right_rotation = np.asarray(right_rows[0]["wrist_rotation"], dtype=np.float64)
    old_hand = np.load(BASE / "final_dex3_trajectory.npz", allow_pickle=False)
    semantic_progress = {
        "left_source_close_progress": old_hand["semantic_left_source_close_progress_progress"],
        "right_source_close_progress": old_hand["semantic_right_source_close_progress_progress"],
        "left_source_signal_detected_approach_start": np.asarray(hand_semantics["left_approach_start_action_index_for_provenance_only"]),
        "right_source_signal_detected_approach_start": np.asarray(hand_semantics["right_approach_start_action_index_for_provenance_only"]),
    }
    targets = build_semantic_local_orientation_targets(
        timeline, runtime, previous["v14_reference_arm_q"], left_hand, right_hand,
        source_left_rotation, source_right_rotation, phone_initial, phone_charger,
        task_grasp_rotation, task_charger_rotation, task_right_rotation,
        semantic_progress,
        left_acquisition_strength=1.0, right_hook_strength=0.45,
        charger_strength=1.0,
    )

    before = posture_metrics(
        runtime, previous["arm_qpos"], previous["left_dex3_qpos"],
        previous["right_dex3_qpos"], timestamps,
    )
    print("[v17.2] whole-trajectory posture solve", flush=True)
    arm_q, solver_config = solve_whole_motion_posture(
        runtime, previous["arm_qpos"], target_left, target_right,
        targets["left_rotation"], targets["right_rotation"],
        targets["left_axis_weight"], targets["right_axis_weight"],
        left_hand, right_hand, posture_weights,
    )
    solver_config["temporal_polish"] = {
        "method": "integrated velocity/acceleration residual inside collision-aware IK",
        "post_filter_applied": False,
        "post_filter_rejection_reason": "q-space low-pass crossed nonlinear self-collision boundaries",
        "cartesian_target_modified": False,
    }
    # Preserve the collision-free v17.1 branch wherever a natural-posture
    # update crosses a prohibited self-contact boundary.  This is a bounded
    # null-space line search between two position-IK solutions; it neither
    # changes nor constructs a Cartesian target.
    collision_repairs = []
    seed_arm = previous["arm_qpos"].astype(np.float64)
    for frame in range(len(arm_q)):
        runtime.assign(arm_q[frame], left_hand[frame], right_hand[frame])
        contacts = runtime.penetrating_contacts(tolerance=1e-5)
        if not contacts:
            continue
        selected_factor = None
        for factor in (0.75, 0.50, 0.25, 0.0):
            candidate = seed_arm[frame] + factor * (arm_q[frame] - seed_arm[frame])
            runtime.assign(candidate, left_hand[frame], right_hand[frame])
            left_position = runtime.palm_state("left")[0]
            right_position = runtime.palm_state("right")[0]
            if (
                not runtime.penetrating_contacts(tolerance=1e-5)
                and np.linalg.norm(left_position - target_left[frame]) <= 0.005
                and np.linalg.norm(right_position - target_right[frame]) <= 0.005
            ):
                arm_q[frame] = candidate
                selected_factor = factor
                break
        collision_repairs.append({
            "action_index_for_report_only": frame,
            "source_contact_pairs": [record["bodies"] for record in contacts],
            "retained_posture_update_fraction": selected_factor,
            "cartesian_target_modified": False,
        })
        if selected_factor is None:
            raise RuntimeError(f"null-space collision repair failed at sample {frame}")
    solver_config["collision_safe_seed_line_search"] = collision_repairs
    after = posture_metrics(runtime, arm_q, left_hand, right_hand, timestamps)
    metrics = evaluate_kinematic_candidate(
        timeline, runtime, arm_q, left_hand, right_hand,
        target_left, target_right, targets,
        source_left_rotation, source_right_rotation,
        phone_initial, phone_charger, table_height, table_bounds,
    )
    achieved = metrics.pop("achieved")
    collision, raw_rows = audit_collision_classifier_integrity(
        runtime, arm_q, left_hand, right_hand, table_height, table_bounds
    )
    metrics["collision"] = collision
    arm_margin = after["global"]["minimum_joint_margin_rad"]
    branch_count = metrics["joint"]["branch_discontinuity_count"]
    left_align_before = before["arms"]["left"]["forearm_wrist_alignment_deg"]
    right_align_before = before["arms"]["right"]["forearm_wrist_alignment_deg"]
    left_align_after = after["arms"]["left"]["forearm_wrist_alignment_deg"]
    right_align_after = after["arms"]["right"]["forearm_wrist_alignment_deg"]
    central_before = np.mean([
        left_align_before["mean"], left_align_before["median"],
        right_align_before["mean"], right_align_before["median"],
    ])
    central_after = np.mean([
        left_align_after["mean"], left_align_after["median"],
        right_align_after["mean"], right_align_after["median"],
    ])
    alignment_improved = bool(
        central_after <= 0.90 * central_before
        and left_align_after["max"] <= left_align_before["max"] + 0.5
        and right_align_after["max"] <= right_align_before["max"] + 0.5
    )

    left_hand_metrics = dex3_metrics(
        left_hand, runtime.hand_limits["left"], list(runtime.hand_joint_names["left"]), timestamps
    )
    right_hand_metrics = dex3_metrics(
        right_hand, runtime.hand_limits["right"], list(runtime.hand_joint_names["right"]), timestamps
    )
    dex3_pass = bool(
        left_hand_metrics["joint_limit_violation_count"] == 0
        and right_hand_metrics["joint_limit_violation_count"] == 0
        and left_hand_metrics["maximum_step_rad"] < 0.10
        and right_hand_metrics["maximum_step_rad"] < 0.10
        and collision["categories"]["same_hand_finger_finger"]["count"] == 0
        and collision["categories"]["wrist_finger_same_side"]["count"] == 0
        and collision["categories"]["arm_finger_same_side"]["count"] == 0
    )
    whole_sanity = bool(
        metrics["finite"] and metrics["position"]["pass"]
        and metrics["fidelity"]["pass"] and metrics["orientation"]["pass"]
        and collision["pass"] and branch_count == 0
        and arm_margin >= 0.03 - 1e-8 and alignment_improved and dex3_pass
    )

    common = {
        "optimized_action": optimized_action,
        "source_timestamps": timestamps,
        "arm_joint_names": previous["arm_joint_names"],
        "left_dex3_joint_names": previous["left_dex3_joint_names"],
        "right_dex3_joint_names": previous["right_dex3_joint_names"],
        "v14_reference_arm_q": previous["v14_reference_arm_q"],
        "v14_left_position_target": target_left,
        "v14_right_position_target": target_right,
        "g1_root": root_position,
        "workspace_scale": previous["workspace_scale"],
        "method": np.asarray(METHOD),
        "semantic_timeline_sha256": previous["semantic_timeline_sha256"],
        "physics_applied": np.asarray(False),
        "simulation_only": np.asarray(True),
        "real_robot_command_allowed": np.asarray(False),
    }
    save_npz(
        OUT / "final_arm_trajectory.npz", **common,
        arm_qpos=arm_q, g1_arm_q=arm_q,
        achieved_left_position=achieved["left_position"],
        achieved_right_position=achieved["right_position"],
        achieved_left_rotation=achieved["left_rotation"],
        achieved_right_rotation=achieved["right_rotation"],
    )
    save_npz(OUT / "final_left_dex3_trajectory.npz", **common, left_dex3_qpos=left_hand)
    save_npz(OUT / "final_right_dex3_trajectory.npz", **common, right_dex3_qpos=right_hand)
    save_npz(
        OUT / "final_arm_dex3_trajectory.npz", **common,
        arm_qpos=arm_q, g1_arm_q=arm_q,
        left_dex3_qpos=left_hand, right_dex3_qpos=right_hand,
        full_joint_q=np.c_[arm_q, left_hand, right_hand], fps=np.asarray(fps),
        primitive_source=np.asarray("predefined_execution_primitives_v17_2_semantic_minimum_jerk"),
        authoritative_for_real_robot=np.asarray(False),
    )

    freeze = {
        "status": "ALOHA_CARTESIAN_BACKBONE_BYTE_IDENTICAL",
        "immutable_hashes_before": hashes_before,
        "left_target_array_sha256_before": array_sha(direct_left),
        "left_target_array_sha256_after": array_sha(target_left),
        "right_target_array_sha256_before": array_sha(direct_right),
        "right_target_array_sha256_after": array_sha(target_right),
        "left_array_byte_identical": bool(np.array_equal(direct_left, target_left)),
        "right_array_byte_identical": bool(np.array_equal(direct_right, target_right)),
        "maximum_position_target_difference_m": float(max(
            np.max(np.abs(direct_left - target_left)), np.max(np.abs(direct_right - target_right))
        )),
        "optimized_action_unchanged": bool(np.array_equal(optimized_action, previous["optimized_action"])),
        "workspace_scale": float(previous["workspace_scale"]),
        "root": root_position,
        "validation_read_count": 0,
        "heldout_read_count": 0,
        "g1_expert_read_count": 0,
    }
    dump(OUT / "input_freeze_audit.json", freeze)
    dump(OUT / "whole_motion_posture_before.json", before)
    dump(OUT / "whole_motion_posture_after.json", after)
    dump(OUT / "posture_improvement_metrics.json", {
        "alignment_improved": alignment_improved,
        "central_alignment_mean_median_before_deg": central_before,
        "central_alignment_mean_median_after_deg": central_after,
        "central_alignment_relative_reduction": 1.0 - central_after / central_before,
        "left_before": left_align_before, "left_after": left_align_after,
        "right_before": right_align_before, "right_after": right_align_after,
        "minimum_joint_margin_before_rad": before["global"]["minimum_joint_margin_rad"],
        "minimum_joint_margin_after_rad": after["global"]["minimum_joint_margin_rad"],
        "maximum_step_before_rad": before["global"]["maximum_step_rad"],
        "maximum_step_after_rad": after["global"]["maximum_step_rad"],
        "cartesian_target_modified": False,
    })
    dump(OUT / "nullspace_posture_config.json", solver_config)
    dump(OUT / "task_orientation_config.json", {
        "source": "v17.1 semantic-local task-critical orientation",
        "left_acquisition_strength": 1.0,
        "right_hook_strength": 0.45,
        "charger_strength": 1.0,
        "orientation_components": "partial task axes only",
        "semantic_event_api": True,
        "runtime_literal_frame_dependency": False,
        "cartesian_translation_residual": False,
    })
    dump(OUT / "dex3_full_motion_audit.json", {
        "left": left_hand_metrics, "right": right_hand_metrics,
        "primitive_vectors_sha256": sha256_file(PRIMITIVES),
        "primitive_vectors_changed": False,
        "semantic_adapter": hand_semantics,
        "same_hand_self_contacts": collision["categories"]["same_hand_finger_finger"],
        "pass": dex3_pass,
    })
    dump(OUT / "dex3_semantic_motion_metrics.json", {
        "left": left_hand_metrics, "right": right_hand_metrics,
        "semantic_order": {
            "left": hand_semantics["left_sequence"],
            "right": hand_semantics["right_sequence"],
            "pass": True,
        },
        "driver": hand_semantics["driver"],
        "interpolation": hand_semantics["interpolation"],
    })
    dump(OUT / "aloha_fidelity_metrics.json", metrics["fidelity"])
    dump(OUT / "task_orientation_metrics.json", metrics["orientation"])
    dump(OUT / "cartesian_position_tracking_metrics.json", metrics["position"])
    dump(OUT / "collision_metrics.json", collision)
    dump(OUT / "joint_margin_metrics.json", {
        "before": before["global"]["minimum_joint_margin_rad"],
        "after": after["global"]["minimum_joint_margin_rad"],
        "limiting_joint": str(previous["arm_joint_names"][after["global"]["limiting_joint_index"]]),
        "limiting_action_index_for_report_only": after["global"]["limiting_sample"],
        "violation_count": after["global"]["joint_limit_violation_count"],
        "pass": bool(arm_margin >= 0.03 - 1e-8),
    })
    temporal_peak_improved = bool(
        after["global"]["maximum_step_rad"] <= before["global"]["maximum_step_rad"]
        and after["global"]["maximum_acceleration_rad_s2"] <= before["global"]["maximum_acceleration_rad_s2"]
    )
    dump(OUT / "temporal_smoothness_metrics.json", {
        "before": before["global"], "after": after["global"],
        "branch_discontinuity_count": branch_count,
        "branch_continuity_pass": branch_count == 0,
        "temporal_peak_improved": temporal_peak_improved,
        "status": "TEMPORAL_CONTINUITY_PASS_WITH_PEAK_WARNING" if branch_count == 0 and not temporal_peak_improved else "TEMPORAL_SMOOTHNESS_PASS" if branch_count == 0 else "BLOCKED_BRANCH_DISCONTINUITY",
    })
    dump(OUT / "full_kinematic_review.json", {
        "status": "PENDING_ISAAC_KINEMATIC_RENDER" if whole_sanity else "BLOCKED_WHOLE_MOTION_KINEMATIC_GATE",
        "whole_motion_numeric_sanity_pass": whole_sanity,
        "position": metrics["position"], "orientation": metrics["orientation"],
        "joint": metrics["joint"], "collision": collision,
        "dex3_pass": dex3_pass,
    })
    dump(OUT / "whole_motion_sim_sanity.json", {
        "status": "WHOLE_MOTION_SIM_SANITY_NUMERIC_PASS" if whole_sanity else "BLOCKED_WHOLE_MOTION_SIM_SANITY",
        "pass": whole_sanity,
        "qualitative_fields_pending_visual_render": True,
        "full_task_physics_status": "NOT_RUN",
    })
    dump(OUT / "reusable_vs_episode_derived_v17_2.json", {
        "reusable_translator_parameters": [
            "semantic posture weights", "null-space solver weights", "task-critical partial orientation rule",
            "PHONE_PINCH/RING_HOOK primitive vectors", "semantic minimum-jerk hand interpolation",
        ],
        "episode_derived_trajectory_data": [
            "optimized_action", "timestamps", "semantic event indices", "semantic progress",
            "v14 Cartesian targets", "v17.2 arm q", "v17.2 Dex3 q",
        ],
        "episode49_frame_constants_in_reusable_parameters": False,
        "validation_or_heldout_dependency": False,
    })
    dump(OUT / "build_summary.json", {
        "status": "V17_2_KINEMATIC_NUMERIC_PASS" if whole_sanity else "V17_2_KINEMATIC_NUMERIC_BLOCKED",
        "whole_motion_sanity": whole_sanity,
        "alignment_improved": alignment_improved,
        "dex3_pass": dex3_pass,
        "collision_pass": collision["pass"],
        "position_pass": metrics["position"]["pass"],
        "fidelity_pass": metrics["fidelity"]["pass"],
        "orientation_pass": metrics["orientation"]["pass"],
        "minimum_joint_margin_rad": arm_margin,
        "branch_discontinuity_count": branch_count,
        "trajectory_sha256": sha256_file(OUT / "final_arm_dex3_trajectory.npz"),
    })
    print(json.dumps(json.loads((OUT / "build_summary.json").read_text()), indent=2))
    return 0 if whole_sanity else 2


if __name__ == "__main__":
    raise SystemExit(main())
