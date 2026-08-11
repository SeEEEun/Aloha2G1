"""Task-null-space whole-motion refinement without Cartesian-path mutation.

Every phase boundary is resolved from a :class:`SemanticTimeline`.  The
functions in this module never construct, offset, or rewrite a Cartesian
target and contain no Episode-49 semantic frame constants.
"""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares

from aloha_g1_v15.kinematics import ActiveG1Dex3
from aloha_g1_v17.trajectory import _collision_scalars
from aloha_magsafe_semantics.schema import SemanticTimeline


def _minimum_jerk(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return value ** 3 * (10.0 - 15.0 * value + 6.0 * value ** 2)


def _ramp(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(max(1, length), dtype=np.float64)
    return _minimum_jerk(np.linspace(0.0, 1.0, length))


def _write_blend(output: np.ndarray, start: int, end: int, first: np.ndarray, last: np.ndarray) -> None:
    if end < start:
        raise ValueError("invalid semantic blend interval")
    progress = _ramp(end - start + 1)
    output[start : end + 1] = (
        (1.0 - progress[:, None]) * np.asarray(first, dtype=np.float64)
        + progress[:, None] * np.asarray(last, dtype=np.float64)
    )


def _source_close_start(aperture: np.ndarray, lo: int, event: int) -> int:
    """Find the last genuinely open sample before a named grasp event."""
    values = np.asarray(aperture, dtype=np.float64)
    low, high = np.quantile(values, [0.05, 0.95])
    close = np.clip((high - values) / max(float(high - low), 1e-8), 0.0, 1.0)
    candidates = np.flatnonzero(close[int(lo) : int(event) + 1] <= 0.10)
    return int(lo + candidates[-1]) if len(candidates) else int(lo)


def _final_release_start(
    aperture: np.ndarray,
    timestamps: np.ndarray,
    phase_start: int,
    release_complete: int,
) -> tuple[int, dict[str, Any]]:
    """Detect the final sustained opening onset inside the named release phase.

    Early non-terminal opening pulses are ignored.  The returned onset is the
    last low/open-progress sample before the final sustained transition to the
    endpoint plateau.  Smoothing duration is 0.233333 seconds, never a fixed
    sample count.
    """
    values = np.asarray(aperture, dtype=np.float64)
    time = np.asarray(timestamps, dtype=np.float64)
    dt = float(np.median(np.diff(time)))
    samples = max(3, int(round((7.0 / 30.0) / max(dt, 1e-8))))
    if samples % 2 == 0:
        samples += 1
    kernel = np.ones(samples, dtype=np.float64) / samples
    smooth = np.convolve(np.pad(values, samples // 2, mode="edge"), kernel, mode="valid")
    segment = smooth[int(phase_start) : int(release_complete) + 1]
    low, high = np.quantile(values, [0.05, 0.95])
    opening = np.clip((segment - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    high_rows = np.flatnonzero(opening >= 0.80)
    final_high = int(high_rows[-1]) if len(high_rows) else len(opening) - 1
    low_rows = np.flatnonzero(opening[: final_high + 1] <= 0.15)
    onset_local = int(low_rows[-1]) if len(low_rows) else 0
    onset = int(phase_start) + onset_local
    return onset, {
        "method": "final sustained normalized opening suffix",
        "smoothing_sec": 7.0 / 30.0,
        "smoothing_samples_for_this_episode": samples,
        "normalized_low_threshold": 0.15,
        "normalized_high_threshold": 0.80,
        "detected_release_start_action_index_for_provenance_only": onset,
        "runtime_literal_frame_dependency": False,
    }


def build_smooth_dex3_trajectories(
    timeline: SemanticTimeline,
    primitives: dict[str, np.ndarray],
    source_left_gripper: np.ndarray,
    source_right_gripper: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one smooth semantic hand-adapter trajectory per side."""
    length = timeline.trajectory_length
    event = lambda name: int(timeline.event(name).action_index)
    left_open = np.asarray(primitives["LEFT_OPEN"], dtype=np.float64)
    left_pre = np.asarray(primitives["LEFT_PHONE_PREGRASP"], dtype=np.float64)
    left_pinch = np.asarray(primitives["LEFT_PHONE_PINCH"], dtype=np.float64)
    right_open = np.asarray(primitives["RIGHT_OPEN"], dtype=np.float64)
    right_pre = np.asarray(primitives["RIGHT_RING_PREHOOK"], dtype=np.float64)
    right_hook = np.asarray(primitives["RIGHT_RING_HOOK"], dtype=np.float64)
    left = np.repeat(left_open[None], length, axis=0)
    right = np.repeat(right_open[None], length, axis=0)

    left_grasp = event("left_phone_grasp_start")
    rotation_start = event("phone_rotation_to_portrait_start")
    charger = event("phone_charger_attachment_complete")
    left_release = event("left_phone_release_complete")
    portrait = event("phone_portrait_reached")
    right_grasp = event("right_accessory_grasp_start")
    removed = event("accessory_removed")
    right_release = event("right_accessory_release_complete")
    left_start = _source_close_start(source_left_gripper, timeline.start_index, left_grasp)
    right_start = _source_close_start(source_right_gripper, portrait, right_grasp)
    right_release_start, release_audit = _final_release_start(
        source_right_gripper, timestamps, removed, right_release
    )

    _write_blend(left, left_start, left_grasp, left_open, left_pre)
    _write_blend(left, left_grasp, rotation_start, left_pre, left_pinch)
    left[rotation_start : charger + 1] = left_pinch
    _write_blend(left, charger, left_release, left_pinch, left_open)
    left[left_release:] = left_open

    _write_blend(right, right_start, right_grasp, right_open, right_pre)
    _write_blend(right, right_grasp, removed, right_pre, right_hook)
    right[removed : right_release_start + 1] = right_hook
    _write_blend(right, right_release_start, right_release, right_hook, right_open)
    right[right_release:] = right_open

    return left, right, {
        "driver": "GENERIC_SEMANTIC_TIMELINE_SOURCE_GRIPPER_SUFFIX",
        "interpolation": "minimum_jerk",
        "left_sequence": ["OPEN", "PREGRASP", "PINCH", "HOLD", "ROTATION_HOLD", "TRANSPORT_HOLD", "CHARGER_HOLD", "RELEASE", "OPEN_RETURN"],
        "right_sequence": ["OPEN", "PREHOOK", "HOOK", "REMOVAL_HOLD", "ACCESSORY_HOLD", "RELEASE", "OPEN_RETURN"],
        "left_approach_start_action_index_for_provenance_only": left_start,
        "right_approach_start_action_index_for_provenance_only": right_start,
        "right_release_detection": release_audit,
        "primitive_vectors_changed": False,
        "per_frame_finger_ik": False,
        "cartesian_arm_path_dependency": False,
        "runtime_literal_frame_dependency": False,
    }


def build_semantic_posture_weights(
    timeline: SemanticTimeline,
    left_approach_start: int,
    right_approach_start: int,
) -> dict[str, np.ndarray]:
    """Return reusable natural-posture emphasis from named task phases."""
    n = timeline.trajectory_length
    event = lambda name: int(timeline.event(name).action_index)
    left = np.full(n, 0.65, dtype=np.float64)
    right = np.full(n, 0.75, dtype=np.float64)
    grasp = event("left_phone_grasp_start")
    rotation_start = event("phone_rotation_to_portrait_start")
    portrait = event("phone_portrait_reached")
    phone_move = event("phone_move_to_charger_start")
    charger = event("phone_charger_attachment_complete")
    left_release = event("left_phone_release_complete")
    right_grasp = event("right_accessory_grasp_start")
    removed = event("accessory_removed")
    right_release = event("right_accessory_release_complete")
    left[int(left_approach_start) : grasp + 1] = 1.00
    left[grasp : rotation_start + 1] = 0.55
    left[rotation_start : portrait + 1] = 0.30
    left[portrait : phone_move + 1] = 0.65
    left[phone_move : charger + 1] = 0.45
    left[charger : left_release + 1] = 0.35
    left[left_release:] = 1.00
    right[portrait : int(right_approach_start)] = 0.90
    right[int(right_approach_start) : right_grasp + 1] = 1.00
    right[right_grasp : removed + 1] = 0.35
    right[removed : right_release + 1] = 0.55
    right[right_release:] = 1.00
    return {
        "left": left,
        "right": right,
        "definition": np.asarray("semantic phase/progress; no literal frame rule"),
    }


def _alignment_vector(runtime: ActiveG1Dex3, side: str) -> np.ndarray:
    elbow = runtime.model_to_scene_position(runtime.data.xpos[runtime.body_id(f"{side}_elbow_link")])
    wrist = runtime.wrist_pose(side)
    forearm = wrist[:3, 3] - elbow
    forearm /= max(float(np.linalg.norm(forearm)), 1e-12)
    axis = wrist[:3, 1]
    sign = 1.0 if float(np.dot(axis, forearm)) >= 0.0 else -1.0
    return axis - sign * forearm


def solve_whole_motion_posture(
    runtime: ActiveG1Dex3,
    seed_arm_q: np.ndarray,
    left_position: np.ndarray,
    right_position: np.ndarray,
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
    left_axis_weight: np.ndarray,
    right_axis_weight: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    posture_weights: dict[str, np.ndarray],
    *,
    hard_margin_rad: float = 0.03,
    preferred_margin_rad: float = 0.05,
    max_deviation_rad: float = 0.35,
    max_step_rad: float = 0.25,
    position_gain: float = 10000.0,
    orientation_gain: float = 42.0,
    alignment_gain: float = 0.55,
    wrist_neutral_gain: float = 0.12,
    elbow_neutral_gain: float = 0.05,
    shoulder_neutral_gain: float = 0.01,
    temporal_gain: float = 0.06,
    acceleration_gain: float = 0.03,
    joint_center_gain: float = 4.0,
    collision_gain: float = 150000.0,
    passes: tuple[str, ...] = ("forward",),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Alternating temporal solve over all samples with immutable XYZ targets."""
    seed_arm_q = np.asarray(seed_arm_q, dtype=np.float64)
    output = seed_arm_q.copy()
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    n = len(output)
    pass_rows: list[dict[str, Any]] = []

    for pass_name in passes:
        reference = output.copy()
        indices = range(n) if pass_name == "forward" else range(n - 1, -1, -1)
        maximum_position_error = 0.0
        for frame in indices:
            for side, target_p, target_r, axis_weight, block in (
                ("left", left_position[frame], left_rotation[frame], left_axis_weight[frame], slice(0, 7)),
                ("right", right_position[frame], right_rotation[frame], right_axis_weight[frame], slice(7, 14)),
            ):
                base = seed_arm_q[frame, block]
                value0 = output[frame, block].copy()
                direction = 1 if pass_name == "forward" else -1
                neighbor_index = frame - direction
                second_index = frame - 2 * direction
                neighbor = output[neighbor_index, block] if 0 <= neighbor_index < n else value0
                second = output[second_index, block] if 0 <= second_index < n else neighbor
                other_neighbor = reference[frame + direction, block] if 0 <= frame + direction < n else value0
                lo = np.maximum(limits[block, 0] + hard_margin_rad, base - max_deviation_rad)
                hi = np.minimum(limits[block, 1] - hard_margin_rad, base + max_deviation_rad)
                if 0 <= neighbor_index < n:
                    lo = np.maximum(lo, neighbor - max_step_rad)
                    hi = np.minimum(hi, neighbor + max_step_rad)
                if np.any(lo >= hi):
                    # The final continuation bound has priority over the
                    # seed-deviation preference.  Dropping the continuation
                    # bound would create the exact isolated joint flip this
                    # whole-trajectory solve is meant to prevent.
                    lo = np.maximum(limits[block, 0] + hard_margin_rad, neighbor - max_step_rad)
                    hi = np.minimum(limits[block, 1] - hard_margin_rad, neighbor + max_step_rad)
                if np.any(lo >= hi):
                    raise RuntimeError(f"continuation trust region infeasible at {frame}/{side}")
                x0 = np.clip(0.50 * value0 + 0.30 * neighbor + 0.20 * other_neighbor, lo + 1e-9, hi - 1e-9)
                whole = output[frame].copy()
                posture = float(posture_weights[side][frame])
                # Freeze the neutral forearm-axis choice at the seed pose for
                # this sample.  Re-selecting the +/- axis inside the nonlinear
                # residual is discontinuous near 90 degrees and can trap the
                # solver at the visibly worst posture.
                whole[block] = value0
                runtime.assign(whole, left_hand_q[frame], right_hand_q[frame])
                elbow_seed = runtime.model_to_scene_position(
                    runtime.data.xpos[runtime.body_id(f"{side}_elbow_link")]
                )
                wrist_seed = runtime.wrist_pose(side)
                forearm_seed = wrist_seed[:3, 3] - elbow_seed
                forearm_seed /= max(float(np.linalg.norm(forearm_seed)), 1e-12)
                sign_seed = 1.0 if float(np.dot(wrist_seed[:3, 1], forearm_seed)) >= 0.0 else -1.0
                alignment_target = sign_seed * forearm_seed

                def residual(value: np.ndarray) -> np.ndarray:
                    whole[block] = value
                    runtime.assign(whole, left_hand_q[frame], right_hand_q[frame])
                    position, rotation, _ = runtime.palm_state(side)
                    axes = []
                    for axis in range(3):
                        axes.extend(
                            orientation_gain * float(axis_weight[axis])
                            * (rotation[:, axis] - target_r[:, axis])
                        )
                    margin = np.minimum(value - limits[block, 0], limits[block, 1] - value)
                    barrier = np.maximum(0.0, preferred_margin_rad - margin)
                    elbow_target = 0.65
                    temporal_center = 0.5 * (neighbor + other_neighbor)
                    acceleration = value - 2.0 * neighbor + second
                    return np.r_[
                        position_gain * (position - target_p),
                        np.asarray(axes),
                        alignment_gain * posture * (rotation[:, 1] - alignment_target),
                        wrist_neutral_gain * posture * value[4:7],
                        elbow_neutral_gain * posture * (value[3] - elbow_target),
                        shoulder_neutral_gain * posture * value[1:3],
                        0.08 * (value - base),
                        temporal_gain * (value - temporal_center),
                        acceleration_gain * acceleration,
                        joint_center_gain * barrier,
                        collision_gain * _collision_scalars(runtime, side),
                    ]

                solved = least_squares(
                    residual, x0, bounds=(lo, hi), max_nfev=45,
                    ftol=2e-10, xtol=2e-10, gtol=2e-10, x_scale="jac",
                )
                output[frame, block] = solved.x
                whole[block] = solved.x
                runtime.assign(whole, left_hand_q[frame], right_hand_q[frame])
                position = runtime.palm_state(side)[0]
                maximum_position_error = max(maximum_position_error, float(np.linalg.norm(position - target_p)))
        pass_rows.append({
            "direction": pass_name,
            "maximum_position_error_m_during_pass": maximum_position_error,
            "maximum_joint_step_rad_after_pass": float(np.max(np.abs(np.diff(output, axis=0)))),
        })
        print(
            f"[v17.2 posture] {pass_name} pass complete | "
            f"max position {maximum_position_error * 1000.0:.3f} mm | "
            f"max step {np.max(np.abs(np.diff(output, axis=0))):.4f} rad",
            flush=True,
        )
    return output, {
        "method": "whole-trajectory temporal-continuation task-null-space least squares",
        "passes": pass_rows,
        "hard_margin_rad": hard_margin_rad,
        "preferred_margin_rad": preferred_margin_rad,
        "max_deviation_rad_from_v17_1_seed": max_deviation_rad,
        "max_step_trust_region_rad": max_step_rad,
        "position_gain": position_gain,
        "orientation_gain": orientation_gain,
        "alignment_gain": alignment_gain,
        "wrist_neutral_gain": wrist_neutral_gain,
        "elbow_neutral_gain": elbow_neutral_gain,
        "shoulder_neutral_gain": shoulder_neutral_gain,
        "temporal_gain": temporal_gain,
        "acceleration_gain": acceleration_gain,
        "joint_center_gain": joint_center_gain,
        "collision_gain": collision_gain,
        "cartesian_target_mutation_allowed": False,
        "semantic_literal_frame_dependency": False,
    }


def posture_metrics(
    runtime: ActiveG1Dex3,
    arm_q: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any]:
    """Compute full-distribution robot posture metrics for both arms."""
    arm_q = np.asarray(arm_q, dtype=np.float64)
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    dt = float(np.median(np.diff(timestamps)))
    result: dict[str, Any] = {"arms": {}}
    torso_id = runtime.body_id("torso_link")
    for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
        align, elbow_bend, elbow_center, wrist_center = [], [], [], []
        for aq, lq, rq in zip(arm_q, left_hand_q, right_hand_q):
            runtime.assign(aq, lq, rq)
            shoulder = runtime.model_to_scene_position(runtime.data.xpos[runtime.body_id(f"{side}_shoulder_roll_link")])
            elbow = runtime.model_to_scene_position(runtime.data.xpos[runtime.body_id(f"{side}_elbow_link")])
            wrist = runtime.wrist_pose(side)
            torso = runtime.model_to_scene_position(runtime.data.xpos[torso_id])
            upper = elbow - shoulder
            forearm = wrist[:3, 3] - elbow
            upper /= max(float(np.linalg.norm(upper)), 1e-12)
            forearm /= max(float(np.linalg.norm(forearm)), 1e-12)
            align.append(math.degrees(math.acos(np.clip(abs(float(np.dot(wrist[:3, 1], forearm))), 0.0, 1.0))))
            elbow_bend.append(math.degrees(math.acos(np.clip(float(np.dot(-upper, forearm)), -1.0, 1.0))))
            elbow_center.append(float(np.linalg.norm(elbow - torso)))
            wrist_center.append(float(np.linalg.norm(wrist[:3, 3] - torso)))
        alignment = np.asarray(align)
        bend = np.asarray(elbow_bend)
        q = arm_q[:, block]
        result["arms"][side] = {
            "forearm_wrist_alignment_deg": {
                "mean": float(np.mean(alignment)), "median": float(np.median(alignment)),
                "p95": float(np.quantile(alignment, 0.95)), "max": float(np.max(alignment)),
            },
            "elbow_bend_deg": {
                "mean": float(np.mean(bend)), "median": float(np.median(bend)),
                "p05": float(np.quantile(bend, 0.05)), "p95": float(np.quantile(bend, 0.95)),
                "min": float(np.min(bend)), "max": float(np.max(bend)),
            },
            "wrist_joint_vector_norm_rad": {
                "mean": float(np.mean(np.linalg.norm(q[:, 4:7], axis=1))),
                "p95": float(np.quantile(np.linalg.norm(q[:, 4:7], axis=1), 0.95)),
                "max": float(np.max(np.linalg.norm(q[:, 4:7], axis=1))),
            },
            "minimum_elbow_to_torso_center_distance_m": float(np.min(elbow_center)),
            "minimum_wrist_to_torso_center_distance_m": float(np.min(wrist_center)),
            "joint_ranges_rad": {
                "shoulder_pitch": [float(np.min(q[:, 0])), float(np.max(q[:, 0]))],
                "shoulder_roll": [float(np.min(q[:, 1])), float(np.max(q[:, 1]))],
                "shoulder_yaw": [float(np.min(q[:, 2])), float(np.max(q[:, 2]))],
                "elbow": [float(np.min(q[:, 3])), float(np.max(q[:, 3]))],
                "wrist_roll": [float(np.min(q[:, 4])), float(np.max(q[:, 4]))],
                "wrist_pitch": [float(np.min(q[:, 5])), float(np.max(q[:, 5]))],
                "wrist_yaw": [float(np.min(q[:, 6])), float(np.max(q[:, 6]))],
            },
        }
    margin = np.minimum(arm_q - limits[:, 0], limits[:, 1] - arm_q)
    velocity = np.diff(arm_q, axis=0) / dt
    acceleration = np.diff(arm_q, n=2, axis=0) / (dt * dt)
    minimum = np.unravel_index(int(np.argmin(margin)), margin.shape)
    result["global"] = {
        "minimum_joint_margin_rad": float(margin[minimum]),
        "limiting_sample": int(minimum[0]),
        "limiting_joint_index": int(minimum[1]),
        "maximum_step_rad": float(np.max(np.abs(np.diff(arm_q, axis=0)))),
        "rms_velocity_rad_s": float(np.sqrt(np.mean(velocity ** 2))),
        "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
        "rms_acceleration_rad_s2": float(np.sqrt(np.mean(acceleration ** 2))),
        "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "joint_limit_violation_count": int(np.sum(margin < 0.0)),
    }
    return result
