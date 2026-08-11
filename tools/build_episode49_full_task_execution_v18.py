#!/usr/bin/env python3
"""Build the single Episode-49 v18 execution-oriented G1 candidate.

The v14 left/right Cartesian arrays are immutable inputs, never optimization
variables.  Arm changes are restricted to redundant posture, joint-margin,
partial-orientation continuity, and collision repair.  Dex3 is a pair of
predefined semantic hand adapters: the user-approved physical left
thumb/index phone pinch and one physical right thumb/index accessory attempt.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from pxr import Usd, UsdGeom
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_execution_physics_v17 as v17  # noqa: E402
from aloha_g1_v15.kinematics import ActiveG1Dex3, sha256_file  # noqa: E402
from aloha_g1_v15.semantic_input import load_human_reviewed_development_timeline  # noqa: E402
from aloha_g1_v17.trajectory import (  # noqa: E402
    _collision_scalars,
    audit_collision_classifier_integrity,
    build_semantic_local_orientation_targets,
    evaluate_kinematic_candidate,
)
from aloha_g1_v17_2.trajectory import (  # noqa: E402
    _final_release_start,
    _source_close_start,
    _write_blend,
    posture_metrics,
)


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_full_task_execution_v18"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
V171 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1"
V172 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2"
PHOTO = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
PRIOR = OUT / "prior_failure_lessons_v18.json"
MAGNET = ROOT / "isaaclab_magsafe_fixed_scene/magnet_config_v2.json"
METHOD = "ALOHA_PRIMARY_EP49_FULL_TASK_EXECUTION_V18"


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_npz(path: Path, **value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **value)
    os.replace(temporary, path)


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def minimum_jerk(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return value ** 3 * (10.0 - 15.0 * value + 6.0 * value ** 2)


def two_stage_blend(
    output: np.ndarray,
    start: int,
    end: int,
    open_q: np.ndarray,
    pre_q: np.ndarray,
    closed_q: np.ndarray,
    *,
    pregrasp_fraction: float,
) -> None:
    """C2 OPEN->PREGRASP->CLOSED blend over a semantic/source interval."""
    if end <= start:
        output[start] = closed_q
        return
    progress = np.linspace(0.0, 1.0, end - start + 1)
    into_pre = minimum_jerk(np.clip(progress / pregrasp_fraction, 0.0, 1.0))
    into_close = minimum_jerk(np.clip(
        (progress - pregrasp_fraction) / (1.0 - pregrasp_fraction), 0.0, 1.0
    ))
    segment = (1.0 - into_pre[:, None]) * open_q + into_pre[:, None] * pre_q
    output[start : end + 1] = (
        (1.0 - into_close[:, None]) * segment + into_close[:, None] * closed_q
    )


def stage_bbox(stage: Usd.Stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
    return np.asarray(box.GetMin(), dtype=np.float64), np.asarray(box.GetMax(), dtype=np.float64)


def contact_point(runtime: ActiveG1Dex3, side: str, role: str) -> np.ndarray:
    return runtime.contact_pose(f"{side}_{role}")[0]


def make_right_thumb_index_primitive(
    runtime: ActiveG1Dex3,
    arm_at_grasp: np.ndarray,
    left_q_at_grasp: np.ndarray,
    accessory_min: np.ndarray,
    accessory_max: np.ndarray,
    safe_open: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One geometry-level right thumb/index primitive, never per-frame IK.

    The immutable v14 palm is outside digit reach.  We therefore solve only a
    collision-free distal thumb/index aperture pointing toward the closest
    accessible accessory rim and retain the measured reach warning.  This is
    a fixed execution primitive, not a claim of contact feasibility.
    """
    limits = runtime.hand_limits["right"]
    task_lo = limits[:5, 0] + 0.03
    task_hi = limits[:5, 1] - 0.03
    target_aperture_m = float(accessory_max[0] - accessory_min[0])
    target_aperture_m = float(np.clip(target_aperture_m, 0.045, 0.055))
    accessory_center = 0.5 * (accessory_min + accessory_max)
    target = accessory_center.copy()
    target[2] = accessory_max[2]
    middle = np.asarray(safe_open[5:], dtype=np.float64)

    def state(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
        q = np.r_[value, middle]
        runtime.assign(arm_at_grasp, left_q_at_grasp, q)
        index = contact_point(runtime, "right", "A")
        thumb = contact_point(runtime, "right", "B")
        return index, thumb, runtime.penetrating_contacts()

    def residual(value: np.ndarray) -> np.ndarray:
        index, thumb, contacts = state(value)
        midpoint = 0.5 * (index + thumb)
        aperture = np.linalg.norm(index - thumb)
        # The midpoint term asks for the nearest feasible outer/rim approach;
        # it cannot move the immutable palm or arm trajectory.
        collision_depth = max(
            [max(0.0, -float(row["distance_m"])) for row in contacts] or [0.0]
        )
        return np.r_[
            8.0 * (midpoint - target),
            20.0 * (aperture - target_aperture_m),
            0.12 * (value - np.asarray(safe_open[:5], dtype=np.float64)),
            5000.0 * collision_depth,
        ]

    seed = np.asarray([0.06, 1.14, -0.17, -0.98, -0.37], dtype=np.float64)
    seed = np.clip(seed, task_lo + 1e-8, task_hi - 1e-8)
    result = least_squares(
        residual, seed, bounds=(task_lo, task_hi), max_nfev=180,
        ftol=1e-12, xtol=1e-12, gtol=1e-12, x_scale="jac",
    )
    selected = np.r_[result.x, middle]
    index, thumb, contacts = state(result.x)
    midpoint = 0.5 * (index + thumb)
    nearest = np.minimum(np.maximum(midpoint, accessory_min), accessory_max)
    palm = runtime.palm_state("right")[0]
    palm_nearest = np.minimum(np.maximum(palm, accessory_min), accessory_max)
    return selected, {
        "status": "RIGHT_ACCESSORY_PRIMITIVE_GEOMETRY_WARNING",
        "name": "RIGHT_THUMB_INDEX_ACCESSORY_PINCH",
        "physical_task_fingers": ["RIGHT_THUMB", "RIGHT_INDEX"],
        "physical_mapping": {
            "RIGHT_INDEX": ["right_hand_index_0_joint", "right_hand_index_1_joint"],
            "RIGHT_THUMB": ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"],
            "RIGHT_THIRD_NON_TASK": ["right_hand_middle_0_joint", "right_hand_middle_1_joint"],
        },
        "joint_names": list(runtime.hand_joint_names["right"]),
        "q_rad": selected,
        "open_q_rad": safe_open,
        "third_q_unchanged_from_collision_safe_open": True,
        "index_contact_world_m": index,
        "thumb_contact_world_m": thumb,
        "pinch_center_world_m": midpoint,
        "thumb_index_aperture_m": float(np.linalg.norm(index - thumb)),
        "accessory_bbox_world_m": [accessory_min, accessory_max],
        "pinch_center_to_accessory_bbox_m": float(np.linalg.norm(midpoint - nearest)),
        "palm_to_accessory_bbox_m": float(np.linalg.norm(palm - palm_nearest)),
        "prohibited_static_collision_records": len(contacts),
        "optimizer": {
            "method": "single deterministic active-geometry least-squares calibration",
            "success": bool(result.success), "cost": float(result.cost),
            "variables": list(runtime.hand_joint_names["right"][:5]),
            "per_frame_finger_ik": False,
            "cartesian_arm_variable_count": 0,
        },
        "warning_reason": (
            "The immutable v14 right palm path remains outside physical digit "
            "reach of the authored accessory; the single primitive is retained "
            "as an honest execution attempt and never drives arm translation."
        ),
    }


def build_hands(
    timeline: Any,
    optimized_action: np.ndarray,
    timestamps: np.ndarray,
    left_primitive: dict[str, Any],
    right_open: np.ndarray,
    right_pinch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    event = lambda name: int(timeline.event(name).action_index)
    length = timeline.trajectory_length
    left_open = np.asarray(left_primitive["LEFT_PHONE_OPEN"], dtype=np.float64).copy()
    left_pre = np.asarray(left_primitive["LEFT_PHONE_PREGRASP"], dtype=np.float64).copy()
    left_pinch = np.asarray(left_primitive["LEFT_PHONE_FINGERTIP_PINCH"], dtype=np.float64)
    # The open non-task third digit otherwise intersects the table while the
    # immutable palm approaches low over the supported phone.  Tuck only that
    # non-task digit during OPEN/PREGRASP; it returns continuously to the exact
    # approved Candidate-A value by the named grasp event.  Thumb/index q and
    # the final seven-DOF Candidate-A vector are untouched.
    left_open[5:] = np.asarray([-1.0, -1.0], dtype=np.float64)
    left_pre[5:] = left_open[5:]
    right_pre = 0.5 * (right_open + right_pinch)
    left = np.repeat(left_open[None], length, axis=0)
    right = np.repeat(right_open[None], length, axis=0)

    left_grasp = event("left_phone_grasp_start")
    charger = event("phone_charger_attachment_complete")
    left_release = event("left_phone_release_complete")
    portrait = event("phone_portrait_reached")
    right_grasp = event("right_accessory_grasp_start")
    removed = event("accessory_removed")
    right_release = event("right_accessory_release_complete")
    left_start = _source_close_start(
        optimized_action[:, 6], timeline.start_index, left_grasp
    )
    right_start = _source_close_start(
        optimized_action[:, 13], portrait, right_grasp
    )
    right_release_start, release_audit = _final_release_start(
        optimized_action[:, 13], timestamps, removed, right_release
    )

    # Unlike v17.2, Candidate A is fully reached at the named grasp event.
    # This models table-supported acquisition before the source-derived lift.
    two_stage_blend(
        left, left_start, left_grasp, left_open, left_pre, left_pinch,
        pregrasp_fraction=0.55,
    )
    left[left_grasp : charger + 1] = left_pinch
    _write_blend(left, charger, left_release, left_pinch, left_open)
    left[left_release:] = left_open

    two_stage_blend(
        right, right_start, right_grasp,
        right_open, right_pre, right_pinch, pregrasp_fraction=0.55,
    )
    right[right_grasp : right_release_start + 1] = right_pinch
    _write_blend(right, right_release_start, right_release, right_pinch, right_open)
    right[right_release:] = right_open
    return left, right, {
        "driver": "GENERIC_SEMANTIC_TIMELINE_PLUS_SOURCE_GRIPPER_INTENT",
        "interpolation": "C2 minimum-jerk",
        "left_sequence": ["OPEN", "PREGRASP", "CANDIDATE_A_PINCH", "HOLD", "ROTATION_HOLD", "TRANSPORT_HOLD", "CHARGER_HOLD", "RELEASE", "OPEN"],
        "right_sequence": ["OPEN", "PREGRASP", "THUMB_INDEX_ACCESSORY_PINCH", "HOLD", "REMOVAL_HOLD", "RELEASE", "OPEN"],
        "left_pinch_reached_at_named_grasp_event": True,
        "left_candidate_A_q_changed": False,
        "left_non_task_third_collision_safe_open_q_rad": left_open[5:],
        "left_non_task_third_reason": "table-clear OPEN/PREGRASP posture; never a phone contact or pinch-frame input",
        "left_approach_start_action_index_for_provenance_only": left_start,
        "right_approach_start_action_index_for_provenance_only": right_start,
        "right_release_detection": release_audit,
        "runtime_literal_semantic_frame_dependency": False,
    }


def c2_margin_blend(
    base: np.ndarray,
    safe: np.ndarray,
    limits: np.ndarray,
    timestamps: np.ndarray,
    threshold_rad: float = 0.03,
    ramp_seconds: float = 1.0 / 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use a verified null-space solution only where base margin is deficient."""
    margin = np.minimum(base - limits[:, 0], limits[:, 1] - base)
    weight = np.zeros_like(base)
    dt = float(np.median(np.diff(timestamps)))
    ramp = max(1, int(round(ramp_seconds / dt)))
    components: list[dict[str, Any]] = []
    for joint in range(base.shape[1]):
        rows = np.flatnonzero(margin[:, joint] < threshold_rad + 1e-7)
        if not len(rows):
            continue
        groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        for group in groups:
            start, end = int(group[0]), int(group[-1])
            weight[start : end + 1, joint] = 1.0
            for offset in range(1, ramp + 1):
                normalized = 1.0 - offset / float(ramp + 1)
                value = float(minimum_jerk(np.asarray([normalized]))[0])
                if start - offset >= 0:
                    weight[start - offset, joint] = max(weight[start - offset, joint], value)
                if end + offset < len(weight):
                    weight[end + offset, joint] = max(weight[end + offset, joint], value)
            components.append({
                "joint_index": joint,
                "deficient_start_action_index_for_provenance_only": start,
                "deficient_end_action_index_for_provenance_only": end,
            })
    value = base + weight * (safe - base)
    return value, {
        "method": "C2 blend to the previously collision-safe v17.2 null-space solution only where joint margin is deficient",
        "threshold_rad": threshold_rad,
        "ramp_seconds": ramp_seconds,
        "ramp_samples_resolved_from_timestamps": ramp,
        "components_for_provenance_only": components,
        "cartesian_target_modified": False,
        "literal_semantic_frame_dependency": False,
    }


def collision_frames(
    runtime: ActiveG1Dex3,
    arm: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[int, list[int]]:
    count = 0
    frames: list[int] = []
    for frame in range(len(arm)):
        runtime.assign(arm[frame], left[frame], right[frame])
        rows = runtime.penetrating_contacts()
        count += len(rows)
        if rows:
            frames.append(frame)
    return count, frames


def project_immutable_cartesian_positions(
    runtime: ActiveG1Dex3,
    arm: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project margin-blended posture back to the exact protected XYZ path."""
    output = np.asarray(arm, dtype=np.float64).copy()
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for frame in range(len(output)):
        for side, target, block in (
            ("left", target_left[frame], slice(0, 7)),
            ("right", target_right[frame], slice(7, 14)),
        ):
            runtime.assign(output[frame], left_hand[frame], right_hand[frame])
            before = float(np.linalg.norm(runtime.palm_state(side)[0] - target))
            # The contract is a 5 mm task-space gate, not artificial exact-q
            # reconstruction.  Avoid re-solving already-valid samples because
            # redundant IK can select a distant shoulder/wrist branch while
            # improving a sub-millimetre residual that has no task value.
            if before <= 4.5e-3:
                continue
            original = output[frame, block].copy()
            previous = output[frame - 1, block] if frame else original
            following = arm[frame + 1, block] if frame + 1 < len(output) else original
            lo = np.maximum(limits[block, 0] + 0.03, original - 0.45)
            hi = np.minimum(limits[block, 1] - 0.03, original + 0.45)
            whole = output[frame].copy()

            def residual(value: np.ndarray) -> np.ndarray:
                whole[block] = value
                runtime.assign(whole, left_hand[frame], right_hand[frame])
                position = runtime.palm_state(side)[0]
                return np.r_[
                    14000.0 * (position - target),
                    0.75 * (value - original),
                    0.10 * (value - 0.5 * (previous + following)),
                ]

            solved = least_squares(
                residual, np.clip(original, lo + 1e-9, hi - 1e-9),
                bounds=(lo, hi), max_nfev=60,
                ftol=1e-11, xtol=1e-11, gtol=1e-11, x_scale="jac",
            )
            output[frame, block] = solved.x
            whole[block] = solved.x
            runtime.assign(whole, left_hand[frame], right_hand[frame])
            after = float(np.linalg.norm(runtime.palm_state(side)[0] - target))
            rows.append({
                "action_index_for_provenance_only": frame,
                "side": side, "error_before_m": before, "error_after_m": after,
            })
    return output, {
        "method": "position-only bounded IK projection after null-space margin blending",
        "projected_sample_side_count": len(rows),
        "maximum_error_before_m": max([row["error_before_m"] for row in rows] or [0.0]),
        "maximum_error_after_m": max([row["error_after_m"] for row in rows] or [0.0]),
        "rows_for_provenance_only": rows,
        "cartesian_target_modified": False,
    }


def collision_nullspace_repair(
    runtime: ActiveG1Dex3,
    arm: np.ndarray,
    base: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair only frames surrounding measured self-contact in task null space."""
    output = np.asarray(arm, dtype=np.float64).copy()
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    before_count, before_frames = collision_frames(runtime, output, left_hand, right_hand)
    dt = float(np.median(np.diff(timestamps)))
    padding = max(2, int(round(0.2 / dt)))
    active: set[int] = set()
    for frame in before_frames:
        active.update(range(max(0, frame - padding), min(len(output), frame + padding + 1)))
    sweep_rows: list[dict[str, Any]] = []
    for sweep in range(4 if active else 0):
        for frame in sorted(active):
            for side, target, block in (
                ("left", target_left[frame], slice(0, 7)),
                ("right", target_right[frame], slice(7, 14)),
            ):
                whole = output[frame].copy()
                seed = base[frame, block]
                previous = output[frame - 1, block] if frame else seed
                following = output[frame + 1, block] if frame + 1 < len(output) else seed
                previous2 = output[frame - 2, block] if frame > 1 else previous
                lo = np.maximum(limits[block, 0] + 0.03, seed - 0.55)
                hi = np.minimum(limits[block, 1] - 0.03, seed + 0.55)

                def residual(value: np.ndarray) -> np.ndarray:
                    whole[block] = value
                    runtime.assign(whole, left_hand[frame], right_hand[frame])
                    position = runtime.palm_state(side)[0]
                    margin = np.minimum(
                        value - limits[block, 0], limits[block, 1] - value
                    )
                    return np.r_[
                        8000.0 * (position - target),
                        0.12 * (value - seed),
                        0.25 * (value - 0.5 * (previous + following)),
                        0.08 * (value - 2.0 * previous + previous2),
                        5.0 * np.maximum(0.0, 0.05 - margin),
                        180000.0 * _collision_scalars(runtime, side),
                    ]

                x0 = np.clip(output[frame, block], lo + 1e-9, hi - 1e-9)
                solved = least_squares(
                    residual, x0, bounds=(lo, hi), max_nfev=80,
                    ftol=1e-11, xtol=1e-11, gtol=1e-11, x_scale="jac",
                )
                output[frame, block] = solved.x
        count, frames = collision_frames(runtime, output, left_hand, right_hand)
        sweep_rows.append({"sweep": sweep + 1, "contact_records": count, "contact_frames": frames})
        if count == 0:
            break
    after_count, after_frames = collision_frames(runtime, output, left_hand, right_hand)
    return output, {
        "method": "measured-contact-local task-null-space least-squares repair",
        "before_contact_records": before_count,
        "before_contact_frames_for_provenance_only": before_frames,
        "semantic_time_padding_seconds": 0.2,
        "padding_samples_resolved_from_timestamps": padding,
        "sweeps": sweep_rows,
        "after_contact_records": after_count,
        "after_contact_frames_for_provenance_only": after_frames,
        "cartesian_target_modified": False,
    }


def temporal_metrics(q: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    dt = float(np.median(np.diff(timestamps)))
    step = np.diff(q, axis=0)
    velocity = step / dt
    acceleration = np.diff(q, n=2, axis=0) / dt ** 2
    jerk = np.diff(q, n=3, axis=0) / dt ** 3
    sign = np.sign(velocity)
    reversal = sign[1:] * sign[:-1] < 0.0
    centered = q - np.mean(q, axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequency = np.fft.rfftfreq(len(q), d=dt)
    high = frequency >= 5.0
    energy = float(np.sum(spectrum[high]) / max(np.sum(spectrum[1:]), 1e-12))

    def stats(value: np.ndarray) -> dict[str, float]:
        absolute = np.abs(value)
        return {
            "rms": float(np.sqrt(np.mean(value ** 2))),
            "p95_absolute": float(np.quantile(absolute, 0.95)),
            "maximum_absolute": float(np.max(absolute)),
        }
    return {
        "maximum_joint_step_rad": float(np.max(np.abs(step))),
        "velocity_rad_s": stats(velocity),
        "acceleration_rad_s2": stats(acceleration),
        "jerk_rad_s3": stats(jerk),
        "frame_to_frame_sign_reversal_rate": float(np.mean(reversal)),
        "high_frequency_energy_fraction_ge_5hz": energy,
    }


def dex3_metrics(q: np.ndarray, limits: np.ndarray, names: list[str], timestamps: np.ndarray) -> dict[str, Any]:
    dt = float(np.median(np.diff(timestamps)))
    margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    velocity = np.diff(q, axis=0) / dt
    acceleration = np.diff(q, n=2, axis=0) / dt ** 2
    return {
        "joint_names": names,
        "per_joint_peak_to_peak_rad": dict(zip(names, np.ptp(q, axis=0).tolist())),
        "maximum_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
        "maximum_velocity_rad_s": float(np.max(np.abs(velocity))),
        "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "minimum_joint_margin_rad": float(np.min(margin)),
        "joint_limit_violation_count": int(np.sum(margin < -1e-9)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not PRIOR.is_file():
        raise RuntimeError("prior_failure_lessons_v18.json must be created before v18 solver code runs")
    immutable = {
        "optimized_action": v17.SOURCE,
        "v14_cartesian_targets": v17.V14_TARGET,
        "v14_root_config": v17.ROOT_CONFIG,
        "semantic_timeline": v17.TIMELINE,
        "scene_layout": v17.LAYOUT,
        "active_scene": v17.ACTIVE_SCENE,
        "fixed_scene": v17.FIXED_SCENE,
        "magnet_config": MAGNET,
        "approved_left_candidate_A": PHOTO / "left_phone_fingertip_pinch_primitive.json",
        "v17_1_trajectory": V171 / "final_arm_dex3_trajectory.npz",
        "v17_2_trajectory": V172 / "final_arm_dex3_trajectory.npz",
    }
    missing = [str(path) for path in immutable.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    hashes_before = {name: sha256_file(path) for name, path in immutable.items()}
    if hashes_before["optimized_action"] != "a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c":
        raise RuntimeError("authoritative source archive SHA mismatch")

    with np.load(v17.SOURCE, allow_pickle=False) as source:
        optimized_action = source["optimized_action"].copy()
        timestamps = source["timestamp"].copy()
        fps = float(source["fps"])
    with np.load(v17.PHASE_LIBRARY, allow_pickle=False) as phase:
        source_left_position = phase["left_tcp_position"].copy()
        source_right_position = phase["right_tcp_position"].copy()
        source_left_rotation = phase["left_tcp_rotation"].copy()
        source_right_rotation = phase["right_tcp_rotation"].copy()
    with np.load(V171 / "final_arm_dex3_trajectory.npz", allow_pickle=False) as archive:
        v171 = {name: archive[name].copy() for name in archive.files}
    with np.load(V172 / "final_arm_dex3_trajectory.npz", allow_pickle=False) as archive:
        v172 = {name: archive[name].copy() for name in archive.files}
    with np.load(v17.V14_TARGET, allow_pickle=False) as archive:
        direct_left = archive["corrected_left_position"].copy()
        direct_right = archive["corrected_right_position"].copy()
    target_left = v171["v14_left_position_target"].copy()
    target_right = v171["v14_right_position_target"].copy()
    if not np.array_equal(target_left, direct_left) or not np.array_equal(target_right, direct_right):
        raise RuntimeError("BLOCKED_V18_CARTESIAN_MUTATION")
    if optimized_action.shape != (990, 14) or fps != 30.0:
        raise RuntimeError("Episode-49 source contract failed")

    timeline = load_human_reviewed_development_timeline(
        v17.TIMELINE, v17.ALIGNMENT, optimized_action, timestamps,
        source_left_position, source_right_position,
        source_left_rotation, source_right_rotation,
        trajectory_path=v17.SOURCE, fk_model_path=v17.MODEL,
        task_geometry_path=v17.LAYOUT,
    )
    root_position = v171["g1_root"].astype(np.float64)
    runtime = ActiveG1Dex3(v17.MODEL, v17.DEX3_MAPPING, v17.PALM_CONFIG, root_position)
    stage = Usd.Stage.Open(str(v17.ACTIVE_SCENE))
    accessory_min, accessory_max = stage_bbox(stage, "/World/MagSafeScene/Accessory")
    phone_initial = v17.usd_pose(stage, "/World/MagSafeScene/Phone")
    pad = v17.usd_pose(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    phone_charger = v17.phone_on_pad_pose(pad)
    event = lambda name: int(timeline.event(name).action_index)

    approved = json.loads(immutable["approved_left_candidate_A"].read_text(encoding="utf-8"))
    approved_q = np.asarray(approved["selected_static_q_rad"], dtype=np.float64)
    required_q = np.asarray([
        -0.517737046259834, 0.7470531658390861, 0.050425648920224796,
        -0.6619250941551765, -1.7053299997433065, -0.1, -0.1,
    ])
    if not np.array_equal(approved_q, required_q):
        raise RuntimeError("approved Candidate A q mismatch")
    left_primitives = approved["all_primitives_q_rad"]

    old_primitives = json.loads(
        (V171 / "dex3_magsafe_execution_primitives_v17_1.sim.json").read_text(encoding="utf-8")
    )["primitives"]
    right_open = np.asarray(old_primitives["RIGHT_OPEN"], dtype=np.float64)
    # It is a physical thumb-index aperture despite the legacy filename; the
    # old physical middle/C hook vector is never used.
    right_pinch, right_audit = make_right_thumb_index_primitive(
        runtime,
        v171["v14_reference_arm_q"][event("right_accessory_grasp_start")],
        approved_q,
        accessory_min, accessory_max, right_open,
    )
    left_hand, right_hand, hand_audit = build_hands(
        timeline, optimized_action, timestamps, left_primitives,
        right_open, right_pinch,
    )

    # Start from the lower-jitter v17.1 left task orientation and the v14
    # source-relative right branch.  This explicitly removes the historical
    # middle/C-hook wrist registration.  Only deficient margin components are
    # transitioned to the already verified v17.2 null-space solution.
    base_arm = np.c_[v171["arm_qpos"][:, :7], v171["v14_reference_arm_q"][:, 7:]]
    margin_arm, margin_audit = c2_margin_blend(
        base_arm, v172["arm_qpos"], np.asarray(runtime.info["joint_limits"]), timestamps
    )
    projected_arm, position_projection = project_immutable_cartesian_positions(
        runtime, margin_arm, target_left, target_right, left_hand, right_hand
    )
    arm_q, collision_repair = collision_nullspace_repair(
        runtime, projected_arm, base_arm, target_left, target_right,
        left_hand, right_hand, timestamps,
    )

    # Orientation evaluation uses the same source-relative task axes as v17.1
    # for the left.  Right achieved motion is the neutral/source-relative v14
    # branch; the old C-hook endpoint is not an optimization target.
    anchors = json.loads(v17.V14_ANCHORS.read_text(encoding="utf-8"))
    charger_rows = [
        row for row in anchors.values()
        if isinstance(row, dict)
        and row.get("action_index") == event("phone_charger_attachment_complete")
        and "wrist_rotation" in row
    ]
    v16_carrier = json.loads(v17.V16_LEFT_CARRIER.read_text(encoding="utf-8"))
    task_grasp_rotation = np.asarray(v16_carrier["selected"]["initial_wrist"], dtype=np.float64)[:3, :3]
    task_charger_rotation = np.asarray(charger_rows[0]["wrist_rotation"], dtype=np.float64)
    runtime.assign(v171["v14_reference_arm_q"][event("right_accessory_grasp_start")], approved_q, right_pinch)
    neutral_right_rotation = runtime.wrist_pose("right")[:3, :3]
    semantic_progress = {
        "left_source_close_progress": np.linspace(0.0, 1.0, len(optimized_action)),
        "right_source_close_progress": np.linspace(0.0, 1.0, len(optimized_action)),
        "left_source_signal_detected_approach_start": np.asarray(hand_audit["left_approach_start_action_index_for_provenance_only"]),
        "right_source_signal_detected_approach_start": np.asarray(hand_audit["right_approach_start_action_index_for_provenance_only"]),
    }
    targets = build_semantic_local_orientation_targets(
        timeline, runtime, v171["v14_reference_arm_q"], left_hand, right_hand,
        source_left_rotation, source_right_rotation, phone_initial, phone_charger,
        task_grasp_rotation, task_charger_rotation, neutral_right_rotation,
        semantic_progress,
        left_acquisition_strength=1.0,
        right_hook_strength=0.0,
        charger_strength=1.0,
    )
    targets["right_rotation"] = targets["v14_right_wrist"][:, :3, :3].copy()
    targets["right_axis_weight"][:] = 0.0

    layout = json.loads(v17.LAYOUT.read_text(encoding="utf-8"))
    table_height = float(layout["table"]["surface_height"])
    table_bounds = (0.0, float(layout["table"]["size_x"]), 0.0, float(layout["table"]["size_y"]))
    metrics = evaluate_kinematic_candidate(
        timeline, runtime, arm_q, left_hand, right_hand,
        target_left, target_right, targets,
        source_left_rotation, source_right_rotation,
        phone_initial, phone_charger, table_height, table_bounds,
    )
    achieved = metrics.pop("achieved")
    collision, raw_contacts = audit_collision_classifier_integrity(
        runtime, arm_q, left_hand, right_hand, table_height, table_bounds
    )
    metrics["collision"] = collision
    before_posture = posture_metrics(
        runtime, v172["arm_qpos"], v172["left_dex3_qpos"], v172["right_dex3_qpos"], timestamps
    )
    after_posture = posture_metrics(runtime, arm_q, left_hand, right_hand, timestamps)
    before_temporal = temporal_metrics(v172["arm_qpos"], timestamps)
    after_temporal = temporal_metrics(arm_q, timestamps)
    left_hand_metrics = dex3_metrics(
        left_hand, runtime.hand_limits["left"], list(runtime.hand_joint_names["left"]), timestamps
    )
    right_hand_metrics = dex3_metrics(
        right_hand, runtime.hand_limits["right"], list(runtime.hand_joint_names["right"]), timestamps
    )

    common = {
        "optimized_action": optimized_action,
        "source_timestamps": timestamps,
        "arm_joint_names": v171["arm_joint_names"],
        "left_dex3_joint_names": v171["left_dex3_joint_names"],
        "right_dex3_joint_names": v171["right_dex3_joint_names"],
        "v14_reference_arm_q": v171["v14_reference_arm_q"],
        "v14_left_position_target": target_left,
        "v14_right_position_target": target_right,
        "g1_root": root_position,
        "workspace_scale": v171["workspace_scale"],
        "method": np.asarray(METHOD),
        "semantic_timeline_sha256": v171["semantic_timeline_sha256"],
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
        full_joint_q=np.c_[arm_q, left_hand, right_hand],
        fps=np.asarray(fps),
        primitive_source=np.asarray("approved_left_candidate_A_plus_single_right_thumb_index_adapter"),
        authoritative_for_real_robot=np.asarray(False),
    )

    hashes_after = {name: sha256_file(path) for name, path in immutable.items()}
    freeze = {
        "status": "V18_SOURCE_AND_SCIENTIFIC_INPUTS_FROZEN",
        "hashes_before": hashes_before, "hashes_after": hashes_after,
        "all_immutable_file_hashes_equal": hashes_before == hashes_after,
        "optimized_action_archive_sha256": hashes_after["optimized_action"],
        "optimized_action_shape": list(optimized_action.shape),
        "optimized_action_finite": bool(np.isfinite(optimized_action).all()),
        "validation_read_count": 0, "heldout_read_count": 0, "g1_expert_read_count": 0,
        "dds": False, "publisher": False, "real_robot_command": False,
    }
    cartesian_freeze = {
        "status": "ALOHA_CARTESIAN_BACKBONE_BYTE_IDENTICAL",
        "v14_file_sha256": hashes_after["v14_cartesian_targets"],
        "left_array_sha256_before": array_sha(direct_left),
        "left_array_sha256_after": array_sha(target_left),
        "right_array_sha256_before": array_sha(direct_right),
        "right_array_sha256_after": array_sha(target_right),
        "left_byte_identical": bool(np.array_equal(direct_left, target_left)),
        "right_byte_identical": bool(np.array_equal(direct_right, target_right)),
        "maximum_cartesian_target_difference_m": float(max(
            np.max(np.abs(direct_left - target_left)),
            np.max(np.abs(direct_right - target_right)),
        )),
        "cartesian_waypoints_added": 0,
        "cartesian_residuals_added": 0,
    }
    if not freeze["all_immutable_file_hashes_equal"] or cartesian_freeze["maximum_cartesian_target_difference_m"] != 0.0:
        raise RuntimeError("BLOCKED_V18_CARTESIAN_MUTATION")
    dump(OUT / "source_freeze_audit.json", freeze)
    dump(OUT / "v14_cartesian_freeze_audit.json", cartesian_freeze)
    dump(OUT / "v17_2_jitter_audit.json", {
        "status": "V17_2_JITTER_CONFIRMED_IN_TARGET_Q",
        "v17_2": before_temporal,
        "root_cause": [
            "frame-local null-space collision repairs",
            "discontinuous task-axis weight changes at named-boundary implementation",
            "redundant posture branch motion rather than Cartesian XYZ motion",
        ],
        "cartesian_XYZ_jitter_was_not_smoothed": True,
    })
    temporal_pass = bool(
        after_temporal["acceleration_rad_s2"]["rms"] < 0.7 * before_temporal["acceleration_rad_s2"]["rms"]
        and after_temporal["jerk_rad_s3"]["rms"] < 0.7 * before_temporal["jerk_rad_s3"]["rms"]
    )
    dump(OUT / "v18_temporal_stabilization.json", {
        "status": "V18_REDUNDANT_POSTURE_TEMPORALLY_STABILIZED" if temporal_pass else "V18_TEMPORAL_STABILIZATION_WARNING",
        "before_v17_2": before_temporal,
        "after_v18": after_temporal,
        "rms_acceleration_reduction_fraction": 1.0 - after_temporal["acceleration_rad_s2"]["rms"] / before_temporal["acceleration_rad_s2"]["rms"],
        "rms_jerk_reduction_fraction": 1.0 - after_temporal["jerk_rad_s3"]["rms"] / before_temporal["jerk_rad_s3"]["rms"],
        "margin_blend": margin_audit,
        "immutable_cartesian_projection": position_projection,
        "collision_nullspace_repair": collision_repair,
        "quaternion_sign_policy": "orientation is evaluated as rotation matrices; any reporting quaternion is canonicalized for consecutive positive dot product",
        "cartesian_XYZ_filtered_or_retimed": False,
    })

    # Correct physical thumb/index pinch task frame at the named grasp pose.
    grasp = event("left_phone_grasp_start")
    runtime.assign(arm_q[grasp], approved_q, right_hand[grasp])
    thumb_p, thumb_n = runtime.contact_pose("left_A")
    index_p, index_n = runtime.contact_pose("left_B")
    wrist = runtime.wrist_pose("left")
    center = 0.5 * (thumb_p + index_p)
    closing = index_p - thumb_p
    closing /= max(np.linalg.norm(closing), 1e-12)
    approach = wrist[:3, 0] - closing * float(np.dot(wrist[:3, 0], closing))
    approach /= max(np.linalg.norm(approach), 1e-12)
    lateral = np.cross(closing, approach)
    lateral /= max(np.linalg.norm(lateral), 1e-12)
    dump(OUT / "left_thumb_index_task_frame_v18.json", {
        "status": "PHYSICAL_LEFT_THUMB_INDEX_TASK_FRAME_VERIFIED",
        "source": PHOTO / "left_phone_contact_frames.json",
        "physical_thumb": {"role": "left_A", "link": runtime.contacts["left_A"].link, "contact_world_m_at_grasp": thumb_p, "normal_world": thumb_n},
        "physical_index": {"role": "left_B", "link": runtime.contacts["left_B"].link, "contact_world_m_at_grasp": index_p, "normal_world": index_n},
        "physical_third": {"role": "left_C", "link": runtime.contacts["left_C"].link, "task_role": "NON_TASK"},
        "pinch_center_world_m_at_grasp": center,
        "closing_axis_thumb_to_index_world": closing,
        "palm_approach_axis_world": approach,
        "lateral_axis_world": lateral,
        "third_finger_used_to_define_frame": False,
        "wrist_or_arm_used_to_fake_fingertip_alignment": False,
    })
    dump(OUT / "left_candidate_A_integration_audit.json", {
        "status": "USER_APPROVED_LEFT_CANDIDATE_A_INTEGRATED_UNCHANGED",
        "primitive_path": immutable["approved_left_candidate_A"],
        "primitive_sha256": hashes_after["approved_left_candidate_A"],
        "approved_q_rad": approved_q,
        "trajectory_q_at_named_grasp": left_hand[grasp],
        "exact_q_match": bool(np.array_equal(left_hand[grasp], approved_q)),
        "task_fingers": ["PHYSICAL_THUMB", "PHYSICAL_INDEX"],
        "third": "NON_TASK",
        "interpolation": hand_audit,
    })
    dump(OUT / "right_dex3_physical_identity_v18.json", {
        "status": "RIGHT_PHYSICAL_FINGER_IDENTITY_VERIFIED",
        "active_model": v17.MODEL,
        "joint_order": list(runtime.hand_joint_names["right"]),
        "RIGHT_INDEX": {"legacy_role": "A", "link": runtime.contacts["right_A"].link, "joints": runtime.contacts["right_A"].joint_names},
        "RIGHT_THUMB": {"legacy_role": "B", "link": runtime.contacts["right_B"].link, "joints": runtime.contacts["right_B"].joint_names},
        "RIGHT_THIRD": {"legacy_role": "C", "link": runtime.contacts["right_C"].link, "joints": runtime.contacts["right_C"].joint_names, "task_role": "NON_TASK"},
        "historical_C_hook_reused": False,
    })
    dump(OUT / "right_accessory_primitive_v18.json", right_audit | {
        "semantic_interpolation": hand_audit["right_sequence"],
        "source_gripper_intent_used": True,
        "episode_specific_cartesian_dependency": False,
    })
    magnet_payload = json.loads(MAGNET.read_text(encoding="utf-8"))
    dump(OUT / "magnet_physics_freeze_audit.json", {
        "status": "AUTHORED_MAGNET_PHYSICS_FROZEN",
        "path": MAGNET, "sha256_before": hashes_before["magnet_config"],
        "sha256_after": hashes_after["magnet_config"],
        "byte_identical": hashes_before["magnet_config"] == hashes_after["magnet_config"],
        "parameter_status": magnet_payload.get("metadata", {}).get("parameter_status"),
        "scripted_semantic_attach": False, "scripted_semantic_detach": False,
    })
    dump(OUT / "aloha_fidelity_v18.json", metrics["fidelity"])
    dump(OUT / "collision_audit_v18.json", collision)
    dump(OUT / "joint_temporal_metrics_v18.json", {
        "v17_2": before_temporal, "v18": after_temporal,
        "joint": metrics["joint"],
        "v17_2_posture": before_posture,
        "v18_posture": after_posture,
        "left_dex3": left_hand_metrics,
        "right_dex3": right_hand_metrics,
        "branch_continuity_pass": metrics["joint"]["branch_discontinuity_count"] == 0,
    })
    # Detailed collision accounting is useful for later review and retains
    # wrist/arm-finger rows rather than silently filtering them.
    with (OUT / "collision_records_v18.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in raw_contacts for key in row}) if raw_contacts else ["frame", "classification"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_contacts)

    kinematic_pass = bool(
        metrics["position"]["pass"]
        and metrics["fidelity"]["pass"]
        and metrics["joint"]["joint_limit_violation_count"] == 0
        and metrics["joint"]["branch_discontinuity_count"] == 0
        and collision["pass"]
        and temporal_pass
        and left_hand_metrics["joint_limit_violation_count"] == 0
        and right_hand_metrics["joint_limit_violation_count"] == 0
    )
    dump(OUT / "kinematic_full_review_v18.json", {
        "status": "V18_KINEMATIC_NUMERIC_PASS_PENDING_RENDER" if kinematic_pass else "BLOCKED_V18_KINEMATIC_NUMERIC_GATE",
        "pass": kinematic_pass,
        "position": metrics["position"], "orientation": metrics["orientation"],
        "fidelity": metrics["fidelity"], "joint": metrics["joint"],
        "collision": collision, "temporal_stabilization_pass": temporal_pass,
        "right_accessory_geometry_warning": right_audit["status"],
    })
    dump(OUT / "whole_motion_execution_sanity_v18.json", {
        "status": "PENDING_KINEMATIC_RENDER_AND_TRUE_PHYSICS",
        "numeric_kinematic_pass": kinematic_pass,
        "all_990_samples_built": len(arm_q) == 990,
        "full_task_physics_status": "NOT_RUN",
    })
    dump(OUT / "full_task_physics_status_v18.json", {
        "status": "NOT_RUN", "full_task_true_physics_pass": False,
        "one_uninterrupted_run_required": True,
    })
    dump(OUT / "build_summary_v18.json", {
        "status": "V18_SINGLE_TRAJECTORY_KINEMATIC_NUMERIC_PASS" if kinematic_pass else "V18_SINGLE_TRAJECTORY_KINEMATIC_NUMERIC_BLOCKED",
        "trajectory_samples": len(arm_q),
        "trajectory_sha256": sha256_file(OUT / "final_arm_dex3_trajectory.npz"),
        "cartesian_backbone_status": cartesian_freeze["status"],
        "position": metrics["position"], "fidelity": metrics["fidelity"],
        "orientation": metrics["orientation"], "joint": metrics["joint"],
        "collision": collision,
        "temporal": {"before": before_temporal, "after": after_temporal, "pass": temporal_pass},
        "right_accessory": right_audit["status"],
    })
    print(json.dumps(json.loads((OUT / "build_summary_v18.json").read_text()), indent=2))
    return 0 if kinematic_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
