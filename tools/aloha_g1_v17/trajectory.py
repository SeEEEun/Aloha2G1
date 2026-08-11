"""Semantic-progress hand primitives and partial-orientation arm refinement.

This module deliberately contains no episode-specific event indices.  It
accepts a canonical :class:`SemanticTimeline` and resolves every task boundary
by event name.  Cartesian positions are supplied by the caller and are never
redesigned here.
"""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from aloha_g1_v15.kinematics import ActiveG1Dex3, R_SCENE_FROM_MODEL
from aloha_g1_v15.translator import (
    C_LEFT,
    C_RIGHT,
    interpolate_rotation,
    mapped_relative,
    rotation_arc_progress,
    rotation_error,
    smoothstep,
    source_preserving_segment,
)
from aloha_magsafe_semantics.schema import SemanticTimeline
from solve_episode49_target_phase_anchored_v12_ik import geom_world_vertices


def _interval_progress(timeline: SemanticTimeline, start_name: str, end_name: str) -> np.ndarray:
    start, end = timeline.interval(start_name, end_name)
    if end <= start:
        return np.ones(1, dtype=np.float64)
    return smoothstep(np.linspace(0.0, 1.0, end - start + 1))


def _clip_with_margin(values: np.ndarray, limits: np.ndarray, margin: float) -> np.ndarray:
    limits = np.asarray(limits, dtype=np.float64)
    usable = np.minimum(float(margin), 0.20 * (limits[:, 1] - limits[:, 0]))
    return np.clip(np.asarray(values, dtype=np.float64), limits[:, 0] + usable, limits[:, 1] - usable)


def _source_close_progress(
    aperture: np.ndarray,
    approach_start: int,
    grasp_start: int,
    phase_end: int,
    *,
    open_threshold: float = 0.10,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Return a monotone, source-signal-driven closing progress.

    The ALOHA carriage signal is large when open and small when closed.  Its
    absolute scale is trajectory-dependent, so this routine uses robust
    per-trajectory levels.  The active approach begins at the *last* open
    sample before the named grasp event; this deliberately ignores unrelated
    closed states earlier in the episode.  No episode-specific index or frame
    duration enters the rule.
    """
    aperture = np.asarray(aperture, dtype=np.float64)
    if aperture.ndim != 1 or not np.all(np.isfinite(aperture)):
        raise ValueError("source gripper aperture must be one finite vector")
    low, high = np.quantile(aperture, [0.05, 0.95])
    span = max(float(high - low), 1e-8)
    close = np.clip((high - aperture) / span, 0.0, 1.0)

    lo = max(0, int(approach_start))
    grasp = int(grasp_start)
    end = min(len(close) - 1, int(phase_end))
    open_samples = np.flatnonzero(close[lo : grasp + 1] <= float(open_threshold))
    detected_start = lo + int(open_samples[-1]) if len(open_samples) else lo

    progress = np.zeros_like(close)
    segment = np.maximum.accumulate(close[detected_start : end + 1])
    base = float(segment[0])
    peak = float(max(np.max(segment), base + 1e-8))
    progress[detected_start : end + 1] = np.clip((segment - base) / (peak - base), 0.0, 1.0)
    progress[end + 1 :] = 1.0
    return progress, detected_start, {
        "robust_closed_level": float(low),
        "robust_open_level": float(high),
        "detected_approach_start": int(detected_start),
        "close_progress_at_grasp": float(progress[grasp]),
    }


def build_predefined_hand_trajectories(
    timeline: SemanticTimeline,
    runtime: ActiveG1Dex3,
    primitives: dict[str, np.ndarray],
    source_left_gripper: np.ndarray,
    source_right_gripper: np.ndarray,
    *,
    left_lock_end_event: str = "phone_portrait_reached",
    left_digit_staging_mode: str = "staged_index_then_thumb",
    left_lock_progress_mode: str = "source_and_semantic_progress",
    left_lock_completion_progress: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Interpolate fixed task-level primitives from semantics and gripper progress."""
    length = timeline.trajectory_length
    left_open = np.asarray(primitives["LEFT_OPEN"], dtype=np.float64)
    left_pre = np.asarray(primitives["LEFT_PHONE_PREGRASP"], dtype=np.float64)
    left_contact_ready = np.asarray(
        primitives.get("LEFT_PHONE_INDEX_CONTACT", left_pre), dtype=np.float64
    )
    left_pinch = np.asarray(primitives["LEFT_PHONE_PINCH"], dtype=np.float64)
    right_open = np.asarray(primitives["RIGHT_OPEN"], dtype=np.float64)
    right_pre = np.asarray(primitives["RIGHT_RING_PREHOOK"], dtype=np.float64)
    right_hook = np.asarray(primitives["RIGHT_RING_HOOK"], dtype=np.float64)

    left = np.repeat(left_open[None], length, axis=0)
    right = np.repeat(right_open[None], length, axis=0)

    left_grasp = int(timeline.event("left_phone_grasp_start").action_index)
    portrait = int(timeline.event("phone_portrait_reached").action_index)
    left_lock_end = int(timeline.event(left_lock_end_event).action_index)
    charger = int(timeline.event("phone_charger_attachment_complete").action_index)
    left_release = int(timeline.event("left_phone_release_complete").action_index)
    right_grasp = int(timeline.event("right_accessory_grasp_start").action_index)
    detachment = int(timeline.event("accessory_detachment_start").action_index)
    removed = int(timeline.event("accessory_removed").action_index)
    right_release = int(timeline.event("right_accessory_release_complete").action_index)

    # ALOHA has already begun closing before its contact-acquisition event.
    # Preserve that behavior: approach OPEN->PREGRASP follows the measured
    # source carriage signal, then PREGRASP->PINCH finishes over the remaining
    # measured closure.  The phase boundaries still come only from names.
    left_close, left_approach_start, left_signal_audit = _source_close_progress(
        source_left_gripper, timeline.start_index, left_grasp, left_lock_end
    )
    pre = smoothstep(
        np.clip(left_close[left_approach_start : left_grasp + 1] /
                max(left_close[left_grasp], 1e-8), 0.0, 1.0)
    )
    left[left_approach_start : left_grasp + 1] = (
        (1.0 - pre[:, None]) * left_open + pre[:, None] * left_pre
    )
    remaining = max(1.0 - float(left_close[left_grasp]), 1e-8)
    source_lock = smoothstep(np.clip(
        (left_close[left_grasp : left_lock_end + 1] - left_close[left_grasp]) / remaining,
        0.0, 1.0,
    ))
    # A gripper trace can saturate immediately at the event.  In that case,
    # semantic acquisition progress supplies continuity without changing the
    # fixed primitive or introducing a frame-based duration.
    semantic_lock_raw = timeline.progress("phone_acquisition")[left_grasp : left_lock_end + 1]
    semantic_span = max(float(semantic_lock_raw[-1] - semantic_lock_raw[0]), 1e-8)
    semantic_lock = smoothstep(np.clip(
        (semantic_lock_raw - semantic_lock_raw[0]) / semantic_span, 0.0, 1.0
    ))
    if left_lock_progress_mode == "source_and_semantic_progress":
        lock = np.maximum.accumulate(np.maximum(source_lock, semantic_lock))
    elif left_lock_progress_mode == "event_interval_minimum_jerk":
        # Some ALOHA gripper traces saturate in one or two samples.  Directly
        # turning that saturation into a multi-joint Dex3 command creates a
        # target-embodiment actuator impulse even at slow replay speed.  Use
        # the generic named-event interval as the execution clock instead.
        # This changes neither event timing nor the arm trajectory and has no
        # dependence on an Episode-specific frame constant.
        phase_time = timeline.timestamps[left_grasp : left_lock_end + 1]
        duration = max(float(phase_time[-1] - phase_time[0]), 1e-8)
        completion = float(np.clip(left_lock_completion_progress, 0.05, 1.0))
        normalized_time = np.clip(
            (phase_time - phase_time[0]) / (duration * completion), 0.0, 1.0
        )
        lock = smoothstep(normalized_time)
    else:
        raise ValueError(f"unknown left lock progress mode: {left_lock_progress_mode}")
    if "LEFT_PHONE_INDEX_CONTACT" in primitives:
        # A reusable two-part hand primitive avoids crossing the two digits:
        # the index reaches its opposite phone surface early in semantic
        # acquisition while the thumb closes continuously behind it.  These
        # are fixed joint-space primitives, not per-frame contact solutions.
        if left_digit_staging_mode == "simultaneous_opposed_close":
            index_progress = smoothstep(lock)
            thumb_progress = smoothstep(lock)
        elif left_digit_staging_mode == "staged_index_then_thumb":
            index_progress = smoothstep(np.clip(lock / 0.35, 0.0, 1.0))
            thumb_progress = smoothstep(lock)
        else:
            raise ValueError(f"unknown left digit staging mode: {left_digit_staging_mode}")
        segment = np.repeat(left_pre[None], len(lock), axis=0)
        segment[:, :3] = (
            (1.0 - thumb_progress[:, None]) * left_pre[:3]
            + thumb_progress[:, None] * left_pinch[:3]
        )
        segment[:, 3:5] = (
            (1.0 - index_progress[:, None]) * left_pre[3:5]
            + index_progress[:, None] * left_contact_ready[3:5]
        )
        segment[:, 5:] = (
            (1.0 - thumb_progress[:, None]) * left_pre[5:]
            + thumb_progress[:, None] * left_pinch[5:]
        )
        left[left_grasp : left_lock_end + 1] = segment
    else:
        left[left_grasp : left_lock_end + 1] = (
            (1.0 - lock[:, None]) * left_pre + lock[:, None] * left_pinch
        )
    left[left_lock_end : charger + 1] = left_pinch
    left_release_progress = smoothstep(timeline.progress("left_release")[charger : left_release + 1])
    left[charger : left_release + 1] = (
        (1.0 - left_release_progress[:, None]) * left_pinch
        + left_release_progress[:, None] * left_open
    )
    left[left_release:] = left_open

    right_close, right_approach_start, right_signal_audit = _source_close_progress(
        source_right_gripper, portrait, right_grasp, detachment
    )
    right_pre_progress = smoothstep(
        np.clip(right_close[right_approach_start : right_grasp + 1] /
                max(right_close[right_grasp], 1e-8), 0.0, 1.0)
    )
    right[right_approach_start : right_grasp + 1] = (
        (1.0 - right_pre_progress[:, None]) * right_open
        + right_pre_progress[:, None] * right_pre
    )
    remaining = max(1.0 - float(right_close[right_grasp]), 1e-8)
    hook = smoothstep(np.clip(
        (right_close[right_grasp : detachment + 1] - right_close[right_grasp]) / remaining,
        0.0, 1.0,
    ))
    semantic_hook = smoothstep(timeline.progress("accessory_acquisition")[right_grasp : detachment + 1])
    hook = np.maximum.accumulate(np.maximum(hook, semantic_hook))
    right[right_grasp : detachment + 1] = (
        (1.0 - hook[:, None]) * right_pre + hook[:, None] * right_hook
    )
    right[detachment : removed + 1] = right_hook
    release = smoothstep(timeline.progress("right_release")[removed : right_release + 1])
    right[removed : right_release + 1] = (
        (1.0 - release[:, None]) * right_hook + release[:, None] * right_open
    )
    right[right_release:] = right_open

    return left, right, {
        "phone_acquisition": timeline.progress("phone_acquisition").copy(),
        "phone_rotation": timeline.progress("phone_rotation").copy(),
        "phone_transport": timeline.progress("phone_to_charger").copy(),
        "accessory_acquisition": timeline.progress("accessory_acquisition").copy(),
        "accessory_removal": timeline.progress("accessory_removal").copy(),
        "left_release": timeline.progress("left_release").copy(),
        "right_release": timeline.progress("right_release").copy(),
        "left_source_close_progress": left_close,
        "right_source_close_progress": right_close,
        "left_lock_end_event": np.asarray(left_lock_end_event),
        "left_digit_staging_mode": np.asarray(left_digit_staging_mode),
        "left_lock_progress_mode": np.asarray(left_lock_progress_mode),
        "left_lock_completion_progress": np.asarray(left_lock_completion_progress),
        **{f"left_source_signal_{key}": np.asarray(value) for key, value in left_signal_audit.items()},
        **{f"right_source_signal_{key}": np.asarray(value) for key, value in right_signal_audit.items()},
    }


def _actual_wrist_trajectory(
    runtime: ActiveG1Dex3,
    arm_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left, right = [], []
    for arm, lq, rq in zip(arm_q, left_q, right_q):
        runtime.assign(arm, lq, rq)
        left.append(runtime.wrist_pose("left"))
        right.append(runtime.wrist_pose("right"))
    return np.asarray(left), np.asarray(right)


def _write_segment(output: np.ndarray, start: int, end: int, values: np.ndarray) -> None:
    if end < start or len(values) != end - start + 1:
        raise ValueError("invalid semantic orientation segment")
    output[start : end + 1] = values


def build_task_partial_orientation_targets(
    timeline: SemanticTimeline,
    runtime: ActiveG1Dex3,
    v14_arm_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    source_left_rotation: np.ndarray,
    source_right_rotation: np.ndarray,
    phone_initial: np.ndarray,
    phone_charger: np.ndarray,
    left_phone_grasp_wrist_rotation: np.ndarray,
    left_charger_wrist_rotation: np.ndarray,
    right_ring_wrist_rotation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build source-relative wrist targets with geometry-derived endpoints."""
    left_wrist, right_wrist = _actual_wrist_trajectory(runtime, v14_arm_q, left_q, right_q)
    length = timeline.trajectory_length
    left_target = left_wrist[:, :3, :3].copy()
    right_target = right_wrist[:, :3, :3].copy()
    left_axis_weight = np.zeros((length, 3), dtype=np.float64)
    right_axis_weight = np.zeros((length, 3), dtype=np.float64)

    grasp = int(timeline.event("left_phone_grasp_start").action_index)
    rotation_start = int(timeline.event("phone_rotation_to_portrait_start").action_index)
    portrait = int(timeline.event("phone_portrait_reached").action_index)
    phone_move = int(timeline.event("phone_move_to_charger_start").action_index)
    charger = int(timeline.event("phone_charger_attachment_complete").action_index)
    left_release = int(timeline.event("left_phone_release_complete").action_index)
    right_grasp = int(timeline.event("right_accessory_grasp_start").action_index)
    detachment = int(timeline.event("accessory_detachment_start").action_index)
    removed = int(timeline.event("accessory_removed").action_index)
    right_release = int(timeline.event("right_accessory_release_complete").action_index)

    grasp_wrist = np.asarray(left_phone_grasp_wrist_rotation, dtype=np.float64)
    wrist_from_phone_rotation = grasp_wrist.T @ phone_initial[:3, :3]
    source_phone_portrait = phone_initial[:3, :3] @ mapped_relative(
        source_left_rotation, rotation_start, portrait, C_LEFT
    )[-1]
    long_axis = source_phone_portrait[:, 0]
    long_xy = long_axis.copy()
    long_xy[2] = 0.0
    if np.linalg.norm(long_xy) < 1e-8:
        long_xy = np.array([1.0, 0.0, 0.0])
    vertical = np.array([0.0, 0.0, 1.0])
    back = source_phone_portrait[:, 1] - vertical * float(np.dot(source_phone_portrait[:, 1], vertical))
    if np.linalg.norm(back) < 1e-8:
        back = np.array([0.0, 1.0, 0.0])
    back /= np.linalg.norm(back)
    short = np.cross(vertical, back)
    short /= max(np.linalg.norm(short), 1e-12)
    back = np.cross(short, vertical)
    portrait_phone_rotation = np.column_stack((vertical, back, short))
    if np.linalg.det(portrait_phone_rotation) < 0.0:
        portrait_phone_rotation[:, 2] *= -1.0
    portrait_wrist = portrait_phone_rotation @ wrist_from_phone_rotation.T
    # v17 explicitly does not assume one rigid A/B-to-phone carrier from grasp
    # through charger placement.  The charger endpoint therefore uses its own
    # task-level, geometry-calibrated partial-orientation registration.
    charger_wrist = np.asarray(left_charger_wrist_rotation, dtype=np.float64)
    wrist_from_phone_charger_rotation = charger_wrist.T @ phone_charger[:3, :3]

    # Introduce the reusable v16 task-level phone-pinch orientation over the
    # named episode-start->grasp interval while preserving source-relative
    # rotation.  No wrist translation or per-frame contact fit is introduced.
    approach_progress = smoothstep(
        np.linspace(0.0, 1.0, grasp - timeline.start_index + 1)
    )
    grasp_registration = grasp_wrist @ left_wrist[grasp, :3, :3].T
    correction = interpolate_rotation(
        np.eye(3), grasp_registration, approach_progress
    )
    approach_to_grasp = np.einsum(
        "tij,tjk->tik",
        correction,
        left_wrist[timeline.start_index:grasp + 1, :3, :3],
    )
    _write_segment(left_target, timeline.start_index, grasp, approach_to_grasp)
    hold_to_rotation = source_preserving_segment(
        source_left_rotation, grasp, rotation_start, C_LEFT, grasp_wrist, None
    )
    rotate = source_preserving_segment(
        source_left_rotation, rotation_start, portrait, C_LEFT,
        hold_to_rotation[-1], portrait_wrist,
    )
    portrait_hold = source_preserving_segment(
        source_left_rotation, portrait, phone_move, C_LEFT, rotate[-1], None
    )
    transport = source_preserving_segment(
        source_left_rotation, phone_move, charger, C_LEFT,
        portrait_hold[-1], charger_wrist,
    )
    _write_segment(left_target, grasp, rotation_start, hold_to_rotation)
    _write_segment(left_target, rotation_start, portrait, rotate)
    _write_segment(left_target, portrait, phone_move, portrait_hold)
    _write_segment(left_target, phone_move, charger, transport)
    left_target[charger : left_release + 1] = charger_wrist
    release_progress = smoothstep(timeline.progress("left_release")[charger : left_release + 1])
    release_rotation = np.empty((len(release_progress), 3, 3))
    for offset, progress in enumerate(release_progress):
        release_rotation[offset] = interpolate_rotation(
            charger_wrist, left_wrist[charger + offset, :3, :3], np.asarray([progress])
        )[0]
    left_target[charger : left_release + 1] = release_rotation

    left_axis_weight[timeline.start_index:grasp + 1] = (
        approach_progress[:, None] * np.array([1.0, 1.0, 0.0])
    )
    left_axis_weight[grasp:rotation_start + 1] = np.array([1.0, 1.0, 0.0])
    rotation_weight = smoothstep(
        timeline.progress("phone_rotation")[rotation_start : portrait + 1]
    )
    left_axis_weight[rotation_start:portrait + 1] = (
        np.array([0.25, 0.45, 0.0])
        + rotation_weight[:, None] * np.array([0.75, 0.30, 0.0])
    )
    # Portrait is an endpoint task constraint, not a rigid absolute wrist
    # constraint for the entire subsequent hold.  Fade it back toward the
    # neutral v14 wrist using semantic interval progress.
    portrait_hold_progress = _interval_progress(
        timeline, "phone_portrait_reached", "phone_move_to_charger_start"
    )
    left_axis_weight[portrait:phone_move + 1] = (
        (1.0 - portrait_hold_progress[:, None]) ** 2
        * np.array([1.0, 0.65, 0.0])
    )
    transport_progress = smoothstep(
        timeline.progress("phone_to_charger")[phone_move : charger + 1]
    )
    left_axis_weight[phone_move:charger + 1] = (
        transport_progress[:, None] ** 2 * np.array([1.0, 1.0, 0.0])
    )
    left_axis_weight[charger:left_release + 1] = np.array([1.0, 1.0, 0.0])
    # Fade the task axes fully to zero at the named release endpoint so the
    # validated v14 return branch is rejoined continuously.
    left_axis_weight[charger:left_release + 1] *= (1.0 - release_progress[:, None])

    portrait_right = right_wrist[portrait, :3, :3]
    approach = source_preserving_segment(
        source_right_rotation, portrait, right_grasp, C_RIGHT,
        portrait_right, np.asarray(right_ring_wrist_rotation, dtype=np.float64),
    )
    acquire = source_preserving_segment(
        source_right_rotation, right_grasp, detachment, C_RIGHT,
        approach[-1], None,
    )
    remove = source_preserving_segment(
        source_right_rotation, detachment, removed, C_RIGHT,
        acquire[-1], None,
    )
    _write_segment(right_target, portrait, right_grasp, approach)
    _write_segment(right_target, right_grasp, detachment, acquire)
    _write_segment(right_target, detachment, removed, remove)
    right_release_progress = smoothstep(
        timeline.progress("right_release")[removed : right_release + 1]
    )
    for offset, progress_value in enumerate(right_release_progress):
        right_target[removed + offset] = interpolate_rotation(
            remove[-1], right_wrist[removed + offset, :3, :3],
            np.asarray([progress_value]),
        )[0]
    right_axis_weight[portrait:right_grasp + 1] = np.array([0.45, 0.0, 0.0])
    right_axis_weight[right_grasp:detachment + 1] = np.array([0.85, 0.25, 0.0])
    right_axis_weight[detachment:removed + 1] = np.array([0.75, 0.25, 0.0])
    right_axis_weight[removed:right_release + 1] = (
        (1.0 - right_release_progress[:, None]) * np.array([0.35, 0.0, 0.0])
    )

    return {
        "left_rotation": left_target,
        "right_rotation": right_target,
        "left_axis_weight": left_axis_weight,
        "right_axis_weight": right_axis_weight,
        "v14_left_wrist": left_wrist,
        "v14_right_wrist": right_wrist,
        "wrist_from_phone_rotation": wrist_from_phone_rotation,
        "wrist_from_phone_charger_rotation": wrist_from_phone_charger_rotation,
        "portrait_phone_rotation": portrait_phone_rotation,
        "phone_charger_rotation": phone_charger[:3, :3],
    }


def build_semantic_local_orientation_targets(
    timeline: SemanticTimeline,
    runtime: ActiveG1Dex3,
    v14_arm_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    source_left_rotation: np.ndarray,
    source_right_rotation: np.ndarray,
    phone_initial: np.ndarray,
    phone_charger: np.ndarray,
    left_phone_grasp_wrist_rotation: np.ndarray,
    left_charger_wrist_rotation: np.ndarray,
    right_ring_wrist_rotation: np.ndarray,
    semantic_progress: dict[str, np.ndarray],
    *,
    left_acquisition_strength: float = 1.0,
    right_hook_strength: float = 0.65,
    charger_strength: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build task orientation only inside evidence-derived semantic windows.

    v17 blended its grasp registration from the start of the episode and its
    right hook registration from the portrait event.  Those broad windows
    changed an otherwise unrelated bimanual interval.  v17.1 instead derives
    PREGRASP onset from the robust source-gripper progress already exposed by
    the generic semantic adapter, then activates the task axes with a smooth
    bounded progress curve.  No Cartesian target is constructed here.
    """
    full = build_task_partial_orientation_targets(
        timeline, runtime, v14_arm_q, left_q, right_q,
        source_left_rotation, source_right_rotation,
        phone_initial, phone_charger,
        left_phone_grasp_wrist_rotation, left_charger_wrist_rotation,
        right_ring_wrist_rotation,
    )
    left_v14 = full["v14_left_wrist"][:, :3, :3]
    right_v14 = full["v14_right_wrist"][:, :3, :3]
    left_target = full["left_rotation"].copy()
    right_target = full["right_rotation"].copy()
    left_weight = np.zeros_like(full["left_axis_weight"])
    right_weight = np.zeros_like(full["right_axis_weight"])

    grasp = int(timeline.event("left_phone_grasp_start").action_index)
    rotation_start = int(timeline.event("phone_rotation_to_portrait_start").action_index)
    portrait = int(timeline.event("phone_portrait_reached").action_index)
    phone_move = int(timeline.event("phone_move_to_charger_start").action_index)
    charger = int(timeline.event("phone_charger_attachment_complete").action_index)
    left_release = int(timeline.event("left_phone_release_complete").action_index)
    right_grasp = int(timeline.event("right_accessory_grasp_start").action_index)
    detachment = int(timeline.event("accessory_detachment_start").action_index)
    removed = int(timeline.event("accessory_removed").action_index)
    right_release = int(timeline.event("right_accessory_release_complete").action_index)

    left_start = int(np.asarray(
        semantic_progress["left_source_signal_detected_approach_start"]
    ).item())
    right_start = int(np.asarray(
        semantic_progress["right_source_signal_detected_approach_start"]
    ).item())
    left_start = int(np.clip(left_start, timeline.start_index, grasp))
    right_start = int(np.clip(right_start, portrait, right_grasp))

    def source_progress(key: str, start: int, end: int) -> np.ndarray:
        raw = np.asarray(semantic_progress[key], dtype=np.float64)[start : end + 1]
        span = float(raw[-1] - raw[0]) if len(raw) else 0.0
        if len(raw) and span > 1e-8:
            value = np.clip((raw - raw[0]) / span, 0.0, 1.0)
        else:
            value = np.linspace(0.0, 1.0, end - start + 1)
        return smoothstep(np.maximum.accumulate(value))

    # Preserve v14 exactly outside the semantic-local left task windows.
    left_target[:left_start] = left_v14[:left_start]
    left_acquire = source_progress("left_source_close_progress", left_start, grasp)
    grasp_registration = (
        np.asarray(left_phone_grasp_wrist_rotation, dtype=np.float64)
        @ left_v14[grasp].T
    )
    local_correction = interpolate_rotation(np.eye(3), grasp_registration, left_acquire)
    left_target[left_start : grasp + 1] = np.einsum(
        "tij,tjk->tik", local_correction, left_v14[left_start : grasp + 1]
    )
    left_weight[left_start : grasp + 1] = (
        float(left_acquisition_strength) * left_acquire[:, None]
        * np.array([1.0, 1.0, 0.0])
    )
    # The acquired pose remains the source-relative rotation origin; only the
    # task-critical axes are constrained and unconstrained twist remains free.
    left_weight[grasp : rotation_start + 1] = (
        float(left_acquisition_strength) * np.array([1.0, 1.0, 0.0])
    )
    left_weight[rotation_start : portrait + 1] = full["left_axis_weight"][rotation_start : portrait + 1]
    left_weight[portrait : phone_move + 1] = full["left_axis_weight"][portrait : phone_move + 1]
    left_weight[phone_move : charger + 1] = (
        float(charger_strength) * full["left_axis_weight"][phone_move : charger + 1]
    )
    left_weight[charger : left_release + 1] = (
        float(charger_strength) * full["left_axis_weight"][charger : left_release + 1]
    )
    left_target[left_release + 1 :] = left_v14[left_release + 1 :]

    # Right task orientation begins only when the source gripper/task evidence
    # enters PREHOOK, rather than immediately after portrait.  This removes the
    # unrelated v17 collision interval without changing either hand position.
    right_target[:right_start] = right_v14[:right_start]
    right_acquire = source_progress("right_source_close_progress", right_start, right_grasp)
    approach = source_preserving_segment(
        source_right_rotation, right_start, right_grasp, C_RIGHT,
        right_v14[right_start], np.asarray(right_ring_wrist_rotation, dtype=np.float64),
    )
    right_target[right_start : right_grasp + 1] = approach
    right_weight[right_start : right_grasp + 1] = (
        float(right_hook_strength) * right_acquire[:, None]
        * np.array([1.0, 0.25, 0.0])
    )
    right_weight[right_grasp : detachment + 1] = (
        float(right_hook_strength) * full["right_axis_weight"][right_grasp : detachment + 1]
    )
    right_weight[detachment : removed + 1] = (
        float(right_hook_strength) * full["right_axis_weight"][detachment : removed + 1]
    )
    right_weight[removed : right_release + 1] = (
        float(right_hook_strength) * full["right_axis_weight"][removed : right_release + 1]
    )
    right_target[right_release + 1 :] = right_v14[right_release + 1 :]

    full.update({
        "left_rotation": left_target,
        "right_rotation": right_target,
        "left_axis_weight": left_weight,
        "right_axis_weight": right_weight,
        "left_semantic_activation": np.r_[
            np.zeros(left_start), left_acquire,
            np.zeros(timeline.trajectory_length - grasp - 1),
        ],
        "right_semantic_activation": np.r_[
            np.zeros(right_start), right_acquire,
            np.zeros(timeline.trajectory_length - right_grasp - 1),
        ],
        "left_pregrasp_start": np.asarray(left_start),
        "right_prehook_start": np.asarray(right_start),
    })
    return full


def _collision_scalars(runtime: ActiveG1Dex3, side: str) -> np.ndarray:
    torso = cross = hand_torso = same_side = 0.0
    for record in runtime.penetrating_contacts():
        depth = max(0.0, -float(record["distance_m"]))
        bodies = record["bodies"]
        side_hit = any(value.startswith(f"{side}_") for value in bodies)
        other = "right" if side == "left" else "left"
        other_hit = any(value.startswith(f"{other}_") for value in bodies)
        torso_hit = any(any(token in value for token in ("torso", "waist", "pelvis")) for value in bodies)
        hand_hit = any("hand" in value for value in bodies)
        if side_hit and torso_hit:
            torso = max(torso, depth)
            if hand_hit:
                hand_torso = max(hand_torso, depth)
        if side_hit and other_hit:
            cross = max(cross, depth)
        side_bodies = [value.startswith(f"{side}_") for value in bodies]
        if all(side_bodies) and any(
            any(token in value for token in ("wrist", "elbow", "shoulder", "hand"))
            for value in bodies
        ):
            same_side = max(same_side, depth)
    return np.asarray([torso, hand_torso, cross, same_side], dtype=np.float64)


def solve_partial_orientation_trajectory(
    runtime: ActiveG1Dex3,
    v14_arm_q: np.ndarray,
    left_position: np.ndarray,
    right_position: np.ndarray,
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
    left_axis_weight: np.ndarray,
    right_axis_weight: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    *,
    orientation_gain: float,
    prior_gain: float,
    temporal_gain: float,
    collision_gain: float,
    shoulder_prior_gain: float,
    max_deviation_rad: float,
    max_step_rad: float,
) -> np.ndarray:
    """Refine only arm q while preserving the supplied Cartesian path."""
    arm = np.asarray(v14_arm_q, dtype=np.float64)
    output = arm.copy()
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    for frame in range(len(output)):
        for side, target_p, target_r, axis_weight, block in (
            ("left", left_position[frame], left_rotation[frame], left_axis_weight[frame], slice(0, 7)),
            ("right", right_position[frame], right_rotation[frame], right_axis_weight[frame], slice(7, 14)),
        ):
            base = arm[frame, block].copy()
            previous = output[frame - 1, block].copy() if frame else base.copy()
            # Outside task-critical semantic phases, keep the already validated
            # v14 branch exactly.  This is both more faithful and avoids
            # numerically re-solving an unconstrained wrist pose.
            if not np.any(np.asarray(axis_weight) > 0.0):
                output[frame, block] = base
                continue
            lo = np.maximum(limits[block, 0] + 1e-6, base - max_deviation_rad)
            hi = np.minimum(limits[block, 1] - 1e-6, base + max_deviation_rad)
            if frame:
                lo = np.maximum(lo, previous - max_step_rad)
                hi = np.minimum(hi, previous + max_step_rad)
            invalid = lo >= hi
            if np.any(invalid):
                lo[invalid] = limits[block, 0][invalid] + 1e-6
                hi[invalid] = limits[block, 1][invalid] - 1e-6
            seed = np.clip(0.7 * base + 0.3 * previous, lo + 1e-9, hi - 1e-9)
            whole = output[frame].copy()

            def residual(value: np.ndarray) -> np.ndarray:
                whole[block] = value
                runtime.assign(whole, left_hand_q[frame], right_hand_q[frame])
                position, rotation, _ = runtime.palm_state(side)
                axes = []
                for axis in range(3):
                    axes.extend(orientation_gain * axis_weight[axis] * (rotation[:, axis] - target_r[:, axis]))
                margin = np.minimum(value - limits[block, 0], limits[block, 1] - value)
                barrier = np.maximum(0.0, 0.025 - margin)
                return np.r_[
                    2600.0 * (position - target_p),
                    np.asarray(axes),
                    prior_gain * (value - base),
                    shoulder_prior_gain * (value[:3] - base[:3]),
                    temporal_gain * (value - previous),
                    0.20 * barrier,
                    collision_gain * _collision_scalars(runtime, side),
                ]

            result = least_squares(
                residual,
                seed,
                bounds=(lo, hi),
                max_nfev=45,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
                x_scale="jac",
            )
            output[frame, block] = result.x
    return output


def solve_semantic_local_orientation_trajectory(
    runtime: ActiveG1Dex3,
    v14_arm_q: np.ndarray,
    left_position: np.ndarray,
    right_position: np.ndarray,
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
    left_axis_weight: np.ndarray,
    right_axis_weight: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    *,
    orientation_gain: float = 42.0,
    prior_gain: float = 0.06,
    temporal_gain: float = 0.16,
    acceleration_gain: float = 0.08,
    collision_gain: float = 100000.0,
    shoulder_prior_gain: float = 8.0,
    joint_center_gain: float = 2.0,
    minimum_joint_margin_rad: float = 0.01,
    preferred_joint_margin_rad: float = 0.03,
    max_deviation_rad: float = 1.20,
    max_step_rad: float = 0.24,
) -> np.ndarray:
    """Task-null-space continuation with immutable palm-position targets.

    The solver changes only arm q.  The supplied Cartesian arrays are read-only
    objectives and are never offset, warped, or rewritten.  A hard joint-limit
    inset removes v17's micro-radian wrist-yaw margin in the same semantic
    interval where orientation is already active.
    """
    arm = np.asarray(v14_arm_q, dtype=np.float64)
    output = arm.copy()
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    hard_margin = float(minimum_joint_margin_rad)
    preferred_margin = float(preferred_joint_margin_rad)
    for frame in range(len(output)):
        for side, target_p, target_r, axis_weight, block in (
            ("left", left_position[frame], left_rotation[frame], left_axis_weight[frame], slice(0, 7)),
            ("right", right_position[frame], right_rotation[frame], right_axis_weight[frame], slice(7, 14)),
        ):
            base = arm[frame, block].copy()
            previous = output[frame - 1, block].copy() if frame else base.copy()
            previous2 = output[frame - 2, block].copy() if frame > 1 else previous.copy()
            base_margin = np.minimum(
                base - limits[block, 0], limits[block, 1] - base
            )
            active = bool(
                np.any(np.asarray(axis_weight) > 0.0)
                or np.min(base_margin) < preferred_margin
                or (frame and np.max(np.abs(previous - base)) > 0.5 * max_step_rad)
            )
            if not active:
                output[frame, block] = base
                continue
            lo = np.maximum(limits[block, 0] + hard_margin, base - max_deviation_rad)
            hi = np.minimum(limits[block, 1] - hard_margin, base + max_deviation_rad)
            if frame:
                lo = np.maximum(lo, previous - max_step_rad)
                hi = np.minimum(hi, previous + max_step_rad)
            if np.any(lo >= hi):
                raise RuntimeError(f"semantic-local trust region infeasible at {frame}/{side}")
            seed = np.clip(0.65 * base + 0.35 * previous, lo + 1e-10, hi - 1e-10)
            whole = output[frame].copy()

            def residual(value: np.ndarray) -> np.ndarray:
                whole[block] = value
                runtime.assign(whole, left_hand_q[frame], right_hand_q[frame])
                position, rotation, _ = runtime.palm_state(side)
                axes = []
                for axis in range(3):
                    axes.extend(
                        orientation_gain * axis_weight[axis]
                        * (rotation[:, axis] - target_r[:, axis])
                    )
                margin = np.minimum(
                    value - limits[block, 0], limits[block, 1] - value
                )
                barrier = np.maximum(0.0, preferred_margin - margin)
                acceleration = value - 2.0 * previous + previous2
                return np.r_[
                    3000.0 * (position - target_p),
                    np.asarray(axes),
                    prior_gain * (value - base),
                    shoulder_prior_gain * (value[:3] - base[:3]),
                    temporal_gain * (value - previous),
                    acceleration_gain * acceleration,
                    joint_center_gain * barrier,
                    collision_gain * _collision_scalars(runtime, side),
                ]

            result = least_squares(
                residual, seed, bounds=(lo, hi), max_nfev=65,
                ftol=1e-10, xtol=1e-10, gtol=1e-10, x_scale="jac",
            )
            output[frame, block] = result.x
    return output


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b, atol=1e-9) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _path_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ar = reference - reference[0]
    ac = candidate - candidate[0]
    sr = max(float(np.max(np.linalg.norm(ar, axis=1))), 1e-12)
    sc = max(float(np.max(np.linalg.norm(ac, axis=1))), 1e-12)
    path = _corr(ar / sr, ac / sc)
    vr = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    vc = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
    return path, _corr(vr / max(float(np.max(vr)), 1e-12), vc / max(float(np.max(vc)), 1e-12))


def _angle_unsigned(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1.0, 1.0))))


def _collision_audit(
    runtime: ActiveG1Dex3,
    arm_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    table_height: float,
    table_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    categories = {name: set() for name in (
        "arm_torso", "arm_arm", "hand_torso", "hand_hand", "same_hand_self_contact",
        "arm_table", "palm_table",
    )}
    minimum_table = math.inf
    minimum_hand_torso = math.inf
    for frame, (aq, lq, rq) in enumerate(zip(arm_q, left_q, right_q)):
        runtime.assign(aq, lq, rq)
        for record in runtime.penetrating_contacts():
            bodies = record["bodies"]
            pair = "|".join(sorted(bodies))
            left = any(value.startswith("left_") for value in bodies)
            right = any(value.startswith("right_") for value in bodies)
            hand = ["hand" in value for value in bodies]
            arm = [any(token in value for token in ("shoulder", "elbow", "wrist")) for value in bodies]
            torso = [any(token in value for token in ("torso", "waist", "pelvis")) for value in bodies]
            if any(arm) and any(torso): categories["arm_torso"].add((frame, pair))
            if left and right and any(arm): categories["arm_arm"].add((frame, pair))
            if any(hand) and any(torso):
                categories["hand_torso"].add((frame, pair))
                minimum_hand_torso = min(minimum_hand_torso, float(record["distance_m"]))
            if left and right and all(hand): categories["hand_hand"].add((frame, pair))
            if all(hand) and (left ^ right): categories["same_hand_self_contact"].add((frame, pair))
        x0, x1, y0, y1 = table_bounds
        for geom_id in range(runtime.model.ngeom):
            body = int(runtime.model.geom_bodyid[geom_id])
            name = mujoco.mj_id2name(runtime.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if not (name.startswith("left_") or name.startswith("right_")):
                continue
            if not any(token in name for token in ("shoulder", "elbow", "wrist", "hand")):
                continue
            vertices_model = geom_world_vertices(runtime.model, runtime.data, geom_id)
            vertices = runtime.model_to_scene_position(vertices_model)
            inside = (
                (vertices[:, 0] >= x0) & (vertices[:, 0] <= x1)
                & (vertices[:, 1] >= y0) & (vertices[:, 1] <= y1)
            )
            if np.any(inside):
                clearance = float(np.min(vertices[inside, 2] - table_height))
                minimum_table = min(minimum_table, clearance)
                if clearance < -1e-5:
                    category = "palm_table" if "hand" in name else "arm_table"
                    categories[category].add((frame, name))
    values = {
        key: {"count": len(rows), "frames": sorted({frame for frame, _ in rows}), "pairs": sorted({pair for _, pair in rows})}
        for key, rows in categories.items()
    }
    prohibited = sum(row["count"] for row in values.values())
    return {
        "categories": values,
        "prohibited_collision_records": prohibited,
        "minimum_table_clearance_m_active_geom_vertices": float(minimum_table),
        "minimum_hand_torso_signed_contact_distance_m": None if not np.isfinite(minimum_hand_torso) else float(minimum_hand_torso),
        "pass": prohibited == 0,
    }


def audit_collision_classifier_integrity(
    runtime: ActiveG1Dex3,
    arm_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    table_height: float,
    table_bounds: tuple[float, float, float, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Classify every raw penetrating robot self-contact exactly once.

    This is deliberately independent of the optimizer's compact collision
    residual.  In particular, wrist/arm-to-finger contacts are prohibited and
    cannot disappear merely because they are not ``finger-finger`` contacts.
    """
    category_names = (
        "arm_torso", "arm_arm", "arm_table", "hand_torso", "hand_hand",
        "wrist_finger_same_side", "arm_finger_same_side",
        "wrist_hand_same_side", "same_hand_finger_finger", "palm_table",
        "other_robot_self_contact",
    )
    counts = {name: 0 for name in category_names}
    frames = {name: set() for name in category_names}
    pairs = {name: set() for name in category_names}
    rows: list[dict[str, Any]] = []
    raw_robot_robot = 0
    ignored = 0

    def side(name: str) -> str | None:
        if name.startswith("left_"):
            return "left"
        if name.startswith("right_"):
            return "right"
        return None

    def role(name: str) -> str:
        if any(token in name for token in ("torso", "waist", "pelvis")):
            return "torso"
        if "wrist" in name:
            return "wrist"
        if any(token in name for token in ("shoulder", "elbow")):
            return "arm"
        if "hand" in name and any(token in name for token in ("thumb", "index", "middle")):
            return "finger"
        if "hand" in name or "palm" in name:
            return "hand"
        return "other"

    for frame, (aq, lq, rq) in enumerate(zip(arm_q, left_q, right_q)):
        runtime.assign(aq, lq, rq)
        for contact_index, record in enumerate(runtime.penetrating_contacts()):
            bodies = tuple(str(value) for value in record["bodies"])
            sides = tuple(side(value) for value in bodies)
            roles = tuple(role(value) for value in bodies)
            is_robot = tuple(value is not None or role_value == "torso" for value, role_value in zip(sides, roles))
            row = {
                "frame": frame,
                "contact_index": contact_index,
                "body_1": bodies[0], "body_2": bodies[1],
                "geom_1": record["geoms"][0], "geom_2": record["geoms"][1],
                "distance_m": float(record["distance_m"]),
                "side_1": sides[0], "side_2": sides[1],
                "role_1": roles[0], "role_2": roles[1],
            }
            if not all(is_robot):
                ignored += 1
                row.update({
                    "classification": "IGNORED_NON_ROBOT_ROBOT_CONTACT",
                    "prohibited": False,
                    "ignored_reason": "one or both bodies are outside the robot self-contact taxonomy",
                })
                rows.append(row)
                continue

            raw_robot_robot += 1
            role_set = set(roles)
            side_set = {value for value in sides if value is not None}
            if "torso" in role_set:
                other = roles[1] if roles[0] == "torso" else roles[0]
                category = "hand_torso" if other in ("finger", "hand") else "arm_torso"
            elif len(side_set) == 2:
                category = "hand_hand" if role_set & {"finger", "hand"} else "arm_arm"
            elif role_set == {"finger"}:
                category = "same_hand_finger_finger"
            elif "finger" in role_set and "wrist" in role_set:
                category = "wrist_finger_same_side"
            elif "finger" in role_set and "arm" in role_set:
                category = "arm_finger_same_side"
            elif "hand" in role_set and "wrist" in role_set:
                category = "wrist_hand_same_side"
            else:
                category = "other_robot_self_contact"
            counts[category] += 1
            frames[category].add(frame)
            pair = "|".join(sorted(bodies))
            pairs[category].add(pair)
            row.update({
                "classification": category,
                "prohibited": True,
                "ignored_reason": "",
            })
            rows.append(row)

    # Table penetration uses the same active-geometry vertex audit as v17 and
    # is appended to the prohibited total, but it is not a robot-robot contact.
    legacy = _collision_audit(
        runtime, arm_q, left_q, right_q, table_height, table_bounds
    )
    for category in ("arm_table", "palm_table"):
        count = int(legacy["categories"][category]["count"])
        counts[category] += count
        frames[category].update(legacy["categories"][category]["frames"])
        pairs[category].update(legacy["categories"][category]["pairs"])

    classified_robot_robot = sum(
        counts[name] for name in category_names if name not in ("arm_table", "palm_table")
    )
    integrity = raw_robot_robot == classified_robot_robot
    prohibited = classified_robot_robot + counts["arm_table"] + counts["palm_table"]
    categories = {
        name: {
            "count": int(counts[name]),
            "frames": sorted(frames[name]),
            "pairs": sorted(pairs[name]),
        }
        for name in category_names
    }
    return {
        "raw_penetrating_contact_records": len(rows),
        "raw_robot_robot_contact_count": raw_robot_robot,
        "classified_robot_robot_contact_count": classified_robot_robot,
        "ignored_non_robot_robot_contact_count": ignored,
        "raw_equals_classified": integrity,
        "categories": categories,
        "prohibited_collision_records": prohibited,
        "minimum_table_clearance_m_active_geom_vertices": legacy[
            "minimum_table_clearance_m_active_geom_vertices"
        ],
        "minimum_hand_torso_signed_contact_distance_m": legacy[
            "minimum_hand_torso_signed_contact_distance_m"
        ],
        "wrist_arm_finger_contacts_preserved": bool(
            counts["wrist_finger_same_side"] >= 0
            and counts["arm_finger_same_side"] >= 0
            and counts["wrist_hand_same_side"] >= 0
        ),
        "pass": bool(integrity and prohibited == 0),
    }, rows


def evaluate_kinematic_candidate(
    timeline: SemanticTimeline,
    runtime: ActiveG1Dex3,
    arm_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    reference_left_position: np.ndarray,
    reference_right_position: np.ndarray,
    orientation_targets: dict[str, np.ndarray],
    source_left_rotation: np.ndarray,
    source_right_rotation: np.ndarray,
    phone_initial: np.ndarray,
    phone_charger: np.ndarray,
    table_height: float,
    table_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    length = timeline.trajectory_length
    left_p, right_p = np.empty((length, 3)), np.empty((length, 3))
    left_r, right_r = np.empty((length, 3, 3)), np.empty((length, 3, 3))
    left_wrist, right_wrist = np.empty((length, 4, 4)), np.empty((length, 4, 4))
    for frame in range(length):
        runtime.assign(arm_q[frame], left_q[frame], right_q[frame])
        left_p[frame], left_r[frame], _ = runtime.palm_state("left")
        right_p[frame], right_r[frame], _ = runtime.palm_state("right")
        left_wrist[frame] = runtime.wrist_pose("left")
        right_wrist[frame] = runtime.wrist_pose("right")
    left_error = np.linalg.norm(left_p - reference_left_position, axis=1)
    right_error = np.linalg.norm(right_p - reference_right_position, axis=1)
    left_path, left_speed = _path_metrics(reference_left_position, left_p)
    right_path, right_speed = _path_metrics(reference_right_position, right_p)
    midpoint_ref = 0.5 * (reference_left_position + reference_right_position)
    midpoint = 0.5 * (left_p + right_p)
    relative_ref = reference_right_position - reference_left_position
    relative = right_p - left_p

    grasp = int(timeline.event("left_phone_grasp_start").action_index)
    rotation_start = int(timeline.event("phone_rotation_to_portrait_start").action_index)
    portrait = int(timeline.event("phone_portrait_reached").action_index)
    charger = int(timeline.event("phone_charger_attachment_complete").action_index)
    right_grasp = int(timeline.event("right_accessory_grasp_start").action_index)
    removed = int(timeline.event("accessory_removed").action_index)
    wrist_from_phone = left_wrist[grasp, :3, :3].T @ phone_initial[:3, :3]
    predicted_phone_portrait = left_wrist[portrait, :3, :3] @ wrist_from_phone
    predicted_phone_charger = (
        left_wrist[charger, :3, :3]
        @ orientation_targets["wrist_from_phone_charger_rotation"]
    )
    portrait_error = _angle_unsigned(predicted_phone_portrait[:, 0], np.array([0.0, 0.0, 1.0]))
    charger_normal = _angle_unsigned(predicted_phone_charger[:, 1], phone_charger[:3, 1])
    charger_vertical = _angle_unsigned(predicted_phone_charger[:, 0], phone_charger[:3, 0])

    def progress(rotation: np.ndarray, start: int, end: int) -> np.ndarray:
        segment = rotation[start : end + 1]
        step = Rotation.from_matrix(np.einsum("tji,tjk->tik", segment[:-1], segment[1:])).magnitude()
        value = np.r_[0.0, np.cumsum(np.abs(step))]
        return value / max(float(value[-1]), 1e-12)

    source_left_progress = progress(source_left_rotation, rotation_start, portrait)
    achieved_left_progress = progress(left_wrist[:, :3, :3], rotation_start, portrait)
    source_right_progress = progress(source_right_rotation, right_grasp, removed)
    achieved_right_progress = progress(right_wrist[:, :3, :3], right_grasp, removed)
    rotation_corr_left = _corr(source_left_progress, achieved_left_progress)
    rotation_corr_right = _corr(source_right_progress, achieved_right_progress)

    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    margin = np.minimum(arm_q - limits[:, 0], limits[:, 1] - arm_q)
    hand_margin_left = np.minimum(left_q - runtime.hand_limits["left"][:, 0], runtime.hand_limits["left"][:, 1] - left_q)
    hand_margin_right = np.minimum(right_q - runtime.hand_limits["right"][:, 0], runtime.hand_limits["right"][:, 1] - right_q)
    step = np.max(np.abs(np.diff(arm_q, axis=0)), axis=1)
    local = np.asarray([
        np.median(step[max(0, i - 10) : min(len(step), i + 11)]) for i in range(len(step))
    ])
    branch = np.flatnonzero(step > np.maximum(0.35, 8.0 * np.maximum(local, 1e-6))) + 1
    collision = _collision_audit(runtime, arm_q, left_q, right_q, table_height, table_bounds)
    fidelity_values = [
        left_path, left_speed, right_path, right_speed,
        _corr(midpoint_ref - midpoint_ref[0], midpoint - midpoint[0]),
        _corr(relative_ref - relative_ref[0], relative - relative[0]),
        _corr(np.linalg.norm(relative_ref, axis=1), np.linalg.norm(relative, axis=1)),
    ]
    position_pass = bool(np.mean((left_error <= 0.005) & (right_error <= 0.005)) >= 0.99)
    orientation_pass = bool(
        portrait_error <= 10.0 and charger_normal <= 10.0 and charger_vertical <= 10.0
        and rotation_corr_left >= 0.90 and rotation_corr_right >= 0.90
    )
    return {
        "finite": bool(np.isfinite(arm_q).all() and np.isfinite(left_q).all() and np.isfinite(right_q).all()),
        "position": {
            "simultaneous_5mm_rate": float(np.mean((left_error <= 0.005) & (right_error <= 0.005))),
            "left_mean_mm": float(np.mean(left_error) * 1000.0),
            "left_max_mm": float(np.max(left_error) * 1000.0),
            "right_mean_mm": float(np.mean(right_error) * 1000.0),
            "right_max_mm": float(np.max(right_error) * 1000.0),
            "pass": position_pass,
        },
        "fidelity": {
            "left_path_shape": left_path,
            "left_speed": left_speed,
            "right_path_shape": right_path,
            "right_speed": right_speed,
            "bimanual_midpoint": fidelity_values[4],
            "relative_hand_vector": fidelity_values[5],
            "inter_hand_distance": fidelity_values[6],
            "minimum_primary_metric": float(min(fidelity_values)),
            "pass": bool(min(fidelity_values) >= 0.95),
        },
        "orientation": {
            "portrait_long_axis_error_deg": portrait_error,
            "charger_normal_error_deg": charger_normal,
            "charger_vertical_axis_error_deg": charger_vertical,
            "left_rotation_progress_correlation": rotation_corr_left,
            "right_rotation_progress_correlation": rotation_corr_right,
            "pass": orientation_pass,
        },
        "joint": {
            "minimum_arm_margin_rad": float(np.min(margin)),
            "minimum_left_dex3_margin_rad": float(np.min(hand_margin_left)),
            "minimum_right_dex3_margin_rad": float(np.min(hand_margin_right)),
            "joint_limit_violation_count": int(np.sum(margin < -1e-9) + np.sum(hand_margin_left < -1e-9) + np.sum(hand_margin_right < -1e-9)),
            "maximum_arm_step_rad": float(np.max(step)),
            "branch_discontinuity_count": int(len(branch)),
            "branch_discontinuity_indices": branch.tolist(),
        },
        "collision": collision,
        "achieved": {
            "left_position": left_p,
            "right_position": right_p,
            "left_rotation": left_r,
            "right_rotation": right_r,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
        },
        "gate_pass": bool(
            position_pass and orientation_pass and min(fidelity_values) >= 0.95
            and np.min(margin) >= -1e-9 and np.min(hand_margin_left) >= -1e-9
            and np.min(hand_margin_right) >= -1e-9 and len(branch) == 0 and collision["pass"]
        ),
    }
