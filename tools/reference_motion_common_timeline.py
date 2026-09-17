#!/usr/bin/env python3
"""Method-blind source event clock for reference-motion supervision.

This module deliberately consumes only authoritative source signals: ALOHA
gripper commands, measured ALOHA FK, timestamps, and the one pooled event
detector configuration.  It has no representation-mode, policy, physics, or
task-outcome input.  Both WRIST and INTERACTION references must consume the
same returned timeline and hand-semantic arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


EVENT_NAMES = (
    "APPROACH_START",
    "LEFT_CLOSE_BEGIN",
    "LEFT_CLOSE_COMPLETE",
    "LEFT_GRASP_CONFIRMED_SOURCE",
    "LEFT_LIFT_BEGIN",
    "LEFT_TRANSPORT",
    "RIGHT_APPROACH_BEGIN",
    "RIGHT_CLOSE_BEGIN",
    "RIGHT_ACQUIRE_SOURCE",
    "DUAL_CONTACT_SOURCE",
    "LEFT_RELEASE_BEGIN",
    "RIGHT_OWNED_SOURCE",
    "RIGHT_TRANSPORT_BEGIN",
    "FINAL_RELEASE_BEGIN",
    "TASK_END",
)


@dataclass(frozen=True)
class SourceEventTimeline:
    frames: dict[str, int]
    times_sec: dict[str, float]
    phase_label: np.ndarray
    phase_time: np.ndarray
    left_hand_semantic: np.ndarray
    right_hand_semantic: np.ndarray
    source_order: tuple[str, ...]
    order_violations: tuple[str, ...]
    evidence: dict[str, Any]


def _transitions(
    signal: np.ndarray,
    *,
    close_threshold: float,
    open_threshold: float,
    smoothing_window: int,
    debounce: int,
    lag: int,
) -> list[dict[str, Any]]:
    """Reproduce the frozen pooled hysteretic detector without method input."""
    smoothed = median_filter(
        np.asarray(signal, dtype=np.float64),
        size=int(smoothing_window),
        mode="nearest",
    )
    midpoint = 0.5 * (float(close_threshold) + float(open_threshold))
    state = "CLOSED" if smoothed[0] <= midpoint else "OPEN"
    rows: list[dict[str, Any]] = []
    frame = 1
    while frame < len(smoothed):
        if state == "CLOSED":
            changed = (
                frame + debounce <= len(smoothed)
                and np.all(smoothed[frame : frame + debounce] >= open_threshold)
            )
            target = "OPEN"
        else:
            changed = (
                frame + debounce <= len(smoothed)
                and np.all(smoothed[frame : frame + debounce] <= close_threshold)
            )
            target = "CLOSED"
        if changed:
            rows.append(
                {
                    "from": state,
                    "to": target,
                    "command_onset_frame": int(frame),
                    "command_confirmed_frame": int(frame + debounce - 1),
                    "aligned_onset_frame": int(min(len(smoothed) - 1, frame + lag)),
                    "aligned_confirmed_frame": int(
                        min(len(smoothed) - 1, frame + debounce - 1 + lag)
                    ),
                }
            )
            state = target
            frame += debounce
        else:
            frame += 1
    return rows


def detect_coarse_events(
    action: np.ndarray,
    detector_config: Mapping[str, Any],
) -> dict[str, int]:
    """Return the original pooled detector's common manipulation boundaries."""
    action = np.asarray(action, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 14 or not np.isfinite(action).all():
        raise ValueError("source action must be finite [T,14]")
    cfg = detector_config
    side_rows: dict[str, list[dict[str, Any]]] = {}
    for side, channel in (("left", 6), ("right", 13)):
        side_cfg = cfg["sides"][side]
        side_rows[side] = _transitions(
            action[:, channel],
            close_threshold=float(side_cfg["close_enter_threshold_m"]),
            open_threshold=float(side_cfg["open_enter_threshold_m"]),
            smoothing_window=int(cfg["smoothing_window_frames"]),
            debounce=int(cfg["debounce_frames"]),
            lag=int(cfg["pooled_command_to_measured_state_lag_frames"]),
        )

    def get(side: str, index: int, source: str, target: str, field: str) -> int:
        rows = side_rows[side]
        if index >= len(rows):
            raise RuntimeError(f"{side} has only {len(rows)} gripper transitions")
        row = rows[index]
        if row["from"] != source or row["to"] != target:
            raise RuntimeError(f"unexpected {side} transition {index}: {row}")
        return int(row[field])

    hold = int(cfg["stable_hold_frames"])
    left_close = get("left", 1, "OPEN", "CLOSED", "aligned_onset_frame")
    left_grasp = get("left", 1, "OPEN", "CLOSED", "aligned_confirmed_frame")
    right_close = get("right", 1, "OPEN", "CLOSED", "aligned_onset_frame")
    right_grasp = get("right", 1, "OPEN", "CLOSED", "aligned_confirmed_frame")
    return {
        "LEFT_CLOSE_ONSET": left_close,
        "LEFT_GRASP": left_grasp,
        "LEFT_STABLE_HOLD": min(len(action) - 1, left_grasp + hold),
        "RIGHT_CLOSE_ONSET": right_close,
        "RIGHT_GRASP": right_grasp,
        "RIGHT_STABLE_HOLD": min(len(action) - 1, right_grasp + hold),
        "LEFT_RELEASE": get("left", 2, "CLOSED", "OPEN", "aligned_onset_frame"),
        "RIGHT_FINAL_RELEASE": get(
            "right", 2, "CLOSED", "OPEN", "aligned_onset_frame"
        ),
        "_left_command_close_onset": get(
            "left", 1, "OPEN", "CLOSED", "command_onset_frame"
        ),
        "_left_command_close_complete": get(
            "left", 1, "OPEN", "CLOSED", "command_confirmed_frame"
        ),
        "_right_command_close_onset": get(
            "right", 1, "OPEN", "CLOSED", "command_onset_frame"
        ),
        "_right_command_close_complete": get(
            "right", 1, "OPEN", "CLOSED", "command_confirmed_frame"
        ),
    }


def _smooth_position(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = min(11, len(values) if len(values) % 2 else len(values) - 1)
    if window < 5:
        return values.copy()
    return savgol_filter(values, window, 2, axis=0, mode="interp")


def _first_sustained_motion(
    position: np.ndarray,
    start: int,
    stop: int,
    fps: float,
    *,
    vertical: bool,
) -> int:
    """Detect source motion onset using a fixed, source-only multi-frame rule."""
    smooth = _smooth_position(position)
    velocity = np.gradient(smooth, 1.0 / fps, axis=0)
    start = max(0, int(start))
    stop = min(len(smooth) - 1, max(start, int(stop)))
    horizon = max(4, int(round(0.20 * fps)))
    for frame in range(start, max(start + 1, stop - horizon + 1)):
        end = min(len(smooth) - 1, frame + horizon)
        delta = smooth[end] - smooth[frame]
        segment = velocity[frame : end + 1]
        if vertical:
            leading = segment[: min(4, len(segment)), 2]
            passed = (
                float(delta[2]) >= 0.005
                and float(np.median(segment[:, 2])) >= 0.02
                and float(np.mean(segment[:, 2] > 0.0)) >= 0.75
                and int(np.count_nonzero(leading >= 0.02)) >= min(3, len(leading))
            )
        else:
            planar = np.linalg.norm(segment[:, :2], axis=1)
            passed = (
                float(np.linalg.norm(delta)) >= 0.006
                and float(np.median(planar)) >= 0.02
            )
        if passed:
            return int(frame)
    return int(start)


def build_common_timeline(
    *,
    action: np.ndarray,
    timestamps: np.ndarray,
    left_tcp_position_world: np.ndarray,
    right_tcp_position_world: np.ndarray,
    detector_config: Mapping[str, Any],
    coarse_events: Mapping[str, int] | None = None,
) -> SourceEventTimeline:
    """Build one phase-aware timeline, shared byte-for-byte by A and B."""
    action = np.asarray(action, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    left = np.asarray(left_tcp_position_world, dtype=np.float64)
    right = np.asarray(right_tcp_position_world, dtype=np.float64)
    if not (len(action) == len(timestamps) == len(left) == len(right)):
        raise ValueError("source arrays are not frame aligned")
    fps = float(1.0 / np.median(np.diff(timestamps)))
    coarse = (
        {key: int(value) for key, value in coarse_events.items()}
        if coarse_events is not None
        else detect_coarse_events(action, detector_config)
    )
    left_close = int(coarse["LEFT_CLOSE_ONSET"])
    left_complete = int(coarse["LEFT_GRASP"])
    right_close = int(coarse["RIGHT_CLOSE_ONSET"])
    right_acquire = int(coarse["RIGHT_GRASP"])
    dual = int(coarse.get("RIGHT_STABLE_HOLD", right_acquire))
    left_release = int(coarse["LEFT_RELEASE"])
    final_release = int(coarse["RIGHT_FINAL_RELEASE"])

    left_lift = _first_sustained_motion(
        left, left_close, max(left_close + 1, right_close), fps, vertical=True
    )
    # In several authoritative demonstrations the doll starts a stable lift
    # during the final few gripper-closing frames.  That table-free motion is
    # ownership evidence and must not be delayed to impose a synthetic order.
    left_grasp = min(left_complete, left_lift)
    left_transport = _first_sustained_motion(
        left,
        min(left_lift + max(1, int(round(0.20 * fps))), right_close),
        right_close,
        fps,
        vertical=False,
    )
    interhand = np.linalg.norm(right - left, axis=1)
    approach_search_start = max(left_transport, left_grasp)
    approach_search_stop = max(approach_search_start + 1, right_close)
    right_speed = np.linalg.norm(np.gradient(_smooth_position(right), 1.0 / fps, axis=0), axis=1)
    candidates = np.flatnonzero(
        (np.arange(len(right)) >= approach_search_start)
        & (np.arange(len(right)) < approach_search_stop)
        & (right_speed >= 0.025)
    )
    right_approach = int(candidates[0]) if len(candidates) else approach_search_start
    right_transport = _first_sustained_motion(
        right,
        min(left_release + 1, final_release),
        final_release,
        fps,
        vertical=False,
    )
    approach_start = max(0, left_close - int(round(1.0 * fps)))
    frames = {
        "APPROACH_START": approach_start,
        "LEFT_CLOSE_BEGIN": left_close,
        "LEFT_CLOSE_COMPLETE": left_complete,
        "LEFT_GRASP_CONFIRMED_SOURCE": left_grasp,
        "LEFT_LIFT_BEGIN": max(left_grasp, left_lift),
        "LEFT_TRANSPORT": max(left_lift, left_transport),
        "RIGHT_APPROACH_BEGIN": max(left_transport, min(right_approach, right_close)),
        "RIGHT_CLOSE_BEGIN": right_close,
        "RIGHT_ACQUIRE_SOURCE": right_acquire,
        "DUAL_CONTACT_SOURCE": max(right_acquire, min(dual, left_release)),
        "LEFT_RELEASE_BEGIN": left_release,
        "RIGHT_OWNED_SOURCE": left_release,
        "RIGHT_TRANSPORT_BEGIN": max(left_release, min(right_transport, final_release)),
        "FINAL_RELEASE_BEGIN": final_release,
        "TASK_END": len(action) - 1,
    }

    expected_pairs = (
        ("APPROACH_START", "LEFT_CLOSE_BEGIN"),
        ("LEFT_CLOSE_BEGIN", "LEFT_CLOSE_COMPLETE"),
        ("LEFT_CLOSE_BEGIN", "LEFT_GRASP_CONFIRMED_SOURCE"),
        ("LEFT_GRASP_CONFIRMED_SOURCE", "LEFT_LIFT_BEGIN"),
        ("LEFT_LIFT_BEGIN", "LEFT_TRANSPORT"),
        ("LEFT_TRANSPORT", "RIGHT_APPROACH_BEGIN"),
        ("RIGHT_APPROACH_BEGIN", "RIGHT_CLOSE_BEGIN"),
        ("RIGHT_CLOSE_BEGIN", "RIGHT_ACQUIRE_SOURCE"),
        ("RIGHT_ACQUIRE_SOURCE", "DUAL_CONTACT_SOURCE"),
        ("DUAL_CONTACT_SOURCE", "LEFT_RELEASE_BEGIN"),
        ("LEFT_RELEASE_BEGIN", "RIGHT_OWNED_SOURCE"),
        ("RIGHT_OWNED_SOURCE", "RIGHT_TRANSPORT_BEGIN"),
        ("RIGHT_TRANSPORT_BEGIN", "FINAL_RELEASE_BEGIN"),
        ("FINAL_RELEASE_BEGIN", "TASK_END"),
    )
    violations = tuple(
        f"{first}>{second}"
        for first, second in expected_pairs
        if frames[first] > frames[second]
    )

    phase_specs = (
        ("APPROACH", "APPROACH_START", "LEFT_CLOSE_BEGIN"),
        ("LEFT_CLOSE", "LEFT_CLOSE_BEGIN", "LEFT_GRASP_CONFIRMED_SOURCE"),
        ("LEFT_HOLD", "LEFT_CLOSE_COMPLETE", "LEFT_LIFT_BEGIN"),
        ("LIFT", "LEFT_LIFT_BEGIN", "LEFT_TRANSPORT"),
        ("LEFT_TRANSPORT", "LEFT_TRANSPORT", "RIGHT_APPROACH_BEGIN"),
        ("HANDOFF_APPROACH", "RIGHT_APPROACH_BEGIN", "RIGHT_CLOSE_BEGIN"),
        ("RECEIVER_CLOSE", "RIGHT_CLOSE_BEGIN", "RIGHT_ACQUIRE_SOURCE"),
        ("DUAL_CONTACT", "RIGHT_ACQUIRE_SOURCE", "LEFT_RELEASE_BEGIN"),
        ("GIVER_RELEASE", "LEFT_RELEASE_BEGIN", "RIGHT_TRANSPORT_BEGIN"),
        ("RIGHT_TRANSPORT", "RIGHT_TRANSPORT_BEGIN", "FINAL_RELEASE_BEGIN"),
        ("FINAL_RELEASE", "FINAL_RELEASE_BEGIN", "TASK_END"),
    )
    labels = np.full(len(action), "PRE_TASK", dtype="U24")
    phase_time = np.zeros(len(action), dtype=np.float64)
    for label, first, second in phase_specs:
        begin, end = frames[first], frames[second]
        if end < begin:
            continue
        labels[begin : end + 1] = label
        if end > begin:
            phase_time[begin : end + 1] = np.linspace(0.0, 1.0, end - begin + 1)
    labels[frames["TASK_END"] :] = "TASK_END"
    phase_time[frames["TASK_END"] :] = 1.0

    pre_frames = int(round(0.3 * fps))
    left_hand = np.full(len(action), "OPEN", dtype="U24")
    right_hand = np.full(len(action), "OPEN", dtype="U24")
    left_hand[max(0, left_close - pre_frames) : left_close] = "PRESHAPE"
    left_hand[left_close:left_complete] = "PROGRESSIVE_CLOSE"
    left_hand[left_complete:left_release] = "HOLD"
    left_hand[left_release:] = "GIVER_RELEASE"
    right_hand[max(0, right_close - pre_frames) : right_close] = "PRESHAPE"
    right_hand[right_close:right_acquire] = "RECEIVER_CLOSE"
    right_hand[right_acquire:final_release] = "DUAL_HOLD"
    right_hand[final_release:] = "FINAL_RELEASE"

    times = {name: float(timestamps[frame]) for name, frame in frames.items()}
    return SourceEventTimeline(
        frames=frames,
        times_sec=times,
        phase_label=labels,
        phase_time=phase_time,
        left_hand_semantic=left_hand,
        right_hand_semantic=right_hand,
        source_order=tuple(sorted(frames, key=frames.get)),
        order_violations=violations,
        evidence={
            "fps_hz": fps,
            "coarse_frames": coarse,
            "left_lift_rule": "first >=0.20 s source-FK window with >=5 mm rise, median vz>=20 mm/s, >=75% positive vz, and >=3/4 leading samples >=20 mm/s; search begins at LEFT close onset so real close/lift overlap is preserved",
            "left_grasp_source_rule": "earlier of pooled close confirmation or sustained table-free lift onset; lift is source ownership evidence, never policy/physics evidence",
            "transport_rule": "first >=0.20 s source-FK window with >=6 mm displacement and median planar speed>=20 mm/s",
            "right_approach_rule": "first source right-TCP speed >=25 mm/s after LEFT transport and before RIGHT close",
            "inter_hand_minimum_m": float(np.min(interhand)),
            "method_identity_consumed": False,
            "policy_or_physics_outcome_consumed": False,
        },
    )


__all__ = [
    "EVENT_NAMES",
    "SourceEventTimeline",
    "build_common_timeline",
    "detect_coarse_events",
]
