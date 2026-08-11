"""Time-scaled, per-trajectory robust ALOHA gripper phase detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHASE_NAMES = np.asarray(("OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE"), dtype="U12")


def odd_window(duration_sec: float, timestamps: np.ndarray) -> int:
    frequency = 1.0 / float(np.median(np.diff(timestamps)))
    samples = max(1, int(round(float(duration_sec) * frequency)))
    return samples if samples % 2 else samples + 1


def smooth_seconds(value: np.ndarray, duration_sec: float, timestamps: np.ndarray) -> tuple[np.ndarray, int]:
    window = odd_window(duration_sec, timestamps)
    if window <= 1:
        return np.asarray(value, dtype=np.float64).copy(), window
    pad = window // 2
    kernel = np.full(window, 1.0 / window)
    return np.convolve(np.pad(value, pad, mode="edge"), kernel, mode="valid"), window


def robust_two_cluster(value: np.ndarray) -> tuple[float, float]:
    value = np.asarray(value, dtype=np.float64)
    lo, hi = np.quantile(value, (0.02, 0.98))
    clipped = np.clip(value, lo, hi)
    centers = np.quantile(clipped, (0.2, 0.8))
    for _ in range(64):
        labels = np.abs(clipped[:, None] - centers[None]).argmin(axis=1)
        updated = np.asarray([
            np.median(clipped[labels == group]) if np.any(labels == group) else centers[group]
            for group in range(2)
        ])
        if np.allclose(updated, centers, atol=1e-12, rtol=0.0):
            break
        centers = updated
    centers.sort()
    return float(centers[0]), float(centers[1])


@dataclass
class GripperResult:
    phase: np.ndarray
    normalized_open: np.ndarray
    smoothed: np.ndarray
    derivative: np.ndarray
    close_score: np.ndarray
    open_score: np.ndarray
    transitions: list[dict[str, object]]
    calibration: dict[str, object]


def detect_gripper_phases(signal: np.ndarray, timestamps: np.ndarray, config: dict[str, float]) -> GripperResult:
    signal = np.asarray(signal, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    filtered, window = smooth_seconds(signal, config["smoothing_sec"], timestamps)
    closed_center, open_center = robust_two_cluster(filtered)
    span = open_center - closed_center
    degenerate = bool(span < max(float(config.get("minimum_cluster_span", 1e-5)), 1e-12))
    if degenerate:
        lo, hi = np.quantile(filtered, (0.05, 0.95))
        closed_center, open_center = float(lo), float(hi)
        span = max(open_center - closed_center, 1e-9)
    normalized = np.clip((filtered - closed_center) / span, 0.0, 1.0)
    derivative = np.gradient(normalized, timestamps)
    derivative_scale = max(float(np.quantile(np.abs(derivative), 0.9)), 1e-6)
    close_score = np.clip(-derivative / derivative_scale, 0.0, 1.0)
    open_score = np.clip(derivative / derivative_scale, 0.0, 1.0)

    open_threshold = float(config["open_normalized_threshold"])
    close_threshold = float(config["close_normalized_threshold"])
    motion_threshold = float(config["motion_score_threshold"])
    debounce = odd_window(config["debounce_sec"], timestamps)
    grasp_duration = odd_window(config["grasp_transition_sec"], timestamps)
    phase = np.empty(len(signal), dtype="U12")
    state = "OPEN" if normalized[0] >= open_threshold else "HOLD" if normalized[0] <= close_threshold else "PREGRASP"
    age = 0
    stable_count = 0
    transitions: list[dict[str, object]] = []

    def transition(index: int, new_state: str, reason: str) -> None:
        nonlocal state, age, stable_count
        old_state = state
        state = new_state
        age = 0
        stable_count = 0
        transitions.append({
            "action_index": int(index),
            "action_time_sec": float(timestamps[index]),
            "from": old_state,
            "to": new_state,
            "reason": reason,
        })

    for index in range(len(signal)):
        value = normalized[index]
        if state == "OPEN":
            if close_score[index] >= motion_threshold and value < open_threshold:
                transition(index, "PREGRASP", "sustained_close_motion_started")
        elif state == "PREGRASP":
            stable_count = stable_count + 1 if value <= close_threshold else 0
            if stable_count >= debounce:
                transition(index - debounce + 1, "GRASP", "closed_cluster_entered")
            elif value >= open_threshold and open_score[index] >= motion_threshold:
                transition(index, "OPEN", "pregrasp_aborted")
        elif state == "GRASP":
            if age >= grasp_duration:
                transition(index, "HOLD", "grasp_transition_dwell_complete")
        elif state == "HOLD":
            if open_score[index] >= motion_threshold and value > close_threshold:
                transition(index, "RELEASE", "sustained_open_motion_started")
        elif state == "RELEASE":
            stable_count = stable_count + 1 if value >= open_threshold else 0
            if stable_count >= debounce:
                transition(index - debounce + 1, "OPEN", "open_cluster_entered")
            elif value <= close_threshold and close_score[index] >= motion_threshold:
                transition(index, "HOLD", "release_aborted")
        phase[index] = state
        age += 1

    return GripperResult(
        phase=phase,
        normalized_open=normalized,
        smoothed=filtered,
        derivative=derivative,
        close_score=close_score,
        open_score=open_score,
        transitions=transitions,
        calibration={
            "method": "per_trajectory_robust_two_cluster",
            "closed_center": closed_center,
            "open_center": open_center,
            "cluster_span": span,
            "degenerate_clusters": degenerate,
            "increasing_is_open": True,
            "smoothing_samples": window,
            "smoothing_sec": float(config["smoothing_sec"]),
            "debounce_samples": debounce,
            "debounce_sec": float(config["debounce_sec"]),
        },
    )

