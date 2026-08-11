"""Global, episode-independent evidence models used by detector v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aloha_magsafe_semantics.gripper_phase import GripperResult, smooth_seconds


def robust_unit(value: np.ndarray, low: float = 0.1, high: float = 0.9) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    lo, hi = np.quantile(value, (low, high))
    if hi - lo <= 1e-12:
        return np.zeros_like(value)
    return np.clip((value - lo) / (hi - lo), 0.0, 1.0)


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def sustained_runs(mask: np.ndarray, timestamps: np.ndarray, minimum_sec: float) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(mask) - 1):
            end = index if active else index - 1
            if timestamps[end] - timestamps[start] >= minimum_sec:
                runs.append((start, end))
            start = None
    return runs


def forward_dwell_score(mask: np.ndarray, timestamps: np.ndarray, dwell_sec: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    score = np.zeros(len(mask), dtype=np.float64)
    end = 0
    for start in range(len(mask)):
        end = max(end, start)
        while end + 1 < len(mask) and mask[end + 1]:
            end += 1
        if mask[start]:
            duration = float(timestamps[end] - timestamps[start])
            score[start] = min(1.0, duration / max(dwell_sec, 1e-9))
        if end == start and not mask[start]:
            end += 1
    return score


def future_window_endpoint(timestamps: np.ndarray, index: int, seconds: float) -> int:
    return min(len(timestamps) - 1, int(np.searchsorted(timestamps, timestamps[index] + seconds, side="right") - 1))


def past_window_start(timestamps: np.ndarray, index: int, seconds: float) -> int:
    return max(0, int(np.searchsorted(timestamps, timestamps[index] - seconds, side="left")))


@dataclass(frozen=True)
class EvidenceCandidate:
    action_index: int
    score: float
    components: dict[str, float]
    source: str


def rotation_segmentation(
    timestamps: np.ndarray,
    rotations: np.ndarray,
    angular_speed: np.ndarray,
    hold: np.ndarray,
    start_index: int,
    end_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Segment onset/dominant rotation/portrait plateau without frame priors."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    angular_speed = np.asarray(angular_speed, dtype=np.float64)
    length = len(timestamps)
    start_index = int(np.clip(start_index, 0, length - 2))
    end_index = int(np.clip(end_index, start_index + 1, length - 1))
    settings = config["rotation_v2"]
    smoothed, window = smooth_seconds(angular_speed, float(settings["smoothing_sec"]), timestamps)
    local = smoothed[start_index : end_index + 1]
    median = float(np.median(local))
    mad = float(np.median(np.abs(local - median)))
    threshold = max(
        float(np.quantile(local, float(settings["motion_quantile"]))),
        median + float(settings["motion_mad_multiplier"]) * max(mad, 1e-9),
    )
    low_threshold = float(np.quantile(local, float(settings["low_speed_quantile"])))
    active = np.zeros(length, dtype=bool)
    active[start_index : end_index + 1] = local >= threshold
    runs = sustained_runs(active, timestamps, float(settings["minimum_rotation_run_sec"]))
    dt = np.gradient(timestamps)
    energy = smoothed * dt
    total_interval_energy = float(np.sum(energy[start_index : end_index + 1]))
    cumulative = np.zeros(length, dtype=np.float64)
    cumulative[start_index : end_index + 1] = np.cumsum(energy[start_index : end_index + 1])
    if total_interval_energy > 1e-12:
        cumulative[start_index : end_index + 1] /= total_interval_energy
    cumulative[end_index + 1 :] = 1.0
    low = smoothed <= max(low_threshold, median + 0.25 * mad)
    low_dwell = forward_dwell_score(low, timestamps, float(settings["plateau_dwell_sec"]))

    onset_candidates: list[EvidenceCandidate] = []
    plateau_candidates: list[EvidenceCandidate] = []
    run_energy = []
    for run_start, run_end in runs:
        value = float(np.sum(energy[run_start : run_end + 1]))
        run_energy.append(value)
    dominant = int(np.argmax(run_energy)) if run_energy else None
    for run_number, (run_start, run_end) in enumerate(runs):
        share = run_energy[run_number] / max(total_interval_energy, 1e-12)
        pre = past_window_start(timestamps, run_start, float(settings["onset_context_sec"]))
        onset_contrast = float(np.clip(
            (np.mean(smoothed[run_start : run_end + 1]) - np.mean(smoothed[pre : run_start + 1]))
            / max(threshold, 1e-9), 0.0, 1.0,
        ))
        onset_components = {
            "sustained_angular_onset": 1.0,
            "angular_contrast": onset_contrast,
            "left_gripper_hold": float(np.mean(hold[run_start : run_end + 1])),
            "dominant_rotation_share": float(np.clip(share * 2.0, 0.0, 1.0)),
        }
        onset_score = float(np.average(list(onset_components.values()), weights=(0.32, 0.23, 0.20, 0.25)))
        onset_candidates.append(EvidenceCandidate(run_start, onset_score, onset_components, "v2_rotation_change_point_onset"))

        candidate_index = min(end_index, run_end + 1)
        future_end = future_window_endpoint(timestamps, candidate_index, float(settings["post_plateau_sec"]))
        relative_angles = np.asarray([
            rotation_angle(rotations[candidate_index].T @ rotations[index])
            for index in range(candidate_index, future_end + 1)
        ])
        stability = float(np.exp(-np.max(relative_angles, initial=0.0) / max(float(settings["stability_angle_rad"]), 1e-9)))
        later_start = candidate_index
        later_end = min(end_index, future_window_endpoint(timestamps, candidate_index, float(settings["later_consistency_sec"])))
        later_energy = float(np.sum(energy[later_start : later_end + 1]))
        later_consistency = float(np.exp(-later_energy / max(run_energy[run_number], 1e-9)))
        completion = float(cumulative[candidate_index])
        components = {
            "cumulative_rotation_completion": completion,
            "dominant_rotation_interval": 1.0 if run_number == dominant else float(np.clip(share, 0.0, 1.0)),
            "plateau_low_speed_dwell": float(low_dwell[candidate_index]),
            "post_plateau_orientation_stability": stability,
            "later_motion_consistency": later_consistency,
            "left_gripper_hold": float(np.mean(hold[candidate_index : future_end + 1])),
        }
        weights = np.asarray((0.19, 0.18, 0.22, 0.18, 0.13, 0.10))
        plateau_score = float(np.dot(weights, np.asarray(list(components.values()))))
        plateau_candidates.append(EvidenceCandidate(candidate_index, plateau_score, components, "v2_rotation_semi_markov_plateau"))

    # A cumulative-completion change point handles overshoot/return and cases
    # where a thresholded run fragments into several small runs.
    completion_threshold = float(settings["completion_threshold"])
    completion_indices = np.flatnonzero(
        (np.arange(length) >= start_index)
        & (np.arange(length) <= end_index)
        & (cumulative >= completion_threshold)
        & (low_dwell >= 0.75)
    )
    if len(completion_indices):
        index = int(completion_indices[0])
        future_end = future_window_endpoint(timestamps, index, float(settings["post_plateau_sec"]))
        relative_angles = [rotation_angle(rotations[index].T @ rotations[j]) for j in range(index, future_end + 1)]
        components = {
            "cumulative_rotation_completion": float(cumulative[index]),
            "dominant_rotation_interval": 0.75,
            "plateau_low_speed_dwell": float(low_dwell[index]),
            "post_plateau_orientation_stability": float(np.exp(-max(relative_angles, default=0.0) / max(float(settings["stability_angle_rad"]), 1e-9))),
            "later_motion_consistency": 0.75,
            "left_gripper_hold": float(np.mean(hold[index : future_end + 1])),
        }
        plateau_candidates.append(EvidenceCandidate(
            index,
            float(np.dot(np.asarray((0.19, 0.18, 0.22, 0.18, 0.13, 0.10)), np.asarray(list(components.values())))),
            components,
            "v2_rotation_monotonic_completion_plateau",
        ))
    return {
        "onset_candidates": onset_candidates,
        "plateau_candidates": plateau_candidates,
        "smoothed_angular_speed": smoothed,
        "orientation_progress_monotonic_envelope": np.maximum.accumulate(cumulative),
        "low_speed_dwell_score": low_dwell,
        "runs": runs,
        "dominant_run": None if dominant is None else runs[dominant],
        "threshold_radps": threshold,
        "low_threshold_radps": low_threshold,
        "smoothing_samples": window,
        "interval": [start_index, end_index],
    }


def terminal_suffix_analysis(
    timestamps: np.ndarray,
    features: dict[str, np.ndarray],
    left_gripper: GripperResult,
    right_gripper: GripperResult,
    prerequisite_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Backward no-later-motion analysis for return/terminal semantics."""
    settings = config["terminal_v2"]
    timestamps = np.asarray(timestamps, dtype=np.float64)
    length = len(timestamps)
    left_speed, _ = smooth_seconds(features["left_linear_speed"], float(settings["smoothing_sec"]), timestamps)
    right_speed, _ = smooth_seconds(features["right_linear_speed"], float(settings["smoothing_sec"]), timestamps)
    left_angular, _ = smooth_seconds(features["left_angular_speed"], float(settings["smoothing_sec"]), timestamps)
    right_angular, _ = smooth_seconds(features["right_angular_speed"], float(settings["smoothing_sec"]), timestamps)
    linear_scale = max(float(np.quantile(np.concatenate((left_speed, right_speed)), 0.85)), 1e-9)
    angular_scale = max(float(np.quantile(np.concatenate((left_angular, right_angular)), 0.85)), 1e-9)
    # Terminal semantics depend on persistent phase transitions, not isolated
    # command jitter inside an unchanged OPEN/HOLD state.
    gripper_energy = np.zeros(length, dtype=np.float64)
    minimum_gripper_transition_sec = float(settings["minimum_suffix_sec"]) * 0.5
    for result in (left_gripper, right_gripper):
        for transition_position, transition in enumerate(result.transitions):
            index = int(transition["action_index"])
            next_index = (
                int(result.transitions[transition_position + 1]["action_index"])
                if transition_position + 1 < len(result.transitions) else length - 1
            )
            if timestamps[next_index] - timestamps[index] >= minimum_gripper_transition_sec:
                gripper_energy[max(0, index - 1) : min(length, index + 2)] = 1.0
    gripper_scale = 1.0
    instantaneous = (
        0.38 * np.maximum(left_speed, right_speed) / linear_scale
        + 0.32 * np.maximum(left_angular, right_angular) / angular_scale
        + 0.30 * gripper_energy / gripper_scale
    )
    instantaneous = np.clip(instantaneous, 0.0, 3.0)
    dt = np.gradient(timestamps)
    future_energy = np.cumsum((instantaneous * dt)[::-1])[::-1]
    remaining_duration = np.maximum(timestamps[-1] - timestamps, np.median(np.diff(timestamps)))
    future_energy_rate = future_energy / remaining_duration
    future_peak = np.maximum.accumulate(instantaneous[::-1])[::-1]
    suffix_spreads: list[np.ndarray] = []
    suffix_rotation_spreads: list[np.ndarray] = []
    for side in ("left", "right"):
        position = np.asarray(features[f"{side}_tcp_position"])
        suffix_min = np.minimum.accumulate(position[::-1], axis=0)[::-1]
        suffix_max = np.maximum.accumulate(position[::-1], axis=0)[::-1]
        suffix_spreads.append(np.linalg.norm(suffix_max - suffix_min, axis=1))
        cumulative = np.asarray(features[f"{side}_cumulative_rotation"])
        suffix_rotation_spreads.append(cumulative[-1] - cumulative)
    position_spread = np.maximum(suffix_spreads[0], suffix_spreads[1])
    rotation_spread = np.maximum(suffix_rotation_spreads[0], suffix_rotation_spreads[1])
    tail_start = int(np.searchsorted(timestamps, timestamps[-1] - max(0.75, float(settings["minimum_suffix_sec"])), side="left"))
    def terminal_mode(phase: np.ndarray) -> str:
        names, counts = np.unique(phase[tail_start:], return_counts=True)
        return str(names[int(np.argmax(counts))])
    left_terminal_phase = terminal_mode(left_gripper.phase)
    right_terminal_phase = terminal_mode(right_gripper.phase)
    stable_gripper = np.minimum(
        forward_dwell_score(left_gripper.phase == left_terminal_phase, timestamps, float(settings["minimum_suffix_sec"])),
        forward_dwell_score(right_gripper.phase == right_terminal_phase, timestamps, float(settings["minimum_suffix_sec"])),
    )
    valid = (
        (future_energy_rate <= float(settings["future_energy_rate_threshold"]))
        & (future_peak <= float(settings["future_peak_threshold"]))
        & (position_spread <= float(settings["position_spread_m"]))
        & (rotation_spread <= float(settings["rotation_spread_rad"]))
        & (stable_gripper >= 0.99)
    )
    valid[: max(0, int(prerequisite_index))] = False
    candidates: list[EvidenceCandidate] = []
    valid_runs = sustained_runs(valid, timestamps, float(settings["minimum_suffix_sec"]))
    for start, _ in valid_runs:
        components = {
            "future_translation_rotation_energy_low": float(np.exp(-future_energy_rate[start] / max(float(settings["future_energy_rate_threshold"]), 1e-9))),
            "no_later_high_energy_interval": float(np.clip(1.0 - future_peak[start] / max(float(settings["future_peak_threshold"]), 1e-9), 0.0, 1.0)),
            "terminal_position_stability": float(np.exp(-position_spread[start] / max(float(settings["position_spread_m"]), 1e-9))),
            "terminal_rotation_stability": float(np.exp(-rotation_spread[start] / max(float(settings["rotation_spread_rad"]), 1e-9))),
            "terminal_gripper_stability": float(stable_gripper[start]),
        }
        score = float(np.average(list(components.values()), weights=(0.25, 0.20, 0.20, 0.15, 0.20)))
        candidates.append(EvidenceCandidate(start, score, components, "v2_backward_terminal_suffix"))
    # Always expose the end sample as a low-information fallback; confidence
    # remains LOW/AMBIGUOUS when a sustained suffix is absent.
    if not candidates:
        index = length - 1
        components = {
            "future_translation_rotation_energy_low": 0.25,
            "no_later_high_energy_interval": 1.0,
            "terminal_position_stability": 0.25,
            "terminal_rotation_stability": 0.25,
            "terminal_gripper_stability": float(stable_gripper[index]),
        }
        candidates.append(EvidenceCandidate(index, 0.35, components, "v2_terminal_end_fallback"))

    start_position = np.asarray(features["left_tcp_position"])[0]
    return_distance = np.linalg.norm(np.asarray(features["left_tcp_position"]) - start_position, axis=1)
    return_scale = max(float(np.quantile(return_distance, 0.85)), 1e-9)
    return_proximity = np.clip(1.0 - return_distance / return_scale, 0.0, 1.0)
    low_left = robust_unit(-left_speed, 0.1, 0.9)
    return_mask = (return_proximity >= float(settings["return_proximity_threshold"])) & (low_left >= float(settings["return_low_speed_score"]))
    return_candidates: list[EvidenceCandidate] = []
    for start, _ in sustained_runs(return_mask, timestamps, float(settings["return_dwell_sec"])):
        no_later = float(np.exp(-future_energy_rate[start] / max(float(settings["future_energy_rate_threshold"]), 1e-9)))
        components = {
            "return_to_start_proximity": float(return_proximity[start]),
            "left_low_speed_dwell": float(low_left[start]),
            "future_motion_consistency": no_later,
        }
        return_candidates.append(EvidenceCandidate(start, float(np.average(list(components.values()), weights=(0.45, 0.30, 0.25))), components, "v2_return_near_home_dwell"))
    if not return_candidates and candidates:
        index = int(candidates[0].action_index)
        components = {
            "return_to_start_proximity": float(return_proximity[index]),
            "left_low_speed_dwell": float(low_left[index]),
            "future_motion_consistency": float(np.exp(-future_energy_rate[index] / max(float(settings["future_energy_rate_threshold"]), 1e-9))),
        }
        return_candidates.append(EvidenceCandidate(
            index,
            float(min(0.42, np.average(list(components.values()), weights=(0.45, 0.30, 0.25)))),
            components,
            "v2_return_terminal_low_evidence_fallback",
        ))
    return {
        "terminal_candidates": candidates,
        "return_candidates": return_candidates,
        "terminal_hold_start": candidates[0].action_index,
        "instantaneous_motion_energy": instantaneous,
        "future_motion_energy": future_energy,
        "future_motion_energy_rate": future_energy_rate,
        "future_peak_energy": future_peak,
        "suffix_position_spread_m": position_spread,
        "suffix_rotation_spread_rad": rotation_spread,
        "valid_terminal_suffix": valid,
    }


def release_candidates(
    side: str,
    timestamps: np.ndarray,
    gripper: GripperResult,
    tcp_position: np.ndarray,
    start_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["release_v2"]
    timestamps = np.asarray(timestamps, dtype=np.float64)
    derivative = np.asarray(gripper.derivative, dtype=np.float64)
    positive = np.maximum(derivative, 0.0)
    derivative_score = robust_unit(positive, 0.55, 0.98)
    open_state = np.asarray(gripper.normalized_open) >= float(settings["open_normalized_threshold"])
    stable_open = forward_dwell_score(open_state, timestamps, float(settings["open_plateau_sec"]))
    low_speed = robust_unit(-np.linalg.norm(np.gradient(tcp_position, timestamps, axis=0), axis=1), 0.1, 0.85)
    indices: set[int] = set()
    for index in range(max(1, start_index), len(timestamps) - 1):
        if positive[index] >= positive[index - 1] and positive[index] >= positive[index + 1] and derivative_score[index] >= 0.25:
            indices.add(index)
        if open_state[index] and not open_state[index - 1]:
            indices.add(index)
    candidates: list[EvidenceCandidate] = []
    for index in sorted(indices):
        end = future_window_endpoint(timestamps, index, float(settings["departure_window_sec"]))
        departure = float(np.linalg.norm(tcp_position[end] - tcp_position[index]))
        departure_score = float(np.clip(departure / max(float(settings["departure_distance_m"]), 1e-9), 0.0, 1.0))
        transition = float(any(
            int(row["action_index"]) >= index - 2 and int(row["action_index"]) <= end and row["to"] in ("RELEASE", "OPEN")
            for row in gripper.transitions
        ))
        components = {
            "robust_opening_derivative": float(derivative_score[index]),
            "release_phase_transition": transition,
            "stable_open_plateau": float(stable_open[index]),
            "post_release_hand_departure": departure_score,
            "local_low_speed_support": float(low_speed[index]),
        }
        score = float(np.average(list(components.values()), weights=(0.26, 0.23, 0.24, 0.19, 0.08)))
        candidates.append(EvidenceCandidate(index, score, components, f"v2_{side}_release_multimodal"))
    strongest_physical = max((max(row.components["robust_opening_derivative"], row.components["release_phase_transition"]) for row in candidates), default=0.0)
    return {
        "candidates": candidates,
        "physical_release_evidence": bool(strongest_physical >= float(settings["minimum_physical_release_evidence"])),
        "strongest_physical_evidence": float(strongest_physical),
        "derivative_score": derivative_score,
        "stable_open_score": stable_open,
    }


def right_removal_segmentation(
    timestamps: np.ndarray,
    position: np.ndarray,
    speed: np.ndarray,
    hold: np.ndarray,
    start_index: int,
    end_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Phase-relative approach/detachment/removal evidence."""
    settings = config["removal_v2"]
    timestamps = np.asarray(timestamps, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    velocity = np.gradient(position, timestamps, axis=0)
    start_index = int(np.clip(start_index, 0, len(timestamps) - 2))
    end_index = int(np.clip(end_index, start_index + 1, len(timestamps) - 1))
    local = position[start_index : end_index + 1] - position[start_index]
    if len(local) >= 2:
        _, _, vh = np.linalg.svd(local - np.mean(local, axis=0), full_matrices=False)
        direction = vh[0]
    else:
        direction = np.asarray((1.0, 0.0, 0.0))
    net = position[end_index] - position[start_index]
    if np.dot(direction, net) < 0:
        direction = -direction
    projected_velocity = velocity @ direction
    positive = np.maximum(projected_velocity, 0.0)
    threshold = max(float(np.quantile(positive[start_index : end_index + 1], float(settings["motion_quantile"]))), 1e-6)
    active = (positive >= threshold) & (hold > 0.5)
    runs = sustained_runs(active, timestamps, float(settings["minimum_run_sec"]))
    onset_candidates: list[EvidenceCandidate] = []
    removed_candidates: list[EvidenceCandidate] = []
    speed_low = robust_unit(-speed, 0.1, 0.8)
    for run_start, run_end in runs:
        displacement = np.maximum.accumulate((position[run_start : run_end + 1] - position[run_start]) @ direction)
        total = float(displacement[-1])
        direction_alignment = np.clip(projected_velocity[run_start : run_end + 1] / np.maximum(speed[run_start : run_end + 1], 1e-9), -1.0, 1.0)
        onset_components = {
            "sustained_removal_onset": 1.0,
            "right_gripper_hold": float(np.mean(hold[run_start : run_end + 1])),
            "removal_direction_stability": float(np.clip(np.mean(direction_alignment), 0.0, 1.0)),
        }
        onset_candidates.append(EvidenceCandidate(run_start, float(np.average(list(onset_components.values()), weights=(0.45, 0.25, 0.30))), onset_components, "v2_phase_relative_detachment_onset"))
        if total > 1e-9:
            reached = np.flatnonzero(displacement / total >= float(settings["completion_fraction"]))
            completion_index = run_start + int(reached[0]) if len(reached) else run_end
        else:
            completion_index = run_end
        components = {
            "cumulative_removal_progress": 1.0 if total >= float(settings["minimum_removal_displacement_m"]) else float(total / max(float(settings["minimum_removal_displacement_m"]), 1e-9)),
            "removal_direction_stability": float(np.clip(np.mean(direction_alignment), 0.0, 1.0)),
            "right_gripper_hold": float(hold[completion_index]),
            "leaves_approach_region": float(np.clip(total / max(float(settings["approach_region_exit_m"]), 1e-9), 0.0, 1.0)),
            "endpoint_speed_drop": float(speed_low[min(run_end + 1, len(speed_low) - 1)]),
        }
        removed_candidates.append(EvidenceCandidate(completion_index, float(np.average(list(components.values()), weights=(0.28, 0.20, 0.18, 0.20, 0.14))), components, "v2_phase_relative_removal_completion"))
    return {
        "detachment_candidates": onset_candidates,
        "removed_candidates": removed_candidates,
        "estimated_removal_direction": direction,
        "projected_velocity": projected_velocity,
        "active_runs": runs,
    }
