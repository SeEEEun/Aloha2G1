"""Evidence extraction and event candidate generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .features import extract_task_space_features
from .gripper_phase import GripperResult, detect_gripper_phases, smooth_seconds


@dataclass(frozen=True)
class EventCandidate:
    event_name: str
    action_index: int
    action_time_sec: float
    score: float
    score_components: dict[str, float]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def robust_unit(value: np.ndarray, low_quantile: float = 0.1, high_quantile: float = 0.9) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    low, high = np.quantile(value, (low_quantile, high_quantile))
    if high - low < 1e-12:
        return np.zeros_like(value)
    return np.clip((value - low) / (high - low), 0.0, 1.0)


def _falling(value: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    derivative = np.gradient(np.asarray(value, dtype=np.float64), timestamps)
    return robust_unit(-derivative, 0.5, 0.95)


def _rising(value: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    derivative = np.gradient(np.asarray(value, dtype=np.float64), timestamps)
    return robust_unit(derivative, 0.5, 0.95)


def _local_maxima(score: np.ndarray, timestamps: np.ndarray, separation_sec: float, limit: int) -> list[int]:
    score = np.asarray(score, dtype=np.float64)
    candidates = [index for index in range(1, len(score) - 1) if score[index] >= score[index - 1] and score[index] >= score[index + 1]]
    candidates.extend((0, len(score) - 1))
    candidates = sorted(set(candidates), key=lambda index: (-score[index], index))
    # Preserve temporal coverage without injecting a semantic time prior.  Flat
    # dwell scores otherwise fill the top-K list from only one portion of a
    # trajectory and make a valid globally ordered path undecodable.
    temporal_seeds: list[int] = []
    bin_count = min(16, max(1, limit // 2))
    for group in np.array_split(np.arange(len(score)), bin_count):
        if len(group):
            temporal_seeds.append(int(group[np.argmax(score[group])]))
    candidates = temporal_seeds + [index for index in candidates if index not in temporal_seeds]
    selected: list[int] = []
    for index in candidates:
        if all(abs(float(timestamps[index] - timestamps[other])) >= separation_sec for other in selected):
            selected.append(index)
        if len(selected) >= limit:
            break
    return sorted(selected)


def _transition_impulses(result: GripperResult, target_states: tuple[str, ...], length: int) -> np.ndarray:
    impulse = np.zeros(length, dtype=np.float64)
    for row in result.transitions:
        if row["to"] in target_states:
            impulse[int(row["action_index"])] = 1.0
    if np.any(impulse):
        kernel = np.asarray((0.15, 0.35, 0.7, 1.0, 0.7, 0.35, 0.15))
        impulse = np.convolve(impulse, kernel, mode="same")
    return np.clip(impulse, 0.0, 1.0)


def _run_boundary_score(
    state: np.ndarray,
    target_open: bool,
    timestamps: np.ndarray,
    minimum_hold_sec: float,
    require_exit: bool = False,
) -> np.ndarray:
    """Score signal-derived state entries by their future dwell duration."""
    state = np.asarray(state, dtype=bool)
    score = np.zeros(len(state), dtype=np.float64)
    start = 0
    for end in range(1, len(state) + 1):
        if end == len(state) or state[end] != state[start]:
            if bool(state[start]) == target_open:
                duration = float(timestamps[end - 1] - timestamps[start]) if end - start > 1 else 0.0
                exited = end < len(state) and state[end] != state[start]
                if start > 0 and state[start - 1] != state[start] and (exited or not require_exit):
                    score[start] = min(1.0, duration / max(minimum_hold_sec, 1e-9))
            start = end
    return score


def _sustained_runs(mask: np.ndarray, timestamps: np.ndarray, minimum_sec: float) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    result: list[tuple[int, int]] = []
    start = 0
    for end in range(1, len(mask) + 1):
        if end == len(mask) or mask[end] != mask[start]:
            duration = float(timestamps[end - 1] - timestamps[start]) if end - start > 1 else 0.0
            if mask[start] and duration >= minimum_sec:
                result.append((start, end - 1))
            start = end
    return result


def _candidates(
    name: str,
    score: np.ndarray,
    components: dict[str, np.ndarray],
    timestamps: np.ndarray,
    config: dict[str, Any],
    source: str,
) -> list[EventCandidate]:
    decoder = config["decoder"]
    separation = float(config["kinematics"]["minimum_event_separation_sec"])
    indices = _local_maxima(score, timestamps, separation, int(decoder["candidates_per_event"]))
    return [EventCandidate(
        event_name=name,
        action_index=int(index),
        action_time_sec=float(timestamps[index]),
        score=float(np.clip(score[index], 0.0, 1.0)),
        score_components={key: float(np.clip(value[index], 0.0, 1.0)) for key, value in components.items()},
        source=source,
    ) for index in indices]


def extract_event_candidates(
    action: np.ndarray,
    timestamps: np.ndarray,
    fk_trajectory: dict[str, Any],
    task_geometry: dict[str, Any],
    detector_config: dict[str, Any],
) -> tuple[dict[str, list[EventCandidate]], dict[str, Any]]:
    action = np.asarray(action, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    features = extract_task_space_features(fk_trajectory, timestamps, task_geometry)
    gripper_config = detector_config["gripper"]
    left_gripper = detect_gripper_phases(action[:, 6], timestamps, gripper_config)
    right_gripper = detect_gripper_phases(action[:, 13], timestamps, gripper_config)
    smoothing_sec = float(detector_config["kinematics"]["feature_smoothing_sec"])
    left_speed, _ = smooth_seconds(features["left_linear_speed"], smoothing_sec, timestamps)
    right_speed, _ = smooth_seconds(features["right_linear_speed"], smoothing_sec, timestamps)
    left_angular, _ = smooth_seconds(features["left_angular_speed"], smoothing_sec, timestamps)
    right_angular, _ = smooth_seconds(features["right_angular_speed"], smoothing_sec, timestamps)
    left_speed_unit = robust_unit(left_speed)
    right_speed_unit = robust_unit(right_speed)
    left_angular_unit = robust_unit(left_angular)
    right_angular_unit = robust_unit(right_angular)
    left_low_speed = 1.0 - robust_unit(left_speed, 0.1, 0.65)
    right_low_speed = 1.0 - robust_unit(right_speed, 0.1, 0.65)
    left_low_angular = 1.0 - robust_unit(left_angular, 0.1, 0.65)
    right_low_angular = 1.0 - robust_unit(right_angular, 0.1, 0.65)
    left_motion_onset = np.maximum(_rising(left_speed, timestamps), left_speed_unit)
    right_motion_onset = np.maximum(_rising(right_speed, timestamps), right_speed_unit)
    left_rotation_onset = np.maximum(_rising(left_angular, timestamps), left_angular_unit)
    right_motion_endpoint = np.maximum(_falling(right_speed, timestamps), right_low_speed)
    left_motion_endpoint = np.maximum(_falling(left_speed, timestamps), left_low_speed)
    left_rotation_endpoint = np.maximum(_falling(left_angular, timestamps), left_low_angular)
    left_close_transition = _transition_impulses(left_gripper, ("PREGRASP", "GRASP"), len(action))
    right_close_transition = _transition_impulses(right_gripper, ("PREGRASP", "GRASP"), len(action))
    left_open_transition = _transition_impulses(left_gripper, ("OPEN",), len(action))
    right_open_transition = _transition_impulses(right_gripper, ("OPEN",), len(action))
    minimum_hold_sec = float(detector_config["kinematics"].get("minimum_gripper_hold_sec", 0.5))
    left_binary_open = left_gripper.normalized_open >= 0.5
    right_binary_open = right_gripper.normalized_open >= 0.5
    left_close_dwell = _run_boundary_score(left_binary_open, False, timestamps, minimum_hold_sec, require_exit=True)
    right_close_dwell = _run_boundary_score(right_binary_open, False, timestamps, minimum_hold_sec, require_exit=True)
    left_open_dwell = _run_boundary_score(left_binary_open, True, timestamps, minimum_hold_sec)
    right_open_dwell = _run_boundary_score(right_binary_open, True, timestamps, minimum_hold_sec)
    left_close_transition = np.maximum(left_close_transition, left_close_dwell)
    right_close_transition = np.maximum(right_close_transition, right_close_dwell)
    left_open_transition = np.maximum(left_open_transition, left_open_dwell)
    right_open_transition = np.maximum(right_open_transition, right_open_dwell)
    left_hold = np.isin(left_gripper.phase, ("GRASP", "HOLD")).astype(np.float64)
    right_hold = np.isin(right_gripper.phase, ("GRASP", "HOLD")).astype(np.float64)
    phone_distance = features.get("left_phone_region_distance")
    phone_proximity = 1.0 - robust_unit(phone_distance, 0.02, 0.75) if phone_distance is not None else np.full(len(action), 0.5)
    charger_velocity = features.get("left_charger_direction_velocity")
    charger_direction = robust_unit(charger_velocity, 0.4, 0.95) if charger_velocity is not None else left_motion_onset
    left_return_proximity = 1.0 - robust_unit(np.linalg.norm(features["left_tcp_position"] - features["left_tcp_position"][0], axis=1), 0.02, 0.8)
    right_return_proximity = 1.0 - robust_unit(np.linalg.norm(features["right_tcp_position"] - features["right_tcp_position"][0], axis=1), 0.02, 0.8)
    terminal_stability = (left_low_speed + right_low_speed + left_low_angular + right_low_angular) * 0.25
    linear_motion_threshold_left = float(np.quantile(left_speed, detector_config["kinematics"]["speed_motion_quantile"]))
    linear_motion_threshold_right = float(np.quantile(right_speed, detector_config["kinematics"]["speed_motion_quantile"]))
    sustained_motion = (left_speed > linear_motion_threshold_left) | (right_speed > linear_motion_threshold_right)
    dwell_sec = float(detector_config["kinematics"]["motion_onset_dwell_sec"])
    left_speed_runs = _sustained_runs(left_speed > linear_motion_threshold_left, timestamps, dwell_sec)
    right_speed_runs = _sustained_runs(right_speed > linear_motion_threshold_right, timestamps, dwell_sec)
    angular_threshold_left = float(np.quantile(left_angular, detector_config["kinematics"]["angular_motion_quantile"]))
    left_angular_runs = _sustained_runs(left_angular > angular_threshold_left, timestamps, dwell_sec)
    left_speed_run_onset = np.zeros(len(action)); left_speed_run_endpoint = np.zeros(len(action))
    right_speed_run_onset = np.zeros(len(action)); right_speed_run_endpoint = np.zeros(len(action))
    left_angular_run_onset = np.zeros(len(action)); left_angular_run_endpoint = np.zeros(len(action))
    for start, end in left_speed_runs:
        left_speed_run_onset[start] = 1.0; left_speed_run_endpoint[min(end + 1, len(action) - 1)] = 1.0
    for start, end in right_speed_runs:
        right_speed_run_onset[start] = 1.0; right_speed_run_endpoint[min(end + 1, len(action) - 1)] = 1.0
    for start, end in left_angular_runs:
        left_angular_run_onset[start] = 1.0; left_angular_run_endpoint[min(end + 1, len(action) - 1)] = 1.0
    terminal_impulse = np.zeros(len(action), dtype=np.float64)
    motion_runs = _sustained_runs(
        sustained_motion,
        timestamps,
        float(detector_config["kinematics"]["motion_onset_dwell_sec"]),
    )
    if motion_runs:
        terminal_index = min(len(action) - 1, motion_runs[-1][1] + 1)
        terminal_impulse[terminal_index] = 1.0
    else:
        terminal_impulse[0] = 1.0

    score_components: dict[str, dict[str, np.ndarray]] = {
        "left_phone_grasp_start": {
            "gripper_close": left_close_transition,
            "phone_region_proximity": phone_proximity,
            "approach_endpoint": left_motion_endpoint,
        },
        "phone_rotation_to_portrait_start": {
            "angular_motion_onset": np.maximum(left_angular_run_onset, 0.35 * _rising(left_angular, timestamps)),
            "left_gripper_hold": left_hold,
            "translation_stability": left_low_speed,
        },
        "phone_portrait_reached": {
            "angular_motion_endpoint": np.maximum(left_angular_run_endpoint, 0.35 * _falling(left_angular, timestamps)),
            "left_gripper_hold": left_hold,
            "angular_dwell": left_low_angular,
        },
        "right_accessory_grasp_start": {
            "gripper_close": right_close_transition,
            "right_approach_endpoint": right_motion_endpoint,
            "right_low_speed": right_low_speed,
        },
        "accessory_detachment_start": {
            "right_motion_onset": np.maximum(right_speed_run_onset, 0.35 * _rising(right_speed, timestamps)),
            "right_gripper_hold": right_hold,
            "right_direction_change": _rising(right_speed, timestamps),
        },
        "accessory_removed": {
            "right_motion_endpoint": np.maximum(right_speed_run_endpoint, 0.35 * _falling(right_speed, timestamps)),
            "right_gripper_hold": right_hold,
            "right_displacement": robust_unit(features["right_displacement_from_start"]),
        },
        "phone_move_to_charger_start": {
            "left_motion_onset": np.maximum(left_speed_run_onset, 0.35 * _rising(left_speed, timestamps)),
            "left_gripper_hold": left_hold,
            "charger_direction_alignment": charger_direction,
        },
        "phone_charger_attachment_complete": {
            "left_motion_endpoint": np.maximum(left_speed_run_endpoint, 0.35 * _falling(left_speed, timestamps)),
            "left_gripper_hold": left_hold,
            "linear_angular_dwell": (left_low_speed + left_low_angular) * 0.5,
        },
        "left_phone_release_complete": {
            "gripper_open": left_open_transition,
            "left_low_speed": left_low_speed,
            "left_open_state": (left_gripper.phase == "OPEN").astype(np.float64),
        },
        "right_accessory_release_complete": {
            "gripper_open": right_open_transition,
            "right_low_speed": right_low_speed,
            "right_open_state": (right_gripper.phase == "OPEN").astype(np.float64),
        },
        "left_arm_return_near_home": {
            "return_proximity": left_return_proximity,
            "left_low_speed": left_low_speed,
            "left_open_state": (left_gripper.phase == "OPEN").astype(np.float64),
        },
        "task_end": {
            "bimanual_low_motion": np.maximum(terminal_stability, terminal_impulse),
            "left_return_proximity": left_return_proximity,
            "right_return_proximity": right_return_proximity,
        },
    }
    weights = {
        "left_phone_grasp_start": (0.48, 0.27, 0.25),
        "phone_rotation_to_portrait_start": (0.62, 0.28, 0.10),
        "phone_portrait_reached": (0.62, 0.23, 0.15),
        "right_accessory_grasp_start": (0.55, 0.27, 0.18),
        "accessory_detachment_start": (0.62, 0.25, 0.13),
        "accessory_removed": (0.58, 0.27, 0.15),
        "phone_move_to_charger_start": (0.58, 0.25, 0.17),
        "phone_charger_attachment_complete": (0.58, 0.25, 0.17),
        "left_phone_release_complete": (0.58, 0.18, 0.24),
        "right_accessory_release_complete": (0.58, 0.18, 0.24),
        "left_arm_return_near_home": (0.48, 0.30, 0.22),
        "task_end": (0.50, 0.25, 0.25),
    }
    candidate_map: dict[str, list[EventCandidate]] = {}
    for event_name, components in score_components.items():
        weight = weights[event_name]
        score = sum(component_weight * component for component_weight, component in zip(weight, components.values()))
        candidate_map[event_name] = _candidates(
            event_name, score, components, timestamps, detector_config, "gripper+task_space_evidence"
        )
    context = {
        "features": features,
        "left_gripper": left_gripper,
        "right_gripper": right_gripper,
        "scores": {
            name: sum(component_weight * component for component_weight, component in zip(weights[name], components.values()))
            for name, components in score_components.items()
        },
        "score_components": score_components,
        "derived": {
            "left_speed_smoothed": left_speed,
            "right_speed_smoothed": right_speed,
            "left_angular_speed_smoothed": left_angular,
            "right_angular_speed_smoothed": right_angular,
            "left_low_speed": left_low_speed,
            "right_low_speed": right_low_speed,
            "terminal_stability": terminal_stability,
            "terminal_impulse": terminal_impulse,
            "left_binary_open": left_binary_open,
            "right_binary_open": right_binary_open,
            "linear_motion_threshold_left": linear_motion_threshold_left,
            "linear_motion_threshold_right": linear_motion_threshold_right,
            "sustained_motion_runs": motion_runs,
            "left_speed_runs": left_speed_runs,
            "right_speed_runs": right_speed_runs,
            "left_angular_runs": left_angular_runs,
        },
    }
    return candidate_map, context
