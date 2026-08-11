"""Constrained beam decoder for a globally consistent MagSafe event sequence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .candidate_detection import EventCandidate, robust_unit
from .event_names import PARTIAL_ORDER_EDGES, REQUIRED_EVENTS
from .io import canonical_json_hash, sha256_array
from .schema import EventRecord, SemanticTimeline


CORE_CHAIN = (
    "left_phone_grasp_start",
    "phone_rotation_to_portrait_start",
    "phone_portrait_reached",
    "right_accessory_grasp_start",
    "accessory_detachment_start",
    "accessory_removed",
    "phone_move_to_charger_start",
    "phone_charger_attachment_complete",
    "left_phone_release_complete",
    "left_arm_return_near_home",
)


@dataclass
class BeamState:
    selections: dict[str, EventCandidate | None]
    score: float


def _duration_bounds(config: dict[str, Any], previous: str, current: str) -> tuple[float, float | None]:
    defaults: dict[tuple[str, str], tuple[float, float | None]] = {
        ("left_phone_grasp_start", "phone_rotation_to_portrait_start"): (0.05, 4.0),
        ("phone_rotation_to_portrait_start", "phone_portrait_reached"): (0.15, 4.0),
        ("phone_portrait_reached", "right_accessory_grasp_start"): (0.0, 10.0),
        ("right_accessory_grasp_start", "accessory_detachment_start"): (0.03, 3.0),
        ("accessory_detachment_start", "accessory_removed"): (0.1, 2.5),
        ("accessory_removed", "phone_move_to_charger_start"): (0.0, 8.0),
        ("phone_move_to_charger_start", "phone_charger_attachment_complete"): (0.3, 12.0),
        ("phone_charger_attachment_complete", "left_phone_release_complete"): (0.05, 8.0),
        ("left_phone_release_complete", "left_arm_return_near_home"): (0.0, 12.0),
    }
    configured = config.get("semantic_duration_bounds_sec", {}).get(f"{previous}->{current}")
    if configured is not None:
        return float(configured[0]), None if configured[1] is None else float(configured[1])
    return defaults[(previous, current)]


def _transition_bonus(
    previous_name: str,
    previous: EventCandidate,
    current_name: str,
    current: EventCandidate,
    context: dict[str, Any],
) -> float:
    features = context["features"]
    start, end = previous.action_index, current.action_index
    if end <= start:
        return 0.0
    bonus = 0.0
    if (previous_name, current_name) == ("phone_rotation_to_portrait_start", "phone_portrait_reached"):
        value = features["left_cumulative_rotation"][end] - features["left_cumulative_rotation"][start]
        scale = np.quantile(np.diff(features["left_cumulative_rotation"]), 0.9) * max(end - start, 1)
        return bonus + float(0.25 * np.clip(value / max(scale, 1e-9), 0.0, 1.0))
    if (previous_name, current_name) == ("accessory_detachment_start", "accessory_removed"):
        value = np.linalg.norm(features["right_tcp_position"][end] - features["right_tcp_position"][start])
        return bonus + float(0.25 * np.clip(value / 0.04, 0.0, 1.0))
    if (previous_name, current_name) == ("phone_move_to_charger_start", "phone_charger_attachment_complete"):
        value = np.linalg.norm(features["left_tcp_position"][end] - features["left_tcp_position"][start])
        return bonus + float(0.25 * np.clip(value / 0.08, 0.0, 1.0))
    return bonus


def _structural_first_bonus(event_name: str, candidate: EventCandidate, viable: list[EventCandidate]) -> float:
    component_by_event = {
        "left_phone_grasp_start": "gripper_close",
        "phone_rotation_to_portrait_start": "angular_motion_onset",
        "phone_portrait_reached": "angular_motion_endpoint",
        "right_accessory_grasp_start": "gripper_close",
        "accessory_detachment_start": "right_motion_onset",
        "accessory_removed": "right_motion_endpoint",
        "phone_move_to_charger_start": "left_motion_onset",
        "left_phone_release_complete": "gripper_open",
    }
    component_name = component_by_event.get(event_name)
    if component_name is None:
        return 0.0
    evidenced = [row for row in viable if row.score_components.get(component_name, 0.0) >= 0.30]
    if not evidenced:
        return 0.0
    earliest = min(evidenced, key=lambda row: row.action_index)
    if candidate.action_index == earliest.action_index:
        return 2.0
    elapsed = max(0.0, candidate.action_time_sec - earliest.action_time_sec)
    return float(0.15 * np.exp(-elapsed))


def _decode_core(
    candidates: dict[str, list[EventCandidate]],
    timestamps: np.ndarray,
    config: dict[str, Any],
    context: dict[str, Any],
) -> BeamState:
    beam = [BeamState({}, 0.0)]
    width = int(config["decoder"]["beam_width"])
    missing_penalty = float(config["decoder"]["missing_event_penalty"])
    for event_position, event_name in enumerate(CORE_CHAIN):
        expanded: list[BeamState] = []
        event_candidates = sorted(candidates[event_name], key=lambda row: (-row.score, row.action_index))
        for state in beam:
            previous_name = CORE_CHAIN[event_position - 1] if event_position else None
            previous = state.selections.get(previous_name) if previous_name else None
            viable: list[EventCandidate] = []
            for candidate in event_candidates:
                if previous is not None:
                    elapsed = candidate.action_time_sec - previous.action_time_sec
                    minimum, maximum = _duration_bounds(config, previous_name, event_name)
                    if elapsed < minimum - 1e-9 or (maximum is not None and elapsed > maximum + 1e-9):
                        continue
                viable.append(candidate)
            for candidate in viable:
                bonus = 0.0 if previous is None else _transition_bonus(previous_name, previous, event_name, candidate, context)
                bonus += _structural_first_bonus(event_name, candidate, viable)
                selections = dict(state.selections)
                selections[event_name] = candidate
                expanded.append(BeamState(selections, state.score + candidate.score + bonus))
            if not viable:
                selections = dict(state.selections)
                selections[event_name] = None
                expanded.append(BeamState(selections, state.score - missing_penalty))
        beam = sorted(expanded, key=lambda state: (-state.score, tuple(
            state.selections[name].action_index if state.selections.get(name) is not None else len(timestamps)
            for name in CORE_CHAIN[: event_position + 1]
        )))[:width]
    return beam[0]


def _select_after(
    event_name: str,
    candidates: dict[str, list[EventCandidate]],
    minimum_time: float,
    maximum_time: float | None = None,
) -> EventCandidate | None:
    viable = [row for row in candidates[event_name] if row.action_time_sec + 1e-9 >= minimum_time]
    if maximum_time is not None:
        viable = [row for row in viable if row.action_time_sec <= maximum_time + 1e-9]
    return max(viable, key=lambda row: (row.score, -row.action_index), default=None)


def _first_sustained(mask: np.ndarray, timestamps: np.ndarray, start_index: int, minimum_sec: float) -> int | None:
    mask = np.asarray(mask, dtype=bool)
    run_start: int | None = None
    for index in range(max(0, start_index), len(mask)):
        if mask[index] and run_start is None:
            run_start = index
        if (not mask[index] or index == len(mask) - 1) and run_start is not None:
            run_end = index if mask[index] else index - 1
            if timestamps[run_end] - timestamps[run_start] >= minimum_sec:
                return run_start
            run_start = None
    return None


def _confidence(candidate: EventCandidate | None, alternatives: list[EventCandidate], config: dict[str, Any]) -> tuple[float, str, float | None]:
    if candidate is None:
        return 0.0, "AMBIGUOUS", None
    other_scores = [row.score for row in alternatives if row.action_index != candidate.action_index]
    second = max(other_scores, default=0.0)
    margin = max(0.0, candidate.score - second)
    strong_primary_evidence = max(candidate.score_components.values(), default=0.0) >= 0.95
    confidence = float(np.clip(
        0.68 * candidate.score
        + 0.17 * min(1.0, margin / 0.25)
        + 0.10
        + (0.12 if strong_primary_evidence else 0.0),
        0.0,
        1.0,
    ))
    decoder = config["decoder"]
    if margin < float(decoder["ambiguity_score_margin"]) and candidate.score < float(decoder["confidence_medium"]):
        confidence_class = "AMBIGUOUS"
    elif confidence >= float(decoder["confidence_high"]):
        confidence_class = "HIGH"
    elif confidence >= float(decoder["confidence_medium"]):
        confidence_class = "MEDIUM"
    elif confidence >= float(decoder["confidence_low"]):
        confidence_class = "LOW"
    else:
        confidence_class = "AMBIGUOUS"
    return confidence, confidence_class, margin


def _event_evidence(
    event_name: str,
    candidate: EventCandidate,
    action: np.ndarray,
    context: dict[str, Any],
    timestamps: np.ndarray,
) -> dict[str, Any]:
    index = candidate.action_index
    before = max(0, index - 1)
    after = min(len(action) - 1, index + 1)
    side = "left" if event_name.startswith("left_") or event_name.startswith("phone_") else "right"
    gripper = context[f"{side}_gripper"]
    features = context["features"]
    direction_alignment = None
    if side == "left" and "left_charger_direction_velocity" in features:
        value = features["left_charger_direction_velocity"][index]
        speed = features["left_linear_speed"][index]
        direction_alignment = float(value / max(speed, 1e-12))
    semantic_distance = None
    if side == "left" and "left_phone_region_distance" in features:
        semantic_distance = float(features["left_phone_region_distance"][index])
    dwell_samples = 1
    current_phase = gripper.phase[index]
    while index + dwell_samples < len(action) and gripper.phase[index + dwell_samples] == current_phase:
        dwell_samples += 1
    dwell_end = min(len(action) - 1, index + dwell_samples - 1)
    return {
        "gripper_state_before": str(gripper.phase[before]),
        "gripper_state_after": str(gripper.phase[after]),
        "gripper_normalized_open_before": float(gripper.normalized_open[before]),
        "gripper_normalized_open_after": float(gripper.normalized_open[after]),
        "gripper_derivative": float(gripper.derivative[index]),
        "tcp_linear_speed": float(features[f"{side}_linear_speed"][index]),
        "tcp_angular_speed": float(features[f"{side}_angular_speed"][index]),
        "cumulative_displacement": float(features[f"{side}_cumulative_path_length"][index]),
        "cumulative_rotation": float(features[f"{side}_cumulative_rotation"][index]),
        "semantic_region_distance": semantic_distance,
        "direction_alignment": direction_alignment,
        "dwell_duration_sec": float(timestamps[dwell_end] - timestamps[index]),
        "detector_score_components": candidate.score_components,
        "candidate_source": candidate.source,
    }


def _progress_between(
    start: int | None,
    end: int | None,
    cumulative: np.ndarray,
    length: int,
) -> np.ndarray:
    progress = np.zeros(length, dtype=np.float64)
    if start is None or end is None or end <= start:
        return progress
    delta = np.asarray(cumulative[start : end + 1], dtype=np.float64) - float(cumulative[start])
    denominator = float(delta[-1])
    if denominator <= 1e-12:
        local = np.linspace(0.0, 1.0, end - start + 1)
    else:
        local = np.maximum.accumulate(np.clip(delta / denominator, 0.0, 1.0))
    progress[start : end + 1] = local
    progress[end + 1 :] = 1.0
    return progress


def _positive_signal_progress(
    start: int | None,
    end: int | None,
    signal: np.ndarray,
    length: int,
) -> np.ndarray:
    progress = np.zeros(length, dtype=np.float64)
    if start is None or end is None or end <= start:
        return progress
    local_signal = np.asarray(signal[start : end + 1], dtype=np.float64)
    positive_change = np.maximum(np.diff(local_signal, prepend=local_signal[0]), 0.0)
    cumulative = np.cumsum(positive_change)
    if cumulative[-1] <= 1e-12:
        local = np.linspace(0.0, 1.0, end - start + 1)
    else:
        local = np.clip(cumulative / cumulative[-1], 0.0, 1.0)
    progress[start : end + 1] = local
    progress[end + 1 :] = 1.0
    return progress


def _phase_labels(length: int, events: dict[str, EventRecord]) -> dict[str, np.ndarray]:
    left = np.full(length, "PRE_PHONE", dtype="U32")
    right = np.full(length, "PRE_ACCESSORY", dtype="U32")
    global_phase = np.full(length, "TASK_APPROACH", dtype="U40")

    def index(name: str, default: int) -> int:
        record = events.get(name)
        return default if record is None or record.action_index is None else int(record.action_index)

    lg = index("left_phone_grasp_start", length)
    rs = index("phone_rotation_to_portrait_start", length)
    pr = index("phone_portrait_reached", length)
    rg = index("right_accessory_grasp_start", length)
    ds = index("accessory_detachment_start", length)
    ar = index("accessory_removed", length)
    pm = index("phone_move_to_charger_start", length)
    pc = index("phone_charger_attachment_complete", length)
    lr = index("left_phone_release_complete", length)
    rr = index("right_accessory_release_complete", length)
    te = index("task_end", length)
    left[lg:rs] = "PHONE_ACQUISITION"
    left[rs:pr] = "PHONE_ROTATION"
    left[pr:pm] = "PORTRAIT_HOLD"
    left[pm:pc] = "PHONE_TRANSPORT"
    left[pc:lr] = "PHONE_ATTACHED_HOLD"
    left[lr:te] = "RELEASE_RETURN"
    left[te:] = "TERMINAL"
    right[pr:rg] = "ACCESSORY_APPROACH"
    right[rg:ds] = "ACCESSORY_ACQUISITION"
    right[ds:ar] = "ACCESSORY_REMOVAL"
    right[ar:rr] = "ACCESSORY_HOLD"
    right[rr:te] = "RELEASED"
    right[te:] = "TERMINAL"
    boundaries = sorted((
        (lg, "PHONE_ACQUISITION"), (rs, "PHONE_ROTATION"), (pr, "ACCESSORY_APPROACH"),
        (rg, "ACCESSORY_ACQUISITION"), (ds, "ACCESSORY_REMOVAL"), (ar, "POST_REMOVAL"),
        (pm, "PHONE_TRANSPORT"), (pc, "RELEASES_AND_RETURN"), (te, "TERMINAL"),
    ))
    for position, (start, name) in enumerate(boundaries):
        end = boundaries[position + 1][0] if position + 1 < len(boundaries) else length
        global_phase[start:end] = name
    return {"left_task_phase": left, "right_task_phase": right, "global_task_phase": global_phase}


def validate_partial_order(events: dict[str, EventRecord]) -> dict[str, Any]:
    violations = []
    for first, second, non_decreasing in PARTIAL_ORDER_EDGES:
        left = events[first].action_index
        right = events[second].action_index
        if left is None or right is None:
            continue
        valid = left <= right if non_decreasing else left < right
        if not valid:
            violations.append({"before": first, "after": second, "indices": [left, right]})
    return {"valid": not violations, "violations": violations}


def decode_semantic_sequence(
    action: np.ndarray,
    timestamps: np.ndarray,
    source_type: str,
    fk_trajectory: dict[str, Any],
    task_geometry: dict[str, Any],
    detector_config: dict[str, Any],
    candidates: dict[str, list[EventCandidate]],
    context: dict[str, Any],
    observation_state: np.ndarray | None = None,
    observation_alignment: dict[str, Any] | None = None,
) -> SemanticTimeline:
    action = np.asarray(action, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    core = _decode_core(candidates, timestamps, detector_config, context)
    selected = dict(core.selections)
    left_release = selected.get("left_phone_release_complete")
    return_components = context["score_components"]["left_arm_return_near_home"]
    return_mask = (return_components["return_proximity"] >= 0.8) & (return_components["left_low_speed"] >= 0.8)
    return_start = 0 if left_release is None else left_release.action_index
    return_index = _first_sustained(
        return_mask,
        timestamps,
        return_start,
        float(detector_config["kinematics"]["terminal_dwell_sec"]),
    )
    if return_index is not None:
        selected["left_arm_return_near_home"] = EventCandidate(
            event_name="left_arm_return_near_home",
            action_index=return_index,
            action_time_sec=float(timestamps[return_index]),
            score=float(np.clip(context["scores"]["left_arm_return_near_home"][return_index], 0.0, 1.0)),
            score_components={key: float(value[return_index]) for key, value in return_components.items()},
            source="earliest_sustained_return_proximity_and_low_speed_dwell",
        )
    removed = selected.get("accessory_removed")
    hold_minimum = float(detector_config["kinematics"].get("accessory_hold_before_release_sec", 0.5))
    minimum_release_time = timestamps[0] if removed is None else removed.action_time_sec + hold_minimum
    right_release = _select_after("right_accessory_release_complete", candidates, minimum_release_time)
    selected["right_accessory_release_complete"] = right_release
    left_return = selected.get("left_arm_return_near_home")
    prerequisite_times = [row.action_time_sec for row in (left_return, right_release) if row is not None]
    task_minimum = max(prerequisite_times, default=float(timestamps[0]))
    terminal_impulse = np.asarray(context["derived"]["terminal_impulse"], dtype=np.float64)
    terminal_start = int(np.argmax(terminal_impulse))
    feature = context["features"]
    suffix_spreads = []
    for side in ("left", "right"):
        position = np.asarray(feature[f"{side}_tcp_position"], dtype=np.float64)
        suffix_min = np.minimum.accumulate(position[::-1], axis=0)[::-1]
        suffix_max = np.maximum.accumulate(position[::-1], axis=0)[::-1]
        suffix_spreads.append(np.linalg.norm(suffix_max - suffix_min, axis=1))
    terminal_spread = np.maximum(suffix_spreads[0], suffix_spreads[1])
    spread_threshold = float(detector_config["kinematics"].get("terminal_position_spread_m", 0.01))
    stable_suffix_indices = np.flatnonzero(terminal_spread <= spread_threshold)
    if len(stable_suffix_indices):
        terminal_start = int(stable_suffix_indices[0])
    task_index = int(np.searchsorted(timestamps, task_minimum, side="left"))
    task_index = min(len(timestamps) - 1, max(terminal_start, task_index))
    task_score = float(np.clip(context["scores"]["task_end"][task_index], 0.0, 1.0))
    task_end = EventCandidate(
        event_name="task_end",
        action_index=task_index,
        action_time_sec=float(timestamps[task_index]),
        score=task_score,
        score_components={
            key: float(np.clip(value[task_index], 0.0, 1.0))
            for key, value in context["score_components"]["task_end"].items()
        },
        source="terminal_dwell_after_all_required_terminal_events",
    )
    selected["task_end"] = task_end

    config_hash = canonical_json_hash(detector_config)
    trajectory_hash = sha256_array(action)
    fk_model_hash = str(fk_trajectory.get("model_sha256", "UNAVAILABLE"))
    task_geometry_hash = canonical_json_hash(task_geometry)
    events: dict[str, EventRecord] = {}
    alternative_count = int(detector_config["decoder"]["alternative_count"])
    for name in REQUIRED_EVENTS:
        candidate = selected.get(name)
        alternatives = sorted(candidates[name], key=lambda row: (-row.score, row.action_index))
        confidence, confidence_class, margin = _confidence(candidate, alternatives, detector_config)
        if name == "right_accessory_release_complete" and candidate is not None:
            open_evidence = candidate.score_components.get("gripper_open", 0.0)
            if open_evidence < 0.35:
                confidence = min(confidence, 0.27)
                confidence_class = "AMBIGUOUS"
        if candidate is None:
            events[name] = EventRecord(
                event_name=name, action_index=None, action_time_sec=None,
                observed_frame=None, observed_time_sec=None, confidence=0.0,
                confidence_class="AMBIGUOUS",
                evidence={"reason": "no globally consistent candidate"},
                provenance={
                    "detector_config_hash": config_hash,
                    "trajectory_hash": trajectory_hash,
                    "FK_model_hash": fk_model_hash,
                    "task_geometry_hash": task_geometry_hash,
                    "reference_timeline_used_for_detection": False,
                },
                alternatives=[row.to_dict() for row in alternatives[:alternative_count]],
                score_difference=None,
            )
            continue
        observed_frame = None
        observed_time = None
        if observation_alignment:
            lag = observation_alignment.get("action_to_observation_lag_frames")
            if lag is not None:
                observed_frame = int(candidate.action_index + int(lag))
                observed_time = float(candidate.action_time_sec + float(observation_alignment.get("latency_seconds", 0.0)))
        events[name] = EventRecord(
            event_name=name,
            action_index=int(candidate.action_index),
            action_time_sec=float(candidate.action_time_sec),
            observed_frame=observed_frame,
            observed_time_sec=observed_time,
            confidence=confidence,
            confidence_class=confidence_class,
            evidence=_event_evidence(name, candidate, action, context, timestamps),
            provenance={
                "detector_config_hash": config_hash,
                "trajectory_hash": trajectory_hash,
                "FK_model_hash": fk_model_hash,
                "task_geometry_hash": task_geometry_hash,
                "reference_timeline_used_for_detection": False,
            },
            alternatives=[row.to_dict() for row in alternatives if row.action_index != candidate.action_index][:alternative_count],
            score_difference=margin,
        )

    feature = context["features"]
    event_index = {name: record.action_index for name, record in events.items()}
    sample_arrays: dict[str, np.ndarray] = {
        "left_gripper_phase": context["left_gripper"].phase,
        "right_gripper_phase": context["right_gripper"].phase,
        "left_phase_progress": _progress_between(event_index["left_phone_grasp_start"], event_index["task_end"], feature["left_cumulative_path_length"], len(action)),
        "right_phase_progress": _progress_between(event_index["phone_portrait_reached"], event_index["task_end"], feature["right_cumulative_path_length"], len(action)),
        "phone_acquisition_progress": _progress_between(event_index["left_phone_grasp_start"], event_index["phone_portrait_reached"], feature["left_cumulative_path_length"] + feature["left_cumulative_rotation"], len(action)),
        "phone_rotation_progress": _progress_between(event_index["phone_rotation_to_portrait_start"], event_index["phone_portrait_reached"], feature["left_cumulative_rotation"], len(action)),
        "accessory_acquisition_progress": _progress_between(event_index["phone_portrait_reached"], event_index["right_accessory_grasp_start"], feature["right_cumulative_path_length"], len(action)),
        "accessory_removal_progress": _progress_between(event_index["accessory_detachment_start"], event_index["accessory_removed"], feature["right_cumulative_path_length"], len(action)),
        "phone_to_charger_progress": _progress_between(event_index["phone_move_to_charger_start"], event_index["phone_charger_attachment_complete"], feature["left_cumulative_path_length"], len(action)),
        "left_release_progress": _positive_signal_progress(
            event_index["phone_charger_attachment_complete"], event_index["left_phone_release_complete"],
            context["left_gripper"].normalized_open, len(action),
        ),
        "right_release_progress": _positive_signal_progress(
            event_index["accessory_removed"], event_index["right_accessory_release_complete"],
            context["right_gripper"].normalized_open, len(action),
        ),
    }
    sample_arrays.update(_phase_labels(len(action), events))
    validation = validate_partial_order(events)
    timeline = SemanticTimeline(
        trajectory_length=len(action),
        timestamps=timestamps,
        events=events,
        sample_arrays=sample_arrays,
        detector_config_hash=config_hash,
        trajectory_hash=trajectory_hash,
        fk_model_hash=fk_model_hash,
        task_geometry_hash=task_geometry_hash,
        source_type=source_type,
        metadata={
            "decoder": "constrained_beam_search_v1",
            "beam_score": core.score,
            "partial_order": validation,
            "reference_timeline_loaded_during_detection": False,
            "observation_state_used_as_motion_source": False,
            "observation_state_supplied_for_diagnostics": observation_state is not None,
            "observation_alignment_applied_after_action_domain_detection": observation_alignment is not None,
            "gripper_calibration": {
                "left": context["left_gripper"].calibration,
                "right": context["right_gripper"].calibration,
            },
            "terminal_position_spread_threshold_m": spread_threshold,
            "terminal_position_spread_at_task_end_m": float(terminal_spread[events["task_end"].action_index]) if events["task_end"].action_index is not None else None,
        },
    )
    return timeline
