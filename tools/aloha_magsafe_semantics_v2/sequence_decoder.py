"""Bidirectional, globally constrained semantic sequence decoder v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aloha_magsafe_semantics.candidate_detection import EventCandidate
from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS
from aloha_magsafe_semantics.io import canonical_json_hash, sha256_array
from aloha_magsafe_semantics.schema import EventRecord, SemanticTimeline
from aloha_magsafe_semantics.sequence_decoder import (
    _event_evidence,
    _phase_labels,
    _positive_signal_progress,
    _progress_between,
    validate_partial_order,
)


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


TARGETED_V2_EVENTS = {
    "phone_rotation_to_portrait_start",
    "phone_portrait_reached",
    "right_accessory_grasp_start",
    "accessory_detachment_start",
    "accessory_removed",
    "left_phone_release_complete",
    "right_accessory_release_complete",
    "left_arm_return_near_home",
    "task_end",
}


@dataclass(frozen=True)
class SequenceState:
    selections: dict[str, EventCandidate | None]
    score: float
    score_terms: dict[str, dict[str, float]]


def _broad_bounds(config: dict[str, Any], first: str, second: str) -> tuple[float, float | None]:
    value = config["duration_bounds_sec"].get(f"{first}->{second}")
    if value is None:
        return 0.0, None
    return float(value[0]), None if value[1] is None else float(value[1])


def _duration_score(config: dict[str, Any], first: str, second: str, duration: float, episode_duration: float) -> float:
    prior = config.get("duration_priors", {}).get(f"{first}->{second}")
    if not prior:
        return 0.5
    absolute_scale = max(float(prior["robust_scale_sec"]), float(config["decoder_v2"]["minimum_duration_prior_scale_sec"]))
    absolute = np.exp(-0.5 * ((duration - float(prior["median_sec"])) / absolute_scale) ** 2)
    normalized_duration = duration / max(episode_duration, 1e-9)
    normalized_scale = max(float(prior["robust_scale_normalized"]), float(config["decoder_v2"]["minimum_duration_prior_scale_normalized"]))
    normalized = np.exp(-0.5 * ((normalized_duration - float(prior["median_normalized"])) / normalized_scale) ** 2)
    return float(0.65 * absolute + 0.35 * normalized)


def _candidate_model_score(name: str, candidate: EventCandidate, config: dict[str, Any]) -> float:
    value = float(candidate.score)
    if name in TARGETED_V2_EVENTS and candidate.source.startswith("v2_"):
        value += float(config["decoder_v2"]["v2_evidence_model_bonus"])
    if candidate.source == "v1_generic_signal_seed_not_reference_timeline":
        value += float(config["decoder_v2"]["generic_seed_consistency_bonus"])
    return value


def _decode_subset(name: str, values: list[EventCandidate], config: dict[str, Any], limit: int) -> list[EventCandidate]:
    ordered = sorted(values, key=lambda row: (-_candidate_model_score(name, row, config), row.action_index))
    if name not in TARGETED_V2_EVENTS:
        return ordered[:limit]
    v2 = [row for row in ordered if row.source.startswith("v2_")]
    seeds = [row for row in ordered if row.source == "v1_generic_signal_seed_not_reference_timeline"]
    base = [row for row in ordered if not row.source.startswith("v2_") and row.source != "v1_generic_signal_seed_not_reference_timeline"]
    # A targeted evidence model must reach the decoder even when a broad v1
    # low-motion score creates many numerically high but semantically weak ties.
    priority_count = min(len(v2), max(1, limit // 2))
    seed_count = min(len(seeds), 1)
    selected = v2[:priority_count] + seeds[:seed_count] + base[: max(0, limit - priority_count - seed_count)]
    return sorted(selected, key=lambda row: (-_candidate_model_score(name, row, config), row.action_index))


def _transition_task_score(first: str, left: EventCandidate, second: str, right: EventCandidate, context: dict[str, Any]) -> float:
    if right.action_index <= left.action_index:
        return 0.0
    features = context["features"]
    start, end = left.action_index, right.action_index
    if (first, second) == ("phone_rotation_to_portrait_start", "phone_portrait_reached"):
        total = float(features["left_cumulative_rotation"][end] - features["left_cumulative_rotation"][start])
        return float(np.clip(total / 0.35, 0.0, 1.0))
    if (first, second) == ("accessory_detachment_start", "accessory_removed"):
        displacement = float(np.linalg.norm(features["right_tcp_position"][end] - features["right_tcp_position"][start]))
        return float(np.clip(displacement / 0.025, 0.0, 1.0))
    if (first, second) == ("phone_move_to_charger_start", "phone_charger_attachment_complete"):
        displacement = float(np.linalg.norm(features["left_tcp_position"][end] - features["left_tcp_position"][start]))
        return float(np.clip(displacement / 0.06, 0.0, 1.0))
    return 0.5


def _decode_core(
    candidates: dict[str, list[EventCandidate]],
    timestamps: np.ndarray,
    config: dict[str, Any],
    context: dict[str, Any],
) -> list[SequenceState]:
    settings = config["decoder_v2"]
    beam: list[SequenceState] = [SequenceState({}, 0.0, {})]
    episode_duration = float(timestamps[-1] - timestamps[0])
    width = int(settings["beam_width"])
    per_event_limit = int(settings["decode_candidates_per_event"])
    for event_position, name in enumerate(CORE_CHAIN):
        expanded: list[SequenceState] = []
        event_candidates = _decode_subset(name, candidates[name], config, per_event_limit)
        previous_name = CORE_CHAIN[event_position - 1] if event_position else None
        for state in beam:
            previous = None if previous_name is None else state.selections.get(previous_name)
            viable = []
            for candidate in event_candidates:
                if previous is not None:
                    duration = candidate.action_time_sec - previous.action_time_sec
                    minimum, maximum = _broad_bounds(config, previous_name, name)
                    if duration < minimum - 1e-9 or (maximum is not None and duration > maximum + 1e-9):
                        continue
                viable.append(candidate)
            if not viable:
                selections = dict(state.selections)
                selections[name] = None
                expanded.append(SequenceState(
                    selections,
                    state.score - float(settings["missing_event_penalty"]),
                    dict(state.score_terms, **{name: {"missing_penalty": -float(settings["missing_event_penalty"])}}),
                ))
                continue
            for candidate in viable:
                model_score = _candidate_model_score(name, candidate, config)
                duration_score = 0.5
                task_score = 0.5
                if previous is not None:
                    duration = candidate.action_time_sec - previous.action_time_sec
                    duration_score = _duration_score(config, previous_name, name, duration, episode_duration)
                    task_score = _transition_task_score(previous_name, previous, name, candidate, context)
                increment = (
                    float(settings["local_evidence_weight"]) * model_score
                    + float(settings["duration_prior_weight"]) * duration_score
                    + float(settings["phase_consistency_weight"]) * task_score
                )
                selections = dict(state.selections)
                selections[name] = candidate
                terms = dict(state.score_terms)
                terms[name] = {
                    "local_evidence": float(model_score),
                    "duration_prior": float(duration_score),
                    "phase_consistency": float(task_score),
                    "increment": float(increment),
                }
                expanded.append(SequenceState(selections, state.score + increment, terms))
        beam = sorted(
            expanded,
            key=lambda state: (
                -state.score,
                tuple(
                    len(timestamps) if state.selections.get(event) is None else state.selections[event].action_index
                    for event in CORE_CHAIN[: event_position + 1]
                ),
            ),
        )[:width]
    return beam


def _complete_sequences(
    core_states: list[SequenceState],
    candidates: dict[str, list[EventCandidate]],
    timestamps: np.ndarray,
    config: dict[str, Any],
) -> list[SequenceState]:
    settings = config["decoder_v2"]
    episode_duration = float(timestamps[-1] - timestamps[0])
    right_candidates = _decode_subset(
        "right_accessory_release_complete", candidates["right_accessory_release_complete"],
        config, int(settings["decode_candidates_per_event"]),
    )
    task_candidates = _decode_subset(
        "task_end", candidates["task_end"], config, int(settings["decode_candidates_per_event"]),
    )
    backward_terminal = [row for row in task_candidates if row.source == "v2_backward_terminal_suffix"]
    if backward_terminal:
        task_candidates = backward_terminal
    completed: list[SequenceState] = []
    for core in core_states[: int(settings["terminal_core_beam_count"])]:
        removed = core.selections.get("accessory_removed")
        left_return = core.selections.get("left_arm_return_near_home")
        viable_right = [row for row in right_candidates if removed is None or row.action_index > removed.action_index]
        if not viable_right:
            viable_right = [None]
        for right_release in viable_right:
            prerequisites = [row.action_index for row in (left_return, right_release) if row is not None]
            minimum_index = max(prerequisites, default=0)
            viable_task = [row for row in task_candidates if row.action_index >= minimum_index]
            if not viable_task:
                viable_task = [None]
            for task_end in viable_task:
                selections = dict(core.selections)
                selections["right_accessory_release_complete"] = right_release
                selections["task_end"] = task_end
                score = core.score
                terms = dict(core.score_terms)
                if right_release is None:
                    score -= float(settings["missing_event_penalty"])
                    terms["right_accessory_release_complete"] = {"missing_penalty": -float(settings["missing_event_penalty"])}
                else:
                    duration = right_release.action_time_sec - (removed.action_time_sec if removed is not None else timestamps[0])
                    duration_score = _duration_score(config, "accessory_removed", "right_accessory_release_complete", duration, episode_duration)
                    local = _candidate_model_score("right_accessory_release_complete", right_release, config)
                    increment = float(settings["local_evidence_weight"]) * local + float(settings["duration_prior_weight"]) * duration_score
                    score += increment
                    terms["right_accessory_release_complete"] = {"local_evidence": local, "duration_prior": duration_score, "increment": increment}
                if task_end is None:
                    score -= float(settings["missing_event_penalty"])
                    terms["task_end"] = {"missing_penalty": -float(settings["missing_event_penalty"])}
                else:
                    previous_time = max(
                        row.action_time_sec for row in (left_return, right_release) if row is not None
                    ) if any(row is not None for row in (left_return, right_release)) else float(timestamps[0])
                    duration = task_end.action_time_sec - previous_time
                    local = _candidate_model_score("task_end", task_end, config)
                    terminal_values = [
                        value for key, value in task_end.score_components.items()
                        if any(token in key for token in ("future_", "no_later", "terminal_"))
                    ]
                    terminal_bonus = float(np.mean(terminal_values)) if terminal_values else 0.0
                    duration_score = _duration_score(config, "terminal_prerequisite", "task_end", duration, episode_duration)
                    increment = (
                        float(settings["local_evidence_weight"]) * local
                        + float(settings["duration_prior_weight"]) * duration_score
                        + float(settings["backward_terminal_weight"]) * terminal_bonus
                    )
                    score += increment
                    terms["task_end"] = {
                        "local_evidence": local,
                        "duration_prior": duration_score,
                        "backward_terminal": terminal_bonus,
                        "increment": increment,
                    }
                completed.append(SequenceState(selections, score, terms))
    return sorted(
        completed,
        key=lambda state: (
            -state.score,
            tuple(
                len(timestamps) if state.selections.get(name) is None else state.selections[name].action_index
                for name in REQUIRED_EVENTS
            ),
        ),
    )[: max(int(settings["beam_width"]), int(settings["global_sequence_alternatives"]))]


def _event_sequence_margin(name: str, selected: SequenceState, sequences: list[SequenceState]) -> float:
    chosen = selected.selections.get(name)
    chosen_index = None if chosen is None else chosen.action_index
    alternatives = [
        state.score for state in sequences[1:]
        if (None if state.selections.get(name) is None else state.selections[name].action_index) != chosen_index
    ]
    return float(max(0.0, selected.score - max(alternatives))) if alternatives else float("inf")


def _confidence(
    name: str,
    candidate: EventCandidate | None,
    event_candidates: list[EventCandidate],
    sequence_margin: float,
    config: dict[str, Any],
    context: dict[str, Any],
) -> tuple[float, str, float | None, dict[str, float]]:
    if candidate is None:
        return 0.0, "AMBIGUOUS", None, {"resolved": 0.0}
    settings = config["confidence_v2"]
    other_scores = [row.score for row in event_candidates if row.action_index != candidate.action_index]
    local_margin = max(0.0, candidate.score - max(other_scores, default=0.0))
    finite_sequence_margin = float(settings["sequence_margin_saturation"]) if not np.isfinite(sequence_margin) else sequence_margin
    separation = float(np.clip(
        0.45 * local_margin / max(float(settings["local_margin_saturation"]), 1e-9)
        + 0.55 * finite_sequence_margin / max(float(settings["sequence_margin_saturation"]), 1e-9),
        0.0,
        1.0,
    ))
    values = sorted((float(value) for key, value in candidate.score_components.items() if "seed" not in key), reverse=True)
    evidence_breadth = float(np.mean(values[: min(3, len(values))])) if values else 0.0
    dwell_stability = float(max(
        (value for key, value in candidate.score_components.items() if any(token in key for token in ("dwell", "stability", "plateau", "no_later"))),
        default=evidence_breadth,
    ))
    confidence = float(np.clip(
        float(settings["local_score_weight"]) * candidate.score
        + float(settings["evidence_breadth_weight"]) * evidence_breadth
        + float(settings["sequence_separation_weight"]) * separation
        + float(settings["dwell_stability_weight"]) * dwell_stability,
        0.0,
        1.0,
    ))
    physical_evidence = True
    if name == "right_accessory_release_complete":
        physical_evidence = bool(context["v2_right_release"]["physical_release_evidence"])
    elif name == "left_phone_release_complete":
        physical_evidence = bool(context["v2_left_release"]["physical_release_evidence"])
    elif name == "phone_portrait_reached":
        physical_evidence = candidate.source.startswith("v2_rotation_") or candidate.score >= float(settings["portrait_fallback_min_score"])
    elif name == "task_end":
        physical_evidence = candidate.source == "v2_backward_terminal_suffix"
    if not physical_evidence:
        confidence = min(confidence, float(settings["ambiguous_cap"]))
        confidence_class = "AMBIGUOUS"
    elif confidence >= float(settings["high_threshold"]):
        confidence_class = "HIGH"
    elif confidence >= float(settings["medium_threshold"]):
        confidence_class = "MEDIUM"
    elif confidence >= float(settings["low_threshold"]):
        confidence_class = "LOW"
    else:
        confidence_class = "AMBIGUOUS"
    attribution = {
        "local_detector_score": float(candidate.score),
        "evidence_breadth": evidence_breadth,
        "dwell_stability": dwell_stability,
        "sequence_separation": separation,
        "local_candidate_margin": local_margin,
        "global_sequence_margin": finite_sequence_margin,
        "physical_evidence_gate": float(physical_evidence),
    }
    return confidence, confidence_class, local_margin, attribution


def _sequence_payload(state: SequenceState) -> dict[str, Any]:
    return {
        "score": float(state.score),
        "events": {
            name: None if state.selections.get(name) is None else int(state.selections[name].action_index)
            for name in REQUIRED_EVENTS
        },
        "score_terms": state.score_terms,
    }


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
    core_states = _decode_core(candidates, timestamps, detector_config, context)
    sequences = _complete_sequences(core_states, candidates, timestamps, detector_config)
    if not sequences:
        raise RuntimeError("v2 global decoder produced no sequence")
    selected = sequences[0]
    config_hash = canonical_json_hash(detector_config)
    trajectory_hash = sha256_array(action)
    fk_model_hash = str(fk_trajectory.get("model_sha256", "UNAVAILABLE"))
    geometry_hash = canonical_json_hash(task_geometry)
    events: dict[str, EventRecord] = {}
    alternative_count = int(detector_config["decoder_v2"]["alternative_count"])
    for name in REQUIRED_EVENTS:
        candidate = selected.selections.get(name)
        alternatives = sorted(candidates[name], key=lambda row: (-row.score, row.action_index))
        sequence_margin = _event_sequence_margin(name, selected, sequences)
        confidence, confidence_class, local_margin, attribution = _confidence(
            name, candidate, alternatives, sequence_margin, detector_config, context,
        )
        provenance = {
            "detector_config_hash": config_hash,
            "trajectory_hash": trajectory_hash,
            "FK_model_hash": fk_model_hash,
            "task_geometry_hash": geometry_hash,
            "reference_timeline_used_for_detection": False,
            "detector_version": 2,
        }
        if candidate is None:
            events[name] = EventRecord(
                event_name=name, action_index=None, action_time_sec=None,
                observed_frame=None, observed_time_sec=None, confidence=0.0,
                confidence_class="AMBIGUOUS", evidence={"reason": "no globally consistent candidate"},
                provenance=provenance,
                alternatives=[row.to_dict() for row in alternatives[:alternative_count]],
                score_difference=None,
            )
            continue
        evidence = _event_evidence(name, candidate, action, context, timestamps)
        evidence.update({
            "v2_confidence_attribution": attribution,
            "global_sequence_score": float(selected.score),
            "global_sequence_margin": None if not np.isfinite(sequence_margin) else float(sequence_margin),
        })
        observed_frame = None
        observed_time = None
        if observation_alignment and observation_alignment.get("action_to_observation_lag_frames") is not None:
            observed_frame = int(candidate.action_index + int(observation_alignment["action_to_observation_lag_frames"]))
            observed_time = float(candidate.action_time_sec + float(observation_alignment.get("latency_seconds", 0.0)))
        events[name] = EventRecord(
            event_name=name,
            action_index=int(candidate.action_index),
            action_time_sec=float(candidate.action_time_sec),
            observed_frame=observed_frame,
            observed_time_sec=observed_time,
            confidence=confidence,
            confidence_class=confidence_class,
            evidence=evidence,
            provenance=provenance,
            alternatives=[row.to_dict() for row in alternatives if row.action_index != candidate.action_index][:alternative_count],
            score_difference=local_margin,
        )

    terminal_candidate = context["v2_terminal"]["terminal_candidates"][0]
    if terminal_candidate.source == "v2_terminal_end_fallback":
        terminal_confidence_class = "AMBIGUOUS"
    elif terminal_candidate.score >= detector_config["confidence_v2"]["high_threshold"]:
        terminal_confidence_class = "HIGH"
    elif terminal_candidate.score >= detector_config["confidence_v2"]["medium_threshold"]:
        terminal_confidence_class = "MEDIUM"
    else:
        terminal_confidence_class = "LOW"
    events["terminal_hold_start"] = EventRecord(
        event_name="terminal_hold_start",
        action_index=int(terminal_candidate.action_index),
        action_time_sec=float(timestamps[terminal_candidate.action_index]),
        observed_frame=None,
        observed_time_sec=None,
        confidence=float(terminal_candidate.score),
        confidence_class=terminal_confidence_class,
        evidence={"detector_score_components": terminal_candidate.components, "candidate_source": terminal_candidate.source},
        provenance={
            "detector_config_hash": config_hash,
            "trajectory_hash": trajectory_hash,
            "FK_model_hash": fk_model_hash,
            "task_geometry_hash": geometry_hash,
            "reference_timeline_used_for_detection": False,
            "detector_version": 2,
        },
    )

    feature = context["features"]
    index = {name: record.action_index for name, record in events.items()}
    sample_arrays: dict[str, np.ndarray] = {
        "left_gripper_phase": context["left_gripper"].phase,
        "right_gripper_phase": context["right_gripper"].phase,
        "left_phase_progress": _progress_between(index["left_phone_grasp_start"], index["task_end"], feature["left_cumulative_path_length"], len(action)),
        "right_phase_progress": _progress_between(index["phone_portrait_reached"], index["task_end"], feature["right_cumulative_path_length"], len(action)),
        "phone_acquisition_progress": _progress_between(index["left_phone_grasp_start"], index["phone_portrait_reached"], feature["left_cumulative_path_length"] + feature["left_cumulative_rotation"], len(action)),
        "phone_rotation_progress": _progress_between(index["phone_rotation_to_portrait_start"], index["phone_portrait_reached"], feature["left_cumulative_rotation"], len(action)),
        "accessory_acquisition_progress": _progress_between(index["phone_portrait_reached"], index["right_accessory_grasp_start"], feature["right_cumulative_path_length"], len(action)),
        "accessory_removal_progress": _progress_between(index["accessory_detachment_start"], index["accessory_removed"], feature["right_cumulative_path_length"], len(action)),
        "phone_to_charger_progress": _progress_between(index["phone_move_to_charger_start"], index["phone_charger_attachment_complete"], feature["left_cumulative_path_length"], len(action)),
        "left_release_progress": _positive_signal_progress(index["phone_charger_attachment_complete"], index["left_phone_release_complete"], context["left_gripper"].normalized_open, len(action)),
        "right_release_progress": _positive_signal_progress(index["accessory_removed"], index["right_accessory_release_complete"], context["right_gripper"].normalized_open, len(action)),
        "phone_orientation_progress_envelope": context["v2_rotation"]["orientation_progress_monotonic_envelope"],
        "terminal_future_motion_energy": context["v2_terminal"]["future_motion_energy"],
        "terminal_future_motion_energy_rate": context["v2_terminal"]["future_motion_energy_rate"],
    }
    sample_arrays.update(_phase_labels(len(action), events))
    partial_order = validate_partial_order(events)
    top_count = int(detector_config["decoder_v2"]["global_sequence_alternatives"])
    return SemanticTimeline(
        trajectory_length=len(action),
        timestamps=timestamps,
        events=events,
        sample_arrays=sample_arrays,
        detector_config_hash=config_hash,
        trajectory_hash=trajectory_hash,
        fk_model_hash=fk_model_hash,
        task_geometry_hash=geometry_hash,
        source_type=source_type,
        metadata={
            "detector_version": 2,
            "decoder": "bidirectional_constrained_beam_with_backward_terminal_v2",
            "selected_sequence_score": float(selected.score),
            "second_best_sequence_score": float(sequences[1].score) if len(sequences) > 1 else None,
            "global_sequence_score_margin": float(selected.score - sequences[1].score) if len(sequences) > 1 else None,
            "top_globally_consistent_sequences": [_sequence_payload(state) for state in sequences[:top_count]],
            "partial_order": partial_order,
            "reference_timeline_loaded_during_detection": False,
            "observation_state_used_as_motion_source": False,
            "observation_state_supplied_for_diagnostics": observation_state is not None,
            "observation_alignment_applied_after_action_domain_detection": observation_alignment is not None,
            "v1_generic_seed_used": True,
            "v1_seed_is_approved_reference": False,
            "gripper_calibration": {
                "left": context["left_gripper"].calibration,
                "right": context["right_gripper"].calibration,
            },
            "rotation_segmentation": {
                "interval": context["v2_rotation"]["interval"],
                "runs": context["v2_rotation"]["runs"],
                "dominant_run": context["v2_rotation"]["dominant_run"],
                "threshold_radps": context["v2_rotation"]["threshold_radps"],
                "low_threshold_radps": context["v2_rotation"]["low_threshold_radps"],
            },
            "terminal_suffix": {
                "terminal_hold_start": int(context["v2_terminal"]["terminal_hold_start"]),
                "candidate_count": len(context["v2_terminal"]["terminal_candidates"]),
            },
            "release_physical_evidence": {
                "left": bool(context["v2_left_release"]["physical_release_evidence"]),
                "right": bool(context["v2_right_release"]["physical_release_evidence"]),
            },
        },
    )
