"""V2 candidate extraction layered on the frozen v1 signal detector."""
from __future__ import annotations

from typing import Any

import numpy as np

from aloha_magsafe_semantics.candidate_detection import (
    EventCandidate,
    extract_event_candidates as extract_v1_candidates,
)
from aloha_magsafe_semantics.detector import detect_magsafe_semantics as detect_v1
from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS

from .evidence import (
    EvidenceCandidate,
    release_candidates,
    right_removal_segmentation,
    rotation_segmentation,
    terminal_suffix_analysis,
)


def _convert(name: str, value: EvidenceCandidate, timestamps: np.ndarray) -> EventCandidate:
    return EventCandidate(
        event_name=name,
        action_index=int(value.action_index),
        action_time_sec=float(timestamps[value.action_index]),
        score=float(np.clip(value.score, 0.0, 1.0)),
        score_components={key: float(np.clip(component, 0.0, 1.0)) for key, component in value.components.items()},
        source=value.source,
    )


def _seed_candidate(name: str, record: Any, timestamps: np.ndarray) -> EventCandidate | None:
    if record.action_index is None:
        return None
    components = dict(record.evidence.get("detector_score_components", {}))
    components["v1_signal_seed_support"] = float(record.confidence)
    return EventCandidate(
        event_name=name,
        action_index=int(record.action_index),
        action_time_sec=float(timestamps[int(record.action_index)]),
        score=float(np.clip(0.18 + 0.72 * record.confidence, 0.0, 1.0)),
        score_components={key: float(np.clip(value, 0.0, 1.0)) for key, value in components.items()},
        source="v1_generic_signal_seed_not_reference_timeline",
    )


def _merge(values: list[EventCandidate], limit: int) -> list[EventCandidate]:
    by_index: dict[int, EventCandidate] = {}
    for row in values:
        old = by_index.get(row.action_index)
        if old is None or (row.score, row.source.startswith("v2_")) > (old.score, old.source.startswith("v2_")):
            by_index[row.action_index] = row
    # Keep both strong candidates and broad temporal coverage.  Coverage is
    # evidence-derived; it is not an episode-index prior.
    ordered = sorted(by_index.values(), key=lambda row: (-row.score, row.action_index))
    if len(ordered) <= limit:
        return sorted(ordered, key=lambda row: row.action_index)
    strong = ordered[: max(1, limit // 2)]
    remaining = [row for row in by_index.values() if row.action_index not in {value.action_index for value in strong}]
    bins = np.array_split(np.asarray(sorted(remaining, key=lambda row: row.action_index), dtype=object), max(1, limit - len(strong)))
    covered = [max(group.tolist(), key=lambda row: row.score) for group in bins if len(group)]
    return sorted((strong + covered)[:limit], key=lambda row: row.action_index)


def extract_event_candidates(
    action: np.ndarray,
    timestamps: np.ndarray,
    fk_trajectory: dict[str, Any],
    task_geometry: dict[str, Any],
    detector_config: dict[str, Any],
) -> tuple[dict[str, list[EventCandidate]], dict[str, Any]]:
    action = np.asarray(action, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    v1_config = detector_config["v1_base_config"]
    candidates, context = extract_v1_candidates(action, timestamps, fk_trajectory, task_geometry, v1_config)
    seed = detect_v1(action, timestamps, "v2_internal_signal_seed", fk_trajectory, task_geometry, v1_config)
    seed_index = {
        name: seed.event(name).action_index
        for name in REQUIRED_EVENTS
    }
    if any(value is None for value in seed_index.values()):
        missing = [name for name, value in seed_index.items() if value is None]
        raise RuntimeError(f"v1 signal seed has unresolved mandatory events: {missing}")
    seed_index = {name: int(value) for name, value in seed_index.items()}
    hold_left = np.isin(context["left_gripper"].phase, ("GRASP", "HOLD")).astype(np.float64)
    hold_right = np.isin(context["right_gripper"].phase, ("GRASP", "HOLD")).astype(np.float64)

    rotation = rotation_segmentation(
        timestamps,
        context["features"]["left_tcp_rotation"],
        context["features"]["left_angular_speed"],
        hold_left,
        seed_index["left_phone_grasp_start"],
        seed_index["right_accessory_grasp_start"],
        detector_config,
    )
    removal = right_removal_segmentation(
        timestamps,
        context["features"]["right_tcp_position"],
        context["features"]["right_linear_speed"],
        hold_right,
        seed_index["right_accessory_grasp_start"],
        seed_index["phone_move_to_charger_start"],
        detector_config,
    )
    # Couple the gripper transition to phase-relative approach completion and
    # imminent removal evidence.  This strengthens a real but non-zero-speed
    # hook acquisition without introducing an absolute event window.
    right_grasp_evidence: list[EventCandidate] = []
    approach_start = seed_index["phone_portrait_reached"]
    approach_end = seed_index["accessory_detachment_start"]
    cumulative_path = context["features"]["right_cumulative_path_length"]
    path_denominator = max(float(cumulative_path[approach_end] - cumulative_path[approach_start]), 1e-9)
    for row in candidates["right_accessory_grasp_start"]:
        if row.action_index < approach_start or row.action_index > approach_end:
            continue
        close = float(row.score_components.get("gripper_close", 0.0))
        if close < 0.25:
            continue
        index = row.action_index
        approach_progress = float(np.clip((cumulative_path[index] - cumulative_path[approach_start]) / path_denominator, 0.0, 1.0))
        future_end = min(len(action), int(np.searchsorted(timestamps, timestamps[index] + detector_config["release_v2"]["open_plateau_sec"], side="right")))
        hold_after = float(np.mean(hold_right[index:future_end])) if future_end > index else float(hold_right[index])
        removal_onset_support = float(np.exp(-max(0.0, timestamps[approach_end] - timestamps[index]) / 1.0))
        components = {
            "robust_right_gripper_close": close,
            "phase_relative_approach_progress": approach_progress,
            "post_transition_gripper_hold": hold_after,
            "imminent_removal_phase_support": removal_onset_support,
        }
        score = float(np.average(list(components.values()), weights=(0.34, 0.22, 0.24, 0.20)))
        right_grasp_evidence.append(EventCandidate(
            event_name="right_accessory_grasp_start",
            action_index=index,
            action_time_sec=float(timestamps[index]),
            score=score,
            score_components=components,
            source="v2_right_grasp_multimodal",
        ))
    left_release = release_candidates(
        "left", timestamps, context["left_gripper"], context["features"]["left_tcp_position"],
        seed_index["phone_charger_attachment_complete"], detector_config,
    )
    right_release = release_candidates(
        "right", timestamps, context["right_gripper"], context["features"]["right_tcp_position"],
        seed_index["accessory_removed"], detector_config,
    )
    prerequisite = max(seed_index["left_phone_release_complete"], seed_index["right_accessory_release_complete"])
    terminal = terminal_suffix_analysis(
        timestamps, context["features"], context["left_gripper"], context["right_gripper"],
        prerequisite, detector_config,
    )

    additions: dict[str, list[EventCandidate]] = {name: [] for name in REQUIRED_EVENTS}
    additions["phone_rotation_to_portrait_start"] = [_convert("phone_rotation_to_portrait_start", value, timestamps) for value in rotation["onset_candidates"]]
    additions["phone_portrait_reached"] = [_convert("phone_portrait_reached", value, timestamps) for value in rotation["plateau_candidates"]]
    additions["right_accessory_grasp_start"] = right_grasp_evidence
    additions["accessory_detachment_start"] = [_convert("accessory_detachment_start", value, timestamps) for value in removal["detachment_candidates"]]
    additions["accessory_removed"] = [_convert("accessory_removed", value, timestamps) for value in removal["removed_candidates"]]
    additions["left_phone_release_complete"] = [_convert("left_phone_release_complete", value, timestamps) for value in left_release["candidates"]]
    additions["right_accessory_release_complete"] = [_convert("right_accessory_release_complete", value, timestamps) for value in right_release["candidates"]]
    additions["left_arm_return_near_home"] = [_convert("left_arm_return_near_home", value, timestamps) for value in terminal["return_candidates"]]
    additions["task_end"] = [_convert("task_end", value, timestamps) for value in terminal["terminal_candidates"]]
    limit = int(detector_config["decoder_v2"]["candidates_per_event"])
    for name in REQUIRED_EVENTS:
        seed_row = _seed_candidate(name, seed.event(name), timestamps)
        merged = list(candidates.get(name, [])) + additions[name]
        if seed_row is not None:
            merged.append(seed_row)
        candidates[name] = _merge(merged, limit)

    context["v1_seed_timeline"] = seed
    context["v1_seed_indices"] = seed_index
    context["v2_rotation"] = rotation
    context["v2_removal"] = removal
    context["v2_left_release"] = left_release
    context["v2_right_release"] = right_release
    context["v2_terminal"] = terminal
    context["v2_reference_timeline_used"] = False
    return candidates, context
