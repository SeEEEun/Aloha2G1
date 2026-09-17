"""Explicit non-physical semantic phase detector shared by ACT-A and ACT-B."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import CANONICAL_PHASES


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(value, dtype=np.float64)))))


def _p2p_detrended(value: np.ndarray) -> float:
    q = np.asarray(value, dtype=np.float64)
    alpha = np.linspace(0.0, 1.0, len(q), dtype=np.float64)[:, None]
    residual = q - (q[:1] + alpha * (q[-1:] - q[:1]))
    return float(np.max(np.ptp(residual, axis=0)))


def _behavior(candidate: np.ndarray, reference: np.ndarray, rule: Mapping[str, Any]) -> dict[str, Any]:
    predicted_motion = candidate - candidate[:1]
    target_motion = reference - reference[:1]
    target_rms = _rms(target_motion)
    target_energy = float(np.sum(target_motion**2))
    predicted_energy = float(np.sum(predicted_motion**2))
    cosine = (
        float(np.sum(predicted_motion * target_motion) / np.sqrt(target_energy * predicted_energy))
        if target_energy > 1e-12 and predicted_energy > 1e-12
        else None
    )
    progress = (
        float(np.sum(predicted_motion * target_motion) / target_energy)
        if target_energy > 1e-12
        else None
    )
    rmse = _rms(candidate - reference)
    p2p = _p2p_detrended(candidate)
    dynamic = target_rms >= float(rule["dynamic_group_threshold_rms_motion_rad"])
    success = (
        cosine is not None
        and progress is not None
        and cosine >= float(rule["dynamic_trajectory_motion_cosine_minimum"])
        and progress >= float(rule["dynamic_target_direction_progress_minimum"])
        if dynamic
        else rmse <= float(rule["static_full_segment_rmse_maximum_rad"])
        and p2p <= float(rule["static_detrended_peak_to_peak_maximum_rad"])
    )
    return {
        "success": bool(success),
        "dynamic": bool(dynamic),
        "target_motion_rms_rad": target_rms,
        "trajectory_motion_cosine": cosine,
        "target_direction_progress": progress,
        "full_segment_rmse_rad": rmse,
        "detrended_peak_to_peak_rad": p2p,
    }


def _selector_mask(
    selector: Mapping[str, Any],
    *,
    source_length: int,
    event_frames: Mapping[str, int],
    semantic_arrays: Mapping[str, Any],
) -> np.ndarray:
    if selector["kind"] == "frame_range":
        start = int(selector["start"])
        end = int(event_frames[selector["end_event_inclusive"]])
        mask = np.zeros(source_length, dtype=bool)
        mask[start : end + 1] = True
        return mask
    if selector["kind"] == "categorical":
        values = np.asarray(semantic_arrays[selector["array"]]).astype(str)
        if values.shape != (source_length,):
            raise ValueError(f"semantic source array has wrong shape: {selector['array']}")
        return np.isin(values, np.asarray(selector["values"]).astype(str))
    raise ValueError(f"unknown semantic selector: {selector}")


def semantic_phase_events(
    candidate_q_rad: Any,
    reference_q_rad: Any,
    *,
    event_frames: Mapping[str, int],
    semantic_arrays: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Detect semantic behavior completion without claiming physical contact."""

    candidate = np.asarray(candidate_q_rad, dtype=np.float64)
    reference = np.asarray(reference_q_rad, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape[1] != 28 or not np.isfinite(candidate).all():
        raise ValueError("candidate semantic trajectory must be finite [T,28]")
    if reference.ndim != 2 or reference.shape[1] != 28 or not np.isfinite(reference).all():
        raise ValueError("reference semantic trajectory must be finite [T,28]")
    source_length = len(reference)
    if len(candidate) > source_length:
        raise ValueError("candidate is longer than its source-aligned reference")
    groups = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in config["joint_groups_in_authoritative_28d_order"].items()
    }
    rule = config["common_behavior_rule"]
    diagnostics: dict[str, Any] = {}
    completed_events: dict[str, int | None] = {}
    for phase in CANONICAL_PHASES:
        phase_config = config["phase_contract"][phase]
        source_mask = _selector_mask(
            phase_config["selector"],
            source_length=source_length,
            event_frames=event_frames,
            semantic_arrays=semantic_arrays,
        )
        source_indices = np.flatnonzero(source_mask)
        if len(source_indices) < 4:
            raise RuntimeError(f"authoritative semantic interval is too short for {phase}")
        complete_interval = int(source_indices[-1]) < len(candidate)
        group_rows = {}
        if complete_interval:
            for group in phase_config["joint_groups"]:
                joint_indices = groups[group]
                group_rows[group] = _behavior(
                    candidate[source_indices][:, joint_indices],
                    reference[source_indices][:, joint_indices],
                    rule,
                )
        success = complete_interval and all(row["success"] for row in group_rows.values())
        completed_events[phase] = int(source_indices[-1]) if success else None
        diagnostics[phase] = {
            "source_interval_first_frame": int(source_indices[0]),
            "source_interval_last_frame": int(source_indices[-1]),
            "source_interval_frames": int(len(source_indices)),
            "candidate_contains_complete_interval": bool(complete_interval),
            "joint_groups": group_rows,
            "semantic_phase_completed": bool(success),
            "physical_contact_or_ownership_claimed": False,
        }
    return {
        "schema_version": "paper_semantic_phase_detection_v1",
        "status": "READY",
        "canonical_phase_events_frame": completed_events,
        "phase_diagnostics": diagnostics,
        "annotation_scope": "SOURCE_DERIVED_SEMANTIC_BEHAVIOR_NOT_MEASURED_OBJECT_CONTACT",
        "physical_contact_or_ownership_inferred": False,
        "policy_specific_logic": False,
    }
