"""Common quantitative metrics for retargeting, policies, and physical runs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import (
    CANONICAL_PHASES,
    FPS,
    PHYSICAL_SUCCESS,
    SEMANTIC_SUCCESS,
    SUCCESS_KINDS,
    na,
    validate_fps,
)


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and non-empty")
    return array


def rms(values: Any) -> float:
    array = _finite_array(values, "RMS input")
    return float(np.sqrt(np.mean(np.square(array))))


def distribution(values: Any, *, unit: str) -> dict[str, Any]:
    """Return the common scalar distribution summary, including true RMSE."""

    array = _finite_array(values, "distribution input").reshape(-1)
    q25, q75 = np.percentile(array, [25.0, 75.0])
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=0)),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "rmse": rms(array),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "count": int(array.size),
        "unit": unit,
    }


def evaluate_task_sequence(
    event_frames: Mapping[str, int] | None,
    *,
    success_kind: str,
    authoritative: bool,
) -> dict[str, Any]:
    """Score the longest correctly ordered prefix of the canonical sequence.

    Contact or ownership is never guessed here.  A caller must explicitly mark
    its event annotations authoritative for the evaluation mode.
    """

    if success_kind not in SUCCESS_KINDS:
        raise ValueError(f"invalid success kind: {success_kind}")
    if not authoritative:
        return na("authoritative phase/contact annotations were not supplied")
    if event_frames is None:
        return na("candidate phase events are absent")
    clean: dict[str, int] = {}
    for phase, frame in event_frames.items():
        if phase in CANONICAL_PHASES and frame is not None:
            integer = int(frame)
            if integer < 0:
                raise ValueError(f"negative event frame for {phase}: {integer}")
            clean[phase] = integer

    completed = 0
    previous = -1
    last_phase: str | None = None
    for phase in CANONICAL_PHASES:
        frame = clean.get(phase)
        if frame is None or frame <= previous:
            break
        completed += 1
        previous = frame
        last_phase = phase
    success = int(completed == len(CANONICAL_PHASES))
    result = {
        "status": "READY",
        "success_kind": success_kind,
        "completed_ordered_phases": completed,
        "total_phases": len(CANONICAL_PHASES),
        "phase_completion_score": completed / len(CANONICAL_PHASES),
        "last_successfully_completed_phase": last_phase,
        "canonical_phase_order": list(CANONICAL_PHASES),
        "event_frames": {phase: clean.get(phase) for phase in CANONICAL_PHASES},
        "task_sequence_success": success,
    }
    if success_kind == SEMANTIC_SUCCESS:
        result["SEMANTIC_TASK_SEQUENCE_SUCCESS"] = success
    else:
        result["PHYSICAL_TASK_SUCCESS"] = success
    return result


def phase_completion_summary(scores: Iterable[float]) -> dict[str, Any]:
    values = _finite_array(list(scores), "phase completion scores", ndim=1)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("phase completion scores must be in [0, 1]")
    result = distribution(values, unit="fraction")
    return {
        key: result[key]
        for key in ("mean", "median", "std", "iqr", "q25", "q75", "count", "unit")
    }


def _position_error(reference: Any, candidate: Any, name: str) -> np.ndarray:
    ref = _finite_array(reference, f"{name} reference", ndim=2)
    pred = _finite_array(candidate, f"{name} candidate", ndim=2)
    if ref.shape != pred.shape or ref.shape[1] != 3:
        raise ValueError(f"{name} positions must match [T,3], got {ref.shape}/{pred.shape}")
    return np.linalg.norm(pred - ref, axis=1) * 1000.0


def so3_geodesic_error_deg(reference: Any, candidate: Any) -> np.ndarray:
    ref = _finite_array(reference, "orientation reference", ndim=3)
    pred = _finite_array(candidate, "orientation candidate", ndim=3)
    if ref.shape != pred.shape or ref.shape[1:] != (3, 3):
        raise ValueError(f"rotations must match [T,3,3], got {ref.shape}/{pred.shape}")
    relative = np.einsum("tji,tjk->tik", ref, pred)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def wrist_metrics(
    reference_positions: Mapping[str, Any],
    candidate_positions: Mapping[str, Any],
    *,
    reference_orientations: Mapping[str, Any] | None = None,
    candidate_orientations: Mapping[str, Any] | None = None,
    orientation_reference_authoritative: bool = False,
) -> dict[str, Any]:
    errors = {
        side: _position_error(reference_positions[side], candidate_positions[side], f"{side} wrist")
        for side in ("left", "right")
    }
    result: dict[str, Any] = {
        side: {"position_error_mm": distribution(errors[side], unit="mm")}
        for side in ("left", "right")
    }
    result["combined"] = {
        "position_error_mm": distribution(
            np.concatenate((errors["left"], errors["right"])), unit="mm"
        )
    }
    if not orientation_reference_authoritative:
        result["orientation_error_deg"] = na("no authoritative wrist orientation reference")
    elif reference_orientations is None or candidate_orientations is None:
        result["orientation_error_deg"] = na("authoritative orientation arrays are incomplete")
    else:
        orientation_errors = {
            side: so3_geodesic_error_deg(
                reference_orientations[side], candidate_orientations[side]
            )
            for side in ("left", "right")
        }
        result["orientation_error_deg"] = {
            side: distribution(orientation_errors[side], unit="deg")
            for side in ("left", "right")
        }
        result["orientation_error_deg"]["combined"] = distribution(
            np.concatenate((orientation_errors["left"], orientation_errors["right"])),
            unit="deg",
        )
    return result


def whole_hand_metrics(
    reference_positions: Mapping[str, Any],
    candidate_positions: Mapping[str, Any],
    *,
    authoritative_definition: bool,
    definition_provenance: str | None,
) -> dict[str, Any]:
    if not authoritative_definition or not definition_provenance:
        return na("authoritative project whole-hand frame definition/provenance is required")
    errors = {
        side: _position_error(
            reference_positions[side], candidate_positions[side], f"{side} whole-hand"
        )
        for side in ("left", "right")
    }
    return {
        "definition_provenance": definition_provenance,
        "left": {"position_error_mm": distribution(errors["left"], unit="mm")},
        "right": {"position_error_mm": distribution(errors["right"], unit="mm")},
        "combined": {
            "position_error_mm": distribution(
                np.concatenate((errors["left"], errors["right"])), unit="mm"
            )
        },
    }


def bimanual_relation_metrics(
    reference_positions: Mapping[str, Any], candidate_positions: Mapping[str, Any]
) -> dict[str, Any]:
    reference_left = _finite_array(reference_positions["left"], "left relation reference", 2)
    reference_right = _finite_array(reference_positions["right"], "right relation reference", 2)
    candidate_left = _finite_array(candidate_positions["left"], "left relation candidate", 2)
    candidate_right = _finite_array(candidate_positions["right"], "right relation candidate", 2)
    shapes = {value.shape for value in (reference_left, reference_right, candidate_left, candidate_right)}
    if len(shapes) != 1 or reference_left.shape[1] != 3:
        raise ValueError(f"bimanual arrays must share [T,3], got {sorted(shapes)}")
    reference_delta = reference_right - reference_left
    candidate_delta = candidate_right - candidate_left
    error_mm = np.linalg.norm(candidate_delta - reference_delta, axis=1) * 1000.0
    return {
        "definition": "d(t) = p_R(t) - p_L(t)",
        "error_mm": distribution(error_mm, unit="mm"),
    }


def handoff_ordering_metrics(
    right_acquire_frame: int | None,
    left_release_frame: int | None,
    *,
    fps: float = FPS,
    authoritative: bool,
) -> dict[str, Any]:
    fps = validate_fps(fps)
    if not authoritative:
        return na("authoritative ownership/hand-phase annotation was not supplied")
    if right_acquire_frame is None or left_release_frame is None:
        return na("RIGHT_ACQUIRE and LEFT_RELEASE must both be present")
    acquire = int(right_acquire_frame)
    release = int(left_release_frame)
    if min(acquire, release) < 0:
        raise ValueError("handoff event frames cannot be negative")
    margin = release - acquire
    return {
        "status": "READY",
        "RIGHT_ACQUIRE_frame": acquire,
        "LEFT_RELEASE_frame": release,
        "score": int(margin > 0),
        "acquisition_to_release_margin_frames": margin,
        "acquisition_to_release_margin_seconds": margin / fps,
        "negative_margin": int(margin < 0),
        "units": {"margin_frames": "frames", "margin_seconds": "s"},
    }


def aggregate_handoff_ordering(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("status") == "READY"]
    if not eligible:
        return na("no episodes have authoritative handoff ordering annotations")
    margins = np.asarray(
        [row["acquisition_to_release_margin_frames"] for row in eligible], dtype=np.float64
    )
    scores = np.asarray([row["score"] for row in eligible], dtype=np.float64)
    return {
        "HOA": float(np.mean(scores)),
        "correct_episodes": int(np.sum(scores)),
        "evaluated_episodes": int(len(scores)),
        "negative_margin_count": int(np.count_nonzero(margins < 0.0)),
        "margin_frames": distribution(margins, unit="frames"),
        "margin_seconds": distribution(margins / FPS, unit="s"),
    }


def action_prediction_metrics(
    reference: Any,
    candidate: Any,
    joint_ranges_rad: Any,
    *,
    valid_lengths: Any | None = None,
) -> dict[str, Any]:
    truth = _finite_array(reference, "reference actions")
    pred = _finite_array(candidate, "candidate actions")
    ranges = _finite_array(joint_ranges_rad, "joint ranges", ndim=1)
    if truth.shape != pred.shape or truth.ndim not in (2, 3) or truth.shape[-1] != 28:
        raise ValueError(f"actions must match [T,28] or [N,K,28], got {truth.shape}/{pred.shape}")
    if ranges.shape != (28,) or np.any(ranges <= 0.0):
        raise ValueError("joint ranges must be positive [28]")
    delta = pred - truth
    if truth.ndim == 2:
        if valid_lengths is not None:
            raise ValueError("valid_lengths is only defined for [N,K,28] chunks")
        valid_delta = delta
        first = delta[:1]
        first4 = delta[: min(4, len(delta))]
    else:
        if valid_lengths is None:
            lengths = np.full(truth.shape[0], truth.shape[1], dtype=np.int64)
        else:
            lengths = np.asarray(valid_lengths, dtype=np.int64).reshape(-1)
            if lengths.shape != (truth.shape[0],) or np.any(lengths < 1) or np.any(
                lengths > truth.shape[1]
            ):
                raise ValueError("valid chunk lengths must be [N] within 1..K")
        valid_delta = np.concatenate(
            [delta[index, : int(length)] for index, length in enumerate(lengths)], axis=0
        )
        first = delta[:, 0]
        first4 = np.concatenate(
            [delta[index, : min(4, int(length))] for index, length in enumerate(lengths)],
            axis=0,
        )
    normalized = valid_delta / ranges
    return {
        "OVERALL_28D_RMSE_rad": rms(valid_delta),
        "ARM_14D_RMSE_rad": rms(valid_delta[..., :14]),
        "DEX3_14D_RMSE_rad": rms(valid_delta[..., 14:]),
        "FIRST_ACTION_RMSE_rad": rms(first),
        "FIRST4_RMSE_rad": rms(first4),
        "FULL_CHUNK_RMSE_rad": rms(valid_delta),
        "NRMSE_percent": {
            "overall_28d": 100.0 * rms(normalized),
            "arm_14d": 100.0 * rms(normalized[..., :14]),
            "dex3_14d": 100.0 * rms(normalized[..., 14:]),
            "definition": "jointwise error divided by authoritative joint range, then RMS * 100",
        },
        "units": {"absolute": "rad", "normalized": "percent of joint range"},
        "valid_joint_frames": int(valid_delta.shape[0] * 28),
    }


def bimanual_path_length_m(positions: Mapping[str, Any]) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for side in ("left", "right"):
        value = _finite_array(positions[side], f"{side} path", ndim=2)
        if value.shape[1] != 3:
            raise ValueError(f"{side} path must be [T,3]")
        lengths[side] = float(np.sum(np.linalg.norm(np.diff(value, axis=0), axis=1)))
    lengths["combined"] = lengths["left"] + lengths["right"]
    return lengths


def path_efficiency_metrics(
    reference_positions: Mapping[str, Any],
    candidate_positions: Mapping[str, Any],
    *,
    success: int | bool | None,
    success_kind: str,
) -> dict[str, Any]:
    if success_kind not in SUCCESS_KINDS:
        raise ValueError(f"RPL/SWPE requires explicit semantic or physical success, got {success_kind}")
    if success is None:
        return na("SWPE requires an explicitly scored success value")
    reference = bimanual_path_length_m(reference_positions)
    candidate = bimanual_path_length_m(candidate_positions)
    if reference["combined"] <= 1e-12:
        return na("reference combined bimanual path length is zero")
    rpl = candidate["combined"] / reference["combined"]
    swpe = int(bool(success)) * reference["combined"] / max(
        candidate["combined"], reference["combined"]
    )
    return {
        "RPL": float(rpl),
        "SWPE": float(swpe),
        "success_kind": success_kind,
        "success_value": int(bool(success)),
        "reference_path_length_m": reference,
        "predicted_path_length_m": candidate,
        "definition": "combined path = L_left + L_right; this metric is not called SPL",
    }


def _direction_reversal_count(velocity: np.ndarray, deadband: float) -> tuple[int, np.ndarray]:
    counts = np.zeros(velocity.shape[1], dtype=np.int64)
    for joint in range(velocity.shape[1]):
        signs = np.sign(velocity[:, joint])
        signs[np.abs(velocity[:, joint]) <= deadband] = 0.0
        nonzero = signs[signs != 0.0]
        counts[joint] = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    return int(np.sum(counts)), counts


def smoothness_metrics(
    q_rad: Any,
    *,
    fps: float = FPS,
    reversal_deadband_rad_s: float = 0.002,
    low_motion_velocity_rad_s: float = 0.01,
) -> dict[str, Any]:
    fps = validate_fps(fps)
    q = _finite_array(q_rad, "joint trajectory", ndim=2)
    if q.shape[1] != 28 or len(q) < 4:
        raise ValueError(f"smoothness requires at least four 28-D frames, got {q.shape}")
    qdot = np.diff(q, axis=0) * fps
    qddot = np.diff(qdot, axis=0) * fps
    jerk = np.diff(qddot, axis=0) * fps
    reversals, per_joint_reversals = _direction_reversal_count(qdot, reversal_deadband_rad_s)
    duration_s = (len(q) - 1) / fps
    reversal_rate = reversals / (duration_s * q.shape[1])

    net_velocity = np.abs(q[-1] - q[0]) / duration_s
    low_motion_mask = net_velocity <= low_motion_velocity_rad_s
    time_fraction = np.linspace(0.0, 1.0, len(q), dtype=np.float64)[:, None]
    trend = q[:1] + time_fraction * (q[-1:] - q[:1])
    detrended_p2p = np.ptp(q - trend, axis=0)
    low_motion_values = detrended_p2p[low_motion_mask]
    low_motion_summary: dict[str, Any]
    if low_motion_values.size:
        low_motion_summary = distribution(low_motion_values, unit="rad")
        low_motion_summary["eligible_joint_count"] = int(low_motion_values.size)
    else:
        low_motion_summary = na("no joints met the fixed low-motion criterion")
    return {
        "JOINT_JERK_RMS_rad_s3": rms(jerk),
        "DIRECTION_REVERSAL_RATE_per_joint_s": float(reversal_rate),
        "direction_reversal_count": reversals,
        "direction_reversal_count_per_joint": per_joint_reversals.tolist(),
        "qdot_RMS_rad_s": rms(qdot),
        "qddot_RMS_rad_s2": rms(qddot),
        "peak_to_peak_low_motion_oscillation": low_motion_summary,
        "fps": fps,
        "reversal_deadband_rad_s": float(reversal_deadband_rad_s),
        "low_motion_velocity_rad_s": float(low_motion_velocity_rad_s),
    }


def chunk_smoothness_metrics(
    q_chunk_rad: Any,
    *,
    valid_lengths: Any | None = None,
    fps: float = FPS,
    reversal_deadband_rad_s: float = 0.002,
    low_motion_velocity_rad_s: float = 0.01,
) -> dict[str, Any]:
    """Score raw chunks without introducing derivatives across query boundaries."""

    chunks = _finite_array(q_chunk_rad, "joint chunks", ndim=3)
    if chunks.shape[2] != 28 or chunks.shape[1] < 4:
        raise ValueError(f"chunk smoothness requires [N,K>=4,28], got {chunks.shape}")
    if valid_lengths is None:
        lengths = np.full(chunks.shape[0], chunks.shape[1], dtype=np.int64)
    else:
        lengths = np.asarray(valid_lengths, dtype=np.int64).reshape(-1)
        if lengths.shape != (chunks.shape[0],) or np.any(lengths < 1) or np.any(
            lengths > chunks.shape[1]
        ):
            raise ValueError("smoothness valid lengths must be [N] within 1..K")
    eligible_indices = np.flatnonzero(lengths >= 4)
    if not len(eligible_indices):
        return na("no query chunk has the four valid frames required for jerk")
    fps = validate_fps(fps)
    qdot_parts = []
    qddot_parts = []
    jerk_parts = []
    reversals = 0
    reversal_duration = 0.0
    low_motion_values = []
    for index in eligible_indices:
        length = lengths[index]
        q = chunks[index, : int(length)]
        qdot = np.diff(q, axis=0) * fps
        qddot = np.diff(qdot, axis=0) * fps
        jerk = np.diff(qddot, axis=0) * fps
        qdot_parts.append(qdot)
        qddot_parts.append(qddot)
        jerk_parts.append(jerk)
        count, _ = _direction_reversal_count(qdot, reversal_deadband_rad_s)
        reversals += count
        duration = (len(q) - 1) / fps
        reversal_duration += duration
        net_velocity = np.abs(q[-1] - q[0]) / duration
        eligible = net_velocity <= low_motion_velocity_rad_s
        alpha = np.linspace(0.0, 1.0, len(q), dtype=np.float64)[:, None]
        detrended = np.ptp(q - (q[:1] + alpha * (q[-1:] - q[:1])), axis=0)
        low_motion_values.extend(detrended[eligible].tolist())
    low_motion = (
        distribution(low_motion_values, unit="rad")
        if low_motion_values
        else na("no joints met the fixed low-motion criterion")
    )
    return {
        "JOINT_JERK_RMS_rad_s3": rms(np.concatenate(jerk_parts, axis=0)),
        "DIRECTION_REVERSAL_RATE_per_joint_s": float(
            reversals / (reversal_duration * chunks.shape[2])
        ),
        "direction_reversal_count": int(reversals),
        "qdot_RMS_rad_s": rms(np.concatenate(qdot_parts, axis=0)),
        "qddot_RMS_rad_s2": rms(np.concatenate(qddot_parts, axis=0)),
        "peak_to_peak_low_motion_oscillation": low_motion,
        "chunk_count": int(chunks.shape[0]),
        "smoothness_eligible_chunk_count": int(len(eligible_indices)),
        "short_chunk_count": int(chunks.shape[0] - len(eligible_indices)),
        "valid_frames": int(np.sum(lengths)),
        "fps": fps,
        "derivatives_cross_chunk_boundaries": False,
        "reversal_deadband_rad_s": float(reversal_deadband_rad_s),
        "low_motion_velocity_rad_s": float(low_motion_velocity_rad_s),
    }


def feasibility_metrics(
    diagnostics: Mapping[str, Any] | None,
    projection_magnitudes_m: Any | None = None,
) -> dict[str, Any]:
    if diagnostics is None:
        return na("validated project feasibility diagnostics were not supplied")
    required = (
        "hard_ik_failure_count",
        "hard_collision_count",
        "joint_limit_failure_count",
        "branch_discontinuity_count",
    )
    missing = [key for key in required if key not in diagnostics]
    if missing:
        return na(f"validated feasibility fields missing: {missing}")
    counts = {key: int(diagnostics[key]) for key in required}
    if any(value < 0 for value in counts.values()):
        raise ValueError("feasibility counts cannot be negative")
    projection: dict[str, Any]
    if projection_magnitudes_m is None:
        projection = na("per-frame feasibility projection magnitudes unavailable")
    else:
        projection_mm = _finite_array(
            projection_magnitudes_m, "projection magnitudes"
        ).reshape(-1) * 1000.0
        if np.any(projection_mm < -1e-12):
            raise ValueError("projection magnitudes cannot be negative")
        projection = distribution(np.maximum(projection_mm, 0.0), unit="mm")
    hard_collision_count = counts["hard_collision_count"]
    result = {
        "FEASIBLE": int(all(value == 0 for value in counts.values())),
        "hard_ik_failure_count": counts["hard_ik_failure_count"],
        "hard_collision_count": hard_collision_count,
        "hard_collision_episode": int(hard_collision_count > 0),
        "joint_limit_failure_count": counts["joint_limit_failure_count"],
        "branch_discontinuity_count": counts["branch_discontinuity_count"],
        "minimum_clearance_m": diagnostics.get("minimum_clearance_m"),
        "projection_magnitude_mm": projection,
        "diagnostics_provenance": diagnostics.get("provenance"),
        "collision_definition_recomputed": False,
    }
    return result


def feasibility_rate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("FEASIBLE") in (0, 1)]
    if not eligible:
        return na("no validated episode feasibility diagnostics")
    values = np.asarray([row["FEASIBLE"] for row in eligible], dtype=np.float64)
    return {
        "FEASIBILITY_RATE": float(np.mean(values)),
        "feasible_episodes": int(np.sum(values)),
        "evaluated_episodes": int(len(values)),
        "hard_collision_count": int(sum(row["hard_collision_count"] for row in eligible)),
        "hard_collision_episodes": int(sum(row["hard_collision_episode"] for row in eligible)),
    }


def aggregate_episode_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return named dataset-level rates while retaining episode-level records."""

    episodes = list(rows)
    sequences = [
        row.get("metrics", {}).get("task_sequence")
        for row in episodes
        if isinstance(row.get("metrics", {}).get("task_sequence"), Mapping)
        and row["metrics"]["task_sequence"].get("status") == "READY"
    ]
    if sequences:
        kinds = {str(row["success_kind"]) for row in sequences}
        if len(kinds) != 1:
            raise ValueError(f"aggregate mixes semantic and physical success: {sorted(kinds)}")
        scores = [float(row["phase_completion_score"]) for row in sequences]
        successes = [int(row["task_sequence_success"]) for row in sequences]
        sequence_aggregate: dict[str, Any] = {
            "status": "READY",
            "success_kind": next(iter(kinds)),
            "TASK_SEQUENCE_SUCCESS_RATE": float(np.mean(successes)),
            "successful_episodes": int(sum(successes)),
            "evaluated_episodes": len(sequences),
            "phase_completion_score": phase_completion_summary(scores),
            "last_successfully_completed_phase_counts": {
                phase: int(
                    sum(row.get("last_successfully_completed_phase") == phase for row in sequences)
                )
                for phase in CANONICAL_PHASES
            },
            "no_phase_completed_count": int(
                sum(row.get("last_successfully_completed_phase") is None for row in sequences)
            ),
        }
        label = (
            "SEMANTIC_TASK_SEQUENCE_SUCCESS_RATE"
            if next(iter(kinds)) == SEMANTIC_SUCCESS
            else "PHYSICAL_TASK_SUCCESS_RATE"
        )
        sequence_aggregate[label] = sequence_aggregate["TASK_SEQUENCE_SUCCESS_RATE"]
    else:
        sequence_aggregate = na("no episodes have authoritative ordered phase annotations")
    handoffs = [row.get("metrics", {}).get("handoff_ordering", {}) for row in episodes]
    feasibility_rows = [row.get("metrics", {}).get("feasibility", {}) for row in episodes]
    return {
        "episode_count": len(episodes),
        "task_sequence": sequence_aggregate,
        "handoff_ordering": aggregate_handoff_ordering(handoffs),
        "feasibility": feasibility_rate(feasibility_rows),
    }


def _array_pair(arrays: Mapping[str, Any], prefix: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    reference = {}
    candidate = {}
    for side in ("left", "right"):
        ref_key = f"reference_{side}_{prefix}"
        candidate_key = f"candidate_{side}_{prefix}"
        if ref_key not in arrays or candidate_key not in arrays:
            return None
        reference[side] = arrays[ref_key]
        candidate[side] = arrays[candidate_key]
    return reference, candidate


def evaluate_episode(
    episode: Mapping[str, Any],
    *,
    joint_ranges_rad: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one standardized episode without policy- or simulator-specific logic."""

    source_episode_id = str(episode.get("source_episode_id", ""))
    if not source_episode_id:
        raise ValueError("source_episode_id is required")
    fps = validate_fps(float(episode.get("fps", FPS)))
    arrays = dict(episode.get("arrays", {}))
    annotations = dict(episode.get("annotations", {}))
    provenance = dict(episode.get("provenance", {}))
    success_kind = episode.get("success_kind")
    metrics: dict[str, Any] = {}

    if success_kind is None:
        metrics["task_sequence"] = na("evaluation mode has no explicit semantic/physical success kind")
    else:
        metrics["task_sequence"] = evaluate_task_sequence(
            annotations.get("candidate_phase_events_frame"),
            success_kind=str(success_kind),
            authoritative=bool(annotations.get("phase_events_authoritative", False)),
        )

    wrist_pair = _array_pair(arrays, "wrist_position_m")
    if wrist_pair is None:
        metrics["wrist"] = na("reference/candidate bilateral wrist positions unavailable")
    else:
        wrist_rotation_pair = _array_pair(arrays, "wrist_rotation")
        metrics["wrist"] = wrist_metrics(
            *wrist_pair,
            reference_orientations=wrist_rotation_pair[0] if wrist_rotation_pair else None,
            candidate_orientations=wrist_rotation_pair[1] if wrist_rotation_pair else None,
            orientation_reference_authoritative=bool(
                provenance.get("wrist_orientation_reference_authoritative", False)
            ),
        )

    hand_pair = _array_pair(arrays, "whole_hand_position_m")
    if hand_pair is None:
        metrics["whole_hand"] = na("reference/candidate bilateral whole-hand positions unavailable")
        metrics["bimanual_relation"] = na("whole-hand arrays required for bimanual relation")
    else:
        metrics["whole_hand"] = whole_hand_metrics(
            *hand_pair,
            authoritative_definition=bool(provenance.get("whole_hand_frame_authoritative", False)),
            definition_provenance=provenance.get("whole_hand_frame_definition"),
        )
        if isinstance(metrics["whole_hand"], Mapping) and metrics["whole_hand"].get("status") == "NA":
            metrics["bimanual_relation"] = na("authoritative whole-hand definition required")
        else:
            metrics["bimanual_relation"] = bimanual_relation_metrics(*hand_pair)

    metrics["handoff_ordering"] = handoff_ordering_metrics(
        annotations.get("right_acquire_frame"),
        annotations.get("left_release_frame"),
        fps=fps,
        authoritative=bool(annotations.get("handoff_events_authoritative", False)),
    )

    reference_action_key = (
        "reference_action_chunk_rad"
        if "reference_action_chunk_rad" in arrays
        else "reference_action_rad"
    )
    candidate_action_key = (
        "candidate_action_chunk_rad"
        if "candidate_action_chunk_rad" in arrays
        else "candidate_action_rad"
    )
    if reference_action_key not in arrays or candidate_action_key not in arrays:
        metrics["action_prediction"] = na("paired action prediction/target arrays unavailable")
    elif joint_ranges_rad is None:
        metrics["action_prediction"] = na("authoritative 28-D joint ranges unavailable")
    else:
        metrics["action_prediction"] = action_prediction_metrics(
            arrays[reference_action_key],
            arrays[candidate_action_key],
            joint_ranges_rad,
            valid_lengths=arrays.get("action_chunk_valid_lengths"),
        )

    if not bool(episode.get("path_efficiency_applicable", True)):
        metrics["path_efficiency"] = na(
            "path efficiency is undefined for disjoint offline query chunks"
        )
    elif hand_pair is None or success_kind is None:
        metrics["path_efficiency"] = na(
            "bilateral EE paths and an explicit semantic/physical success value are required"
        )
    else:
        sequence = metrics["task_sequence"]
        success = sequence.get("task_sequence_success") if isinstance(sequence, Mapping) else None
        metrics["path_efficiency"] = path_efficiency_metrics(
            *hand_pair, success=success, success_kind=str(success_kind)
        )

    if "candidate_q_chunk_rad" in arrays:
        metrics["smoothness"] = chunk_smoothness_metrics(
            arrays["candidate_q_chunk_rad"],
            valid_lengths=arrays.get("action_chunk_valid_lengths"),
            fps=fps,
            reversal_deadband_rad_s=float(
                episode.get("smoothness", {}).get("reversal_deadband_rad_s", 0.002)
            ),
            low_motion_velocity_rad_s=float(
                episode.get("smoothness", {}).get("low_motion_velocity_rad_s", 0.01)
            ),
        )
    elif "candidate_q_rad" in arrays:
        metrics["smoothness"] = smoothness_metrics(
            arrays["candidate_q_rad"],
            fps=fps,
            reversal_deadband_rad_s=float(
                episode.get("smoothness", {}).get("reversal_deadband_rad_s", 0.002)
            ),
            low_motion_velocity_rad_s=float(
                episode.get("smoothness", {}).get("low_motion_velocity_rad_s", 0.01)
            ),
        )
    else:
        metrics["smoothness"] = na("candidate 28-D joint trajectory unavailable")

    metrics["feasibility"] = feasibility_metrics(
        episode.get("feasibility"), arrays.get("feasibility_projection_m")
    )
    if "physical_success" in annotations:
        metrics["physical_success"] = annotations["physical_success"]

    return {
        "schema_version": "paper_episode_metrics_v1",
        "source_episode_id": source_episode_id,
        "evaluation_mode": episode.get("evaluation_mode"),
        "success_kind": success_kind,
        "fps": fps,
        "metrics": metrics,
        "provenance": provenance,
    }
