"""Adapters that feed reviewed timelines through the canonical semantic API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aloha_magsafe_semantics.io import canonical_json_hash, sha256_array, sha256_file
from aloha_magsafe_semantics.schema import EventRecord, SemanticTimeline


TASK_EVENTS = (
    "left_phone_grasp_start",
    "phone_rotation_to_portrait_start",
    "phone_portrait_reached",
    "right_accessory_grasp_start",
    "accessory_detachment_start",
    "accessory_removed",
    "phone_move_to_charger_start",
    "phone_charger_attachment_complete",
    "left_phone_release_complete",
    "right_accessory_release_complete",
)


def _clip_progress(values: np.ndarray) -> np.ndarray:
    values = np.maximum.accumulate(np.asarray(values, dtype=np.float64))
    span = float(values[-1] - values[0])
    if span <= np.finfo(np.float64).eps:
        return np.linspace(0.0, 1.0, len(values))
    return np.clip((values - values[0]) / span, 0.0, 1.0)


def _path_progress(values: np.ndarray, start: int, end: int) -> np.ndarray:
    segment = np.asarray(values[start : end + 1], dtype=np.float64)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
    return _clip_progress(distance)


def _rotation_progress(values: np.ndarray, start: int, end: int) -> np.ndarray:
    segment = np.asarray(values[start : end + 1], dtype=np.float64)
    step = Rotation.from_matrix(np.einsum("tji,tjk->tik", segment[:-1], segment[1:])).magnitude()
    return _clip_progress(np.r_[0.0, np.cumsum(np.abs(step))])


def _gripper_progress(values: np.ndarray, start: int, end: int) -> np.ndarray:
    segment = np.asarray(values[start : end + 1], dtype=np.float64)
    # Direction is inferred from the segment, so the adapter does not assume a
    # universal open/close sign or an episode-specific absolute threshold.
    directed = segment[-1] - segment[0]
    signal = (segment - segment[0]) * (1.0 if directed >= 0.0 else -1.0)
    return _clip_progress(signal)


def _place_interval(length: int, start: int, end: int, local: np.ndarray) -> np.ndarray:
    output = np.zeros(length, dtype=np.float64)
    output[start : end + 1] = local
    output[end + 1 :] = 1.0
    return output


def _combined_progress(
    length: int,
    start: int,
    end: int,
    *components: np.ndarray,
) -> np.ndarray:
    local = np.mean(np.vstack([_clip_progress(value) for value in components]), axis=0)
    local = _clip_progress(local)
    return _place_interval(length, start, end, local)


def _phases(length: int, intervals: list[tuple[str, int, int]]) -> np.ndarray:
    result = np.full(length, "UNASSIGNED", dtype="<U48")
    for name, start, end in intervals:
        result[start : end + 1] = name
    return result


def load_human_reviewed_development_timeline(
    timeline_path: str | Path,
    alignment_path: str | Path,
    action: np.ndarray,
    timestamps: np.ndarray,
    left_position: np.ndarray,
    right_position: np.ndarray,
    left_rotation: np.ndarray,
    right_rotation: np.ndarray,
    *,
    trajectory_path: str | Path,
    fk_model_path: str | Path,
    task_geometry_path: str | Path,
) -> SemanticTimeline:
    """Load a reviewed development timeline without exposing numeric rules.

    Observed-frame bookkeeping is converted to action-domain indices using the
    separately reviewed alignment artifact.  All downstream phase construction
    requests events by name from the returned canonical API.
    """
    timeline_path = Path(timeline_path).resolve()
    alignment_path = Path(alignment_path).resolve()
    action = np.asarray(action)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    length = len(action)
    if action.shape != (length, 14) or timestamps.shape != (length,):
        raise ValueError("action/timestamp shape mismatch")
    reviewed = json.loads(timeline_path.read_text(encoding="utf-8"))
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    observed = {row["event"]: row for row in reviewed["events"]}
    mapped = alignment["event_mapping"]
    events: dict[str, EventRecord] = {}
    for name, row in mapped.items():
        if name not in observed:
            raise ValueError(f"alignment event absent from reviewed timeline: {name}")
        source = observed[name]
        if int(source["frame"]) != int(row["observed_frame"]):
            raise ValueError(f"review/alignment mismatch for {name}")
        index = int(row["aligned_action_index"])
        if not 0 <= index < length:
            raise ValueError(f"aligned index out of range for {name}")
        events[name] = EventRecord(
            event_name=name,
            action_index=index,
            action_time_sec=float(timestamps[index]),
            observed_frame=int(row["observed_frame"]),
            observed_time_sec=float(source.get("timestamp_s", source["timestamp"])),
            confidence=1.0,
            confidence_class="HIGH",
            evidence={
                "review_status": reviewed["status"],
                "alignment_status": alignment["status"],
                "command_to_observation_latency_sec": float(alignment["latency_seconds"]),
            },
            provenance={
                "source": "HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE",
                "semantic_runtime_interface": "GENERIC",
                "hardcoded_runtime_indices": False,
                "timeline_sha256": sha256_file(timeline_path),
                "alignment_sha256": sha256_file(alignment_path),
            },
        )
    missing = [name for name in TASK_EVENTS if name not in events]
    if missing:
        raise ValueError(f"reviewed timeline missing task-critical events: {missing}")

    def interval(a: str, b: str) -> tuple[int, int]:
        return int(events[a].action_index), int(events[b].action_index)

    left_gripper = action[:, 6]
    right_gripper = action[:, 13]
    grasp_start, portrait = interval("left_phone_grasp_start", "phone_portrait_reached")
    rotation_start, rotation_end = interval("phone_rotation_to_portrait_start", "phone_portrait_reached")
    right_grasp, detachment = interval("right_accessory_grasp_start", "accessory_detachment_start")
    removal_start, removal_end = interval("accessory_detachment_start", "accessory_removed")
    transport_start, charger = interval("phone_move_to_charger_start", "phone_charger_attachment_complete")
    left_release_start, left_release_end = charger, int(events["left_phone_release_complete"].action_index)
    right_release_start, right_release_end = removal_end, int(events["right_accessory_release_complete"].action_index)

    sample_arrays = {
        "phone_acquisition_progress": _combined_progress(
            length,
            grasp_start,
            portrait,
            _path_progress(left_position, grasp_start, portrait),
            _gripper_progress(left_gripper, grasp_start, portrait),
        ),
        "phone_rotation_progress": _place_interval(
            length,
            rotation_start,
            rotation_end,
            _rotation_progress(left_rotation, rotation_start, rotation_end),
        ),
        "accessory_acquisition_progress": _combined_progress(
            length,
            right_grasp,
            detachment,
            _path_progress(right_position, right_grasp, detachment),
            _gripper_progress(right_gripper, right_grasp, detachment),
        ),
        "accessory_removal_progress": _place_interval(
            length,
            removal_start,
            removal_end,
            _path_progress(right_position, removal_start, removal_end),
        ),
        "phone_to_charger_progress": _place_interval(
            length,
            transport_start,
            charger,
            _path_progress(left_position, transport_start, charger),
        ),
        "charger_transport_progress": _place_interval(
            length,
            transport_start,
            charger,
            _path_progress(left_position, transport_start, charger),
        ),
        "left_release_progress": _place_interval(
            length,
            left_release_start,
            left_release_end,
            _gripper_progress(left_gripper, left_release_start, left_release_end),
        ),
        "right_release_progress": _place_interval(
            length,
            right_release_start,
            right_release_end,
            _gripper_progress(right_gripper, right_release_start, right_release_end),
        ),
    }
    sample_arrays["left_gripper_phase"] = _phases(length, [
        ("OPEN", 0, grasp_start),
        ("ACQUIRE", grasp_start, portrait),
        ("PINCH_HOLD", portrait, charger),
        ("RELEASE", charger, left_release_end),
        ("OPEN_RETURN", left_release_end, length - 1),
    ])
    sample_arrays["right_gripper_phase"] = _phases(length, [
        ("OPEN", 0, right_grasp),
        ("INSERT", right_grasp, detachment),
        ("HOOK_REMOVE", detachment, removal_end),
        ("HOLD", removal_end, right_release_end),
        ("RELEASE_OPEN", right_release_end, length - 1),
    ])
    sample_arrays["left_task_phase"] = sample_arrays["left_gripper_phase"].copy()
    sample_arrays["right_task_phase"] = sample_arrays["right_gripper_phase"].copy()
    sample_arrays["global_task_phase"] = np.asarray([
        f"{left}|{right}" for left, right in zip(
            sample_arrays["left_task_phase"], sample_arrays["right_task_phase"]
        )
    ], dtype="<U96")

    adapter_payload = {
        "timeline_sha256": sha256_file(timeline_path),
        "alignment_sha256": sha256_file(alignment_path),
        "task_events": TASK_EVENTS,
        "source": "HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE",
        "semantic_runtime_interface": "GENERIC",
    }
    return SemanticTimeline(
        trajectory_length=length,
        timestamps=timestamps,
        events=events,
        sample_arrays=sample_arrays,
        detector_config_hash=canonical_json_hash(adapter_payload),
        trajectory_hash=sha256_array(action),
        fk_model_hash=sha256_file(fk_model_path),
        task_geometry_hash=sha256_file(task_geometry_path),
        source_type="optimized_action",
        metadata={
            "source": "HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE",
            "semantic_runtime_interface": "GENERIC",
            "hardcoded_runtime_indices": False,
            "approved_timeline_explicit_input": str(timeline_path),
            "alignment_explicit_input": str(alignment_path),
            "trajectory_path": str(Path(trajectory_path).resolve()),
        },
    )
