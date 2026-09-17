"""Policy-independent rigid-doll physical success detectors.

This module consumes logged physics signals only.  It has no simulator import,
learned policy, method label, or per-policy threshold path, so its logic can be
unit-tested entirely on CPU with synthetic traces.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .contracts import CANONICAL_PHASES, FPS, PHYSICAL_SUCCESS, validate_fps
from .metrics import evaluate_task_sequence


REQUIRED_EVENT_LOG_KEYS = (
    "object_com_m",
    "object_orientation_xyzw",
    "object_linear_velocity_m_s",
    "object_angular_velocity_rad_s",
    "left_hand_object_contact",
    "right_hand_object_contact",
    "table_contact",
    "bin_contact",
    "hand_joint_q_rad",
    "arm_joint_q_rad",
)


def _array(value: Any, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim == 0 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty time series")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} has length {len(result)}, expected {length}")
    if np.issubdtype(result.dtype, np.number) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _contact(value: Any, threshold_n: float, length: int, name: str) -> np.ndarray:
    array = _array(value, name, length)
    if array.dtype == np.bool_:
        return array.reshape(length, -1).any(axis=1)
    numeric = np.asarray(array, dtype=np.float64).reshape(length, -1)
    return np.max(np.abs(numeric), axis=1) >= threshold_n


def _first_run(mask: Any, frames: int, start: int = 0) -> tuple[int, int] | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    frames = max(1, int(frames))
    begin = max(0, int(start))
    run = 0
    for index in range(begin, len(values)):
        run = run + 1 if values[index] else 0
        if run >= frames:
            return index - frames + 1, index
    return None


def _first_true(mask: Any, start: int = 0) -> int | None:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1)[max(0, int(start)) :])
    return None if not len(indices) else int(indices[0] + max(0, int(start)))


def _frames(seconds: float, fps: float) -> int:
    return max(1, int(math.ceil(float(seconds) * fps - 1e-12)))


def _inside_box(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.all((points >= lower[None, :]) & (points <= upper[None, :]), axis=1)


def validate_event_log(log: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_EVENT_LOG_KEYS if key not in log]
    if missing:
        raise ValueError(f"physical event log is missing required signals: {missing}")
    com = np.asarray(log["object_com_m"], dtype=np.float64)
    if com.ndim != 2 or com.shape[1] != 3 or not np.isfinite(com).all():
        raise ValueError("object_com_m must be finite [T,3]")
    length = len(com)
    expected_shapes = {
        "object_orientation_xyzw": (length, 4),
        "object_linear_velocity_m_s": (length, 3),
        "object_angular_velocity_rad_s": (length, 3),
        "hand_joint_q_rad": (length, 14),
        "arm_joint_q_rad": (length, 14),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(log[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite {shape}, got {value.shape}")
    quaternion_norm = np.linalg.norm(
        np.asarray(log["object_orientation_xyzw"], dtype=np.float64), axis=1
    )
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-3):
        raise ValueError("object_orientation_xyzw must contain unit quaternions")
    for key in ("left_hand_object_contact", "right_hand_object_contact", "table_contact", "bin_contact"):
        _array(log[key], key, length)
    return {"status": "PASS", "frames": length, "required_signals": list(REQUIRED_EVENT_LOG_KEYS)}


def detect_physical_success(
    log: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Detect fixed physical outcomes and the canonical ordered phase prefix."""

    validation = validate_event_log(log)
    fps = validate_fps(float(config["timing"]["control_fps_hz"]))
    thresholds = config["success_thresholds"]
    contact_threshold = float(thresholds["minimum_hand_object_contact_force_n"])
    length = int(validation["frames"])
    com = np.asarray(log["object_com_m"], dtype=np.float64)
    left_contact = _contact(
        log["left_hand_object_contact"], contact_threshold, length, "left hand contact"
    )
    right_contact = _contact(
        log["right_hand_object_contact"], contact_threshold, length, "right hand contact"
    )
    table_contact = _contact(
        log["table_contact"],
        float(thresholds["maximum_force_for_table_unsupported_n"]),
        length,
        "table contact",
    )
    bin_contact = _contact(
        log["bin_contact"],
        float(thresholds["minimum_bin_contact_force_n"]),
        length,
        "bin contact",
    )
    contact_frames = _frames(float(thresholds["contact_stability_duration_s"]), fps)
    lift_frames = _frames(float(thresholds["lift_hold_duration_s"]), fps)
    handoff_frames = _frames(float(thresholds["handoff_hold_duration_s"]), fps)
    transport_frames = _frames(float(thresholds["transport_hold_duration_s"]), fps)
    release_frames = _frames(float(thresholds["release_hold_duration_s"]), fps)

    initial_z = float(config["object"]["initial_center_world_xyz_m"][2])
    lift_z = initial_z + float(thresholds["lift_height_above_initial_m"])
    elevated_z = float(thresholds["minimum_handoff_com_world_z_m"])
    unsupported = ~table_contact
    elevated = (com[:, 2] >= elevated_z) & unsupported

    left_contact_run = _first_run(left_contact, contact_frames)
    right_contact_search_start = left_contact_run[0] if left_contact_run else 0
    right_contact_run = _first_run(right_contact, contact_frames, right_contact_search_start)
    left_lift_mask = left_contact & unsupported & (com[:, 2] >= lift_z)
    right_lift_mask = right_contact & unsupported & (com[:, 2] >= lift_z)
    left_lift_run = _first_run(
        left_lift_mask, lift_frames, left_contact_run[0] if left_contact_run else 0
    )
    right_lift_run = _first_run(
        right_lift_mask, lift_frames, right_contact_run[0] if right_contact_run else 0
    )

    dual_mask = left_contact & right_contact & elevated
    dual_run = _first_run(
        dual_mask, contact_frames, right_contact_run[0] if right_contact_run else 0
    )
    right_owned_mask = right_contact & ~left_contact & elevated
    right_owned_run = _first_run(
        right_owned_mask, handoff_frames, dual_run[0] if dual_run else 0
    )
    left_release_frame = (
        _first_true(~left_contact, start=dual_run[0]) if dual_run is not None else None
    )

    transport_box = thresholds["right_transport_target_volume_world_m"]
    transport_inside = _inside_box(
        com,
        np.asarray(transport_box["lower_xyz_m"], dtype=np.float64),
        np.asarray(transport_box["upper_xyz_m"], dtype=np.float64),
    )
    transport_mask = transport_inside & right_contact & elevated
    transport_run = _first_run(
        transport_mask, transport_frames, right_owned_run[0] if right_owned_run else 0
    )

    bin_box = thresholds["release_bin_interior_volume_world_m"]
    bin_inside = _inside_box(
        com,
        np.asarray(bin_box["lower_xyz_m"], dtype=np.float64),
        np.asarray(bin_box["upper_xyz_m"], dtype=np.float64),
    )
    if "right_hand_open" not in log:
        right_open = np.zeros(length, dtype=bool)
        release_input_complete = False
    else:
        right_open = np.asarray(log["right_hand_open"], dtype=bool).reshape(-1)
        if len(right_open) != length:
            raise ValueError("right_hand_open length mismatch")
        release_input_complete = True
    release_mask = bin_inside & ~right_contact & right_open
    release_run = _first_run(
        release_mask, release_frames, transport_run[0] if transport_run else 0
    ) if release_input_complete else None
    bin_entry_frame = _first_true(bin_inside, start=transport_run[0] if transport_run else 0)

    approach_distance = float(thresholds["hand_approach_distance_m"])
    approach_frames = _frames(float(thresholds["approach_stability_duration_s"]), fps)
    approach_events: dict[str, int | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        key = f"{side}_hand_position_m"
        if key in log:
            hand = np.asarray(log[key], dtype=np.float64)
            if hand.shape != com.shape or not np.isfinite(hand).all():
                raise ValueError(f"{key} must be finite [T,3]")
            run = _first_run(np.linalg.norm(hand - com, axis=1) <= approach_distance, approach_frames)
            approach_events[side] = run[0] if run else None

    physical_event_frames = {
        "LEFT_CONTACT": left_contact_run[0] if left_contact_run else None,
        "LEFT_LIFT": left_lift_run[0] if left_lift_run else None,
        "RIGHT_CONTACT": right_contact_run[0] if right_contact_run else None,
        "LEFT_RELEASE": left_release_frame,
        "RIGHT_OWNED": right_owned_run[0] if right_owned_run else None,
        "BIN_ENTRY": bin_entry_frame,
        "FINAL_RELEASE": release_run[0] if release_run else None,
    }
    canonical_events = {
        "LEFT_APPROACH": approach_events["left"],
        "LEFT_GRASP": physical_event_frames["LEFT_CONTACT"],
        "LEFT_TRANSPORT": physical_event_frames["LEFT_LIFT"],
        "RIGHT_APPROACH": approach_events["right"],
        "DUAL_CONTACT": dual_run[0] if dual_run else None,
        "RIGHT_OWNED": physical_event_frames["RIGHT_OWNED"],
        "RIGHT_TRANSPORT": transport_run[0] if transport_run else None,
        "RELEASE": physical_event_frames["FINAL_RELEASE"],
    }
    sequence = evaluate_task_sequence(
        canonical_events, success_kind=PHYSICAL_SUCCESS, authoritative=True
    )
    outcomes = {
        "LEFT_LIFT_SUCCESS": int(left_lift_run is not None),
        "RIGHT_LIFT_SUCCESS": int(right_lift_run is not None),
        "HANDOFF_SUCCESS": int(right_owned_run is not None),
        "RIGHT_TRANSPORT_SUCCESS": int(transport_run is not None),
        "RELEASE_SUCCESS": int(release_run is not None),
    }
    outcomes["FULL_PHYSICAL_SUCCESS"] = int(
        outcomes["LEFT_LIFT_SUCCESS"]
        and outcomes["HANDOFF_SUCCESS"]
        and outcomes["RIGHT_TRANSPORT_SUCCESS"]
        and outcomes["RELEASE_SUCCESS"]
    )
    return {
        "schema_version": "doll_handoff_physical_success_v1",
        "status": "READY",
        "fps": fps,
        "outcomes": outcomes,
        **outcomes,
        "physical_event_frames": physical_event_frames,
        "canonical_phase_events_frame": canonical_events,
        "task_sequence": sequence,
        "release_input_complete": release_input_complete,
        "diagnostics": {
            "lift_threshold_world_z_m": lift_z,
            "handoff_elevated_threshold_world_z_m": elevated_z,
            "minimum_object_com_world_z_m": float(np.min(com[:, 2])),
            "maximum_object_com_world_z_m": float(np.max(com[:, 2])),
            "left_contact_frame_count": int(np.count_nonzero(left_contact)),
            "right_contact_frame_count": int(np.count_nonzero(right_contact)),
            "table_contact_frame_count": int(np.count_nonzero(table_contact)),
            "bin_contact_frame_count": int(np.count_nonzero(bin_contact)),
        },
        "method_or_policy_specific_logic": False,
    }


def event_log_schema() -> dict[str, Any]:
    return {
        "schema_version": "doll_handoff_physics_event_log_v1",
        "fps": FPS,
        "required_arrays": {
            "object_com_m": "float [T,3] world XYZ",
            "object_orientation_xyzw": "float [T,4] world quaternion",
            "object_linear_velocity_m_s": "float [T,3]",
            "object_angular_velocity_rad_s": "float [T,3]",
            "left_hand_object_contact": "bool [T] or force [T,...] N",
            "right_hand_object_contact": "bool [T] or force [T,...] N",
            "table_contact": "bool [T] or force [T,...] N",
            "bin_contact": "bool [T] or force [T,...] N",
            "hand_joint_q_rad": "float [T,14], left7+right7",
            "arm_joint_q_rad": "float [T,14], left7+right7",
        },
        "required_for_release": {
            "right_hand_open": "bool [T] derived from the same frozen open-state tolerance"
        },
        "recommended_for_canonical_approach_phases": {
            "left_hand_position_m": "float [T,3] authoritative physical whole-hand frame",
            "right_hand_position_m": "float [T,3] authoritative physical whole-hand frame",
        },
        "semantic_events_emitted": [
            "LEFT_CONTACT",
            "LEFT_LIFT",
            "RIGHT_CONTACT",
            "LEFT_RELEASE",
            "RIGHT_OWNED",
            "BIN_ENTRY",
            "FINAL_RELEASE",
        ],
    }
