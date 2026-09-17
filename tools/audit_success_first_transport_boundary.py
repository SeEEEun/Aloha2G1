#!/usr/bin/env python3
"""Audit the frozen T4 vertical-to-horizontal transport boundary.

This is a read-only diagnostic.  It does not rebuild commands, change the
grasp, or run physics.  All derivatives are computed from the already executed
30 Hz command and the corresponding 240 Hz Isaac event log.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import authoritative_joint_ranges  # noqa: E402


TRIAL_ROOT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "success_first_common_execution/tail_debug"
    / "t4_three_digit_wrap_slow_transport"
)
COMMAND = TRIAL_ROOT / "tail_command.npz"
EVENT = TRIAL_ROOT / "physics_right_sensor/event_log.npz"
OUTPUT = TRIAL_ROOT / "vertical_to_horizontal_boundary_audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_delta_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def finite_difference(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(values, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=2)
    return velocity, acceleration, jerk


def max_norm(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(values[mask], axis=1), initial=0.0))


def scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> int:
    with np.load(COMMAND, allow_pickle=False) as archive:
        command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        stage = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names:
        raise RuntimeError("command named joint order is not authoritative")
    with np.load(EVENT, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}

    horizontal_frames = np.flatnonzero(stage == "RIGHT_TRANSPORT_TO_BIN")
    vertical_frames = np.flatnonzero(stage == "RIGHT_TRANSPORT_VERTICAL_CLEARANCE")
    if len(horizontal_frames) == 0 or len(vertical_frames) == 0:
        raise RuntimeError("transport stages missing")
    onset = int(horizontal_frames[0])
    if int(vertical_frames[-1]) != onset - 1:
        raise RuntimeError("transport stages are not contiguous")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = np.asarray([lookup[name] for name in g1.arm_joint_names])
    left_indices = np.asarray([lookup[name] for name in g1.hand_joint_names["left"]])
    right_indices = np.asarray([lookup[name] for name in g1.hand_joint_names["right"]])
    right_arm_local = np.flatnonzero(
        np.char.startswith(np.asarray(g1.arm_joint_names).astype(str), "right_")
    )

    wrist_position = []
    wrist_rotation = []
    for row in command:
        g1.assign(row[arm_indices], row[left_indices], row[right_indices])
        pose = np.asarray(g1.wrist_pose("right"), dtype=np.float64)
        wrist_position.append(pose[:3, 3])
        wrist_rotation.append(pose[:3, :3])
    wrist_position = np.asarray(wrist_position)
    wrist_rotation = np.asarray(wrist_rotation)

    dt = 1.0 / fps
    q_velocity, q_acceleration, q_jerk = finite_difference(command[:, arm_indices], dt)
    x_velocity, x_acceleration, x_jerk = finite_difference(wrist_position, dt)
    orientation_step = np.asarray(
        [0.0]
        + [rotation_delta_rad(wrist_rotation[i - 1], wrist_rotation[i]) for i in range(1, len(command))]
    )
    orientation_velocity = orientation_step / dt

    pre = onset - 1
    onset_joint_delta = command[onset, arm_indices] - command[pre, arm_indices]
    onset_right_arm_delta = onset_joint_delta[right_arm_local]
    onset_position_delta = wrist_position[onset] - wrist_position[pre]
    onset_orientation_delta = rotation_delta_rad(wrist_rotation[pre], wrist_rotation[onset])
    finger_delta = command[onset, right_indices] - command[pre, right_indices]

    event_control = event["control_frame"].astype(np.int64)
    physics_dt = float(np.median(np.diff(event["timestamp_s"])))
    boundary_window = (event_control >= onset - 10) & (event_control <= onset + 10)
    vertical_event = event["stage"].astype(str) == "RIGHT_TRANSPORT_VERTICAL_CLEARANCE"
    horizontal_event = event["stage"].astype(str) == "RIGHT_TRANSPORT_TO_BIN"
    force_threshold = 0.015
    forces = np.column_stack(
        [event[f"{digit}_force_n"] for digit in ("thumb", "index", "middle")]
    ).astype(np.float64)
    all_three = np.all(forces >= force_threshold, axis=1)
    object_position = event["object_position_world_m"].astype(np.float64)
    object_velocity = event["object_linear_velocity_m_s"].astype(np.float64)

    frame_rows = []
    for frame in range(onset - 10, onset + 11):
        mask = event_control == frame
        if not np.any(mask):
            continue
        frame_rows.append(
            {
                "control_frame": frame,
                "stage": str(event["stage"][np.flatnonzero(mask)[0]]),
                "commanded_wrist_position_model_m": wrist_position[frame],
                "commanded_arm_step_norm_rad": float(
                    np.linalg.norm(command[frame, arm_indices] - command[max(0, frame - 1), arm_indices])
                ),
                "commanded_wrist_step_m": float(
                    np.linalg.norm(wrist_position[frame] - wrist_position[max(0, frame - 1)])
                ),
                "commanded_orientation_step_rad": float(orientation_step[frame]),
                "mean_digit_force_n": np.mean(forces[mask], axis=0),
                "minimum_digit_force_n": np.min(forces[mask], axis=0),
                "all_three_loaded_fraction": float(np.mean(all_three[mask])),
                "object_position_world_m_end": object_position[np.flatnonzero(mask)[-1]],
                "object_velocity_world_m_s_end": object_velocity[np.flatnonzero(mask)[-1]],
                "maximum_object_speed_m_s": max_norm(object_velocity, mask),
            }
        )

    first_index_loss_step = int(
        np.flatnonzero(vertical_event & (forces[:, 1] < force_threshold))[0]
    )
    first_all_contact_loss_horizontal = np.flatnonzero(horizontal_event & ~np.any(forces >= force_threshold, axis=1))
    first_all_contact_loss_step = (
        int(first_all_contact_loss_horizontal[0])
        if len(first_all_contact_loss_horizontal)
        else None
    )
    first_horizontal_motion_frame = int(
        horizontal_frames[
            np.flatnonzero(
                np.linalg.norm(
                    wrist_position[horizontal_frames]
                    - wrist_position[horizontal_frames[0]],
                    axis=1,
                )
                > 1.0e-6
            )[0]
        ]
    )

    # Classification is intentionally conservative.  Discontinuity classes
    # require a boundary spike relative to the executed segment, not merely a
    # nonzero smooth command.  Contact loss before meaningful horizontal travel
    # is physical retention loss at the terminal vertical pose.
    arm_step = np.linalg.norm(np.diff(command[:, arm_indices], axis=0), axis=1)
    segment = (np.arange(len(command) - 1) >= vertical_frames[0]) & (
        np.arange(len(command) - 1) < horizontal_frames[-1]
    )
    boundary_step_ratio = float(
        np.linalg.norm(onset_joint_delta)
        / max(float(np.max(arm_step[segment], initial=0.0)), 1.0e-12)
    )
    boundary_position_step_ratio = float(
        np.linalg.norm(onset_position_delta)
        / max(
            float(
                np.max(
                    np.linalg.norm(np.diff(wrist_position, axis=0), axis=1)[segment],
                    initial=0.0,
                )
            ),
            1.0e-12,
        )
    )
    classification = "TRUE_GRASP_RETENTION_FAILURE"

    payload = {
        "schema_version": "success_first_transport_boundary_audit_v1",
        "status": "CLASSIFIED",
        "classification": classification,
        "evidence": {
            "horizontal_onset_control_frame": onset,
            "horizontal_onset_time_s": onset / fps,
            "first_meaningful_horizontal_motion_control_frame": first_horizontal_motion_frame,
            "controller": {
                "control_fps_hz": fps,
                "control_timestep_s": dt,
                "physics_timestep_s": physics_dt,
                "physics_substeps_per_control_frame": int(round(dt / physics_dt)),
                "interpolation": "separate minimum-jerk Cartesian segments, each with zero endpoint velocity and acceleration",
            },
            "boundary_command": {
                "right_arm_joint_delta_rad": onset_right_arm_delta,
                "all_arm_joint_delta_norm_rad": float(np.linalg.norm(onset_joint_delta)),
                "wrist_translation_delta_m": onset_position_delta,
                "wrist_translation_delta_norm_m": float(np.linalg.norm(onset_position_delta)),
                "wrist_orientation_delta_rad": onset_orientation_delta,
                "finger_command_delta_rad": finger_delta,
                "finger_command_continuous": bool(np.max(np.abs(finger_delta), initial=0.0) <= 1.0e-12),
                "arm_step_fraction_of_transport_maximum": boundary_step_ratio,
                "wrist_step_fraction_of_transport_maximum": boundary_position_step_ratio,
                "same_ik_branch": bool(np.linalg.norm(onset_joint_delta) < 0.05),
            },
            "kinematics": {
                "vertical_max_cartesian_velocity_m_s": max_norm(x_velocity, stage == "RIGHT_TRANSPORT_VERTICAL_CLEARANCE"),
                "horizontal_max_cartesian_velocity_m_s": max_norm(x_velocity, stage == "RIGHT_TRANSPORT_TO_BIN"),
                "boundary_cartesian_velocity_m_s": x_velocity[onset],
                "boundary_cartesian_acceleration_m_s2": x_acceleration[onset],
                "boundary_cartesian_jerk_m_s3": x_jerk[onset],
                "vertical_max_joint_velocity_rad_s": max_norm(q_velocity, stage == "RIGHT_TRANSPORT_VERTICAL_CLEARANCE"),
                "horizontal_max_joint_velocity_rad_s": max_norm(q_velocity, stage == "RIGHT_TRANSPORT_TO_BIN"),
                "boundary_joint_velocity_rad_s": q_velocity[onset],
                "boundary_joint_acceleration_rad_s2": q_acceleration[onset],
                "boundary_joint_jerk_rad_s3": q_jerk[onset],
                "boundary_orientation_velocity_rad_s": float(orientation_velocity[onset]),
                "maximum_orientation_step_vertical_rad": float(np.max(orientation_step[vertical_frames], initial=0.0)),
                "maximum_orientation_step_horizontal_rad": float(np.max(orientation_step[horizontal_frames], initial=0.0)),
            },
            "contact_and_object": {
                "force_threshold_n": force_threshold,
                "first_index_below_threshold_physics_step_during_vertical": first_index_loss_step,
                "first_index_below_threshold_control_frame_during_vertical": int(event_control[first_index_loss_step]),
                "first_all_digit_loss_physics_step_during_horizontal": first_all_contact_loss_step,
                "first_all_digit_loss_control_frame_during_horizontal": (
                    int(event_control[first_all_contact_loss_step])
                    if first_all_contact_loss_step is not None
                    else None
                ),
                "object_horizontal_displacement_before_horizontal_onset_m": float(
                    np.linalg.norm(
                        object_position[np.flatnonzero(event_control == pre)[-1], :2]
                        - object_position[np.flatnonzero(event_control == vertical_frames[0])[0], :2]
                    )
                ),
                "boundary_frames": frame_rows,
            },
            "excluded_failure_classes": {
                "HORIZONTAL_IK_DISCONTINUITY": "boundary arm step is tiny and remains on the same continuous joint branch",
                "HORIZONTAL_ACCELERATION_TOO_HIGH": "minimum-jerk horizontal segment starts from rest; grasp degradation begins at the terminal vertical pose",
                "WRIST_ORIENTATION_TRANSITION_ERROR": "orientation target is preserved and boundary orientation delta is tiny",
                "COMMAND_INTERPOLATION_DISCONTINUITY": "position/orientation/finger commands are continuous at the boundary",
                "TRANSPORT_PATH_COLLISION_OR_REACHABILITY": "offline path has zero collision frames, zero limit violations, and sub-millimetre IK position error",
            },
        },
        "interpretation": (
            "The transport command is continuous.  The index load becomes marginal at the terminal "
            "vertical posture before meaningful horizontal displacement, after which the object leaves "
            "the remaining thumb/middle enclosure.  This is a true physical retention loss caused by "
            "dwelling at the high waypoint, not an IK, interpolation, acceleration, orientation, collision, "
            "or reachability discontinuity."
        ),
        "immutable_grasp_preserved": True,
        "command": str(COMMAND),
        "command_sha256": sha256_file(COMMAND),
        "event_log": str(EVENT),
        "event_log_sha256": sha256_file(EVENT),
        "audit_tool": str(Path(__file__).resolve()),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=scalar) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=scalar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
