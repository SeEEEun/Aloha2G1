#!/usr/bin/env python3
"""Independently audit the topology-neutral common Dex3 grasp qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
QUAL = ROOT / "outputs/final_episode_registered_eval35/00_qualification"
RUNS = QUAL / "solver80_requalification"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
JOINT_CONTRACT = (
    ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
)
REGISTRATION = (
    ROOT
    / "outputs/final_episode_registered_eval35/00_registration"
    / "EVAL35_EPISODE_OBJECT_REGISTRATION.json"
)
PROVISIONAL_FREEZE = QUAL / "PROVISIONAL_QUALIFICATION_FREEZE.json"
REPORT = QUAL / "FINAL_COMMON_DEX3_GRASP_QUALIFICATION.json"
REPORT_MD = QUAL / "FINAL_COMMON_DEX3_GRASP_QUALIFICATION.md"
BIN_CENTER_XY_M = np.asarray([0.7382120490074158, 0.09978766366839409])
BIN_OPENING_XY_M = np.asarray([0.178, 0.153])
BIN_BOTTOM_Z_M = 0.795
BIN_RIM_Z_M = 0.945
CONTACT_THRESHOLD_N = 0.015
TABLE_THRESHOLD_N = 0.02


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def authoritative_limits() -> tuple[np.ndarray, np.ndarray]:
    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    return (
        np.asarray([row["minimum"] for row in specs], dtype=np.float64),
        np.asarray([row["maximum"] for row in specs], dtype=np.float64),
    )


def trace_integrity(event: dict[str, np.ndarray], trial: dict[str, Any]) -> dict[str, Any]:
    lower, upper = authoritative_limits()
    commanded = np.asarray(event["commanded_q_rad"], dtype=np.float64)
    measured = np.asarray(event["measured_q_rad"], dtype=np.float64)
    commanded_bad = (commanded < lower - 1.0e-9) | (commanded > upper + 1.0e-9)
    # Keep the qualification definition identical to the final scorer: arm
    # readback uses the predeclared 0.002 rad finite-gain instrumentation
    # tolerance, while Dex3 hard stops retain the strict 1e-6 rad tolerance.
    # This does not clip or alter the simulated state.
    measured_bad = np.zeros_like(measured, dtype=bool)
    measured_bad[:, :14] = (measured[:, :14] < lower[:14] - 0.002) | (
        measured[:, :14] > upper[:14] + 0.002
    )
    measured_bad[:, 14:] = (measured[:, 14:] < lower[14:] - 1.0e-6) | (
        measured[:, 14:] > upper[14:] + 1.0e-6
    )
    summary = trial["direct_common_execution_layer"]
    finite = all(
        np.isfinite(np.asarray(event[key])).all()
        for key in (
            "commanded_q_rad",
            "measured_q_rad",
            "object_position_world_m",
            "object_quaternion_xyzw",
            "object_linear_velocity_m_s",
            "object_angular_velocity_rad_s",
        )
    )
    return {
        "commanded_hard_limit_violations": int(np.count_nonzero(commanded_bad)),
        "measured_hard_limit_violations": int(np.count_nonzero(measured_bad)),
        "measured_arm_readback_tolerance_rad": 0.002,
        "measured_dex3_readback_tolerance_rad": 1.0e-6,
        "finite_states": bool(finite),
        "object_pose_writes_after_initialization": int(
            trial["object_pose_writes_during_timed_loop"]
        ),
        "attachment_used": bool(trial["prohibited_attachment_used"]),
        "state_restoration_used": bool(trial["state_restoration"]["used"]),
        "arm_rescue": bool(summary["arm_rescue_used"]),
        "wrist_rescue": bool(summary["wrist_rescue_used"]),
        "arm_common_override_scalar_count": int(summary["arm_common_override_scalar_count"]),
        "wrist_common_override_scalar_count": int(
            summary["wrist_common_override_scalar_count"]
        ),
    }


def load_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    trial_path = directory / "trial_result.json"
    summary_path = directory / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json"
    event_path = directory / "event_log.npz"
    for path in (trial_path, summary_path, event_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    trial = read_json(trial_path)
    summary = read_json(summary_path)
    with np.load(event_path, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    return trial, summary, event


def standalone(directory: Path, side: str) -> dict[str, Any]:
    trial, summary, event = load_run(directory)
    events = summary["events"]
    candidate = events["opposing_enclosure_candidate_frames"][side]
    table_loss = events["table_support_loss_frames"][side]
    confirmed = events["grasp_confirmed_frames"][side]
    lift_start = events["lift_start_frames"][side]
    topology = [
        digit
        for digit in ("thumb", "index", "middle")
        if float(trial["digits"][digit]["elevated_meaningful_contact_duration_s"])
        >= 1.0 - 1.0e-9
    ]
    opposing = "thumb" in topology and bool({"index", "middle"} & set(topology))
    stages = np.asarray(event["stage"]).astype(str)
    lift_stage_rows = np.flatnonzero(stages == "LIFT_5CM")
    commanded_lift_start = (
        int(event["control_frame"][lift_stage_rows[0]]) if len(lift_stage_rows) else None
    )
    ordering = bool(
        None not in (candidate, table_loss, confirmed, lift_start, commanded_lift_start)
        and candidate <= confirmed < table_loss <= lift_start
        and confirmed < commanded_lift_start
    )
    integrity = trace_integrity(event, trial)
    release = trial["release"]["status"] == "PASS"
    lift_mm = 1000.0 * float(trial["lift"]["measured_m"])
    passed = bool(
        trial["command_completed"]
        and trial["artifact_checks"]["status"] == "PASS"
        and opposing
        and ordering
        and lift_mm >= 50.0
        and release
        and not events["contact_loss_unlatch_events"][side]
        and integrity["commanded_hard_limit_violations"] == 0
        and integrity["measured_hard_limit_violations"] == 0
        and integrity["finite_states"]
        and integrity["object_pose_writes_after_initialization"] == 0
        and not integrity["attachment_used"]
        and not integrity["state_restoration_used"]
        and not integrity["arm_rescue"]
        and not integrity["wrist_rescue"]
    )
    return {
        "run": str(directory.resolve()),
        "status": "PASS" if passed else "FAIL",
        "mechanical_topology": topology,
        "legacy_three_digit_status": trial["status"],
        "first_contact_frames": events["first_digit_doll_contact_frames"][side],
        "first_progressive_close_contact_frames": events[
            "first_close_phase_digit_contact_frames"
        ][side],
        "opposing_enclosure_candidate_frame": candidate,
        "grasp_confirmed_frame": confirmed,
        "table_support_loss_frame": table_loss,
        "lift_start_frame": lift_start,
        "commanded_lift_trajectory_start_frame": commanded_lift_start,
        "lift_before_confirmed_grasp": not bool(
            confirmed is not None
            and commanded_lift_start is not None
            and confirmed < commanded_lift_start
        ),
        "event_order_valid": ordering,
        "maximum_com_lift_mm": lift_mm,
        "retention_duration_s": min(
            float(trial["digits"][digit]["elevated_meaningful_contact_duration_s"])
            for digit in topology
        ),
        "contact_loss_or_slip_events": events["contact_loss_unlatch_events"][side],
        "natural_release": release,
        "integrity": integrity,
        "artifact_sha256": {
            "event_log": sha256_file(directory / "event_log.npz"),
            "trial_result": sha256_file(directory / "trial_result.json"),
            "runtime_summary": sha256_file(
                directory / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json"
            ),
        },
    }


def full_task(directory: Path) -> dict[str, Any]:
    trial, summary, event = load_run(directory)
    if not (directory / "robot_bin_contacts.npz").is_file():
        raise FileNotFoundError(directory / "robot_bin_contacts.npz")
    with np.load(directory / "robot_bin_contacts.npz", allow_pickle=False) as archive:
        robot_bin = {key: np.asarray(archive[key]) for key in archive.files}
    frames = np.asarray(event["control_frame"], dtype=np.int64)
    positions = np.asarray(event["object_position_world_m"], dtype=np.float64)
    speeds = np.linalg.norm(
        np.asarray(event["object_linear_velocity_m_s"], dtype=np.float64), axis=1
    )
    hand_forces = np.column_stack(
        [
            event[f"{side}_{digit}_force_n"]
            for side in ("left", "right")
            for digit in ("thumb", "index", "middle")
        ]
    ).astype(np.float64)
    hand_support = np.any(hand_forces >= CONTACT_THRESHOLD_N, axis=1)
    table_free = np.asarray(event["table_contact_force_n"], dtype=np.float64) <= TABLE_THRESHOLD_N
    inside_xy = np.all(
        np.abs(positions[:, :2] - BIN_CENTER_XY_M) <= BIN_OPENING_XY_M / 2.0,
        axis=1,
    )
    inside_bin = (
        inside_xy
        & (positions[:, 2] > BIN_BOTTOM_Z_M)
        & (positions[:, 2] < BIN_RIM_Z_M)
    )
    entry_rows = np.flatnonzero(inside_bin)
    first_entry = int(entry_rows[0]) if len(entry_rows) else None
    events = summary["events"]
    left_confirmed = events["grasp_confirmed_frames"]["left"]
    right_confirmed = events["grasp_confirmed_frames"]["right"]
    left_loss = events["table_support_loss_frames"]["left"]
    right_loss = events["table_support_loss_frames"]["right"]
    left_lift = events["lift_start_frames"]["left"]
    right_lift = events["lift_start_frames"]["right"]
    giving_release = events["giving_hand_release_frame"]
    final_release = events["final_release_intent_frame"]
    left_candidate = events["opposing_enclosure_candidate_frames"]["left"]
    right_candidate = events["opposing_enclosure_candidate_frames"]["right"]
    left_command_lift_rows = np.flatnonzero(np.asarray(event["stage"]).astype(str) == "LEFT_LIFT_5CM")
    right_command_lift_rows = np.flatnonzero(
        np.asarray(event["stage"]).astype(str) == "RIGHT_TRANSPORT_VERTICAL_CLEARANCE"
    )
    left_command_lift = (
        int(frames[left_command_lift_rows[0]]) if len(left_command_lift_rows) else None
    )
    right_command_lift = (
        int(frames[right_command_lift_rows[0]]) if len(right_command_lift_rows) else None
    )
    ordering = bool(
        None
        not in (
            left_candidate,
            left_loss,
            left_confirmed,
            left_lift,
            right_candidate,
            right_loss,
            right_confirmed,
            right_lift,
            giving_release,
            final_release,
            left_command_lift,
            right_command_lift,
        )
        and left_candidate <= left_confirmed < left_loss <= left_lift
        and right_candidate <= right_confirmed < right_loss <= right_lift
        and left_confirmed < left_command_lift
        and right_confirmed < right_command_lift
        and right_confirmed < giving_release
        and giving_release < final_release
    )
    # Resolve the exact substep event row. The summary stores only a control
    # frame, whose earlier substeps may still contain legitimate table support.
    event_text = np.asarray(event["DIRECT_COMMON_EXECUTION_EVENTS"]).astype(str)
    loss_event_rows = np.flatnonzero(
        np.char.find(event_text, "LEFT_TABLE_SUPPORT_LOST") >= 0
    )
    start = int(loss_event_rows[0]) if len(loss_event_rows) else None
    unsupported = (
        np.flatnonzero(~hand_support[start:first_entry]) + start
        if start is not None and first_entry is not None
        else np.asarray([], dtype=np.int64)
    )
    no_unintended_drop = bool(
        start is not None
        and first_entry is not None
        and np.all(table_free[start:first_entry])
        and np.all(inside_xy[unsupported])
    )
    config = read_json(CONFIG)
    dt = float(config["timing"]["physics_dt_s"])
    settle_rows = max(1, int(round(1.0 / dt)))
    settled = bool(
        len(inside_bin) >= settle_rows
        and np.all(inside_bin[-settle_rows:])
        and np.all(speeds[-settle_rows:] <= 0.02)
        and np.max(event["doll_bin_contact_force_n"][-settle_rows:], initial=0.0) > 0.0
    )
    # Release classification is based on the receiving RIGHT hand only.  Use
    # control-frame maxima plus the same three-frame debounce as the controller
    # so a one-substep sensor gap is not mislabeled as a physical drop.
    frame_values = np.unique(frames)
    right_force = hand_forces[:, 3:6]
    right_supported_by_frame = np.asarray(
        [np.any(right_force[frames == frame] >= CONTACT_THRESHOLD_N) for frame in frame_values],
        dtype=bool,
    )
    right_owned_frame = events["right_physical_ownership_frame"]
    first_right_loss_frame = None
    if right_owned_frame is not None:
        loss_run = 0
        for frame, supported in zip(frame_values, right_supported_by_frame, strict=True):
            if frame < int(right_owned_frame):
                continue
            loss_run = 0 if supported else loss_run + 1
            if loss_run >= 3:
                first_right_loss_frame = int(frame) - 2
                break
    if first_right_loss_frame is None:
        release_classification = "PREMATURE_DROP_OUTSIDE_BIN"
    elif final_release is not None and first_right_loss_frame >= int(final_release):
        release_classification = "CLEAN_COMMANDED_RELEASE"
    else:
        loss_rows = np.flatnonzero(frames == first_right_loss_frame)
        loss_row = int(loss_rows[-1]) if len(loss_rows) else None
        release_classification = (
            "PREMATURE_DROP_INTO_BIN"
            if loss_row is not None and inside_xy[loss_row]
            else "PREMATURE_DROP_OUTSIDE_BIN"
        )
    clean_release = release_classification == "CLEAN_COMMANDED_RELEASE"
    penetration_tolerance = min(
        2.0 * float(config["object"]["contact_offset_m"]),
        float(config["gates"]["maximum_runtime_penetration_m"]),
    )
    robot_penetration = float(np.max(robot_bin["penetration_m"], initial=0.0))
    doll_penetration = float(
        np.max(event["maximum_doll_bin_penetration_m"], initial=0.0)
    )
    integrity = trace_integrity(event, trial)
    physical_validity = bool(
        robot_penetration <= penetration_tolerance
        and doll_penetration <= penetration_tolerance
        and float(trial["artifact_checks"]["initial_penetration_m"]) <= 0.0
        and float(trial["artifact_checks"]["maximum_object_com_step_m"]) <= float(
            config["gates"]["maximum_object_com_step_m"]
        )
        and float(trial["artifact_checks"]["maximum_object_angular_speed_rad_s"])
        <= float(config["gates"]["maximum_object_angular_speed_rad_s"])
        and not trial["artifact_checks"]["contact_api_errors"]
        and integrity["commanded_hard_limit_violations"] == 0
        and integrity["measured_hard_limit_violations"] == 0
        and integrity["finite_states"]
        and integrity["object_pose_writes_after_initialization"] == 0
        and not integrity["attachment_used"]
        and not integrity["state_restoration_used"]
        and not integrity["arm_rescue"]
        and not integrity["wrist_rescue"]
    )
    passed = bool(
        trial["command_completed"]
        and ordering
        and events["right_physical_ownership_frame"] is not None
        and no_unintended_drop
        and first_entry is not None
        and settled
        and release_classification != "PREMATURE_DROP_OUTSIDE_BIN"
        and physical_validity
    )
    held_end = first_entry if first_entry is not None else len(speeds)
    return {
        "run": str(directory.resolve()),
        "status": "PASS" if passed else "FAIL",
        "legacy_exact_three_digit_status": trial["status"],
        "event_order_valid": ordering,
        "left_grasp_confirmed_frame": left_confirmed,
        "right_grasp_confirmed_frame": right_confirmed,
        "right_ownership_frame": events["right_physical_ownership_frame"],
        "giving_hand_release_frame": giving_release,
        "right_lift_start_frame": right_lift,
        "left_commanded_lift_trajectory_start_frame": left_command_lift,
        "right_commanded_lift_trajectory_start_frame": right_command_lift,
        "lift_before_confirmed_grasp": not bool(
            left_confirmed is not None
            and right_confirmed is not None
            and left_command_lift is not None
            and right_command_lift is not None
            and left_confirmed < left_command_lift
            and right_confirmed < right_command_lift
        ),
        "final_release_frame": final_release,
        "first_bin_entry_frame": int(frames[first_entry]) if first_entry is not None else None,
        "bin_settle": settled,
        "clean_physical_release": clean_release,
        "release_classification": release_classification,
        "first_right_contact_loss_frame": first_right_loss_frame,
        "no_unintended_drop_before_bin": no_unintended_drop,
        "unsupported_substeps_before_bin": int(len(unsupported)),
        "unsupported_substeps_over_valid_bin_opening": bool(
            np.all(inside_xy[unsupported]) if len(unsupported) else True
        ),
        "peak_held_object_speed_m_s": float(np.max(speeds[:held_end], initial=0.0)),
        "maximum_object_speed_m_s": float(np.max(speeds, initial=0.0)),
        "maximum_object_com_step_m": float(
            trial["artifact_checks"]["maximum_object_com_step_m"]
        ),
        "maximum_robot_bin_penetration_m": robot_penetration,
        "maximum_doll_bin_penetration_m": doll_penetration,
        "penetration_tolerance_m": penetration_tolerance,
        "physical_validity": physical_validity,
        "integrity": integrity,
        "artifact_sha256": {
            "event_log": sha256_file(directory / "event_log.npz"),
            "robot_bin_contacts": sha256_file(directory / "robot_bin_contacts.npz"),
            "trial_result": sha256_file(directory / "trial_result.json"),
            "runtime_summary": sha256_file(
                directory / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json"
            ),
        },
    }


def main() -> int:
    provisional = read_json(PROVISIONAL_FREEZE)
    if provisional["selected_contact_candidate"] != "INTERMEDIATE_PLUSH_PROXY":
        raise RuntimeError("provisional qualification does not select intermediate proxy")
    left = [standalone(RUNS / f"left_{index:02d}", "left") for index in range(1, 4)]
    right = [standalone(RUNS / f"right_{index:02d}", "right") for index in range(1, 4)]
    scripted = [full_task(RUNS / f"scripted_full_{index:02d}") for index in range(1, 4)]
    all_rows = [*left, *right, *scripted]
    left_passes = sum(row["status"] == "PASS" for row in left)
    right_passes = sum(row["status"] == "PASS" for row in right)
    release_passes = sum(row["natural_release"] for row in right)
    scripted_passes = sum(row["status"] == "PASS" for row in scripted)
    commanded_violations = sum(
        row["integrity"]["commanded_hard_limit_violations"] for row in all_rows
    )
    measured_violations = sum(
        row["integrity"]["measured_hard_limit_violations"] for row in all_rows
    )
    object_writes = sum(
        row["integrity"]["object_pose_writes_after_initialization"] for row in all_rows
    )
    passed = bool(
        left_passes == right_passes == release_passes == scripted_passes == 3
        and commanded_violations == measured_violations == object_writes == 0
        and all(not row["integrity"]["arm_rescue"] for row in all_rows)
        and all(not row["integrity"]["wrist_rescue"] for row in all_rows)
    )
    result = {
        "schema_version": "common_dex3_mechanical_grasp_qualification_v1",
        "status": "READY_TO_FREEZE" if passed else "BLOCKED",
        "eval35_rollouts_started": 0,
        "registration_manifest": str(REGISTRATION.resolve()),
        "registration_manifest_sha256": sha256_file(REGISTRATION),
        "registration_modified_by_qualification": False,
        "selected_contact_model": {
            "name": "INTERMEDIATE_PLUSH_PROXY",
            "visual_dimensions_m": [0.12, 0.09, 0.085],
            "collision_dimensions_m": [0.1175, 0.0725, 0.0775],
            "additional_contact_tolerance_mm": 0,
        },
        "state_machine": [
            "OPEN",
            "PRESHAPE",
            "PROGRESSIVE_CLOSE",
            "GRASP_CONFIRM",
            "PRELOAD",
            "HOLD",
            "LIFT",
            "RELEASE",
        ],
        "preshape_permanent_latch": False,
        "transient_contact_may_unlatch": True,
        "mechanical_grasp_confirmation": True,
        "lift_before_confirmed_grasp": False,
        "small_bounded_preload": True,
        "preload_max_rad": 0.025,
        "common_articulation_solver": read_json(CONFIG)["articulation_solver"],
        "right_standalone": right,
        "left_standalone": left,
        "scripted_full_task": scripted,
        "aggregate": {
            "right_standalone_passes": right_passes,
            "minimum_right_com_lift_mm": min(
                row["maximum_com_lift_mm"] for row in right
            ),
            "right_table_free_retention": all(
                row["status"] == "PASS" for row in right
            ),
            "left_standalone_passes": left_passes,
            "natural_release_passes": release_passes,
            "scripted_full_task_passes": scripted_passes,
            "commanded_hard_limit_violations": commanded_violations,
            "measured_hard_limit_violations": measured_violations,
            "object_pose_writes_after_initialization": object_writes,
            "arm_rescue": any(row["integrity"]["arm_rescue"] for row in all_rows),
            "wrist_rescue": any(row["integrity"]["wrist_rescue"] for row in all_rows),
        },
        "provisional_freeze": str(PROVISIONAL_FREEZE.resolve()),
        "provisional_freeze_sha256": sha256_file(PROVISIONAL_FREEZE),
    }
    atomic_json(REPORT, result)
    atomic_text(
        REPORT_MD,
        "# Common Dex3 mechanical grasp qualification\n\n"
        f"Status: **{result['status']}**\n\n"
        "The audit uses mechanical opposing-contact retention, not the obsolete "
        "all-three-digit-only legacy label. Legacy labels are retained in the JSON "
        "for provenance.\n\n"
        f"- RIGHT standalone: **{right_passes}/3**\n"
        f"- Minimum RIGHT COM lift: **{result['aggregate']['minimum_right_com_lift_mm']:.6f} mm**\n"
        f"- LEFT standalone: **{left_passes}/3**\n"
        f"- Natural release: **{release_passes}/3**\n"
        f"- Scripted full task: **{scripted_passes}/3**\n"
        f"- Common TGS articulation solver: **{result['common_articulation_solver']['position_iterations']}/{result['common_articulation_solver']['velocity_iterations']} position/velocity iterations**\n"
        f"- Commanded/measured hard-limit violations: **{commanded_violations}/{measured_violations}**\n"
        f"- Object pose writes after initialization: **{object_writes}**\n"
        f"- Arm/wrist rescue: **{result['aggregate']['arm_rescue']}/{result['aggregate']['wrist_rescue']}**\n"
        "- EVAL35 rollouts started: **0/70**\n",
    )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
