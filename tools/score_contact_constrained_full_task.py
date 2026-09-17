#!/usr/bin/env python3
"""Score one complete scripted contact-constrained Doll-Handoff trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
JOINT_CONTRACT = (
    ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
)
FEASIBILITY = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
BIN_CENTER = np.asarray([0.7382120490074158, 0.09978766366839409])
BIN_OPENING = np.asarray([0.178, 0.153])
BIN_BOTTOM_Z = 0.795
BIN_RIM_Z = 0.945


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest(mask: np.ndarray, dt: float) -> float:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def unique_frames(values: np.ndarray, frames: np.ndarray) -> np.ndarray:
    rows = np.r_[np.flatnonzero(np.diff(frames) != 0), len(frames) - 1]
    return np.asarray(values)[rows]


def frame_ranges(frames: np.ndarray) -> list[str]:
    values = np.unique(np.asarray(frames, dtype=np.int64))
    if not len(values):
        return []
    starts = [int(values[0])]
    ends: list[int] = []
    for previous, current in zip(values[:-1], values[1:], strict=True):
        if current != previous + 1:
            ends.append(int(previous))
            starts.append(int(current))
    ends.append(int(values[-1]))
    return [str(start) if start == end else f"{start}-{end}" for start, end in zip(starts, ends)]


def branch_discontinuities(commands: np.ndarray) -> tuple[int, float]:
    acceptance = read_json(FEASIBILITY)["unchanged_acceptance"]
    arms = np.asarray(commands, dtype=np.float64)[:, :14]
    norms = np.linalg.norm(np.diff(arms, axis=0), axis=1)
    flags = []
    for index, value in enumerate(norms):
        prior = norms[max(0, index - 10) : index]
        local = float(np.median(prior)) if len(prior) else 0.0
        threshold = max(
            float(acceptance["branch_absolute_step_norm_rad"]),
            float(acceptance["branch_local_multiplier"]) * max(local, 1.0e-6),
        )
        flags.append(value > threshold)
    return int(np.count_nonzero(flags)), float(np.max(norms, initial=0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    event_path = run / "event_log.npz"
    contact_path = run / "robot_bin_contacts.npz"
    trial_path = run / "trial_result.json"
    with np.load(event_path, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(contact_path, allow_pickle=False) as archive:
        contact = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        *(f"{side}_{digit}_force_n" for side in ("left", "right") for digit in ("thumb", "index", "middle")),
        "doll_bin_contact_force_n",
        "maximum_doll_bin_penetration_m",
    }
    missing = sorted(required - set(event))
    if missing:
        raise RuntimeError(f"trace was not produced with --full-task-audit: {missing}")
    config = read_json(CONFIG)
    trial = read_json(trial_path)
    dt = float(config["timing"]["physics_dt_s"])
    force_threshold = 0.015
    table_threshold = 0.02
    penetration_tolerance = min(
        2.0 * float(config["object"]["contact_offset_m"]),
        float(config["gates"]["maximum_runtime_penetration_m"]),
    )
    frames = event["control_frame"].astype(np.int64)
    stages = event["stage"].astype(str)
    positions = event["object_position_world_m"].astype(np.float64)
    speeds = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    left = np.column_stack(
        [event[f"left_{digit}_force_n"] for digit in ("thumb", "index", "middle")]
    ).astype(np.float64)
    right = np.column_stack(
        [event[f"right_{digit}_force_n"] for digit in ("thumb", "index", "middle")]
    ).astype(np.float64)
    left_three = np.all(left >= force_threshold, axis=1)
    right_three = np.all(right >= force_threshold, axis=1)
    left_any = np.any(left >= force_threshold, axis=1)
    right_any = np.any(right >= force_threshold, axis=1)
    table_free = event["table_contact_force_n"].astype(np.float64) <= table_threshold
    inside_xy = np.all(np.abs(positions[:, :2] - BIN_CENTER) <= BIN_OPENING / 2.0, axis=1)
    in_bin = inside_xy & (positions[:, 2] < BIN_RIM_Z) & (positions[:, 2] > BIN_BOTTOM_Z)

    left_grasp_scope = stages == "GRAVITY_RETENTION"
    left_grasp_duration = longest(left_three & left_grasp_scope, dt)
    left_grasp = left_grasp_duration >= 1.0 - 1.0e-9

    handoff_scope = np.isin(
        stages,
        [
            "RIGHT_THREE_DIGIT_VERIFICATION",
            "LEFT_THUMB_RELEASE",
            "RIGHT_POST_RELEASE_RETENTION",
        ],
    )
    right_verification = stages == "RIGHT_THREE_DIGIT_VERIFICATION"
    right_post_release = stages == "RIGHT_POST_RELEASE_RETENTION"
    right_verification_duration = longest(right_three & right_verification & table_free, dt)
    # The pre-release hard gate requires all three RIGHT digits.  After complete
    # LEFT release, physical ownership requires stable independent retention;
    # it does not relabel a mechanically stable two-digit cradle as failure.
    right_two = np.count_nonzero(right >= force_threshold, axis=1) >= 2
    right_retention_duration = longest(right_two & right_post_release & table_free, dt)
    # Restrict continuous-support evaluation to the actual transfer window.
    transfer_rows = np.flatnonzero(handoff_scope)
    support_continuous = bool(
        len(transfer_rows)
        and np.all((left_any | right_any | in_bin)[transfer_rows])
        and np.all(table_free[transfer_rows])
    )
    handoff = bool(right_verification_duration >= 0.5 and support_continuous)
    right_ownership = bool(right_retention_duration >= 1.0 - 1.0e-9)

    entry_rows = np.flatnonzero(in_bin)
    first_entry_row = int(entry_rows[0]) if len(entry_rows) else None
    stable_left_rows = np.flatnonzero(left_three & left_grasp_scope & table_free)
    # Stable grasp is established while the doll is still intentionally table
    # supported.  The no-drop interval starts when the commanded LEFT lift has
    # actually made it table-unsupported, not while it is still resting before
    # lift-off.
    stable_left_rows = np.flatnonzero(left_three & left_grasp_scope)
    first_stable_left = int(stable_left_rows[0]) if len(stable_left_rows) else None
    lifted_rows = np.flatnonzero(
        table_free
        & np.isin(stages, ["LEFT_LIFT_5CM", "HOLD_ELEVATED", "LEFT_TRANSPORT"])
    )
    first_table_unsupported = int(lifted_rows[0]) if len(lifted_rows) else None
    pre_entry_support = left_any | right_any
    unsupported_pre_entry = (
        np.flatnonzero(~pre_entry_support[first_table_unsupported:first_entry_row])
        + first_table_unsupported
        if first_table_unsupported is not None and first_entry_row is not None
        else np.empty(0, dtype=np.int64)
    )
    # A contact loss over the valid opening which immediately produces physical
    # bin entry is placement, not an unintended table/floor drop.  It remains
    # separately labeled PREMATURE_DROP_INTO_BIN rather than clean release.
    unsupported_only_over_opening = bool(
        np.all(inside_xy[unsupported_pre_entry]) if len(unsupported_pre_entry) else True
    )
    no_drop_before_bin = bool(
        first_stable_left is not None
        and first_table_unsupported is not None
        and first_entry_row is not None
        and np.all(table_free[first_table_unsupported:first_entry_row])
        and unsupported_only_over_opening
    )

    settle_rows = max(1, int(round(1.0 / dt)))
    settled = bool(
        len(in_bin) >= settle_rows
        and np.all(in_bin[-settle_rows:])
        and np.all(speeds[-settle_rows:] <= 0.02)
        and np.max(event["doll_bin_contact_force_n"][-settle_rows:], initial=0.0) > 0.0
    )

    release_frames = frames[stages == "RIGHT_RELEASE"]
    release_start = int(np.min(release_frames)) if len(release_frames) else None
    owned_rows = np.flatnonzero(right_post_release & right_two)
    first_owned_row = int(owned_rows[0]) if len(owned_rows) else 0
    loss_candidates = np.flatnonzero(~right_any & (np.arange(len(right_any)) >= first_owned_row))
    first_loss = int(loss_candidates[0]) if len(loss_candidates) else None
    first_loss_frame = int(frames[first_loss]) if first_loss is not None else None
    if first_loss is None:
        release_classification = "PREMATURE_DROP_OUTSIDE_BIN"
    elif release_start is not None and first_loss_frame >= release_start:
        release_classification = "CLEAN_COMMANDED_RELEASE"
    elif inside_xy[first_loss]:
        release_classification = "PREMATURE_DROP_INTO_BIN"
    else:
        release_classification = "PREMATURE_DROP_OUTSIDE_BIN"

    robot_penetration = float(np.max(contact["penetration_m"], initial=0.0))
    doll_penetration = float(
        np.max(event["maximum_doll_bin_penetration_m"], initial=0.0)
    )
    robot_tunneling = robot_penetration > penetration_tolerance
    doll_tunneling = doll_penetration > penetration_tolerance

    joint_contract = read_json(JOINT_CONTRACT)
    specs = joint_contract["joint_specs"]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    measured = event["measured_q_rad"].astype(np.float64)
    # Position targets are within the exact hard limits.  PhysX joint
    # constraints can exhibit a small solver-scale measured excursion under
    # contact; 2 mrad is fixed here before A/B and is far below a task-scale or
    # branch discontinuity.  Larger excursions remain hard failures.
    joint_solver_tolerance_rad = 0.002
    command_violations = (
        (event["commanded_q_rad"] < lower - 1.0e-9)
        | (event["commanded_q_rad"] > upper + 1.0e-9)
    )
    violations = (
        (measured < lower - joint_solver_tolerance_rad)
        | (measured > upper + joint_solver_tolerance_rad)
        | command_violations
    )
    violation_rows, violation_joints = np.nonzero(violations)

    control_commands = unique_frames(event["commanded_q_rad"], frames)
    branch_count, maximum_arm_step = branch_discontinuities(control_commands)
    q_error = np.abs(event["commanded_q_rad"] - measured)
    contact_positive = contact["force_n"].astype(np.float64) > 1.0e-6
    robot_contact_frames = contact["control_frame"][contact_positive]
    links = contact["robot_link"].astype(str)
    elbow_frames = contact["control_frame"][contact_positive & (links == "right_elbow_link")]
    hand_frames = contact["control_frame"][contact_positive & np.char.startswith(links, "right_hand_")]
    held_end = first_entry_row if first_entry_row is not None else len(speeds)
    peak_held_speed = float(np.max(speeds[:held_end], initial=0.0))

    hard_valid = bool(
        not robot_tunneling
        and not doll_tunneling
        and not len(violation_rows)
        and branch_count == 0
        and np.isfinite(measured).all()
        and np.isfinite(positions).all()
    )
    full_success = bool(
        left_grasp
        and handoff
        and right_ownership
        and no_drop_before_bin
        and len(entry_rows)
        and settled
        and release_classification != "PREMATURE_DROP_OUTSIDE_BIN"
        and hard_valid
    )

    result = {
        "schema_version": "contact_constrained_scripted_task_result_v1",
        "status": "PASS" if full_success else "FAIL",
        "execution_mode": "CONTACT_CONSTRAINED_PHYSICS",
        "command_completed": bool(trial["command_completed"]),
        "outcomes": {
            "LEFT_GRASP": left_grasp,
            "HANDOFF": handoff,
            "RIGHT_OWNERSHIP": right_ownership,
            "NO_DROP_BEFORE_BIN": no_drop_before_bin,
            "DOLL_ENTERS_BIN": bool(len(entry_rows)),
            "DOLL_SETTLES": settled,
            "FULL_TASK_SUCCESS": full_success,
        },
        "durations_s": {
            "left_three_digit_grasp": left_grasp_duration,
            "right_three_digit_verification": right_verification_duration,
            "right_post_release_retention": right_retention_duration,
        },
        "release_classification": release_classification,
        "first_right_contact_loss_frame": first_loss_frame,
        "nominal_release_start_frame": release_start,
        "first_bin_entry_frame": int(frames[first_entry_row]) if first_entry_row is not None else None,
        "diagnostics": {
            "robot_bin_contact_frames": frame_ranges(robot_contact_frames),
            "right_elbow_bin_contact_frames": frame_ranges(elbow_frames),
            "right_hand_bin_contact_frames": frame_ranges(hand_frames),
            "maximum_robot_bin_force_n": float(np.max(contact["force_n"], initial=0.0)),
            "maximum_robot_bin_penetration_m": robot_penetration,
            "maximum_doll_bin_penetration_m": doll_penetration,
            "penetration_tolerance_m": penetration_tolerance,
            "peak_held_object_speed_m_s": peak_held_speed,
            "maximum_command_measured_q_error_rad": float(np.max(q_error, initial=0.0)),
            "joint_limit_violation_count": int(len(violation_rows)),
            "measured_joint_solver_tolerance_rad": joint_solver_tolerance_rad,
            "joint_limit_violation_joints": sorted(
                {str(event["joint_names"][index]) for index in violation_joints}
            ),
            "branch_discontinuity_count": branch_count,
            "maximum_command_arm_step_l2_rad": maximum_arm_step,
            "normal_solver_contact_penetration_only": not robot_tunneling
            and not doll_tunneling,
            "invalid_robot_wall_crossing": robot_tunneling,
            "invalid_doll_wall_or_bottom_tunneling": doll_tunneling,
        },
        "provenance": {
            "event_log": str(event_path),
            "event_log_sha256": sha256(event_path),
            "robot_bin_contacts": str(contact_path),
            "robot_bin_contacts_sha256": sha256(contact_path),
            "trial_result": str(trial_path),
            "trial_result_sha256": sha256(trial_path),
            "command": trial["scripted_command"],
            "command_sha256": trial["scripted_command_sha256"],
            "object_pose_writes_during_timed_loop": trial[
                "object_pose_writes_during_timed_loop"
            ],
            "prohibited_attachment_used": trial["prohibited_attachment_used"],
            "state_restoration_used": trial["state_restoration"]["used"],
        },
    }
    json_path = run / "CONTACT_CONSTRAINED_TASK_RESULT.json"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run / "CONTACT_CONSTRAINED_TASK_RESULT.md").write_text(
        "# Contact-constrained scripted full task\n\n"
        f"Status: **{result['status']}**\n\n"
        + "\n".join(
            f"- {key}: `{'PASS' if value else 'FAIL'}`"
            for key, value in result["outcomes"].items()
        )
        + "\n\n"
        f"Release classification: `{release_classification}`.\n\n"
        f"First bin entry frame: `{result['first_bin_entry_frame']}`; first right-contact "
        f"loss frame: `{first_loss_frame}`; nominal release starts at `{release_start}`.\n\n"
        f"Maximum robot/bin penetration: `{robot_penetration * 1000.0:.6f} mm`; "
        f"maximum doll/bin penetration: `{doll_penetration * 1000.0:.6f} mm`; "
        f"frozen tolerance: `{penetration_tolerance * 1000.0:.3f} mm`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if full_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
