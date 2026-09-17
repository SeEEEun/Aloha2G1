#!/usr/bin/env python3
"""Score one continuous common-execution ACT physical trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
ENVIRONMENT = (
    ROOT
    / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
)
JOINT_CONTRACT = (
    ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
)
STAGES = (
    "PHYSICAL_READINESS",
    "LEFT_GRASP",
    "HANDOFF",
    "RIGHT_OWNERSHIP",
    "NO_DROP_TO_BIN",
    "BIN_ENTRY",
    "BIN_SETTLE",
    "FULL_TASK_SUCCESS",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def longest(mask: np.ndarray, dt: float) -> float:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def first_event(events: np.ndarray, name: str) -> int | None:
    rows = np.flatnonzero(np.char.find(events.astype(str), name) >= 0)
    return int(rows[0]) if len(rows) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(run / "robot_bin_contacts.npz", allow_pickle=False) as archive:
        robot_bin = {key: np.asarray(archive[key]) for key in archive.files}
    trial = read_json(run / "trial_result.json")
    required = {
        "RAW_POLICY_COMMAND",
        "POLICY_SAFE_COMMAND",
        "EXECUTED_COMMAND",
        "MEASURED_Q",
        "COMMON_CONTROLLER_OVERRIDE_MASK",
        "COMMON_CONTROLLER_OVERRIDE_MASK_ARM",
        "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST",
        "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3",
        "COMMON_EXECUTION_PHASE",
        "FROZEN_GRASPABILITY_MARGIN",
        "FROZEN_GRASPABILITY_ELIGIBLE",
        "COMMON_EXECUTION_EVENTS",
        "commanded_q_rad",
        "measured_q_rad",
        "object_position_world_m",
        "object_linear_velocity_m_s",
        "table_contact_force_n",
        "doll_bin_contact_force_n",
        "maximum_doll_bin_penetration_m",
        *(f"{side}_{digit}_force_n" for side in ("left", "right") for digit in ("thumb", "index", "middle")),
    }
    missing = sorted(required - set(event))
    if missing:
        raise RuntimeError(f"common execution trace fields missing: {missing}")

    config = read_json(CONFIG)
    environment = read_json(ENVIRONMENT)
    if environment.get("status") != "FROZEN":
        raise RuntimeError("physical environment is not frozen")
    dt = float(config["timing"]["physics_dt_s"])
    gates = config["gates"]
    force_threshold = float(gates["meaningful_digit_force_n"])
    table_threshold = float(gates["maximum_table_force_for_elevated_n"])
    retention_s = float(gates["minimum_retention_contact_s"])
    frames = event["control_frame"].astype(np.int64)
    positions = event["object_position_world_m"].astype(np.float64)
    speeds = np.linalg.norm(event["object_linear_velocity_m_s"].astype(np.float64), axis=1)
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
    right_two = np.count_nonzero(right >= force_threshold, axis=1) >= 2
    table_free = event["table_contact_force_n"].astype(np.float64) <= table_threshold
    bin_spec = environment["bin"]
    bin_center = np.asarray(bin_spec["opening_center_world_xy_m"], dtype=np.float64)
    bin_opening = np.asarray(bin_spec["opening_dimensions_xy_m"], dtype=np.float64)
    inside_xy = np.all(np.abs(positions[:, :2] - bin_center) <= bin_opening / 2.0, axis=1)
    in_bin = (
        inside_xy
        & (positions[:, 2] > float(bin_spec["bottom_world_z_m"]))
        & (positions[:, 2] < float(bin_spec["rim_world_z_m"]))
    )
    events = event["COMMON_EXECUTION_EVENTS"].astype(str)
    readiness_row = first_event(events, "LEFT_ENVELOPE_ENTRY")
    handoff_row = first_event(events, "HANDOFF_REAL_CONTACT_ENTRY")
    right_support_row = first_event(events, "RIGHT_THREE_DIGIT_SUPPORT_CONFIRMED")
    left_release_row = first_event(events, "GIVING_HAND_RELEASE_START")
    right_owned_row = first_event(events, "RIGHT_PHYSICAL_OWNERSHIP")
    release_row = first_event(events, "FROZEN_BIN_REGION_RELEASE_START")
    physical_readiness = readiness_row is not None

    left_scope = np.arange(len(frames)) >= (readiness_row or 0)
    if handoff_row is not None:
        left_scope &= np.arange(len(frames)) <= handoff_row
    left_grasp_duration = longest(left_three & table_free & left_scope, dt)
    left_grasp = bool(physical_readiness and left_grasp_duration >= retention_s - 1.0e-9)

    right_verification_duration = 0.0
    support_continuous = False
    if handoff_row is not None and left_release_row is not None:
        scope = np.zeros(len(frames), dtype=bool)
        scope[handoff_row : left_release_row + 1] = True
        right_verification_duration = longest(right_three & table_free & scope, dt)
        transfer_rows = np.flatnonzero(scope)
        support_continuous = bool(
            len(transfer_rows)
            and np.all((left_any | right_any)[transfer_rows])
            and np.all(table_free[transfer_rows])
        )
    handoff = bool(
        left_grasp
        and right_support_row is not None
        and right_verification_duration >= 0.5 - 1.0e-9
        and support_continuous
    )

    right_scope = np.arange(len(frames)) >= (left_release_row or len(frames))
    if release_row is not None:
        right_scope &= np.arange(len(frames)) <= release_row
    right_retention_duration = longest(right_two & table_free & right_scope, dt)
    right_ownership = bool(
        handoff
        and right_owned_row is not None
        and right_retention_duration >= retention_s - 1.0e-9
    )

    stable_left = np.flatnonzero(left_three & table_free & left_scope)
    first_supported = int(stable_left[0]) if len(stable_left) else None
    entry_rows = np.flatnonzero(in_bin)
    first_entry = int(entry_rows[0]) if len(entry_rows) else None
    no_drop_to_bin = False
    unsupported_rows = np.empty(0, dtype=np.int64)
    if first_supported is not None and first_entry is not None:
        support = left_any | right_any | in_bin
        unsupported_rows = (
            np.flatnonzero(~support[first_supported:first_entry]) + first_supported
        )
        no_drop_to_bin = bool(
            np.all(table_free[first_supported:first_entry])
            and not len(unsupported_rows)
        )

    settle_rows = max(1, int(round(1.0 / dt)))
    bin_entry = bool(len(entry_rows))
    bin_settle = bool(
        len(in_bin) >= settle_rows
        and np.all(in_bin[-settle_rows:])
        and np.all(speeds[-settle_rows:] <= 0.02)
        and np.max(event["doll_bin_contact_force_n"][-settle_rows:], initial=0.0) > 0.0
    )

    command_equal = bool(np.array_equal(event["EXECUTED_COMMAND"], event["commanded_q_rad"]))
    measured_equal = bool(np.array_equal(event["MEASURED_Q"], event["measured_q_rad"]))
    arm_override_count = int(
        np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_ARM"])
    )
    wrist_override_count = int(
        np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_WRIST"])
    )
    dex3_override_count = int(
        np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"])
    )
    execution_phases = event["COMMON_EXECUTION_PHASE"].astype(str)
    intervention_by_phase = {}
    for phase in (
        "APPROACH",
        "GRASP",
        "HANDOFF",
        "RIGHT_OWNERSHIP",
        "TRANSPORT",
        "RELEASE",
    ):
        scope = execution_phases == phase
        intervention_by_phase[phase] = {
            "physics_rows": int(np.count_nonzero(scope)),
            "ARM": int(
                np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_ARM"][scope])
            ),
            "WRIST": int(
                np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_WRIST"][scope])
            ),
            "DEX3": int(
                np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"][scope])
            ),
        }
    mask_equal = bool(
        np.array_equal(
            event["COMMON_CONTROLLER_OVERRIDE_MASK"],
            event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"],
        )
    )
    maximum_robot_penetration = float(
        np.max(robot_bin.get("penetration_m", np.asarray([])), initial=0.0)
    )
    maximum_doll_penetration = float(
        np.max(event["maximum_doll_bin_penetration_m"], initial=0.0)
    )
    penetration_tolerance = float(environment["penetration"]["normal_solver_tolerance_m"])
    joint_contract = read_json(JOINT_CONTRACT)
    specs = joint_contract["joint_specs"]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    executed = event["EXECUTED_COMMAND"].astype(np.float64)
    measured = event["MEASURED_Q"].astype(np.float64)
    joint_valid = bool(
        np.all(executed >= lower - 1.0e-9)
        and np.all(executed <= upper + 1.0e-9)
        and np.all(measured >= lower - 0.002)
        and np.all(measured <= upper + 0.002)
    )
    hard_valid = bool(
        command_equal
        and measured_equal
        and mask_equal
        and arm_override_count == 0
        and wrist_override_count == 0
        and maximum_robot_penetration <= penetration_tolerance
        and maximum_doll_penetration <= penetration_tolerance
        and joint_valid
        and np.isfinite(executed).all()
        and np.isfinite(measured).all()
        and bool(trial.get("command_completed", False))
        and not bool(trial.get("state_restoration", {}).get("used", False))
        and int(trial.get("object_pose_writes_during_timed_loop", -1)) == 0
    )
    full_success = bool(
        physical_readiness
        and left_grasp
        and handoff
        and right_ownership
        and no_drop_to_bin
        and bin_entry
        and bin_settle
        and hard_valid
    )
    outcomes = {
        "PHYSICAL_READINESS": physical_readiness,
        "LEFT_GRASP": left_grasp,
        "HANDOFF": handoff,
        "RIGHT_OWNERSHIP": right_ownership,
        "NO_DROP_TO_BIN": no_drop_to_bin,
        "BIN_ENTRY": bin_entry,
        "BIN_SETTLE": bin_settle,
        "FULL_TASK_SUCCESS": full_success,
    }
    first_failure = next((name for name in STAGES[:-1] if not outcomes[name]), None)
    if first_failure is None and not hard_valid:
        first_failure = "HARD_PHYSICAL_VALIDITY"
    result = {
        "schema_version": "representation_neutral_common_execution_task_result_v1",
        "status": "PASS" if full_success else "FAIL",
        "outcomes": outcomes,
        "first_failure_stage": first_failure,
        "durations_s": {
            "left_table_free_three_digit_support": left_grasp_duration,
            "right_three_digit_handoff_support": right_verification_duration,
            "right_independent_two_digit_retention": right_retention_duration,
        },
        "events": {
            "readiness_control_frame": int(frames[readiness_row]) if readiness_row is not None else None,
            "handoff_entry_control_frame": int(frames[handoff_row]) if handoff_row is not None else None,
            "right_support_control_frame": int(frames[right_support_row]) if right_support_row is not None else None,
            "right_ownership_control_frame": int(frames[right_owned_row]) if right_owned_row is not None else None,
            "bin_release_control_frame": int(frames[release_row]) if release_row is not None else None,
            "first_bin_entry_control_frame": int(frames[first_entry]) if first_entry is not None else None,
        },
        "hard_physical_validity": hard_valid,
        "fairness_audit": {
            "raw_policy_command_logged": True,
            "executed_command_logged": True,
            "measured_q_logged": True,
            "override_mask_logged": True,
            "executed_equals_engine_command": command_equal,
            "measured_log_identity": measured_equal,
            "arm_common_override_scalar_count": arm_override_count,
            "wrist_common_override_scalar_count": wrist_override_count,
            "dex3_common_override_scalar_count": dex3_override_count,
            "arm_common_intervention_fraction": float(
                arm_override_count / max(1, len(executed) * 8)
            ),
            "wrist_common_intervention_fraction": float(
                wrist_override_count / max(1, len(executed) * 6)
            ),
            "dex3_common_intervention_fraction": float(
                dex3_override_count / max(1, len(executed) * 14)
            ),
            "maximum_raw_to_policy_safe_arm_delta_rad": float(
                np.max(
                    np.abs(event["RAW_POLICY_COMMAND"][:, [0, 1, 2, 3, 7, 8, 9, 10]] - event["POLICY_SAFE_COMMAND"][:, [0, 1, 2, 3, 7, 8, 9, 10]]),
                    initial=0.0,
                )
            ),
            "maximum_raw_to_policy_safe_wrist_delta_rad": float(
                np.max(
                    np.abs(event["RAW_POLICY_COMMAND"][:, [4, 5, 6, 11, 12, 13]] - event["POLICY_SAFE_COMMAND"][:, [4, 5, 6, 11, 12, 13]]),
                    initial=0.0,
                )
            ),
            "substantial_arm_intervention_flag": arm_override_count > 0,
            "wrist_rescue_used": wrist_override_count > 0,
            "intervention_scalar_count_by_phase_and_component": intervention_by_phase,
        },
        "diagnostics": {
            "unsupported_rows_before_bin_count": int(len(unsupported_rows)),
            "maximum_robot_bin_penetration_m": maximum_robot_penetration,
            "maximum_doll_bin_penetration_m": maximum_doll_penetration,
            "penetration_tolerance_m": penetration_tolerance,
            "joint_limits_valid": joint_valid,
            "command_completed": bool(trial.get("command_completed", False)),
        },
    }
    atomic_json(run / "COMMON_EXECUTION_TASK_RESULT.json", result)
    lines = [
        "# Common-execution physical task result",
        "",
        f"Status: **{result['status']}**",
        f"First failure: **{first_failure or 'NONE'}**",
        "",
        "| Stage | Pass |",
        "|---|---:|",
        *[f"| {name.replace('_', ' ').title()} | {'YES' if value else 'NO'} |" for name, value in outcomes.items()],
    ]
    (run / "COMMON_EXECUTION_TASK_RESULT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
