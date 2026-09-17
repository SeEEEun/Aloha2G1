#!/usr/bin/env python3
"""Score one classifier-free contact-constrained ACT physical rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
ENVIRONMENT = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
FEASIBILITY = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
STAGES = ("LEFT_GRASP", "HANDOFF", "RIGHT_OWNERSHIP", "TRANSPORT", "BIN_ENTRY", "SETTLE")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def longest(mask: np.ndarray, dt: float) -> float:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def first_event(events: np.ndarray, name: str) -> int | None:
    rows = np.flatnonzero(np.char.find(events.astype(str), name) >= 0)
    return int(rows[0]) if len(rows) else None


def unique_control(values: np.ndarray, frames: np.ndarray) -> np.ndarray:
    rows = np.r_[np.flatnonzero(np.diff(frames) != 0), len(frames) - 1]
    return np.asarray(values)[rows]


def branch_discontinuities(commands: np.ndarray) -> tuple[int, float]:
    acceptance = read_json(FEASIBILITY)["unchanged_acceptance"]
    norms = np.linalg.norm(np.diff(np.asarray(commands)[:, :14], axis=0), axis=1)
    flags = []
    for index, value in enumerate(norms):
        prior = norms[max(0, index - 10):index]
        local = float(np.median(prior)) if len(prior) else 0.0
        threshold = max(
            float(acceptance["branch_absolute_step_norm_rad"]),
            float(acceptance["branch_local_multiplier"]) * max(local, 1.0e-6),
        )
        flags.append(value > threshold)
    return int(np.count_nonzero(flags)), float(np.max(norms, initial=0.0))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        "RAW_POLICY_COMMAND", "POLICY_SAFE_COMMAND", "EXECUTED_COMMAND", "MEASURED_Q",
        "COMMON_CONTROLLER_OVERRIDE_MASK", "COMMON_CONTROLLER_OVERRIDE_MASK_ARM",
        "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST", "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3",
        "COMMON_ARM_HARD_LIMIT_PROJECTION_MASK",
        "DIRECT_COMMON_EXECUTION_PHASE", "DIRECT_COMMON_TASK_INTENT",
        "DIRECT_COMMON_EXECUTION_EVENTS", "commanded_q_rad", "measured_q_rad",
        "object_position_world_m", "object_linear_velocity_m_s", "table_contact_force_n",
        "doll_bin_contact_force_n", "maximum_doll_bin_penetration_m",
        *(f"{side}_{digit}_force_n" for side in ("left", "right") for digit in ("thumb", "index", "middle")),
    }
    missing = sorted(required - set(event))
    if missing:
        raise RuntimeError(f"direct physical trace fields missing: {missing}")
    config = read_json(CONFIG)
    environment = read_json(ENVIRONMENT)
    dt = float(config["timing"]["physics_dt_s"])
    force_threshold = float(config["gates"]["meaningful_digit_force_n"])
    table_threshold = float(config["gates"]["maximum_table_force_for_elevated_n"])
    retention_s = float(config["gates"]["minimum_retention_contact_s"])
    frames = event["control_frame"].astype(np.int64)
    positions = event["object_position_world_m"].astype(np.float64)
    speeds = np.linalg.norm(event["object_linear_velocity_m_s"].astype(np.float64), axis=1)
    left = np.column_stack([event[f"left_{digit}_force_n"] for digit in ("thumb", "index", "middle")]).astype(np.float64)
    right = np.column_stack([event[f"right_{digit}_force_n"] for digit in ("thumb", "index", "middle")]).astype(np.float64)
    left_three = np.all(left >= force_threshold, axis=1)
    right_three = np.all(right >= force_threshold, axis=1)
    left_any = np.any(left >= force_threshold, axis=1)
    right_any = np.any(right >= force_threshold, axis=1)
    right_two = np.count_nonzero(right >= force_threshold, axis=1) >= 2
    table_free = event["table_contact_force_n"].astype(np.float64) <= table_threshold
    bin_spec = environment["bin"]
    center = np.asarray(bin_spec["opening_center_world_xy_m"], dtype=np.float64)
    opening = np.asarray(bin_spec["opening_dimensions_xy_m"], dtype=np.float64)
    inside_xy = np.all(np.abs(positions[:, :2] - center) <= opening / 2.0, axis=1)
    in_bin = inside_xy & (positions[:, 2] > float(bin_spec["bottom_world_z_m"])) & (positions[:, 2] < float(bin_spec["rim_world_z_m"]))
    events = event["DIRECT_COMMON_EXECUTION_EVENTS"].astype(str)
    left_start = first_event(events, "LEFT_CLOSE_INTENT_START")
    left_owned = first_event(events, "LEFT_PHYSICAL_OWNERSHIP")
    right_start = first_event(events, "RIGHT_HANDOFF_CLOSE_INTENT_START")
    right_support = first_event(events, "RIGHT_THREE_DIGIT_SUPPORT_CONFIRMED")
    left_release = first_event(events, "GIVING_HAND_RELEASE_START")
    right_owned = first_event(events, "RIGHT_PHYSICAL_OWNERSHIP")
    release = first_event(events, "FINAL_RELEASE_INTENT_START")

    row_index = np.arange(len(frames))
    left_scope = row_index >= (left_start if left_start is not None else len(frames))
    if right_start is not None:
        left_scope &= row_index < right_start
    left_duration = longest(left_three & table_free & left_scope, dt)
    left_grasp = bool(left_owned is not None and left_duration >= retention_s - 1.0e-9)

    handoff_scope = np.zeros(len(frames), dtype=bool)
    if right_start is not None:
        end = right_owned if right_owned is not None else (release if release is not None else len(frames) - 1)
        handoff_scope[right_start:end + 1] = True
    right_three_duration = longest(right_three & table_free & handoff_scope, dt)
    transfer_rows = np.flatnonzero(handoff_scope)
    continuous_transfer = bool(
        len(transfer_rows)
        and np.all((left_any | right_any)[transfer_rows])
        and np.all(table_free[transfer_rows])
    )
    handoff = bool(
        left_grasp and right_support is not None and left_release is not None
        and right_three_duration >= 0.5 - 1.0e-9 and continuous_transfer
    )
    right_scope = row_index >= (left_release if left_release is not None else len(frames))
    if release is not None:
        right_scope &= row_index < release
    right_duration = longest(right_two & table_free & right_scope, dt)
    right_ownership = bool(handoff and right_owned is not None and right_duration >= retention_s - 1.0e-9)

    entry_candidates = np.flatnonzero(in_bin & (row_index >= (right_owned if right_owned is not None else len(frames))))
    first_entry = int(entry_candidates[0]) if len(entry_candidates) else None
    unsupported = np.empty(0, dtype=np.int64)
    no_drop = False
    if left_owned is not None and first_entry is not None:
        scope = np.arange(left_owned, first_entry, dtype=np.int64)
        unsupported = scope[~(left_any | right_any | in_bin)[scope]]
        no_drop = bool(np.all(table_free[scope]) and not len(unsupported))
    transport = bool(right_ownership and no_drop)
    bin_entry = bool(transport and first_entry is not None)
    settle_rows = max(1, int(round(1.0 / dt)))
    settle = bool(
        bin_entry and len(in_bin) >= settle_rows and np.all(in_bin[-settle_rows:])
        and np.all(speeds[-settle_rows:] <= 0.02)
        and np.max(event["doll_bin_contact_force_n"][-settle_rows:], initial=0.0) > 0.0
    )

    command_equal = bool(np.array_equal(event["EXECUTED_COMMAND"], event["commanded_q_rad"]))
    measured_equal = bool(np.array_equal(event["MEASURED_Q"], event["measured_q_rad"]))
    arm_override = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_ARM"]))
    wrist_override = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_WRIST"]))
    dex3_override = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"]))
    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    joint_names = [str(row["joint_name"]) for row in specs]
    executed = event["EXECUTED_COMMAND"].astype(np.float64)
    measured = event["MEASURED_Q"].astype(np.float64)
    policy_safe = event["POLICY_SAFE_COMMAND"].astype(np.float64)
    arm_limit_mask = event["COMMON_ARM_HARD_LIMIT_PROJECTION_MASK"].astype(bool)
    expected_projected_arm = np.clip(policy_safe[:, :14], lower[:14], upper[:14])
    expected_arm_limit_mask = np.zeros_like(arm_limit_mask, dtype=bool)
    expected_arm_limit_mask[:, :14] = expected_projected_arm != policy_safe[:, :14]
    arm_limit_mask_valid = bool(
        np.array_equal(arm_limit_mask, expected_arm_limit_mask)
    )
    arm_limit_projection_exact = bool(
        np.array_equal(executed[:, :14], expected_projected_arm)
    )
    mask_equal = bool(
        np.array_equal(
            event["COMMON_CONTROLLER_OVERRIDE_MASK"].astype(bool),
            event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"].astype(bool)
            | arm_limit_mask,
        )
    )
    arm_limit_projection_count = int(np.count_nonzero(arm_limit_mask[:, :14]))
    wrist_limit_projection_count = int(
        np.count_nonzero(arm_limit_mask[:, [4, 5, 6, 11, 12, 13]])
    )
    maximum_arm_limit_correction = float(
        np.max(np.abs(expected_projected_arm - policy_safe[:, :14]), initial=0.0)
    )
    commanded_arm_limits_valid = bool(
        np.all(executed[:, :14] >= lower[:14] - 1.0e-9)
        and np.all(executed[:, :14] <= upper[:14] + 1.0e-9)
    )
    commanded_dex3_limits_valid = bool(
        np.all(executed[:, 14:] >= lower[14:] - 1.0e-9)
        and np.all(executed[:, 14:] <= upper[14:] + 1.0e-9)
    )
    measured_arm_limits_valid = bool(
        np.all(measured[:, :14] >= lower[:14] - 0.002)
        and np.all(measured[:, :14] <= upper[:14] + 0.002)
    )
    # Dex3 is the corrected common layer.  Unlike the broad articulation
    # solver diagnostic, it is held to the authoritative measured hard bounds
    # with only floating-point comparison tolerance.
    measured_dex3_limits_valid = bool(
        np.all(measured[:, 14:] >= lower[14:] - 1.0e-6)
        and np.all(measured[:, 14:] <= upper[14:] + 1.0e-6)
    )
    joint_valid = bool(
        commanded_arm_limits_valid
        and commanded_dex3_limits_valid
        and measured_arm_limits_valid
        and measured_dex3_limits_valid
    )
    dex3_violation_mask = (measured[:, 14:] < lower[14:] - 1.0e-6) | (
        measured[:, 14:] > upper[14:] + 1.0e-6
    )
    dex3_violation_rows, dex3_violation_joints = np.nonzero(dex3_violation_mask)
    first_measured_dex3_violation = None
    if len(dex3_violation_rows):
        row = int(dex3_violation_rows[0])
        local_joint = int(dex3_violation_joints[0])
        joint = 14 + local_joint
        first_measured_dex3_violation = {
            "physics_row": row,
            "control_frame": int(frames[row]),
            "joint_index": joint,
            "joint_name": joint_names[joint],
            "measured_value_rad": float(measured[row, joint]),
            "minimum_rad": float(lower[joint]),
            "maximum_rad": float(upper[joint]),
        }
    robot_penetration = float(np.max(robot_bin.get("penetration_m", np.asarray([])), initial=0.0))
    doll_penetration = float(np.max(event["maximum_doll_bin_penetration_m"], initial=0.0))
    penetration_tolerance = float(environment["penetration"]["normal_solver_tolerance_m"])
    commands = unique_control(executed, frames)
    branch_count, maximum_arm_step = branch_discontinuities(commands)
    hard_valid = bool(
        command_equal and measured_equal and mask_equal and arm_override == 0 and wrist_override == 0
        and arm_limit_mask_valid and arm_limit_projection_exact
        and robot_penetration <= penetration_tolerance and doll_penetration <= penetration_tolerance
        and joint_valid and branch_count == 0 and np.isfinite(executed).all()
        and np.isfinite(measured).all() and bool(trial.get("command_completed", False))
        and not bool(trial.get("state_restoration", {}).get("used", False))
        and int(trial.get("object_pose_writes_during_timed_loop", -1)) == 0
    )
    full = bool(left_grasp and handoff and right_ownership and transport and bin_entry and settle and hard_valid)
    outcomes = {
        "LEFT_GRASP_SUCCESS": left_grasp, "HANDOFF_SUCCESS": handoff,
        "RIGHT_OWNERSHIP_SUCCESS": right_ownership, "NO_DROP_TO_BIN": transport,
        "BIN_ENTRY_SUCCESS": bin_entry, "BIN_SETTLE_SUCCESS": settle,
        "FULL_TASK_SUCCESS": full,
    }
    stage_values = (left_grasp, handoff, right_ownership, transport, bin_entry, settle)
    first_failure = next((name for name, passed in zip(STAGES, stage_values, strict=True) if not passed), None)
    if first_failure is None and not hard_valid:
        first_failure = "HARD_PHYSICAL_VALIDITY"

    loss_rows = np.flatnonzero(~right_any & (row_index >= (right_owned if right_owned is not None else len(frames))))
    first_loss = int(loss_rows[0]) if len(loss_rows) else None
    if first_loss is None or not right_ownership:
        release_classification = "PREMATURE_DROP_OUTSIDE_BIN"
    elif release is not None and first_loss >= release:
        release_classification = "CLEAN_COMMANDED_RELEASE"
    elif inside_xy[first_loss]:
        release_classification = "PREMATURE_DROP_INTO_BIN"
    else:
        release_classification = "PREMATURE_DROP_OUTSIDE_BIN"
    phase = event["DIRECT_COMMON_EXECUTION_PHASE"].astype(str)
    phase_audit = {}
    for value in np.unique(phase):
        scope = phase == value
        phase_audit[str(value)] = {
            "physics_rows": int(np.count_nonzero(scope)),
            "arm_override_scalars": int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_ARM"][scope])),
            "wrist_override_scalars": int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_WRIST"][scope])),
            "dex3_override_scalars": int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"][scope])),
        }
    result = {
        "schema_version": "direct_physical_eval35_run_result_v1",
        "status": "INVALID" if not hard_valid else ("PASS" if full else "FAIL"),
        "outcomes": outcomes, "first_failure_stage": first_failure,
        "release_classification": release_classification,
        "durations_s": {"left_three_digit_table_free": left_duration, "right_three_digit_handoff": right_three_duration, "right_independent_two_digit": right_duration},
        "events": {"left_close_row": left_start, "left_ownership_row": left_owned, "right_close_row": right_start, "right_support_row": right_support, "left_release_row": left_release, "right_ownership_row": right_owned, "final_release_row": release, "first_bin_entry_row": first_entry},
        "hard_physical_validity": hard_valid,
        "fairness_audit": {
            "pregrasp_classifier_used": False, "graspability_atlas_used": False,
            "wrist_distance_gate_used": False, "arm_rescue_used": False,
            "wrist_rescue_used": False, "arm_common_override_scalar_count": arm_override,
            "wrist_common_override_scalar_count": wrist_override,
            "dex3_common_override_scalar_count": dex3_override,
            "common_arm_hard_limit_projection_scalar_count": arm_limit_projection_count,
            "common_wrist_hard_limit_projection_scalar_count": wrist_limit_projection_count,
            "common_arm_hard_limit_projection_fraction": float(
                arm_limit_projection_count / max(1, len(executed) * 14)
            ),
            "maximum_common_arm_hard_limit_correction_rad": maximum_arm_limit_correction,
            "common_arm_hard_limit_projector": "NEAREST_VALID_VALUE_COMPONENTWISE",
            "common_arm_hard_limit_projector_same_for_a_b": True,
            "arm_intervention_fraction": float(arm_override / max(1, len(executed) * 8)),
            "wrist_intervention_fraction": float(wrist_override / max(1, len(executed) * 6)),
            "dex3_intervention_fraction": float(dex3_override / max(1, len(executed) * 14)),
            "phase_intervention": phase_audit,
            "raw_to_executed_arm_rmse_rad": float(np.sqrt(np.mean((event["RAW_POLICY_COMMAND"][:, :14] - executed[:, :14]) ** 2))),
            "raw_to_executed_dex3_rmse_rad": float(np.sqrt(np.mean((event["RAW_POLICY_COMMAND"][:, 14:] - executed[:, 14:]) ** 2))),
        },
        "diagnostics": {
            "unsupported_rows_before_bin_count": int(len(unsupported)),
            "maximum_robot_bin_penetration_m": robot_penetration,
            "maximum_doll_bin_penetration_m": doll_penetration,
            "penetration_tolerance_m": penetration_tolerance,
            "peak_held_object_speed_m_s": float(np.max(speeds[:first_entry] if first_entry is not None else speeds, initial=0.0)),
            "maximum_command_measured_q_error_rad": float(np.max(np.abs(executed - measured), initial=0.0)),
            "joint_limits_valid": joint_valid,
            "commanded_arm_limits_valid": commanded_arm_limits_valid,
            "commanded_dex3_limits_valid": commanded_dex3_limits_valid,
            "measured_arm_limits_valid": measured_arm_limits_valid,
            "measured_dex3_limits_valid": measured_dex3_limits_valid,
            "measured_dex3_hard_limit_violation_scalar_count": int(len(dex3_violation_rows)),
            "first_measured_dex3_hard_limit_violation": first_measured_dex3_violation,
            "branch_discontinuity_count": branch_count,
            "maximum_command_arm_step_l2_rad": maximum_arm_step,
            "command_completed": bool(trial.get("command_completed", False)),
            "arm_hard_limit_projection_mask_valid": arm_limit_mask_valid,
            "arm_hard_limit_projection_exact": arm_limit_projection_exact,
        },
    }
    atomic_json(run / "DIRECT_PHYSICAL_TASK_RESULT.json", result)
    (run / "DIRECT_PHYSICAL_TASK_RESULT.md").write_text(
        "# Direct contact-constrained physical task result\n\n"
        f"Status: **{result['status']}**  \nFirst failure: **{first_failure or 'NONE'}**\n\n"
        + "\n".join(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in outcomes.items())
        + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if not hard_valid else 0


if __name__ == "__main__":
    raise SystemExit(main())
