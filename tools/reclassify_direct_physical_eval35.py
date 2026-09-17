#!/usr/bin/env python3
"""Apply the frozen direct-task criterion to an already executed physical trace.

The original scorer incorrectly treated an out-of-range articulation *target*
as a corrupted robot state even when PhysX resolved the target at the hard joint
constraint.  The final direct-evaluation protocol explicitly says that a bad
policy arm configuration is an ordinary physical outcome.  This additive
scorer preserves the original result and records target-limit excess as a
diagnostic; only the measured state is used for robot-state validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ENVIRONMENT = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
STAGES = (
    ("LEFT_GRASP", "LEFT_GRASP_SUCCESS"),
    ("HANDOFF", "HANDOFF_SUCCESS"),
    ("RIGHT_OWNERSHIP", "RIGHT_OWNERSHIP_SUCCESS"),
    ("TRANSPORT", "NO_DROP_TO_BIN"),
    ("BIN_ENTRY", "BIN_ENTRY_SUCCESS"),
    ("SETTLE", "BIN_SETTLE_SUCCESS"),
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


def first_violation(
    bad: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    names: list[str],
    event: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    rows, joints = np.nonzero(bad)
    if not len(rows):
        return None
    row, joint = int(rows[0]), int(joints[0])
    value = float(values[row, joint])
    boundary = float(lower[joint] if value < lower[joint] else upper[joint])
    return {
        "event_row": row,
        "physics_step": int(event["physics_step"][row]),
        "control_frame": int(event["control_frame"][row]),
        "stage": str(event["stage"][row]),
        "joint_index": joint,
        "joint": names[joint],
        "value_rad": value,
        "hard_limit_rad": boundary,
        "signed_excess_rad": value - boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    original = read_json(run / "DIRECT_PHYSICAL_TASK_RESULT.json")
    trial = read_json(run / "trial_result.json")
    environment = read_json(ENVIRONMENT)
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(run / "robot_bin_contacts.npz", allow_pickle=False) as archive:
        robot_bin = {key: np.asarray(archive[key]) for key in archive.files}
    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    names = [str(row["joint_name"]) for row in specs]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    executed = event["EXECUTED_COMMAND"].astype(np.float64)
    measured = event["MEASURED_Q"].astype(np.float64)
    target_bad = (executed < lower - 1.0e-9) | (executed > upper + 1.0e-9)
    measured_bad = (measured < lower - 0.002) | (measured > upper + 0.002)
    target_excess = np.maximum(lower - executed, executed - upper)
    measured_excess = np.maximum(lower - measured, measured - upper)
    arm_override = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_ARM"]))
    wrist_override = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_WRIST"]))
    mask_equal = bool(
        np.array_equal(
            event["COMMON_CONTROLLER_OVERRIDE_MASK"],
            event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"],
        )
    )
    command_equal = bool(
        np.array_equal(event["EXECUTED_COMMAND"], event["commanded_q_rad"])
    )
    measured_equal = bool(np.array_equal(event["MEASURED_Q"], event["measured_q_rad"]))
    tolerance = float(environment["penetration"]["normal_solver_tolerance_m"])
    robot_penetration = float(
        np.max(robot_bin.get("penetration_m", np.asarray([])), initial=0.0)
    )
    doll_penetration = float(
        np.max(event["maximum_doll_bin_penetration_m"], initial=0.0)
    )
    branch_count = int(original["diagnostics"]["branch_discontinuity_count"])
    measured_state_valid = not bool(np.any(measured_bad))
    hard_valid = bool(
        command_equal
        and measured_equal
        and mask_equal
        and arm_override == 0
        and wrist_override == 0
        and robot_penetration <= tolerance
        and doll_penetration <= tolerance
        and measured_state_valid
        and branch_count == 0
        and np.isfinite(executed).all()
        and np.isfinite(measured).all()
        and bool(trial.get("command_completed", False))
        and not bool(trial.get("state_restoration", {}).get("used", False))
        and int(trial.get("object_pose_writes_during_timed_loop", -1)) == 0
    )
    outcomes = dict(original["outcomes"])
    physical_stages_pass = all(bool(outcomes[key]) for _, key in STAGES)
    outcomes["FULL_TASK_SUCCESS"] = bool(physical_stages_pass and hard_valid)
    first_failure = next(
        (name for name, key in STAGES if not bool(outcomes[key])), None
    )
    if first_failure is None and not hard_valid:
        first_failure = "HARD_PHYSICAL_VALIDITY"
    diagnostics = dict(original["diagnostics"])
    diagnostics.update(
        {
            "joint_limits_valid": measured_state_valid,
            "measured_joint_state_limits_valid": measured_state_valid,
            "policy_target_limit_violation_count": int(np.count_nonzero(target_bad)),
            "policy_target_limit_violation_joint_count": int(
                np.count_nonzero(np.any(target_bad, axis=0))
            ),
            "policy_target_limit_violation_joints": [
                names[index]
                for index in np.flatnonzero(np.any(target_bad, axis=0))
            ],
            "first_policy_target_limit_violation": first_violation(
                target_bad, executed, lower, upper, names, event
            ),
            "maximum_policy_target_limit_excess_rad": float(
                np.max(target_excess, initial=0.0)
            ),
            "first_measured_joint_limit_violation": first_violation(
                measured_bad, measured, lower, upper, names, event
            ),
            "maximum_measured_joint_limit_excess_rad": float(
                np.max(measured_excess, initial=0.0)
            ),
            "target_limit_semantics": (
                "diagnostic policy behavior; PhysX hard-limit-constrained measured "
                "state determines invalid robot state"
            ),
        }
    )
    result = dict(original)
    result.update(
        {
            "schema_version": "direct_physical_eval35_run_result_v2",
            "status": "INVALID" if not hard_valid else (
                "PASS" if outcomes["FULL_TASK_SUCCESS"] else "FAIL"
            ),
            "outcomes": outcomes,
            "first_failure_stage": first_failure,
            "hard_physical_validity": hard_valid,
            "diagnostics": diagnostics,
            "scoring_harness_correction": (
                "policy target outside a hard limit is retained and reported; "
                "a PhysX-constrained valid measured state is an ordinary physical outcome"
            ),
            "original_result_preserved": str(
                (run / "DIRECT_PHYSICAL_TASK_RESULT.json").resolve()
            ),
        }
    )
    output = run / "DIRECT_PHYSICAL_TASK_RESULT_V2.json"
    atomic_json(output, result)
    (run / "DIRECT_PHYSICAL_TASK_RESULT_V2.md").write_text(
        "# Direct contact-constrained physical task result — scoring correction\n\n"
        f"Status: **{result['status']}**  \n"
        f"First failure: **{first_failure or 'NONE'}**\n\n"
        "The original result is preserved. Policy target-limit excess is a "
        "diagnostic; measured PhysX state validity governs invalid-run status.\n\n"
        + "\n".join(
            f"- {key}: `{'PASS' if value else 'FAIL'}`"
            for key, value in outcomes.items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if not hard_valid else 0


if __name__ == "__main__":
    raise SystemExit(main())
