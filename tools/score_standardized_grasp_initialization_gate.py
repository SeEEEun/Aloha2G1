#!/usr/bin/env python3
"""Score the fresh-reset standardized grasp without counting task outcome."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/standardized_grasp_ab_dev35/00_control/physical_initialization_gate"
OUT = ROOT / "outputs/standardized_grasp_ab_dev35/00_control/INITIALIZATION_PHYSICAL_GATE.json"
CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    trial = json.loads((RUN / "trial_result.json").read_text())
    runtime = json.loads((RUN / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json").read_text())
    contract = json.loads(CONTRACT.read_text())
    specs = sorted(contract["joint_specs"], key=lambda row: int(row["index"]))
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    with np.load(RUN / "event_log.npz", allow_pickle=False) as trace:
        control = np.asarray(trace["control_frame"], dtype=np.int64)
        samples = np.r_[np.flatnonzero(np.diff(control)), len(control) - 1]
        retained = samples[15:]
        forces = np.column_stack([
            trace["left_thumb_force_n"], trace["left_index_force_n"], trace["left_middle_force_n"]
        ]).astype(np.float64)
        executed = np.asarray(trace["EXECUTED_COMMAND"], dtype=np.float64)
        measured = np.asarray(trace["MEASURED_Q"], dtype=np.float64)
        object_position = np.asarray(trace["object_position_world_m"], dtype=np.float64)
        table = np.asarray(trace["table_contact_force_n"], dtype=np.float64)
        commanded_violations = int(np.count_nonzero((executed < lower) | (executed > upper)))
        measured_violations = int(np.count_nonzero((measured < lower - 1e-6) | (measured > upper + 1e-6)))
        sustained = np.mean(forces[retained] >= 0.05, axis=0)
        displacement = np.linalg.norm(object_position - object_position[0], axis=1)
        finite = bool(all(np.isfinite(value).all() for value in (executed, measured, object_position)))
    registration = trial["object_task_frame_registration"]["runtime_initial_pose_verification"]
    checks = {
        "exact_pre_frame0_registration": bool(registration["within_0_1_mm_and_0_1_deg"]),
        "real_physx_three_digit_contact": bool(np.all(sustained >= 0.95)),
        "table_free": bool(np.max(table[retained], initial=0.0) <= 0.05),
        "retained_for_at_least_2_5_s": bool(len(retained) >= 75),
        "bounded_object_motion_under_hold": bool(np.max(displacement, initial=0.0) <= 0.025),
        "commanded_hard_limits": commanded_violations == 0,
        "measured_hard_limits": measured_violations == 0,
        "finite": finite,
        "no_attachment": trial["prohibited_attachment_used"] is False,
        "zero_pose_writes": int(trial["object_pose_writes_during_timed_loop"]) == 0,
        "left_controller_state_hold": runtime["final_grasp_state"]["left"] == "HOLD",
        "arm_rescue_zero": int(runtime["arm_common_override_scalar_count"]) == 0,
        "wrist_rescue_zero": int(runtime["wrist_common_override_scalar_count"]) == 0,
    }
    result = {
        "schema_version": "standardized_grasp_physical_initialization_gate_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "CONTROL_CONDITION_ONLY_NOT_A_OR_B_TASK_OUTCOME",
        "checks": checks,
        "sustained_contact_fraction": dict(zip(("thumb", "index", "middle"), map(float, sustained))),
        "maximum_object_displacement_mm": 1000.0 * float(np.max(displacement)),
        "minimum_object_z_m": float(np.min(object_position[:, 2])),
        "maximum_table_force_n": float(np.max(table)),
        "commanded_hard_limit_violations": commanded_violations,
        "measured_hard_limit_violations": measured_violations,
        "object_pose_writes_after_initialization": int(trial["object_pose_writes_during_timed_loop"]),
        "runtime_initial_pose_verification": registration,
        "artifacts": {
            name: {"path": str((RUN / name).resolve()), "sha256": sha256(RUN / name)}
            for name in ("event_log.npz", "trial_result.json", "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.incomplete")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
