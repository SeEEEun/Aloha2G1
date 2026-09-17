#!/usr/bin/env python3
"""Consolidate the B01 named-mapping, isolation, and solver audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_episode_registered_eval35"
AUDIT = OUT / "00_forensic_audit"
OLD_B01 = (
    OUT
    / "provenance_common_execution_bug_solver32/03_act_b_results/rollouts/"
    "eval_00_doll_handoff_20260820_ep002"
)
ZERO = AUDIT / "dex3_zero_contact/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"
DOLL_ISOLATED = AUDIT / "b01_doll_isolated_replay/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"
TABLE_ISOLATED = AUDIT / "b01_doll_table_isolated_replay/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"
SOLVER64 = AUDIT / "b01_solver64x4_replay/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"
SOLVER80 = AUDIT / "solver80_contact_replay/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"
SOLVER128 = AUDIT / "b01_solver128x4_replay/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"
CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    zero = read(ZERO)
    doll = read(DOLL_ISOLATED)
    table = read(TABLE_ISOLATED)
    solver64 = read(SOLVER64)
    solver80 = read(SOLVER80)
    solver128 = read(SOLVER128)
    contract = read(CONTRACT)
    trial = read(OLD_B01 / "trial_result.json")
    result = read(OLD_B01 / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json")
    with np.load(OLD_B01 / "event_log.npz", allow_pickle=False) as event:
        names = event["joint_names"].astype(str).tolist()
        measured = np.asarray(event["MEASURED_Q"], dtype=np.float64)
        command = np.asarray(event["EXECUTED_COMMAND"], dtype=np.float64)
        target_index = names.index("left_hand_middle_1_joint")
        maximum_row = int(np.argmax(measured[:, target_index]))

    mapping = next(
        row for row in zero["mapping"]
        if row["project_joint_name"] == "left_hand_middle_1_joint"
    )
    sweep = next(
        row for row in zero["per_joint"]
        if row["joint"] == "left_hand_middle_1_joint"
    )
    limit_rows = []
    for mapping_row, sweep_row in zip(zero["mapping"], zero["per_joint"], strict=True):
        limit_rows.append(
            {
                "joint": mapping_row["project_joint_name"],
                "policy_index": mapping_row["policy_action_index"],
                "physx_dof_index": mapping_row["physx_articulation_dof_index"],
                "project_limit_rad": [mapping_row["project_lower_rad"], mapping_row["project_upper_rad"]],
                "source_usd_limit_rad": [mapping_row["source_usd_lower_rad"], mapping_row["source_usd_upper_rad"]],
                "safety_stop_rad": [mapping_row["safety_lower_rad"], mapping_row["safety_upper_rad"]],
                "physx_runtime_limit_rad": [mapping_row["physx_runtime_lower_rad"], mapping_row["physx_runtime_upper_rad"]],
                "mapping_pass": sweep_row["mapping_pass"],
                "sign_pass": sweep_row["sign_pass"],
                "readback_pass": sweep_row["readback_pass"],
                "runtime_hard_limit_pass": sweep_row["runtime_limit_pass"],
            }
        )

    report = {
        "schema_version": "final_eval35_b01_dex3_articulation_forensic_v1",
        "status": "ROOT_CAUSE_CONFIRMED_COMMON_PHYSICAL_ARTICULATION_BUG",
        "fix_classification": "TYPE 3",
        "act_a_rerun_required": True,
        "b01_invalid_run_counted_as_task_failure": False,
        "joint_stack_mapping": {
            "authoritative_project_joint_name": "left_hand_middle_1_joint",
            "authoritative_project_limits_rad": [mapping["project_lower_rad"], mapping["project_upper_rad"]],
            "policy_action_vector_index": mapping["policy_action_index"],
            "prepared_command_archive_index": mapping["prepared_archive_index"],
            "execution_layer_vector_index": mapping["execution_layer_index"],
            "usd_articulation_dof_name": mapping["project_joint_name"],
            "usd_joint_prim_path": mapping["usd_joint_prim_path"],
            "physx_articulation_dof_index": mapping["physx_articulation_dof_index"],
            "drive_target_at_maximum_rad": float(command[maximum_row, target_index]),
            "drive_axis": mapping["usd_drive_axis"],
            "sign_convention": "negative flexion; zero-contact command/readback slope is positive",
            "zero_contact_sign_slope": sweep["sign_slope"],
            "runtime_limits_rad": [mapping["physx_runtime_lower_rad"], mapping["physx_runtime_upper_rad"]],
            "measured_state_readback_index": mapping["measured_state_readback_index"],
            "reported_state_label": mapping["reported_state_label"],
        },
        "original_b01": {
            "maximum_measured_rad": float(measured[maximum_row, target_index]),
            "maximum_measured_row": maximum_row,
            "maximum_measured_control_frame": int(maximum_row // 8),
            "command_at_maximum_rad": float(command[maximum_row, target_index]),
            "measured_violating_samples": int(result["integrity"]["measured_dex3_hard_limit_violation_scalar_count"]),
            "commanded_violations": int(result["integrity"]["commanded_hard_limit_violation_scalar_count"]),
            "doll_contact_at_excursion": "NONE",
            "task_status": "INVALID_PHYSICS_NOT_A_TASK_FAILURE",
        },
        "hypothesis_tests": {
            "left_right_swap": False,
            "index_middle_swap": False,
            "joint_index_offset": False,
            "stale_ordering": False,
            "duplicated_dof_index": False,
            "wrong_state_readback_index": False,
            "command_readback_mismatch": False,
            "sign_inversion": False,
            "degrees_radians_mismatch": False,
            "usd_urdf_naming_mismatch": False,
            "wrong_joint_axis": False,
            "runtime_physx_limit_mismatch": False,
            "drive_target_sign_mismatch": False,
            "state_tensor_indexing_mismatch": False,
            "under_resolved_external_table_contact_at_hard_stop": True,
        },
        "isolation_evidence": {
            "doll_isolated_reproduced_violation_samples": doll["replay"]["authoritative_limit_violation_samples"],
            "doll_isolated_maximum_measured_rad": doll["replay"]["maximum_measured_rad"],
            "doll_and_table_isolated_violation_samples": table["replay"]["authoritative_limit_violation_samples"],
            "doll_and_table_isolated_maximum_measured_rad": table["replay"]["maximum_measured_rad"],
            "conclusion": "the external hand-table contact impulse, not doll contact or free-space mapping, triggers the stop excursion",
        },
        "solver_fix_evidence": {
            "source_solver": {"position_iterations": 32, "velocity_iterations": 1},
            "tested_64x4_violation_samples": solver64["replay"]["authoritative_limit_violation_samples"],
            "tested_64x4_maximum_measured_rad": solver64["replay"]["maximum_measured_rad"],
            "selected_solver": {"position_iterations": 80, "velocity_iterations": 4},
            "selected_80x4_violation_samples": solver80["replay"]["authoritative_limit_violation_samples"],
            "selected_80x4_maximum_measured_rad": solver80["replay"]["maximum_measured_rad"],
            "selected_128x4_violation_samples": solver128["replay"]["authoritative_limit_violation_samples"],
            "selected_128x4_maximum_measured_rad": solver128["replay"]["maximum_measured_rad"],
            "selection_rule": "lowest tested setting that both contains the reproduced B01 table impulse inside the authoritative hard limit and passes the unchanged non-EVAL scripted physical-validity gates",
            "commands_changed": False,
            "collision_changed": False,
            "trajectory_changed": False,
        },
        "zero_contact_summary": zero["summary"],
        "all_14_joint_limits": limit_rows,
        "sources": {
            "old_b01": str(OLD_B01.resolve()),
            "old_b01_trace_sha256": sha(OLD_B01 / "event_log.npz"),
            "joint_contract": str(CONTRACT.resolve()),
            "joint_contract_sha256": sha(CONTRACT),
            "zero_contact_audit": str(ZERO.resolve()),
            "doll_isolated_replay": str(DOLL_ISOLATED.resolve()),
            "table_isolated_replay": str(TABLE_ISOLATED.resolve()),
            "solver64_replay": str(SOLVER64.resolve()),
            "solver80_replay": str(SOLVER80.resolve()),
            "solver128_replay": str(SOLVER128.resolve()),
        },
    }
    write(AUDIT / "DEX3_ARTICULATION_FORENSIC_AUDIT.json", json.dumps(report, indent=2, sort_keys=True) + "\n")

    md = [
        "# B01 Dex3 articulation forensic audit",
        "",
        "Status: **ROOT CAUSE CONFIRMED — TYPE 3 COMMON PHYSICAL ARTICULATION BUG**",
        "",
        "The named zero-contact sweep passes all 14 joints for mapping, sign, readback, and runtime limits. B01's `left_hand_middle_1_joint` is policy/archive/execution index 18 and PhysX DOF 36 on the authored Z-axis revolute joint. Therefore the +0.320730 rad sample is a physical articulation excursion, not a mislabeled tensor element.",
        "",
        "Removing the doll alone reproduced the limit crossing. Removing the table as well eliminated it. The causal load is therefore hand–table contact, while the defect is that the source 32/1 articulation solver did not keep the revolute joint within its hard stop under that contact impulse.",
        "",
        "A bounded solver enforcement audit found 64/4 still produced two authoritative-limit violations (+0.004223 rad maximum), while 80/4 produced zero (-0.002687 rad maximum) and passed the unchanged non-EVAL scripted task with 1.911 mm maximum doll/bin penetration. No command, trajectory, collision geometry, registration, controller, or task scorer changed.",
        "",
        "Because the fix changes the common PhysX articulation solver, the previous ACT-A35 is preserved as provenance but labeled `SUPERSEDED_COMMON_EXECUTION_BUG`; a complete matched A35+B35 rerun is required.",
        "",
        "## 14-joint sweep",
        "",
        f"- Mapping: {zero['summary']['mapping_pass_count']} / 14 PASS",
        f"- Sign: {zero['summary']['sign_pass_count']} / 14 PASS",
        f"- Readback: {zero['summary']['readback_pass_count']} / 14 PASS",
        f"- Runtime hard limits: {zero['summary']['runtime_hard_limit_pass_count']} / 14 PASS",
    ]
    write(AUDIT / "DEX3_ARTICULATION_FORENSIC_AUDIT.md", "\n".join(md) + "\n")
    print(json.dumps({"status": report["status"], "fix_type": report["fix_classification"], "act_a_rerun": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
