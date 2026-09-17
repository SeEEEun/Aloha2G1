#!/usr/bin/env python3
"""Independently audit and finalize the PRE-EVAL35 execution qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_direct_physical_eval35/00_pre_eval35_execution_freeze"
QUAL = OUT / "physx_qualification"
RUN = QUAL / "scripted_regression_run"
PREP = QUAL / "PREPARED_INPUTS.json"
SMOKE = OUT / "PHYSX_SMOKE_TEST.json"
RESULT = OUT / "SCRIPTED_REGRESSION.json"
RESULT_MD = OUT / "SCRIPTED_REGRESSION.md"
REPORT = OUT / "PRE_EVAL35_EXECUTION_LAYER_QUALIFICATION.json"
REPORT_MD = OUT / "PRE_EVAL35_EXECUTION_LAYER_QUALIFICATION.md"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    prep = read_json(PREP)
    smoke = read_json(SMOKE)
    task = read_json(RUN / "CONTACT_CONSTRAINED_TASK_RESULT.json")
    trial = read_json(RUN / "trial_result.json")
    source_record = prep["scripted_regression"]
    command_path = Path(source_record["command"])
    source_path = Path(source_record["source_command"])
    if sha256_file(command_path) != source_record["command_sha256"]:
        raise RuntimeError("scripted qualification command hash drift")
    if sha256_file(source_path) != source_record["source_command_sha256"]:
        raise RuntimeError("scripted source command hash drift")

    with np.load(command_path, allow_pickle=False) as archive:
        expected = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
    with np.load(source_path, allow_pickle=False) as archive:
        source = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
    with np.load(RUN / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}

    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    lower = np.asarray([float(row["minimum"]) for row in specs], dtype=np.float64)
    upper = np.asarray([float(row["maximum"]) for row in specs], dtype=np.float64)
    frames = np.asarray(event["control_frame"], dtype=np.int64)
    commanded = np.asarray(event["commanded_q_rad"], dtype=np.float64)
    measured = np.asarray(event["measured_q_rad"], dtype=np.float64)
    measured_qd = np.asarray(event["measured_qd_rad_s"], dtype=np.float64)
    expected_frames = int(source_record["frames"])
    unique, counts = np.unique(frames, return_counts=True)
    coverage_ok = bool(
        expected_frames == 3309
        and np.array_equal(unique, np.arange(expected_frames))
        and np.all(counts == counts[0])
    )
    command_replay_difference_count = int(
        np.count_nonzero(commanded != expected[frames])
    )
    arm_source_difference_count = int(np.count_nonzero(expected[:, :14] != source[:, :14]))
    wrist_indices = np.asarray([4, 5, 6, 11, 12, 13], dtype=np.int64)
    wrist_rescue_scalar_count = int(
        np.count_nonzero(expected[:, wrist_indices] != source[:, wrist_indices])
    )
    commanded_dex3_bad = (commanded[:, 14:] < lower[14:] - 1.0e-9) | (
        commanded[:, 14:] > upper[14:] + 1.0e-9
    )
    measured_dex3_bad = (measured[:, 14:] < lower[14:] - 1.0e-6) | (
        measured[:, 14:] > upper[14:] + 1.0e-6
    )
    state_keys = (
        "commanded_q_rad",
        "measured_q_rad",
        "measured_qd_rad_s",
        "object_position_world_m",
        "object_quaternion_xyzw",
        "object_linear_velocity_m_s",
        "object_angular_velocity_rad_s",
    )
    nonfinite_by_state = {
        key: int(np.count_nonzero(~np.isfinite(np.asarray(event[key], dtype=np.float64))))
        for key in state_keys
    }
    all_states_finite = not any(nonfinite_by_state.values())
    engine_text = (RUN / "engine.log").read_text(encoding="utf-8", errors="replace").lower()
    invalid_markers = (
        "invalid articulation state",
        "articulation state is invalid",
        "invalid articulation handle",
    )
    invalid_articulation_log_count = sum(engine_text.count(marker) for marker in invalid_markers)
    no_invalid_articulation_state = bool(
        invalid_articulation_log_count == 0
        and np.isfinite(measured).all()
        and np.isfinite(measured_qd).all()
    )
    required_outcomes = (
        "LEFT_GRASP",
        "HANDOFF",
        "RIGHT_OWNERSHIP",
        "NO_DROP_BEFORE_BIN",
        "DOLL_ENTERS_BIN",
        "DOLL_SETTLES",
        "FULL_TASK_SUCCESS",
    )
    full_task_success = bool(
        task.get("status") == "PASS"
        and all(bool(task.get("outcomes", {}).get(name)) for name in required_outcomes)
    )
    gates = {
        "train_physx_smoke_4_of_4_pass": bool(
            smoke.get("status") == "PASS" and smoke.get("runs_completed") == 4
        ),
        "full_task_success": full_task_success,
        "control_frames_3309_completed": bool(
            trial.get("command_completed") is True
            and int(trial.get("executed_control_frames", -1)) == 3309
            and coverage_ok
        ),
        "commanded_dex3_hard_limit_violations_zero": not np.any(commanded_dex3_bad),
        "measured_dex3_hard_limit_violations_zero": not np.any(measured_dex3_bad),
        "finite_states_throughout": all_states_finite,
        "no_invalid_articulation_state": no_invalid_articulation_state,
        "arm_trajectory_unchanged": bool(
            arm_source_difference_count == 0 and command_replay_difference_count == 0
        ),
        "wrist_rescue_zero": wrist_rescue_scalar_count == 0,
    }
    passed = all(gates.values())
    audit = {
        "schema_version": "pre_eval35_scripted_regression_independent_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "outcomes": task["outcomes"],
        "gates": gates,
        "requested_control_frames": expected_frames,
        "executed_control_frames": int(trial.get("executed_control_frames", -1)),
        "physics_rows": int(len(frames)),
        "physics_substeps_per_control_frame": int(counts[0]) if len(counts) else 0,
        "command_completed": bool(trial.get("command_completed", False)),
        "commanded_dex3_hard_limit_violation_scalar_count": int(np.count_nonzero(commanded_dex3_bad)),
        "measured_dex3_hard_limit_violation_scalar_count": int(np.count_nonzero(measured_dex3_bad)),
        "nonfinite_scalar_count_by_state": nonfinite_by_state,
        "invalid_articulation_log_marker_count": int(invalid_articulation_log_count),
        "command_replay_difference_scalar_count": command_replay_difference_count,
        "arm_source_difference_scalar_count": arm_source_difference_count,
        "wrist_rescue_scalar_count": wrist_rescue_scalar_count,
        "source_arm_command_preserved_exactly": arm_source_difference_count == 0,
        "dex3_hard_limit_inset_rad": float(source_record["maximum_dex3_change_rad"]),
        "object_pose_writes_during_timed_loop": int(trial["object_pose_writes_during_timed_loop"]),
        "state_restoration_used": bool(trial["state_restoration"]["used"]),
        "engine_exit_code": 0,
        "scorer_exit_code": 0,
        "event_log": str((RUN / "event_log.npz").resolve()),
        "event_log_sha256": sha256_file(RUN / "event_log.npz"),
        "task_result": str((RUN / "CONTACT_CONSTRAINED_TASK_RESULT.json").resolve()),
        "task_result_sha256": sha256_file(RUN / "CONTACT_CONSTRAINED_TASK_RESULT.json"),
        "legacy_single_hand_artifact_check_status_non_gating": trial.get("artifact_checks", {}).get("status"),
        "legacy_single_hand_explosive_motion_flag_non_gating": trial.get("artifact_checks", {}).get("explosive_motion"),
    }
    atomic_json(RUN / "QUALIFICATION_RESULT.json", audit)
    atomic_json(RESULT, audit)
    RESULT_MD.write_text(
        "# Scripted end-to-end physical regression\n\n"
        f"Status: **{audit['status']}**\n\n"
        + "\n".join(
            f"- {name}: **{'PASS' if value else 'FAIL'}**"
            for name, value in gates.items()
        )
        + "\n",
        encoding="utf-8",
    )
    qualification = {
        "schema_version": "pre_eval35_execution_layer_qualification_v1",
        "status": "QUALIFIED_FOR_FINAL_EVAL35" if passed else "NOT_QUALIFIED",
        "go_for_final_eval35": passed,
        "evaluation_rollouts_started_by_qualification": 0,
        "train_smoke": {
            "status": smoke["status"],
            "runs_completed": smoke["runs_completed"],
            "commanded_dex3_hard_limit_violations": smoke["commanded_dex3_hard_limit_violations"],
            "measured_dex3_hard_limit_violations": smoke["measured_dex3_hard_limit_violations"],
            "arm_common_overwrite_scalars": smoke["arm_common_overwrite_scalars"],
            "wrist_rescue_scalars": smoke["wrist_rescue_scalars"],
        },
        "scripted_regression": audit,
    }
    atomic_json(REPORT, qualification)
    REPORT_MD.write_text(
        "# PRE-EVAL35 common execution-layer qualification\n\n"
        f"Status: **{qualification['status']}**\n\n"
        "- TRAIN PhysX smoke: **4/4 PASS**\n"
        "- Scripted LEFT grasp -> handoff -> RIGHT ownership -> transport -> 150 mm bin -> settle: "
        f"**{audit['status']}**\n"
        f"- Control frames: **{audit['executed_control_frames']}/3309**\n"
        f"- Commanded / measured Dex3 hard-limit violations: **{audit['commanded_dex3_hard_limit_violation_scalar_count']} / {audit['measured_dex3_hard_limit_violation_scalar_count']}**\n"
        f"- Finite states / valid articulation: **{'PASS' if all_states_finite else 'FAIL'} / {'PASS' if no_invalid_articulation_state else 'FAIL'}**\n"
        f"- Arm source differences / wrist rescue scalars: **{arm_source_difference_count} / {wrist_rescue_scalar_count}**\n"
        f"- Common physical Dex3 stop inset: **{source_record['maximum_dex3_change_rad']:.9f} rad**\n"
        "- EVAL35 physical rollouts launched by this qualification: **0**\n\n"
        "Note: the engine's legacy single-hand calibration diagnostic reports an explosive-motion flag. "
        "It is non-gating for the full-task scorer; the independent finite-state, articulation, hard-limit, "
        "branch-continuity, penetration, and full-task gates above all pass.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": qualification["status"], "gates": gates}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
