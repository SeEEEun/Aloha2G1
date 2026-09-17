#!/usr/bin/env python3
"""Run fixed TRAIN40 smoke and scripted regression; never execute EVAL35."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
OUT = ROOT / "outputs/final_direct_physical_eval35/00_pre_eval35_execution_freeze"
QUAL = OUT / "physx_qualification"
PREP = QUAL / "PREPARED_INPUTS.json"
PROVISIONAL = QUAL / "PROVISIONAL_EXECUTION_MANIFEST.json"
DIRECT_LAUNCHER = ROOT / "tools/run_direct_physical_execution_isaac.py"
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
SCORER = ROOT / "tools/score_contact_constrained_full_task.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
SMOKE_JSON = OUT / "PHYSX_SMOKE_TEST.json"
SMOKE_MD = OUT / "PHYSX_SMOKE_TEST.md"
REGRESSION_JSON = OUT / "SCRIPTED_REGRESSION.json"
REGRESSION_MD = OUT / "SCRIPTED_REGRESSION.md"


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


def run_process(command: list[str], output: Path) -> int:
    with output.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False
        )
    return int(process.returncode)


def ranges() -> tuple[np.ndarray, np.ndarray, list[str]]:
    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    return (
        np.asarray([float(row["minimum"]) for row in specs]),
        np.asarray([float(row["maximum"]) for row in specs]),
        [str(row["joint_name"]) for row in specs],
    )


def analyze_direct(run: Path, method: str, stable: str) -> dict[str, Any]:
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    trial = read_json(run / "trial_result.json")
    summary = read_json(run / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")
    lower, upper, names = ranges()
    executed = np.asarray(event["EXECUTED_COMMAND"], dtype=np.float64)
    measured = np.asarray(event["MEASURED_Q"], dtype=np.float64)
    dex_command_bad = (executed[:, 14:] < lower[14:] - 1.0e-9) | (executed[:, 14:] > upper[14:] + 1.0e-9)
    dex_measured_bad = (measured[:, 14:] < lower[14:] - 1.0e-6) | (measured[:, 14:] > upper[14:] + 1.0e-6)
    bad_rows, bad_joints = np.nonzero(dex_measured_bad)
    first_bad = None
    if len(bad_rows):
        row, local = int(bad_rows[0]), int(bad_joints[0])
        joint = 14 + local
        first_bad = {
            "physics_row": row,
            "control_frame": int(event["control_frame"][row]),
            "joint_name": names[joint],
            "measured_rad": float(measured[row, joint]),
            "lower_rad": float(lower[joint]),
            "upper_rad": float(upper[joint]),
        }
    events = event["DIRECT_COMMON_EXECUTION_EVENTS"].astype(str)
    joined = "|".join(events.tolist())
    initial = measured[0]
    left_change = float(np.max(np.abs(measured[:, 14:21] - initial[14:21]), initial=0.0))
    right_change = float(np.max(np.abs(measured[:, 21:28] - initial[21:28]), initial=0.0))
    finite = bool(np.isfinite(executed).all() and np.isfinite(measured).all())
    valid = bool(
        finite
        and not np.any(dex_command_bad)
        and not np.any(dex_measured_bad)
        and summary.get("arm_common_override_scalar_count") == 0
        and summary.get("wrist_common_override_scalar_count") == 0
        and bool(trial.get("command_completed", False))
        and int(trial.get("object_pose_writes_during_timed_loop", -1)) == 0
        and not bool(trial.get("state_restoration", {}).get("used", False))
        and "LEFT_CLOSE_INTENT_START" in joined
        and "RIGHT_HANDOFF_CLOSE_INTENT_START" in joined
        and "FINAL_RELEASE_INTENT_START" in joined
        and left_change > 0.05
        and right_change > 0.05
    )
    return {
        "method_stream": method,
        "stable_episode_id": stable,
        "status": "PASS" if valid else "FAIL",
        "evaluation_episode": False,
        "command_completed": bool(trial.get("command_completed", False)),
        "executed_physics_rows": len(executed),
        "commanded_dex3_hard_limit_violation_scalar_count": int(np.count_nonzero(dex_command_bad)),
        "measured_dex3_hard_limit_violation_scalar_count": int(np.count_nonzero(dex_measured_bad)),
        "first_measured_dex3_hard_limit_violation": first_bad,
        "arm_common_override_scalar_count": int(summary.get("arm_common_override_scalar_count", -1)),
        "wrist_common_override_scalar_count": int(summary.get("wrist_common_override_scalar_count", -1)),
        "left_dex3_measured_motion_max_rad": left_change,
        "right_dex3_measured_motion_max_rad": right_change,
        "left_close_executed": "LEFT_CLOSE_INTENT_START" in joined,
        "right_handoff_close_executed": "RIGHT_HANDOFF_CLOSE_INTENT_START" in joined,
        "giver_release_started_only_if_support_gate_passed": (
            "GIVING_HAND_RELEASE_START" not in joined or "RIGHT_THREE_DIGIT_SUPPORT_CONFIRMED" in joined
        ),
        "final_release_executed": "FINAL_RELEASE_INTENT_START" in joined,
        "all_commands_and_measured_states_finite": finite,
        "numerical_failure": not finite,
        "object_pose_writes_during_timed_loop": int(trial.get("object_pose_writes_during_timed_loop", -1)),
        "state_restoration_used": bool(trial.get("state_restoration", {}).get("used", False)),
        "physical_task_success_not_a_smoke_requirement": True,
        "event_log": str((run / "event_log.npz").resolve()),
        "event_log_sha256": sha256_file(run / "event_log.npz"),
    }


def analyze_scripted(run: Path) -> dict[str, Any]:
    result = read_json(run / "CONTACT_CONSTRAINED_TASK_RESULT.json")
    trial = read_json(run / "trial_result.json")
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    lower, upper, names = ranges()
    command = np.asarray(event["commanded_q_rad"], dtype=np.float64)
    measured = np.asarray(event["measured_q_rad"], dtype=np.float64)
    dex_command_bad = (command[:, 14:] < lower[14:] - 1.0e-9) | (command[:, 14:] > upper[14:] + 1.0e-9)
    dex_measured_bad = (measured[:, 14:] < lower[14:] - 1.0e-6) | (measured[:, 14:] > upper[14:] + 1.0e-6)
    bad_rows, bad_joints = np.nonzero(dex_measured_bad)
    first_bad = None
    if len(bad_rows):
        row, local = int(bad_rows[0]), int(bad_joints[0])
        joint = 14 + local
        first_bad = {
            "physics_row": row,
            "control_frame": int(event["control_frame"][row]),
            "joint_name": names[joint],
            "measured_rad": float(measured[row, joint]),
            "lower_rad": float(lower[joint]),
            "upper_rad": float(upper[joint]),
        }
    outcomes = result["outcomes"]
    pass_task = bool(
        result.get("status") == "PASS"
        and all(
            outcomes.get(name, False)
            for name in (
                "LEFT_GRASP", "HANDOFF", "RIGHT_OWNERSHIP", "NO_DROP_BEFORE_BIN", "DOLL_ENTERS_BIN", "DOLL_SETTLES", "FULL_TASK_SUCCESS"
            )
        )
    )
    valid = bool(
        pass_task
        and not np.any(dex_command_bad)
        and not np.any(dex_measured_bad)
        and np.isfinite(command).all()
        and np.isfinite(measured).all()
        and bool(trial.get("command_completed", False))
        and int(trial.get("object_pose_writes_during_timed_loop", -1)) == 0
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "contact_constrained_task_result_status": result.get("status"),
        "outcomes": outcomes,
        "commanded_dex3_hard_limit_violation_scalar_count": int(np.count_nonzero(dex_command_bad)),
        "measured_dex3_hard_limit_violation_scalar_count": int(np.count_nonzero(dex_measured_bad)),
        "first_measured_dex3_hard_limit_violation": first_bad,
        "command_completed": bool(trial.get("command_completed", False)),
        "object_pose_writes_during_timed_loop": int(trial.get("object_pose_writes_during_timed_loop", -1)),
        "event_log": str((run / "event_log.npz").resolve()),
        "event_log_sha256": sha256_file(run / "event_log.npz"),
        "task_result": str((run / "CONTACT_CONSTRAINED_TASK_RESULT.json").resolve()),
        "task_result_sha256": sha256_file(run / "CONTACT_CONSTRAINED_TASK_RESULT.json"),
    }


def main() -> int:
    prep = read_json(PREP)
    if prep.get("status") != "PREPARED_BEFORE_PHYSX_OUTCOMES" or len(prep.get("smoke_records", [])) != 4:
        raise RuntimeError("qualification inputs not predeclared")
    if any(row.get("evaluation_episode") is not False for row in prep["smoke_records"]):
        raise RuntimeError("EVAL data is forbidden in qualification smoke")
    provisional = read_json(PROVISIONAL)
    if provisional.get("status") != "PRE_EVAL35_QUALIFICATION_PROVISIONAL":
        raise RuntimeError("qualification execution manifest missing")
    for row in provisional["files"]:
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"qualification dependency drift: {row['path']}")

    smoke_results = []
    for record in prep["smoke_records"]:
        run = QUAL / "smoke_runs" / f"{record['method_stream'].lower().replace('-', '_')}_{record['smoke_index']:02d}_{record['stable_episode_id']}"
        result_path = run / "QUALIFICATION_RESULT.json"
        if result_path.is_file():
            result = read_json(result_path)
            if result.get("status") != "PASS":
                raise RuntimeError(f"cached smoke failed: {result_path}")
            smoke_results.append(result)
            continue
        if run.exists():
            raise FileExistsError(f"incomplete qualification smoke: {run}")
        run.mkdir(parents=True)
        command = [
            str(ISAAC), str(DIRECT_LAUNCHER), "--qualification-mode",
            "--direct-freeze-manifest", str(PROVISIONAL), "--config", str(CONFIG),
            "--side", "right", "--geometry", "FROZEN_COMPRESSED_SHORT_55", "--profile", "P14",
            "--output-dir", str(run), "--scripted-command-path", record["command"],
            "--object-spawn-side", "left", "--audit-robot-bin", "--full-task-audit",
            "--bin-height-m", "0.150", "--bin-rim-bevel-m", "0.003", "--headless",
        ]
        started = time.monotonic()
        code = run_process(command, run / "engine.log")
        required = [run / name for name in ("event_log.npz", "trial_result.json", "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")]
        if code not in (0, 2) or any(not path.is_file() for path in required):
            raise RuntimeError(f"qualification smoke infrastructure failure: {run}")
        result = analyze_direct(run, record["method_stream"], record["stable_episode_id"])
        result.update(
            {
                "engine_exit_code": code,
                "wall_seconds": time.monotonic() - started,
                "command": record["command"],
                "command_sha256": record["command_sha256"],
                "qualification_selection_rule": prep["selection_rule"],
            }
        )
        atomic_json(result_path, result)
        if result["status"] != "PASS":
            raise RuntimeError(f"qualification smoke validity failed: {result_path}")
        smoke_results.append(result)
        print(json.dumps({"smoke_completed": len(smoke_results), "status": result["status"], "run": str(run)}), flush=True)
    smoke_pass = len(smoke_results) == 4 and all(row["status"] == "PASS" for row in smoke_results)
    smoke_value = {
        "schema_version": "pre_eval35_train40_physx_smoke_v1",
        "status": "PASS" if smoke_pass else "FAIL",
        "evaluation_data_used": False,
        "selection_rule_predeclared_before_outcomes": prep["selection_rule"],
        "selected_train_episodes": prep["selected_train_episodes"],
        "runs": smoke_results,
        "runs_completed": len(smoke_results),
        "commanded_dex3_hard_limit_violations": sum(row["commanded_dex3_hard_limit_violation_scalar_count"] for row in smoke_results),
        "measured_dex3_hard_limit_violations": sum(row["measured_dex3_hard_limit_violation_scalar_count"] for row in smoke_results),
        "arm_common_overwrite_scalars": sum(row["arm_common_override_scalar_count"] for row in smoke_results),
        "wrist_rescue_scalars": sum(row["wrist_common_override_scalar_count"] for row in smoke_results),
        "physical_task_success_used_for_qualification_or_tuning": False,
    }
    atomic_json(SMOKE_JSON, smoke_value)
    SMOKE_MD.write_text(
        "# TRAIN40 PhysX primitive smoke test\n\n"
        f"Status: **{smoke_value['status']}**\n\n"
        f"- Deterministic selection: `{prep['selection_rule']}`\n"
        f"- Episodes: `{', '.join(prep['selected_train_episodes'])}`\n"
        f"- A/B-format runs completed: **{len(smoke_results)}/4**\n"
        f"- Commanded Dex3 hard-limit violations: **{smoke_value['commanded_dex3_hard_limit_violations']}**\n"
        f"- Measured Dex3 hard-limit violations: **{smoke_value['measured_dex3_hard_limit_violations']}**\n"
        f"- Arm overwrite scalars: **{smoke_value['arm_common_overwrite_scalars']}**\n"
        f"- Wrist rescue scalars: **{smoke_value['wrist_rescue_scalars']}**\n"
        "- Both-hand close motion and final release executed in contact-constrained PhysX.\n"
        "- Physical task success/failure was not used to tune or qualify this safety interface.\n",
        encoding="utf-8",
    )
    if not smoke_pass:
        raise RuntimeError("PhysX TRAIN40 smoke failed")

    regression_run = QUAL / "scripted_regression_run"
    regression_result = regression_run / "QUALIFICATION_RESULT.json"
    if regression_result.is_file():
        scripted = read_json(regression_result)
    else:
        if regression_run.exists():
            raise FileExistsError(f"incomplete scripted regression: {regression_run}")
        regression_run.mkdir(parents=True)
        source = prep["scripted_regression"]
        command = [
            str(ISAAC), str(ENGINE), "--config", str(CONFIG), "--side", "right",
            "--geometry", "FROZEN_COMPRESSED_SHORT_55", "--profile", "P14",
            "--output-dir", str(regression_run), "--scripted-command-path", source["command"],
            "--object-spawn-side", "left", "--audit-robot-bin", "--full-task-audit",
            "--bin-height-m", "0.150", "--bin-rim-bevel-m", "0.003", "--headless",
            "--dex3-hard-limit-contract", str(JOINT_CONTRACT),
            "--dex3-hard-limit-inset-rad", "0.005",
        ]
        started = time.monotonic()
        code = run_process(command, regression_run / "engine.log")
        if code not in (0, 2) or not (regression_run / "event_log.npz").is_file():
            raise RuntimeError("scripted regression engine failure")
        scored = subprocess.run(
            [str(ISAAC), str(SCORER), "--run-dir", str(regression_run)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        (regression_run / "scorer.log").write_text(scored.stdout + scored.stderr, encoding="utf-8")
        if not (regression_run / "CONTACT_CONSTRAINED_TASK_RESULT.json").is_file():
            raise RuntimeError("scripted regression scorer failure")
        scripted = analyze_scripted(regression_run)
        scripted.update(
            {
                "engine_exit_code": code,
                "scorer_exit_code": int(scored.returncode),
                "wall_seconds": time.monotonic() - started,
                "command": source["command"],
                "command_sha256": source["command_sha256"],
                "source_arm_command_preserved_exactly": bool(source["arm_command_exact"]),
                "maximum_limit_guard_change_rad": float(source["maximum_dex3_change_rad"]),
            }
        )
        atomic_json(regression_result, scripted)
    atomic_json(REGRESSION_JSON, scripted)
    REGRESSION_MD.write_text(
        "# Scripted end-to-end physical regression\n\n"
        f"Status: **{scripted['status']}**\n\n"
        + "\n".join(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in scripted["outcomes"].items())
        + f"\n- Commanded Dex3 hard-limit violations: **{scripted['commanded_dex3_hard_limit_violation_scalar_count']}**"
        + f"\n- Measured Dex3 hard-limit violations: **{scripted['measured_dex3_hard_limit_violation_scalar_count']}**"
        + f"\n- Arm command preserved exactly: **{'YES' if scripted['source_arm_command_preserved_exactly'] else 'NO'}**"
        + f"\n- Maximum boundary-guard adjustment: **{scripted['maximum_limit_guard_change_rad']:.7f} rad**\n",
        encoding="utf-8",
    )
    print(json.dumps({"physx_smoke": smoke_value["status"], "scripted_regression": scripted["status"]}, indent=2))
    return 0 if scripted["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
