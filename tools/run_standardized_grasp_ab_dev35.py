#!/usr/bin/env python3
"""Run/resume frozen standardized-grasp A35 followed by B35."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
OUT = ROOT / "outputs/standardized_grasp_ab_dev35"
FREEZE = OUT / "02_freeze/STANDARDIZED_GRASP_FINAL_FREEZE.json"
COMMANDS = OUT / "01_prepared_commands/STANDARDIZED_GRASP_AB_COMMAND_MANIFEST.json"
OBJECT = OUT / "00_control/STANDARDIZED_OBJECT_REGISTRATION.json"
INITIAL = OUT / "00_control/QUALIFIED_STANDARDIZED_INITIAL_GRASP.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
LAUNCHER = ROOT / "tools/run_direct_physical_execution_isaac.py"
SCORER = ROOT / "tools/score_standardized_grasp_post_grasp_run.py"
STATUS = OUT / "CURRENT_STATUS.md"
STARTED = OUT / "FINAL_ROLLOUTS_STARTED.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]: return json.loads(path.read_text())
def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(temporary, path)


def root(method: str) -> Path: return OUT / ("03_a_results" if method == "A" else "04_b_results") / "rollouts"
def completed(method: str) -> int: return len(list(root(method).glob("eval_*/RUN_MANIFEST.json")))


def update_status(last: str) -> None:
    STATUS.write_text(
        "# Standardized-grasp DEV35 status\n\n"
        "- Previous end-to-end results: `PRE_STANDARDIZED_GRASP_DIAGNOSTIC_ONLY`\n"
        "- Physical initialization gate: PASS\n"
        f"- Freeze SHA256: `{sha256(FREEZE)}`\n"
        f"- A — WRIST valid: {completed('A')} / 35\n- B — INTERACTION valid: {completed('B')} / 35\n"
        f"- Total valid: {completed('A') + completed('B')} / 70\n- Last: `{last}`\n"
        "- Scope: DEV35 STANDARDIZED-GRASP POST-GRASP evaluation; not end-to-end grasp acquisition.\n",
        encoding="utf-8",
    )


def verify() -> dict[tuple[str, int], dict[str, Any]]:
    freeze = read(FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_EVAL35" or freeze.get("required_rollouts") != 70:
        raise RuntimeError("standardized-grasp experiment is not frozen")
    for row in freeze["files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"frozen dependency drift: {path}")
    command_manifest = read(COMMANDS)
    rows = {(row["method"], int(row["eval_index"])): row for row in command_manifest["records"]}
    if set(rows) != {(method, index) for method in "AB" for index in range(35)}:
        raise RuntimeError("frozen command membership mismatch")
    return rows


def preserve_invalid(output: Path) -> None:
    if not output.exists(): return
    number = 1
    while output.with_name(output.name + f".INFRASTRUCTURE_ATTEMPT_{number:02d}").exists(): number += 1
    shutil.move(str(output), str(output.with_name(output.name + f".INFRASTRUCTURE_ATTEMPT_{number:02d}")))


def run_one(method: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    output = root(method) / f"eval_{index:02d}_{row['stable_episode_id']}"
    complete = output / "RUN_MANIFEST.json"
    if complete.is_file():
        value = read(complete)
        if value.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL"}: raise RuntimeError(f"corrupt completed run: {output}")
        return value
    preserve_invalid(output); output.mkdir(parents=True)
    command = Path(row["command"]); initial = read(INITIAL)
    invocation = [
        str(ISAAC), str(LAUNCHER), "--direct-freeze-manifest", str(FREEZE), "--config", str(CONFIG),
        "--side", "right", "--geometry", "INTERMEDIATE_PLUSH_PROXY", "--profile", "P14",
        "--output-dir", str(output), "--scripted-command-path", str(command), "--object-spawn-side", "left",
        "--object-registration-config", str(OBJECT), "--audit-robot-bin", "--full-task-audit",
        "--bin-height-m", "0.150", "--bin-rim-bevel-m", "0.003", "--headless",
    ]
    invoke = {
        "schema_version": "standardized_grasp_post_grasp_invocation_v1", "method": method, "eval_index": index,
        "stable_episode_id": row["stable_episode_id"], "representation_mode": row["representation_mode"],
        "command": str(command), "command_sha256": sha256(command), "freeze": str(FREEZE), "freeze_sha256": sha256(FREEZE),
        "standardized_initial_state": {"path": str(INITIAL), "sha256": sha256(INITIAL), "object_pose": {"position_xyz_m": initial["object_position_world_m"], "quaternion_xyzw": initial["object_quaternion_xyzw"]}},
        "object_registration": {"path": str(OBJECT), "sha256": sha256(OBJECT), "same_for_A_B": True},
        "common_ik_first_infeasible_frame": row["ik"]["first_infeasible_frame"] if row["ik"]["first_infeasible_frame"] is not None else -1,
        "arm_rescue": False, "wrist_rescue": False, "attachment": False, "object_pose_writes_after_initialization": 0,
        "invocation": invocation,
    }
    atomic_json(output / "INVOCATION_MANIFEST.json", invoke)
    started = time.monotonic()
    with (output / "engine.log").open("w") as log:
        engine = subprocess.run(invocation, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    required = [output / name for name in ("event_log.npz", "robot_bin_contacts.npz", "trial_result.json", "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")]
    if engine.returncode not in (0, 2) or any(not path.is_file() for path in required):
        atomic_json(output / "INFRASTRUCTURE_FAILURE.json", {"engine_returncode": engine.returncode, "missing": [str(path) for path in required if not path.is_file()]})
        raise RuntimeError(f"Isaac infrastructure failure: {output}")
    score = subprocess.run([str(ISAAC), str(SCORER), "--run-dir", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
    (output / "scorer.log").write_text(score.stdout + score.stderr)
    result_path = output / "STANDARDIZED_GRASP_POST_GRASP_RESULT.json"
    if score.returncode not in (0, 3) or not result_path.is_file(): raise RuntimeError(f"scoring failure: {output}")
    result = read(result_path)
    if result["status"] == "INVALID": raise RuntimeError(f"invalid frozen physical rollout: {output}")
    status = "PHYSICAL_PASS" if result["outcomes"]["POST_GRASP_FULL_TASK_SUCCESS"] else "PHYSICAL_FAIL"
    value = {
        "schema_version": "standardized_grasp_post_grasp_run_v1", "status": status, "method": method,
        "eval_index": index, "stable_episode_id": row["stable_episode_id"], "wall_seconds": time.monotonic() - started,
        "outcomes": result["outcomes"], "first_failure_stage": result["first_failure_stage"],
        "common_ik_failure_stage": result["common_ik_failure_stage"], "contact_topology": result["contact_topology"],
        "event_frames": result["event_frames"], "integrity": result["integrity"], "diagnostics": result["diagnostics"],
        "artifacts": {path.name: {"path": str(path), "sha256": sha256(path)} for path in [*required, result_path, output / "STANDARDIZED_GRASP_POST_GRASP_RESULT.md"]},
    }
    atomic_json(complete, value); update_status(str(complete)); return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--method", choices=("A", "B")); parser.add_argument("--eval-index", type=int, choices=range(35)); args = parser.parse_args()
    rows = verify()
    lock = {"schema_version": "standardized_grasp_rollout_start_lock_v1", "freeze_sha256": sha256(FREEZE), "status": "NO_TUNING_AFTER_THIS_POINT"}
    if STARTED.is_file() and read(STARTED) != lock: raise RuntimeError("rollout lock freeze mismatch")
    if not STARTED.is_file(): atomic_json(STARTED, lock)
    methods = (args.method,) if args.method else ("A", "B"); indices = (args.eval_index,) if args.eval_index is not None else range(35)
    for method in methods:
        if method == "B" and args.method is None and completed("A") != 35: raise RuntimeError("B cannot start before A35")
        for index in indices:
            result = run_one(method, index, rows[(method, index)])
            print(json.dumps({"method": method, "eval_index": index, "status": result["status"], "A": completed("A"), "B": completed("B")}), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
