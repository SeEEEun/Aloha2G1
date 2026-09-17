#!/usr/bin/env python3
"""Run/resume the hard-frozen episode-registered ACT-A35 then ACT-B35 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
LAUNCHER = ROOT / "tools/run_direct_physical_execution_isaac.py"
SCORER = ROOT / "tools/score_episode_registered_physical_eval35_run.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
OUT = ROOT / "outputs/final_episode_registered_eval35"
FREEZE = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"
REGISTRATION = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
COMMANDS = ROOT / "outputs/final_direct_physical_eval35/00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
STARTED = OUT / "FINAL_ROLLOUTS_STARTED.json"
STATUS = OUT / "CURRENT_STATUS.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_freeze() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]]]:
    freeze = read_json(FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_EVAL35" or freeze.get("required_rollouts") != 70:
        raise RuntimeError("final episode-registered evaluation is not hard-frozen")
    for row in freeze.get("files", []):
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen dependency drift: {path}")
    commands = read_json(COMMANDS)
    records = {(str(row["method"]), int(row["eval_index"])): row for row in commands["records"]}
    expected = {(f"ACT-{letter}40", index) for letter in ("A", "B") for index in range(35)}
    if set(records) != expected:
        raise RuntimeError("prepared commands are not exact matched ACT-A/B EVAL35")
    registration = read_json(REGISTRATION)
    entries = {str(row["stable_episode_id"]): row for row in registration["entries"]}
    if len(entries) != 35 or registration.get("A_B_identical_object_pose_count") != 35:
        raise RuntimeError("episode registration is not exact/equal 35")
    return freeze, records, entries


def result_root(method: str) -> Path:
    return OUT / ("02_act_a_results" if method == "A" else "03_act_b_results") / "rollouts"


def completed(method: str) -> int:
    return len(list(result_root(method).glob("eval_*/RUN_MANIFEST.json")))


def update_status(last: str) -> None:
    prior = STATUS.read_text(encoding="utf-8") if STATUS.is_file() else ""
    qualification = "PHYSICAL_CONTACT_QUALIFICATION_READY_TO_FREEZE" if "PHYSICAL_CONTACT_QUALIFICATION_READY_TO_FREEZE" in prior else "PASS"
    STATUS.write_text(
        "# Final Episode-Conditioned EVAL35 Status\n\n"
        "- Previous partial physical evaluation: `SUPERSEDED_PRE_EPISODE_REGISTRATION_RESULT`\n"
        "- Registration: 35 / 35 source-derived entries\n"
        "- Matched A/B object-pose equality: 35 / 35 (0 mm / 0 deg)\n"
        f"- Qualification: `{qualification}`\n"
        f"- Final freeze SHA256: `{sha256_file(FREEZE)}`\n"
        f"- ACT-A final valid physical rollouts: {completed('A')} / 35\n"
        f"- ACT-B final valid physical rollouts: {completed('B')} / 35\n"
        f"- Final EVAL35 rollouts completed: {completed('A') + completed('B')} / 70\n"
        f"- Last persisted result: `{last}`\n"
        "- ARM rescue: NO; WRIST rescue: NO.\n",
        encoding="utf-8",
    )


def start_lock() -> None:
    value = {
        "schema_version": "episode_registered_eval35_rollout_start_lock_v1",
        "status": "NO_SCIENTIFIC_TUNING_AFTER_THIS_POINT",
        "freeze_manifest": str(FREEZE),
        "freeze_manifest_sha256": sha256_file(FREEZE),
        "required_rollouts": 70,
    }
    if STARTED.is_file() and read_json(STARTED) != value:
        raise RuntimeError("rollout start lock does not match current freeze")
    if not STARTED.is_file():
        atomic_json(STARTED, value)


def preserve_incomplete(output: Path) -> None:
    if not output.exists():
        return
    suffix = 1
    while True:
        target = output.with_name(output.name + f".INFRASTRUCTURE_ATTEMPT_{suffix:02d}")
        if not target.exists():
            shutil.move(str(output), str(target))
            return
        suffix += 1


def run_one(method: str, index: int, row: dict[str, Any], registration: dict[str, Any]) -> dict[str, Any]:
    output = result_root(method) / f"eval_{index:02d}_{row['stable_episode_id']}"
    complete = output / "RUN_MANIFEST.json"
    required_cached = (
        output / "event_log.npz", output / "robot_bin_contacts.npz",
        output / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json",
        output / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json",
    )
    if complete.is_file():
        cached = read_json(complete)
        if cached.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL"} or any(not path.is_file() for path in required_cached):
            raise RuntimeError(f"completed-run provenance is corrupt: {output}")
        return cached
    preserve_incomplete(output)
    output.mkdir(parents=True)
    command = Path(row["physical_command"])
    if sha256_file(command) != row["physical_command_sha256"]:
        raise RuntimeError(f"prepared command drift: {command}")
    if registration["methods"][f"ACT-{method}40"]["command_sha256"] != row["physical_command_sha256"]:
        raise RuntimeError("registration/command binding mismatch")
    invocation = [
        str(ISAAC), str(LAUNCHER), "--direct-freeze-manifest", str(FREEZE),
        "--config", str(CONFIG), "--side", "right", "--geometry", "INTERMEDIATE_PLUSH_PROXY",
        "--profile", "P14", "--output-dir", str(output), "--scripted-command-path", str(command),
        "--object-spawn-side", "left", "--episode-registration-manifest", str(REGISTRATION),
        "--episode-stable-id", row["stable_episode_id"], "--audit-robot-bin", "--full-task-audit",
        "--bin-height-m", "0.150", "--bin-rim-bevel-m", "0.003", "--headless",
    ]
    invocation_value = {
        "schema_version": "episode_registered_physical_eval35_invocation_v1",
        "method": f"ACT-{method}40", "eval_index": index,
        "stable_episode_id": row["stable_episode_id"], "provenance": row["provenance"],
        "command": str(command), "command_sha256": row["physical_command_sha256"],
        "freeze_manifest": str(FREEZE), "freeze_manifest_sha256": sha256_file(FREEZE),
        "episode_registration_manifest": str(REGISTRATION),
        "episode_registration_manifest_sha256": sha256_file(REGISTRATION),
        "episode_registration": {
            "target_object_pose": registration["target_object_pose"],
            "entry_sha256": registration["entry_sha256"],
            "A_B_identical_object_pose": registration["A_B_identical_object_pose"],
            "manual_episode_nudge": False, "policy_output_derived": False,
        },
        "geometry": "INTERMEDIATE_PLUSH_PROXY", "contact_tolerance_mm": 0,
        "execution": "finite-gain articulation targets + contact-constrained PhysX",
        "arm_rescue_allowed": False, "wrist_rescue_allowed": False,
        "common_arm_hard_limit_projector": "NEAREST_VALID_VALUE_COMPONENTWISE",
        "object_pose_writes_after_initialization": 0,
        "invocation": invocation,
    }
    atomic_json(output / "INVOCATION_MANIFEST.json", invocation_value)
    started = time.monotonic()
    with (output / "engine.log").open("w", encoding="utf-8") as stream:
        engine = subprocess.run(invocation, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    engine_required = (
        output / "event_log.npz", output / "robot_bin_contacts.npz", output / "trial_result.json",
        output / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json",
    )
    if engine.returncode not in (0, 2) or any(not path.is_file() for path in engine_required):
        atomic_json(output / "INFRASTRUCTURE_FAILURE.json", {
            "engine_returncode": engine.returncode,
            "missing": [str(path) for path in engine_required if not path.is_file()],
        })
        raise RuntimeError(f"Isaac infrastructure failure: {output / 'engine.log'}")
    scored = subprocess.run([str(ISAAC), str(SCORER), "--run-dir", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
    (output / "scorer.log").write_text(scored.stdout + scored.stderr, encoding="utf-8")
    result_path = output / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json"
    if scored.returncode not in (0, 3) or not result_path.is_file():
        atomic_json(output / "INFRASTRUCTURE_FAILURE.json", {"scorer_returncode": scored.returncode})
        raise RuntimeError(f"scorer infrastructure failure: {output / 'scorer.log'}")
    result = read_json(result_path)
    if result["status"] == "INVALID":
        raise RuntimeError(f"invalid physical run under frozen experiment: {output}")
    runtime = read_json(output / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")
    if (
        runtime.get("arm_common_override_scalar_count") != 0
        or runtime.get("wrist_common_override_scalar_count") != 0
        or runtime.get("common_arm_hard_limit_projector") != "NEAREST_VALID_VALUE_COMPONENTWISE"
        or runtime.get("common_arm_hard_limit_projector_same_for_a_b") is not True
        or runtime.get("arm_hard_limit_violation_scalar_count_after_projection") != 0
    ):
        raise RuntimeError("COMMON_EXECUTION_LAYER_INVALID")
    status = "PHYSICAL_PASS" if result["outcomes"]["FULL_TASK_SUCCESS"] else "PHYSICAL_FAIL"
    artifacts = [*engine_required, result_path, output / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.md"]
    value = {
        "schema_version": "episode_registered_physical_eval35_run_v1", "status": status,
        "method": f"ACT-{method}40", "eval_index": index, "stable_episode_id": row["stable_episode_id"],
        "provenance": row["provenance"], "wall_seconds": time.monotonic() - started,
        "engine_exit_code": engine.returncode, "scorer_exit_code": scored.returncode,
        "outcomes": result["outcomes"], "first_failure_stage": result["first_failure_stage"],
        "release_classification": result["release_classification"], "contact_topology": result["contact_topology"],
        "event_frames": result["event_frames"], "final_state": result["final_state"],
        "fairness_audit": result["fairness_audit"], "integrity": result["integrity"],
        "episode_registration_entry_sha256": registration["entry_sha256"],
        "artifacts": {path.name: {"path": str(path), "sha256": sha256_file(path)} for path in artifacts},
    }
    atomic_json(complete, value)
    update_status(str(complete))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("A", "B"))
    parser.add_argument("--eval-index", type=int, choices=range(35))
    args = parser.parse_args()
    _, records, registrations = verify_freeze()
    start_lock()
    methods = (args.method,) if args.method else ("A", "B")
    indices = (args.eval_index,) if args.eval_index is not None else tuple(range(35))
    for method in methods:
        if method == "B" and args.method is None and completed("A") != 35:
            raise RuntimeError("ACT-B cannot begin before ACT-A35 completes")
        for index in indices:
            row = records[(f"ACT-{method}40", index)]
            result = run_one(method, index, row, registrations[row["stable_episode_id"]])
            print(json.dumps({
                "completed_total": completed("A") + completed("B"),
                "method": method, "eval_index": index, "status": result["status"],
            }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
