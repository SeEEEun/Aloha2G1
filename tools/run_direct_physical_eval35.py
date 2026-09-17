#!/usr/bin/env python3
"""Run one or all frozen classifier-free ACT-A/B EVAL35 physical rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
LAUNCHER = ROOT / "tools/run_direct_physical_execution_isaac.py"
SCORER = ROOT / "tools/score_direct_physical_eval35_run.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
OBJECT_REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
OUT = ROOT / "outputs/final_direct_physical_eval35"
COMMANDS = OUT / "00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
FREEZE = OUT / "00_freeze/DIRECT_EVAL35_FREEZE_MANIFEST.json"
RUNS = OUT / "01_rollouts"
STARTED = OUT / "ROLLOUTS_STARTED.json"
STATUS = OUT / "CURRENT_STATUS.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    import hashlib
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


def verify_freeze() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    freeze = read_json(FREEZE)
    if freeze.get("status") != "FROZEN_BEFORE_EVAL35":
        raise RuntimeError("direct physical EVAL35 is not frozen")
    for row in freeze.get("files", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"direct EVAL35 frozen dependency drift: {path}")
    commands = read_json(COMMANDS)
    records = {(str(row["method"]), int(row["eval_index"])): row for row in commands["records"]}
    expected = {(f"ACT-{method}40", index) for method in ("A", "B") for index in range(35)}
    if set(records) != expected:
        raise RuntimeError("physical commands are not exact ACT-A/B x EVAL35")
    return freeze, records


def start_lock() -> None:
    value = {"status": "NO_TUNING_AFTER_THIS_POINT", "freeze_manifest": str(FREEZE), "freeze_manifest_sha256": sha256_file(FREEZE), "required_rollouts": 70}
    if STARTED.is_file():
        prior = read_json(STARTED)
        if prior != value:
            raise RuntimeError("rollout-start lock differs from current freeze")
    else:
        atomic_json(STARTED, value)


def completed_count() -> int:
    return len(list(RUNS.glob("act_*40/eval_*/RUN_MANIFEST.json")))


def update_status(last: str) -> None:
    STATUS.write_text(
        "# Final direct physical EVAL35 status\n\n"
        f"- Completed physical rollouts: **{completed_count()}/70**\n"
        f"- Last persisted result: `{last}`\n"
        f"- Freeze: `{FREEZE}`\n"
        "- No classifier, atlas, wrist-distance gate, arm rescue, or wrist rescue.\n",
        encoding="utf-8",
    )


def run_one(method: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    output = RUNS / f"act_{method.lower()}40/eval_{index:02d}_{row['stable_episode_id']}"
    complete = output / "RUN_MANIFEST.json"
    if complete.is_file():
        cached = read_json(complete)
        required = (output / "event_log.npz", output / "robot_bin_contacts.npz", output / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json", output / "DIRECT_PHYSICAL_TASK_RESULT.json")
        if cached.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL", "INVALID_PHYSICS"} or any(not path.is_file() for path in required):
            raise RuntimeError(f"incomplete cached run: {output}")
        return cached
    if output.exists():
        raise FileExistsError(f"refusing to overwrite incomplete run: {output}")
    output.mkdir(parents=True)
    command = Path(row["physical_command"])
    if sha256_file(command) != row["physical_command_sha256"]:
        raise RuntimeError(f"physical command drift: {command}")
    invocation = [
        str(ISAAC), str(LAUNCHER), "--direct-freeze-manifest", str(FREEZE),
        "--config", str(CONFIG), "--side", "right", "--geometry",
        "FROZEN_COMPRESSED_SHORT_55", "--profile", "P14", "--output-dir",
        str(output), "--scripted-command-path", str(command), "--object-spawn-side",
        "left", "--object-registration-config", str(OBJECT_REGISTRATION),
        "--audit-robot-bin", "--full-task-audit", "--bin-height-m",
        "0.150", "--bin-rim-bevel-m", "0.003", "--headless",
    ]
    atomic_json(output / "INVOCATION_MANIFEST.json", {
        "schema_version": "direct_physical_eval35_invocation_v1", "method": f"ACT-{method}40",
        "eval_index": index, "stable_episode_id": row["stable_episode_id"],
        "provenance": row["provenance"], "command": str(command),
        "command_sha256": row["physical_command_sha256"],
        "object_registration": str(OBJECT_REGISTRATION),
        "object_registration_sha256": sha256_file(OBJECT_REGISTRATION),
        "freeze_manifest": str(FREEZE), "freeze_manifest_sha256": sha256_file(FREEZE),
        "execution": "finite-gain articulation targets + contact-constrained PhysX",
        "common_initial_state": "COMMON_G1_POLICY_INITIAL_STATE_AB_V1",
        "pregrasp_classifier_used": False, "atlas_gate_used": False,
        "wrist_distance_gate_used": False, "arm_rescue_allowed": False,
        "wrist_rescue_allowed": False,
        "common_override_scope": "DEX3_PLUS_COMMON_ARM_HARD_LIMIT_SAFETY_ONLY",
        "common_arm_hard_limit_projector": "NEAREST_VALID_VALUE_COMPONENTWISE",
        "state_restoration": False, "object_pose_writes_after_reset": False,
        "invocation": invocation,
    })
    started = time.monotonic()
    with (output / "engine.log").open("w", encoding="utf-8") as stream:
        engine = subprocess.run(invocation, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
    required_engine = (output / "event_log.npz", output / "robot_bin_contacts.npz", output / "trial_result.json", output / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")
    if engine.returncode not in (0, 2) or any(not path.is_file() for path in required_engine):
        atomic_json(output / "INFRASTRUCTURE_FAILURE.json", {"engine_returncode": engine.returncode, "missing": [str(path) for path in required_engine if not path.is_file()]})
        raise RuntimeError(f"Isaac infrastructure failure: {output / 'engine.log'}")
    scored = subprocess.run([str(ISAAC), str(SCORER), "--run-dir", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
    (output / "scorer.log").write_text(scored.stdout + scored.stderr, encoding="utf-8")
    result_path = output / "DIRECT_PHYSICAL_TASK_RESULT.json"
    if scored.returncode not in (0, 3) or not result_path.is_file():
        raise RuntimeError(f"physical scorer infrastructure failure: {output / 'scorer.log'}")
    result = read_json(result_path)
    status = "INVALID_PHYSICS" if result["status"] == "INVALID" else ("PHYSICAL_PASS" if result["outcomes"]["FULL_TASK_SUCCESS"] else "PHYSICAL_FAIL")
    runtime = read_json(output / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")
    if (
        runtime.get("arm_common_override_scalar_count") != 0
        or runtime.get("wrist_common_override_scalar_count") != 0
        or runtime.get("common_arm_hard_limit_projector")
        != "NEAREST_VALID_VALUE_COMPONENTWISE"
        or runtime.get("common_arm_hard_limit_projector_same_for_a_b") is not True
        or runtime.get("arm_hard_limit_violation_scalar_count_after_projection") != 0
    ):
        status = "COMMON_EXECUTION_LAYER_INVALID"
    artifacts = [*required_engine, result_path, output / "DIRECT_PHYSICAL_TASK_RESULT.md"]
    value = {
        "schema_version": "direct_physical_eval35_run_v1", "status": status,
        "method": f"ACT-{method}40", "eval_index": index,
        "stable_episode_id": row["stable_episode_id"], "provenance": row["provenance"],
        "wall_seconds": time.monotonic() - started, "engine_exit_code": engine.returncode,
        "scorer_exit_code": scored.returncode, "outcomes": result["outcomes"],
        "first_failure_stage": result["first_failure_stage"],
        "release_classification": result["release_classification"],
        "fairness_audit": result["fairness_audit"], "diagnostics": result["diagnostics"],
        "artifacts": {path.name: {"path": str(path), "sha256": sha256_file(path)} for path in artifacts},
    }
    atomic_json(complete, value)
    update_status(str(complete))
    if status == "COMMON_EXECUTION_LAYER_INVALID":
        raise RuntimeError("COMMON_EXECUTION_LAYER_INVALID")
    if status == "INVALID_PHYSICS":
        raise RuntimeError(f"invalid physics at {method}:{index}; final evaluation cannot continue")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("A", "B"))
    parser.add_argument("--eval-index", type=int, choices=range(35))
    args = parser.parse_args()
    _, records = verify_freeze()
    start_lock()
    methods = (args.method,) if args.method else ("A", "B")
    indices = (args.eval_index,) if args.eval_index is not None else tuple(range(35))
    for method in methods:
        for index in indices:
            result = run_one(method, index, records[(f"ACT-{method}40", index)])
            print(json.dumps({"completed": completed_count(), "method": method, "eval_index": index, "status": result["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
