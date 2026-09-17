#!/usr/bin/env python3
"""Run one frozen ACT-A/B EVAL10 command in contact-constrained PhysX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
SCORER = ROOT / "tools/score_contact_constrained_full_task.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
COMMAND_MANIFEST = ROOT / "outputs/final_contact_constrained_eval/05_act_ab_results/PHYSICAL_COMMAND_MANIFEST.json"
FREEZE = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FREEZE_MANIFEST.json"
OUT = ROOT / "outputs/final_contact_constrained_eval/05_act_ab_results/runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    parser.add_argument("--eval-index", type=int, choices=range(10), required=True)
    parser.add_argument("--output-root", type=Path, default=OUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    freeze = read_json(FREEZE)
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("physical environment is not frozen")
    commands = read_json(COMMAND_MANIFEST)
    method_name = f"ACT-{args.method.upper()}40"
    matches = [
        row
        for row in commands["records"]
        if row["method"] == method_name and int(row["eval_index"]) == args.eval_index
    ]
    if len(matches) != 1:
        raise RuntimeError("physical command identity lookup failed")
    row = matches[0]
    command = Path(row["physical_command"])
    if sha256(command) != row["physical_command_sha256"]:
        raise RuntimeError("physical command hash drift")
    output = args.output_root.resolve() / f"act_{args.method}40/eval_{args.eval_index:02d}_{row['stable_episode_id']}"
    completed = output / "RUN_MANIFEST.json"
    if completed.is_file():
        result = read_json(completed)
        required = [output / "event_log.npz", output / "robot_bin_contacts.npz", output / "CONTACT_CONSTRAINED_TASK_RESULT.json"]
        if result.get("status") not in {"PHYSICAL_PASS", "PHYSICAL_FAIL"} or any(not path.is_file() for path in required):
            raise RuntimeError(f"incomplete existing run: {output}")
        print(json.dumps({"status": result["status"], "cache_hit": str(output)}, indent=2))
        return 0
    if output.exists():
        raise FileExistsError(f"refusing to overwrite incomplete physical run: {output}")
    output.mkdir(parents=True)
    invocation = [
        str(ISAAC),
        str(ENGINE),
        "--config",
        str(CONFIG),
        "--side",
        "right",
        "--geometry",
        "FROZEN_COMPRESSED_SHORT_55",
        "--profile",
        "P14",
        "--output-dir",
        str(output),
        "--scripted-command-path",
        str(command),
        "--object-spawn-side",
        "left",
        "--audit-robot-bin",
        "--full-task-audit",
        "--bin-height-m",
        "0.150",
        "--bin-rim-bevel-m",
        "0.003",
        "--headless",
    ]
    write_json(
        output / "INVOCATION_MANIFEST.json",
        {
            "schema_version": "contact_constrained_eval10_invocation_v1",
            "method": method_name,
            "eval_index": args.eval_index,
            "provenance": row["provenance"],
            "stable_episode_id": row["stable_episode_id"],
            "command": str(command),
            "command_sha256": sha256(command),
            "environment_freeze": str(FREEZE),
            "environment_freeze_sha256": sha256(FREEZE),
            "engine": str(ENGINE),
            "engine_sha256": sha256(ENGINE),
            "scorer": str(SCORER),
            "scorer_sha256": sha256(SCORER),
            "execution": "finite-gain articulation position targets + PhysX measured state",
            "direct_state_writes_after_reset": False,
            "state_restoration": False,
            "object_pose_writes_after_reset": False,
            "common_controller_override_fraction": 0.0,
            "invocation": invocation,
        },
    )
    started = time.monotonic()
    with (output / "engine.log").open("w", encoding="utf-8") as stream:
        engine_result = subprocess.run(
            invocation,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    required = [output / "event_log.npz", output / "robot_bin_contacts.npz", output / "trial_result.json"]
    if engine_result.returncode not in (0, 2) or any(not path.is_file() for path in required):
        write_json(
            output / "INFRASTRUCTURE_FAILURE.json",
            {"returncode": engine_result.returncode, "missing": [str(path) for path in required if not path.is_file()]},
        )
        raise RuntimeError(f"Isaac infrastructure failure; see {output / 'engine.log'}")
    scored = subprocess.run(
        [str(ISAAC), str(SCORER), "--run-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "scorer.log").write_text(scored.stdout + scored.stderr, encoding="utf-8")
    result_path = output / "CONTACT_CONSTRAINED_TASK_RESULT.json"
    if not result_path.is_file():
        raise RuntimeError(f"scorer infrastructure failure: {output / 'scorer.log'}")
    result = read_json(result_path)
    manifest = {
        "schema_version": "contact_constrained_eval10_run_v1",
        "status": "PHYSICAL_PASS" if result["outcomes"]["FULL_TASK_SUCCESS"] else "PHYSICAL_FAIL",
        "method": method_name,
        "eval_index": args.eval_index,
        "provenance": row["provenance"],
        "stable_episode_id": row["stable_episode_id"],
        "wall_seconds": time.monotonic() - started,
        "engine_exit_code": engine_result.returncode,
        "scorer_exit_code": scored.returncode,
        "full_task_success": result["outcomes"]["FULL_TASK_SUCCESS"],
        "first_failure_stage": next(
            (name for name, passed in result["outcomes"].items() if not passed),
            None,
        ),
        "artifacts": {
            path.name: {"path": str(path), "sha256": sha256(path)}
            for path in [*required, result_path, output / "CONTACT_CONSTRAINED_TASK_RESULT.md"]
        },
    }
    write_json(completed, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    # A physical failure is an auditable evaluation result, not a harness error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
