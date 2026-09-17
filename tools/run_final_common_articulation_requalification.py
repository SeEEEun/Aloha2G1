#!/usr/bin/env python3
"""Rerun the nine frozen non-EVAL grasp tasks after the common solver fix."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path("/home/jbnu/aloha_g1_dataset")
ISAAC = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
LAUNCHER = ROOT / "tools/run_direct_physical_execution_isaac.py"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
QUAL = ROOT / "outputs/final_episode_registered_eval35/00_qualification"
FREEZE = QUAL / "PROVISIONAL_QUALIFICATION_FREEZE.json"
COMMANDS = QUAL / "contact_seeking_commands"
REGISTRATION = QUAL / "SCRIPTED_QUALIFICATION_OBJECT_REGISTRATION.json"
RUNS = QUAL / "solver80_requalification"


def valid(directory: Path) -> bool:
    needed = (
        "event_log.npz",
        "trial_result.json",
        "DIRECT_EXECUTION_RUNTIME_SUMMARY.json",
    )
    if not all((directory / name).is_file() for name in needed):
        return False
    trial = json.loads((directory / "trial_result.json").read_text(encoding="utf-8"))
    solver = trial.get("runtime_articulation_solver", {})
    return (
        solver.get("runtime_position_iterations") == 80
        and solver.get("runtime_velocity_iterations") == 4
        and trial.get("object_pose_writes_during_timed_loop") == 0
    )


def run(name: str, side: str, command: Path, scripted: bool) -> None:
    output = RUNS / name
    if valid(output):
        print(f"[requalification] reuse {name}", flush=True)
        return
    if output.exists():
        raise RuntimeError(f"incomplete requalification output requires forensic review: {output}")
    output.mkdir(parents=True)
    invocation = [
        str(ISAAC),
        str(LAUNCHER),
        "--qualification-mode",
        "--direct-freeze-manifest",
        str(FREEZE),
        "--config",
        str(CONFIG),
        "--side",
        side,
        "--geometry",
        "INTERMEDIATE_PLUSH_PROXY",
        "--profile",
        "P14",
        "--output-dir",
        str(output),
        "--scripted-command-path",
        str(command),
        "--object-spawn-side",
        "left" if scripted else side,
        "--headless",
    ]
    if scripted:
        invocation.extend(
            [
                "--object-registration-config",
                str(REGISTRATION),
                "--audit-robot-bin",
                "--full-task-audit",
                "--bin-height-m",
                "0.150",
                "--bin-rim-bevel-m",
                "0.003",
            ]
        )
    (output / "INVOCATION.json").write_text(
        json.dumps({"command": invocation}, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "engine.log").open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            invocation,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode not in (0, 2) or not valid(output):
        raise RuntimeError(
            f"requalification infrastructure failure ({process.returncode}): {output}"
        )
    print(f"[requalification] completed {name} engine={process.returncode}", flush=True)


def main() -> int:
    plan = []
    for side in ("right", "left"):
        command = COMMANDS / f"{side}_standalone_contact_seeking.npz"
        for index in range(1, 4):
            plan.append((f"{side}_{index:02d}", side, command, False))
    command = COMMANDS / "scripted_full_task_contact_seeking.npz"
    for index in range(1, 4):
        plan.append((f"scripted_full_{index:02d}", "right", command, True))
    for args in plan:
        run(*args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
