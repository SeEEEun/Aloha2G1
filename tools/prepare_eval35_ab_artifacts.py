#!/usr/bin/env python3
"""Prepare (but never physically run) all EVAL35 A/B artifacts after freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path("/home/jbnu/aloha_g1_dataset")
LEROBOT_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
SYSTEM_PYTHON = Path("/usr/bin/python3")


def run(command: list[str]) -> None:
    process = subprocess.run(command, cwd=ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"EVAL35 preparation step failed ({process.returncode}): {command}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-execution-freeze",
        action="store_true",
        help="Require an already-frozen execution layer instead of invoking its finalizer.",
    )
    args = parser.parse_args()
    if not args.skip_execution_freeze:
        run(
            [
                str(SYSTEM_PYTHON),
                "tools/finalize_common_execution_layer_freeze.py",
                "--freeze",
            ]
        )
    run([str(LEROBOT_PYTHON), "tools/convert_eval35_new25_frozen_ab.py"])
    run(
        [
            str(LEROBOT_PYTHON),
            "tools/precompute_eval35_new25_act.py",
            "--method",
            "a",
        ]
    )
    run(
        [
            str(LEROBOT_PYTHON),
            "tools/precompute_eval35_new25_act.py",
            "--method",
            "b",
        ]
    )
    run([str(LEROBOT_PYTHON), "tools/prepare_eval35_physical_commands.py"])
    print(
        json.dumps(
            {
                "status": "EVAL35_AB_ARTIFACTS_READY",
                "physical_rollouts_started": False,
                "next_command": "python3 tools/run_final_common_execution_eval35.py",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
