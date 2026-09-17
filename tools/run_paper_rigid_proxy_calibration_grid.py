#!/usr/bin/env python3
"""Run the predeclared 3-material x 2-hand policy-independent Isaac grid."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation.contracts import sha256_file
from tools.evaluation.io import atomic_json


DEFAULT_CONFIG = ROOT / "configs/paper_eval_doll_rigid_proxy_v1.json"
DEFAULT_PRIMITIVES = ROOT / "outputs/paper_physics_task_eval/environment_calibration/primitives_review1"
DEFAULT_OUTPUT = ROOT / "outputs/paper_physics_task_eval/environment_calibration/material_grid_review1"
ISAAC_PYTHON = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
RUNNER = ROOT / "tools/run_dex3_rigid_doll_grasp_isaac.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_free() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip():
        raise RuntimeError(f"GPU is not exclusively available: {result.stdout.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--primitives", type=Path, default=DEFAULT_PRIMITIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = args.config.resolve()
    primitives = args.primitives.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration grid: {output}")
    output.mkdir(parents=True)
    rows = []
    for material in ("LOW", "MEDIUM", "HIGH"):
        for side in ("left", "right"):
            gpu_free()
            destination = output / material.lower() / side
            log = output / "logs" / f"{material.lower()}_{side}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(ISAAC_PYTHON),
                str(RUNNER),
                "--config",
                str(config),
                "--side",
                side,
                "--material",
                material,
                "--primitive",
                str(primitives / f"{side}_fixed_grasp_primitive.npz"),
                "--output-dir",
                str(destination),
                "--headless",
            ]
            started = time.monotonic()
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            report_path = destination / "calibration_result.json"
            report = read_json(report_path) if report_path.is_file() else None
            row = {
                "material": material,
                "side": side,
                "exit_code": completed.returncode,
                "wall_seconds": time.monotonic() - started,
                "report": str(report_path) if report else None,
                "report_sha256": sha256_file(report_path) if report else None,
                "result_status": report.get("status") if report else "PROCESS_FAILURE",
                "lift_pass": bool(report and report.get(f"{side.upper()}_LIFT_PASS")),
                "policy_or_checkpoint_used": bool(report and report.get("policy_or_checkpoint_used")),
                "log": str(log),
                "log_sha256": sha256_file(log),
            }
            rows.append(row)
            atomic_json(
                output / "grid_progress.json",
                {
                    "schema_version": "paper_rigid_proxy_calibration_grid_v1",
                    "status": "RUNNING",
                    "policy_independent": True,
                    "config_sha256": sha256_file(config),
                    "runs": rows,
                },
            )
            print(
                f"{material} {side}: status={row['result_status']} "
                f"lift_pass={row['lift_pass']} wall={row['wall_seconds']:.1f}s",
                flush=True,
            )
            time.sleep(3.0)
    result = {
        "schema_version": "paper_rigid_proxy_calibration_grid_v1",
        "status": "CALIBRATION_GRID_COMPLETE" if len(rows) == 6 and all(row["report"] for row in rows) else "INCOMPLETE",
        "expected_grid": [[material, side] for material in ("LOW", "MEDIUM", "HIGH") for side in ("left", "right")],
        "run_count": len(rows),
        "policy_independent": True,
        "policy_results_consulted": False,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "runs": rows,
    }
    atomic_json(output / "grid_result.json", result)
    if result["status"] != "CALIBRATION_GRID_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
