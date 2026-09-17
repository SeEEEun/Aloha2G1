#!/usr/bin/env python3
"""Later-only serial launcher for the common 8x2 physical ACT-A/B matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluation.contracts import sha256_file
from tools.evaluation.io import atomic_json


ISAAC_PYTHON = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
RUNNER = ROOT / "tools/run_paper_physical_doll_rollout.py"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
DEFAULT_CONFIG = ROOT / "configs/doll_handoff_rigid_proxy_v1.json"
DEFAULT_MANIFEST = ROOT / "configs/doll_handoff_rigid_proxy_v1.sha256.json"
DEFAULT_OUTPUT = ROOT / "outputs/paper_metrics/physical_rollouts"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_gpu_unoccupied() -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot verify exclusive GPU availability with nvidia-smi")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if rows:
        raise RuntimeError(f"GPU compute process already active; physical batch will not overlap it: {rows}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--config-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=("a", "b"), default=["a", "b"])
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(8)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(index not in range(8) for index in args.episodes):
        raise ValueError("held-out episodes must be in 0..7")
    config = _read(args.config.resolve())
    manifest = _read(args.config_manifest.resolve())
    if (
        not bool(config.get("freeze", {}).get("frozen"))
        or manifest.get("status") != "FROZEN"
        or manifest.get("config_sha256") != sha256_file(args.config.resolve())
    ):
        raise RuntimeError("physical batch requires the hash-verified frozen proxy")
    heldout = _read(HELDOUT)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for episode in args.episodes:
        identity = heldout["entries"][episode]["stable_episode_id"]
        for method in args.methods:
            _assert_gpu_unoccupied()
            target = output / method / f"heldout_{episode:02d}_{identity}"
            if target.exists():
                raise FileExistsError(f"refusing to reuse or overwrite physical output: {target}")
            command = [
                str(ISAAC_PYTHON),
                str(RUNNER),
                "--method",
                method,
                "--heldout-episode",
                str(episode),
                "--config",
                str(args.config.resolve()),
                "--config-manifest",
                str(args.config_manifest.resolve()),
                "--output",
                str(target),
                "--headless",
            ]
            log_path = output / "logs" / f"{method}_heldout_{episode:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            report_path = target / "rollout_report.json"
            report = _read(report_path) if report_path.is_file() else None
            row = {
                "method": method,
                "heldout_episode": episode,
                "source_episode_id": identity,
                "output": str(target),
                "exit_code": process.returncode,
                "rollout_status": report.get("status") if report else "PROCESS_FAILURE",
                "wall_seconds": time.monotonic() - started,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
            }
            runs.append(row)
            atomic_json(
                output / "batch_progress.json",
                {
                    "schema_version": "paper_physical_rollout_batch_progress_v1",
                    "serial_execution": True,
                    "config_sha256": manifest["config_sha256"],
                    "runs": runs,
                },
            )
    expected = len(args.methods) * len(args.episodes)
    result = {
        "schema_version": "paper_physical_rollout_batch_v1",
        "status": (
            "PASS"
            if len(runs) == expected
            and all(row["exit_code"] == 0 and row["rollout_status"] == "PASS" for row in runs)
            else "FAIL_OR_INCOMPLETE"
        ),
        "run_count": len(runs),
        "expected_run_count": expected,
        "serial_execution": True,
        "same_runner_for_a_b": True,
        "same_rigid_proxy_config_sha256": manifest["config_sha256"],
        "policy_specific_physics_logic": False,
        "runs": runs,
    }
    atomic_json(output / "batch_result.json", result)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
