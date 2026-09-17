#!/usr/bin/env python3
"""Run the fixed HELDOUT8 x ACT-A/B method-consistent Isaac matrix serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
OUTPUT = PAPER / "method_consistent_source_video_rollout"
ROLLOUTS = OUTPUT / "rollouts"
ISAAC_PYTHON = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
RUNNER = ROOT / "tools/run_paper_core_source_conditioned_rollout.py"
MANIFEST = PAPER / "heldout8_manifest.json"
CONTRACT = OUTPUT / "METHOD_CONSISTENT_INITIALIZATION_CONTRACT.json"
FREEZE = OUTPUT / "METHOD_CONSISTENT_INITIALIZATION_CONTRACT.sha256.json"
EVALUATION = OUTPUT / "METHOD_CONSISTENT_EVALUATION_CONTRACT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("a", "b"), default=["a", "b"])
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--startup-cooldown-seconds", type=float, default=5.0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_frozen_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = read_json(FREEZE)
    if freeze.get("status") != "FROZEN" or freeze["contract_sha256"] != sha256_file(CONTRACT):
        raise RuntimeError("method-consistent initialization contract changed")
    contract = read_json(CONTRACT)
    evaluation = read_json(EVALUATION)
    if (
        contract.get("status")
        != "METHOD_CONSISTENT_INITIALIZATION_FROZEN_BEFORE_INFERENCE"
        or evaluation.get("status") != "FROZEN_BEFORE_METHOD_CONSISTENT_INFERENCE"
    ):
        raise RuntimeError("method-consistent contracts are not frozen")
    for table in ("table1", "table2"):
        for row in evaluation["frozen_inputs"][table]:
            path = Path(row["path"])
            if sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"frozen {table} changed: {path}")
    return contract, evaluation


def run_status(runs: list[dict[str, Any]], expected: int) -> str:
    if len(runs) == expected and all(
        row["process_exit_code"] == 0 and row["rollout_status"] == "PASS"
        for row in runs
    ):
        return "PASS"
    eligibility = {
        method: [row for row in runs if row["method"] == method and row.get("frame0_eligible")]
        for method in ("a", "b")
    }
    if not eligibility["a"] and not eligibility["b"] and len(runs) == expected:
        return "NOT_EXECUTABLE"
    return "INCOMPLETE_OR_SAFETY_ABORT"


def main() -> None:
    args = parse_args()
    if any(value not in range(8) for value in args.episodes):
        raise ValueError("held-out episodes must be in 0..7")
    if args.startup_cooldown_seconds < 0.0 or args.startup_cooldown_seconds > 30.0:
        raise ValueError("startup cooldown must be in [0,30] seconds")
    contract, _ = verify_frozen_contracts()
    manifest = read_json(MANIFEST)
    representative_final = int(manifest["representative_episode"])
    representative_output = next(
        index
        for index, row in enumerate(manifest["entries"])
        if int(row["final_dataset_index"]) == representative_final
    )
    runs: list[dict[str, Any]] = []
    for output_episode in args.episodes:
        entry = manifest["entries"][output_episode]
        for method in args.methods:
            target = ROLLOUTS / method / (
                f"heldout_{output_episode:02d}_source_{int(entry['final_dataset_index']):02d}"
            )
            report_path = target / "rollout_report.json"
            if target.exists():
                if not report_path.is_file():
                    raise FileExistsError(f"partial existing output requires audit: {target}")
                report = read_json(report_path)
                if report.get("experiment_name") != "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT":
                    raise RuntimeError(f"wrong experiment output at {target}")
                runs.append(
                    {
                        "method": method,
                        "heldout_output_episode": output_episode,
                        "source_final_episode": int(entry["final_dataset_index"]),
                        "output": str(target),
                        "process_exit_code": 0,
                        "rollout_status": report["status"],
                        "frame0_eligible": bool(report.get("frame0_eligible")),
                        "executed_frames": int(report["executed_frames"]),
                        "requested_frames": int(report["requested_frames"]),
                        "reused_existing_output": True,
                    }
                )
                continue
            command = [
                str(ISAAC_PYTHON),
                str(RUNNER),
                "--method",
                method,
                "--heldout-episode",
                str(output_episode),
                "--output",
                str(target),
                "--initialization-mode",
                "method_consistent_v1",
                "--headless",
            ]
            if output_episode != representative_output:
                command.append("--no-video")
            log_path = OUTPUT / "logs" / f"{method}_heldout_{output_episode:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            report = read_json(report_path) if report_path.is_file() else None
            row = {
                "method": method,
                "heldout_output_episode": output_episode,
                "source_final_episode": int(entry["final_dataset_index"]),
                "stable_episode_id": entry["stable_episode_id"],
                "output": str(target),
                "command": command,
                "process_exit_code": completed.returncode,
                "wall_seconds": time.monotonic() - started,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
                "rollout_status": report["status"] if report else "PROCESS_FAILURE",
                "frame0_eligible": bool(report.get("frame0_eligible")) if report else False,
                "executed_frames": int(report["executed_frames"]) if report else 0,
                "requested_frames": int(report["requested_frames"])
                if report
                else int(entry["frames"]),
                "representative_video_run": output_episode == representative_output,
                "reused_existing_output": False,
            }
            runs.append(row)
            expected = len(args.methods) * len(args.episodes)
            atomic_json(
                OUTPUT / "batch_progress.json",
                {
                    "schema_version": "paper_core_method_consistent_batch_progress_v1",
                    "experiment_name": "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT",
                    "initialization_contract_sha256": sha256_file(CONTRACT),
                    "representative_source_final_episode": representative_final,
                    "representative_heldout_output_episode": representative_output,
                    "runs": runs,
                    "current_status": run_status(runs, expected),
                },
            )
            if args.startup_cooldown_seconds:
                time.sleep(args.startup_cooldown_seconds)

    expected = len(args.methods) * len(args.episodes)
    result = {
        "schema_version": "paper_core_method_consistent_rollout_batch_v1",
        "status": run_status(runs, expected),
        "experiment_name": "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT",
        "run_count": len(runs),
        "expected_run_count": expected,
        "serial_execution": True,
        "initialization_contract": str(CONTRACT),
        "initialization_contract_sha256": sha256_file(CONTRACT),
        "frame0_eligible": {
            method: int(sum(row["method"] == method and row.get("frame0_eligible", False) for row in runs))
            for method in ("a", "b")
        },
        "completed": {
            method: int(sum(row["method"] == method and row["rollout_status"] == "PASS" for row in runs))
            for method in ("a", "b")
        },
        "frame0_ineligible": {
            method: int(sum(row["method"] == method and row["rollout_status"] == "FRAME0_INELIGIBLE" for row in runs))
            for method in ("a", "b")
        },
        "safety_aborts_after_eligibility": {
            method: int(sum(row["method"] == method and row["rollout_status"] == "SAFETY_ABORT" for row in runs))
            for method in ("a", "b")
        },
        "runs": runs,
        "real_hardware": False,
    }
    atomic_json(OUTPUT / "batch_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
