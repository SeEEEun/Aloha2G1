#!/usr/bin/env python3
"""Launch the fixed 16-run ACT-A/B source-conditioned Isaac matrix serially."""

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
ISAAC_PYTHON = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")
RUNNER = ROOT / "tools/run_paper_core_source_conditioned_rollout.py"
MANIFEST = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
OUTPUT = ROOT / "outputs/paper_core_ab/source_conditioned_rollout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("a", "b"), default=["a", "b"])
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(8)))
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if any(episode not in range(8) for episode in args.episodes):
        raise ValueError("held-out episodes must be in 0..7")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    representative_final = int(manifest["representative_episode"])
    representative_output = next(
        index
        for index, row in enumerate(manifest["entries"])
        if int(row["final_dataset_index"]) == representative_final
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for output_episode in args.episodes:
        entry = manifest["entries"][output_episode]
        for method in args.methods:
            target = OUTPUT / method / f"heldout_{output_episode:02d}_source_{int(entry['final_dataset_index']):02d}"
            if target.exists():
                report_path = target / "rollout_report.json"
                if not report_path.is_file():
                    raise FileExistsError(f"partial existing output requires manual audit: {target}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                runs.append(
                    {
                        "method": method,
                        "heldout_output_episode": output_episode,
                        "source_final_episode": int(entry["final_dataset_index"]),
                        "output": str(target),
                        "process_exit_code": 0,
                        "rollout_status": report["status"],
                        "reused_complete_existing_output": True,
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
            report_path = target / "rollout_report.json"
            report = (
                json.loads(report_path.read_text(encoding="utf-8"))
                if report_path.is_file()
                else None
            )
            run = {
                "method": method,
                "heldout_output_episode": output_episode,
                "source_final_episode": int(entry["final_dataset_index"]),
                "output": str(target),
                "command": command,
                "process_exit_code": completed.returncode,
                "wall_seconds": time.monotonic() - started,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
                "rollout_status": report["status"] if report else "PROCESS_FAILURE",
                "executed_frames": report["executed_frames"] if report else 0,
                "requested_frames": report["requested_frames"] if report else int(entry["frames"]),
                "representative_video_run": output_episode == representative_output,
                "reused_complete_existing_output": False,
            }
            runs.append(run)
            atomic_json(
                OUTPUT / "batch_progress.json",
                {
                    "schema_version": "paper_core_source_rollout_batch_progress_v1",
                    "representative_selection_frozen_before_policy_results": manifest[
                        "representative_episode_rule_frozen_before_policy_results"
                    ],
                    "representative_source_final_episode": representative_final,
                    "representative_heldout_output_episode": representative_output,
                    "runs": runs,
                },
            )
    expected = len(args.methods) * len(args.episodes)
    matrix = {
        "schema_version": "paper_core_source_rollout_batch_v1",
        "status": (
            "PASS"
            if len(runs) == expected
            and all(row["process_exit_code"] == 0 and row["rollout_status"] == "PASS" for row in runs)
            else "INCOMPLETE_OR_SAFETY_ABORT"
        ),
        "run_count": len(runs),
        "expected_run_count": expected,
        "serial_execution": True,
        "same_runner_for_a_b": True,
        "representative_selection_frozen_before_policy_results": manifest[
            "representative_episode_rule_frozen_before_policy_results"
        ],
        "representative_source_final_episode": representative_final,
        "representative_heldout_output_episode": representative_output,
        "runs": runs,
    }
    atomic_json(OUTPUT / "batch_result.json", matrix)
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
