#!/usr/bin/env python3
"""Launch one of the two frozen paper-core ACT training jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
CONTRACT = PAPER / "act_a_b_training_contract.json"
TRAIN_EXECUTABLE = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("a", "b"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract["status"] != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("paper ACT training contract is not frozen")
    record = contract["records"][args.method]
    config = Path(record["config"])
    if sha256_file(config) != record["config_sha256"]:
        raise RuntimeError("frozen training config hash mismatch")
    variant = PAPER / f"act_{args.method}40"
    train_dir = variant / "train"
    log_path = variant / "training_stdout.log"
    launch_path = variant / "training_launch.json"
    for path in (train_dir, log_path, launch_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite training artifact: {path}")

    command = [str(TRAIN_EXECUTABLE), "--config_path", str(config)]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    launch = {
        "schema_version": "paper_core_act_training_launch_v1",
        "method": "ACT-A40" if args.method == "a" else "ACT-B40",
        "status": "STARTING",
        "command_argv": command,
        "config": str(config),
        "config_sha256": record["config_sha256"],
        "training_contract": str(CONTRACT),
        "training_contract_sha256": sha256_file(CONTRACT),
        "common_config_canonical_sha256": contract["common_config_canonical_sha256"],
        "policy_architecture_canonical_sha256": contract["policy_architecture_canonical_sha256"],
        "working_directory": str(ROOT),
        "environment": {
            key: environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "PYTHONUNBUFFERED",
            )
        },
        "stdout_log": str(log_path),
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "real_hardware": False,
    }
    atomic_json(launch_path, launch)
    start = time.monotonic()
    with log_path.open("x", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        launch["status"] = "RUNNING"
        launch["pid"] = process.pid
        atomic_json(launch_path, launch)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
        return_code = process.wait()

    launch.update(
        {
            "return_code": return_code,
            "elapsed_seconds": time.monotonic() - start,
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "status": "COMPLETE" if return_code == 0 else "FAILED",
            "training_log_sha256": sha256_file(log_path),
        }
    )
    atomic_json(launch_path, launch)
    if return_code != 0:
        raise SystemExit(return_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
