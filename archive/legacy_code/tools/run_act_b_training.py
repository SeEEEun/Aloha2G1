#!/usr/bin/env python3
"""Launch the single frozen-config ACT-B run with an auditable console log."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT_ROOT = ROOT / "outputs/policy_b_act"
CONFIG = OUTPUT_ROOT / "config/train_config.json"
TRAIN_DIR = OUTPUT_ROOT / "train"
LOG = OUTPUT_ROOT / "training_stdout.log"
LAUNCH_RECORD = OUTPUT_ROOT / "training_launch.json"
TRAIN_EXECUTABLE = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train")
EXPECTED_CONFIG_SHA256 = "6abfdc98790c5908c550347dac6281ef11153eace3fe9d9f589f0b0ce8a2f804"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if not CONFIG.is_file():
        raise FileNotFoundError(CONFIG)
    actual_hash = sha256_file(CONFIG)
    if actual_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(f"Frozen training config hash mismatch: {actual_hash}")
    if TRAIN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite ACT-B training directory: {TRAIN_DIR}")
    if LOG.exists():
        raise FileExistsError(f"Refusing to overwrite ACT-B training log: {LOG}")

    command = [str(TRAIN_EXECUTABLE), "--config_path", str(CONFIG)]
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
    record = {
        "schema_version": "act_b_training_launch_v1",
        "status": "STARTING",
        "command_argv": command,
        "command_shell": " ".join(command),
        "config": str(CONFIG),
        "config_sha256": actual_hash,
        "working_directory": str(ROOT),
        "environment": {
            key: environment[key]
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "TOKENIZERS_PARALLELISM",
                "PYTHONUNBUFFERED",
            ]
        },
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_log": str(LOG),
    }
    atomic_json(LAUNCH_RECORD, record)
    start = time.monotonic()
    with LOG.open("x", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        record["status"] = "RUNNING"
        record["pid"] = process.pid
        atomic_json(LAUNCH_RECORD, record)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
        return_code = process.wait()

    record["return_code"] = return_code
    record["elapsed_seconds"] = time.monotonic() - start
    record["end_time_utc"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "COMPLETE" if return_code == 0 else "FAILED"
    record["training_log_sha256"] = sha256_file(LOG)
    atomic_json(LAUNCH_RECORD, record)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
