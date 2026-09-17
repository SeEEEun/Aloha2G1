#!/usr/bin/env python3
"""Audit the bounded POLICY_B_G1VISUAL adaptation run and freeze its checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs/policy_b_g1visual/training/policy_b_g1visual_adaptation_005000"
DEFAULT_LOG = ROOT / "outputs/policy_b_g1visual/training/training_console.log"
START = ROOT / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2/checkpoints/020000/pretrained_model"
START_HASH = "bfe3e2aa6529967a12831a6f0bb91104b704835b2f43733072489b9ea2b68395"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_human_step(value: str) -> int:
    multipliers = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}
    suffix = value[-1].upper() if value[-1].isalpha() else ""
    number = float(value[:-1] if suffix else value)
    return int(round(number * multipliers.get(suffix, 1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run = args.run.resolve()
    log = args.log.resolve()
    output = args.output.resolve() if args.output else run.parent / "training_audit.json"
    if sha256_file(START / "model.safetensors") != START_HASH:
        raise RuntimeError("frozen start checkpoint changed")
    checkpoint_rows = []
    for checkpoint in sorted((run / "checkpoints").glob("[0-9]*")):
        model_dir = checkpoint / "pretrained_model"
        model = model_dir / "model.safetensors"
        state = checkpoint / "training_state/training_step.json"
        if not model.is_file() or not state.is_file():
            raise RuntimeError(f"incomplete checkpoint {checkpoint}")
        training_step = json.loads(state.read_text(encoding="utf-8"))
        step = int(training_step.get("step", training_step.get("training_step", -1)))
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        finite_tensors = True
        tensor_count = 0
        scalar_count = 0
        # Use the PyTorch backend so bfloat16 checkpoint tensors can be
        # inspected without an unsupported NumPy dtype conversion.
        with safe_open(model, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                tensor = archive.get_tensor(key)
                tensor_count += 1
                scalar_count += int(tensor.numel())
                if not bool(torch.isfinite(tensor).all()):
                    finite_tensors = False
                    break
        checks = {
            "directory_step_matches_state": step == int(checkpoint.name),
            "state_28": config["input_features"]["observation.state"]["shape"] == [28],
            "action_28": config["output_features"]["action"]["shape"] == [28],
            "chunk_50": config["chunk_size"] == 50,
            "max_state_action_32": config["max_state_dim"] == 32 and config["max_action_dim"] == 32,
            "primary_cam_high": "observation.images.cam_high" in config["input_features"],
            "weights_finite": finite_tensors,
        }
        if not all(checks.values()):
            raise RuntimeError(f"checkpoint {checkpoint.name} audit failed: {checks}")
        checkpoint_rows.append({
            "step": step,
            "checkpoint": str(model_dir),
            "model_sha256": sha256_file(model),
            "timestamp": model.stat().st_mtime,
            "tensor_count": tensor_count,
            "scalar_count": scalar_count,
            "checks": checks,
        })
    if [row["step"] for row in checkpoint_rows] != [1000, 2000, 3000, 4000, 5000]:
        raise RuntimeError(f"adaptation checkpoint schedule incomplete: {checkpoint_rows}")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    matches = [
        {"step": parse_human_step(step), "loss": float(loss)}
        for step, loss in re.findall(
            r"step:(\d+(?:\.\d+)?[KMG]?).*?\bloss:([0-9.eE+-]+)",
            log_text,
            flags=re.IGNORECASE,
        )
    ]
    if not matches or matches[-1]["step"] != 5000:
        raise RuntimeError("training log lacks a final step-5000 finite loss")
    if not all(np.isfinite(row["loss"]) for row in matches):
        raise RuntimeError("training loss contains NaN/Inf")
    selected = checkpoint_rows[-1]
    result = {
        "schema_version": "policy_b_g1visual_training_audit_v1",
        "status": "PASS",
        "experiment": "POLICY_B_G1VISUAL",
        "start_checkpoint": str(START),
        "start_model_sha256": START_HASH,
        "adaptation_steps": 5000,
        "available_checkpoints": checkpoint_rows,
        "selected_checkpoint": selected,
        "selection": "bounded run final checkpoint; no validation evidence favors an earlier checkpoint before phase probe",
        "loss_log_points": matches,
        "final_logged_loss": matches[-1]["loss"],
        "loss_all_finite": True,
        "training_log": str(log),
        "training_log_sha256": sha256_file(log),
        "real_robot_invoked": False,
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": "PASS",
        "selected_checkpoint": selected["checkpoint"],
        "model_sha256": selected["model_sha256"],
        "final_loss": matches[-1]["loss"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
