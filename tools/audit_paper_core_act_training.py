#!/usr/bin/env python3
"""Audit paired ACT-A40/B40 training completion and checkpoint health."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
CONTRACT = PAPER / "act_a_b_training_contract.json"
EXPECTED_STEPS = (20_000, 40_000, 60_000, 80_000, 100_000)
LOG_PATTERN = re.compile(
    r"step:(?P<step>\d+)K .*?loss:(?P<loss>[-+0-9.eE]+) "
    r"grdn:(?P<gradient>[-+0-9.eE]+).*?l1_loss:(?P<l1>[-+0-9.eE]+) "
    r"kld_loss:(?P<kld>[-+0-9.eE]+)"
)


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


def finite_safetensors(path: Path) -> dict[str, Any]:
    tensor_count = 0
    element_count = 0
    with safe_open(path, framework="pt", device="cpu") as archive:
        for key in archive.keys():
            value = archive.get_tensor(key)
            tensor_count += 1
            element_count += value.numel()
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite tensor in {path}: {key}")
    return {
        "finite": True,
        "tensor_count": tensor_count,
        "element_count": element_count,
    }


def audit_method(method: str, contract: dict[str, Any]) -> dict[str, Any]:
    variant = PAPER / f"act_{method}40"
    launch_path = variant / "training_launch.json"
    log_path = variant / "training_stdout.log"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("status") != "COMPLETE" or launch.get("return_code") != 0:
        raise RuntimeError(f"ACT-{method.upper()}40 training did not complete successfully")
    if sha256_file(log_path) != launch["training_log_sha256"]:
        raise RuntimeError(f"ACT-{method.upper()}40 training log hash changed")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in text:
        raise RuntimeError(f"ACT-{method.upper()}40 log contains a traceback")
    records = []
    for match in LOG_PATTERN.finditer(text):
        record = {
            "step_k": int(match.group("step")),
            "loss": float(match.group("loss")),
            "gradient_norm": float(match.group("gradient")),
            "l1_loss": float(match.group("l1")),
            "kld_loss": float(match.group("kld")),
        }
        if not all(math.isfinite(value) for key, value in record.items() if key != "step_k"):
            raise RuntimeError(f"non-finite training metric ACT-{method.upper()}40: {record}")
        records.append(record)
    if not records:
        raise RuntimeError(f"no parsed health records ACT-{method.upper()}40")
    checkpoint_records = []
    for step in EXPECTED_STEPS:
        checkpoint = variant / "train/checkpoints" / f"{step:06d}"
        model = checkpoint / "pretrained_model/model.safetensors"
        state = checkpoint / "training_state/training_step.json"
        if not model.is_file() or not state.is_file():
            raise FileNotFoundError(f"missing periodic checkpoint {checkpoint}")
        saved_step = json.loads(state.read_text(encoding="utf-8"))["step"]
        if int(saved_step) != step:
            raise RuntimeError(f"checkpoint training step mismatch: {checkpoint}")
        checkpoint_records.append(
            {
                "step": step,
                "pretrained_model": str(checkpoint / "pretrained_model"),
                "model_sha256": sha256_file(model),
                "model_bytes": model.stat().st_size,
                "training_state_step": int(saved_step),
            }
        )
    final_health = finite_safetensors(
        variant / "train/checkpoints/100000/pretrained_model/model.safetensors"
    )
    configured = contract["records"][method]
    if sha256_file(Path(configured["config"])) != configured["config_sha256"]:
        raise RuntimeError(f"ACT-{method.upper()}40 frozen config changed")
    return {
        "method": f"ACT-{method.upper()}40",
        "status": "PASS",
        "training_steps": 100_000,
        "launch": str(launch_path),
        "launch_sha256": sha256_file(launch_path),
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "elapsed_seconds": launch["elapsed_seconds"],
        "health_record_count": len(records),
        "all_logged_losses_finite": True,
        "all_logged_gradients_finite": True,
        "last_logged_health": records[-1],
        "minimum_logged_loss": min(row["loss"] for row in records),
        "maximum_logged_gradient_norm": max(row["gradient_norm"] for row in records),
        "periodic_checkpoints": checkpoint_records,
        "final_checkpoint_tensor_audit": final_health,
        "final_checkpoint": checkpoint_records[-1]["pretrained_model"],
        "final_checkpoint_model_sha256": checkpoint_records[-1]["model_sha256"],
    }


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    methods = {method: audit_method(method, contract) for method in ("a", "b")}
    result = {
        "schema_version": "paper_core_act_a_b_training_audit_v1",
        "status": "PASS",
        "training_contract": str(CONTRACT),
        "training_contract_sha256": sha256_file(CONTRACT),
        "paired_config_hash": contract["common_config_canonical_sha256"],
        "paired_architecture_hash": contract["policy_architecture_canonical_sha256"],
        "identical_budget": contract["fixed_budget"],
        "methods": methods,
        "checkpoint_strict_reload_and_finite_inference": "verified during common Experiment-2 evaluator",
        "real_hardware": False,
    }
    atomic_json(PAPER / "act_a_b_training_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
