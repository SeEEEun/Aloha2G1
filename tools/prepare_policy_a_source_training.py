#!/usr/bin/env python3
"""Gate Policy-A source training and freeze the exact Policy-B-equivalent config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from copy import deepcopy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
B_CONFIG = ROOT / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2/checkpoints/020000/pretrained_model/train_config.json"
A_DATASET = ROOT / "datasets/doll_handoff_trajectory_a_50"
A_OUTPUT = ROOT / "outputs/policy_a_doll_handoff_trajectory_a_50_lag1_state_v2"
A_VALIDATION = ROOT / "outputs/pre_mount_readiness/dataset_a/dataset_a_validation.json"
OUTPUT = ROOT / "outputs/pre_mount_readiness/policy_a_source"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    validation = read_json(A_VALIDATION)
    base = read_json(B_CONFIG)
    config = deepcopy(base)
    config["dataset"]["repo_id"] = "local/doll_handoff_trajectory_a_50"
    config["dataset"]["root"] = str(A_DATASET)
    config["output_dir"] = str(A_OUTPUT)
    config["job_name"] = "policy_a_doll_handoff_trajectory_a_50_lag1_state_v2"
    config_path = OUTPUT / "policy_a_source_train_config.json"
    atomic_json(config_path, config)
    invariants = {
        "pretrained_base": config["policy"]["pretrained_path"] == base["policy"]["pretrained_path"],
        "steps": config["steps"] == base["steps"] == 20000,
        "batch_size": config["batch_size"] == base["batch_size"] == 16,
        "optimizer": config["optimizer"] == base["optimizer"],
        "scheduler": config["scheduler"] == base["scheduler"],
        "chunk_50": config["policy"]["chunk_size"] == base["policy"]["chunk_size"] == 50,
        "state_28": config["policy"]["input_features"]["observation.state"]["shape"] == [28],
        "action_28": config["policy"]["output_features"]["action"]["shape"] == [28],
        "normalization": config["policy"]["normalization_mapping"] == base["policy"]["normalization_mapping"],
        "source_camera_key": next(iter(config["policy"]["input_features"])) == "observation.images.cam_high",
        "seed": config["seed"] == base["seed"],
        "image_augmentation": config["dataset"]["image_transforms"]["enable"] == base["dataset"]["image_transforms"]["enable"],
    }
    if not all(invariants.values()):
        raise RuntimeError(f"Policy A/B source-domain condition mismatch: {invariants}")
    structurally_valid = validation.get("status") == "PASS" and A_DATASET.is_dir()
    command = (
        "/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train "
        f"--config_path={config_path}"
    )
    gate = {
        "schema_version": "policy_a_source_training_gate_v1",
        "status": "READY_TO_TRAIN_MANUALLY" if structurally_valid else "NOT_STARTED_BLOCKED_BY_DATASET_A_HARD_FAIL",
        "dataset_a_validation": str(A_VALIDATION),
        "dataset_a_validation_status": validation.get("status"),
        "dataset_a_hard_fail_count": len(validation.get("hard_fails", [])),
        "dataset_a_exists": A_DATASET.is_dir(),
        "source_conditions_equal_to_original_policy_b": all(invariants.values()),
        "invariants": invariants,
        "config": str(config_path),
        "command": command,
        "command_executed": False,
        "checkpoint_created": False,
        "nine_phase_probe": "NOT_RUN_NO_POLICY_A_CHECKPOINT",
        "gpu_training_started": False,
        "real_robot_invoked": False,
    }
    atomic_json(OUTPUT / "training_gate.json", gate)
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
