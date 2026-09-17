#!/usr/bin/env python3
"""Prepare two bounded, paired-domain Policy-B visual rehearsal experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
START = (
    ROOT
    / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2/checkpoints/020000/pretrained_model"
)
EXPECTED_START_HASH = "bfe3e2aa6529967a12831a6f0bb91104b704835b2f43733072489b9ea2b68395"
DATASET = ROOT / "datasets/doll_handoff_proposed_b_rehearsal_100"
DATASET_VALIDATION = ROOT / "outputs/policy_b_g1visual/rehearsal/dataset_validation.json"
OUTPUT = ROOT / "outputs/policy_b_g1visual/rehearsal/training"
R1_RUN = OUTPUT / "R1_paired_expert_001500"
R2_RUN = OUTPUT / "R2_paired_visual_connector_001500"


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


def experiment_config(run: Path, job_name: str) -> dict[str, Any]:
    config = json.loads((START / "train_config.json").read_text(encoding="utf-8"))
    config["dataset"]["repo_id"] = "local/doll_handoff_proposed_b_rehearsal_100"
    config["dataset"]["root"] = str(DATASET)
    config["dataset"]["video_backend"] = "torchcodec"
    config["dataset"]["image_transforms"]["enable"] = False
    config["output_dir"] = str(run)
    config["job_name"] = job_name
    config["steps"] = 1500
    config["save_freq"] = 500
    config["log_freq"] = 50
    config["env_eval_freq"] = 0
    config["eval_steps"] = 0
    config["resume"] = False
    config["seed"] = 20260825
    config["batch_size"] = 16
    config["num_workers"] = 4
    config["optimizer"]["lr"] = 1.0e-5
    config["scheduler"] = {
        "type": "cosine_decay_with_warmup",
        "peak_lr": 1.0e-5,
        "decay_lr": 1.0e-6,
        "num_warmup_steps": 50,
        "num_decay_steps": 1500,
    }
    config["policy"]["pretrained_path"] = str(START)
    config["policy"]["pretrained_revision"] = None
    config["policy"]["optimizer_lr"] = 1.0e-5
    config["policy"]["scheduler_warmup_steps"] = 50
    config["policy"]["scheduler_decay_steps"] = 1500
    config["policy"]["scheduler_decay_lr"] = 1.0e-6
    return config


def main() -> int:
    if sha256_file(START / "model.safetensors") != EXPECTED_START_HASH:
        raise RuntimeError("frozen original Policy-B checkpoint hash mismatch")
    validation = json.loads(DATASET_VALIDATION.read_text(encoding="utf-8"))
    if validation["status"] != "PASS" or not validation["paired_supervision_exact"]:
        raise RuntimeError("paired rehearsal dataset has not passed exact-supervision validation")
    r1 = experiment_config(R1_RUN, "POLICY_B_G1VISUAL_REHEARSAL_R1")
    r2 = experiment_config(R2_RUN, "POLICY_B_G1VISUAL_REHEARSAL_R2_CONNECTOR")
    invariant_checks = {
        "state_28": r1["policy"]["input_features"]["observation.state"]["shape"] == [28],
        "action_28": r1["policy"]["output_features"]["action"]["shape"] == [28],
        "max_state_action_32": r1["policy"]["max_state_dim"] == 32
        and r1["policy"]["max_action_dim"] == 32,
        "chunk_50": r1["policy"]["chunk_size"] == 50,
        "only_cam_high_real_input": list(r1["policy"]["input_features"])[0]
        == "observation.images.cam_high",
        "empty_camera_padding_2": r1["policy"]["empty_cameras"] == 2,
        "normalization_unchanged": r1["policy"]["normalization_mapping"]
        == {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
        "start_from_original_policy": r1["policy"]["pretrained_path"] == str(START),
        "paired_dataset": r1["dataset"]["root"] == str(DATASET),
        "no_image_augmentation": r1["dataset"]["image_transforms"]["enable"] is False,
        "amp": r1["policy"]["use_amp"] is True,
    }
    if not all(invariant_checks.values()) or r1 != {**r2, "output_dir": str(R1_RUN), "job_name": "POLICY_B_G1VISUAL_REHEARSAL_R1"}:
        # The second comparison is controlled: only the process-level trainable
        # subset and output identity differ, not data or hyperparameters.
        expected_r2 = dict(r1)
        expected_r2["output_dir"] = str(R2_RUN)
        expected_r2["job_name"] = "POLICY_B_G1VISUAL_REHEARSAL_R2_CONNECTOR"
        if r2 != expected_r2:
            raise RuntimeError("R1/R2 configs differ beyond output/job identity")
    r1_path = OUTPUT / "R1_config.json"
    r2_path = OUTPUT / "R2_config.json"
    atomic_json(r1_path, r1)
    atomic_json(r2_path, r2)
    executable = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train")
    connector_launcher = ROOT / "tools/run_smolvla_connector_only_train.py"
    r1_command = f"{executable} --config_path={r1_path}"
    r2_command = (
        f"/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python {connector_launcher} "
        f"--config_path={r2_path}"
    )
    plan = {
        "status": "READY_TO_TRAIN",
        "root_cause": "VISUAL_GAP_DOMINANT",
        "start_checkpoint": str(START),
        "start_model_sha256": EXPECTED_START_HASH,
        "paired_dataset": str(DATASET),
        "paired_dataset_validation": str(DATASET_VALIDATION),
        "balanced_frames": {"ALOHA_RGB": 34478, "G1_RGB": 34478},
        "common_setup": {
            "steps": 1500,
            "save_every_steps": 500,
            "batch_size": 16,
            "peak_learning_rate": 1.0e-5,
            "final_learning_rate": 1.0e-6,
            "warmup_steps": 50,
            "optimizer": "AdamW",
            "amp": True,
            "seed": 20260825,
        },
        "R1": {
            "purpose": "lower-LR paired rehearsal with the checkpoint's current trainable set",
            "trainable_policy": "installed SmolVLA train_expert_only=True plus train_state_proj=True",
            "expected_trainable_components": [
                "action expert",
                "state projection",
                "action input/output projections",
                "action-time MLP",
            ],
            "config": str(r1_path),
            "config_sha256": sha256_file(r1_path),
            "run": str(R1_RUN),
            "command": r1_command,
        },
        "R2": {
            "purpose": "paired visual-side projection-only adaptation",
            "trainable_policy": "only the installed SmolVLM connector object; every other parameter frozen",
            "semantic_module": "policy.model.vlm_with_expert.vlm.model.connector",
            "expected_checkpoint_tensor": "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
            "expected_trainable_scalars": 11796480,
            "peft_lora_not_used": "the installed environment has no peft package",
            "config": str(r2_path),
            "config_sha256": sha256_file(r2_path),
            "run": str(R2_RUN),
            "command": r2_command,
        },
        "invariant_checks": invariant_checks,
        "frozen_inputs_modified": False,
        "real_robot_invoked": False,
    }
    atomic_json(OUTPUT / "experiment_plan.json", plan)
    (OUTPUT / "R1_COMMAND.txt").write_text(r1_command + "\n", encoding="utf-8")
    (OUTPUT / "R2_COMMAND.txt").write_text(r2_command + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
