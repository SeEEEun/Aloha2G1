#!/usr/bin/env python3
"""Freeze one bounded Policy-B G1-visual adaptation configuration."""

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
DATASET = ROOT / "datasets/doll_handoff_proposed_b_g1visual_50"
OUTPUT = ROOT / "outputs/policy_b_g1visual/training"
RUN = OUTPUT / "policy_b_g1visual_adaptation_005000"


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


def main() -> int:
    if sha256_file(START / "model.safetensors") != EXPECTED_START_HASH:
        raise RuntimeError("frozen Policy-B start checkpoint hash mismatch")
    package = json.loads(
        (DATASET / "meta/g1visual_packaging_manifest.json").read_text(encoding="utf-8")
    )
    if not package["label_identity"]["STATE_EQUAL"] or not package["label_identity"]["ACTION_EQUAL"]:
        raise RuntimeError("G1-visual dataset label identity has not passed")
    config = json.loads((START / "train_config.json").read_text(encoding="utf-8"))
    config["dataset"]["repo_id"] = "local/doll_handoff_proposed_b_g1visual_50"
    config["dataset"]["root"] = str(DATASET)
    config["dataset"]["video_backend"] = "torchcodec"
    config["dataset"]["image_transforms"]["enable"] = False
    config["output_dir"] = str(RUN)
    config["job_name"] = "POLICY_B_G1VISUAL"
    config["steps"] = 5000
    config["save_freq"] = 1000
    config["log_freq"] = 50
    config["env_eval_freq"] = 0
    config["eval_steps"] = 0
    config["resume"] = False
    config["seed"] = 20260825
    config["batch_size"] = 16
    config["num_workers"] = 4
    config["optimizer"]["lr"] = 2.5e-5
    config["scheduler"] = {
        "type": "cosine_decay_with_warmup",
        "peak_lr": 2.5e-5,
        "decay_lr": 2.5e-6,
        "num_warmup_steps": 100,
        "num_decay_steps": 5000,
    }
    config["policy"]["pretrained_path"] = str(START)
    config["policy"]["pretrained_revision"] = None
    config["policy"]["optimizer_lr"] = 2.5e-5
    config["policy"]["scheduler_warmup_steps"] = 100
    config["policy"]["scheduler_decay_steps"] = 5000
    config["policy"]["scheduler_decay_lr"] = 2.5e-6
    # Preserve architecture and trainable-parameter policy from Policy B.
    invariant_checks = {
        "state_28": config["policy"]["input_features"]["observation.state"]["shape"] == [28],
        "action_28": config["policy"]["output_features"]["action"]["shape"] == [28],
        "max_state_32": config["policy"]["max_state_dim"] == 32,
        "max_action_32": config["policy"]["max_action_dim"] == 32,
        "chunk_50": config["policy"]["chunk_size"] == 50,
        "n_action_steps_50": config["policy"]["n_action_steps"] == 50,
        "only_primary_real_camera": list(config["policy"]["input_features"])[0]
        == "observation.images.cam_high",
        "empty_camera_padding_unchanged": config["policy"]["empty_cameras"] == 2,
        "image_preprocessing_unchanged": config["policy"]["resize_imgs_with_padding"] == [512, 512],
        "normalization_unchanged": config["policy"]["normalization_mapping"]
        == {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
        "freeze_vision_encoder_unchanged": config["policy"]["freeze_vision_encoder"] is True,
        "train_expert_only_unchanged": config["policy"]["train_expert_only"] is True,
        "amp_unchanged": config["policy"]["use_amp"] is True,
    }
    if not all(invariant_checks.values()):
        raise RuntimeError(f"Policy-B adaptation invariant failed: {invariant_checks}")
    config_path = OUTPUT / "policy_b_g1visual_adaptation_config.json"
    atomic_json(config_path, config)
    executable = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train")
    command = f"{executable} --config_path={config_path}"
    (OUTPUT / "TRAIN_COMMAND.txt").write_text(command + "\n", encoding="utf-8")
    decision = {
        "schema_version": "policy_b_g1visual_adaptation_v1",
        "status": "READY_TO_TRAIN",
        "hypothesis": "adapt Policy B to G1-visible SOURCE_LIKE_CAM_HIGH observations",
        "start_checkpoint": str(START),
        "start_model_sha256": EXPECTED_START_HASH,
        "dataset": str(DATASET),
        "dataset_content_tree_sha256": package[
            "dataset_content_tree_sha256_excluding_this_manifest"
        ],
        "output_run": str(RUN),
        "adaptation_steps": 5000,
        "learning_rate": {
            "peak": 2.5e-5,
            "warmup_steps": 100,
            "decay_steps": 5000,
            "final": 2.5e-6,
            "justification": (
                "approximately the original 20k run's continuation-scale cosine LR; "
                "avoids restarting the trained model at the original 1e-4 peak"
            ),
        },
        "batch_size": 16,
        "optimizer": config["optimizer"],
        "amp": True,
        "architecture_changed": False,
        "normalization_semantics_changed": False,
        "only_training_observation_image": "observation.images.cam_high",
        "invariant_checks": invariant_checks,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "command": command,
        "not_a_hyperparameter_search": True,
        "real_robot_invoked": False,
    }
    atomic_json(OUTPUT / "adaptation_plan.json", decision)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
