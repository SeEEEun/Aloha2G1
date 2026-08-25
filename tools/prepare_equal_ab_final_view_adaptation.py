#!/usr/bin/env python3
"""Prepare, but never launch, equal A/B final-helmet rehearsal training configs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from deployment_camera_config import camera_manifest_record, load_camera_config, sha256_file  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_config(base: dict[str, Any], variant: str, row: dict[str, Any], shared: dict[str, Any], output: Path) -> dict[str, Any]:
    adaptation = shared["visual_adaptation"]
    config = deepcopy(base)
    config["dataset"]["repo_id"] = f"local/doll_handoff_policy_{variant.lower()}_finalview_rehearsal_100"
    config["dataset"]["root"] = str((ROOT / row["paired_dataset"]).resolve())
    config["dataset"]["video_backend"] = "torchcodec"
    config["dataset"]["image_transforms"]["enable"] = bool(adaptation["image_augmentation"])
    config["output_dir"] = str((output / f"policy_{variant.lower()}_equal_finalview_001500").resolve())
    config["job_name"] = f"POLICY_{variant}_EQUAL_FINAL_HELMET_REHEARSAL"
    config["steps"] = int(adaptation["steps"])
    config["save_freq"] = 500
    config["log_freq"] = 50
    config["env_eval_freq"] = 0
    config["eval_steps"] = 0
    config["resume"] = False
    config["seed"] = int(adaptation["seed"])
    config["batch_size"] = int(adaptation["batch_size"])
    config["optimizer"]["lr"] = float(adaptation["peak_learning_rate"])
    config["scheduler"] = {
        "type": "cosine_decay_with_warmup",
        "peak_lr": float(adaptation["peak_learning_rate"]),
        "decay_lr": float(adaptation["final_learning_rate"]),
        "num_warmup_steps": int(adaptation["warmup_steps"]),
        "num_decay_steps": int(adaptation["steps"]),
    }
    start = str((ROOT / row["source_checkpoint"]).resolve())
    config["policy"]["pretrained_path"] = start
    config["policy"]["pretrained_revision"] = None
    config["policy"]["optimizer_lr"] = float(adaptation["peak_learning_rate"])
    config["policy"]["scheduler_warmup_steps"] = int(adaptation["warmup_steps"])
    config["policy"]["scheduler_decay_steps"] = int(adaptation["steps"])
    config["policy"]["scheduler_decay_lr"] = float(adaptation["final_learning_rate"])
    return config


def normalized_for_equality(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["dataset"]["repo_id"] = "<VARIANT_PAIRED_DATASET>"
    result["dataset"]["root"] = "<VARIANT_PAIRED_DATASET>"
    result["output_dir"] = "<VARIANT_OUTPUT>"
    result["job_name"] = "<VARIANT_JOB>"
    result["policy"]["pretrained_path"] = "<VARIANT_SOURCE_CHECKPOINT>"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path, default=ROOT / "configs/final_view_ab_pipeline.json")
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pre_mount_readiness/final_view_ab/adaptation")
    args = parser.parse_args()
    pipeline = read_json(args.pipeline_config)
    camera_path = args.camera_config or ROOT / pipeline["camera_config"]
    camera = load_camera_config(camera_path, purpose="equal A/B final-view adaptation preparation", allow_pending=True)
    output = args.output.resolve()
    variants = pipeline["dataset_variants"]
    gates: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    fallback = ROOT / variants["B"]["source_checkpoint"] / "train_config.json"
    for variant in ("A", "B"):
        row = variants[variant]
        checkpoint = ROOT / row["source_checkpoint"]
        paired = ROOT / row["paired_dataset"]
        model = checkpoint / "model.safetensors"
        paired_manifest = paired / "meta/paired_visual_rehearsal_manifest.json"
        ready = model.is_file() and (checkpoint / "train_config.json").is_file() and paired_manifest.is_file()
        gates[variant] = {
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_ready": model.is_file(),
            "paired_dataset": str(paired),
            "paired_dataset_ready": paired_manifest.is_file(),
        }
        template_path = checkpoint / "train_config.json" if (checkpoint / "train_config.json").is_file() else fallback
        if not template_path.is_file():
            raise FileNotFoundError("neither variant source checkpoint nor frozen Policy-B template is available")
        configs[variant] = make_config(
            read_json(template_path), variant, row, pipeline["shared_contract"], output
        )
        config_path = output / f"policy_{variant.lower()}_config.json"
        atomic_json(config_path, configs[variant])
        gates[variant]["executable_config"] = ready and camera.is_final_helmet
        gates[variant]["config"] = str(config_path)
        gates[variant]["config_sha256"] = sha256_file(config_path)

    equal = normalized_for_equality(configs["A"]) == normalized_for_equality(configs["B"])
    if not equal:
        raise RuntimeError("A/B final-view configs differ beyond variant paths/identity")
    executable = all(gates[v]["executable_config"] for v in ("A", "B"))
    plan = {
        "schema_version": "equal_ab_final_view_adaptation_plan_v1",
        "status": "READY_TO_LAUNCH_MANUALLY" if executable else "PREPARED_DISABLED_GATES_UNMET",
        "camera": camera_manifest_record(camera),
        "camera_final": camera.is_final_helmet,
        "a_b_hyperparameters_equal": equal,
        "shared_strategy": pipeline["shared_contract"]["visual_adaptation"],
        "checkpoint_selection_rule": pipeline["shared_contract"]["checkpoint_selection_rule"],
        "gates": gates,
        "commands": {
            variant: (
                f"/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/lerobot-train "
                f"--config_path={gates[variant]['config']}"
            )
            for variant in ("A", "B")
        },
        "commands_executed": False,
        "gpu_training_started": False,
        "real_robot_invoked": False,
    }
    atomic_json(output / "plan.json", plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
