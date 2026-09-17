#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import draccus
import lerobot
import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401 - registers config choice
from lerobot.configs.train import TrainPipelineConfig

from tools.g1_policy_dataset_packaging_v1.training import config_fairness


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse and fairness-check paired LeRobot training configs")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(args.config_a).resolve(), Path(args.config_b).resolve()]
    rows = []
    for path in paths:
        persisted = json.loads(path.read_text(encoding="utf-8"))
        with path.open(encoding="utf-8") as stream, draccus.config_type("json"):
            config = draccus.load(TrainPipelineConfig, stream)
        config.validate()
        rows.append(
            {
                "path": str(path),
                "parse": "PASS",
                "validate": "PASS",
                "dataset_root": config.dataset.root,
                "policy_type": config.policy.type,
                "state_shape": list(config.policy.input_features["observation.state"].shape),
                "action_shape": list(config.policy.output_features["action"].shape),
                "pretrained_path": str(config.policy.pretrained_path),
                "batch_size": config.batch_size,
                "steps": config.steps,
                "seed": config.seed,
                "configured_device_in_file": persisted["policy"]["device"],
                "runtime_validation_device": config.policy.device,
            }
        )
    result = {
        "schema_version": "g1_policy_dataset_packaging_v1_training_config_validation",
        "status": "PASS",
        "lerobot_version": lerobot.__version__,
        "configs": rows,
        "fairness": config_fairness(*paths),
        "training_executed": False,
        "note": (
            "Runtime validation may fall back to CPU when CUDA is hidden from the validation subprocess; "
            "the persisted configs remain cuda and no training is started."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
