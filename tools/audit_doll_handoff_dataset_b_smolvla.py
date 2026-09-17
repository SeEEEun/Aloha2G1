#!/usr/bin/env python3
"""LeRobot/SmolVLA preflight and smoke-checkpoint audit for Dataset B.

This helper is deliberately executed inside the installed ``lerobot-smolvla``
environment.  It does not train a policy; training is launched separately by
the finalization driver and this helper only validates configuration/model
compatibility or reloads the resulting short-smoke checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


FPS = 30.0
ACTION_CHUNK = 50
STATE_DIM = 28
ACTION_DIM = 28


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_pipeline_config(path: Path):
    import draccus
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.train import TrainPipelineConfig

    with path.open(encoding="utf-8") as stream, draccus.config_type("json"):
        config = draccus.load(TrainPipelineConfig, stream)
    config.validate()
    return config


def safetensor_shapes(path: Path, needles: tuple[str, ...]) -> dict[str, list[int]]:
    matches: dict[str, list[int]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if any(needle in key for needle in needles):
                matches[key] = list(handle.get_slice(key).get_shape())
    return matches


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    import lerobot

    dataset = Path(args.dataset).resolve()
    base = Path(args.base_model).resolve()
    configs = [Path(path).resolve() for path in args.config]
    info = read_json(dataset / "meta/info.json")
    stats = read_json(dataset / "meta/stats.json")
    base_config = read_json(base / "config.json")
    model_path = base / "model.safetensors"
    model_shapes = safetensor_shapes(
        model_path, ("state_proj", "action_in_proj", "action_out_proj")
    )
    parsed = []
    for path in configs:
        persisted = read_json(path)
        config = load_pipeline_config(path)
        parsed.append(
            {
                "path": str(path),
                "parse": "PASS",
                "validate": "PASS",
                "policy_type": config.policy.type,
                "dataset_root": str(config.dataset.root),
                "state_shape": list(
                    config.policy.input_features["observation.state"].shape
                ),
                "action_shape": list(config.policy.output_features["action"].shape),
                "max_state_dim": int(config.policy.max_state_dim),
                "max_action_dim": int(config.policy.max_action_dim),
                "pretrained_path": str(config.policy.pretrained_path),
                "batch_size": int(config.batch_size),
                "steps": int(config.steps),
                "learning_rate": float(config.optimizer.lr),
                "device": persisted["policy"]["device"],
                "use_amp": bool(persisted["policy"]["use_amp"]),
                "normalization_mapping": persisted["policy"][
                    "normalization_mapping"
                ],
            }
        )
    state_stats = stats["observation.state"]
    action_stats = stats["action"]
    statistics_finite = all(
        math.isfinite(float(value))
        for feature in (state_stats, action_stats)
        for field in ("min", "max", "mean", "std")
        for value in feature[field]
    )
    checks = {
        "lerobot_version_0_6_1": lerobot.__version__ == "0.6.1",
        "dataset_50_episodes": int(info["total_episodes"]) == 50,
        "dataset_34478_frames": int(info["total_frames"]) == 34478,
        "dataset_single_cam_high": list(
            key
            for key, feature in info["features"].items()
            if feature.get("dtype") == "video"
        )
        == ["observation.images.cam_high"],
        "dataset_state_28": info["features"]["observation.state"]["shape"]
        == [STATE_DIM],
        "dataset_action_28": info["features"]["action"]["shape"]
        == [ACTION_DIM],
        "state_action_named_order_equal": info["features"]["observation.state"][
            "names"
        ]
        == info["features"]["action"]["names"],
        "base_source_state_6": base_config["input_features"]["observation.state"][
            "shape"
        ]
        == [6],
        "base_source_action_6": base_config["output_features"]["action"]["shape"]
        == [6],
        "base_max_state_32": int(base_config["max_state_dim"]) == 32,
        "base_max_action_32": int(base_config["max_action_dim"]) == 32,
        "base_state_projection_accepts_32": model_shapes.get(
            "model.state_proj.weight"
        )
        == [960, 32],
        "base_action_input_accepts_32": model_shapes.get(
            "model.action_in_proj.weight"
        )
        == [720, 32],
        "base_action_output_emits_32": model_shapes.get(
            "model.action_out_proj.weight"
        )
        == [32, 720],
        "dataset_dims_fit_padded_architecture": STATE_DIM <= 32
        and ACTION_DIM <= 32,
        "dataset_statistics_finite": statistics_finite,
        "all_configs_parse_validate": len(parsed) == 2
        and all(
            row["policy_type"] == "smolvla"
            and row["dataset_root"] == str(dataset)
            and row["state_shape"] == [STATE_DIM]
            and row["action_shape"] == [ACTION_DIM]
            and row["max_state_dim"] == 32
            and row["max_action_dim"] == 32
            and row["pretrained_path"] == str(base)
            and row["normalization_mapping"]
            == {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
            for row in parsed
        ),
    }
    result = {
        "schema_version": "doll_handoff_dataset_b_smolvla_preflight_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "lerobot_version": lerobot.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checks": checks,
        "configs": parsed,
        "pretrained_logical_source_interface": {
            "observation_keys": list(base_config["input_features"]),
            "state_dimension": 6,
            "action_dimension": 6,
        },
        "dataset_interface": {
            "observation_keys": [
                "observation.images.cam_high",
                "observation.state",
            ],
            "state_dimension": STATE_DIM,
            "action_dimension": ACTION_DIM,
        },
        "padded_checkpoint_architecture": {
            "max_state_dimension": 32,
            "max_action_dimension": 32,
            "projection_tensor_shapes": model_shapes,
        },
        "adaptation": {
            "mechanism": (
                "LeRobot SmolVLA logical feature config is set to 28D; the model pads "
                "state/action vectors to the checkpoint's unchanged 32D projection "
                "architecture and slices predictions/losses back to logical action 28D."
            ),
            "aloha_14d_reshape": False,
            "adapt_to_pi_aloha": False,
            "use_delta_joint_actions_aloha": False,
            "missing_camera_handling": (
                "The sole source cam_high stream is real Dataset-B input. Two additional "
                "configured visual keys are absent from the dataset and are created by "
                "SmolVLA as fully masked empty-camera tensors; they are not observations."
            ),
        },
        "normalization": {
            "mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
            "source": str((dataset / "meta/stats.json").resolve()),
            "dataset_specific_override_in_training": True,
            "aloha_statistics_reused": False,
        },
    }
    write_json(Path(args.output), result)
    if result["status"] != "PASS":
        raise RuntimeError(f"SmolVLA preflight failed: {checks}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return cloned


def _scalar_loss(output: Any) -> float:
    value = output[0] if isinstance(output, tuple) else output
    if isinstance(value, dict):
        value = value["loss"]
    return float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _normalization_matches_dataset(checkpoint: Path, dataset: Path) -> dict[str, Any]:
    stats = read_json(dataset / "meta/stats.json")
    tensor_path = checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    comparisons: dict[str, Any] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for feature in ("observation.state", "action"):
            for field in ("min", "max", "mean", "std"):
                key = f"{feature}.{field}"
                expected = np.asarray(stats[feature][field], dtype=np.float32)
                actual = handle.get_tensor(key).numpy()
                comparisons[key] = {
                    "shape": list(actual.shape),
                    "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
                    "matches": bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-6)),
                }
        comparisons["checkpoint_tensor_keys"] = sorted(keys)
    comparisons["all_dataset_statistics_match"] = all(
        value["matches"]
        for key, value in comparisons.items()
        if key != "checkpoint_tensor_keys" and isinstance(value, dict)
    )
    return comparisons


def checkpoint_audit(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.collate import lerobot_collate_fn

    checkpoint = Path(args.checkpoint).resolve()
    dataset_root = Path(args.dataset).resolve()
    deltas = {
        "observation.state": [0.0],
        "action": [index / FPS for index in range(ACTION_CHUNK)],
    }
    dataset = LeRobotDataset(
        repo_id="local/doll_handoff_proposed_b_50",
        root=dataset_root,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="torchcodec",
    )
    raw = lerobot_collate_fn([dataset[len(dataset) - 1]])
    if raw is None:
        raise RuntimeError("smoke checkpoint audit batch is None")
    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    batch = preprocessor(raw)
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(batch)
        prediction = postprocessor(normalized)
    padding_mask = batch["action_is_pad"].bool()
    altered = _clone_batch(batch)
    expanded = padding_mask.unsqueeze(-1).expand_as(altered["action"])
    altered["action"] = torch.where(
        expanded, altered["action"] + 10_000.0, altered["action"]
    )
    torch.manual_seed(1077)
    torch.cuda.manual_seed_all(1077)
    with torch.inference_mode():
        original_loss = _scalar_loss(policy.forward(batch))
    torch.manual_seed(1077)
    torch.cuda.manual_seed_all(1077)
    with torch.inference_mode():
        altered_loss = _scalar_loss(policy.forward(altered))
    loss_delta = abs(original_loss - altered_loss)
    normalization = _normalization_matches_dataset(checkpoint, dataset_root)
    config = read_json(checkpoint / "config.json")
    checks = {
        "checkpoint_reload": True,
        "dataset_frame_read_and_video_decode": list(raw["observation.images.cam_high"].shape)
        == [1, 3, 480, 640],
        "logical_state_28": config["input_features"]["observation.state"]["shape"]
        == [STATE_DIM],
        "logical_action_28": config["output_features"]["action"]["shape"]
        == [ACTION_DIM],
        "padded_model_state_32": int(config["max_state_dim"]) == 32,
        "padded_model_action_32": int(config["max_action_dim"]) == 32,
        "prediction_shape_1x50x28": list(prediction.shape)
        == [1, ACTION_CHUNK, ACTION_DIM],
        "prediction_finite": bool(torch.isfinite(prediction).all()),
        "forward_loss_finite": math.isfinite(original_loss),
        "padding_present": int(padding_mask.sum().detach().cpu()) > 0,
        "padding_loss_mask": loss_delta <= 1e-4,
        "dataset_specific_normalization_loaded": normalization[
            "all_dataset_statistics_match"
        ],
    }
    result = {
        "schema_version": "doll_handoff_dataset_b_smolvla_checkpoint_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checkpoint": str(checkpoint),
        "checks": checks,
        "dataset_length": len(dataset),
        "raw_batch_shapes": {
            key: list(value.shape)
            for key, value in raw.items()
            if isinstance(value, torch.Tensor)
        },
        "processed_state_shape": list(batch["observation.state"].shape),
        "processed_action_shape": list(batch["action"].shape),
        "prediction_shape": list(prediction.shape),
        "prediction_minimum": float(prediction.min().detach().cpu()),
        "prediction_maximum": float(prediction.max().detach().cpu()),
        "padding_true_count": int(padding_mask.sum().detach().cpu()),
        "padding_loss_original": original_loss,
        "padding_loss_after_padded_target_perturbation": altered_loss,
        "padding_loss_absolute_delta": loss_delta,
        "normalization_comparison": normalization,
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
    }
    write_json(Path(args.output), result)
    if result["status"] != "PASS":
        raise RuntimeError(f"checkpoint audit failed: {checks}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--dataset", required=True)
    preflight_parser.add_argument("--base-model", required=True)
    preflight_parser.add_argument("--config", action="append", required=True)
    preflight_parser.add_argument("--output", required=True)
    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--dataset", required=True)
    checkpoint_parser.add_argument("--checkpoint", required=True)
    checkpoint_parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        preflight(args)
    elif args.mode == "checkpoint":
        checkpoint_audit(args)
    else:  # pragma: no cover
        raise ValueError(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
