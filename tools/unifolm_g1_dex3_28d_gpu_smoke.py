#!/usr/bin/env python3
"""Deferred, explicitly authorized UniFoLM G1+Dex3 28D GPU smoke.

This file is preparation only.  It refuses to import Torch or touch CUDA unless
UNIFOLM_GPU_AUTHORIZED=YES is present.  It loads the pinned 23D checkpoint while
replacing exactly four incompatible interface tensors, runs one forward and
backward, verifies finite gradients, saves/reloads an adapter checkpoint, emits
one [1,25,28] action chunk, and optionally runs a bounded 1--3 episode overfit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SRC = PROJECT_ROOT / "external/third_party/unifolm-vla/src"
AUTH_ENV = "UNIFOLM_GPU_AUTHORIZED"


def require_human_gpu_authorization() -> None:
    if os.environ.get(AUTH_ENV) != "YES":
        raise SystemExit(
            "REFUSED: GPU smoke requires a new human authorization and "
            f"{AUTH_ENV}=YES. No Torch/CUDA import occurred."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--vla-config", type=Path, required=True)
    parser.add_argument("--vlm-pretrained", required=True)
    parser.add_argument(
        "--hdf5-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/policy_class_diagnostic/unifolm_b/full50_conversion/hdf5",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/policy_class_diagnostic/unifolm_b/embodiment_28d/dataset_statistics_28d.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--tiny-episodes", default="0,1,2")
    parser.add_argument("--tiny-samples-per-episode", type=int, default=4)
    parser.add_argument("--tiny-steps", type=int, default=30)
    parser.add_argument("--skip-tiny-overfit", action="store_true")
    return parser.parse_args()


def make_action_chunk(action: Any, frame_index: int, horizon: int) -> Any:
    import numpy as np

    indices = np.minimum(np.arange(frame_index, frame_index + horizon), len(action) - 1)
    return action[indices]


def load_raw_sample(hdf5_path: Path, frame_index: int) -> dict[str, Any]:
    import h5py

    with h5py.File(hdf5_path, "r") as source:
        frame_count = int(source.attrs["frame_count"])
        if not 0 <= frame_index < frame_count:
            raise ValueError(f"frame {frame_index} outside {hdf5_path} ({frame_count} frames)")
        return {
            "episode_index": int(source.attrs["episode_id"]),
            "frame_index": frame_index,
            "frame_count": frame_count,
            "image": source["observations/images/cam_high"][frame_index],
            "state": source["observations/qpos"][frame_index],
            "actions": make_action_chunk(source["action"][:], frame_index, 25),
            "task": source["language_instruction"][()].decode("utf-8"),
        }


def prepare_batch(model: Any, raw: dict[str, Any], stats: dict[str, Any], device: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    from tools.unifolm_g1_dex3_28d_adapter import (
        ACTION_HORIZON,
        TASK_INSTRUCTION,
        normalize_bounds,
        validate_model_io,
    )

    if raw["task"] != TASK_INSTRUCTION:
        raise ValueError(f"task changed: {raw['task']!r}")
    state = normalize_bounds(raw["state"], stats["proprio"]).astype(np.float32)
    actions = normalize_bounds(raw["actions"], stats["action"]).astype(np.float32)
    validate_model_io(state[None, None, :], actions[None, :, :], ACTION_HORIZON)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.fromarray(raw["image"]).convert("RGB")},
            {"type": "text", "text": TASK_INSTRUCTION},
        ],
    }]
    processor = model.processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    for key, value in tuple(batch.items()):
        if hasattr(value, "to"):
            batch[key] = value.to(device)
    batch["state"] = torch.from_numpy(state).unsqueeze(0).to(device)
    batch["action"] = torch.from_numpy(actions).unsqueeze(0).to(device)
    return dict(batch)


def sha256_tensors(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def save_reload_adapter(model: Any, path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    import torch

    parameters = dict(model.named_parameters())
    saved = {key: parameters[key].detach().cpu().clone() for key in keys}
    before_hash = sha256_tensors(saved)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "G1_DEX3_28D_INTERFACE_V1", "state_dict": saved}, path)
    with torch.no_grad():
        for key in keys:
            parameters[key].zero_()
    reloaded = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    with torch.no_grad():
        for key in keys:
            parameters[key].copy_(reloaded[key].to(parameters[key].device, dtype=parameters[key].dtype))
    after = {key: parameters[key].detach().cpu().clone() for key in keys}
    after_hash = sha256_tensors(after)
    if before_hash != after_hash:
        raise RuntimeError("adapter checkpoint reload was not exact")
    return {"path": str(path.resolve()), "sha256": before_hash, "reload_exact": True}


def one_optimization_step(model: Any, batch: dict[str, Any], optimizer: Any, seed: int) -> dict[str, Any]:
    import torch

    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    result = model(qwen_inputs=batch)
    loss = result["action_loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss}")
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        finite = bool(torch.isfinite(parameter.grad).all().item())
        gradients[name] = {"finite": finite, "norm": float(parameter.grad.float().norm().item())}
        if not finite:
            raise RuntimeError(f"non-finite gradient for {name}")
    optimizer.step()
    return {"loss": float(loss.detach().float().item()), "gradients": gradients}


def main() -> None:
    require_human_gpu_authorization()
    args = parse_args()

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise SystemExit("Authorized smoke requires an explicitly selected, available CUDA device")
    if not args.vla_checkpoint.is_file() or not args.vla_config.is_file():
        raise FileNotFoundError("VLA checkpoint/config must exist locally")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from tools.unifolm_g1_dex3_28d_adapter import (
        ACTION_HORIZON,
        DIMENSION_TIED_CHECKPOINT_KEYS,
        TASK_INSTRUCTION,
        denormalize_bounds,
        install_upstream_dimension_overrides,
        load_reusable_checkpoint_weights,
    )

    install_upstream_dimension_overrides(UPSTREAM_SRC)
    from unifolm_vla.model.framework import build_framework

    cfg = OmegaConf.load(args.vla_config)
    cfg.framework.framework_py = "unifolm_vla"
    cfg.framework.qwenvl.base_vlm = args.vlm_pretrained
    cfg.framework.action_model.action_dim = 28
    cfg.framework.action_model.state_dim = 28
    cfg.framework.action_model.action_horizon = ACTION_HORIZON
    cfg.framework.action_model.future_action_window_size = ACTION_HORIZON - 1
    cfg.trainer.repeated_diffusion_steps = 1

    device = torch.device(args.device)
    model = build_framework(cfg)
    load_report = load_reusable_checkpoint_weights(model, args.vla_checkpoint)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    trainable_keys = tuple(DIMENSION_TIED_CHECKPOINT_KEYS)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_parameters = dict(model.named_parameters())
    for key in trainable_keys:
        named_parameters[key].requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [named_parameters[key] for key in trainable_keys], lr=args.learning_rate
    )

    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))["g1_dex3_28d"]
    first = load_raw_sample(args.hdf5_dir / "episode_000000.hdf5", 0)
    smoke_batch = prepare_batch(model, first, statistics, device)
    smoke = one_optimization_step(model, smoke_batch, optimizer, seed=280328)

    adapter_checkpoint = save_reload_adapter(
        model, args.output_dir / "smoke_28d_interface.pt", trainable_keys
    )
    with torch.inference_mode():
        predicted = model.predict_action(qwen_inputs=smoke_batch)["normalized_actions"]
    if predicted.shape != (1, ACTION_HORIZON, 28) or not np.isfinite(predicted).all():
        raise RuntimeError(f"invalid inference output {predicted.shape}")
    physical = denormalize_bounds(predicted, statistics["action"])
    np.save(args.output_dir / "smoke_action_chunk_28d.npy", physical)

    tiny_report: dict[str, Any] = {"executed": False}
    if not args.skip_tiny_overfit:
        episode_ids = tuple(int(item) for item in args.tiny_episodes.split(",") if item)
        if not 1 <= len(episode_ids) <= 3:
            raise ValueError("tiny overfit must use 1--3 episodes")
        if not 1 <= args.tiny_samples_per_episode <= 8 or not 1 <= args.tiny_steps <= 100:
            raise ValueError("tiny smoke bounds exceeded")
        tiny_batches = []
        sample_manifest = []
        for episode_id in episode_ids:
            path = args.hdf5_dir / f"episode_{episode_id:06d}.hdf5"
            probe = load_raw_sample(path, 0)
            frame_ids = np.linspace(
                0, probe["frame_count"] - 1, args.tiny_samples_per_episode, dtype=int
            ).tolist()
            for frame_id in frame_ids:
                raw = load_raw_sample(path, frame_id)
                tiny_batches.append(prepare_batch(model, raw, statistics, device))
                sample_manifest.append({"episode_index": episode_id, "frame_index": frame_id})
        losses = []
        for step in range(args.tiny_steps):
            sample_index = step % len(tiny_batches)
            outcome = one_optimization_step(
                model, tiny_batches[sample_index], optimizer, seed=280328 + sample_index
            )
            losses.append(outcome["loss"])
        window = min(5, len(losses))
        first_mean = float(np.mean(losses[:window]))
        last_mean = float(np.mean(losses[-window:]))
        if not last_mean < first_mean:
            raise RuntimeError(
                f"tiny overfit did not reduce deterministic mean loss: {first_mean} -> {last_mean}"
            )
        tiny_report = {
            "executed": True,
            "episodes": list(episode_ids),
            "samples": sample_manifest,
            "steps": args.tiny_steps,
            "first_window_mean_loss": first_mean,
            "last_window_mean_loss": last_mean,
            "loss_decreased": True,
        }
        save_reload_adapter(model, args.output_dir / "tiny_overfit_28d_interface.pt", trainable_keys)

    report = {
        "status": "PASS",
        "authorization": f"{AUTH_ENV}=YES",
        "device": str(device),
        "task_instruction": TASK_INSTRUCTION,
        "pretrained_load": load_report,
        "trainable_parameters": sum(named_parameters[key].numel() for key in trainable_keys),
        "trainable_keys": list(trainable_keys),
        "one_batch_forward_backward": smoke,
        "checkpoint": adapter_checkpoint,
        "inference": {
            "normalized_shape": list(predicted.shape),
            "physical_shape": list(physical.shape),
            "finite": True,
            "file": str((args.output_dir / "smoke_action_chunk_28d.npy").resolve()),
        },
        "tiny_overfit": tiny_report,
        "full_50_episode_training_started": False,
    }
    (args.output_dir / "gpu_smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
