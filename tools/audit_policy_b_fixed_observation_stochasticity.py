#!/usr/bin/env python3
"""Repeat frozen Policy-B inference on byte-identical Isaac observations.

The audit records every 50x28 policy chunk and every sampled flow-noise tensor.
It never executes actions and has no real-robot transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


EXPECTED_RGB = (480, 640, 3)
EXPECTED_STATE = (28,)
TASK = "Pick up the doll with the left hand, handoff it to the right hand, and place it in the trash bin."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--indices", default="0,11,90")
    parser.add_argument("--labels", default="initial,left_approach,doll_region_plateau")
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=2026082500)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def summarize_chunks(chunks: np.ndarray, names: list[str]) -> dict[str, Any]:
    # chunks [repeat, offset, joint]
    variance = np.var(chunks.astype(np.float64), axis=0)
    std = np.sqrt(variance)
    mean = np.mean(chunks.astype(np.float64), axis=0)
    dispersion = np.sqrt(np.mean(np.square(chunks.astype(np.float64) - mean[None]), axis=2))
    prefix = {}
    for horizon in (4, 8, 12, 16):
        values = chunks[:, :horizon].astype(np.float64)
        center = np.mean(values, axis=0, keepdims=True)
        per_repeat_rmse = np.sqrt(np.mean(np.square(values - center), axis=(1, 2)))
        per_joint_std = np.sqrt(np.mean(np.var(values, axis=0), axis=0))
        prefix[str(horizon)] = {
            "repeat_to_mean_rmse_rad": distribution(per_repeat_rmse),
            "per_joint_rms_std_rad": {
                name: float(per_joint_std[index]) for index, name in enumerate(names)
            },
            "maximum_per_joint_rms_std_rad": float(np.max(per_joint_std)),
        }
    first = chunks[:, 0].astype(np.float64)
    first_std = np.std(first, axis=0)
    first_range = np.ptp(first, axis=0)
    overall_std = np.sqrt(np.mean(variance, axis=0))
    order = np.argsort(overall_std)[::-1]
    return {
        "variance_by_chunk_offset_joint_rad2": variance,
        "standard_deviation_by_chunk_offset_joint_rad": std,
        "offset_dispersion_rmse_rad": {
            "per_offset_mean": np.mean(dispersion, axis=0),
            "all_repeat_offset": distribution(dispersion),
        },
        "first_action": {
            "per_joint_std_rad": {name: float(first_std[i]) for i, name in enumerate(names)},
            "per_joint_range_rad": {name: float(first_range[i]) for i, name in enumerate(names)},
            "maximum_std_rad": float(np.max(first_std)),
            "maximum_range_rad": float(np.max(first_range)),
            "rms_std_across_joints_rad": float(np.sqrt(np.mean(np.square(first_std)))),
        },
        "prefix": prefix,
        "dominant_unstable_joints": [
            {
                "joint": names[int(index)],
                "rms_std_across_50_offsets_rad": float(overall_std[index]),
                "first_action_std_rad": float(first_std[index]),
                "first_action_range_rad": float(first_range[index]),
            }
            for index in order[:10]
        ],
    }


def main() -> None:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    args = parse_args()
    if args.repeats != 64:
        raise ValueError("this audit requires exactly 64 stochastic repeats")
    indices = [int(value) for value in args.indices.split(",")]
    labels = args.labels.split(",")
    if len(indices) != 3 or len(labels) != 3:
        raise ValueError("exactly three indices and labels are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    random.seed(args.base_seed)
    np.random.seed(args.base_seed)
    torch.manual_seed(args.base_seed)
    torch.cuda.manual_seed_all(args.base_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Policy-B audit")
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    names: list[str] | None = None
    all_stochastic_chunks = []
    all_stochastic_noise = []
    all_fixed_chunks = []
    observation_reports = []

    for observation_position, (index, label) in enumerate(zip(indices, labels)):
        metadata_path = args.observation_dir / f"inference_{index:04d}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rgb_path = Path(metadata["rgb_path"])
        state_path = Path(metadata["state_path"])
        if sha256_file(rgb_path) != metadata["rgb_sha256"]:
            raise RuntimeError(f"RGB hash mismatch: {rgb_path}")
        if sha256_file(state_path) != metadata["state_sha256"]:
            raise RuntimeError(f"state hash mismatch: {state_path}")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"unreadable RGB: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with np.load(state_path, allow_pickle=False) as archive:
            state = archive["measured_state"].astype(np.float32)
            current_names = archive["joint_names"].astype(str).tolist()
        if rgb.shape != EXPECTED_RGB or state.shape != EXPECTED_STATE:
            raise RuntimeError("frozen observation shape mismatch")
        if metadata["task"] != TASK:
            raise RuntimeError("task string differs from authoritative training/deployment task")
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("joint ordering differs between fixed observations")
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        processed = preprocessor(
            {
                "observation.images.cam_high": image_tensor,
                "observation.state": torch.from_numpy(state.copy()),
                "task": TASK,
            }
        )
        device = processed["observation.state"].device
        stochastic_chunks = []
        stochastic_noise = []
        noise_records = []
        for repeat in range(args.repeats):
            seed = args.base_seed + observation_position * 10_000 + repeat
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cpu_rng_before = torch.get_rng_state().cpu().numpy().tobytes()
            cuda_rng_before = torch.cuda.get_rng_state().cpu().numpy().tobytes()
            noise = policy.model.sample_noise(
                (1, 50, int(policy.config.max_action_dim)), device
            )
            with torch.no_grad():
                normalized = policy.predict_action_chunk(
                    processed,
                    noise=noise,
                    inference_delay=0,
                    prev_chunk_left_over=None,
                    execution_horizon=4,
                )
                action = postprocessor(normalized.clone())
            chunk = action[0].detach().cpu().numpy().astype(np.float32)
            if chunk.shape != (50, 28) or not np.isfinite(chunk).all():
                raise RuntimeError("malformed Policy-B chunk")
            noise_np = noise[0].detach().cpu().numpy().astype(np.float32)
            stochastic_chunks.append(chunk)
            stochastic_noise.append(noise_np)
            noise_records.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "cpu_rng_state_before_noise_sha256": sha256_bytes(cpu_rng_before),
                    "cuda_rng_state_before_noise_sha256": sha256_bytes(cuda_rng_before),
                    "flow_noise_sha256": sha256_bytes(noise_np.tobytes()),
                    "raw_policy_chunk_sha256": sha256_bytes(chunk.tobytes()),
                }
            )

        stochastic_chunks_array = np.stack(stochastic_chunks)
        stochastic_noise_array = np.stack(stochastic_noise)
        fixed_noise = torch.from_numpy(stochastic_noise_array[0:1]).to(device=device)
        fixed_chunks = []
        for repeat in range(args.repeats):
            with torch.no_grad():
                normalized = policy.predict_action_chunk(
                    processed,
                    noise=fixed_noise,
                    inference_delay=0,
                    prev_chunk_left_over=None,
                    execution_horizon=4,
                )
                action = postprocessor(normalized.clone())
            fixed_chunks.append(action[0].detach().cpu().numpy().astype(np.float32))
        fixed_chunks_array = np.stack(fixed_chunks)
        stochastic_summary = summarize_chunks(stochastic_chunks_array, names)
        fixed_max_difference = float(
            np.max(np.abs(fixed_chunks_array - fixed_chunks_array[0:1]))
        )
        report = {
            "label": label,
            "inference_index": index,
            "simulation_timestamp_s": metadata["simulation_timestamp_s"],
            "rgb_path": str(rgb_path),
            "rgb_sha256": metadata["rgb_sha256"],
            "state_path": str(state_path),
            "state_sha256": metadata["state_sha256"],
            "task": TASK,
            "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
            "repeats": args.repeats,
            "unique_seed_noise_audit": stochastic_summary,
            "noise_records": noise_records,
            "fixed_noise_diagnostic": {
                "repeats": args.repeats,
                "flow_noise_sha256": noise_records[0]["flow_noise_sha256"],
                "maximum_output_difference_rad": fixed_max_difference,
                "bitwise_identical": bool(
                    np.all(fixed_chunks_array == fixed_chunks_array[0:1])
                ),
            },
        }
        observation_reports.append(report)
        all_stochastic_chunks.append(stochastic_chunks_array)
        all_stochastic_noise.append(stochastic_noise_array)
        all_fixed_chunks.append(fixed_chunks_array)
        print(
            f"{label}: first_std_max={stochastic_summary['first_action']['maximum_std_rad']:.8f} "
            f"first_range_max={stochastic_summary['first_action']['maximum_range_rad']:.8f} "
            f"fixed_maxdiff={fixed_max_difference:.3e}",
            flush=True,
        )

    assert names is not None
    stochastic_chunks = np.stack(all_stochastic_chunks)
    stochastic_noise = np.stack(all_stochastic_noise)
    fixed_chunks = np.stack(all_fixed_chunks)
    write_npz(
        output / "fixed_observation_inference_samples.npz",
        observation_labels=np.asarray(labels),
        observation_indices=np.asarray(indices, dtype=np.int64),
        seeds=np.asarray(
            [
                [args.base_seed + observation * 10_000 + repeat for repeat in range(args.repeats)]
                for observation in range(3)
            ],
            dtype=np.int64,
        ),
        flow_noise=stochastic_noise,
        raw_policy_chunks=stochastic_chunks,
        fixed_noise_raw_policy_chunks=fixed_chunks,
        joint_names=np.asarray(names),
    )

    aggregate_joint_std = np.sqrt(
        np.mean(np.var(stochastic_chunks.astype(np.float64), axis=1), axis=(0, 1))
    )
    aggregate_order = np.argsort(aggregate_joint_std)[::-1]
    aggregate = {
        "schema_version": 1,
        "audit": "FIXED_OBSERVATION_REPEATED_POLICY_B_INFERENCE",
        "policy_execution_or_robot_transport": False,
        "observation_distribution": "ISAAC_SOURCE_LIKE_CAM_HIGH_PLUS_MEASURED_G1_DEX3_STATE",
        "observations": observation_reports,
        "joint_names": names,
        "dominant_unstable_joints": [
            {
                "joint": names[int(index)],
                "aggregate_rms_std_rad": float(aggregate_joint_std[index]),
            }
            for index in aggregate_order[:10]
        ],
        "all_fixed_noise_repeats_bitwise_identical": bool(
            np.all(fixed_chunks == fixed_chunks[:, :1])
        ),
        "samples": str(output / "fixed_observation_inference_samples.npz"),
        "samples_sha256": None,
        "classification_deferred_until_low_motion_reference": True,
        "real_robot_command_allowed": False,
    }
    aggregate["samples_sha256"] = sha256_file(
        output / "fixed_observation_inference_samples.npz"
    )
    write_json(output / "fixed_observation_stochasticity.json", aggregate)

    variance = np.var(stochastic_chunks.astype(np.float64), axis=1)
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for observation, axis in enumerate(axes):
        image = axis.imshow(
            np.sqrt(variance[observation]).T,
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
        )
        axis.set_ylabel("joint index")
        axis.set_title(f"{labels[observation]}: raw chunk std by offset/joint (rad)")
        figure.colorbar(image, ax=axis, label="std (rad)")
    axes[-1].set_xlabel("chunk offset")
    figure.tight_layout()
    figure.savefig(output / "chunk_offset_joint_stochasticity.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 5))
    x = np.arange(len(names))
    for observation, label in enumerate(labels):
        first_std = np.std(stochastic_chunks[observation, :, 0].astype(np.float64), axis=0)
        axis.plot(x, first_std, marker="o", markersize=3, label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("first-action standard deviation (rad)")
    axis.set_title("64 repeated inferences with identical observation and distinct flow noise")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "first_action_stochasticity_per_joint.png", dpi=170)
    plt.close(figure)

    print(json.dumps({
        "report": str(output / "fixed_observation_stochasticity.json"),
        "samples": str(output / "fixed_observation_inference_samples.npz"),
        "fixed_noise_bitwise_identical": aggregate["all_fixed_noise_repeats_bitwise_identical"],
        "dominant_joints": aggregate["dominant_unstable_joints"][:5],
    }, indent=2))


if __name__ == "__main__":
    main()
