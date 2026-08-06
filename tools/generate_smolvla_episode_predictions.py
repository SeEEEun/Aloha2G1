#!/usr/bin/env python3
"""Generate teacher-forced offline SmolVLA predictions without robot imports."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path("/home/jbnu/aloha_g1_dataset")
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

# Reuse the already dry-run/full-run validated LeRobot 0.6.1 loading,
# reset, boundary, seed, and finite-shape validation implementation.
from evaluate_smolvla_checkpoints import (  # noqa: E402
    ACTION_DIM, CHUNK_SIZE, TASK, episode_bounds, finite_action, seed_all,
    LeRobotDataset, SmolVLAPolicy, lerobot_version, make_pre_post_processors,
)

DEFAULT_CHECKPOINT = ROOT / (
    "outputs/smolvla_magsafe_batch16_20k_20260729_140407/"
    "checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET = ROOT / "lerobot_magsafe_50_cam_high_v3"
DEFAULT_OUTPUT = ROOT / "evaluation/smolvla_20k_full_predictions"
DEFAULT_REPO = "local/magsafe_aloha_50_cam_high_v3"
DEFAULT_EPISODES = [0, 24, 49]
FPS = 30.0
LOG = logging.getLogger("smolvla_prediction")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-forced offline SmolVLA episode prediction")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--episodes", type=int, nargs="+", default=DEFAULT_EPISODES)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    incomplete = path.with_name(path.name + ".incomplete")
    with incomplete.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(incomplete, path)


def atomic_json(path: Path, payload: Any) -> None:
    incomplete = path.with_name(path.name + ".incomplete")
    incomplete.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(incomplete, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    incomplete = path.with_name(path.name + ".incomplete")
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with incomplete.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(incomplete, path)


def transition_timing(expert: np.ndarray, pred: np.ndarray, joint: int) -> dict[str, Any]:
    low, high = float(expert[:, joint].min()), float(expert[:, joint].max())
    threshold = (low + high) / 2.0
    if high - low <= 1e-8:
        return {"threshold": threshold, "expert_frames": [], "predicted_frames": [], "frame_differences": []}
    e = expert[:, joint] >= threshold
    p = pred[:, joint] >= threshold
    ef = np.flatnonzero(e[1:] != e[:-1]) + 1
    pf = np.flatnonzero(p[1:] != p[:-1]) + 1
    differences = [int(pf[np.argmin(np.abs(pf - f))] - f) for f in ef] if len(pf) else []
    return {
        "threshold_from_expert_midrange": threshold,
        "expert_frames": ef.astype(int).tolist(),
        "predicted_frames": pf.astype(int).tolist(),
        "nearest_frame_differences": differences,
    }


def analyze(payload: dict[str, Any], latency: np.ndarray, peak_memory: int) -> dict[str, Any]:
    expert = payload["expert_action"]
    pred = payload["teacher_forced_predicted_action"]
    chunks = payload["predicted_chunks"]
    valid = payload["chunk_valid_length"]
    errors = pred - expert
    chunk_abs_sum = chunk_sq_sum = 0.0
    chunk_count = 0
    for frame, length in enumerate(valid):
        length = int(length)
        target = expert[frame:frame + length]
        delta = chunks[frame, :length] - target
        chunk_abs_sum += float(np.abs(delta).sum())
        chunk_sq_sum += float(np.square(delta).sum())
        chunk_count += int(delta.size)
    dp = np.diff(pred, axis=0)
    de = np.diff(expert, axis=0)
    velocity = dp * FPS
    acceleration = np.diff(velocity, axis=0) * FPS
    per_joint = {}
    for j in range(ACTION_DIM):
        per_joint[str(j)] = {
            "mae": float(np.mean(np.abs(errors[:, j]))),
            "rmse": float(np.sqrt(np.mean(np.square(errors[:, j])))),
            "expert_min": float(expert[:, j].min()),
            "expert_max": float(expert[:, j].max()),
            "predicted_min": float(pred[:, j].min()),
            "predicted_max": float(pred[:, j].max()),
        }
    return {
        "episode_index": int(payload["episode_index"]),
        "frame_count": len(pred),
        "first_action_mae": float(np.mean(np.abs(errors))),
        "first_action_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "chunk_mae": chunk_abs_sum / chunk_count,
        "chunk_rmse": float(np.sqrt(chunk_sq_sum / chunk_count)),
        "action_delta_mae": float(np.mean(np.abs(dp - de))),
        "per_joint": per_joint,
        "first_prediction_minus_initial_state": (pred[0] - payload["observation_state"][0]).tolist(),
        "first_prediction_initial_state_l2": float(np.linalg.norm(pred[0] - payload["observation_state"][0])),
        "max_frame_action_jump": float(np.max(np.abs(dp))),
        "max_frame_action_jump_joint": int(np.unravel_index(np.argmax(np.abs(dp)), dp.shape)[1]),
        "max_predicted_velocity": float(np.max(np.abs(velocity))),
        "max_predicted_acceleration": float(np.max(np.abs(acceleration))) if len(acceleration) else 0.0,
        "gripper_transition_timing": {
            "left_joint_6": transition_timing(expert, pred, 6),
            "right_joint_6": transition_timing(expert, pred, 13),
        },
        "nan_inf_count": int(np.size(pred) - np.isfinite(pred).sum()),
        "inference_latency_mean_ms": float(np.mean(latency)),
        "inference_latency_p95_ms": float(np.percentile(latency, 95)),
        "peak_cuda_memory_bytes": int(peak_memory),
        "joint_limits": {
            "status": "PENDING_ISAACLAB_ASSET_RUNTIME_QUERY",
            "note": "Expert min/max above are observed episode ranges, not joint limits.",
        },
    }


def plots(output: Path, episode: int, expert: np.ndarray, pred: np.ndarray) -> None:
    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(pred))
    fig, axes = plt.subplots(7, 2, figsize=(16, 19), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(x, expert[:, j], label="expert", lw=1)
        ax.plot(x, pred[:, j], label="teacher-forced predicted", lw=.8)
        ax.set_title(f"joint {j}")
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output / f"episode_{episode:06d}_expert_vs_predicted.png", dpi=130)
    plt.close(fig)

    rmse = np.sqrt(np.mean(np.square(pred - expert), axis=1))
    jump = np.max(np.abs(np.diff(pred, axis=0)), axis=1)
    velocity = np.diff(pred, axis=0) * FPS
    acceleration = np.diff(velocity, axis=0) * FPS
    series = [
        ("frame_action_rmse", rmse),
        ("frame_action_jump", jump),
        ("predicted_velocity", velocity),
        ("predicted_acceleration", acceleration),
    ]
    for name, y in series:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y)
        ax.set_title(f"episode {episode}: {name}")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(output / f"episode_{episode:06d}_{name}.png", dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    for j, label in ((6, "left expert"), (13, "right expert")):
        ax.plot(x, expert[:, j], label=label)
    for j, label in ((6, "left predicted"), (13, "right predicted")):
        ax.plot(x, pred[:, j], "--", label=label)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / f"episode_{episode:06d}_grippers.png", dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run == args.execute:
        raise ValueError("Specify exactly one of --dry-run or --execute")
    if lerobot_version != "0.6.1":
        raise RuntimeError(f"Expected LeRobot 0.6.1, got {lerobot_version}")
    if args.device != "cuda":
        raise ValueError("This checkpoint and its saved processors require --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (args.checkpoint / name).is_file():
            raise FileNotFoundError(args.checkpoint / name)
    episodes = sorted(set(args.episodes))
    if args.dry_run and episodes != [0]:
        raise ValueError("Prediction dry-run must use only --episodes 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, download_videos=False)
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    policy.eval()
    reports: list[dict[str, Any]] = []
    for episode in episodes:
        start, episode_end = episode_bounds(dataset, episode)
        end = episode_end
        if args.dry_run:
            end = min(end, start + 10)
        count = end - start
        LOG.info("episode=%d frames=%d mode=%s", episode, count, "dry-run" if args.dry_run else "execute")
        state = np.empty((count, ACTION_DIM), np.float32)
        expert = np.empty_like(state)
        chunks = np.empty((count, CHUNK_SIZE, ACTION_DIM), np.float32)
        timestamps = np.empty(count, np.float32)
        frame_indices = np.empty(count, np.int64)
        valid = np.empty(count, np.int64)
        latency = np.empty(count, np.float64)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for local, global_idx in enumerate(range(start, end)):
            seed_all(args.seed + local)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            item = dataset[global_idx]
            state[local] = finite_action(item["observation.state"], "observation_state", (ACTION_DIM,))
            expert[local] = finite_action(item["action"], "expert_action", (ACTION_DIM,))
            timestamps[local] = float(item["timestamp"])
            frame_indices[local] = int(item["frame_index"])
            raw = {
                "observation.images.cam_high": item["observation.images.cam_high"],
                "observation.state": item["observation.state"],
                "task": TASK,
            }
            batch = preprocessor(raw)
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.inference_mode():
                prediction = postprocessor(policy.predict_action_chunk(batch))
            torch.cuda.synchronize()
            latency[local] = (time.perf_counter() - before) * 1000.0
            chunks[local] = finite_action(
                prediction, "predicted_chunks", (1, CHUNK_SIZE, ACTION_DIM)
            )[0]
            metric_end = end if args.dry_run else episode_end
            valid[local] = min(CHUNK_SIZE, metric_end - global_idx)
            if (local + 1) % 100 == 0 or local + 1 == count:
                LOG.info("episode=%d progress=%d/%d", episode, local + 1, count)
        payload = {
            "episode_index": np.asarray(episode, dtype=np.int64),
            "frame_index": frame_indices,
            "timestamp": timestamps,
            "observation_state": state,
            "expert_action": expert,
            "teacher_forced_predicted_action": chunks[:, 0].copy(),
            "predicted_chunks": chunks,
            "chunk_valid_length": valid,
            "task": np.asarray(TASK),
            "fps": np.asarray(FPS, dtype=np.float32),
            "checkpoint": np.asarray(str(args.checkpoint)),
        }
        for key in ("observation_state", "expert_action", "teacher_forced_predicted_action", "predicted_chunks"):
            if not np.isfinite(payload[key]).all():
                raise RuntimeError(f"episode {episode}: {key} contains NaN/inf")
        report = analyze(payload, latency, torch.cuda.max_memory_allocated())
        report["mode"] = "teacher-forced offline prediction"
        reports.append(report)
        suffix = "_dry_run" if args.dry_run else ""
        atomic_npz(args.output_dir / f"episode_{episode:06d}_prediction{suffix}.npz", payload)
        atomic_json(args.output_dir / f"episode_{episode:06d}_analysis{suffix}.json", report)
        if args.execute:
            plots(args.output_dir / "plots", episode, expert, chunks[:, 0])
    if args.execute:
        # Preserve reports from earlier ordered invocations by re-reading their analysis files.
        all_reports = []
        for path in sorted(args.output_dir.glob("episode_*_analysis.json")):
            all_reports.append(json.loads(path.read_text(encoding="utf-8")))
        summary = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in all_reports]
        atomic_csv(args.output_dir / "prediction_summary.csv", summary)
        atomic_json(
            args.output_dir / "prediction_report.json",
            {
                "evaluation_type": "teacher-forced offline prediction",
                "checkpoint_selection_basis": "training-set sanity evaluation; not validation/generalization",
                "checkpoint": str(args.checkpoint),
                "episodes": [r["episode_index"] for r in all_reports],
                "results": all_reports,
                "errors": [],
                "warnings": [
                    "Joint limits are queried from the actual Isaac Lab articulation before physics replay."
                ],
            },
        )


if __name__ == "__main__":
    main()
