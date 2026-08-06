#!/usr/bin/env python3
"""Offline, read-only SmolVLA checkpoint evaluation on the training dataset.

This program never imports or constructs a robot, teleoperator, or environment.
It only reads a LeRobotDataset and local pretrained-policy directories.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

LEROBOT_CHECKOUT = Path("/home/jbnu/lerobot-smolvla")
if str(LEROBOT_CHECKOUT / "src") not in sys.path:
    sys.path.insert(0, str(LEROBOT_CHECKOUT / "src"))

from lerobot.__version__ import __version__ as lerobot_version  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

ROOT = Path("/home/jbnu/aloha_g1_dataset")
DATASET_ROOT = ROOT / "lerobot_magsafe_50_cam_high_v3"
REPO_ID = "local/magsafe_aloha_50_cam_high_v3"
OUTPUT = ROOT / "evaluation/smolvla_first50_checkpoints"
CHECKPOINTS = [
    ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/005000/pretrained_model",
    ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/010000/pretrained_model",
    ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/015000/pretrained_model",
    ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/020000/pretrained_model",
]
TASK = "Remove the MagSafe accessory from the phone and place the phone on the MagSafe charger."
SEED = 1000
STRIDE = 50
ACTION_DIM = 14
CHUNK_SIZE = 50
FK_MODULE = ROOT / "tools/export_stationary_fk.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--checkpoint-limit", type=int, default=4)
    p.add_argument("--episode-limit", type=int, default=50)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--skip-chunk-metrics", action="store_true")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_action(x: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.shape != shape:
        raise RuntimeError(f"{name} shape mismatch: expected {shape}, got {x.shape}")
    if not np.isfinite(x).all():
        bad = np.argwhere(~np.isfinite(x))[0].tolist()
        raise RuntimeError(f"{name} contains NaN/inf at index {bad}")
    return x


def scalar(v: Any) -> int:
    if torch.is_tensor(v):
        return int(v.reshape(-1)[0].item())
    if isinstance(v, (list, tuple, np.ndarray)):
        return int(np.asarray(v).reshape(-1)[0])
    return int(v)


def episode_bounds(dataset: LeRobotDataset, episode: int) -> tuple[int, int]:
    """Use the public metadata episode boundary fields in LeRobot 0.6.1."""
    row = dataset.meta.episodes[episode]
    return scalar(row["dataset_from_index"]), scalar(row["dataset_to_index"])


def load_fk() -> tuple[Any | None, dict[str, Any]]:
    status: dict[str, Any] = {"enabled": False, "source": str(FK_MODULE)}
    try:
        import importlib.util
        import mujoco

        spec = importlib.util.spec_from_file_location("project_stationary_aloha_fk", FK_MODULE)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load module spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model_path = Path(mod.DEFAULT_MODEL)
        model = mujoco.MjModel.from_xml_path(str(model_path))
        mod.validate_model(model)
        status.update(
            enabled=True,
            model=str(model_path),
            end_effector_bodies=list(mod.EXPECTED_BODIES),
            tcp_offset_local_m=np.asarray(mod.TCP_OFFSET_LOCAL).tolist(),
        )
        return (mod, model), status
    except Exception as exc:
        status["reason"] = f"{type(exc).__name__}: {exc}"
        return None, status


def quat_geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(a * b, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def metric_values(pred: np.ndarray, expert: np.ndarray, fk: Any | None) -> dict[str, float]:
    err = pred - expert
    out = {
        "first_action_mse_loss": float(np.mean(err**2)),
        "first_action_mae": float(np.mean(np.abs(err))),
        "first_action_rmse": float(np.sqrt(np.mean(err**2))),
        "left_arm_mae": float(np.mean(np.abs(err[:, 0:6]))),
        "left_gripper_mae": float(np.mean(np.abs(err[:, 6]))),
        "right_arm_mae": float(np.mean(np.abs(err[:, 7:13]))),
        "right_gripper_mae": float(np.mean(np.abs(err[:, 13]))),
    }
    if len(pred) > 1:
        out["action_delta_mae"] = float(np.mean(np.abs(np.diff(pred, axis=0) - np.diff(expert, axis=0))))
    else:
        out["action_delta_mae"] = math.nan
    if fk is not None:
        mod, model = fk
        p_fk = mod.compute_trajectory(model, pred.astype(np.float64))
        e_fk = mod.compute_trajectory(model, expert.astype(np.float64))
        lp = p_fk["left_position_xyz_m"] - e_fk["left_position_xyz_m"]
        rp = p_fk["right_position_xyz_m"] - e_fk["right_position_xyz_m"]
        rel = p_fk["right_minus_left_position_xyz_m"] - e_fk["right_minus_left_position_xyz_m"]
        out.update(
            left_hand_position_rmse_mm=float(np.sqrt(np.mean(lp**2)) * 1000.0),
            right_hand_position_rmse_mm=float(np.sqrt(np.mean(rp**2)) * 1000.0),
            left_hand_orientation_geodesic_deg=float(np.mean(quat_geodesic_deg(
                p_fk["left_quaternion_wxyz"], e_fk["left_quaternion_wxyz"]))),
            right_hand_orientation_geodesic_deg=float(np.mean(quat_geodesic_deg(
                p_fk["right_quaternion_wxyz"], e_fk["right_quaternion_wxyz"]))),
            bimanual_relative_position_rmse_mm=float(np.sqrt(np.mean(rel**2)) * 1000.0),
        )
        out["fk_hand_position_rmse_mm"] = float(
            np.sqrt((out["left_hand_position_rmse_mm"] ** 2 + out["right_hand_position_rmse_mm"] ** 2) / 2)
        )
    return out


def mean_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != "episode_index"}
    return {
        k: float(np.nanmean([float(r.get(k, math.nan)) for r in rows]))
        for k in sorted(keys)
        if not np.all(np.isnan([float(r.get(k, math.nan)) for r in rows]))
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [str(r["checkpoint_step"]) for r in rows]
    fields = ["first_action_mse_loss", "first_action_mae", "first_action_rmse"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field in zip(axes, fields, strict=True):
        ax.plot(labels, [r[field] for r in rows], marker="o")
        ax.set_title(field)
        ax.set_xlabel("checkpoint step")
        ax.grid(alpha=0.3)
    fig.suptitle("SmolVLA training-set sanity evaluation: loss / MAE / RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if lerobot_version != "0.6.1":
        raise RuntimeError(f"Expected LeRobot 0.6.1, found {lerobot_version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the saved checkpoint processor/config but is unavailable")
    checkpoints = CHECKPOINTS[: args.checkpoint_limit]
    for path in checkpoints:
        for name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                     "policy_postprocessor.json"):
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)
    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, download_videos=False)
    if dataset.meta.total_episodes != 50 or dataset.meta.total_frames != 50302:
        raise RuntimeError(
            f"Dataset mismatch: episodes={dataset.meta.total_episodes}, frames={dataset.meta.total_frames}"
        )
    episodes = list(range(min(args.episode_limit, dataset.meta.total_episodes)))
    samples: list[tuple[int, int, int]] = []
    for ep in episodes:
        start, end = episode_bounds(dataset, ep)
        for global_idx in range(start, end, STRIDE):
            samples.append((ep, global_idx - start, global_idx))
    if args.sample_limit is not None:
        samples = samples[: args.sample_limit]
    if not samples:
        raise RuntimeError("No samples selected")

    fk, fk_status = load_fk()
    report: dict[str, Any] = {
        "evaluation_type": "training-set sanity evaluation",
        "warning": "All 50 episodes were used for training; this is not validation or generalization performance.",
        "environment": {
            "lerobot_version": lerobot_version,
            "lerobot_checkout": str(LEROBOT_CHECKOUT),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "seed": SEED,
        },
        "dataset": {
            "repo_id": REPO_ID, "root": str(DATASET_ROOT), "episodes": len(episodes),
            "total_frames": dataset.meta.total_frames, "sample_stride_frames": STRIDE,
            "samples_per_checkpoint": len(samples), "task": TASK,
        },
        "checkpoints": [str(p) for p in checkpoints],
        "public_chunk_api": hasattr(SmolVLAPolicy, "predict_action_chunk"),
        "chunk_metrics_enabled": hasattr(SmolVLAPolicy, "predict_action_chunk") and not args.skip_chunk_metrics,
        "fk": fk_status,
        "errors": [],
        "warnings": [],
    }
    if fk is None:
        report["warnings"].append(f"FK skipped: {fk_status.get('reason')}")

    all_episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {}
    checkpoint_predictions: list[np.ndarray] = []
    checkpoint_chunks: list[np.ndarray] = []

    for cp_index, checkpoint in enumerate(checkpoints):
        step = checkpoint.parent.name
        print(f"[checkpoint {cp_index + 1}/{len(checkpoints)}] {step}", flush=True)
        seed_all(SEED)
        policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
        policy.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pred_list, expert_list, latency_ms, chunk_pred_list, chunk_exp_list = [], [], [], [], []
        sample_ep: list[int] = []
        for sample_i, (ep, frame, global_idx) in enumerate(samples):
            seed_all(SEED + sample_i)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            item = dataset[global_idx]
            expert = finite_action(item["action"], "expert_action", (ACTION_DIM,))
            raw = {
                "observation.images.cam_high": item["observation.images.cam_high"],
                "observation.state": item["observation.state"],
                "task": TASK,
            }
            batch = preprocessor(raw)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                if report["chunk_metrics_enabled"]:
                    normalized = policy.predict_action_chunk(batch)
                    action_out = postprocessor(normalized)
                    chunk = finite_action(action_out, "predicted_chunk", (1, CHUNK_SIZE, ACTION_DIM))[0]
                    predicted = chunk[0]
                else:
                    normalized = policy.select_action(batch)
                    predicted = finite_action(postprocessor(normalized), "predicted_action", (1, ACTION_DIM))[0]
                    chunk = None
            torch.cuda.synchronize()
            latency_ms.append((time.perf_counter() - started) * 1000.0)
            predicted = finite_action(predicted, "predicted_action", (ACTION_DIM,))
            pred_list.append(predicted)
            expert_list.append(expert)
            sample_ep.append(ep)
            if chunk is not None:
                start, end = episode_bounds(dataset, ep)
                available = min(CHUNK_SIZE, end - global_idx)
                expert_chunk = np.stack([
                    finite_action(dataset.reader.hf_dataset[global_idx + j]["action"],
                                  "expert_chunk_action", (ACTION_DIM,))
                    for j in range(available)
                ])
                chunk_pred_list.append(chunk[:available])
                chunk_exp_list.append(expert_chunk)
            if (sample_i + 1) % 50 == 0 or sample_i + 1 == len(samples):
                print(f"  samples {sample_i + 1}/{len(samples)}", flush=True)

        pred_arr = finite_action(np.stack(pred_list), "all predicted actions", (len(samples), ACTION_DIM))
        exp_arr = finite_action(np.stack(expert_list), "all expert actions", (len(samples), ACTION_DIM))
        checkpoint_predictions.append(pred_arr)
        ep_rows: list[dict[str, Any]] = []
        for ep in episodes:
            mask = np.asarray(sample_ep) == ep
            row: dict[str, Any] = {"checkpoint_step": step, "checkpoint": str(checkpoint),
                                   "episode_index": ep, "sample_count": int(mask.sum())}
            row.update(metric_values(pred_arr[mask], exp_arr[mask], fk))
            ep_latency = np.asarray(latency_ms)[mask]
            row["inference_latency_mean_ms"] = float(np.mean(ep_latency))
            row["inference_latency_p95_ms"] = float(np.percentile(ep_latency, 95))
            if chunk_pred_list:
                ids = np.flatnonzero(mask)
                cp = np.concatenate([chunk_pred_list[i] for i in ids])
                ce = np.concatenate([chunk_exp_list[i] for i in ids])
                row["chunk_mae"] = float(np.mean(np.abs(cp - ce)))
                row["chunk_rmse"] = float(np.sqrt(np.mean((cp - ce) ** 2)))
            ep_rows.append(row)
        all_episode_rows.extend(ep_rows)
        summary = {"checkpoint_step": step, "checkpoint": str(checkpoint), "sample_count": len(samples)}
        summary.update(metric_values(pred_arr, exp_arr, fk))
        # Do not form a delta between the final sample of episode N and the
        # first sample of episode N+1.
        adjacent_same_episode = np.asarray(sample_ep[1:]) == np.asarray(sample_ep[:-1])
        pred_delta = np.diff(pred_arr, axis=0)[adjacent_same_episode]
        expert_delta = np.diff(exp_arr, axis=0)[adjacent_same_episode]
        summary["action_delta_mae"] = float(np.mean(np.abs(pred_delta - expert_delta)))
        summary["inference_latency_mean_ms"] = float(np.mean(latency_ms))
        summary["inference_latency_p95_ms"] = float(np.percentile(latency_ms, 95))
        summary["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        summary["abnormal_action_count"] = int((~np.isfinite(pred_arr)).any(axis=1).sum())
        if chunk_pred_list:
            cp, ce = np.concatenate(chunk_pred_list), np.concatenate(chunk_exp_list)
            summary["predicted_chunk_shape"] = f"[{CHUNK_SIZE},{ACTION_DIM}]"
            summary["chunk_compared_action_count"] = len(cp)
            summary["chunk_mae"] = float(np.mean(np.abs(cp - ce)))
            summary["chunk_rmse"] = float(np.sqrt(np.mean((cp - ce) ** 2)))
            checkpoint_chunks.append(np.stack(chunk_pred_list) if len({x.shape for x in chunk_pred_list}) == 1
                                     else np.empty((0, CHUNK_SIZE, ACTION_DIM), dtype=np.float32))
        summary_rows.append(summary)
        del policy, preprocessor, postprocessor
        gc.collect()
        torch.cuda.empty_cache()

    ep_indices = np.asarray([x[0] for x in samples], dtype=np.int64)
    frame_indices = np.asarray([x[1] for x in samples], dtype=np.int64)
    global_indices = np.asarray([x[2] for x in samples], dtype=np.int64)
    expert_actions = np.stack([
        finite_action(dataset.reader.hf_dataset[int(i)]["action"], "saved expert_action", (ACTION_DIM,))
        for i in global_indices
    ])
    npz_payload.update(
        sample_index=np.arange(len(samples), dtype=np.int64),
        episode_index=ep_indices,
        frame_index=frame_indices,
        global_dataset_index=global_indices,
        expert_action=expert_actions.astype(np.float32),
        predicted_action=np.stack(checkpoint_predictions).astype(np.float32),
        checkpoint_step=np.asarray([r["checkpoint_step"] for r in summary_rows]),
    )
    np.savez_compressed(args.output_dir / "sampled_predictions.npz", **npz_payload)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "per_episode.csv", all_episode_rows)
    make_plot(args.output_dir / "loss_mae_rmse_comparison.png", summary_rows)

    best_rmse = min(summary_rows, key=lambda x: x["first_action_rmse"])
    best_fk = min(summary_rows, key=lambda x: x.get("fk_hand_position_rmse_mm", math.inf)) if fk else None
    normal = [r["checkpoint_step"] for r in summary_rows if r["abnormal_action_count"] == 0]
    report["results"] = summary_rows
    report["selection_criteria"] = {
        "lowest_first_action_rmse_checkpoint": best_rmse["checkpoint_step"],
        "lowest_fk_hand_position_rmse_checkpoint": best_fk["checkpoint_step"] if best_fk else None,
        "checkpoints_without_abnormal_actions": normal,
    }
    (args.output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["selection_criteria"], indent=2), flush=True)


if __name__ == "__main__":
    main()
