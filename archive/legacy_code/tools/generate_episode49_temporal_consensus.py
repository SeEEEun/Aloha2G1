#!/usr/bin/env python3
"""Build an episode-49 SmolVLA overlapping-chunk temporal-consensus trajectory.

Offline only: this module imports no robot, G1, or Isaac Lab interfaces.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path("/home/jbnu/aloha_g1_dataset")
LEROBOT = Path("/home/jbnu/lerobot-smolvla")
sys.path.insert(0, str(LEROBOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from lerobot.__version__ import __version__ as lerobot_version  # noqa: E402
from evaluate_smolvla_checkpoints import (  # noqa: E402
    ACTION_DIM, TASK, LeRobotDataset, SmolVLAPolicy, episode_bounds,
    make_pre_post_processors, seed_all,
)
from validate_smolvla_in_stationary_aloha_mujoco import (  # noqa: E402
    dataset_limits, dynamic_replay, fk, load_validated_model, mapped_limit_counts, mapped_qpos,
)

CHECKPOINT = ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/020000/pretrained_model"
DATASET_ROOT = ROOT / "lerobot_magsafe_50_cam_high_v3"
RTC_FILE = ROOT / "evaluation/smolvla_episode49_rtc/episode_000049_rtc_trajectory.npz"
H10_FILE = ROOT / "evaluation/smolvla_20k_chunk_stitched_preflight/episode_000049_chunk_stitched.npz"
XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
OUTPUT = ROOT / "evaluation/smolvla_episode49_temporal_consensus"
REPO_ID = "local/magsafe_aloha_50_cam_high_v3"
FPS = 30.0
CHUNK = 50
ARM = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER = (6, 13)
REGULARIZATION = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--optimization-steps", type=int, default=350)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def raw_batch(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.cam_high": item["observation.images.cam_high"],
        "observation.state": item["observation.state"],
        "task": TASK,
    }


def generate_chunks(policy: Any, pre: Any, post: Any, dataset: Any, start: int,
                    frames: int) -> tuple[np.ndarray, np.ndarray]:
    """One episode reset; four seeded public chunk predictions at every frame."""
    policy.reset()
    pre.reset()
    post.reset()
    chunks = np.empty((frames, 4, CHUNK, ACTION_DIM), np.float32)
    latency = np.empty((frames, 4), np.float32)
    for t in range(frames):
        batch = pre(raw_batch(dataset[start + t]))
        for sample_seed in range(4):
            seed_all(sample_seed)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.no_grad():
                normalized = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            latency[t, sample_seed] = time.perf_counter() - begin
            if normalized.shape != (1, CHUNK, ACTION_DIM) or not torch.isfinite(normalized).all():
                raise RuntimeError(f"Invalid chunk at frame={t}, seed={sample_seed}: {normalized.shape}")
            raw = post(normalized).squeeze(0).detach().float().cpu().numpy()
            if raw.shape != (CHUNK, ACTION_DIM) or not np.isfinite(raw).all():
                raise RuntimeError(f"Invalid postprocessed chunk at frame={t}, seed={sample_seed}")
            chunks[t, sample_seed] = raw
        if t % 25 == 0:
            print(f"chunk inference {t}/{frames}", flush=True)
    return chunks, latency


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    xx, ww = x[order], w[order]
    return float(xx[np.searchsorted(np.cumsum(ww), .5 * ww.sum(), side="left")])


def robust_huber_location(x: np.ndarray, w: np.ndarray) -> float:
    center = weighted_median(x, w)
    scale = max(1.4826 * weighted_median(np.abs(x - center), w), 1e-5)
    for _ in range(8):
        residual = np.abs(x - center)
        huber = np.minimum(1.0, 1.345 * scale / np.maximum(residual, 1e-9))
        ww = w * huber
        center = float(np.sum(ww * x) / np.sum(ww))
    return center


def temporal_consensus(chunks: np.ndarray, tau: float = 12.0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frames = len(chunks)
    out = np.empty((frames, ACTION_DIM), np.float32)
    count = np.empty(frames, np.int64)
    gripper_report: dict[str, Any] = {}
    for t in range(frames):
        starts = np.arange(max(0, t - CHUNK + 1), t + 1)
        horizons = t - starts
        values = chunks[starts, :, horizons, :].reshape(-1, ACTION_DIM)
        weights = np.repeat(np.exp(-horizons / tau), 4)
        count[t] = len(values)
        for j in ARM:
            out[t, j] = robust_huber_location(values[:, j], weights)
    # Grippers: forecast-weighted majority plus hysteresis; output cluster medians, never averages.
    for j in GRIPPER:
        all_values = chunks[..., j].reshape(-1)
        low_center = float(np.median(all_values[all_values <= np.median(all_values)]))
        high_center = float(np.median(all_values[all_values > np.median(all_values)]))
        threshold = .5 * (low_center + high_center)
        state: bool | None = None
        transitions: list[int] = []
        for t in range(frames):
            starts = np.arange(max(0, t - CHUNK + 1), t + 1)
            horizons = t - starts
            values = chunks[starts, :, horizons, j].reshape(-1)
            weights = np.repeat(np.exp(-horizons / tau), 4)
            high_fraction = float(weights[values >= threshold].sum() / weights.sum())
            new_state = high_fraction >= .6 if state is not True else not (high_fraction <= .4)
            if state is not None and new_state != state:
                transitions.append(t)
            state = new_state
            cluster = values[(values >= threshold) == state]
            out[t, j] = float(np.median(cluster)) if len(cluster) else (high_center if state else low_center)
        gripper_report[str(j)] = {
            "low_cluster_median": low_center, "high_cluster_median": high_center,
            "threshold": threshold, "hysteresis_low": .4, "hysteresis_high": .6,
            "transition_frames": transitions,
        }
    return out, count, gripper_report


def optimize_global(consensus: np.ndarray, limits: np.ndarray, strength: float,
                    steps: int, device: str) -> np.ndarray:
    """Bounded global arm optimization; gripper consensus remains discrete."""
    target = torch.as_tensor(consensus[:, ARM], dtype=torch.float64, device=device)
    lo = torch.as_tensor(limits[ARM, 0], dtype=torch.float64, device=device)
    hi = torch.as_tensor(limits[ARM, 1], dtype=torch.float64, device=device)
    ratio = ((target - lo) / (hi - lo)).clamp(1e-7, 1 - 1e-7)
    parameter = torch.nn.Parameter(torch.logit(ratio))
    optimizer = torch.optim.Adam([parameter], lr=.035)
    delta = torch.as_tensor(.015, dtype=torch.float64, device=device)
    for _ in range(steps):
        optimizer.zero_grad()
        x = lo + (hi - lo) * torch.sigmoid(parameter)
        residual = x - target
        robust = (delta * delta * (torch.sqrt(1 + (residual / delta) ** 2) - 1)).mean()
        velocity = torch.diff(x, dim=0)
        acceleration = torch.diff(x, n=2, dim=0)
        jerk = torch.diff(x, n=3, dim=0)
        loss = robust + strength * (
            velocity.square().mean() + 4.0 * acceleration.square().mean()
            + 16.0 * jerk.square().mean()
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        x = lo + (hi - lo) * torch.sigmoid(parameter)
    result = consensus.copy()
    result[:, ARM] = x.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError(f"Non-finite global solution for regularization {strength}")
    return result


def high_frequency_per_frame_joint(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    frequency = np.fft.rfftfreq(len(x), d=1 / FPS)
    spectrum[~((frequency >= 3) & (frequency <= 15))] = 0
    return np.fft.irfft(spectrum, n=len(x), axis=0) ** 2


def boundary_jitter(rtc: np.ndarray, starts: np.ndarray) -> dict[str, Any]:
    boundary = np.zeros(len(rtc), dtype=bool)
    boundary[starts[1:]] = True
    acceleration = np.zeros_like(rtc, dtype=np.float64)
    acceleration[2:] = np.diff(rtc, n=2, axis=0) * FPS**2
    hf = high_frequency_per_frame_joint(rtc)
    acc_energy = acceleration**2
    def split(x: np.ndarray) -> dict[str, Any]:
        total = x.sum(axis=0)
        at_boundary = x[boundary].sum(axis=0)
        return {
            "boundary_mean_per_joint": x[boundary].mean(axis=0).tolist(),
            "non_boundary_mean_per_joint": x[~boundary].mean(axis=0).tolist(),
            "boundary_fraction_per_joint": np.divide(
                at_boundary, total, out=np.zeros_like(total), where=total > 0).tolist(),
            "boundary_fraction_all_joints": float(at_boundary.sum() / total.sum()) if total.sum() else 0.0,
        }
    return {"boundary_frames": np.flatnonzero(boundary).tolist(),
            "acceleration_squared": split(acc_energy), "high_frequency_energy": split(hf)}


def transitions(expert: np.ndarray, x: np.ndarray, j: int) -> dict[str, Any]:
    threshold = float((expert[:, j].min() + expert[:, j].max()) / 2)
    e = np.flatnonzero((expert[1:, j] >= threshold) != (expert[:-1, j] >= threshold)) + 1
    p = np.flatnonzero((x[1:, j] >= threshold) != (x[:-1, j] >= threshold)) + 1
    return {"threshold": threshold, "expert": e.tolist(), "trajectory": p.tolist()}


def metrics(x: np.ndarray, expert: np.ndarray, model: Any, label: str) -> dict[str, Any]:
    step = np.abs(np.diff(x, axis=0))
    velocity = step * FPS
    acceleration = np.abs(np.diff(x, n=2, axis=0)) * FPS**2
    jerk = np.abs(np.diff(x, n=3, axis=0)) * FPS**3
    qpos, mapped_gripper_frames = mapped_qpos(x)
    violations, per_qpos = mapped_limit_counts(qpos, model)
    poses = fk(model, qpos)
    left = np.linalg.norm(np.diff(poses["left_position_m"], axis=0), axis=1) * 1000
    right = np.linalg.norm(np.diff(poses["right_position_m"], axis=0), axis=1) * 1000
    def stats(name: str, value: np.ndarray) -> dict[str, float]:
        return {f"max_{name}": float(value.max(initial=0)),
                f"p95_{name}": float(np.percentile(value, 95)),
                f"p99_{name}": float(np.percentile(value, 99))}
    error = x - expert
    row = {"label": label, "frames": len(x), "action_mae": float(np.mean(np.abs(error))),
           "action_rmse": float(np.sqrt(np.mean(error**2))),
           "high_frequency_energy_3_15hz": float(high_frequency_per_frame_joint(x).mean()),
           "joint_limit_violations": int(violations), "per_qpos_limit_violations": per_qpos,
           "nan_inf_count": int((~np.isfinite(x)).sum()),
           "mapped_gripper_frames": int(mapped_gripper_frames),
           "left_fk_max_jump_mm": float(left.max(initial=0)),
           "left_fk_p99_jump_mm": float(np.percentile(left, 99)),
           "right_fk_max_jump_mm": float(right.max(initial=0)),
           "right_fk_p99_jump_mm": float(np.percentile(right, 99)),
           "left_gripper_transitions": transitions(expert, x, 6),
           "right_gripper_transitions": transitions(expert, x, 13)}
    row.update(stats("joint_step", step))
    row.update(stats("velocity", velocity))
    row.update(stats("acceleration", acceleration))
    row.update(stats("jerk", jerk))
    return row


def render_videos(model: Any, h10: np.ndarray, rtc: np.ndarray, consensus: np.ndarray,
                  optimized: np.ndarray, output: Path) -> None:
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -24, 1.9
    camera.lookat[:] = [.25, 0, .18]
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    q = {name: mapped_qpos(x)[0] for name, x in
         (("H10", h10), ("RTC", rtc), ("Consensus", consensus), ("Optimized", optimized))}
    def open_writer(path: Path, width: int) -> subprocess.Popen[bytes]:
        return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x480", "-r", "30", "-i", "pipe:0", "-an", "-vcodec", "mpeg4",
            "-pix_fmt", "yuv420p", "-q:v", "3", "-y", str(path)], stdin=subprocess.PIPE)
    single = open_writer(output / "aloha_temporal_consensus.mp4", 640)
    comparison = open_writer(output / "h10_vs_rtc_vs_consensus.mp4", 1920)
    try:
        for t in range(len(optimized)):
            images: dict[str, np.ndarray] = {}
            for name in ("H10", "RTC", "Optimized"):
                data.qpos[:] = q[name][t]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((8, 8, 420, 38), fill=(0, 0, 0))
                draw.text((16, 15), f"{name} frame {t}/{len(optimized)-1}", fill=(255, 255, 255))
                images[name] = np.asarray(image)
            assert single.stdin and comparison.stdin
            single.stdin.write(images["Optimized"].tobytes())
            comparison.stdin.write(np.concatenate(
                [images["H10"], images["RTC"], images["Optimized"]], axis=1).tobytes())
        single.stdin.close()
        comparison.stdin.close()
        if single.wait() or comparison.wait():
            raise RuntimeError("ffmpeg encoding failed")
    finally:
        renderer.close()
        for process in (single, comparison):
            if process.poll() is None:
                process.terminate()


def plots(path: Path, trajectories: dict[str, np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
    for j, ax in enumerate(axes.flat):
        for name, x in trajectories.items():
            ax.plot(x[:, j], label=name, linewidth=.6)
        ax.set_title(f"joint {j}")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(path / "joint_trajectories.png", dpi=130)
    plt.close(fig)
    for order, name, scale in ((1, "joint_step", 1), (2, "acceleration", FPS**2),
                               (3, "jerk", FPS**3)):
        fig, ax = plt.subplots(figsize=(14, 4))
        for label, x in trajectories.items():
            ax.plot(np.max(np.abs(np.diff(x, n=order, axis=0)), axis=1) * scale, label=label)
        ax.legend()
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(path / f"{name}.png", dpi=140)
        plt.close(fig)


def rank_candidate(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["joint_limit_violations"], row["nan_inf_count"],
            max(row["left_fk_max_jump_mm"], row["right_fk_max_jump_mm"]),
            row["p99_acceleration"], row["high_frequency_energy_3_15hz"],
            row["consensus_rmse"])


def main() -> int:
    args = parse_args()
    if args.dry_run == args.execute:
        raise ValueError("Specify exactly one of --dry-run or --execute")
    if lerobot_version != "0.6.1":
        raise RuntimeError(f"Expected LeRobot 0.6.1, got {lerobot_version}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint inference")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    for source in (args.checkpoint, args.dataset_root, RTC_FILE, H10_FILE, XML):
        if not source.exists():
            raise FileNotFoundError(source)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "plots").mkdir()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, download_videos=False)
    start, end = episode_bounds(dataset, 49)
    total = end - start
    frames = min(total, args.max_frames or total)
    if args.dry_run:
        frames = min(frames, 12)
    with np.load(RTC_FILE, allow_pickle=False) as z:
        rtc = z["rtc_action"][:frames].astype(np.float32)
        state = z["observation_state"][:frames].astype(np.float32)
        frame_index = z["frame_index"][:frames].astype(np.int64)
        timestamp = z["timestamp"][:frames].astype(np.float32)
        rtc_starts = z["chunk_start_frames"]
        rtc_starts = rtc_starts[rtc_starts < frames].astype(np.int64)
        expert = z["expert_action"][:frames].astype(np.float32)
    with np.load(H10_FILE, allow_pickle=False) as z:
        h10 = z["chunk_stitched_h10"][:frames].astype(np.float32)
        if not np.allclose(z["expert_action"][:frames], expert):
            raise RuntimeError("Expert alignment mismatch")
    if not args.dry_run and (total != 990 or frames != 990):
        raise RuntimeError(f"Expected full episode length 990, got {total}/{frames}")

    model, mapping = load_validated_model(XML)
    limits = dataset_limits(model)
    jitter = boundary_jitter(rtc, rtc_starts)

    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, local_files_only=True).to(args.device)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    chunks, latency = generate_chunks(policy, pre, post, dataset, start, frames)
    if chunks.shape != (frames, 4, CHUNK, ACTION_DIM):
        raise RuntimeError(f"Chunk tensor mismatch: {chunks.shape}")
    incomplete = args.output_dir / "all_predicted_chunks.npz.incomplete"
    with incomplete.open("wb") as f:
        np.savez_compressed(f, predicted_chunks=chunks, sample_seeds=np.arange(4, dtype=np.int64),
                            frame_index=frame_index, inference_latency_s=latency)
    os.replace(incomplete, args.output_dir / "all_predicted_chunks.npz")

    consensus, counts, gripper_report = temporal_consensus(chunks)
    rows: list[dict[str, Any]] = []
    candidates: dict[float, np.ndarray] = {}
    steps = min(args.optimization_steps, 40) if args.dry_run else args.optimization_steps
    for strength in REGULARIZATION:
        optimized = optimize_global(consensus, limits, strength, steps, args.device)
        row = metrics(optimized, expert, model, f"optimized_lambda_{strength:g}")
        row["regularization"] = strength
        row["consensus_rmse"] = float(np.sqrt(np.mean((optimized - consensus)**2)))
        rows.append(row)
        candidates[strength] = optimized
        print(f"lambda={strength:g} limit={row['joint_limit_violations']} "
              f"fk={max(row['left_fk_max_jump_mm'],row['right_fk_max_jump_mm']):.2f}mm "
              f"p99_acc={row['p99_acceleration']:.2f}", flush=True)
    selected_row = min(rows, key=rank_candidate)
    strength = float(selected_row["regularization"])
    optimized = candidates[strength]

    comparison = [
        metrics(expert, expert, model, "expert_action"),
        metrics(h10, expert, model, "chunk_stitched_h10"),
        metrics(rtc, expert, model, "rtc_action"),
        metrics(consensus, expert, model, "temporal_consensus"),
        metrics(optimized, expert, model, "optimized_temporal_consensus"),
    ]
    rows.extend(comparison)
    dynamic: dict[str, Any] = {"status": "NOT_RUN_IN_DRY_RUN"}
    if not args.dry_run:
        if selected_row["joint_limit_violations"] or selected_row["nan_inf_count"]:
            dynamic = {"status": "SAFETY_BLOCKED"}
        else:
            qpos, _ = mapped_qpos(optimized)
            ctrl = optimized.copy()
            ctrl[:, 6] = qpos[:, 6]
            ctrl[:, 13] = qpos[:, 14]
            dyn, actual = dynamic_replay(model, qpos[0], ctrl)
            dynamic = {"status": "SUCCESS", **dyn}
            np.savez_compressed(args.output_dir / "dynamic_replay_tracking.npz",
                                target=ctrl, actual=actual)

    rtc_row = next(r for r in comparison if r["label"] == "rtc_action")
    opt_row = next(r for r in comparison if r["label"] == "optimized_temporal_consensus")
    reductions = {
        "p99_acceleration_percent": float(
            (rtc_row["p99_acceleration"] - opt_row["p99_acceleration"]) /
            rtc_row["p99_acceleration"] * 100),
        "high_frequency_energy_percent": float(
            (rtc_row["high_frequency_energy_3_15hz"] - opt_row["high_frequency_energy_3_15hz"]) /
            rtc_row["high_frequency_energy_3_15hz"] * 100),
        "max_fk_jump_percent": float(
            (max(rtc_row["left_fk_max_jump_mm"], rtc_row["right_fk_max_jump_mm"]) -
             max(opt_row["left_fk_max_jump_mm"], opt_row["right_fk_max_jump_mm"])) /
            max(rtc_row["left_fk_max_jump_mm"], rtc_row["right_fk_max_jump_mm"]) * 100),
    }
    quantitative_ready = bool(
        opt_row["joint_limit_violations"] == 0 and opt_row["nan_inf_count"] == 0
        and reductions["p99_acceleration_percent"] > 5
        and reductions["high_frequency_energy_percent"] > 5
        and reductions["max_fk_jump_percent"] > 0
    )
    status = ("ALOHA_TEMPORAL_CONSENSUS_READY_FOR_G1"
              if quantitative_ready else "ALOHA_TEMPORAL_CONSENSUS_BLOCKED")

    payload = dict(temporal_consensus_action=consensus, optimized_action=optimized,
                   rtc_action=rtc, observation_state=state, expert_action=expert,
                   prediction_count_per_frame=counts,
                   selected_regularization=np.asarray(strength), fps=np.asarray(FPS),
                   frame_index=frame_index, timestamp=timestamp)
    incomplete = args.output_dir / "episode_000049_temporal_consensus.npz.incomplete"
    with incomplete.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(incomplete, args.output_dir / "episode_000049_temporal_consensus.npz")
    atomic_csv(args.output_dir / "candidate_comparison.csv", rows)
    if not args.dry_run:
        plots(args.output_dir / "plots", {
            "expert": expert, "H10": h10, "RTC": rtc,
            "consensus": consensus, "optimized": optimized})
        render_videos(model, h10, rtc, consensus, optimized, args.output_dir)
    report = {
        "evaluation_type": "training-set teacher-forced offline overlapping-chunk temporal-consensus sanity evaluation",
        "trajectory_name": "SmolVLA overlapping-chunk temporal-consensus trajectory",
        "expert_used_for_generation_or_selection": False,
        "expert_used_only_for_final_evaluation": True,
        "frames": frames, "raw_chunk_shape": list(chunks.shape),
        "stochastic_seeds": [0, 1, 2, 3], "forecast_weight": "exp(-horizon/12)",
        "policy_reset_count": 1, "moving_average_or_clipping_used": False,
        "mujoco_xml": str(XML), "validated_mapping": mapping,
        "rtc_boundary_jitter": jitter, "gripper_consensus": gripper_report,
        "latency_ms": {"mean": float(latency.mean() * 1000),
                       "p95": float(np.percentile(latency, 95) * 1000),
                       "max": float(latency.max() * 1000)},
        "regularization_candidates": list(REGULARIZATION),
        "selected_regularization": strength, "selected_metrics": selected_row,
        "comparison": comparison, "rtc_reduction_percent": reductions,
        "dynamic_replay": dynamic, "quantitative_ready": quantitative_ready,
        "visual_review_required": True, "g1_progress_status": status,
        "hardware_commands_sent": False, "g1_conversion_executed": False,
        "isaac_lab_executed": False,
    }
    atomic_json(args.output_dir / "temporal_consensus_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
