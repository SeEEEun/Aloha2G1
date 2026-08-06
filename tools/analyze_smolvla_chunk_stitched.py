#!/usr/bin/env python3
"""Generate and preflight chunk-stitched teacher-forced SmolVLA trajectories."""
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
sys.path.insert(0, str(ROOT / "tools"))
from evaluate_smolvla_checkpoints import (  # noqa: E402
    ACTION_DIM, CHUNK_SIZE, TASK, episode_bounds, finite_action, seed_all,
    LeRobotDataset, SmolVLAPolicy, make_pre_post_processors,
)

DATASET_ROOT = ROOT / "lerobot_magsafe_50_cam_high_v3"
REPO_ID = "local/magsafe_aloha_50_cam_high_v3"
PRED_ROOT = ROOT / "evaluation/smolvla_20k_full_predictions"
OUT = ROOT / "evaluation/smolvla_20k_chunk_stitched_preflight"
CHECKPOINT_ROOT = ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints"
STEPS = ["005000", "010000", "015000", "020000"]
EPISODES = [0, 24, 49]
FPS = 30.0
SEED = 1000
LIMITS = np.asarray([
    [-3.0543298721, 3.0543298721], [0.0, 3.14158988], [0.0, 2.35618997],
    [-1.57080007, 1.57080007], [-1.57080007, 1.57080007], [-3.14158988, 3.14158988],
    [0.0, 0.044], [-3.0543298721, 3.0543298721], [0.0, 3.14158988],
    [0.0, 2.35618997], [-1.57080007, 1.57080007], [-1.57080007, 1.57080007],
    [-3.14158988, 3.14158988], [0.0, 0.044],
], dtype=np.float32)
VELOCITY_LIMITS = np.asarray([8.0] * 6 + [.5] + [8.0] * 6 + [.5], np.float32)
LOG = logging.getLogger("chunk_stitched")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".incomplete")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".incomplete")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".incomplete")
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def adapter(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact existing replay adapter: raw tolerance check, carriage clamp only."""
    raw = np.asarray(raw, np.float32)
    target = raw.copy()
    tolerance_bad = ((raw[:, [6, 13]] < -0.001) | (raw[:, [6, 13]] > 0.0441))
    target[:, [6, 13]] = np.minimum(np.maximum(raw[:, [6, 13]], 0.0), 0.044)
    return target, {
        "raw_gripper_tolerance_violation_count": int(tolerance_bad.sum()),
        "raw_gripper_tolerance_violation_per_side": tolerance_bad.sum(axis=0).astype(int).tolist(),
    }


def generate(dataset: LeRobotDataset, policy: Any, pre: Any, post: Any,
             episode: int, horizon: int, max_frames: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    start, end = episode_bounds(dataset, episode)
    if max_frames is not None:
        end = min(end, start + max_frames)
    total = end - start
    trajectory = np.empty((total, ACTION_DIM), np.float32)
    starts = np.empty(total, np.int64)
    policy.reset()
    pre.reset()
    post.reset()
    for local in range(0, total, horizon):
        global_idx = start + local
        seed_all(SEED + local)
        item = dataset[global_idx]
        batch = pre({
            "observation.images.cam_high": item["observation.images.cam_high"],
            "observation.state": item["observation.state"],
            "task": TASK,
        })
        with torch.inference_mode():
            chunk = finite_action(
                post(policy.predict_action_chunk(batch)), "chunk", (1, CHUNK_SIZE, ACTION_DIM)
            )[0]
        take = min(horizon, total - local)
        trajectory[local:local + take] = chunk[:take]
        starts[local:local + take] = local
        if local % (horizon * 20) == 0:
            LOG.info("episode=%d H=%d frame=%d/%d", episode, horizon, local, total)
    return trajectory, starts


def transitions(x: np.ndarray, joint: int, threshold: float) -> list[int]:
    state = x[:, joint] >= threshold
    return (np.flatnonzero(state[1:] != state[:-1]) + 1).astype(int).tolist()


def fk_jump(raw: np.ndarray) -> dict[str, float]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("aloha_fk", ROOT / "tools/export_stationary_fk.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(mod.DEFAULT_MODEL))
    poses = mod.compute_trajectory(model, raw.astype(np.float64))
    lp = poses["left_position_xyz_m"]
    rp = poses["right_position_xyz_m"]
    return {
        "fk_left_max_frame_jump_mm": float(np.max(np.linalg.norm(np.diff(lp, axis=0), axis=1)) * 1000),
        "fk_right_max_frame_jump_mm": float(np.max(np.linalg.norm(np.diff(rp, axis=0), axis=1)) * 1000),
    }


def metrics(raw: np.ndarray, initial: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    sim, adapter_report = adapter(raw)
    delta_raw = np.abs(np.diff(raw, axis=0))
    delta_sim = np.abs(np.diff(sim, axis=0))
    velocity = delta_sim * FPS
    acceleration = np.abs(np.diff(sim, n=2, axis=0)) * FPS * FPS
    violation = (sim < LIMITS[:, 0]) | (sim > LIMITS[:, 1])
    expert_delta = np.abs(np.diff(expert, axis=0))
    result = {
        **adapter_report,
        "converted_joint_limit_violation_count": int(violation.sum()),
        "converted_violation_per_joint": violation.sum(axis=0).astype(int).tolist(),
        "arm_violations": int(violation[:, [*range(6), *range(7, 13)]].sum()),
        "gripper_violations": int(violation[:, [6, 13]].sum()),
        "raw_max_jump": float(delta_raw.max()),
        "raw_p99_jump": float(np.percentile(delta_raw, 99)),
        "converted_max_jump": float(delta_sim.max()),
        "converted_p99_jump": float(np.percentile(delta_sim, 99)),
        "expert_max_jump": float(expert_delta.max()),
        "expert_p99_9_jump": float(np.percentile(expert_delta, 99.9)),
        "max_velocity": float(velocity.max()),
        "p99_velocity": float(np.percentile(velocity, 99)),
        "physical_velocity_limit_violation_count": int((velocity > VELOCITY_LIMITS).sum()),
        "max_acceleration": float(acceleration.max()) if len(acceleration) else 0.0,
        "first_target_vs_initial_state_linf": float(np.max(np.abs(sim[0] - initial))),
        "nan_inf_count": int(raw.size - np.isfinite(raw).sum()),
        "raw_gripper_minmax": {
            "left": [float(raw[:, 6].min()), float(raw[:, 6].max())],
            "right": [float(raw[:, 13].min()), float(raw[:, 13].max())],
        },
        "converted_gripper_minmax": {
            "left": [float(sim[:, 6].min()), float(sim[:, 6].max())],
            "right": [float(sim[:, 13].min()), float(sim[:, 13].max())],
        },
    }
    for j, name in ((6, "left"), (13, "right")):
        threshold = float((expert[:, j].min() + expert[:, j].max()) / 2)
        result[f"{name}_gripper_transition_count"] = len(transitions(raw, j, threshold))
        result[f"{name}_gripper_transition_frames"] = transitions(raw, j, threshold)
    result.update(fk_jump(raw))
    return result


def plot_episode(path: Path, ep: int, trajectories: dict[str, np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for name, x in trajectories.items():
        axes[0].plot(np.max(np.abs(np.diff(x, axis=0)), axis=1), label=name, alpha=.8)
        axes[1].plot(x[:, 13], label=name, alpha=.8)
    axes[0].set_title("frame max joint jump")
    axes[1].set_title("right gripper raw action")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path / f"episode_{ep:06d}_continuity.png", dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(REPO_ID, root=DATASET_ROOT, download_videos=False)
    episodes = [0] if cfg.dry_run else EPISODES
    max_frames = 10 if cfg.dry_run else None
    checkpoint = CHECKPOINT_ROOT / "020000/pretrained_model"
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    comparisons: list[dict[str, Any]] = []
    joint3_rows: list[dict[str, Any]] = []
    generated: dict[int, dict[str, np.ndarray]] = {}
    starts: dict[int, dict[str, np.ndarray]] = {}
    episode_sources: dict[int, dict[str, Any]] = {}
    for ep in episodes:
        src_path = PRED_ROOT / f"episode_{ep:06d}_prediction.npz"
        with np.load(src_path, allow_pickle=False) as z:
            src = {k: z[k].copy() for k in z.files}
        if max_frames:
            source_length = len(src["frame_index"])
            for key, value in list(src.items()):
                if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == source_length:
                    src[key] = value[:max_frames]
        h5, s5 = generate(dataset, policy, pre, post, ep, 5, max_frames)
        h10, s10 = generate(dataset, policy, pre, post, ep, 10, max_frames)
        generated[ep] = {"chunk_stitched_h5": h5, "chunk_stitched_h10": h10}
        starts[ep] = {"chunk_stitched_h5": s5, "chunk_stitched_h10": s10}
        episode_sources[ep] = src
        trajectories = {
            "expert_action": src["expert_action"],
            "independent_first_action": src["teacher_forced_predicted_action"],
            "chunk_stitched_h5": h5,
            "chunk_stitched_h10": h10,
        }
        sim_targets = {}
        for name, raw in trajectories.items():
            row = {"episode_index": ep, "trajectory": name, **metrics(
                raw, src["observation_state"][0], src["expert_action"]
            )}
            comparisons.append(row)
            sim_targets[name] = adapter(raw)[0]
            violation = (sim_targets[name][:, 3] < LIMITS[3, 0]) | (
                sim_targets[name][:, 3] > LIMITS[3, 1]
            )
            for frame in np.flatnonzero(violation):
                joint3_rows.append({
                    "episode_index": ep, "frame_index": int(src["frame_index"][frame]),
                    "trajectory": name, "expert_action_value": float(src["expert_action"][frame, 3]),
                    "predicted_raw_value": float(raw[frame, 3]),
                    "converted_target_value": float(sim_targets[name][frame, 3]),
                    "lower_limit": float(LIMITS[3, 0]), "upper_limit": float(LIMITS[3, 1]),
                    "previous_value": float(raw[max(0, frame - 1), 3]),
                    "next_value": float(raw[min(len(raw) - 1, frame + 1), 3]),
                    "prediction_chunk_start_frame": (
                        int(starts[ep][name][frame]) if name in starts[ep] else int(frame)
                    ),
                    "cause": (
                        "EXPERT_RAW_OUTSIDE_ASSET_LIMIT__MAPPING_INVALID"
                        if name == "expert_action"
                        else "MODEL_RAW_ACTION_OUTSIDE_ASSET_LIMIT"
                    ),
                })
        if not cfg.dry_run:
            atomic_npz(cfg.output_dir / f"episode_{ep:06d}_chunk_stitched.npz", {
                "expert_action": trajectories["expert_action"].astype(np.float32),
                "independent_first_action": trajectories["independent_first_action"].astype(np.float32),
                "chunk_stitched_h5": h5, "chunk_stitched_h10": h10,
                "sim_target_expert": sim_targets["expert_action"],
                "sim_target_h5": sim_targets["chunk_stitched_h5"],
                "sim_target_h10": sim_targets["chunk_stitched_h10"],
                "frame_index": src["frame_index"].astype(np.int64),
                "timestamp": src["timestamp"],
                "h5_chunk_start_frame": s5, "h10_chunk_start_frame": s10,
            })
            plot_episode(cfg.output_dir / "plots", ep, trajectories)
    if cfg.dry_run:
        LOG.info("dry-run successful")
        return

    del policy, pre, post
    torch.cuda.empty_cache()
    checkpoint_rows = []
    existing_summary = {
        r["checkpoint_step"]: r for r in csv.DictReader(
            open(ROOT / "evaluation/smolvla_first50_checkpoints/summary.csv", encoding="utf-8")
        )
    }
    for step in STEPS:
        if step == "020000":
            h10 = generated[0]["chunk_stitched_h10"]
        else:
            cp = CHECKPOINT_ROOT / f"{step}/pretrained_model"
            pol = SmolVLAPolicy.from_pretrained(cp, local_files_only=True)
            pp, po = make_pre_post_processors(pol.config, pretrained_path=str(cp))
            h10, _ = generate(dataset, pol, pp, po, 0, 10)
            del pol, pp, po
            torch.cuda.empty_cache()
        m = metrics(h10, episode_sources[0]["observation_state"][0], episode_sources[0]["expert_action"])
        checkpoint_rows.append({
            "checkpoint": step,
            "first_action_rmse": float(existing_summary[step]["first_action_rmse"]),
            "chunk_rmse": float(existing_summary[step]["chunk_rmse"]),
            "converted_joint_limit_violations": m["converted_joint_limit_violation_count"],
            "max_converted_jump": m["converted_max_jump"],
            "p99_converted_jump": m["converted_p99_jump"],
            "max_velocity": m["max_velocity"],
            "gripper_violations": m["gripper_violations"],
            "arm_violations": m["arm_violations"],
        })
    flat_rows = []
    for row in comparisons:
        flat_rows.append({k: v for k, v in row.items() if not isinstance(v, (dict, list))})
    write_csv(cfg.output_dir / "expert_vs_prediction_preflight.csv", flat_rows)
    write_csv(cfg.output_dir / "checkpoint_safety_comparison.csv", checkpoint_rows)
    write_csv(cfg.output_dir / "joint3_violation_details.csv", joint3_rows)
    expert_rows = [r for r in comparisons if r["trajectory"] == "expert_action"]
    mapping_valid = all(r["converted_joint_limit_violation_count"] == 0 for r in expert_rows)
    atomic_json(cfg.output_dir / "gripper_mapping_report.json", {
        "source": str(ROOT / "isaaclab_magsafe_fixed_scene/replay_magsafe_aloha_episode.py"),
        "adapter": {
            "raw_tolerance": [-0.001, 0.0441],
            "sim_carriage_target": "min(max(raw_gripper, 0.0), 0.044)",
            "carriage_mapping": "same converted value duplicated to left/right carriage joints",
            "raw_model_action_modified": False,
        },
        "expert_adapter_validation": "SUCCESS" if mapping_valid else "MAPPING_INVALID",
        "episodes": [{
            "episode_index": ep,
            "expert": next(r for r in comparisons if r["episode_index"] == ep and r["trajectory"] == "expert_action"),
            "independent": next(r for r in comparisons if r["episode_index"] == ep and r["trajectory"] == "independent_first_action"),
        } for ep in EPISODES],
    })
    h5_rows = [r for r in comparisons if r["trajectory"] == "chunk_stitched_h5"]
    h10_rows = [r for r in comparisons if r["trajectory"] == "chunk_stitched_h10"]
    allowed = mapping_valid and all(
        r["converted_joint_limit_violation_count"] == 0 and r["nan_inf_count"] == 0
        for r in h10_rows
    )
    atomic_json(cfg.output_dir / "trajectory_generation_report.json", {
        "name": "chunk-stitched teacher-forced offline trajectory",
        "policy_reset": "once at episode start",
        "horizons": [5, 10],
        "chunk_api": "SmolVLAPolicy.predict_action_chunk public API",
        "overlap_aggregation": "not used; first H actions from each chunk are concatenated",
        "raw_action_clipping": False,
        "results": comparisons,
    })
    atomic_json(cfg.output_dir / "final_preflight_report.json", {
        "expert_adapter_validation": "SUCCESS" if mapping_valid else "MAPPING_INVALID",
        "physics_dry_run_allowed": allowed,
        "physics_steps_executed": 0,
        "recommended_trajectory": (
            "chunk_stitched_h10" if allowed else "NONE_SAFETY_PREFLIGHT_BLOCKED"
        ),
        "checkpoint_comparison": checkpoint_rows,
        "joint3_violation_rows": len(joint3_rows),
        "h5_mean_converted_p99_jump": float(np.mean([r["converted_p99_jump"] for r in h5_rows])),
        "h10_mean_converted_p99_jump": float(np.mean([r["converted_p99_jump"] for r in h10_rows])),
        "outputs": str(cfg.output_dir),
    })


if __name__ == "__main__":
    main()
