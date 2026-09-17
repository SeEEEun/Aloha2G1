#!/usr/bin/env python3
"""Quantify source-clock-forced Policy-B diagnostic rollouts against frozen targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/policy_b_g1visual/teacher_forced_diagnostic"
TRAJECTORIES = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories"
GROUPS = {
    "left_arm": np.arange(0, 7),
    "right_arm": np.arange(7, 14),
    "left_dex3": np.arange(14, 21),
    "right_dex3": np.arange(21, 28),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
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
    os.replace(temporary, path)


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def motion_metrics(actual: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    actual = actual[:, indices]
    target = target[:, indices]
    actual_motion = actual - actual[:1]
    target_motion = target - target[:1]
    denominator = float(np.linalg.norm(actual_motion) * np.linalg.norm(target_motion))
    target_energy = float(np.sum(target_motion * target_motion))
    return {
        "actual_motion_rms_rad": rms(actual_motion),
        "target_motion_rms_rad": rms(target_motion),
        "actual_to_target_motion_ratio": rms(actual_motion) / max(rms(target_motion), 1e-12),
        "trajectory_motion_cosine": (
            float(np.sum(actual_motion * target_motion) / denominator) if denominator > 1e-12 else None
        ),
        "target_direction_projection_progress": (
            float(np.sum(actual_motion * target_motion) / target_energy) if target_energy > 1e-12 else None
        ),
        "absolute_rmse_rad": rms(actual - target),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episodes", default="0,24,49")
    args = parser.parse_args()
    root = args.root.resolve()
    episodes = [int(value) for value in args.episodes.split(",")]
    aggregate = []
    for episode in episodes:
        stage = root / f"episode_{episode:06d}" / "full_policy_b_diagnostic_rollout"
        report = read_json(stage / "stage_report.json")
        trace = np.load(stage / "rollout_trace.npz", allow_pickle=False)
        trajectory = np.load(TRAJECTORIES / f"episode_{episode:06d}.npz", allow_pickle=False)
        names = trace["joint_names"].astype(str).tolist()
        source_names = trajectory["replay_joint_names"].astype(str).tolist()
        order = [source_names.index(name) for name in names]
        target = trajectory["replay_named_joint_qpos"][:, order].astype(np.float64)
        actual = trace["actual_q"].astype(np.float64)
        count = min(len(actual), len(target))
        actual, target = actual[:count], target[:count]
        events = dict(zip(trajectory["event_names"].astype(str), trajectory["event_frames"].astype(int)))
        ownership = trajectory["ownership_state"].astype(str)
        right_transport_start = int(np.flatnonzero(ownership == "RIGHT_TRANSPORT")[0])
        windows = {
            "left_grasp_transition": (
                events["LEFT_CLOSE_ONSET"], min(count, events["LEFT_STABLE_HOLD"] + 12), "left_dex3"
            ),
            "left_transport": (
                events["LEFT_STABLE_HOLD"], min(count, events["HANDOFF_APPROACH"]), "left_arm"
            ),
            "right_handoff_approach": (
                events["HANDOFF_APPROACH"], min(count, events["RIGHT_ACQUIRE"] + 1), "right_arm"
            ),
            "dual_hand_transition_left_release": (
                events["RIGHT_ACQUIRE"], min(count, events["LEFT_RELEASE"] + 12), "left_dex3"
            ),
            "dual_hand_transition_right_acquire": (
                max(0, events["RIGHT_CLOSE_ONSET"]), min(count, events["RIGHT_STABLE_HOLD"] + 8), "right_dex3"
            ),
            "right_transport": (
                right_transport_start, min(count, events["RIGHT_FINAL_RELEASE"]), "right_arm"
            ),
            "right_release": (
                events["RIGHT_FINAL_RELEASE"], min(count, events["RIGHT_FINAL_RELEASE"] + 50), "right_dex3"
            ),
        }
        phase_metrics = {}
        for phase, (start, end, group) in windows.items():
            if end - start < 2:
                phase_metrics[phase] = {"available": False, "start": start, "end": end, "group": group}
            else:
                phase_metrics[phase] = {
                    "available": True, "start": start, "end_exclusive": end, "group": group,
                    **motion_metrics(actual[start:end], target[start:end], GROUPS[group]),
                }

        timestamps = np.arange(count) / 30.0
        figure, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        for axis, (group, indices) in zip(axes, GROUPS.items()):
            actual_motion = np.sqrt(np.mean((actual[:, indices] - actual[:1, indices]) ** 2, axis=1))
            target_motion = np.sqrt(np.mean((target[:, indices] - target[:1, indices]) ** 2, axis=1))
            axis.plot(timestamps, target_motion, color="black", label="frozen target")
            axis.plot(timestamps, actual_motion, color="tab:orange", label="measured forced-RGB rollout")
            axis.set_ylabel(f"{group}\nRMS q displacement")
            axis.grid(alpha=0.25)
        axes[0].legend()
        axes[-1].set_xlabel("source episode clock / executed time (s)")
        figure.suptitle(f"Source-RGB time-forced diagnostic — episode {episode:02d}")
        figure.tight_layout()
        plot = stage / "source_clock_target_vs_measured_motion.png"
        figure.savefig(plot, dpi=180)
        plt.close(figure)

        comparison = stage / "source_rgb_vs_g1_overview.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(stage / "source_rgb_time_forced.mp4"),
                "-i", str(stage / "overview.mp4"),
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", str(comparison),
            ],
            check=True,
        )
        result = {
            "episode_index": episode,
            "classification": "SOURCE_RGB_TIME_FORCED_POLICY_ROLLOUT",
            "closed_loop_vision": False,
            "status": report["status"],
            "executed_frames": int(report["executed_control_frames"]),
            "duration_s": float(report["duration_s"]),
            "inference_calls": int(report["inference_call_count"]),
            "phase_metrics": phase_metrics,
            "videos": report["videos"],
            "source_rgb_vs_g1_overview": str(comparison),
            "target_vs_measured_plot": str(plot),
            "physical_task_success_claimed": False,
        }
        atomic_json(stage / "source_rgb_time_forced_analysis.json", result)
        aggregate.append(result)
    summary = {
        "schema_version": "source_rgb_time_forced_policy_rollout_analysis_v1",
        "classification": "SOURCE_RGB_TIME_FORCED_POLICY_ROLLOUT",
        "episodes": episodes,
        "rollouts": aggregate,
        "SOURCE_RGB_TIME_FORCED_PROGRESS": "PENDING_VISUAL_REVIEW",
        "interpretation": (
            "This asks whether an in-distribution visual clock elicits later phases while Isaac measured state "
            "remains live. It is not closed-loop vision and cannot establish task success."
        ),
    }
    atomic_json(root / "source_rgb_time_forced_summary.json", summary)
    print(json.dumps({
        "status": "ANALYSIS_COMPLETE_VISUAL_REVIEW_PENDING",
        "episodes": episodes,
        "summary": str(root / "source_rgb_time_forced_summary.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
