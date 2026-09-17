#!/usr/bin/env python3
"""Render the selected ACT-B nine-phase offline probe as centered motion traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SELECTION = ROOT / "outputs/policy_b_act/selected_checkpoint.json"
EVALUATION_ROOT = ROOT / "outputs/policy_b_act/checkpoint_evaluation"
OUTPUT = ROOT / "outputs/policy_b_act/phase_probe"
PHASES = (
    "initial_left_approach",
    "immediately_before_left_grasp",
    "left_grasp_left_owned",
    "left_transport",
    "handoff_approach",
    "dual_contact_transfer",
    "right_owned",
    "right_transport",
    "release",
)
RELEVANT = {
    "initial_left_approach": np.arange(0, 7),
    "immediately_before_left_grasp": np.arange(14, 21),
    "left_grasp_left_owned": np.arange(14, 21),
    "left_transport": np.arange(0, 7),
    "handoff_approach": np.arange(7, 14),
    "dual_contact_transfer": np.arange(14, 28),
    "right_owned": np.arange(14, 21),
    "right_transport": np.arange(7, 14),
    "release": np.arange(21, 28),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    step = int(selection["selected_checkpoint_step"])
    evaluation = json.loads(
        (EVALUATION_ROOT / f"checkpoint_{step:06d}_evaluation.json").read_text(encoding="utf-8")
    )
    prediction_path = Path(evaluation["prediction_artifact"])
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = archive["act_prediction"].astype(np.float64)
        target = archive["authoritative_target"].astype(np.float64)
        phase_names = archive["phase"].astype(str)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    time = np.arange(50) / 30.0
    figure, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True)
    cmap = plt.get_cmap("tab20")
    for axis, phase in zip(axes.flat, PHASES, strict=True):
        samples = np.flatnonzero(phase_names == phase)
        joints = RELEVANT[phase]
        target_motion = target[samples][:, :, joints] - target[samples][:, :1, joints]
        act_motion = prediction[samples][:, :, joints] - prediction[samples][:, :1, joints]
        target_mean = np.mean(target_motion, axis=0)
        act_mean = np.mean(act_motion, axis=0)
        for local_joint in range(len(joints)):
            color = cmap(local_joint)
            axis.plot(time, target_mean[:, local_joint], color=color, linewidth=1.7, alpha=0.85)
            axis.plot(time, act_mean[:, local_joint], color=color, linewidth=1.0, alpha=0.9, linestyle="--")
        metrics = evaluation["phases"][phase]
        axis.set_title(
            f"{metrics['phase_label']}\n"
            f"success {metrics['successful_qualitative_probes']}/6 | "
            f"RMSE {metrics['full_available_chunk_rmse_rad']:.3f} rad",
            fontsize=10,
        )
        axis.grid(alpha=0.2)
        axis.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("future chunk time (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("mean motion from row 0 (rad)")
    figure.suptitle(
        "ACT-B same frozen 9-phase probe — solid Dataset target, dashed ACT (six episodes/phase)",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / "nine_phase_centered_motion_overview.png", dpi=180)
    plt.close(figure)

    # A compact target-vs-prediction displacement heatmap makes stop/reverse
    # patterns and wrong-sign phase behavior visible without mixing base poses.
    figure, axes = plt.subplots(9, 2, figsize=(14, 25), sharex=True)
    maximum = 0.0
    matrices = []
    for phase in PHASES:
        samples = np.flatnonzero(phase_names == phase)
        joints = RELEVANT[phase]
        target_mean = np.mean(target[samples][:, :, joints] - target[samples][:, :1, joints], axis=0).T
        act_mean = np.mean(prediction[samples][:, :, joints] - prediction[samples][:, :1, joints], axis=0).T
        matrices.append((target_mean, act_mean))
        maximum = max(maximum, float(np.max(np.abs(target_mean))), float(np.max(np.abs(act_mean))))
    maximum = max(maximum, 1e-6)
    for row, (phase, (target_mean, act_mean)) in enumerate(zip(PHASES, matrices, strict=True)):
        axes[row, 0].imshow(target_mean, aspect="auto", cmap="coolwarm", vmin=-maximum, vmax=maximum)
        image = axes[row, 1].imshow(act_mean, aspect="auto", cmap="coolwarm", vmin=-maximum, vmax=maximum)
        axes[row, 0].set_ylabel(phase.replace("_", " "), fontsize=8)
        axes[row, 0].set_yticks([])
        axes[row, 1].set_yticks([])
    axes[0, 0].set_title("Dataset target centered motion")
    axes[0, 1].set_title("ACT raw centered motion")
    axes[-1, 0].set_xlabel("future row 0..49")
    axes[-1, 1].set_xlabel("future row 0..49")
    figure.colorbar(image, ax=axes, shrink=0.35, label="motion from first action (rad)")
    figure.suptitle("Nine semantic phases: relevant joint-group mean across six frozen probes", y=0.995)
    figure.savefig(OUTPUT / "nine_phase_motion_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_step": step,
                "phase_score": evaluation["phase_score"],
                "outputs": [
                    str(OUTPUT / "nine_phase_centered_motion_overview.png"),
                    str(OUTPUT / "nine_phase_motion_heatmap.png"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
