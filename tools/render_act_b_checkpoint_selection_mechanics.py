#!/usr/bin/env python3
"""Visualize raw mechanics for the bounded ACT-B checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
EVAL = ROOT / "outputs/policy_b_act/checkpoint_evaluation"
SELECTIONS = ROOT / "outputs/policy_b_offline_phase_probe/selected_frames.json"
PHASES = (
    "initial_left_approach",
    "left_grasp_left_owned",
    "left_transport",
    "handoff_approach",
    "right_transport",
    "release",
)


def main() -> None:
    summary = json.loads((EVAL / "checkpoint_comparison.json").read_text(encoding="utf-8"))
    selections = json.loads(SELECTIONS.read_text(encoding="utf-8"))
    representative = [
        next(
            index
            for index, row in enumerate(selections)
            if int(row["episode_index"]) == 24 and row["phase"] == phase
        )
        for phase in PHASES
    ]
    steps = [int(value) for value in summary["evaluated_steps"]]
    predictions = []
    target = None
    smol = None
    for step in steps:
        with np.load(EVAL / f"checkpoint_{step:06d}_phase_predictions.npz", allow_pickle=False) as archive:
            predictions.append(archive["act_prediction"].astype(np.float64))
            if target is None:
                target = archive["authoritative_target"].astype(np.float64)
                smol = archive["smolvla_prediction"].astype(np.float64)
    assert target is not None and smol is not None
    time = np.arange(49) / 30.0
    figure, axes = plt.subplots(len(PHASES), len(steps), figsize=(18, 17), sharex=True)
    for row, (phase, sample_index) in enumerate(zip(PHASES, representative, strict=True)):
        for column, (step, prediction) in enumerate(zip(steps, predictions, strict=True)):
            axis = axes[row, column]
            target_velocity = np.diff(target[sample_index, :, :14], axis=0) * 30.0
            smol_velocity = np.diff(smol[sample_index, :, :14], axis=0) * 30.0
            act_velocity = np.diff(prediction[sample_index, :, :14], axis=0) * 30.0
            axis.plot(time, np.sqrt(np.mean(target_velocity**2, axis=1)), color="#2f855a", linewidth=1.6, label="Dataset")
            axis.plot(time, np.sqrt(np.mean(smol_velocity**2, axis=1)), color="#c53030", linewidth=1.0, alpha=0.72, label="SmolVLA")
            axis.plot(time, np.sqrt(np.mean(act_velocity**2, axis=1)), color="#2563eb", linewidth=1.3, label=f"ACT {step // 1000}k")
            axis.grid(alpha=0.2)
            if row == 0:
                result = summary["results"][column]
                axis.set_title(
                    f"ACT {step // 1000}k | phase {result['phase_score']['successful_qualitative_probes']}/54\n"
                    f"full RMSE {result['overall_accuracy']['full_available_chunk_rmse_rad']:.4f} rad"
                )
            if column == 0:
                axis.set_ylabel(phase.replace("_", " ") + "\narm qdot RMS")
            if row == len(PHASES) - 1:
                axis.set_xlabel("raw chunk time (s)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Bounded checkpoint selection: raw arm-speed traces (no smoothing)", fontsize=14)
    figure.tight_layout()
    output = EVAL / "checkpoint_raw_mechanics_comparison.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
