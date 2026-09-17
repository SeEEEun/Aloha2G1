#!/usr/bin/env python3
"""Run ACT-B fixed-input and raw single-chunk offline diagnostics.

All outputs are unfiltered policy predictions.  This script does not use RTC,
crossfade, Ruckig, temporal ensembling, replanning, a controller, physics, or
hardware communication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_act_b_checkpoints import aggregate_dynamics, chunk_dynamics


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DATASET_ROOT = ROOT / "datasets/doll_handoff_proposed_b_50"
OUTPUT = ROOT / "outputs/policy_b_act/raw_diagnostics"
COMPARISON = ROOT / "outputs/policy_b_act/comparison"
FIXED_INPUTS = (
    ROOT
    / "outputs/policy_execution_stability_review/fixed_observation_capture"
    / "full_policy_b_diagnostic_rollout/fixed_inference_observations"
)
LOW_MOTION_REFERENCE = (
    ROOT / "outputs/policy_execution_stability_review/low_motion_reference/low_motion_reference.json"
)
RAW_LABELS = (
    "initial",
    "left_approach",
    "doll_plateau",
    "left_transport",
    "handoff_approach",
    "right_transport",
    "release",
)
PHASE_FOR_LABEL = {
    "left_approach": "initial_left_approach",
    "doll_plateau": "left_grasp_left_owned",
    "left_transport": "left_transport",
    "handoff_approach": "handoff_approach",
    "right_transport": "right_transport",
    "release": "release",
}
FIXED_CONDITIONS = (
    ("initial", 0),
    ("left_approach", 11),
    ("doll_region_plateau", 90),
)
REPEATS = 64
FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-predictions", type=Path, required=True)
    parser.add_argument("--smolvla-reference", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values), dtype=np.float64))))


def target_stationary_p2p(values: np.ndarray, stationary_joint_mask: np.ndarray) -> tuple[float, float]:
    """Detrended p-p on joints the Dataset target keeps slow for the full chunk."""
    selected = np.asarray(values, dtype=np.float64)[:, stationary_joint_mask]
    if selected.shape[1] == 0:
        raise RuntimeError("target-conditioned low-motion selection has no joints")
    time = np.arange(len(selected), dtype=np.float64)
    design = np.column_stack((time, np.ones_like(time)))
    residual = selected - design @ np.linalg.lstsq(design, selected, rcond=None)[0]
    p2p = np.ptp(residual, axis=0)
    return float(np.mean(p2p)), float(np.max(p2p))


def summarize_repeats(chunks: np.ndarray) -> dict[str, Any]:
    values = np.asarray(chunks, dtype=np.float64)
    if values.shape != (REPEATS, 50, 28):
        raise RuntimeError(f"unexpected repeat tensor {values.shape}")
    first_std = np.std(values[:, 0], axis=0)
    full_std = np.std(values, axis=0)
    reference = chunks[0:1]
    return {
        "exact_output_equality": bool(np.array_equal(chunks, np.broadcast_to(reference, chunks.shape))),
        "unique_output_byte_sequences": len({chunk.tobytes() for chunk in chunks}),
        "maximum_absolute_difference_from_first_rad": float(np.max(np.abs(values - values[:1]))),
        "first_action_standard_deviation": {
            "rms_across_joints_rad": rms(first_std),
            "maximum_joint_rad": float(np.max(first_std)),
            "mean_across_joints_rad": float(np.mean(first_std)),
        },
        "full_chunk_standard_deviation": {
            "rms_across_offsets_and_joints_rad": rms(full_std),
            "maximum_offset_joint_rad": float(np.max(full_std)),
            "mean_across_offsets_and_joints_rad": float(np.mean(full_std)),
        },
    }


def write_flat_metrics(
    path: Path,
    metrics: dict[str, dict[str, dict[str, Any]]],
) -> None:
    rows = []
    for label in RAW_LABELS:
        for source in ("dataset_target", "smolvla_raw_fixed_noise", "act_raw"):
            row: dict[str, Any] = {"condition": label, "source": source}
            row.update(metrics[label][source])
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_phase(
    label: str,
    target: np.ndarray,
    smol: np.ndarray,
    act: np.ndarray,
    output: Path,
) -> None:
    time = np.arange(50) / FPS
    colors = ("#2f855a", "#c53030", "#2563eb")
    sources = (("Dataset target", target), ("SmolVLA raw", smol), ("ACT raw", act))
    figure, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    for column, ((title, values), color) in enumerate(zip(sources, colors, strict=True)):
        for joint in range(14):
            axes[0, column].plot(time, values[:, joint], color=color, alpha=0.48, linewidth=0.9)
        for joint in range(14, 28):
            axes[1, column].plot(time, values[:, joint], color=color, alpha=0.48, linewidth=0.9)
        axes[0, column].set_title(title)
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[1, column].set_xlabel("chunk time (s)")
    axes[0, 0].set_ylabel("arm joint position (rad)")
    axes[1, 0].set_ylabel("Dex3 joint position (rad)")
    figure.suptitle(f"Raw 50-step chunks — {label.replace('_', ' ')} (no smoothing)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def plot_aggregate_bars(aggregates: dict[str, Any], output: Path) -> None:
    definitions = (
        ("arm_direction_reversals_per_s_mean", "Arm reversals/s"),
        ("dex3_direction_reversals_per_s_mean", "Dex3 reversals/s"),
        ("adjacent_step_max_abs_rad", "Max step (rad)"),
        ("qddot_rms_rad_s2", "qddot RMS"),
        ("jerk_rms_rad_s3", "jerk RMS"),
        ("target_stationary_detrended_peak_to_peak_max_rad", "Target-stationary p-p (rad)"),
    )
    sources = ("dataset_target", "smolvla_raw_fixed_noise", "act_raw")
    labels = ("Dataset", "SmolVLA", "ACT")
    colors = ("#2f855a", "#c53030", "#2563eb")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (key, title) in zip(axes.flat, definitions, strict=True):
        values = [aggregates[source][key]["mean_across_chunks"] for source in sources]
        axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Seven raw 50-step chunks: Dataset vs SmolVLA vs ACT")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    phase_predictions_path = args.phase_predictions.resolve()
    smol_reference_path = args.smolvla_reference.resolve()
    for path in (checkpoint / "model.safetensors", phase_predictions_path, smol_reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    if policy.config.chunk_size != 50 or policy.config.action_feature.shape != (28,):
        raise RuntimeError("selected ACT checkpoint contract changed")

    dropout_modules = [
        {"name": name, "class": module.__class__.__name__, "training": bool(module.training)}
        for name, module in policy.named_modules()
        if isinstance(
            module,
            (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d),
        )
    ]
    active_dropout = [row for row in dropout_modules if row["training"]]
    stochastic_modules_active = list(active_dropout)

    fixed_reports = {}
    fixed_chunks = []
    fixed_rgb_hashes = []
    fixed_state_hashes = []
    for label, inference_index in FIXED_CONDITIONS:
        metadata_path = FIXED_INPUTS / f"inference_{inference_index:04d}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rgb_path = Path(metadata["rgb_path"])
        state_path = Path(metadata["state_path"])
        if sha256_file(rgb_path) != metadata["rgb_sha256"]:
            raise RuntimeError(f"fixed RGB hash mismatch: {rgb_path}")
        if sha256_file(state_path) != metadata["state_sha256"]:
            raise RuntimeError(f"fixed state hash mismatch: {state_path}")
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"unreadable RGB {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with np.load(state_path, allow_pickle=False) as archive:
            state = archive["measured_state"].astype(np.float32)
        if rgb.shape != (480, 640, 3) or state.shape != (28,):
            raise RuntimeError("frozen input shape changed")
        image = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255.0)
        )
        processed = preprocessor(
            {
                "observation.images.cam_high": image,
                "observation.state": torch.from_numpy(state.copy()),
            }
        )
        repeats = []
        for _ in range(REPEATS):
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(processed)
                physical = postprocessor(normalized)
            chunk = physical[0].detach().float().cpu().numpy().astype(np.float32)
            if chunk.shape != (50, 28) or not np.isfinite(chunk).all():
                raise RuntimeError("ACT repeatability output malformed")
            repeats.append(chunk)
        repeats_array = np.stack(repeats)
        fixed_chunks.append(repeats_array)
        fixed_reports[label] = {
            **summarize_repeats(repeats_array),
            "inference_index": inference_index,
            "rgb_path": str(rgb_path),
            "rgb_sha256": metadata["rgb_sha256"],
            "state_path": str(state_path),
            "state_sha256": metadata["state_sha256"],
        }
        fixed_rgb_hashes.append(metadata["rgb_sha256"])
        fixed_state_hashes.append(metadata["state_sha256"])
        print(
            f"repeatability {label}: exact={fixed_reports[label]['exact_output_equality']} "
            f"maxdiff={fixed_reports[label]['maximum_absolute_difference_from_first_rad']:.3e}",
            flush=True,
        )

    fixed_chunks_array = np.stack(fixed_chunks)
    fixed_stochasticity = "NEGLIGIBLE" if (
        all(row["exact_output_equality"] for row in fixed_reports.values())
        and not stochastic_modules_active
    ) else "NON_NEGLIGIBLE"
    repeatability_report = {
        "schema_version": "act_b_fixed_input_repeatability_v1",
        "status": "PASS" if fixed_stochasticity == "NEGLIGIBLE" else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "model_eval_mode": not policy.training,
        "repeats_per_condition": REPEATS,
        "rgb_bytes_frozen": True,
        "state_bytes_frozen": True,
        "checkpoint_frozen": True,
        "normalization_frozen": True,
        "dropout_module_count": len(dropout_modules),
        "dropout_modules_active": active_dropout,
        "stochastic_modules_active": stochastic_modules_active,
        "act_inference_vae_latent": "deterministic zero latent; VAE encoder is training-only",
        "conditions": fixed_reports,
        "ACT_FIXED_INPUT_STOCHASTICITY": fixed_stochasticity,
    }
    atomic_npz(
        OUTPUT / "fixed_input_repeat_samples.npz",
        condition_labels=np.asarray([row[0] for row in FIXED_CONDITIONS]),
        raw_act_chunks=fixed_chunks_array,
        rgb_sha256=np.asarray(fixed_rgb_hashes),
        state_sha256=np.asarray(fixed_state_hashes),
    )
    atomic_json(OUTPUT / "fixed_input_repeatability.json", repeatability_report)

    with np.load(smol_reference_path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(str)
        frames = archive["frame_index"].astype(np.int64)
        targets = archive["authoritative_target"].astype(np.float32)
        smol = archive["smolvla_raw_fixed_noise"].astype(np.float32)
    if tuple(labels) != RAW_LABELS or targets.shape != (7, 50, 28) or smol.shape != targets.shape:
        raise RuntimeError("SmolVLA raw comparison reference contract changed")

    with np.load(phase_predictions_path, allow_pickle=False) as archive:
        phase_act = archive["act_prediction"].astype(np.float32)
        phase_target = archive["authoritative_target"].astype(np.float32)
        phase_episode = archive["episode_index"].astype(np.int64)
        phase_frame = archive["frame_index"].astype(np.int64)
        phase_names = archive["phase"].astype(str)
    act_chunks = [None] * len(RAW_LABELS)
    for output_index, label in enumerate(RAW_LABELS[1:], start=1):
        matches = np.flatnonzero(
            (phase_episode == 24) & (phase_names == PHASE_FOR_LABEL[label])
        )
        if len(matches) != 1 or int(phase_frame[matches[0]]) != int(frames[output_index]):
            raise RuntimeError("ACT phase prediction mapping changed")
        if not np.array_equal(phase_target[matches[0]], targets[output_index]):
            raise RuntimeError("Dataset target differs between comparison archives")
        act_chunks[output_index] = phase_act[matches[0]]

    dataset = LeRobotDataset(
        "local/doll_handoff_proposed_b_50", root=DATASET_ROOT, video_backend="torchcodec"
    )
    episode_start = int(dataset.meta.episodes[24]["dataset_from_index"])
    initial_sample = dataset[episode_start]
    if int(initial_sample["episode_index"]) != 24 or int(initial_sample["frame_index"]) != 0:
        raise RuntimeError("episode-24 initial Dataset-B sample changed")
    processed = preprocessor(
        {
            "observation.images.cam_high": initial_sample["observation.images.cam_high"].clone(),
            "observation.state": initial_sample["observation.state"].clone(),
        }
    )
    with torch.inference_mode():
        initial_normalized = policy.predict_action_chunk(processed)
        initial_physical = postprocessor(initial_normalized)
    act_chunks[0] = initial_physical[0].detach().float().cpu().numpy().astype(np.float32)
    act = np.stack(act_chunks)
    if act.shape != (7, 50, 28) or not np.isfinite(act).all():
        raise RuntimeError("raw ACT comparison chunks malformed")

    low_motion_reference = json.loads(LOW_MOTION_REFERENCE.read_text(encoding="utf-8"))
    deadband = float(low_motion_reference["selection"]["direction_reversal_deadband_rad_s"])
    stationary_speed_cutoff = float(
        low_motion_reference["selection"]["arm_speed_rms_cutoff_rad_s"]
    )
    per_condition: dict[str, dict[str, dict[str, Any]]] = {}
    source_rows: dict[str, list[dict[str, Any]]] = {
        "dataset_target": [],
        "smolvla_raw_fixed_noise": [],
        "act_raw": [],
    }
    for index, label in enumerate(RAW_LABELS):
        per_condition[label] = {}
        target_per_joint_qdot_rms = np.sqrt(
            np.mean(np.square(np.diff(targets[index], axis=0) * FPS), axis=0)
        )
        stationary_joint_mask = target_per_joint_qdot_rms <= stationary_speed_cutoff
        if not np.any(stationary_joint_mask):
            raise RuntimeError(f"no target-stationary joint available for {label}")
        for source, chunks in (
            ("dataset_target", targets),
            ("smolvla_raw_fixed_noise", smol),
            ("act_raw", act),
        ):
            row = chunk_dynamics(chunks[index], deadband)
            stationary_mean, stationary_max = target_stationary_p2p(
                chunks[index], stationary_joint_mask
            )
            row["target_stationary_joint_count"] = int(np.count_nonzero(stationary_joint_mask))
            row["target_stationary_detrended_peak_to_peak_mean_rad"] = stationary_mean
            row["target_stationary_detrended_peak_to_peak_max_rad"] = stationary_max
            per_condition[label][source] = row
            source_rows[source].append(row)
        plot_phase(
            label,
            targets[index],
            smol[index],
            act[index],
            OUTPUT / "trace_plots" / f"{index:02d}_{label}.png",
        )
    aggregates = {source: aggregate_dynamics(rows) for source, rows in source_rows.items()}
    ratios = {}
    for key in (
        "arm_direction_reversals_per_s_mean",
        "dex3_direction_reversals_per_s_mean",
        "adjacent_step_max_abs_rad",
        "qdot_rms_rad_s",
        "qddot_rms_rad_s2",
        "jerk_rms_rad_s3",
        "mean_4_10_hz_energy_rad2",
        "target_stationary_detrended_peak_to_peak_max_rad",
    ):
        act_value = aggregates["act_raw"][key]["mean_across_chunks"]
        smol_value = aggregates["smolvla_raw_fixed_noise"][key]["mean_across_chunks"]
        target_value = aggregates["dataset_target"][key]["mean_across_chunks"]
        ratios[key] = {
            "act_to_smolvla": act_value / max(smol_value, 1e-20),
            "act_to_dataset_target": act_value / max(target_value, 1e-20),
        }
    raw_report = {
        "schema_version": "act_b_raw_single_chunk_smoothness_v1",
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "raw_prediction_shape": list(act.shape),
        "fps": FPS,
        "chunk_duration_seconds": 50 / FPS,
        "episode_index": 24,
        "conditions": list(RAW_LABELS),
        "frame_indices": frames,
        "direction_reversal_deadband_rad_s": deadband,
        "target_stationary_joint_speed_cutoff_rad_s": stationary_speed_cutoff,
        "target_stationary_peak_to_peak_definition": (
            "For each condition, select joints whose Dataset-target qdot RMS over all 50 rows is at most "
            "the frozen low-motion cutoff; detrend each selected 50-row position trace, then report p-p. "
            "The identical target-derived joint mask is used for Dataset, SmolVLA, and ACT."
        ),
        "no_physics_smoothing": True,
        "no_execution_adapter": True,
        "no_rtc": True,
        "no_ruckig": True,
        "no_crossfade": True,
        "no_low_pass_filter": True,
        "no_temporal_ensemble": True,
        "no_replanning": True,
        "per_condition": per_condition,
        "aggregates": aggregates,
        "act_metric_ratios": ratios,
        "human_visible_smoothness_classification": "DEFERRED_TO_KINEMATIC_REPLAY_REVIEW",
    }
    atomic_npz(
        COMPARISON / "raw_target_vs_smolvla_vs_act_chunks.npz",
        labels=labels,
        episode_index=np.full(7, 24, dtype=np.int64),
        frame_index=frames,
        dataset_target=targets,
        smolvla_raw_fixed_noise=smol,
        act_raw=act,
    )
    atomic_json(OUTPUT / "raw_single_chunk_smoothness.json", raw_report)
    write_flat_metrics(OUTPUT / "raw_single_chunk_metrics.csv", per_condition)
    plot_aggregate_bars(aggregates, OUTPUT / "raw_single_chunk_metric_overview.png")
    print(
        json.dumps(
            {
                "status": "PASS",
                "repeatability": repeatability_report["ACT_FIXED_INPUT_STOCHASTICITY"],
                "raw_report": str(OUTPUT / "raw_single_chunk_smoothness.json"),
                "comparison_chunks": str(COMPARISON / "raw_target_vs_smolvla_vs_act_chunks.npz"),
                "ratios": ratios,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
