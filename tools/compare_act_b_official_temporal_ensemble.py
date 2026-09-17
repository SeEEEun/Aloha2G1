#!/usr/bin/env python3
"""Compare official ACT E0 queue execution with official temporal ensembling.

The comparison is teacher-forced and offline on frozen Dataset-B episode 24.
E0 uses the configured 50-action queue.  E1 uses LeRobot's unmodified online
ACTTemporalEnsembler with the reference coefficient 0.01 and n_action_steps=1.
No custom averaging, coefficient sweep, controller, physics, or robot is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DATASET_ROOT = ROOT / "datasets/doll_handoff_proposed_b_50"
OUTPUT = ROOT / "outputs/policy_b_act/temporal_ensemble"
LOW_MOTION_REFERENCE = (
    ROOT / "outputs/policy_execution_stability_review/low_motion_reference/low_motion_reference.json"
)
EPISODE = 24
FPS = 30.0
REFERENCE_COEFFICIENT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
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


def reversal_rates(velocity: np.ndarray, deadband: float) -> np.ndarray:
    duration = len(velocity) / FPS
    result = []
    for joint in range(velocity.shape[1]):
        signs = np.where(
            velocity[:, joint] > deadband,
            1,
            np.where(velocity[:, joint] < -deadband, -1, 0),
        )
        nonzero = signs[signs != 0]
        reversals = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
        result.append(reversals / duration)
    return np.asarray(result, dtype=np.float64)


def dynamics(q: np.ndarray, deadband: float) -> dict[str, float]:
    values = np.asarray(q, dtype=np.float64)
    step = np.diff(values, axis=0)
    qdot = step * FPS
    qddot = np.diff(qdot, axis=0) * FPS
    jerk = np.diff(qddot, axis=0) * FPS
    reversal = reversal_rates(qdot, deadband)
    time = np.arange(len(values), dtype=np.float64)
    design = np.column_stack((time, np.ones_like(time)))
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / FPS)
    power = np.abs(np.fft.rfft(residual, axis=0)) ** 2 / len(values) ** 2
    band = (frequencies >= 4.0) & (frequencies <= 10.0)
    return {
        "adjacent_step_mean_abs_rad": float(np.mean(np.abs(step))),
        "adjacent_step_max_abs_rad": float(np.max(np.abs(step))),
        "arm_direction_reversals_per_s_mean": float(np.mean(reversal[:14])),
        "dex3_direction_reversals_per_s_mean": float(np.mean(reversal[14:])),
        "qdot_rms_rad_s": rms(qdot),
        "qdot_max_abs_rad_s": float(np.max(np.abs(qdot))),
        "qddot_rms_rad_s2": rms(qddot),
        "qddot_max_abs_rad_s2": float(np.max(np.abs(qddot))),
        "jerk_rms_rad_s3": rms(jerk),
        "jerk_max_abs_rad_s3": float(np.max(np.abs(jerk))),
        "mean_4_10_hz_energy_rad2": float(np.mean(np.sum(power[band], axis=0))),
    }


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors

    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")

    e0 = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    if (
        e0.config.chunk_size != 50
        or e0.config.n_action_steps != 50
        or e0.config.temporal_ensemble_coeff is not None
    ):
        raise RuntimeError("ACT-E0 checkpoint is not the frozen 50-step non-ensemble config")
    e1_config = copy.deepcopy(e0.config)
    e1_config.n_action_steps = 1
    e1_config.temporal_ensemble_coeff = REFERENCE_COEFFICIENT
    e1 = ACTPolicy(e1_config)
    load_result = e1.load_state_dict(e0.state_dict(), strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"E1 strict weight load failed: {load_result}")
    device = next(e0.parameters()).device
    e1.to(device)
    e0.eval()
    e1.eval()
    e0.reset()
    e1.reset()
    if not isinstance(e1.temporal_ensembler, ACTTemporalEnsembler):
        raise RuntimeError("E1 is not using official ACTTemporalEnsembler")
    preprocessor, postprocessor = make_pre_post_processors(
        e0.config, pretrained_path=str(checkpoint)
    )

    dataset = LeRobotDataset(
        "local/doll_handoff_proposed_b_50", root=DATASET_ROOT, video_backend="torchcodec"
    )
    episode = dict(dataset.meta.episodes[EPISODE])
    start = int(episode["dataset_from_index"])
    stop = int(episode["dataset_to_index"])
    frame_count = stop - start
    targets = []
    e0_actions = []
    e1_actions = []
    for relative_frame, global_index in enumerate(range(start, stop)):
        sample = dataset[global_index]
        if (
            int(sample["episode_index"]) != EPISODE
            or int(sample["frame_index"]) != relative_frame
        ):
            raise RuntimeError("Dataset-B episode sequence changed")
        processed = preprocessor(
            {
                "observation.images.cam_high": sample["observation.images.cam_high"].clone(),
                "observation.state": sample["observation.state"].clone(),
            }
        )
        with torch.inference_mode():
            e0_normalized = e0.select_action(processed)
            e1_normalized = e1.select_action(processed)
            e0_physical = postprocessor(e0_normalized.clone())
            e1_physical = postprocessor(e1_normalized.clone())
        targets.append(sample["action"].detach().float().cpu().numpy())
        e0_actions.append(e0_physical[0].detach().float().cpu().numpy())
        e1_actions.append(e1_physical[0].detach().float().cpu().numpy())
        if relative_frame % 100 == 0:
            print(f"temporal ensemble offline frame {relative_frame}/{frame_count}", flush=True)

    target = np.stack(targets).astype(np.float32)
    e0_array = np.stack(e0_actions).astype(np.float32)
    e1_array = np.stack(e1_actions).astype(np.float32)
    expected = (frame_count, 28)
    if any(values.shape != expected or not np.isfinite(values).all() for values in (target, e0_array, e1_array)):
        raise RuntimeError("official execution comparison array malformed")

    deadband = float(
        json.loads(LOW_MOTION_REFERENCE.read_text(encoding="utf-8"))["selection"][
            "direction_reversal_deadband_rad_s"
        ]
    )
    metrics = {
        "dataset_target": dynamics(target, deadband),
        "ACT_E0_official_non_ensemble": dynamics(e0_array, deadband),
        "ACT_E1_official_temporal_ensemble": dynamics(e1_array, deadband),
    }
    for name, values in (("ACT_E0_official_non_ensemble", e0_array), ("ACT_E1_official_temporal_ensemble", e1_array)):
        metrics[name].update(
            {
                "action_rmse_rad": rms(values - target),
                "arm_action_rmse_rad": rms(values[:, :14] - target[:, :14]),
                "dex3_action_rmse_rad": rms(values[:, 14:] - target[:, 14:]),
            }
        )
    boundaries = np.arange(50, frame_count, 50)
    boundary_steps = {
        "ACT_E0_official_non_ensemble": np.abs(np.diff(e0_array, axis=0)[boundaries - 1]),
        "ACT_E1_official_temporal_ensemble": np.abs(np.diff(e1_array, axis=0)[boundaries - 1]),
    }
    boundary_summary = {
        name: {
            "boundary_count": len(values),
            "mean_abs_rad": float(np.mean(values)),
            "maximum_abs_rad": float(np.max(values)),
        }
        for name, values in boundary_steps.items()
    }

    arrays_path = OUTPUT / "official_e0_vs_e1_episode24.npz"
    atomic_npz(
        arrays_path,
        frame_index=np.arange(frame_count, dtype=np.int64),
        dataset_target=target,
        act_e0=e0_array,
        act_e1=e1_array,
        e0_chunk_boundary_frame=boundaries,
    )
    report = {
        "schema_version": "act_b_official_temporal_ensemble_v1",
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "episode_index": EPISODE,
        "frame_count": frame_count,
        "observation_semantics": "teacher-forced frozen Dataset-B RGB plus authoritative RETARGETED_G1_STATE_SURROGATE",
        "ACT_E0": {
            "implementation": "official ACTPolicy.select_action queue",
            "chunk_size": 50,
            "n_action_steps": 50,
            "temporal_ensemble_coeff": None,
            "policy_queries": int((frame_count + 49) // 50),
        },
        "ACT_E1": {
            "implementation": "official ACTPolicy.select_action plus ACTTemporalEnsembler.update",
            "chunk_size": 50,
            "n_action_steps": 1,
            "temporal_ensemble_coeff": REFERENCE_COEFFICIENT,
            "coefficient_source": "LeRobot ACT source states original ACT default/reference is 0.01",
            "policy_queries": frame_count,
        },
        "custom_averaging_used": False,
        "coefficient_sweep_performed": False,
        "raw_predictions_preserved_separately": str(
            ROOT / "outputs/policy_b_act/comparison/raw_target_vs_smolvla_vs_act_chunks.npz"
        ),
        "metrics": metrics,
        "e0_boundary_steps": boundary_summary,
        "arrays": str(arrays_path),
        "arrays_sha256": sha256_file(arrays_path),
    }
    atomic_json(OUTPUT / "official_e0_vs_e1_report.json", report)

    rows = []
    fieldnames = ["source"] + sorted({key for values in metrics.values() for key in values})
    for source, values in metrics.items():
        rows.append({"source": source, **values})
    with (OUTPUT / "official_e0_vs_e1_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    time = np.arange(frame_count) / FPS
    figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for joint in range(14):
        axes[0].plot(time, e0_array[:, joint], color="#c53030", alpha=0.22, linewidth=0.65)
        axes[0].plot(time, e1_array[:, joint], color="#2563eb", alpha=0.22, linewidth=0.65)
    for joint in range(14, 28):
        axes[1].plot(time, e0_array[:, joint], color="#c53030", alpha=0.22, linewidth=0.65)
        axes[1].plot(time, e1_array[:, joint], color="#2563eb", alpha=0.22, linewidth=0.65)
    for axis in axes:
        for boundary in boundaries / FPS:
            axis.axvline(boundary, color="black", alpha=0.08, linewidth=0.5)
        axis.grid(alpha=0.18)
    axes[0].set_ylabel("arm q (rad)")
    axes[1].set_ylabel("Dex3 q (rad)")
    axes[1].set_xlabel("teacher-forced episode time (s)")
    axes[0].set_title("red=ACT-E0 official 50-step queue | blue=ACT-E1 official temporal ensemble k=0.01")
    figure.tight_layout()
    figure.savefig(OUTPUT / "official_e0_vs_e1_trajectory.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
