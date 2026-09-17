#!/usr/bin/env python3
"""Evaluate three saved ACT-B checkpoints on frozen phase and smoothness probes.

The script performs read-only checkpoint inference with the official LeRobot
ACT pre/post processors.  It intentionally does not select by training loss and
does not execute any action in simulation or on hardware.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DATASET_ROOT = ROOT / "datasets/doll_handoff_proposed_b_50"
RUN_ROOT = ROOT / "outputs/policy_b_act/train"
OUTPUT = ROOT / "outputs/policy_b_act/checkpoint_evaluation"
SELECTED_FRAMES = ROOT / "outputs/policy_b_offline_phase_probe/selected_frames.json"
SMOL_PROBES = ROOT / "outputs/policy_b_offline_phase_probe/probe_predictions.npz"
LOW_MOTION_REFERENCE = (
    ROOT / "outputs/policy_execution_stability_review/low_motion_reference/low_motion_reference.json"
)
CHECKPOINT_STEPS = (20_000, 60_000, 100_000)
CHUNK_SIZE = 50
ACTION_DIM = 28
FPS = 30.0

PHASE_LABELS = OrderedDict(
    [
        ("initial_left_approach", "initial / left approach"),
        ("immediately_before_left_grasp", "before left grasp"),
        ("left_grasp_left_owned", "left grasp / LEFT_OWNED"),
        ("left_transport", "left transport"),
        ("handoff_approach", "handoff approach"),
        ("dual_contact_transfer", "dual-contact transfer"),
        ("right_owned", "RIGHT_OWNED"),
        ("right_transport", "right transport"),
        ("release", "release"),
    ]
)
GROUPS = {
    "left_arm": np.arange(0, 7),
    "right_arm": np.arange(7, 14),
    "arms": np.arange(0, 14),
    "left_dex3": np.arange(14, 21),
    "right_dex3": np.arange(21, 28),
    "dex3": np.arange(14, 28),
    "all": np.arange(0, 28),
}
RELEVANT_GROUPS = {
    "initial_left_approach": ("left_arm",),
    "immediately_before_left_grasp": ("left_dex3",),
    "left_grasp_left_owned": ("left_dex3",),
    "left_transport": ("left_arm",),
    "handoff_approach": ("right_arm",),
    "dual_contact_transfer": ("left_dex3", "right_dex3"),
    "right_owned": ("left_dex3",),
    "right_transport": ("right_arm",),
    "release": ("right_dex3",),
}
REPRESENTATIVE_PHASES = (
    "initial_left_approach",
    "left_grasp_left_owned",
    "left_transport",
    "handoff_approach",
    "right_transport",
    "release",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, default=list(CHECKPOINT_STEPS))
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write empty checkpoint evaluation CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values), dtype=np.float64))))


def cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return float(np.dot(a, b) / denominator)


def motion_agreement(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    pred = prediction[:, indices].astype(np.float64)
    truth = target[:, indices].astype(np.float64)
    pred_motion = pred - pred[:1]
    target_motion = truth - truth[:1]
    target_energy = float(np.sum(target_motion**2))
    progress = float(np.sum(pred_motion * target_motion) / target_energy) if target_energy > 1e-12 else None
    trajectory_cosine = cosine(pred_motion[1:], target_motion[1:])
    success = bool(
        trajectory_cosine is not None
        and progress is not None
        and trajectory_cosine >= 0.75
        and progress >= 0.5
    )
    return {
        "trajectory_motion_cosine": trajectory_cosine,
        "target_direction_projection_progress": progress,
        "qualitative_motion_success": success,
    }


def reversal_rates(velocity: np.ndarray, deadband: float) -> np.ndarray:
    duration = velocity.shape[0] / FPS
    rates = []
    for joint in range(velocity.shape[1]):
        signs = np.where(
            velocity[:, joint] > deadband,
            1,
            np.where(velocity[:, joint] < -deadband, -1, 0),
        )
        nonzero = signs[signs != 0]
        count = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
        rates.append(count / duration)
    return np.asarray(rates, dtype=np.float64)


def spectral_energy(q: np.ndarray) -> dict[str, float]:
    values = np.asarray(q, dtype=np.float64)
    time_axis = np.arange(len(values), dtype=np.float64)
    design = np.column_stack((time_axis, np.ones_like(time_axis)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    detrended = values - design @ coefficients
    transform = np.fft.rfft(detrended, axis=0)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / FPS)
    power = np.abs(transform) ** 2 / (len(values) ** 2)
    band = (frequencies >= 4.0) & (frequencies <= 10.0)
    non_dc = frequencies > 0.0
    band_energy = np.sum(power[band], axis=0)
    total_energy = np.sum(power[non_dc], axis=0)
    return {
        "mean_4_10_hz_energy_rad2": float(np.mean(band_energy)),
        "maximum_joint_4_10_hz_energy_rad2": float(np.max(band_energy)),
        "mean_4_10_hz_fraction": float(np.mean(band_energy / np.maximum(total_energy, 1e-20))),
    }


def minimum_motion_window(q: np.ndarray, width: int = 30) -> tuple[int, np.ndarray]:
    candidates = []
    for start in range(0, len(q) - width + 1):
        window = q[start : start + width]
        arm_velocity = np.diff(window[:, :14], axis=0) * FPS
        candidates.append((rms(arm_velocity), start, window))
    _, start, window = min(candidates, key=lambda row: row[0])
    return start, window


def chunk_dynamics(q: np.ndarray, deadband: float) -> dict[str, Any]:
    values = np.asarray(q, dtype=np.float64)
    if values.shape != (CHUNK_SIZE, ACTION_DIM) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid action chunk {values.shape}")
    step = np.diff(values, axis=0)
    qdot = step * FPS
    qddot = np.diff(qdot, axis=0) * FPS
    jerk = np.diff(qddot, axis=0) * FPS
    reversals = reversal_rates(qdot, deadband)
    window_start, window = minimum_motion_window(values)
    window_time = np.arange(len(window), dtype=np.float64)
    design = np.column_stack((window_time, np.ones_like(window_time)))
    residual = window - design @ np.linalg.lstsq(design, window, rcond=None)[0]
    return {
        "adjacent_step_mean_abs_rad": float(np.mean(np.abs(step))),
        "adjacent_step_max_abs_rad": float(np.max(np.abs(step))),
        "arm_direction_reversals_per_s_mean": float(np.mean(reversals[:14])),
        "arm_direction_reversals_per_s_max": float(np.max(reversals[:14])),
        "dex3_direction_reversals_per_s_mean": float(np.mean(reversals[14:])),
        "dex3_direction_reversals_per_s_max": float(np.max(reversals[14:])),
        "qdot_rms_rad_s": rms(qdot),
        "qdot_max_abs_rad_s": float(np.max(np.abs(qdot))),
        "qddot_rms_rad_s2": rms(qddot),
        "qddot_max_abs_rad_s2": float(np.max(np.abs(qddot))),
        "jerk_rms_rad_s3": rms(jerk),
        "jerk_max_abs_rad_s3": float(np.max(np.abs(jerk))),
        "position_peak_to_peak_max_rad": float(np.max(np.ptp(values, axis=0))),
        "minimum_arm_motion_window_start": int(window_start),
        "low_motion_detrended_peak_to_peak_mean_rad": float(np.mean(np.ptp(residual, axis=0))),
        "low_motion_detrended_peak_to_peak_max_rad": float(np.max(np.ptp(residual, axis=0))),
        **spectral_energy(values),
    }


def aggregate_dynamics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [key for key in rows[0] if key != "minimum_arm_motion_window_start"]
    result = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "mean_across_chunks": float(np.mean(values)),
            "maximum_across_chunks": float(np.max(values)),
        }
    return result


def add_target_stationary_metrics(
    row: dict[str, Any], values: np.ndarray, stationary_joint_mask: np.ndarray
) -> None:
    selected = np.asarray(values, dtype=np.float64)[:, stationary_joint_mask]
    if selected.shape[1] == 0:
        raise RuntimeError("target-conditioned stationary joint selection is empty")
    time = np.arange(len(selected), dtype=np.float64)
    design = np.column_stack((time, np.ones_like(time)))
    residual = selected - design @ np.linalg.lstsq(design, selected, rcond=None)[0]
    p2p = np.ptp(residual, axis=0)
    row["target_stationary_joint_count"] = int(selected.shape[1])
    row["target_stationary_detrended_peak_to_peak_mean_rad"] = float(np.mean(p2p))
    row["target_stationary_detrended_peak_to_peak_max_rad"] = float(np.max(p2p))


def checkpoint_path(step: int) -> Path:
    return RUN_ROOT / "checkpoints" / f"{step:06d}" / "pretrained_model"


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    args = parse_args()
    steps = tuple(args.steps)
    if len(steps) > 3:
        raise ValueError("checkpoint selection is intentionally limited to at most three checkpoints")
    checkpoints = [checkpoint_path(step) for step in steps]
    missing = [str(path) for path in checkpoints if not (path / "model.safetensors").is_file()]
    if missing:
        raise FileNotFoundError(f"missing ACT-B checkpoints: {missing}")

    selections = json.loads(SELECTED_FRAMES.read_text(encoding="utf-8"))
    if len(selections) != 54:
        raise RuntimeError("frozen SmolVLA phase selection is not 54 frames")
    with np.load(SMOL_PROBES, allow_pickle=False) as archive:
        target = archive["authoritative_target"].astype(np.float32)
        smolvla = archive["policy_prediction_A"].astype(np.float32)
        old_episode = archive["episode_index"].astype(np.int64)
        old_frame = archive["frame_index"].astype(np.int64)
        old_phase = archive["phase"].astype(str)
    if target.shape != (54, CHUNK_SIZE, ACTION_DIM) or smolvla.shape != target.shape:
        raise RuntimeError("frozen phase prediction arrays have unexpected shape")

    dataset = LeRobotDataset(REPO_ID := "local/doll_handoff_proposed_b_50", root=DATASET_ROOT, video_backend="torchcodec")
    samples = []
    for index, selection in enumerate(selections):
        if (
            int(selection["episode_index"]) != int(old_episode[index])
            or int(selection["frame_index"]) != int(old_frame[index])
            or str(selection["phase"]) != str(old_phase[index])
        ):
            raise RuntimeError("frozen phase selection and prediction archive differ")
        sample = dataset[int(selection["global_dataset_index"])]
        if int(sample["episode_index"]) != int(selection["episode_index"]) or int(sample["frame_index"]) != int(selection["frame_index"]):
            raise RuntimeError("Dataset-B frame identity changed")
        if not np.array_equal(sample["observation.state"].numpy(), archive_state := np.asarray(sample["observation.state"], dtype=np.float32)):
            raise RuntimeError("unreachable state identity failure")
        if not np.array_equal(sample["action"].numpy(), target[index, 0]):
            raise RuntimeError("authoritative target first action changed")
        samples.append(
            {
                "observation.images.cam_high": sample["observation.images.cam_high"].clone(),
                "observation.state": torch.from_numpy(archive_state.copy()),
            }
        )

    low_motion_reference = json.loads(LOW_MOTION_REFERENCE.read_text(encoding="utf-8"))
    deadband = float(low_motion_reference["selection"]["direction_reversal_deadband_rad_s"])
    stationary_speed_cutoff = float(
        low_motion_reference["selection"]["arm_speed_rms_cutoff_rad_s"]
    )
    representative_indices = [
        next(
            index
            for index, row in enumerate(selections)
            if int(row["episode_index"]) == 24 and row["phase"] == phase
        )
        for phase in REPRESENTATIVE_PHASES
    ]
    stationary_masks = [
        np.sqrt(
            np.mean(
                np.square(np.diff(target[index].astype(np.float64), axis=0) * FPS),
                axis=0,
            )
        )
        <= stationary_speed_cutoff
        for index in representative_indices
    ]
    target_dynamics = [chunk_dynamics(target[index], deadband) for index in representative_indices]
    smol_dynamics = [chunk_dynamics(smolvla[index], deadband) for index in representative_indices]
    for local_index, sample_index in enumerate(representative_indices):
        add_target_stationary_metrics(
            target_dynamics[local_index], target[sample_index], stationary_masks[local_index]
        )
        add_target_stationary_metrics(
            smol_dynamics[local_index], smolvla[sample_index], stationary_masks[local_index]
        )

    results = []
    flat_rows = []
    for checkpoint_step, checkpoint in zip(steps, checkpoints, strict=True):
        policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=str(checkpoint),
        )
        policy.eval()
        if policy.config.chunk_size != CHUNK_SIZE or policy.config.action_feature.shape != (ACTION_DIM,):
            raise RuntimeError("ACT checkpoint logical action contract changed")
        predictions = []
        for sample in samples:
            processed = preprocessor({key: value.clone() for key, value in sample.items()})
            with torch.inference_mode():
                normalized = policy.predict_action_chunk(processed)
                physical = postprocessor(normalized)
            values = physical.detach().float().cpu().numpy()
            if values.shape != (1, CHUNK_SIZE, ACTION_DIM) or not np.isfinite(values).all():
                raise RuntimeError(f"checkpoint {checkpoint_step}: invalid inference {values.shape}")
            predictions.append(values[0])
        prediction = np.stack(predictions).astype(np.float32)
        prediction_path = OUTPUT / f"checkpoint_{checkpoint_step:06d}_phase_predictions.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            act_prediction=prediction,
            authoritative_target=target,
            smolvla_prediction=smolvla,
            episode_index=old_episode,
            frame_index=old_frame,
            phase=old_phase,
        )

        phase_results = {}
        qualitative_success_total = 0
        for phase, label in PHASE_LABELS.items():
            indices = [index for index, row in enumerate(selections) if row["phase"] == phase]
            phase_prediction = prediction[indices]
            phase_target = target[indices]
            successes = []
            agreement_rows = []
            for sample_index in indices:
                group_agreements = {
                    group: motion_agreement(prediction[sample_index], target[sample_index], GROUPS[group])
                    for group in RELEVANT_GROUPS[phase]
                }
                success = all(row["qualitative_motion_success"] for row in group_agreements.values())
                successes.append(success)
                agreement_rows.append(group_agreements)
            successful = int(sum(successes))
            qualitative_success_total += successful
            phase_results[phase] = {
                "phase_label": label,
                "sample_count": len(indices),
                "successful_qualitative_probes": successful,
                "qualitative_success_definition": "every relevant dynamic group has trajectory-motion cosine >=0.75 and target-direction projection progress >=0.5",
                "relevant_groups": list(RELEVANT_GROUPS[phase]),
                "first_action_rmse_rad": rms(phase_prediction[:, :1] - phase_target[:, :1]),
                "first_4_frame_rmse_rad": rms(phase_prediction[:, :4] - phase_target[:, :4]),
                "full_available_chunk_rmse_rad": rms(phase_prediction - phase_target),
                "arm_full_available_chunk_rmse_rad": rms(phase_prediction[:, :, :14] - phase_target[:, :, :14]),
                "dex3_full_available_chunk_rmse_rad": rms(phase_prediction[:, :, 14:] - phase_target[:, :, 14:]),
                "per_sample_relevant_group_agreement": agreement_rows,
            }
            flat_rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "phase": phase,
                    "phase_label": label,
                    "successful_qualitative_probes": successful,
                    "sample_count": len(indices),
                    "first_action_rmse_rad": phase_results[phase]["first_action_rmse_rad"],
                    "first_4_frame_rmse_rad": phase_results[phase]["first_4_frame_rmse_rad"],
                    "full_available_chunk_rmse_rad": phase_results[phase]["full_available_chunk_rmse_rad"],
                    "arm_full_available_chunk_rmse_rad": phase_results[phase]["arm_full_available_chunk_rmse_rad"],
                    "dex3_full_available_chunk_rmse_rad": phase_results[phase]["dex3_full_available_chunk_rmse_rad"],
                }
            )

        act_dynamics = [chunk_dynamics(prediction[index], deadband) for index in representative_indices]
        for local_index, sample_index in enumerate(representative_indices):
            add_target_stationary_metrics(
                act_dynamics[local_index], prediction[sample_index], stationary_masks[local_index]
            )
        result = {
            "checkpoint_step": checkpoint_step,
            "checkpoint": str(checkpoint),
            "model_sha256": sha256_file(checkpoint / "model.safetensors"),
            "checkpoint_reload": True,
            "strict_weight_reload": True,
            "policy_eval_mode": not policy.training,
            "prediction_shape": list(prediction.shape),
            "prediction_finite": bool(np.isfinite(prediction).all()),
            "overall_accuracy": {
                "first_action_rmse_rad": rms(prediction[:, :1] - target[:, :1]),
                "first_4_frame_rmse_rad": rms(prediction[:, :4] - target[:, :4]),
                "full_available_chunk_rmse_rad": rms(prediction - target),
                "arm_full_available_chunk_rmse_rad": rms(prediction[:, :, :14] - target[:, :, :14]),
                "dex3_full_available_chunk_rmse_rad": rms(prediction[:, :, 14:] - target[:, :, 14:]),
            },
            "phase_score": {
                "successful_qualitative_probes": qualitative_success_total,
                "total_probes": len(selections),
                "fraction": qualitative_success_total / len(selections),
                "phases_with_all_6_successful": sum(
                    phase_results[phase]["successful_qualitative_probes"] == 6 for phase in PHASE_LABELS
                ),
                "total_phases": len(PHASE_LABELS),
            },
            "phases": phase_results,
            "representative_raw_chunk_dynamics": {
                "phases": list(REPRESENTATIVE_PHASES),
                "episode_index": 24,
                "deadband_rad_s": deadband,
                "target_stationary_joint_speed_cutoff_rad_s": stationary_speed_cutoff,
                "dataset_target": target_dynamics,
                "smolvla_raw_fixed_seed": smol_dynamics,
                "act_raw": act_dynamics,
                "aggregates": {
                    "dataset_target": aggregate_dynamics(target_dynamics),
                    "smolvla_raw_fixed_seed": aggregate_dynamics(smol_dynamics),
                    "act_raw": aggregate_dynamics(act_dynamics),
                },
            },
            "prediction_artifact": str(prediction_path),
        }
        atomic_json(OUTPUT / f"checkpoint_{checkpoint_step:06d}_evaluation.json", result)
        results.append(result)
        del policy, preprocessor, postprocessor
        torch.cuda.empty_cache()
        print(
            f"checkpoint={checkpoint_step} phase={qualitative_success_total}/54 "
            f"rmse={result['overall_accuracy']['full_available_chunk_rmse_rad']:.6f} "
            f"act_arm_reversal={result['representative_raw_chunk_dynamics']['aggregates']['act_raw']['arm_direction_reversals_per_s_mean']['mean_across_chunks']:.3f}",
            flush=True,
        )

    summary = {
        "schema_version": "act_b_checkpoint_evaluation_v1",
        "status": "PASS",
        "evaluated_checkpoint_count": len(results),
        "evaluated_steps": list(steps),
        "selection_performed": False,
        "selection_note": "Use accuracy, semantic phase score, and raw mechanics together; training loss is not a selection field.",
        "results": results,
    }
    atomic_json(OUTPUT / "checkpoint_comparison.json", summary)
    write_csv(OUTPUT / "phase_metrics.csv", flat_rows)
    print(json.dumps({"status": "PASS", "output": str(OUTPUT), "steps": steps}, indent=2))


if __name__ == "__main__":
    main()
