#!/usr/bin/env python3
"""Evaluate paired ACT-A40/ACT-B40 checkpoints on the frozen HELDOUT8 split.

This is the common evaluator frozen before any paper-model prediction.  It:

* uses the same source RGB at the same nine semantic frames for A and B;
* supplies each policy its own authoritative embodiment state;
* compares actions only to the corresponding method's held-out trajectory;
* evaluates both policy outputs against common source wrist/interaction geometry;
* applies one checkpoint-selection and phase-behavior rule to both methods.

It is offline only.  It does not instantiate Isaac, execute a controller, or
touch a real robot.
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
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/paper_core_ab/offline_heldout8"
TABLES = ROOT / "outputs/paper_core_ab/tables"
HELDOUT_MANIFEST = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EVALUATION_CONTRACT = ROOT / "outputs/paper_core_ab/offline_evaluation_contract.json"
SELECTION_CONTRACT = ROOT / "outputs/paper_core_ab/checkpoint_selection_rule.json"

METHODS = OrderedDict(
    [
        (
            "a",
            {
                "label": "ACT-A40",
                "dataset": ROOT / "datasets/doll_handoff_fair_a_heldout8",
                "repo_id": "local/doll_handoff_fair_a_heldout8",
                "train": ROOT / "outputs/paper_core_ab/act_a40/train",
            },
        ),
        (
            "b",
            {
                "label": "ACT-B40",
                "dataset": ROOT / "datasets/doll_handoff_proposed_b_heldout8",
                "repo_id": "local/doll_handoff_proposed_b_heldout8",
                "train": ROOT / "outputs/paper_core_ab/act_b40/train",
            },
        ),
    ]
)
CHECKPOINT_STEPS = (20_000, 60_000, 100_000)
CHUNK_SIZE = 50
ACTION_DIM = 28
FPS = 30.0

PROBE_LABELS = OrderedDict(
    [
        ("initial_approach", "initial / approach"),
        ("pre_grasp", "pre-grasp"),
        ("left_grasp_owned", "left grasp / owned"),
        ("left_transport", "left transport"),
        ("handoff_approach", "handoff approach"),
        ("dual_contact_transfer", "dual-contact transfer"),
        ("right_owned", "right owned"),
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
BEHAVIORS = OrderedDict(
    [
        ("LEFT_APPROACH", ("initial_approach", ("left_arm",))),
        ("LEFT_GRASP", ("pre_grasp", ("left_dex3",))),
        ("LEFT_TRANSPORT", ("left_transport", ("left_arm",))),
        ("RIGHT_HANDOFF_APPROACH", ("handoff_approach", ("right_arm",))),
        (
            "DUAL_HAND_CONFIGURATION",
            ("dual_contact_transfer", ("left_dex3", "right_dex3")),
        ),
        ("RIGHT_OWNED", ("right_owned", ("left_dex3",))),
        ("RIGHT_TRANSPORT", ("right_transport", ("right_arm",))),
        ("RELEASE", ("release", ("right_dex3",))),
    ]
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("RMS of empty array")
    return float(np.sqrt(np.mean(np.square(array))))


def stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    if not array.size or not np.isfinite(array).all():
        raise RuntimeError("statistics require a finite, non-empty array")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return float(np.dot(a, b) / denominator)


def detrended_peak_to_peak(values: np.ndarray) -> float:
    q = np.asarray(values, dtype=np.float64)
    time = np.arange(len(q), dtype=np.float64)
    design = np.column_stack((time, np.ones_like(time)))
    residual = q - design @ np.linalg.lstsq(design, q, rcond=None)[0]
    return float(np.max(np.ptp(residual, axis=0)))


def group_behavior(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction[:, indices], dtype=np.float64)
    truth = np.asarray(target[:, indices], dtype=np.float64)
    pred_motion = pred - pred[:1]
    target_motion = truth - truth[:1]
    target_rms_motion = rms(target_motion)
    target_energy = float(np.sum(target_motion**2))
    progress = (
        float(np.sum(pred_motion * target_motion) / target_energy)
        if target_energy > 1e-12
        else None
    )
    trajectory_cosine = cosine(pred_motion[1:], target_motion[1:])
    dynamic = target_rms_motion >= 0.01
    relevant_rmse = rms(pred - truth)
    predicted_p2p = detrended_peak_to_peak(pred)
    if dynamic:
        success = bool(
            trajectory_cosine is not None
            and progress is not None
            and trajectory_cosine >= 0.75
            and progress >= 0.5
        )
        rule = "DYNAMIC"
    else:
        success = relevant_rmse <= 0.05 and predicted_p2p <= 0.05
        rule = "STATIC"
    return {
        "rule": rule,
        "target_rms_motion_rad": target_rms_motion,
        "trajectory_motion_cosine": trajectory_cosine,
        "target_direction_projection_progress": progress,
        "relevant_group_rmse_rad": relevant_rmse,
        "predicted_detrended_peak_to_peak_rad": predicted_p2p,
        "success": bool(success),
    }


def reversal_rates(velocity: np.ndarray, deadband: float) -> np.ndarray:
    qdot = np.asarray(velocity, dtype=np.float64)
    duration_s = qdot.shape[0] / FPS
    rates = []
    for joint in range(qdot.shape[1]):
        signs = np.where(
            qdot[:, joint] > deadband,
            1,
            np.where(qdot[:, joint] < -deadband, -1, 0),
        )
        nonzero = signs[signs != 0]
        count = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
        rates.append(count / duration_s)
    return np.asarray(rates, dtype=np.float64)


def chunk_dynamics(values: np.ndarray, deadband: float) -> dict[str, float]:
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (CHUNK_SIZE, ACTION_DIM) or not np.isfinite(q).all():
        raise RuntimeError(f"invalid raw action chunk: {q.shape}")
    step = np.diff(q, axis=0)
    qdot = step * FPS
    qddot = np.diff(qdot, axis=0) * FPS
    jerk = np.diff(qddot, axis=0) * FPS
    reversals = reversal_rates(qdot, deadband)
    return {
        "arm_direction_reversals_per_s_mean": float(np.mean(reversals[:14])),
        "dex3_direction_reversals_per_s_mean": float(np.mean(reversals[14:])),
        "direction_reversals_per_s_mean": float(np.mean(reversals)),
        "max_adjacent_step_rad": float(np.max(np.abs(step))),
        "qdot_rms_rad_s": rms(qdot),
        "qdot_max_abs_rad_s": float(np.max(np.abs(qdot))),
        "qddot_rms_rad_s2": rms(qddot),
        "qddot_max_abs_rad_s2": float(np.max(np.abs(qddot))),
        "jerk_rms_rad_s3": rms(jerk),
        "jerk_max_abs_rad_s3": float(np.max(np.abs(jerk))),
        "low_motion_detrended_peak_to_peak_rad": detrended_peak_to_peak(q),
    }


def aggregate_dynamics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean_across_chunks": float(np.mean([row[key] for row in rows])),
            "maximum_across_chunks": float(np.max([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def parquet_arrays(dataset_root: Path) -> dict[str, np.ndarray]:
    data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no parquet data under {dataset_root}")
    table = pq.read_table(data_files)
    result = {
        "action": np.asarray(table["action"].to_pylist(), dtype=np.float32),
        "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
        "episode_index": np.asarray(table["episode_index"], dtype=np.int64),
        "frame_index": np.asarray(table["frame_index"], dtype=np.int64),
    }
    if result["action"].shape[1:] != (ACTION_DIM,) or result["state"].shape != result["action"].shape:
        raise RuntimeError(f"invalid dataset arrays under {dataset_root}")
    if not np.isfinite(result["action"]).all() or not np.isfinite(result["state"]).all():
        raise RuntimeError(f"non-finite held-out arrays under {dataset_root}")
    return result


def prepare_probes() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = load_json(HELDOUT_MANIFEST)
    if manifest["status"] != "PASS" or manifest["episode_count"] != 8:
        raise RuntimeError("HELDOUT8 manifest is not frozen and valid")
    datasets = {
        method: LeRobotDataset(
            row["repo_id"], root=row["dataset"], video_backend="torchcodec"
        )
        for method, row in METHODS.items()
    }
    arrays = {method: parquet_arrays(row["dataset"]) for method, row in METHODS.items()}
    episode_rows = {
        method: {int(row["episode_index"]): row for row in datasets[method].meta.episodes}
        for method in METHODS
    }
    feature_names = {
        method: load_json(METHODS[method]["dataset"] / "meta/info.json")["features"]["action"]["names"]
        for method in METHODS
    }
    if feature_names["a"] != feature_names["b"] or len(feature_names["a"]) != ACTION_DIM:
        raise RuntimeError("A/B held-out named action contracts differ")

    probes: list[dict[str, Any]] = []
    exact_rgb_count = 0
    for output_episode, entry in enumerate(manifest["entries"]):
        final_episode = int(entry["final_dataset_index"])
        source_audit = manifest["complete_source_phase_audit"][str(final_episode)]
        if not source_audit["complete"]:
            raise RuntimeError(f"incomplete source semantic episode {final_episode}")
        rows = {method: episode_rows[method][output_episode] for method in METHODS}
        if rows["a"]["length"] != rows["b"]["length"] or rows["a"]["length"] != entry["frames"]:
            raise RuntimeError(f"A/B episode length mismatch for source {final_episode}")
        length = int(entry["frames"])
        with np.load(entry["a_trajectory_path"], allow_pickle=False) as archive_a, np.load(
            entry["b_trajectory_path"], allow_pickle=False
        ) as archive_b:
            source_interaction = {}
            source_wrist = {}
            for side in ("left", "right"):
                interaction_a = archive_a[f"target_{side}_interaction_frame_position_world"].astype(np.float64)
                interaction_b = archive_b[f"source_{side}_interaction_frame_position_world"].astype(np.float64)
                if not np.array_equal(interaction_a, interaction_b):
                    raise RuntimeError(f"A/B source interaction targets differ ep{final_episode} {side}")
                source_interaction[side] = interaction_b
                source_wrist[side] = archive_a[f"target_{side}_wrist_position_model"].astype(np.float64)
            event_map = {
                str(name): int(frame)
                for name, frame in zip(archive_b["event_names"], archive_b["event_frames"], strict=True)
            }
            left_phase = archive_b["left_hand_phase"].astype(str)
            post_release_open = np.flatnonzero(
                (np.arange(length) > event_map["LEFT_RELEASE"]) & (left_phase == "OPEN")
            )
            if not len(post_release_open):
                raise RuntimeError(f"no post-release left-open frame ep{final_episode}")
            prototypes = {
                method: {
                    "right_open": arrays[method]["action"][int(rows[method]["dataset_from_index"]) + event_map["RIGHT_CLOSE_ONSET"] - 1, 21:28].copy(),
                    "right_hold": arrays[method]["action"][int(rows[method]["dataset_from_index"]) + event_map["RIGHT_STABLE_HOLD"], 21:28].copy(),
                    "left_hold": arrays[method]["action"][int(rows[method]["dataset_from_index"]) + event_map["LEFT_STABLE_HOLD"], 14:21].copy(),
                    "left_open": arrays[method]["action"][int(rows[method]["dataset_from_index"]) + int(post_release_open[0]), 14:21].copy(),
                }
                for method in METHODS
            }

            for phase in PROBE_LABELS:
                frame = int(source_audit["nine_probe_frames"][phase])
                valid = min(CHUNK_SIZE, length - frame)
                if valid <= 0:
                    raise RuntimeError(f"invalid probe frame ep{final_episode} {phase}")
                samples = {}
                targets = {}
                states = {}
                for method in METHODS:
                    global_index = int(rows[method]["dataset_from_index"]) + frame
                    sample = datasets[method][global_index]
                    if int(sample["episode_index"]) != output_episode or int(sample["frame_index"]) != frame:
                        raise RuntimeError(f"held-out readback identity failure {method} ep{final_episode} f{frame}")
                    start = global_index
                    target = arrays[method]["action"][start : start + valid].copy()
                    state = arrays[method]["state"][start].copy()
                    if not np.array_equal(sample["observation.state"].numpy(), state):
                        raise RuntimeError(f"state readback mismatch {method} ep{final_episode} f{frame}")
                    samples[method] = sample
                    targets[method] = target
                    states[method] = state
                if not torch.equal(
                    samples["a"]["observation.images.cam_high"],
                    samples["b"]["observation.images.cam_high"],
                ):
                    raise RuntimeError(f"decoded A/B RGB mismatch ep{final_episode} f{frame}")
                exact_rgb_count += 1
                probes.append(
                    {
                        "output_episode": output_episode,
                        "final_episode": final_episode,
                        "stable_episode_id": entry["stable_episode_id"],
                        "frame": frame,
                        "phase": phase,
                        "valid": valid,
                        "image": samples["a"]["observation.images.cam_high"].clone(),
                        "state": states,
                        "target": targets,
                        "source_wrist_model": {
                            side: source_wrist[side][frame : frame + valid].copy()
                            for side in ("left", "right")
                        },
                        "source_interaction_world": {
                            side: source_interaction[side][frame : frame + valid].copy()
                            for side in ("left", "right")
                        },
                        "prototypes": prototypes,
                    }
                )
    if len(probes) != 72 or exact_rgb_count != 72:
        raise RuntimeError(f"probe construction incomplete: {len(probes)} / {exact_rgb_count}")
    audit = {
        "probe_count": len(probes),
        "episode_count": 8,
        "nine_probes_per_episode": True,
        "decoded_rgb_tensor_exact_equal_count": exact_rgb_count,
        "decoded_rgb_tensor_exact_equal_total": len(probes),
        "a_b_feature_names_exact": feature_names["a"] == feature_names["b"],
        "feature_names": feature_names["a"],
        "method_specific_state_exact_readback": True,
        "source_interaction_targets_exact_between_archives": True,
        "valid_chunk_min": min(row["valid"] for row in probes),
        "valid_chunk_max": max(row["valid"] for row in probes),
    }
    return probes, audit


def infer_checkpoint(
    method: str, step: int, probes: list[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint = METHODS[method]["train"] / "checkpoints" / f"{step:06d}" / "pretrained_model"
    model_file = checkpoint / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    if policy.config.chunk_size != CHUNK_SIZE or policy.config.action_feature.shape != (ACTION_DIM,):
        raise RuntimeError(f"checkpoint contract mismatch {method} step {step}")
    if policy.config.temporal_ensemble_coeff is not None:
        raise RuntimeError("offline raw predictions require temporal ensembling disabled")
    policy.eval()
    predictions = []
    round_trip_errors = []
    normalization_reconstruction_errors = []
    for probe_index, probe in enumerate(probes):
        batch = {
            "observation.images.cam_high": probe["image"].clone(),
            "observation.state": torch.from_numpy(probe["state"][method].copy()),
        }
        processed = preprocessor(batch)
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(processed)
            physical = postprocessor(normalized)
        value = physical.detach().float().cpu().numpy()
        if value.shape != (1, CHUNK_SIZE, ACTION_DIM) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid prediction {method} step {step}: {value.shape}")
        predictions.append(value[0])
        if probe_index == 0:
            round_trip_processed = preprocessor(
                {
                    "observation.images.cam_high": probe["image"].clone(),
                    "observation.state": torch.from_numpy(probe["state"][method].copy()),
                    "action": torch.from_numpy(value[0].copy()),
                }
            )
            renormalized = round_trip_processed["action"]
            inverse = postprocessor(renormalized.clone())
            normalization_reconstruction_errors.append(
                float(
                    torch.max(
                        torch.abs(renormalized.detach().float().cpu() - normalized.detach().float().cpu())
                    )
                )
            )
            round_trip_errors.append(
                float(
                    np.max(
                        np.abs(
                            inverse.detach().float().cpu().numpy().reshape(CHUNK_SIZE, ACTION_DIM)
                            - value[0]
                        )
                    )
                )
            )
    del policy, preprocessor, postprocessor
    torch.cuda.empty_cache()
    return np.stack(predictions).astype(np.float32), {
        "authoritative_preprocessor_postprocessor": True,
        "normalized_reconstruction_max_abs": max(normalization_reconstruction_errors),
        "physical_round_trip_max_abs_rad": max(round_trip_errors),
        "finite": True,
    }


def action_accuracy(prediction: np.ndarray, probes: list[dict[str, Any]], method: str) -> dict[str, float]:
    differences = []
    first = []
    first4 = []
    arms = []
    dex3 = []
    for index, probe in enumerate(probes):
        valid = int(probe["valid"])
        delta = prediction[index, :valid].astype(np.float64) - probe["target"][method].astype(np.float64)
        differences.append(delta)
        first.append(delta[:1])
        first4.append(delta[: min(4, valid)])
        arms.append(delta[:, :14])
        dex3.append(delta[:, 14:])
    return {
        "first_action_rmse_rad": rms(np.concatenate(first)),
        "first_4_frame_rmse_rad": rms(np.concatenate(first4)),
        "full_valid_chunk_rmse_rad": rms(np.concatenate(differences)),
        "arm_full_valid_chunk_rmse_rad": rms(np.concatenate(arms)),
        "dex3_full_valid_chunk_rmse_rad": rms(np.concatenate(dex3)),
        "scored_joint_frames": int(sum(value.size for value in differences)),
    }


def phase_metrics(prediction: np.ndarray, probes: list[dict[str, Any]], method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    probe_metrics = {}
    for phase, label in PROBE_LABELS.items():
        indices = [index for index, row in enumerate(probes) if row["phase"] == phase]
        differences = []
        first = []
        first4 = []
        arms = []
        dex3 = []
        for index in indices:
            valid = probes[index]["valid"]
            delta = prediction[index, :valid] - probes[index]["target"][method]
            differences.append(delta)
            first.append(delta[:1])
            first4.append(delta[: min(4, valid)])
            arms.append(delta[:, :14])
            dex3.append(delta[:, 14:])
        probe_metrics[phase] = {
            "label": label,
            "probe_count": len(indices),
            "first_action_rmse_rad": rms(np.concatenate(first)),
            "first_4_frame_rmse_rad": rms(np.concatenate(first4)),
            "full_valid_chunk_rmse_rad": rms(np.concatenate(differences)),
            "arm_full_valid_chunk_rmse_rad": rms(np.concatenate(arms)),
            "dex3_full_valid_chunk_rmse_rad": rms(np.concatenate(dex3)),
        }

    behavior_results = {}
    total_success = 0
    for behavior, (phase, group_names) in BEHAVIORS.items():
        rows = []
        for index, probe in enumerate(probes):
            if probe["phase"] != phase:
                continue
            valid = int(probe["valid"])
            group_rows = {
                group: group_behavior(
                    prediction[index, :valid],
                    probe["target"][method],
                    GROUPS[group],
                )
                for group in group_names
            }
            success = all(value["success"] for value in group_rows.values())
            rows.append(
                {
                    "final_episode": probe["final_episode"],
                    "frame": probe["frame"],
                    "success": success,
                    "groups": group_rows,
                }
            )
            total_success += int(success)
        behavior_results[behavior] = {
            "probe": phase,
            "groups": list(group_names),
            "successful": int(sum(row["success"] for row in rows)),
            "total": len(rows),
            "per_episode": rows,
        }
    return probe_metrics, {
        "successful": total_success,
        "total": len(BEHAVIORS) * 8,
        "fraction": total_success / (len(BEHAVIORS) * 8),
        "behaviors": behavior_results,
    }


def evaluate_candidate(
    method: str,
    step: int,
    prediction: np.ndarray,
    probes: list[dict[str, Any]],
    deadband: float,
    normalization_audit: dict[str, Any],
) -> dict[str, Any]:
    accuracy = action_accuracy(prediction, probes, method)
    probe_metrics, phase_score = phase_metrics(prediction, probes, method)
    dynamics_rows = [chunk_dynamics(value, deadband) for value in prediction]
    checkpoint = METHODS[method]["train"] / "checkpoints" / f"{step:06d}" / "pretrained_model"
    return {
        "method": METHODS[method]["label"],
        "checkpoint_step": step,
        "checkpoint": str(checkpoint),
        "model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "strict_checkpoint_reload": True,
        "eval_mode": True,
        "prediction_shape": list(prediction.shape),
        "prediction_finite": bool(np.isfinite(prediction).all()),
        "normalization_round_trip": normalization_audit,
        "action_accuracy": accuracy,
        "per_probe_phase_accuracy": probe_metrics,
        "phase_score": phase_score,
        "raw_chunk_smoothness": aggregate_dynamics(dynamics_rows),
        "raw_chunk_smoothness_per_probe": [
            {
                "final_episode": probe["final_episode"],
                "frame": probe["frame"],
                "phase": probe["phase"],
                **row,
            }
            for probe, row in zip(probes, dynamics_rows, strict=True)
        ],
    }


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            -int(row["phase_score"]["successful"]),
            float(row["action_accuracy"]["full_valid_chunk_rmse_rad"]),
            float(row["raw_chunk_smoothness"]["jerk_rms_rad_s3"]["mean_across_chunks"]),
            int(row["checkpoint_step"]),
        ),
    )
    return {
        "selected_step": int(ranked[0]["checkpoint_step"]),
        "selected_checkpoint": ranked[0]["checkpoint"],
        "selected_model_sha256": ranked[0]["model_sha256"],
        "ranking_order_steps": [int(row["checkpoint_step"]) for row in ranked],
        "ranking_keys": [
            {
                "step": int(row["checkpoint_step"]),
                "phase_score": int(row["phase_score"]["successful"]),
                "full_valid_chunk_rmse_rad": float(row["action_accuracy"]["full_valid_chunk_rmse_rad"]),
                "raw_jerk_rms_rad_s3": float(row["raw_chunk_smoothness"]["jerk_rms_rad_s3"]["mean_across_chunks"]),
            }
            for row in ranked
        ],
    }


def predicted_geometry(method: str, prediction: np.ndarray, probes: list[dict[str, Any]], joint_names: list[str]) -> dict[str, Any]:
    try:
        from tools.doll_handoff_retargeting.common import load_common_config, load_scene
        from tools.doll_handoff_retargeting.models import G1Kinematics
    except ModuleNotFoundError:  # Called through a direct ``tools/<script>.py`` entrypoint.
        from doll_handoff_retargeting.common import load_common_config, load_scene
        from doll_handoff_retargeting.models import G1Kinematics

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(joint_names)}
    arm_indices = np.asarray([lookup[str(name)] for name in g1.arm_joint_names], dtype=np.int64)
    hand_indices = {
        side: np.asarray([lookup[name] for name in g1.hand_joint_names[side]], dtype=np.int64)
        for side in ("left", "right")
    }
    if len(set(np.concatenate((arm_indices, hand_indices["left"], hand_indices["right"])).tolist())) != ACTION_DIM:
        raise RuntimeError("named FK mapping is not a bijection over 28 joints")

    wrist_errors = []
    hand_errors = []
    relation_errors = []
    per_probe = []
    for probe_index, probe in enumerate(probes):
        valid = int(probe["valid"])
        pred = prediction[probe_index, :valid].astype(np.float64)
        predicted_wrist = {side: [] for side in ("left", "right")}
        predicted_hand = {side: [] for side in ("left", "right")}
        for q in pred:
            g1.assign(
                q[arm_indices],
                q[hand_indices["left"]],
                q[hand_indices["right"]],
            )
            for side in ("left", "right"):
                predicted_wrist[side].append(
                    g1.model_to_world_position(g1.wrist_pose(side)[:3, 3])
                )
                predicted_hand[side].append(
                    g1.model_to_world_position(g1.whole_hand_grasp_pose(side)[:3, 3])
                )
        predicted_wrist = {side: np.asarray(value) for side, value in predicted_wrist.items()}
        predicted_hand = {side: np.asarray(value) for side, value in predicted_hand.items()}
        wrist_target = {
            side: g1.model_to_world_position(probe["source_wrist_model"][side])
            for side in ("left", "right")
        }
        wrist = {
            side: np.linalg.norm(predicted_wrist[side] - wrist_target[side], axis=1)
            for side in ("left", "right")
        }
        whole_hand = {
            side: np.linalg.norm(
                predicted_hand[side] - probe["source_interaction_world"][side], axis=1
            )
            for side in ("left", "right")
        }
        relation = np.linalg.norm(
            (predicted_hand["right"] - predicted_hand["left"])
            - (
                probe["source_interaction_world"]["right"]
                - probe["source_interaction_world"]["left"]
            ),
            axis=1,
        )
        wrist_errors.extend((wrist["left"], wrist["right"]))
        hand_errors.extend((whole_hand["left"], whole_hand["right"]))
        relation_errors.append(relation)
        per_probe.append(
            {
                "final_episode": probe["final_episode"],
                "frame": probe["frame"],
                "phase": probe["phase"],
                "valid_frames": valid,
                "wrist_error_mm": stats(np.concatenate((wrist["left"], wrist["right"])), 1000.0),
                "whole_hand_error_mm": stats(
                    np.concatenate((whole_hand["left"], whole_hand["right"])), 1000.0
                ),
                "bimanual_relation_error_mm": stats(relation, 1000.0),
            }
        )
    return {
        "named_mapping_pass": True,
        "canonical_joint_names": joint_names,
        "g1_arm_indices_in_canonical": arm_indices,
        "g1_left_hand_indices_in_canonical": hand_indices["left"],
        "g1_right_hand_indices_in_canonical": hand_indices["right"],
        "source_wrist_trajectory_fidelity_error_mm": stats(np.concatenate(wrist_errors), 1000.0),
        "whole_hand_interaction_frame_error_mm": stats(np.concatenate(hand_errors), 1000.0),
        "bimanual_relation_error_mm": stats(np.concatenate(relation_errors), 1000.0),
        "per_probe": per_probe,
    }


def hand_configuration_and_ordering(method: str, prediction: np.ndarray, probes: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = {"left_grasp_owned", "dual_contact_transfer", "right_owned", "release"}
    configuration_rows = []
    ordering_rows = []
    for index, probe in enumerate(probes):
        valid = int(probe["valid"])
        pred = prediction[index, :valid].astype(np.float64)
        target = probe["target"][method].astype(np.float64)
        if probe["phase"] in relevant:
            configuration_rows.append(
                {
                    "final_episode": probe["final_episode"],
                    "phase": probe["phase"],
                    "left_dex3_rmse_rad": rms(pred[:, 14:21] - target[:, 14:21]),
                    "right_dex3_rmse_rad": rms(pred[:, 21:28] - target[:, 21:28]),
                    "all_dex3_rmse_rad": rms(pred[:, 14:28] - target[:, 14:28]),
                }
            )
        if probe["phase"] != "dual_contact_transfer":
            continue
        proto = probe["prototypes"][method]
        right_hold_distance = np.linalg.norm(pred[:, 21:28] - proto["right_hold"], axis=1)
        right_open_distance = np.linalg.norm(pred[:, 21:28] - proto["right_open"], axis=1)
        left_open_distance = np.linalg.norm(pred[:, 14:21] - proto["left_open"], axis=1)
        left_hold_distance = np.linalg.norm(pred[:, 14:21] - proto["left_hold"], axis=1)
        right_indices = np.flatnonzero(right_hold_distance < right_open_distance)
        left_indices = np.flatnonzero(left_open_distance < left_hold_distance)
        right_acquire = int(right_indices[0]) if len(right_indices) else None
        left_release = int(left_indices[0]) if len(left_indices) else None
        success = bool(
            right_acquire is not None
            and left_release is not None
            and right_acquire < left_release
        )
        ordering_rows.append(
            {
                "final_episode": probe["final_episode"],
                "dual_probe_frame": probe["frame"],
                "right_acquire_predicted_offset": right_acquire,
                "left_release_predicted_offset": left_release,
                "right_acquire_before_left_release": success,
            }
        )
    return {
        "configuration_consistency": {
            "mean_left_dex3_rmse_rad": float(np.mean([row["left_dex3_rmse_rad"] for row in configuration_rows])),
            "mean_right_dex3_rmse_rad": float(np.mean([row["right_dex3_rmse_rad"] for row in configuration_rows])),
            "mean_all_dex3_rmse_rad": float(np.mean([row["all_dex3_rmse_rad"] for row in configuration_rows])),
            "per_probe": configuration_rows,
        },
        "handoff_ordering": {
            "successful_episodes": int(sum(row["right_acquire_before_left_release"] for row in ordering_rows)),
            "total_episodes": len(ordering_rows),
            "per_episode": ordering_rows,
        },
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    lines.extend("| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="stop after checkpoint selection; finish named G1 FK with the Isaac environment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_json(EVALUATION_CONTRACT)
    selection_contract = load_json(SELECTION_CONTRACT)
    if contract["status"] != "FROZEN_BEFORE_ANY_PAPER_MODEL_PREDICTION":
        raise RuntimeError("offline evaluation contract is not frozen")
    if tuple(selection_contract["candidate_steps"]) != CHECKPOINT_STEPS:
        raise RuntimeError("checkpoint candidate contract changed")
    deadband = float(contract["raw_smoothness_contract"]["direction_reversal_deadband_rad_s"])
    probes, input_audit = prepare_probes()
    atomic_json(OUTPUT / "heldout_input_audit.json", input_audit)
    joint_names = input_audit["feature_names"]

    all_results: dict[str, Any] = {}
    selected_predictions: dict[str, np.ndarray] = {}
    for method in METHODS:
        candidates = []
        predictions_by_step = {}
        for step in CHECKPOINT_STEPS:
            prediction, normalization_audit = infer_checkpoint(method, step, probes)
            predictions_by_step[step] = prediction
            metadata = {
                "final_episode": np.asarray([row["final_episode"] for row in probes], dtype=np.int64),
                "output_episode": np.asarray([row["output_episode"] for row in probes], dtype=np.int64),
                "frame": np.asarray([row["frame"] for row in probes], dtype=np.int64),
                "valid_frames": np.asarray([row["valid"] for row in probes], dtype=np.int64),
                "phase": np.asarray([row["phase"] for row in probes]),
            }
            atomic_npz(
                OUTPUT / method / f"checkpoint_{step:06d}_raw_predictions.npz",
                prediction=prediction,
                **metadata,
            )
            candidate = evaluate_candidate(
                method, step, prediction, probes, deadband, normalization_audit
            )
            candidates.append(candidate)
            atomic_json(OUTPUT / method / f"checkpoint_{step:06d}_evaluation.json", candidate)
        selection = select_candidate(candidates)
        selected_step = int(selection["selected_step"])
        selected = next(row for row in candidates if row["checkpoint_step"] == selected_step)
        selected_prediction = predictions_by_step[selected_step]
        selected_predictions[method] = selected_prediction
        all_results[method] = {
            "method": METHODS[method]["label"],
            "checkpoint_selection": selection,
            "selected_checkpoint_evaluation": selected,
            "selected_prediction_archive": str(
                OUTPUT / method / f"checkpoint_{selected_step:06d}_raw_predictions.npz"
            ),
        }
        atomic_json(OUTPUT / method / "selected_checkpoint_inference_result.json", all_results[method])

    inference_result = {
        "schema_version": "paper_core_ab_heldout_act_inference_selection_v1",
        "status": "PASS",
        "evaluation_contract": str(EVALUATION_CONTRACT),
        "evaluation_contract_sha256": sha256_file(EVALUATION_CONTRACT),
        "checkpoint_selection_contract": str(SELECTION_CONTRACT),
        "checkpoint_selection_contract_sha256": sha256_file(SELECTION_CONTRACT),
        "input_audit": input_audit,
        "methods": all_results,
        "named_g1_fk_pending": bool(args.inference_only),
    }
    atomic_json(OUTPUT / "experiment2_inference_selection.json", inference_result)
    if args.inference_only:
        print(
            json.dumps(
                {
                    method: {
                        "selected_step": all_results[method]["checkpoint_selection"]["selected_step"],
                        "phase_score": all_results[method]["selected_checkpoint_evaluation"]["phase_score"],
                        "action_accuracy": all_results[method]["selected_checkpoint_evaluation"]["action_accuracy"],
                    }
                    for method in METHODS
                },
                indent=2,
                default=json_default,
            )
        )
        return

    for method in METHODS:
        geometry = predicted_geometry(method, selected_predictions[method], probes, joint_names)
        hands = hand_configuration_and_ordering(method, selected_predictions[method], probes)
        all_results[method]["common_source_geometry"] = geometry
        all_results[method]["hand_configuration_and_ordering"] = hands
        atomic_json(OUTPUT / method / "selected_checkpoint_result.json", all_results[method])

    table_rows = []
    for label, accessor, units in [
        ("first-action RMSE", ("action_accuracy", "first_action_rmse_rad"), "rad"),
        ("first-4 RMSE", ("action_accuracy", "first_4_frame_rmse_rad"), "rad"),
        ("full valid chunk RMSE", ("action_accuracy", "full_valid_chunk_rmse_rad"), "rad"),
        ("arm RMSE", ("action_accuracy", "arm_full_valid_chunk_rmse_rad"), "rad"),
        ("Dex3 RMSE", ("action_accuracy", "dex3_full_valid_chunk_rmse_rad"), "rad"),
    ]:
        table_rows.append(
            {
                "metric": label,
                "ACT-A40": f"{all_results['a']['selected_checkpoint_evaluation'][accessor[0]][accessor[1]]:.6f}",
                "ACT-B40": f"{all_results['b']['selected_checkpoint_evaluation'][accessor[0]][accessor[1]]:.6f}",
                "unit": units,
            }
        )
    for label, key in [
        ("predicted source wrist error mean", "source_wrist_trajectory_fidelity_error_mm"),
        ("predicted source wrist error p95", "source_wrist_trajectory_fidelity_error_mm"),
        ("predicted whole-hand error mean", "whole_hand_interaction_frame_error_mm"),
        ("predicted whole-hand error p95", "whole_hand_interaction_frame_error_mm"),
        ("predicted bimanual relation error mean", "bimanual_relation_error_mm"),
        ("predicted bimanual relation error p95", "bimanual_relation_error_mm"),
    ]:
        statistic = "p95" if label.endswith("p95") else "mean"
        table_rows.append(
            {
                "metric": label,
                "ACT-A40": f"{all_results['a']['common_source_geometry'][key][statistic]:.3f}",
                "ACT-B40": f"{all_results['b']['common_source_geometry'][key][statistic]:.3f}",
                "unit": "mm",
            }
        )
    for label, key, statistic, unit in [
        (
            "raw arm direction reversals mean",
            "arm_direction_reversals_per_s_mean",
            "mean_across_chunks",
            "reversals/s/joint",
        ),
        (
            "raw Dex3 direction reversals mean",
            "dex3_direction_reversals_per_s_mean",
            "mean_across_chunks",
            "reversals/s/joint",
        ),
        (
            "raw all-joint direction reversals mean",
            "direction_reversals_per_s_mean",
            "mean_across_chunks",
            "reversals/s/joint",
        ),
        (
            "raw maximum adjacent step",
            "max_adjacent_step_rad",
            "maximum_across_chunks",
            "rad",
        ),
        ("raw qdot RMS", "qdot_rms_rad_s", "mean_across_chunks", "rad/s"),
        ("raw qddot RMS", "qddot_rms_rad_s2", "mean_across_chunks", "rad/s^2"),
        ("raw jerk RMS", "jerk_rms_rad_s3", "mean_across_chunks", "rad/s^3"),
        (
            "raw low-motion detrended peak-to-peak",
            "low_motion_detrended_peak_to_peak_rad",
            "mean_across_chunks",
            "rad",
        ),
    ]:
        table_rows.append(
            {
                "metric": label,
                "ACT-A40": f"{all_results['a']['selected_checkpoint_evaluation']['raw_chunk_smoothness'][key][statistic]:.6f}",
                "ACT-B40": f"{all_results['b']['selected_checkpoint_evaluation']['raw_chunk_smoothness'][key][statistic]:.6f}",
                "unit": unit,
            }
        )
    table_rows.extend(
        [
            {
                "metric": "phase-behavior score",
                "ACT-A40": f"{all_results['a']['selected_checkpoint_evaluation']['phase_score']['successful']}/64",
                "ACT-B40": f"{all_results['b']['selected_checkpoint_evaluation']['phase_score']['successful']}/64",
                "unit": "probes",
            },
            {
                "metric": "right-acquire-before-left-release",
                "ACT-A40": f"{all_results['a']['hand_configuration_and_ordering']['handoff_ordering']['successful_episodes']}/8",
                "ACT-B40": f"{all_results['b']['hand_configuration_and_ordering']['handoff_ordering']['successful_episodes']}/8",
                "unit": "episodes",
            },
        ]
    )
    atomic_csv(TABLES / "table2_heldout_act_prediction.csv", table_rows)
    (TABLES / "table2_heldout_act_prediction.md").write_text(
        markdown_table(table_rows), encoding="utf-8"
    )
    result = {
        "schema_version": "paper_core_ab_heldout_act_prediction_v1",
        "status": "PASS",
        "evaluation_contract": str(EVALUATION_CONTRACT),
        "evaluation_contract_sha256": sha256_file(EVALUATION_CONTRACT),
        "checkpoint_selection_contract": str(SELECTION_CONTRACT),
        "checkpoint_selection_contract_sha256": sha256_file(SELECTION_CONTRACT),
        "input_audit": input_audit,
        "methods": all_results,
        "table2": table_rows,
        "claims": {
            "offline_heldout_only": True,
            "physical_manipulation_success_claimed": False,
            "real_robot_claimed": False,
        },
    }
    atomic_json(OUTPUT / "experiment2_result.json", result)
    atomic_json(TABLES / "table2_heldout_act_prediction.json", {"rows": table_rows})
    print(json.dumps({
        method: {
            "selected_step": all_results[method]["checkpoint_selection"]["selected_step"],
            "phase_score": all_results[method]["selected_checkpoint_evaluation"]["phase_score"],
            "action_accuracy": all_results[method]["selected_checkpoint_evaluation"]["action_accuracy"],
            "geometry": all_results[method]["common_source_geometry"],
        }
        for method in METHODS
    }, indent=2, default=json_default))


if __name__ == "__main__":
    main()
