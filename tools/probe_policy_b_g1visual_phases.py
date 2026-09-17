#!/usr/bin/env python3
"""Offline phase probe for a G1-visual Policy-B adaptation checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

import probe_policy_b_dataset_phases as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets/doll_handoff_proposed_b_g1visual_50"
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_g1visual/offline_phase_probe"
TRAJECTORIES = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories"
EPISODES = [0, 12, 13, 24, 36, 49]
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
ESTABLISHED_COSINE_GATE = 0.75
ESTABLISHED_DIRECTION_PROGRESS_GATE = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    dataset_root = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_hash_before = sha256_file(checkpoint / "model.safetensors")
    if model_hash_before != args.expected_model_sha256:
        raise RuntimeError(
            f"adapted checkpoint hash mismatch: {model_hash_before} != {args.expected_model_sha256}"
        )
    dataset = LeRobotDataset(
        repo_id="local/doll_handoff_proposed_b_g1visual_50",
        root=dataset_root,
        video_backend="torchcodec",
    )
    if len(dataset) != 34478 or dataset.num_episodes != 50:
        raise RuntimeError("G1-visual Dataset B identity mismatch")
    policy_joint_names = list(dataset.features["action"]["names"])
    if policy_joint_names != list(dataset.features["observation.state"]["names"]):
        raise RuntimeError("state/action named-joint order mismatch")
    if dataset.meta.camera_keys != ["observation.images.cam_high"]:
        raise RuntimeError(
            f"initial adaptation dataset must expose only cam_high, got {dataset.meta.camera_keys}"
        )

    episode_arrays: dict[int, dict[str, Any]] = {}
    hand_accumulator: dict[str, dict[str, list[np.ndarray]]] = {
        "left": defaultdict(list), "right": defaultdict(list)
    }
    for episode in range(dataset.num_episodes):
        meta = dataset.meta.episodes[episode]
        start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        table = dataset.hf_dataset[start:end]
        states = np.asarray(table["observation.state"], dtype=np.float32)
        actions = np.asarray(table["action"], dtype=np.float32)
        trajectory_path = TRAJECTORIES / f"episode_{episode:06d}.npz"
        trajectory = np.load(trajectory_path, allow_pickle=False)
        if not np.array_equal(states[0], actions[0]) or not np.array_equal(states[1:], actions[:-1]):
            raise RuntimeError(f"episode {episode}: lag-1 state contract mismatch")
        if not np.array_equal(actions, base.npz_actions_in_policy_order(trajectory, policy_joint_names)):
            raise RuntimeError(f"episode {episode}: frozen action mismatch")
        episode_arrays[episode] = {
            "start": start, "end": end, "states": states, "actions": actions,
            "trajectory": trajectory, "trajectory_path": trajectory_path,
        }
        for side, indices in (("left", base.GROUPS["left_dex3"]), ("right", base.GROUPS["right_dex3"])):
            labels = trajectory[f"{side}_hand_phase"]
            for label in np.unique(labels):
                hand_accumulator[side][str(label)].append(actions[labels == label][:, indices])
    hand_prototypes = {
        side: {label: np.concatenate(values).mean(axis=0) for label, values in phases.items()}
        for side, phases in hand_accumulator.items()
    }

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    checks = {
        "state_28": config["input_features"]["observation.state"]["shape"] == [28],
        "action_28": config["output_features"]["action"]["shape"] == [28],
        "chunk_50": int(config["chunk_size"]) == 50,
        "primary_cam_high_expected": "observation.images.cam_high" in config["input_features"],
    }
    if not all(checks.values()) or not torch.cuda.is_available():
        raise RuntimeError(f"checkpoint interface/CUDA gate failed: {checks}")
    set_seed(args.seed)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()

    selected: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    per_sample: list[dict[str, Any]] = []
    for episode in EPISODES:
        data = episode_arrays[episode]
        for phase, frame in base.select_phase_frames(data["trajectory"]).items():
            sample = dataset[data["start"] + frame]
            image = sample["observation.images.cam_high"]
            state = data["states"][frame]
            target = data["actions"][frame : frame + base.CHUNK_SIZE]
            if tuple(image.shape) != (3, 480, 640) or target.shape != (50, 28):
                raise RuntimeError("phase sample schema mismatch")
            if sample["task"] != base.TASK or not np.array_equal(sample["observation.state"].numpy(), state):
                raise RuntimeError("phase sample identity mismatch")
            seed = args.seed + len(selected)
            prediction = base.infer(
                policy, preprocessor, postprocessor, image, state, sample["task"], seed
            )
            reference_path = (
                output / "g1visual_references"
                / f"{phase}_episode_{episode:02d}_frame_{frame:04d}.png"
            )
            base.save_training_image(
                reference_path, image, f"G1 visual | {phase} | ep {episode:02d} | frame {frame}"
            )
            record = {
                "episode_index": episode,
                "frame_index": frame,
                "phase": phase,
                "phase_label": base.PHASE_LABELS[phase],
                "reference_image": str(reference_path),
                "state": "exact stored RETARGETED_G1_STATE_SURROGATE",
                "rgb": "Isaac-rendered G1 SOURCE_LIKE_CAM_HIGH training frame",
                "target": "authoritative frozen Dataset-B future action[0:50]",
                "seed": seed,
                "group_metrics": {
                    group: base.group_metrics(prediction, target, indices)
                    for group, indices in base.GROUPS.items()
                },
                "predicted_left_hand_phase_sequence": base.nearest_hand_phases(
                    prediction[:, base.GROUPS["left_dex3"]], hand_prototypes["left"]
                ),
                "predicted_right_hand_phase_sequence": base.nearest_hand_phases(
                    prediction[:, base.GROUPS["right_dex3"]], hand_prototypes["right"]
                ),
            }
            selected.append({
                "episode_index": episode, "frame_index": frame, "phase": phase,
                "phase_label": base.PHASE_LABELS[phase], "reference_image": str(reference_path),
            })
            predictions.append(prediction)
            targets.append(target.copy())
            per_sample.append(record)
            print(
                f"G1-visual probe {len(selected):02d}/54 ep={episode:02d} phase={phase}",
                flush=True,
            )

    prediction_array = np.stack(predictions).astype(np.float32)
    target_array = np.stack(targets).astype(np.float32)
    phase_summary: dict[str, Any] = {}
    phase_rows: list[dict[str, Any]] = []
    all_phases_present = True
    for phase, label in base.PHASE_LABELS.items():
        sample_indices = [index for index, row in enumerate(selected) if row["phase"] == phase]
        group_summary = {}
        for group, indices in base.GROUPS.items():
            rows = [base.group_metrics(prediction_array[i], target_array[i], indices) for i in sample_indices]
            group_summary[group] = {key: summarize(rows, key) for key in rows[0]}
        evidence = {}
        phase_pass = True
        for group in RELEVANT_GROUPS[phase]:
            indices = base.GROUPS[group]
            rows = [base.group_metrics(prediction_array[i], target_array[i], indices) for i in sample_indices]
            per_episode_pass = [
                row["trajectory_motion_cosine"] is not None
                and row["trajectory_motion_cosine"] >= ESTABLISHED_COSINE_GATE
                and row["target_direction_projection_progress"] is not None
                and row["target_direction_projection_progress"] >= ESTABLISHED_DIRECTION_PROGRESS_GATE
                for row in rows
            ]
            evidence[group] = {
                "episode_pass_count": int(sum(per_episode_pass)),
                "episode_total": len(per_episode_pass),
                "all_episodes_pass": all(per_episode_pass),
                "trajectory_motion_cosine": summarize(rows, "trajectory_motion_cosine"),
                "target_direction_projection_progress": summarize(
                    rows, "target_direction_projection_progress"
                ),
            }
            phase_pass &= all(per_episode_pass)
        all_phases_present &= phase_pass
        phase_summary[phase] = {
            "phase_label": label,
            "sample_count": len(sample_indices),
            "target_behavior_present": True,
            "policy_behavior_present": phase_pass,
            "relevant_group_evidence": evidence,
            "groups": group_summary,
        }
        phase_rows.append({
            "phase": label,
            "target_behavior_present": "YES",
            "policy_behavior_present": "YES" if phase_pass else "NO",
            "arm_rmse_rad": base.rms(
                prediction_array[sample_indices, :, :14] - target_array[sample_indices, :, :14]
            ),
            "dex3_rmse_rad": base.rms(
                prediction_array[sample_indices, :, 14:] - target_array[sample_indices, :, 14:]
            ),
            "relevant_groups": "+".join(RELEVANT_GROUPS[phase]),
        })

    np.savez_compressed(
        output / "probe_predictions.npz",
        policy_prediction=prediction_array,
        authoritative_target=target_array,
        episode_index=np.asarray([row["episode_index"] for row in selected]),
        frame_index=np.asarray([row["frame_index"] for row in selected]),
        phase=np.asarray([row["phase"] for row in selected]),
        joint_names=np.asarray(policy_joint_names),
    )
    atomic_json(output / "selected_frames.json", selected)
    atomic_json(output / "per_sample_metrics.json", per_sample)
    atomic_json(output / "phase_metrics.json", phase_summary)
    with (output / "phase_table.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    base.save_phase_contact_sheets(output, selected)
    base.save_phase_plots(output, selected, prediction_array, target_array)
    decision = {
        "status": "PASS" if all_phases_present else "FAIL",
        "G1_VISUAL_PHASE_LEARNING_CONFIRMED": all_phases_present,
        "checkpoint": str(checkpoint),
        "model_sha256": model_hash_before,
        "dataset": str(dataset_root),
        "samples": 54,
        "episodes": EPISODES,
        "phases": list(base.PHASE_LABELS),
        "evidence_rule": {
            "provenance": "unchanged from the established original Policy-B phase-learning decision",
            "per_episode_relevant_group_trajectory_motion_cosine_min": ESTABLISHED_COSINE_GATE,
            "per_episode_relevant_group_target_direction_projection_progress_min": ESTABLISHED_DIRECTION_PROGRESS_GATE,
            "required_episode_evidence": "6/6 for each relevant group in every phase",
        },
        "phase_results": phase_summary,
        "real_robot_invoked": False,
    }
    atomic_json(output / "phase_learning_decision.json", decision)
    if sha256_file(checkpoint / "model.safetensors") != model_hash_before:
        raise RuntimeError("checkpoint changed during read-only probe")
    print(json.dumps({
        "status": decision["status"],
        "G1_VISUAL_PHASE_LEARNING_CONFIRMED": all_phases_present,
        "output": str(output),
    }, indent=2))
    return 0 if all_phases_present else 2


if __name__ == "__main__":
    raise SystemExit(main())
