#!/usr/bin/env python3
"""Read-only, phase-conditioned offline probe of the frozen Policy-B checkpoint.

This tool never writes into Dataset B or the checkpoint.  It selects frames by
the frozen Proposed-B semantic annotations, decodes the corresponding LeRobot
training observation, and compares one raw 50x28 Policy-B prediction with the
authoritative same-episode future action chunk.  A paired current-target-state
condition is evaluated with common random numbers to isolate the one-frame
state-semantics change.  One already frozen Isaac initial observation is used
only as a valid matched RGB/state distribution-shift reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DATASET_ROOT = ROOT / "datasets/doll_handoff_proposed_b_50"
TRAJECTORY_ROOT = (
    ROOT
    / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories"
)
SEMANTIC_MANIFEST = (
    ROOT
    / "outputs/doll_handoff_dataset_b_semantic_audit_2026-08-23/FINAL_DATASET_B_SEMANTIC_MANIFEST.json"
)
CHECKPOINT = (
    ROOT
    / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2/checkpoints/020000/pretrained_model"
)
ISAAC_STAGE0 = ROOT / "outputs/policy_b_isaac_validation/stage0_inference"
ISAAC_ROLLOUT = (
    ROOT
    / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout"
)
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_offline_phase_probe"
TASK = (
    "Pick up the doll with the left hand, handoff it to the right hand, "
    "and place it in the trash bin."
)
EPISODES = [0, 12, 13, 24, 36, 49]
CHUNK_SIZE = 50
EXPECTED_MODEL_SHA256 = (
    "bfe3e2aa6529967a12831a6f0bb91104b704835b2f43733072489b9ea2b68395"
)

GROUPS: OrderedDict[str, np.ndarray] = OrderedDict(
    [
        ("left_arm", np.arange(0, 7)),
        ("right_arm", np.arange(7, 14)),
        ("left_dex3", np.arange(14, 21)),
        ("right_dex3", np.arange(21, 28)),
    ]
)

PHASE_LABELS = OrderedDict(
    [
        ("initial_left_approach", "initial / left approach"),
        ("immediately_before_left_grasp", "immediately before left grasp"),
        ("left_grasp_left_owned", "left grasp / LEFT_OWNED"),
        ("left_transport", "left transport"),
        ("handoff_approach", "handoff approach"),
        ("dual_contact_transfer", "dual-contact / ownership transfer"),
        ("right_owned", "RIGHT_OWNED"),
        ("right_transport", "right transport toward bin"),
        ("release", "release"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def event_map(trajectory: Any) -> dict[str, int]:
    return {
        str(name): int(frame)
        for name, frame in zip(
            trajectory["event_names"].tolist(), trajectory["event_frames"].tolist()
        )
    }


def state_segment(values: np.ndarray, label: str) -> tuple[int, int]:
    indices = np.flatnonzero(values == label)
    if not len(indices):
        raise RuntimeError(f"semantic state {label!r} is absent")
    if np.any(np.diff(indices) != 1):
        raise RuntimeError(f"semantic state {label!r} is not contiguous")
    return int(indices[0]), int(indices[-1])


def select_phase_frames(trajectory: Any) -> dict[str, int]:
    events = event_map(trajectory)
    ownership = trajectory["ownership_state"]
    left_owned_start, left_owned_end = state_segment(ownership, "LEFT_OWNED")
    right_transport_start, right_transport_end = state_segment(
        ownership, "RIGHT_TRANSPORT"
    )
    left_transport_start = max(events["LEFT_STABLE_HOLD"], left_owned_start)
    left_transport_end = min(events["HANDOFF_APPROACH"] - 1, left_owned_end)
    if left_transport_end < left_transport_start:
        raise RuntimeError("left transport annotation interval is empty")
    frames = {
        # The complete 50-row target remains strictly before close onset.
        "initial_left_approach": events["LEFT_CLOSE_ONSET"] - CHUNK_SIZE,
        # Prefix rows 1..3 enter the annotated GRASP transition.
        "immediately_before_left_grasp": events["LEFT_CLOSE_ONSET"] - 1,
        "left_grasp_left_owned": events["LEFT_GRASP"],
        "left_transport": (left_transport_start + left_transport_end) // 2,
        "handoff_approach": events["HANDOFF_APPROACH"],
        "dual_contact_transfer": events["RIGHT_ACQUIRE"],
        "right_owned": events["LEFT_RELEASE"],
        # Keep most of the 50-row chunk inside transport when the interval permits.
        "right_transport": min(
            right_transport_start + 10,
            max(right_transport_start, right_transport_end - CHUNK_SIZE + 1),
        ),
        "release": events["RIGHT_FINAL_RELEASE"],
    }
    length = len(ownership)
    for phase, frame in frames.items():
        if frame < 0 or frame + CHUNK_SIZE > length:
            raise RuntimeError(
                f"episode length {length} cannot provide full chunk for {phase} at {frame}"
            )
    return frames


def npz_actions_in_policy_order(
    trajectory: Any, policy_joint_names: list[str]
) -> np.ndarray:
    source_names = trajectory["replay_joint_names"].tolist()
    source = trajectory["replay_named_joint_qpos"]
    lookup = {name: index for index, name in enumerate(source_names)}
    if set(lookup) != set(policy_joint_names):
        raise RuntimeError("trajectory and policy joint-name sets differ")
    return source[:, [lookup[name] for name in policy_joint_names]].astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    image: torch.Tensor,
    state: np.ndarray,
    task: str,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    if hasattr(policy, "reset"):
        policy.reset()
    raw = {
        "observation.images.cam_high": image.detach().clone(),
        "observation.state": torch.from_numpy(
            np.asarray(state, dtype=np.float32).copy()
        ),
        "task": task,
    }
    processed = preprocessor(raw)
    with torch.inference_mode():
        normalized = policy.predict_action_chunk(processed)
        prediction = postprocessor(normalized)
    result = prediction.detach().float().cpu().numpy()
    if result.shape != (1, CHUNK_SIZE, 28):
        raise RuntimeError(f"unexpected prediction shape {result.shape}")
    result = result[0]
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite policy prediction")
    return result


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return None
    return float(np.dot(a, b) / denominator)


def group_metrics(prediction: np.ndarray, target: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    pred = prediction[:, indices]
    truth = target[:, indices]
    pred_centered = pred - pred[:1]
    truth_centered = truth - truth[:1]
    target_energy = float(np.sum(np.square(truth_centered, dtype=np.float64)))
    projection_progress = (
        float(np.sum(pred_centered * truth_centered) / target_energy)
        if target_energy > 1e-12
        else None
    )
    return {
        "first_action_rmse_rad": rms(pred[:1] - truth[:1]),
        "prefix4_rmse_rad": rms(pred[:4] - truth[:4]),
        "full_chunk_rmse_rad": rms(pred - truth),
        "predicted_motion_magnitude_rad_rms": rms(pred_centered),
        "target_motion_magnitude_rad_rms": rms(truth_centered),
        "predicted_net_motion_rad_rms": rms(pred[-1] - pred[0]),
        "target_net_motion_rad_rms": rms(truth[-1] - truth[0]),
        "trajectory_motion_cosine": cosine(pred_centered[1:], truth_centered[1:]),
        "net_motion_cosine": cosine(pred[-1] - pred[0], truth[-1] - truth[0]),
        "target_direction_projection_progress": projection_progress,
    }


def nearest_hand_phases(
    values: np.ndarray, prototypes: dict[str, np.ndarray]
) -> list[str]:
    labels = sorted(prototypes)
    centers = np.stack([prototypes[label] for label in labels])
    distance = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
    return [labels[index] for index in np.argmin(distance, axis=1)]


def summarize_numbers(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def save_training_image(path: Path, image: torch.Tensor, label: str) -> None:
    rgb = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    rendered = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    draw.rectangle((0, 0, rendered.width, 24), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 240, 64))
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path)


def save_phase_contact_sheets(output: Path, selected: list[dict[str, Any]]) -> None:
    reference_dir = output / "source_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for phase in PHASE_LABELS:
        phase_rows = [row for row in selected if row["phase"] == phase]
        images = [Image.open(row["reference_image"]).convert("RGB") for row in phase_rows]
        thumb_size = (320, 240)
        sheet = Image.new("RGB", (thumb_size[0] * 3, thumb_size[1] * 2), "white")
        for index, frame in enumerate(images):
            frame.thumbnail(thumb_size)
            x = (index % 3) * thumb_size[0]
            y = (index // 3) * thumb_size[1]
            sheet.paste(frame, (x, y))
        sheet.save(reference_dir / f"{phase}_contact_sheet.png")


def save_phase_plots(
    output: Path,
    selected: list[dict[str, Any]],
    predictions_a: np.ndarray,
    targets: np.ndarray,
    representative_episode: int = 24,
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.arange(CHUNK_SIZE) / 30.0
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for phase, title in PHASE_LABELS.items():
        sample_indices = [
            index for index, row in enumerate(selected) if row["phase"] == phase
        ]
        fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
        for axis, (group, indices) in zip(axes.flat, GROUPS.items()):
            pred = predictions_a[sample_indices][:, :, indices]
            truth = targets[sample_indices][:, :, indices]
            pred_motion = np.sqrt(np.mean((pred - pred[:, :1]) ** 2, axis=2))
            truth_motion = np.sqrt(np.mean((truth - truth[:, :1]) ** 2, axis=2))
            axis.plot(offsets, truth_motion.mean(axis=0), color="black", label="target mean")
            axis.fill_between(
                offsets,
                truth_motion.mean(axis=0) - truth_motion.std(axis=0),
                truth_motion.mean(axis=0) + truth_motion.std(axis=0),
                color="black",
                alpha=0.12,
            )
            axis.plot(offsets, pred_motion.mean(axis=0), color="tab:orange", label="policy mean")
            axis.fill_between(
                offsets,
                pred_motion.mean(axis=0) - pred_motion.std(axis=0),
                pred_motion.mean(axis=0) + pred_motion.std(axis=0),
                color="tab:orange",
                alpha=0.16,
            )
            axis.set_title(group)
            axis.set_ylabel("RMS displacement from row 0 (rad)")
            axis.grid(alpha=0.25)
        axes[1, 0].set_xlabel("future offset (s)")
        axes[1, 1].set_xlabel("future offset (s)")
        axes[0, 0].legend()
        fig.suptitle(f"{title}: phase-conditioned motion across {len(sample_indices)} episodes")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{phase}_aggregate_motion.png", dpi=170)
        plt.close(fig)

        representative = next(
            index
            for index in sample_indices
            if selected[index]["episode_index"] == representative_episode
        )
        fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)
        for axis, (group, indices) in zip(axes, GROUPS.items()):
            for local, joint_index in enumerate(indices):
                axis.plot(
                    offsets,
                    targets[representative, :, joint_index],
                    color=colors[local],
                    linewidth=1.5,
                )
                axis.plot(
                    offsets,
                    predictions_a[representative, :, joint_index],
                    color=colors[local],
                    linestyle="--",
                    linewidth=1.0,
                )
            axis.set_title(f"{group}: target solid / Policy B dashed")
            axis.set_ylabel("absolute q (rad)")
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("future offset (s)")
        fig.suptitle(
            f"{title}: episode {representative_episode:02d}, frame "
            f"{selected[representative]['frame_index']}"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"{phase}_representative_ep024.png", dpi=170)
        plt.close(fig)


def save_summary_plots(output: Path, phase_summary: dict[str, Any]) -> None:
    plot_dir = output / "plots"
    phases = list(PHASE_LABELS)
    labels = [PHASE_LABELS[phase] for phase in phases]
    arm_rmse = [phase_summary[p]["combined_arm_full_chunk_rmse_rad"]["mean"] for p in phases]
    dex_rmse = [phase_summary[p]["combined_dex3_full_chunk_rmse_rad"]["mean"] for p in phases]
    x = np.arange(len(phases))
    fig, axis = plt.subplots(figsize=(15, 6))
    axis.bar(x - 0.2, arm_rmse, 0.4, label="arms 14D")
    axis.bar(x + 0.2, dex_rmse, 0.4, label="Dex3 14D")
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylabel("full-chunk RMSE (rad)")
    axis.set_title("In-distribution Policy-B phase probe")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "phase_rmse_summary.png", dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("refusing to write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_hash_before = sha256_file(CHECKPOINT / "model.safetensors")
    if model_hash_before != EXPECTED_MODEL_SHA256:
        raise RuntimeError("frozen checkpoint model hash mismatch")
    semantic_hash_before = sha256_file(SEMANTIC_MANIFEST)
    dataset_info_hash_before = sha256_file(DATASET_ROOT / "meta/info.json")
    semantic_manifest = json.loads(SEMANTIC_MANIFEST.read_text(encoding="utf-8"))

    dataset = LeRobotDataset(
        repo_id="local/doll_handoff_proposed_b_50_semantic_v2",
        root=DATASET_ROOT,
        video_backend="torchcodec",
    )
    if len(dataset) != 34478 or dataset.num_episodes != 50:
        raise RuntimeError("Dataset-B identity mismatch")
    policy_joint_names = list(dataset.features["action"]["names"])
    if policy_joint_names != list(dataset.features["observation.state"]["names"]):
        raise RuntimeError("Dataset-B state/action named-joint order mismatch")

    # Load all per-episode state/actions once.  No dataset file is mutated.
    episode_arrays: dict[int, dict[str, Any]] = {}
    hand_accumulator: dict[str, dict[str, list[np.ndarray]]] = {
        "left": defaultdict(list),
        "right": defaultdict(list),
    }
    phase_annotation_counts: dict[str, dict[str, int]] = {
        "left": defaultdict(int),
        "right": defaultdict(int),
    }
    for episode_index in range(dataset.num_episodes):
        meta = dataset.meta.episodes[episode_index]
        start = int(meta["dataset_from_index"])
        end = int(meta["dataset_to_index"])
        table = dataset.hf_dataset[start:end]
        states = np.asarray(table["observation.state"], dtype=np.float32)
        actions = np.asarray(table["action"], dtype=np.float32)
        trajectory_path = TRAJECTORY_ROOT / f"episode_{episode_index:06d}.npz"
        trajectory = np.load(trajectory_path, allow_pickle=False)
        if len(actions) != len(trajectory["timestamp"]):
            raise RuntimeError(f"episode {episode_index}: trajectory length mismatch")
        if not np.array_equal(states[0], actions[0]) or not np.array_equal(
            states[1:], actions[:-1]
        ):
            raise RuntimeError(f"episode {episode_index}: lag-1 state contract mismatch")
        action_from_trajectory = npz_actions_in_policy_order(
            trajectory, policy_joint_names
        )
        if not np.array_equal(actions, action_from_trajectory):
            raise RuntimeError(f"episode {episode_index}: frozen action mismatch")
        episode_arrays[episode_index] = {
            "start": start,
            "end": end,
            "states": states,
            "actions": actions,
            "trajectory": trajectory,
            "trajectory_path": trajectory_path,
        }
        for side, indices in (("left", GROUPS["left_dex3"]), ("right", GROUPS["right_dex3"])):
            labels = trajectory[f"{side}_hand_phase"]
            for label in np.unique(labels):
                mask = labels == label
                hand_accumulator[side][str(label)].append(actions[mask][:, indices])
                phase_annotation_counts[side][str(label)] += int(np.sum(mask))

    hand_prototypes = {
        side: {
            label: np.concatenate(values, axis=0).mean(axis=0)
            for label, values in accumulator.items()
        }
        for side, accumulator in hand_accumulator.items()
    }

    config = json.loads((CHECKPOINT / "config.json").read_text(encoding="utf-8"))
    if config["input_features"]["observation.state"]["shape"] != [28]:
        raise RuntimeError("checkpoint state interface is not 28D")
    if config["output_features"]["action"]["shape"] != [28]:
        raise RuntimeError("checkpoint action interface is not 28D")
    if int(config["chunk_size"]) != CHUNK_SIZE:
        raise RuntimeError("checkpoint chunk size is not 50")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Policy-B checkpoint")
    set_seed(args.seed)
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(CHECKPOINT)
    )
    policy.eval()

    selected: list[dict[str, Any]] = []
    prediction_a: list[np.ndarray] = []
    prediction_b: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    state_a_values: list[np.ndarray] = []
    state_b_values: list[np.ndarray] = []
    metrics_rows: list[dict[str, Any]] = []
    detailed_metrics: list[dict[str, Any]] = []

    sample_index = 0
    for episode_index in EPISODES:
        episode = episode_arrays[episode_index]
        trajectory = episode["trajectory"]
        frames = select_phase_frames(trajectory)
        for phase, frame_index in frames.items():
            global_index = episode["start"] + frame_index
            sample = dataset[global_index]
            if int(sample["episode_index"]) != episode_index or int(sample["frame_index"]) != frame_index:
                raise RuntimeError("LeRobot frame identity mismatch")
            if sample["task"] != TASK:
                raise RuntimeError("task instruction mismatch")
            image = sample["observation.images.cam_high"]
            if image.shape != (3, 480, 640) or image.dtype != torch.float32:
                raise RuntimeError("training RGB tensor schema mismatch")
            if float(image.min()) < 0.0 or float(image.max()) > 1.0:
                raise RuntimeError("training RGB tensor range mismatch")
            state_a = episode["states"][frame_index]
            state_b = episode["actions"][frame_index]
            target = episode["actions"][frame_index : frame_index + CHUNK_SIZE]
            if target.shape != (CHUNK_SIZE, 28):
                raise RuntimeError("selected target chunk is not fully available")
            if not np.array_equal(sample["observation.state"].numpy(), state_a):
                raise RuntimeError("decoded sample state differs from parquet state")
            if not np.array_equal(sample["action"].numpy(), target[0]):
                raise RuntimeError("decoded sample action differs from target row 0")

            seed = args.seed + sample_index
            pred_a = infer(
                policy,
                preprocessor,
                postprocessor,
                image,
                state_a,
                sample["task"],
                seed,
            )
            pred_b = infer(
                policy,
                preprocessor,
                postprocessor,
                image,
                state_b,
                sample["task"],
                seed,
            )
            reference_path = (
                output
                / "source_references"
                / f"{phase}_episode_{episode_index:02d}_frame_{frame_index:04d}.png"
            )
            save_training_image(
                reference_path,
                image,
                f"{phase} | ep {episode_index:02d} | frame {frame_index}",
            )
            annotation_slice = slice(frame_index, frame_index + CHUNK_SIZE)
            selection = {
                "sample_index": sample_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "global_dataset_index": global_index,
                "phase": phase,
                "phase_label": PHASE_LABELS[phase],
                "timestamp_seconds": frame_index / 30.0,
                "source_raw_episode": str(trajectory["source_directory_name"]),
                "trajectory_path": str(episode["trajectory_path"]),
                "ownership_state_at_frame": str(trajectory["ownership_state"][frame_index]),
                "left_hand_phase_at_frame": str(trajectory["left_hand_phase"][frame_index]),
                "right_hand_phase_at_frame": str(trajectory["right_hand_phase"][frame_index]),
                "event_frames": event_map(trajectory),
                "target_chunk_ownership_states": trajectory["ownership_state"][annotation_slice].tolist(),
                "target_chunk_left_hand_phases": trajectory["left_hand_phase"][annotation_slice].tolist(),
                "target_chunk_right_hand_phases": trajectory["right_hand_phase"][annotation_slice].tolist(),
                "reference_image": str(reference_path),
                "condition_a": "TRAIN_RGB + exact stored RETARGETED_G1_STATE_SURROGATE",
                "condition_b": "TRAIN_RGB + same-row retargeted q_target[t] current-style state (not measured)",
                "paired_common_random_seed": seed,
                "state_a_equals_dataset_observation_state": True,
                "state_b_is_measured_g1": False,
                "state_a_to_b_rmse_rad": rms(state_a - state_b),
            }
            selected.append(selection)
            prediction_a.append(pred_a)
            prediction_b.append(pred_b)
            target_chunks.append(target.copy())
            state_a_values.append(state_a.copy())
            state_b_values.append(state_b.copy())

            sample_detail: dict[str, Any] = selection.copy()
            sample_detail["conditions"] = {}
            flat_row: dict[str, Any] = {
                "sample_index": sample_index,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "phase": phase,
                "ownership_state": selection["ownership_state_at_frame"],
                "left_hand_phase": selection["left_hand_phase_at_frame"],
                "right_hand_phase": selection["right_hand_phase_at_frame"],
                "state_a_to_b_rmse_rad": selection["state_a_to_b_rmse_rad"],
            }
            for condition, prediction in (("A", pred_a), ("B", pred_b)):
                condition_metrics: dict[str, Any] = {}
                for group, indices in GROUPS.items():
                    values = group_metrics(prediction, target, indices)
                    condition_metrics[group] = values
                    for key, value in values.items():
                        flat_row[f"{condition}_{group}_{key}"] = value
                condition_metrics["left_dex3_nearest_prototype_sequence"] = nearest_hand_phases(
                    prediction[:, GROUPS["left_dex3"]], hand_prototypes["left"]
                )
                condition_metrics["right_dex3_nearest_prototype_sequence"] = nearest_hand_phases(
                    prediction[:, GROUPS["right_dex3"]], hand_prototypes["right"]
                )
                sample_detail["conditions"][condition] = condition_metrics
            sample_detail["target_left_dex3_nearest_prototype_sequence"] = nearest_hand_phases(
                target[:, GROUPS["left_dex3"]], hand_prototypes["left"]
            )
            sample_detail["target_right_dex3_nearest_prototype_sequence"] = nearest_hand_phases(
                target[:, GROUPS["right_dex3"]], hand_prototypes["right"]
            )
            sample_detail["paired_A_B_prediction_rmse_rad"] = rms(pred_a - pred_b)
            flat_row["paired_A_B_prediction_rmse_rad"] = sample_detail[
                "paired_A_B_prediction_rmse_rad"
            ]
            detailed_metrics.append(sample_detail)
            metrics_rows.append(flat_row)
            sample_index += 1
            print(
                f"probe {sample_index:02d}/{len(EPISODES) * len(PHASE_LABELS)} "
                f"ep={episode_index:02d} phase={phase} frame={frame_index}",
                flush=True,
            )

    prediction_a_array = np.stack(prediction_a).astype(np.float32)
    prediction_b_array = np.stack(prediction_b).astype(np.float32)
    target_array = np.stack(target_chunks).astype(np.float32)
    state_a_array = np.stack(state_a_values).astype(np.float32)
    state_b_array = np.stack(state_b_values).astype(np.float32)

    # One valid Condition-C pair: the exact frozen Stage-0 Isaac RGB and its
    # simultaneously measured qpos.  Later phases have no phase-matched Isaac
    # observation because the closed-loop rollout never reached them.
    isaac_image = np.asarray(Image.open(ISAAC_STAGE0 / "input_rgb.png").convert("RGB"))
    if isaac_image.shape != (480, 640, 3):
        raise RuntimeError("Isaac Stage-0 RGB schema mismatch")
    isaac_tensor = torch.from_numpy(isaac_image.copy()).permute(2, 0, 1).float() / 255.0
    isaac_state = np.load(ISAAC_STAGE0 / "input_state.npy").astype(np.float32)
    reference_index = next(
        index
        for index, row in enumerate(selected)
        if row["episode_index"] == 24 and row["phase"] == "initial_left_approach"
    )
    isaac_seed = int(selected[reference_index]["paired_common_random_seed"])
    prediction_c = infer(
        policy,
        preprocessor,
        postprocessor,
        isaac_tensor,
        isaac_state,
        TASK,
        isaac_seed,
    )
    existing_isaac_chunks = np.load(
        ISAAC_ROLLOUT / "inference_chunks.npz", allow_pickle=False
    )["policy_raw_action"]
    existing_isaac_inferences = json.loads(
        (ISAAC_ROLLOUT / "inference_timestamps.json").read_text(encoding="utf-8")
    )
    valid_existing_c = []
    for inference_index in (0, 22, 44):
        valid_existing_c.append(
            {
                "inference_index": inference_index,
                "simulation_timestamp_seconds": float(
                    existing_isaac_inferences[inference_index]["simulation_timestamp_s"]
                ),
                "input_rgb_sha256": existing_isaac_inferences[inference_index][
                    "input_rgb_sha256"
                ],
                "input_state_sha256": existing_isaac_inferences[inference_index][
                    "input_state_sha256"
                ],
                "semantic_scope": "initial/left-approach portion reached by Isaac rollout",
                "prediction_motion_magnitude_by_group": {
                    group: rms(
                        existing_isaac_chunks[inference_index, :, indices]
                        - existing_isaac_chunks[inference_index, :1, indices]
                    )
                    for group, indices in GROUPS.items()
                },
            }
        )

    phase_summary: dict[str, Any] = {}
    phase_csv_rows: list[dict[str, Any]] = []
    for phase, label in PHASE_LABELS.items():
        indices = [index for index, row in enumerate(selected) if row["phase"] == phase]
        summary: dict[str, Any] = {
            "phase_label": label,
            "sample_count": len(indices),
            "episode_indices": [selected[index]["episode_index"] for index in indices],
            "target_behavior_present_by_annotation": True,
            "policy_behavior_present": "PENDING_EVIDENCE_REVIEW",
            "groups": {},
        }
        csv_row: dict[str, Any] = {
            "phase": phase,
            "phase_label": label,
            "sample_count": len(indices),
            "target_behavior_present": "YES",
            "policy_behavior_present": "PENDING_EVIDENCE_REVIEW",
        }
        for group, joint_indices in GROUPS.items():
            group_rows = [
                group_metrics(prediction_a_array[index], target_array[index], joint_indices)
                for index in indices
            ]
            summary["groups"][group] = {
                key: summarize_numbers(group_rows, key)
                for key in group_rows[0]
            }
            for key in (
                "first_action_rmse_rad",
                "prefix4_rmse_rad",
                "full_chunk_rmse_rad",
                "predicted_motion_magnitude_rad_rms",
                "target_motion_magnitude_rad_rms",
                "trajectory_motion_cosine",
                "target_direction_projection_progress",
            ):
                csv_row[f"{group}_{key}_mean"] = summary["groups"][group][key][
                    "mean"
                ]
        combined_arm_rows = [
            {
                "value": group_metrics(
                    prediction_a_array[index], target_array[index], np.arange(14)
                )["full_chunk_rmse_rad"]
            }
            for index in indices
        ]
        combined_dex_rows = [
            {
                "value": group_metrics(
                    prediction_a_array[index], target_array[index], np.arange(14, 28)
                )["full_chunk_rmse_rad"]
            }
            for index in indices
        ]
        summary["combined_arm_full_chunk_rmse_rad"] = summarize_numbers(
            combined_arm_rows, "value"
        )
        summary["combined_dex3_full_chunk_rmse_rad"] = summarize_numbers(
            combined_dex_rows, "value"
        )
        ab_rows = [
            {"value": rms(prediction_a_array[index] - prediction_b_array[index])}
            for index in indices
        ]
        summary["paired_A_B_prediction_rmse_rad"] = summarize_numbers(ab_rows, "value")
        csv_row["arm_rmse_full_mean_rad"] = summary[
            "combined_arm_full_chunk_rmse_rad"
        ]["mean"]
        csv_row["dex3_rmse_full_mean_rad"] = summary[
            "combined_dex3_full_chunk_rmse_rad"
        ]["mean"]
        csv_row["paired_A_B_prediction_rmse_mean_rad"] = summary[
            "paired_A_B_prediction_rmse_rad"
        ]["mean"]
        phase_summary[phase] = summary
        phase_csv_rows.append(csv_row)

    distribution_shift = {
        "comparison_scope": {
            "A": "Exact training RGB and exact stored lag-1 RETARGETED_G1_STATE_SURROGATE",
            "B": "Same training RGB with same-row q_target[t] as a temporally matched current-target-style G1 state; not measured feedback",
            "C": "Frozen SOURCE_LIKE_CAM_HIGH Isaac RGB with simultaneously measured Isaac G1/Dex3 qpos",
        },
        "paired_randomness_control": "A and B reset to the same per-sample seed before inference",
        "A_B_all_selected_frames": {
            "sample_count": len(selected),
            "state_input_rmse_rad": summarize_numbers(
                [{"value": row["state_a_to_b_rmse_rad"]} for row in selected],
                "value",
            ),
            "prediction_chunk_rmse_rad": summarize_numbers(
                [
                    {"value": rms(prediction_a_array[i] - prediction_b_array[i])}
                    for i in range(len(selected))
                ],
                "value",
            ),
            "phase_breakdown": {
                phase: phase_summary[phase]["paired_A_B_prediction_rmse_rad"]
                for phase in PHASE_LABELS
            },
        },
        "C_valid_matched_initial_pair": {
            "rgb_path": str(ISAAC_STAGE0 / "input_rgb.png"),
            "state_path": str(ISAAC_STAGE0 / "input_state.npy"),
            "common_seed_reference_sample": selected[reference_index],
            "prediction_vs_A_reference_rmse_rad": rms(
                prediction_c - prediction_a_array[reference_index]
            ),
            "prediction_motion_magnitude_by_group": {
                group: rms(prediction_c[:, indices] - prediction_c[:1, indices])
                for group, indices in GROUPS.items()
            },
            "authoritative_target_rmse": None,
            "target_rmse_reason": "No Dataset-B target is temporally paired with the Isaac observation.",
        },
        "C_existing_initial_approach_evidence": valid_existing_c,
        "C_later_phase_rows": "NOT_AVAILABLE_NO_PHASE_MATCHED_ISAAC_OBSERVATION",
        "invalid_hybrid_pairing_performed": False,
    }

    np.savez_compressed(
        output / "probe_predictions.npz",
        policy_prediction_A=prediction_a_array,
        policy_prediction_B=prediction_b_array,
        policy_prediction_C_initial=prediction_c.astype(np.float32),
        authoritative_target=target_array,
        state_A=state_a_array,
        state_B=state_b_array,
        joint_names=np.asarray(policy_joint_names),
        episode_index=np.asarray([row["episode_index"] for row in selected]),
        frame_index=np.asarray([row["frame_index"] for row in selected]),
        phase=np.asarray([row["phase"] for row in selected]),
    )
    atomic_json(output / "selected_frames.json", selected)
    atomic_json(output / "per_sample_metrics.json", detailed_metrics)
    atomic_json(output / "phase_metrics.json", phase_summary)
    atomic_json(output / "distribution_shift_comparison.json", distribution_shift)
    atomic_json(
        output / "hand_phase_prototypes.json",
        {
            "joint_order": {
                "left": [policy_joint_names[i] for i in GROUPS["left_dex3"]],
                "right": [policy_joint_names[i] for i in GROUPS["right_dex3"]],
            },
            "annotation_frame_counts": phase_annotation_counts,
            "prototypes": hand_prototypes,
            "definition": "Mean absolute Dataset-B action q for each frozen per-frame hand-phase annotation",
        },
    )
    write_csv(output / "per_sample_metrics.csv", metrics_rows)
    write_csv(output / "phase_metrics.csv", phase_csv_rows)
    save_phase_contact_sheets(output, selected)
    save_phase_plots(output, selected, prediction_a_array, target_array)
    save_summary_plots(output, phase_summary)

    integrity = {
        "status": "PASS",
        "read_only_probe": True,
        "dataset_modified": False,
        "policy_modified": False,
        "camera_modified": False,
        "retargeting_modified": False,
        "deployment_code_modified": False,
        "real_robot_invoked": False,
        "checkpoint": str(CHECKPOINT),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": sha256_file(CHECKPOINT / "model.safetensors"),
        "semantic_manifest": str(SEMANTIC_MANIFEST),
        "semantic_manifest_sha256_before": semantic_hash_before,
        "semantic_manifest_sha256_after": sha256_file(SEMANTIC_MANIFEST),
        "dataset_info_sha256_before": dataset_info_hash_before,
        "dataset_info_sha256_after": sha256_file(DATASET_ROOT / "meta/info.json"),
        "semantic_manifest_status": semantic_manifest["status"],
        "selected_episode_count": len(EPISODES),
        "selected_phase_count": len(PHASE_LABELS),
        "selected_frame_count": len(selected),
        "prediction_shape_A": list(prediction_a_array.shape),
        "prediction_shape_B": list(prediction_b_array.shape),
        "target_shape": list(target_array.shape),
        "all_finite": bool(
            np.isfinite(prediction_a_array).all()
            and np.isfinite(prediction_b_array).all()
            and np.isfinite(target_array).all()
        ),
        "full_target_chunk_available_for_every_probe": True,
        "task_instruction": TASK,
        "state_A": "exact Dataset-B observation.state at the selected frame",
        "state_B": "same-row Dataset-B q_target[t], current-target-style but not measured",
        "state_C": "measured Isaac qpos paired only with its actual Isaac RGB",
    }
    if (
        integrity["model_sha256_before"] != integrity["model_sha256_after"]
        or integrity["semantic_manifest_sha256_before"]
        != integrity["semantic_manifest_sha256_after"]
        or integrity["dataset_info_sha256_before"]
        != integrity["dataset_info_sha256_after"]
    ):
        raise RuntimeError("a frozen input changed during the read-only probe")
    atomic_json(output / "probe_integrity.json", integrity)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "selected_frames": len(selected),
                "prediction_shape": list(prediction_a_array.shape),
                "phase_metrics": str(output / "phase_metrics.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
