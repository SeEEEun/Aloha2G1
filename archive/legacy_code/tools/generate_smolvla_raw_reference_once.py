#!/usr/bin/env python3
"""Generate one frozen-noise SmolVLA reference for ACT-B comparison.

This is a read-only offline policy probe.  It uses one sampled flow-noise tensor
for all seven frozen Dataset-B observations and never invokes an execution
adapter, simulator, controller, or robot transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/home/jbnu/aloha_g1_dataset")
DATASET_ROOT = ROOT / "datasets/doll_handoff_proposed_b_50"
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs/policy_b_doll_handoff_proposed_b_50_lag1_state_v2/checkpoints/020000/pretrained_model"
)
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_act/comparison/smolvla_raw_fixed_noise_reference.npz"
SELECTIONS_PATH = ROOT / "outputs/policy_b_offline_phase_probe/selected_frames.json"
PROBE_ARCHIVE = ROOT / "outputs/policy_b_offline_phase_probe/probe_predictions.npz"
EPISODE = 24
SEED = 2026082601
TASK = "Pick up the doll with the left hand, handoff it to the right hand, and place it in the trash bin."

RAW_SELECTIONS = (
    ("initial", None),
    ("left_approach", "initial_left_approach"),
    ("doll_plateau", "left_grasp_left_owned"),
    ("left_transport", "left_transport"),
    ("handoff_approach", "handoff_approach"),
    ("right_transport", "right_transport"),
    ("release", "release"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the existing SmolVLA checkpoint")

    selections = json.loads(SELECTIONS_PATH.read_text(encoding="utf-8"))
    episode_rows = [row for row in selections if int(row["episode_index"]) == EPISODE]
    by_phase = {str(row["phase"]): row for row in episode_rows}
    dataset = LeRobotDataset(
        "local/doll_handoff_proposed_b_50",
        root=DATASET_ROOT,
        video_backend="torchcodec",
    )
    episode_metadata = dict(dataset.meta.episodes[EPISODE])
    episode_start = int(episode_metadata["dataset_from_index"])
    trajectory_path = Path(episode_rows[0]["trajectory_path"])
    with np.load(trajectory_path, allow_pickle=False) as archive:
        episode_action = archive["replay_named_joint_qpos"].astype(np.float32)
    if episode_action.shape != (int(episode_metadata["length"]), 28):
        raise RuntimeError("authoritative episode-24 action shape changed")

    sample_specs: list[dict[str, Any]] = []
    targets = []
    for label, phase in RAW_SELECTIONS:
        if phase is None:
            frame_index = 0
            global_index = episode_start
        else:
            row = by_phase[phase]
            frame_index = int(row["frame_index"])
            global_index = int(row["global_dataset_index"])
        available = min(50, len(episode_action) - frame_index)
        target = episode_action[frame_index : frame_index + available]
        if available < 50:
            target = np.concatenate(
                (target, np.repeat(target[-1:], 50 - available, axis=0)), axis=0
            )
        targets.append(target)
        sample_specs.append(
            {
                "label": label,
                "phase": phase,
                "episode_index": EPISODE,
                "frame_index": frame_index,
                "global_dataset_index": global_index,
                "available_target_actions": available,
            }
        )

    # Confirm targets are the same frozen arrays used in the prior phase audit.
    with np.load(PROBE_ARCHIVE, allow_pickle=False) as archive:
        prior_targets = archive["authoritative_target"].astype(np.float32)
        prior_episodes = archive["episode_index"].astype(np.int64)
        prior_frames = archive["frame_index"].astype(np.int64)
    for index, specification in enumerate(sample_specs[1:], start=1):
        matches = np.flatnonzero(
            (prior_episodes == EPISODE)
            & (prior_frames == int(specification["frame_index"]))
        )
        if len(matches) != 1 or not np.array_equal(targets[index], prior_targets[matches[0]]):
            raise RuntimeError("frozen Dataset-B target differs from prior phase archive")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )
    policy.eval()
    device = next(policy.parameters()).device
    # Reset immediately before noise creation so model construction cannot
    # advance the recorded reference seed.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    fixed_noise = policy.model.sample_noise(
        (1, 50, int(policy.config.max_action_dim)), device
    )
    fixed_noise_numpy = fixed_noise.detach().float().cpu().numpy()
    predictions = []
    states = []
    for specification in sample_specs:
        sample = dataset[int(specification["global_dataset_index"])]
        if (
            int(sample["episode_index"]) != EPISODE
            or int(sample["frame_index"]) != int(specification["frame_index"])
        ):
            raise RuntimeError("Dataset-B sample identity changed")
        state = sample["observation.state"].detach().float().cpu().numpy()
        processed = preprocessor(
            {
                "observation.images.cam_high": sample["observation.images.cam_high"].clone(),
                "observation.state": sample["observation.state"].clone(),
                "task": TASK,
            }
        )
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(processed, noise=fixed_noise)
            physical = postprocessor(normalized)
        prediction = physical[0].detach().float().cpu().numpy()
        if prediction.shape != (50, 28) or not np.isfinite(prediction).all():
            raise RuntimeError(f"malformed SmolVLA output {prediction.shape}")
        predictions.append(prediction)
        states.append(state)

    predictions_array = np.stack(predictions).astype(np.float32)
    targets_array = np.stack(targets).astype(np.float32)
    states_array = np.stack(states).astype(np.float32)
    labels_array = np.asarray([row["label"] for row in sample_specs])
    frames_array = np.asarray([row["frame_index"] for row in sample_specs], dtype=np.int64)
    atomic_npz(
        output,
        labels=labels_array,
        episode_index=np.full(len(sample_specs), EPISODE, dtype=np.int64),
        frame_index=frames_array,
        global_dataset_index=np.asarray(
            [row["global_dataset_index"] for row in sample_specs], dtype=np.int64
        ),
        observation_state=states_array,
        authoritative_target=targets_array,
        smolvla_raw_fixed_noise=predictions_array,
        fixed_flow_noise=fixed_noise_numpy.astype(np.float32),
    )
    manifest = {
        "schema_version": "smolvla_raw_reference_for_act_b_v1",
        "status": "PASS",
        "offline_inference_only": True,
        "execution_adapter_used": False,
        "controller_used": False,
        "simulator_used": False,
        "hardware_transport_used": False,
        "smolvla_tuning_performed": False,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.safetensors"),
        "flow_noise_seed": SEED,
        "one_identical_flow_noise_tensor_used_for_all_observations": True,
        "flow_noise_sha256": hashlib.sha256(fixed_noise_numpy.tobytes()).hexdigest(),
        "sample_specs": sample_specs,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "prediction_shape": list(predictions_array.shape),
    }
    atomic_json(output.with_suffix(".json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
