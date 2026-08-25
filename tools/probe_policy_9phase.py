#!/usr/bin/env python3
"""Common read-only nine-phase offline probe for Policy A or Policy B."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = "Pick up the doll with the left hand, handoff it to the right hand, and place it in the trash bin."
EPISODES = (0, 12, 13, 24, 36, 49)
CHUNK = 50
PHASES = OrderedDict(
    (
        ("initial_left_approach", "initial / left approach"),
        ("immediately_before_left_grasp", "immediately before left grasp"),
        ("left_grasp_left_owned", "left grasp / LEFT_OWNED"),
        ("left_transport", "left transport"),
        ("handoff_approach", "handoff approach"),
        ("dual_contact_transfer", "dual-contact / ownership transfer"),
        ("right_owned", "RIGHT_OWNED"),
        ("right_transport", "right transport toward bin"),
        ("release", "release"),
    )
)
GROUPS = {
    "left_arm": np.arange(0, 7),
    "right_arm": np.arange(7, 14),
    "left_dex3": np.arange(14, 21),
    "right_dex3": np.arange(21, 28),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def event_map(trajectory: Any) -> dict[str, int]:
    return {
        str(name): int(frame)
        for name, frame in zip(trajectory["event_names"], trajectory["event_frames"], strict=True)
    }


def segment(labels: np.ndarray, value: str) -> tuple[int, int]:
    indices = np.flatnonzero(labels.astype(str) == value)
    if not len(indices) or np.any(np.diff(indices) != 1):
        raise RuntimeError(f"missing or non-contiguous semantic phase {value}")
    return int(indices[0]), int(indices[-1])


def select_frames(trajectory: Any) -> dict[str, int]:
    events = event_map(trajectory)
    ownership = np.asarray(trajectory["ownership_state"]).astype(str)
    left_start, left_end = segment(ownership, "LEFT_OWNED")
    right_start, right_end = segment(ownership, "RIGHT_TRANSPORT")
    transport_start = max(events["LEFT_STABLE_HOLD"], left_start)
    transport_end = min(events["HANDOFF_APPROACH"] - 1, left_end)
    frames = {
        "initial_left_approach": events["LEFT_CLOSE_ONSET"] - CHUNK,
        "immediately_before_left_grasp": events["LEFT_CLOSE_ONSET"] - 1,
        "left_grasp_left_owned": events["LEFT_GRASP"],
        "left_transport": (transport_start + transport_end) // 2,
        "handoff_approach": events["HANDOFF_APPROACH"],
        "dual_contact_transfer": events["RIGHT_ACQUIRE"],
        "right_owned": events["LEFT_RELEASE"],
        "right_transport": min(right_start + 10, max(right_start, right_end - CHUNK + 1)),
        "release": events["RIGHT_FINAL_RELEASE"],
    }
    for phase, frame in frames.items():
        if frame < 0 or frame + CHUNK > len(ownership):
            raise RuntimeError(f"{phase}: full 50-row target is unavailable at {frame}")
    return frames


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer(policy: Any, preprocessor: Any, postprocessor: Any, image: torch.Tensor, state: np.ndarray, seed: int) -> np.ndarray:
    set_seed(seed)
    if hasattr(policy, "reset"):
        policy.reset()
    raw = {
        "observation.images.cam_high": image.detach().clone(),
        "observation.state": torch.from_numpy(state.astype(np.float32, copy=True)),
        "task": TASK,
    }
    with torch.inference_mode():
        prediction = postprocessor(policy.predict_action_chunk(preprocessor(raw)))
    result = prediction.detach().float().cpu().numpy()
    if result.shape != (1, CHUNK, 28) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid policy prediction {result.shape}")
    return result[0]


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(value, dtype=np.float64)))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-variant", choices=("A", "B"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    dataset_root = args.dataset.resolve()
    trajectories = args.trajectories.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    model_hash_before = sha256_file(checkpoint / "model.safetensors")
    info_hash_before = sha256_file(dataset_root / "meta/info.json")
    sources = json.loads(args.source_manifest.read_text(encoding="utf-8"))["episodes"]
    dataset = LeRobotDataset(
        repo_id=f"local/doll_handoff_policy_{args.policy_variant.lower()}_probe",
        root=dataset_root,
        video_backend="torchcodec",
    )
    if dataset.num_episodes != 50 or len(dataset) != sum(int(row["source_frame_count"]) for row in sources):
        raise RuntimeError("dataset/common-source identity mismatch")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config["input_features"]["observation.state"]["shape"] != [28] or config["output_features"]["action"]["shape"] != [28]:
        raise RuntimeError("checkpoint is not 28D state/action")
    if int(config["chunk_size"]) != CHUNK or not torch.cuda.is_available():
        raise RuntimeError("probe requires a 50-row checkpoint and CUDA")
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    policy.eval()

    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for episode in EPISODES:
        metadata = dataset.meta.episodes[episode]
        start, end = int(metadata["dataset_from_index"]), int(metadata["dataset_to_index"])
        table = dataset.hf_dataset[start:end]
        states = np.asarray(table["observation.state"], dtype=np.float32)
        actions = np.asarray(table["action"], dtype=np.float32)
        source = sources[episode]
        candidates = (
            trajectories / f"episode_{episode:06d}.npz",
            trajectories / f"{source['stable_episode_id']}.npz",
        )
        trajectory_path = next((path for path in candidates if path.is_file()), candidates[-1])
        trajectory = np.load(trajectory_path, allow_pickle=False)
        replay_names = trajectory["replay_joint_names"].astype(str).tolist()
        dataset_names = list(dataset.features["action"]["names"])
        replay = np.asarray(trajectory["replay_named_joint_qpos"], dtype=np.float32)
        replay = replay[:, [replay_names.index(name) for name in dataset_names]]
        if not np.array_equal(actions, replay) or not np.array_equal(states[0], actions[0]) or not np.array_equal(states[1:], actions[:-1]):
            raise RuntimeError(f"episode {episode}: frozen label or lag-1 identity failed")
        for phase, frame in select_frames(trajectory).items():
            sample = dataset[start + frame]
            if sample["task"] != TASK:
                raise RuntimeError("task instruction mismatch")
            target = actions[frame : frame + CHUNK]
            prediction = infer(policy, preprocessor, postprocessor, sample["observation.images.cam_high"], states[frame], args.seed + len(rows))
            group_rmse = {name: rms(prediction[:, indices] - target[:, indices]) for name, indices in GROUPS.items()}
            rows.append({
                "episode_index": episode,
                "frame_index": frame,
                "phase": phase,
                "phase_label": PHASES[phase],
                "source_raw_episode": source["raw_directory"],
                "full_chunk_rmse_rad": rms(prediction - target),
                "first_action_rmse_rad": rms(prediction[0] - target[0]),
                "group_full_chunk_rmse_rad": group_rmse,
            })
            predictions.append(prediction)
            targets.append(target)
            print(f"Policy {args.policy_variant} probe ep={episode:02d} phase={phase}", flush=True)
    prediction_array = np.asarray(predictions, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    phase_metrics = {}
    for phase, label in PHASES.items():
        selected = [row for row in rows if row["phase"] == phase]
        phase_metrics[phase] = {
            "label": label,
            "samples": len(selected),
            "full_chunk_rmse_rad_mean": float(np.mean([row["full_chunk_rmse_rad"] for row in selected])),
            "first_action_rmse_rad_mean": float(np.mean([row["first_action_rmse_rad"] for row in selected])),
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", prediction=prediction_array, target=target_array)
    report = {
        "schema_version": "common_policy_9phase_probe_v1",
        "status": "PASS",
        "policy_variant": args.policy_variant,
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256_before": model_hash_before,
        "checkpoint_model_sha256_after": sha256_file(checkpoint / "model.safetensors"),
        "dataset": str(dataset_root),
        "dataset_info_sha256_before": info_hash_before,
        "dataset_info_sha256_after": sha256_file(dataset_root / "meta/info.json"),
        "episodes": list(EPISODES),
        "phase_count": len(PHASES),
        "sample_count": len(rows),
        "prediction_shape": list(prediction_array.shape),
        "selection_rule_matches_original_policy_b_probe": True,
        "phase_metrics": phase_metrics,
        "samples": rows,
        "read_only": True,
        "real_robot_invoked": False,
    }
    if report["checkpoint_model_sha256_before"] != report["checkpoint_model_sha256_after"] or report["dataset_info_sha256_before"] != report["dataset_info_sha256_after"]:
        raise RuntimeError("frozen probe input changed")
    atomic_json(output / "probe_report.json", report)
    print(json.dumps({"status": "PASS", "samples": len(rows), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
