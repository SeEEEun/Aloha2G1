#!/usr/bin/env python3
"""Read Dataset B through the installed LeRobotDataset implementation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


TASK = (
    "Pick up the doll with the left hand, handoff it to the right hand, "
    "and place it in the trash bin."
)
FPS = 30
CHUNK = 50


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=jsonable) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/doll_handoff_proposed_b_50")
    return parser.parse_args()


def scalar(value: Any) -> int | float:
    return value.item() if hasattr(value, "item") else value


def validate_item(item: dict[str, Any], expected_episode: int) -> dict[str, Any]:
    import torch

    image = item["observation.images.cam_high"]
    state = item["observation.state"]
    action = item["action"]
    checks = {
        "episode_index": int(scalar(item["episode_index"])) == expected_episode,
        "image_shape": tuple(image.shape) == (3, 480, 640),
        "image_finite": bool(torch.isfinite(image).all()),
        "state_shape": tuple(state.shape) == (28,),
        "action_shape": tuple(action.shape) == (28,),
        "state_finite": bool(torch.isfinite(state).all()),
        "action_finite": bool(torch.isfinite(action).all()),
        "task": item.get("task") == TASK,
    }
    if not all(checks.values()):
        raise RuntimeError(f"LeRobot item check failed: {checks}")
    return {
        "global_index": int(scalar(item["index"])),
        "episode_index": int(scalar(item["episode_index"])),
        "frame_index": int(scalar(item["frame_index"])),
        "timestamp": float(scalar(item["timestamp"])),
        "image_shape": list(image.shape),
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "task": item["task"],
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    root = args.dataset.resolve()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.import_utils import get_safe_default_video_backend

    episode_rows = []
    for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    if len(episode_rows) != 50:
        raise RuntimeError(f"expected 50 episode rows, got {len(episode_rows)}")
    offsets = [int(row["dataset_from_index"]) for row in episode_rows]
    lengths = [int(row["length"]) for row in episode_rows]
    default_backend = get_safe_default_video_backend()
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        download_videos=False,
        video_backend=default_backend,
    )
    if len(dataset) != sum(lengths) or dataset.meta.total_episodes != 50:
        raise RuntimeError("LeRobotDataset length/episode metadata mismatch")
    if tuple(dataset.meta.features["observation.state"]["shape"]) != (28,):
        raise RuntimeError("LeRobot state feature shape mismatch")
    if tuple(dataset.meta.features["action"]["shape"]) != (28,):
        raise RuntimeError("LeRobot action feature shape mismatch")

    # Decode one real frame from every episode with the backend training will use.
    every_episode = []
    for episode, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        global_index = offset + length // 2
        every_episode.append(validate_item(dataset[global_index], episode))

    deterministic = [
        (0, 0),
        (0, lengths[0] // 2),
        (0, lengths[0] - 1),
        (25, 0),
        (25, lengths[25] // 2),
        (25, lengths[25] - 1),
        (49, 0),
        (49, lengths[49] // 2),
        (49, lengths[49] - 1),
    ]
    random_episodes = sorted(random.Random(20260823).sample(range(1, 49), 6))
    deterministic.extend(
        (episode, lengths[episode] // 2) for episode in random_episodes
    )
    selected_reads = [
        validate_item(dataset[offsets[episode] + frame], episode)
        for episode, frame in deterministic
    ]

    # PyAV is also audited explicitly as the independent fallback decoder.
    pyav = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        download_videos=False,
        video_backend="pyav",
    )
    pyav_reads = [
        validate_item(pyav[offsets[episode] + lengths[episode] // 2], episode)
        for episode in (0, 25, 49, *random_episodes[:3])
    ]

    deltas = {
        "observation.state": [0.0],
        "action": [index / FPS for index in range(CHUNK)],
    }
    chunked = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend=default_backend,
    )
    chunk_rows = []
    for episode in (0, 25, 49):
        index = offsets[episode] + lengths[episode] - 1
        item = chunked[index]
        action_shape = tuple(item["action"].shape)
        state_shape = tuple(item["observation.state"].shape)
        padding = item["action_is_pad"].detach().cpu().numpy().astype(bool)
        if action_shape != (50, 28) or state_shape != (1, 28):
            raise RuntimeError(
                f"chunk feature shape mismatch ep{episode}: {state_shape}/{action_shape}"
            )
        if bool(padding[0]) or not bool(np.all(padding[1:])):
            raise RuntimeError(f"last-frame action padding mismatch ep{episode}")
        chunk_rows.append(
            {
                "episode_index": episode,
                "global_index": index,
                "state_shape": list(state_shape),
                "action_shape": list(action_shape),
                "action_is_pad": padding.tolist(),
            }
        )
    result = {
        "schema_version": "doll_handoff_lerobot_readback_v2",
        "repo_id": args.repo_id,
        "status": "PASS",
        "python": sys.executable,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("lerobot", "torch", "torchcodec", "av", "datasets", "pyarrow")
        },
        "dataset_path": str(root),
        "dataset_length": len(dataset),
        "episode_count": int(dataset.meta.total_episodes),
        "feature_schema": dataset.meta.features,
        "default_video_backend": default_backend,
        "every_episode_middle_frame_read_count": len(every_episode),
        "every_episode_middle_frame_reads": every_episode,
        "first_middle_last_and_random_reads": selected_reads,
        "random_episode_indices": random_episodes,
        "pyav_fallback_reads": pyav_reads,
        "chunk_and_padding_reads": chunk_rows,
        "video_decoding": "PASS with default backend for all 50 episodes and PyAV fallback sample",
        "task_loading": "PASS",
        "state_tensor_shape": [28],
        "action_tensor_shape": [28],
        "action_chunk_tensor_shape": [50, 28],
    }
    write_json(args.output.resolve(), result)
    print(json.dumps({key: result[key] for key in (
        "status", "dataset_length", "episode_count", "default_video_backend",
        "every_episode_middle_frame_read_count", "random_episode_indices"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
