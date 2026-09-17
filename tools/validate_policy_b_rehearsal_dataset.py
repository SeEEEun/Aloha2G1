#!/usr/bin/env python3
"""Validate the balanced ALOHA/G1 visual rehearsal dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/doll_handoff_proposed_b_rehearsal_100"
ORIGINAL = ROOT / "datasets/doll_handoff_proposed_b_50"
G1VISUAL = ROOT / "datasets/doll_handoff_proposed_b_g1visual_50"
OUTPUT = ROOT / "outputs/policy_b_g1visual/rehearsal/dataset_validation.json"
TASK = (
    "Pick up the doll with the left hand, handoff it to the right hand, "
    "and place it in the trash bin."
)


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


def logical_hash(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def episode_arrays(dataset: Any, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = dataset.meta.episodes[episode]
    start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    table = dataset.hf_dataset[start:end]
    return (
        np.asarray(table["observation.state"], dtype=np.float32),
        np.asarray(table["action"], dtype=np.float32),
        np.asarray(table["timestamp"], dtype=np.float32),
    )


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paired = LeRobotDataset(
        repo_id="local/doll_handoff_proposed_b_rehearsal_100",
        root=DATASET,
        video_backend="torchcodec",
    )
    original = LeRobotDataset(
        repo_id="local/doll_handoff_proposed_b_50", root=ORIGINAL, video_backend="torchcodec"
    )
    g1visual = LeRobotDataset(
        repo_id="local/doll_handoff_proposed_b_g1visual_50",
        root=G1VISUAL,
        video_backend="torchcodec",
    )
    exact_rows = []
    paired_states: list[np.ndarray] = []
    paired_actions: list[np.ndarray] = []
    for episode in range(50):
        o_state, o_action, o_time = episode_arrays(original, episode)
        g_state, g_action, g_time = episode_arrays(g1visual, episode)
        a_state, a_action, a_time = episode_arrays(paired, episode)
        b_state, b_action, b_time = episode_arrays(paired, episode + 50)
        row = {
            "source_episode": episode,
            "length": len(o_state),
            "original_half_state_exact": bool(np.array_equal(o_state, a_state)),
            "original_half_action_exact": bool(np.array_equal(o_action, a_action)),
            "original_half_timestamp_exact": bool(np.array_equal(o_time, a_time)),
            "g1_half_state_exact": bool(np.array_equal(g_state, b_state)),
            "g1_half_action_exact": bool(np.array_equal(g_action, b_action)),
            "g1_half_timestamp_exact": bool(np.array_equal(g_time, b_time)),
            "cross_domain_state_exact": bool(np.array_equal(o_state, g_state)),
            "cross_domain_action_exact": bool(np.array_equal(o_action, g_action)),
        }
        exact_rows.append(row)
        paired_states.extend([a_state, b_state])
        paired_actions.extend([a_action, b_action])

    sample_episodes = [0, 12, 24, 49, 50, 62, 74, 99]
    reads = []
    for episode in sample_episodes:
        meta = paired.meta.episodes[episode]
        start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
        index = (start + end - 1) // 2
        sample = paired[index]
        reads.append(
            {
                "episode": episode,
                "index": index,
                "image_shape": list(sample["observation.images.cam_high"].shape),
                "state_shape": list(sample["observation.state"].shape),
                "action_shape": list(sample["action"].shape),
                "task": sample["task"],
                "finite": bool(
                    np.isfinite(sample["observation.state"].numpy()).all()
                    and np.isfinite(sample["action"].numpy()).all()
                ),
            }
        )

    all_exact = all(
        all(value for key, value in row.items() if key.endswith("_exact")) for row in exact_rows
    )
    all_reads = all(
        row["image_shape"] == [3, 480, 640]
        and row["state_shape"] == [28]
        and row["action_shape"] == [28]
        and row["task"] == TASK
        and row["finite"]
        for row in reads
    )
    lengths = np.asarray([row["length"] for row in exact_rows], dtype=np.int64)
    result = {
        "status": "PASS" if all_exact and all_reads else "FAIL",
        "dataset": str(DATASET),
        "episodes": paired.num_episodes,
        "frames": len(paired),
        "domain_a_episodes": [0, 49],
        "domain_b_episodes": [50, 99],
        "domain_a_frames": int(lengths.sum()),
        "domain_b_frames": int(lengths.sum()),
        "balanced_frame_fraction": {"ALOHA_RGB": 0.5, "G1_RGB": 0.5},
        "paired_supervision_exact": all_exact,
        "action_logical_sha256": logical_hash(paired_actions),
        "state_logical_sha256": logical_hash(paired_states),
        "camera_keys": paired.meta.camera_keys,
        "state_names": paired.features["observation.state"]["names"],
        "action_names": paired.features["action"]["names"],
        "episode_checks": exact_rows,
        "video_reads": reads,
        "source_datasets_modified": False,
        "metadata_harmonization": {
            "field": "robot_type",
            "merge_view_value": "unitree_g1_fixed_base_dex3_retargeted",
            "reason": "same target robot/control schema; original G1-visual source metadata remains unchanged",
        },
    }
    if paired.num_episodes != 100 or len(paired) != 68956:
        result["status"] = "FAIL"
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": result["status"], "episodes": paired.num_episodes, "frames": len(paired)}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
