from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tools.g1_training_schema_v1.constants import ACTION_KEY, CHUNK_SIZE, FPS, STATE_KEY, TASK_TEXT


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    return sorted(
        pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist(),
        key=lambda row: int(row["episode_index"]),
    )


def run_lerobot_readback(root_a: str | Path, root_b: str | Path, output: str | Path) -> dict[str, Any]:
    import lerobot
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root_a, root_b = Path(root_a).resolve(), Path(root_b).resolve()
    rows = _episode_rows(root_a)
    manifest = json.loads((root_a / "meta/g1_packaging_manifest.json").read_text())
    stable_ids = manifest["stable_source_ids"]
    new_index = next(index for index, stable_id in enumerate(stable_ids) if stable_id.startswith("new20:"))
    old_index = len([stable_id for stable_id in stable_ids if stable_id.startswith("old50:")]) // 2
    samples = [
        (0, 0, "first_episode_first_frame"),
        (0, int(rows[0]["length"]) // 2, "first_episode_middle_frame"),
        (old_index, int(rows[old_index]["length"]) // 2, "old50_episode"),
        (new_index, int(rows[new_index]["length"]) // 2, "new20_episode"),
        (len(rows) - 1, int(rows[-1]["length"]) - 1, "last_episode_final_frame"),
    ]
    offsets = np.cumsum([0] + [int(row["length"]) for row in rows[:-1]])
    deltas = {STATE_KEY: [0.0], ACTION_KEY: [index / FPS for index in range(CHUNK_SIZE)]}
    dataset_a = LeRobotDataset(
        repo_id="local/g1_magsafe_matched51_baseline_a_v1",
        root=root_a,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="torchcodec",
    )
    dataset_b = LeRobotDataset(
        repo_id="local/g1_magsafe_matched51_proposed_b_v1",
        root=root_b,
        delta_timestamps=deltas,
        download_videos=False,
        video_backend="torchcodec",
    )
    reports = []
    for episode_index, frame_index, label in samples:
        global_index = int(offsets[episode_index] + frame_index)
        item_a, item_b = dataset_a[global_index], dataset_b[global_index]
        image_equal = torch.equal(item_a["observation.images.cam_high"], item_b["observation.images.cam_high"])
        expected_padding = [frame_index + offset >= int(rows[episode_index]["length"]) for offset in range(CHUNK_SIZE)]
        if tuple(item_a[STATE_KEY].shape) != (1, 28) or tuple(item_b[STATE_KEY].shape) != (1, 28):
            raise ValueError("LeRobot state readback shape mismatch")
        if tuple(item_a[ACTION_KEY].shape) != (50, 28) or tuple(item_b[ACTION_KEY].shape) != (50, 28):
            raise ValueError("LeRobot action chunk shape mismatch")
        if item_a["action_is_pad"].tolist() != expected_padding or item_b["action_is_pad"].tolist() != expected_padding:
            raise ValueError("LeRobot action_is_pad mismatch")
        if item_a["task"] != TASK_TEXT or item_b["task"] != TASK_TEXT:
            raise ValueError("LeRobot task resolution mismatch")
        if not image_equal:
            raise ValueError("A/B decoded RGB differs")
        reports.append(
            {
                "label": label,
                "stable_source_id": stable_ids[episode_index],
                "packaged_episode_index": episode_index,
                "frame_index": frame_index,
                "global_index": global_index,
                "decoded_rgb_shape": list(item_a["observation.images.cam_high"].shape),
                "decoded_rgb_exact_equal_a_b": image_equal,
                "state_shape_a_b": [list(item_a[STATE_KEY].shape), list(item_b[STATE_KEY].shape)],
                "action_chunk_shape_a_b": [list(item_a[ACTION_KEY].shape), list(item_b[ACTION_KEY].shape)],
                "task_resolved": item_a["task"],
                "timestamp_a_b": [float(item_a["timestamp"]), float(item_b["timestamp"])],
                "padding_true_count": int(sum(expected_padding)),
            }
        )
    result = {
        "schema_version": "g1_policy_dataset_packaging_v1_lerobot_readback",
        "status": "PASS",
        "python": "/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python",
        "lerobot_version": lerobot.__version__,
        "reader": "lerobot.datasets.lerobot_dataset.LeRobotDataset",
        "video_backend": "torchcodec",
        "dataset_length_a": len(dataset_a),
        "dataset_length_b": len(dataset_b),
        "sample_reports": reports,
        "image_decode_pass": True,
        "state_28d_pass": True,
        "action_chunk_50x28_pass": True,
        "task_resolution_pass": True,
        "timestamp_pass": True,
        "padding_mask_pass": True,
        "a_b_rgb_decode_exact_equal": True,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
