"""Paths, integrity snapshots, and serialization for Proposed hand v2.1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aloha_g1_hand_v2.common import (
    ROOT,
    V1_ROOT,
    V2_ROOT,
    array_sha256,
    atomic_csv,
    atomic_json,
    json_default,
    sha256_file,
    tree_sha256,
)


V2_1_ROOT = ROOT / "outputs/g1_dataset_retargeting_hand_v2_1"
SOURCE_CONFIG = ROOT / "configs/aloha_g1_hand_v2_1.json"


def load_source_config() -> dict[str, Any]:
    return json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))


def integrity_snapshot() -> dict[str, Any]:
    """Hash immutable Dataset A and every frozen Proposed arm artifact."""
    dataset_a_hash, dataset_a_files = tree_sha256(V1_ROOT / "baseline")
    arm_files: dict[str, dict[str, str]] = {}
    for episode_id in range(50):
        path = V1_ROOT / "proposed" / f"episode_{episode_id:06d}" / "g1_arm_action.npz"
        with np.load(path, allow_pickle=False) as payload:
            action_hash = array_sha256(payload["action"])
        arm_files[f"episode_{episode_id:06d}"] = {
            "file_sha256": sha256_file(path),
            "action_array_sha256": action_hash,
        }
    digest = hashlib.sha256()
    for episode, values in sorted(arm_files.items()):
        digest.update(episode.encode("ascii"))
        digest.update(values["file_sha256"].encode("ascii"))
        digest.update(values["action_array_sha256"].encode("ascii"))
    return {
        "dataset_a_root": str(V1_ROOT / "baseline"),
        "dataset_a_tree_sha256": dataset_a_hash,
        "dataset_a_file_count": len(dataset_a_files),
        "proposed_arm_combined_sha256": digest.hexdigest(),
        "proposed_arm_files": arm_files,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


__all__ = [
    "ROOT",
    "V1_ROOT",
    "V2_ROOT",
    "V2_1_ROOT",
    "SOURCE_CONFIG",
    "array_sha256",
    "atomic_csv",
    "atomic_json",
    "integrity_snapshot",
    "json_default",
    "load_source_config",
    "markdown_table",
    "sha256_file",
]
