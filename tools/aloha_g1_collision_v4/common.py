"""Frozen dependencies and deterministic serialization for collision-v4."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from aloha_g1_arm_v2.common import (
    ROOT,
    atomic_csv,
    atomic_json,
    atomic_npz,
    load_json,
    sha256_file,
    source_integrity,
)


V3_ROOT = ROOT / "outputs/g1_dataset_feasibility_v3"
ARM_V2_ROOT = ROOT / "outputs/g1_dataset_retargeting_arm_v2"
HAND_V2_1_ROOT = ROOT / "outputs/g1_dataset_retargeting_hand_v2_1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/g1_dataset_collision_v4"
DEFAULT_CONFIG = ROOT / "configs/aloha_g1_collision_v4.json"
METHOD_TO_DATASET = {"baseline": "dataset_a", "proposed": "dataset_b"}
DATASET_TO_METHOD = {value: key for key, value in METHOD_TO_DATASET.items()}
METHODS = tuple(METHOD_TO_DATASET)
SIDES = ("left", "right")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = load_json(Path(path))
    if value.get("schema_version") != "aloha_g1_collision_v4_search":
        raise ValueError("unexpected collision-v4 configuration schema")
    if not value.get("offline_only", False):
        raise ValueError("collision-v4 must remain offline-only")
    for key in (
        "training_allowed",
        "physics_sweep_allowed",
        "real_robot_command_allowed",
        "global_mapping_change_allowed",
        "hand_change_allowed",
        "acceptance_threshold_change_allowed",
    ):
        if value.get(key, False):
            raise ValueError(f"forbidden collision-v4 option enabled: {key}")
    return value


def _array_digest(digest: Any, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())


def dataset_definition_sha256(dataset_name: str) -> str:
    """Hash frozen representation targets and hand semantics, excluding arm q."""
    digest = hashlib.sha256()
    arm_keys = (
        "target_left_wrist_position",
        "target_right_wrist_position",
        "target_left_wrist_rotation",
        "target_right_wrist_rotation",
        "target_left_task_tool_position",
        "target_right_task_tool_position",
        "representation",
        "method",
    )
    hand_keys = (
        "left_action",
        "right_action",
        "left_phase",
        "right_phase",
        "mapper",
    )
    for episode_id in range(50):
        digest.update(f"episode:{episode_id}".encode("ascii"))
        directory = V3_ROOT / dataset_name / f"episode_{episode_id:06d}"
        with np.load(directory / "g1_arm_action.npz", allow_pickle=False) as payload:
            for key in arm_keys:
                digest.update(key.encode("ascii"))
                _array_digest(digest, payload[key])
        with np.load(directory / "g1_hand_action.npz", allow_pickle=False) as payload:
            for key in hand_keys:
                digest.update(key.encode("ascii"))
                _array_digest(digest, payload[key])
    return digest.hexdigest()


def freeze_dependencies(output_root: Path) -> dict[str, Any]:
    paths = {
        "common_arm_v2": ARM_V2_ROOT
        / "candidates/frozen_common_arm_v2_config.json",
        "feasibility_v3": V3_ROOT / "solver/frozen_feasibility_v3_config.json",
        "hand_v2_1": HAND_V2_1_ROOT
        / "config/proposed_hand_v2_1_candidate.json",
        "split": ARM_V2_ROOT / "split/calibration_validation_split.json",
        "collision_semantics": V3_ROOT
        / "audit/collision_gate_semantics.json",
        "v3_readiness": V3_ROOT / "summary/training_readiness.json",
        "v3_integrity": V3_ROOT / "summary/integrity.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    arm = load_json(paths["common_arm_v2"])
    feasibility = load_json(paths["feasibility_v3"])
    hand = load_json(paths["hand_v2_1"])
    split = load_json(paths["split"])
    readiness = load_json(paths["v3_readiness"])
    integrity = load_json(paths["v3_integrity"])
    semantics = load_json(paths["collision_semantics"])
    if readiness.get("conclusion") != "COMMON_FEASIBILITY_V3_READY_FOR_TRAINING_SCHEMA":
        raise RuntimeError("Feasibility-v3 dependency is not accepted")
    if not all(
        integrity.get(key, False)
        for key in (
            "common_arm_v2_unchanged",
            "hand_v2_1_unchanged",
            "source_hashes_unchanged",
            "split_byte_identical",
        )
    ):
        raise RuntimeError("Feasibility-v3 integrity contract is not satisfied")
    if semantics.get("gate_change") != "NONE":
        raise RuntimeError("collision-v4 requires unchanged audited v3 semantics")
    if len(split["calibration_episode_ids"]) != 40 or len(
        split["validation_episode_ids"]
    ) != 10:
        raise RuntimeError("frozen 40/10 split is invalid")
    expected = feasibility["solver_parameters"]
    if not np.isclose(float(expected["orientation_slack_bound_rad"]), 0.65):
        raise RuntimeError("unexpected Feasibility-v3 slack bound")

    destination = output_root / "dependencies"
    destination.mkdir(parents=True, exist_ok=True)
    copies = {
        "frozen_common_arm_v2_config.json": paths["common_arm_v2"],
        "frozen_feasibility_v3_config.json": paths["feasibility_v3"],
        "frozen_proposed_hand_v2_1.json": paths["hand_v2_1"],
        "frozen_calibration_validation_split.json": paths["split"],
        "frozen_collision_gate_semantics.json": paths["collision_semantics"],
    }
    copy_checks: dict[str, Any] = {}
    for name, source in copies.items():
        target = destination / name
        shutil.copy2(source, target)
        copy_checks[name] = {
            "source": str(source),
            "copy": str(target),
            "source_sha256": sha256_file(source),
            "copy_sha256": sha256_file(target),
            "byte_identical": sha256_file(source) == sha256_file(target),
        }

    source = source_integrity(arm["runtime_config"])
    definitions = {
        dataset_name: dataset_definition_sha256(dataset_name)
        for dataset_name in ("dataset_a", "dataset_b")
    }
    checksums = {
        "schema_version": "common_collision_v4_dependency_checksums",
        "copies": copy_checks,
        "source_dataset": source,
        "dataset_definition_sha256": definitions,
        "global_mapping_sha256": sha256_file(paths["common_arm_v2"]),
        "feasibility_v3_solver_sha256": sha256_file(paths["feasibility_v3"]),
        "hand_v2_1_sha256": sha256_file(paths["hand_v2_1"]),
        "collision_semantics_sha256": sha256_file(paths["collision_semantics"]),
        "split_sha256": sha256_file(paths["split"]),
        "all_valid": bool(
            source["unchanged"]
            and all(row["byte_identical"] for row in copy_checks.values())
        ),
    }
    atomic_json(destination / "dependency_checksums.json", checksums)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "arm": arm,
        "feasibility": feasibility,
        "hand": hand,
        "split": split,
        "semantics": semantics,
        "readiness": readiness,
        "integrity": integrity,
        "checksums": checksums,
    }


def stable_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "median", "min", "max", "p95")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
    }


__all__ = [
    "ROOT",
    "V3_ROOT",
    "ARM_V2_ROOT",
    "HAND_V2_1_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_CONFIG",
    "METHOD_TO_DATASET",
    "DATASET_TO_METHOD",
    "METHODS",
    "SIDES",
    "atomic_csv",
    "atomic_json",
    "atomic_npz",
    "load_json",
    "sha256_file",
    "source_integrity",
    "load_config",
    "freeze_dependencies",
    "dataset_definition_sha256",
    "stable_stats",
]
