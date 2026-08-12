"""Immutable dependencies and deterministic serialization for feasibility v3."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

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


V2_ARM_ROOT = ROOT / "outputs/g1_dataset_retargeting_arm_v2"
V2_INTEGRATED_ROOT = ROOT / "outputs/g1_dataset_retargeting_integrated_v2"
HAND_ROOT = ROOT / "outputs/g1_dataset_retargeting_hand_v2_1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/g1_dataset_feasibility_v3"
DEFAULT_CONFIG = ROOT / "configs/aloha_g1_feasibility_v3.json"
METHOD_TO_DATASET = {"baseline": "dataset_a", "proposed": "dataset_b"}
DATASET_TO_METHOD = {value: key for key, value in METHOD_TO_DATASET.items()}


def load_search_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    value = load_json(path)
    if value.get("schema_version") != "aloha_g1_common_feasibility_v3_search":
        raise ValueError(f"unexpected feasibility-v3 schema: {path}")
    if not value.get("offline_only") or value.get("real_robot_command_allowed"):
        raise ValueError("feasibility-v3 must remain offline-only")
    if value.get("training_allowed") or value.get("physics_sweep_allowed"):
        raise ValueError("training and physics sweep must remain disabled")
    if value["shared_solver"].get("method_specific_parameters_allowed"):
        raise ValueError("method-specific feasibility parameters are forbidden")
    return value


def resolve_dependency(search: Mapping[str, Any], key: str) -> Path:
    value = Path(search["dependencies"][key])
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _representation_contract(method: str) -> dict[str, Any]:
    folder = V2_ARM_ROOT / method / "episode_000000"
    with np.load(folder / "g1_arm_action.npz", allow_pickle=False) as payload:
        representation = str(payload["representation"])
        selected_sha = str(payload["selected_config_sha256"])
    return {
        "method": method,
        "representation": representation,
        "selected_common_arm_config_sha256": selected_sha,
        "target_arrays": [
            "left_wrist_position",
            "right_wrist_position",
            "left_wrist_rotation",
            "right_wrist_rotation",
            "left_task_tool_position",
            "right_task_tool_position",
        ],
        "source": str(folder / "g1_arm_action.npz"),
    }


def freeze_dependencies(
    search: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    arm_source = resolve_dependency(search, "common_arm_v2")
    hand_source = resolve_dependency(search, "proposed_hand_v2_1")
    readiness_source = resolve_dependency(search, "proposed_hand_v2_1_readiness")
    split_source = resolve_dependency(search, "common_arm_v2_split")
    first_source = resolve_dependency(search, "common_arm_v2_first_failures")
    for path in (arm_source, hand_source, readiness_source, split_source, first_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    arm = load_json(arm_source)
    hand = load_json(hand_source)
    readiness = load_json(readiness_source)
    split = load_json(split_source)
    frozen_contract = search["frozen_upstream_contract"]
    mapping = arm["global_mapping"]
    if not np.isclose(
        float(mapping["uniform_scale"]), float(frozen_contract["uniform_scale"])
    ):
        raise RuntimeError("Common Arm-v2 scale differs from frozen v3 contract")
    if not np.isclose(
        float(mapping["task_ready_anchor_interpolation_fraction"]),
        float(frozen_contract["task_ready_anchor_fraction"]),
    ):
        raise RuntimeError("Common Arm-v2 anchor differs from frozen v3 contract")
    if arm.get("status") != "IMMUTABLE_AFTER_CALIBRATION_SELECTION":
        raise RuntimeError("Common Arm-v2 dependency is not frozen")
    if readiness.get("conclusion") != "PROPOSED_HAND_V2_1_READY_FOR_COMMON_ARM_RERUN":
        raise RuntimeError("Hand-v2.1 dependency is not ready")
    if len(split["calibration_episode_ids"]) != 40 or len(split["validation_episode_ids"]) != 10:
        raise RuntimeError("Common Arm-v2 40/10 split contract is invalid")
    destination = output_root / "dependencies"
    destination.mkdir(parents=True, exist_ok=True)
    frozen_arm = destination / "frozen_common_arm_v2_config.json"
    frozen_hand = destination / "frozen_proposed_hand_v2_1.json"
    shutil.copy2(arm_source, frozen_arm)
    shutil.copy2(hand_source, frozen_hand)
    representations = {
        method: _representation_contract(method)
        for method in ("baseline", "proposed")
    }
    source = source_integrity(arm["runtime_config"])
    arm_identical = sha256_file(frozen_arm) == sha256_file(arm_source)
    hand_identical = sha256_file(frozen_hand) == sha256_file(hand_source)
    checksums = {
        "schema_version": "common_feasibility_v3_dependency_checksums",
        "sources": {
            "common_arm_v2": {
                "path": str(arm_source),
                "sha256": sha256_file(arm_source),
            },
            "proposed_hand_v2_1": {
                "path": str(hand_source),
                "sha256": sha256_file(hand_source),
            },
            "hand_readiness": {
                "path": str(readiness_source),
                "sha256": sha256_file(readiness_source),
            },
            "split": {"path": str(split_source), "sha256": sha256_file(split_source)},
            "first_failures": {
                "path": str(first_source),
                "sha256": sha256_file(first_source),
            },
        },
        "frozen_copies": {
            "common_arm_v2": {
                "path": str(frozen_arm),
                "sha256": sha256_file(frozen_arm),
                "byte_identical": arm_identical,
            },
            "proposed_hand_v2_1": {
                "path": str(frozen_hand),
                "sha256": sha256_file(frozen_hand),
                "byte_identical": hand_identical,
            },
        },
        "source_dataset": source,
        "representations": representations,
        "nominal_g1_arm_q": arm["global_mapping"]["nominal_g1_arm_q"],
        "global_mapping": copy.deepcopy(arm["global_mapping"]),
        "all_dependencies_valid": bool(source["unchanged"] and arm_identical and hand_identical),
    }
    atomic_json(destination / "dependency_checksums.json", checksums)
    return {
        "arm": arm,
        "hand": hand,
        "readiness": readiness,
        "split": split,
        "first_failures": load_json(first_source),
        "checksums": checksums,
        "paths": {
            "arm_source": arm_source,
            "hand_source": hand_source,
            "frozen_arm": frozen_arm,
            "frozen_hand": frozen_hand,
        },
    }


def implementation_fingerprint(paths: list[Path]) -> tuple[str, dict[str, str]]:
    values = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), values
