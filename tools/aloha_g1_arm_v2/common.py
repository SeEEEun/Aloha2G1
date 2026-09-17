"""Shared paths, integrity contracts, and deterministic serialization."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aloha_g1_dataset_v1.core import array_sha256, json_default, sha256_file


ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = ROOT / "outputs/g1_dataset_retargeting_v1"
HAND_ROOT = ROOT / "outputs/g1_dataset_retargeting_hand_v2_1"
DEFAULT_ARM_ROOT = ROOT / "outputs/g1_dataset_retargeting_arm_v2"
DEFAULT_INTEGRATED_ROOT = ROOT / "outputs/g1_dataset_retargeting_integrated_v2"
DEFAULT_SEARCH_CONFIG = ROOT / "configs/aloha_g1_arm_v2.json"


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    names = fieldnames or (list(rows[0]) if rows else [])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def tree_sha256(root: str | Path) -> tuple[str, dict[str, str]]:
    root = Path(root).resolve()
    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), files


def load_search_config(path: str | Path = DEFAULT_SEARCH_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    value = load_json(path)
    if value.get("schema_version") != "aloha_g1_common_arm_v2_search":
        raise ValueError(f"unexpected Common Arm-v2 config schema: {path}")
    if not value.get("offline_only") or value.get("real_robot_command_allowed"):
        raise ValueError("Common Arm-v2 must remain offline-only")
    if value.get("training_allowed") or value.get("physics_sweep_allowed"):
        raise ValueError("training and physics sweep must be disabled")
    return value


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def deterministic_split(
    episode_ids: Iterable[int], salt: str, validation_count: int
) -> dict[str, Any]:
    scores: list[tuple[int, int, str]] = []
    for episode_id in sorted(int(value) for value in episode_ids):
        hexdigest = hashlib.sha256(f"{salt}:{episode_id}".encode("utf-8")).hexdigest()
        scores.append((int(hexdigest[:16], 16), episode_id, hexdigest))
    ordered = sorted(scores)
    validation = sorted(row[1] for row in ordered[:validation_count])
    calibration = sorted(row[1] for row in ordered[validation_count:])
    return {
        "algorithm": "SHA-256(salt + ':' + decimal episode ID); lowest digest prefixes are validation",
        "salt": salt,
        "calibration_episode_ids": calibration,
        "validation_episode_ids": validation,
        "episode_hashes": {
            str(episode_id): hexdigest for _, episode_id, hexdigest in sorted(scores, key=lambda row: row[1])
        },
    }


def dependency_paths(search: dict[str, Any]) -> dict[str, Path]:
    hand = search["hand_v2_1"]
    return {
        "candidate": resolve_from_root(hand["candidate"]),
        "readiness": resolve_from_root(hand["readiness"]),
        "tool_compatibility": resolve_from_root(hand["tool_compatibility"]),
    }


def validate_hand_dependency(search: dict[str, Any]) -> dict[str, Any]:
    paths = dependency_paths(search)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"PROPOSED_HAND_V2_1_DEPENDENCY_INVALID: missing {missing}")
    candidate = load_json(paths["candidate"])
    readiness = load_json(paths["readiness"])
    compatibility = load_json(paths["tool_compatibility"])
    required = search["hand_v2_1"]["required_conclusion"]
    checks = {
        "candidate_schema": candidate.get("schema_version") == "proposed_hand_v2_1_candidate",
        "candidate_scope": candidate.get("method_scope") == "Proposed/Dataset B only",
        "candidate_not_direct_default": candidate.get("diagnostic_exact_contact_ik", {}).get("default") is False,
        "readiness_conclusion": readiness.get("conclusion") == required,
        "ready_for_common_arm": readiness.get("ready_for_common_arm_rerun") is True,
        "tool_rerun_required": compatibility.get("tool_transform_changed_requires_arm_rerun") is True,
        "candidate_left_tool_matches": np.array_equal(
            np.asarray(candidate.get("left_static_wrist_to_pinch")),
            np.asarray(compatibility.get("sides", {}).get("left", {}).get("candidate_wrist_to_pinch")),
        ),
        "candidate_right_tool_matches": np.array_equal(
            np.asarray(candidate.get("right_static_wrist_to_pinch")),
            np.asarray(compatibility.get("sides", {}).get("right", {}).get("candidate_wrist_to_pinch")),
        ),
    }
    for side in ("left", "right"):
        states = candidate.get("states", {}).get(side, {})
        checks[f"{side}_five_states"] = set(states) == {
            "OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE"
        }
        checks[f"{side}_states_finite_shape"] = all(
            np.asarray(value).shape == (7,) and np.isfinite(value).all()
            for value in states.values()
        )
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(
            "PROPOSED_HAND_V2_1_DEPENDENCY_INVALID: failed checks " + str(failed)
        )
    return {
        "status": "PROPOSED_HAND_V2_1_DEPENDENCY_VALID",
        "checks": checks,
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": {key: sha256_file(value) for key, value in paths.items()},
        "candidate": candidate,
        "readiness": readiness,
        "tool_compatibility": compatibility,
    }


def freeze_hand_dependency(
    dependency: dict[str, Any], output_root: str | Path
) -> dict[str, Any]:
    output = Path(output_root).resolve() / "dependencies"
    output.mkdir(parents=True, exist_ok=True)
    source = {key: Path(value) for key, value in dependency["paths"].items()}
    destinations = {
        "candidate": output / "proposed_hand_v2_1_candidate.json",
        "readiness": output / "proposed_hand_v2_1_readiness.json",
    }
    for key, destination in destinations.items():
        temporary = destination.with_suffix(destination.suffix + ".incomplete")
        shutil.copyfile(source[key], temporary)
        os.replace(temporary, destination)
    frozen = {
        "status": "IMMUTABLE_DEPENDENCY_FROZEN",
        "source_sha256": dependency["sha256"],
        "frozen_files": {key: str(path) for key, path in destinations.items()},
        "frozen_sha256": {key: sha256_file(path) for key, path in destinations.items()},
        "byte_identical": {
            key: sha256_file(destinations[key]) == dependency["sha256"][key]
            for key in destinations
        },
    }
    if not all(frozen["byte_identical"].values()):
        raise RuntimeError("PROPOSED_HAND_V2_1_DEPENDENCY_INVALID: frozen copy differs")
    atomic_json(output / "dependency_manifest.json", frozen)
    return frozen


def source_integrity(base_config: dict[str, Any]) -> dict[str, Any]:
    source = base_config["source_dataset"]
    root = Path(source["root"]).resolve()
    info = root / "meta/info.json"
    data = root / "data/chunk-000/file-000.parquet"
    actual = {"info_sha256": sha256_file(info), "data_sha256": sha256_file(data)}
    expected = {
        "info_sha256": source["info_sha256"],
        "data_sha256": source["data_sha256"],
    }
    return {
        "dataset_root": str(root),
        "expected": expected,
        "actual": actual,
        "unchanged": actual == expected,
    }


def combine_action_hashes(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as payload:
            value = array_sha256(payload["action"])
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def scalar_stats(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if len(data) == 0:
        return {key: 0.0 for key in ("mean", "median", "min", "max", "p95")}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p95": float(np.percentile(data, 95)),
    }


__all__ = [
    "DEFAULT_ARM_ROOT",
    "DEFAULT_INTEGRATED_ROOT",
    "DEFAULT_SEARCH_CONFIG",
    "HAND_ROOT",
    "ROOT",
    "V1_ROOT",
    "array_sha256",
    "atomic_csv",
    "atomic_json",
    "atomic_npz",
    "combine_action_hashes",
    "dependency_paths",
    "deterministic_split",
    "freeze_hand_dependency",
    "load_json",
    "load_search_config",
    "resolve_from_root",
    "scalar_stats",
    "sha256_file",
    "source_integrity",
    "tree_sha256",
    "validate_hand_dependency",
]
