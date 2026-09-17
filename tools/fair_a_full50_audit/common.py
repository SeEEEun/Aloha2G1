"""Immutable input paths and small helpers for the Fair-A full-50 audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY / "outputs/fair_a_full50_hard_fail_audit"
A_ROOT = REPOSITORY / "outputs/dataset_a_final50_retargeting"
A_TRAJECTORY_ROOT = A_ROOT / "baseline/trajectories"
A_METRIC_ROOT = A_ROOT / "baseline/metrics"
A_GATE = REPOSITORY / "outputs/pre_mount_readiness/dataset_a/dataset_a_validation.json"
FINAL_SOURCE_MANIFEST = (
    REPOSITORY / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
)
B_FINAL_TRAJECTORY_ROOT = (
    REPOSITORY / "outputs/doll_handoff_dataset_b_final/retargeted_actions/trajectories"
)
SIDES = ("left", "right")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_rows() -> list[dict[str, Any]]:
    rows = list(load_json(FINAL_SOURCE_MANIFEST)["episodes"])
    rows.sort(key=lambda row: int(row["final_dataset_index"]))
    if [int(row["final_dataset_index"]) for row in rows] != list(range(50)):
        raise RuntimeError("final common source manifest is not an exact 0..49 set")
    return rows


def source_row(episode: int) -> dict[str, Any]:
    return source_rows()[int(episode)]


def stable_episode_id(episode: int) -> str:
    return str(source_row(episode)["stable_episode_id"])


def a_trajectory_path(episode: int) -> Path:
    return A_TRAJECTORY_ROOT / f"{stable_episode_id(episode)}.npz"


def a_metric_path(episode: int, suffix: str = "") -> Path:
    return A_METRIC_ROOT / f"{stable_episode_id(episode)}{suffix}.json"


def b_final_trajectory_path(episode: int) -> Path:
    return B_FINAL_TRAJECTORY_ROOT / f"episode_{int(episode):06d}.npz"


def load_a_trajectory(episode: int) -> dict[str, np.ndarray]:
    path = a_trajectory_path(episode)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_npz(path: str | Path, values: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)

