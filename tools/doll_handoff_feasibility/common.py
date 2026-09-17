"""Shared paths, integrity checks, and numerical helpers for the feasibility audit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY = Path(__file__).resolve().parents[2]
FROZEN_ROOT = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
)
OUTPUT_ROOT = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/g1_feasibility_resolver_2026-08-21"
)
MOTION_FREEZE = (
    FROZEN_ROOT
    / "review/dataset_b_gate/motion_freeze/motion_freeze_manifest.json"
)
EXPECTED_FROZEN_HASHES = {
    "implementation_sha256": (
        "36fd0c2dfda0a5b0b0ed8b63a88f5d311d7267eaa1479d2d70b97dd8dd0f093a"
    ),
    "trajectory_file_set_sha256": (
        "c46c619afe83cb955564a9f6792eb9737fa4eca37ea158307b76a266b79f1da8"
    ),
    "cartesian_target_array_set_sha256": (
        "76bdca77b78e0b2f4b86e40bbd3a12a4984ea409aa382caa46fe954e59abb792"
    ),
}
SIDES = ("left", "right")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def verify_frozen_contract() -> dict[str, Any]:
    if OUTPUT_ROOT.resolve() == FROZEN_ROOT.resolve():
        raise RuntimeError("resolver output must not overwrite the frozen BEFORE root")
    freeze = load_json(MOTION_FREEZE)
    if freeze.get("status") != "PROPOSED_B_MOTION_FROZEN_FOR_DATASET_AUDIT":
        raise RuntimeError("unexpected Proposed-B motion-freeze status")
    for key, expected in EXPECTED_FROZEN_HASHES.items():
        actual = str(freeze.get(key))
        if actual != expected:
            raise RuntimeError(
                f"frozen contract mismatch for {key}: {actual} != {expected}"
            )
    if int(freeze.get("frozen_episode_count", -1)) != 50:
        raise RuntimeError("frozen motion does not contain exactly 50 episodes")
    if float(freeze.get("handoff_cartesian_residual_m", float("nan"))) != 0.0:
        raise RuntimeError("frozen handoff Cartesian residual is not zero")
    return freeze


def stable_episode_id(episode: int) -> str:
    return f"doll_handoff_20260820_ep{int(episode):03d}"


def trajectory_path(episode: int) -> Path:
    return FROZEN_ROOT / "proposed/trajectories" / f"{stable_episode_id(episode)}.npz"


def metric_path(episode: int) -> Path:
    return FROZEN_ROOT / "proposed/metrics" / f"{stable_episode_id(episode)}.json"


def load_trajectory(episode: int) -> dict[str, np.ndarray]:
    path = trajectory_path(episode)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"refusing to write headerless empty CSV: {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def contiguous_segments(frames: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(int(value) for value in frames))
    if not ordered:
        return []
    output: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            output.append((start, previous))
            start = frame
        previous = frame
    output.append((start, previous))
    return output


def rotation_error_rad(actual: np.ndarray, desired: np.ndarray) -> float:
    relative = np.asarray(actual, dtype=np.float64).T @ np.asarray(
        desired, dtype=np.float64
    )
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return xyzw[[3, 0, 1, 2]]


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def scalar_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: float("nan") for name in ("mean", "median", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
