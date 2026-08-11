"""Generic NPZ and LeRobot parquet trajectory loaders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _arrow_array(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist())


def validate_trajectory(action: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if action.ndim != 2 or action.shape[1] != 14:
        raise ValueError(f"action must have shape [T,14], got {action.shape}")
    if len(action) < 2 or timestamps.shape != (len(action),):
        raise ValueError("timestamps must have one value per action sample")
    if not np.isfinite(action).all() or not np.isfinite(timestamps).all():
        raise ValueError("action/timestamps contain NaN or Inf")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing")
    return action.astype(np.float64, copy=False), timestamps


def load_trajectory(
    path: str | Path,
    source_type: str,
    action_key: str | None = None,
    observation_state: bool = False,
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if source_type not in ("optimized_action", "raw_action"):
        raise ValueError("source_type must be optimized_action or raw_action")
    if path.suffix == ".npz":
        key = action_key or "optimized_action"
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise KeyError(f"{path}: missing NPZ key {key}; keys={archive.files}")
            action = archive[key].copy()
            if "timestamp" in archive.files:
                timestamps = archive["timestamp"].copy()
            elif "timestamps" in archive.files:
                timestamps = archive["timestamps"].copy()
            else:
                fps = float(archive["fps"]) if "fps" in archive.files else 0.0
                if fps <= 0:
                    raise ValueError(f"{path}: timestamp and fps are both absent")
                timestamps = np.arange(len(action), dtype=np.float64) / fps
            state = archive["observation_state"].copy() if observation_state and "observation_state" in archive.files else None
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        columns = [action_key or "action", "timestamp"]
        if observation_state:
            columns.append("observation.state")
        table = pq.read_table(path, columns=columns)
        action = _arrow_array(table, columns[0])
        timestamps = _arrow_array(table, "timestamp").reshape(-1)
        state = _arrow_array(table, "observation.state") if observation_state else None
    else:
        raise ValueError(f"unsupported trajectory file: {path}")
    action, timestamps = validate_trajectory(action, timestamps)
    if state is not None:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != action.shape or not np.isfinite(state).all():
            raise ValueError("observation.state must be finite and match action shape")
    dt = np.diff(timestamps)
    return {
        "action": action,
        "timestamps": timestamps,
        "observation_state": state,
        "source_type": source_type,
        "path": str(path),
        "file_sha256": sha256_file(path),
        "trajectory_hash": sha256_array(action),
        "frames": int(len(action)),
        "frequency_hz_median": float(1.0 / np.median(dt)),
    }


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".incomplete")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)

