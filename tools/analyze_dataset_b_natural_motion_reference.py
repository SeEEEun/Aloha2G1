#!/usr/bin/env python3
"""Freeze Dataset-B action-motion statistics without crossing episode boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_causal_execution/dataset_b_motion_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not flat.size:
        return {"count": 0, "median": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "count": int(flat.size),
        "median": float(np.median(flat)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "maximum": float(np.max(flat)),
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    info_path = dataset / "meta/info.json"
    data_path = dataset / "data/chunk-000/file-000.parquet"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    names = list(info["features"]["action"]["names"])
    if len(names) != 28 or len(set(names)) != 28:
        raise RuntimeError("Dataset-B must expose 28 unique named action joints")
    table = pq.read_table(data_path, columns=["action", "episode_index", "frame_index", "timestamp"])
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    episode = np.asarray(table["episode_index"], dtype=np.int64)
    frame = np.asarray(table["frame_index"], dtype=np.int64)
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)
    if action.shape != (34478, 28) or len(np.unique(episode)) != 50:
        raise RuntimeError(f"Unexpected frozen Dataset-B shape {action.shape} / episodes {len(np.unique(episode))}")
    if not np.isfinite(action).all():
        raise RuntimeError("Dataset-B action contains NaN/Inf")

    step_parts: list[np.ndarray] = []
    velocity_parts: list[np.ndarray] = []
    acceleration_parts: list[np.ndarray] = []
    jerk_parts: list[np.ndarray] = []
    episode_lengths: list[int] = []
    for episode_index in sorted(np.unique(episode).tolist()):
        mask = episode == episode_index
        q = action[mask]
        f = frame[mask]
        t = timestamp[mask]
        episode_lengths.append(len(q))
        if not np.array_equal(f, np.arange(len(q), dtype=np.int64)):
            raise RuntimeError(f"Episode {episode_index} frame indices are not contiguous")
        if len(t) > 1 and not np.allclose(np.diff(t), 1.0 / fps, atol=2e-6, rtol=0.0):
            raise RuntimeError(f"Episode {episode_index} timestamps do not match {fps:g} Hz")
        step = np.abs(np.diff(q, axis=0))
        velocity = step * fps
        acceleration = np.abs(np.diff(q, n=2, axis=0)) * fps**2
        jerk = np.abs(np.diff(q, n=3, axis=0)) * fps**3
        step_parts.append(step)
        velocity_parts.append(velocity)
        acceleration_parts.append(acceleration)
        jerk_parts.append(jerk)

    step = np.concatenate(step_parts, axis=0)
    velocity = np.concatenate(velocity_parts, axis=0)
    acceleration = np.concatenate(acceleration_parts, axis=0)
    jerk = np.concatenate(jerk_parts, axis=0)
    rows: list[dict[str, Any]] = []
    for joint_index, name in enumerate(names):
        rows.append(
            {
                "joint_index": joint_index,
                "joint": name,
                "adjacent_step_rad": stats(step[:, joint_index]),
                "velocity_rad_s": stats(velocity[:, joint_index]),
                "acceleration_rad_s2": stats(acceleration[:, joint_index]),
                "jerk_rad_s3": stats(jerk[:, joint_index]),
            }
        )

    per_frame_max = {
        "adjacent_step_rad": stats(np.max(step, axis=1)),
        "velocity_rad_s": stats(np.max(velocity, axis=1)),
        "acceleration_rad_s2": stats(np.max(acceleration, axis=1)),
        "jerk_rad_s3": stats(np.max(jerk, axis=1)),
    }
    scalar = {
        "adjacent_step_rad": stats(step),
        "velocity_rad_s": stats(velocity),
        "acceleration_rad_s2": stats(acceleration),
        "jerk_rad_s3": stats(jerk),
    }
    report = {
        "schema_version": "dataset_b_natural_motion_reference_v1",
        "status": "PASS",
        "dataset": str(dataset),
        "dataset_info_sha256": sha256_file(info_path),
        "dataset_parquet_sha256": sha256_file(data_path),
        "episodes": int(len(np.unique(episode))),
        "frames": int(len(action)),
        "fps": fps,
        "episode_lengths": stats(np.asarray(episode_lengths, dtype=np.float64)),
        "episode_boundary_handling": "All finite differences are computed independently within each episode; no boundary is crossed.",
        "absolute_value_statistics": True,
        "joint_names": names,
        "scalar_across_all_frames_and_joints": scalar,
        "per_frame_maximum_across_joints": per_frame_max,
        "per_joint": rows,
        "use_for_stitching_acceptance": {
            "primary_boundary_metric": "per-frame maximum absolute adjacent joint step",
            "reference_p95_rad": per_frame_max["adjacent_step_rad"]["p95"],
            "reference_p99_rad": per_frame_max["adjacent_step_rad"]["p99"],
            "reference_maximum_rad": per_frame_max["adjacent_step_rad"]["maximum"],
            "acceleration_reference_p95_rad_s2": per_frame_max["acceleration_rad_s2"]["p95"],
            "acceleration_reference_p99_rad_s2": per_frame_max["acceleration_rad_s2"]["p99"],
            "acceleration_reference_maximum_rad_s2": per_frame_max["acceleration_rad_s2"]["maximum"],
        },
    }
    atomic_json(output / "dataset_b_natural_motion_reference.json", report)
    csv_path = output / "per_joint_motion_reference.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["joint_index", "joint"]
        for metric in ("adjacent_step_rad", "velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3"):
            for statistic in ("median", "p95", "p99", "maximum"):
                fieldnames.append(f"{metric}_{statistic}")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {"joint_index": row["joint_index"], "joint": row["joint"]}
            for metric in ("adjacent_step_rad", "velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3"):
                for statistic in ("median", "p95", "p99", "maximum"):
                    flat[f"{metric}_{statistic}"] = row[metric][statistic]
            writer.writerow(flat)
    os.replace(temporary, csv_path)
    print(json.dumps(report["use_for_stitching_acceptance"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
