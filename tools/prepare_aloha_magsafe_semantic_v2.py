#!/usr/bin/env python3
"""Freeze v1 and create the semantic-v2 scientific split.

This stage deliberately does not read any v1 semantic timeline.  Development
coverage is selected from manifest duration and raw gripper-signal variation
only.  The resulting split is written before any v2 detector tuning begins.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v2"
MANIFEST = ROOT / "reports/magsafe_lerobot_v3_manifest.csv"
BACKUP_GLOB = "semantic_detector_v1_before_v2_*"
SPLIT_SEED = 20260807


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def manifest_rows() -> list[dict[str, Any]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    for value in source:
        rows.append({
            "episode_id": int(value["output_episode_index"]),
            "source_folder": str(Path(value["source_folder"]).resolve()),
            "source_parquet": str(Path(value["source_parquet"]).resolve()),
            "frame_count": int(value["source_frame_count"]),
            "first_timestamp": float(value["source_first_timestamp"]),
            "last_timestamp": float(value["source_last_timestamp"]),
        })
    ids = [row["episode_id"] for row in rows]
    if len(rows) != 50 or sorted(ids) != list(range(50)) or len(ids) != len(set(ids)):
        raise RuntimeError(f"authoritative manifest mismatch: count={len(rows)} ids={ids}")
    return sorted(rows, key=lambda row: row["episode_id"])


def _column(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.float64)


def raw_selection_features(row: dict[str, Any]) -> dict[str, Any]:
    """Read only raw action/timestamps; no semantic detector output is touched."""
    table = pq.read_table(row["source_parquet"], columns=["action", "timestamp"])
    action = _column(table, "action")
    timestamps = _column(table, "timestamp").reshape(-1)
    if action.ndim != 2 or action.shape[1] != 14 or len(action) != len(timestamps):
        raise RuntimeError(f"invalid raw action shape for episode {row['episode_id']}: {action.shape}")
    dt = np.diff(timestamps)
    duration = float(timestamps[-1] - timestamps[0])
    result: dict[str, Any] = {
        "episode_id": row["episode_id"],
        "frame_count": int(len(action)),
        "duration_sec": duration,
        "frequency_hz_median": float(1.0 / np.median(dt)),
    }
    for side, channel in (("left", 6), ("right", 13)):
        signal = action[:, channel]
        derivative = np.diff(signal) / np.maximum(dt, 1e-12)
        span = float(np.quantile(signal, 0.98) - np.quantile(signal, 0.02))
        result.update({
            f"{side}_gripper_range": span,
            f"{side}_gripper_std": float(np.std(signal)),
            f"{side}_gripper_total_variation": float(np.sum(np.abs(np.diff(signal)))),
            f"{side}_gripper_derivative_q95": float(np.quantile(np.abs(derivative), 0.95)),
        })
    return result


def robust_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = [
        "duration_sec", "frame_count",
        "left_gripper_range", "right_gripper_range",
        "left_gripper_std", "right_gripper_std",
        "left_gripper_total_variation", "right_gripper_total_variation",
        "left_gripper_derivative_q95", "right_gripper_derivative_q95",
    ]
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    median = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=0)
    scale = np.where(q75 - q25 > 1e-12, q75 - q25, np.std(matrix, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - median) / scale, names


def choose_development(rows: list[dict[str, Any]], matrix: np.ndarray) -> list[int]:
    ids = np.asarray([row["episode_id"] for row in rows], dtype=np.int64)
    duration = np.asarray([row["duration_sec"] for row in rows])
    initial = {
        49,
        int(ids[np.argmin(duration)]),
        int(ids[np.argmax(duration)]),
        int(ids[np.argmin(np.abs(duration - np.median(duration)))]),
    }
    selected = sorted(initial)
    while len(selected) < 10:
        selected_rows = [int(np.flatnonzero(ids == episode_id)[0]) for episode_id in selected]
        best: tuple[float, int] | None = None
        for row_index, episode_id in enumerate(ids.tolist()):
            if episode_id in selected:
                continue
            distance = min(float(np.linalg.norm(matrix[row_index] - matrix[other])) for other in selected_rows)
            candidate = (distance, -episode_id)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        selected.append(-best[1])
        selected.sort()
    return selected


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows()
    feature_rows = [raw_selection_features(row) for row in rows]
    matrix, feature_names = robust_matrix(feature_rows)
    development = choose_development(feature_rows, matrix)
    remaining = np.asarray(sorted(set(range(50)) - set(development)), dtype=np.int64)
    generator = np.random.default_rng(SPLIT_SEED)
    validation = sorted(generator.choice(remaining, size=10, replace=False).astype(int).tolist())
    heldout = sorted(set(remaining.tolist()) - set(validation))
    development_duration = {row["episode_id"]: row["duration_sec"] for row in feature_rows if row["episode_id"] in development}
    ordered_dev = sorted(development, key=lambda episode_id: (development_duration[episode_id], episode_id))
    smoke = [ordered_dev[0], ordered_dev[len(ordered_dev) // 2], ordered_dev[-1]]
    split = {
        "protocol": "FROZEN_DEVELOPMENT_VALIDATION_HELDOUT_V2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_manifest": str(MANIFEST.resolve()),
        "authoritative_manifest_sha256": sha256(MANIFEST),
        "selection_input_scope": ["manifest duration", "raw action gripper channels 6 and 13"],
        "semantic_timeline_or_approved_reference_used_for_selection": False,
        "development": development,
        "validation": validation,
        "heldout_test": heldout,
        "development_smoke": smoke,
        "validation_seed": SPLIT_SEED,
        "development_selection": "Episode 49 plus duration extrema/median and deterministic farthest-point coverage of robust duration/gripper features",
        "validation_selection": "fixed-seed sample without replacement from non-development IDs",
        "heldout_selection": "remaining IDs; single evaluation after selected config freeze",
        "checks": {
            "development_count": len(development),
            "validation_count": len(validation),
            "heldout_count": len(heldout),
            "no_overlap": not (set(development) & set(validation) or set(development) & set(heldout) or set(validation) & set(heldout)),
            "union_is_all_ids": sorted(development + validation + heldout) == list(range(50)),
            "episode49_development_only": 49 in development and 49 not in validation and 49 not in heldout,
        },
        "selection_feature_names": feature_names,
        "selection_features_by_episode": feature_rows,
    }
    if not all(split["checks"].values()):
        raise RuntimeError(f"invalid split: {split['checks']}")
    split["split_hash"] = canonical_hash({key: split[key] for key in ("development", "validation", "heldout_test", "development_smoke", "validation_seed")})
    atomic_json(OUT / "dataset_split_v2.json", split)

    latest_backups = sorted((ROOT / "backups").glob(BACKUP_GLOB))
    if not latest_backups:
        raise RuntimeError("required pre-v2 backup is missing")
    backup = latest_backups[-1]
    v1_code = sorted((ROOT / "tools/aloha_magsafe_semantics").glob("*.py"))
    v1_files = v1_code + [
        ROOT / "configs/aloha_magsafe_semantic_detector_v1.json",
        ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/batch_semantic_summary.json",
        ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1/fifty_episode_readiness.json",
    ]
    immutable = [
        ROOT / "configs/episode49_task_timeline.approved.json",
        ROOT / "configs/episode49_action_observation_alignment.approved.json",
        ROOT / "configs/g1_root_forward_v14.approved.json",
        ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz",
        ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/position_only_exact_v14.npz",
        ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/position_only_nullspace_v14.npz",
        ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json",
        ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda",
    ]
    freeze = {
        "status": "V1_FROZEN_BEFORE_V2",
        "backup_directory": str(backup.resolve()),
        "backup_created_before_v2_code": True,
        "v1_files": {str(path.relative_to(ROOT)): sha256(path) for path in v1_files},
        "immutable_v14_scene_files_before": {str(path.relative_to(ROOT)): sha256(path) for path in immutable},
        "v1_authoritative_results": {
            "total_episodes": 50,
            "partial_order_valid": 50,
            "mandatory_event_index_missing": 0,
            "high_medium_complete": 19,
            "complete_with_low_or_ambiguous": 31,
            "episodes_with_ambiguous_mandatory_events": 30,
            "forbidden_runtime_frame_dependencies": 0,
            "approved_reference_independent_detector_hash": "b467ef2bd0307d12de7650b18469698586d267370db2e4f56bba1248af10b377",
        },
    }
    atomic_json(OUT / "v1_freeze_audit.json", freeze)
    atomic_json(OUT / "input_audit.json", {
        "status": "PASS",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "total_episode_count": len(rows),
        "exact_episode_ids": [row["episode_id"] for row in rows],
        "ids_are_zero_through_49": [row["episode_id"] for row in rows] == list(range(50)),
        "mapping": rows,
        "split_hash": split["split_hash"],
        "no_downstream_generation": True,
    })
    print(json.dumps({
        "output": str(OUT),
        "development": development,
        "validation": validation,
        "heldout_test": heldout,
        "smoke": smoke,
        "split_hash": split["split_hash"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
