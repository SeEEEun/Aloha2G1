"""Portable bundle loading and atomic JSON helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import json_default, resolve_path, validate_unique_episode_ids
from .metrics import aggregate_episode_metrics, evaluate_episode


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "paper_evaluation_bundle_v1":
        raise ValueError(f"unsupported evaluation bundle: {path}")
    entries = list(manifest.get("episodes", []))
    validate_unique_episode_ids(entries)
    episodes: list[dict[str, Any]] = []
    for entry in entries:
        arrays_path = resolve_path(path.parent, entry["arrays_path"])
        if not arrays_path.is_file():
            raise FileNotFoundError(arrays_path)
        episodes.append(
            {
                "source_episode_id": entry["source_episode_id"],
                "fps": entry.get("fps", manifest.get("fps", 30.0)),
                "evaluation_mode": entry.get(
                    "evaluation_mode", manifest.get("evaluation_mode")
                ),
                "success_kind": entry.get("success_kind", manifest.get("success_kind")),
                "arrays": load_npz(arrays_path),
                "annotations": dict(entry.get("annotations", {})),
                "feasibility": entry.get("feasibility"),
                "smoothness": dict(entry.get("smoothness", manifest.get("smoothness", {}))),
                "provenance": {
                    **dict(manifest.get("provenance", {})),
                    **dict(entry.get("provenance", {})),
                    "arrays_path": str(arrays_path),
                },
            }
        )
    return manifest, episodes


def evaluate_bundle(path: Path, joint_ranges_rad: Any | None) -> dict[str, Any]:
    manifest, episodes = load_bundle(path)
    rows = [evaluate_episode(row, joint_ranges_rad=joint_ranges_rad) for row in episodes]
    return {
        "schema_version": "paper_evaluation_result_v1",
        "status": "READY",
        "evaluation_mode": manifest.get("evaluation_mode"),
        "success_kind": manifest.get("success_kind"),
        "source_bundle": str(Path(path).resolve()),
        "episode_count": len(rows),
        "source_episode_ids": [row["source_episode_id"] for row in rows],
        "episodes": rows,
        "aggregate": aggregate_episode_metrics(rows),
    }


def result_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = list(payload.get("episodes", []))
    validate_unique_episode_ids(rows)
    return rows
