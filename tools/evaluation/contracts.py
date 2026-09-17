"""Shared data contracts and authoritative project references.

An evaluation bundle is a JSON manifest with one entry per source episode::

    {
      "schema_version": "paper_evaluation_bundle_v1",
      "evaluation_mode": "retargeting|offline_act|source_conditioned|physical",
      "success_kind": "SEMANTIC_SUCCESS|PHYSICAL_SUCCESS|null",
      "episodes": [{
        "source_episode_id": "stable exact identity",
        "fps": 30.0,
        "arrays_path": "episode.npz",
        "annotations": {...},
        "feasibility": {...},
        "provenance": {...}
      }]
    }

Array keys are optional so one evaluator can score partial artifacts without
inventing unavailable measurements.  Supported keys are documented in
``outputs/paper_metrics/contracts/evaluation_bundle_v1.json``.  A metric whose
required input is absent is returned as ``status=NA`` with a reason.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
FPS = 30.0
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_RESAMPLES = 10_000

SEMANTIC_SUCCESS = "SEMANTIC_SUCCESS"
PHYSICAL_SUCCESS = "PHYSICAL_SUCCESS"
SUCCESS_KINDS = {SEMANTIC_SUCCESS, PHYSICAL_SUCCESS}

CANONICAL_PHASES = (
    "LEFT_APPROACH",
    "LEFT_GRASP",
    "LEFT_TRANSPORT",
    "RIGHT_APPROACH",
    "DUAL_CONTACT",
    "RIGHT_OWNED",
    "RIGHT_TRANSPORT",
    "RELEASE",
)

AUTHORITATIVE_REFERENCES = {
    "source_episode_manifest": ROOT
    / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json",
    "joint_ranges": ROOT
    / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json",
    "whole_hand_frame": ROOT
    / "outputs/dataset_a_final50_retargeting/config/tool_frame_report.json",
    "whole_hand_geometry": ROOT
    / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json",
    "scene_layout": ROOT / "isaaclab_doll_handoff_scene/scene_layout.json",
    "feasibility_config": ROOT / "configs/doll_handoff_g1_feasibility_resolver.json",
}


def na(reason: str) -> dict[str, str]:
    """Return a JSON-safe, explicit unavailable value."""

    return {"status": "NA", "reason": str(reason)}


def is_na(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("status") == "NA"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def authoritative_joint_ranges(
    path: Path = AUTHORITATIVE_REFERENCES["joint_ranges"],
) -> tuple[list[str], np.ndarray]:
    """Read the frozen named 28-D joint ranges used by Dataset B and A/B."""

    payload = read_json(path)
    names = list(payload["joint_names"])
    specs = list(payload["joint_specs"])
    if len(names) != 28 or [row["joint_name"] for row in specs] != names:
        raise RuntimeError(f"authoritative 28-D joint contract is malformed: {path}")
    ranges = np.asarray(
        [float(row["maximum"]) - float(row["minimum"]) for row in specs],
        dtype=np.float64,
    )
    if ranges.shape != (28,) or not np.isfinite(ranges).all() or np.any(ranges <= 0.0):
        raise RuntimeError(f"invalid authoritative joint ranges: {path}")
    return names, ranges


def validate_unique_episode_ids(entries: Iterable[Mapping[str, Any]]) -> list[str]:
    ids = [str(row.get("source_episode_id", "")) for row in entries]
    if any(not value for value in ids):
        raise ValueError("every episode must have a non-empty source_episode_id")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate source episode identities: {duplicates}")
    return ids


def validate_fps(value: float) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not np.isclose(fps, FPS, atol=1e-9, rtol=0.0):
        raise ValueError(f"paper evaluator requires authoritative 30 Hz timing, got {value}")
    return fps
