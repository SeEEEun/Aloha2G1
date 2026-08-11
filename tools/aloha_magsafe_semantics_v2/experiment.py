"""Shared scientific-protocol helpers for semantic detector v2."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS
from aloha_magsafe_semantics.features import compute_stationary_aloha_fk
from aloha_magsafe_semantics.io import load_trajectory, sha256_file

from .candidate_detection import extract_event_candidates
from .detector import detect_magsafe_semantics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v2"
MANIFEST = ROOT / "reports/magsafe_lerobot_v3_manifest.csv"
MODEL = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
POSE_CONFIG = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
GEOMETRY_PATH = ROOT / "configs/aloha_magsafe_task_geometry.semantic_v1.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def read_manifest() -> list[dict[str, Any]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows = [{
        "episode_id": int(row["output_episode_index"]),
        "source_folder": str(Path(row["source_folder"]).resolve()),
        "source_parquet": str(Path(row["source_parquet"]).resolve()),
        "frame_count": int(row["source_frame_count"]),
        "duration_sec": float(row["source_last_timestamp"]) - float(row["source_first_timestamp"]),
    } for row in source]
    if sorted(row["episode_id"] for row in rows) != list(range(50)):
        raise RuntimeError("authoritative manifest IDs are not 0..49")
    return sorted(rows, key=lambda row: row["episode_id"])


def load_split() -> dict[str, Any]:
    return json.loads((OUT / "dataset_split_v2.json").read_text(encoding="utf-8"))


@dataclass
class AccessLedger:
    stage: str
    allowed_ids: set[int]
    prohibited_ids: set[int]
    accesses: list[dict[str, Any]] = field(default_factory=list)

    def check(self, episode_id: int, purpose: str) -> None:
        if episode_id in self.prohibited_ids or episode_id not in self.allowed_ids:
            raise PermissionError(f"stage {self.stage} may not access episode {episode_id} for {purpose}")
        self.accesses.append({
            "episode_id": int(episode_id),
            "purpose": purpose,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    def payload(self) -> dict[str, Any]:
        accessed = sorted({row["episode_id"] for row in self.accesses})
        return {
            "stage": self.stage,
            "allowed_ids": sorted(self.allowed_ids),
            "prohibited_ids": sorted(self.prohibited_ids),
            "accessed_ids": accessed,
            "prohibited_access_count": 0,
            "access_count": len(self.accesses),
        }


def pose() -> dict[str, Any]:
    return json.loads(POSE_CONFIG.read_text(encoding="utf-8"))["stationary_aloha"]


def geometry() -> dict[str, Any]:
    return json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))


def process_episode(
    row: dict[str, Any],
    config: dict[str, Any],
    ledger: AccessLedger,
    purpose: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    episode_id = int(row["episode_id"])
    ledger.check(episode_id, purpose)
    loaded = load_trajectory(row["source_parquet"], "raw_action")
    source_pose = pose()
    fk = compute_stationary_aloha_fk(
        loaded["action"], loaded["timestamps"], MODEL,
        source_pose["position_xyz_m"], source_pose["orientation_wxyz"],
    )
    fk["model_sha256"] = sha256_file(MODEL)
    task_geometry = geometry()
    timeline = detect_magsafe_semantics(
        loaded["action"], loaded["timestamps"], "raw_action", fk, task_geometry, config,
        observation_state=None, observation_alignment=None,
    )
    candidates, context = extract_event_candidates(
        loaded["action"], loaded["timestamps"], fk, task_geometry, config,
    )
    return timeline, loaded, candidates, context


def load_episode_inputs(
    row: dict[str, Any],
    ledger: AccessLedger,
    purpose: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load raw motion and compute FK once for multi-config experiments."""
    episode_id = int(row["episode_id"])
    ledger.check(episode_id, purpose)
    loaded = load_trajectory(row["source_parquet"], "raw_action")
    source_pose = pose()
    fk = compute_stationary_aloha_fk(
        loaded["action"], loaded["timestamps"], MODEL,
        source_pose["position_xyz_m"], source_pose["orientation_wxyz"],
    )
    fk["model_sha256"] = sha256_file(MODEL)
    return loaded, fk, geometry()


def detect_loaded(
    loaded: dict[str, Any],
    fk: dict[str, Any],
    task_geometry: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    return detect_magsafe_semantics(
        loaded["action"], loaded["timestamps"], "raw_action", fk, task_geometry, config,
        observation_state=None, observation_alignment=None,
    )


def timeline_summary(episode_id: int, timeline: Any, row: dict[str, Any]) -> dict[str, Any]:
    low = [name for name in REQUIRED_EVENTS if timeline.event(name).confidence_class == "LOW"]
    ambiguous = [name for name in REQUIRED_EVENTS if timeline.event(name).confidence_class == "AMBIGUOUS"]
    result: dict[str, Any] = {
        "episode_id": int(episode_id),
        "frame_count": int(timeline.trajectory_length),
        "duration_sec": float(timeline.timestamps[-1] - timeline.timestamps[0]),
        "source_parquet": row["source_parquet"],
        "detector_config_hash": timeline.detector_config_hash,
        "mandatory_complete": timeline.mandatory_complete(),
        "high_medium_complete": timeline.high_medium_complete(),
        "partial_order_valid": bool(timeline.metadata["partial_order"]["valid"]),
        "low_events": low,
        "ambiguous_events": ambiguous,
        "global_sequence_margin": timeline.metadata.get("global_sequence_score_margin"),
    }
    for name in REQUIRED_EVENTS:
        event = timeline.event(name)
        result[f"{name}_index"] = event.action_index
        result[f"{name}_time_sec"] = event.action_time_sec
        result[f"{name}_confidence"] = event.confidence
        result[f"{name}_class"] = event.confidence_class
        result[f"{name}_source"] = event.evidence.get("candidate_source")
    return result


def save_episode(directory: Path, timeline: Any, candidates: dict[str, Any], context: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "semantic_timeline.auto.json", timeline.to_dict())
    np.savez_compressed(
        directory / "semantic_phases.npz",
        timestamps=timeline.timestamps,
        event_names=np.asarray(list(timeline.events), dtype="U48"),
        event_action_indices=np.asarray([
            -1 if timeline.events[name].action_index is None else timeline.events[name].action_index
            for name in timeline.events
        ], dtype=np.int64),
        **timeline.sample_arrays,
    )
    atomic_json(directory / "event_candidates.json", {
        "detector_config_hash": timeline.detector_config_hash,
        "reference_timeline_used_for_detection": False,
        "events": {name: [row.to_dict() for row in values] for name, values in candidates.items()},
    })
    feature = context["features"]
    atomic_json(directory / "feature_metrics.json", {
        "frame_count": timeline.trajectory_length,
        "duration_sec": float(timeline.timestamps[-1] - timeline.timestamps[0]),
        "left_path_m": float(feature["left_cumulative_path_length"][-1]),
        "right_path_m": float(feature["right_cumulative_path_length"][-1]),
        "left_rotation_rad": float(feature["left_cumulative_rotation"][-1]),
        "right_rotation_rad": float(feature["right_cumulative_rotation"][-1]),
        "rotation_segmentation": timeline.metadata["rotation_segmentation"],
        "terminal_suffix": timeline.metadata["terminal_suffix"],
        "release_physical_evidence": timeline.metadata["release_physical_evidence"],
    })


def aggregate_status(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(summaries),
        "mandatory_complete_count": sum(row["mandatory_complete"] for row in summaries),
        "high_medium_complete_count": sum(row["high_medium_complete"] for row in summaries),
        "partial_order_valid_count": sum(row["partial_order_valid"] for row in summaries),
        "low_episode_count": sum(bool(row["low_events"]) for row in summaries),
        "ambiguous_episode_count": sum(bool(row["ambiguous_events"]) for row in summaries),
        "same_detector_config": len({row["detector_config_hash"] for row in summaries}) == 1,
    }
