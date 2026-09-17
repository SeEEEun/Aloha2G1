"""Read-only adapter for the project's frozen Doll-Handoff retarget artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import AUTHORITATIVE_REFERENCES, FPS, ROOT, SEMANTIC_SUCCESS, sha256_file
from .metrics import aggregate_episode_metrics, evaluate_episode


DEFAULT_A_ROOT = ROOT / "outputs/fair_a_full50_hard_fail_audit/after_full_pose/trajectories"
DEFAULT_A_FEASIBILITY = ROOT / "outputs/fair_a_full50_hard_fail_audit/after/full50_per_episode.csv"
DEFAULT_B_FEASIBILITY = ROOT / "outputs/doll_handoff_dataset_b_final/final50/per_episode.csv"


def _csv_index(path: Path, identity: str = "stable_episode_id") -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {str(row[identity]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate {identity} in {path}")
    return result


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(()).item())


def _event_map(payload: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {
        str(name): int(frame)
        for name, frame in zip(
            payload["event_names"].astype(str), payload["event_frames"].astype(np.int64), strict=True
        )
    }


def _first_state(payload: Mapping[str, np.ndarray], state: str) -> int | None:
    values = payload["ownership_state"].astype(str)
    indices = np.flatnonzero(values == state)
    return int(indices[0]) if len(indices) else None


def _canonical_events(payload: Mapping[str, np.ndarray]) -> dict[str, int | None]:
    events = _event_map(payload)
    return {
        "LEFT_APPROACH": 0,
        "LEFT_GRASP": events.get("LEFT_GRASP"),
        "LEFT_TRANSPORT": events.get("LEFT_STABLE_HOLD"),
        "RIGHT_APPROACH": events.get("HANDOFF_APPROACH"),
        "DUAL_CONTACT": events.get("RIGHT_ACQUIRE"),
        "RIGHT_OWNED": events.get("LEFT_RELEASE"),
        "RIGHT_TRANSPORT": _first_state(payload, "RIGHT_TRANSPORT"),
        "RELEASE": events.get("RIGHT_FINAL_RELEASE"),
    }


def _a_feasibility(row: Mapping[str, str], path: Path) -> dict[str, Any]:
    hard_reasons = json.loads(row["hard_reasons"])
    return {
        "hard_ik_failure_count": int(any("IK" in reason for reason in hard_reasons)),
        "hard_collision_count": int(row["hard_collision_frames_after"]),
        "joint_limit_failure_count": int(row["joint_limit_violation_count"]),
        "branch_discontinuity_count": int(row["branch_discontinuity_count"]),
        "minimum_clearance_m": None,
        "provenance": f"{path}: validated common feasibility audit; clearance absent in summary",
    }


def _b_feasibility(row: Mapping[str, str], path: Path) -> dict[str, Any]:
    return {
        "hard_ik_failure_count": int(str(row["after_hard_ik"]).lower() == "true"),
        "hard_collision_count": int(row["after_hard_collision_frames"]),
        "joint_limit_failure_count": int(row["joint_limit_violations_after"]),
        "branch_discontinuity_count": int(row["branch_discontinuities_after"]),
        "minimum_clearance_m": float(row["after_minimum_torso_clearance_m"]),
        "provenance": f"{path}: finalized validated Dataset-B diagnostics",
    }


def _episode(
    method: str,
    source_id: str,
    a: Mapping[str, np.ndarray],
    b: Mapping[str, np.ndarray],
    feasibility: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    candidate = a if method == "A" else b
    frames = len(candidate["replay_named_joint_qpos"])
    if frames != len(a["replay_named_joint_qpos"]) or frames != len(b["replay_named_joint_qpos"]):
        raise RuntimeError(f"A/B frame mismatch for {source_id}")
    for key in ("target_left_interaction_frame_position_world", "target_right_interaction_frame_position_world"):
        if not np.array_equal(a[key], b[key]):
            raise RuntimeError(f"A/B authoritative whole-hand target differs for {source_id}: {key}")
    arrays: dict[str, Any] = {
        "reference_left_wrist_position_m": a["source_left_realization_frame_position_world"],
        "reference_right_wrist_position_m": a["source_right_realization_frame_position_world"],
        "candidate_left_wrist_position_m": candidate["achieved_left_wrist_position_world"],
        "candidate_right_wrist_position_m": candidate["achieved_right_wrist_position_world"],
        "reference_left_whole_hand_position_m": a["target_left_interaction_frame_position_world"],
        "reference_right_whole_hand_position_m": a["target_right_interaction_frame_position_world"],
        "candidate_left_whole_hand_position_m": candidate["achieved_left_physical_grasp_frame_position_world"],
        "candidate_right_whole_hand_position_m": candidate["achieved_right_physical_grasp_frame_position_world"],
        "candidate_q_rad": candidate["replay_named_joint_qpos"],
        "feasibility_projection_m": candidate["feasibility_projection_translation_m"],
    }
    if method == "A":
        arrays.update(
            {
                "reference_left_wrist_rotation": a["source_left_realization_frame_orientation_model"],
                "reference_right_wrist_rotation": a["source_right_realization_frame_orientation_model"],
                "candidate_left_wrist_rotation": a["achieved_left_realization_frame_orientation_model"],
                "candidate_right_wrist_rotation": a["achieved_right_realization_frame_orientation_model"],
            }
        )
    events = _event_map(candidate)
    return {
        "source_episode_id": source_id,
        "fps": FPS,
        "evaluation_mode": "retargeting",
        "success_kind": SEMANTIC_SUCCESS,
        "arrays": arrays,
        "annotations": {
            "candidate_phase_events_frame": _canonical_events(candidate),
            "phase_events_authoritative": True,
            "right_acquire_frame": events.get("RIGHT_ACQUIRE"),
            "left_release_frame": events.get("LEFT_RELEASE"),
            "handoff_events_authoritative": True,
            "annotation_scope": "SOURCE_DERIVED_INTERACTION_SEMANTICS_NOT_MEASURED_OBJECT_CONTACT",
        },
        "feasibility": feasibility,
        "provenance": {
            "trajectory_path": str(path),
            "trajectory_sha256": sha256_file(path),
            "whole_hand_frame_authoritative": True,
            "whole_hand_frame_definition": str(AUTHORITATIVE_REFERENCES["whole_hand_frame"]),
            "wrist_orientation_reference_authoritative": method == "A",
            "wrist_orientation_note": (
                "A archive carries achieved authoritative wrist rotations"
                if method == "A"
                else "B frozen archive does not persist achieved wrist rotation; orientation metric is NA"
            ),
        },
    }


def evaluate_frozen_retargeting(
    *,
    joint_ranges_rad: np.ndarray,
    source_manifest: Path = AUTHORITATIVE_REFERENCES["source_episode_manifest"],
    a_root: Path = DEFAULT_A_ROOT,
    a_feasibility_path: Path = DEFAULT_A_FEASIBILITY,
    b_feasibility_path: Path = DEFAULT_B_FEASIBILITY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the currently frozen full-50 artifacts without modifying them."""

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    a_files = sorted(a_root.glob("*.npz"))
    a_paths: dict[str, Path] = {}
    for path in a_files:
        payload = _npz(path)
        source_id = _scalar_string(payload["source_episode_id"])
        if source_id in a_paths:
            raise RuntimeError(f"duplicate A source identity: {source_id}")
        a_paths[source_id] = path
    a_diag = _csv_index(a_feasibility_path)
    b_diag = _csv_index(b_feasibility_path)
    a_results = []
    b_results = []
    for entry in manifest["episodes"]:
        source_id = str(entry["stable_episode_id"])
        a_path = a_paths.get(source_id)
        b_path = Path(entry["retargeted_trajectory_path"])
        if a_path is None or not b_path.is_file():
            raise FileNotFoundError(f"missing exact A/B trajectory for {source_id}")
        a = _npz(a_path)
        b = _npz(b_path)
        if _scalar_string(a["source_episode_id"]) != source_id or _scalar_string(b["source_episode_id"]) != source_id:
            raise RuntimeError(f"trajectory/source manifest identity mismatch for {source_id}")
        a_input = _episode("A", source_id, a, b, _a_feasibility(a_diag[source_id], a_feasibility_path), a_path)
        b_input = _episode("B", source_id, a, b, _b_feasibility(b_diag[source_id], b_feasibility_path), b_path)
        a_results.append(evaluate_episode(a_input, joint_ranges_rad=joint_ranges_rad))
        b_results.append(evaluate_episode(b_input, joint_ranges_rad=joint_ranges_rad))
    common = {
        "schema_version": "paper_evaluation_result_v1",
        "status": "READY",
        "evaluation_mode": "retargeting",
        "success_kind": SEMANTIC_SUCCESS,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "episode_count": len(manifest["episodes"]),
        "read_only_adapter": True,
    }
    return (
        {
            **common,
            "method": "FAIR_A",
            "episodes": a_results,
            "aggregate": aggregate_episode_metrics(a_results),
        },
        {
            **common,
            "method": "PROPOSED_B",
            "episodes": b_results,
            "aggregate": aggregate_episode_metrics(b_results),
        },
    )
