#!/usr/bin/env python3
"""Freeze the completed Proposed-B trajectories for Dataset-B auditing.

This script is deliberately read-only with respect to retargeting inputs and
trajectory artifacts.  It writes a manifest and sentinel beneath the completed
review tree, after verifying the original approval freeze and all 50 episodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
)
STATUS = "PROPOSED_B_MOTION_FROZEN_FOR_DATASET_AUDIT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.review_root.resolve()
    approval_path = root / "frozen_approval/freeze_manifest.json"
    if not approval_path.is_file():
        raise FileNotFoundError(approval_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("status") != "PROPOSED_B_APPROVED_FOR_50_EPISODE_BATCH":
        raise RuntimeError("the completed batch is not tied to the approved freeze")
    if approval["protected_constraints"]["handoff_cartesian_residual_m"] != 0.0:
        raise RuntimeError("non-zero handoff Cartesian residual in approval freeze")

    protected_files: dict[str, str] = {}
    for label, record in approval["snapshot_files"].items():
        path = Path(record["snapshot"])
        actual = sha256_file(path)
        if actual != record["snapshot_sha256"]:
            raise RuntimeError(f"approved snapshot drift: {label}")
        protected_files[label] = actual

    trajectory_dir = root / "proposed/trajectories"
    metric_dir = root / "proposed/metrics"
    trajectories = sorted(trajectory_dir.glob("doll_handoff_20260820_ep*.npz"))
    if len(trajectories) != 50:
        raise RuntimeError(f"expected 50 trajectories, found {len(trajectories)}")

    episode_records: list[dict[str, Any]] = []
    target_keys = (
        "target_left_wrist_position_model",
        "target_right_wrist_position_model",
        "target_left_wrist_rotation_model",
        "target_right_wrist_rotation_model",
        "target_left_interaction_frame_position_world",
        "target_right_interaction_frame_position_world",
        "target_left_interaction_frame_position_task",
        "target_right_interaction_frame_position_task",
    )
    action_keys = ("g1_arm_qpos", "left_dex3_qpos", "right_dex3_qpos")
    semantic_keys = ("left_hand_phase", "right_hand_phase", "ownership_state")
    all_target_hashes: dict[str, str] = {}
    all_action_hashes: dict[str, str] = {}
    all_file_hashes: dict[str, str] = {}

    for expected_episode, path in enumerate(trajectories):
        stable_id = f"doll_handoff_20260820_ep{expected_episode:03d}"
        if path.stem != stable_id:
            raise RuntimeError(
                f"episode sequence mismatch: expected {stable_id}, got {path.stem}"
            )
        manifest_path = metric_dir / f"{stable_id}.manifest.json"
        metrics_path = metric_dir / f"{stable_id}.json"
        validation_path = metric_dir / f"{stable_id}.validation.json"
        for required in (manifest_path, metrics_path, validation_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        runtime_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as payload:
            missing = [k for k in (*target_keys, *action_keys, *semantic_keys) if k not in payload]
            if missing:
                raise RuntimeError(f"{stable_id} missing frozen arrays: {missing}")
            component_hashes = {key: array_hash(payload[key]) for key in payload.files}
            target_hash = canonical_hash({key: component_hashes[key] for key in target_keys})
            action_hash = canonical_hash({key: component_hashes[key] for key in action_keys})
            semantic_hash = canonical_hash(
                {key: component_hashes[key] for key in semantic_keys}
            )
            frame_count = int(payload["g1_arm_qpos"].shape[0])
        trajectory_sha = sha256_file(path)
        if trajectory_sha != runtime_manifest["trajectory_sha256"]:
            raise RuntimeError(f"runtime trajectory checksum mismatch: {stable_id}")
        if runtime_manifest["implementation_sha256"] != approval["implementation_sha256"]:
            raise RuntimeError(f"implementation provenance mismatch: {stable_id}")
        if runtime_manifest["cartesian_target_sha256"] != json.loads(
            metrics_path.read_text(encoding="utf-8")
        )["cartesian_target_sha256"]:
            raise RuntimeError(f"Cartesian target provenance mismatch: {stable_id}")

        all_target_hashes[stable_id] = target_hash
        all_action_hashes[stable_id] = action_hash
        all_file_hashes[stable_id] = trajectory_sha
        episode_records.append(
            {
                "episode_index": expected_episode,
                "stable_episode_id": stable_id,
                "frame_count": frame_count,
                "trajectory_path": str(path),
                "trajectory_file_sha256": trajectory_sha,
                "cartesian_target_manifest_sha256": runtime_manifest[
                    "cartesian_target_sha256"
                ],
                "cartesian_target_array_set_sha256": target_hash,
                "g1_action_array_set_sha256": action_hash,
                "interaction_semantic_array_set_sha256": semantic_hash,
                "all_array_sha256": component_hashes,
                "metrics_sha256": sha256_file(metrics_path),
                "validation_sha256": sha256_file(validation_path),
                "runtime_manifest_sha256": sha256_file(manifest_path),
            }
        )

    old_artifact_manifest = root / "review/artifact_manifest.json"
    old_review_sentinel = root / "review/PROPOSED_B_50_REVIEW_COMPLETE"
    if not old_artifact_manifest.is_file() or not old_review_sentinel.is_file():
        raise RuntimeError("completed 50-episode review provenance is missing")

    output_dir = root / "review/dataset_b_gate/motion_freeze"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "motion_freeze_manifest.json"
    manifest = {
        "schema_version": "proposed_b_motion_dataset_audit_freeze_v1",
        "status": STATUS,
        "scope": "COMPLETED_INTERACTION_CENTRIC_PROPOSED_B_MOTION_ONLY",
        "frozen_episode_count": 50,
        "motion_mutation_allowed": False,
        "solver_tolerance_change_allowed": False,
        "canonical_scene_diagnostic_is_dataset_rejection_gate": False,
        "handoff_cartesian_residual_m": 0.0,
        "approval_freeze_manifest": str(approval_path),
        "approval_freeze_manifest_sha256": sha256_file(approval_path),
        "implementation_sha256": approval["implementation_sha256"],
        "implementation_files_sha256": approval["implementation_files"],
        "protected_snapshot_files_sha256": protected_files,
        "resolved_config_sha256": approval["approved_resolved_file_sha256"],
        "protected_semantic_hashes": approval["protected_semantic_hashes"],
        "trajectory_file_set_sha256": canonical_hash(all_file_hashes),
        "cartesian_target_array_set_sha256": canonical_hash(all_target_hashes),
        "g1_action_array_set_sha256": canonical_hash(all_action_hashes),
        "completed_review_artifact_manifest_sha256": sha256_file(
            old_artifact_manifest
        ),
        "completed_review_sentinel_sha256": sha256_file(old_review_sentinel),
        "episodes": episode_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    sentinel = output_dir / STATUS
    sentinel.write_text(
        f"{STATUS}\n"
        f"manifest: {manifest_path}\n"
        f"manifest_sha256: {manifest_sha}\n"
        f"implementation_sha256: {approval['implementation_sha256']}\n"
        f"trajectory_file_set_sha256: {manifest['trajectory_file_set_sha256']}\n"
        f"cartesian_target_array_set_sha256: {manifest['cartesian_target_array_set_sha256']}\n"
        "trajectory_mutation: FORBIDDEN\n"
        "solver_tolerance_change: FORBIDDEN\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "implementation_sha256": manifest["implementation_sha256"],
                "trajectory_file_set_sha256": manifest["trajectory_file_set_sha256"],
                "cartesian_target_array_set_sha256": manifest[
                    "cartesian_target_array_set_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
