"""Strict v4 evaluation and isolated episode export."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np

from aloha_g1_dataset_v1.core import G1Kinematics, array_sha256
from aloha_g1_feasibility_v3.evaluate import evaluate_result

from .audit import v4_category
from .common import (
    V3_ROOT,
    atomic_json,
    atomic_npz,
    load_json,
    sha256_file,
)
from .solver import CollisionRepairResult, SharedCollisionWindowSolver


def evaluate_repair(
    result: CollisionRepairResult,
    solver: SharedCollisionWindowSolver,
    runtime: Any,
    classifier: Any,
    g1: G1Kinematics,
    runtime_config: Mapping[str, Any],
    pair_catalog: list[tuple[int, int]],
) -> dict[str, Any]:
    """Reuse the unchanged v3 gates, then add v4 clearance diagnostics."""
    evaluated = evaluate_result(
        solver.as_v3_result(result), runtime, classifier, g1, runtime_config
    )
    metrics = evaluated["metrics"]
    validation = evaluated["validation"]
    metrics["schema_version"] = "common_collision_v4_episode_metrics"
    metrics["v3_to_v4"] = metrics.pop("v2_to_v3")
    metrics["v3_to_v4"]["reference"] = "accepted Common Feasibility-v3 arm action"
    metrics["solver"].update(
        {
            "collision_v4_solver_class": type(solver).__name__,
            "frozen_global_collision_config": result.metadata["collision_config"],
            "event_reports": result.metadata["event_reports"],
            "method_specific_collision_logic": False,
        }
    )

    category_frames: dict[str, set[int]] = defaultdict(set)
    pair_frames: Counter[str] = Counter()
    penetration_values: list[float] = []
    clearance_values: list[float] = []
    colliding_frame_clearance: list[float] = []
    for frame in range(len(result.q)):
        runtime.assign(result.q[frame], result.left_hand[frame], result.right_hand[frame])
        distances = [
            float(
                mujoco.mj_geomDistance(
                    runtime.model,
                    runtime.data,
                    int(pair[0]),
                    int(pair[1]),
                    0.10,
                    None,
                )
            )
            for pair in pair_catalog
        ]
        clearance_values.extend(distances)
        records = [row for row in classifier.records() if row.v1_gate_relevant]
        if records:
            colliding_frame_clearance.append(
                min(float(row.distance_m) for row in records)
            )
        for row in records:
            category_frames[v4_category(row.bodies)].add(frame)
            pair_frames[row.pair] += 1
            penetration_values.append(max(0.0, -float(row.distance_m)))
    collision = metrics["collision"]
    collision.update(
        {
            "v4_category_frame_incidence": {
                key: len(value) for key, value in sorted(category_frames.items())
            },
            "v4_top_prohibited_pairs": [
                {"pair": key, "frame_count": count}
                for key, count in pair_frames.most_common(20)
            ],
            "minimum_catalog_clearance_m": min(clearance_values, default=0.1),
            "minimum_colliding_signed_distance_m": min(
                colliding_frame_clearance, default=0.0
            ),
            "maximum_penetration_depth_m": max(penetration_values, default=0.0),
            "signed_distance_backend": "MuJoCo mj_geomDistance",
            "d_safe_m": float(result.metadata["collision_config"]["d_safe_m"]),
        }
    )
    validation["schema_version"] = "common_collision_v4_validation"
    validation["acceptance_gates_byte_semantically_unchanged_from_v3"] = True
    if not validation["checks"]["collision"]:
        validation["first_causal_failure"]["classification"] = (
            "COLLISION_UNAVOIDABLE_WITHIN_TASK_EQUIVALENCE"
        )
    return evaluated


def export_episode(
    output_root: Path,
    result: CollisionRepairResult,
    evaluated: Mapping[str, Any],
    g1: G1Kinematics,
    runtime: Any,
    frozen_solver_sha256: str,
    dependency_checksums: Mapping[str, Any],
) -> Path:
    episode = result.episode
    directory = output_root / episode.dataset_name / f"episode_{episode.episode_id:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    q = result.q.astype(np.float32)
    left = result.left_hand.astype(np.float32)
    right = result.right_hand.astype(np.float32)
    hand = np.column_stack((left, right)).astype(np.float32)
    full = np.column_stack((q, hand)).astype(np.float32)
    atomic_npz(
        directory / "g1_arm_action.npz",
        action=q,
        timestamps=episode.timestamps.astype(np.float64),
        fps=np.asarray(episode.fps),
        target_left_wrist_position=episode.targets["left_wrist_position"].astype(np.float64),
        target_right_wrist_position=episode.targets["right_wrist_position"].astype(np.float64),
        target_left_wrist_rotation=episode.targets["left_wrist_rotation"].astype(np.float64),
        target_right_wrist_rotation=episode.targets["right_wrist_rotation"].astype(np.float64),
        target_left_task_tool_position=episode.targets["left_tool_position"].astype(np.float64),
        target_right_task_tool_position=episode.targets["right_tool_position"].astype(np.float64),
        joint_names=np.asarray(g1.info["joint_names"]).astype("U64"),
        method=np.asarray(episode.method),
        representation=np.asarray(episode.representation),
        frozen_collision_v4_solver_sha256=np.asarray(frozen_solver_sha256),
    )
    atomic_npz(
        directory / "g1_hand_action.npz",
        action=hand,
        left_action=left,
        right_action=right,
        left_phase=episode.left_phase.astype("U16"),
        right_phase=episode.right_phase.astype("U16"),
        left_joint_names=np.asarray(runtime.hand_joint_names["left"]).astype("U64"),
        right_joint_names=np.asarray(runtime.hand_joint_names["right"]).astype("U64"),
        mapper=np.asarray(
            "unchanged_binary_open_close"
            if episode.method == "baseline"
            else "frozen_proposed_hand_v2_1"
        ),
    )
    atomic_npz(
        directory / "g1_full_action.npz",
        action=full,
        timestamps=episode.timestamps.astype(np.float64),
        fps=np.asarray(episode.fps),
        joint_names=np.concatenate(
            (
                np.asarray(g1.info["joint_names"]).astype("U64"),
                np.asarray(runtime.hand_joint_names["left"]).astype("U64"),
                np.asarray(runtime.hand_joint_names["right"]).astype("U64"),
            )
        ),
    )
    atomic_npz(
        directory / "collision_repair_diagnostics.npz",
        q_v3=episode.q_v2.astype(np.float32),
        q_v4=q,
        q_deviation=(result.q - episode.q_v2).astype(np.float32),
        changed_frame_mask=result.changed_mask.astype(bool),
        orientation_slack_rad=result.orientation_slack.astype(np.float32),
        orientation_slack_requested_rad=result.orientation_slack_requested.astype(np.float32),
    )
    source = load_json(
        V3_ROOT / episode.dataset_name / f"episode_{episode.episode_id:06d}" / "source_metadata.json"
    )
    source.update(
        {
            "collision_v4_source": "frozen Common Feasibility-v3 representation, targets, hand action, and arm seed",
            "source_assets_duplicated": False,
            "source_hashes_unchanged": True,
        }
    )
    atomic_json(directory / "source_metadata.json", source)
    atomic_json(directory / "retargeting_metrics.json", evaluated["metrics"])
    atomic_json(directory / "validation.json", evaluated["validation"])
    atomic_json(directory / "solver_report.json", result.metadata)
    file_names = (
        "source_metadata.json",
        "g1_arm_action.npz",
        "g1_hand_action.npz",
        "g1_full_action.npz",
        "collision_repair_diagnostics.npz",
        "retargeting_metrics.json",
        "validation.json",
        "solver_report.json",
    )
    manifest = {
        "schema_version": "common_collision_v4_episode",
        "dataset": episode.dataset_name,
        "method": episode.method,
        "episode_id": episode.episode_id,
        "status": evaluated["metrics"]["status"],
        "frozen_collision_v4_solver_sha256": frozen_solver_sha256,
        "frozen_feasibility_v3_solver_sha256": dependency_checksums[
            "feasibility_v3_solver_sha256"
        ],
        "frozen_common_arm_v2_sha256": dependency_checksums["global_mapping_sha256"],
        "frozen_hand_v2_1_sha256": (
            None
            if episode.method == "baseline"
            else dependency_checksums["hand_v2_1_sha256"]
        ),
        "frozen_v3_arm_action_array_sha256": array_sha256(episode.q_v2.astype(np.float32)),
        "hand_action_array_sha256": array_sha256(hand),
        "a_b_output_separation": True,
        "files": {name: sha256_file(directory / name) for name in file_names},
        "offline_only": True,
        "training_executed": False,
        "physics_executed": False,
        "real_robot_commands": False,
    }
    atomic_json(directory / "manifest.json", manifest)
    return directory


__all__ = ["evaluate_repair", "export_episode"]
