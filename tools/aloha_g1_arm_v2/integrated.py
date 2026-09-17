"""Frozen hand integration, 100-trajectory validation, and exporters."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aloha_g1_dataset_v1.core import (
    G1Kinematics,
    HandMapper,
    branch_flags,
    inverse_transform,
    phase_runs,
    physical_pinch_frame,
    raw_wrist_pose,
    rotation_errors,
    stats,
)
from aloha_g1_hand_v2.collision_eval import (
    CollisionClassifier,
    body_digit,
)
from aloha_g1_hand_v2_1.third_neutral import minimum_jerk_state_commands

from .audit import ArmEpisodeResult
from .common import array_sha256, atomic_json, atomic_npz, sha256_file


INTEGRATED_STATUSES = (
    "PASS",
    "FAIL_IK",
    "FAIL_COLLISION",
    "FAIL_TEMPORAL",
    "FAIL_LIMIT",
    "FAIL_DATA",
    "FAIL_OTHER",
)


def validate_candidate_joint_order(
    hand_candidate: Mapping[str, Any], runtime: Any
) -> dict[str, bool]:
    checks = {
        side: list(hand_candidate["joint_order"][side])
        == list(runtime.hand_joint_names[side])
        for side in ("left", "right")
    }
    if not all(checks.values()):
        raise RuntimeError(f"Hand-v2.1 joint order mismatch: {checks}")
    return checks


def map_proposed_hand(
    arm_result: ArmEpisodeResult,
    hand_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    frames = int(hand_candidate["interpolation"]["selected_transition_frames"])
    states = {
        side: {
            phase: np.asarray(value, dtype=np.float64)
            for phase, value in hand_candidate["states"][side].items()
        }
        for side in ("left", "right")
    }
    labels = {
        side: np.asarray(arm_result.detected[side].phase).astype(str)
        for side in ("left", "right")
    }
    return {
        "left": minimum_jerk_state_commands(
            labels["left"], states["left"], frames, arm_result.episode.fps
        ),
        "right": minimum_jerk_state_commands(
            labels["right"], states["right"], frames, arm_result.episode.fps
        ),
        "left_phase": labels["left"],
        "right_phase": labels["right"],
        "transition_frames": frames,
        "mapper": "frozen_proposed_hand_v2_1",
        "label_calibration_status": hand_candidate["label_status"],
    }


def map_baseline_hand(
    arm_result: ArmEpisodeResult,
    hand_mapper: HandMapper,
) -> dict[str, Any]:
    mapped = hand_mapper.map(
        "baseline", arm_result.episode, arm_result.detected
    )
    mapped["mapper"] = "unchanged_v1_binary_open_close"
    mapped["label_calibration_status"] = hand_mapper.config["hand_mapping"][
        "label_calibration_status"
    ]
    return mapped


def _full_geometry_and_collisions(
    runtime: Any,
    classifier: CollisionClassifier,
    config: Mapping[str, Any],
    arm: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, Any]:
    count = len(arm)
    positions = {
        f"{side}_{frame}": np.empty((count, 3), dtype=np.float64)
        for side in ("left", "right")
        for frame in ("wrist", "pinch")
    }
    rotations = {
        f"{side}_{frame}_rotation": np.empty((count, 3, 3), dtype=np.float64)
        for side in ("left", "right")
        for frame in ("wrist", "pinch")
    }
    flags = {
        key: np.zeros(count, dtype=bool)
        for key in (
            "prohibited",
            "arm_only",
            "hand_related",
            "hand_hand",
            "cross_arm",
            "third_finger",
            "thumb_index",
            "same_hand_self",
        )
    }
    category_frames: Counter[str] = Counter()
    pair_events: Counter[str] = Counter()
    labels = {
        side: tuple(config["target_frames"][f"{side}_physical_pinch_contacts"])
        for side in ("left", "right")
    }
    for index in range(count):
        runtime.assign(arm[index], left[index], right[index])
        for side in ("left", "right"):
            wrist = raw_wrist_pose(runtime, side)
            pinch = physical_pinch_frame(runtime, side, labels[side])
            positions[f"{side}_wrist"][index] = wrist[:3, 3]
            positions[f"{side}_pinch"][index] = pinch[:3, 3]
            rotations[f"{side}_wrist_rotation"][index] = wrist[:3, :3]
            rotations[f"{side}_pinch_rotation"][index] = pinch[:3, :3]
        records = classifier.records()
        gate = [record for record in records if record.v1_gate_relevant]
        hand = [
            record
            for record in records
            if any(body_digit(name) is not None for name in record.bodies)
        ]
        arm_only = [
            record
            for record in gate
            if all(body_digit(name) is None for name in record.bodies)
        ]
        flags["prohibited"][index] = bool(gate)
        flags["arm_only"][index] = bool(arm_only)
        flags["hand_related"][index] = bool(hand)
        flags["hand_hand"][index] = any(
            record.category == "HAND_HAND" for record in hand
        )
        flags["cross_arm"][index] = any(
            record.category == "CROSS_ARM" for record in records
        )
        flags["third_finger"][index] = any(
            any(body_digit(name) == "THIRD" for name in record.bodies)
            for record in hand
        )
        flags["thumb_index"][index] = any(
            any(body_digit(name) in {"THUMB", "INDEX"} for name in record.bodies)
            for record in hand
        )
        flags["same_hand_self"][index] = any(
            record.enhanced_same_hand for record in hand
        )
        for category in {record.category for record in records}:
            category_frames[category] += 1
        for record in records:
            pair_events[record.pair] += 1
    return {
        **positions,
        **rotations,
        **{f"{key}_collision_flag": value for key, value in flags.items()},
        "category_frame_incidence": dict(sorted(category_frames.items())),
        "top_pairs": [
            {"pair": pair, "events": value}
            for pair, value in pair_events.most_common(20)
        ],
    }


def evaluate_integrated_episode(
    dataset_name: str,
    arm_result: ArmEpisodeResult,
    hand: Mapping[str, Any],
    runtime: Any,
    classifier: CollisionClassifier,
    g1: G1Kinematics,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    arm = np.asarray(arm_result.solved["q"], dtype=np.float64)
    left = np.asarray(hand["left"], dtype=np.float64)
    right = np.asarray(hand["right"], dtype=np.float64)
    hand_action = np.column_stack((left, right))
    full = np.column_stack((arm, hand_action))
    geometry = _full_geometry_and_collisions(
        runtime, classifier, config, arm, left, right
    )
    fps = float(arm_result.episode.fps)
    step = np.abs(np.diff(full, axis=0))
    velocity = step * fps
    acceleration = np.abs(np.diff(full, n=2, axis=0)) * fps**2
    arm_step = np.abs(np.diff(arm, axis=0))
    branches = branch_flags(
        arm,
        float(config["validation"]["branch_absolute_step_norm_rad"]),
        float(config["validation"]["branch_local_multiplier"]),
    )
    limits = {
        "arm": g1.limits,
        "left": runtime.hand_limits["left"],
        "right": runtime.hand_limits["right"],
    }
    violation_count = int(
        np.count_nonzero((arm < limits["arm"][:, 0]) | (arm > limits["arm"][:, 1]))
        + np.count_nonzero((left < limits["left"][:, 0]) | (left > limits["left"][:, 1]))
        + np.count_nonzero((right < limits["right"][:, 0]) | (right > limits["right"][:, 1]))
    )
    targets = arm_result.targets
    pinch_error = {
        side: np.linalg.norm(
            geometry[f"{side}_pinch"] - np.asarray(targets[f"{side}_tool_position"]),
            axis=1,
        )
        for side in ("left", "right")
    }
    pinch_orientation = {
        side: rotation_errors(
            geometry[f"{side}_pinch_rotation"],
            np.asarray(targets[f"{side}_tool_rotation"]),
        )
        for side in ("left", "right")
    }
    actual_left = geometry["left_pinch"]
    actual_right = geometry["right_pinch"]
    target_left = np.asarray(targets["left_tool_position"])
    target_right = np.asarray(targets["right_tool_position"])
    actual_midpoint = 0.5 * (actual_left + actual_right)
    target_midpoint = 0.5 * (target_left + target_right)
    actual_relative = actual_right - actual_left
    target_relative = target_right - target_left
    actual_distance = np.linalg.norm(actual_relative, axis=1)
    target_distance = np.linalg.norm(target_relative, axis=1)
    task_masks = {
        side: np.isin(
            np.asarray(arm_result.detected[side].phase).astype(str),
            ["GRASP", "HOLD"],
        )
        for side in ("left", "right")
    }
    target_runs = {
        side: phase_runs(
            np.asarray(hand[f"{side}_phase"]).astype(str),
            arm_result.episode.timestamps,
        )
        for side in ("left", "right")
    }
    finite = bool(
        np.isfinite(full).all()
        and all(
            np.isfinite(geometry[f"{side}_pinch"]).all()
            for side in ("left", "right")
        )
    )
    temporal = bool(
        np.count_nonzero(branches) == 0
        and float(np.max(step, initial=0.0))
        <= float(config["validation"]["maximum_joint_step_rad"])
        and float(np.max(velocity, initial=0.0))
        <= float(config["validation"]["maximum_velocity_rad_s"])
        and float(np.max(acceleration, initial=0.0))
        <= float(config["validation"]["maximum_acceleration_rad_s2"])
    )
    collision_count = int(
        np.count_nonzero(geometry["prohibited_collision_flag"])
    )
    checks = {
        "data": finite and full.shape == (len(arm), 28),
        "limits": violation_count == 0,
        "ik": arm_result.metrics["ik_success_rate"]
        >= float(config["ik"]["required_success_rate"]),
        "collision": collision_count
        <= int(config["validation"]["prohibited_collision_frames_allowed"]),
        "temporal": temporal,
        "semantic": bool(
            all(
                len(hand[f"{side}_phase"]) == len(arm)
                for side in ("left", "right")
            )
        ),
    }
    status = "PASS"
    first = None
    for candidate, key in (
        ("FAIL_DATA", "data"),
        ("FAIL_LIMIT", "limits"),
        ("FAIL_IK", "ik"),
        ("FAIL_COLLISION", "collision"),
        ("FAIL_TEMPORAL", "temporal"),
        ("FAIL_OTHER", "semantic"),
    ):
        if not checks[key]:
            status = candidate
            first = key
            break
    metrics = {
        "dataset": dataset_name,
        "method": arm_result.method,
        "episode_id": int(arm_result.episode.episode_id),
        "status": status,
        "frame_count": len(arm),
        "fps": fps,
        "finite_values": finite,
        "action_shape": list(full.shape),
        "joint_limit_violation_count": violation_count,
        "ik_success_rate": arm_result.metrics["ik_success_rate"],
        "strict_ik_failure": not checks["ik"],
        "branch_discontinuity_count": int(np.count_nonzero(branches)),
        "maximum_joint_step_rad": float(np.max(step, initial=0.0)),
        "maximum_arm_joint_step_rad": float(np.max(arm_step, initial=0.0)),
        "maximum_velocity_rad_s": float(np.max(velocity, initial=0.0)),
        "maximum_acceleration_rad_s2": float(np.max(acceleration, initial=0.0)),
        "collision": {
            "prohibited_collision_frames": collision_count,
            "arm_only_collision_frames": int(
                np.count_nonzero(geometry["arm_only_collision_flag"])
            ),
            "hand_related_collision_frames": int(
                np.count_nonzero(geometry["hand_related_collision_flag"])
            ),
            "HAND_HAND_frames": int(
                np.count_nonzero(geometry["hand_hand_collision_flag"])
            ),
            "CROSS_ARM_frames": int(
                np.count_nonzero(geometry["cross_arm_collision_flag"])
            ),
            "third_finger_collision_frames": int(
                np.count_nonzero(geometry["third_finger_collision_flag"])
            ),
            "thumb_index_collision_frames": int(
                np.count_nonzero(geometry["thumb_index_collision_flag"])
            ),
            "same_hand_self_contact_frames": int(
                np.count_nonzero(geometry["same_hand_self_collision_flag"])
            ),
            "category_frame_incidence": geometry["category_frame_incidence"],
            "top_pairs": geometry["top_pairs"],
        },
        "task_space": {
            "wrist_error_mean_m": arm_result.metrics["wrist_error_mean_m"],
            "left_wrist_error_m": arm_result.metrics["left_wrist_error_m"],
            "right_wrist_error_m": arm_result.metrics["right_wrist_error_m"],
            "physical_pinch_frame_error_mean_m": float(
                0.5 * (np.mean(pinch_error["left"]) + np.mean(pinch_error["right"]))
            ),
            "task_critical_pinch_error_mean_m": float(
                0.5
                * (
                    np.mean(pinch_error["left"][task_masks["left"]])
                    + np.mean(pinch_error["right"][task_masks["right"]])
                )
            ),
            "left_physical_pinch_error_m": stats(pinch_error["left"]),
            "right_physical_pinch_error_m": stats(pinch_error["right"]),
            "left_pinch_orientation_error_rad": stats(pinch_orientation["left"]),
            "right_pinch_orientation_error_rad": stats(pinch_orientation["right"]),
        },
        "bimanual": {
            "midpoint_error_mean_m": float(
                np.mean(np.linalg.norm(actual_midpoint - target_midpoint, axis=1))
            ),
            "relative_vector_error_mean_m": float(
                np.mean(np.linalg.norm(actual_relative - target_relative, axis=1))
            ),
            "distance_change_error_mean_m": float(
                np.mean(
                    np.abs(
                        (actual_distance - actual_distance[0])
                        - (target_distance - target_distance[0])
                    )
                )
            ),
        },
        "semantic": {
            "mapper": hand["mapper"],
            "phase_completeness": checks["semantic"],
            "left_transition_count": max(0, len(target_runs["left"]) - 1),
            "right_transition_count": max(0, len(target_runs["right"]) - 1),
            "left_phase_runs": target_runs["left"],
            "right_phase_runs": target_runs["right"],
            "transition_validity": bool(
                np.isfinite(hand_action).all() and violation_count == 0
            ),
            "label_calibration_status": hand["label_calibration_status"],
        },
    }
    validation = {
        "status": status,
        "pass": status == "PASS",
        "first_causal_failure": first,
        "checks": checks,
        "thresholds": {
            "required_ik_success_rate": config["ik"]["required_success_rate"],
            "maximum_joint_step_rad": config["validation"]["maximum_joint_step_rad"],
            "maximum_velocity_rad_s": config["validation"]["maximum_velocity_rad_s"],
            "maximum_acceleration_rad_s2": config["validation"]["maximum_acceleration_rad_s2"],
            "prohibited_collision_frames_allowed": config["validation"][
                "prohibited_collision_frames_allowed"
            ],
        },
        "physics_validation": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "real_robot_execution": "NOT_PERFORMED",
    }
    return {
        "arm": arm,
        "left_hand": left,
        "right_hand": right,
        "hand_action": hand_action,
        "full_action": full,
        "geometry": geometry,
        "metrics": metrics,
        "validation": validation,
        "hand": hand,
    }


def export_arm_episode(
    output_root: Path,
    result: ArmEpisodeResult,
    selected_config_sha256: str,
    implementation_sha256: str,
) -> Path:
    directory = output_root / result.method / f"episode_{result.episode.episode_id:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    arm = np.asarray(result.solved["q"], dtype=np.float32)
    atomic_npz(
        directory / "g1_arm_action.npz",
        action=arm,
        timestamps=result.episode.timestamps,
        fps=np.asarray(result.episode.fps),
        target_left_wrist_position=result.targets["left_wrist_position"],
        target_right_wrist_position=result.targets["right_wrist_position"],
        target_left_wrist_rotation=result.targets["left_wrist_rotation"],
        target_right_wrist_rotation=result.targets["right_wrist_rotation"],
        achieved_left_wrist_position=result.wrist["left_position"],
        achieved_right_wrist_position=result.wrist["right_position"],
        target_left_task_tool_position=result.targets["left_tool_position"],
        target_right_task_tool_position=result.targets["right_tool_position"],
        method=np.asarray(result.method),
        representation=np.asarray(result.targets["representation"]),
        selected_config_sha256=np.asarray(selected_config_sha256),
        implementation_sha256=np.asarray(implementation_sha256),
        real_robot_command_allowed=np.asarray(False),
    )
    atomic_json(directory / "arm_metrics.json", result.metrics)
    files = ("g1_arm_action.npz", "arm_metrics.json")
    atomic_json(
        directory / "manifest.json",
        {
            "schema_version": "common_arm_v2_episode",
            "method": result.method,
            "episode_id": result.episode.episode_id,
            "selected_config_sha256": selected_config_sha256,
            "implementation_sha256": implementation_sha256,
            "files": {name: sha256_file(directory / name) for name in files},
            "offline_only": True,
            "training_executed": False,
            "physics_executed": False,
            "real_robot_commands": False,
        },
    )
    return directory


def export_integrated_episode(
    output_root: Path,
    dataset_name: str,
    arm_result: ArmEpisodeResult,
    integrated: Mapping[str, Any],
    g1: G1Kinematics,
    runtime: Any,
    selected_config_sha256: str,
    hand_dependency_sha256: str | None,
) -> Path:
    directory = output_root / dataset_name / f"episode_{arm_result.episode.episode_id:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    arm = np.asarray(integrated["arm"], dtype=np.float32)
    left = np.asarray(integrated["left_hand"], dtype=np.float32)
    right = np.asarray(integrated["right_hand"], dtype=np.float32)
    hand = np.column_stack((left, right)).astype(np.float32)
    full = np.column_stack((arm, hand)).astype(np.float32)
    arm_names = np.asarray(g1.info["joint_names"]).astype("U64")
    left_names = np.asarray(runtime.hand_joint_names["left"]).astype("U64")
    right_names = np.asarray(runtime.hand_joint_names["right"]).astype("U64")
    atomic_json(
        directory / "source_metadata.json",
        {
            "authoritative_dataset_root": str(
                Path(arm_result.episode.image_reference["dataset_root"]).resolve()
            ),
            "episode_id": arm_result.episode.episode_id,
            "frame_count": len(arm),
            "fps": arm_result.episode.fps,
            "language_instruction": arm_result.episode.task,
            "source_action_sha256": array_sha256(
                arm_result.episode.action.astype(np.float32)
            ),
            "source_state_sha256": array_sha256(
                arm_result.episode.state.astype(np.float32)
            ),
            "image_reference": arm_result.episode.image_reference,
            "images_duplicated": False,
        },
    )
    atomic_npz(
        directory / "g1_arm_action.npz",
        action=arm,
        timestamps=arm_result.episode.timestamps,
        fps=np.asarray(arm_result.episode.fps),
        joint_names=arm_names,
        selected_config_sha256=np.asarray(selected_config_sha256),
        real_robot_command_allowed=np.asarray(False),
    )
    atomic_npz(
        directory / "g1_hand_action.npz",
        action=hand,
        left_action=left,
        right_action=right,
        timestamps=arm_result.episode.timestamps,
        fps=np.asarray(arm_result.episode.fps),
        left_joint_names=left_names,
        right_joint_names=right_names,
        left_phase=np.asarray(integrated["hand"]["left_phase"]).astype("U12"),
        right_phase=np.asarray(integrated["hand"]["right_phase"]).astype("U12"),
        mapper=np.asarray(integrated["hand"]["mapper"]),
        hand_dependency_sha256=np.asarray(hand_dependency_sha256 or "NOT_APPLICABLE"),
        real_robot_command_allowed=np.asarray(False),
    )
    atomic_npz(
        directory / "g1_full_action.npz",
        action=full,
        timestamps=arm_result.episode.timestamps,
        fps=np.asarray(arm_result.episode.fps),
        joint_names=np.concatenate((arm_names, left_names, right_names)),
        dataset=np.asarray(dataset_name),
        real_robot_command_allowed=np.asarray(False),
    )
    atomic_json(directory / "retargeting_metrics.json", integrated["metrics"])
    atomic_json(directory / "validation.json", integrated["validation"])
    files = (
        "source_metadata.json",
        "g1_arm_action.npz",
        "g1_hand_action.npz",
        "g1_full_action.npz",
        "retargeting_metrics.json",
        "validation.json",
    )
    atomic_json(
        directory / "manifest.json",
        {
            "schema_version": "integrated_g1_retargeting_v2_episode",
            "dataset": dataset_name,
            "episode_id": arm_result.episode.episode_id,
            "status": integrated["validation"]["status"],
            "selected_config_sha256": selected_config_sha256,
            "hand_dependency_sha256": hand_dependency_sha256,
            "method_separation": "Dataset A and Dataset B are separate output roots",
            "files": {name: sha256_file(directory / name) for name in files},
            "offline_only": True,
            "training_executed": False,
            "physics_executed": False,
            "real_robot_commands": False,
        },
    )
    return directory


__all__ = [
    "INTEGRATED_STATUSES",
    "evaluate_integrated_episode",
    "export_arm_episode",
    "export_integrated_episode",
    "map_baseline_hand",
    "map_proposed_hand",
    "validate_candidate_joint_order",
]
