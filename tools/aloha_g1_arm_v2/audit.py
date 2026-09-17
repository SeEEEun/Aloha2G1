"""Workspace, candidate, arm-only collision, and first-failure analysis."""
from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from aloha_g1_dataset_v1.core import (
    G1Kinematics,
    RepresentationBuilder,
    SourceEpisode,
    branch_flags,
    rotation_errors,
    stats,
)
from aloha_g1_hand_v2.collision_eval import (
    CollisionClassifier,
    body_digit,
)

from .solver import AcceptanceAwareTemporalIK, static_pose_solve


@dataclass
class PreparedEpisode:
    episode: SourceEpisode
    source_fk: dict[str, Any]
    detected: dict[str, Any]


@dataclass
class ArmEpisodeResult:
    method: str
    episode: SourceEpisode
    source_fk: dict[str, Any]
    detected: dict[str, Any]
    targets: dict[str, Any]
    solved: dict[str, Any]
    wrist: dict[str, np.ndarray]
    metrics: dict[str, Any]
    arm_collision_flags: np.ndarray
    arm_collision_categories: dict[str, int]
    arm_collision_pairs: dict[str, int]


def candidate_config(
    base: Mapping[str, Any],
    hand_candidate: Mapping[str, Any],
    nominal_q: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config["schema_version"] = "aloha_g1_retargeting_arm_v2_runtime"
    config["nominal_g1_arm_q"] = np.asarray(nominal_q, dtype=np.float64).tolist()
    mapping = config["workspace_mapping"]
    mapping["uniform_scale"] = float(scale)
    mapping["midpoint_scale"] = float(scale)
    mapping["relative_vector_scale"] = float(scale)
    config["target_frames"]["left_wrist_to_physical_pinch"] = hand_candidate[
        "left_static_wrist_to_pinch"
    ]
    config["target_frames"]["right_wrist_to_physical_pinch"] = hand_candidate[
        "right_static_wrist_to_pinch"
    ]
    config["ik"]["backend"] = (
        "shared_temporally_regularized_damped_least_squares_v2_"
        "acceptance_aware_best_state"
    )
    return config


def configure_g1(
    g1: G1Kinematics,
    runtime_config: Mapping[str, Any],
) -> None:
    g1.nominal_q = np.asarray(runtime_config["nominal_g1_arm_q"], dtype=np.float64)
    g1.tool_local = {
        side: np.asarray(
            runtime_config["target_frames"][f"{side}_wrist_to_physical_pinch"],
            dtype=np.float64,
        )
        for side in ("left", "right")
    }


def build_targets(
    g1: G1Kinematics,
    runtime_config: Mapping[str, Any],
    method: str,
    source_fk: Mapping[str, Any],
) -> dict[str, Any]:
    configure_g1(g1, runtime_config)
    return RepresentationBuilder(dict(runtime_config), g1).build(method, dict(source_fk))


def arm_only_collision_at_current_state(
    classifier: CollisionClassifier,
) -> list[Any]:
    return [
        record
        for record in classifier.records()
        if record.v1_gate_relevant
        and all(body_digit(name) is None for name in record.bodies)
    ]


def arm_collision_sweep(
    runtime: Any,
    classifier: CollisionClassifier,
    arm: np.ndarray,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    flags = np.zeros(len(arm), dtype=bool)
    categories: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    left = runtime.open_hand_q["left"]
    right = runtime.open_hand_q["right"]
    for index, row in enumerate(arm):
        runtime.assign(row, left, right)
        records = arm_only_collision_at_current_state(classifier)
        if records:
            flags[index] = True
            for category in {record.category for record in records}:
                categories[category] += 1
            for record in records:
                pairs[record.pair] += 1
    return flags, dict(sorted(categories.items())), dict(sorted(pairs.items()))


def _actual_tool_from_wrist(
    wrist: Mapping[str, np.ndarray],
    tool_local: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        rotation = np.asarray(wrist[f"{side}_rotation"])
        position = np.asarray(wrist[f"{side}_position"])
        local = np.asarray(tool_local[side])
        output[f"{side}_position"] = position + np.einsum(
            "tij,j->ti", rotation, local[:3, 3]
        )
        output[f"{side}_rotation"] = np.einsum(
            "tij,jk->tik", rotation, local[:3, :3]
        )
    return output


def evaluate_arm_episode(
    method: str,
    prepared: PreparedEpisode,
    runtime_config: Mapping[str, Any],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
) -> ArmEpisodeResult:
    configure_g1(g1, runtime_config)
    targets = RepresentationBuilder(dict(runtime_config), g1).build(
        method, prepared.source_fk
    )
    solved = AcceptanceAwareTemporalIK(dict(runtime_config), g1).solve(targets)
    arm = np.asarray(solved["q"])
    wrist = g1.evaluate_wrist(arm, targets)
    collision, categories, pairs = arm_collision_sweep(
        collision_runtime, classifier, arm
    )
    ik = runtime_config["ik"]
    components = {
        "left_position": wrist["left_position_error"] <= float(ik["position_tolerance_m"]),
        "right_position": wrist["right_position_error"] <= float(ik["position_tolerance_m"]),
        "left_orientation": wrist["left_orientation_error"] <= float(ik["orientation_tolerance_rad"]),
        "right_orientation": wrist["right_orientation_error"] <= float(ik["orientation_tolerance_rad"]),
    }
    success = np.logical_and.reduce(list(components.values()))
    violations = (arm < g1.limits[:, 0] - 1e-9) | (arm > g1.limits[:, 1] + 1e-9)
    margin = np.minimum(arm - g1.limits[:, 0], g1.limits[:, 1] - arm)
    actual_tool = _actual_tool_from_wrist(wrist, g1.tool_local)
    target_left = np.asarray(targets["left_tool_position"])
    target_right = np.asarray(targets["right_tool_position"])
    actual_left = actual_tool["left_position"]
    actual_right = actual_tool["right_position"]
    target_midpoint = 0.5 * (target_left + target_right)
    actual_midpoint = 0.5 * (actual_left + actual_right)
    target_relative = target_right - target_left
    actual_relative = actual_right - actual_left
    target_distance = np.linalg.norm(target_relative, axis=1)
    actual_distance = np.linalg.norm(actual_relative, axis=1)
    step = np.abs(np.diff(arm, axis=0))
    velocity = step * prepared.episode.fps
    acceleration = np.abs(np.diff(arm, n=2, axis=0)) * prepared.episode.fps**2
    validation = runtime_config["validation"]
    branches = branch_flags(
        arm,
        float(validation["branch_absolute_step_norm_rad"]),
        float(validation["branch_local_multiplier"]),
    )
    metrics = {
        "method": method,
        "episode_id": int(prepared.episode.episode_id),
        "frame_count": len(arm),
        "finite": bool(np.isfinite(arm).all()),
        "ik_success_rate": float(np.mean(success)),
        "ik_failed_frame_count": int(np.count_nonzero(~success)),
        "ik_component_success_rate": {
            key: float(np.mean(value)) for key, value in components.items()
        },
        "joint_limit_violation_count": int(np.count_nonzero(violations)),
        "minimum_joint_limit_margin_rad": float(np.min(margin)),
        "arm_only_collision_frames": int(np.count_nonzero(collision)),
        "arm_collision_categories": categories,
        "arm_collision_pairs": pairs,
        "wrist_error_mean_m": float(
            0.5
            * (
                np.mean(wrist["left_position_error"])
                + np.mean(wrist["right_position_error"])
            )
        ),
        "wrist_error_median_m": float(
            np.median(
                np.concatenate(
                    (wrist["left_position_error"], wrist["right_position_error"])
                )
            )
        ),
        "left_wrist_error_m": stats(wrist["left_position_error"]),
        "right_wrist_error_m": stats(wrist["right_position_error"]),
        "left_orientation_error_rad": stats(wrist["left_orientation_error"]),
        "right_orientation_error_rad": stats(wrist["right_orientation_error"]),
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
        "maximum_joint_step_rad": float(np.max(step, initial=0.0)),
        "maximum_velocity_rad_s": float(np.max(velocity, initial=0.0)),
        "maximum_acceleration_rad_s2": float(np.max(acceleration, initial=0.0)),
        "branch_discontinuity_count": int(np.count_nonzero(branches)),
        "iteration_budget_exhausted_frames": int(
            sum(row["budget_exhausted"] for row in solved["reprojection_metadata"])
        ),
        "numerical_fallback_frames": int(
            sum(row["numerical_lstsq_fallback"] for row in solved["reprojection_metadata"])
        ),
    }
    return ArmEpisodeResult(
        method=method,
        episode=prepared.episode,
        source_fk=prepared.source_fk,
        detected=prepared.detected,
        targets=targets,
        solved=solved,
        wrist=wrist,
        metrics=metrics,
        arm_collision_flags=collision,
        arm_collision_categories=categories,
        arm_collision_pairs=pairs,
    )


def representative_indices(source_fk: Mapping[str, Any], count: int) -> np.ndarray:
    length = len(source_fk["left_position"])
    priority: list[int] = [0, length - 1]
    left = np.asarray(source_fk["left_position"])
    right = np.asarray(source_fk["right_position"])
    midpoint = 0.5 * (left + right)
    relative = right - left
    for values in (midpoint, relative, left, right):
        displacement = np.linalg.norm(values - values[0], axis=1)
        priority.append(int(np.argmax(displacement)))
    for axis in range(3):
        priority.extend((int(np.argmin(midpoint[:, axis])), int(np.argmax(midpoint[:, axis]))))
        priority.extend((int(np.argmin(relative[:, axis])), int(np.argmax(relative[:, axis]))))
    unique: list[int] = []
    for value in priority:
        if value not in unique:
            unique.append(value)
        if len(unique) == count:
            return np.asarray(sorted(unique), dtype=np.int64)
    for value in np.linspace(0, length - 1, count, dtype=np.int64):
        if int(value) not in unique:
            unique.append(int(value))
        if len(unique) == count:
            break
    return np.asarray(sorted(unique), dtype=np.int64)


def static_screen_candidate(
    candidate_id: str,
    runtime_config: Mapping[str, Any],
    prepared: Mapping[int, PreparedEpisode],
    calibration_ids: list[int],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
    screen_config: Mapping[str, Any],
) -> dict[str, Any]:
    configure_g1(g1, runtime_config)
    total = Counter()
    errors: list[float] = []
    per_method: dict[str, Counter[str]] = {
        "baseline": Counter(),
        "proposed": Counter(),
    }
    for method in ("baseline", "proposed"):
        for episode_id in calibration_ids:
            item = prepared[episode_id]
            targets = RepresentationBuilder(dict(runtime_config), g1).build(
                method, item.source_fk
            )
            indices = representative_indices(
                item.source_fk,
                int(screen_config["representative_frames_per_episode"]),
            )
            previous = g1.nominal_q.copy()
            for index in indices:
                result = static_pose_solve(
                    g1,
                    {
                        side: np.asarray(targets[f"{side}_wrist_position"])[index]
                        for side in ("left", "right")
                    },
                    {
                        side: np.asarray(targets[f"{side}_wrist_rotation"])[index]
                        for side in ("left", "right")
                    },
                    [previous, g1.nominal_q],
                    iterations=int(screen_config["static_probe_iterations"]),
                    position_tolerance_m=float(
                        screen_config["static_position_tolerance_m"]
                    ),
                    orientation_tolerance_rad=float(
                        screen_config["static_orientation_tolerance_rad"]
                    ),
                )
                previous = np.asarray(result["q"])
                per_method[method]["queries"] += 1
                per_method[method]["accepted"] += int(result["accepted"])
                per_method[method]["joint_limit_violations"] += int(
                    result["minimum_joint_limit_margin_rad"] < -1e-9
                )
                errors.append(float(max(result["position_errors_m"])))
                collision_runtime.assign(
                    result["q"],
                    collision_runtime.open_hand_q["left"],
                    collision_runtime.open_hand_q["right"],
                )
                collision = bool(arm_only_collision_at_current_state(classifier))
                per_method[method]["arm_collision_queries"] += int(collision)
    output_methods = {
        method: {
            **dict(values),
            "success_rate": float(values["accepted"] / max(values["queries"], 1)),
        }
        for method, values in per_method.items()
    }
    total.update({
        "queries": sum(value["queries"] for value in per_method.values()),
        "accepted": sum(value["accepted"] for value in per_method.values()),
        "joint_limit_violations": sum(
            value["joint_limit_violations"] for value in per_method.values()
        ),
        "arm_collision_queries": sum(
            value["arm_collision_queries"] for value in per_method.values()
        ),
    })
    return {
        "candidate_id": candidate_id,
        "methods": output_methods,
        "query_count": int(total["queries"]),
        "minimum_method_success_rate": min(
            output_methods[method]["success_rate"]
            for method in ("baseline", "proposed")
        ),
        "joint_limit_violations": int(total["joint_limit_violations"]),
        "arm_collision_queries": int(total["arm_collision_queries"]),
        "position_error_mean_m": float(np.mean(errors)),
        "position_error_max_m": float(np.max(errors)),
    }


def summarize_arm_results(results: Mapping[str, list[ArmEpisodeResult]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, rows in results.items():
        output[method] = {
            "episode_count": len(rows),
            "mean_ik_success_rate": float(
                np.mean([row.metrics["ik_success_rate"] for row in rows])
            ),
            "median_ik_success_rate": float(
                np.median([row.metrics["ik_success_rate"] for row in rows])
            ),
            "joint_limit_violation_count": int(
                sum(row.metrics["joint_limit_violation_count"] for row in rows)
            ),
            "arm_only_collision_frames": int(
                sum(row.metrics["arm_only_collision_frames"] for row in rows)
            ),
            "arm_collision_episode_count": int(
                sum(row.metrics["arm_only_collision_frames"] > 0 for row in rows)
            ),
            "wrist_error_mean_m": float(
                np.mean([row.metrics["wrist_error_mean_m"] for row in rows])
            ),
            "midpoint_error_mean_m": float(
                np.mean([row.metrics["midpoint_error_mean_m"] for row in rows])
            ),
            "relative_vector_error_mean_m": float(
                np.mean([row.metrics["relative_vector_error_mean_m"] for row in rows])
            ),
            "distance_change_error_mean_m": float(
                np.mean([row.metrics["distance_change_error_mean_m"] for row in rows])
            ),
            "finite": bool(all(row.metrics["finite"] for row in rows)),
            "maximum_joint_step_rad": float(
                max(row.metrics["maximum_joint_step_rad"] for row in rows)
            ),
            "maximum_velocity_rad_s": float(
                max(row.metrics["maximum_velocity_rad_s"] for row in rows)
            ),
            "maximum_acceleration_rad_s2": float(
                max(row.metrics["maximum_acceleration_rad_s2"] for row in rows)
            ),
        }
    return output


def full_candidate_score(
    summary: Mapping[str, Any],
    scale: float,
    anchor_distance_rad: float,
    reference_scale: float,
) -> tuple[Any, ...]:
    baseline = summary["baseline"]
    proposed = summary["proposed"]
    minimum_success = min(
        baseline["mean_ik_success_rate"], proposed["mean_ik_success_rate"]
    )
    return (
        -minimum_success,
        baseline["joint_limit_violation_count"]
        + proposed["joint_limit_violation_count"],
        baseline["arm_only_collision_frames"]
        + proposed["arm_only_collision_frames"],
        baseline["wrist_error_mean_m"] + proposed["wrist_error_mean_m"],
        proposed["midpoint_error_mean_m"] + proposed["relative_vector_error_mean_m"],
        abs(float(scale) - float(reference_scale)),
        float(anchor_distance_rad),
    )


def deterministic_fk_workspace(
    g1: G1Kinematics,
    sample_power: int = 12,
) -> dict[str, Any]:
    """Sobol joint-limit sampling for fixed-base wrist reach diagnostics."""
    sampler = qmc.Sobol(d=7, scramble=False)
    unit = sampler.random_base2(sample_power)
    positions: dict[str, np.ndarray] = {}
    distances: dict[str, np.ndarray] = {}
    shoulder_positions: dict[str, np.ndarray] = {}
    original = g1.nominal_q.copy()
    for side, block, shoulder_name in (
        ("left", slice(0, 7), "left_shoulder_pitch_link"),
        ("right", slice(7, 14), "right_shoulder_pitch_link"),
    ):
        lower = g1.limits[block, 0]
        upper = g1.limits[block, 1]
        q_values = qmc.scale(unit, lower, upper)
        side_positions = np.empty((len(q_values), 3), dtype=np.float64)
        side_shoulders = np.empty_like(side_positions)
        shoulder_id = mujoco.mj_name2id(
            g1.model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name
        )
        for index, side_q in enumerate(q_values):
            q = original.copy()
            q[block] = side_q
            state = g1.wrist_state(q)
            side_positions[index] = state[f"{side}_position"]
            side_shoulders[index] = g1.data.xpos[shoulder_id]
        positions[side] = side_positions
        shoulder_positions[side] = side_shoulders
        distances[side] = np.linalg.norm(side_positions - side_shoulders, axis=1)
    g1.nominal_q = original
    return {
        "sampling": {
            "type": "deterministic_unscrambled_sobol_joint_limit_sampling",
            "sample_count_per_side": int(2**sample_power),
            "sample_power": int(sample_power),
        },
        "positions": positions,
        "shoulder_positions": shoulder_positions,
        "shoulder_wrist_distances": distances,
        "bounds": {
            side: {
                "minimum_xyz_m": positions[side].min(axis=0).tolist(),
                "maximum_xyz_m": positions[side].max(axis=0).tolist(),
                "maximum_shoulder_wrist_distance_m": float(np.max(distances[side])),
                "p99_shoulder_wrist_distance_m": float(
                    np.percentile(distances[side], 99)
                ),
            }
            for side in ("left", "right")
        },
    }


def first_failure_record(
    method: str,
    episode_id: int,
    arm: np.ndarray,
    target_positions: Mapping[str, np.ndarray],
    target_rotations: Mapping[str, np.ndarray],
    achieved_positions: Mapping[str, np.ndarray],
    achieved_rotations: Mapping[str, np.ndarray],
    phases: Mapping[str, np.ndarray],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
    workspace: Mapping[str, Any],
    ik_config: Mapping[str, Any],
    solver_metadata: list[Mapping[str, Any]] | None,
    probe_iterations: int,
) -> dict[str, Any]:
    position_errors = {
        side: np.linalg.norm(
            achieved_positions[side] - target_positions[side], axis=1
        )
        for side in ("left", "right")
    }
    orientation_error = {
        side: rotation_errors(achieved_rotations[side], target_rotations[side])
        for side in ("left", "right")
    }
    failed = np.zeros(len(arm), dtype=bool)
    for side in ("left", "right"):
        failed |= position_errors[side] > float(ik_config["position_tolerance_m"])
        failed |= orientation_error[side] > float(ik_config["orientation_tolerance_rad"])
    indices = np.flatnonzero(failed)
    if not len(indices):
        for index, row in enumerate(arm):
            collision_runtime.assign(
                row,
                collision_runtime.open_hand_q["left"],
                collision_runtime.open_hand_q["right"],
            )
            records = arm_only_collision_at_current_state(classifier)
            if records:
                category = (
                    "ARM_TORSO_COLLISION_BLOCK"
                    if any(record.category == "ARM_TORSO" for record in records)
                    else "CROSS_ARM_COLLISION_BLOCK"
                )
                return {
                    "method": method,
                    "episode": episode_id,
                    "frame": index,
                    "side": "bimanual",
                    "semantic_phase": {
                        side: str(phases[side][index]) for side in ("left", "right")
                    },
                    "classification": category,
                    "collision_pair": records[0].pair,
                    "best_effort_q_status": "FINITE_IN_LIMITS",
                }
        return {
            "method": method,
            "episode": episode_id,
            "frame": None,
            "side": None,
            "classification": "NO_ARM_IK_FAILURE",
        }

    index = int(indices[0])
    failed_components: list[str] = []
    for side in ("left", "right"):
        if position_errors[side][index] > float(ik_config["position_tolerance_m"]):
            failed_components.append(f"{side}_position")
        if orientation_error[side][index] > float(ik_config["orientation_tolerance_rad"]):
            failed_components.append(f"{side}_orientation")
    sides = sorted({value.split("_", 1)[0] for value in failed_components})
    side = sides[0] if len(sides) == 1 else "bimanual"
    q = np.asarray(arm[index])
    g1.assign_arm(q)
    shoulder_target: dict[str, float] = {}
    outside = False
    for current in ("left", "right"):
        shoulder_id = mujoco.mj_name2id(
            g1.model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{current}_shoulder_pitch_link",
        )
        distance = float(
            np.linalg.norm(target_positions[current][index] - g1.data.xpos[shoulder_id])
        )
        shoulder_target[current] = distance
        outside = outside or distance > (
            float(workspace["bounds"][current]["maximum_shoulder_wrist_distance_m"])
            + float(ik_config["position_tolerance_m"])
        )
    previous = arm[max(index - 1, 0)]
    probe = static_pose_solve(
        g1,
        {current: target_positions[current][index] for current in ("left", "right")},
        {current: target_rotations[current][index] for current in ("left", "right")},
        [q, previous, g1.nominal_q],
        iterations=probe_iterations,
        position_tolerance_m=float(ik_config["position_tolerance_m"]),
        orientation_tolerance_rad=float(ik_config["orientation_tolerance_rad"]),
    )
    collision_runtime.assign(
        q,
        collision_runtime.open_hand_q["left"],
        collision_runtime.open_hand_q["right"],
    )
    collision_records = arm_only_collision_at_current_state(classifier)
    margin = np.minimum(q - g1.limits[:, 0], g1.limits[:, 1] - q)
    frame_step = float(np.max(np.abs(q - previous))) if index else 0.0
    only_orientation = all(value.endswith("orientation") for value in failed_components)
    if not np.isfinite(q).all():
        classification = "NUMERICAL_FAILURE"
    elif outside:
        classification = "TARGET_OUTSIDE_REACHABLE_WORKSPACE"
    elif float(np.min(margin)) <= 1e-5:
        classification = "JOINT_LIMIT_BLOCK"
    elif only_orientation or (
        max(probe["position_errors_m"]) <= float(ik_config["position_tolerance_m"])
        and max(probe["orientation_errors_rad"]) > float(ik_config["orientation_tolerance_rad"])
    ):
        classification = "ORIENTATION_OVERCONSTRAINT"
    elif collision_records and any(
        record.category == "ARM_TORSO" for record in collision_records
    ):
        classification = "ARM_TORSO_COLLISION_BLOCK"
    elif collision_records:
        classification = "CROSS_ARM_COLLISION_BLOCK"
    elif probe["accepted"] and frame_step >= 0.95 * float(ik_config["max_frame_joint_step_rad"]):
        classification = "TEMPORAL_CONTINUITY_BLOCK"
    elif probe["accepted"]:
        classification = "IK_SEED_OR_BRANCH_FAILURE"
    elif solver_metadata is not None and solver_metadata[index]["budget_exhausted"]:
        classification = "ITERATION_BUDGET_EXHAUSTED"
    else:
        classification = "POSITION_TOLERANCE_FAILURE"
    metadata = solver_metadata[index] if solver_metadata is not None else None
    return {
        "method": method,
        "episode": episode_id,
        "frame": index,
        "side": side,
        "semantic_phase": {
            current: str(phases[current][index]) for current in ("left", "right")
        },
        "classification": classification,
        "failed_components": failed_components,
        "target_position": {
            current: target_positions[current][index].tolist()
            for current in ("left", "right")
        },
        "target_orientation": {
            current: target_rotations[current][index].tolist()
            for current in ("left", "right")
        },
        "shoulder_target_distance_m": shoulder_target,
        "nearest_joint_limit_margin_rad": float(np.min(margin)),
        "position_error_m": {
            current: float(position_errors[current][index])
            for current in ("left", "right")
        },
        "orientation_error_rad": {
            current: float(orientation_error[current][index])
            for current in ("left", "right")
        },
        "collision_pair": collision_records[0].pair if collision_records else None,
        "iterations": None if metadata is None else int(metadata["iterations"]),
        "iteration_logging": (
            "NOT_RECORDED_IN_V1_ARTIFACT; static probe recorded separately"
            if metadata is None
            else "COMMON_ARM_V2_REPROJECTION_METADATA"
        ),
        "static_probe_iterations": int(probe["iterations"]),
        "static_probe_accepted": bool(probe["accepted"]),
        "seed_provenance": "previous-q, frame best effort, then immutable nominal static probes",
        "best_effort_q_status": (
            "FINITE_IN_LIMITS"
            if np.isfinite(q).all() and float(np.min(margin)) >= -1e-9
            else "INVALID"
        ),
    }


def rotation_distribution(rotations: np.ndarray) -> dict[str, Any]:
    relative = np.einsum("ij,tjk->tik", rotations[0].T, rotations)
    angles = Rotation.from_matrix(relative).magnitude()
    return stats(angles)


__all__ = [
    "ArmEpisodeResult",
    "PreparedEpisode",
    "arm_collision_sweep",
    "arm_only_collision_at_current_state",
    "build_targets",
    "candidate_config",
    "configure_g1",
    "deterministic_fk_workspace",
    "evaluate_arm_episode",
    "first_failure_record",
    "full_candidate_score",
    "representative_indices",
    "rotation_distribution",
    "static_screen_candidate",
    "summarize_arm_results",
]
