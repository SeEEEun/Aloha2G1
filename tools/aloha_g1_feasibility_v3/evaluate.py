"""Strict validation, geometry metrics, and episode export for feasibility v3."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aloha_g1_dataset_v1.core import (
    G1Kinematics,
    array_sha256,
    branch_flags,
    physical_pinch_frame,
    raw_wrist_pose,
    rotation_errors,
    stats,
)
from aloha_g1_hand_v2.collision_eval import (
    CollisionClassifier,
    body_digit,
    is_torso,
)

from .common import (
    V2_INTEGRATED_ROOT,
    atomic_json,
    atomic_npz,
    load_json,
    sha256_file,
)
from .solver import FeasibilityResult, SIDES


STATUSES = (
    "PASS",
    "FAIL_IK",
    "FAIL_COLLISION",
    "FAIL_TEMPORAL",
    "FAIL_LIMIT",
    "FAIL_DATA",
    "FAIL_OTHER",
)


def _geometry(
    result: FeasibilityResult,
    runtime: Any,
    classifier: CollisionClassifier,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    count = len(result.q)
    values: dict[str, Any] = {}
    for side in SIDES:
        for frame_name in ("wrist", "pinch"):
            values[f"{side}_{frame_name}_position"] = np.empty(
                (count, 3), dtype=np.float64
            )
            values[f"{side}_{frame_name}_rotation"] = np.empty(
                (count, 3, 3), dtype=np.float64
            )
    flag_names = (
        "prohibited",
        "arm_only",
        "hand_prohibited",
        "hand_comprehensive",
        "cross_arm",
        "torso",
        "third_finger",
        "thumb_index",
        "same_hand_internal",
    )
    flags = {
        name: np.zeros(count, dtype=bool) for name in flag_names
    }
    categories: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    labels = {
        side: tuple(
            runtime_config["target_frames"][
                f"{side}_physical_pinch_contacts"
            ]
        )
        for side in SIDES
    }
    for frame in range(count):
        runtime.assign(
            result.q[frame], result.left_hand[frame], result.right_hand[frame]
        )
        for side in SIDES:
            wrist = raw_wrist_pose(runtime, side)
            pinch = physical_pinch_frame(runtime, side, labels[side])
            values[f"{side}_wrist_position"][frame] = wrist[:3, 3]
            values[f"{side}_wrist_rotation"][frame] = wrist[:3, :3]
            values[f"{side}_pinch_position"][frame] = pinch[:3, 3]
            values[f"{side}_pinch_rotation"][frame] = pinch[:3, :3]
        records = classifier.records()
        prohibited = [row for row in records if row.v1_gate_relevant]
        comprehensive_hand = [
            row
            for row in records
            if any(body_digit(name) is not None for name in row.bodies)
        ]
        hand_prohibited = [
            row
            for row in prohibited
            if any(body_digit(name) is not None for name in row.bodies)
        ]
        arm_only = [
            row
            for row in prohibited
            if all(body_digit(name) is None for name in row.bodies)
        ]
        flags["prohibited"][frame] = bool(prohibited)
        flags["arm_only"][frame] = bool(arm_only)
        flags["hand_prohibited"][frame] = bool(hand_prohibited)
        flags["hand_comprehensive"][frame] = bool(comprehensive_hand)
        flags["cross_arm"][frame] = any(
            row.category in {"CROSS_ARM", "HAND_HAND"}
            for row in prohibited
        )
        flags["torso"][frame] = any(
            any(is_torso(name) for name in row.bodies)
            for row in prohibited
        )
        flags["third_finger"][frame] = any(
            any(body_digit(name) == "THIRD" for name in row.bodies)
            for row in comprehensive_hand
        )
        flags["thumb_index"][frame] = any(
            any(body_digit(name) in {"THUMB", "INDEX"} for name in row.bodies)
            for row in comprehensive_hand
        )
        flags["same_hand_internal"][frame] = any(
            row.enhanced_same_hand for row in records
        )
        for category in {row.category for row in records}:
            categories[category] += 1
        for row in records:
            pairs[row.pair] += 1
    values.update({f"{key}_flag": value for key, value in flags.items()})
    values["category_frame_incidence"] = dict(sorted(categories.items()))
    values["top_pairs"] = [
        {"pair": pair, "events": events}
        for pair, events in pairs.most_common(20)
    ]
    return values


def _v2_wrist(g1: G1Kinematics, q: np.ndarray) -> dict[str, np.ndarray]:
    output = {
        f"{side}_{kind}": np.empty(
            (len(q), 3, 3) if kind == "rotation" else (len(q), 3),
            dtype=np.float64,
        )
        for side in SIDES
        for kind in ("position", "rotation")
    }
    for frame, row in enumerate(q):
        state = g1.wrist_state(row)
        for side in SIDES:
            output[f"{side}_position"][frame] = state[f"{side}_position"]
            output[f"{side}_rotation"][frame] = state[f"{side}_rotation"]
    return output


def evaluate_result(
    result: FeasibilityResult,
    runtime: Any,
    classifier: CollisionClassifier,
    g1: G1Kinematics,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    episode = result.episode
    geometry = _geometry(result, runtime, classifier, runtime_config)
    target_position = {
        side: episode.targets[f"{side}_wrist_position"] for side in SIDES
    }
    target_rotation = {
        side: episode.targets[f"{side}_wrist_rotation"] for side in SIDES
    }
    position_error = {
        side: np.linalg.norm(
            geometry[f"{side}_wrist_position"] - target_position[side],
            axis=1,
        )
        for side in SIDES
    }
    orientation_error = {
        side: rotation_errors(
            geometry[f"{side}_wrist_rotation"], target_rotation[side]
        )
        for side in SIDES
    }
    solver_config = result.metadata["solver_parameters"]
    position_tolerance = float(solver_config["position_tolerance_m"])
    orientation_tolerance = float(solver_config["orientation_tolerance_rad"])
    slack_bound = float(solver_config["orientation_slack_bound_rad"])
    success = np.ones(len(result.q), dtype=bool)
    for side in SIDES:
        success &= position_error[side] <= position_tolerance
        success &= orientation_error[side] <= orientation_tolerance + slack_bound

    target_tool_rotation = {
        side: np.einsum(
            "tij,jk->tik",
            target_rotation[side],
            np.asarray(g1.tool_local[side])[:3, :3],
        )
        for side in SIDES
    }
    pinch_error = {
        side: np.linalg.norm(
            geometry[f"{side}_pinch_position"]
            - episode.targets[f"{side}_tool_position"],
            axis=1,
        )
        for side in SIDES
    }
    pinch_orientation_error = {
        side: rotation_errors(
            geometry[f"{side}_pinch_rotation"], target_tool_rotation[side]
        )
        for side in SIDES
    }
    # The source-semantic mask is shared diagnostic metadata, not an A objective.
    with np.load(
        V2_INTEGRATED_ROOT
        / "dataset_b"
        / f"episode_{episode.episode_id:06d}"
        / "g1_hand_action.npz",
        allow_pickle=False,
    ) as payload:
        semantic_phase = {
            side: payload[f"{side}_phase"].astype(str) for side in SIDES
        }
    task_mask = {
        side: np.isin(semantic_phase[side], ["GRASP", "HOLD"])
        for side in SIDES
    }
    actual_left = geometry["left_pinch_position"]
    actual_right = geometry["right_pinch_position"]
    target_left = episode.targets["left_tool_position"]
    target_right = episode.targets["right_tool_position"]
    actual_midpoint = 0.5 * (actual_left + actual_right)
    target_midpoint = 0.5 * (target_left + target_right)
    actual_relative = actual_right - actual_left
    target_relative = target_right - target_left
    actual_distance = np.linalg.norm(actual_relative, axis=1)
    target_distance = np.linalg.norm(target_relative, axis=1)

    full = np.column_stack(
        (result.q, result.left_hand, result.right_hand)
    )
    fps = float(episode.fps)
    step = np.abs(np.diff(full, axis=0))
    velocity = step * fps
    acceleration = np.abs(np.diff(full, n=2, axis=0)) * fps**2
    validation = runtime_config["validation"]
    branches = branch_flags(
        result.q,
        float(validation["branch_absolute_step_norm_rad"]),
        float(validation["branch_local_multiplier"]),
    )
    margins = np.minimum(
        result.q - g1.limits[:, 0], g1.limits[:, 1] - result.q
    )
    violations = (result.q < g1.limits[:, 0] - 1e-9) | (
        result.q > g1.limits[:, 1] + 1e-9
    )
    active = margins <= float(
        solver_config["active_joint_bound_threshold_rad"]
    )
    active_names = Counter()
    active_sides = Counter()
    for _, joint in np.argwhere(active):
        name = str(g1.info["joint_names"][joint])
        active_names[name] += 1
        active_sides["left" if joint < 7 else "right"] += 1

    q_delta = result.q - episode.q_v2
    v2_wrist = _v2_wrist(g1, episode.q_v2)
    wrist_position_delta = np.concatenate(
        [
            np.linalg.norm(
                geometry[f"{side}_wrist_position"]
                - v2_wrist[f"{side}_position"],
                axis=1,
            )
            for side in SIDES
        ]
    )
    wrist_orientation_delta = np.concatenate(
        [
            rotation_errors(
                geometry[f"{side}_wrist_rotation"],
                v2_wrist[f"{side}_rotation"],
            )
            for side in SIDES
        ]
    )
    v2_task_success = np.ones(len(result.q), dtype=bool)
    for side in SIDES:
        old_position = np.linalg.norm(
            v2_wrist[f"{side}_position"] - target_position[side], axis=1
        )
        old_orientation = rotation_errors(
            v2_wrist[f"{side}_rotation"], target_rotation[side]
        )
        v2_task_success &= old_position <= position_tolerance
        v2_task_success &= old_orientation <= orientation_tolerance
    exact_arm = np.all(result.q == episode.q_v2, axis=1)

    collision_count = int(np.count_nonzero(geometry["prohibited_flag"]))
    temporal_ok = bool(
        np.count_nonzero(branches) == 0
        and float(np.max(step, initial=0.0))
        <= float(validation["maximum_joint_step_rad"])
        and float(np.max(velocity, initial=0.0))
        <= float(validation["maximum_velocity_rad_s"])
        and float(np.max(acceleration, initial=0.0))
        <= float(validation["maximum_acceleration_rad_s2"])
    )
    finite = bool(np.isfinite(full).all())
    checks = {
        "data": finite and full.shape == (len(result.q), 28),
        "limits": int(np.count_nonzero(violations)) == 0,
        "ik": float(np.mean(success))
        >= float(solver_config["required_ik_success_rate"]),
        "collision": collision_count
        <= int(validation["prohibited_collision_frames_allowed"]),
        "temporal": temporal_ok,
        "semantic": bool(
            len(episode.left_phase) == len(result.q)
            and len(episode.right_phase) == len(result.q)
        ),
    }
    status = "PASS"
    first_gate = None
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
            first_gate = key
            break
    first_failure = {
        "gate": first_gate,
        "classification": None,
        "frame": None,
        "side": None,
    }
    if first_gate == "ik":
        failed = np.flatnonzero(~success)
        frame = int(failed[0])
        side_index = int(
            np.argmax(
                [
                    position_error[side][frame] / position_tolerance
                    + orientation_error[side][frame]
                    / (orientation_tolerance + slack_bound)
                    for side in SIDES
                ]
            )
        )
        side = SIDES[side_index]
        if float(np.min(margins[frame])) <= float(
            solver_config["active_joint_bound_threshold_rad"]
        ):
            classification = "JOINT_LIMIT_BLOCK"
        elif position_error[side][frame] > position_tolerance:
            classification = "SOURCE_DEMONSTRATION_OUTSIDE_TARGET_FEASIBILITY"
        else:
            classification = "ORIENTATION_SLACK_BOUND_EXHAUSTED"
        first_failure.update(
            {
                "classification": classification,
                "frame": frame,
                "side": side,
                "position_error_m": float(position_error[side][frame]),
                "orientation_error_rad": float(
                    orientation_error[side][frame]
                ),
                "minimum_joint_limit_margin_rad": float(
                    np.min(margins[frame])
                ),
            }
        )
    elif first_gate == "collision":
        frame = int(np.flatnonzero(geometry["prohibited_flag"])[0])
        first_failure.update(
            {
                "classification": "PROHIBITED_COLLISION_REMAINS",
                "frame": frame,
            }
        )
    elif first_gate == "temporal":
        first_failure["classification"] = (
            "SOURCE_DEMONSTRATION_OUTSIDE_TARGET_FEASIBILITY"
        )

    metrics = {
        "schema_version": "common_feasibility_v3_episode_metrics",
        "dataset": episode.dataset_name,
        "method": episode.method,
        "representation": episode.representation,
        "episode_id": episode.episode_id,
        "status": status,
        "frame_count": len(result.q),
        "fps": fps,
        "finite_values": finite,
        "action_shape": list(full.shape),
        "ik_success_rate": float(np.mean(success)),
        "ik_failed_frame_count": int(np.count_nonzero(~success)),
        "joint_limit_violation_count": int(np.count_nonzero(violations)),
        "active_joint_bound_frames": int(np.count_nonzero(np.any(active, axis=1))),
        "active_joint_bound_entries": int(np.count_nonzero(active)),
        "minimum_joint_limit_margin_rad": float(np.min(margins)),
        "active_bound_joint_names": dict(active_names.most_common()),
        "active_bound_sides": dict(active_sides),
        "orientation_slack_rad": stats(result.orientation_slack),
        "orientation_slack_frame_count": int(
            np.count_nonzero(np.any(result.orientation_slack > 0.0, axis=1))
        ),
        "orientation_slack_bound_rad": slack_bound,
        "orientation_slack_requested_rad": stats(
            result.orientation_slack_requested
        ),
        "orientation_slack_excess_rad": stats(
            np.maximum(
                0.0,
                result.orientation_slack_requested - slack_bound,
            )
        ),
        "orientation_slack_bound_exhausted_frame_count": int(
            np.count_nonzero(
                np.any(
                    result.orientation_slack_requested
                    > slack_bound + 1e-12,
                    axis=1,
                )
            )
        ),
        "position_error_mean_m": float(
            0.5
            * (
                np.mean(position_error["left"])
                + np.mean(position_error["right"])
            )
        ),
        "orientation_error_mean_rad": float(
            0.5
            * (
                np.mean(orientation_error["left"])
                + np.mean(orientation_error["right"])
            )
        ),
        "left_position_error_m": stats(position_error["left"]),
        "right_position_error_m": stats(position_error["right"]),
        "left_orientation_error_rad": stats(orientation_error["left"]),
        "right_orientation_error_rad": stats(orientation_error["right"]),
        "collision": {
            "prohibited_collision_frames": collision_count,
            "arm_collision_frames": int(
                np.count_nonzero(geometry["arm_only_flag"])
            ),
            "hand_prohibited_collision_frames": int(
                np.count_nonzero(geometry["hand_prohibited_flag"])
            ),
            "hand_comprehensive_collision_frames": int(
                np.count_nonzero(geometry["hand_comprehensive_flag"])
            ),
            "cross_arm_collision_frames": int(
                np.count_nonzero(geometry["cross_arm_flag"])
            ),
            "torso_collision_frames": int(
                np.count_nonzero(geometry["torso_flag"])
            ),
            "third_finger_collision_frames": int(
                np.count_nonzero(geometry["third_finger_flag"])
            ),
            "thumb_index_collision_frames": int(
                np.count_nonzero(geometry["thumb_index_flag"])
            ),
            "same_hand_internal_contact_frames": int(
                np.count_nonzero(geometry["same_hand_internal_flag"])
            ),
            "category_frame_incidence": geometry[
                "category_frame_incidence"
            ],
            "top_pairs": geometry["top_pairs"],
        },
        "temporal": {
            "branch_discontinuity_count": int(np.count_nonzero(branches)),
            "maximum_joint_step_rad": float(np.max(step, initial=0.0)),
            "maximum_velocity_rad_s": float(
                np.max(velocity, initial=0.0)
            ),
            "maximum_acceleration_rad_s2": float(
                np.max(acceleration, initial=0.0)
            ),
            "hand_projection": result.metadata[
                "hand_temporal_projection"
            ],
        },
        "task_space": {
            "physical_pinch_error_mean_m": float(
                0.5
                * (
                    np.mean(pinch_error["left"])
                    + np.mean(pinch_error["right"])
                )
            ),
            "task_critical_pinch_error_mean_m": float(
                0.5
                * (
                    np.mean(pinch_error["left"][task_mask["left"]])
                    + np.mean(pinch_error["right"][task_mask["right"]])
                )
            ),
            "left_pinch_error_m": stats(pinch_error["left"]),
            "right_pinch_error_m": stats(pinch_error["right"]),
            "left_pinch_orientation_error_rad": stats(
                pinch_orientation_error["left"]
            ),
            "right_pinch_orientation_error_rad": stats(
                pinch_orientation_error["right"]
            ),
            "source_semantic_mask_is_diagnostic_only_for_a": True,
        },
        "bimanual": {
            "midpoint_error_mean_m": float(
                np.mean(
                    np.linalg.norm(actual_midpoint - target_midpoint, axis=1)
                )
            ),
            "relative_vector_error_mean_m": float(
                np.mean(
                    np.linalg.norm(actual_relative - target_relative, axis=1)
                )
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
        "v2_to_v3": {
            "q_deviation_norm_rad": stats(
                np.linalg.norm(q_delta, axis=1)
            ),
            "q_deviation_max_abs_joint_rad": float(
                np.max(np.abs(q_delta), initial=0.0)
            ),
            "changed_arm_frame_count": int(
                np.count_nonzero(np.linalg.norm(q_delta, axis=1) > 1e-8)
            ),
            "wrist_position_delta_m": stats(wrist_position_delta),
            "wrist_orientation_delta_rad": stats(
                wrist_orientation_delta
            ),
            "v2_strict_task_success_frames": int(
                np.count_nonzero(v2_task_success)
            ),
            "v2_strict_task_success_frames_exactly_unchanged": int(
                np.count_nonzero(v2_task_success & exact_arm)
            ),
        },
        "solver": {
            key: value
            for key, value in result.metadata.items()
            if key != "repair_records"
        },
        "strict_checks": checks,
        "semantic": {
            "left_phase_complete": len(episode.left_phase) == len(result.q),
            "right_phase_complete": len(episode.right_phase) == len(result.q),
            "left_transition_count": int(
                np.count_nonzero(episode.left_phase[1:] != episode.left_phase[:-1])
            ),
            "right_transition_count": int(
                np.count_nonzero(episode.right_phase[1:] != episode.right_phase[:-1])
            ),
        },
    }
    validation_output = {
        "schema_version": "common_feasibility_v3_validation",
        "status": status,
        "pass": status == "PASS",
        "checks": checks,
        "first_causal_failure": first_failure,
        "thresholds_unchanged_from_v2": {
            "position_tolerance_m": position_tolerance,
            "orientation_tolerance_rad": orientation_tolerance,
            "required_ik_success_rate": solver_config[
                "required_ik_success_rate"
            ],
            "maximum_joint_step_rad": validation[
                "maximum_joint_step_rad"
            ],
            "maximum_velocity_rad_s": validation[
                "maximum_velocity_rad_s"
            ],
            "maximum_acceleration_rad_s2": validation[
                "maximum_acceleration_rad_s2"
            ],
            "collision_penetration_tolerance_m": validation[
                "collision_penetration_tolerance_m"
            ],
        },
        "bounded_orientation_slack_rad": slack_bound,
        "source_feasibility_classification": (
            None
            if status == "PASS"
            else first_failure["classification"]
        ),
        "physics": "NOT_PERFORMED",
        "training": "NOT_PERFORMED",
        "real_robot_execution": "NOT_PERFORMED",
    }
    return {
        "metrics": metrics,
        "validation": validation_output,
        "geometry": geometry,
    }


def export_episode(
    output_root: Path,
    result: FeasibilityResult,
    evaluated: Mapping[str, Any],
    g1: G1Kinematics,
    runtime: Any,
    frozen_solver_sha256: str,
    dependency_checksums: Mapping[str, Any],
) -> Path:
    episode = result.episode
    directory = (
        output_root
        / episode.dataset_name
        / f"episode_{episode.episode_id:06d}"
    )
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
        target_left_wrist_position=episode.targets[
            "left_wrist_position"
        ].astype(np.float64),
        target_right_wrist_position=episode.targets[
            "right_wrist_position"
        ].astype(np.float64),
        target_left_wrist_rotation=episode.targets[
            "left_wrist_rotation"
        ].astype(np.float64),
        target_right_wrist_rotation=episode.targets[
            "right_wrist_rotation"
        ].astype(np.float64),
        target_left_task_tool_position=episode.targets[
            "left_tool_position"
        ].astype(np.float64),
        target_right_task_tool_position=episode.targets[
            "right_tool_position"
        ].astype(np.float64),
        joint_names=np.asarray(g1.info["joint_names"]).astype("U64"),
        method=np.asarray(episode.method),
        representation=np.asarray(episode.representation),
        frozen_solver_sha256=np.asarray(frozen_solver_sha256),
    )
    atomic_npz(
        directory / "g1_hand_action.npz",
        action=hand,
        left_action=left,
        right_action=right,
        left_phase=episode.left_phase.astype("U16"),
        right_phase=episode.right_phase.astype("U16"),
        left_joint_names=np.asarray(runtime.hand_joint_names["left"]).astype(
            "U64"
        ),
        right_joint_names=np.asarray(runtime.hand_joint_names["right"]).astype(
            "U64"
        ),
        mapper=np.asarray(
            "unchanged_binary_open_close_with_shared_temporal_feasibility"
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
        directory / "solver_diagnostics.npz",
        orientation_slack_rad=result.orientation_slack.astype(np.float32),
        orientation_slack_requested_rad=(
            result.orientation_slack_requested.astype(np.float32)
        ),
        orientation_slack_excess_rad=np.maximum(
            0.0,
            result.orientation_slack_requested
            - float(
                result.metadata["solver_parameters"][
                    "orientation_slack_bound_rad"
                ]
            ),
        ).astype(np.float32),
        q_v2=result.episode.q_v2.astype(np.float32),
        q_deviation=(result.q - result.episode.q_v2).astype(np.float32),
    )
    source_metadata = load_json(
        V2_INTEGRATED_ROOT
        / episode.dataset_name
        / f"episode_{episode.episode_id:06d}"
        / "source_metadata.json"
    )
    source_metadata.update(
        {
            "feasibility_v3_source": "frozen integrated-v2 representation and targets",
            "source_assets_duplicated": False,
            "source_action_sha256_verified": True,
        }
    )
    atomic_json(directory / "source_metadata.json", source_metadata)
    atomic_json(directory / "retargeting_metrics.json", evaluated["metrics"])
    atomic_json(directory / "validation.json", evaluated["validation"])
    atomic_json(
        directory / "solver_report.json",
        {
            "solver_class": result.metadata["solver_class"],
            "solver_parameters": result.metadata["solver_parameters"],
            "hand_temporal_projection": result.metadata[
                "hand_temporal_projection"
            ],
            "repair_records": result.metadata["repair_records"],
        },
    )
    files = (
        "source_metadata.json",
        "g1_arm_action.npz",
        "g1_hand_action.npz",
        "g1_full_action.npz",
        "solver_diagnostics.npz",
        "retargeting_metrics.json",
        "validation.json",
        "solver_report.json",
    )
    manifest = {
        "schema_version": "common_feasibility_v3_episode",
        "dataset": episode.dataset_name,
        "method": episode.method,
        "episode_id": episode.episode_id,
        "status": evaluated["metrics"]["status"],
        "frozen_solver_sha256": frozen_solver_sha256,
        "frozen_common_arm_v2_sha256": dependency_checksums["sources"][
            "common_arm_v2"
        ]["sha256"],
        "frozen_hand_v2_1_sha256": (
            None
            if episode.method == "baseline"
            else dependency_checksums["sources"]["proposed_hand_v2_1"][
                "sha256"
            ]
        ),
        "frozen_v2_arm_action_array_sha256": array_sha256(
            episode.q_v2.astype(np.float32)
        ),
        "a_b_output_separation": True,
        "files": {name: sha256_file(directory / name) for name in files},
        "offline_only": True,
        "training_executed": False,
        "physics_executed": False,
        "real_robot_commands": False,
    }
    atomic_json(directory / "manifest.json", manifest)
    return directory
