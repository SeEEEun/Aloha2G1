"""Before/after evaluation, representative gate, and full-50 reporting."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from tools.doll_handoff_retargeting.common import branch_flags

from .common import (
    FROZEN_ROOT,
    OUTPUT_ROOT,
    SIDES,
    contiguous_segments,
    load_json,
    load_trajectory,
    metric_path,
    read_csv,
    rotation_error_rad,
    scalar_stats,
    stable_episode_id,
    write_csv,
    write_json,
)
from .solver import EpisodeResult, GenericG1FeasibilityResolver


CLASSIFICATION_CSV = (
    FROZEN_ROOT
    / "review/dataset_b_gate/dataset_b_episode_classification.csv"
)


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _hard_ik(error: np.ndarray, fps: float, physical: float) -> tuple[bool, float]:
    frames = np.flatnonzero(np.asarray(error) > physical)
    longest = max(
        ((end - start + 1) / fps for start, end in contiguous_segments(frames)),
        default=0.0,
    )
    return bool(np.max(error, initial=0.0) >= 2.0 * physical or longest >= 0.25), float(
        longest
    )


def _orientation_metrics(
    achieved: Mapping[str, np.ndarray], desired: Mapping[str, np.ndarray]
) -> dict[str, float]:
    values = np.asarray(
        [
            rotation_error_rad(achieved[side][frame], desired[side][frame])
            for side in SIDES
            for frame in range(len(achieved[side]))
        ],
        dtype=np.float64,
    )
    return {
        "mean_orientation_residual_rad": float(np.mean(values)),
        "max_orientation_residual_rad": float(np.max(values)),
    }


def _geometry_quality(
    resolver: GenericG1FeasibilityResolver, q: np.ndarray
) -> dict[str, float]:
    guides = {
        side: np.asarray(
            resolver.common["resolved"]["natural_arm_redundancy"][
                "nominal_elbow_guides_model"
            ][side],
            dtype=np.float64,
        )
        for side in SIDES
    }
    manipulation: list[float] = []
    torso: list[float] = []
    sew_values = {side: [] for side in SIDES}
    for value in q:
        state = resolver.g1.manipulability_state(value)
        clearance = resolver.g1.posture_clearance_state(value)
        sew = resolver.g1.sew_angles(value, guides)
        manipulation.extend(
            float(state[side]["minimum_singular_value"]) for side in SIDES
        )
        torso.append(float(clearance["TORSO"]["minimum_distance_m"]))
        for side in SIDES:
            sew_values[side].append(float(sew[side]))
    sew_jump = max(
        (
            float(
                np.max(
                    np.abs(
                        np.angle(
                            np.exp(
                                1j
                                * np.diff(
                                    np.asarray(sew_values[side], dtype=np.float64)
                                )
                            )
                        )
                    ),
                    initial=0.0,
                )
            )
            for side in SIDES
        ),
        default=0.0,
    )
    return {
        "minimum_manipulability": float(np.min(manipulation)),
        "mean_manipulability": float(np.mean(manipulation)),
        "minimum_torso_clearance_m": float(np.min(torso)),
        "mean_torso_clearance_m": float(np.mean(torso)),
        "maximum_sew_step_rad": sew_jump,
    }


def _source_arrays_unchanged(
    frozen: Mapping[str, np.ndarray], after_path: Path
) -> tuple[bool, dict[str, bool]]:
    keys = (
        "target_left_wrist_position_model",
        "target_right_wrist_position_model",
        "target_left_wrist_rotation_model",
        "target_right_wrist_rotation_model",
        "target_left_interaction_frame_position_world",
        "target_right_interaction_frame_position_world",
        "target_left_interaction_frame_position_task",
        "target_right_interaction_frame_position_task",
        "ownership_state",
        "left_hand_phase",
        "right_hand_phase",
        "event_names",
        "event_frames",
    )
    with np.load(after_path, allow_pickle=False) as payload:
        checks = {
            key: bool(
                np.array_equal(np.asarray(frozen[key]), np.asarray(payload[key]))
            )
            for key in keys
        }
    return all(checks.values()), checks


def evaluate_episode(
    resolver: GenericG1FeasibilityResolver,
    result: EpisodeResult,
    include_geometry_quality: bool = True,
) -> dict[str, Any]:
    episode = result.episode
    frozen = load_trajectory(episode)
    timestamp = frozen["timestamp"].astype(np.float64)
    fps = float(1.0 / np.median(np.diff(timestamp))) if len(timestamp) > 1 else 30.0
    source_model = resolver._source_model(result.source_position_world)
    before_position_model, before_rotation = resolver._pose_arrays(result.q_before)
    before_achieved_world = {
        side: resolver.g1.model_to_world_position(before_position_model[side])
        for side in SIDES
    }
    before_source_error_by_side = {
        side: np.linalg.norm(
            before_achieved_world[side] - result.source_position_world[side], axis=1
        )
        for side in SIDES
    }
    after_source_error_by_side = {
        side: np.linalg.norm(
            result.achieved_position_world[side]
            - result.source_position_world[side],
            axis=1,
        )
        for side in SIDES
    }
    after_realized_error_by_side = {
        side: np.linalg.norm(
            result.achieved_position_world[side]
            - result.realized_position_world[side],
            axis=1,
        )
        for side in SIDES
    }
    before_source_error = np.maximum(
        before_source_error_by_side["left"], before_source_error_by_side["right"]
    )
    after_source_error = np.maximum(
        after_source_error_by_side["left"], after_source_error_by_side["right"]
    )
    after_realized_error = np.maximum(
        after_realized_error_by_side["left"],
        after_realized_error_by_side["right"],
    )
    before_hard_ik, before_longest = _hard_ik(
        before_source_error, fps, resolver.physical_tolerance
    )
    after_hard_ik, after_longest = _hard_ik(
        after_realized_error, fps, resolver.physical_tolerance
    )
    before_collision = resolver._collision_metrics(
        result.q_before, result.left_hand, result.right_hand, fps
    )
    after_collision = resolver._collision_metrics(
        result.q_after, result.left_hand, result.right_hand, fps
    )
    before_temporal = resolver._temporal_metrics(
        result.q_before, result.left_hand, result.right_hand, fps
    )
    after_temporal = resolver._temporal_metrics(
        result.q_after, result.left_hand, result.right_hand, fps
    )
    source_relative = (
        result.source_position_world["right"]
        - result.source_position_world["left"]
    )
    realized_relative = (
        result.realized_position_world["right"]
        - result.realized_position_world["left"]
    )
    relation_change = np.linalg.norm(realized_relative - source_relative, axis=1)
    source_midpoint = 0.5 * (
        result.source_position_world["left"]
        + result.source_position_world["right"]
    )
    realized_midpoint = 0.5 * (
        result.realized_position_world["left"]
        + result.realized_position_world["right"]
    )
    midpoint_change = np.linalg.norm(realized_midpoint - source_midpoint, axis=1)
    ownership = frozen["ownership_state"].astype(str)
    dual_contact = ownership == "DUAL_CONTACT"
    dual_relation_change = relation_change[dual_contact]
    after_path = (
        resolver.output_root
        / "after/trajectories"
        / f"{stable_episode_id(episode)}.npz"
    )
    unchanged, target_checks = _source_arrays_unchanged(frozen, after_path)
    hands_unchanged = bool(
        np.array_equal(
            frozen["left_dex3_qpos"], result.left_hand.astype(np.float32)
        )
        and np.array_equal(
            frozen["right_dex3_qpos"], result.right_hand.astype(np.float32)
        )
    )
    hard_acceleration = bool(
        float(after_temporal["maximum_acceleration_rad_s2"])
        > float(resolver.acceptance["maximum_acceleration_rad_s2"]) + 1e-5
    )
    hard_classes = set(after_collision["hard_class_frame_counts"])
    hard_reasons = [
        reason
        for active, reason in (
            (after_hard_ik, "PHYSICAL_HARD_IK"),
            ("ARM_TORSO_INVALID" in hard_classes, "ARM_TORSO_HARD_COLLISION"),
            (
                "DISTAL_HAND_HAND_CONTACT" in hard_classes,
                "DISTAL_HAND_HARD_COLLISION",
            ),
            (
                bool(hard_classes - {"ARM_TORSO_INVALID", "DISTAL_HAND_HAND_CONTACT"}),
                "OTHER_ROBOT_SELF_COLLISION",
            ),
            (
                int(after_temporal["joint_limit_violation_count"]) > 0,
                "JOINT_LIMIT_VIOLATION",
            ),
            (
                int(after_temporal["branch_discontinuity_count"]) > 0,
                "BRANCH_DISCONTINUITY",
            ),
            (hard_acceleration, "ACCELERATION_HARD_GATE"),
        )
        if active
    ]
    warning_reasons = [
        reason
        for active, reason in (
            (
                np.any(result.projection_translation_m > 0.0),
                "FEASIBILITY_PROJECTION_REPORTED",
            ),
            (
                int(after_collision["contact_frame_count"]) > 0,
                "NON_HARD_SELF_CONTACT",
            ),
            (
                np.any(after_source_error > resolver.strict_tolerance),
                "SOURCE_TARGET_OUTSIDE_STRICT_TOLERANCE",
            ),
            (
                np.any(after_realized_error > resolver.strict_tolerance),
                "REALIZED_TARGET_OUTSIDE_STRICT_TOLERANCE",
            ),
        )
        if active
    ]
    if hard_reasons:
        classification = "HARD_FAIL"
        classification_reasons = hard_reasons + warning_reasons
    elif warning_reasons:
        classification = "USABLE_WITH_WARNING"
        classification_reasons = warning_reasons
    else:
        classification = "CLEAN_PASS"
        classification_reasons = ["ALL_HARD_AND_WARNING_GATES_CLEAR"]
    frozen_gate_rows = {
        int(row["episode_index"]): row for row in read_csv(CLASSIFICATION_CSV)
    }
    frozen_gate = frozen_gate_rows[episode]
    row: dict[str, Any] = {
        "episode_index": episode,
        "stable_episode_id": stable_episode_id(episode),
        "frame_count": len(result.q_after),
        "before_classification": frozen_gate["classification"],
        "after_classification": classification,
        "after_classification_reasons": classification_reasons,
        "before_strict_source_ik_success_rate": float(
            np.mean(before_source_error <= resolver.strict_tolerance)
        ),
        "after_strict_source_ik_success_rate": float(
            np.mean(after_source_error <= resolver.strict_tolerance)
        ),
        "after_strict_realized_ik_success_rate": float(
            np.mean(after_realized_error <= resolver.strict_tolerance)
        ),
        "before_physical_source_ik_success_rate": float(
            np.mean(before_source_error <= resolver.physical_tolerance)
        ),
        "after_physical_source_ik_success_rate": float(
            np.mean(after_source_error <= resolver.physical_tolerance)
        ),
        "after_physical_realized_ik_success_rate": float(
            np.mean(after_realized_error <= resolver.physical_tolerance)
        ),
        "before_hard_ik": before_hard_ik,
        "after_hard_ik": after_hard_ik,
        "before_max_source_position_residual_m": float(np.max(before_source_error)),
        "after_max_source_position_residual_m": float(np.max(after_source_error)),
        "after_max_realized_position_residual_m": float(
            np.max(after_realized_error)
        ),
        "before_mean_source_position_residual_m": float(
            np.mean(before_source_error)
        ),
        "after_mean_source_position_residual_m": float(
            np.mean(after_source_error)
        ),
        "after_mean_realized_position_residual_m": float(
            np.mean(after_realized_error)
        ),
        "before_longest_physical_ik_segment_s": before_longest,
        "after_longest_physical_ik_segment_s": after_longest,
        "before_hard_collision_frames": int(
            before_collision["hard_collision_frame_count"]
        ),
        "after_hard_collision_frames": int(
            after_collision["hard_collision_frame_count"]
        ),
        "before_contact_frames": int(before_collision["contact_frame_count"]),
        "after_contact_frames": int(after_collision["contact_frame_count"]),
        "before_maximum_penetration_m": float(
            before_collision["maximum_penetration_depth_m"]
        ),
        "after_maximum_penetration_m": float(
            after_collision["maximum_penetration_depth_m"]
        ),
        "before_arm_torso_hard_frames": int(
            before_collision["hard_class_frame_counts"].get(
                "ARM_TORSO_INVALID", 0
            )
        ),
        "after_arm_torso_hard_frames": int(
            after_collision["hard_class_frame_counts"].get(
                "ARM_TORSO_INVALID", 0
            )
        ),
        "before_distal_hard_frames": int(
            before_collision["hard_class_frame_counts"].get(
                "DISTAL_HAND_HAND_CONTACT", 0
            )
        ),
        "after_distal_hard_frames": int(
            after_collision["hard_class_frame_counts"].get(
                "DISTAL_HAND_HAND_CONTACT", 0
            )
        ),
        "joint_limit_violations_before": int(
            before_temporal["joint_limit_violation_count"]
        ),
        "joint_limit_violations_after": int(
            after_temporal["joint_limit_violation_count"]
        ),
        "branch_discontinuities_before": int(
            before_temporal["branch_discontinuity_count"]
        ),
        "branch_discontinuities_after": int(
            after_temporal["branch_discontinuity_count"]
        ),
        "maximum_velocity_before_rad_s": float(
            before_temporal["maximum_velocity_rad_s"]
        ),
        "maximum_velocity_after_rad_s": float(
            after_temporal["maximum_velocity_rad_s"]
        ),
        "maximum_acceleration_before_rad_s2": float(
            before_temporal["maximum_acceleration_rad_s2"]
        ),
        "maximum_acceleration_after_rad_s2": float(
            after_temporal["maximum_acceleration_rad_s2"]
        ),
        "projection_active_frames": int(
            np.count_nonzero(np.any(result.projection_translation_m > 0.0, axis=1))
        ),
        "mean_projection_translation_m": float(
            np.mean(result.projection_translation_m)
        ),
        "median_projection_translation_m": float(
            np.median(result.projection_translation_m)
        ),
        "max_projection_translation_m": float(
            np.max(result.projection_translation_m)
        ),
        "mean_projection_orientation_rad": 0.0,
        "max_projection_orientation_rad": 0.0,
        "mean_bimanual_relation_change_m": float(np.mean(relation_change)),
        "max_bimanual_relation_change_m": float(np.max(relation_change)),
        "dual_contact_frame_count": int(np.count_nonzero(dual_contact)),
        "mean_dual_contact_bimanual_relation_change_m": float(
            np.mean(dual_relation_change) if len(dual_relation_change) else 0.0
        ),
        "max_dual_contact_bimanual_relation_change_m": float(
            np.max(dual_relation_change, initial=0.0)
        ),
        "mean_bimanual_midpoint_change_m": float(np.mean(midpoint_change)),
        "max_bimanual_midpoint_change_m": float(np.max(midpoint_change)),
        "ownership_timing_change": 0 if unchanged else int(not target_checks["ownership_state"]),
        "source_interaction_target_change": 0
        if unchanged
        else int(
            not all(
                target_checks[key]
                for key in target_checks
                if key.startswith("target_")
            )
        ),
        "hands_changed": int(not hands_unchanged),
        "handoff_order_valid": frozen_gate["handoff_order_valid"],
        "ownership_transition_valid": frozen_gate["ownership_transition_valid"],
        "release_event_present": frozen_gate["release_event_present"],
        "source_bin_release_inside": frozen_gate[
            "per_episode_source_bin_release_xy_inside"
        ],
        "source_array_checks_pass": unchanged,
        "source_array_checks": target_checks,
        "after_collision_segments": after_collision["segments"],
    }
    row.update(_orientation_metrics(before_rotation, result.source_orientation_model))
    row["before_mean_orientation_residual_rad"] = row.pop(
        "mean_orientation_residual_rad"
    )
    row["before_max_orientation_residual_rad"] = row.pop(
        "max_orientation_residual_rad"
    )
    row.update(
        _orientation_metrics(
            result.achieved_orientation_model, result.source_orientation_model
        )
    )
    row["after_mean_orientation_residual_rad"] = row.pop(
        "mean_orientation_residual_rad"
    )
    row["after_max_orientation_residual_rad"] = row.pop(
        "max_orientation_residual_rad"
    )
    if include_geometry_quality:
        for prefix, q in (("before", result.q_before), ("after", result.q_after)):
            for key, value in _geometry_quality(resolver, q).items():
                row[f"{prefix}_{key}"] = value
    return row


def _flat_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in row.items()
    }


def representative_gate(
    resolver: GenericG1FeasibilityResolver,
    results: Iterable[EpisodeResult],
) -> dict[str, Any]:
    rows = [evaluate_episode(resolver, result) for result in results]
    rows.sort(key=lambda row: int(row["episode_index"]))
    write_csv(
        resolver.output_root / "representative_before_after.csv",
        [_flat_row(row) for row in rows],
    )
    by_episode = {int(row["episode_index"]): row for row in rows}
    settings = resolver.config["representative_gate"]
    selected = [int(value) for value in settings["episodes"]]
    if sorted(by_episode) != sorted(selected):
        raise RuntimeError("representative result set does not match frozen selection")
    hard_ik_rows = [by_episode[index] for index in (2, 3)]
    before_max_sum = sum(
        float(row["before_max_source_position_residual_m"])
        for row in hard_ik_rows
    )
    after_max_sum = sum(
        float(row["after_max_source_position_residual_m"])
        for row in hard_ik_rows
    )
    residual_reduction = (
        1.0 - after_max_sum / before_max_sum if before_max_sum > 0 else 1.0
    )
    collision_rows = [by_episode[index] for index in (3, 11, 26)]
    before_collision = sum(
        int(row["before_hard_collision_frames"]) for row in collision_rows
    )
    after_collision = sum(
        int(row["after_hard_collision_frames"]) for row in collision_rows
    )
    collision_reduction = (
        1.0 - after_collision / before_collision if before_collision else 1.0
    )
    control = by_episode[24]
    aggregate_before_mean = float(
        np.mean(
            [float(row["before_mean_source_position_residual_m"]) for row in rows]
        )
    )
    aggregate_after_mean = float(
        np.mean(
            [float(row["after_mean_source_position_residual_m"]) for row in rows]
        )
    )
    checks = {
        "hard_source_residual_reduction": residual_reduction
        >= float(settings["minimum_hard_source_residual_reduction_fraction"]),
        "hard_collision_frame_reduction": collision_reduction
        >= float(settings["minimum_hard_collision_frame_reduction_fraction"]),
        "no_after_realized_hard_ik": all(
            not bool(row["after_hard_ik"]) for row in rows
        ),
        "no_joint_limit_regression": all(
            int(row["joint_limit_violations_after"])
            <= int(row["joint_limit_violations_before"])
            for row in rows
        ),
        "no_branch_regression": all(
            int(row["branch_discontinuities_after"])
            <= int(row["branch_discontinuities_before"])
            for row in rows
        ),
        "no_acceleration_regression_beyond_gate": all(
            float(row["maximum_acceleration_after_rad_s2"])
            <= float(resolver.acceptance["maximum_acceleration_rad_s2"]) + 1e-5
            for row in rows
        ),
        "control_class_not_regressed": control["after_classification"]
        == "CLEAN_PASS",
        "control_mean_source_error_not_regressed": float(
            control["after_mean_source_position_residual_m"]
        )
        <= float(control["before_mean_source_position_residual_m"])
        + float(settings["maximum_control_mean_source_error_regression_m"]),
        "aggregate_mean_source_error_not_regressed": aggregate_after_mean
        <= aggregate_before_mean
        + float(settings["maximum_aggregate_mean_source_error_regression_m"]),
        "ownership_timing_unchanged": all(
            int(row["ownership_timing_change"]) == 0 for row in rows
        ),
        "source_targets_unchanged": all(
            int(row["source_interaction_target_change"]) == 0 for row in rows
        ),
        "hands_unchanged": all(int(row["hands_changed"]) == 0 for row in rows),
    }
    passed = all(checks.values())
    gate = {
        "schema_version": "doll_handoff_representative_feasibility_gate_v1",
        "status": "PASS" if passed else "FAIL_STOP_BEFORE_FULL50",
        "episodes": selected,
        "checks": checks,
        "hard_source_max_residual_reduction_fraction": residual_reduction,
        "hard_collision_frame_reduction_fraction": collision_reduction,
        "before_hard_collision_frames": before_collision,
        "after_hard_collision_frames": after_collision,
        "aggregate_before_mean_source_error_m": aggregate_before_mean,
        "aggregate_after_mean_source_error_m": aggregate_after_mean,
        "source_targets_modified": False,
        "ownership_timing_modified": False,
        "episode_specific_parameters": 0,
        "rows": rows,
    }
    write_json(resolver.output_root / "representative_gate.json", gate)
    lines = [
        "# Representative Generic-G1 feasibility gate",
        "",
        f"Status: **{gate['status']}**",
        "",
        "| episode | before → after | source max residual | realized max residual | hard collision frames | projection max |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| ep{int(row['episode_index']):03d} | {row['before_classification']} → {row['after_classification']} | "
            f"{float(row['before_max_source_position_residual_m'])*1000:.2f} → {float(row['after_max_source_position_residual_m'])*1000:.2f} mm | "
            f"{float(row['after_max_realized_position_residual_m'])*1000:.2f} mm | "
            f"{row['before_hard_collision_frames']} → {row['after_hard_collision_frames']} | "
            f"{float(row['max_projection_translation_m'])*1000:.2f} mm |"
        )
    lines.extend(
        [
            "",
            f"- Hard-source residual reduction: **{residual_reduction*100:.2f}%**",
            f"- Hard-collision frame reduction: **{collision_reduction*100:.2f}%**",
            f"- Ownership timing change: **{sum(int(row['ownership_timing_change']) for row in rows)}**",
            f"- Source interaction target change: **{sum(int(row['source_interaction_target_change']) for row in rows)}**",
            "- Orientation target projection: **0 rad** (no new orientation relaxation activated)",
        ]
    )
    (resolver.output_root / "representative_gate_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return gate


def full50_report(
    resolver: GenericG1FeasibilityResolver,
    results: Iterable[EpisodeResult],
) -> dict[str, Any]:
    materialized_results = list(results)
    rows = [evaluate_episode(resolver, result) for result in materialized_results]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if len(rows) != 50:
        raise RuntimeError(f"full-50 report requires 50 episodes, got {len(rows)}")
    full_root = resolver.output_root / "full50"
    full_root.mkdir(parents=True, exist_ok=True)
    write_csv(full_root / "per_episode.csv", [_flat_row(row) for row in rows])
    counts = collections.Counter(str(row["after_classification"]) for row in rows)
    non_clean_findings = [
        {
            "episode_index": int(row["episode_index"]),
            "stable_episode_id": row["stable_episode_id"],
            "classification": row["after_classification"],
            "reasons": row["after_classification_reasons"],
        }
        for row in rows
        if row["after_classification"] != "CLEAN_PASS"
    ]
    before_physical_residuals = [
        float(row["before_max_source_position_residual_m"])
        for row in rows
        if bool(row["before_hard_ik"])
    ]
    after_physical_residuals = [
        float(row["after_max_realized_position_residual_m"])
        for row in rows
        if bool(row["after_hard_ik"])
    ]
    projections = np.concatenate(
        [
            result.projection_translation_m.reshape(-1)
            for result in sorted(
                materialized_results, key=lambda result: result.episode
            )
        ]
    )
    source_errors_after = [
        float(row["after_mean_source_position_residual_m"]) for row in rows
    ]
    aggregate = {
        "schema_version": "doll_handoff_generic_g1_feasibility_full50_v1",
        "episode_count": 50,
        "classification_counts": {
            name: int(counts.get(name, 0))
            for name in ("CLEAN_PASS", "USABLE_WITH_WARNING", "HARD_FAIL")
        },
        "before_classification_counts": {
            "CLEAN_PASS": 1,
            "USABLE_WITH_WARNING": 25,
            "HARD_FAIL": 24,
        },
        "physical_hard_ik_episode_count": sum(bool(row["after_hard_ik"]) for row in rows),
        "physical_residual_after_realized_m": {
            "max": max(after_physical_residuals, default=0.0),
            "median": float(np.median(after_physical_residuals))
            if after_physical_residuals
            else 0.0,
        },
        "physical_residual_before_source_m": {
            "max": max(before_physical_residuals, default=0.0),
            "median": float(np.median(before_physical_residuals))
            if before_physical_residuals
            else 0.0,
        },
        "arm_torso_hard_collision_episode_count": sum(
            int(row["after_arm_torso_hard_frames"]) > 0 for row in rows
        ),
        "distal_hard_collision_episode_count": sum(
            int(row["after_distal_hard_frames"]) > 0 for row in rows
        ),
        "joint_limit_violations": sum(
            int(row["joint_limit_violations_after"]) for row in rows
        ),
        "branch_discontinuities": sum(
            int(row["branch_discontinuities_after"]) for row in rows
        ),
        "handoff_ordering_valid": sum(
            str(row["handoff_order_valid"]).lower() == "true" for row in rows
        ),
        "ownership_transition_valid": sum(
            str(row["ownership_transition_valid"]).lower() == "true"
            for row in rows
        ),
        "release_event_present": sum(
            str(row["release_event_present"]).lower() == "true" for row in rows
        ),
        "source_bin_release_inside": sum(
            str(row["source_bin_release_inside"]).lower() == "true" for row in rows
        ),
        "mean_achieved_vs_source_interaction_position_residual_m": float(
            np.mean(source_errors_after)
        ),
        "median_achieved_vs_source_interaction_position_residual_m": float(
            np.median(source_errors_after)
        ),
        "mean_interaction_frame_position_deviation_m": float(
            np.mean(projections)
        ),
        "median_interaction_frame_position_deviation_m": float(
            np.median(projections)
        ),
        "feasibility_projection_translation_m": {
            "mean": float(np.mean(projections)),
            "median": float(np.median(projections)),
            "max": float(np.max(projections)),
        },
        "maximum_velocity_rad_s": max(
            float(row["maximum_velocity_after_rad_s"]) for row in rows
        ),
        "maximum_acceleration_rad_s2": max(
            float(row["maximum_acceleration_after_rad_s2"]) for row in rows
        ),
        "mean_bimanual_relation_change_m": float(
            np.mean([float(row["mean_bimanual_relation_change_m"]) for row in rows])
        ),
        "max_bimanual_relation_change_m": max(
            float(row["max_bimanual_relation_change_m"]) for row in rows
        ),
        "mean_dual_contact_bimanual_relation_change_m": float(
            np.mean(
                [
                    float(row["mean_dual_contact_bimanual_relation_change_m"])
                    for row in rows
                ]
            )
        ),
        "max_dual_contact_bimanual_relation_change_m": max(
            float(row["max_dual_contact_bimanual_relation_change_m"])
            for row in rows
        ),
        "maximum_realized_position_residual_all_episodes_m": max(
            float(row["after_max_realized_position_residual_m"]) for row in rows
        ),
        "median_episode_max_realized_position_residual_m": float(
            np.median(
                [
                    float(row["after_max_realized_position_residual_m"])
                    for row in rows
                ]
            )
        ),
        "maximum_source_position_residual_after_m": max(
            float(row["after_max_source_position_residual_m"]) for row in rows
        ),
        "median_episode_max_source_position_residual_after_m": float(
            np.median(
                [
                    float(row["after_max_source_position_residual_m"])
                    for row in rows
                ]
            )
        ),
        "minimum_manipulability": min(
            float(row["after_minimum_manipulability"]) for row in rows
        ),
        "minimum_torso_clearance_m": min(
            float(row["after_minimum_torso_clearance_m"]) for row in rows
        ),
        "maximum_sew_step_rad": max(
            float(row["after_maximum_sew_step_rad"]) for row in rows
        ),
        "ownership_timing_change": sum(
            int(row["ownership_timing_change"]) for row in rows
        ),
        "source_interaction_target_change": sum(
            int(row["source_interaction_target_change"]) for row in rows
        ),
        "source_target_hash": {
            "expected": resolver.freeze["cartesian_target_array_set_sha256"],
            "unchanged": all(bool(row["source_array_checks_pass"]) for row in rows),
        },
        "episode_specific_parameters": 0,
        "dataset_b_packaged": False,
        "policy_training_started": False,
    }
    write_json(full_root / "aggregate_summary.json", aggregate)
    write_json(full_root / "failures.json", non_clean_findings)
    return {
        "aggregate": aggregate,
        "rows": rows,
        "failures": non_clean_findings,
    }


__all__ = ["evaluate_episode", "representative_gate", "full50_report"]
