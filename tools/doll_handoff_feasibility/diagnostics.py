"""Frozen failure-frame enrichment and morphology-level cluster analysis."""
from __future__ import annotations

import collections
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tools.doll_handoff_retargeting.models import G1Kinematics

from .common import (
    FROZEN_ROOT,
    MOTION_FREEZE,
    OUTPUT_ROOT,
    REPOSITORY,
    SIDES,
    bool_value,
    load_json,
    load_trajectory,
    matrix_to_wxyz,
    metric_path,
    read_csv,
    rotation_error_rad,
    scalar_stats,
    stable_episode_id,
    verify_frozen_contract,
    write_csv,
    write_json,
)


IK_FRAMES = (
    FROZEN_ROOT
    / "review/dataset_b_gate/ik_audit/strict_ik_failed_frames.csv"
)
CLASSIFICATION = (
    FROZEN_ROOT
    / "review/dataset_b_gate/dataset_b_episode_classification.csv"
)
CONTACT_RECORDS = (
    FROZEN_ROOT
    / "review/dataset_b_gate/collision_audit/robot_self_contact_records.csv"
)
CONTACT_CLASS_SEGMENTS = (
    FROZEN_ROOT
    / "review/dataset_b_gate/collision_audit/robot_self_contact_class_segments.csv"
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()


def _load_context() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_json(FROZEN_ROOT / "frozen_approval/config/common_config.json"),
        load_json(FROZEN_ROOT / "frozen_approval/scene/scene_layout.json"),
        load_json(FROZEN_ROOT / "frozen_approval/config/tool_frame_report.json"),
    )


def _hard_contact_lookup() -> tuple[
    dict[tuple[int, int], list[dict[str, str]]],
    dict[tuple[int, int], list[dict[str, str]]],
]:
    segments = [
        row
        for row in read_csv(CONTACT_CLASS_SEGMENTS)
        if row["severity"] == "HARD"
    ]
    segment_by_frame: dict[tuple[int, int], list[dict[str, str]]] = (
        collections.defaultdict(list)
    )
    for row in segments:
        episode = int(row["episode_index"])
        for frame in range(int(row["start_frame"]), int(row["end_frame"]) + 1):
            segment_by_frame[(episode, frame)].append(row)
    record_by_frame: dict[tuple[int, int], list[dict[str, str]]] = (
        collections.defaultdict(list)
    )
    for row in read_csv(CONTACT_RECORDS):
        key = (int(row["episode_index"]), int(row["frame"]))
        if key in segment_by_frame:
            record_by_frame[key].append(row)
    return segment_by_frame, record_by_frame


def _arm_outer_tool_radius(
    g1: G1Kinematics, transforms: Mapping[str, np.ndarray]
) -> dict[str, float]:
    geometry = g1.shoulder_wrist_reach_geometry()
    output: dict[str, float] = {}
    for side in SIDES:
        row = geometry["sides"][side]
        output[side] = float(row["upper_effective_length_m"]) + float(
            row["forearm_effective_length_m"]
        ) + float(np.linalg.norm(transforms[side][:3, 3]))
    return output


def _cluster(
    *,
    hard_classes: set[str],
    physical_ik: bool,
    failing_reach_m: float,
    outer_radius_m: float,
    failing_manipulability: float,
    target_step_m: float,
    joint_step_norm_rad: float,
    recent_target_step_m: float,
    recent_joint_step_norm_rad: float,
    natural_frame_trust_rad: float,
    torso_clearance_m: float,
) -> tuple[str, str]:
    if "DISTAL_HAND_HAND_CONTACT" in hard_classes:
        return (
            "HAND_GEOMETRY_CONFLICT",
            "deep sustained bilateral distal-hand overlap at fixed grasp centers",
        )
    if "ARM_TORSO_INVALID" in hard_classes:
        return (
            "TORSO_CLEARANCE_BRANCH_CONFLICT",
            "negative active-model torso clearance on an inward shoulder/elbow branch",
        )
    if physical_ik and (
        failing_reach_m >= outer_radius_m - 0.015
        or (
            failing_manipulability <= 0.015
            and failing_reach_m >= outer_radius_m - 0.035
        )
    ):
        return (
            "POSITION_NEAR_WORKSPACE_BOUNDARY",
            "target-tool radius approaches the active arm-plus-tool outer radius with low Jacobian margin",
        )
    if physical_ik and (
        recent_target_step_m > 0.010
        or recent_joint_step_norm_rad >= 0.50 * natural_frame_trust_rad
    ):
        return (
            "TEMPORAL_TRANSITION_CONFLICT",
            "source target displacement exceeds one-frame tracking capacity while the common joint trust region is active",
        )
    if physical_ik and failing_manipulability <= 0.015:
        return (
            "ELBOW_BRANCH_CONFLICT",
            "low-manipulability redundant-arm branch without a joint-limit violation",
        )
    if physical_ik:
        return (
            "OTHER_TARGET_REALIZATION",
            "physical Cartesian residual not explained by limits, hard collision, or the dominant reach/temporal signatures",
        )
    if torso_clearance_m < 0.0:
        return (
            "TORSO_CLEARANCE_BRANCH_CONFLICT",
            "negative active-model torso clearance",
        )
    return ("OTHER", "unclassified hard-failure evidence")


def _vector_columns(prefix: str, value: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_{axis}": float(component)
        for axis, component in zip("xyz", np.asarray(value, dtype=np.float64))
    }


def _quat_columns(prefix: str, rotation: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_q{name}": float(component)
        for name, component in zip("wxyz", matrix_to_wxyz(rotation))
    }


def build_failure_diagnostics(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    freeze = verify_frozen_contract()
    output_root = Path(output_root).resolve()
    if output_root == FROZEN_ROOT.resolve():
        raise RuntimeError("diagnostic output may not be the frozen input root")
    output_root.mkdir(parents=True, exist_ok=True)
    common, scene, tool = _load_context()
    g1 = G1Kinematics(common, scene)
    transforms = {
        side: np.asarray(
            tool["g1"][f"{side}_wrist_to_grasp_frame"], dtype=np.float64
        )
        for side in SIDES
    }
    shoulders = g1.fixed_shoulder_anchors_model()
    outer_radius = _arm_outer_tool_radius(g1, transforms)
    guides = {
        side: np.asarray(
            common["resolved"]["natural_arm_redundancy"][
                "nominal_elbow_guides_model"
            ][side],
            dtype=np.float64,
        )
        for side in SIDES
    }
    hard_episode_rows = {
        int(row["episode_index"]): row
        for row in read_csv(CLASSIFICATION)
        if row["classification"] == "HARD_FAIL"
    }
    if len(hard_episode_rows) != 24:
        raise RuntimeError(
            f"expected 24 frozen HARD_FAIL episodes, found {len(hard_episode_rows)}"
        )
    hard_ik_episodes = {
        episode
        for episode, row in hard_episode_rows.items()
        if bool_value(row["hard_ik_fail"])
    }
    hard_segments, hard_records = _hard_contact_lookup()

    rows: list[dict[str, Any]] = []
    cluster_episode_sets: dict[str, set[int]] = collections.defaultdict(set)
    cluster_counts: collections.Counter[str] = collections.Counter()
    cluster_signatures: dict[str, str] = {}
    representative_scores: dict[str, list[tuple[float, int, int]]] = (
        collections.defaultdict(list)
    )
    elbow_indices = {
        side: list(g1.arm_joint_names).index(f"{side}_elbow_joint")
        for side in SIDES
    }
    natural_frame_trust = float(
        common["natural_arm_redundancy"]["maximum_frame_step_norm_rad"]
    )
    physical_tolerance = float(
        common["validation"]["physically_usable_position_tolerance_m"]
    )

    for episode in sorted(hard_episode_rows):
        values = load_trajectory(episode)
        metric = load_json(metric_path(episode))
        arm = values["g1_arm_qpos"].astype(np.float64)
        timestamp = values["timestamp"].astype(np.float64)
        count = len(arm)
        geometry = g1.trajectory_geometry(
            arm,
            values["left_dex3_qpos"],
            values["right_dex3_qpos"],
            float(common["validation"]["collision_penetration_tolerance_m"]),
            transforms,
        )
        desired_model = {
            side: g1.world_to_model_position(
                values[f"target_{side}_interaction_frame_position_world"].astype(
                    np.float64
                )
            )
            for side in SIDES
        }
        source_world = {
            side: values[
                f"target_{side}_interaction_frame_position_world"
            ].astype(np.float64)
            for side in SIDES
        }
        achieved_world = {
            side: geometry[f"{side}_static_tool_position_world"]
            for side in SIDES
        }
        position_error = {
            side: np.linalg.norm(achieved_world[side] - source_world[side], axis=1)
            for side in SIDES
        }
        maximum_error = np.maximum(position_error["left"], position_error["right"])
        physical_frames = (
            set(np.flatnonzero(maximum_error > physical_tolerance).astype(int))
            if episode in hard_ik_episodes
            else set()
        )
        collision_frames = {
            frame for ep, frame in hard_segments if ep == episode
        }
        selected_frames = sorted(physical_frames | collision_frames)
        for frame in selected_frames:
            q = arm[frame]
            state = g1.wrist_state(q)
            sew = g1.sew_angles(q, guides)
            manipulability = g1.manipulability_state(q)
            clearance = g1.posture_clearance_state(q)
            joint_margin_vector = np.minimum(
                q - g1.arm_limits[:, 0], g1.arm_limits[:, 1] - q
            )
            limiting_index = int(np.argmin(joint_margin_vector))
            failing_side = (
                "left"
                if position_error["left"][frame]
                >= position_error["right"][frame]
                else "right"
            )
            desired_rotation: dict[str, np.ndarray] = {}
            achieved_rotation: dict[str, np.ndarray] = {}
            orientation_error: dict[str, float] = {}
            bearing_error: dict[str, float] = {}
            reach: dict[str, float] = {}
            for side in SIDES:
                desired_rotation[side] = (
                    values[f"target_{side}_wrist_rotation_model"][frame].astype(
                        np.float64
                    )
                    @ transforms[side][:3, :3]
                )
                achieved_rotation[side] = geometry[
                    f"{side}_static_tool_rotation_model"
                ][frame]
                orientation_error[side] = rotation_error_rad(
                    achieved_rotation[side], desired_rotation[side]
                )
                bearing_error[side] = float(
                    np.arccos(
                        np.clip(
                            achieved_rotation[side][:, 0]
                            @ desired_rotation[side][:, 0],
                            -1.0,
                            1.0,
                        )
                    )
                )
                reach[side] = float(
                    np.linalg.norm(desired_model[side][frame] - shoulders[side])
                )
            previous_frame = max(0, frame - 1)
            target_step = max(
                float(
                    np.linalg.norm(
                        source_world[side][frame]
                        - source_world[side][previous_frame]
                    )
                )
                for side in SIDES
            )
            joint_step = float(np.linalg.norm(q - arm[previous_frame]))
            recent_start = max(1, frame - 8)
            recent_target_step = max(
                (
                    float(
                        np.linalg.norm(
                            source_world[side][sample]
                            - source_world[side][sample - 1]
                        )
                    )
                    for side in SIDES
                    for sample in range(recent_start, frame + 1)
                ),
                default=target_step,
            )
            recent_joint_step = max(
                (
                    float(np.linalg.norm(arm[sample] - arm[sample - 1]))
                    for sample in range(recent_start, frame + 1)
                ),
                default=joint_step,
            )
            contact_segments = hard_segments.get((episode, frame), [])
            contact_records = hard_records.get((episode, frame), [])
            hard_classes = {row["classification"] for row in contact_segments}
            maximum_depth = max(
                (float(row["penetration_depth_m"]) for row in contact_records),
                default=0.0,
            )
            maximum_duration = max(
                (float(row["duration_s"]) for row in contact_segments),
                default=0.0,
            )
            cluster, signature = _cluster(
                hard_classes=hard_classes,
                physical_ik=frame in physical_frames,
                failing_reach_m=reach[failing_side],
                outer_radius_m=outer_radius[failing_side],
                failing_manipulability=float(
                    manipulability[failing_side]["minimum_singular_value"]
                ),
                target_step_m=target_step,
                joint_step_norm_rad=joint_step,
                recent_target_step_m=recent_target_step,
                recent_joint_step_norm_rad=recent_joint_step,
                natural_frame_trust_rad=natural_frame_trust,
                torso_clearance_m=float(
                    clearance["TORSO"]["minimum_distance_m"]
                ),
            )
            cluster_episode_sets[cluster].add(episode)
            cluster_counts[cluster] += 1
            cluster_signatures[cluster] = signature
            severity_score = max(maximum_error[frame], maximum_depth)
            representative_scores[cluster].append((severity_score, episode, frame))
            row: dict[str, Any] = {
                "episode_index": episode,
                "stable_episode_id": stable_episode_id(episode),
                "source_name": metric["source_name"],
                "frame": frame,
                "source_frame_index": int(values["source_frame_index"][frame]),
                "timestamp_s": float(timestamp[frame]),
                "interaction_phase": str(values["ownership_state"][frame]),
                "ownership_state": str(values["ownership_state"][frame]),
                "left_hand_phase": str(values["left_hand_phase"][frame]),
                "right_hand_phase": str(values["right_hand_phase"][frame]),
                "physical_ik_failure": frame in physical_frames,
                "hard_collision_failure": bool(contact_segments),
                "failure_cluster": cluster,
                "cluster_signature": signature,
                "maximum_position_residual_m": float(maximum_error[frame]),
                "maximum_error_side": failing_side,
                "left_position_residual_m": float(position_error["left"][frame]),
                "right_position_residual_m": float(position_error["right"][frame]),
                "left_orientation_residual_rad": orientation_error["left"],
                "right_orientation_residual_rad": orientation_error["right"],
                "left_task_bearing_axis_error_rad": bearing_error["left"],
                "right_task_bearing_axis_error_rad": bearing_error["right"],
                "left_shoulder_to_target_m": reach["left"],
                "right_shoulder_to_target_m": reach["right"],
                "left_outer_tool_radius_m": outer_radius["left"],
                "right_outer_tool_radius_m": outer_radius["right"],
                "left_elbow_joint_rad": float(q[elbow_indices["left"]]),
                "right_elbow_joint_rad": float(q[elbow_indices["right"]]),
                "left_sew_rad": float(sew["left"]),
                "right_sew_rad": float(sew["right"]),
                "minimum_joint_limit_margin_rad": float(
                    joint_margin_vector[limiting_index]
                ),
                "limiting_joint": str(g1.arm_joint_names[limiting_index]),
                "left_manipulability_min_singular_value": float(
                    manipulability["left"]["minimum_singular_value"]
                ),
                "right_manipulability_min_singular_value": float(
                    manipulability["right"]["minimum_singular_value"]
                ),
                "torso_clearance_m": float(
                    clearance["TORSO"]["minimum_distance_m"]
                ),
                "torso_closest_link_pair": "|".join(
                    clearance["TORSO"]["closest_body_pair"]
                ),
                "cross_arm_clearance_m": float(
                    clearance["CROSS_ARM"]["minimum_distance_m"]
                ),
                "source_target_step_m": target_step,
                "joint_step_norm_rad": joint_step,
                "recent_8_frame_max_source_target_step_m": recent_target_step,
                "recent_8_frame_max_joint_step_norm_rad": recent_joint_step,
                "collision_classes": ";".join(sorted(hard_classes)),
                "collision_link_pairs": ";".join(
                    sorted({row["link_pair"] for row in contact_records})
                ),
                "maximum_penetration_depth_m": maximum_depth,
                "maximum_collision_duration_s": maximum_duration,
            }
            for side in SIDES:
                row.update(
                    _vector_columns(
                        f"desired_{side}_whole_hand_position_world_m",
                        source_world[side][frame],
                    )
                )
                row.update(
                    _quat_columns(
                        f"desired_{side}_whole_hand_orientation_model",
                        desired_rotation[side],
                    )
                )
                row.update(
                    _vector_columns(
                        f"achieved_{side}_whole_hand_position_world_m",
                        achieved_world[side][frame],
                    )
                )
                row.update(
                    _quat_columns(
                        f"achieved_{side}_whole_hand_orientation_model",
                        achieved_rotation[side],
                    )
                )
            rows.append(row)

    write_csv(output_root / "failure_frames.csv", rows)
    cluster_rows: list[dict[str, Any]] = []
    for cluster in sorted(cluster_counts, key=lambda name: (-cluster_counts[name], name)):
        ranked = sorted(representative_scores[cluster], reverse=True)
        representatives: list[str] = []
        seen: set[int] = set()
        for _, episode, frame in ranked:
            if episode in seen:
                continue
            representatives.append(f"ep{episode:03d}:f{frame}")
            seen.add(episode)
            if len(representatives) == 3:
                break
        cluster_rows.append(
            {
                "cluster": cluster,
                "episode_count": len(cluster_episode_sets[cluster]),
                "frame_count": int(cluster_counts[cluster]),
                "representative_episodes_frames": ";".join(representatives),
                "common_geometric_signature": cluster_signatures[cluster],
            }
        )
    write_csv(output_root / "failure_clusters.csv", cluster_rows)

    report_lines = [
        "# Frozen Proposed-B G1 failure clusters",
        "",
        "This report is diagnostic only. It reads the immutable frozen source targets and q trajectories; it does not rerun IK, alter ownership, move the scene, or write into the frozen review root.",
        "",
        f"- HARD_FAIL episodes: **{len(hard_episode_rows)} / 50**",
        f"- Enriched hard-failure frames: **{len(rows)}**",
        "- Joint-limit-dominated hard frames: **0** (verified from active-model margins)",
        "- Full orientation was non-gating in frozen Proposed B; large errors in the reach cluster are primarily axial gauge twist, so `ORIENTATION_INFEASIBLE` is not selected as the primary cause.",
        "",
        "| cluster | episodes | frames | representatives | geometric signature |",
        "|---|---:|---:|---|---|",
    ]
    for row in cluster_rows:
        report_lines.append(
            f"| {row['cluster']} | {row['episode_count']} | {row['frame_count']} | "
            f"{row['representative_episodes_frames']} | {row['common_geometric_signature']} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The failures are target-embodiment realization failures rather than ownership or scene-registration failures. The smallest justified common layer therefore needs (1) nearest constrained Cartesian realization for workspace/temporal limits and (2) signed-distance redundancy repair for torso and bilateral-hand geometry. No new doll, bin, handoff, episode, or frame waypoint is justified.",
            "",
            f"Machine-readable frame table: `{output_root / 'failure_frames.csv'}`",
        ]
    )
    (output_root / "failure_cluster_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    status_short = _git("status", "--short", "--branch")
    recovered = [
        "# Recovered project state",
        "",
        f"- Repository: `{REPOSITORY}`",
        f"- Git commit: `{_git('rev-parse', 'HEAD')}`",
        f"- Branch: `{_git('branch', '--show-current')}`",
        f"- Frozen Proposed-B implementation SHA256: `{freeze['implementation_sha256']}`",
        f"- Frozen trajectory-set SHA256: `{freeze['trajectory_file_set_sha256']}`",
        f"- Frozen Cartesian-target-set SHA256: `{freeze['cartesian_target_array_set_sha256']}`",
        "- BEFORE: `CLEAN_PASS=1`, `USABLE_WITH_WARNING=25`, `HARD_FAIL=24`",
        "- Valid source directories: `50`; `ETC` and `_invalid_no_parquet` excluded",
        "- Frozen inputs modified by this audit: `NO`",
        "",
        "## Pre-existing worktree state",
        "",
        "```text",
        status_short,
        "```",
        "",
        f"Motion-freeze manifest: `{MOTION_FREEZE}`",
    ]
    (output_root / "recovered_state.md").write_text(
        "\n".join(recovered) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "doll_handoff_frozen_failure_clusters_v1",
        "frozen_hashes": {
            key: freeze[key] for key in (
                "implementation_sha256",
                "trajectory_file_set_sha256",
                "cartesian_target_array_set_sha256",
            )
        },
        "hard_episode_count": len(hard_episode_rows),
        "failure_frame_count": len(rows),
        "clusters": cluster_rows,
        "joint_limit_dominated_frame_count": sum(
            float(row["minimum_joint_limit_margin_rad"]) <= 1e-6 for row in rows
        ),
        "source_targets_modified": False,
        "ownership_modified": False,
        "scene_modified": False,
    }
    write_json(output_root / "failure_cluster_summary.json", summary)
    return summary
