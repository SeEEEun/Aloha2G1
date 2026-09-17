#!/usr/bin/env python3
"""Physical Dataset-B gate audit for the frozen 50-episode Proposed-B batch.

The audit replays frozen q trajectories through the approved active model.  It
does not solve IK, change tolerances, mutate targets, or write trajectories.
Canonical-object geometry is reported in a separate diagnostic namespace and
is never used as a hard motion gate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    atomic_csv,
    atomic_json,
    sha256_file,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


DEFAULT_ROOT = (
    REPOSITORY
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
)
SIDES = ("left", "right")
DATASET_CLASSES = ("CLEAN_PASS", "USABLE_WITH_WARNING", "HARD_FAIL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def stats(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {key: math.nan for key in ("mean", "median", "std", "min", "max")}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def rotation_error_rad(actual: np.ndarray, desired: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", actual, desired)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def contiguous_segments(frames: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(int(value) for value in frames))
    if not ordered:
        return []
    output: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            output.append((start, previous))
            start = frame
        previous = frame
    output.append((start, previous))
    return output


def event_lookup(values: Mapping[str, np.ndarray]) -> dict[str, int | None]:
    output: dict[str, int | None] = {}
    for name, frame in zip(values["event_names"].astype(str), values["event_frames"]):
        output[str(name)] = None if int(frame) < 0 else int(frame)
    return output


def is_torso(body: str) -> bool:
    return any(token in body for token in ("torso", "pelvis", "waist"))


def is_arm(body: str) -> bool:
    return any(
        token in body
        for token in ("shoulder", "upper_arm", "elbow", "forearm", "wrist")
    )


def is_finger(body: str) -> bool:
    return "hand_" in body and any(
        token in body for token in ("thumb", "index", "middle")
    )


def is_palm_or_hand(body: str) -> bool:
    return "hand_palm" in body or "wrist" in body or "hand_" in body


def classify_contact(body_1: str, body_2: str) -> str:
    bodies = (body_1, body_2)
    sides = {
        side
        for side in SIDES
        if any(body.startswith(f"{side}_") for body in bodies)
    }
    if any(is_torso(body) for body in bodies) and any(is_arm(body) for body in bodies):
        return "ARM_TORSO_INVALID"
    if len(sides) == 2 and all(is_finger(body) for body in bodies):
        return "DISTAL_HAND_HAND_CONTACT"
    if len(sides) == 2 and any(is_palm_or_hand(body) for body in bodies):
        if any("hand_palm" in body or "wrist" in body for body in bodies):
            return "PALM_HAND_INVALID"
        return "CROSS_ARM_INVALID"
    if len(sides) == 2:
        return "CROSS_ARM_INVALID"
    return "OTHER_INVALID"


def contact_segment_severity(
    classification: str,
    maximum_depth_m: float,
    duration_s: float,
    policy: Mapping[str, float],
) -> tuple[str, str]:
    """Apply one global evidence policy; no episode or phase changes the rule."""
    if classification == "DISTAL_HAND_HAND_CONTACT":
        hard = (
            maximum_depth_m >= policy["distal_hard_depth_m"]
            and duration_s >= policy["distal_hard_duration_s"]
        )
        if hard:
            return "HARD", "DEEP_SUSTAINED_DISTAL_HAND_PENETRATION"
        return "WARNING", "DISTAL_HANDOFF_CONTACT_REQUIRES_REVIEW"
    if classification == "ARM_TORSO_INVALID":
        hard = maximum_depth_m >= policy["proximal_hard_depth_m"] or (
            maximum_depth_m >= policy["proximal_sustained_depth_m"]
            and duration_s >= policy["proximal_sustained_duration_s"]
        )
        if hard:
            return "HARD", "SUBSTANTIAL_OR_SUSTAINED_ARM_TORSO_PENETRATION"
        return "WARNING", "SHORT_OR_SHALLOW_ARM_TORSO_PENETRATION"
    hard = maximum_depth_m >= policy["other_hard_depth_m"] or (
        duration_s >= policy["other_hard_duration_s"]
    )
    if hard:
        return "HARD", f"SUBSTANTIAL_{classification}"
    return "WARNING", f"SHORT_OR_SHALLOW_{classification}"


def object_rectangle_clearance(
    point_xy: np.ndarray, radius: float, rectangle_center: np.ndarray, size: np.ndarray
) -> float:
    outside = np.maximum(np.abs(point_xy - rectangle_center) - 0.5 * size, 0.0)
    return float(np.linalg.norm(outside) - radius)


def main() -> int:
    args = parse_args()
    root = args.review_root.resolve()
    output = root / "review/dataset_b_gate"
    ik_dir = output / "ik_audit"
    collision_dir = output / "collision_audit"
    scene_dir = output / "canonical_scene_diagnostic"
    for path in (ik_dir, collision_dir, scene_dir):
        path.mkdir(parents=True, exist_ok=True)

    freeze_path = output / "motion_freeze/motion_freeze_manifest.json"
    freeze_sentinel = output / "motion_freeze/PROPOSED_B_MOTION_FROZEN_FOR_DATASET_AUDIT"
    if not freeze_path.is_file() or not freeze_sentinel.is_file():
        raise RuntimeError("explicit Dataset-B motion freeze is missing")
    freeze = load_json(freeze_path)
    if freeze.get("status") != "PROPOSED_B_MOTION_FROZEN_FOR_DATASET_AUDIT":
        raise RuntimeError("unexpected motion-freeze status")
    if float(freeze["handoff_cartesian_residual_m"]) != 0.0:
        raise RuntimeError("handoff Cartesian residual is not zero")

    common = load_json(root / "frozen_approval/config/common_config.json")
    scene = load_json(root / "frozen_approval/scene/scene_layout.json")
    tool = load_json(root / "frozen_approval/config/tool_frame_report.json")
    recalibration = load_json(
        REPOSITORY / "outputs/doll_handoff_retargeting/scene_recalibration.json"
    )
    if int(recalibration["source_count"]) != 50:
        raise RuntimeError("scene recalibration does not cover all 50 sources")
    per_episode_calibration = {
        int(row["episode_index"]): row for row in recalibration["per_episode"]
    }

    strict_position_m = float(common["shared_temporal_ik"]["position_tolerance_m"])
    physical_position_m = float(
        common["validation"]["physically_usable_position_tolerance_m"]
    )
    acceleration_limit = float(common["validation"]["maximum_acceleration_rad_s2"])
    collision_tolerance = float(
        common["validation"]["collision_penetration_tolerance_m"]
    )
    classification_policy: dict[str, Any] = {
        "solver_thresholds_unchanged": True,
        "strict_ik_position_m": strict_position_m,
        "physically_meaningful_ik_position_m": physical_position_m,
        "hard_ik_peak_m": 2.0 * physical_position_m,
        "hard_ik_contiguous_duration_s": 0.25,
        "ik_hard_rule": (
            "peak >= 2x the existing physical threshold OR contiguous error above "
            "the existing physical threshold for >=0.25 s"
        ),
        "collision_detection_penetration_tolerance_m": collision_tolerance,
        "distal_hard_depth_m": physical_position_m,
        "distal_hard_duration_s": 0.25,
        "proximal_hard_depth_m": physical_position_m,
        "proximal_sustained_depth_m": 0.005,
        "proximal_sustained_duration_s": 1.0,
        "other_hard_depth_m": 0.005,
        "other_hard_duration_s": 0.25,
        "acceleration_limit_rad_s2": acceleration_limit,
        "hard_acceleration_multiplier": 2.0,
        "hard_acceleration_contiguous_duration_s": 0.25,
        "canonical_object_pose_material_delta_m": physical_position_m,
        "canonical_scene_can_cause_hard_rejection": False,
        "rationale": {
            "ik": "uses the already frozen 10 mm strict and 15 mm physical diagnostics; no solver gate is relaxed",
            "distal": "15 mm is a deep overlap relative to the active Dex3 distal-pad geometry and must also persist to become hard",
            "arm_torso": "the neutral active model has no self contact; >=15 mm is substantial, while >=5 mm for >=1 s is sustained",
            "canonical_scene": "per-episode source object variation is visualization evidence only",
        },
    }
    atomic_json(output / "classification_policy.json", classification_policy)

    static_transforms = {
        side: np.asarray(tool["g1"][f"{side}_wrist_to_grasp_frame"], dtype=np.float64)
        for side in SIDES
    }
    g1 = G1Kinematics(common, scene)
    open_hands = {
        side: np.asarray(tool["g1"]["canonical_states"][side]["OPEN"], dtype=np.float64)
        for side in SIDES
    }
    neutral_geometry = g1.trajectory_geometry(
        g1.stand_qpos[g1.arm_qpos_ids][None, :],
        open_hands["left"][None, :],
        open_hands["right"][None, :],
        collision_tolerance,
    )
    neutral_clearance = g1.posture_clearance_state(
        g1.stand_qpos[g1.arm_qpos_ids]
    )
    neutral_reference = {
        "active_model_stand_has_collision_records": bool(
            neutral_geometry["collision_records"]
        ),
        "active_model_stand_collision_record_count": len(
            neutral_geometry["collision_records"]
        ),
        "active_model_stand_minimum_torso_clearance_m": float(
            neutral_clearance["TORSO"]["minimum_distance_m"]
        ),
        "active_model_stand_closest_torso_body_pair": list(
            neutral_clearance["TORSO"]["closest_body_pair"]
        ),
    }

    ik_frame_rows: list[dict[str, Any]] = []
    ik_segment_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    contact_segment_rows: list[dict[str, Any]] = []
    contact_class_segment_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    episode_data: list[dict[str, Any]] = []

    for episode in range(50):
        stable = f"doll_handoff_20260820_ep{episode:03d}"
        trajectory_path = root / "proposed/trajectories" / f"{stable}.npz"
        metrics_path = root / "proposed/metrics" / f"{stable}.json"
        if not trajectory_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(f"missing frozen episode artifact: {stable}")
        with np.load(trajectory_path, allow_pickle=False) as saved:
            values = {key: np.asarray(saved[key]) for key in saved.files}
        metric = load_json(metrics_path)
        frame_count = int(values["g1_arm_qpos"].shape[0])
        if frame_count != int(metric["frame_count"]):
            raise RuntimeError(f"frame-count drift: {stable}")
        timestamp = values["timestamp"].astype(np.float64)
        fps = float(metric["fps"])
        if len(timestamp) > 1:
            measured_fps = 1.0 / float(np.median(np.diff(timestamp)))
            if abs(measured_fps - fps) > 1e-3:
                raise RuntimeError(f"timestamp/FPS mismatch: {stable}")
        geometry = g1.trajectory_geometry(
            values["g1_arm_qpos"],
            values["left_dex3_qpos"],
            values["right_dex3_qpos"],
            collision_tolerance,
            static_transforms,
        )

        position_error: dict[str, np.ndarray] = {}
        orientation_error: dict[str, np.ndarray] = {}
        dynamic_grasp_error: dict[str, np.ndarray] = {}
        for side in SIDES:
            target_position = values[
                f"target_{side}_interaction_frame_position_world"
            ].astype(np.float64)
            achieved_position = geometry[
                f"{side}_static_tool_position_world"
            ].astype(np.float64)
            position_error[side] = np.linalg.norm(
                achieved_position - target_position, axis=1
            )
            desired_tool_rotation = np.einsum(
                "...ij,jk->...ik",
                values[f"target_{side}_wrist_rotation_model"].astype(np.float64),
                static_transforms[side][:3, :3],
            )
            orientation_error[side] = rotation_error_rad(
                geometry[f"{side}_static_tool_rotation_model"],
                desired_tool_rotation,
            )
            dynamic_grasp_error[side] = np.linalg.norm(
                values[f"achieved_{side}_physical_grasp_frame_position_world"].astype(
                    np.float64
                )
                - target_position,
                axis=1,
            )
        maximum_position_error = np.maximum(
            position_error["left"], position_error["right"]
        )
        calculated_strict_success = maximum_position_error <= strict_position_m
        saved_success = values["ik_success_per_frame"].astype(bool)
        success_flag_mismatch_count = int(
            np.count_nonzero(calculated_strict_success != saved_success)
        )
        if success_flag_mismatch_count > 1:
            raise RuntimeError(
                f"frozen strict-IK flag mismatch in {stable}: "
                f"{success_flag_mismatch_count} frames"
            )
        strict_frames = np.flatnonzero(~calculated_strict_success)
        physical_frames = np.flatnonzero(
            maximum_position_error > physical_position_m
        )
        numerical_only_frames = np.flatnonzero(
            (maximum_position_error > strict_position_m)
            & (maximum_position_error <= physical_position_m)
        )
        for frame in strict_frames:
            frame = int(frame)
            kind = (
                "STRICT_NUMERICAL_IK_FAIL_USABLE_IK_WARNING"
                if maximum_position_error[frame] <= physical_position_m
                else "PHYSICALLY_MEANINGFUL_IK_FAIL"
            )
            frame_row = {
                    "episode_index": episode,
                    "stable_episode_id": stable,
                    "source_name": metric["source_name"],
                    "frame": frame,
                    "source_frame_index": int(values["source_frame_index"][frame]),
                    "timestamp_s": float(timestamp[frame]),
                    "ownership_state": str(values["ownership_state"][frame]),
                    "left_hand_phase": str(values["left_hand_phase"][frame]),
                    "right_hand_phase": str(values["right_hand_phase"][frame]),
                    "left_achieved_vs_target_position_error_m": float(
                        position_error["left"][frame]
                    ),
                    "right_achieved_vs_target_position_error_m": float(
                        position_error["right"][frame]
                    ),
                    "maximum_achieved_vs_target_position_error_m": float(
                        maximum_position_error[frame]
                    ),
                    "maximum_error_side": (
                        "left"
                        if position_error["left"][frame]
                        >= position_error["right"][frame]
                        else "right"
                    ),
                    "left_orientation_error_rad_non_gating": float(
                        orientation_error["left"][frame]
                    ),
                    "right_orientation_error_rad_non_gating": float(
                        orientation_error["right"][frame]
                    ),
                    "classification": kind,
                }
            for side in SIDES:
                target_position = values[
                    f"target_{side}_interaction_frame_position_world"
                ][frame].astype(np.float64)
                achieved_position = geometry[
                    f"{side}_static_tool_position_world"
                ][frame].astype(np.float64)
                for axis, target_value, achieved_value in zip(
                    "xyz", target_position, achieved_position
                ):
                    frame_row[f"{side}_target_world_m_{axis}"] = float(target_value)
                    frame_row[f"{side}_achieved_world_m_{axis}"] = float(
                        achieved_value
                    )
            ik_frame_rows.append(frame_row)
        for kind, frames in (
            ("STRICT_NUMERICAL_IK_FAIL", strict_frames),
            ("PHYSICALLY_MEANINGFUL_IK_FAIL", physical_frames),
        ):
            for start, end in contiguous_segments(frames):
                local = maximum_position_error[start : end + 1]
                peak_local = int(start + np.argmax(local))
                ik_segment_rows.append(
                    {
                        "episode_index": episode,
                        "stable_episode_id": stable,
                        "classification": kind,
                        "start_frame": start,
                        "end_frame": end,
                        "frame_count": end - start + 1,
                        "duration_s": float((end - start + 1) / fps),
                        "start_time_s": float(timestamp[start]),
                        "end_time_s": float(timestamp[end]),
                        "peak_frame": peak_local,
                        "peak_error_m": float(maximum_position_error[peak_local]),
                        "ownership_states": ";".join(
                            sorted(set(values["ownership_state"][start : end + 1].astype(str)))
                        ),
                    }
                )
        physical_segments = contiguous_segments(physical_frames)
        longest_physical_duration = max(
            ((end - start + 1) / fps for start, end in physical_segments),
            default=0.0,
        )
        hard_ik = bool(
            float(np.max(maximum_position_error, initial=0.0))
            >= float(classification_policy["hard_ik_peak_m"])
            or longest_physical_duration
            >= float(classification_policy["hard_ik_contiguous_duration_s"])
        )

        per_episode_contacts: list[dict[str, Any]] = []
        for contact_index, record in enumerate(geometry["collision_records"]):
            frame = int(record["frame"])
            classification = classify_contact(record["body_1"], record["body_2"])
            row = {
                "episode_index": episode,
                "stable_episode_id": stable,
                "source_name": metric["source_name"],
                "frame": frame,
                "source_frame_index": int(values["source_frame_index"][frame]),
                "timestamp_s": float(timestamp[frame]),
                "ownership_state": str(values["ownership_state"][frame]),
                "left_hand_phase": str(values["left_hand_phase"][frame]),
                "right_hand_phase": str(values["right_hand_phase"][frame]),
                "classification": classification,
                "link_pair": record["pair"],
                "body_1": record["body_1"],
                "body_2": record["body_2"],
                "geom_1": record["geom_1"],
                "geom_2": record["geom_2"],
                "contact_index_in_frame_scan": contact_index,
                "penetration_depth_m": float(record["penetration_m"]),
            }
            for prefix, vector in (
                ("contact_position_world_m", record["contact_position_world_m"]),
                ("contact_normal_world", record["contact_normal_world"]),
            ):
                vector = np.asarray(vector, dtype=np.float64)
                for axis, value in zip("xyz", vector):
                    row[f"{prefix}_{axis}"] = float(value)
            per_episode_contacts.append(row)
            contact_rows.append(row)

        contact_groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in per_episode_contacts:
            contact_groups[(row["classification"], row["link_pair"])].append(row)
        episode_contact_segments: list[dict[str, Any]] = []
        for (classification, link_pair), rows in sorted(contact_groups.items()):
            frame_to_rows: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
            for row in rows:
                frame_to_rows[int(row["frame"])].append(row)
            for start, end in contiguous_segments(frame_to_rows):
                segment_records = [
                    row
                    for frame in range(start, end + 1)
                    for row in frame_to_rows[frame]
                ]
                maximum_depth = max(float(row["penetration_depth_m"]) for row in segment_records)
                duration = float((end - start + 1) / fps)
                severity, reason = contact_segment_severity(
                    classification, maximum_depth, duration, classification_policy
                )
                deepest = max(
                    segment_records, key=lambda row: float(row["penetration_depth_m"])
                )
                segment_row = {
                    "episode_index": episode,
                    "stable_episode_id": stable,
                    "classification": classification,
                    "severity": severity,
                    "reason": reason,
                    "link_pair": link_pair,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": end - start + 1,
                    "duration_s": duration,
                    "start_time_s": float(timestamp[start]),
                    "end_time_s": float(timestamp[end]),
                    "contact_record_count": len(segment_records),
                    "maximum_penetration_depth_m": maximum_depth,
                    "mean_penetration_depth_m": float(
                        np.mean(
                            [float(row["penetration_depth_m"]) for row in segment_records]
                        )
                    ),
                    "deepest_contact_frame": int(deepest["frame"]),
                    "ownership_states": ";".join(
                        sorted(set(str(row["ownership_state"]) for row in segment_records))
                    ),
                    "left_hand_phases": ";".join(
                        sorted(set(str(row["left_hand_phase"]) for row in segment_records))
                    ),
                    "right_hand_phases": ";".join(
                        sorted(set(str(row["right_hand_phase"]) for row in segment_records))
                    ),
                    "deepest_contact_position_world_m": [
                        float(deepest[f"contact_position_world_m_{axis}"])
                        for axis in "xyz"
                    ],
                    "deepest_contact_normal_world": [
                        float(deepest[f"contact_normal_world_{axis}"])
                        for axis in "xyz"
                    ],
                }
                for contact_row in segment_records:
                    contact_row["pair_segment_start_frame"] = start
                    contact_row["pair_segment_end_frame"] = end
                    contact_row["pair_segment_duration_s"] = duration
                    contact_row["pair_segment_severity"] = severity
                    contact_row["pair_segment_reason"] = reason
                episode_contact_segments.append(segment_row)
                contact_segment_rows.append(segment_row)

        # Severity is decided on continuous contact of the physical class, not
        # accidentally reset when contact migrates between neighboring links.
        episode_class_segments: list[dict[str, Any]] = []
        class_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in per_episode_contacts:
            class_groups[row["classification"]].append(row)
        for classification, rows in sorted(class_groups.items()):
            frame_to_rows: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
            for row in rows:
                frame_to_rows[int(row["frame"])].append(row)
            for start, end in contiguous_segments(frame_to_rows):
                segment_records = [
                    row
                    for frame in range(start, end + 1)
                    for row in frame_to_rows[frame]
                ]
                deepest = max(
                    segment_records, key=lambda row: float(row["penetration_depth_m"])
                )
                maximum_depth = float(deepest["penetration_depth_m"])
                duration = float((end - start + 1) / fps)
                severity, reason = contact_segment_severity(
                    classification, maximum_depth, duration, classification_policy
                )
                class_row = {
                    "episode_index": episode,
                    "stable_episode_id": stable,
                    "classification": classification,
                    "severity": severity,
                    "reason": reason,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_count": end - start + 1,
                    "duration_s": duration,
                    "start_time_s": float(timestamp[start]),
                    "end_time_s": float(timestamp[end]),
                    "contact_record_count": len(segment_records),
                    "link_pairs": ";".join(
                        sorted(set(str(row["link_pair"]) for row in segment_records))
                    ),
                    "maximum_penetration_depth_m": maximum_depth,
                    "mean_penetration_depth_m": float(
                        np.mean(
                            [float(row["penetration_depth_m"]) for row in segment_records]
                        )
                    ),
                    "deepest_contact_frame": int(deepest["frame"]),
                    "ownership_states": ";".join(
                        sorted(set(str(row["ownership_state"]) for row in segment_records))
                    ),
                    "deepest_contact_position_world_m": [
                        float(deepest[f"contact_position_world_m_{axis}"])
                        for axis in "xyz"
                    ],
                    "deepest_contact_normal_world": [
                        float(deepest[f"contact_normal_world_{axis}"])
                        for axis in "xyz"
                    ],
                }
                episode_class_segments.append(class_row)
                contact_class_segment_rows.append(class_row)

        q = values["g1_arm_qpos"].astype(np.float64)
        acceleration = np.abs(np.diff(q, n=2, axis=0)) * fps**2
        acceleration_per_frame = np.max(acceleration, axis=1) if len(acceleration) else np.empty(0)
        acceleration_frames = np.flatnonzero(acceleration_per_frame > acceleration_limit) + 2
        acceleration_segments = contiguous_segments(acceleration_frames)
        longest_acceleration_duration = max(
            ((end - start + 1) / fps for start, end in acceleration_segments),
            default=0.0,
        )
        hard_acceleration = bool(
            float(np.max(acceleration_per_frame, initial=0.0))
            > classification_policy["hard_acceleration_multiplier"] * acceleration_limit
            or longest_acceleration_duration
            >= classification_policy["hard_acceleration_contiguous_duration_s"]
        )

        events = event_lookup(values)
        left_grasp_frame = events.get("LEFT_GRASP")
        final_release_frame = events.get("RIGHT_FINAL_RELEASE")
        calibration = per_episode_calibration[episode]
        canonical_doll_xy = np.asarray(scene["doll"]["center_world_xy_m"], dtype=np.float64)
        episode_doll_xy = np.asarray(
            calibration["doll_initial_center_task_xy_m"], dtype=np.float64
        )
        canonical_bin_xy = np.asarray(scene["bin"]["center_world_xy_m"], dtype=np.float64)
        episode_bin_xy = np.asarray(calibration["bin_center_task_xy_m"], dtype=np.float64)
        doll_radius = 0.5 * float(scene["doll"]["diameter_m"])
        doll_z = float(scene["table"]["surface_height_m"]) + doll_radius
        canonical_doll = np.r_[canonical_doll_xy, doll_z]
        episode_doll = np.r_[episode_doll_xy, doll_z]
        doll_delta = float(np.linalg.norm(episode_doll_xy - canonical_doll_xy))
        bin_delta = float(np.linalg.norm(episode_bin_xy - canonical_bin_xy))
        material_doll_mismatch = bool(
            doll_delta > classification_policy["canonical_object_pose_material_delta_m"]
        )
        left_grasp_position = (
            values["achieved_left_physical_grasp_frame_position_world"][left_grasp_frame].astype(
                np.float64
            )
            if left_grasp_frame is not None
            else np.full(3, np.nan)
        )
        release_position = (
            values["achieved_right_physical_grasp_frame_position_world"][final_release_frame].astype(
                np.float64
            )
            if final_release_frame is not None
            else np.full(3, np.nan)
        )
        opening = np.asarray(scene["bin"]["opening_dimensions_xy_m"], dtype=np.float64)
        outer_bin = np.asarray(scene["bin"]["outer_dimensions_xyz_m"][:2], dtype=np.float64)
        canonical_release_inside = bool(
            np.isfinite(release_position).all()
            and np.all(np.abs(release_position[:2] - canonical_bin_xy) <= 0.5 * opening)
        )
        episode_release_inside = bool(
            np.isfinite(release_position).all()
            and np.all(np.abs(release_position[:2] - episode_bin_xy) <= 0.5 * opening)
        )
        scene_row = {
            "episode_index": episode,
            "stable_episode_id": stable,
            "source_name": metric["source_name"],
            "diagnostic_scope": "CANONICAL_SCENE_DIAGNOSTIC",
            "used_to_modify_trajectory": False,
            "used_as_hard_dataset_gate": False,
            "canonical_doll_xy_m": canonical_doll_xy.tolist(),
            "per_episode_source_doll_xy_m": episode_doll_xy.tolist(),
            "per_episode_doll_to_canonical_center_delta_m": doll_delta,
            "material_fixed_doll_visual_mismatch": material_doll_mismatch,
            "left_grasp_frame_to_fixed_canonical_doll_center_m": float(
                np.linalg.norm(left_grasp_position - canonical_doll)
            ),
            "left_grasp_frame_to_per_episode_source_doll_center_m": float(
                np.linalg.norm(left_grasp_position - episode_doll)
            ),
            "canonical_bin_xy_m": canonical_bin_xy.tolist(),
            "per_episode_source_bin_xy_m": episode_bin_xy.tolist(),
            "per_episode_bin_to_canonical_center_delta_m": bin_delta,
            "initial_doll_to_fixed_bin_outer_clearance_m": object_rectangle_clearance(
                episode_doll_xy, doll_radius, canonical_bin_xy, outer_bin
            ),
            "right_release_grasp_frame_world_m": release_position.tolist(),
            "right_release_xy_inside_fixed_canonical_bin": canonical_release_inside,
            "right_release_xy_inside_per_episode_source_bin_estimate": episode_release_inside,
            "right_release_horizontal_distance_to_fixed_bin_center_m": float(
                np.linalg.norm(release_position[:2] - canonical_bin_xy)
            ),
            "right_release_horizontal_distance_to_source_bin_estimate_m": float(
                np.linalg.norm(release_position[:2] - episode_bin_xy)
            ),
        }
        scene_rows.append(scene_row)

        hard_reasons: list[str] = []
        warning_reasons: list[str] = []
        if not bool(metric["finite"]) or int(metric["nan_inf_count"]):
            hard_reasons.append("NONFINITE_TRAJECTORY")
        if int(metric["joint_limit_violation_count"]):
            hard_reasons.append("JOINT_LIMIT_VIOLATION")
        if int(metric["branch_discontinuity_count"]):
            hard_reasons.append("BRANCH_DISCONTINUITY")
        if hard_ik:
            hard_reasons.append("PHYSICALLY_MEANINGFUL_PERSISTENT_OR_LARGE_IK_RESIDUAL")
        elif len(physical_frames):
            warning_reasons.append("SHORT_PHYSICALLY_MEANINGFUL_IK_RESIDUAL")
        elif len(numerical_only_frames):
            warning_reasons.append("STRICT_NUMERICAL_IK_FAIL_USABLE_IK_WARNING")
        hard_contact_segments = [
            row for row in episode_class_segments if row["severity"] == "HARD"
        ]
        warning_contact_segments = [
            row for row in episode_class_segments if row["severity"] == "WARNING"
        ]
        if hard_contact_segments:
            hard_reasons.extend(
                sorted(
                    {
                        f"{row['classification']}:{row['reason']}"
                        for row in hard_contact_segments
                    }
                )
            )
        if warning_contact_segments:
            warning_reasons.extend(
                sorted(
                    {
                        f"{row['classification']}:{row['reason']}"
                        for row in warning_contact_segments
                    }
                )
            )
        if hard_acceleration:
            hard_reasons.append("SEVERE_SUSTAINED_ACCELERATION_OR_TRAJECTORY_JUMP")
        elif len(acceleration_frames):
            warning_reasons.append("ISOLATED_ACCELERATION_SPIKE")
        if not bool(metric["semantics"]["right_grasp_before_left_release"]):
            hard_reasons.append("RIGHT_ACQUIRE_NOT_BEFORE_LEFT_RELEASE")
        if not bool(metric["semantics"]["ownership_transition_validity"]):
            hard_reasons.append("OWNERSHIP_TRANSITION_INVALID")
        if material_doll_mismatch:
            warning_reasons.append("CANONICAL_FIXED_DOLL_POSE_MISMATCH_DIAGNOSTIC_ONLY")
        if not canonical_release_inside:
            warning_reasons.append("CANONICAL_FIXED_BIN_RELEASE_XY_MISS_DIAGNOSTIC_ONLY")
        hard_reasons = sorted(set(hard_reasons))
        warning_reasons = sorted(set(warning_reasons))
        episode_classification = (
            "HARD_FAIL"
            if hard_reasons
            else "USABLE_WITH_WARNING"
            if warning_reasons
            else "CLEAN_PASS"
        )

        task = metric["task_space"]
        episode_data.append(
            {
                "episode_index": episode,
                "stable_episode_id": stable,
                "source_name": metric["source_name"],
                "frame_count": frame_count,
                "classification": episode_classification,
                "hard_fail_reasons": hard_reasons,
                "warning_reasons": warning_reasons,
                "strict_ik_success_rate_recomputed": float(
                    np.mean(calculated_strict_success)
                ),
                "saved_strict_ik_success_rate": float(metric["ik_success_rate"]),
                "strict_ik_failed_frame_count": int(len(strict_frames)),
                "strict_numerical_only_frame_count": int(len(numerical_only_frames)),
                "physically_meaningful_ik_failed_frame_count": int(len(physical_frames)),
                "maximum_achieved_vs_target_error_m": float(
                    np.max(maximum_position_error, initial=0.0)
                ),
                "mean_achieved_vs_target_error_m": float(
                    np.mean(np.column_stack(tuple(position_error.values())))
                ),
                "longest_physical_ik_failure_duration_s": float(
                    longest_physical_duration
                ),
                "hard_ik_fail": hard_ik,
                "strict_success_flag_replay_mismatch_count": success_flag_mismatch_count,
                "mean_non_gating_orientation_error_rad": float(
                    np.mean(np.column_stack(tuple(orientation_error.values())))
                ),
                "mean_physical_grasp_frame_error_m": float(
                    np.mean(np.column_stack(tuple(dynamic_grasp_error.values())))
                ),
                "joint_limit_violation_count": int(metric["joint_limit_violation_count"]),
                "branch_discontinuity_count": int(metric["branch_discontinuity_count"]),
                "maximum_joint_velocity_rad_s": float(metric["maximum_joint_velocity_rad_s"]),
                "maximum_joint_acceleration_rad_s2": float(
                    metric["maximum_joint_acceleration_rad_s2"]
                ),
                "acceleration_limit_exceedance_frame_count": int(len(acceleration_frames)),
                "hard_acceleration_fail": hard_acceleration,
                "robot_self_contact_record_count": len(per_episode_contacts),
                "robot_self_contact_frame_count": len(
                    {int(row["frame"]) for row in per_episode_contacts}
                ),
                "hard_robot_self_contact_segment_count": len(hard_contact_segments),
                "warning_robot_self_contact_segment_count": len(
                    warning_contact_segments
                ),
                "contact_frame_counts_by_class": {
                    classification: len(
                        {
                            int(row["frame"])
                            for row in per_episode_contacts
                            if row["classification"] == classification
                        }
                    )
                    for classification in (
                        "DISTAL_HAND_HAND_CONTACT",
                        "PALM_HAND_INVALID",
                        "CROSS_ARM_INVALID",
                        "ARM_TORSO_INVALID",
                        "OTHER_INVALID",
                    )
                },
                "mean_bimanual_relation_error_m": float(
                    metric["bimanual"]["inter_hand_relation_error_m"]["mean"]
                ),
                "mean_bimanual_relative_vector_error_m": float(
                    metric["bimanual"]["relative_vector_error_m"]["mean"]
                ),
                "handoff_order_valid": bool(
                    metric["semantics"]["right_grasp_before_left_release"]
                ),
                "ownership_transition_valid": bool(
                    metric["semantics"]["ownership_transition_validity"]
                ),
                "release_event_present": final_release_frame is not None,
                "canonical_release_xy_inside_bin": canonical_release_inside,
                "per_episode_source_bin_release_xy_inside": episode_release_inside,
                "canonical_fixed_doll_material_mismatch": material_doll_mismatch,
                "canonical_scene_diagnostic": scene_row,
                "legacy_top_level_status": metric["status"],
                "left_static_grasp_objective_mean_error_m": float(
                    task["left_static_canonical_grasp_frame_target_error_m"]["mean"]
                ),
                "right_static_grasp_objective_mean_error_m": float(
                    task["right_static_canonical_grasp_frame_target_error_m"]["mean"]
                ),
            }
        )

    # All trajectories have now been replayed; emit evidence before aggregation.
    atomic_csv(ik_dir / "strict_ik_failed_frames.csv", ik_frame_rows)
    atomic_csv(ik_dir / "ik_failure_segments.csv", ik_segment_rows)
    atomic_csv(collision_dir / "robot_self_contact_records.csv", contact_rows)
    atomic_csv(collision_dir / "robot_self_contact_segments.csv", contact_segment_rows)
    atomic_csv(
        collision_dir / "robot_self_contact_class_segments.csv",
        contact_class_segment_rows,
    )
    atomic_csv(scene_dir / "per_episode_object_pose_diagnostics.csv", scene_rows)

    flat_rows: list[dict[str, Any]] = []
    for row in episode_data:
        flat = {
            key: value
            for key, value in row.items()
            if key not in {"hard_fail_reasons", "warning_reasons", "contact_frame_counts_by_class", "canonical_scene_diagnostic"}
        }
        flat["hard_fail_reasons"] = ";".join(row["hard_fail_reasons"])
        flat["warning_reasons"] = ";".join(row["warning_reasons"])
        for name, count in row["contact_frame_counts_by_class"].items():
            flat[f"{name.lower()}_frames"] = count
        flat_rows.append(flat)
    atomic_csv(output / "dataset_b_episode_classification.csv", flat_rows)
    atomic_json(output / "dataset_b_episode_classification.json", episode_data)

    classification_counts = collections.Counter(
        row["classification"] for row in episode_data
    )
    if sum(classification_counts.values()) != 50:
        raise RuntimeError("classification did not cover all 50 episodes")
    hard_failures = [row for row in episode_data if row["classification"] == "HARD_FAIL"]
    six_legacy_ik_fails = [2, 3, 4, 5, 13, 23]
    numerical_only_legacy = [
        episode
        for episode in six_legacy_ik_fails
        if episode_data[episode]["physically_meaningful_ik_failed_frame_count"] == 0
    ]
    physical_hard_legacy = [
        episode
        for episode in six_legacy_ik_fails
        if episode_data[episode]["hard_ik_fail"]
    ]
    contact_class_counts = {
        name: {
            "record_count": sum(1 for row in contact_rows if row["classification"] == name),
            "episode_count": len(
                {
                    int(row["episode_index"])
                    for row in contact_rows
                    if row["classification"] == name
                }
            ),
            "frame_count": len(
                {
                    (int(row["episode_index"]), int(row["frame"]))
                    for row in contact_rows
                    if row["classification"] == name
                }
            ),
            "hard_episode_count": len(
                {
                    int(row["episode_index"])
                    for row in contact_class_segment_rows
                    if row["classification"] == name and row["severity"] == "HARD"
                }
            ),
        }
        for name in (
            "DISTAL_HAND_HAND_CONTACT",
            "PALM_HAND_INVALID",
            "CROSS_ARM_INVALID",
            "ARM_TORSO_INVALID",
            "OTHER_INVALID",
        )
    }
    distal_warning_episodes = sorted(
        {
            int(row["episode_index"])
            for row in contact_class_segment_rows
            if row["classification"] == "DISTAL_HAND_HAND_CONTACT"
            and row["severity"] == "WARNING"
        }
    )
    distal_hard_episodes = sorted(
        {
            int(row["episode_index"])
            for row in contact_class_segment_rows
            if row["classification"] == "DISTAL_HAND_HAND_CONTACT"
            and row["severity"] == "HARD"
        }
    )
    arm_torso_hard_episodes = sorted(
        {
            int(row["episode_index"])
            for row in contact_class_segment_rows
            if row["classification"] == "ARM_TORSO_INVALID"
            and row["severity"] == "HARD"
        }
    )
    cross_arm_hard_episodes = sorted(
        {
            int(row["episode_index"])
            for row in contact_class_segment_rows
            if row["classification"] == "CROSS_ARM_INVALID"
            and row["severity"] == "HARD"
        }
    )
    material_doll_episodes = [
        int(row["episode_index"])
        for row in scene_rows
        if row["material_fixed_doll_visual_mismatch"]
    ]
    release_outside_canonical = [
        int(row["episode_index"])
        for row in scene_rows
        if not row["right_release_xy_inside_fixed_canonical_bin"]
    ]
    release_outside_source_estimate = [
        int(row["episode_index"])
        for row in scene_rows
        if not row["right_release_xy_inside_per_episode_source_bin_estimate"]
    ]

    aggregate = {
        "schema_version": "proposed_b_dataset_gate_audit_v1",
        "status": "DATASET_B_NOT_READY" if hard_failures else "DATASET_B_READY",
        "motion": {
            "frozen": True,
            "freeze_status": freeze["status"],
            "freeze_manifest": str(freeze_path),
            "freeze_manifest_sha256": sha256_file(freeze_path),
            "implementation_sha256": freeze["implementation_sha256"],
            "trajectory_file_set_sha256": freeze["trajectory_file_set_sha256"],
            "cartesian_target_array_set_sha256": freeze[
                "cartesian_target_array_set_sha256"
            ],
            "changed_by_audit": False,
        },
        "episode_count": 50,
        "classification_counts": {
            name: int(classification_counts.get(name, 0)) for name in DATASET_CLASSES
        },
        "hard_fail_episodes": [int(row["episode_index"]) for row in hard_failures],
        "hard_fail_reasons": {
            f"ep{int(row['episode_index']):03d}": row["hard_fail_reasons"]
            for row in hard_failures
        },
        "strict_ik_audit": {
            "strict_threshold_m_unchanged": strict_position_m,
            "physical_threshold_m_unchanged": physical_position_m,
            "legacy_top_level_fail_episodes": six_legacy_ik_fails,
            "legacy_fail_numerical_only_episodes": numerical_only_legacy,
            "legacy_fail_physical_hard_episodes": physical_hard_legacy,
            "all_episodes_with_hard_ik": [
                int(row["episode_index"]) for row in episode_data if row["hard_ik_fail"]
            ],
            "strict_failed_frame_count": len(ik_frame_rows),
            "physically_meaningful_failed_frame_count": sum(
                int(row["physically_meaningful_ik_failed_frame_count"])
                for row in episode_data
            ),
            "strict_numerical_only_frame_count": sum(
                int(row["strict_numerical_only_frame_count"]) for row in episode_data
            ),
            "strict_success_rate_distribution": stats(
                row["strict_ik_success_rate_recomputed"] for row in episode_data
            ),
            "maximum_residual_m_distribution": stats(
                row["maximum_achieved_vs_target_error_m"] for row in episode_data
            ),
            "per_failed_frame_csv": str(ik_dir / "strict_ik_failed_frames.csv"),
        },
        "robot_self_collision": {
            "canonical_object_contacts_included": False,
            "neutral_reference": neutral_reference,
            "class_counts": contact_class_counts,
            "distal_warning_episodes": distal_warning_episodes,
            "distal_hard_episodes": distal_hard_episodes,
            "arm_torso_hard_episodes": arm_torso_hard_episodes,
            "cross_arm_hard_episodes": cross_arm_hard_episodes,
            "hard_contact_episodes": sorted(
                {
                    int(row["episode_index"])
                    for row in contact_class_segment_rows
                    if row["severity"] == "HARD"
                }
            ),
            "warning_only_contact_episodes": sorted(
                {
                    int(row["episode_index"])
                    for row in contact_class_segment_rows
                    if row["severity"] == "WARNING"
                }
                - {
                    int(row["episode_index"])
                    for row in contact_class_segment_rows
                    if row["severity"] == "HARD"
                }
            ),
            "records_csv": str(collision_dir / "robot_self_contact_records.csv"),
            "segments_csv": str(collision_dir / "robot_self_contact_segments.csv"),
            "class_segments_csv": str(
                collision_dir / "robot_self_contact_class_segments.csv"
            ),
        },
        "interaction_semantics": {
            "right_acquire_before_left_release_valid": sum(
                bool(row["handoff_order_valid"]) for row in episode_data
            ),
            "ownership_transition_valid": sum(
                bool(row["ownership_transition_valid"]) for row in episode_data
            ),
            "release_event_present": sum(
                bool(row["release_event_present"]) for row in episode_data
            ),
            "canonical_release_xy_inside_bin": sum(
                bool(row["canonical_release_xy_inside_bin"]) for row in episode_data
            ),
            "mean_physical_grasp_frame_error_m_distribution": stats(
                row["mean_physical_grasp_frame_error_m"] for row in episode_data
            ),
            "mean_bimanual_relation_error_m_distribution": stats(
                row["mean_bimanual_relation_error_m"] for row in episode_data
            ),
        },
        "canonical_scene_diagnostic": {
            "primary_dataset_gate": False,
            "label": "CANONICAL_SCENE_DIAGNOSTIC",
            "material_fixed_doll_mismatch_episodes": material_doll_episodes,
            "material_fixed_doll_mismatch_count": len(material_doll_episodes),
            "canonical_release_outside_bin_episodes": release_outside_canonical,
            "per_episode_source_bin_estimate_release_outside_episodes": release_outside_source_estimate,
            "object_pose_csv": str(scene_dir / "per_episode_object_pose_diagnostics.csv"),
        },
        "temporal": {
            "joint_limit_clean_episode_count": sum(
                int(row["joint_limit_violation_count"]) == 0 for row in episode_data
            ),
            "branch_discontinuity_clean_episode_count": sum(
                int(row["branch_discontinuity_count"]) == 0 for row in episode_data
            ),
            "isolated_acceleration_warning_episodes": [
                int(row["episode_index"])
                for row in episode_data
                if row["acceleration_limit_exceedance_frame_count"]
                and not row["hard_acceleration_fail"]
            ],
            "hard_acceleration_episodes": [
                int(row["episode_index"])
                for row in episode_data
                if row["hard_acceleration_fail"]
            ],
        },
        "dataset_b": {
            "ready": not bool(hard_failures),
            "packaging_started": False,
            "policy_training_started": False,
            "xr_data_used": False,
            "recommended_handling": (
                "B: diagnose one common target-realization issue and rerun all 50; "
                "do not exclude only B episodes or apply episode-specific corrections"
                if hard_failures
                else "Proceed to G1-compatible state/action schema and Dataset-B packaging"
            ),
        },
        "classification_policy": classification_policy,
    }
    atomic_json(output / "dataset_b_readiness_audit.json", aggregate)

    collision_lines = [
        "# Frozen Proposed-B robot self-contact audit",
        "",
        "Canonical doll/bin contacts are excluded from this table. Every record is a replayed robot↔robot MuJoCo penetration from the frozen trajectory.",
        "",
        f"- Active-model neutral stand contact records: **{neutral_reference['active_model_stand_collision_record_count']}**",
        f"- Neutral minimum torso clearance: **{neutral_reference['active_model_stand_minimum_torso_clearance_m'] * 1000.0:.3f} mm**",
        f"- Detection tolerance (unchanged): **{collision_tolerance * 1000.0:.3f} mm**",
        "",
        "| class | records | episode count | unique frames | hard episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in contact_class_counts.items():
        collision_lines.append(
            f"| {name} | {counts['record_count']} | {counts['episode_count']} | "
            f"{counts['frame_count']} | {counts['hard_episode_count']} |"
        )
    collision_lines.extend(
        [
            "",
            "Hard/warning severity uses the single dataset-wide policy in `classification_policy.json`; ownership phase is context only and never changes the penetration threshold.",
            "",
            f"Raw records: `{collision_dir / 'robot_self_contact_records.csv'}`",
            f"Contiguous segments: `{collision_dir / 'robot_self_contact_segments.csv'}`",
            f"Class-continuous severity segments: `{collision_dir / 'robot_self_contact_class_segments.csv'}`",
        ]
    )
    (collision_dir / "robot_self_collision_audit.md").write_text(
        "\n".join(collision_lines) + "\n", encoding="utf-8"
    )

    ik_lines = [
        "# Frozen Proposed-B IK residual audit",
        "",
        "The approved solver was not rerun and no tolerance was changed. Errors are recomputed from frozen q labels, the active G1 model, static wrist→whole-hand grasp transform, and frozen interaction-frame targets.",
        "",
        f"- Strict position gate: **{strict_position_m * 1000.0:.1f} mm**",
        f"- Existing physical diagnostic gate: **{physical_position_m * 1000.0:.1f} mm**",
        f"- Legacy FAIL_IK numerical-only episodes: **{numerical_only_legacy}**",
        f"- Legacy FAIL_IK physically hard episodes: **{physical_hard_legacy}**",
        "",
        "| episode | strict success | physical frames | max error | longest physical segment | classification |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for episode in six_legacy_ik_fails:
        row = episode_data[episode]
        ik_lines.append(
            f"| ep{episode:03d} | {row['strict_ik_success_rate_recomputed'] * 100.0:.3f}% | "
            f"{row['physically_meaningful_ik_failed_frame_count']} | "
            f"{row['maximum_achieved_vs_target_error_m'] * 1000.0:.3f} mm | "
            f"{row['longest_physical_ik_failure_duration_s']:.3f} s | "
            f"{'HARD_IK_FAIL' if row['hard_ik_fail'] else 'USABLE_IK_WARNING'} |"
        )
    ik_lines.extend(
        [
            "",
            f"Every strict failed frame: `{ik_dir / 'strict_ik_failed_frames.csv'}`",
            f"Contiguous residual segments: `{ik_dir / 'ik_failure_segments.csv'}`",
        ]
    )
    (ik_dir / "ik_residual_audit.md").write_text(
        "\n".join(ik_lines) + "\n", encoding="utf-8"
    )

    scene_lines = [
        "# Canonical-scene object geometry diagnostic",
        "",
        "**CANONICAL_SCENE_DIAGNOSTIC — never a hard Dataset-B gate.**",
        "",
        "The canonical scene uses one global median doll/bin pose. The source-image homography records a distinct initial doll estimate for every episode; neither those estimates nor this report changes a target or action label.",
        "",
        f"- Material fixed-doll visualization mismatch (> {classification_policy['canonical_object_pose_material_delta_m'] * 1000.0:.1f} mm): **{len(material_doll_episodes)} / 50**",
        f"- Episodes: **{material_doll_episodes}**",
        f"- Release XY inside canonical bin: **{50 - len(release_outside_canonical)} / 50**",
        f"- Canonical release misses: **{release_outside_canonical}**",
        f"- Release XY inside per-episode source-bin estimate: **{50 - len(release_outside_source_estimate)} / 50**",
        f"- Source-bin-estimate misses: **{release_outside_source_estimate}**",
        "",
        f"Per-episode geometry: `{scene_dir / 'per_episode_object_pose_diagnostics.csv'}`",
    ]
    (scene_dir / "canonical_scene_diagnostic.md").write_text(
        "\n".join(scene_lines) + "\n", encoding="utf-8"
    )

    report_lines = [
        "# Proposed-B Dataset-B readiness gate",
        "",
        "## Frozen motion",
        "",
        "- Status: **PROPOSED_B_MOTION_FROZEN_FOR_DATASET_AUDIT**",
        f"- Implementation SHA256: `{aggregate['motion']['implementation_sha256']}`",
        f"- Trajectory-set SHA256: `{aggregate['motion']['trajectory_file_set_sha256']}`",
        f"- Cartesian-target-set SHA256: `{aggregate['motion']['cartesian_target_array_set_sha256']}`",
        "- Motion/config/solver changes in this audit: **NO**",
        "",
        "## 50-episode classification",
        "",
        f"- CLEAN_PASS: **{classification_counts.get('CLEAN_PASS', 0)} / 50**",
        f"- USABLE_WITH_WARNING: **{classification_counts.get('USABLE_WITH_WARNING', 0)} / 50**",
        f"- HARD_FAIL: **{classification_counts.get('HARD_FAIL', 0)} / 50**",
        "",
        "## Interaction semantics vs target realization",
        "",
        f"- RIGHT acquisition before LEFT release: **{aggregate['interaction_semantics']['right_acquire_before_left_release_valid']} / 50**",
        f"- Ownership transition valid: **{aggregate['interaction_semantics']['ownership_transition_valid']} / 50**",
        f"- Release event present: **{aggregate['interaction_semantics']['release_event_present']} / 50**",
        f"- Release XY inside canonical bin (diagnostic): **{aggregate['interaction_semantics']['canonical_release_xy_inside_bin']} / 50**",
        "- Fixed-doll mismatch is a visualization diagnostic, not evidence that the Interaction-Centric representation failed.",
        "",
        "## Hard failures",
        "",
    ]
    if hard_failures:
        report_lines.extend(
            f"- ep{int(row['episode_index']):03d}: " + "; ".join(row["hard_fail_reasons"])
            for row in hard_failures
        )
    else:
        report_lines.append("- None")
    report_lines.extend(
        [
            "",
            "## Dataset-B gate",
            "",
            f"- Status: **{aggregate['status']}**",
            f"- Recommendation: {aggregate['dataset_b']['recommended_handling']}",
            "- Dataset packaging: **NOT_STARTED_BY_DESIGN**",
            "- Policy training: **NOT_STARTED_BY_DESIGN**",
            "- XR data used: **NO**",
        ]
    )
    report_path = output / "dataset_b_readiness_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    console_lines = [
        "PROPOSED B MOTION",
        "FROZEN: YES",
        f"implementation SHA256: {aggregate['motion']['implementation_sha256']}",
        f"trajectory-set SHA256: {aggregate['motion']['trajectory_file_set_sha256']}",
        f"Cartesian-target-set SHA256: {aggregate['motion']['cartesian_target_array_set_sha256']}",
        "",
        "50 EPISODES",
        f"CLEAN_PASS: {classification_counts.get('CLEAN_PASS', 0)} / 50",
        f"USABLE_WITH_WARNING: {classification_counts.get('USABLE_WITH_WARNING', 0)} / 50",
        f"HARD_FAIL: {classification_counts.get('HARD_FAIL', 0)} / 50",
        "",
        "STRICT IK FAILS",
        f"numerical-only (original six): {len(numerical_only_legacy)} / 6 {numerical_only_legacy}",
        f"physical hard fails (original six): {len(physical_hard_legacy)} / 6 {physical_hard_legacy}",
        f"all physical-hard IK episodes: {aggregate['strict_ik_audit']['all_episodes_with_hard_ik']}",
        "",
        "ROBOT SELF-COLLISION",
        f"distal warnings: {len(distal_warning_episodes)} episodes {distal_warning_episodes}",
        f"distal hard: {len(distal_hard_episodes)} episodes {distal_hard_episodes}",
        f"arm/torso hard: {len(arm_torso_hard_episodes)} episodes {arm_torso_hard_episodes}",
        f"cross-arm hard: {len(cross_arm_hard_episodes)} episodes {cross_arm_hard_episodes}",
        "",
        "FIXED DOLL / SCENE MISMATCH",
        f"episodes affected (>15 mm): {len(material_doll_episodes)} / 50 {material_doll_episodes}",
        "classification: CANONICAL_SCENE_DIAGNOSTIC",
        "primary dataset gate: NO",
        "",
        "INTERACTION SEMANTICS",
        f"handoff: {aggregate['interaction_semantics']['right_acquire_before_left_release_valid']} / 50",
        f"ownership: {aggregate['interaction_semantics']['ownership_transition_valid']} / 50",
        f"release: {aggregate['interaction_semantics']['canonical_release_xy_inside_bin']} / 50 canonical; {50 - len(release_outside_source_estimate)} / 50 per-episode source-bin estimate",
        "",
        "DATASET B",
        "NOT READY" if hard_failures else "READY",
        "",
        "HARD-FAIL EPISODES",
    ]
    if hard_failures:
        console_lines.extend(
            f"ep{int(row['episode_index']):03d}: " + "; ".join(row["hard_fail_reasons"])
            for row in hard_failures
        )
    else:
        console_lines.append("NONE")
    console_lines.extend(
        [
            "",
            "NEXT STEP",
            aggregate["dataset_b"]["recommended_handling"],
            "",
            "DATASET PACKAGING: NOT_STARTED_BY_DESIGN",
            "POLICY TRAINING: NOT_STARTED_BY_DESIGN",
            "XR DATA USED: NO",
        ]
    )
    (output / "dataset_b_gate_summary.txt").write_text(
        "\n".join(console_lines) + "\n", encoding="utf-8"
    )
    final_marker = (
        "BLOCKED_BY_TRUE_B_HARD_FAILURES"
        if hard_failures
        else "READY_FOR_DATASET_B_PACKAGING"
    )
    (output / final_marker).write_text(
        f"{final_marker}\n"
        f"report: {report_path}\n"
        f"hard_fail_count: {len(hard_failures)}\n"
        f"trajectory_file_set_sha256: {aggregate['motion']['trajectory_file_set_sha256']}\n",
        encoding="utf-8",
    )

    # A final hash manifest covers only newly generated gate artifacts.  Frozen
    # trajectories remain protected by the earlier motion manifest.
    artifact_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"dataset_b_gate_artifact_manifest.json"}
    )
    artifact_manifest = {
        "schema_version": "dataset_b_gate_artifact_manifest_v1",
        "motion_freeze_manifest_sha256": sha256_file(freeze_path),
        "artifacts": {
            str(path.relative_to(root)): sha256_file(path) for path in artifact_paths
        },
    }
    artifact_manifest["artifact_set_sha256"] = hashlib.sha256(
        json.dumps(
            artifact_manifest["artifacts"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    atomic_json(output / "dataset_b_gate_artifact_manifest.json", artifact_manifest)

    print(
        json.dumps(
            {
                "status": aggregate["status"],
                "classification_counts": aggregate["classification_counts"],
                "hard_fail_episodes": aggregate["hard_fail_episodes"],
                "legacy_strict_ik_numerical_only": numerical_only_legacy,
                "legacy_strict_ik_physical_hard": physical_hard_legacy,
                "material_fixed_doll_mismatch_count": len(material_doll_episodes),
                "canonical_release_inside_bin": 50 - len(release_outside_canonical),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
