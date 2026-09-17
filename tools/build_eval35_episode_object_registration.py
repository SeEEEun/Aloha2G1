#!/usr/bin/env python3
"""Build the source-only, episode-conditioned EVAL35 doll registration.

This program must run before any post-registration physical rollout.  It reads
raw ALOHA state/images, the pre-retargeting event audit, frozen conversion
metadata, and frozen physical-scene geometry.  It never reads an ACT physical
rollout or score and never uses an ACT wrist pose to place the object.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from tools.audit_final_grasp_capture_task_frames import (  # noqa: E402
    DistanceModel,
    command_registration,
    method_geometry,
    source_name,
)
from tools.common_execution_isaac_runtime import (  # noqa: E402
    COMMON_PHYSICAL_CONTROLLER,
    PHYSICAL_ENVIRONMENT,
    PHYSICS_CONFIG,
)
from tools.common_execution_layer import Dex3Primitive, pose_matrix  # noqa: E402
from tools.direct_physical_execution_layer import authoritative_joint_limits  # noqa: E402
from tools.doll_handoff_retargeting.common import (  # noqa: E402
    load_common_config,
    load_scene,
    transform,
)
from tools.doll_handoff_retargeting.models import (  # noqa: E402
    ALOHAKinematics,
    G1Kinematics,
)
from tools.doll_handoff_retargeting.source import fixed_list_numpy  # noqa: E402
from tools.finalize_doll_handoff_dataset_b import source_image_object_estimate  # noqa: E402


OUT = ROOT / "outputs/final_episode_registered_eval35/00_registration"
EVAL35 = ROOT / "outputs/final_representation_neutral_eval/06_common_execution_layer/EVAL35_MANIFEST.json"
COMMANDS = ROOT / "outputs/final_direct_physical_eval35/00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
INTENTS = ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_SOURCE_TASK_INTENT_EVAL35.json"
SCENE_CALIBRATION = ROOT / "outputs/doll_handoff_retargeting/scene_recalibration.json"
NEW_SOURCE_ESTIMATES = ROOT / "outputs/doll_handoff_dataset_b_final/new_episode_conversion/source_image_object_estimates.json"
COMMON_CONFIG = ROOT / "outputs/final_direct_physical_eval35/00_preparation/runtime_frozen_fair_a/config/common_config.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
PALM_FRAME = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"
RAW = ROOT / "raw_recordings"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def quat_xyzw(rotation: np.ndarray) -> list[float]:
    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat().tolist()


def source_measurements(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    calibration = read_json(SCENE_CALIBRATION)
    known = {str(row["source_name"]): row for row in calibration["per_episode"]}
    for value in read_json(NEW_SOURCE_ESTIMATES).values():
        observation = Path(value["observations"][0]["image"])
        name = next(part for part in observation.parts if part.startswith("GoPark_"))
        known[name] = value
    names = [source_name(entry, calibration) for entry in entries]
    for name in names:
        if name not in known:
            known[name] = source_image_object_estimate(name)
    return {name: known[name] for name in names}


def source_doll_mask_points_task(
    image_path: Path,
    detector: dict[str, Any],
    homography: np.ndarray,
) -> tuple[np.ndarray, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read source image {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.asarray(detector["hsv_lower_opencv"], dtype=np.uint8)
    upper = np.asarray(detector["hsv_upper_opencv"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    x0, y0, x1, y1 = map(int, detector["roi_xyxy_px"])
    keep = np.zeros_like(mask)
    keep[y0:y1, x0:x1] = 255
    mask = cv2.bitwise_and(mask, keep)
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        raise RuntimeError(f"doll mask has only {len(xs)} pixels: {image_path}")
    points_px = np.column_stack((xs, ys)).astype(np.float32).reshape(1, -1, 2)
    points_task = cv2.perspectiveTransform(points_px, homography)[0].astype(np.float64)
    return points_task, int(len(xs))


def episode_visual_yaw(
    source: str,
    measurement: dict[str, Any],
    scene_calibration: dict[str, Any],
) -> dict[str, Any]:
    """Resolve 180-degree PCA ambiguity with a common +X half-plane rule."""
    homography = np.asarray(
        scene_calibration["metric_reference"]["image_to_task_homography"],
        dtype=np.float64,
    )
    detector = scene_calibration["detectors"]["doll"]
    frame_rows: list[dict[str, Any]] = []
    doubled: list[complex] = []
    for observation in measurement["observations"]:
        frame = int(observation["frame"])
        image_path = Path(observation["image"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        points, pixels = source_doll_mask_points_task(image_path, detector, homography)
        centered = points - np.median(points, axis=0)
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        major = eigenvectors[:, order[0]]
        if major[0] < 0.0 or (abs(float(major[0])) < 1e-12 and major[1] < 0.0):
            major = -major
        yaw = math.atan2(float(major[1]), float(major[0]))
        ratio = float(eigenvalues[order[0]] / max(eigenvalues[order[1]], 1e-15))
        doubled.append(complex(math.cos(2.0 * yaw), math.sin(2.0 * yaw)))
        frame_rows.append(
            {
                "frame": frame,
                "image": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "mask_pixels": pixels,
                "major_axis_task_xy": major.tolist(),
                "yaw_deg_modulo_180": math.degrees(yaw),
                "pca_eigenvalue_ratio": ratio,
            }
        )
    mean_double = sum(doubled) / len(doubled)
    if abs(mean_double) < 1e-12:
        raise RuntimeError(f"source object yaw is degenerate for {source}")
    yaw = 0.5 * math.atan2(mean_double.imag, mean_double.real)
    # One shared deterministic ambiguity rule: the local major axis has +world-X.
    if math.cos(yaw) < 0.0:
        yaw += math.pi
    if yaw > math.pi / 2.0:
        yaw -= math.pi
    concentration = float(abs(mean_double))
    minimum_ratio = min(float(row["pca_eigenvalue_ratio"]) for row in frame_rows)
    return {
        "yaw_rad": yaw,
        "yaw_deg": math.degrees(yaw),
        "quaternion_xyzw": Rotation.from_euler("z", yaw).as_quat().tolist(),
        "method": "source_cam_high_green_mask_task_plane_PCA_median_double_angle",
        "ambiguity_rule": "major axis direction selected in the +world-X half-plane",
        "frame_consistency_double_angle_concentration": concentration,
        "minimum_pca_eigenvalue_ratio": minimum_ratio,
        "confidence": (
            "HIGH" if concentration >= 0.95 and minimum_ratio >= 1.20
            else "MEDIUM" if concentration >= 0.80 and minimum_ratio >= 1.08
            else "LOW"
        ),
        "frames": frame_rows,
    }


def load_source_state(source: str) -> tuple[np.ndarray, Path]:
    parquets = sorted((RAW / source / "data").rglob("*.parquet"))
    if len(parquets) != 1:
        raise RuntimeError(f"expected one source parquet for {source}, got {len(parquets)}")
    # Direct ParquetFile access avoids PyArrow dataset discovery (and its
    # unrelated optional Pandas import) while reading the exact same column.
    table = pq.ParquetFile(parquets[0]).read(columns=["observation.state"])
    state = fixed_list_numpy(table["observation.state"], 14).astype(np.float64)
    if not np.isfinite(state).all():
        raise RuntimeError(f"non-finite source state: {source}")
    return state, parquets[0]


def robust_source_grasp_evidence(
    source: str,
    event_frames: dict[str, Any],
    aloha: ALOHAKinematics,
) -> dict[str, Any]:
    state, parquet = load_source_state(source)
    onset = int(event_frames["LEFT_CLOSE_ONSET"])
    stable = int(event_frames["LEFT_STABLE_HOLD"])
    release = int(event_frames["LEFT_RELEASE"])
    if not 0 <= onset < stable < release <= len(state):
        raise RuntimeError(f"invalid source LEFT grasp chronology for {source}")
    # Frozen common interval: first 0.5 s after stable hold, bounded by release.
    end = min(len(state) - 1, release - 1, stable + 14)
    if end - stable + 1 < 6:
        raise RuntimeError(f"source stable grasp window too short for {source}")
    fk = aloha.fk(state)
    positions = fk["left_tcp_position_world"][stable : end + 1]
    rotations = fk["left_tcp_rotation_world"][stable : end + 1]
    median_position = np.median(positions, axis=0)
    mean_rotation = Rotation.from_matrix(rotations).mean().as_matrix()
    gripper = state[stable : end + 1, 6]
    lift_delta = float(np.max(positions[:, 2]) - positions[0, 2])
    return {
        "close_onset_frame": onset,
        "stable_grasp_start_frame": stable,
        "stable_grasp_end_frame": end,
        "window_frame_count": end - stable + 1,
        "source_left_tool_pose_robust": {
            "position_xyz_m": median_position.tolist(),
            "quaternion_xyzw": quat_xyzw(mean_rotation),
            "position_estimator": "componentwise median",
            "orientation_estimator": "scipy Rotation.mean",
        },
        "source_left_gripper_state_median_m": float(np.median(gripper)),
        "source_left_gripper_state_range_m": [float(np.min(gripper)), float(np.max(gripper))],
        "source_left_tool_z_range_m": [float(np.min(positions[:, 2])), float(np.max(positions[:, 2]))],
        "source_left_lift_delta_within_window_m": lift_delta,
        "ownership_evidence": "common pre-retargeting event chronology LEFT_STABLE_HOLD before LEFT_RELEASE",
        "evidence_source": str(parquet.resolve()),
        "evidence_source_sha256": sha256_file(parquet),
        "derivation_confidence": "HIGH",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eval35 = read_json(EVAL35)
    command_manifest = read_json(COMMANDS)
    intent_manifest = read_json(INTENTS)
    scene_calibration = read_json(SCENE_CALIBRATION)
    physics = read_json(PHYSICS_CONFIG)
    physical_environment = read_json(PHYSICAL_ENVIRONMENT)
    common_controller = read_json(COMMON_PHYSICAL_CONTROLLER)
    common = load_common_config(COMMON_CONFIG)
    scene = load_scene(common)
    aloha = ALOHAKinematics(common, scene)
    g1 = G1Kinematics(common, scene)
    palm = read_json(PALM_FRAME)
    lower, upper, joint_names = authoritative_joint_limits(read_json(JOINT_CONTRACT))
    primitive = Dex3Primitive.from_frozen_dependencies(
        physics, physical_environment, common_controller
    )

    entries = sorted(eval35["eval_entries"], key=lambda row: int(row["eval_index"]))
    if len(entries) != 35 or len({row["stable_episode_id"] for row in entries}) != 35:
        raise RuntimeError("EVAL35 membership is not exactly 35 unique episodes")
    command_rows = {
        (str(row["method"]), int(row["eval_index"])): row
        for row in command_manifest["records"]
    }
    expected_commands = {
        (f"ACT-{method}40", index) for method in ("A", "B") for index in range(35)
    }
    if set(command_rows) != expected_commands:
        raise RuntimeError("prepared command membership is not matched A/B EVAL35")
    intent_rows = {int(row["eval_index"]): row for row in intent_manifest["records"]}
    if set(intent_rows) != set(range(35)):
        raise RuntimeError("common source-intent membership is not EVAL35")
    measurements = source_measurements(entries)

    visual_dimensions = np.asarray(physics["object"]["visual_dimensions_m"], dtype=np.float64)
    collision_geometry = physics["geometry_candidates"][0]
    collision_dimensions = np.asarray(collision_geometry["dimensions_m"], dtype=np.float64)
    table_z = float(physics["object"]["table_surface_world_z_m"])
    object_z = table_z + visual_dimensions[2] / 2.0
    collision_z_offset = float((collision_dimensions[2] - visual_dimensions[2]) / 2.0)
    fixed_bin_xy = np.asarray(
        physical_environment["bin"]["opening_center_world_xy_m"], dtype=np.float64
    )
    fixed_bin_opening = np.asarray(
        physical_environment["bin"]["opening_dimensions_xy_m"], dtype=np.float64
    )
    distance_model = DistanceModel(g1, collision_dimensions, table_z, joint_names)

    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for entry in entries:
        index = int(entry["eval_index"])
        stable_id = str(entry["stable_episode_id"])
        source = source_name(entry, scene_calibration)
        measurement = measurements[source]
        xy = np.asarray(measurement["doll_initial_center_task_xy_m"], dtype=np.float64)
        yaw = episode_visual_yaw(source, measurement, scene_calibration)
        target_pose = pose_matrix([*xy, object_z], yaw["quaternion_xyzw"])
        grasp = robust_source_grasp_evidence(
            source, intent_rows[index]["event_frames"], aloha
        )
        methods: dict[str, Any] = {}
        for method in ("ACT-A40", "ACT-B40"):
            command_row = command_rows[(method, index)]
            if str(command_row["stable_episode_id"]) != stable_id:
                raise RuntimeError(f"command/source identity mismatch at {index}")
            conversion_path = Path(
                intent_rows[index]["source_artifacts"][0 if method == "ACT-A40" else 1]
            )
            conversion = command_registration(conversion_path)
            geometry = method_geometry(
                command_row,
                target_pose,
                target_pose,
                distance_model,
                g1,
                palm,
                primitive,
                lower,
                upper,
                joint_names,
                collision_z_offset,
            )
            release_xy = np.asarray(
                conversion["right_release_interaction_target_world_xyz_m"][:2],
                dtype=np.float64,
            )
            delta_bin = release_xy - fixed_bin_xy
            methods[method] = {
                "command_path": str(Path(command_row["physical_command"]).resolve()),
                "command_sha256": command_row["physical_command_sha256"],
                "conversion_registration": conversion,
                "initial_left_palm_to_registered_doll": geometry[
                    "initial_left_palm_to_expected_doll"
                ],
                "minimum_left_surface_distance_during_close_hold_m": geometry[
                    "minimum_left_surface_distance_during_close_hold_m"
                ]["episode_matched_registration"],
                "fixed_bin_final_placement_consistency": {
                    "right_release_interaction_xy_m": release_xy.tolist(),
                    "frozen_bin_opening_center_xy_m": fixed_bin_xy.tolist(),
                    "delta_xy_m": delta_bin.tolist(),
                    "center_distance_m": float(np.linalg.norm(delta_bin)),
                    "inside_opening_footprint_by_center": bool(
                        np.all(np.abs(delta_bin) <= fixed_bin_opening / 2.0)
                    ),
                    "bin_was_moved_per_episode": False,
                },
            }
        pose = {
            "position_xyz_m": target_pose[:3, 3].tolist(),
            "quaternion_xyzw": yaw["quaternion_xyzw"],
        }
        registration = {
            "eval_index": index,
            "eval_number": index + 1,
            "stable_episode_id": stable_id,
            "source_recording": source,
            "provenance": entry["provenance"],
            "source_object_task_frame_source": {
                "priority": 2,
                "artifact": str(SCENE_CALIBRATION.resolve()),
                "artifact_sha256": sha256_file(SCENE_CALIBRATION),
                "position_method": "existing common source cam_high object artifact; median frames 0/10/20",
                "orientation_method": yaw["method"],
                "physical_A_B_outcomes_read": False,
                "ACT_outputs_used_for_object_pose": False,
            },
            "source_grasp_window": grasp,
            "source_object_pose": pose,
            "source_to_target_transform": {
                "translation_xyz_m": common["task_registration"][
                    "global_translation_correction_world_xyz_m"
                ],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "uniform_metric_scale": common["task_registration"]["uniform_metric_scale"],
                "source": common["task_registration"]["calibration_report"],
                "source_sha256": sha256_file(
                    Path(common["task_registration"]["calibration_report"])
                ),
                "established_convention": "identity metric task frame",
            },
            "target_object_pose": pose,
            "orientation_evidence": yaw,
            "runtime_initialization_rule": {
                "order": [
                    "sim.reset",
                    "load registration entry",
                    "write dynamic doll root pose once",
                    "verify root pose",
                    "control frame 0",
                ],
                "object_root_pose_writes_after_initialization": 0,
                "bin_pose_fixed": True,
                "G1_root_fixed": True,
                "table_pose_fixed": True,
            },
            "A_B_identical_object_pose": True,
            "A_B_translation_difference_mm": 0.0,
            "A_B_rotation_difference_deg": 0.0,
            "registration_confidence": (
                "HIGH" if yaw["confidence"] == "HIGH" else "MEDIUM"
            ),
            "methods": methods,
        }
        registration["entry_sha256"] = canonical_sha256(registration)
        rows.append(registration)
        csv_rows.append(
            {
                "eval_index": index,
                "eval_number": f"{index + 1:02d}",
                "stable_episode_id": stable_id,
                "source_recording": source,
                "close_onset": grasp["close_onset_frame"],
                "stable_grasp_start": grasp["stable_grasp_start_frame"],
                "stable_grasp_end": grasp["stable_grasp_end_frame"],
                "source_object_task_frame_source": str(SCENE_CALIBRATION.resolve()),
                "target_x_m": pose["position_xyz_m"][0],
                "target_y_m": pose["position_xyz_m"][1],
                "target_z_m": pose["position_xyz_m"][2],
                "target_qx": pose["quaternion_xyzw"][0],
                "target_qy": pose["quaternion_xyzw"][1],
                "target_qz": pose["quaternion_xyzw"][2],
                "target_qw": pose["quaternion_xyzw"][3],
                "A_command_path": methods["ACT-A40"]["command_path"],
                "B_command_path": methods["ACT-B40"]["command_path"],
                "A_B_pose_equal": "YES",
                "A_B_translation_difference_mm": 0.0,
                "A_B_rotation_difference_deg": 0.0,
                "registration_confidence": registration["registration_confidence"],
                "entry_sha256": registration["entry_sha256"],
            }
        )

    unique_poses = {
        canonical_sha256(row["target_object_pose"]) for row in rows
    }
    manifest = {
        "schema_version": "eval35_episode_source_derived_object_registration_v1",
        "status": "PASS",
        "created_before_post_registration_eval35": True,
        "physical_eval35_outcomes_read": False,
        "registration_semantics": "EPISODE_CONDITIONED_SOURCE_DERIVED_TASK_REGISTRATION",
        "EVAL35_count": len(rows),
        "unique_source_episode_count": len({row["source_recording"] for row in rows}),
        "source_derived_count": len(rows),
        "matched_A_count": len(rows),
        "matched_B_count": len(rows),
        "A_B_identical_object_pose_count": sum(
            bool(row["A_B_identical_object_pose"]) for row in rows
        ),
        "maximum_A_B_translation_difference_mm": 0.0,
        "maximum_A_B_rotation_difference_deg": 0.0,
        "one_global_canonical_object_pose": False,
        "unique_episode_object_pose_count": len(unique_poses),
        "manual_episode_nudges": False,
        "policy_output_derived_object_placement": False,
        "bin_pose": {
            "fixed_for_all_episodes": True,
            "opening_center_world_xy_m": fixed_bin_xy.tolist(),
            "opening_dimensions_xy_m": fixed_bin_opening.tolist(),
            "bottom_world_z_m": physical_environment["bin"]["bottom_world_z_m"],
            "external_height_m": physical_environment["bin"]["external_height_m"],
        },
        "source_orientation_rule": {
            "method": "same frozen source image masks and task-plane homography; per-episode PCA over frames 0/10/20",
            "proposed_B_axes_used": False,
            "ACT_outputs_used": False,
            "rollout_outcomes_used": False,
            "manual_adjustment": False,
        },
        "visual_collision_audit": {
            "visual_dimensions_m": visual_dimensions.tolist(),
            "collision_dimensions_m": collision_dimensions.tolist(),
            "visual_minus_collision_half_extent_m": (
                0.5 * (visual_dimensions - collision_dimensions)
            ).tolist(),
            "visual_contact_can_occur_outside_collider": bool(
                np.any(visual_dimensions > collision_dimensions)
            ),
        },
        "authoritative_inputs": {
            str(path.resolve()): sha256_file(path)
            for path in (
                EVAL35,
                COMMANDS,
                INTENTS,
                SCENE_CALIBRATION,
                NEW_SOURCE_ESTIMATES,
                COMMON_CONFIG,
                JOINT_CONTRACT,
                PALM_FRAME,
                PHYSICS_CONFIG,
                PHYSICAL_ENVIRONMENT,
                COMMON_PHYSICAL_CONTROLLER,
            )
        },
        "entries": rows,
    }
    manifest["registration_content_sha256"] = canonical_sha256(manifest)
    json_path = OUT / "EVAL35_EPISODE_OBJECT_REGISTRATION.json"
    csv_path = OUT / "EVAL35_EPISODE_OBJECT_REGISTRATION.csv"
    md_path = OUT / "EVAL35_EPISODE_OBJECT_REGISTRATION.md"
    atomic_json(json_path, manifest)
    atomic_csv(csv_path, csv_rows, list(csv_rows[0]))

    half = manifest["visual_collision_audit"]["visual_minus_collision_half_extent_m"]
    yaw_low = sum(row["orientation_evidence"]["confidence"] == "LOW" for row in rows)
    md_path.write_text(
        "# EVAL35 episode-conditioned object registration\n\n"
        "Status: **PASS**\n\n"
        f"- EVAL35 entries: **{len(rows)}/35**\n"
        f"- Unique source recordings: **{manifest['unique_source_episode_count']}/35**\n"
        f"- Source-derived object poses: **{manifest['source_derived_count']}/35**\n"
        f"- A/B identical matched object poses: **{manifest['A_B_identical_object_pose_count']}/35**\n"
        "- Maximum matched A/B pose difference: **0.000000 mm / 0.000000 deg**\n"
        f"- Unique object poses: **{len(unique_poses)}** (one global canonical pose: **NO**)\n"
        "- Manual episode nudges: **NO**\n"
        "- Policy-output-derived placement: **NO**\n"
        "- Physical A/B outcomes read: **NO**\n"
        "- Bin/table/G1 pose episode-varying: **NO**\n"
        f"- Low-confidence source yaw entries: **{yaw_low}/35**\n\n"
        "Object XY is the existing pre-policy common cam_high source artifact. "
        "Episode yaw is the deterministic major axis of the same doll mask after "
        "the established task-plane homography; its 180-degree ambiguity is always "
        "resolved toward +world-X. The source and target task frames retain the "
        "project's frozen identity metric registration.\n\n"
        "## Visual/collision audit\n\n"
        f"- Visual dimensions: `{visual_dimensions.tolist()}` m\n"
        f"- Collision dimensions: `{collision_dimensions.tolist()}` m\n"
        f"- Visual minus collision half extent X/Y/Z: `{half}` m\n"
        "- Fingers may look visually touching while outside the collider: **YES**\n\n"
        "The fixed-bin placement diagnostics and all A/B palm/digit distance fields "
        "are retained per entry in the JSON manifest.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "entries": len(rows),
                "unique_sources": manifest["unique_source_episode_count"],
                "unique_object_poses": len(unique_poses),
                "A_B_identical": manifest["A_B_identical_object_pose_count"],
                "low_confidence_yaw": yaw_low,
                "registration_content_sha256": manifest["registration_content_sha256"],
                "output": str(json_path.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
