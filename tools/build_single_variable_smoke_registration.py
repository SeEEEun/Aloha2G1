#!/usr/bin/env python3
"""Build the source-only common registration for TRAIN smoke episodes 0/24/49."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "outputs/doll_handoff_retargeting/scene_recalibration.json"
PHYSICAL = ROOT / "outputs/final_episode_registered_eval35/01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
OUT = ROOT / "outputs/single_variable_ab_common_execution/00_registration"
INDICES = (0, 24, 49)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_yaw(measurement: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    homography = np.asarray(
        calibration["metric_reference"]["image_to_task_homography"],
        dtype=np.float64,
    )
    detector = calibration["detectors"]["doll"]
    doubled: list[complex] = []
    evidence: list[dict[str, Any]] = []
    for observation in measurement["observations"]:
        image_path = Path(observation["image"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.asarray(detector["hsv_lower_opencv"], dtype=np.uint8),
            np.asarray(detector["hsv_upper_opencv"], dtype=np.uint8),
        )
        x0, y0, x1, y1 = map(int, detector["roi_xyxy_px"])
        keep = np.zeros_like(mask)
        keep[y0:y1, x0:x1] = 255
        ys, xs = np.where(cv2.bitwise_and(mask, keep) > 0)
        if len(xs) < 500:
            raise RuntimeError(f"insufficient doll pixels in {image_path}")
        pixels = np.column_stack((xs, ys)).astype(np.float32).reshape(1, -1, 2)
        points = cv2.perspectiveTransform(pixels, homography)[0].astype(np.float64)
        centered = points - np.median(points, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        if major[0] < 0.0 or (abs(float(major[0])) < 1e-12 and major[1] < 0.0):
            major = -major
        yaw = math.atan2(float(major[1]), float(major[0]))
        doubled.append(complex(math.cos(2.0 * yaw), math.sin(2.0 * yaw)))
        evidence.append(
            {
                "frame": int(observation["frame"]),
                "image": str(image_path.resolve()),
                "image_sha256": file_hash(image_path),
                "mask_pixels": int(len(xs)),
                "yaw_deg_modulo_180": math.degrees(yaw),
            }
        )
    mean = sum(doubled) / len(doubled)
    yaw = 0.5 * math.atan2(mean.imag, mean.real)
    if math.cos(yaw) < 0.0:
        yaw += math.pi
    if yaw > math.pi / 2.0:
        yaw -= math.pi
    return {
        "yaw_rad": yaw,
        "yaw_deg": math.degrees(yaw),
        "quaternion_xyzw": Rotation.from_euler("z", yaw).as_quat().tolist(),
        "method": "source_cam_high_doll_mask_task_plane_PCA_double_angle",
        "ambiguity_rule": "major axis selected in +world-X half-plane",
        "double_angle_concentration": float(abs(mean)),
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a common source-derived TRAIN-smoke task registration."
    )
    parser.add_argument(
        "--workspace-translation-m",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="one common target-task translation, applied before the representation switch",
    )
    parser.add_argument(
        "--workspace-yaw-deg",
        type=float,
        default=0.0,
        help="one common target-task yaw rotation in degrees",
    )
    parser.add_argument(
        "--workspace-rpy-deg",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "one common target-task XYZ extrinsic rotation in degrees; when "
            "provided, --workspace-yaw-deg must remain zero"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT / "COMMON_TASK_REGISTRATION_TRAIN_SMOKE.json",
    )
    args = parser.parse_args()
    calibration = json.loads(CALIBRATION.read_text())
    physical = json.loads(PHYSICAL.read_text())
    by_index = {int(row["episode_index"]): row for row in calibration["per_episode"]}
    table_z = float(physical["bin"]["bottom_world_z_m"])
    visual_height = float(physical["doll"]["visual_dimensions_m"][2])
    center_z = table_z + 0.5 * visual_height
    if args.workspace_rpy_deg is not None and abs(float(args.workspace_yaw_deg)) > 1e-12:
        parser.error("use either --workspace-rpy-deg or --workspace-yaw-deg, not both")
    workspace_rpy_deg = (
        np.asarray(args.workspace_rpy_deg, dtype=np.float64)
        if args.workspace_rpy_deg is not None
        else np.asarray([0.0, 0.0, args.workspace_yaw_deg], dtype=np.float64)
    )
    rotation = Rotation.from_euler("xyz", workspace_rpy_deg, degrees=True).as_matrix()
    translation = np.asarray(args.workspace_translation_m, dtype=np.float64)
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation
    source_bin_position = np.asarray(
        [*physical["bin"]["opening_center_world_xy_m"], physical["bin"]["bottom_world_z_m"]],
        dtype=np.float64,
    )
    target_bin_position = rotation @ source_bin_position + translation
    source_table_origin = np.asarray([0.0, 0.0, table_z], dtype=np.float64)
    target_table_origin = rotation @ source_table_origin + translation
    entries: list[dict[str, Any]] = []
    for index in INDICES:
        source = by_index[index]
        yaw = source_yaw(source, calibration)
        source_pose = {
            "position_xyz_m": [
                float(source["doll_initial_center_task_xy_m"][0]),
                float(source["doll_initial_center_task_xy_m"][1]),
                center_z,
            ],
            "quaternion_xyzw": yaw["quaternion_xyzw"],
        }
        source_position = np.asarray(source_pose["position_xyz_m"], dtype=np.float64)
        source_rotation = Rotation.from_quat(source_pose["quaternion_xyzw"]).as_matrix()
        target_pose = {
            "position_xyz_m": (rotation @ source_position + translation).tolist(),
            "quaternion_xyzw": Rotation.from_matrix(rotation @ source_rotation)
            .as_quat()
            .tolist(),
        }
        entry = {
            "episode_index": index,
            "source_name": str(source["source_name"]),
            "source_object_pose": source_pose,
            "source_object_pose_derivation": {
                "xy": "pre-policy cam_high task-plane doll-mask median",
                "z": "qualified table height plus half visual doll height",
                "orientation": yaw,
            },
            "source_task_frame": {
                "origin_world_xyz_m": [0.0, 0.0, table_z],
                "axes": "world XYZ; meters; right-handed",
            },
            "source_to_target_transform_matrix": transform_matrix.tolist(),
            # Keep the episode-derived registration separate from the one
            # common G1-workspace placement.  The former is consumed before
            # the representation switch; the latter rigidly moves the
            # completed WRIST or INTERACTION target and therefore cannot
            # change either representation's internal hand/object geometry.
            "pre_workspace_source_to_target_transform_matrix": np.eye(
                4, dtype=np.float64
            ).tolist(),
            "common_workspace_transform_matrix": transform_matrix.tolist(),
            "source_to_target_direction": "T_target_from_source",
            "target_object_pose": target_pose,
            "source_bin_pose": {
                "position_xyz_m": source_bin_position.tolist(),
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "target_bin_pose": {
                "position_xyz_m": target_bin_position.tolist(),
                "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
            },
            "source_table_task_origin_xyz_m": source_table_origin.tolist(),
            "target_table_task_origin_xyz_m": target_table_origin.tolist(),
            "matched_A_B_pose_equal": True,
            "manual_nudge": False,
            "policy_output_used": False,
        }
        entry["entry_sha256"] = canonical_hash(entry)
        entries.append(entry)
    payload = {
        "schema_version": "single_variable_common_task_registration_smoke_v1",
        "status": "PASS",
        "scope": "TRAIN_ONLY_SMOKE_0_24_49",
        "registration_semantics": "EPISODE_CONDITIONED_SOURCE_DERIVED_TASK_REGISTRATION",
        "transform_convention": "column homogeneous pose; p_target=T_target_from_source @ p_source",
        "rotation_convention": "right-handed rotation matrix; serialized object quaternion XYZW",
        "units": "meter_radian",
        "common_for_A_B": True,
        "one_global_object_pose": False,
        "manual_episode_nudges": False,
        "policy_output_derived": False,
        "common_workspace_registration": {
            "translation_xyz_m": translation.tolist(),
            "rotation_quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
            "rotation_rpy_deg": workspace_rpy_deg.tolist(),
            "rotation_yaw_deg": float(workspace_rpy_deg[2]),
            "rigid_SE3": True,
            "scale": 1.0,
            "applied_to": [
                "task_frame",
                "episode_object_pose",
                "bin_pose",
                "table_task_origin",
                "WRIST_targets",
                "INTERACTION_targets",
            ],
            "selected_from": "TRAIN_ONLY_SMOKE_0_24_49",
            "A_B_identical": True,
        },
        "entries": entries,
        "inputs": {
            "scene_recalibration": str(CALIBRATION),
            "scene_recalibration_sha256": file_hash(CALIBRATION),
            "qualified_physical_environment": str(PHYSICAL),
            "qualified_physical_environment_sha256": file_hash(PHYSICAL),
        },
    }
    payload["content_sha256"] = canonical_hash(payload)
    path = args.output.resolve()
    atomic_json(path, payload)
    lines = [
        "# Common task registration — TRAIN smoke",
        "",
        "Status: **PASS**",
        "",
        "This source-only manifest binds one episode-conditioned task/object frame before the representation switch, then applies one identical rigid G1-workspace transform to the completed A/B target frames before the shared solver.",
        "",
        "- Episodes: **0, 24, 49**",
        "- A/B registration equality: **3/3**",
        "- Units: **meters/radians**",
        "- Object quaternion serialization: **XYZW**",
        "- Transform direction: **target from source**",
        "- Manual nudges: **NO**",
        "- Policy/physical outcomes used: **NO**",
        f"- Common workspace translation: **{translation.tolist()} m**",
        f"- Common workspace RPY: **{workspace_rpy_deg.tolist()} deg**",
        "- Rigid scale: **1.0**",
        "",
        "| index | source | target XYZ (m) | target yaw (deg) | entry SHA256 |",
        "|---:|---|---|---:|---|",
    ]
    for row in entries:
        lines.append(
            f"| {row['episode_index']} | {row['source_name']} | "
            f"`{row['target_object_pose']['position_xyz_m']}` | "
            f"{row['source_object_pose_derivation']['orientation']['yaw_deg']:.6f} | "
            f"`{row['entry_sha256']}` |"
        )
    path.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(path)
    print(payload["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
