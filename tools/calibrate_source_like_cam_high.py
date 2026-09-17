#!/usr/bin/env python3
"""Fit the fixed external SOURCE_LIKE_CAM_HIGH to the real workspace rails.

Only camera parameters are fitted.  The tabletop, black frame, G1 root, doll,
and bin are read from the existing Doll-Handoff scene and never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_recordings"
LAYOUT_PATH = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
IMAGE_RELATIVE = Path(
    "images/observation.images.cam_high/episode_000000/frame_000000.png"
)
REFERENCE_EPISODE = "GoPark_20260820_152058"
AUDIT_EPISODES = (
    "GoPark_20260820_152058",
    "GoPark_20260820_152242",
    "GoPark_20260820_154342",
    "GoPark_20260820_155934",
    "GoPark_20260820_161731",
    "GoPark_20260820_163946",
    "GoPark_20260823_135848",
    "GoPark_20260823_140035",
)

# Existing one-time measurements on the median-aligned 640x480 source view.
# Order: tabletop inner opening lower-left, lower-right, upper-right, upper-left.
MEASURED_CORNERS_PX = np.asarray(
    [[122.0, 445.0], [526.0, 445.0], [470.0, 195.0], [170.0, 199.0]],
    dtype=np.float64,
)
MEASUREMENT_UNCERTAINTY_PX = 2.0
WIDTH = 640
HEIGHT = 480
# Isaac's camera sensor reports pixel-center intrinsics at (width/2, height/2).
# Use that same convention in the fit so the frozen JSON and sensor readback are
# numerically identical instead of carrying an unexplained half-pixel offset.
PRINCIPAL_POINT = np.asarray([WIDTH / 2.0, HEIGHT / 2.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photo", type=Path, action="append", default=[])
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_basis(yaw_deg: float, downward_pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return ROS camera-to-world rotation columns (right, down, forward)."""
    yaw = math.radians(yaw_deg)
    downward = math.radians(-downward_pitch_deg)
    forward = np.asarray(
        [
            math.sin(yaw) * math.cos(downward),
            math.cos(yaw) * math.cos(downward),
            -math.sin(downward),
        ]
    )
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    roll = math.radians(roll_deg)
    rolled_right = math.cos(roll) * right + math.sin(roll) * down
    rolled_down = -math.sin(roll) * right + math.cos(roll) * down
    return np.column_stack((rolled_right, rolled_down, forward))


def project(
    points_world: np.ndarray,
    camera_position: np.ndarray,
    yaw_deg: float,
    downward_pitch_deg: float,
    roll_deg: float,
    focal_px: float,
) -> np.ndarray:
    rotation = camera_basis(yaw_deg, downward_pitch_deg, roll_deg)
    camera_coordinates = (points_world - camera_position) @ rotation
    if np.any(camera_coordinates[:, 2] <= 0.0):
        raise ValueError("Workspace corner behind the fitted camera")
    return PRINCIPAL_POINT + focal_px * camera_coordinates[:, :2] / camera_coordinates[:, 2:3]


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def geometry_metrics(points: np.ndarray) -> dict[str, Any]:
    ll, lr, ur, ul = points
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    return {
        "center_px": np.mean(points, axis=0).tolist(),
        "bounding_box_xyxy_px": [*bbox_min.tolist(), *bbox_max.tolist()],
        "bounding_box_width_px": float(bbox_max[0] - bbox_min[0]),
        "bounding_box_height_px": float(bbox_max[1] - bbox_min[1]),
        "front_apparent_width_px": float(np.linalg.norm(lr - ll)),
        "back_apparent_width_px": float(np.linalg.norm(ur - ul)),
        "left_apparent_depth_px": float(np.linalg.norm(ul - ll)),
        "right_apparent_depth_px": float(np.linalg.norm(ur - lr)),
        "trapezoid_front_to_back_width_ratio": float(
            np.linalg.norm(lr - ll) / np.linalg.norm(ur - ul)
        ),
        "projected_area_px2": polygon_area(points),
    }


def estimate_static_view_stability(reference: np.ndarray) -> dict[str, Any]:
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    polygon = np.rint(MEASURED_CORNERS_PX).astype(np.int32)
    mask = np.zeros(gray_reference.shape, dtype=np.uint8)
    cv2.polylines(mask, [polygon], True, 255, 20, cv2.LINE_AA)
    window = cv2.createHanningWindow((WIDTH, HEIGHT), cv2.CV_32F)
    rail_weight = mask.astype(np.float32) / np.float32(255.0)
    reference_signal = cv2.Laplacian(gray_reference, cv2.CV_32F) * rail_weight
    rows = []
    for episode in AUDIT_EPISODES:
        path = RAW / episode / IMAGE_RELATIVE
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        if image.shape[:2] != (HEIGHT, WIDTH):
            raise RuntimeError(f"Unexpected source shape {image.shape}: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        signal = cv2.Laplacian(gray, cv2.CV_32F) * rail_weight
        shift, response = cv2.phaseCorrelate(reference_signal, signal, window)
        rows.append(
            {
                "episode": episode,
                "image": str(path.relative_to(ROOT)),
                "image_sha256": sha256_file(path),
                "translation_from_reference_px": [float(shift[0]), float(shift[1])],
                "phase_correlation_response": float(response),
            }
        )
    shifts = np.asarray([row["translation_from_reference_px"] for row in rows])
    norms = np.linalg.norm(shifts, axis=1)
    return {
        "method": "phase correlation of Laplacian rail-band pixels on early frame 0",
        "episodes": rows,
        "median_translation_px": np.median(shifts, axis=0).tolist(),
        "maximum_absolute_x_translation_px": float(np.max(np.abs(shifts[:, 0]))),
        "maximum_absolute_y_translation_px": float(np.max(np.abs(shifts[:, 1]))),
        "maximum_translation_norm_px": float(np.max(norms)),
        "source_camera_static_within_corner_measurement_uncertainty": bool(
            np.max(norms) <= MEASUREMENT_UNCERTAINTY_PX
        ),
    }


def annotate_source(image: np.ndarray, projected: np.ndarray, output: Path) -> None:
    canvas = image.copy()
    source_poly = np.rint(MEASURED_CORNERS_PX).astype(np.int32)
    projected_poly = np.rint(projected).astype(np.int32)
    cv2.polylines(canvas, [source_poly], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(canvas, [projected_poly], True, (255, 0, 255), 2, cv2.LINE_AA)
    labels = ("LL", "LR", "UR", "UL")
    for label, measured, fitted in zip(labels, source_poly, projected_poly, strict=True):
        cv2.circle(canvas, tuple(measured), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(canvas, tuple(fitted), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(
            canvas,
            label,
            (int(measured[0]) + 5, int(measured[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, "yellow=source measurement  magenta=fitted pinhole projection", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Failed to write {output}")


def render_report(config: dict[str, Any], output: Path) -> None:
    fit = config["calibration"]
    pose = config["extrinsics"]
    intrinsics = config["intrinsics"]
    stability = config["source_reference"]["static_view_audit"]
    lines = [
        "# SOURCE_LIKE_CAM_HIGH calibration",
        "",
        "Status: **CAMERA_FIT_FROZEN**",
        "",
        "This fit changes only the camera. The Doll-Handoff table, black frame, G1 root, doll, and bin remain the scene-authored values. Robot motion and Policy-B behavior were not part of the objective.",
        "",
        "## Evidence and constraints",
        "",
        "- Camera: fixed external `cam_high`, in front of the table and on its left/right centerline.",
        "- Optical-center height: tabletop + 1.04 m (fixed, not optimized).",
        "- Source pixel target: four inner black-workspace corners measured in the real 640×480 ALOHA `cam_high` view.",
        f"- Static-view audit: {len(stability['episodes'])} recordings; maximum registered shift {stability['maximum_translation_norm_px']:.3f} px, within the documented {MEASUREMENT_UNCERTAINTY_PX:.1f} px corner-measurement uncertainty.",
        "- Attached installation photos were used only to corroborate a fixed, downward-oblique external mount; pitch and FOV were not read visually from the photos.",
        "- No authoritative intrinsics/extrinsics were found in the recording metadata or recorder configuration. Resolution is authoritative; a centered, square-pixel pinhole focal length is therefore estimated.",
        "",
        "## Frozen camera",
        "",
        f"- World position (m): `{pose['position_world_xyz_m']}`",
        f"- Front distance from tabletop front edge: {pose['front_distance_from_table_m']:.6f} m",
        f"- Lateral offset: {pose['lateral_offset_from_table_centerline_m']:.6f} m",
        f"- Downward pitch: {pose['pitch_deg']:.6f}°",
        f"- Yaw: {pose['yaw_deg']:.6f}°",
        f"- Roll: {pose['roll_deg']:.6f}°",
        f"- Focal length: {intrinsics['fx_px']:.6f} px",
        f"- Horizontal / vertical FOV: {intrinsics['horizontal_fov_deg']:.6f}° / {intrinsics['vertical_fov_deg']:.6f}°",
        f"- Resolution: {intrinsics['width_px']}×{intrinsics['height_px']}; crop: none",
        "",
        "## Reprojection result",
        "",
        f"- Corner component RMSE: {fit['corner_component_rmse_px']:.6f} px",
        f"- Corner Euclidean RMSE: {fit['corner_euclidean_rmse_px']:.6f} px",
        f"- Normalized corner RMSE (800 px image diagonal): {fit['normalized_corner_rmse']:.8f}",
        f"- Maximum corner error: {fit['maximum_corner_error_px']:.6f} px",
        "",
        "The non-zero residual is retained rather than relaxing physical constraints. It reflects the centered pinhole/no-distortion approximation, rail-corner measurement uncertainty, and the fixed height/centerline/front-side constraints.",
        "",
        "`isaac_calibrated.png` and the final source/Isaac overlay are generated by the Isaac render verification using this frozen configuration.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    table = layout["table"]
    table_width, table_depth = [float(value) for value in table["size_xy_m"]]
    tabletop_z = float(table["surface_height_m"])
    world_points = np.asarray(
        [
            [0.0, 0.0, tabletop_z],
            [table_width, 0.0, tabletop_z],
            [table_width, table_depth, tabletop_z],
            [0.0, table_depth, tabletop_z],
        ],
        dtype=np.float64,
    )
    center_x = table_width / 2.0
    fixed_z = tabletop_z + 1.04

    def residual(parameters: np.ndarray) -> np.ndarray:
        camera_y, yaw, pitch, roll, focal = parameters
        predicted = project(
            world_points,
            np.asarray([center_x, camera_y, fixed_z]),
            yaw,
            pitch,
            roll,
            focal,
        )
        return (predicted - MEASURED_CORNERS_PX).ravel()

    solution = least_squares(
        residual,
        x0=np.asarray([-0.35, 0.0, -50.0, 0.0, 500.0]),
        bounds=(
            np.asarray([-3.0, -15.0, -89.0, -15.0, 100.0]),
            np.asarray([-0.05, 15.0, -10.0, 15.0, 3000.0]),
        ),
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        max_nfev=100_000,
    )
    camera_y, yaw, pitch, roll, focal = [float(value) for value in solution.x]
    camera_position = np.asarray([center_x, camera_y, fixed_z])
    predicted = project(world_points, camera_position, yaw, pitch, roll, focal)
    component_error = predicted - MEASURED_CORNERS_PX
    corner_error = np.linalg.norm(component_error, axis=1)
    rotation = camera_basis(yaw, pitch, roll)
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    horizontal_fov = math.degrees(2.0 * math.atan(WIDTH / (2.0 * focal)))
    vertical_fov = math.degrees(2.0 * math.atan(HEIGHT / (2.0 * focal)))
    reference_path = RAW / REFERENCE_EPISODE / IMAGE_RELATIVE
    reference = cv2.imread(str(reference_path))
    if reference is None:
        raise FileNotFoundError(reference_path)
    stability = estimate_static_view_stability(reference)
    annotate_source(reference, predicted, output / "source_reference.png")
    photos = []
    for index, photo in enumerate(args.photo, start=1):
        photo = photo.resolve()
        if not photo.is_file():
            raise FileNotFoundError(photo)
        destination = output / f"physical_mount_reference_{index}.png"
        shutil.copy2(photo, destination)
        photos.append(
            {
                "input_path": str(photo),
                "archived_path": str(destination),
                "sha256": sha256_file(destination),
                "use": "qualitative fixed external/downward-oblique mounting evidence only",
            }
        )
    config = {
        "schema_version": "source_like_cam_high_v1",
        "name": "SOURCE_LIKE_CAM_HIGH",
        "status": "FROZEN_FOR_POLICY_B_ISAAC_VALIDATION",
        "coordinate_convention": {
            "world": layout["coordinate_frame"],
            "camera_axes": "ROS/OpenCV: +X right, +Y down, +Z optical forward",
        },
        "physical_constraints": {
            "external_fixed_camera": True,
            "front_side_of_table": True,
            "table_centerline": True,
            "height_above_tabletop_m": 1.04,
            "height_optimized": False,
            "lateral_offset_optimized": False,
        },
        "extrinsics": {
            "position_world_xyz_m": camera_position.tolist(),
            "position_task_frame_xyz_m": [center_x, camera_y, 1.04],
            "height_above_tabletop_m": 1.04,
            "front_distance_from_table_m": -camera_y,
            "lateral_offset_from_table_centerline_m": 0.0,
            "pitch_deg": pitch,
            "pitch_definition": "negative means downward elevation from world horizontal",
            "yaw_deg": yaw,
            "roll_deg": roll,
            "rotation_camera_ros_to_world_matrix": rotation.tolist(),
            "orientation_world_xyzw_ros_camera": quaternion_xyzw.tolist(),
            "orientation_world_wxyz_ros_camera": [
                float(quaternion_xyzw[3]),
                float(quaternion_xyzw[0]),
                float(quaternion_xyzw[1]),
                float(quaternion_xyzw[2]),
            ],
        },
        "intrinsics": {
            "model": "estimated centered square-pixel pinhole; no distortion; Isaac width/2,height/2 pixel-center convention",
            "authoritative_source_intrinsics_found": False,
            "width_px": WIDTH,
            "height_px": HEIGHT,
            "fx_px": focal,
            "fy_px": focal,
            "cx_px": float(PRINCIPAL_POINT[0]),
            "cy_px": float(PRINCIPAL_POINT[1]),
            "matrix": [
                [focal, 0.0, float(PRINCIPAL_POINT[0])],
                [0.0, focal, float(PRINCIPAL_POINT[1])],
                [0.0, 0.0, 1.0],
            ],
            "horizontal_fov_deg": horizontal_fov,
            "vertical_fov_deg": vertical_fov,
            "clipping_range_m": [0.01, 100.0],
            "crop": None,
        },
        "source_reference": {
            "reference_image": str(reference_path.relative_to(ROOT)),
            "reference_image_sha256": sha256_file(reference_path),
            "corner_order": ["lower_left", "lower_right", "upper_right", "upper_left"],
            "measured_workspace_inner_corners_px": MEASURED_CORNERS_PX.tolist(),
            "measurement_uncertainty_px": MEASUREMENT_UNCERTAINTY_PX,
            "static_view_audit": stability,
            "attached_physical_mount_photos": photos,
        },
        "calibration": {
            "objective": "least-squares black-frame inner-corner pixel reprojection only",
            "optimized_parameters": [
                "front_distance_from_table",
                "pitch",
                "yaw",
                "roll",
                "common_focal_length_px",
            ],
            "forbidden_objective_inputs_used": [],
            "optimizer_success": bool(solution.success),
            "optimizer_message": solution.message,
            "optimizer_cost": float(solution.cost),
            "workspace_corners_world_xyz_m": world_points.tolist(),
            "projected_workspace_inner_corners_px": predicted.tolist(),
            "component_errors_px": component_error.tolist(),
            "corner_errors_px": corner_error.tolist(),
            "corner_component_rmse_px": float(np.sqrt(np.mean(np.square(component_error)))),
            "corner_euclidean_rmse_px": float(np.sqrt(np.mean(np.square(corner_error)))),
            "normalized_corner_rmse": float(
                np.sqrt(np.mean(np.square(corner_error))) / math.hypot(WIDTH, HEIGHT)
            ),
            "maximum_corner_error_px": float(np.max(corner_error)),
            "source_geometry": geometry_metrics(MEASURED_CORNERS_PX),
            "projected_geometry": geometry_metrics(predicted),
        },
        "freeze_rules": {
            "camera_moved_during_stages_0_to_5": False,
            "policy_input_camera": "SOURCE_LIKE_CAM_HIGH only",
            "g1_head_camera_used": False,
        },
    }
    config_path = output / "source_like_cam_high.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_report(config, output / "camera_calibration_report.md")
    print(
        json.dumps(
            {
                "status": config["status"],
                "position_world_xyz_m": config["extrinsics"]["position_world_xyz_m"],
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "roll_deg": roll,
                "focal_length_px": focal,
                "horizontal_fov_deg": horizontal_fov,
                "corner_euclidean_rmse_px": config["calibration"]["corner_euclidean_rmse_px"],
                "normalized_corner_rmse": config["calibration"]["normalized_corner_rmse"],
                "static_view_max_shift_px": stability["maximum_translation_norm_px"],
            },
            indent=2,
        )
    )
    if not solution.success or not stability[
        "source_camera_static_within_corner_measurement_uncertainty"
    ]:
        raise RuntimeError("Camera fit/source-static audit failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
