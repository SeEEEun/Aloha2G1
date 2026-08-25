#!/usr/bin/env python3
"""Shared, fail-closed deployment-camera configuration utilities.

The module supports the historical SOURCE_LIKE_CAM_HIGH diagnostic JSON and
the helmet-D455 schema.  It deliberately refuses unresolved helmet presets for
rendering, rollout, or live inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PENDING_MOUNT = "CAMERA_NOT_MOUNTED"
PENDING_EXTRINSIC = "EXTRINSIC_NOT_CALIBRATED"
FINAL_STATUS = "FROZEN_HELMET_D455_FINAL"
LEGACY_DIAGNOSTIC_STATUS = "FROZEN_FOR_POLICY_B_ISAAC_VALIDATION"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix(value: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite {shape} matrix")
    return array


def _pose_matrix(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    quaternion_array = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position_array.shape != (3,) or quaternion_array.shape != (4,):
        raise ValueError("camera position/quaternion shape is invalid")
    if not np.isclose(np.linalg.norm(quaternion_array), 1.0, atol=1e-5):
        raise ValueError("camera quaternion is not unit length")
    x, y, z, w = quaternion_array / np.linalg.norm(quaternion_array)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position_array
    return result


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Numerically stable unit quaternion conversion without a SciPy runtime dependency."""

    matrix = _matrix(rotation, (3, 3), "rotation")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray([0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale])
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray([(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray([(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    return quaternion / np.linalg.norm(quaternion)


@dataclass(frozen=True)
class DeploymentCamera:
    path: Path
    file_sha256: str
    schema_version: str
    name: str
    status: str
    width: int
    height: int
    intrinsic_matrix: np.ndarray
    distortion_model: str
    distortion_coefficients: np.ndarray
    clipping_range_m: tuple[float, float]
    world_from_camera: np.ndarray
    task_from_camera: np.ndarray
    mount: dict[str, Any]
    raw: dict[str, Any]

    @property
    def position_world_xyz_m(self) -> np.ndarray:
        return self.world_from_camera[:3, 3].copy()

    @property
    def orientation_world_xyzw_ros_camera(self) -> np.ndarray:
        return _matrix_to_quaternion_xyzw(self.world_from_camera[:3, :3])

    @property
    def is_final_helmet(self) -> bool:
        return self.status == FINAL_STATUS

    @property
    def resolved(self) -> bool:
        return self.status not in {
            PENDING_MOUNT,
            PENDING_EXTRINSIC,
            "HELMET_D455_FINAL_PENDING",
        } and self.mount.get("status") != PENDING_MOUNT


def _validate_parent_contract(mount: dict[str, Any]) -> None:
    movable = mount.get("parent_link_can_move")
    if movable is True:
        locked = mount.get("locked_parent_pose_id")
        fk = mount.get("parent_link_fk_provider")
        if locked in (None, "", PENDING_MOUNT) and fk in (None, "", PENDING_MOUNT):
            raise RuntimeError(
                "movable camera parent requires a locked parent pose or parent-link FK provider"
            )


def load_camera_config(
    path: str | Path,
    *,
    purpose: str,
    allow_pending: bool = False,
) -> DeploymentCamera:
    """Load a camera JSON for one named consumer.

    ``allow_pending`` is restricted to audit/dry-run tools.  Any operational
    consumer must leave it false and therefore cannot accidentally turn the
    pending helmet template into a camera pose.
    """

    resolved = Path(path).resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    schema = str(raw.get("schema_version", ""))
    if schema == "camera_preset_pointer_v1":
        target = Path(str(raw["camera_config"]))
        if not target.is_absolute():
            candidates = [Path.cwd() / target]
            candidates.extend(parent / target for parent in resolved.parents)
            target = next((item for item in candidates if item.is_file()), candidates[0])
        return load_camera_config(target, purpose=purpose, allow_pending=allow_pending)
    name = str(raw.get("name", ""))
    status = str(raw.get("status", ""))
    if not name:
        raise ValueError(f"camera config has no name: {resolved}")

    if schema == "source_like_cam_high_v1":
        if status != LEGACY_DIAGNOSTIC_STATUS:
            raise RuntimeError("legacy SOURCE_LIKE_CAM_HIGH config is not frozen")
        intrinsics = raw["intrinsics"]
        intrinsic_matrix = _matrix(intrinsics["matrix"], (3, 3), "intrinsics.matrix")
        width = int(intrinsics["width_px"])
        height = int(intrinsics["height_px"])
        world_from_camera = _pose_matrix(
            raw["extrinsics"]["position_world_xyz_m"],
            raw["extrinsics"]["orientation_world_xyzw_ros_camera"],
        )
        task_from_world = np.eye(4)
        task_from_world[2, 3] = -float(
            raw["calibration"]["workspace_corners_world_xyz_m"][0][2]
        )
        task_from_camera = task_from_world @ world_from_camera
        mount = {
            "type": "external_fixed_diagnostic",
            "parent_link": None,
            "parent_link_can_move": False,
            "diagnostic_only": True,
        }
        distortion_model = "none"
        distortion = np.zeros(5, dtype=np.float64)
        clipping = tuple(map(float, intrinsics.get("clipping_range_m", [0.01, 100.0])))
    elif schema == "helmet_d455_camera_config_v1":
        unresolved = (
            status in {PENDING_MOUNT, PENDING_EXTRINSIC, "HELMET_D455_FINAL_PENDING"}
            or raw.get("mount", {}).get("status") == PENDING_MOUNT
            or raw.get("extrinsics", {}).get("status") == PENDING_EXTRINSIC
        )
        if unresolved and not allow_pending:
            raise RuntimeError(
                f"{purpose}: FINAL_CAMERA_NOT_CALIBRATED ({name}, {status}); "
                "physical D455 capture/solve/validation is required"
            )
        if unresolved:
            # A structurally useful object for audit-only callers.  Identity
            # values are explicit non-calibration placeholders and must never
            # escape into operational consumers because of the guard above.
            stream = raw["streams"]["color"]
            width, height = int(stream["width_px"]), int(stream["height_px"])
            intrinsic_matrix = np.eye(3, dtype=np.float64)
            distortion_model = "unresolved"
            distortion = np.zeros(5, dtype=np.float64)
            world_from_camera = np.eye(4, dtype=np.float64)
            task_from_camera = np.eye(4, dtype=np.float64)
            clipping = tuple(map(float, raw.get("rendering", {}).get("clipping_range_m", [0.01, 10.0])))
            mount = dict(raw.get("mount", {}))
        else:
            if status not in {FINAL_STATUS, "SOLVED_PENDING_VALIDATION"}:
                raise RuntimeError(f"unsupported helmet camera status: {status}")
            color = raw["intrinsics"]["color"]
            intrinsic_matrix = _matrix(color["matrix"], (3, 3), "intrinsics.color.matrix")
            width = int(color["width_px"])
            height = int(color["height_px"])
            distortion_model = str(color.get("distortion_model", "none"))
            distortion = np.asarray(color.get("distortion_coefficients", []), dtype=np.float64)
            if distortion.ndim != 1 or not np.isfinite(distortion).all():
                raise ValueError("color distortion coefficients must be a finite vector")
            task_from_camera = _matrix(
                raw["extrinsics"]["task_from_camera_matrix"],
                (4, 4),
                "extrinsics.task_from_camera_matrix",
            )
            world_from_task = _matrix(
                raw["coordinate_frames"]["world_from_task_matrix"],
                (4, 4),
                "coordinate_frames.world_from_task_matrix",
            )
            world_from_camera = world_from_task @ task_from_camera
            clipping = tuple(map(float, raw["rendering"]["clipping_range_m"]))
            mount = dict(raw["mount"])
            _validate_parent_contract(mount)
    else:
        raise ValueError(f"unsupported camera schema {schema!r}: {resolved}")

    if width <= 0 or height <= 0 or len(clipping) != 2 or clipping[0] <= 0 or clipping[1] <= clipping[0]:
        raise ValueError("camera dimensions or clipping range are invalid")
    if not allow_pending and (width, height) != (640, 480):
        raise RuntimeError(
            f"{purpose}: current SmolVLA contract requires 640x480 RGB, got {width}x{height}"
        )
    return DeploymentCamera(
        path=resolved,
        file_sha256=sha256_file(resolved),
        schema_version=schema,
        name=name,
        status=status,
        width=width,
        height=height,
        intrinsic_matrix=intrinsic_matrix,
        distortion_model=distortion_model,
        distortion_coefficients=distortion,
        clipping_range_m=(float(clipping[0]), float(clipping[1])),
        world_from_camera=world_from_camera,
        task_from_camera=task_from_camera,
        mount=mount,
        raw=raw,
    )


def apply_configured_distortion(rgb_ideal: np.ndarray, camera: DeploymentCamera) -> np.ndarray:
    """Map an ideal pinhole render into the configured OpenCV color model."""

    image = np.asarray(rgb_ideal)
    if image.shape != (camera.height, camera.width, 3) or image.dtype != np.uint8:
        raise ValueError("rendered RGB shape/dtype does not match camera config")
    coefficients = camera.distortion_coefficients
    if camera.distortion_model in {"none", "unresolved"} or not coefficients.size or np.allclose(coefficients, 0):
        return image.copy()
    if camera.distortion_model.lower() not in {
        "brown_conrady",
        "modified_brown_conrady",
        "inverse_brown_conrady",
        "opencv",
    }:
        raise RuntimeError(f"renderer does not support distortion model {camera.distortion_model!r}")
    yy, xx = np.indices((camera.height, camera.width), dtype=np.float32)
    distorted_pixels = np.column_stack((xx.reshape(-1), yy.reshape(-1))).reshape(-1, 1, 2)
    ideal_pixels = cv2.undistortPoints(
        distorted_pixels,
        camera.intrinsic_matrix,
        coefficients,
        P=camera.intrinsic_matrix,
    ).reshape(camera.height, camera.width, 2)
    return cv2.remap(
        image,
        ideal_pixels[..., 0].astype(np.float32),
        ideal_pixels[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def camera_manifest_record(camera: DeploymentCamera) -> dict[str, Any]:
    return {
        "name": camera.name,
        "status": camera.status,
        "schema_version": camera.schema_version,
        "config": str(camera.path),
        "config_sha256": camera.file_sha256,
        "width_px": camera.width,
        "height_px": camera.height,
        "intrinsic_matrix": camera.intrinsic_matrix.tolist(),
        "distortion_model": camera.distortion_model,
        "distortion_coefficients": camera.distortion_coefficients.tolist(),
        "position_world_xyz_m": camera.position_world_xyz_m.tolist(),
        "orientation_world_xyzw_ros_camera": camera.orientation_world_xyzw_ros_camera.tolist(),
        "mount": camera.mount,
    }


__all__ = [
    "DeploymentCamera",
    "FINAL_STATUS",
    "LEGACY_DIAGNOSTIC_STATUS",
    "PENDING_EXTRINSIC",
    "PENDING_MOUNT",
    "apply_configured_distortion",
    "camera_manifest_record",
    "load_camera_config",
    "sha256_file",
]
