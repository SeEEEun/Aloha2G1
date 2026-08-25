"""Offline geometry and ChArUco helpers for the helmet D455 kit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transform(rotation: np.ndarray | None = None, translation: Iterable[float] | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        result[:3, :3] = rotation
    if translation is not None:
        translation = np.asarray(list(translation), dtype=np.float64)
        if translation.shape != (3,):
            raise ValueError("translation must have three values")
        result[:3, 3] = translation
    return result


def invert_transform(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def dictionary_from_spec(spec: dict[str, Any]) -> Any:
    name = str(spec["dictionary"])
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"OpenCV has no ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def board_from_spec(spec: dict[str, Any]) -> Any:
    dictionary = dictionary_from_spec(spec)
    squares_x = int(spec["squares_x"])
    squares_y = int(spec["squares_y"])
    square = float(spec["square_length_m"])
    marker = float(spec["marker_length_m"])
    if not (squares_x >= 3 and squares_y >= 3 and 0 < marker < square):
        raise ValueError("invalid ChArUco board dimensions")
    if hasattr(cv2.aruco, "CharucoBoard") and not hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square, marker, dictionary)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square, marker, dictionary)


def board_chessboard_corners(board: Any) -> np.ndarray:
    corners = getattr(board, "chessboardCorners", None)
    if corners is None and hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    result = np.asarray(corners, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise RuntimeError("OpenCV ChArUco board did not expose 3D chess corners")
    return result


def opencv_from_canonical_board(spec: dict[str, Any]) -> np.ndarray:
    """Map the documented bottom-left/up/out frame into OpenCV's board frame.

    OpenCV's generated/detected ChArUco object points use the outer top-left
    origin, +Y down the printed page, and +Z into the printed face.  The kit's
    physical frame is deliberately easier to measure: bottom-left, +Y up, +Z
    out toward the viewer.  This fixed transform makes that contract explicit
    and independent of which old/new ArUco detector API is installed.
    """

    height = int(spec["squares_y"]) * float(spec["square_length_m"])
    return transform(np.diag([1.0, -1.0, -1.0]), [0.0, height, 0.0])


def camera_parameters(camera_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    color = camera_config["intrinsics"]["color"]
    matrix = np.asarray(color["matrix"], dtype=np.float64)
    distortion = np.asarray(color.get("distortion_coefficients", []), dtype=np.float64)
    if matrix.shape != (3, 3) or distortion.ndim != 1:
        raise ValueError("malformed color intrinsics")
    return matrix, distortion


def detect_charuco_pose(
    image: np.ndarray,
    board_spec: dict[str, Any],
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> dict[str, Any]:
    """Detect a board and estimate ``camera_from_board``."""

    if image is None or image.ndim not in (2, 3):
        raise ValueError("image is empty")
    board = board_from_spec(board_spec)
    dictionary = dictionary_from_spec(board_spec)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if hasattr(cv2.aruco, "detectMarkers"):
        marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
    else:
        marker_corners, marker_ids, rejected = cv2.aruco.ArucoDetector(
            dictionary
        ).detectMarkers(gray)
    if marker_ids is None or len(marker_ids) < int(board_spec.get("minimum_markers", 6)):
        raise RuntimeError(f"insufficient ChArUco markers: {0 if marker_ids is None else len(marker_ids)}")
    if hasattr(cv2.aruco, "interpolateCornersCharuco"):
        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
            cameraMatrix=np.asarray(intrinsic_matrix, dtype=np.float64),
            distCoeffs=np.asarray(distortion_coefficients, dtype=np.float64),
        )
    else:
        parameters = cv2.aruco.CharucoParameters()
        parameters.cameraMatrix = np.asarray(intrinsic_matrix, dtype=np.float64)
        parameters.distCoeffs = np.asarray(distortion_coefficients, dtype=np.float64)
        detector = cv2.aruco.CharucoDetector(board, parameters)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray,
            markerCorners=marker_corners,
            markerIds=marker_ids,
        )
        count = 0 if charuco_ids is None else len(charuco_ids)
    minimum = int(board_spec.get("minimum_charuco_corners", 12))
    if charuco_ids is None or int(count) < minimum:
        raise RuntimeError(f"insufficient ChArUco corners: {int(count)} < {minimum}")
    object_points = board_chessboard_corners(board)[charuco_ids.reshape(-1)]
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        charuco_corners.reshape(-1, 2),
        np.asarray(intrinsic_matrix, dtype=np.float64),
        np.asarray(distortion_coefficients, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("ChArUco pose estimation failed")
    rotation, _ = cv2.Rodrigues(rvec)
    camera_from_opencv_board = transform(rotation, np.asarray(tvec).reshape(3))
    camera_from_board = camera_from_opencv_board @ opencv_from_canonical_board(board_spec)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsic_matrix,
        distortion_coefficients,
    )
    error = projected.reshape(-1, 2) - charuco_corners.reshape(-1, 2)
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
    return {
        "camera_from_board": camera_from_board,
        "camera_from_opencv_board": camera_from_opencv_board,
        "marker_count": int(len(marker_ids)),
        "charuco_corner_count": int(count),
        "charuco_ids": charuco_ids.reshape(-1).astype(int),
        "charuco_corners_px": charuco_corners.reshape(-1, 2),
        "pose_reprojection_rmse_px": float(np.sqrt(np.mean(np.square(error)))),
        "pose_reprojection_max_px": float(np.max(np.linalg.norm(error, axis=1))),
        "annotated": annotated,
        "rejected_marker_candidates": int(len(rejected)),
    }


def average_transforms(values: Iterable[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    transforms = np.asarray(list(values), dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or len(transforms) < 1:
        raise ValueError("at least one 4x4 transform is required")
    translations = transforms[:, :3, 3]
    rotations = Rotation.from_matrix(transforms[:, :3, :3])
    mean_rotation = rotations.mean()
    mean_translation = translations.mean(axis=0)
    result = transform(mean_rotation.as_matrix(), mean_translation)
    angular = (mean_rotation.inv() * rotations).magnitude()
    linear = np.linalg.norm(translations - mean_translation, axis=1)
    return result, {
        "sample_count": int(len(transforms)),
        "translation_std_m": translations.std(axis=0).tolist(),
        "translation_rms_spread_m": float(np.sqrt(np.mean(np.square(linear)))),
        "translation_max_spread_m": float(np.max(linear)),
        "rotation_rms_spread_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(angular))))),
        "rotation_max_spread_deg": float(np.rad2deg(np.max(angular))),
    }


def solve_task_from_camera(
    camera_from_board_samples: Iterable[np.ndarray],
    task_from_board: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    task_from_board = np.asarray(task_from_board, dtype=np.float64)
    if task_from_board.shape != (4, 4):
        raise ValueError("task_from_board must be 4x4")
    samples = [task_from_board @ invert_transform(value) for value in camera_from_board_samples]
    return average_transforms(samples)


def project_task_points(
    points_task_xyz: np.ndarray,
    task_from_camera: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_task_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("task points must have shape [N,3]")
    camera_from_task = invert_transform(np.asarray(task_from_camera, dtype=np.float64))
    points_camera = points @ camera_from_task[:3, :3].T + camera_from_task[:3, 3]
    if np.any(points_camera[:, 2] <= 0):
        raise RuntimeError("one or more validation points are behind the camera")
    rvec, _ = cv2.Rodrigues(camera_from_task[:3, :3])
    projected, _ = cv2.projectPoints(
        points,
        rvec,
        camera_from_task[:3, 3],
        np.asarray(intrinsic_matrix, dtype=np.float64),
        np.asarray(distortion_coefficients, dtype=np.float64),
    )
    return projected.reshape(-1, 2), points_camera[:, 2]


def detect_black_workspace_corners(image: np.ndarray) -> np.ndarray:
    """Return ordered [lower-left, lower-right, upper-right, upper-left] corners."""

    if image is None or image.ndim != 3:
        raise ValueError("black-frame validation expects a BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            area = abs(cv2.contourArea(polygon))
            if area > 0.05 * image.shape[0] * image.shape[1]:
                candidates.append((area, polygon.reshape(4, 2).astype(np.float64)))
    if not candidates:
        raise RuntimeError("no sufficiently large black workspace quadrilateral detected")
    points = max(candidates, key=lambda item: item[0])[1]
    # Split by vertical position, then horizontal position.
    top = points[np.argsort(points[:, 1])[:2]]
    bottom = points[np.argsort(points[:, 1])[2:]]
    upper_left, upper_right = top[np.argsort(top[:, 0])]
    lower_left, lower_right = bottom[np.argsort(bottom[:, 0])]
    return np.asarray([lower_left, lower_right, upper_right, upper_left])


def reprojection_metrics(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if expected.shape != observed.shape or expected.ndim != 2 or expected.shape[1] != 2:
        raise ValueError("reprojection arrays must both have shape [N,2]")
    error = observed - expected
    distance = np.linalg.norm(error, axis=1)
    return {
        "component_error_px": error.tolist(),
        "component_rmse_px": float(np.sqrt(np.mean(np.square(error)))),
        "euclidean_rmse_px": float(np.sqrt(np.mean(np.square(distance)))),
        "maximum_euclidean_error_px": float(np.max(distance)),
    }


def task_from_camera_fields(value: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(value, dtype=np.float64)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return {
        "task_from_camera_matrix": matrix.tolist(),
        "position_task_frame_xyz_m": matrix[:3, 3].tolist(),
        "orientation_task_xyzw_ros_camera": quaternion.tolist(),
        "rotation_camera_ros_to_task_matrix": matrix[:3, :3].tolist(),
    }


__all__ = [
    "atomic_json",
    "average_transforms",
    "board_chessboard_corners",
    "board_from_spec",
    "camera_parameters",
    "detect_black_workspace_corners",
    "detect_charuco_pose",
    "dictionary_from_spec",
    "invert_transform",
    "load_json",
    "opencv_from_canonical_board",
    "project_task_points",
    "reprojection_metrics",
    "sha256_file",
    "solve_task_from_camera",
    "task_from_camera_fields",
    "transform",
]
