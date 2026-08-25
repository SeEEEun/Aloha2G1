#!/usr/bin/env python3
"""Validate ChArUco consistency and black-workspace reprojection, then optionally freeze."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path
import sys

import cv2
import numpy as np

from deployment_camera_config import FINAL_STATUS, load_camera_config
from helmet_d455.calibration import (
    atomic_json,
    camera_parameters,
    detect_black_workspace_corners,
    load_json,
    project_task_points,
    reprojection_metrics,
    sha256_file,
)


def _mount_checks(mount: dict) -> list[str]:
    errors = []
    unresolved = {None, "", "CAMERA_NOT_MOUNTED", "EXTRINSIC_NOT_CALIBRATED"}
    if mount.get("status") in unresolved:
        errors.append("camera mount status is unresolved")
    if mount.get("parent_link") in unresolved:
        errors.append("camera parent link is unresolved")
    movable = mount.get("parent_link_can_move")
    if not isinstance(movable, bool):
        errors.append("parent_link_can_move must be measured and set to true or false")
    elif movable:
        locked = mount.get("locked_parent_pose_id") not in unresolved
        fk = mount.get("parent_link_fk_provider") not in unresolved
        if not (locked or fk):
            errors.append("movable parent requires locked_parent_pose_id or parent_link_fk_provider")
        if fk and mount.get("parent_from_camera_matrix") in unresolved:
            errors.append("parent-link FK runtime requires calibrated parent_from_camera_matrix")
    return errors


def _observed_corners(args: argparse.Namespace, capture: dict) -> tuple[np.ndarray, str, bool]:
    if args.observed_corners_json:
        value = load_json(args.observed_corners_json)
        if isinstance(value, dict):
            value = value.get("observed_corners_px", value.get("corners_px"))
        return np.asarray(value, dtype=np.float64), str(args.observed_corners_json.resolve()), False
    candidates = [args.black_image] if args.black_image else [Path(row) for row in capture.get("black_workspace_images", [])]
    if not candidates:
        raise RuntimeError("black-workspace validation requires --black-image or observed corners JSON")
    observations = []
    for candidate in candidates:
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        observations.append(detect_black_workspace_corners(image))
    return np.mean(np.stack(observations), axis=0), ",".join(str(path.resolve()) for path in candidates), True


def _match_unlabeled_quad(expected: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Assign automatically detected corners to known projected task corners."""

    candidates = [observed[list(order)] for order in permutations(range(4))]
    return min(candidates, key=lambda value: float(np.square(value - expected).sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--black-image", type=Path)
    parser.add_argument("--observed-corners-json", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--freeze",
        type=Path,
        help="write helmet_d455_final.json; refused for synthetic or incomplete calibration",
    )
    args = parser.parse_args()

    candidate = load_json(args.candidate)
    if candidate.get("status") != "SOLVED_PENDING_VALIDATION":
        raise RuntimeError("candidate status must be SOLVED_PENDING_VALIDATION")
    capture_path = args.capture_manifest or Path(candidate["calibration"]["charuco"]["capture_manifest"])
    capture = load_json(capture_path)
    intrinsic, distortion = camera_parameters(candidate)
    task_from_camera = np.asarray(candidate["extrinsics"]["task_from_camera_matrix"], dtype=np.float64)
    charuco = candidate["calibration"]["charuco"]
    charuco_checks = {
        "valid_frame_count": int(charuco["valid_frames"]),
        "minimum_valid_frame_count": int(charuco["minimum_valid_frames"]),
        "aggregate_pose_reprojection_rmse_px": float(charuco["aggregate_pose_reprojection_rmse_px"]),
        "maximum_allowed_pose_reprojection_rmse_px": float(charuco["maximum_pose_reprojection_rmse_px"]),
        "translation_max_spread_m": float(charuco["pose_spread"]["translation_max_spread_m"]),
        "maximum_allowed_translation_spread_m": float(charuco["maximum_translation_spread_m"]),
        "rotation_max_spread_deg": float(charuco["pose_spread"]["rotation_max_spread_deg"]),
        "maximum_allowed_rotation_spread_deg": float(charuco["maximum_rotation_spread_deg"]),
    }
    charuco_pass = (
        charuco_checks["valid_frame_count"] >= charuco_checks["minimum_valid_frame_count"]
        and charuco_checks["aggregate_pose_reprojection_rmse_px"] <= charuco_checks["maximum_allowed_pose_reprojection_rmse_px"]
        and charuco_checks["translation_max_spread_m"] <= charuco_checks["maximum_allowed_translation_spread_m"]
        and charuco_checks["rotation_max_spread_deg"] <= charuco_checks["maximum_allowed_rotation_spread_deg"]
    )

    black_spec = candidate["calibration"]["black_workspace_frame"]
    expected, depth = project_task_points(
        np.asarray(black_spec["corners_task_xyz_m"], dtype=np.float64),
        task_from_camera,
        intrinsic,
        distortion,
    )
    black_error = None
    black_source = None
    black_failure = None
    try:
        observed, black_source, unordered = _observed_corners(args, capture)
        if observed.shape != (4, 2):
            raise ValueError("observed black-workspace corners must have shape [4,2]")
        if unordered:
            observed = _match_unlabeled_quad(expected, observed)
        black_error = reprojection_metrics(expected, observed)
        black_pass = (
            black_error["euclidean_rmse_px"] <= float(black_spec["maximum_euclidean_rmse_px"])
            and black_error["maximum_euclidean_error_px"] <= float(black_spec["maximum_corner_error_px"])
        )
    except (RuntimeError, ValueError) as error:
        observed = None
        black_pass = False
        black_failure = str(error)

    mount_errors = _mount_checks(candidate["mount"])
    is_synthetic = bool(capture.get("synthetic") or charuco.get("synthetic"))
    physical_device = (
        "D455" in str(candidate["device"].get("product", "")).upper()
        and not str(candidate["device"].get("serial", "")).startswith("SYNTHETIC")
    )
    geometry_pass = charuco_pass and black_pass and not mount_errors
    status = (
        "PASS_SYNTHETIC_OFFLINE_ONLY_NOT_FREEZABLE"
        if geometry_pass and is_synthetic
        else "PASS_PHYSICAL_READY_TO_FREEZE"
        if geometry_pass and physical_device
        else "HARD_FAIL"
    )
    report = {
        "schema_version": "helmet_d455_calibration_validation_v1",
        "status": status,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256_file(args.candidate),
        "capture_manifest": str(capture_path.resolve()),
        "capture_manifest_sha256": sha256_file(capture_path),
        "synthetic": is_synthetic,
        "physical_d455_identity_confirmed": physical_device,
        "charuco": {"status": "PASS" if charuco_pass else "HARD_FAIL", **charuco_checks},
        "black_workspace_projection": {
            "status": "PASS" if black_pass else "HARD_FAIL",
            "expected_corners_px": expected.tolist(),
            "observed_corners_px": None if observed is None else observed.tolist(),
            "observed_source": black_source,
            "point_depths_camera_m": depth.tolist(),
            "metrics": black_error,
            "failure": black_failure,
            "thresholds": {
                "maximum_euclidean_rmse_px": float(black_spec["maximum_euclidean_rmse_px"]),
                "maximum_corner_error_px": float(black_spec["maximum_corner_error_px"]),
            },
        },
        "mount_parent_contract": {
            "status": "PASS" if not mount_errors else "HARD_FAIL",
            "errors": mount_errors,
            "requirement": "movable parent must use a named locked pose or timestamp-aligned FK plus calibrated parent_from_camera",
        },
        "freeze_permitted": bool(geometry_pass and physical_device and not is_synthetic),
    }
    atomic_json(args.report, report)

    if args.freeze:
        if args.freeze.name != "helmet_d455_final.json":
            raise RuntimeError("final frozen filename must be helmet_d455_final.json")
        if not report["freeze_permitted"]:
            raise RuntimeError(f"refusing to freeze: validation status is {status}")
        final = deepcopy(candidate)
        final.update(name="HELMET_D455_FINAL", status=FINAL_STATUS)
        final["extrinsics"]["status"] = FINAL_STATUS
        final["calibration"]["validation"] = {
            "report": str(args.report.resolve()),
            "report_sha256": sha256_file(args.report),
            "status": "PASS",
        }
        final["rendering"]["final_50_episode_render_allowed"] = True
        final["freeze"].update(
            {
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_candidate": str(args.candidate.resolve()),
                "source_candidate_sha256": sha256_file(args.candidate),
            }
        )
        atomic_json(args.freeze, final)
        # Exercise the same fail-closed loader used by render, rollout, and live inference.
        load_camera_config(args.freeze, purpose="post-freeze validation")
        print(args.freeze)
    else:
        print(args.report)
    return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    sys.exit(main())
