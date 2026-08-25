#!/usr/bin/env python3
"""Solve task-from-camera from measured ChArUco board poses."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

from helmet_d455.calibration import (
    atomic_json,
    camera_parameters,
    detect_charuco_pose,
    load_json,
    sha256_file,
    solve_task_from_camera,
    task_from_camera_fields,
)


def _matrix_from_file(path: Path) -> np.ndarray:
    value = load_json(path)
    if isinstance(value, dict):
        value = value.get("task_from_board_matrix", value.get("matrix"))
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("task-from-board JSON must contain a finite 4x4 matrix")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("configs/helmet_d455_mount_template.json"))
    parser.add_argument("--board-spec", type=Path, default=Path("configs/helmet_d455_charuco_board.json"))
    parser.add_argument(
        "--task-from-board-json",
        type=Path,
        help="physical measurement of the board frame in the task frame; mandatory for real capture",
    )
    parser.add_argument(
        "--mount-json",
        type=Path,
        help="measured mount/parent contract; mandatory before a physical result can be frozen",
    )
    parser.add_argument("--annotated-dir", type=Path)
    args = parser.parse_args()

    capture = load_json(args.capture_manifest)
    template = load_json(args.template)
    board_spec = load_json(args.board_spec)
    synthetic = bool(capture.get("synthetic"))
    if args.task_from_board_json:
        task_from_board = _matrix_from_file(args.task_from_board_json)
        task_from_board_source = str(args.task_from_board_json.resolve())
    elif synthetic:
        task_from_board = np.asarray(capture["task_from_board_matrix"], dtype=np.float64)
        task_from_board_source = "synthetic_fixture_ground_truth"
    else:
        raise RuntimeError(
            "physical solve requires --task-from-board-json from a measured board pose; guessing is forbidden"
        )

    intrinsic, distortion = camera_parameters(capture)
    annotated_dir = args.annotated_dir or args.output.parent / f"{args.output.stem}_annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    poses: list[np.ndarray] = []
    observations = []
    rejected = []
    for frame in capture.get("frames", []):
        image = cv2.imread(str(frame["color_path"]), cv2.IMREAD_COLOR)
        try:
            result = detect_charuco_pose(image, board_spec, intrinsic, distortion)
        except (RuntimeError, ValueError) as error:
            rejected.append({"index": int(frame["index"]), "reason": str(error)})
            continue
        annotated = annotated_dir / f"frame_{int(frame['index']):04d}.png"
        cv2.imwrite(str(annotated), result.pop("annotated"))
        pose = np.asarray(result.pop("camera_from_board"), dtype=np.float64)
        poses.append(pose)
        observations.append(
            {
                "index": int(frame["index"]),
                "camera_from_board_matrix": pose.tolist(),
                "annotated_path": str(annotated.resolve()),
                **{key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()},
            }
        )

    minimum = int(template["calibration"]["charuco"]["minimum_valid_frames"])
    if len(poses) < minimum:
        raise RuntimeError(f"only {len(poses)} valid ChArUco frames; at least {minimum} are required")
    task_from_camera, spread = solve_task_from_camera(poses, task_from_board)
    reprojection = [float(row["pose_reprojection_rmse_px"]) for row in observations]

    candidate = deepcopy(template)
    candidate.update(
        name="HELMET_D455_CALIBRATION_CANDIDATE",
        status="SOLVED_PENDING_VALIDATION",
    )
    device = dict(capture["device"])
    candidate["device"] = {
        "product": device.get("name", device.get("product", "UNKNOWN")),
        "serial": device.get("serial", "UNKNOWN"),
        "firmware_version": device.get("firmware_version", "UNKNOWN"),
        "recommended_firmware_version": device.get("recommended_firmware_version", "NOT_SUPPORTED"),
        "usb_mode": device.get("usb_mode", "UNKNOWN"),
        "physical_port": device.get("physical_port", "NOT_SUPPORTED"),
    }
    candidate["streams"] = deepcopy(capture["streams"])
    candidate["intrinsics"] = deepcopy(capture["intrinsics"])
    candidate["depth_to_color_extrinsics"] = deepcopy(capture["depth_to_color_extrinsics"])
    candidate["timestamping"] = {
        "device_timestamp_domains": sorted({str(row.get("timestamp_domain", "UNKNOWN")) for row in capture["frames"]}),
        "per_frame_device_and_host_timestamps": True,
        "capture_manifest": str(args.capture_manifest.resolve()),
    }
    candidate["capture_settings"] = deepcopy(capture["capture_settings"])
    if args.mount_json:
        mount_data = load_json(args.mount_json)
        candidate["mount"] = deepcopy(mount_data.get("mount", mount_data))
    elif synthetic:
        candidate["mount"] = {
            "status": "SYNTHETIC_FIXED_TEST_RIG",
            "type": "synthetic_fixed_test_rig",
            "parent_link": "synthetic_world",
            "parent_link_can_move": False,
            "locked_parent_pose_id": "synthetic_fixture",
            "parent_link_fk_provider": None,
            "parent_from_camera_matrix": task_from_camera.tolist(),
            "runtime_requirement": "synthetic offline test only",
        }
    fields = task_from_camera_fields(task_from_camera)
    candidate["extrinsics"] = {
        "status": "SOLVED_PENDING_VALIDATION",
        "method": "CHARUCO_BOARD_POSE_TO_TASK_FRAME",
        **fields,
    }
    candidate["calibration"]["charuco"].update(
        {
            "task_from_board_matrix": task_from_board.tolist(),
            "task_from_board_source": task_from_board_source,
            "capture_manifest": str(args.capture_manifest.resolve()),
            "capture_manifest_sha256": sha256_file(args.capture_manifest),
            "board_spec": str(args.board_spec.resolve()),
            "board_spec_sha256": sha256_file(args.board_spec),
            "valid_frames": len(poses),
            "rejected_frames": rejected,
            "observations": observations,
            "aggregate_pose_reprojection_rmse_px": float(np.sqrt(np.mean(np.square(reprojection)))),
            "maximum_frame_pose_reprojection_rmse_px": float(max(reprojection)),
            "pose_spread": spread,
            "solved_at_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic": synthetic,
        }
    )
    candidate["rendering"]["final_50_episode_render_allowed"] = False
    candidate["freeze"]["candidate_is_synthetic"] = synthetic
    atomic_json(args.output, candidate)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
