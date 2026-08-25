#!/usr/bin/env python3
"""Capture synchronized D455 calibration frames, or a marked synthetic fixture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from helmet_d455.calibration import (
    atomic_json,
    board_from_spec,
    invert_transform,
    load_json,
    opencv_from_canonical_board,
    project_task_points,
    sha256_file,
    transform,
)
from helmet_d455.realsense import (
    device_record,
    extrinsics_record,
    intrinsics_record,
    sensor_settings,
    start_pipeline,
)


def _board_image(board: object, size: tuple[int, int]) -> np.ndarray:
    if hasattr(board, "generateImage"):
        return board.generateImage(size, marginSize=0, borderBits=1)
    return board.draw(size, marginSize=0, borderBits=1)


def _synthetic_capture(output: Path, template: dict, board_spec: dict, count: int) -> dict:
    width, height = 640, 480
    intrinsic = np.asarray([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros(5)
    board_width = float(board_spec["printed_pattern_width_mm"]) / 1000.0
    board_height = float(board_spec["printed_pattern_height_mm"]) / 1000.0
    workspace = template["calibration"]["black_workspace_frame"]["physical_dimensions_m"]
    board_bottom_left_task = np.asarray(
        [(float(workspace["x"]) - board_width) / 2.0, (float(workspace["y"]) - board_height) / 2.0, 0.0]
    )
    task_from_board = transform(translation=board_bottom_left_task)
    # ROS optical +X is image-right, +Y is image-down and +Z is forward.
    # Task +Z is up/out of the tabletop, so a straight-down synthetic camera
    # uses diag(+1,-1,-1) and sits over the workspace center.
    task_from_camera = transform(
        rotation=np.diag([1.0, -1.0, -1.0]),
        translation=[float(workspace["x"]) / 2.0, float(workspace["y"]) / 2.0, 1.4],
    )
    camera_from_task = invert_transform(task_from_camera)
    camera_from_board = camera_from_task @ task_from_board
    camera_from_opencv_board = camera_from_board @ invert_transform(
        opencv_from_canonical_board(board_spec)
    )
    board = board_from_spec(board_spec)
    raster = _board_image(board, (1400, 1000))
    src = np.asarray([[0, 0], [1399, 0], [1399, 999], [0, 999]], dtype=np.float32)
    frames = []
    for index in range(count):
        # Repeated identical ideal captures isolate plumbing/geometry from
        # detector-noise characterization; real captures must span poses.
        outer = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [float(board_spec["printed_pattern_width_mm"]) / 1000.0, 0.0, 0.0],
                [float(board_spec["printed_pattern_width_mm"]) / 1000.0, float(board_spec["printed_pattern_height_mm"]) / 1000.0, 0.0],
                [0.0, float(board_spec["printed_pattern_height_mm"]) / 1000.0, 0.0],
            ],
            dtype=np.float64,
        )
        rvec, _ = cv2.Rodrigues(camera_from_opencv_board[:3, :3])
        dst, _ = cv2.projectPoints(outer, rvec, camera_from_opencv_board[:3, 3], intrinsic, distortion)
        homography = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        canvas = np.full((height, width), 235, dtype=np.uint8)
        warped = cv2.warpPerspective(raster, homography, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
        mask = cv2.warpPerspective(np.full_like(raster, 255), homography, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        canvas[mask > 0] = warped[mask > 0]
        color = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        depth = np.full((height, width), 1400, dtype=np.uint16)
        color_path = output / "color" / f"charuco_{index:04d}.png"
        depth_path = output / "depth" / f"charuco_{index:04d}.npy"
        color_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(color_path), color):
            raise RuntimeError(f"failed to save {color_path}")
        np.save(depth_path, depth)
        frames.append(
            {
                "index": index,
                "color_path": str(color_path.resolve()),
                "depth_path": str(depth_path.resolve()),
                "host_monotonic_timestamp_ns": 1_000_000_000 + index * 33_333_333,
                "host_wall_timestamp_ns": 2_000_000_000 + index * 33_333_333,
                "color_device_timestamp_ms": index * 33.333333,
                "depth_device_timestamp_ms": index * 33.333333,
                "timestamp_domain": "synthetic_clock",
                "color_frame_number": index,
                "depth_frame_number": index,
                "ground_truth_camera_from_board_matrix": camera_from_board.tolist(),
            }
        )

    corners = np.asarray(template["calibration"]["black_workspace_frame"]["corners_task_xyz_m"])
    projected, _ = project_task_points(corners, task_from_camera, intrinsic, distortion)
    black = np.full((height, width, 3), 225, dtype=np.uint8)
    cv2.polylines(black, [np.rint(projected).astype(np.int32)], True, (5, 5, 5), 3, cv2.LINE_AA)
    black_path = output / "black_workspace.png"
    cv2.imwrite(str(black_path), black)

    color_intrinsics = {
        "width_px": width,
        "height_px": height,
        "fx_px": 700.0,
        "fy_px": 700.0,
        "cx_px": 320.0,
        "cy_px": 240.0,
        "matrix": intrinsic.tolist(),
        "distortion_model": "none",
        "distortion_coefficients": distortion.tolist(),
    }
    return {
        "capture_kind": "SYNTHETIC_OFFLINE_TEST_NOT_A_PHYSICAL_CALIBRATION",
        "synthetic": True,
        "device": {
            "name": "SYNTHETIC_D455_MODEL",
            "serial": "SYNTHETIC_NOT_A_DEVICE_SERIAL",
            "firmware_version": "SYNTHETIC",
            "usb_mode": "SYNTHETIC",
        },
        "streams": template["streams"],
        "intrinsics": {"color": color_intrinsics, "depth": color_intrinsics},
        "depth_to_color_extrinsics": {
            "rotation": np.eye(3).tolist(),
            "translation_m": [0.0, 0.0, 0.0],
            "target_from_source_matrix": np.eye(4).tolist(),
            "direction": "color_from_depth",
        },
        "capture_settings": {"color": "SYNTHETIC", "depth": "SYNTHETIC"},
        "task_from_board_matrix": task_from_board.tolist(),
        "synthetic_ground_truth_task_from_camera_matrix": task_from_camera.tolist(),
        "frames": frames,
        "black_workspace_images": [str(black_path.resolve())],
    }


def _physical_capture(args: argparse.Namespace, template: dict) -> dict:
    color = template["streams"]["color"]
    depth = template["streams"]["depth"]
    active = start_pipeline(
        serial=args.serial,
        color=(int(color["width_px"]), int(color["height_px"]), int(color["fps"])),
        depth=(int(depth["width_px"]), int(depth["height_px"]), int(depth["fps"])),
        color_exposure=args.color_exposure,
        color_gain=args.color_gain,
        depth_exposure=args.depth_exposure,
        depth_gain=args.depth_gain,
    )
    try:
        for _ in range(args.warmup_frames):
            active.pipeline.wait_for_frames(5000)
        frames = []
        for index in range(args.frames):
            frame_set = active.pipeline.wait_for_frames(5000)
            wall = time.time_ns()
            monotonic = time.monotonic_ns()
            color_frame = frame_set.get_color_frame()
            depth_frame = frame_set.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("D455 returned an incomplete synchronized frame set")
            rgb = np.asanyarray(color_frame.get_data())
            depth_array = np.asanyarray(depth_frame.get_data()).copy()
            color_path = args.output_dir / "color" / f"charuco_{index:04d}.png"
            depth_path = args.output_dir / "depth" / f"charuco_{index:04d}.npy"
            color_path.parent.mkdir(parents=True, exist_ok=True)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(color_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"failed to save {color_path}")
            np.save(depth_path, depth_array)
            frames.append(
                {
                    "index": index,
                    "color_path": str(color_path.resolve()),
                    "depth_path": str(depth_path.resolve()),
                    "host_monotonic_timestamp_ns": monotonic,
                    "host_wall_timestamp_ns": wall,
                    "color_device_timestamp_ms": float(color_frame.get_timestamp()),
                    "depth_device_timestamp_ms": float(depth_frame.get_timestamp()),
                    "timestamp_domain": str(color_frame.get_frame_timestamp_domain()),
                    "color_frame_number": int(color_frame.get_frame_number()),
                    "depth_frame_number": int(depth_frame.get_frame_number()),
                }
            )
        return {
            "capture_kind": "PHYSICAL_D455_CALIBRATION_CAPTURE",
            "synthetic": False,
            "device": device_record(active),
            "streams": template["streams"],
            "intrinsics": {
                "color": intrinsics_record(active.color_profile.get_intrinsics()),
                "depth": intrinsics_record(active.depth_profile.get_intrinsics()),
            },
            "depth_to_color_extrinsics": {
                **extrinsics_record(active.depth_profile.get_extrinsics_to(active.color_profile)),
                "direction": "color_from_depth",
            },
            "capture_settings": {
                "color": sensor_settings(active.color_sensor),
                "depth": sensor_settings(active.depth_sensor),
                "fixed_exposure_requested": any(
                    value is not None
                    for value in (args.color_exposure, args.color_gain, args.depth_exposure, args.depth_gain)
                ),
            },
            "task_from_board_matrix": "MUST_BE_SUPPLIED_TO_SOLVER_FROM_PHYSICAL_MEASUREMENT",
            "frames": frames,
            "black_workspace_images": [],
        }
    finally:
        active.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path("configs/helmet_d455_mount_template.json"))
    parser.add_argument("--board-spec", type=Path, default=Path("configs/helmet_d455_charuco_board.json"))
    parser.add_argument("--serial")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--color-exposure", type=float)
    parser.add_argument("--color-gain", type=float)
    parser.add_argument("--depth-exposure", type=float)
    parser.add_argument("--depth-gain", type=float)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = load_json(args.template)
    board_spec = load_json(args.board_spec)
    body = (
        _synthetic_capture(args.output_dir, template, board_spec, args.frames)
        if args.simulate
        else _physical_capture(args, template)
    )
    report = {
        "schema_version": "helmet_d455_calibration_capture_v1",
        "status": "CAPTURE_COMPLETE_SYNTHETIC_ONLY" if args.simulate else "CAPTURE_COMPLETE_PHYSICAL",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "template": str(args.template.resolve()),
        "template_sha256": sha256_file(args.template),
        "board_spec": str(args.board_spec.resolve()),
        "board_spec_sha256": sha256_file(args.board_spec),
        **body,
    }
    manifest = args.output_dir / "capture_manifest.json"
    atomic_json(manifest, report)
    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
