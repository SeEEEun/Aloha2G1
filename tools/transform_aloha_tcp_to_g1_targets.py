#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


DEFAULT_INPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/aloha_tcp_trajectory.npz")
DEFAULT_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_cartesian_targets.npz")
MODEL_CANDIDATES = [
    Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml"),
    Path("/home/jbnu/jaeyoung/TWIST/assets/unitree_g1/g1_with_hands.xml"),
    Path("/home/jbnu/jaeyoung/unitree/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"),
    Path("/home/jbnu/jaeyoung/TWIST/assets/g1/g1_29dof_rev_1_0.urdf"),
]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_EE_FRAME = "left_wrist_yaw_link"
RIGHT_EE_FRAME = "right_wrist_yaw_link"
LEFT_PALM_OFFSET = np.array([0.0415, 0.003, 0.0], dtype=np.float64)
RIGHT_PALM_OFFSET = np.array([0.0415, -0.003, 0.0], dtype=np.float64)
ALIGN_ROTATION_CONVENTION = "intrinsic XYZ (roll, pitch, yaw in degrees); implemented as Rz(yaw) @ Ry(pitch) @ Rx(roll)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform ALOHA TCP FK trajectory into G1 Cartesian targets.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--g1-model", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.3)
    parser.add_argument("--align-rotation-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--left-tool-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--right-tool-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fail(message: str) -> RuntimeError:
    return RuntimeError(message)


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise fail(f"Missing {description}: {path}")
    if not path.is_file():
        raise fail(f"Expected {description} to be a file: {path}")
    return path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(require_file(path, "input NPZ"), allow_pickle=False)
    return {key: data[key] for key in data.files}


def ensure_keys(payload: dict[str, np.ndarray], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise fail(f"Missing required NPZ keys: {missing}")


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise fail("Encountered zero quaternion")
    return quat / norm


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat_normalize(quat)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return quat_normalize((1.0 - t) * q0 + t * q1)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return quat_normalize(s0 * q0 + s1 * q1)


def make_quat_sequence_continuous(quats: np.ndarray) -> np.ndarray:
    result = np.array(quats, dtype=np.float64, copy=True)
    for i in range(1, result.shape[0]):
        if float(np.dot(result[i - 1], result[i])) < 0.0:
            result[i] *= -1.0
    return result


def rotation_from_rpy_deg(rpy_deg: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float64))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def find_model_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return require_file(explicit.expanduser().resolve(), "G1 model")
    for candidate in MODEL_CANDIDATES:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise fail(f"Could not find a G1 model from candidates: {[str(p) for p in MODEL_CANDIDATES]}")


def load_neutral_state(model: mujoco.MjModel) -> tuple[np.ndarray, str]:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id >= 0:
        return np.array(model.key_qpos[key_id], dtype=np.float64), "keyframe: stand"
    if hasattr(model, "qpos0"):
        return np.array(model.qpos0, dtype=np.float64), "qpos0"
    raise fail("Could not find neutral pose source. Expected keyframe 'stand' or qpos0.")


def compute_body_pose(data: mujoco.MjData, body_id: int, offset_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array(data.xpos[body_id], dtype=np.float64)
    rot = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    tcp = pos + rot @ offset_local
    quat = mat_to_quat_wxyz(rot)
    return tcp, quat


def compute_g1_start_pose(model: mujoco.MjModel) -> dict[str, np.ndarray | str]:
    neutral_qpos, source = load_neutral_state(model)
    data = mujoco.MjData(model)
    data.qpos[:] = neutral_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_FRAME)
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_FRAME)
    if left_body < 0 or right_body < 0:
        raise fail(f"Required G1 EE body not found: {LEFT_EE_FRAME}, {RIGHT_EE_FRAME}")

    left_pos, left_quat = compute_body_pose(data, left_body, LEFT_PALM_OFFSET)
    right_pos, right_quat = compute_body_pose(data, right_body, RIGHT_PALM_OFFSET)

    joint_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    joint_values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise fail(f"Missing required G1 arm joint: {name}")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        joint_values.append(float(neutral_qpos[qpos_adr]))

    return {
        "neutral_pose_source": source,
        "g1_left_start_position_xyz_m": left_pos,
        "g1_left_start_quaternion_wxyz": left_quat,
        "g1_right_start_position_xyz_m": right_pos,
        "g1_right_start_quaternion_wxyz": right_quat,
        "g1_neutral_arm_q": np.asarray(joint_values, dtype=np.float64),
        "g1_arm_joint_names": np.asarray(joint_names),
    }


def validate_input(data: dict[str, np.ndarray]) -> int:
    required = [
        "timestamp_s",
        "left_position_xyz_m",
        "left_quaternion_wxyz",
        "right_position_xyz_m",
        "right_quaternion_wxyz",
        "left_gripper_m",
        "right_gripper_m",
        "right_minus_left_position_xyz_m",
        "hands_distance_m",
    ]
    ensure_keys(data, required)
    frame_count = int(data["timestamp_s"].shape[0])
    for key in required:
        if data[key].shape[0] != frame_count:
            raise fail(f"{key} first dimension mismatch")
        if not np.isfinite(data[key]).all():
            raise fail(f"{key} contains NaN or inf")
    if frame_count >= 2 and not np.all(np.diff(data["timestamp_s"]) > 0.0):
        raise fail("timestamp_s is not strictly increasing")
    return frame_count


def transform(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray | str]]:
    input_data = load_npz(args.input.expanduser().resolve())
    frame_count = validate_input(input_data)
    if not (0.0 <= args.orientation_weight <= 1.0):
        raise fail("--orientation-weight must be within [0.0, 1.0]")

    model_path = find_model_path(args.g1_model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    g1_info = compute_g1_start_pose(model)

    align_rot = rotation_from_rpy_deg(np.asarray(args.align_rotation_rpy_deg, dtype=np.float64))
    left_tool_rot = rotation_from_rpy_deg(np.asarray(args.left_tool_rpy_deg, dtype=np.float64))
    right_tool_rot = rotation_from_rpy_deg(np.asarray(args.right_tool_rpy_deg, dtype=np.float64))

    aloha_left_pos = np.asarray(input_data["left_position_xyz_m"], dtype=np.float64)
    aloha_right_pos = np.asarray(input_data["right_position_xyz_m"], dtype=np.float64)
    aloha_left_quat = np.asarray(input_data["left_quaternion_wxyz"], dtype=np.float64)
    aloha_right_quat = np.asarray(input_data["right_quaternion_wxyz"], dtype=np.float64)
    aloha_rel = np.asarray(input_data["right_minus_left_position_xyz_m"], dtype=np.float64)
    aloha_dist = np.asarray(input_data["hands_distance_m"], dtype=np.float64)

    left_start_pos = np.asarray(g1_info["g1_left_start_position_xyz_m"], dtype=np.float64)
    right_start_pos = np.asarray(g1_info["g1_right_start_position_xyz_m"], dtype=np.float64)
    left_start_quat = np.asarray(g1_info["g1_left_start_quaternion_wxyz"], dtype=np.float64)
    right_start_quat = np.asarray(g1_info["g1_right_start_quaternion_wxyz"], dtype=np.float64)
    left_start_rot = quat_to_mat(left_start_quat)
    right_start_rot = quat_to_mat(right_start_quat)

    left_delta = aloha_left_pos - aloha_left_pos[0]
    right_delta = aloha_right_pos - aloha_right_pos[0]
    g1_left_pos = left_start_pos[None, :] + args.scale * (left_delta @ align_rot.T)
    g1_right_pos = right_start_pos[None, :] + args.scale * (right_delta @ align_rot.T)

    left_rot0_inv = quat_to_mat(aloha_left_quat[0]).T
    right_rot0_inv = quat_to_mat(aloha_right_quat[0]).T
    full_left_quat = np.zeros((frame_count, 4), dtype=np.float64)
    full_right_quat = np.zeros((frame_count, 4), dtype=np.float64)
    align_left = align_rot
    align_right = align_rot

    for i in range(frame_count):
        left_delta_rot = left_rot0_inv @ quat_to_mat(aloha_left_quat[i])
        right_delta_rot = right_rot0_inv @ quat_to_mat(aloha_right_quat[i])
        left_full_rot = left_start_rot @ align_left @ left_delta_rot @ left_tool_rot
        right_full_rot = right_start_rot @ align_right @ right_delta_rot @ right_tool_rot
        full_left_quat[i] = quat_slerp(left_start_quat, mat_to_quat_wxyz(left_full_rot), args.orientation_weight)
        full_right_quat[i] = quat_slerp(right_start_quat, mat_to_quat_wxyz(right_full_rot), args.orientation_weight)

    full_left_quat = make_quat_sequence_continuous(full_left_quat)
    full_right_quat = make_quat_sequence_continuous(full_right_quat)

    g1_rel = g1_right_pos - g1_left_pos
    g1_dist = np.linalg.norm(g1_rel, axis=1)
    aloha_delta_rel = aloha_rel - aloha_rel[0]
    g1_delta_rel = g1_rel - g1_rel[0]
    expected_delta_rel = args.scale * (aloha_delta_rel @ align_rot.T)
    relative_motion_error = np.linalg.norm(g1_delta_rel - expected_delta_rel, axis=1)
    hands_distance_error = np.abs((g1_dist - g1_dist[0]) - args.scale * (aloha_dist - aloha_dist[0]))

    payload = {
        "timestamp_s": np.asarray(input_data["timestamp_s"], dtype=np.float64),
        "g1_left_target_position_xyz_m": g1_left_pos,
        "g1_left_target_quaternion_wxyz": full_left_quat,
        "g1_right_target_position_xyz_m": g1_right_pos,
        "g1_right_target_quaternion_wxyz": full_right_quat,
        "g1_left_gripper_source_m": np.asarray(input_data["left_gripper_m"], dtype=np.float64),
        "g1_right_gripper_source_m": np.asarray(input_data["right_gripper_m"], dtype=np.float64),
        "aloha_left_position_xyz_m": aloha_left_pos,
        "aloha_left_quaternion_wxyz": aloha_left_quat,
        "aloha_right_position_xyz_m": aloha_right_pos,
        "aloha_right_quaternion_wxyz": aloha_right_quat,
        "aloha_left_gripper_m": np.asarray(input_data["left_gripper_m"], dtype=np.float64),
        "aloha_right_gripper_m": np.asarray(input_data["right_gripper_m"], dtype=np.float64),
        "aloha_right_minus_left_position_xyz_m": aloha_rel,
        "aloha_hands_distance_m": aloha_dist,
        "g1_right_minus_left_position_xyz_m": g1_rel,
        "g1_hands_distance_m": g1_dist,
        "aloha_delta_relative_position_xyz_m": aloha_delta_rel,
        "g1_delta_relative_position_xyz_m": g1_delta_rel,
        "relative_motion_error_m": relative_motion_error,
        "hands_distance_error_m": hands_distance_error,
        "g1_left_start_position_xyz_m": left_start_pos,
        "g1_left_start_quaternion_wxyz": left_start_quat,
        "g1_right_start_position_xyz_m": right_start_pos,
        "g1_right_start_quaternion_wxyz": right_start_quat,
        "g1_neutral_arm_q": np.asarray(g1_info["g1_neutral_arm_q"], dtype=np.float64),
        "g1_arm_joint_names": np.asarray(g1_info["g1_arm_joint_names"]),
        "g1_model_path": np.asarray(str(model_path)),
        "g1_left_ee_frame": np.asarray(LEFT_EE_FRAME),
        "g1_right_ee_frame": np.asarray(RIGHT_EE_FRAME),
        "g1_left_palm_offset_m": LEFT_PALM_OFFSET,
        "g1_right_palm_offset_m": RIGHT_PALM_OFFSET,
        "coordinate_frame_source": np.asarray("aloha_stationary_world"),
        "coordinate_frame_target": np.asarray("unitree_g1_world_from_stand"),
        "quaternion_order": np.asarray("wxyz"),
        "scale": np.asarray(args.scale, dtype=np.float64),
        "orientation_weight": np.asarray(args.orientation_weight, dtype=np.float64),
        "align_rotation_rpy_deg": np.asarray(args.align_rotation_rpy_deg, dtype=np.float64),
        "align_rotation_convention": np.asarray(ALIGN_ROTATION_CONVENTION),
        "left_tool_rpy_deg": np.asarray(args.left_tool_rpy_deg, dtype=np.float64),
        "right_tool_rpy_deg": np.asarray(args.right_tool_rpy_deg, dtype=np.float64),
    }
    return payload, g1_info


def validate_payload(payload: dict[str, np.ndarray]) -> None:
    frame_count = int(payload["timestamp_s"].shape[0])
    for key, value in payload.items():
        if value.ndim > 0 and value.shape[0] == frame_count and np.issubdtype(value.dtype, np.number):
            if not np.isfinite(value).all():
                raise fail(f"{key} contains NaN or inf")
    for key in [
        "g1_left_target_quaternion_wxyz",
        "g1_right_target_quaternion_wxyz",
        "g1_left_start_quaternion_wxyz",
        "g1_right_start_quaternion_wxyz",
    ]:
        arr = np.asarray(payload[key], dtype=np.float64)
        if arr.ndim == 1:
            norms = np.array([np.linalg.norm(arr)])
        else:
            norms = np.linalg.norm(arr, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise fail(f"{key} quaternion norm check failed")
    if frame_count >= 2 and not np.all(np.diff(payload["timestamp_s"]) > 0.0):
        raise fail("timestamp_s is not strictly increasing")
    if np.any(payload["relative_motion_error_m"] < 0.0):
        raise fail("relative_motion_error_m contains negative values")


def print_model_info(model_path: Path, g1_info: dict[str, np.ndarray | str]) -> None:
    print(f"g1 model path: {model_path}")
    print(f"neutral pose source: {g1_info['neutral_pose_source']}")
    print(f"arm joint names: {list(g1_info['g1_arm_joint_names'])}")
    print(f"left ee frame: {LEFT_EE_FRAME}")
    print(f"right ee frame: {RIGHT_EE_FRAME}")
    print(f"left palm offset: {LEFT_PALM_OFFSET.tolist()}")
    print(f"right palm offset: {RIGHT_PALM_OFFSET.tolist()}")
    print(
        "g1 start left pose:",
        np.asarray(g1_info["g1_left_start_position_xyz_m"]).tolist(),
        np.asarray(g1_info["g1_left_start_quaternion_wxyz"]).tolist(),
    )
    print(
        "g1 start right pose:",
        np.asarray(g1_info["g1_right_start_position_xyz_m"]).tolist(),
        np.asarray(g1_info["g1_right_start_quaternion_wxyz"]).tolist(),
    )


def print_transform_settings(args: argparse.Namespace) -> None:
    print(f"scale: {args.scale}")
    print(f"align rotation rpy deg: {list(args.align_rotation_rpy_deg)}")
    print(f"align rotation convention: {ALIGN_ROTATION_CONVENTION}")
    print(f"orientation weight: {args.orientation_weight}")
    print(f"left tool rpy deg: {list(args.left_tool_rpy_deg)}")
    print(f"right tool rpy deg: {list(args.right_tool_rpy_deg)}")


def print_stats(payload: dict[str, np.ndarray]) -> None:
    frame_count = int(payload["timestamp_s"].shape[0])
    duration = float(payload["timestamp_s"][-1] - payload["timestamp_s"][0]) if frame_count > 1 else 0.0
    left = payload["g1_left_target_position_xyz_m"]
    right = payload["g1_right_target_position_xyz_m"]
    dist = payload["g1_hands_distance_m"]
    err = payload["relative_motion_error_m"]
    print(f"frame count: {frame_count}")
    print(f"duration_s: {duration:.6f}")
    print(f"left target xyz min: {left.min(axis=0).tolist()}")
    print(f"left target xyz max: {left.max(axis=0).tolist()}")
    print(f"right target xyz min: {right.min(axis=0).tolist()}")
    print(f"right target xyz max: {right.max(axis=0).tolist()}")
    print(f"hand distance min/max/mean: {float(dist.min()):.6f} {float(dist.max()):.6f} {float(dist.mean()):.6f}")
    print(f"relative motion error mean/max: {float(err.mean()):.6f} {float(err.max()):.6f}")


def print_sample_frames(payload: dict[str, np.ndarray]) -> None:
    indices = sorted({idx for idx in [0, 100, 300, 500, payload["timestamp_s"].shape[0] - 1] if 0 <= idx < payload["timestamp_s"].shape[0]})
    for idx in indices:
        print(f"frame {idx}:")
        print(f"  aloha left position: {payload['aloha_left_position_xyz_m'][idx].tolist()}")
        print(f"  aloha right position: {payload['aloha_right_position_xyz_m'][idx].tolist()}")
        print(f"  g1 left target position: {payload['g1_left_target_position_xyz_m'][idx].tolist()}")
        print(f"  g1 right target position: {payload['g1_right_target_position_xyz_m'][idx].tolist()}")
        print(f"  g1 left quaternion: {payload['g1_left_target_quaternion_wxyz'][idx].tolist()}")
        print(f"  g1 right quaternion: {payload['g1_right_target_quaternion_wxyz'][idx].tolist()}")
        print(f"  g1 hand distance: {float(payload['g1_hands_distance_m'][idx]):.6f}")
        print(f"  relative motion error: {float(payload['relative_motion_error_m'][idx]):.6f}")


def print_saved_report(output_path: Path) -> None:
    saved = np.load(output_path, allow_pickle=False)
    print(f"saved keys: {sorted(saved.files)}")
    for key in [
        "timestamp_s",
        "g1_left_target_position_xyz_m",
        "g1_left_target_quaternion_wxyz",
        "g1_right_target_position_xyz_m",
        "g1_right_target_quaternion_wxyz",
        "relative_motion_error_m",
        "g1_hands_distance_m",
    ]:
        print(f"{key}: shape={saved[key].shape} dtype={saved[key].dtype}")
    print(f"file size bytes: {os.path.getsize(output_path)}")
    print(
        "first g1 pose:",
        saved["g1_left_target_position_xyz_m"][0].tolist(),
        saved["g1_left_target_quaternion_wxyz"][0].tolist(),
        saved["g1_right_target_position_xyz_m"][0].tolist(),
        saved["g1_right_target_quaternion_wxyz"][0].tolist(),
    )
    print(
        "last g1 pose:",
        saved["g1_left_target_position_xyz_m"][-1].tolist(),
        saved["g1_left_target_quaternion_wxyz"][-1].tolist(),
        saved["g1_right_target_position_xyz_m"][-1].tolist(),
        saved["g1_right_target_quaternion_wxyz"][-1].tolist(),
    )
    print(
        f"relative_motion_error mean/max: {float(saved['relative_motion_error_m'].mean()):.6f} {float(saved['relative_motion_error_m'].max()):.6f}"
    )
    print(
        f"g1 hand distance min/max: {float(saved['g1_hands_distance_m'].min()):.6f} {float(saved['g1_hands_distance_m'].max()):.6f}"
    )


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise fail(f"Output already exists. Use --overwrite to replace: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload, g1_info = transform(args)
    model_path = find_model_path(args.g1_model)
    validate_payload(payload)
    print_model_info(model_path, g1_info)
    print_transform_settings(args)
    print_stats(payload)
    print_sample_frames(payload)
    np.savez(output_path, **payload)
    print_saved_report(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
