#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

TWIST_PYTHON = "/home/jbnu/miniconda3/envs/twist/bin/python"

try:
    import numpy as np
    import mujoco
except ModuleNotFoundError:
    if Path(sys.executable).resolve() != Path(TWIST_PYTHON).resolve():
        os.execv(TWIST_PYTHON, [TWIST_PYTHON, __file__, *sys.argv[1:]])
    raise

import argparse
import math
import time

try:
    from mujoco import viewer as mujoco_viewer
except ImportError:
    mujoco_viewer = None


DEFAULT_INPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_cartesian_targets.npz")
DEFAULT_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEFAULT_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_sparse_ik_validation.npz")
LEFT_EE_BODY = "left_wrist_yaw_link"
RIGHT_EE_BODY = "right_wrist_yaw_link"
LEFT_PALM_OFFSET = np.array([0.0415, 0.003, 0.0], dtype=np.float64)
RIGHT_PALM_OFFSET = np.array([0.0415, -0.003, 0.0], dtype=np.float64)
LEFT_ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
FULL_ARM_NAMES = LEFT_ARM_NAMES + RIGHT_ARM_NAMES
MODE_NAMES = ["full_pose", "reduced_orientation", "position_only", "best_effort"]
SOLVER_NAME = "damped_least_squares_with_augmented_regularization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate G1 Cartesian targets and sparse bimanual IK.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("targets", "ik", "both"), default="both")
    parser.add_argument("--frame-duration-s", type=float, default=2.0)
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.15)
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--no-viewer", action="store_true")
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


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
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
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat_normalize(quat)


def rotation_log_world(current_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    relative = target_rot @ current_rot.T
    trace = float(np.trace(relative))
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = math.acos(cos_angle)
    if angle < 1e-8:
        return 0.5 * np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ],
            dtype=np.float64,
        )
    denom = 2.0 * math.sin(angle)
    axis = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    ) / denom
    return axis * angle


def load_and_validate_input(path: Path) -> tuple[dict[str, np.ndarray], int]:
    payload = load_npz(path)
    required = [
        "timestamp_s",
        "g1_left_target_position_xyz_m",
        "g1_left_target_quaternion_wxyz",
        "g1_right_target_position_xyz_m",
        "g1_right_target_quaternion_wxyz",
        "g1_left_start_position_xyz_m",
        "g1_left_start_quaternion_wxyz",
        "g1_right_start_position_xyz_m",
        "g1_right_start_quaternion_wxyz",
        "g1_neutral_arm_q",
        "g1_arm_joint_names",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise fail(f"Missing required NPZ keys: {missing}")

    frame_count = int(payload["timestamp_s"].shape[0])
    if frame_count <= 0:
        raise fail("Input trajectory has no frames")

    for key in [
        "g1_left_target_position_xyz_m",
        "g1_right_target_position_xyz_m",
    ]:
        if payload[key].shape != (frame_count, 3):
            raise fail(f"{key} must have shape ({frame_count}, 3), got {payload[key].shape}")
    for key in [
        "g1_left_target_quaternion_wxyz",
        "g1_right_target_quaternion_wxyz",
    ]:
        if payload[key].shape != (frame_count, 4):
            raise fail(f"{key} must have shape ({frame_count}, 4), got {payload[key].shape}")
    if payload["timestamp_s"].shape != (frame_count,):
        raise fail(f"timestamp_s must have shape ({frame_count},)")

    for key, value in payload.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == frame_count and np.issubdtype(value.dtype, np.number):
            if not np.isfinite(value).all():
                raise fail(f"{key} contains NaN or inf")

    if not np.all(np.diff(payload["timestamp_s"]) > 0.0):
        raise fail("timestamp_s is not strictly increasing")

    for key in ["g1_left_target_quaternion_wxyz", "g1_right_target_quaternion_wxyz"]:
        norms = np.linalg.norm(payload[key], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise fail(f"{key} quaternion norms are not close to 1")
    return payload, frame_count


def actual_joint_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<unnamed:{i}>" for i in range(model.njnt)]


def resolve_joint_ids(model: mujoco.MjModel, expected_names: list[str]) -> list[int]:
    resolved: list[int] = []
    names = actual_joint_names(model)
    print("Actual model joint names:")
    for i, name in enumerate(names):
        print(f"  {i:02d}: {name}")
    for name in expected_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 and name.endswith("_joint"):
            alt = name[:-6]
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, alt)
        if joint_id < 0:
            raise fail(f"Could not resolve required joint '{name}'. Actual names: {names}")
        resolved.append(joint_id)
    return resolved


def validate_model(model_path: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(require_file(model_path, "G1 model")))
    left_joint_ids = resolve_joint_ids(model, LEFT_ARM_NAMES)
    right_joint_ids = resolve_joint_ids(model, RIGHT_ARM_NAMES)
    left_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY)
    right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY)
    if left_body_id < 0 or right_body_id < 0:
        raise fail(f"Required wrist bodies not found: {LEFT_EE_BODY}, {RIGHT_EE_BODY}")
    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_key_id < 0:
        raise fail("Model does not contain required keyframe 'stand'")

    arm_joint_ids = left_joint_ids + right_joint_ids
    qpos_ids = []
    dof_ids = []
    joint_limits = []
    joint_names = []
    for joint_id in arm_joint_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        low, high = [float(x) for x in model.jnt_range[joint_id]]
        if not low < high:
            raise fail(f"Joint '{name}' does not have a valid range: {low}, {high}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
        joint_limits.append([low, high])
        joint_names.append(name)

    stand_qpos = np.array(model.key_qpos[stand_key_id], dtype=np.float64)
    stand_arm_q = stand_qpos[qpos_ids]
    return {
        "model": model,
        "left_body_id": left_body_id,
        "right_body_id": right_body_id,
        "left_joint_ids": left_joint_ids,
        "right_joint_ids": right_joint_ids,
        "arm_joint_ids": arm_joint_ids,
        "arm_qpos_ids": np.asarray(qpos_ids, dtype=np.int64),
        "arm_dof_ids": np.asarray(dof_ids, dtype=np.int64),
        "joint_limits": np.asarray(joint_limits, dtype=np.float64),
        "joint_names": np.asarray(joint_names),
        "stand_key_id": stand_key_id,
        "stand_qpos": stand_qpos,
        "stand_arm_q": stand_arm_q,
    }


def sample_frame_indices(frame_count: int) -> np.ndarray:
    indices = sorted({0, frame_count - 1, 100, 300, 500})
    return np.asarray([i for i in indices if 0 <= i < frame_count], dtype=np.int64)


def compute_palm_pose_and_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    palm_offset_local: np.ndarray,
    arm_dof_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_rot = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    body_pos = np.array(data.xpos[body_id], dtype=np.float64)
    palm_pos = body_pos + body_rot @ palm_offset_local
    palm_quat = mat_to_quat_wxyz(body_rot)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, palm_pos, body_id)
    jac = np.vstack([jacp[:, arm_dof_ids], jacr[:, arm_dof_ids]])
    return palm_pos, palm_quat, jac


def arm_limit_metrics(q: np.ndarray, limits: np.ndarray) -> tuple[float, np.ndarray, bool]:
    margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    near_mask = margins <= 0.02
    violation = bool(np.any(margins < -1e-9))
    return float(np.min(margins)), near_mask, violation


def assign_arm_qpos(data: mujoco.MjData, stand_qpos: np.ndarray, arm_qpos_ids: np.ndarray, arm_q: np.ndarray) -> None:
    data.qpos[:] = stand_qpos
    data.qpos[arm_qpos_ids] = arm_q
    data.qvel[:] = 0.0


def current_bimanual_state(model_info: dict[str, object], data: mujoco.MjData) -> dict[str, np.ndarray]:
    arm_dof_ids = model_info["arm_dof_ids"]
    left_pos, left_quat, left_jac = compute_palm_pose_and_jacobian(
        model_info["model"], data, model_info["left_body_id"], LEFT_PALM_OFFSET, arm_dof_ids[:7]
    )
    right_pos, right_quat, right_jac = compute_palm_pose_and_jacobian(
        model_info["model"], data, model_info["right_body_id"], RIGHT_PALM_OFFSET, arm_dof_ids[7:]
    )
    return {
        "left_pos": left_pos,
        "left_quat": left_quat,
        "right_pos": right_pos,
        "right_quat": right_quat,
        "left_jac": left_jac,
        "right_jac": right_jac,
    }


def build_error_and_jacobian(
    current: dict[str, np.ndarray],
    target_left_pos: np.ndarray,
    target_left_quat: np.ndarray,
    target_right_pos: np.ndarray,
    target_right_quat: np.ndarray,
    position_weight: float,
    orientation_weight: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    left_rot = quat_to_mat(current["left_quat"])
    right_rot = quat_to_mat(current["right_quat"])
    left_target_rot = quat_to_mat(target_left_quat)
    right_target_rot = quat_to_mat(target_right_quat)

    left_pos_err = target_left_pos - current["left_pos"]
    right_pos_err = target_right_pos - current["right_pos"]
    left_rot_err_vec = rotation_log_world(left_rot, left_target_rot)
    right_rot_err_vec = rotation_log_world(right_rot, right_target_rot)

    zero = np.zeros((6, 7), dtype=np.float64)
    left_block = np.hstack([current["left_jac"], zero])
    right_block = np.hstack([zero, current["right_jac"]])
    J = np.vstack([left_block, right_block])
    error = np.concatenate([left_pos_err, left_rot_err_vec, right_pos_err, right_rot_err_vec])
    weights = np.concatenate(
        [
            np.full(3, position_weight, dtype=np.float64),
            np.full(3, orientation_weight, dtype=np.float64),
            np.full(3, position_weight, dtype=np.float64),
            np.full(3, orientation_weight, dtype=np.float64),
        ]
    )
    J_weighted = weights[:, None] * J
    error_weighted = weights * error
    return (
        error_weighted,
        J_weighted,
        float(np.linalg.norm(left_pos_err)),
        float(np.linalg.norm(right_pos_err)),
        float(np.linalg.norm(left_rot_err_vec)),
        float(np.linalg.norm(right_rot_err_vec)),
    )


def mode_success(mode_index: int, left_pos_err: float, right_pos_err: float, left_ori_err: float, right_ori_err: float, pos_tol: float, ori_tol: float) -> bool:
    if mode_index in (0, 1):
        return left_pos_err <= pos_tol and right_pos_err <= pos_tol and left_ori_err <= ori_tol and right_ori_err <= ori_tol
    return left_pos_err <= pos_tol and right_pos_err <= pos_tol


def solve_sparse_frame(
    model_info: dict[str, object],
    target_left_pos: np.ndarray,
    target_left_quat: np.ndarray,
    target_right_pos: np.ndarray,
    target_right_quat: np.ndarray,
    init_arm_q: np.ndarray,
    previous_arm_q: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    model = model_info["model"]
    stand_qpos = model_info["stand_qpos"]
    arm_qpos_ids = model_info["arm_qpos_ids"]
    joint_limits = model_info["joint_limits"]
    neutral_arm_q = model_info["stand_arm_q"]
    data = mujoco.MjData(model)
    best_overall: dict[str, object] | None = None

    mode_orientation_weights = [args.orientation_weight, 0.03, 0.0]
    attempt_initials = [init_arm_q, neutral_arm_q]

    for mode_index, mode_ori_w in enumerate(mode_orientation_weights):
        for initial_q in attempt_initials:
            q = np.array(initial_q, dtype=np.float64, copy=True)
            best_mode_result: dict[str, object] | None = None
            for iteration in range(1, args.max_iterations + 1):
                assign_arm_qpos(data, stand_qpos, arm_qpos_ids, q)
                mujoco.mj_forward(model, data)
                current = current_bimanual_state(model_info, data)
                error, J, left_pos_err, right_pos_err, left_ori_err, right_ori_err = build_error_and_jacobian(
                    current,
                    target_left_pos,
                    target_left_quat,
                    target_right_pos,
                    target_right_quat,
                    args.position_weight,
                    mode_ori_w,
                )
                pos_score = left_pos_err + right_pos_err
                limit_margin, near_mask, violation = arm_limit_metrics(q, joint_limits)
                result = {
                    "arm_q": q.copy(),
                    "left_pos": current["left_pos"].copy(),
                    "left_quat": current["left_quat"].copy(),
                    "right_pos": current["right_pos"].copy(),
                    "right_quat": current["right_quat"].copy(),
                    "left_pos_err": left_pos_err,
                    "right_pos_err": right_pos_err,
                    "left_ori_err": left_ori_err,
                    "right_ori_err": right_ori_err,
                    "iteration_count": iteration,
                    "mode_index": mode_index,
                    "limit_margin": limit_margin,
                    "near_mask": near_mask.copy(),
                    "violation": violation,
                    "converged": mode_success(mode_index, left_pos_err, right_pos_err, left_ori_err, right_ori_err, args.position_tolerance_m, args.orientation_tolerance_rad),
                }
                if best_mode_result is None or pos_score < (best_mode_result["left_pos_err"] + best_mode_result["right_pos_err"]):
                    best_mode_result = result
                if result["converged"] and not violation:
                    return result

                reg_rows = []
                reg_error = []
                prev_w = math.sqrt(0.02)
                neutral_w = math.sqrt(0.005)
                reg_rows.append(prev_w * np.eye(14, dtype=np.float64))
                reg_error.append(prev_w * (previous_arm_q - q))
                reg_rows.append(neutral_w * np.eye(14, dtype=np.float64))
                reg_error.append(neutral_w * (neutral_arm_q - q))
                J_aug = np.vstack([J, *reg_rows])
                error_aug = np.concatenate([error, *reg_error])
                system = J_aug @ J_aug.T + (args.damping ** 2) * np.eye(J_aug.shape[0], dtype=np.float64)
                dq = J_aug.T @ np.linalg.solve(system, error_aug)
                dq = np.clip(dq, -args.max_joint_step_rad, args.max_joint_step_rad)
                q = np.clip(q + dq, joint_limits[:, 0], joint_limits[:, 1])

            if best_mode_result is not None:
                if best_overall is None or (best_mode_result["left_pos_err"] + best_mode_result["right_pos_err"]) < (best_overall["left_pos_err"] + best_overall["right_pos_err"]):
                    best_overall = best_mode_result

    if best_overall is None:
        raise fail("Sparse IK produced no best-effort result")
    best_overall = dict(best_overall)
    best_overall["mode_index"] = 3
    best_overall["converged"] = False
    return best_overall


def add_geom_capacity(scn: mujoco.MjvScene, needed: int) -> bool:
    return scn.ngeom + needed <= scn.maxgeom


def add_sphere(scn: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba: tuple[float, float, float, float]) -> None:
    if not add_geom_capacity(scn, 1):
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def add_line(scn: mujoco.MjvScene, start: np.ndarray, end: np.ndarray, width: float, rgba: tuple[float, float, float, float]) -> None:
    if not add_geom_capacity(scn, 1):
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_makeConnector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        float(start[0]),
        float(start[1]),
        float(start[2]),
        float(end[0]),
        float(end[1]),
        float(end[2]),
    )
    scn.ngeom += 1


def add_axes(scn: mujoco.MjvScene, pos: np.ndarray, quat: np.ndarray, length: float = 0.05) -> None:
    rot = quat_to_mat(quat)
    colors = [
        (1.0, 0.2, 0.2, 1.0),
        (0.2, 1.0, 0.2, 1.0),
        (0.2, 0.4, 1.0, 1.0),
    ]
    for axis in range(3):
        add_line(scn, pos, pos + length * rot[:, axis], 0.004, colors[axis])


def configure_viewer_camera(viewer: object) -> None:
    cam = getattr(viewer, "cam", None)
    if cam is None:
        return
    cam.azimuth = 138.0
    cam.elevation = -18.0
    cam.distance = 2.1
    cam.lookat[:] = np.array([0.0, 0.0, 0.85], dtype=np.float64)


def viewer_running(viewer: object) -> bool:
    checker = getattr(viewer, "is_running", None)
    return bool(checker()) if callable(checker) else True


def render_targets_sequence(model_info: dict[str, object], payload: dict[str, np.ndarray], sample_indices_arr: np.ndarray, frame_duration_s: float) -> None:
    if mujoco_viewer is None:
        raise fail("mujoco.viewer is unavailable")
    model = model_info["model"]
    data = mujoco.MjData(model)
    with mujoco_viewer.launch_passive(model, data) as viewer:
        configure_viewer_camera(viewer)
        while viewer_running(viewer):
            for frame_idx in sample_indices_arr:
                if not viewer_running(viewer):
                    return
                assign_arm_qpos(data, model_info["stand_qpos"], model_info["arm_qpos_ids"], model_info["stand_arm_q"])
                mujoco.mj_forward(model, data)
                current = current_bimanual_state(model_info, data)
                scn = viewer.user_scn
                scn.ngeom = 0
                left_target_pos = payload["g1_left_target_position_xyz_m"][frame_idx]
                right_target_pos = payload["g1_right_target_position_xyz_m"][frame_idx]
                left_target_quat = payload["g1_left_target_quaternion_wxyz"][frame_idx]
                right_target_quat = payload["g1_right_target_quaternion_wxyz"][frame_idx]
                add_sphere(scn, left_target_pos, 0.015, (1.0, 0.4, 0.2, 0.9))
                add_sphere(scn, right_target_pos, 0.015, (0.2, 0.6, 1.0, 0.9))
                add_axes(scn, left_target_pos, left_target_quat)
                add_axes(scn, right_target_pos, right_target_quat)
                add_sphere(scn, current["left_pos"], 0.012, (1.0, 0.8, 0.2, 0.9))
                add_sphere(scn, current["right_pos"], 0.012, (0.2, 1.0, 1.0, 0.9))
                add_line(scn, current["left_pos"], left_target_pos, 0.003, (1.0, 0.8, 0.2, 1.0))
                add_line(scn, current["right_pos"], right_target_pos, 0.003, (0.2, 1.0, 1.0, 1.0))
                viewer.sync()
                end_time = time.perf_counter() + frame_duration_s
                while viewer_running(viewer) and time.perf_counter() < end_time:
                    time.sleep(0.01)


def render_ik_sequence(model_info: dict[str, object], result: dict[str, np.ndarray], frame_duration_s: float) -> None:
    if mujoco_viewer is None:
        raise fail("mujoco.viewer is unavailable")
    model = model_info["model"]
    data = mujoco.MjData(model)
    with mujoco_viewer.launch_passive(model, data) as viewer:
        configure_viewer_camera(viewer)
        while viewer_running(viewer):
            for idx in range(result["sample_frame_indices"].shape[0]):
                if not viewer_running(viewer):
                    return
                assign_arm_qpos(data, model_info["stand_qpos"], model_info["arm_qpos_ids"], result["g1_bimanual_arm_q"][idx])
                mujoco.mj_forward(model, data)
                scn = viewer.user_scn
                scn.ngeom = 0
                add_sphere(scn, result["target_left_position_xyz_m"][idx], 0.015, (1.0, 0.4, 0.2, 0.9))
                add_sphere(scn, result["target_right_position_xyz_m"][idx], 0.015, (0.2, 0.6, 1.0, 0.9))
                add_axes(scn, result["target_left_position_xyz_m"][idx], result["target_left_quaternion_wxyz"][idx])
                add_axes(scn, result["target_right_position_xyz_m"][idx], result["target_right_quaternion_wxyz"][idx])
                add_sphere(scn, result["achieved_left_position_xyz_m"][idx], 0.012, (1.0, 0.8, 0.2, 0.9))
                add_sphere(scn, result["achieved_right_position_xyz_m"][idx], 0.012, (0.2, 1.0, 1.0, 0.9))
                add_axes(scn, result["achieved_left_position_xyz_m"][idx], result["achieved_left_quaternion_wxyz"][idx])
                add_axes(scn, result["achieved_right_position_xyz_m"][idx], result["achieved_right_quaternion_wxyz"][idx])
                add_line(scn, result["achieved_left_position_xyz_m"][idx], result["target_left_position_xyz_m"][idx], 0.003, (1.0, 0.8, 0.2, 1.0))
                add_line(scn, result["achieved_right_position_xyz_m"][idx], result["target_right_position_xyz_m"][idx], 0.003, (0.2, 1.0, 1.0, 1.0))
                viewer.sync()
                end_time = time.perf_counter() + frame_duration_s
                while viewer_running(viewer) and time.perf_counter() < end_time:
                    time.sleep(0.01)


def run_sparse_ik(
    model_info: dict[str, object],
    payload: dict[str, np.ndarray],
    sample_indices_arr: np.ndarray,
    args: argparse.Namespace,
    model_path: Path,
) -> dict[str, np.ndarray]:
    sample_count = int(sample_indices_arr.shape[0])
    left_q = np.zeros((sample_count, 7), dtype=np.float64)
    right_q = np.zeros((sample_count, 7), dtype=np.float64)
    both_q = np.zeros((sample_count, 14), dtype=np.float64)
    converged = np.zeros(sample_count, dtype=bool)
    ik_mode = np.empty(sample_count, dtype="<U20")
    iteration_count = np.zeros(sample_count, dtype=np.int64)
    left_pos_err = np.zeros(sample_count, dtype=np.float64)
    right_pos_err = np.zeros(sample_count, dtype=np.float64)
    left_ori_err = np.zeros(sample_count, dtype=np.float64)
    right_ori_err = np.zeros(sample_count, dtype=np.float64)
    min_margin = np.zeros(sample_count, dtype=np.float64)
    near_mask = np.zeros((sample_count, 14), dtype=bool)
    achieved_left_pos = np.zeros((sample_count, 3), dtype=np.float64)
    achieved_left_quat = np.zeros((sample_count, 4), dtype=np.float64)
    achieved_right_pos = np.zeros((sample_count, 3), dtype=np.float64)
    achieved_right_quat = np.zeros((sample_count, 4), dtype=np.float64)

    previous_solution = np.asarray(model_info["stand_arm_q"], dtype=np.float64)
    for out_idx, frame_idx in enumerate(sample_indices_arr):
        targets = {
            "left_pos": payload["g1_left_target_position_xyz_m"][frame_idx],
            "left_quat": payload["g1_left_target_quaternion_wxyz"][frame_idx],
            "right_pos": payload["g1_right_target_position_xyz_m"][frame_idx],
            "right_quat": payload["g1_right_target_quaternion_wxyz"][frame_idx],
        }
        init_q = previous_solution if out_idx > 0 else np.asarray(model_info["stand_arm_q"], dtype=np.float64)
        result = solve_sparse_frame(
            model_info,
            targets["left_pos"],
            targets["left_quat"],
            targets["right_pos"],
            targets["right_quat"],
            init_q,
            previous_solution,
            args,
        )
        both_q[out_idx] = result["arm_q"]
        left_q[out_idx] = result["arm_q"][:7]
        right_q[out_idx] = result["arm_q"][7:]
        converged[out_idx] = bool(result["converged"])
        ik_mode[out_idx] = MODE_NAMES[int(result["mode_index"])]
        iteration_count[out_idx] = int(result["iteration_count"])
        left_pos_err[out_idx] = float(result["left_pos_err"])
        right_pos_err[out_idx] = float(result["right_pos_err"])
        left_ori_err[out_idx] = float(result["left_ori_err"])
        right_ori_err[out_idx] = float(result["right_ori_err"])
        min_margin[out_idx] = float(result["limit_margin"])
        near_mask[out_idx] = np.asarray(result["near_mask"], dtype=bool)
        achieved_left_pos[out_idx] = result["left_pos"]
        achieved_left_quat[out_idx] = result["left_quat"]
        achieved_right_pos[out_idx] = result["right_pos"]
        achieved_right_quat[out_idx] = result["right_quat"]
        previous_solution = np.asarray(result["arm_q"], dtype=np.float64)
        print(f"frame {int(frame_idx)}")
        print(f"  IK mode: {ik_mode[out_idx]}")
        print(f"  converged: {bool(converged[out_idx])}")
        print(f"  iteration count: {int(iteration_count[out_idx])}")
        print(f"  left position error: {left_pos_err[out_idx]:.6f}")
        print(f"  right position error: {right_pos_err[out_idx]:.6f}")
        print(f"  left orientation error: {left_ori_err[out_idx]:.6f}")
        print(f"  right orientation error: {right_ori_err[out_idx]:.6f}")
        print(f"  left arm q: {left_q[out_idx].tolist()}")
        print(f"  right arm q: {right_q[out_idx].tolist()}")
        print(f"  minimum joint-limit margin: {min_margin[out_idx]:.6f}")

    return {
        "sample_frame_indices": sample_indices_arr,
        "timestamp_s": payload["timestamp_s"][sample_indices_arr],
        "g1_left_arm_q": left_q,
        "g1_right_arm_q": right_q,
        "g1_bimanual_arm_q": both_q,
        "g1_arm_joint_names": np.asarray(model_info["joint_names"]),
        "converged": converged,
        "ik_mode": ik_mode,
        "iteration_count": iteration_count,
        "left_position_error_m": left_pos_err,
        "right_position_error_m": right_pos_err,
        "left_orientation_error_rad": left_ori_err,
        "right_orientation_error_rad": right_ori_err,
        "joint_limit_min_margin_rad": min_margin,
        "joint_limit_near_mask": near_mask,
        "target_left_position_xyz_m": payload["g1_left_target_position_xyz_m"][sample_indices_arr],
        "target_left_quaternion_wxyz": payload["g1_left_target_quaternion_wxyz"][sample_indices_arr],
        "target_right_position_xyz_m": payload["g1_right_target_position_xyz_m"][sample_indices_arr],
        "target_right_quaternion_wxyz": payload["g1_right_target_quaternion_wxyz"][sample_indices_arr],
        "achieved_left_position_xyz_m": achieved_left_pos,
        "achieved_left_quaternion_wxyz": achieved_left_quat,
        "achieved_right_position_xyz_m": achieved_right_pos,
        "achieved_right_quaternion_wxyz": achieved_right_quat,
        "model_path": np.asarray(str(model_path)),
        "neutral_keyframe": np.asarray("stand"),
        "solver": np.asarray(SOLVER_NAME),
        "damping": np.asarray(args.damping, dtype=np.float64),
        "max_iterations": np.asarray(args.max_iterations, dtype=np.int64),
        "max_joint_step_rad": np.asarray(args.max_joint_step_rad, dtype=np.float64),
        "position_weight": np.asarray(args.position_weight, dtype=np.float64),
        "orientation_weight": np.asarray(args.orientation_weight, dtype=np.float64),
        "position_tolerance_m": np.asarray(args.position_tolerance_m, dtype=np.float64),
        "orientation_tolerance_rad": np.asarray(args.orientation_tolerance_rad, dtype=np.float64),
        "left_palm_offset_m": LEFT_PALM_OFFSET,
        "right_palm_offset_m": RIGHT_PALM_OFFSET,
        "quaternion_order": np.asarray("wxyz"),
    }


def validate_output(result: dict[str, np.ndarray], model_info: dict[str, object]) -> None:
    n = int(result["sample_frame_indices"].shape[0])
    if result["g1_left_arm_q"].shape != (n, 7) or result["g1_right_arm_q"].shape != (n, 7) or result["g1_bimanual_arm_q"].shape != (n, 14):
        raise fail("IK joint array shapes are invalid")
    limits = np.asarray(model_info["joint_limits"], dtype=np.float64)
    q = result["g1_bimanual_arm_q"]
    if not np.all(q >= limits[:, 0] - 1e-9) or not np.all(q <= limits[:, 1] + 1e-9):
        raise fail("Some IK joints are outside limits")
    for key in [
        "g1_left_arm_q",
        "g1_right_arm_q",
        "g1_bimanual_arm_q",
        "left_position_error_m",
        "right_position_error_m",
        "left_orientation_error_rad",
        "right_orientation_error_rad",
        "joint_limit_min_margin_rad",
        "target_left_position_xyz_m",
        "target_left_quaternion_wxyz",
        "target_right_position_xyz_m",
        "target_right_quaternion_wxyz",
        "achieved_left_position_xyz_m",
        "achieved_left_quaternion_wxyz",
        "achieved_right_position_xyz_m",
        "achieved_right_quaternion_wxyz",
    ]:
        if not np.isfinite(result[key]).all():
            raise fail(f"{key} contains NaN or inf")
    for key in [
        "target_left_quaternion_wxyz",
        "target_right_quaternion_wxyz",
        "achieved_left_quaternion_wxyz",
        "achieved_right_quaternion_wxyz",
    ]:
        norms = np.linalg.norm(result[key], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise fail(f"{key} quaternion norms are invalid")
    if np.any(result["left_position_error_m"] < 0.0) or np.any(result["right_position_error_m"] < 0.0):
        raise fail("Position error contains negative values")
    if np.any(result["left_orientation_error_rad"] < 0.0) or np.any(result["right_orientation_error_rad"] < 0.0):
        raise fail("Orientation error contains negative values")


def print_report(result_path: Path) -> None:
    data = np.load(result_path, allow_pickle=False)
    print(f"saved keys: {sorted(data.files)}")
    for key in ["sample_frame_indices", "timestamp_s", "g1_left_arm_q", "g1_right_arm_q", "g1_bimanual_arm_q", "converged", "ik_mode", "joint_limit_near_mask"]:
        print(f"{key}: shape={data[key].shape} dtype={data[key].dtype}")
    print(f"file size bytes: {result_path.stat().st_size}")
    print(f"frame-wise converged: {data['converged'].tolist()}")
    modes = {}
    for item in data["ik_mode"].tolist():
        modes[item] = modes.get(item, 0) + 1
    print(f"mode counts: {modes}")
    print(f"position error mean/max: {float(np.mean(np.r_[data['left_position_error_m'], data['right_position_error_m']])):.6f} / {float(np.max(np.r_[data['left_position_error_m'], data['right_position_error_m']])):.6f}")
    print(f"orientation error mean/max: {float(np.mean(np.r_[data['left_orientation_error_rad'], data['right_orientation_error_rad']])):.6f} / {float(np.max(np.r_[data['left_orientation_error_rad'], data['right_orientation_error_rad']])):.6f}")
    print(f"minimum joint-limit margin: {float(np.min(data['joint_limit_min_margin_rad'])):.6f}")
    non_converged = data["sample_frame_indices"][~data["converged"]]
    print(f"non-converged frames: {non_converged.tolist()}")


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise fail(f"Output already exists. Use --overwrite to replace: {output_path}")

    payload, frame_count = load_and_validate_input(input_path)
    model_info = validate_model(model_path)
    sample_indices_arr = sample_frame_indices(frame_count)
    print(f"Representative frames: {sample_indices_arr.tolist()}")
    print(f"Stand keyframe arm q: {np.asarray(model_info['stand_arm_q']).tolist()}")
    print("Actual joint limits:")
    for name, limits in zip(model_info["joint_names"], model_info["joint_limits"]):
        print(f"  {name}: {limits.tolist()}")

    if not args.no_viewer and args.mode in ("targets", "both"):
        render_targets_sequence(model_info, payload, sample_indices_arr, args.frame_duration_s)

    if args.mode == "targets":
        return 0

    result = run_sparse_ik(model_info, payload, sample_indices_arr, args, model_path)
    validate_output(result, model_info)
    np.savez(output_path, **result)
    print_report(output_path)

    if not args.no_viewer and args.mode in ("ik", "both"):
        render_ik_sequence(model_info, result, args.frame_duration_s)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
