"""Named-joint Stationary ALOHA FK and task-space feature extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TCP_OFFSET_DEFAULT = np.asarray((0.1487, 0.0, -0.00105), dtype=np.float64)


def _matrix_from_quat_wxyz(value: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(value, dtype=np.float64) / np.linalg.norm(value)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    vector = np.asarray((
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    )) * 0.5
    sine = float(np.linalg.norm(vector))
    angle = float(np.arctan2(sine, cosine))
    if sine < 1e-12:
        return vector
    return vector * (angle / sine)


def _body_id(model: Any, name: str) -> int:
    import mujoco

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body_id < 0:
        raise KeyError(f"MuJoCo body missing: {name}")
    return body_id


def _joint_qpos_address(model: Any, name: str) -> int:
    import mujoco

    joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    if joint_id < 0:
        raise KeyError(f"MuJoCo joint missing: {name}")
    return int(model.jnt_qposadr[joint_id])


def stationary_aloha_joint_mapping(model: Any) -> dict[str, Any]:
    left_arm = [f"follower_left_joint_{index}" for index in range(6)]
    right_arm = [f"follower_right_joint_{index}" for index in range(6)]
    left_gripper = ["follower_left_right_carriage_joint", "follower_left_left_carriage_joint"]
    right_gripper = ["follower_right_right_carriage_joint", "follower_right_left_carriage_joint"]
    names = left_arm + left_gripper + right_arm + right_gripper
    addresses = {name: _joint_qpos_address(model, name) for name in names}
    if len(set(addresses.values())) != len(addresses):
        raise RuntimeError("duplicate qpos address in Stationary ALOHA mapping")
    return {
        "left_arm_joint_names": left_arm,
        "right_arm_joint_names": right_arm,
        "left_gripper_joint_names": left_gripper,
        "right_gripper_joint_names": right_gripper,
        "joint_name_to_qpos_address": addresses,
        "dataset_channel_order": [
            *(f"left_joint_{index}" for index in range(7)),
            *(f"right_joint_{index}" for index in range(7)),
        ],
    }


def compute_stationary_aloha_fk(
    action: np.ndarray,
    timestamps: np.ndarray,
    model_xml: str | Path,
    root_position_xyz_m: np.ndarray | None = None,
    root_orientation_wxyz: np.ndarray | None = None,
    tcp_offset: np.ndarray = TCP_OFFSET_DEFAULT,
) -> dict[str, Any]:
    import mujoco

    action = np.asarray(action, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    model = mujoco.MjModel.from_xml_path(str(Path(model_xml).resolve()))
    data = mujoco.MjData(model)
    mapping = stationary_aloha_joint_mapping(model)
    left_body = _body_id(model, "follower_left_link_6")
    right_body = _body_id(model, "follower_right_link_6")
    root_rotation = np.eye(3) if root_orientation_wxyz is None else _matrix_from_quat_wxyz(root_orientation_wxyz)
    root_position = np.zeros(3) if root_position_xyz_m is None else np.asarray(root_position_xyz_m, dtype=np.float64)
    tcp_offset = np.asarray(tcp_offset, dtype=np.float64)
    positions = {side: np.empty((len(action), 3), dtype=np.float64) for side in ("left", "right")}
    rotations = {side: np.empty((len(action), 3, 3), dtype=np.float64) for side in ("left", "right")}
    qpos = np.empty((len(action), model.nq), dtype=np.float64)
    addresses = mapping["joint_name_to_qpos_address"]
    for sample_index, row in enumerate(action):
        data.qpos[:] = 0.0
        for joint_index, joint_name in enumerate(mapping["left_arm_joint_names"]):
            data.qpos[addresses[joint_name]] = row[joint_index]
        for joint_name in mapping["left_gripper_joint_names"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            low, high = model.jnt_range[joint_id]
            data.qpos[addresses[joint_name]] = np.clip(row[6], low, high)
        for joint_index, joint_name in enumerate(mapping["right_arm_joint_names"]):
            data.qpos[addresses[joint_name]] = row[7 + joint_index]
        for joint_name in mapping["right_gripper_joint_names"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            low, high = model.jnt_range[joint_id]
            data.qpos[addresses[joint_name]] = np.clip(row[13], low, high)
        mujoco.mj_forward(model, data)
        qpos[sample_index] = data.qpos
        for side, body_id in (("left", left_body), ("right", right_body)):
            local_rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            local_position = np.asarray(data.xpos[body_id], dtype=np.float64) + local_rotation @ tcp_offset
            positions[side][sample_index] = root_position + root_rotation @ local_position
            rotations[side][sample_index] = root_rotation @ local_rotation
    return {
        "left_tcp_position": positions["left"],
        "right_tcp_position": positions["right"],
        "left_tcp_rotation": rotations["left"],
        "right_tcp_rotation": rotations["right"],
        "qpos": qpos,
        "joint_mapping": mapping,
        "tcp_offset_m": tcp_offset,
        "root_position_xyz_m": root_position,
        "root_orientation_wxyz": np.asarray(root_orientation_wxyz if root_orientation_wxyz is not None else (1, 0, 0, 0)),
        "timestamps": timestamps,
    }


def _kinematics(position: np.ndarray, rotation: np.ndarray, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    velocity = np.gradient(position, timestamps, axis=0, edge_order=1)
    acceleration = np.gradient(velocity, timestamps, axis=0, edge_order=1)
    speed = np.linalg.norm(velocity, axis=1)
    angular_delta = np.zeros((len(position), 3), dtype=np.float64)
    for index in range(1, len(position)):
        angular_delta[index] = _rotation_vector(rotation[index - 1].T @ rotation[index])
    angular_velocity = angular_delta / np.maximum(np.gradient(timestamps)[:, None], 1e-12)
    angular_speed = np.linalg.norm(angular_velocity, axis=1)
    step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    cumulative_path = np.concatenate(([0.0], np.cumsum(step)))
    cumulative_rotation = np.concatenate(([0.0], np.cumsum(np.linalg.norm(angular_delta[1:], axis=1))))
    tangent = velocity / np.maximum(speed[:, None], 1e-12)
    tangent_rate = np.gradient(tangent, timestamps, axis=0, edge_order=1)
    curvature = np.linalg.norm(tangent_rate, axis=1) / np.maximum(speed, 1e-6)
    return {
        "linear_velocity": velocity,
        "linear_acceleration": acceleration,
        "linear_speed": speed,
        "angular_velocity": angular_velocity,
        "angular_speed": angular_speed,
        "cumulative_path_length": cumulative_path,
        "cumulative_rotation": cumulative_rotation,
        "velocity_direction": tangent,
        "curvature": np.clip(curvature, 0.0, np.quantile(curvature, 0.995) if len(curvature) else 0.0),
        "displacement_from_start": np.linalg.norm(position - position[0], axis=1),
    }


def extract_task_space_features(
    fk_trajectory: dict[str, Any],
    timestamps: np.ndarray,
    task_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    result: dict[str, Any] = {}
    for side in ("left", "right"):
        position = np.asarray(fk_trajectory[f"{side}_tcp_position"], dtype=np.float64)
        rotation = np.asarray(fk_trajectory[f"{side}_tcp_rotation"], dtype=np.float64)
        result[f"{side}_tcp_position"] = position
        result[f"{side}_tcp_rotation"] = rotation
        for name, value in _kinematics(position, rotation, timestamps).items():
            result[f"{side}_{name}"] = value
    left = result["left_tcp_position"]
    right = result["right_tcp_position"]
    result["bimanual_midpoint"] = (left + right) * 0.5
    result["right_left_vector"] = right - left
    result["inter_hand_distance"] = np.linalg.norm(right - left, axis=1)
    result["relative_velocity"] = result["right_linear_velocity"] - result["left_linear_velocity"]
    left_rotation = result["left_tcp_rotation"]
    right_rotation = result["right_tcp_rotation"]
    relative_rotation = np.empty_like(left_rotation)
    for index in range(len(left_rotation)):
        relative_rotation[index] = left_rotation[index].T @ right_rotation[index]
    result["bimanual_relative_rotation"] = relative_rotation
    if task_geometry:
        phone_region = task_geometry.get("initial_phone_left_grasp_region_world")
        if phone_region is not None:
            center = np.asarray(phone_region["center_xyz_m"], dtype=np.float64)
            result["left_phone_region_distance"] = np.linalg.norm(left - center, axis=1)
        charger_direction = task_geometry.get("charger_task_direction_world")
        if charger_direction is not None:
            direction = np.asarray(charger_direction, dtype=np.float64)
            direction /= max(np.linalg.norm(direction), 1e-12)
            result["left_charger_direction_velocity"] = result["left_linear_velocity"] @ direction
    return result
