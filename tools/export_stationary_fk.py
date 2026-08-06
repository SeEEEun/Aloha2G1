#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np


DEFAULT_DATASET = Path("/home/jbnu/aloha_g1_dataset/GoPark")
DEFAULT_MODEL = Path(
    "/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml"
)
DEFAULT_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/aloha_tcp_trajectory.npz")
PYARROW_PYTHON = Path("/home/jbnu/miniconda3/envs/trossen_ai_data_collection_ui_env/bin/python")
TCP_OFFSET_LOCAL = np.array([0.1487, 0.0, -0.00105], dtype=np.float64)
EXPECTED_BODIES = ["follower_left_link_6", "follower_right_link_6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stationary ALOHA TCP FK trajectory to NPZ.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--source", choices=("observation.state", "action"), default="observation.state")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def require_dir(path: Path, description: str) -> Path:
    if not path.exists():
        raise fail(f"Missing {description}: {path}")
    if not path.is_dir():
        raise fail(f"Expected {description} to be a directory: {path}")
    return path


def load_info_json(dataset_dir: Path) -> dict:
    info_path = require_file(dataset_dir / "meta" / "info.json", "dataset info.json")
    return json.loads(info_path.read_text(encoding="utf-8"))


def episode_parquet_path(dataset_dir: Path, info: dict, episode: int) -> Path:
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk_size = int(info.get("chunks_size", 1000))
    return dataset_dir / template.format(episode_chunk=episode // chunk_size, episode_index=episode)


def load_parquet_via_pyarrow(parquet_path: Path) -> dict:
    require_file(PYARROW_PYTHON, "pyarrow helper python")
    helper = f"""
import json
import pyarrow.parquet as pq
table = pq.read_table({str(parquet_path)!r}, columns=['observation.state','action','timestamp','frame_index'])
print(json.dumps(table.to_pydict()))
"""
    result = subprocess.run(
        [str(PYARROW_PYTHON), "-c", helper],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def as_array_2d(values: list, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 14:
        raise fail(f"{name} must have shape (T, 14), but got {array.shape}")
    if not np.isfinite(array).all():
        raise fail(f"{name} contains NaN or inf")
    return array


def as_array_1d(values: list, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if not np.isfinite(array.astype(np.float64)).all():
        raise fail(f"{name} contains NaN or inf")
    return array


def validate_model(model: mujoco.MjModel) -> tuple[int, int]:
    left_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EXPECTED_BODIES[0])
    right_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EXPECTED_BODIES[1])
    if left_body < 0 or right_body < 0:
        raise fail(f"Required FK bodies not found: {EXPECTED_BODIES}")
    if model.nq < 16:
        raise fail(f"Model nq={model.nq} is smaller than required 16")
    return left_body, right_body


def map_to_qpos(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    left_gripper = float(np.clip(x[6], 0.0, 0.044))
    right_gripper = float(np.clip(x[13], 0.0, 0.044))
    qpos = np.array(
        [
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            left_gripper,
            left_gripper,
            x[7],
            x[8],
            x[9],
            x[10],
            x[11],
            x[12],
            right_gripper,
            right_gripper,
        ],
        dtype=np.float64,
    )
    return qpos, left_gripper, right_gripper


def mat_to_quat_wxyz(mat3: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat3.reshape(9))
    return quat


def compute_trajectory(model: mujoco.MjModel, positions: np.ndarray) -> dict[str, np.ndarray]:
    left_body_id, right_body_id = validate_model(model)
    data = mujoco.MjData(model)
    frame_count = positions.shape[0]
    left_pos = np.zeros((frame_count, 3), dtype=np.float64)
    right_pos = np.zeros((frame_count, 3), dtype=np.float64)
    left_quat = np.zeros((frame_count, 4), dtype=np.float64)
    right_quat = np.zeros((frame_count, 4), dtype=np.float64)
    left_gripper = np.zeros(frame_count, dtype=np.float64)
    right_gripper = np.zeros(frame_count, dtype=np.float64)

    for index, x in enumerate(positions):
        qpos, left_g, right_g = map_to_qpos(x)
        data.qpos[:] = 0.0
        data.qpos[:16] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        left_rot = np.array(data.xmat[left_body_id], dtype=np.float64).reshape(3, 3)
        right_rot = np.array(data.xmat[right_body_id], dtype=np.float64).reshape(3, 3)
        left_pos[index] = np.array(data.xpos[left_body_id], dtype=np.float64) + left_rot @ TCP_OFFSET_LOCAL
        right_pos[index] = np.array(data.xpos[right_body_id], dtype=np.float64) + right_rot @ TCP_OFFSET_LOCAL
        left_quat[index] = mat_to_quat_wxyz(left_rot)
        right_quat[index] = mat_to_quat_wxyz(right_rot)
        left_gripper[index] = left_g
        right_gripper[index] = right_g

    rel = right_pos - left_pos
    distance = np.linalg.norm(rel, axis=1)
    return {
        "left_position_xyz_m": left_pos,
        "left_quaternion_wxyz": left_quat,
        "right_position_xyz_m": right_pos,
        "right_quaternion_wxyz": right_quat,
        "left_gripper_m": left_gripper,
        "right_gripper_m": right_gripper,
        "right_minus_left_position_xyz_m": rel,
        "hands_distance_m": distance,
    }


def validate_before_save(payload: dict[str, np.ndarray]) -> None:
    frame_count = payload["timestamp_s"].shape[0]
    for key, value in payload.items():
        if value.shape[0] != frame_count:
            raise fail(f"{key} first dimension mismatch: expected {frame_count}, got {value.shape}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise fail(f"{key} contains NaN or inf")
    if frame_count >= 2 and not np.all(np.diff(payload["timestamp_s"]) > 0.0):
        raise fail("timestamp_s is not strictly increasing")
    for key in ["left_quaternion_wxyz", "right_quaternion_wxyz"]:
        norms = np.linalg.norm(payload[key], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            raise fail(f"{key} is not normalized within tolerance")
    if np.any(payload["hands_distance_m"] < 0.0):
        raise fail("hands_distance_m contains negative values")


def print_validation_report(path: Path) -> None:
    data = np.load(path, allow_pickle=False)
    frame_count = int(data["timestamp_s"].shape[0])
    duration = float(data["timestamp_s"][-1] - data["timestamp_s"][0]) if frame_count > 1 else 0.0
    print(f"output: {path}")
    print(f"frame count: {frame_count}")
    print(f"duration_s: {duration:.6f}")
    for key in [
        "timestamp_s",
        "frame_index",
        "source_joint_state",
        "action",
        "observation_state",
        "left_position_xyz_m",
        "left_quaternion_wxyz",
        "right_position_xyz_m",
        "right_quaternion_wxyz",
        "left_gripper_m",
        "right_gripper_m",
        "right_minus_left_position_xyz_m",
        "hands_distance_m",
    ]:
        print(f"{key}: shape={data[key].shape} dtype={data[key].dtype}")
    for key in ["left_position_xyz_m", "left_quaternion_wxyz", "right_position_xyz_m", "right_quaternion_wxyz"]:
        if not np.isfinite(data[key]).all():
            raise fail(f"{key} contains NaN or inf after save")
    print(
        "quat norms:",
        float(np.linalg.norm(data["left_quaternion_wxyz"], axis=1).min()),
        float(np.linalg.norm(data["left_quaternion_wxyz"], axis=1).max()),
        float(np.linalg.norm(data["right_quaternion_wxyz"], axis=1).min()),
        float(np.linalg.norm(data["right_quaternion_wxyz"], axis=1).max()),
    )
    print(
        "gripper min/max:",
        float(data["left_gripper_m"].min()),
        float(data["left_gripper_m"].max()),
        float(data["right_gripper_m"].min()),
        float(data["right_gripper_m"].max()),
    )
    print(
        "hand distance min/max:",
        float(data["hands_distance_m"].min()),
        float(data["hands_distance_m"].max()),
    )
    print(
        "first left/right pose:",
        data["left_position_xyz_m"][0].tolist(),
        data["left_quaternion_wxyz"][0].tolist(),
        data["right_position_xyz_m"][0].tolist(),
        data["right_quaternion_wxyz"][0].tolist(),
    )
    print(
        "last left/right pose:",
        data["left_position_xyz_m"][-1].tolist(),
        data["left_quaternion_wxyz"][-1].tolist(),
        data["right_position_xyz_m"][-1].tolist(),
        data["right_quaternion_wxyz"][-1].tolist(),
    )


def main() -> int:
    args = parse_args()
    dataset_dir = require_dir(args.dataset.expanduser().resolve(), "dataset directory")
    model_path = require_file(args.model.expanduser().resolve(), "MuJoCo model")
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise fail(f"Output already exists. Use --overwrite to replace: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = load_info_json(dataset_dir)
    parquet_path = require_file(episode_parquet_path(dataset_dir, info, args.episode), "episode parquet")
    raw = load_parquet_via_pyarrow(parquet_path)
    observation_state = as_array_2d(raw["observation.state"], "observation.state")
    action = as_array_2d(raw["action"], "action")
    source_joint_state = as_array_2d(raw[args.source], args.source)
    timestamp = as_array_1d(raw["timestamp"], "timestamp", np.float64)
    frame_index = as_array_1d(raw["frame_index"], "frame_index", np.int64)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    fk = compute_trajectory(model, source_joint_state)
    payload = {
        "timestamp_s": timestamp,
        "frame_index": frame_index,
        "source_joint_state": source_joint_state,
        "action": action,
        "observation_state": observation_state,
        **fk,
    }
    validate_before_save(payload)
    np.savez(output_path, **payload)
    print_validation_report(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
