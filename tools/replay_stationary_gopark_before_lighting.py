#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mujoco
import numpy as np
import pandas as pd

try:
    from mujoco import viewer as mujoco_viewer
except ImportError:
    mujoco_viewer = None


DEFAULT_DATASET = Path("/home/jbnu/aloha_g1_dataset/GoPark")
DEFAULT_MODEL = Path(
    "/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml"
)
MODEL_SEARCH_ROOTS = (
    Path(
        "/home/jbnu/miniconda3/envs/trossen_ai_data_collection_ui_env/lib/python3.10/site-packages"
    ),
    Path("/home/jbnu/.lerobot_trossen_ai_data_collection_ui"),
)
EXPECTED_ACTUATORS = [
    "follower_left_joint_0",
    "follower_left_joint_1",
    "follower_left_joint_2",
    "follower_left_joint_3",
    "follower_left_joint_4",
    "follower_left_joint_5",
    "follower_left_gripper",
    "follower_right_joint_0",
    "follower_right_joint_1",
    "follower_right_joint_2",
    "follower_right_joint_3",
    "follower_right_joint_4",
    "follower_right_joint_5",
    "follower_right_gripper",
]
EXPECTED_QPOS_JOINTS = [
    "follower_left_joint_0",
    "follower_left_joint_1",
    "follower_left_joint_2",
    "follower_left_joint_3",
    "follower_left_joint_4",
    "follower_left_joint_5",
    "follower_left_right_carriage_joint",
    "follower_left_left_carriage_joint",
    "follower_right_joint_0",
    "follower_right_joint_1",
    "follower_right_joint_2",
    "follower_right_joint_3",
    "follower_right_joint_4",
    "follower_right_joint_5",
    "follower_right_right_carriage_joint",
    "follower_right_left_carriage_joint",
]
EXPECTED_BODIES = [
    "follower_left_link_6",
    "follower_right_link_6",
]
TCP_OFFSET_LOCAL = np.array([0.1487, 0.0, -0.00105], dtype=np.float64)
GRIPPER_MIN = 0.0
GRIPPER_MAX = 0.044


@dataclass(frozen=True)
class EpisodeData:
    positions: np.ndarray
    timestamps: np.ndarray | None
    frame_index: np.ndarray
    fps: float | None
    parquet_path: Path


@dataclass(frozen=True)
class TimingInfo:
    frame_dt: np.ndarray
    source_label: str
    nominal_fps: float
    expected_duration_seconds: float


@dataclass(frozen=True)
class FkPose:
    position_xyz: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class FrameFkSample:
    frame_index: int
    left: FkPose
    right: FkPose
    delta_right_minus_left_xyz: np.ndarray
    distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kinematic replay for a GoPark episode on the Trossen AI Stationary MuJoCo model."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--source",
        choices=("observation.state", "action"),
        default="observation.state",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-viewer", action="store_true")
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
    with info_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def episode_parquet_path(dataset_dir: Path, info: dict, episode: int) -> Path:
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    episode_chunk = episode // int(info.get("chunks_size", 1000))
    relative = template.format(episode_chunk=episode_chunk, episode_index=episode)
    return dataset_dir / relative


def _stack_series_column(series: pd.Series, column_name: str) -> np.ndarray:
    values = [np.asarray(value, dtype=np.float64) for value in series.to_list()]
    array = np.stack(values, axis=0)
    if array.ndim != 2 or array.shape[1] != 14:
        raise fail(
            f"Column '{column_name}' must have shape (T, 14), but got {array.shape} from {len(values)} rows."
        )
    return array


def load_episode_data(dataset_dir: Path, episode: int, source: str) -> EpisodeData:
    require_dir(dataset_dir, "dataset directory")
    info = load_info_json(dataset_dir)
    parquet_path = require_file(episode_parquet_path(dataset_dir, info, episode), "episode parquet")
    dataframe = pd.read_parquet(parquet_path, columns=[source, "timestamp", "frame_index"])

    missing_columns = [name for name in (source, "timestamp", "frame_index") if name not in dataframe.columns]
    if missing_columns:
        raise fail(f"Parquet is missing required columns: {missing_columns}")

    positions = _stack_series_column(dataframe[source], source)
    if not np.isfinite(positions).all():
        raise fail(f"Column '{source}' contains NaN or inf values.")

    timestamps = np.asarray(dataframe["timestamp"], dtype=np.float64).reshape(-1)
    if timestamps.shape[0] != positions.shape[0]:
        raise fail("Timestamp length does not match trajectory length.")

    frame_index = np.asarray(dataframe["frame_index"], dtype=np.int64).reshape(-1)
    if frame_index.shape[0] != positions.shape[0]:
        raise fail("frame_index length does not match trajectory length.")

    fps_value = info.get("fps")
    fps = float(fps_value) if fps_value is not None else None
    return EpisodeData(
        positions=positions,
        timestamps=timestamps,
        frame_index=frame_index,
        fps=fps,
        parquet_path=parquet_path,
    )


def compute_timing(episode_data: EpisodeData, speed: float) -> TimingInfo:
    if speed <= 0.0 or not math.isfinite(speed):
        raise fail(f"--speed must be a positive finite value, but got {speed}.")

    timestamps = episode_data.timestamps
    use_timestamp = False
    if timestamps is not None and timestamps.size >= 2 and np.isfinite(timestamps).all():
        diffs = np.diff(timestamps)
        if diffs.size > 0 and np.all(diffs > 0.0) and np.isfinite(diffs).all():
            use_timestamp = True

    if use_timestamp:
        base_dt = np.diff(timestamps)
        frame_dt = np.concatenate(([base_dt[0]], base_dt))
        mean_dt = float(np.mean(base_dt))
        nominal_fps = 1.0 / mean_dt
        source_label = f"mean timestamp dt={mean_dt:.6f}s"
        expected_duration_seconds = float(np.sum(base_dt) / speed)
    else:
        if episode_data.fps is None or episode_data.fps <= 0.0 or not math.isfinite(episode_data.fps):
            raise fail("Timestamps are invalid and dataset fps is missing or invalid in meta/info.json.")
        dt = 1.0 / episode_data.fps
        frame_dt = np.full(episode_data.positions.shape[0], dt, dtype=np.float64)
        nominal_fps = float(episode_data.fps)
        source_label = f"fps={episode_data.fps:.6f}"
        expected_duration_seconds = float(max(episode_data.positions.shape[0] - 1, 0) * dt / speed)

    return TimingInfo(
        frame_dt=frame_dt,
        source_label=source_label,
        nominal_fps=nominal_fps,
        expected_duration_seconds=expected_duration_seconds,
    )


def actuator_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"<unnamed:{i}>" for i in range(model.nu)]


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<unnamed:{i}>" for i in range(model.njnt)]


def body_names(model: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<unnamed:{i}>" for i in range(model.nbody)]


def _validate_name_exists(model: mujoco.MjModel, obj_type: mujoco._enums.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        if obj_type == mujoco.mjtObj.mjOBJ_BODY:
            actual = body_names(model)
            label = "body"
        elif obj_type == mujoco.mjtObj.mjOBJ_JOINT:
            actual = joint_names(model)
            label = "joint"
        else:
            actual = actuator_names(model)
            label = "actuator"
        raise fail(f"Expected {label} '{name}' was not found. Actual {label} names: {actual}")
    return obj_id


def validate_model_or_raise(model: mujoco.MjModel) -> None:
    actual_actuators = actuator_names(model)
    if actual_actuators != EXPECTED_ACTUATORS:
        raise fail(
            "Actuator order mismatch.\n"
            f"Expected: {EXPECTED_ACTUATORS}\n"
            f"Actual:   {actual_actuators}"
        )

    if model.nq < len(EXPECTED_QPOS_JOINTS):
        raise fail(f"Model nq={model.nq} is too small for the required 16-qpos mapping.")

    for name in EXPECTED_QPOS_JOINTS:
        _validate_name_exists(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in EXPECTED_BODIES:
        _validate_name_exists(model, mujoco.mjtObj.mjOBJ_BODY, name)

    for expected_qpos_index, joint_name in enumerate(EXPECTED_QPOS_JOINTS):
        joint_id = _validate_name_exists(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        if qpos_adr != expected_qpos_index:
            raise fail(
                f"Joint '{joint_name}' qpos index mismatch: expected {expected_qpos_index}, got {qpos_adr}."
            )


def find_stationary_xml_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in MODEL_SEARCH_ROOTS:
        if not root.exists():
            continue
        try:
            candidates.extend(root.rglob("stationary_ai.xml"))
        except PermissionError:
            continue
    unique_sorted = sorted({path.resolve() for path in candidates})
    return unique_sorted


def load_validated_model(model_path: Path | None) -> tuple[mujoco.MjModel, Path]:
    if model_path is not None:
        resolved_model_path = model_path.expanduser().resolve()
        if not resolved_model_path.exists():
            raise fail(f"--model path does not exist: {resolved_model_path}")
        if not resolved_model_path.is_file():
            raise fail(f"--model path is not a file: {resolved_model_path}")
        try:
            model = mujoco.MjModel.from_xml_path(str(resolved_model_path))
            validate_model_or_raise(model)
            return model, resolved_model_path
        except Exception as exc:
            raise fail(f"Failed to load or validate model from --model {resolved_model_path}: {exc}") from exc

    candidates = find_stationary_xml_candidates()
    if not candidates:
        roots = [str(root) for root in MODEL_SEARCH_ROOTS]
        raise fail(f"Could not find stationary_ai.xml under search roots: {roots}")

    errors: list[str] = []
    for xml_path in candidates:
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            validate_model_or_raise(model)
            return model, xml_path
        except Exception as exc:
            errors.append(f"{xml_path}: {exc}")

    joined = "\n".join(errors)
    raise fail(f"Found stationary_ai.xml candidates, but none passed validation:\n{joined}")


def map_row_to_qpos(row: np.ndarray) -> tuple[np.ndarray, bool, bool]:
    if row.shape != (14,):
        raise fail(f"Expected frame vector shape (14,), but got {row.shape}.")

    left_gripper = float(np.clip(row[6], GRIPPER_MIN, GRIPPER_MAX))
    right_gripper = float(np.clip(row[13], GRIPPER_MIN, GRIPPER_MAX))
    left_clipped = not math.isclose(left_gripper, float(row[6]), rel_tol=0.0, abs_tol=1e-12)
    right_clipped = not math.isclose(right_gripper, float(row[13]), rel_tol=0.0, abs_tol=1e-12)
    qpos = np.array(
        [
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            left_gripper,
            left_gripper,
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
            row[12],
            right_gripper,
            right_gripper,
        ],
        dtype=np.float64,
    )
    return qpos, left_clipped, right_clipped


def qpos_trajectory(positions: np.ndarray) -> tuple[np.ndarray, int]:
    qpos_frames = np.zeros((positions.shape[0], 16), dtype=np.float64)
    clip_count = 0
    for index, row in enumerate(positions):
        qpos, left_clipped, right_clipped = map_row_to_qpos(row)
        qpos_frames[index] = qpos
        if left_clipped or right_clipped:
            clip_count += 1
    return qpos_frames, clip_count


def configure_viewer_camera(viewer: object) -> None:
    cam = getattr(viewer, "cam", None)
    if cam is None:
        return
    cam.azimuth = 135.0
    cam.elevation = -24.0
    cam.distance = 1.9
    cam.lookat[:] = np.array([0.25, 0.0, 0.18], dtype=np.float64)


def mj_rotation_to_quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation_matrix.reshape(9))
    return quat


def compute_tcp_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> FkPose:
    body_id = _validate_name_exists(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    body_position = np.array(data.xpos[body_id], dtype=np.float64)
    body_rotation = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    tcp_position = body_position + body_rotation @ TCP_OFFSET_LOCAL
    tcp_quaternion = mj_rotation_to_quat_wxyz(body_rotation)
    return FkPose(position_xyz=tcp_position, quaternion_wxyz=tcp_quaternion)


def sample_fk_frames(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_frames: np.ndarray,
    desired_indices: Sequence[int],
) -> list[FrameFkSample]:
    samples: list[FrameFkSample] = []
    valid_indices = sorted({index for index in desired_indices if 0 <= index < qpos_frames.shape[0]})
    for index in valid_indices:
        data.qpos[:] = 0.0
        data.qpos[:16] = qpos_frames[index]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        left = compute_tcp_pose(model, data, "follower_left_link_6")
        right = compute_tcp_pose(model, data, "follower_right_link_6")
        delta = right.position_xyz - left.position_xyz
        distance = float(np.linalg.norm(delta))
        samples.append(
            FrameFkSample(
                frame_index=index,
                left=left,
                right=right,
                delta_right_minus_left_xyz=delta,
                distance=distance,
            )
        )
    return samples


def format_vector(vector: np.ndarray) -> str:
    return np.array2string(np.asarray(vector, dtype=np.float64), precision=6, separator=", ")


def print_summary(
    xml_path: Path,
    episode_data: EpisodeData,
    source: str,
    timing: TimingInfo,
    positions: np.ndarray,
    clip_count: int,
) -> None:
    joint_mins = positions.min(axis=0)
    joint_maxs = positions.max(axis=0)
    print(f"stationary_ai.xml: {xml_path}")
    print(f"dataset: {episode_data.parquet_path.parent.parent.parent}")
    print(f"episode: {episode_data.parquet_path.stem.split('_')[-1]}")
    print(f"source: {source}")
    print(f"frame count: {positions.shape[0]}")
    print(f"timing: {timing.source_label}")
    print(f"nominal fps: {timing.nominal_fps:.6f}")
    print(f"expected replay time (speed-adjusted): {timing.expected_duration_seconds:.3f} s")
    print("joint min/max:")
    for index in range(positions.shape[1]):
        print(f"  [{index:02d}] min={joint_mins[index]: .6f} max={joint_maxs[index]: .6f}")
    print(f"gripper clip frames: {clip_count}")


def print_fk_samples(samples: Iterable[FrameFkSample]) -> None:
    print("FK samples:")
    for sample in samples:
        print(f"frame {sample.frame_index}:")
        print(f"  left TCP xyz: {format_vector(sample.left.position_xyz)}")
        print(f"  left quat wxyz: {format_vector(sample.left.quaternion_wxyz)}")
        print(f"  right TCP xyz: {format_vector(sample.right.position_xyz)}")
        print(f"  right quat wxyz: {format_vector(sample.right.quaternion_wxyz)}")
        print(f"  right-left xyz: {format_vector(sample.delta_right_minus_left_xyz)}")
        print(f"  tcp distance: {sample.distance:.6f}")


def viewer_is_running(viewer: object) -> bool:
    checker = getattr(viewer, "is_running", None)
    if callable(checker):
        return bool(checker())
    return True


def replay_with_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_frames: np.ndarray,
    timing: TimingInfo,
    speed: float,
) -> None:
    if mujoco_viewer is None:
        raise fail("mujoco.viewer is unavailable in this environment; use --no-viewer.")

    with mujoco_viewer.launch_passive(model, data) as viewer:
        configure_viewer_camera(viewer)
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        viewer.sync()
        previous_wall_time = time.perf_counter()

        for frame_index, qpos in enumerate(qpos_frames):
            if not viewer_is_running(viewer):
                print("Viewer closed. Exiting replay.")
                return

            data.qpos[:] = 0.0
            data.qpos[:16] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            viewer.sync()

            if frame_index + 1 >= qpos_frames.shape[0]:
                continue

            wait_seconds = float(timing.frame_dt[frame_index + 1] / speed)
            target_time = previous_wall_time + wait_seconds
            while True:
                if not viewer_is_running(viewer):
                    print("Viewer closed. Exiting replay.")
                    return
                now = time.perf_counter()
                remaining = target_time - now
                if remaining <= 0.0:
                    previous_wall_time = target_time
                    break
                time.sleep(min(remaining, 0.005))


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.expanduser().resolve()
    episode_data = load_episode_data(dataset_dir, args.episode, args.source)
    timing = compute_timing(episode_data, args.speed)
    model, xml_path = load_validated_model(args.model)
    data = mujoco.MjData(model)
    qpos_frames, clip_count = qpos_trajectory(episode_data.positions)

    print_summary(xml_path, episode_data, args.source, timing, episode_data.positions, clip_count)
    sample_indices = [0, 100, 300, 500, qpos_frames.shape[0] - 1]
    samples = sample_fk_frames(model, data, qpos_frames, sample_indices)
    print_fk_samples(samples)

    if args.no_viewer:
        return 0

    replay_with_viewer(model, data, qpos_frames, timing, args.speed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
