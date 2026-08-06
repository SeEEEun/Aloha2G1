"""GUI preview: unchanged MagSafe magnetic v2 scene plus Stationary ALOHA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent
ALOHA_USD = Path(
    "/home/jbnu/robot_assets/stationary_aloha/usd_imported/stationary_aloha_imported.usd"
)
OUTPUT = ROOT / "generated" / "magsafe_aloha_model_preview.usda"
EPISODE = Path(
    "/home/jbnu/aloha_g1_dataset/raw_recordings/GoPark_20260723_162750/"
    "data/chunk-000/episode_000000.parquet"
)
EPISODE_INFO = EPISODE.parents[2] / "meta" / "info.json"
DATASET_NAMES = [
    *(f"left_joint_{index}" for index in range(7)),
    *(f"right_joint_{index}" for index in range(7)),
]
USD_NAMES = [
    *(f"follower_left_joint_{index}" for index in range(6)),
    "follower_left_right_carriage_joint",
    "follower_left_left_carriage_joint",
    *(f"follower_right_joint_{index}" for index in range(6)),
    "follower_right_right_carriage_joint",
    "follower_right_left_carriage_joint",
]

parser = argparse.ArgumentParser(description="Static/default-pose Stationary ALOHA model preview.")
parser.add_argument("--camera", choices=("overview", "front", "side", "top"), default="overview")
parser.add_argument("--pose", choices=("default", "episode_frame0"), default="episode_frame0")
parser.add_argument("--hold-seconds", type=float, default=None, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robot_model_preview_common import (
    compose_stage,
    print_stage_report,
    run_viewer,
    suppress_stationary_aloha_fixture,
)


def main() -> None:
    stage = compose_stage(OUTPUT, "StationaryALOHA", ALOHA_USD, "stationary_aloha")
    suppress_stationary_aloha_fixture(stage)
    print(f"[PREVIEW] output={OUTPUT}")
    print("[PREVIEW] source MJCF tabletop/frame suppressed only in preview layer; imported USD is unchanged")
    print_stage_report(stage, "/World/StationaryALOHA", ALOHA_USD)
    initializer = _episode_frame0_initializer if args_cli.pose == "episode_frame0" else None
    print(f"[PREVIEW] pose={args_cli.pose}")
    run_viewer(simulation_app, OUTPUT, args_cli.camera, args_cli.hold_seconds, initializer)


def _load_frame0() -> list[float]:
    import pyarrow.parquet as pq

    metadata = json.loads(EPISODE_INFO.read_text())
    names = metadata["features"]["observation.state"]["names"]
    if names != DATASET_NAMES:
        raise RuntimeError(f"Unexpected observation.state ordering: {names}")
    table = pq.ParquetFile(EPISODE).read_row_group(
        0, columns=["observation.state", "frame_index"]
    )
    if table.num_rows < 1 or int(table["frame_index"][0].as_py()) != 0:
        raise RuntimeError(f"First parquet row is not frame 0: {EPISODE}")
    values = [float(value) for value in table["observation.state"][0].as_py()]
    if len(values) != 14:
        raise RuntimeError(f"Frame 0 observation.state must have 14 values, got {len(values)}")
    return values


def _episode_frame0_initializer(sim) -> None:
    import inspect
    import math
    import torch
    from isaaclab.assets import Articulation, ArticulationCfg

    frame0 = _load_frame0()
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/StationaryALOHA/Asset/Geometry/tabletop_link",
            spawn=None,
            actuators={},
        )
    )
    sim.reset()
    state_writer_signature = inspect.signature(robot.write_joint_state_to_sim)
    print(f"[POSE] Articulation.write_joint_state_to_sim{state_writer_signature}")
    actual_names = list(robot.data.joint_names)
    missing = [name for name in USD_NAMES if name not in actual_names]
    if missing:
        raise RuntimeError(
            f"Dataset/MJCF mapping is not safe; imported USD joints are missing: {missing}. "
            f"Actual USD order: {actual_names}"
        )
    # Dataset order is six revolute joints plus one gripper value per side.
    # Each gripper value is duplicated to the imported left/right carriage pair,
    # matching stationary_ai.xml's equality/mimic constraint.
    left_gripper = min(max(frame0[6], 0.0), 0.044)
    right_gripper = min(max(frame0[13], 0.0), 0.044)
    mapped = [
        *frame0[0:6],
        left_gripper,
        left_gripper,
        *frame0[7:13],
        right_gripper,
        right_gripper,
    ]
    if not math.isclose(right_gripper, frame0[13], abs_tol=1e-12):
        print(f"[POSE] right gripper clipped from {frame0[13]:.9f} to {right_gripper:.9f}")
    joint_pos = robot.data.default_joint_pos.torch.clone().to(
        device=robot.device, dtype=torch.float32
    )
    joint_vel = torch.zeros_like(joint_pos)
    expected_shape = (1, 16)
    if tuple(joint_pos.shape) != expected_shape:
        raise RuntimeError(
            f"Expected joint_pos shape {expected_shape}, got {tuple(joint_pos.shape)}"
        )
    if joint_vel.shape != joint_pos.shape:
        raise RuntimeError(
            f"joint_vel shape {tuple(joint_vel.shape)} does not match "
            f"joint_pos shape {tuple(joint_pos.shape)}"
        )
    if joint_pos.device != torch.device(robot.device) or joint_vel.device != torch.device(
        robot.device
    ):
        raise RuntimeError(
            f"Joint-state device mismatch: robot={robot.device}, "
            f"position={joint_pos.device}, velocity={joint_vel.device}"
        )
    if joint_pos.dtype != torch.float32 or joint_vel.dtype != torch.float32:
        raise RuntimeError(
            f"Joint-state dtype must be float32, got "
            f"position={joint_pos.dtype}, velocity={joint_vel.dtype}"
        )
    mapping_lines = []
    for usd_name, value in zip(USD_NAMES, mapped, strict=True):
        usd_index = actual_names.index(usd_name)
        joint_pos[0, usd_index] = value
        mapping_lines.append(f"{usd_name}={value:.9f}")
    robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel)
    robot.reset()
    robot.update(0.0)
    for _ in range(2):
        sim.step(render=True)
    sim.pause()
    print(f"[POSE] source={EPISODE} frame=0 values={frame0}")
    print(f"[POSE] imported_usd_joint_order={actual_names}")
    print("[POSE] applied " + ", ".join(mapping_lines))
    print(
        f"[POSE] joint_pos_shape={tuple(joint_pos.shape)} "
        f"joint_vel_shape={tuple(joint_vel.shape)} "
        f"device={joint_pos.device} dtype={joint_pos.dtype}"
    )
    print("[POSE] simulation_steps=2 timeline=PAUSED trajectory_replay=OFF")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
