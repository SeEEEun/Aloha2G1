#!/usr/bin/env python3
"""Visual-only Stationary ALOHA replay of episode-49 SmolVLA H10 output."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from replay_stationary_gopark import (  # noqa: E402
    configure_viewer_camera,
    load_validated_model,
    map_row_to_qpos,
)

DEFAULT_TRAJECTORY = Path(
    "/home/jbnu/aloha_g1_dataset/evaluation/mujoco_stationary_aloha_validation/"
    "episode_000049/smolvla_h10_trajectory.npz"
)
DEFAULT_XML = Path(
    "/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/"
    "stationary_ai/stationary_ai.xml"
)
FPS = 30.0

# GLFW key codes used by mujoco.viewer callbacks.
KEY_SPACE = 32
KEY_R = 82
KEY_L = 76
KEY_RIGHT = 262
KEY_LEFT = 263


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--key", default="raw")
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def load_trajectory(path: Path, key: str) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise RuntimeError(f"Trajectory key '{key}' not found; actual keys={data.files}")
        raw = np.asarray(data[key], dtype=np.float64)
        stored_mapped = np.asarray(data["mapped_qpos"], dtype=np.float64) if "mapped_qpos" in data.files else None
    if raw.ndim != 2 or raw.shape[1] != 14:
        raise RuntimeError(f"Trajectory '{key}' must have shape [T,14], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise RuntimeError(f"Trajectory '{key}' contains NaN/Inf")
    return raw, stored_mapped


def map_with_existing_adapter(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Call the established replay mapping without modifying the raw array."""
    mapped = np.empty((len(raw), 16), dtype=np.float64)
    mapped_gripper_frames = 0
    for index, row in enumerate(raw):
        qpos, left_mapped, right_mapped = map_row_to_qpos(row)
        mapped[index] = qpos
        mapped_gripper_frames += int(left_mapped or right_mapped)
    return mapped, mapped_gripper_frames


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        raise ValueError("--speed must be positive and finite")
    raw, stored_mapped = load_trajectory(args.trajectory.resolve(), args.key)
    qpos, mapped_gripper_frames = map_with_existing_adapter(raw)
    if stored_mapped is not None:
        if stored_mapped.shape != qpos.shape or not np.allclose(stored_mapped, qpos, atol=1e-7, rtol=0):
            raise RuntimeError("Recomputed established mapping differs from stored mapped_qpos")

    model, xml_path = load_validated_model(DEFAULT_XML)
    data = __import__("mujoco").MjData(model)
    from mujoco import mj_forward, viewer

    end = len(qpos) - 1 if args.end_frame is None else args.end_frame
    if not (0 <= args.start_frame <= end < len(qpos)):
        raise ValueError(f"Invalid range {args.start_frame}:{end}; trajectory frames={len(qpos)}")

    paused = False
    looping = bool(args.loop)
    restart_requested = False
    step_requested = 0

    def key_callback(keycode: int) -> None:
        nonlocal paused, looping, restart_requested, step_requested
        if keycode == KEY_SPACE:
            paused = not paused
        elif keycode == KEY_LEFT:
            paused = True
            step_requested = -1
        elif keycode == KEY_RIGHT:
            paused = True
            step_requested = 1
        elif keycode in (KEY_R, ord("r")):
            restart_requested = True
        elif keycode in (KEY_L, ord("l")):
            looping = not looping
            print(f"\nloop={'ON' if looping else 'OFF'}")

    print(f"XML: {xml_path}")
    print(f"trajectory: {args.trajectory.resolve()}")
    print(f"key={args.key}; raw_shape={raw.shape}; mapped_qpos_shape={qpos.shape}")
    print(f"fps={FPS}; speed={args.speed}; range={args.start_frame}:{end}")
    print(f"existing gripper mapping affected {mapped_gripper_frames} frames; raw input remains unchanged")
    print("SPACE pause/resume | LEFT/RIGHT one frame | R restart | L loop on/off")

    frame = args.start_frame
    next_tick = time.perf_counter()
    with viewer.launch_passive(model, data, key_callback=key_callback) as window:
        configure_viewer_camera(window)
        while window.is_running():
            if restart_requested:
                frame = args.start_frame
                restart_requested = False
                next_tick = time.perf_counter()
            if step_requested:
                frame = min(max(frame + step_requested, args.start_frame), end)
                step_requested = 0

            data.qpos[:] = 0.0
            data.qpos[:16] = qpos[frame]
            data.qvel[:] = 0.0
            mj_forward(model, data)
            window.sync()
            seconds = frame / FPS
            print(
                f"frame {frame:04d}/{end:04d} | time {seconds:7.3f}s | "
                f"{'PAUSED' if paused else 'PLAY'} | loop={'ON' if looping else 'OFF'}",
                end="\r",
                flush=True,
            )

            if paused:
                time.sleep(0.02)
                next_tick = time.perf_counter()
                continue

            if frame >= end:
                if looping:
                    frame = args.start_frame
                else:
                    paused = True
                    print(f"\nReached end frame {end}; paused.")
                next_tick = time.perf_counter()
                continue

            frame += 1
            next_tick += 1.0 / (FPS * args.speed)
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.perf_counter()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
