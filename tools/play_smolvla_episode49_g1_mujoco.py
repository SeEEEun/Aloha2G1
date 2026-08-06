#!/usr/bin/env python3
"""Visual-only G1 MuJoCo qpos player for the safety-blocked episode 49 result."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_TRAJECTORY = Path(
    "/home/jbnu/aloha_g1_dataset/converted_runs/"
    "smolvla_20k_episode49_h10_g1/g1_smolvla_episode49_h10_trajectory.npz"
)
DEFAULT_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
QPOS_KEY = "full_g1_joint_trajectory"
WIDTH, HEIGHT = 640, 480


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--pause-at", type=int, nargs="*", default=[])
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--video-output", type=Path)
    return p.parse_args()


def load(path: Path) -> tuple[np.ndarray, int, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        if QPOS_KEY not in z.files:
            raise RuntimeError(f"Missing {QPOS_KEY}; actual keys={z.files}")
        qpos = z[QPOS_KEY].astype(np.float64)
        task_start = int(z["task_start_frame"])
        fps = float(z["fps"])
    if qpos.ndim != 2 or qpos.shape[1] != 50 or not np.isfinite(qpos).all():
        raise RuntimeError(f"{QPOS_KEY} must be finite [T,50], got {qpos.shape}")
    return qpos, task_start, fps


def frame_label(frame: int, task_start: int) -> str:
    if frame < 90:
        segment = "approach"
    elif frame < task_start:
        segment = "hold"
    else:
        segment = "task"
    task_frame = frame - task_start
    task_text = str(task_frame) if task_frame >= 0 else "N/A"
    warning = "  REVIEW" if task_frame in {240, 249, 328, 329, 330, 331, 332} else ""
    return f"full {frame} | task {task_text} | {segment}{warning}"


def set_qpos(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    """Kinematics only: no ctrl assignment and no mj_step."""
    if model.nq != q.shape[0]:
        raise RuntimeError(f"Model nq={model.nq}, trajectory width={q.shape[0]}")
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.azimuth = 138.0
    cam.elevation = -18.0
    cam.distance = 2.1
    cam.lookat[:] = np.array([0.0, 0.0, 0.85])
    return cam


def add_overlay(rgb: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 430, 48), fill=(0, 0, 0))
    draw.text((22, 21), text, fill=(255, 255, 255))
    return np.asarray(image)


def record(model: mujoco.MjModel, qpos: np.ndarray, frames: range, task_start: int,
           source_fps: float, speed: float, output: Path) -> None:
    if output is None:
        raise RuntimeError("--video-output is required with --record-video")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    data = mujoco.MjData(model)
    cam = camera()
    output_fps = 30
    repeats = max(1, int(round(output_fps / (source_fps * speed))))
    cmd = [
        "ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(output_fps), "-i", "pipe:0",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-y", str(output),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for i, frame in enumerate(frames):
            set_qpos(model, data, qpos[frame])
            renderer.update_scene(data, camera=cam)
            rgb = add_overlay(renderer.render(), frame_label(frame, task_start))
            for _ in range(repeats):
                process.stdin.write(rgb.tobytes())
            if i % 30 == 0:
                print(frame_label(frame, task_start), flush=True)
        process.stdin.close()
        code = process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with exit code {code}")
    finally:
        renderer.close()
        if process.poll() is None:
            process.terminate()


def view(model: mujoco.MjModel, qpos: np.ndarray, frames: range, task_start: int,
         fps: float, speed: float, loop: bool, pause_at: set[int]) -> None:
    from mujoco import viewer
    data = mujoco.MjData(model)
    paused = False

    def callback(key: int) -> None:
        nonlocal paused
        if key == 32:
            paused = not paused

    with viewer.launch_passive(model, data, key_callback=callback) as window:
        window.cam.azimuth = 138.0
        window.cam.elevation = -18.0
        window.cam.distance = 2.1
        window.cam.lookat[:] = [0.0, 0.0, 0.85]
        print("Kinematic qpos viewer only. SPACE pause/resume; close window to exit.")
        while window.is_running():
            for frame in frames:
                while window.is_running() and paused:
                    window.sync(); time.sleep(.02)
                if not window.is_running():
                    return
                set_qpos(model, data, qpos[frame])
                label = frame_label(frame, task_start)
                print(label, end="\r", flush=True)
                window.sync()
                if frame in pause_at:
                    print(f"\nPaused at {label}; press SPACE to continue.")
                    paused = True
                deadline = time.perf_counter() + 1.0/(fps*speed)
                while window.is_running() and not paused and time.perf_counter() < deadline:
                    time.sleep(min(.005, deadline-time.perf_counter()))
            if not loop:
                print()
                return


def main() -> int:
    args = parse_args()
    if not (args.speed > 0 and np.isfinite(args.speed)):
        raise ValueError("--speed must be positive and finite")
    qpos, task_start, fps = load(args.trajectory.resolve())
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    if model.nq != qpos.shape[1]:
        raise RuntimeError(f"Existing G1 model/trajectory mismatch: nq={model.nq}, shape={qpos.shape}")
    end = len(qpos)-1 if args.end_frame is None else args.end_frame
    if not (0 <= args.start_frame <= end < len(qpos)):
        raise ValueError(f"Invalid frame range {args.start_frame}:{end} for {len(qpos)} frames")
    frames = range(args.start_frame, end+1)
    print(f"model={DEFAULT_MODEL}")
    print(f"trajectory_key={QPOS_KEY}; shape={qpos.shape}; task_start_frame={task_start}; fps={fps}")
    if args.record_video:
        record(model, qpos, frames, task_start, fps, args.speed, args.video_output)
    else:
        view(model, qpos, frames, task_start, fps, args.speed, args.loop, set(args.pause_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
