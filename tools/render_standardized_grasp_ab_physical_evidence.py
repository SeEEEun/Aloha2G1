#!/usr/bin/env python3
"""Render standardized-grasp mosaics from saved measured robot/PhysX states only."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.doll_handoff_retargeting.render import _add_box, _add_geom

OUT = ROOT / "outputs/standardized_grasp_ab_dev35"
COMMON = ROOT / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21/frozen_approval/config/common_config.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
RESULTS = OUT / "05_results/FINAL_STANDARDIZED_GRASP_NUMERIC_RESULTS.json"
REPLAYS = OUT / "07_physical_replays"
VISUALS = OUT / "08_paper_visuals"
WIDTH, HEIGHT = 540, 405
MWIDTH, MHEIGHT = 3840, 2160
COLS, ROWS = 7, 5
GX, GY = 30, 67
FPS = 30


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_image(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
    if not cv2.imwrite(str(temporary), value):
        raise RuntimeError(f"cannot write image {temporary}")
    os.replace(temporary, path)


def rotation_xyzw(value: Iterable[float]) -> np.ndarray:
    x, y, z, w = np.asarray(tuple(value), dtype=np.float64)
    x, y, z, w = np.asarray([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def make_camera(eye: Iterable[float], target: Iterable[float]) -> mujoco.MjvCamera:
    eye, target = np.asarray(tuple(eye)), np.asarray(tuple(target))
    relative = eye - target
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target; camera.distance = float(np.linalg.norm(relative))
    camera.azimuth = math.degrees(math.atan2(-relative[1], -relative[0]))
    camera.elevation = -math.degrees(math.atan2(relative[2], np.linalg.norm(relative[:2])))
    return camera


class PhysicalRenderer:
    def __init__(self) -> None:
        self.common = load_common_config(COMMON); self.layout = load_scene(self.common); self.config = read_json(CONFIG)
        self.g1 = G1Kinematics(self.common, self.layout); self.model = self.g1.model; self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=WIDTH, height=HEIGHT)
        contract_names = [str(row["joint_name"]) for row in read_json(CONTRACT)["joint_specs"]]
        model_names = [*map(str, self.g1.arm_joint_names), *self.g1.hand_joint_names["left"], *self.g1.hand_joint_names["right"]]
        lookup = {name: index for index, name in enumerate(contract_names)}
        if set(model_names) != set(contract_names): raise RuntimeError("saved-state/G1 renderer joint-name mismatch")
        self.reorder = np.asarray([lookup[name] for name in model_names])
        presets = self.layout["camera"]["presets"]
        self.camera_parameters = {"top": presets["top"], "overview": presets["overview"]}
        self.cameras = {name: make_camera(row["eye_world_xyz_m"], row["target_world_xyz_m"]) for name, row in self.camera_parameters.items()}
        self.root_position = np.asarray(self.layout["g1"]["root_position_world_xyz_m"])
        self.root_quaternion = np.asarray(self.layout["g1"]["root_orientation_world_wxyz"])
        self.model.vis.headlight.ambient[:] = (.52, .52, .52); self.model.vis.headlight.diffuse[:] = (.78, .78, .78); self.model.vis.headlight.specular[:] = (.08, .08, .08)

    def close(self) -> None: self.renderer.close()

    def set_state(self, q_contract: np.ndarray, position: np.ndarray, quaternion: np.ndarray) -> None:
        q = np.asarray(q_contract)[self.reorder]
        self.data.qpos[:] = self.g1.stand_qpos; self.data.qpos[self.g1.arm_qpos_ids] = q[:14]
        self.data.qpos[self.g1.hand_qpos_ids["left"]] = q[14:21]; self.data.qpos[self.g1.hand_qpos_ids["right"]] = q[21:28]
        self.data.qpos[:3] = self.root_position; self.data.qpos[3:7] = self.root_quaternion; self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data); self.doll_position = np.asarray(position); self.doll_rotation = rotation_xyzw(quaternion)

    def add_scene(self) -> None:
        scene, layout = self.renderer.scene, self.layout; table = layout["table"]
        surface = float(table["surface_height_m"]); width, depth = map(float, table["size_xy_m"]); thickness = float(table["top_thickness_m"])
        _add_box(scene, (width, depth, thickness), (.5 * width, .5 * depth, surface - .5 * thickness), np.asarray([.72, .72, .70, 1], np.float32))
        for rail in layout["black_frame"]["rails"].values(): _add_box(scene, rail["size_xyz_m"], rail["center_xyz_m"], np.asarray([.07, .07, .08, 1], np.float32))
        dimensions = np.asarray(self.config["object"]["visual_dimensions_m"])
        _add_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, .5 * dimensions, self.doll_position, np.asarray([.18, .62, .20, 1], np.float32), rotation=self.doll_rotation)
        cfg = layout["bin"]; ox, oy, _ = map(float, cfg["outer_dimensions_xyz_m"]); ix, iy = map(float, cfg["opening_dimensions_xy_m"])
        wall, bottom = float(cfg["wall_thickness_m"]), float(cfg["bottom_thickness_m"]); cx, cy = map(float, cfg["center_world_xy_m"]); color = np.asarray([.86, .83, .68, 1], np.float32); h = .150
        _add_box(scene, (ox, oy, bottom), (cx, cy, surface + .5 * bottom), color)
        _add_box(scene, (ox, wall, h), (cx, cy - .5 * (iy + wall), surface + .5 * h), color); _add_box(scene, (ox, wall, h), (cx, cy + .5 * (iy + wall), surface + .5 * h), color)
        _add_box(scene, (wall, iy, h), (cx - .5 * (ix + wall), cy, surface + .5 * h), color); _add_box(scene, (wall, iy, h), (cx + .5 * (ix + wall), cy, surface + .5 * h), color)
        _add_box(scene, (6, 6, .025), (.4175, .2, -.0125), np.asarray([.84, .85, .86, 1], np.float32)); scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    def view(self, name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, self.cameras[name]); self.add_scene()
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)


def run_dirs(method: str) -> list[Path]:
    directory = OUT / ("03_a_results" if method == "A" else "04_b_results") / "rollouts"
    values = sorted(path.parent for path in directory.glob("eval_*/RUN_MANIFEST.json"))
    if len(values) != 35: raise RuntimeError(f"{method}: {len(values)}/35 traces")
    return values


def control_trace(run: Path) -> dict[str, np.ndarray]:
    with np.load(run / "event_log.npz", allow_pickle=False) as archive: event = {key: np.asarray(archive[key]) for key in archive.files}
    frames = event["control_frame"].astype(int); rows = np.r_[np.flatnonzero(np.diff(frames) != 0), len(frames) - 1]
    return {"frame": frames[rows], "q": event["MEASURED_Q"][rows], "position": event["object_position_world_m"][rows], "quaternion": event["object_quaternion_xyzw"][rows]}


def state_at(trace: dict[str, np.ndarray], frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = int(np.argmin(np.abs(trace["frame"] - int(frame))))
    return trace["q"][index], trace["position"][index], trace["quaternion"][index]


def label(image: np.ndarray, method: str, display: int, failure: str, initial: bool = False) -> np.ndarray:
    result = image.copy(); cv2.rectangle(result, (0, 0), (WIDTH, 35), (18, 18, 22), -1)
    status = "STANDARDIZED START" if initial else ("SUCCESS" if failure == "SUCCESS" else f"FAIL · {failure}")
    cv2.putText(result, f"{method}{display:02d}/35  {status}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


class Writer:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True); self.path = path; self.tmp = path.with_name(path.stem + ".incomplete" + path.suffix)
        self.process = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{MWIDTH}x{MHEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", str(self.tmp)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None; self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def finish(self) -> None:
        assert self.process.stdin is not None; self.process.stdin.close()
        if self.process.wait() != 0: raise RuntimeError(f"ffmpeg failed: {self.path}")
        os.replace(self.tmp, self.path)


def probe(path: Path) -> dict[str, Any]:
    return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)], text=True))["streams"][0]


def render_mosaics() -> dict[str, Any]:
    renderer = PhysicalRenderer(); results = read_json(RESULTS); products: dict[str, Any] = {}
    directories = {method: run_dirs(method) for method in "AB"}
    traces = {method: [control_trace(run) for run in directories[method]] for method in "AB"}
    manifests = {method: [read_json(run / "RUN_MANIFEST.json") for run in directories[method]] for method in "AB"}
    maximum = max(len(trace["frame"]) for method in "AB" for trace in traces[method])
    writers = {
        "A_top": Writer(REPLAYS / "A_POST_GRASP_DEV35_TOP_35SPLIT.mp4"),
        "A_overview": Writer(REPLAYS / "A_POST_GRASP_DEV35_OVERVIEW_35SPLIT.mp4"),
        "B_top": Writer(REPLAYS / "B_POST_GRASP_DEV35_TOP_35SPLIT.mp4"),
        "B_overview": Writer(REPLAYS / "B_POST_GRASP_DEV35_OVERVIEW_35SPLIT.mp4"),
        "paired": Writer(REPLAYS / "AB_POST_GRASP_MATCHED_PHYSICAL_REVIEW.mp4"),
    }
    try:
        for frame_index in range(maximum):
            canvases = {key: np.full((MHEIGHT, MWIDTH, 3), 232, np.uint8) for key in writers}
            for cell in range(35):
                row, col = divmod(cell, COLS); x, y = GX + col * WIDTH, GY + row * HEIGHT
                views: dict[tuple[str, str], np.ndarray] = {}
                for method in "AB":
                    trace = traces[method][cell]; index = min(frame_index, len(trace["frame"]) - 1)
                    renderer.set_state(trace["q"][index], trace["position"][index], trace["quaternion"][index])
                    failure = manifests[method][cell]["first_failure_stage"]
                    for view in ("top", "overview"):
                        views[(method, view)] = label(renderer.view(view), method, cell + 1, failure, frame_index == 0)
                        canvases[f"{method}_{view}"][y:y+HEIGHT, x:x+WIDTH] = views[(method, view)]
                        cv2.rectangle(canvases[f"{method}_{view}"], (x, y), (x + WIDTH - 1, y + HEIGHT - 1), (45, 45, 45), 1)
                pair = np.concatenate([cv2.resize(views[("A", "overview")], (WIDTH // 2, HEIGHT)), cv2.resize(views[("B", "overview")], (WIDTH - WIDTH // 2, HEIGHT))], axis=1)
                cv2.line(pair, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT - 1), (255, 255, 255), 1)
                canvases["paired"][y:y+HEIGHT, x:x+WIDTH] = pair; cv2.rectangle(canvases["paired"], (x, y), (x + WIDTH - 1, y + HEIGHT - 1), (45, 45, 45), 1)
            for key, writer in writers.items(): writer.write(canvases[key])
            if frame_index % 25 == 0: print(f"actual-state mosaics {frame_index + 1}/{maximum}", flush=True)
        for key, writer in writers.items():
            writer.finish(); products[key] = {"path": str(writer.path), "probe": probe(writer.path)}
    finally:
        renderer.close()
    products["source_fields"] = ["MEASURED_Q", "object_position_world_m", "object_quaternion_xyzw"]
    products["command_only_replay"] = False; products["result_freeze_sha256"] = results["freeze_sha256"]
    return products


def composite(images: list[np.ndarray], labels: list[str], columns: int = 4) -> np.ndarray:
    rows = math.ceil(len(images) / columns); canvas = np.full((rows * (HEIGHT + 38), columns * WIDTH, 3), 245, np.uint8)
    for index, (image, text) in enumerate(zip(images, labels, strict=True)):
        row, col = divmod(index, columns); x, y = col * WIDTH, row * (HEIGHT + 38)
        canvas[y:y+HEIGHT, x:x+WIDTH] = image; cv2.putText(canvas, text, (x + 8, y + HEIGHT + 26), cv2.FONT_HERSHEY_SIMPLEX, .52, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def render_storyboards() -> dict[str, Any]:
    renderer = PhysicalRenderer(); products: dict[str, Any] = {}
    try:
        for method in "AB":
            directories = run_dirs(method); records = [read_json(path / "RUN_MANIFEST.json") for path in directories]
            successes = [index for index, row in enumerate(records) if row["outcomes"]["POST_GRASP_FULL_TASK_SUCCESS"]]
            if not successes:
                products[method] = "NOT AVAILABLE"; continue
            index = successes[0]; record = records[index]; trace = control_trace(directories[index]); events = record["event_frames"]
            right_close = events["right_close"] if events["right_close"] is not None else int(trace["frame"][len(trace["frame"]) // 2])
            frames = [0, events["lift"], max(events["lift"] or 0, right_close - 15), right_close, events["right_confirm"], events["right_owned"], events["bin_entry"], int(trace["frame"][-1])]
            names = ["standardized grasp", "lift", "handoff approach", "dual contact", "right confirmation", "right ownership", "bin entry", "settle"]
            images = []
            for frame in frames:
                renderer.set_state(*state_at(trace, int(frame or 0))); images.append(renderer.view("overview"))
            path = VISUALS / f"{method}_POST_GRASP_SUCCESS_STORYBOARD.png"
            atomic_image(path, composite(images, [f"{name} · {method}{index + 1:02d}" for name in names])); products[method] = str(path)
    finally:
        renderer.close()
    return products


def main() -> int:
    mosaics = render_mosaics(); storyboards = render_storyboards(); checks = []
    for key, value in mosaics.items():
        if not isinstance(value, dict) or "probe" not in value: continue
        probe_value = value["probe"]; checks.append(probe_value["codec_name"] == "h264" and int(probe_value["width"]) == MWIDTH and int(probe_value["height"]) == MHEIGHT and probe_value["avg_frame_rate"] == "30/1")
    report = {
        "status": "PASS" if len(checks) == 5 and all(checks) else "FAIL",
        "actual_saved_physical_states": True, "command_only_replay": False,
        "layout": "7x5", "resolution": [MWIDTH, MHEIGHT], "fps": FPS,
        "same_camera_A_B": True, "same_episode_order": True, "mosaics": mosaics, "storyboards": storyboards,
    }
    atomic_text(REPLAYS / "STANDARDIZED_GRASP_PHYSICAL_REPLAY_VERIFICATION.json", json.dumps(report, indent=2) + "\n")
    atomic_text(REPLAYS / "STANDARDIZED_GRASP_PHYSICAL_REPLAY_VERIFICATION.md", "# Standardized-grasp physical replay verification\n\n" + f"Status: **{report['status']}**\n\nThe five videos reconstruct only saved `MEASURED_Q` and PhysX doll pose states. All use identical A/B cameras and episode order; failures remain visible.\n")
    atomic_text(VISUALS / "STORYBOARD_SELECTION.json", json.dumps({"rule": "lowest-index successful episode per method", "physical_state_modified": False, "camera_only": True, "products": storyboards}, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
