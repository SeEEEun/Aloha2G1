#!/usr/bin/env python3
"""Visualization-only 50-way ALOHA / frozen Policy-B G1 mosaics.

This tool discovers inputs through the final Dataset-B manifests.  It never
retargets, runs IK, changes actions, packages a dataset, or trains a policy.
All writes are confined to the selected output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.doll_handoff_retargeting.render import (  # noqa: E402
    _add_box,
    add_approved_scene,
)


SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
ACTION_FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
POLICY_DATASET = ROOT / "datasets/doll_handoff_proposed_b_50"
PACKAGING_MANIFEST = POLICY_DATASET / "meta/g1_packaging_manifest.json"
TRAINING_CONFIG = (
    ROOT
    / "outputs/doll_handoff_dataset_b_semantic_audit_2026-08-23"
    / "training/policy_b_semantic_final_config.json"
)
COMMON_CONFIG = (
    ROOT
    / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
    / "config/common_config.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/portfolio_50way"

WIDTH, HEIGHT, FPS = 3840, 2160, 30.0
COLS, ROWS = 8, 7
TILE_MARGIN = 2
BACKGROUND_BGR = (238, 238, 238)
G1_WIDTH, G1_HEIGHT = 640, 480


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path, count_frames: bool = True) -> dict[str, Any]:
    entries = "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,nb_frames"
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        command += ["-count_frames"]
        entries += ",nb_read_frames"
    command += ["-show_entries", entries, "-of", "json", str(path)]
    result = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    rate = result.get("avg_frame_rate") or result.get("r_frame_rate")
    numerator, denominator = map(float, rate.split("/"))
    frames = result.get("nb_read_frames") or result.get("nb_frames")
    return {
        "codec": result.get("codec_name"),
        "width": int(result["width"]),
        "height": int(result["height"]),
        "fps": numerator / denominator,
        "frame_count": int(frames) if frames not in (None, "N/A") else None,
        "pixel_format": result.get("pix_fmt"),
    }


@dataclass(frozen=True)
class Episode:
    index: int
    episode_id: str
    source_raw_episode: str
    source_video: Path
    source_parquet: Path
    trajectory: Path
    frame_count: int
    fps: float


def discover() -> tuple[list[Episode], dict[str, Any]]:
    required = [SOURCE_MANIFEST, ACTION_FREEZE, PACKAGING_MANIFEST, TRAINING_CONFIG]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"FAIL: missing authoritative metadata: {missing}")
    source = read_json(SOURCE_MANIFEST)
    freeze = read_json(ACTION_FREEZE)
    package = read_json(PACKAGING_MANIFEST)
    training = read_json(TRAINING_CONFIG)
    configured_root = Path(training["dataset"]["root"]).resolve()
    if configured_root != POLICY_DATASET.resolve():
        raise RuntimeError(
            f"FAIL: latest Policy-B training root differs: {configured_root}"
        )
    if not (
        source.get("source_count")
        == freeze.get("episode_count")
        == package.get("episode_count")
        == 50
    ):
        raise RuntimeError("FAIL: authoritative manifests do not all declare 50 episodes")
    if freeze.get("status") != "FINAL_PROPOSED_B_ACTION_LABELS_FROZEN":
        raise RuntimeError(f"FAIL: action set is not frozen: {freeze.get('status')}")
    records = source["episodes"]
    alignment = package["episode_alignment"]
    videos = {int(item["final_dataset_episode_index"]): item for item in package["video_assets"]}
    episodes: list[Episode] = []
    for index in range(50):
        record = records[index]
        aligned = alignment[index]
        if int(record["final_dataset_index"]) != index:
            raise RuntimeError(f"FAIL: source index mismatch at {index}")
        if int(aligned["final_dataset_episode_index"]) != index:
            raise RuntimeError(f"FAIL: Policy-B alignment mismatch at {index}")
        frame_count = int(record["source_frame_count"])
        if not (
            int(aligned["source_frame_count"])
            == int(aligned["rgb_frame_count"])
            == int(aligned["action_frame_count"])
            == frame_count
        ):
            raise RuntimeError(f"FAIL: logical frame correspondence mismatch at ep{index:03d}")
        source_video = POLICY_DATASET / videos[index]["dataset_video"]
        trajectory = Path(record["retargeted_trajectory_path"])
        source_parquet = Path(record["source_parquet_path"])
        for path in (source_video, trajectory, source_parquet):
            if not path.is_file():
                raise RuntimeError(f"FAIL: missing ep{index:03d} input: {path}")
        video_probe = probe(source_video)
        if video_probe["frame_count"] != frame_count or abs(video_probe["fps"] - FPS) > 0.01:
            raise RuntimeError(f"FAIL: source video timing mismatch ep{index:03d}: {video_probe}")
        with np.load(trajectory, allow_pickle=False) as arrays:
            q = np.asarray(arrays["replay_named_joint_qpos"])
            names = list(map(str, arrays["replay_joint_names"]))
            if q.shape != (frame_count, 28) or not np.isfinite(q).all():
                raise RuntimeError(f"FAIL: invalid frozen trajectory ep{index:03d}: {q.shape}")
            if set(names) != set(freeze["joint_names"]) or len(set(names)) != 28:
                raise RuntimeError(f"FAIL: frozen joint-name set mismatch ep{index:03d}")
        episodes.append(
            Episode(
                index=index,
                episode_id=f"ep{index:03d}",
                source_raw_episode=str(record["raw_directory"]),
                source_video=source_video.resolve(),
                source_parquet=source_parquet.resolve(),
                trajectory=trajectory.resolve(),
                frame_count=frame_count,
                fps=float(record["source_fps"]),
            )
        )
    metadata = {
        "source_manifest": str(SOURCE_MANIFEST.resolve()),
        "action_freeze_manifest": str(ACTION_FREEZE.resolve()),
        "packaging_manifest": str(PACKAGING_MANIFEST.resolve()),
        "latest_policy_b_training_config": str(TRAINING_CONFIG.resolve()),
        "source_aloha_path": str(
            (POLICY_DATASET / "videos/observation.images.cam_high/chunk-000").resolve()
        ),
        "policy_b_dataset_path": str(POLICY_DATASET.resolve()),
        "frozen_trajectory_path": str(Path(freeze["trajectory_directory"]).resolve()),
        "camera_key": "observation.images.cam_high",
        "fps": FPS,
        "frame_count_range": [min(e.frame_count for e in episodes), max(e.frame_count for e in episodes)],
        "joint_names": list(freeze["joint_names"]),
        "action_schema": "absolute q_target[t], 28D",
        "state_schema": "q_target[max(t-1,0)], 28D prior-target surrogate",
        "correspondence": "PASS 50/50 ep000..ep049",
    }
    return episodes, metadata


def input_hashes(episodes: Iterable[Episode]) -> dict[str, Any]:
    return {
        "source_manifest": sha256(SOURCE_MANIFEST),
        "action_freeze_manifest": sha256(ACTION_FREEZE),
        "packaging_manifest": sha256(PACKAGING_MANIFEST),
        "source_videos": {e.episode_id: sha256(e.source_video) for e in episodes},
        "frozen_trajectories": {e.episode_id: sha256(e.trajectory) for e in episodes},
    }


class G1PortfolioRenderer:
    """Pure FK replay of stored named qpos; no solver or retargeter is called."""

    def __init__(self) -> None:
        common = load_common_config(COMMON_CONFIG)
        self.layout = load_scene(common)
        self.g1 = G1Kinematics(common, self.layout)
        self.model = self.g1.model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=G1_HEIGHT, width=G1_WIDTH)
        self.camera = self._camera()
        self.root_position = np.asarray(self.layout["g1"]["root_position_world_xyz_m"])
        self.root_quaternion = np.asarray(self.layout["g1"]["root_orientation_world_wxyz"])
        self.model_joint_names = [
            *map(str, self.g1.arm_joint_names),
            *self.g1.hand_joint_names["left"],
            *self.g1.hand_joint_names["right"],
        ]
        # Bright portfolio lighting; model geometry and qpos remain untouched on disk.
        self.model.vis.headlight.ambient[:] = (0.52, 0.52, 0.52)
        self.model.vis.headlight.diffuse[:] = (0.78, 0.78, 0.78)
        self.model.vis.headlight.specular[:] = (0.08, 0.08, 0.08)
        self.model.vis.rgba.haze[:] = (0.93, 0.94, 0.95, 1.0)

    @staticmethod
    def _camera() -> mujoco.MjvCamera:
        # Identical upper-body/workspace camera for every episode.
        eye = np.asarray([1.17, -1.04, 1.47], dtype=np.float64)
        target = np.asarray([0.4175, 0.18, 1.02], dtype=np.float64)
        relative = eye - target
        horizontal = float(np.linalg.norm(relative[:2]))
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = target
        camera.distance = float(np.linalg.norm(relative))
        camera.azimuth = math.degrees(math.atan2(-relative[1], -relative[0]))
        camera.elevation = -math.degrees(math.atan2(relative[2], horizontal))
        return camera

    def close(self) -> None:
        self.renderer.close()

    def frame(self, q: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = self.g1.stand_qpos
        self.data.qpos[self.g1.arm_qpos_ids] = q[:14]
        self.data.qpos[self.g1.hand_qpos_ids["left"]] = q[14:21]
        self.data.qpos[self.g1.hand_qpos_ids["right"]] = q[21:28]
        self.data.qpos[:3] = self.root_position
        self.data.qpos[3:7] = self.root_quaternion
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        # Fixed canonical objects only: no object trajectory exists in Dataset B.
        add_approved_scene(self.renderer.scene, self.layout)
        _add_box(
            self.renderer.scene,
            (6.0, 6.0, 0.025),
            (0.4175, 0.20, -0.0125),
            np.asarray([0.84, 0.85, 0.86, 1.0], dtype=np.float32),
        )
        _add_box(
            self.renderer.scene,
            (10.0, 0.035, 4.0),
            (0.4175, 1.02, 1.50),
            np.asarray([0.78, 0.80, 0.82, 1.0], dtype=np.float32),
        )
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        image = self.renderer.render()
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ffmpeg_writer(path: Path, width: int, height: int, preset: str = "fast") -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incomplete.mp4")
    if temporary.exists():
        temporary.unlink()
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-preset", preset,
        "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    process._portfolio_final_path = path  # type: ignore[attr-defined]
    process._portfolio_temporary_path = temporary  # type: ignore[attr-defined]
    return process


def close_writer(process: subprocess.Popen[bytes]) -> None:
    assert process.stdin is not None
    process.stdin.close()
    return_code = process.wait()
    temporary = process._portfolio_temporary_path  # type: ignore[attr-defined]
    final = process._portfolio_final_path  # type: ignore[attr-defined]
    if return_code:
        raise RuntimeError(f"ffmpeg encode failed ({return_code}): {temporary}")
    os.replace(temporary, final)


def render_g1_tiles(episodes: list[Episode], directory: Path, resume: bool) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    renderer = G1PortfolioRenderer()
    outputs: list[Path] = []
    try:
        for episode in episodes:
            output = directory / f"{episode.episode_id}.mp4"
            if output.exists():
                valid = probe(output)
                if resume and valid["frame_count"] == episode.frame_count:
                    print(f"G1 {episode.episode_id}: reuse verified cache", flush=True)
                    outputs.append(output)
                    continue
                raise FileExistsError(f"refusing to overwrite existing tile: {output}")
            print(f"G1 {episode.episode_id}: rendering {episode.frame_count} frozen frames", flush=True)
            with np.load(episode.trajectory, allow_pickle=False) as arrays:
                q = np.asarray(arrays["replay_named_joint_qpos"], dtype=np.float64)
                stored_names = list(map(str, arrays["replay_joint_names"]))
            # Visualization-only name mapping.  Dataset-B explicitly requires
            # name-based reordering between canonical policy and model orders.
            lookup = {name: column for column, name in enumerate(stored_names)}
            if set(lookup) != set(renderer.model_joint_names):
                raise RuntimeError(f"G1 model joint-name mapping failed: {episode.episode_id}")
            q = q[:, [lookup[name] for name in renderer.model_joint_names]]
            writer = ffmpeg_writer(output, G1_WIDTH, G1_HEIGHT, preset="fast")
            assert writer.stdin is not None
            for row in q:
                writer.stdin.write(renderer.frame(row).tobytes())
            close_writer(writer)
            result = probe(output)
            if result["frame_count"] != episode.frame_count:
                raise RuntimeError(f"G1 tile decode mismatch: {episode.episode_id} {result}")
            outputs.append(output)
    finally:
        renderer.close()
    return outputs


def tile_cell(index: int) -> tuple[int, int, int, int]:
    if index < 48:
        row, col = divmod(index, COLS)
    else:
        row, col = 6, index - 48 + 3
    x0, x1 = round(col * WIDTH / COLS), round((col + 1) * WIDTH / COLS)
    y0, y1 = round(row * HEIGHT / ROWS), round((row + 1) * HEIGHT / ROWS)
    return x0 + TILE_MARGIN, y0 + TILE_MARGIN, x1 - TILE_MARGIN, y1 - TILE_MARGIN


def place_scaled(canvas: np.ndarray, image: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bounds
    available_w, available_h = x1 - x0, y1 - y0
    scale = min(available_w / image.shape[1], available_h / image.shape[0])
    width, height = max(2, round(image.shape[1] * scale)), max(2, round(image.shape[0] * scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    left, top = x0 + (available_w - width) // 2, y0 + (available_h - height) // 2
    canvas[top : top + height, left : left + width] = resized


def draw_label(canvas: np.ndarray, episode_index: int, bounds: tuple[int, int, int, int]) -> None:
    x0, y0, _, _ = bounds
    label = f"EP {episode_index:02d}"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    (width, height), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + width + 13, y0 + height + baseline + 11), (245, 245, 245), -1)
    cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + width + 13, y0 + height + baseline + 11), (45, 45, 45), 1)
    cv2.putText(canvas, label, (x0 + 9, y0 + height + 7), font, scale, (25, 25, 25), thickness, cv2.LINE_AA)


def compose(paths: list[Path], episodes: list[Episode], output: Path) -> None:
    if len(paths) != 50:
        raise RuntimeError(f"mosaic needs 50 paths, got {len(paths)}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final video: {output}")
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("one or more tile videos failed to open")
    maximum = max(e.frame_count for e in episodes)
    cached: list[np.ndarray | None] = [None] * 50
    writer = ffmpeg_writer(output, WIDTH, HEIGHT, preset="medium")
    assert writer.stdin is not None
    try:
        for frame_index in range(maximum):
            canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND_BGR, dtype=np.uint8)
            for index, (capture, episode) in enumerate(zip(captures, episodes)):
                if frame_index < episode.frame_count:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"decode failed: {paths[index]} frame {frame_index}")
                    cached[index] = frame
                frame = cached[index]
                if frame is None:
                    raise RuntimeError(f"empty tile: {paths[index]}")
                bounds = tile_cell(index)
                place_scaled(canvas, frame, bounds)
                draw_label(canvas, index, bounds)
            writer.stdin.write(canvas.tobytes())
            if frame_index % 100 == 0:
                print(f"{output.name}: composed {frame_index + 1}/{maximum}", flush=True)
        close_writer(writer)
    finally:
        for capture in captures:
            capture.release()


def read_video_frame(path: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"cannot read {path} frame {frame}")
    return image


def make_visual_evidence(video: Path, preview: Path, contact_sheet: Path, frame_count: int) -> dict[str, Any]:
    indices = [0, round(0.25 * (frame_count - 1)), round(0.5 * (frame_count - 1)), round(0.75 * (frame_count - 1)), frame_count - 1]
    frames = [read_video_frame(video, index) for index in indices]
    if not cv2.imwrite(str(preview), frames[2], [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"failed to write preview: {preview}")
    thumbs = [cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA) for frame in frames]
    sheet = np.hstack(thumbs)
    if not cv2.imwrite(str(contact_sheet), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"failed to write contact sheet: {contact_sheet}")
    return {"sample_frame_indices": indices, "preview": str(preview), "contact_sheet": str(contact_sheet)}


def sample_tile_metrics(paths: list[Path], episodes: list[Episode]) -> dict[str, Any]:
    rows = []
    for path, episode in zip(paths, episodes):
        values = []
        deviations = []
        for index in (0, episode.frame_count // 2, episode.frame_count - 1):
            frame = read_video_frame(path, index)
            values.append(float(frame.mean()))
            deviations.append(float(frame.std()))
        rows.append({"episode_id": episode.episode_id, "mean_rgb": float(np.mean(values)), "mean_stddev": float(np.mean(deviations))})
    return {
        "episodes": rows,
        "minimum_mean_rgb": min(row["mean_rgb"] for row in rows),
        "minimum_mean_stddev": min(row["mean_stddev"] for row in rows),
        "black_or_empty_count": sum(row["mean_rgb"] < 8 or row["mean_stddev"] < 3 for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("aloha", "g1", "all"), required=True)
    parser.add_argument("--dataset", choices=("policy_b",), default="policy_b")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="reuse only already-valid individual G1 tile caches")
    parser.add_argument("--discover-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, metadata = discover()
    print(json.dumps(metadata, indent=2), flush=True)
    if args.discover_only:
        return 0
    before = input_hashes(episodes)
    g1_paths: list[Path] | None = None
    source_paths = [episode.source_video for episode in episodes]
    if args.robot in ("g1", "all"):
        g1_paths = render_g1_tiles(episodes, output_dir / "cache/g1", args.resume)
    produced: dict[str, Any] = {}
    maximum = max(e.frame_count for e in episodes)
    if args.robot in ("aloha", "all"):
        video = output_dir / "aloha_50episodes_mosaic_4k.mp4"
        compose(source_paths, episodes, video)
        produced["aloha"] = {
            "video": str(video),
            "visual_evidence": make_visual_evidence(
                video,
                output_dir / "aloha_50episodes_mosaic_preview.png",
                output_dir / "aloha_50episodes_contact_sheet.png",
                maximum,
            ),
            "tile_metrics": sample_tile_metrics(source_paths, episodes),
        }
    if args.robot in ("g1", "all"):
        assert g1_paths is not None
        video = output_dir / "policyB_g1_50episodes_mosaic_4k.mp4"
        compose(g1_paths, episodes, video)
        produced["g1"] = {
            "video": str(video),
            "visual_evidence": make_visual_evidence(
                video,
                output_dir / "policyB_g1_50episodes_mosaic_preview.png",
                output_dir / "policyB_g1_50episodes_contact_sheet.png",
                maximum,
            ),
            "tile_metrics": sample_tile_metrics(g1_paths, episodes),
            "render_note": "stored frozen 28D action replay; fixed canonical doll/bin/table; no object motion synthesized",
        }
    after = input_hashes(episodes)
    if before != after:
        raise RuntimeError("FAIL: authoritative input checksum changed during visualization")
    final_probes = {name: probe(Path(item["video"])) for name, item in produced.items()}
    for name, result in final_probes.items():
        if not (
            result["codec"] == "h264"
            and result["width"] == WIDTH
            and result["height"] == HEIGHT
            and abs(result["fps"] - FPS) < 0.01
            and result["pixel_format"] == "yuv420p"
            and result["frame_count"] == maximum
        ):
            raise RuntimeError(f"FAIL: final {name} probe: {result}")
    tiles = []
    for episode in episodes:
        tiles.append(
            {
                "tile_index": episode.index,
                "episode_id": episode.episode_id,
                "grid_row": episode.index // 8 if episode.index < 48 else 6,
                "grid_column": episode.index % 8 if episode.index < 48 else episode.index - 45,
                "source_path": str(episode.source_video),
                "source_raw_episode": episode.source_raw_episode,
                "source_parquet_path": str(episode.source_parquet),
                "g1_converted_trajectory_path": str(episode.trajectory),
                "frame_count": episode.frame_count,
                "fps": episode.fps,
                "duration": episode.frame_count / episode.fps,
                "render_status": "PASS",
            }
        )
    validation = {
        "status": "PASS",
        "aloha_tile_count": 50,
        "g1_tile_count": 50 if g1_paths is not None else None,
        "episode_ids_unique_complete": [e.index for e in episodes] == list(range(50)),
        "aloha_g1_correspondence": "50/50",
        "nan_frames": 0,
        "joint_mapping_failures": 0,
        "input_checksums_unchanged": before == after,
        "source_files_modified": False,
        "recompute_retargeting": False,
        "recompute_ik": False,
        "modify_action": False,
        "final_video_probes": final_probes,
        "black_or_empty": {name: item["tile_metrics"]["black_or_empty_count"] for name, item in produced.items()},
    }
    if any(value for value in validation["black_or_empty"].values()):
        raise RuntimeError(f"FAIL: black or empty tile detected: {validation['black_or_empty']}")
    manifest = {
        "schema_version": "portfolio_50way_visualization_v1",
        "visualization_only": True,
        "authoritative_data": metadata,
        "layout": {"columns": COLS, "rows": ROWS, "used_slots": 50, "last_row_columns": [3, 4], "scale_mode": "scale+pad", "margin_px": TILE_MARGIN},
        "output": {"resolution": [WIDTH, HEIGHT], "fps": FPS, "codec": "H.264", "pixel_format": "yuv420p", "duration_policy": "original 30 Hz; shorter episodes freeze last frame; stop at longest episode"},
        "tiles": tiles,
        "produced": produced,
        "input_checksums_before": before,
        "input_checksums_after": after,
        "validation": validation,
    }
    write_json(output_dir / "mosaic_manifest.json", manifest)
    print(json.dumps(validation, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
