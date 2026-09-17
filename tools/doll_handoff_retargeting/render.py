"""Synchronized source/Baseline/Proposed review rendering in the approved scene.

The renderer is kinematic and diagnostic only.  Scene coordinates are read from
``scene_layout.json`` and are never fed back into trajectory generation.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from .common import METHODS, OUTPUT, SIDES, atomic_json, load_json, sha256_file
from .models import G1Kinematics


COLORS = {
    "table": np.asarray([0.42, 0.31, 0.20, 1.0], dtype=np.float32),
    "frame": np.asarray([0.018, 0.022, 0.025, 1.0], dtype=np.float32),
    "doll": np.asarray([0.16, 0.58, 0.18, 1.0], dtype=np.float32),
    "eyes": np.asarray([0.025, 0.03, 0.025, 1.0], dtype=np.float32),
    "bin": np.asarray([0.87, 0.86, 0.79, 1.0], dtype=np.float32),
    "ground": np.asarray([0.12, 0.13, 0.15, 1.0], dtype=np.float32),
    "left_grasp": np.asarray([1.0, 0.72, 0.08, 0.92], dtype=np.float32),
    "right_grasp": np.asarray([0.08, 0.78, 1.0, 0.92], dtype=np.float32),
}


def _add_geom(
    scene: mujoco.MjvScene,
    kind: mujoco.mjtGeom,
    size: Iterable[float],
    position: Iterable[float],
    rgba: np.ndarray,
    rotation: np.ndarray | None = None,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo visual scene exhausted user-geometry capacity")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        kind,
        np.asarray(list(size), dtype=np.float64),
        np.asarray(list(position), dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel()
        if rotation is None
        else np.asarray(rotation, dtype=np.float64).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def _add_box(
    scene: mujoco.MjvScene,
    dimensions: Iterable[float],
    center: Iterable[float],
    rgba: np.ndarray,
) -> None:
    _add_geom(
        scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        0.5 * np.asarray(list(dimensions), dtype=np.float64),
        center,
        rgba,
    )


def add_approved_scene(
    visual_scene: mujoco.MjvScene,
    layout: Mapping[str, Any],
    left_grasp_world: np.ndarray | None = None,
    right_grasp_world: np.ndarray | None = None,
) -> None:
    """Append only geometry named by the immutable approved scene config."""
    surface = float(layout["table"]["surface_height_m"])
    width, depth = map(float, layout["table"]["size_xy_m"])
    thickness = float(layout["table"]["top_thickness_m"])
    _add_box(
        visual_scene,
        (width, depth, thickness),
        (0.5 * width, 0.5 * depth, surface - 0.5 * thickness),
        COLORS["table"],
    )
    for rail in layout["black_frame"]["rails"].values():
        _add_box(
            visual_scene,
            rail["size_xyz_m"],
            rail["center_xyz_m"],
            COLORS["frame"],
        )

    ground = layout["ground"]
    ground_thickness = float(ground["thickness_m"])
    _add_box(
        visual_scene,
        (*map(float, ground["size_xy_m"]), ground_thickness),
        (0.5 * width, 0.25 * depth, float(ground["surface_height_m"]) - 0.5 * ground_thickness),
        COLORS["ground"],
    )

    doll = layout["doll"]
    doll_radius = 0.5 * float(doll["diameter_m"])
    doll_center = np.asarray(
        [
            *map(float, doll["center_world_xy_m"]),
            surface + float(doll["initial_table_clearance_m"]) + doll_radius,
        ]
    )
    _add_geom(
        visual_scene,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (doll_radius, 0.0, 0.0),
        doll_center,
        COLORS["doll"],
    )
    for local in doll["nub_centers_local_xyz_m"]:
        _add_geom(
            visual_scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (float(doll["nub_radius_m"]), 0.0, 0.0),
            doll_center + np.asarray(local, dtype=np.float64),
            COLORS["doll"],
        )
    for local in doll["eye_centers_local_xyz_m"]:
        _add_geom(
            visual_scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (float(doll["eye_radius_m"]), 0.0, 0.0),
            doll_center + np.asarray(local, dtype=np.float64),
            COLORS["eyes"],
        )

    bin_cfg = layout["bin"]
    outer_x, outer_y, outer_z = map(float, bin_cfg["outer_dimensions_xyz_m"])
    opening_x, opening_y = map(float, bin_cfg["opening_dimensions_xy_m"])
    wall = float(bin_cfg["wall_thickness_m"])
    bottom = float(bin_cfg["bottom_thickness_m"])
    center_x, center_y = map(float, bin_cfg["center_world_xy_m"])
    # Exactly five pieces: bottom plus four walls.  No top/interior visual or collider.
    _add_box(
        visual_scene,
        (outer_x, outer_y, bottom),
        (center_x, center_y, surface + 0.5 * bottom),
        COLORS["bin"],
    )
    wall_z = surface + 0.5 * outer_z
    _add_box(
        visual_scene,
        (outer_x, wall, outer_z),
        (center_x, center_y - 0.5 * (opening_y + wall), wall_z),
        COLORS["bin"],
    )
    _add_box(
        visual_scene,
        (outer_x, wall, outer_z),
        (center_x, center_y + 0.5 * (opening_y + wall), wall_z),
        COLORS["bin"],
    )
    _add_box(
        visual_scene,
        (wall, opening_y, outer_z),
        (center_x - 0.5 * (opening_x + wall), center_y, wall_z),
        COLORS["bin"],
    )
    _add_box(
        visual_scene,
        (wall, opening_y, outer_z),
        (center_x + 0.5 * (opening_x + wall), center_y, wall_z),
        COLORS["bin"],
    )

    for position, color in (
        (left_grasp_world, COLORS["left_grasp"]),
        (right_grasp_world, COLORS["right_grasp"]),
    ):
        if position is not None and np.isfinite(position).all():
            _add_geom(
                visual_scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                (0.009, 0.0, 0.0),
                position,
                color,
            )


def camera_from_layout(layout: Mapping[str, Any], name: str) -> mujoco.MjvCamera:
    preset = layout["camera"]["presets"][name]
    eye = np.asarray(preset["eye_world_xyz_m"], dtype=np.float64)
    target = np.asarray(preset["target_world_xyz_m"], dtype=np.float64)
    relative = eye - target
    horizontal = float(np.linalg.norm(relative[:2]))
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = float(np.linalg.norm(relative))
    # MuJoCo's free-camera eye direction in the horizontal plane is
    # ``[-cos(azimuth), -sin(azimuth)]``.  Resolve the approved eye point
    # explicitly instead of borrowing an azimuth convention from Isaac.
    camera.azimuth = math.degrees(math.atan2(-relative[1], -relative[0]))
    camera.elevation = -math.degrees(math.atan2(relative[2], horizontal))
    return camera


def _overlay(
    image: np.ndarray,
    title: str,
    source_name: str,
    frame: int,
    timestamp: float,
    lines: list[str],
    status: str,
) -> np.ndarray:
    image = np.ascontiguousarray(image)
    color = (80, 220, 80) if status == "PASS" else (80, 170, 255)
    cv2.rectangle(image, (0, 0), (image.shape[1], 106), (14, 14, 18), -1)
    cv2.putText(
        image,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{source_name} | frame {frame:04d} | t={timestamp:6.2f}s",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        " | ".join(lines[:2]),
        (10, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        " | ".join(lines[2:]),
        (10, 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (100, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _handoff_label(events: Mapping[str, Any], frame: int, stride: int) -> str:
    frames = events["frames"]
    approach = frames.get("HANDOFF_APPROACH")
    right_grasp = frames.get("RIGHT_GRASP")
    left_release = frames.get("LEFT_RELEASE")
    final_release = frames.get("RIGHT_FINAL_RELEASE")
    tolerance = max(1, stride // 2)
    if right_grasp is not None and abs(frame - int(right_grasp)) <= tolerance:
        return "RIGHT_GRASP"
    if left_release is not None and abs(frame - int(left_release)) <= tolerance:
        return "LEFT_RELEASE"
    if final_release is not None and abs(frame - int(final_release)) <= tolerance:
        return "RIGHT_FINAL_RELEASE"
    if approach is not None and right_grasp is not None and int(approach) <= frame < int(right_grasp):
        return "HANDOFF_APPROACH"
    if right_grasp is not None and left_release is not None and int(right_grasp) < frame < int(left_release):
        return "DUAL_HOLD"
    if left_release is not None and frame > int(left_release):
        return "RIGHT_HOLD"
    return ""


def trajectory_grasp_array(
    trajectory: Mapping[str, np.ndarray], side: str
) -> np.ndarray:
    """Read the current whole-hand frame with frozen-pinch compatibility."""
    current = f"achieved_{side}_physical_grasp_frame_position_world"
    legacy = f"achieved_{side}_physical_pinch_position_world"
    if current in trajectory:
        return np.asarray(trajectory[current])
    if legacy in trajectory:
        return np.asarray(trajectory[legacy])
    raise KeyError(f"trajectory has neither {current!r} nor legacy {legacy!r}")


def trajectory_inter_grasp_distance(trajectory: Mapping[str, np.ndarray]) -> np.ndarray:
    """Read current whole-hand distance with frozen-pinch compatibility."""
    if "inter_grasp_frame_distance_m" in trajectory:
        return np.asarray(trajectory["inter_grasp_frame_distance_m"])
    if "inter_pinch_distance_m" in trajectory:
        return np.asarray(trajectory["inter_pinch_distance_m"])
    return np.linalg.norm(
        trajectory_grasp_array(trajectory, "right")
        - trajectory_grasp_array(trajectory, "left"),
        axis=1,
    )


class ReviewRenderer:
    def __init__(self, common: Mapping[str, Any], layout: Mapping[str, Any], g1: G1Kinematics):
        self.common = common
        self.layout = layout
        self.g1 = g1
        render = common["render"]
        self.width = int(render["panel_width"])
        self.height = int(render["panel_height"])
        self.stride = int(render["sample_stride_frames"])
        self.fps = float(render["output_fps"])
        self.data = {method: mujoco.MjData(g1.model) for method in METHODS}
        self.renderers = {
            method: mujoco.Renderer(g1.model, height=self.height, width=self.width)
            for method in METHODS
        }
        self.root_position = np.asarray(
            layout["g1"]["root_position_world_xyz_m"], dtype=np.float64
        )
        self.root_quaternion = np.asarray(
            layout["g1"]["root_orientation_world_wxyz"], dtype=np.float64
        )

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()

    def _assign(self, method: str, trajectory: Mapping[str, np.ndarray], frame: int) -> None:
        data = self.data[method]
        data.qpos[:] = self.g1.stand_qpos
        data.qpos[self.g1.arm_qpos_ids] = trajectory["g1_arm_qpos"][frame]
        data.qpos[self.g1.hand_qpos_ids["left"]] = trajectory["left_dex3_qpos"][frame]
        data.qpos[self.g1.hand_qpos_ids["right"]] = trajectory["right_dex3_qpos"][frame]
        data.qpos[:3] = self.root_position
        data.qpos[3:7] = self.root_quaternion
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.g1.model, data)

    def robot_panel(
        self,
        method: str,
        trajectory: Mapping[str, np.ndarray] | None,
        metrics: Mapping[str, Any],
        events: Mapping[str, Any],
        source_name: str,
        frame: int,
        timestamp: float,
        camera_name: str,
    ) -> np.ndarray:
        if trajectory is None or not len(trajectory["g1_arm_qpos"]):
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return _overlay(
                image,
                "G1 BASELINE A" if method == "baseline" else "G1 PROPOSED B",
                source_name,
                frame,
                timestamp,
                ["IK NO TRAJECTORY", str(metrics.get("status", "FAIL_EXCEPTION")), "CONVERSION_EXCEPTION"],
                str(metrics.get("status", "FAIL_EXCEPTION")),
            )
        self._assign(method, trajectory, frame)
        renderer = self.renderers[method]
        renderer.update_scene(self.data[method], camera_from_layout(self.layout, camera_name))
        add_approved_scene(
            renderer.scene,
            self.layout,
            trajectory_grasp_array(trajectory, "left")[frame],
            trajectory_grasp_array(trajectory, "right")[frame],
        )
        image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        left_phase = str(np.asarray(trajectory["left_hand_phase"])[frame])
        right_phase = str(np.asarray(trajectory["right_hand_phase"])[frame])
        success = bool(np.asarray(trajectory["ik_success_per_frame"])[frame])
        distance = float(trajectory_inter_grasp_distance(trajectory)[frame])
        lines = [
            f"L={left_phase} R={right_phase}",
            f"IK={'OK' if success else 'FAIL'}",
            f"inter-grasp={distance:.3f}m",
        ]
        if method == "proposed":
            label = _handoff_label(events, frame, self.stride)
            if label:
                lines.append(label)
        return _overlay(
            image,
            (
                "G1 TRAJECTORY-CENTRIC A"
                if method == "baseline"
                else "G1 INTERACTION-CENTRIC B"
            ),
            source_name,
            frame,
            timestamp,
            lines,
            str(metrics.get("status", "FAIL")),
        )


def _load_trajectory(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as values:
        result = {name: np.asarray(values[name]) for name in values.files}
    if "g1_arm_qpos" not in result or not len(result["g1_arm_qpos"]):
        return None
    return result


def _decoded_video(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode rendered video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return count, fps, width, height


def _writer(path: Path, fps: float, dimensions: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, dimensions
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open MP4 writer: {path}")
    return writer


def render_episode(
    renderer: ReviewRenderer,
    output: Path,
    source_record: Mapping[str, Any],
    events: Mapping[str, Any],
    episode_index: int,
    cameras: tuple[str, ...] = ("overview",),
) -> dict[str, Any]:
    stable = str(source_record["stable_episode_id"])
    source_name = str(source_record["source_name"])
    trajectories = {
        method: _load_trajectory(output / method / "trajectories" / f"{stable}.npz")
        for method in METHODS
    }
    metrics = {
        method: load_json(output / method / "metrics" / f"{stable}.json")
        for method in METHODS
    }
    source_images = sorted(
        Path(source_record["camera_assets"]["observation.images.cam_high"]["directory"]).glob(
            "frame_*.png"
        )
    )
    frame_count = int(source_record["frame_count"])
    frames = list(range(0, frame_count, renderer.stride))
    if frames[-1] != frame_count - 1:
        frames.append(frame_count - 1)
    timestamps = np.arange(frame_count, dtype=np.float64) / float(source_record["fps"])
    outputs: dict[str, str] = {}
    for camera_name in cameras:
        comparison_path = (
            output / "comparison" / "videos" / f"{stable}_comparison_{camera_name}.mp4"
        )
        method_paths = {
            method: output / method / "renders" / f"{stable}_{camera_name}.mp4"
            for method in METHODS
        }
        comparison_writer = _writer(
            comparison_path, renderer.fps, (3 * renderer.width, renderer.height)
        )
        method_writers = {
            method: _writer(path, renderer.fps, (renderer.width, renderer.height))
            for method, path in method_paths.items()
        }
        for frame in frames:
            source = cv2.imread(str(source_images[frame]), cv2.IMREAD_COLOR)
            if source is None:
                source = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)
            source = cv2.resize(source, (renderer.width, renderer.height))
            source = _overlay(
                source,
                "SOURCE ALOHA / REAL CAM_HIGH",
                source_name,
                frame,
                float(timestamps[frame]),
                [
                    "SOURCE EVENTS=" + ("VALID" if not events.get("anomalies") else "ANOMALY"),
                    _handoff_label(events, frame, renderer.stride) or "SOURCE MOTION",
                    "30Hz source / synchronized",
                ],
                "PASS" if not events.get("anomalies") else "WARN",
            )
            panels = []
            for method in METHODS:
                panel = renderer.robot_panel(
                    method,
                    trajectories[method],
                    metrics[method],
                    events,
                    source_name,
                    frame,
                    float(timestamps[frame]),
                    camera_name,
                )
                panels.append(panel)
                method_writers[method].write(panel)
            comparison_writer.write(np.hstack((source, *panels)))
        comparison_writer.release()
        for writer in method_writers.values():
            writer.release()
        expected = len(frames)
        for path in (comparison_path, *method_paths.values()):
            decoded = _decoded_video(path)
            if decoded[0] != expected or abs(decoded[1] - renderer.fps) > 0.1:
                raise RuntimeError(
                    f"render validation failed for {path}: decoded={decoded}, expected={expected}"
                )
        outputs[f"comparison_{camera_name}"] = str(comparison_path.resolve())
        outputs.update(
            {
                f"{method}_{camera_name}": str(path.resolve())
                for method, path in method_paths.items()
            }
        )
    return {
        "episode_index": episode_index,
        "stable_episode_id": stable,
        "source_name": source_name,
        "source_frame_count": frame_count,
        "render_stride": renderer.stride,
        "rendered_frame_count": len(frames),
        "render_fps": renderer.fps,
        "source_semantic_valid": not bool(events.get("anomalies")),
        "baseline_status": metrics["baseline"].get("status"),
        "proposed_status": metrics["proposed"].get("status"),
        "major_semantic_fail": bool(events.get("anomalies"))
        or not bool(metrics["proposed"].get("semantics", {}).get("right_grasp_before_left_release", False)),
        "outputs": outputs,
        "output_sha256": {
            name: sha256_file(path) for name, path in outputs.items()
        },
        "scene_config": renderer.common["scene_config"],
        "scene_config_sha256": renderer.common["scene_config_sha256"],
        "static_scene_doll_not_trajectory_waypoint": True,
        "physics": False,
        "dataset_packaging": False,
        "policy_training": False,
    }


def render_many(
    common: Mapping[str, Any],
    layout: Mapping[str, Any],
    g1: G1Kinematics,
    episode_indices: Iterable[int],
    output: Path = OUTPUT,
    top_for_smoke: bool = True,
    camera_override: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    source_manifest = load_json(output / "source_audit/source_manifest.json")
    event_rows = load_json(output / "event_audit/events.json")
    records = {int(row["episode_index"]): row for row in source_manifest["records"]}
    indices = list(map(int, episode_indices))
    smoke = set(map(int, common["render"]["smoke_episode_indices"]))
    review = ReviewRenderer(common, layout, g1)
    entries: list[dict[str, Any]] = []
    try:
        for offset, episode_index in enumerate(indices, 1):
            cameras = (
                camera_override
                if camera_override is not None
                else (
                    ("overview", "top")
                    if top_for_smoke and episode_index in smoke
                    else ("overview",)
                )
            )
            entry = render_episode(
                review,
                output,
                records[episode_index],
                event_rows[str(episode_index)],
                episode_index,
                cameras,
            )
            entries.append(entry)
            print(
                f"[RENDER {offset:02d}/{len(indices):02d}] episode={episode_index:03d} "
                f"views={','.join(cameras)} output={entry['outputs']['comparison_overview']}",
                flush=True,
            )
            atomic_json(
                output / "comparison/visual_review_manifest.in_progress.json",
                {"complete": False, "entries": entries},
            )
    finally:
        review.close()
    representative = set(map(int, common["render"]["representative_episode_indices"]))
    failure_indices = sorted(
        entry["episode_index"]
        for entry in entries
        if entry["baseline_status"] != "PASS"
        or entry["proposed_status"] != "PASS"
        or entry["major_semantic_fail"]
    )
    manifest = {
        "schema_version": "doll_handoff_visual_review_manifest_v1",
        "status": "PASS" if len(entries) == len(indices) else "FAIL",
        "rendered_episode_count": len(entries),
        "requested_episode_count": len(indices),
        "representative_episode_indices": sorted(representative),
        "representative_videos_present": sorted(
            representative.intersection(entry["episode_index"] for entry in entries)
        ),
        "failure_video_episode_indices": failure_indices,
        "failure_videos_present_count": len(failure_indices),
        "every_rendered_failure_has_video": all(
            bool(entry["outputs"].get("comparison_overview"))
            for entry in entries
            if entry["episode_index"] in failure_indices
        ),
        "layout": "Source ALOHA | G1 Baseline A | G1 Proposed B",
        "synchronization": "source frame/time, stride 3, 10 fps",
        "scene": common["scene_config"],
        "scene_sha256": common["scene_config_sha256"],
        "entries": entries,
        "dataset_packaging": "NOT_STARTED_BY_DESIGN",
        "policy_training": "NOT_STARTED_BY_DESIGN",
    }
    atomic_json(output / "comparison/visual_review_manifest.json", manifest)
    in_progress = output / "comparison/visual_review_manifest.in_progress.json"
    if in_progress.exists():
        in_progress.unlink()
    return manifest


__all__ = [
    "ReviewRenderer",
    "add_approved_scene",
    "camera_from_layout",
    "render_episode",
    "render_many",
    "trajectory_grasp_array",
    "trajectory_inter_grasp_distance",
]
