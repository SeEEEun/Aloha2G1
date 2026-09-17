"""Render Source ALOHA | frozen BEFORE | generic-feasibility AFTER reviews."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from tools.doll_handoff_retargeting.render import (
    _decoded_video,
    _handoff_label,
    _writer,
    add_approved_scene,
    camera_from_layout,
)

from .common import (
    FROZEN_ROOT,
    OUTPUT_ROOT,
    SIDES,
    load_json,
    load_trajectory,
    sha256_file,
    stable_episode_id,
    write_json,
)
from .solver import GenericG1FeasibilityResolver


REPRESENTATIVE_EPISODES = (2, 3, 11, 26, 24, 13, 36)
CAMERAS = ("overview", "top", "side")


def _overlay(
    image: np.ndarray,
    title: str,
    episode: int,
    frame: int,
    timestamp: float,
    lines: list[str],
    status: str,
) -> np.ndarray:
    image = np.ascontiguousarray(image)
    hard = "HARD" in status or "FAIL" in status
    color = (70, 90, 255) if hard else ((80, 170, 255) if "WARN" in status else (80, 220, 80))
    cv2.rectangle(image, (0, 0), (image.shape[1], 132), (14, 14, 18), -1)
    entries = [
        (title, 22, 0.52, (255, 255, 255)),
        (f"ep{episode:03d} | frame {frame:04d} | t={timestamp:6.2f}s", 44, 0.40, (210, 210, 210)),
    ]
    entries.extend(
        (line, 66 + 20 * index, 0.37, color if index == 1 else (100, 220, 255))
        for index, line in enumerate(lines[:4])
    )
    for text, y, scale, text_color in entries:
        cv2.putText(
            image,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return image


class BeforeAfterRenderer:
    def __init__(self, resolver: GenericG1FeasibilityResolver) -> None:
        self.resolver = resolver
        self.g1 = resolver.g1
        self.layout = resolver.scene
        render = resolver.common["render"]
        self.width = int(render["panel_width"])
        self.height = int(render["panel_height"])
        self.stride = int(render["sample_stride_frames"])
        self.fps = float(render["output_fps"])
        self.data = mujoco.MjData(self.g1.model)
        self.renderer = mujoco.Renderer(
            self.g1.model, height=self.height, width=self.width
        )
        self.root_position = np.asarray(
            self.layout["g1"]["root_position_world_xyz_m"], dtype=np.float64
        )
        self.root_quaternion = np.asarray(
            self.layout["g1"]["root_orientation_world_wxyz"], dtype=np.float64
        )

    def close(self) -> None:
        self.renderer.close()

    def robot_panel(
        self,
        trajectory: Mapping[str, np.ndarray],
        grasp_world: Mapping[str, np.ndarray],
        episode: int,
        frame: int,
        timestamp: float,
        camera: str,
        title: str,
        status: str,
        lines: list[str],
    ) -> np.ndarray:
        self.data.qpos[:] = self.g1.stand_qpos
        self.data.qpos[self.g1.arm_qpos_ids] = trajectory["g1_arm_qpos"][frame]
        self.data.qpos[self.g1.hand_qpos_ids["left"]] = trajectory[
            "left_dex3_qpos"
        ][frame]
        self.data.qpos[self.g1.hand_qpos_ids["right"]] = trajectory[
            "right_dex3_qpos"
        ][frame]
        self.data.qpos[:3] = self.root_position
        self.data.qpos[3:7] = self.root_quaternion
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.g1.model, self.data)
        self.renderer.update_scene(
            self.data, camera_from_layout(self.layout, camera)
        )
        add_approved_scene(
            self.renderer.scene,
            self.layout,
            grasp_world["left"][frame],
            grasp_world["right"][frame],
        )
        image = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        return _overlay(
            image,
            title,
            episode,
            frame,
            timestamp,
            lines,
            status,
        )


def _collision_arrays(metrics: Mapping[str, Any], count: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.full(count, "CLEAR", dtype="U96")
    depth = np.zeros(count, dtype=np.float64)
    hard = set(map(int, metrics["hard_frames"]))
    for raw_frame, records in metrics["by_frame"].items():
        frame = int(raw_frame)
        classes = sorted({str(row["classification"]) for row in records})
        depth[frame] = max(float(row["penetration_depth_m"]) for row in records)
        state[frame] = ("HARD:" if frame in hard else "CONTACT:") + "+".join(classes)
    return state, depth


def _episode_rows(output_root: Path) -> dict[int, dict[str, str]]:
    path = output_root / "full50/per_episode.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["episode_index"]): row for row in csv.DictReader(stream)}


def render_representative_reviews(
    output_root: Path = OUTPUT_ROOT,
    episodes: tuple[int, ...] = REPRESENTATIVE_EPISODES,
    cameras: tuple[str, ...] = CAMERAS,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    resolver = GenericG1FeasibilityResolver(output_root=output_root)
    rows = _episode_rows(output_root)
    source_records = {
        int(row["episode_index"]): row
        for row in load_json(FROZEN_ROOT / "source_audit/source_manifest.json")[
            "records"
        ]
    }
    events = load_json(FROZEN_ROOT / "event_audit/events.json")
    renderer = BeforeAfterRenderer(resolver)
    entries: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            stable = stable_episode_id(episode)
            before = load_trajectory(episode)
            after_result = resolver.load_exported_episode(episode)
            after_path = output_root / "after/trajectories" / f"{stable}.npz"
            with np.load(after_path, allow_pickle=False) as payload:
                after = {name: np.asarray(payload[name]) for name in payload.files}
            count = len(before["g1_arm_qpos"])
            timestamp = before["timestamp"].astype(np.float64)
            source_world = after_result.source_position_world
            before_model, _ = resolver._pose_arrays(
                before["g1_arm_qpos"].astype(np.float64)
            )
            before_world = {
                side: resolver.g1.model_to_world_position(before_model[side])
                for side in SIDES
            }
            after_world = after_result.achieved_position_world
            before_error = np.maximum(
                *[
                    np.linalg.norm(before_world[side] - source_world[side], axis=1)
                    for side in SIDES
                ]
            )
            after_source_error = np.maximum(
                *[
                    np.linalg.norm(after_world[side] - source_world[side], axis=1)
                    for side in SIDES
                ]
            )
            after_realized_error = np.maximum(
                *[
                    np.linalg.norm(
                        after_world[side]
                        - after_result.realized_position_world[side],
                        axis=1,
                    )
                    for side in SIDES
                ]
            )
            fps = float(1.0 / np.median(np.diff(timestamp)))
            before_collision = resolver._collision_metrics(
                before["g1_arm_qpos"].astype(np.float64),
                before["left_dex3_qpos"].astype(np.float64),
                before["right_dex3_qpos"].astype(np.float64),
                fps,
            )
            after_collision = resolver._collision_metrics(
                after_result.q_after,
                after_result.left_hand,
                after_result.right_hand,
                fps,
            )
            before_collision_state, before_depth = _collision_arrays(
                before_collision, count
            )
            after_collision_state, after_depth = _collision_arrays(
                after_collision, count
            )
            source_images = sorted(
                Path(
                    source_records[episode]["camera_assets"][
                        "observation.images.cam_high"
                    ]["directory"]
                ).glob("frame_*.png")
            )
            if len(source_images) < count:
                raise RuntimeError(f"ep{episode:03d} source images are incomplete")
            sampled = list(range(0, count, renderer.stride))
            if sampled[-1] != count - 1:
                sampled.append(count - 1)
            outputs: dict[str, str] = {}
            for camera in cameras:
                path = output_root / "videos" / f"ep{episode:03d}_source_before_after_{camera}.mp4"
                writer = _writer(
                    path,
                    renderer.fps,
                    (3 * renderer.width, renderer.height),
                )
                try:
                    for frame in sampled:
                        source = cv2.imread(str(source_images[frame]), cv2.IMREAD_COLOR)
                        if source is None:
                            source = np.zeros(
                                (renderer.height, renderer.width, 3), dtype=np.uint8
                            )
                        source = cv2.resize(source, (renderer.width, renderer.height))
                        event = _handoff_label(
                            events[str(episode)], frame, renderer.stride
                        ) or "SOURCE MOTION"
                        source = _overlay(
                            source,
                            "SOURCE ALOHA / CAM_HIGH",
                            episode,
                            frame,
                            float(timestamp[frame]),
                            [event, "ownership reference: source-derived", "30 Hz synchronized"],
                            "PASS",
                        )
                        ownership = str(before["ownership_state"][frame])
                        phases = (
                            f"phase L={before['left_hand_phase'][frame]} "
                            f"R={before['right_hand_phase'][frame]}"
                        )
                        before_panel = renderer.robot_panel(
                            before,
                            before_world,
                            episode,
                            frame,
                            float(timestamp[frame]),
                            camera,
                            "FROZEN PROPOSED B / BEFORE",
                            str(rows[episode]["before_classification"]),
                            [
                                f"{phases} | owner={ownership}",
                                f"IK source residual={before_error[frame]*1000:.2f} mm",
                                f"collision={before_collision_state[frame]} depth={before_depth[frame]*1000:.2f} mm",
                                "projection=0.00 mm (frozen source target)",
                            ],
                        )
                        projection = float(
                            np.max(after_result.projection_translation_m[frame])
                        )
                        after_panel = renderer.robot_panel(
                            after,
                            after_world,
                            episode,
                            frame,
                            float(timestamp[frame]),
                            camera,
                            "PROPOSED B + GENERIC G1 FEASIBILITY / AFTER",
                            str(rows[episode]["after_classification"]),
                            [
                                f"{phases} | owner={ownership}",
                                f"IK source/realized={after_source_error[frame]*1000:.2f}/{after_realized_error[frame]*1000:.2f} mm",
                                f"collision={after_collision_state[frame]} depth={after_depth[frame]*1000:.2f} mm",
                                f"projection={projection*1000:.2f} mm",
                            ],
                        )
                        writer.write(np.hstack((source, before_panel, after_panel)))
                finally:
                    writer.release()
                decoded = _decoded_video(path)
                expected = (len(sampled), renderer.fps, 3 * renderer.width, renderer.height)
                if decoded[0] != expected[0] or abs(decoded[1] - expected[1]) > 0.1:
                    raise RuntimeError(
                        f"video validation failed for {path}: {decoded} != {expected}"
                    )
                outputs[camera] = str(path.resolve())
                print(
                    f"[VIDEO ep{episode:03d}] {camera} frames={len(sampled)}",
                    flush=True,
                )
            entries.append(
                {
                    "episode_index": episode,
                    "selection_reason": resolver.config["representative_gate"][
                        "selection"
                    ].get(
                        str(episode),
                        "retained full-50 hard collision requiring human review",
                    ),
                    "before_classification": rows[episode]["before_classification"],
                    "after_classification": rows[episode]["after_classification"],
                    "views": list(cameras),
                    "render_stride": renderer.stride,
                    "output_fps": renderer.fps,
                    "outputs": outputs,
                    "sha256": {
                        name: sha256_file(path) for name, path in outputs.items()
                    },
                }
            )
    finally:
        renderer.close()
    manifest = {
        "schema_version": "doll_handoff_generic_feasibility_visual_review_v1",
        "layout_modified": False,
        "episodes": list(episodes),
        "representative_gate_episodes": [2, 3, 11, 26, 24],
        "retained_hard_failure_episodes": [13, 36],
        "views": list(cameras),
        "panel_order": ["SOURCE_ALOHA", "FROZEN_PROPOSED_B_BEFORE", "GENERIC_FEASIBILITY_AFTER"],
        "entries": entries,
    }
    write_json(output_root / "videos/manifest.json", manifest)
    return manifest


__all__ = ["render_representative_reviews"]
