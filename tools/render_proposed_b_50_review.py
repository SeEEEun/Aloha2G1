#!/usr/bin/env python3
"""Render Source ALOHA | frozen Interaction-Centric Proposed B reviews."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    atomic_json,
    load_common_config,
    load_json,
    load_scene,
    sha256_file,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.doll_handoff_retargeting.render import (  # noqa: E402
    _decoded_video,
    _handoff_label,
    _load_trajectory,
    _overlay,
    _writer,
    add_approved_scene,
    camera_from_layout,
    trajectory_grasp_array,
    trajectory_inter_grasp_distance,
)


DEFAULT_ROOT = (
    REPOSITORY / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
)


class ProposedBRenderer:
    def __init__(
        self,
        common: Mapping[str, Any],
        layout: Mapping[str, Any],
        g1: G1Kinematics,
    ) -> None:
        self.common = common
        self.layout = layout
        self.g1 = g1
        render = common["render"]
        self.width = int(render["panel_width"])
        self.height = int(render["panel_height"])
        self.stride = int(render["sample_stride_frames"])
        self.fps = float(render["output_fps"])
        self.data = mujoco.MjData(g1.model)
        self.renderer = mujoco.Renderer(
            g1.model, height=self.height, width=self.width
        )
        self.root_position = np.asarray(
            layout["g1"]["root_position_world_xyz_m"], dtype=np.float64
        )
        self.root_quaternion = np.asarray(
            layout["g1"]["root_orientation_world_wxyz"], dtype=np.float64
        )

    def close(self) -> None:
        self.renderer.close()

    def panel(
        self,
        trajectory: Mapping[str, np.ndarray],
        metrics: Mapping[str, Any],
        events: Mapping[str, Any],
        source_name: str,
        frame: int,
        timestamp: float,
        camera_name: str,
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
            self.data, camera_from_layout(self.layout, camera_name)
        )
        add_approved_scene(
            self.renderer.scene,
            self.layout,
            trajectory_grasp_array(trajectory, "left")[frame],
            trajectory_grasp_array(trajectory, "right")[frame],
        )
        image = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        left_phase = str(np.asarray(trajectory["left_hand_phase"])[frame])
        right_phase = str(np.asarray(trajectory["right_hand_phase"])[frame])
        ownership = str(np.asarray(trajectory["ownership_state"])[frame])
        success = bool(np.asarray(trajectory["ik_success_per_frame"])[frame])
        distance = float(trajectory_inter_grasp_distance(trajectory)[frame])
        handoff = _handoff_label(events, frame, self.stride)
        return _overlay(
            image,
            "G1 INTERACTION-CENTRIC PROPOSED B",
            source_name,
            frame,
            timestamp,
            [
                f"L={left_phase} R={right_phase}",
                f"IK={'OK' if success else 'FAIL'} STATUS={metrics['status']}",
                f"OWNER={ownership}",
                f"inter-grasp={distance:.3f}m" + (f" {handoff}" if handoff else ""),
            ],
            str(metrics["status"]),
        )


def source_panel(
    image_path: Path,
    width: int,
    height: int,
    source_name: str,
    frame: int,
    timestamp: float,
    events: Mapping[str, Any],
    stride: int,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.resize(image, (width, height))
    label = _handoff_label(events, frame, stride) or "SOURCE MOTION"
    return _overlay(
        image,
        "SOURCE ALOHA / REAL CAM_HIGH",
        source_name,
        frame,
        timestamp,
        [
            "SOURCE EVENTS=" + ("VALID" if not events.get("anomalies") else "ANOMALY"),
            label,
            "30Hz source / synchronized",
        ],
        "PASS" if not events.get("anomalies") else "WARN",
    )


def render_episode(
    renderer: ProposedBRenderer,
    root: Path,
    record: Mapping[str, Any],
    events: Mapping[str, Any],
    episode: int,
    cameras: tuple[str, ...],
) -> dict[str, Any]:
    stable = str(record["stable_episode_id"])
    source_name = str(record["source_name"])
    trajectory_path = root / "proposed/trajectories" / f"{stable}.npz"
    metric_path = root / "proposed/metrics" / f"{stable}.json"
    trajectory = _load_trajectory(trajectory_path)
    if trajectory is None:
        raise RuntimeError(f"missing usable Proposed-B trajectory: {trajectory_path}")
    metrics = load_json(metric_path)
    source_images = sorted(
        Path(record["camera_assets"]["observation.images.cam_high"]["directory"]).glob(
            "frame_*.png"
        )
    )
    frame_count = int(record["frame_count"])
    if len(source_images) < frame_count:
        raise RuntimeError(
            f"source image count mismatch for ep{episode:03d}: "
            f"{len(source_images)} < {frame_count}"
        )
    frames = list(range(0, frame_count, renderer.stride))
    if frames[-1] != frame_count - 1:
        frames.append(frame_count - 1)
    timestamps = np.arange(frame_count, dtype=np.float64) / float(record["fps"])
    outputs: dict[str, str] = {}
    for camera_name in cameras:
        path = root / "review/videos" / f"{stable}_source_proposed_b_{camera_name}.mp4"
        writer = _writer(
            path, renderer.fps, (2 * renderer.width, renderer.height)
        )
        try:
            for frame in frames:
                source = source_panel(
                    source_images[frame],
                    renderer.width,
                    renderer.height,
                    source_name,
                    frame,
                    float(timestamps[frame]),
                    events,
                    renderer.stride,
                )
                proposed = renderer.panel(
                    trajectory,
                    metrics,
                    events,
                    source_name,
                    frame,
                    float(timestamps[frame]),
                    camera_name,
                )
                writer.write(np.hstack((source, proposed)))
        finally:
            writer.release()
        decoded = _decoded_video(path)
        if decoded[0] != len(frames) or abs(decoded[1] - renderer.fps) > 0.1:
            raise RuntimeError(
                f"render validation failed for {path}: decoded={decoded}, "
                f"expected_frames={len(frames)}"
            )
        outputs[camera_name] = str(path.resolve())
    return {
        "episode_index": episode,
        "stable_episode_id": stable,
        "source_name": source_name,
        "status": metrics["status"],
        "representative": episode
        in set(map(int, renderer.common["render"]["representative_episode_indices"])),
        "failure": metrics["status"] != "PASS",
        "source_frame_count": frame_count,
        "rendered_frame_count": len(frames),
        "render_stride": renderer.stride,
        "render_fps": renderer.fps,
        "views": list(cameras),
        "outputs": outputs,
        "output_sha256": {name: sha256_file(path) for name, path in outputs.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    common = load_common_config(root / "config/common_config.json")
    layout = load_scene(common)
    g1 = G1Kinematics(common, layout)
    source = load_json(root / "source_audit/source_manifest.json")
    events = load_json(root / "event_audit/events.json")
    records = {int(item["episode_index"]): item for item in source["records"]}
    representative = set(map(int, common["render"]["representative_episode_indices"]))
    failures = {
        episode
        for episode in range(50)
        if load_json(
            root
            / "proposed/metrics"
            / f"doll_handoff_20260820_ep{episode:03d}.json"
        )["status"]
        != "PASS"
    }
    requested = sorted(representative | failures)
    renderer = ProposedBRenderer(common, layout, g1)
    entries: list[dict[str, Any]] = []
    in_progress = root / "review/visual_review_manifest.in_progress.json"
    try:
        for offset, episode in enumerate(requested, 1):
            cameras = (
                ("overview", "top", "side")
                if episode in representative
                else ("overview",)
            )
            entry = render_episode(
                renderer,
                root,
                records[episode],
                events[str(episode)],
                episode,
                cameras,
            )
            entries.append(entry)
            atomic_json(
                in_progress,
                {"complete": False, "requested": requested, "entries": entries},
            )
            print(
                f"[B REVIEW {offset:02d}/{len(requested):02d}] "
                f"episode={episode:03d} status={entry['status']} "
                f"views={','.join(cameras)}",
                flush=True,
            )
    finally:
        renderer.close()
    by_episode = {int(entry["episode_index"]): entry for entry in entries}
    every_failure = all(
        episode in by_episode and "overview" in by_episode[episode]["outputs"]
        for episode in failures
    )
    every_representative = all(
        episode in by_episode
        and set(("overview", "top", "side")).issubset(by_episode[episode]["outputs"])
        for episode in representative
    )
    manifest = {
        "schema_version": "interaction_centric_proposed_b_visual_review_v1",
        "status": "PASS" if every_failure and every_representative else "FAIL",
        "layout": "Source ALOHA | G1 Interaction-Centric Proposed B",
        "synchronization": "source frame/time, stride 3, 10 fps",
        "representative_episode_indices": sorted(representative),
        "failure_episode_indices": sorted(failures),
        "requested_episode_indices": requested,
        "rendered_episode_count": len(entries),
        "rendered_video_count": sum(len(entry["outputs"]) for entry in entries),
        "every_failure_has_overview_video": every_failure,
        "every_representative_has_overview_top_side": every_representative,
        "scene": common["scene_config"],
        "scene_sha256": common["scene_config_sha256"],
        "entries": entries,
        "baseline_a": "NOT_RUN",
        "dataset_packaging": "NOT_RUN",
        "policy_training": "NOT_RUN",
    }
    atomic_json(root / "review/visual_review_manifest.json", manifest)
    if in_progress.exists():
        in_progress.unlink()
    print(json.dumps({key: manifest[key] for key in (
        "status",
        "rendered_episode_count",
        "rendered_video_count",
        "every_failure_has_overview_video",
        "every_representative_has_overview_top_side",
    )}, indent=2))
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
