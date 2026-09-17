#!/usr/bin/env python3
"""Render Source ALOHA | Proposed-B before | Proposed-B after smoke reviews.

This tool only reads already-converted smoke trajectories.  It never invokes
retargeting, changes task-space targets, packages a dataset, or starts training.
"""
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


DEFAULT_AUDIT_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/natural_arm_audit"
DEFAULT_BEFORE = DEFAULT_AUDIT_ROOT / "before_final_common_resolver_off"
DEFAULT_AFTER = DEFAULT_AUDIT_ROOT / "after_v8_joint_space_trust_region"
DEFAULT_OUTPUT = DEFAULT_AUDIT_ROOT / "review"


def _parse_episodes(value: str) -> list[int]:
    result = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not result or any(index < 0 or index >= 50 for index in result):
        raise ValueError("episodes must be a comma-separated subset of 0..49")
    return result


def _load_metric(root: Path, stable: str) -> dict[str, Any]:
    return load_json(root / "proposed/metrics" / f"{stable}.json")


class NaturalArmRenderer:
    def __init__(
        self,
        common: Mapping[str, Any],
        layout: Mapping[str, Any],
        g1: G1Kinematics,
    ) -> None:
        render = common["render"]
        self.common = common
        self.layout = layout
        self.g1 = g1
        self.width = int(render["panel_width"])
        self.height = int(render["panel_height"])
        self.stride = int(render["sample_stride_frames"])
        self.fps = float(render["output_fps"])
        self.data = {label: mujoco.MjData(g1.model) for label in ("before", "after")}
        self.renderers = {
            label: mujoco.Renderer(g1.model, height=self.height, width=self.width)
            for label in ("before", "after")
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

    def panel(
        self,
        label: str,
        trajectory: Mapping[str, np.ndarray],
        metrics: Mapping[str, Any],
        events: Mapping[str, Any],
        source_name: str,
        frame: int,
        timestamp: float,
        camera_name: str,
    ) -> np.ndarray:
        data = self.data[label]
        data.qpos[:] = self.g1.stand_qpos
        data.qpos[self.g1.arm_qpos_ids] = trajectory["g1_arm_qpos"][frame]
        data.qpos[self.g1.hand_qpos_ids["left"]] = trajectory["left_dex3_qpos"][frame]
        data.qpos[self.g1.hand_qpos_ids["right"]] = trajectory["right_dex3_qpos"][frame]
        data.qpos[:3] = self.root_position
        data.qpos[3:7] = self.root_quaternion
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.g1.model, data)
        renderer = self.renderers[label]
        renderer.update_scene(data, camera_from_layout(self.layout, camera_name))
        add_approved_scene(
            renderer.scene,
            self.layout,
            trajectory_grasp_array(trajectory, "left")[frame],
            trajectory_grasp_array(trajectory, "right")[frame],
        )
        image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        left_phase = str(np.asarray(trajectory["left_hand_phase"])[frame])
        right_phase = str(np.asarray(trajectory["right_hand_phase"])[frame])
        ownership = (
            str(np.asarray(trajectory["ownership_state"])[frame])
            if "ownership_state" in trajectory
            else _handoff_label(events, frame, self.stride) or "SOURCE_DERIVED"
        )
        success = bool(np.asarray(trajectory["ik_success_per_frame"])[frame])
        distance = float(trajectory_inter_grasp_distance(trajectory)[frame])
        title = (
            "B BEFORE | RESOLVER OFF"
            if label == "before"
            else "B AFTER | COMMON NATURAL-ARM"
        )
        return _overlay(
            image,
            title,
            source_name,
            frame,
            timestamp,
            [
                f"L={left_phase} R={right_phase}",
                f"IK={'OK' if success else 'FAIL'}",
                f"inter-grasp={distance:.3f}m",
                ownership,
            ],
            str(metrics.get("status", "FAIL")),
        )


def render_episode(
    renderer: NaturalArmRenderer,
    before_root: Path,
    after_root: Path,
    output_root: Path,
    record: Mapping[str, Any],
    events: Mapping[str, Any],
    cameras: tuple[str, ...],
) -> dict[str, Any]:
    stable = str(record["stable_episode_id"])
    source_name = str(record["source_name"])
    roots = {"before": before_root, "after": after_root}
    trajectories = {
        label: _load_trajectory(root / "proposed/trajectories" / f"{stable}.npz")
        for label, root in roots.items()
    }
    if any(value is None for value in trajectories.values()):
        raise RuntimeError(f"missing before/after trajectory for {stable}")
    typed_trajectories = {
        label: value for label, value in trajectories.items() if value is not None
    }
    metrics = {label: _load_metric(root, stable) for label, root in roots.items()}
    target_hash_equal = (
        metrics["before"]["cartesian_target_sha256"]
        == metrics["after"]["cartesian_target_sha256"]
    )
    if not target_hash_equal:
        raise RuntimeError(f"Cartesian target changed between before/after: {stable}")

    source_images = sorted(
        Path(record["camera_assets"]["observation.images.cam_high"]["directory"]).glob(
            "frame_*.png"
        )
    )
    frame_count = int(record["frame_count"])
    frames = list(range(0, frame_count, renderer.stride))
    if frames[-1] != frame_count - 1:
        frames.append(frame_count - 1)
    timestamps = np.arange(frame_count, dtype=np.float64) / float(record["fps"])
    outputs: dict[str, str] = {}
    for camera_name in cameras:
        comparison_path = output_root / "videos" / f"{stable}_natural_arm_{camera_name}.mp4"
        individual_paths = {
            label: output_root / label / f"{stable}_{camera_name}.mp4"
            for label in ("before", "after")
        }
        comparison_writer = _writer(
            comparison_path, renderer.fps, (3 * renderer.width, renderer.height)
        )
        individual_writers = {
            label: _writer(path, renderer.fps, (renderer.width, renderer.height))
            for label, path in individual_paths.items()
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
                    "CARTESIAN TARGETS IDENTICAL",
                ],
                "PASS" if not events.get("anomalies") else "WARN",
            )
            panels: list[np.ndarray] = []
            for label in ("before", "after"):
                panel = renderer.panel(
                    label,
                    typed_trajectories[label],
                    metrics[label],
                    events,
                    source_name,
                    frame,
                    float(timestamps[frame]),
                    camera_name,
                )
                panels.append(panel)
                individual_writers[label].write(panel)
            comparison_writer.write(np.hstack((source, *panels)))
        comparison_writer.release()
        for writer in individual_writers.values():
            writer.release()
        expected = len(frames)
        for path in (comparison_path, *individual_paths.values()):
            decoded = _decoded_video(path)
            if decoded[0] != expected or abs(decoded[1] - renderer.fps) > 0.1:
                raise RuntimeError(
                    f"render validation failed for {path}: decoded={decoded}, expected={expected}"
                )
        outputs[f"comparison_{camera_name}"] = str(comparison_path.resolve())
        outputs.update(
            {
                f"{label}_{camera_name}": str(path.resolve())
                for label, path in individual_paths.items()
            }
        )
    return {
        "episode_index": int(record["episode_index"]),
        "stable_episode_id": stable,
        "source_name": source_name,
        "target_cartesian_sha256": metrics["after"]["cartesian_target_sha256"],
        "target_hash_equal": target_hash_equal,
        "before_status": metrics["before"]["status"],
        "after_status": metrics["after"]["status"],
        "outputs": outputs,
        "sha256": {name: sha256_file(path) for name, path in outputs.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after-root", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", default="0,24,49")
    parser.add_argument("--cameras", default="overview,top,side")
    args = parser.parse_args()
    before_root = args.before_root.resolve()
    after_root = args.after_root.resolve()
    output_root = args.output_root.resolve()
    episodes = _parse_episodes(args.episodes)
    cameras = tuple(value.strip() for value in args.cameras.split(",") if value.strip())

    common = load_common_config()
    layout = load_scene(common)
    unknown = sorted(set(cameras) - set(layout["camera"]["presets"]))
    if unknown:
        raise ValueError(f"unknown camera presets: {unknown}")
    source_manifest = load_json(before_root / "source_audit/source_manifest.json")
    event_rows = load_json(before_root / "event_audit/events.json")
    records = {int(row["episode_index"]): row for row in source_manifest["records"]}
    g1 = G1Kinematics(common, layout)
    renderer = NaturalArmRenderer(common, layout, g1)
    entries: list[dict[str, Any]] = []
    try:
        for offset, episode in enumerate(episodes, 1):
            entry = render_episode(
                renderer,
                before_root,
                after_root,
                output_root,
                records[episode],
                event_rows[str(episode)],
                cameras,
            )
            entries.append(entry)
            print(
                f"[NATURAL-ARM RENDER {offset:02d}/{len(episodes):02d}] "
                f"episode={episode:03d} views={','.join(cameras)} "
                f"output={entry['outputs']['comparison_overview']}",
                flush=True,
            )
    finally:
        renderer.close()
    manifest = {
        "schema_version": "doll_handoff_natural_arm_review_v1",
        "status": "PASS",
        "layout": "Source ALOHA | Proposed B resolver off | Proposed B common natural-arm",
        "before_root": str(before_root),
        "after_root": str(after_root),
        "scene_config": common["scene_config"],
        "scene_config_sha256": common["scene_config_sha256"],
        "primary_cartesian_targets_changed": False,
        "episode_specific_corrections": False,
        "phase_specific_cartesian_corrections": False,
        "entries": entries,
        "full_50_episode_run": "NOT_STARTED_BY_THIS_ADD_ON",
        "dataset_packaging": "NOT_STARTED_BY_DESIGN",
        "policy_training": "NOT_STARTED_BY_DESIGN",
    }
    manifest_path = output_root / "visual_review_manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"NATURAL_ARM_VISUAL_REVIEW_COMPLETE manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
