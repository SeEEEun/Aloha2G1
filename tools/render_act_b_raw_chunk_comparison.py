#!/usr/bin/env python3
"""Render raw Dataset/SmolVLA/ACT chunks as direct kinematic G1 poses.

There are deliberately zero MuJoCo physics steps and no interpolation between
the 50 policy outputs.  Each array row is rendered once at 30 fps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    atomic_json,
    load_common_config,
    load_scene,
)
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.doll_handoff_retargeting.render import (  # noqa: E402
    _decoded_video,
    _writer,
    add_approved_scene,
    camera_from_layout,
)


DEFAULT_INPUT = ROOT / "outputs/policy_b_act/comparison/raw_target_vs_smolvla_vs_act_chunks.npz"
METRICS = ROOT / "outputs/policy_b_act/raw_diagnostics/raw_single_chunk_smoothness.json"
OUTPUT = ROOT / "outputs/policy_b_act/comparison"
ACTION_FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
FPS = 30.0
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
SOURCES = (
    ("dataset_target", "DATASET TARGET", (80, 210, 120)),
    ("smolvla_raw_fixed_noise", "SMOLVLA RAW", (80, 100, 235)),
    ("act_raw", "ACT RAW", (235, 145, 60)),
)
REQUIRED_OVERVIEW = ("left_approach", "doll_plateau", "handoff_approach")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--camera", default="overview")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlay(
    image: np.ndarray,
    source_title: str,
    phase: str,
    frame: int,
    color: tuple[int, int, int],
    dynamics: Mapping[str, Any],
) -> np.ndarray:
    result = np.ascontiguousarray(image.copy())
    cv2.rectangle(result, (0, 0), (result.shape[1], 92), (16, 16, 20), -1)
    cv2.putText(
        result,
        source_title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        f"{phase.replace('_', ' ')} | raw row {frame:02d}/49 | t={frame / FPS:.3f}s",
        (12, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        (
            f"arm rev={dynamics['arm_direction_reversals_per_s_mean']:.2f}/s  "
            f"max step={dynamics['adjacent_step_max_abs_rad']:.4f} rad  "
            f"jerk RMS={dynamics['jerk_rms_rad_s3']:.1f}"
        ),
        (12, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "KINEMATIC REPLAY | NO PHYSICS / FILTER / EXECUTION ADAPTER",
        (12, result.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (150, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return result


class RawChunkRenderer:
    def __init__(self, common: Mapping[str, Any], layout: Mapping[str, Any], camera: str):
        self.layout = layout
        self.g1 = G1Kinematics(common, layout)
        policy_names = json.loads(ACTION_FREEZE.read_text(encoding="utf-8"))["joint_names"]
        if policy_names[:14] != self.g1.arm_joint_names.tolist():
            raise RuntimeError("Dataset-B arm order differs from G1 kinematic model")
        self.left_policy_indices = np.asarray(
            [policy_names.index(name) for name in self.g1.hand_joint_names["left"]],
            dtype=np.int64,
        )
        self.right_policy_indices = np.asarray(
            [policy_names.index(name) for name in self.g1.hand_joint_names["right"]],
            dtype=np.int64,
        )
        self.camera = camera_from_layout(layout, camera)
        self.data = [mujoco.MjData(self.g1.model) for _ in SOURCES]
        self.renderers = [
            mujoco.Renderer(self.g1.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            for _ in SOURCES
        ]
        self.root_position = np.asarray(
            layout["g1"]["root_position_world_xyz_m"], dtype=np.float64
        )
        self.root_quaternion = np.asarray(
            layout["g1"]["root_orientation_world_wxyz"], dtype=np.float64
        )
        self.mujoco_physics_step_count = 0

    def close(self) -> None:
        for renderer in self.renderers:
            renderer.close()

    def panel(self, slot: int, q: np.ndarray) -> np.ndarray:
        if q.shape != (28,) or not np.isfinite(q).all():
            raise RuntimeError(f"invalid logical G1 pose {q.shape}")
        data = self.data[slot]
        data.qpos[:] = self.g1.stand_qpos
        data.qpos[:3] = self.root_position
        data.qpos[3:7] = self.root_quaternion
        data.qpos[self.g1.arm_qpos_ids] = q[:14]
        # Dataset-B logical order is thumb/middle/index while this MuJoCo
        # model enumerates thumb/index/middle.  Reorder strictly by names.
        data.qpos[self.g1.hand_qpos_ids["left"]] = q[self.left_policy_indices]
        data.qpos[self.g1.hand_qpos_ids["right"]] = q[self.right_policy_indices]
        data.qvel[:] = 0.0
        # Forward kinematics only; mj_step is intentionally never called.
        mujoco.mj_forward(self.g1.model, data)
        self.renderers[slot].update_scene(data, self.camera)
        add_approved_scene(self.renderers[slot].scene, self.layout, None, None)
        return cv2.cvtColor(self.renderers[slot].render(), cv2.COLOR_RGB2BGR)


def render_triptych_frame(
    renderer: RawChunkRenderer,
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    label: str,
    label_index: int,
    frame: int,
) -> np.ndarray:
    panels = []
    for slot, (key, title, color) in enumerate(SOURCES):
        image = renderer.panel(slot, arrays[key][label_index, frame])
        panels.append(
            overlay(
                image,
                title,
                label,
                frame,
                color,
                metrics["per_condition"][label][key],
            )
        )
    return np.hstack(panels)


def make_contact_sheet(frames: list[np.ndarray], output: Path) -> None:
    if len(frames) != 6:
        raise RuntimeError("contact sheet requires six frames")
    half = [cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)) for frame in frames]
    sheet = np.vstack((np.hstack(half[:3]), np.hstack(half[3:])))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"failed to write {output}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file() or not METRICS.is_file():
        raise FileNotFoundError(input_path if not input_path.is_file() else METRICS)
    with np.load(input_path, allow_pickle=False) as archive:
        labels = archive["labels"].astype(str).tolist()
        arrays = {key: archive[key].astype(np.float64) for key, _, _ in SOURCES}
    for key, values in arrays.items():
        if values.shape != (7, 50, 28) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid {key} chunks {values.shape}")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    common = load_common_config()
    layout = load_scene(common)
    renderer = RawChunkRenderer(common, layout, args.camera)
    manifest_outputs: dict[str, Any] = {}
    try:
        for label_index, label in enumerate(labels):
            path = OUTPUT / "raw_chunk_videos" / f"{label}_target_vs_smolvla_vs_act.mp4"
            writer = _writer(path, FPS, (3 * PANEL_WIDTH, PANEL_HEIGHT))
            selected_frames = []
            try:
                for frame in range(50):
                    triptych = render_triptych_frame(
                        renderer, arrays, metrics, label, label_index, frame
                    )
                    writer.write(triptych)
                    if frame in (0, 10, 20, 30, 40, 49):
                        selected_frames.append(triptych.copy())
            finally:
                writer.release()
            decoded = _decoded_video(path)
            if decoded != (50, FPS, 3 * PANEL_WIDTH, PANEL_HEIGHT):
                raise RuntimeError(f"decoded video contract mismatch {path}: {decoded}")
            contact_sheet = path.with_suffix(".contact_sheet.png")
            make_contact_sheet(selected_frames, contact_sheet)
            manifest_outputs[label] = {
                "video": str(path),
                "video_sha256": sha256_file(path),
                "contact_sheet": str(contact_sheet),
                "decoded": list(decoded),
            }

        overview_path = OUTPUT / "raw_target_vs_smolvla_vs_act_overview.mp4"
        writer = _writer(overview_path, FPS, (3 * PANEL_WIDTH, PANEL_HEIGHT))
        overview_contact_frames = []
        try:
            for label in REQUIRED_OVERVIEW:
                label_index = labels.index(label)
                for frame in range(50):
                    triptych = render_triptych_frame(
                        renderer, arrays, metrics, label, label_index, frame
                    )
                    writer.write(triptych)
                    if frame in (0, 25, 49):
                        overview_contact_frames.append(triptych.copy())
        finally:
            writer.release()
        decoded = _decoded_video(overview_path)
        if decoded != (150, FPS, 3 * PANEL_WIDTH, PANEL_HEIGHT):
            raise RuntimeError(f"decoded overview contract mismatch: {decoded}")
        # Nine samples fit a 3x3 sheet; resize each triptych to 960x240.
        samples = [cv2.resize(frame, (960, 240)) for frame in overview_contact_frames]
        overview_sheet = np.vstack(
            (np.hstack(samples[:3]), np.hstack(samples[3:6]), np.hstack(samples[6:9]))
        )
        overview_contact_path = OUTPUT / "raw_target_vs_smolvla_vs_act_overview.contact_sheet.png"
        if not cv2.imwrite(str(overview_contact_path), overview_sheet):
            raise RuntimeError("failed to write overview contact sheet")
        manifest_outputs["required_overview"] = {
            "phases": list(REQUIRED_OVERVIEW),
            "video": str(overview_path),
            "video_sha256": sha256_file(overview_path),
            "contact_sheet": str(overview_contact_path),
            "decoded": list(decoded),
        }
    finally:
        renderer.close()

    manifest = {
        "schema_version": "act_b_raw_kinematic_replay_v1",
        "status": "PASS",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "camera": args.camera,
        "fps": FPS,
        "one_raw_array_row_per_video_frame": True,
        "interpolation_used": False,
        "mujoco_forward_kinematics_calls": 7 * 50 * 3 + 3 * 50 * 3,
        "mujoco_physics_steps": renderer.mujoco_physics_step_count,
        "physics_smoothing": False,
        "execution_adapter": False,
        "temporal_smoothing": False,
        "replanning": False,
        "hardware_transport": False,
        "outputs": manifest_outputs,
    }
    atomic_json(OUTPUT / "raw_kinematic_replay_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
