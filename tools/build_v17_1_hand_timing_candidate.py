#!/usr/bin/env python3
"""Build a Dex3-only semantic timing diagnostic over the fixed v17.1 arm q."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_execution_physics_v17 as v17  # noqa: E402
from aloha_g1_v15.kinematics import ActiveG1Dex3  # noqa: E402
from aloha_g1_v15.semantic_input import load_human_reviewed_development_timeline  # noqa: E402
from aloha_g1_v17.trajectory import build_predefined_hand_trajectories  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0.05 <= args.completion <= 1.0:
        raise ValueError("completion must be in [0.05,1]")

    source_path = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
    phase_path = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12/aloha_phase_motion_library.npz"
    base_path = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz"
    primitive_path = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/dex3_magsafe_execution_primitives_v17_1.sim.json"

    with np.load(source_path, allow_pickle=False) as archive:
        action = archive["optimized_action"].copy()
        timestamps = archive["timestamp"].copy()
    with np.load(phase_path, allow_pickle=False) as archive:
        fk_values = [archive[key].copy() for key in (
            "left_tcp_position", "right_tcp_position", "left_tcp_rotation", "right_tcp_rotation"
        )]
    timeline = load_human_reviewed_development_timeline(
        v17.TIMELINE, v17.ALIGNMENT, action, timestamps, *fk_values,
        trajectory_path=source_path, fk_model_path=v17.MODEL, task_geometry_path=v17.LAYOUT,
    )
    with np.load(base_path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    primitives = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in json.loads(primitive_path.read_text(encoding="utf-8"))["primitives"].items()
    }
    runtime = ActiveG1Dex3(v17.MODEL, v17.DEX3_MAPPING, v17.PALM_CONFIG, payload["g1_root"])
    left, right, _ = build_predefined_hand_trajectories(
        timeline, runtime, primitives, action[:, 6], action[:, 13],
        left_lock_end_event="phone_rotation_to_portrait_start",
        left_digit_staging_mode="simultaneous_opposed_close",
        left_lock_progress_mode="event_interval_minimum_jerk",
        left_lock_completion_progress=float(args.completion),
    )
    payload["left_dex3_qpos"] = left
    payload["right_dex3_qpos"] = right
    payload["full_joint_q"] = np.c_[payload["g1_arm_q"], left, right]
    payload["hand_timing_diagnostic"] = np.asarray(True)
    payload["left_lock_completion_progress"] = np.asarray(float(args.completion))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "completion": args.completion,
        "maximum_left_step_rad": float(np.max(np.abs(np.diff(left, axis=0)))),
        "arm_q_byte_identical": bool(np.array_equal(payload["g1_arm_q"], payload["arm_qpos"])),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
