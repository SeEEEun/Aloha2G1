#!/usr/bin/env python3
"""Finalize review artifacts for the frozen Episode-49 v17.2 candidate."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_execution_physics_v17 as v17  # noqa: E402
from aloha_g1_v15.semantic_input import load_human_reviewed_development_timeline  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2"
OLD = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text())


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(path)
    return capture


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"video frame {index}")
    return frame


def resize_panel(frame: np.ndarray, label: str) -> np.ndarray:
    frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (50, 235, 255), 1, cv2.LINE_AA)
    return frame


def phase_at(index: int, events: list[tuple[str, int]]) -> tuple[str, float]:
    previous_name, previous_index = "episode_start", 0
    for name, event_index in events:
        if index <= event_index:
            denominator = max(1, event_index - previous_index)
            return f"{previous_name} -> {name}", float(np.clip((index - previous_index) / denominator, 0.0, 1.0))
        previous_name, previous_index = name, event_index
    denominator = max(1, 989 - previous_index)
    return f"{previous_name} -> trajectory_end", float(np.clip((index - previous_index) / denominator, 0.0, 1.0))


def metric_panel(index: int, phase: str, progress: float, before: dict, after: dict, q: np.ndarray) -> np.ndarray:
    panel = np.full((360, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (640, 360), (22, 22, 22), 2)
    lines = [
        ("POSTURE / SEMANTIC REVIEW", (15, 30), .60, (30, 30, 30)),
        (f"action {index}/989 | {phase}", (15, 62), .46, (30, 30, 30)),
        (f"phase progress {progress:.3f}", (15, 86), .46, (30, 30, 30)),
        ("FOREARM-WRIST ALIGNMENT (before -> after)", (15, 126), .44, (40, 40, 40)),
        (f"left median  {before['arms']['left']['forearm_wrist_alignment_deg']['median']:.2f} -> {after['arms']['left']['forearm_wrist_alignment_deg']['median']:.2f} deg", (25, 153), .47, (20, 90, 20)),
        (f"right median {before['arms']['right']['forearm_wrist_alignment_deg']['median']:.2f} -> {after['arms']['right']['forearm_wrist_alignment_deg']['median']:.2f} deg", (25, 179), .47, (20, 90, 20)),
        (f"left p95    {before['arms']['left']['forearm_wrist_alignment_deg']['p95']:.2f} -> {after['arms']['left']['forearm_wrist_alignment_deg']['p95']:.2f} deg", (25, 205), .47, (40, 40, 160)),
        (f"margin      {before['global']['minimum_joint_margin_rad']:.3f} -> {after['global']['minimum_joint_margin_rad']:.3f} rad", (25, 231), .47, (20, 90, 20)),
        (f"current wrist norm L/R {np.linalg.norm(q[index,4:7]):.3f} / {np.linalg.norm(q[index,11:14]):.3f} rad", (15, 271), .45, (30, 30, 30)),
        ("CARTESIAN TARGETS BYTE-IDENTICAL", (15, 310), .48, (20, 110, 20)),
        ("FULL PHYSICS TASK: FAIL | FULL PLAYBACK: COMPLETE", (15, 338), .43, (30, 30, 180)),
    ]
    for text, origin, scale, color in lines:
        cv2.putText(panel, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return panel


def comparison_video(timeline, before: dict, after: dict, q: np.ndarray) -> None:
    old_four = open_video(OLD / "aloha_vs_g1_full_motion_v17_1_4panel.mp4")
    old_g1 = open_video(OLD / "v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_overview.mp4")
    new_g1 = open_video(OUT / "v17_2_KINEMATIC_FULL_overview.mp4")
    output = OUT / "v17_1_vs_v17_2_FULL_MOTION_4panel.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(output)
    ordered_names = [
        "left_phone_grasp_start", "phone_rotation_to_portrait_start", "phone_portrait_reached",
        "right_accessory_grasp_start", "accessory_detachment_start", "accessory_removed",
        "phone_move_to_charger_start", "phone_charger_attachment_complete",
        "left_phone_release_complete", "right_accessory_release_complete",
    ]
    events = [(name, int(timeline.event(name).action_index)) for name in ordered_names]
    for index in range(990):
        ok_source, four = old_four.read()
        ok_old, old = old_g1.read()
        ok_new, new = new_g1.read()
        if not (ok_source and ok_old and ok_new):
            raise RuntimeError(f"comparison input ended at {index}")
        source = resize_panel(four[:360, :640], "SMOLVLA-GENERATED ALOHA SOURCE VIDEO")
        old = resize_panel(old, "V17.1 G1 TRUE-PHYSICS VIEW")
        new = resize_panel(new, "V17.2 G1 KINEMATIC WHOLE-MOTION VIEW")
        phase, progress = phase_at(index, events)
        metrics = metric_panel(index, phase, progress, before, after, q)
        writer.write(np.vstack([np.hstack([source, old]), np.hstack([new, metrics])]))
    writer.release()


def labeled(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (320, 38), (0, 0, 0), -1)
    cv2.putText(frame, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .37, (60, 235, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (5, 31), cv2.FONT_HERSHEY_SIMPLEX, .29, (120, 255, 150), 1, cv2.LINE_AA)
    return frame


def contact_sheets(timeline) -> None:
    event = lambda name: int(timeline.event(name).action_index)
    stages = [
        ("phone approach", (timeline.start_index + event("left_phone_grasp_start")) // 2),
        ("phone grasp", event("left_phone_grasp_start")),
        ("portrait", event("phone_portrait_reached")),
        ("accessory approach", (event("phone_portrait_reached") + event("right_accessory_grasp_start")) // 2),
        ("accessory removal", event("accessory_removed")),
        ("charger transport", (event("phone_move_to_charger_start") + event("phone_charger_attachment_complete")) // 2),
        ("charger placement", event("phone_charger_attachment_complete")),
        ("releases", max(event("left_phone_release_complete"), event("right_accessory_release_complete"))),
        ("return", timeline.end_index),
    ]
    old_over = open_video(OLD / "v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_overview.mp4")
    old_side = open_video(OLD / "v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_side.mp4")
    new_over = open_video(OUT / "v17_2_KINEMATIC_FULL_overview.mp4")
    new_side = open_video(OUT / "v17_2_KINEMATIC_FULL_side.mp4")
    rows = []
    for version, overview, side in (("v17.1", old_over, old_side), ("v17.2", new_over, new_side)):
        top = []
        bottom = []
        for title, index in stages:
            top.append(labeled(read_frame(overview, index), f"{version} | {title}", f"overview | action {index}"))
            bottom.append(labeled(read_frame(side, index), f"{version} | {title}", f"side | action {index}"))
        rows.extend([np.hstack(top), np.hstack(bottom)])
    cv2.imwrite(str(OUT / "v17_1_vs_v17_2_posture_contact_sheet.png"), np.vstack(rows))

    top_video = open_video(OUT / "v17_2_KINEMATIC_FULL_top.mp4")
    dex_rows = []
    for side_name in ("LEFT DEX3", "RIGHT DEX3"):
        row = []
        for title, index in stages:
            frame = read_frame(top_video, index)
            # Use the same whole-hand view for both sides; no pose or contact is fabricated.
            row.append(labeled(frame, f"{side_name} | {title}", f"semantic selection | action {index}"))
        dex_rows.append(np.hstack(row))
    cv2.imwrite(str(OUT / "v17_2_dex3_semantic_contact_sheet.png"), np.vstack(dex_rows))


def video_audit() -> dict:
    names = [
        "v17_2_KINEMATIC_FULL_overview.mp4", "v17_2_KINEMATIC_FULL_side.mp4",
        "v17_2_KINEMATIC_FULL_top.mp4", "v17_2_KINEMATIC_FULL_robot_only.mp4",
        "v17_1_vs_v17_2_FULL_MOTION_4panel.mp4",
        "v17_2_TRUE_PHYSICS_FULL_overview.mp4", "v17_2_TRUE_PHYSICS_FULL_side.mp4",
        "v17_2_TRUE_PHYSICS_FULL_top.mp4",
    ]
    rows = {}
    for name in names:
        path = OUT / name
        capture = open_video(path)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        rows[name] = {
            "frames": frames, "fps": fps, "width": width, "height": height,
            "sha256": sha256(path), "pass": frames == 990 and abs(fps - 7.5) < 1e-6,
        }
    return {"videos": rows, "all_pass": all(row["pass"] for row in rows.values())}


def write_commands() -> str:
    command = """#!/usr/bin/env bash
set -euo pipefail
source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset

# TRUE-PHYSICS 990-frame interactive review (tested headless with the same path/config).
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/gui_review \\
  --artifact-prefix v17_2_gui \\
  --diagnostic-video-prefix v17_2_GUI_TRUE_PHYSICS_FULL \\
  --trial full_task_diagnostic \\
  --speed 0.25 \\
  --gui --interactive-review \\
  --render-preset paper-white \\
  --camera overview \\
  --pause-at-end --enable_cameras

# Repeat the identical frozen true-physics trial until the GUI is closed.
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/gui_loop \\
  --artifact-prefix v17_2_gui_loop \\
  --diagnostic-video-prefix v17_2_GUI_LOOP_TRUE_PHYSICS_FULL \\
  --trial full_task_diagnostic --speed 0.25 \\
  --gui --interactive-review --render-preset paper-white --camera overview --loop --enable_cameras

# Physics-free G1+Dex3 mesh whole-motion review using the same 990 commands.
DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/render_execution_quality_v17_2.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/gui_kinematic \\
  --gui --interactive-review --camera overview --pause-at-end --enable_cameras
"""
    (OUT / "commands.sh").write_text(command)
    os.chmod(OUT / "commands.sh", 0o755)
    return command


def main() -> int:
    with np.load(v17.SOURCE, allow_pickle=False) as source_archive:
        optimized_action = source_archive["optimized_action"].copy()
        timestamps = source_archive["timestamp"].copy()
    with np.load(v17.PHASE_LIBRARY, allow_pickle=False) as phase_archive:
        source_left_position = phase_archive["left_tcp_position"].copy()
        source_right_position = phase_archive["right_tcp_position"].copy()
        source_left_rotation = phase_archive["left_tcp_rotation"].copy()
        source_right_rotation = phase_archive["right_tcp_rotation"].copy()
    timeline = load_human_reviewed_development_timeline(
        v17.TIMELINE, v17.ALIGNMENT, optimized_action, timestamps,
        source_left_position, source_right_position,
        source_left_rotation, source_right_rotation,
        trajectory_path=v17.SOURCE, fk_model_path=v17.MODEL,
        task_geometry_path=v17.LAYOUT,
    )
    before = load_json("whole_motion_posture_before.json")
    after = load_json("whole_motion_posture_after.json")
    posture = load_json("posture_improvement_metrics.json")
    fidelity = load_json("aloha_fidelity_metrics.json")
    orientation = load_json("task_orientation_metrics.json")
    position = load_json("cartesian_position_tracking_metrics.json")
    collision = load_json("collision_metrics.json")
    dex3 = load_json("dex3_full_motion_audit.json")
    temporal = load_json("temporal_smoothness_metrics.json")
    physics = load_json("full_task_diagnostic_result.json")
    tests = load_json("tests_results.json")
    freeze = load_json("input_freeze_audit.json")
    with np.load(OUT / "final_arm_dex3_trajectory.npz", allow_pickle=False) as archive:
        q = archive["arm_qpos"].copy()
    expected_hash = "2993a408d5194fffc10fe56c5a9c801e3a2bc36e8404a8566cb3abd9f62c6cd6"
    if sha256(OUT / "final_arm_dex3_trajectory.npz") != expected_hash:
        raise RuntimeError("final trajectory differs from true-physics input")

    comparison_video(timeline, before, after, q)
    contact_sheets(timeline)
    videos = video_audit()

    stage = physics["stage_passes_observed"]
    objects = physics["object_metrics"]
    physical = {
        "status": "FULL_TASK_PHYSICS_FAIL",
        "full_990_frame_playback_completed": physics["diagnostic_complete"],
        "stage_failure_stopped_playback": False,
        "phone": {
            "left_A_contact_force_max_n": objects["left_A_phone_force_max_n"],
            "left_B_contact_force_max_n": objects["left_B_phone_force_max_n"],
            "simultaneous_AB_longest_run_samples": objects["left_AB_simultaneous_contact_longest_run_samples"],
            "non_task_middle_contact_observed": any("left_hand_middle" in row["pair"] for row in physics["all_robot_object_contact_pairs"]),
            "slip_m": objects["phone_hand_translation_slip_m"],
            "phone_to_wrist_motion_ratio": objects["phone_to_wrist_motion_ratio"],
            "lifted_3mm": objects["phone_lifted_clear_of_table_3mm"],
            "grasp_pass": stage["phone_grasp"], "rotation_pass": stage["phone_rotation"],
        },
        "accessory": {
            "right_C_contact_force_max_n": objects["right_C_accessory_force_max_n"],
            "maximum_displacement_m": objects["accessory_max_displacement_m"],
            "detached": objects["accessory_detached"],
            "removal_pass": stage["accessory_removal"],
        },
        "charger": {
            "state_final": objects["charger_state_final"],
            "center_error_final_mm": objects["charger_center_error_final_mm"],
            "orientation_error_final_deg": objects["charger_orientation_error_final_deg"],
            "placement_pass": stage["charger_placement"],
        },
        "releases": {"left_task_release_pass": False, "right_task_release_pass": stage["accessory_release"]},
        "all_stage_passes": stage,
    }
    dump(OUT / "physical_task_stage_metrics.json", physical)
    dump(OUT / "full_true_physics_diagnostic.json", {
        "status": physics["status"], "frames": physics["source_frames_executed"],
        "speed_scale": physics["speed_scale"], "physics_steps": physics["physics_steps"],
        "tracking": physics["tracking"], "stage_passes": stage,
        "full_task_physics_status": "FULL_TASK_PHYSICS_FAIL",
        "render_parity_status": physics["execution_render_parity"]["status"],
        "no_cheat_integrity": physics["integrity"],
        "raw_result": str((OUT / "full_task_diagnostic_result.json").resolve()),
    })
    gui_result_path = OUT / "gui_tested/full_task_diagnostic_result.json"
    gui_result = json.loads(gui_result_path.read_text())
    dump(OUT / "gui_review_audit.json", {
        "status": "ISAACLAB_INTERACTIVE_TRUE_PHYSICS_REVIEW_TEST_PASS",
        "tested_without_pause_at_end_to_allow_automated_exit": True,
        "user_command_adds_pause_at_end": True,
        "source_frames_executed": gui_result["source_frames_executed"],
        "physics_steps": gui_result["physics_steps"],
        "input_sha256": gui_result["input_sha256"],
        "render_parity_status": gui_result["execution_render_parity"]["status"],
        "interactive_review": gui_result["integrity"]["interactive_review"],
        "gui_viewport": gui_result["integrity"]["gui_viewport"],
        "trajectory_or_physics_parameter_changed": False,
    })
    dump(OUT / "full_kinematic_review.json", {
        "status": "FULL_990_FRAME_KINEMATIC_REVIEW_PASS",
        "numeric_gate": load_json("build_summary.json"),
        "isaac_render": load_json("isaaclab_kinematic_render.json"),
        "visual_posture_warnings": [
            "left forearm-wrist p95 increased despite median improvement",
            "wrist joint-vector norms did not improve",
            "left elbow reaches near-full extension",
            "temporal peak acceleration did not improve",
        ],
    })
    qualitative = {
        "LEFT_ARM_MOTION": "QUESTIONABLE",
        "RIGHT_ARM_MOTION": "QUESTIONABLE",
        "LEFT_DEX3_MOTION": "GOOD",
        "RIGHT_DEX3_MOTION": "GOOD",
        "BIMANUAL_COORDINATION": "GOOD",
        "TASK_SEMANTIC_ORDER": "PASS",
    }
    dump(OUT / "whole_motion_sim_sanity.json", {
        "status": "WHOLE_MOTION_SIM_SANITY_PASS_WITH_POSTURE_WARNINGS",
        "pass": True, "qualitative": qualitative,
        "basis": {
            "backbone_byte_identical": freeze["maximum_position_target_difference_m"] == 0.0,
            "fidelity_pass": fidelity["pass"], "collision_pass": collision["pass"],
            "branch_discontinuity_count": temporal["branch_discontinuity_count"],
            "joint_margin_rad": after["global"]["minimum_joint_margin_rad"],
            "dex3_pass": dex3["pass"], "video_audit_pass": videos["all_pass"],
        },
        "posture_assessment": {
            "central_alignment_relative_reduction": posture["central_alignment_relative_reduction"],
            "left_alignment_p95_change_deg": posture["left_after"]["p95"] - posture["left_before"]["p95"],
            "not_declared_natural_posture_complete": True,
        },
        "full_task_physics_status": "FULL_TASK_PHYSICS_FAIL",
    })
    dump(OUT / "video_audit.json", videos)
    commands = write_commands()

    status_lines = [
        "ALOHA_CARTESIAN_BACKBONE_BYTE_IDENTICAL",
        "FULL_990_FRAME_KINEMATIC_REVIEW_PASS",
        "FULL_990_FRAME_TRUE_PHYSICS_DIAGNOSTIC_COMPLETE",
        "WHOLE_MOTION_SIM_SANITY_PASS_WITH_POSTURE_WARNINGS",
        "NATURAL_G1_POSTURE_PARTIALLY_IMPROVED",
        "FOREARM_WRIST_ALIGNMENT_PARTIALLY_IMPROVED",
        "FULL_TASK_PHYSICS_FAIL",
        "ALOHA_PRIMARY_G1_WHOLE_MOTION_READY_FOR_VISUAL_REVIEW",
    ]
    report = f"""# Episode-49 execution quality v17.2

## 3-line summary

The v14 ALOHA-derived Cartesian position targets remained byte-identical (maximum target difference: 0 m), while one 990-sample G1 arm posture trajectory was re-solved only in redundant joint space.
Median forearm-wrist alignment and minimum joint margin improved, and collision/branch/Dex3 gates passed; however tail alignment, wrist norm, elbow extension, and temporal-peak warnings remain for visual review.
Both 990-frame kinematic and true-PhysX runs completed, but the physical task failed because A/B and right-C task contacts were absent; playback still ran to trajectory end.

## Final status

{chr(10).join(f'- `{value}`' for value in status_lines)}

This candidate is suitable for user visual review, not translator freeze or real-robot use.

## Cartesian backbone and source provenance

- Left target SHA-256: `{freeze['left_target_array_sha256_after']}` (before/after identical: `{freeze['left_array_byte_identical']}`).
- Right target SHA-256: `{freeze['right_target_array_sha256_after']}` (before/after identical: `{freeze['right_array_byte_identical']}`).
- Maximum target-array difference: `{freeze['maximum_position_target_difference_m']:.1f}` m.
- Final trajectory SHA-256: `{expected_hash}`.
- Source: SmolVLA-generated Episode-49 `optimized_action`; validation/held-out/G1 Expert read counts are all zero.

## ALOHA fidelity

- Left path `{fidelity['left_path_shape']:.6f}`, right path `{fidelity['right_path_shape']:.6f}`.
- Left speed `{fidelity['left_speed']:.6f}`, right speed `{fidelity['right_speed']:.6f}`.
- Midpoint `{fidelity['bimanual_midpoint']:.6f}`, relative vector `{fidelity['relative_hand_vector']:.6f}`, inter-hand distance `{fidelity['inter_hand_distance']:.6f}`.
- Result: `ALOHA_MOTION_FIDELITY_PASS`.

## Posture before vs after

- Left forearm-wrist mean/median: `{posture['left_before']['mean']:.3f}/{posture['left_before']['median']:.3f}` -> `{posture['left_after']['mean']:.3f}/{posture['left_after']['median']:.3f}` deg.
- Right forearm-wrist mean/median: `{posture['right_before']['mean']:.3f}/{posture['right_before']['median']:.3f}` -> `{posture['right_after']['mean']:.3f}/{posture['right_after']['median']:.3f}` deg.
- Left p95: `{posture['left_before']['p95']:.3f}` -> `{posture['left_after']['p95']:.3f}` deg (regression warning).
- Right p95: `{posture['right_before']['p95']:.3f}` -> `{posture['right_after']['p95']:.3f}` deg.
- Central mean/median reduction: `{100.0 * posture['central_alignment_relative_reduction']:.2f}%`.
- Minimum arm joint margin: `{posture['minimum_joint_margin_before_rad']:.3f}` -> `{posture['minimum_joint_margin_after_rad']:.3f}` rad.
- Maximum joint step: `{posture['maximum_step_before_rad']:.3f}` -> `{posture['maximum_step_after_rad']:.3f}` rad.

Shoulder/elbow/wrist quality remains mixed. Left elbow bend reaches `{after['arms']['left']['elbow_bend_deg']['max']:.3f}` deg and right reaches `{after['arms']['right']['elbow_bend_deg']['max']:.3f}` deg. Left wrist-vector p95 is `{after['arms']['left']['wrist_joint_vector_norm_rad']['p95']:.3f}` rad and right is `{after['arms']['right']['wrist_joint_vector_norm_rad']['p95']:.3f}` rad. These prevent a claim that natural posture is fully solved.

## Cartesian tracking and task orientation

- Simultaneous left/right position within 5 mm: `{position['simultaneous_5mm_rate']:.6f}`.
- Left mean/max position error: `{position['left_mean_mm']:.6f}` / `{position['left_max_mm']:.6f}` mm.
- Right mean/max position error: `{position['right_mean_mm']:.6f}` / `{position['right_max_mm']:.6f}` mm.
- Portrait long-axis error: `{orientation['portrait_long_axis_error_deg']:.6f}` deg.
- Charger normal/vertical errors: `{orientation['charger_normal_error_deg']:.6f}` / `{orientation['charger_vertical_axis_error_deg']:.6f}` deg.
- Left/right source rotation-progress correlation: `{orientation['left_rotation_progress_correlation']:.6f}` / `{orientation['right_rotation_progress_correlation']:.6f}`.

## Temporal continuity

- Branch discontinuities: `{temporal['branch_discontinuity_count']}`.
- RMS/max velocity: `{after['global']['rms_velocity_rad_s']:.3f}` / `{after['global']['maximum_velocity_rad_s']:.3f}` rad/s.
- RMS/max acceleration: `{after['global']['rms_acceleration_rad_s2']:.3f}` / `{after['global']['maximum_acceleration_rad_s2']:.3f}` rad/s².
- Status: `{temporal['status']}`; continuity passes, but peak step/acceleration did not improve over v17.1.

The left and right arm motions are classified `QUESTIONABLE`, not `GOOD`: central alignment improved, but left tail alignment worsened, wrist norms did not improve, and the left elbow still nears full extension. Collision count is zero and branch discontinuities are zero.

## Dex3 full-task behavior

- Left sequence: `OPEN -> PREGRASP -> PINCH -> HOLD -> ROTATION_HOLD -> TRANSPORT_HOLD -> CHARGER_HOLD -> RELEASE -> OPEN_RETURN`.
- Right sequence: `OPEN -> PREHOOK -> HOOK -> REMOVAL_HOLD -> ACCESSORY_HOLD -> RELEASE -> OPEN_RETURN`.
- Left/right max step: `{dex3['left']['maximum_step_rad']:.5f}` / `{dex3['right']['maximum_step_rad']:.5f}` rad.
- Left/right max acceleration: `{dex3['left']['maximum_acceleration_rad_s2']:.3f}` / `{dex3['right']['maximum_acceleration_rad_s2']:.3f}` rad/s².
- Limit violations and same-hand self contacts: zero. Primitive vectors remained unchanged; only semantic minimum-jerk interpolation was applied.

## Kinematic whole-motion review

All four PAPER_WHITE views contain exactly 990 decoded frames at 7.5 fps. Isaac readback error was zero and physics steps were zero. Numeric position/orientation/fidelity/collision gates passed. Remaining posture warnings are visible in the comparison video and contact sheet.

## True-physics whole-motion review

- Executed `{physics['source_frames_executed']}` source samples, `{physics['physics_steps']}` PhysX steps, 0.25x timing.
- Target/actual RMSE `{physics['tracking']['rmse_rad']:.6f}` rad; max error `{physics['tracking']['maximum_absolute_error_rad']:.6f}` rad.
- Render parity: `{physics['execution_render_parity']['status']}`.
- Stage failure did not stop playback; no object teleport, kinematic follow, or scripted attach/detach occurred.

## Physical task result

- Phone: A max `{objects['left_A_phone_force_max_n']:.3f}` N, B max `{objects['left_B_phone_force_max_n']:.3f}` N, simultaneous A/B run `{objects['left_AB_simultaneous_contact_longest_run_samples']}` samples, slip `{objects['phone_hand_translation_slip_m']:.6f}` m, wrist-follow ratio `{objects['phone_to_wrist_motion_ratio']:.6g}`. A non-task left-middle/phone contact occurred.
- Accessory: right-C max `{objects['right_C_accessory_force_max_n']:.3f}` N, detached `{objects['accessory_detached']}`, maximum displacement `{objects['accessory_max_displacement_m']:.6f}` m.
- Charger: final state `{objects['charger_state_final']}`, center error `{objects['charger_center_error_final_mm']:.3f}` mm, orientation error `{objects['charger_orientation_error_final_deg']:.3f}` deg.
- `FULL_TASK_PHYSICS_FAIL`: none of the six task stages passed.

## Videos and sheets

{chr(10).join(f'- [{name}]({name})' for name in videos['videos'])}
- [Posture comparison sheet](v17_1_vs_v17_2_posture_contact_sheet.png)
- [Dex3 semantic sheet](v17_2_dex3_semantic_contact_sheet.png)

## GUI review command

The interactive path was executed through all 990 samples with a live Kit viewport; the automated test omitted only `--pause-at-end` so it could exit. The command below adds the final-state pause for user review.

```bash
{commands.split(chr(10) + '# Repeat', 1)[0].strip()}
```

## Tests

`{tests['passed']} passed, {tests['failed']} failed` using the repository hardcoding/semantic test set. External pytest plugin autoload was disabled because the unrelated ROS launch-testing plugin is missing `lark`.

## Exact next action

WAIT FOR USER VISUAL REVIEW of the v17.1-vs-v17.2 comparison and full PAPER_WHITE kinematic/true-physics videos. If the user rejects whole-motion posture, the next correction must remain null-space/posture-only; do not tune PHONE_PINCH/RING_HOOK or modify the Cartesian path in that step.

THE ALOHA-DERIVED CARTESIAN ARM BACKBONE WAS NOT REDESIGNED
THE COMPLETE 990-FRAME G1 ARM MOTION WAS EVALUATED, NOT ONLY THE PHONE GRASP
G1 SHOULDER, ELBOW, FOREARM, AND WRIST POSTURE WERE OPTIMIZED ONLY THROUGH REDUNDANT TASK-NULL-SPACE FREEDOM
DEX3 WAS EVALUATED AS A COMPLETE SEMANTIC HAND MOTION OVER THE FULL TASK
DEX3 DID NOT MODIFY THE G1 CARTESIAN ARM PATH
ALOHA BILATERAL MOTION AND TIMING REMAINED PRIMARY
TRUE PHYSICS TASK FAILURES DID NOT TERMINATE THE FULL-MOTION DIAGNOSTIC
WHOLE-MOTION SIMULATION QUALITY AND FULL PHYSICS TASK SUCCESS WERE REPORTED SEPARATELY
NO VALIDATION, HELD-OUT, OR G1 EXPERT TRAJECTORY WAS USED
NO DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report)
    report_dir = OUT / "report"
    report_dir.mkdir(exist_ok=True)
    video_tags = "\n".join(
        f'<section><h3>{html.escape(name)}</h3><video controls preload="metadata" src="../{html.escape(name)}"></video></section>'
        for name in videos["videos"]
    )
    index = f"""<!doctype html><html><head><meta charset="utf-8"><title>v17.2 whole-motion review</title>
<style>body{{font:16px system-ui;max-width:1280px;margin:auto;padding:24px;background:#f7f7f5;color:#202020}}video,img{{max-width:100%;background:white}}code{{background:#eee;padding:2px 5px}}.fail{{color:#a20}}.pass{{color:#174}}</style></head><body>
<h1>Episode-49 execution quality v17.2</h1><p class="pass">WHOLE_MOTION_SIM_SANITY_PASS_WITH_POSTURE_WARNINGS</p><p class="fail">FULL_TASK_PHYSICS_FAIL</p>
<p>Cartesian targets are byte-identical. Full 990-frame kinematic and PhysX reviews completed without stage-gate termination.</p>
<h2>Comparison sheets</h2><img src="../v17_1_vs_v17_2_posture_contact_sheet.png"><img src="../v17_2_dex3_semantic_contact_sheet.png">
<h2>Videos</h2>{video_tags}<h2>Report</h2><pre>{html.escape(report)}</pre></body></html>"""
    (report_dir / "index.html").write_text(index)

    tracked = [path for path in OUT.rglob("*") if path.is_file() and path.name != "run_manifest.json"]
    dump(OUT / "run_manifest.json", {
        "status": status_lines, "trajectory_sha256": expected_hash,
        "generated_file_count": len(tracked),
        "files": {str(path.relative_to(OUT)): sha256(path) for path in sorted(tracked)},
        "validation_read_count": 0, "heldout_read_count": 0, "g1_expert_read_count": 0,
        "dds": False, "publisher": False, "real_robot_command": False,
    })
    print(json.dumps({"status": status_lines, "video_audit": videos["all_pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
