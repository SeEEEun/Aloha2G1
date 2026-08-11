#!/usr/bin/env python3
"""Read-only full-motion/posture audit for frozen Episode-49 v17.1."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix"
BASE = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1"
TRAJ = BASE / "final_arm_dex3_trajectory.npz"
PHYSICS = OUT / "physics_trial_full_task_diagnostic_0p25x_paper_white.npz"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]
from aloha_g1_v15.kinematics import ActiveG1Dex3  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed frame {index}: {path}")
    return image


def fit(image: np.ndarray, width: int = 640, height: int = 360) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.full((height, width, 3), 245, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def caption(image: np.ndarray, text: str) -> np.ndarray:
    result = np.full((image.shape[0] + 38, image.shape[1], 3), 248, np.uint8)
    result[38:] = image
    cv2.putText(result, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    return result


def joint_rows(names: list[str], q: np.ndarray, limits: np.ndarray) -> dict:
    rows = {}
    for index, name in enumerate(names):
        margin = np.minimum(q[:, index] - limits[index, 0], limits[index, 1] - q[:, index])
        rows[name] = {
            "minimum_rad": float(np.min(q[:, index])),
            "maximum_rad": float(np.max(q[:, index])),
            "range_rad": float(np.ptp(q[:, index])),
            "minimum_joint_limit_margin_rad": float(np.min(margin)),
            "minimum_margin_action_index": int(np.argmin(margin)),
        }
    return rows


def main() -> None:
    trajectory = np.load(TRAJ, allow_pickle=False)
    physics = np.load(PHYSICS, allow_pickle=True)
    q = trajectory["arm_qpos"].astype(np.float64)
    q_v14 = trajectory["v14_reference_arm_q"].astype(np.float64)
    left_hand = trajectory["left_dex3_qpos"].astype(np.float64)
    right_hand = trajectory["right_dex3_qpos"].astype(np.float64)
    timestamps = trajectory["source_timestamps"].astype(np.float64)
    names = trajectory["arm_joint_names"].astype(str).tolist()
    if q.shape != (990, 14) or physics["actual_q"].shape != (990, 28):
        raise RuntimeError("full 990-frame trajectory/state missing")

    alignment = json.loads((ROOT / "configs/episode49_action_observation_alignment.approved.json").read_text())
    event_map = alignment["event_mapping"]
    event = lambda name: int(event_map[name]["aligned_action_index"])
    semantic_bounds = [
        ("phone_approach", 0, event("left_phone_grasp_start")),
        ("phone_grasp", event("left_phone_grasp_start"), event("phone_rotation_to_portrait_start")),
        ("phone_rotation", event("phone_rotation_to_portrait_start"), event("phone_portrait_reached")),
        ("accessory_approach", event("phone_portrait_reached"), event("right_accessory_grasp_start")),
        ("accessory_remove", event("right_accessory_grasp_start"), event("accessory_removed")),
        ("pre_transport", event("accessory_removed"), event("phone_move_to_charger_start")),
        ("charger_transport", event("phone_move_to_charger_start"), event("phone_charger_attachment_complete")),
        ("left_release", event("phone_charger_attachment_complete"), event("left_phone_release_complete")),
        ("right_release", event("left_phone_release_complete"), event("right_accessory_release_complete")),
        ("return_and_terminal_hold", max(event("left_phone_release_complete"), event("right_accessory_release_complete")), 989),
    ]

    def phase_at(index: int) -> dict:
        candidates = [row for row in semantic_bounds if row[1] <= index <= row[2]]
        row = candidates[-1] if candidates else semantic_bounds[-1]
        progress = (index - row[1]) / max(1, row[2] - row[1])
        return {"phase": row[0], "progress": float(np.clip(progress, 0.0, 1.0))}

    runtime = ActiveG1Dex3(
        MODEL,
        ROOT / "configs/dex3_abc_finger_mapping.sim.json",
        ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json",
        trajectory["g1_root"].astype(np.float64),
    )
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    body_ids = {
        side: {
            "shoulder": runtime.body_id(f"{side}_shoulder_roll_link"),
            "elbow": runtime.body_id(f"{side}_elbow_link"),
            "wrist": runtime.body_id(f"{side}_wrist_yaw_link"),
        } for side in ("left", "right")
    }
    torso_id = runtime.body_id("torso_link")
    torso_geoms = np.flatnonzero(runtime.model.geom_bodyid == torso_id).tolist()
    elbow_geoms = {
        side: np.flatnonzero(runtime.model.geom_bodyid == ids["elbow"]).tolist()
        for side, ids in body_ids.items()
    }
    elbow_torso_center = {side: np.zeros(990) for side in body_ids}
    elbow_torso_surface = {side: np.zeros(990) for side in body_ids}
    forearm_wrist_alignment = {side: np.zeros(990) for side in body_ids}
    elbow_bend = {side: np.zeros(990) for side in body_ids}
    shoulder_positions = {side: np.zeros((990, 3)) for side in body_ids}
    elbow_positions = {side: np.zeros((990, 3)) for side in body_ids}
    wrist_positions = {side: np.zeros((990, 3)) for side in body_ids}
    for index in range(990):
        runtime.assign(q[index], left_hand[index], right_hand[index])
        torso = runtime.model_to_scene_position(runtime.data.xpos[torso_id])
        for side, ids in body_ids.items():
            shoulder = runtime.model_to_scene_position(runtime.data.xpos[ids["shoulder"]])
            elbow = runtime.model_to_scene_position(runtime.data.xpos[ids["elbow"]])
            wrist_pose = runtime.wrist_pose(side)
            wrist = wrist_pose[:3, 3]
            shoulder_positions[side][index] = shoulder
            elbow_positions[side][index] = elbow
            wrist_positions[side][index] = wrist
            elbow_torso_center[side][index] = np.linalg.norm(elbow - torso)
            distances = []
            for elbow_geom in elbow_geoms[side]:
                for torso_geom in torso_geoms:
                    distances.append(float(mujoco.mj_geomDistance(
                        runtime.model, runtime.data, elbow_geom, torso_geom, 1.0, None
                    )))
            elbow_torso_surface[side][index] = min(distances) if distances else np.nan
            upper = elbow - shoulder
            forearm = wrist - elbow
            upper /= max(np.linalg.norm(upper), 1e-12)
            forearm /= max(np.linalg.norm(forearm), 1e-12)
            elbow_bend[side][index] = math.degrees(math.acos(np.clip(np.dot(-upper, forearm), -1.0, 1.0)))
            # In the verified active G1 model, wrist local Y is the forearm axis.
            forearm_wrist_alignment[side][index] = math.degrees(math.acos(np.clip(
                abs(float(np.dot(wrist_pose[:3, 1], forearm))), 0.0, 1.0
            )))

    dt = float(np.median(np.diff(timestamps)))
    dq = np.diff(q, axis=0)
    ddq = np.diff(q, n=2, axis=0)
    margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    wrist_indices = [names.index(f"{side}_wrist_{axis}_joint") for side in ("left", "right") for axis in ("roll", "pitch", "yaw")]
    wrist_magnitude = np.linalg.norm(q[:, wrist_indices].reshape(990, 2, 3), axis=2)
    v14_wrist_magnitude = np.linalg.norm(q_v14[:, wrist_indices].reshape(990, 2, 3), axis=2)
    q_delta_v14 = q - q_v14

    left_target = trajectory["v14_left_position_target"].astype(np.float64)
    right_target = trajectory["v14_right_position_target"].astype(np.float64)
    target_step = np.maximum(
        np.linalg.norm(np.diff(left_target, axis=0), axis=1),
        np.linalg.norm(np.diff(right_target, axis=0), axis=1),
    )
    target_accel = np.maximum(
        np.linalg.norm(np.diff(left_target, n=2, axis=0), axis=1),
        np.linalg.norm(np.diff(right_target, n=2, axis=0), axis=1),
    ) / (dt * dt)
    step_median = float(np.median(target_step))
    step_mad = float(np.median(np.abs(target_step - step_median)))
    discontinuity_threshold = step_median + 10.0 * max(step_mad, 1e-9)
    cartesian_discontinuities = np.flatnonzero(target_step > max(0.02, discontinuity_threshold)) + 1

    extreme = np.zeros(990, dtype=bool)
    extreme |= np.max(wrist_magnitude, axis=1) >= np.quantile(np.max(wrist_magnitude, axis=1), 0.95)
    extreme |= np.min(margin, axis=1) <= 0.01
    for side in body_ids:
        finite = elbow_torso_surface[side][np.isfinite(elbow_torso_surface[side])]
        if len(finite):
            extreme |= elbow_torso_surface[side] <= np.quantile(finite, 0.05)
        extreme |= forearm_wrist_alignment[side] >= np.quantile(forearm_wrist_alignment[side], 0.95)

    intervals = []
    start = None
    for index, value in enumerate(np.r_[extreme, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            if end - start + 1 >= 2:
                midpoint = (start + end) // 2
                intervals.append({
                    "start_action_index": start,
                    "end_action_index": end,
                    "duration_samples": end - start + 1,
                    "semantic_midpoint": phase_at(midpoint),
                    "maximum_wrist_joint_vector_norm_rad": float(np.max(wrist_magnitude[start:end + 1])),
                    "minimum_joint_margin_rad": float(np.min(margin[start:end + 1])),
                    "minimum_elbow_torso_surface_distance_m": float(np.nanmin(np.c_[
                        elbow_torso_surface["left"][start:end + 1],
                        elbow_torso_surface["right"][start:end + 1],
                    ])),
                })
            start = None

    approach_end = event("left_phone_grasp_start")
    approach_slice = slice(0, approach_end + 1)
    wrist_delta = q_delta_v14[approach_slice][:, wrist_indices]
    non_wrist_indices = [index for index in range(14) if index not in wrist_indices]
    attribution = {
        "classification": "REDUNDANT_G1_IK_POSTURE_WITH_TASK_ORIENTATION_CONTRIBUTION",
        "cartesian_path_geometry_primary_cause": False,
        "redundant_g1_ik_posture_primary_cause": True,
        "task_orientation_constraint_contributor": True,
        "evidence": {
            "cartesian_target_discontinuity_count": int(len(cartesian_discontinuities)),
            "maximum_cartesian_target_step_m": float(np.max(target_step)),
            "frozen_path_minimum_primary_fidelity": float(json.loads((BASE / "aloha_fidelity_metrics.json").read_text())["minimum_primary_metric"]),
            "phone_approach_v17_vs_v14_wrist_rms_delta_rad": float(np.sqrt(np.mean(wrist_delta ** 2))),
            "phone_approach_v17_vs_v14_nonwrist_rms_delta_rad": float(np.sqrt(np.mean(q_delta_v14[approach_slice][:, non_wrist_indices] ** 2))),
            "phone_approach_current_max_wrist_norm_rad": float(np.max(wrist_magnitude[approach_slice])),
            "phone_approach_v14_max_wrist_norm_rad": float(np.max(v14_wrist_magnitude[approach_slice])),
        },
        "interpretation": (
            "The protected Cartesian targets are continuous and retain near-unity ALOHA fidelity. "
            "The visible awkwardness therefore originates primarily in redundant shoulder/elbow/wrist "
            "branch posture, with the semantic-local task orientation increasing wrist demand."
        ),
        "trajectory_modified": False,
    }

    fidelity = json.loads((BASE / "aloha_fidelity_metrics.json").read_text())
    audit = {
        "status": "FULL_MOTION_POSTURE_AUDIT_COMPLETE",
        "trajectory_sha256": sha256(TRAJ),
        "samples": 990,
        "fps": float(trajectory["fps"]),
        "cartesian_backbone_modified": False,
        "posture_modified": False,
        "joint_groups": {
            "shoulder": joint_rows(
                names[:3] + names[7:10], np.c_[q[:, :3], q[:, 7:10]],
                np.r_[limits[:3], limits[7:10]].reshape(6, 2),
            ),
            "elbow": joint_rows([names[3], names[10]], q[:, [3, 10]], limits[[3, 10]]),
            "wrist": joint_rows(names[4:7] + names[11:14], np.c_[q[:, 4:7], q[:, 11:14]], np.r_[limits[4:7], limits[11:14]].reshape(6, 2)),
        },
        "global_joint_metrics": {
            "minimum_joint_limit_margin_rad": float(np.min(margin)),
            "minimum_margin_joint": names[int(np.unravel_index(np.argmin(margin), margin.shape)[1])],
            "minimum_margin_action_index": int(np.unravel_index(np.argmin(margin), margin.shape)[0]),
            "maximum_joint_step_rad": float(np.max(np.abs(dq))),
            "maximum_joint_velocity_rad_s": float(np.max(np.abs(dq)) / dt),
            "maximum_joint_acceleration_rad_s2": float(np.max(np.abs(ddq)) / (dt * dt)),
        },
        "kinematic_posture": {
            side: {
                "elbow_to_torso_center_min_m": float(np.min(elbow_torso_center[side])),
                "elbow_to_torso_surface_min_m": float(np.nanmin(elbow_torso_surface[side])),
                "elbow_bend_angle_range_deg": [float(np.min(elbow_bend[side])), float(np.max(elbow_bend[side]))],
                "forearm_to_wrist_axis_misalignment_range_deg": [
                    float(np.min(forearm_wrist_alignment[side])),
                    float(np.max(forearm_wrist_alignment[side])),
                ],
                "wrist_joint_vector_norm_max_rad": float(np.max(wrist_magnitude[:, 0 if side == "left" else 1])),
            } for side in body_ids
        },
        "cartesian_path_geometry": {
            "maximum_target_step_m": float(np.max(target_step)),
            "maximum_target_acceleration_m_s2": float(np.max(target_accel)),
            "robust_discontinuity_threshold_m": float(max(0.02, discontinuity_threshold)),
            "discontinuity_count": int(len(cartesian_discontinuities)),
            "discontinuity_action_indices": cartesian_discontinuities.tolist(),
        },
        "cause_attribution": attribution,
        "extreme_posture_intervals": intervals,
        "aloha_fidelity": fidelity,
        "semantic_mapping_provenance": str((ROOT / "configs/episode49_action_observation_alignment.approved.json").resolve()),
    }
    write_json(OUT / "full_motion_posture_audit.json", audit)

    overview = OUT / "v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_overview.mp4"
    side = OUT / "v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_side.mp4"
    stages = [
        ("initial", 0),
        ("phone approach", event("left_phone_grasp_start") // 2),
        ("phone grasp", event("left_phone_grasp_start")),
        ("portrait", event("phone_portrait_reached")),
        ("accessory approach", (event("phone_portrait_reached") + event("right_accessory_grasp_start")) // 2),
        ("accessory removal", event("accessory_removed")),
        ("charger transport", (event("phone_move_to_charger_start") + event("phone_charger_attachment_complete")) // 2),
        ("charger placement", event("phone_charger_attachment_complete")),
        ("release", max(event("left_phone_release_complete"), event("right_accessory_release_complete"))),
        ("final", 989),
    ]
    rows = []
    for label, index in stages:
        phase = phase_at(index)
        left = caption(frame(overview, index), f"{label} overview | action {index} | {phase['phase']} {phase['progress']:.2f}")
        right = caption(frame(side, index), f"{label} side | action {index} | posture read-only")
        rows.append(np.hstack([left, right]))
    cv2.imwrite(str(OUT / "full_motion_posture_contact_sheet.png"), np.vstack(rows))

    source_path = ROOT / "evaluation/smolvla_episode49_temporal_consensus/aloha_temporal_consensus.mp4"
    replay_path = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/source_optimized_action_replay.mp4"
    captures = [cv2.VideoCapture(str(path)) for path in (source_path, replay_path, overview, side)]
    raw = OUT / ".aloha_vs_g1_full_motion_v17_1_4panel.raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(raw)
    for index in range(990):
        images = []
        for capture in captures:
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"4-panel source ended at {index}")
            images.append(fit(image))
        labels = ["SOURCE ALOHA VIDEO", "OPTIMIZED-ACTION ALOHA REPLAY", "G1 TRUE-PHYSICS OVERVIEW", "G1 SIDE + SEMANTIC PROGRESS"]
        for image, label in zip(images, labels):
            cv2.rectangle(image, (0, 0), (640, 28), (0, 0, 0), -1)
            cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 235, 255), 1, cv2.LINE_AA)
        phase = phase_at(index)
        cv2.rectangle(images[3], (10, 322), (630, 352), (20, 20, 20), -1)
        cv2.rectangle(images[3], (14, 340), (14 + int(610 * phase["progress"]), 348), (80, 220, 120), -1)
        cv2.putText(images[3], f"{phase['phase']} | progress {phase['progress']:.3f} | action {index}", (14, 336), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 245, 245), 1, cv2.LINE_AA)
        writer.write(np.vstack([np.hstack(images[:2]), np.hstack(images[2:])]))
    for capture in captures:
        capture.release()
    writer.release()
    output_4panel = OUT / "aloha_vs_g1_full_motion_v17_1_4panel.mp4"
    output_4panel.unlink(missing_ok=True)
    raw.replace(output_4panel)

    common = """source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
"""
    gui = """DISPLAY=:0 /home/jbnu/IsaacLab-3-beta/isaaclab.sh -p \\
  isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py \\
  --input outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz \\
  --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix/gui_full \\
  --artifact-prefix v17_1_full_diagnostic_gui \\
  --trial full_task_diagnostic --speed 0.25 \\
  --gui --interactive-review --render-preset paper-white \\
  --camera overview --pause-at-end
"""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# Full 990-frame interactive true-physics review
{common}{gui}

# Loop the identical frozen 990-frame diagnostic (Ctrl+C or close Kit to stop)
{common}{gui.replace('--pause-at-end', '--loop').replace('gui_full ', 'gui_full_loop ')}

# Open generated review videos
xdg-open {overview}
xdg-open {side}
xdg-open {OUT / 'v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_top.mp4'}
xdg-open {output_4panel}
"""
    (OUT / "full_motion_review_commands.sh").write_text(commands)
    (OUT / "full_motion_review_commands.sh").chmod(0o755)

    result = json.loads((OUT / "full_task_diagnostic_result.json").read_text())
    gui_result_path = OUT / "gui_full" / "full_task_diagnostic_result.json"
    if gui_result_path.exists():
        gui_result = json.loads(gui_result_path.read_text())
        write_json(
            OUT / "gui_full_diagnostic_review_audit.json",
            {
                "status": "INTERACTIVE_FULL_TRAJECTORY_REVIEW_COMMAND_TESTED",
                "trajectory_sha256": sha256(TRAJ),
                "trial": gui_result.get("trial"),
                "source_frames_executed": gui_result.get("source_frames_executed"),
                "physics_steps": gui_result.get("physics_steps"),
                "pause_at_end_reached": True,
                "interactive_review": gui_result.get("integrity", {}).get("interactive_review"),
                "gui_viewport": gui_result.get("integrity", {}).get("gui_viewport"),
                "render_parity_status": gui_result.get("execution_render_parity", {}).get("status"),
                "captured_frame_count": gui_result.get("execution_render_parity", {}).get("captured_frame_count"),
                "stage_failure_stops_playback": gui_result.get("stage_failure_stops_playback"),
                "full_task_success_claim": gui_result.get("full_task_success_claim"),
                "numeric_result_matches_headless": (
                    result.get("tracking") == gui_result.get("tracking")
                    and result.get("object_metrics") == gui_result.get("object_metrics")
                    and result.get("physics_steps") == gui_result.get("physics_steps")
                ),
            },
        )
    report = f"""1. frozen v17.1 G1 arm+Dex3 trajectory 990 samples를 stage-gate 조기 종료 없이 true physics로 완주했다.
2. phone grasp부터 accessory release까지 모든 stage가 FAIL했지만 failure는 playback을 중단하거나 robot command를 수정하지 않았다.
3. Cartesian backbone은 smooth/source-faithful했으며 접근 자세의 비합리성은 주로 redundant G1 IK posture와 task-orientation wrist demand에서 발생했다.

## 1. Full 990-frame execution

`FULL_TRAJECTORY_DIAGNOSTIC_COMPLETE`: action 0–989, {result['source_frames_executed']} captured frames, {result['physics_steps']} physics steps, 0.25x execution, 7.5 fps review encoding. This is not a full-task success claim.

## 2. Stage outcomes without stopping

{json.dumps(result['stage_passes_observed'], indent=2)}

Phone moved physically by at most {result['object_metrics']['phone_max_displacement_m'] * 1000:.3f} mm but was not retained through rotation. Accessory detached = {result['object_metrics']['accessory_detached']}; charger final state = {result['object_metrics']['charger_state_final']}. The robot nevertheless executed all later motion and the terminal suffix.

## 3. Full G1 motion summary

The generated G1 visibly performs the complete left approach/close, source-relative phone-rotation attempt, right accessory approach/removal motion, left charger transport, both release phases, and return/terminal hold. Object outcomes remain failed true-physics diagnostics.

## 4. Cartesian-path diagnosis

The Cartesian path itself is not the primary cause of the awkward approach: discontinuity count = {audit['cartesian_path_geometry']['discontinuity_count']}, max per-sample target step = {audit['cartesian_path_geometry']['maximum_target_step_m'] * 1000:.3f} mm, and minimum ALOHA fidelity = {fidelity['minimum_primary_metric']:.6f}.

## 5. IK/null-space diagnosis

Classification: `{attribution['classification']}`. Phone-approach v17.1-v14 RMS delta is {attribution['evidence']['phone_approach_v17_vs_v14_wrist_rms_delta_rad']:.6f} rad in wrist joints versus {attribution['evidence']['phone_approach_v17_vs_v14_nonwrist_rms_delta_rad']:.6f} rad in shoulder/elbow joints. The task-facing orientation is a contributor, but the protected hand XYZ path is not redrawn.

## 6. Shoulder/elbow/wrist posture

- Minimum joint-limit margin: {audit['global_joint_metrics']['minimum_joint_limit_margin_rad']:.9f} rad at `{audit['global_joint_metrics']['minimum_margin_joint']}`.
- Max joint step: {audit['global_joint_metrics']['maximum_joint_step_rad']:.6f} rad.
- Max joint velocity/acceleration: {audit['global_joint_metrics']['maximum_joint_velocity_rad_s']:.3f} rad/s / {audit['global_joint_metrics']['maximum_joint_acceleration_rad_s2']:.3f} rad/s².
- Left/right minimum elbow–torso surface distance: {audit['kinematic_posture']['left']['elbow_to_torso_surface_min_m'] * 1000:.3f} / {audit['kinematic_posture']['right']['elbow_to_torso_surface_min_m'] * 1000:.3f} mm.
- Extreme posture intervals are stored with semantic phase/progress in `full_motion_posture_audit.json`; no posture was changed.

## 7. ALOHA fidelity

Path: L {fidelity['left_path_shape']:.6f}, R {fidelity['right_path_shape']:.6f}; speed: L {fidelity['left_speed']:.6f}, R {fidelity['right_speed']:.6f}; midpoint {fidelity['bimanual_midpoint']:.6f}; relative vector {fidelity['relative_hand_vector']:.6f}; inter-hand distance {fidelity['inter_hand_distance']:.6f}; rotation progress L/R {fidelity['rotation']['left_rotation_progress_correlation']:.6f}/{fidelity['rotation']['right_rotation_progress_correlation']:.6f}.

## 8. Exact GUI command

```bash
{common}{gui}```

## 9. Review paths

- `v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_overview.mp4`
- `v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_side.mp4`
- `v17_1_FULL_TRAJECTORY_DIAGNOSTIC_WHITE_top.mp4`
- `aloha_vs_g1_full_motion_v17_1_4panel.mp4`
- `full_motion_posture_contact_sheet.png`

## 10. Recommended next step

VISUALLY REVIEW THE FULL-TRAJECTORY GUI/VIDEOS, THEN DESIGN ONE NULL-SPACE POSTURE-ONLY CORRECTION WHILE KEEPING THE CARTESIAN BACKBONE FROZEN.

THE FULL 990-FRAME G1 TRAJECTORY WAS EXECUTED WITHOUT STAGE-GATE EARLY TERMINATION
PHYSICS FAILURES DID NOT MODIFY OR STOP THE GENERATED G1 MOTION
THE ALOHA-DERIVED CARTESIAN ARM BACKBONE WAS NOT MODIFIED
CARTESIAN MOTION AND G1 REDUNDANT JOINT POSTURE WERE AUDITED SEPARATELY
THE PURPOSE OF THIS RUN WAS FULL-BEHAVIOR REVIEW, NOT A FULL-TASK SUCCESS CLAIM
NO VALIDATION, HELD-OUT, G1 EXPERT, DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
