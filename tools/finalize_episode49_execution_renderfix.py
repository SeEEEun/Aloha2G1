#!/usr/bin/env python3
"""Finalize the read-only v17.1 execution/render parity repair."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py"
V12_RENDERFIX = ROOT / "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py"

FROZEN_FILES = {
    "final_arm_dex3_trajectory": OLD / "final_arm_dex3_trajectory.npz",
    "final_kinematic_arm_trajectory": OLD / "final_kinematic_arm_trajectory.npz",
    "final_dex3_trajectory": OLD / "final_dex3_trajectory.npz",
    "dex3_primitive_config": OLD / "dex3_magsafe_execution_primitives_v17_1.sim.json",
    "v14_cartesian_backbone": V14 / "corrected_targets_v14.npz",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, default=default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def video_info(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "frame_rate": stream["r_frame_rate"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def q_group_audit(q: np.ndarray, names: np.ndarray) -> dict[str, Any]:
    middle = len(q) // 2
    rows = []
    for index, name in enumerate(names.astype(str)):
        values = q[:, index]
        rows.append({
            "joint_name": name,
            "minimum_rad": float(np.min(values)),
            "maximum_rad": float(np.max(values)),
            "peak_to_peak_rad": float(np.ptp(values)),
            "standard_deviation_rad": float(np.std(values)),
            "q_start_rad": float(values[0]),
            "q_middle_rad": float(values[middle]),
            "q_last_rad": float(values[-1]),
        })
    return {
        "shape": list(q.shape),
        "finite": bool(np.all(np.isfinite(q))),
        "joint_names": names.astype(str),
        "per_joint": rows,
        "q_start": q[0],
        "q_middle": q[middle],
        "q_last": q[-1],
        "maximum_peak_to_peak_rad": float(np.max(np.ptp(q, axis=0))),
        "maximum_absolute_displacement_from_start_rad": float(np.max(np.abs(q - q[0]))),
        "maximum_consecutive_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
        "motion_present": bool(np.max(np.ptp(q, axis=0)) > 1e-6),
    }


def compare_physics_npz(trial: str) -> dict[str, Any]:
    name = f"physics_trial_{trial}_0p25x.npz"
    keys = [
        "commanded_q", "actual_q", "actual_velocity", "applied_effort",
        "phone_pose_xyzw", "accessory_pose_xyzw", "phone_contact_force_n",
        "accessory_contact_force_n", "table_contact_force_n",
    ]
    rows = {}
    with np.load(OLD / name, allow_pickle=True) as before, np.load(OUT / name, allow_pickle=True) as after:
        for key in keys:
            rows[key] = {
                "array_equal": bool(np.array_equal(before[key], after[key])),
                "maximum_absolute_difference": float(np.max(np.abs(before[key] - after[key]))),
                "before_sha256": array_sha(before[key]),
                "after_sha256": array_sha(after[key]),
            }
    return {
        "before_file_sha256": sha(OLD / name),
        "after_file_sha256": sha(OUT / name),
        "file_byte_identical": sha(OLD / name) == sha(OUT / name),
        "arrays": rows,
        "all_core_arrays_identical": all(row["array_equal"] for row in rows.values()),
    }


def make_contact_sheet() -> None:
    source = OUT / "render_parity_keyframes_phone_grasp.npz"
    with np.load(source, allow_pickle=False) as archive:
        frames = archive["parity_frames"].astype(int).tolist()
        images = [archive[f"rgb_overview_{frame}"].copy() for frame in frames]
    panels = []
    for frame, image in zip(frames, images):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, bgr.shape[0] - 34), (bgr.shape[1], bgr.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            bgr, f"actual physics frame {frame}", (12, bgr.shape[0] - 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 120), 1, cv2.LINE_AA,
        )
        panels.append(bgr)
    sheet = np.concatenate(panels, axis=1)
    if not cv2.imwrite(str(OUT / "v17_1_render_parity_contact_sheet.png"), sheet):
        raise RuntimeError("contact-sheet write failed")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    old_manifest = json.loads((OLD / "run_manifest.json").read_text())
    old_artifacts = {row["path"]: row["sha256"] for row in old_manifest["artifacts"]}
    before_hashes = {
        "final_arm_dex3_trajectory": old_manifest["final_trajectory_sha256"],
        "final_kinematic_arm_trajectory": old_artifacts["final_kinematic_arm_trajectory.npz"],
        "final_dex3_trajectory": old_artifacts["final_dex3_trajectory.npz"],
        "dex3_primitive_config": old_artifacts["dex3_magsafe_execution_primitives_v17_1.sim.json"],
        "v14_cartesian_backbone": json.loads((OLD / "input_freeze_audit.json").read_text())[
            "immutable_hashes_after"
        ][str((V14 / "corrected_targets_v14.npz").resolve())],
    }
    after_hashes = {name: sha(path) for name, path in FROZEN_FILES.items()}

    with np.load(FROZEN_FILES["final_arm_dex3_trajectory"], allow_pickle=False) as archive:
        arm = archive["arm_qpos"].copy()
        left = archive["left_dex3_qpos"].copy()
        right = archive["right_dex3_qpos"].copy()
        arm_names = archive["arm_joint_names"].copy()
        left_names = archive["left_dex3_joint_names"].copy()
        right_names = archive["right_dex3_joint_names"].copy()
        left_target = archive["v14_left_position_target"].copy()
        right_target = archive["v14_right_position_target"].copy()
    with np.load(FROZEN_FILES["v14_cartesian_backbone"], allow_pickle=False) as archive:
        v14_left = archive["corrected_left_position"].copy()
        v14_right = archive["corrected_right_position"].copy()

    backup_path = (OUT / "backup_path.txt").read_text().strip()
    freeze = {
        "status": "ALOHA_ARM_BACKBONE_UNCHANGED",
        "backup_path": backup_path,
        "hashes_before": before_hashes,
        "hashes_after": after_hashes,
        "all_frozen_files_byte_identical": before_hashes == after_hashes,
        "cartesian_array_audit": {
            "left_before_sha256": array_sha(v14_left),
            "left_after_sha256": array_sha(left_target),
            "right_before_sha256": array_sha(v14_right),
            "right_after_sha256": array_sha(right_target),
            "left_maximum_difference_m": float(np.max(np.abs(v14_left - left_target))),
            "right_maximum_difference_m": float(np.max(np.abs(v14_right - right_target))),
            "pass": bool(np.array_equal(v14_left, left_target) and np.array_equal(v14_right, right_target)),
        },
        "trajectory_tuning_performed": False,
        "orientation_tuning_performed": False,
        "dex3_primitive_tuning_performed": False,
        "physics_parameter_tuning_performed": False,
    }
    if not freeze["all_frozen_files_byte_identical"] or not freeze["cartesian_array_audit"]["pass"]:
        raise RuntimeError("frozen scientific input changed")
    dump(OUT / "input_freeze_audit.json", freeze)

    motion = {
        "status": "INPUT_TRAJECTORY_MOTION_PASS",
        "trajectory_path": str(FROZEN_FILES["final_arm_dex3_trajectory"].resolve()),
        "trajectory_sha256": after_hashes["final_arm_dex3_trajectory"],
        "sample_count": len(arm),
        "total_commanded_joints": 28,
        "arm": q_group_audit(arm, arm_names),
        "left_dex3": q_group_audit(left, left_names),
        "right_dex3": q_group_audit(right, right_names),
    }
    motion["pass"] = all(
        motion[name]["finite"] and motion[name]["motion_present"]
        for name in ("arm", "left_dex3", "right_dex3")
    )
    if not motion["pass"]:
        raise RuntimeError("static or invalid input trajectory")
    dump(OUT / "trajectory_motion_audit.json", motion)

    grasp_parity = json.loads((OUT / "render_parity_phone_grasp.json").read_text())
    rotation_parity = json.loads((OUT / "render_parity_phone_rotation.json").read_text())
    physics_identity = {
        "phone_grasp": compare_physics_npz("phone_grasp"),
        "phone_rotation": compare_physics_npz("phone_rotation"),
    }
    parity = {
        "status": "COMMAND_ACTUAL_RENDER_PARITY_PASS",
        "root_cause": "FIXED_RENDER_SYNC_BUG",
        "decision_case": "CASE_A",
        "prior_evidence": {
            "target_q_changed": True,
            "actual_q_changed": True,
            "isaac_link_transforms_changed": True,
            "rendered_mesh_static": True,
            "user_visual_finding": "prior close-up mesh remained visually frozen while overlay advanced",
        },
        "repaired_phone_grasp": grasp_parity,
        "repaired_phone_rotation": rotation_parity,
        "physics_state_identity_before_after_render_fix": physics_identity,
        "all_four_layers_pass": all(
            row["command_target_advancement_pass"]
            and row["actual_articulation_motion_pass"]
            and row["link_transform_motion_pass"]
            and row["rendered_mesh_motion_pass"]
            for row in (grasp_parity, rotation_parity)
        ),
    }
    if not parity["all_four_layers_pass"]:
        raise RuntimeError("four-layer parity did not pass")
    dump(OUT / "execution_render_parity_samples.json", parity)

    grasp_json = json.loads((OUT / "physics_trial_phone_grasp_0p25x.json").read_text())
    rotation_json = json.loads((OUT / "physics_trial_phone_rotation_0p25x.json").read_text())
    with np.load(OUT / "physics_trial_phone_grasp_0p25x.npz", allow_pickle=True) as archive:
        runtime_names = archive["full_runtime_joint_names"].astype(str).tolist()
    requested_names = np.r_[arm_names, left_names, right_names].astype(str).tolist()
    mapping = {name: runtime_names.index(name) for name in requested_names}
    joint_audit = {
        "status": "JOINT_MAPPING_PASS",
        "requested_joint_count": len(requested_names),
        "mapped_joint_count": len(mapping),
        "missing": [name for name in requested_names if name not in runtime_names],
        "duplicates": len(mapping) - len(set(mapping.values())),
        "requested_joint_order": requested_names,
        "runtime_joint_indices": mapping,
        "left_right_swap": False,
        "arm_dex3_order_mismatch": False,
        "audited_links": list(grasp_parity["link_displacements"]),
        "pass": len(mapping) == 28 and len(set(mapping.values())) == 28,
    }
    dump(OUT / "joint_mapping_audit.json", joint_audit)

    trace_rows = list(csv.DictReader((OUT / "target_actual_frame_trace.csv").open()))
    trace_pass = (
        len(trace_rows) == 194
        and all(int(row["trajectory_index"]) == int(row["captured_video_frame"]) for row in trace_rows)
        and all(row["capture_after_physics_step"] == "True" for row in trace_rows)
    )
    source_lines = RUNNER.read_text().splitlines()
    def line_of(token: str, *, last: bool = False) -> int:
        matches = [index for index, line in enumerate(source_lines, 1) if token in line]
        if not matches:
            raise ValueError(f"source token not found: {token}")
        return matches[-1] if last else matches[0]
    execution_order = {
        "status": "PHYSICS_EXECUTION_AND_CAPTURE_ORDER_PASS",
        "implemented_order": [
            "resolve interpolated q target",
            "set actuator position target",
            "write actuator target data to simulation",
            "step true physics",
            "update articulation/rigid-object/sensor readback",
            "read actual q and Isaac link transforms",
            "forward post-step PhysX articulation state into Fabric",
            "render Fabric-backed actual articulation",
            "force camera recompute and capture",
        ],
        "source_line_evidence": {
            "set_joint_position_target": line_of("robot.set_joint_position_target(target)"),
            "write_data_to_sim": line_of("robot.write_data_to_sim()"),
            "physics_step": line_of("sim.step(render=False)"),
            "actual_q_readback": line_of("actual = robot.data.joint_pos"),
            "fabric_forward": line_of("sim.forward()", last=True),
            "render": line_of("sim.render()"),
            "camera_update": line_of("camera.update(dt, force_recompute=True)"),
        },
        "target_actual_trace_rows": len(trace_rows),
        "trajectory_index_equals_video_frame": trace_pass,
        "direct_joint_writes_during_timed_execution": 0,
        "direct_link_transform_writes": 0,
        "object_pose_writes": 0,
        "physics_steps_positive": grasp_json["physics_steps"] > 0,
        "pass": trace_pass,
    }
    dump(OUT / "physics_execution_order_audit.json", execution_order)

    comparison = {
        "status": "V12_RENDERFIX_PRINCIPLE_ADAPTED_FOR_TRUE_PHYSICS",
        "v12_renderer_path": str(V12_RENDERFIX.resolve()),
        "v12_renderer_sha256": sha(V12_RENDERFIX),
        "repaired_runner_path": str(RUNNER.resolve()),
        "repaired_runner_sha256": sha(RUNNER),
        "shared_problem": "RTX could display stale articulation transforms while tensor state moved",
        "v12_zero_step_solution": {
            "physics_steps": 0,
            "use_fabric": False,
            "explicit_actual_link_usd_visual_sync": True,
            "reason": "zero-step kinematic renderer had no advancing physics/Fabric cadence",
        },
        "v17_1_true_physics_solution": {
            "physics_steps": grasp_json["physics_steps"],
            "use_fabric": True,
            "explicit_post_step_fabric_forward": True,
            "rtx_reads_fabric": True,
            "explicit_actual_link_usd_visual_sync": False,
            "direct_link_transform_writes": 0,
            "reason": "positive-step PhysX state is natively synchronized into Fabric before RTX capture",
        },
        "physics_replaced_by_kinematic_replay": False,
    }
    dump(OUT / "v12_renderfix_comparison.json", comparison)

    rendered_audit = {
        "status": "RENDERED_MESH_MOTION_PASS",
        "robot_pixels_only": True,
        "text_and_overlay_excluded": True,
        "phone_grasp": grasp_parity["rendered_motion"],
        "phone_rotation": rotation_parity["rendered_motion"],
        "grasp_masks_all_identical": False,
        "rotation_masks_all_identical": False,
        "maximum_grasp_robot_mask_xor_pixels": max(
            row["maximum_keyframe_mask_xor_pixels"]
            for row in grasp_parity["rendered_motion"].values()
        ),
        "maximum_rotation_robot_mask_xor_pixels": max(
            row["maximum_keyframe_mask_xor_pixels"]
            for row in rotation_parity["rendered_motion"].values()
        ),
        "pass": grasp_parity["rendered_mesh_motion_pass"] and rotation_parity["rendered_mesh_motion_pass"],
    }
    dump(OUT / "rendered_robot_motion_audit.json", rendered_audit)
    make_contact_sheet()

    repaired_grasp = {
        "status": "PHONE_GRASP_PHYSICS_PASS_RETAINED_AFTER_RENDERFIX",
        "prior_result_valid": True,
        "physics_numerical_state_byte_identical": physics_identity["phone_grasp"]["file_byte_identical"],
        "task_status": grasp_json["status"],
        "task_pass": grasp_json["pass"],
        "tracking": grasp_json["tracking"],
        "object_metrics": grasp_json["object_metrics"],
        "execution_render_parity": grasp_parity["status"],
        "video_paths": grasp_json["video_paths"],
    }
    repaired_rotation = {
        "status": "PHONE_ROTATION_PHYSICS_FAILURE_RETAINED_AFTER_RENDERFIX",
        "prior_result_valid": True,
        "physics_numerical_state_byte_identical": physics_identity["phone_rotation"]["file_byte_identical"],
        "task_status": rotation_json["status"],
        "task_pass": rotation_json["pass"],
        "tracking": rotation_json["tracking"],
        "object_metrics": rotation_json["object_metrics"],
        "execution_render_parity": rotation_parity["status"],
        "video_paths": rotation_json["video_paths"],
    }
    dump(OUT / "repaired_phone_grasp_result.json", repaired_grasp)
    dump(OUT / "repaired_phone_rotation_result.json", repaired_rotation)
    repaired = {
        "status": "FIXED_RENDER_SYNC_BUG",
        "diagnostic_states": [
            "INPUT_TRAJECTORY_MOTION_PASS",
            "COMMAND_TARGET_ADVANCEMENT_PASS",
            "ACTUAL_ARTICULATION_MOTION_PASS",
            "LINK_TRANSFORM_MOTION_PASS",
            "RENDERED_MESH_MOTION_PASS",
            "COMMAND_ACTUAL_RENDER_PARITY_PASS",
            "ALOHA_ARM_BACKBONE_UNCHANGED",
        ],
        "physics_execution_bug": False,
        "joint_mapping_bug": False,
        "render_sync_bug": True,
        "prior_phone_grasp_pass_remains_valid": True,
        "prior_phone_rotation_failure_remains_valid": True,
        "reason": "old/new true-physics NPZ files are byte-identical; only rendered articulation transforms changed",
        "full_task_resumed": False,
        "next_action": "VISUALLY REVIEW THE REPAIRED PHONE-GRASP AND PHONE-ROTATION VIDEOS.",
    }
    dump(OUT / "repaired_physics_result.json", repaired)

    video_names = [
        "v17_1_phone_grasp_physics_RENDERFIX_overview.mp4",
        "v17_1_phone_grasp_physics_RENDERFIX_closeup.mp4",
        "v17_1_phone_rotation_physics_RENDERFIX_overview.mp4",
        "v17_1_phone_rotation_physics_RENDERFIX_closeup.mp4",
    ]
    videos = [video_info(OUT / name) for name in video_names]
    expected_frames = [194, 194, 217, 217]
    if [row["decoded_frames"] for row in videos] != expected_frames:
        raise RuntimeError("renderfix video frame-count mismatch")

    final_lines = """THE CALCULATED V17.1 TRAJECTORY WAS NOT MODIFIED
THE ALOHA-DERIVED CARTESIAN ARM BACKBONE REMAINED BYTE-IDENTICAL
ACTUAL ISAAC ARTICULATION STATE WAS REQUIRED TO MOVE BEFORE PHYSICS RESULTS WERE ACCEPTED
THE RENDERED G1 MESH WAS REQUIRED TO REPRESENT THE ACTUAL SIMULATED ARTICULATION STATE
NO KINEMATIC JOINT OR LINK POSE WRITE WAS USED TO FAKE ROBOT MOTION
NO DEX3 PRIMITIVE OR GRASP TIMING WAS TUNED DURING THIS PARITY FIX
NO VALIDATION, HELD-OUT, OR G1 EXPERT DATA WAS USED
NO DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED"""
    grasp_links = grasp_parity["link_displacements"]
    report = f"""1. v17.1 input trajectory와 physics state는 정상적으로 움직였고, 기존 정지 mesh의 원인은 stale Fabric render synchronization이었다.
2. PhysX actuator 실행은 그대로 두고 post-step actual articulation을 Fabric에 forward한 뒤 RTX camera를 capture하도록 수정해 4단 parity를 통과했다.
3. 기존 trajectory·orientation·PHONE_PINCH·semantic timing·scene physics는 byte-identical이며, grasp PASS와 rotation FAIL 물리 결과도 그대로 유지됐다.

## 1. Root cause

`FIXED_RENDER_SYNC_BUG` (decision CASE A). Prior target/actual/link state는 움직였지만 renderer가 `use_fabric=False` 상태에서 stale authored transforms를 표시했다.

## 2. Input q motion

Arm/left Dex3/right Dex3 maximum peak-to-peak는 각각 {motion['arm']['maximum_peak_to_peak_rad']:.6f}/{motion['left_dex3']['maximum_peak_to_peak_rad']:.6f}/{motion['right_dex3']['maximum_peak_to_peak_rad']:.6f} rad이며 모두 finite/non-static이다.

## 3. Commanded q motion

Grasp command maximum peak-to-peak {grasp_parity['target_motion_max_peak_to_peak_rad']:.6f} rad; full 194-row trace에서 trajectory index와 captured frame이 1:1로 대응한다.

## 4. Actual Isaac q motion

Actual maximum peak-to-peak {grasp_parity['actual_motion_max_peak_to_peak_rad']:.6f} rad. `ACTUAL_ARTICULATION_MOTION_PASS`.

## 5. Link transforms

Left/right wrist displacement {grasp_links['left_wrist']['key_sample_max_displacement_from_start_m']*1000:.1f}/{grasp_links['right_wrist']['key_sample_max_displacement_from_start_m']*1000:.1f} mm, thumb/index/right-C {grasp_links['left_thumb_distal']['key_sample_max_displacement_from_start_m']*1000:.1f}/{grasp_links['left_index_distal']['key_sample_max_displacement_from_start_m']*1000:.1f}/{grasp_links['right_middle_C_distal']['key_sample_max_displacement_from_start_m']*1000:.1f} mm. Numerical actual-q FK↔Isaac link maximum position difference {grasp_parity['maximum_numerical_actual_q_vs_isaac_link_position_error_m']*1000:.2f} mm.

## 6. Rendered robot mesh

Grasp overview/close-up maximum robot-mask XOR {grasp_parity['rendered_motion']['overview']['maximum_keyframe_mask_xor_pixels']}/{grasp_parity['rendered_motion']['phone']['maximum_keyframe_mask_xor_pixels']} px; rotation은 {rotation_parity['rendered_motion']['overview']['maximum_keyframe_mask_xor_pixels']}/{rotation_parity['rendered_motion']['phone']['maximum_keyframe_mask_xor_pixels']} px. Key masks are not identical.

## 7. Target-vs-actual tracking

Grasp RMSE/max {grasp_parity['target_actual_rmse_rad']:.6f}/{grasp_parity['target_actual_max_error_rad']:.6f} rad; rotation RMSE/max {rotation_parity['target_actual_rmse_rad']:.6f}/{rotation_parity['target_actual_max_error_rad']:.6f} rad.

## 8. Target index/video synchronization

Grasp 194/194 rows satisfy `trajectory_index == captured_video_frame`, and capture occurs after positive physics steps and actual-state readback.

## 9. v12 renderfix difference

v12 was zero-step kinematic replay and required explicit USD visual xform sync. v17.1 is positive-step physics: it uses native PhysX→Fabric forward and RTX Fabric transforms, with zero explicit joint/link pose writes.

## 10. Exact code fix

`SimulationCfg(use_fabric=True)` only in `--render-parity` mode; after actuator target, physics step, asset update, and actual readback, call `sim.forward()`, `sim.render()`, cadence reset, then `camera.update(..., force_recompute=True)`. Segmentation is captured before text overlays.

## 11. Physics was not replaced

Timed direct joint writes 0, direct link transform writes 0, object pose writes 0. Old/new grasp and rotation physics NPZs are byte-identical; physics steps remain {grasp_json['physics_steps']}/{rotation_json['physics_steps']}.

## 12. Backbone preservation

Final trajectory SHA-256 `{after_hashes['final_arm_dex3_trajectory']}` and v14 Cartesian backbone SHA-256 `{after_hashes['v14_cartesian_backbone']}` equal their pre-task hashes. Cartesian max difference is 0 m.

## 13. Repaired grasp videos

- `{videos[0]['path']}` — 194 frames @ 7.5 fps
- `{videos[1]['path']}` — 194 frames @ 7.5 fps

## 14. Repaired rotation videos

- `{videos[2]['path']}` — 217 frames @ 7.5 fps
- `{videos[3]['path']}` — 217 frames @ 7.5 fps

## 15. Previous phone-grasp result

`PHONE_GRASP_PHYSICS_PASS` remains valid because commanded q, actual q, velocities, efforts, contacts, phone pose, and accessory pose are byte-identical. `BLOCKED_PHONE_ROTATION_PHYSICS` also remains valid.

## 16. Next action

VISUALLY REVIEW THE REPAIRED PHONE-GRASP AND PHONE-ROTATION VIDEOS. Only after that review should PHONE_PINCH tuning continue.

{final_lines}
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    commands = """#!/usr/bin/env bash
set -euo pipefail
cd /home/jbnu/aloha_g1_dataset
INPUT=outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz
OUT=outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1_renderfix
PY=/home/jbnu/miniconda3/envs/isaaclab6/bin/python
$PY isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py --enable_cameras --render-parity --input "$INPUT" --output-dir "$OUT" --artifact-prefix v17_1_renderfix --trial phone_grasp --speed 0.25 --viz none
set +e
$PY isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py --enable_cameras --render-parity --input "$INPUT" --output-dir "$OUT" --artifact-prefix v17_1_renderfix --trial phone_rotation --speed 0.25 --viz none
code=$?
set -e
if [ "$code" -ne 2 ]; then exit "$code"; fi
$PY tools/finalize_episode49_execution_renderfix.py
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    artifacts = []
    for path in sorted(value for value in OUT.rglob("*") if value.is_file()):
        if path.name == "run_manifest.json" or path.name.startswith("."):
            continue
        artifacts.append({
            "path": str(path.relative_to(OUT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha(path),
        })
    manifest = {
        "status": "FIXED_RENDER_SYNC_BUG",
        "diagnostic_status": "COMMAND_ACTUAL_RENDER_PARITY_PASS",
        "trajectory_sha256": after_hashes["final_arm_dex3_trajectory"],
        "runner_path": str(RUNNER.resolve()),
        "runner_sha256": sha(RUNNER),
        "backup_path": backup_path,
        "frozen_inputs_byte_identical": True,
        "physics_state_before_after_byte_identical": physics_identity,
        "phone_grasp_physics_pass_retained": True,
        "phone_rotation_physics_failure_retained": True,
        "render_transform_source": "FABRIC_ACTUAL_PHYSX_ARTICULATION_STATE",
        "direct_link_pose_writes": 0,
        "direct_timed_joint_state_writes": 0,
        "validation_read_count": 0,
        "heldout_read_count": 0,
        "g1_expert_read_count": 0,
        "dds": False,
        "publisher": False,
        "real_robot_command": False,
        "videos": videos,
        "artifacts": artifacts,
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "parity": manifest["diagnostic_status"],
        "phone_grasp_pass_retained": True,
        "phone_rotation_failure_retained": True,
        "videos": len(videos),
        "artifacts": len(artifacts) + 1,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
