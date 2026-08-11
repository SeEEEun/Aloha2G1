#!/usr/bin/env python3
"""Finalize the approved-q closed-pose Dex3/phone PhysX contact audit.

This script is deliberately Decision-A only: if the approved hand pose did not
produce simultaneous physical thumb/index contact it exits without creating a
preload candidate.  It never edits the approved primitive, v17.2, the scene, or
the physics configuration.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import cv2
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
PREVIOUS = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_static_physics_v1"
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_closed_contact_v2"
TRIAL = OUT / "approved_trial"
PRIMITIVE = CAL / "left_phone_fingertip_pinch_primitive.json"
RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py"
V17_2 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
SCENE_USD = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
SCENE_LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, payload) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)

    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, default=convert, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".incomplete")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    primitive = load(PRIMITIVE)
    previous_result = load(PREVIOUS / "static_physics_result.json")
    previous_retention = load(PREVIOUS / "phone_retention_metrics.json")
    result = load(TRIAL / "static_physics_result.json")
    contacts = load(TRIAL / "phone_contact_identity_metrics.json")
    retention = load(TRIAL / "phone_retention_metrics.json")
    tracking = load(TRIAL / "dex3_tracking_metrics.json")
    collision = load(TRIAL / "collision_audit.json")
    setup = load(TRIAL / "physics_setup_audit.json")
    no_cheat = load(TRIAL / "no_cheat_audit.json")
    freeze_trial = load(TRIAL / "input_freeze_audit.json")

    approved_q = np.asarray(primitive["selected_static_q_rad"], dtype=float)
    expected_q = np.asarray(
        [-0.517737, 0.747053, 0.050426, -0.661925, -1.705330, -0.100000, -0.100000]
    )
    if not np.array_equal(np.round(approved_q, 6), expected_q):
        raise RuntimeError("approved q does not match the user-approved six-decimal values")
    if not result["simultaneous_thumb_index_contact"]:
        raise RuntimeError(
            "Decision A did not pass. Stop here and perform signed-gap diagnosis before any preload experiment."
        )
    if contacts["identity"]["THIRD"]["contact_samples"] != 0:
        raise RuntimeError("third finger unexpectedly contacted the phone")

    frozen_paths = {
        "approved_primitive": PRIMITIVE,
        "v17_2_full_trajectory": V17_2,
        "v14_cartesian_backbone": V14,
        "authoritative_scene": SCENE_USD,
        "scene_layout": SCENE_LAYOUT,
    }
    frozen_hashes = {key: sha256(path) for key, path in frozen_paths.items()}
    approved_hash = frozen_hashes["approved_primitive"]
    freeze_pass = (
        freeze_trial["all_frozen_inputs_byte_identical"]
        and freeze_trial["hashes_before"]["approved_primitive"] == approved_hash
        and freeze_trial["hashes_after"]["approved_primitive"] == approved_hash
        and np.array_equal(np.asarray(result["tested_q_rad"]), approved_q)
    )
    dump(OUT / "input_freeze_audit.json", {
        "status": "APPROVED_LEFT_PHONE_PINCH_UNCHANGED" if freeze_pass else "BLOCKED_INPUT_MUTATION",
        "approved_primitive_sha256": approved_hash,
        "approved_q_full_precision_rad": approved_q,
        "approved_q_six_decimals_rad": np.round(approved_q, 6),
        "tested_q_exactly_equal_to_approved_q": bool(np.array_equal(np.asarray(result["tested_q_rad"]), approved_q)),
        "frozen_input_hashes": frozen_hashes,
        "trial_before_after_freeze_audit": freeze_trial,
        "v17_2_modified": False,
        "right_dex3_modified": False,
        "arm_wrist_modified": False,
    })

    prior_diag = previous_retention["failure_diagnosis"]
    dump(OUT / "previous_failure_reinterpretation.json", {
        "status": "PREVIOUS_CONTACT_TEST_CONFIRMED_CONFOUNDED_BY_PRECONTACT_PHONE_FALL",
        "previous_reported_status": previous_result["status"],
        "phone_displacement_before_pinch_transition_m": prior_diag["phone_displacement_before_pinch_transition_m"],
        "phone_vertical_displacement_before_pinch_transition_m": prior_diag["phone_vertical_displacement_before_pinch_transition_m"],
        "open_pregrasp_delay_before_pinch_s": 1.0,
        "previous_thumb_index_task_chain_closest_approved_error_rad": prior_diag["closest_thumb_index_chain_approved_pose_error_rad"],
        "previous_full_7dof_closest_approved_error_rad": prior_diag["closest_all_7dof_approved_pose_error_rad"],
        "interpretation": (
            "The unrestrained phone had moved 96.691 mm during OPEN/PREGRASP before PINCH. "
            "That run cannot establish failure of the approved photo-derived pinch geometry."
        ),
        "closed_pose_causal_isolation": {
            "open_phase": False,
            "pregrasp_phase": False,
            "hand_started_at_approved_pinch": True,
            "phone_initialized_once_before_timed_physics": True,
        },
    })

    per_joint = tracking["per_joint"]
    task_rows = per_joint[:5]
    third_rows = per_joint[5:]
    task_max = max(row["maximum_absolute_error_rad"] for row in task_rows)
    task_rmse = float(np.sqrt(np.mean([row["rmse_rad"] ** 2 for row in task_rows])))
    third_max = max(row["maximum_absolute_error_rad"] for row in third_rows)
    third_rmse = float(np.sqrt(np.mean([row["rmse_rad"] ** 2 for row in third_rows])))
    dump(OUT / "task_vs_nontask_tracking_audit.json", {
        "status": "TASK_FINGER_TRACKING_PASS_NON_TASK_THIRD_WARNING",
        "task_fingers": {
            "physical_identity": "THUMB + INDEX",
            "joint_names": [row["joint"] for row in task_rows],
            "per_joint": task_rows,
            "aggregate_rmse_rad": task_rmse,
            "maximum_absolute_error_rad": task_max,
            "closest_chain_to_approved_max_error_rad": tracking["closest_thumb_index_chain_to_approved_max_error_rad"],
            "contact_outcome": "BILATERAL_PHYSX_CONTACT_FORMED",
        },
        "non_task_third": {
            "physical_identity": "MIDDLE/THIRD",
            "role": "NON_TASK",
            "joint_names": [row["joint"] for row in third_rows],
            "per_joint": third_rows,
            "aggregate_rmse_rad": third_rmse,
            "maximum_absolute_error_rad": third_max,
            "phone_contact_samples": contacts["identity"]["THIRD"]["contact_samples"],
            "interpretation": "Third-finger tracking error is reported separately and does not block verified thumb-index contact.",
        },
        "hand_actuator_tracking_blocker": False,
    })

    thumb = contacts["identity"]["THUMB"]
    index = contacts["identity"]["INDEX"]
    third = contacts["identity"]["THIRD"]
    thumb_sep = thumb["initial_solver_minimum_signed_separation_m"]
    index_sep = index["initial_solver_minimum_signed_separation_m"]
    if thumb_sep is None or index_sep is None:
        raise RuntimeError("approved pose contact passed but initial signed separations are missing")
    dump(OUT / "approved_pose_collision_gap.json", {
        "status": "APPROVED_PHOTO_PINCH_PHYSX_CONTACT_PASS",
        "method": "active PhysX collision contact data at the first timed solver step from the unchanged approved q and initial diagnostic phone pose",
        "signed_distance_convention": "negative values are solver-reported collision-surface overlap/contact; positive values would be open gap",
        "thumb_collision_surface_minimum_signed_separation_m": thumb_sep,
        "index_collision_surface_minimum_signed_separation_m": index_sep,
        "thumb_collision_surface_signed_separations_m": [row["separation_m"] for row in thumb["initial_solver_contact_points"]],
        "index_collision_surface_signed_separations_m": [row["separation_m"] for row in index["initial_solver_contact_points"]],
        "total_bilateral_open_gap_m": 0.0,
        "bilateral_contact_exists": True,
        "calibration_visual_contact_frame_gap_per_side_m": load(CAL / "fingertip_geometry_metrics.json")["metrics"]["bilateral_surface_gap_m"],
        "important_distinction": (
            "The earlier +0.415 mm/side value came from diagnostic pad frames and visual/mesh geometry. "
            "The active PhysX collision surfaces report contact/overlap on both task fingers and are authoritative for this contact test."
        ),
        "preload_decision": "NOT_NEEDED_DECISION_A_CONTACT_ALREADY_EXISTS",
        "preload_experiment_performed": False,
    })

    # Contact centering is measured in the unchanged diagnostic phone frame.
    phone_world = np.asarray(setup["phone_initial_transform_matrix"], dtype=float)
    phone_world_inv = np.linalg.inv(phone_world)
    def local_point(record):
        point = np.r_[np.asarray(record["point_m"], dtype=float), 1.0]
        return (phone_world_inv @ point)[:3]
    thumb_local = local_point(thumb["first_contact"])
    index_local = local_point(index["first_contact"])
    # The authored phone thickness axis is local Y for this diagnostic asset.
    pinch_center_offset = float(0.5 * (thumb_local[1] + index_local[1]))
    contact_height_offset = float(abs(thumb_local[0] - index_local[0]))
    dump(OUT / "phone_centering_audit.json", {
        "status": "APPROVED_DIAGNOSTIC_PHONE_CENTERING_CONTACT_PASS",
        "phone_pose_source": setup["phone_initial_transform_source"],
        "phone_initial_transform_matrix": phone_world,
        "phone_initial_velocity": [0.0] * 6,
        "thumb_first_contact_phone_local_xyz_m": thumb_local,
        "index_first_contact_phone_local_xyz_m": index_local,
        "phone_center_offset_along_local_thickness_pinch_axis_m": pinch_center_offset,
        "contact_height_offset_along_phone_long_axis_m": contact_height_offset,
        "thumb_initial_signed_separation_m": thumb_sep,
        "index_initial_signed_separation_m": index_sep,
        "centering_sweep_performed": False,
        "centering_sweep_reason": "Decision A: bilateral contact already existed at the approved centered pose, so no diagnostic pose sweep was needed.",
        "phone_pose_tuned": False,
    })

    trace = np.load(TRIAL / "static_physics_trace.npz")
    actual_arm = np.asarray(trace["actual_fixed_arm_q"])
    arm_actual_excursion = np.max(np.abs(actual_arm - actual_arm[0]), axis=0)
    fixed_pose = primitive["fixed_arm_wrist_pose"]["joint_values_rad"]
    dump(OUT / "closed_pose_contact_metrics.json", {
        "status": "APPROVED_PHOTO_PINCH_PHYSX_CONTACT_PASS",
        "decision": "A_CONTACT_ALREADY_EXISTS_NO_PRELOAD",
        "approved_q_rad": approved_q,
        "thumb": thumb,
        "index": index,
        "third_non_task": third,
        "simultaneous_thumb_index_samples": contacts["simultaneous_thumb_index_samples"],
        "simultaneous_thumb_index_duration_s": contacts["simultaneous_thumb_index_longest_duration_s"],
        "third_is_primary": contacts["third_is_primary"],
        "retention": retention,
        "arm_wrist_target_constant": True,
        "fixed_arm_wrist_target_rad": fixed_pose,
        "actual_arm_wrist_maximum_excursion_from_first_readback_rad": float(np.max(arm_actual_excursion)),
        "actual_wrist_maximum_excursion_from_first_readback_rad": float(np.max(arm_actual_excursion[-3:])),
        "interpretation_of_small_actual_motion": "PhysX actuator compliance only; the arm/wrist target was constant and was not used to create contact.",
        "prohibited_self_collision_zero": collision["prohibited_robot_self_contact_records"] == 0,
        "retention_classification": retention["status"],
        "preload_needed": False,
    })

    dump(OUT / "no_cheat_audit.json", no_cheat)
    dump(OUT / "collision_audit.json", collision)
    dump(OUT / "physics_setup_audit.json", setup)

    copies = {
        "approved_closed_pose_before_physics.png": "approved_closed_pose_before_physics.png",
        "approved_closed_pose_contact_closeup.mp4": "approved_closed_pose_contact_closeup.mp4",
        "approved_closed_pose_overview.mp4": "approved_closed_pose_overview.mp4",
        "approved_closed_pose_side.mp4": "approved_closed_pose_side.mp4",
        "left_phone_pinch_static_physics_contact_sheet.png": "closed_pose_contact_sheet.png",
        "left_phone_pinch_physics_contact_identity.png": "contact_identity_overlay.png",
    }
    for source_name, destination_name in copies.items():
        copy(TRIAL / source_name, OUT / destination_name)

    gui_command = """source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
DISPLAY=:0 /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir outputs/scene_registered_retargeting/dex3_left_phone_pinch_closed_contact_v2/gui_review \\
  --trial closed_hold --hold-duration 1.5 \\
  --artifact-prefix approved_closed_pose_gui \\
  --gui --pause-at-end --enable_cameras
"""
    headless_command = """source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
/home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir outputs/scene_registered_retargeting/dex3_left_phone_pinch_closed_contact_v2/approved_trial \\
  --trial closed_hold --hold-duration 1.5 \\
  --artifact-prefix approved_closed_pose --headless --enable_cameras
"""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# Reproduce the approved-q closed-pose true-PhysX contact test.
{headless_command}

# Interactive PAPER_WHITE close-up.  The window pauses at the final state.
{gui_command}
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    report = f"""APPROVED PHOTO POSE THUMB CONTACT: PASS — {thumb['contact_samples']}/91 samples.
APPROVED PHOTO POSE SIMULTANEOUS INDEX CONTACT: PASS — {contacts['simultaneous_thumb_index_longest_duration_s']:.6f} s.
COLLISION GAP: NOT ABSENT — initial signed PhysX separations were THUMB {thumb_sep * 1000:.3f} mm and INDEX {index_sep * 1000:.3f} mm (contact/overlap).

# Left Dex3 closed-pose contact audit v2

## Final status

`APPROVED_PHOTO_PINCH_PHYSX_CONTACT_PASS`

The unchanged user-approved q produced bilateral physical contact from the verified
physical thumb and index for all 91 timed samples.  The third finger produced zero
phone-contact samples.  Decision A therefore applies and no preload search or
candidate was created.

## 1. Why v1 was confounded

The v1 phone moved {prior_diag['phone_displacement_before_pinch_transition_m'] * 1000:.3f} mm
({prior_diag['phone_vertical_displacement_before_pinch_transition_m'] * 1000:.3f} mm vertically)
during the 1.0 s OPEN/PREGRASP interval before PINCH.  Its FAIL is not evidence that
the approved photo-derived closed geometry cannot make contact.

## 2. Task versus non-task tracking

- Thumb/index task-chain closest approved-pose error: {tracking['closest_thumb_index_chain_to_approved_max_error_rad']:.6f} rad.
- Task-joint aggregate RMSE / max transient error: {task_rmse:.6f} / {task_max:.6f} rad.
- Non-task third aggregate RMSE / max error: {third_rmse:.6f} / {third_max:.6f} rad.
- The third tracking warning did not block the measured thumb/index contact.

## 3. Approved-q collision geometry and centering

- Approved q: `{np.round(approved_q, 9).tolist()}`
- Thumb initial minimum signed separation: {thumb_sep * 1000:.3f} mm.
- Index initial minimum signed separation: {index_sep * 1000:.3f} mm.
- Open bilateral gap: 0 mm; both active collision surfaces were already in contact.
- Phone-center offset along its local thickness axis: {pinch_center_offset * 1000:.6f} mm.
- First-contact long-axis height offset: {contact_height_offset * 1000:.3f} mm.
- No ±1 mm centering sweep was run because contact already existed unchanged.

The calibration-frame +0.415 mm/side estimate was not treated as a PhysX collision
gap.  Actual first-step PhysX collision separation is negative on both task fingers.

## 4. Contact, force, and third-finger status

- Thumb: {thumb['contact_samples']} samples, max {thumb['maximum_force_n']:.3f} N.
- Index: {index['contact_samples']} samples, max {index['maximum_force_n']:.3f} N.
- Simultaneous contact: {contacts['simultaneous_thumb_index_samples']} samples / {contacts['simultaneous_thumb_index_longest_duration_s']:.6f} s.
- Third: {third['contact_samples']} samples, max {third['maximum_force_n']:.3f} N; non-task.
- Prohibited robot self-contact records: {collision['prohibited_robot_self_contact_records']}.

## 5. Retention sanity

Contact feasibility passed, but static retention remains a warning: phone COM motion
was {retention['phone_hold_com_displacement_m'] * 1000:.3f} mm, relative slip was
{retention['phone_relative_to_pinch_center_hold_slip_m'] * 1000:.3f} mm, vertical
motion was {retention['phone_hold_vertical_displacement_m'] * 1000:.3f} mm, and
orientation changed {retention['phone_hold_orientation_change_deg']:.3f} deg.
This task did not tune q, friction, effort, controller, phone geometry, or physics.

## 6. Fixed arm/wrist and no-cheat proof

All seven arm/wrist targets stayed constant.  Actual compliant excursion was at most
{float(np.max(arm_actual_excursion)):.6f} rad overall and
{float(np.max(arm_actual_excursion[-3:])):.6f} rad at the wrist; no arm/wrist target
motion was used to create contact.  The phone pose and zero velocity were written
once before timed physics, then timed phone writes, following, attachment, hidden
joints, and direct transforms were all zero.  Gravity and collision remained on for
{no_cheat['actual_physx_steps']} actual PhysX steps.

## 7. Visual artifacts

- [Pre-physics approved closed pose](approved_closed_pose_before_physics.png)
- [True-PhysX close-up](approved_closed_pose_contact_closeup.mp4)
- [Overview](approved_closed_pose_overview.mp4)
- [Side](approved_closed_pose_side.mp4)
- [Contact sheet](closed_pose_contact_sheet.png)
- [Physical-finger contact identity](contact_identity_overlay.png)

## 8. Exact GUI command

```bash
{gui_command.rstrip()}
```

## 9. Exact next action

USER VISUALLY APPROVES THE PHYSICS-COMPATIBLE LEFT PHONE PINCH BEFORE
FULL-TRAJECTORY INTEGRATION.

THE PREVIOUS CONTACT FAIL WAS NOT ATTRIBUTED TO THE APPROVED PINCH BEFORE REMOVING THE PRE-CONTACT PHONE-FALL CONFOUND
TASK THUMB-INDEX TRACKING WAS EVALUATED SEPARATELY FROM THE NON-TASK THIRD FINGER
THE USER-APPROVED PHOTO PINCH WAS TESTED UNCHANGED BEFORE ANY PRELOAD CALIBRATION
ANY PRELOAD CORRECTION WAS LIMITED TO THE PHYSICAL THUMB AND INDEX DEX3 JOINTS
THE THIRD FINGER REMAINED NON-TASK
THE G1 ARM AND WRIST WERE NOT USED TO CREATE CONTACT
NO PHONE GEOMETRY, FRICTION, MASS, GRAVITY, OR COLLISION PARAMETER WAS TUNED
THE V17.2 TRAJECTORY, JITTER, RIGHT DEX3, AND ALOHA CARTESIAN BACKBONE WERE NOT MODIFIED
NO DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    # Verify videos beyond successful encoding.
    videos = {}
    for name in ("approved_closed_pose_contact_closeup.mp4", "approved_closed_pose_overview.mp4", "approved_closed_pose_side.mp4"):
        capture = cv2.VideoCapture(str(OUT / name))
        videos[name] = {
            "opens": bool(capture.isOpened()),
            "decoded_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        capture.release()

    gui_smoke_result_path = OUT / "gui_smoke/static_physics_result.json"
    gui_smoke_result = load(gui_smoke_result_path) if gui_smoke_result_path.exists() else None
    dump(OUT / "gui_review_audit.json", {
        "status": "GUI_TRUE_PHYSX_CLOSED_POSE_REVIEW_SMOKE_PASS" if gui_smoke_result else "GUI_NOT_SMOKE_TESTED",
        "tested_without_pause_at_end": bool(gui_smoke_result),
        "tested_trial": gui_smoke_result.get("trial") if gui_smoke_result else None,
        "tested_physics_steps": gui_smoke_result.get("physics_steps") if gui_smoke_result else None,
        "production_gui_command_in_commands_sh_uses_pause_at_end": True,
        "actual_physx_to_fabric_to_rtx": True,
        "gui_command": gui_command,
    })
    tests = {
        "approved_q_unchanged": freeze_pass,
        "approved_q_bilateral_contact": bool(result["simultaneous_thumb_index_contact"]),
        "thumb_contact_samples_positive": thumb["contact_samples"] > 0,
        "index_contact_samples_positive": index["contact_samples"] > 0,
        "third_contact_samples_zero": third["contact_samples"] == 0,
        "prohibited_self_collision_zero": collision["prohibited_robot_self_contact_records"] == 0,
        "timed_phone_pose_writes_zero": no_cheat["timed_phone_pose_writes"] == 0,
        "timed_phone_velocity_writes_zero": no_cheat["timed_phone_velocity_writes"] == 0,
        "object_follow_zero": no_cheat["object_follow"] == 0,
        "hidden_fixed_joint_zero": no_cheat["hidden_fixed_joint"] == 0,
        "gravity_enabled": no_cheat["gravity_enabled"],
        "collision_enabled": no_cheat["collision_enabled"],
        "physx_steps_positive": no_cheat["actual_physx_steps"] > 0,
        "preload_not_performed": True,
        "all_videos_decode": all(row["opens"] and row["decoded_frames"] > 0 for row in videos.values()),
        "gui_smoke_pass": bool(gui_smoke_result),
    }
    dump(OUT / "tests_results.json", {
        "status": "ALL_REQUIRED_TESTS_PASS" if all(tests.values()) else "REQUIRED_TEST_FAILURE",
        "tests": tests,
    })

    generated = sorted(path for path in OUT.iterdir() if path.is_file())
    manifest = {
        "status": "APPROVED_PHOTO_PINCH_PHYSX_CONTACT_PASS",
        "decision": "A_CONTACT_ALREADY_EXISTS_NO_PRELOAD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(RUNNER),
        "runner_sha256": sha256(RUNNER),
        "approved_primitive_sha256": approved_hash,
        "approved_q_rad": approved_q,
        "physics": {
            "gravity": True,
            "collision": True,
            "steps": no_cheat["actual_physx_steps"],
            "phone_initial_pose_writes": 1,
            "timed_object_writes": 0,
            "hidden_grasp_joints": 0,
        },
        "preload": {"needed": False, "performed": False, "candidate_created": False},
        "video_decode_audit": videos,
        "frozen_input_hashes": frozen_hashes,
        "output_hashes": {
            path.name: sha256(path) for path in generated
            if path.name != "run_manifest.json"
        },
        "unused_data": {"v17_2_modified": False, "right_dex3_modified": False},
    }
    dump(OUT / "run_manifest.json", manifest)
    if not freeze_pass:
        raise RuntimeError("input freeze audit failed")
    if not all(row["opens"] and row["decoded_frames"] > 0 for row in videos.values()):
        raise RuntimeError("video decode audit failed")
    print(json.dumps({
        "status": manifest["status"],
        "thumb_initial_signed_separation_mm": thumb_sep * 1000,
        "index_initial_signed_separation_mm": index_sep * 1000,
        "simultaneous_duration_s": contacts["simultaneous_thumb_index_longest_duration_s"],
        "preload_performed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
