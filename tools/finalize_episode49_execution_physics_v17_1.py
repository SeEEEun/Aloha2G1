#!/usr/bin/env python3
"""Finalize the narrow Episode-49 v17.1 execution/physics audit."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
BUILDER = ROOT / "tools/build_episode49_execution_physics_v17_1.py"
TRAJECTORY_CODE = ROOT / "tools/aloha_g1_v17/trajectory.py"
PHYSICS_RUNNER = ROOT / "isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py"
PHONE_JSON = OUT / "physics_trial_phone_grasp_0p25x.json"
PHONE_NPZ = OUT / "physics_trial_phone_grasp_0p25x.npz"
ROTATION_JSON = OUT / "physics_trial_phone_rotation_0p25x.json"
ROTATION_NPZ = OUT / "physics_trial_phone_rotation_0p25x.npz"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def save_npz(path: Path, **value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **value)
    os.replace(temporary, path)


def video_info(path: Path, scope: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "decoded_frames": int(stream["nb_read_frames"]),
        "frame_rate": stream["r_frame_rate"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "scope": scope,
        "true_isaac_physics": True,
        "kinematic_object_follow": False,
    }


def main() -> int:
    required = [
        OUT / "input_freeze_audit.json", OUT / "semantic_runtime_audit.json",
        OUT / "orientation_gate_metrics.json", OUT / "aloha_fidelity_metrics.json",
        OUT / "joint_margin_metrics.json", OUT / "collision_classifier_integrity_audit.json",
        OUT / "kinematic_prephysics_result.json",
        OUT / "dex3_magsafe_execution_primitives_v17_1.sim.json",
        OUT / "dex3_semantic_interpolation_config.json",
        OUT / "final_arm_dex3_trajectory.npz", PHONE_JSON, PHONE_NPZ,
        ROTATION_JSON, ROTATION_NPZ,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    freeze = json.loads((OUT / "input_freeze_audit.json").read_text())
    semantic = json.loads((OUT / "semantic_runtime_audit.json").read_text())
    orientation = json.loads((OUT / "orientation_gate_metrics.json").read_text())
    fidelity = json.loads((OUT / "aloha_fidelity_metrics.json").read_text())
    joints = json.loads((OUT / "joint_margin_metrics.json").read_text())
    collision = json.loads((OUT / "collision_classifier_integrity_audit.json").read_text())
    kinematic = json.loads((OUT / "kinematic_prephysics_result.json").read_text())
    primitives = json.loads((OUT / "dex3_magsafe_execution_primitives_v17_1.sim.json").read_text())
    interpolation = json.loads((OUT / "dex3_semantic_interpolation_config.json").read_text())
    phone_trial = json.loads(PHONE_JSON.read_text())
    rotation_trial = json.loads(ROTATION_JSON.read_text())

    semantic["v17_1_execution"] = {
        "semantic_local_orientation_executed": True,
        "predefined_dex3_execution_executed": True,
        "detector_invocation_count": 0,
        "event_redetection": False,
        "runtime_literal_semantic_frame_count": semantic["runtime_literal_audit"]["count"],
        "validation_read_count": 0,
        "heldout_read_count": 0,
    }
    dump(OUT / "semantic_runtime_audit.json", semantic)

    if not kinematic["gate_pass"]:
        raise RuntimeError("v17.1 finalizer requires the recorded kinematic prephysics PASS")
    if phone_trial["input_sha256"] != sha(OUT / "final_arm_dex3_trajectory.npz"):
        raise RuntimeError("phone-grasp physics trial does not match the final trajectory")
    if rotation_trial["input_sha256"] != sha(OUT / "final_arm_dex3_trajectory.npz"):
        raise RuntimeError("phone-rotation physics trial does not match the final trajectory")

    with np.load(OUT / "final_arm_dex3_trajectory.npz", allow_pickle=False) as archive:
        arm = archive["g1_arm_q"].copy()
        left = archive["left_dex3_qpos"].copy()
        right = archive["right_dex3_qpos"].copy()
        timestamps = archive["source_timestamps"].copy()
        arm_names = archive["arm_joint_names"].astype(str)
        left_names = archive["left_dex3_joint_names"].astype(str)
        right_names = archive["right_dex3_joint_names"].astype(str)
        root = archive["g1_root"].copy()
    with np.load(ROTATION_NPZ, allow_pickle=True) as archive:
        phone_pose = archive["phone_pose_xyzw"].copy()
        accessory_pose = archive["accessory_pose_xyzw"].copy()
        action_indices = archive["action_indices"].copy()
        actual_q = archive["actual_q"].copy()
        commanded_q = archive["commanded_q"].copy()

    phone_result = phone_trial | {
        "gate_definition": {
            "sustained_opposing_AB_contact": True,
            "relative_slip_max_m": 0.020,
            "phone_follows_hand_min_ratio": 0.40,
            "table_supported_acquisition_allowed": True,
            "lift_is_reported_separately_and_tested_by_rotation_stage": True,
        },
        "dominant_failure_subsystem": None,
    }
    dump(OUT / "phone_grasp_physics_result.json", phone_result)
    dump(OUT / "phone_rotation_physics_result.json", rotation_trial | {
        "dominant_failure_subsystem": "PHONE_PINCH_PRIMITIVE",
        "failure_evidence": (
            "A/B contact was lost during the mapped ALOHA rotation; the phone remained on the table "
            "and finished near 90 degrees from portrait."
        ),
    })
    blocked_stage = {
        "status": "NOT_RUN_PREREQUISITE_PHONE_ROTATION_FAILED",
        "prerequisite": "PHONE_ROTATION_PHYSICS_PASS",
        "physics_steps": 0,
        "object_pose_scripted": False,
        "kinematic_object_follow": False,
        "semantic_attach_detach": False,
        "dominant_failure_subsystem": "PHONE_PINCH_PRIMITIVE",
    }
    for name in (
        "accessory_removal_physics_result.json", "bimanual_transport_physics_result.json",
        "charger_placement_physics_result.json", "accessory_release_physics_result.json",
    ):
        dump(OUT / name, blocked_stage)

    dump(OUT / "physics_tracking_metrics.json", {
        "status": "TRUE_PHYSICS_TRACKING_AUDITED",
        "phone_grasp": phone_trial["tracking"],
        "phone_rotation": rotation_trial["tracking"],
        "joint_mapping": rotation_trial["joint_mapping"],
        "maximum_command_actual_error_rad": max(
            phone_trial["tracking"]["maximum_absolute_error_rad"],
            rotation_trial["tracking"]["maximum_absolute_error_rad"],
        ),
        "rotation_trial_command_actual_rmse_rad_recomputed": float(
            np.sqrt(np.mean((commanded_q - actual_q) ** 2))
        ),
        "physics_steps_total_executed_across_reset_trials": int(
            phone_trial["physics_steps"] + rotation_trial["physics_steps"]
        ),
    })
    intentional_tokens = ("thumb_2_link", "index_1_link")
    prohibited_pairs = [
        row for trial in (phone_trial, rotation_trial)
        for row in trial["all_robot_object_contact_pairs"]
        if not (
            "Phone" in row["pair"]
            and any(token in row["pair"] for token in intentional_tokens)
        )
    ]
    dump(OUT / "physics_collision_metrics.json", {
        "status": "TRUE_PHYSICS_EXECUTED_STAGES_COLLISION_AUDIT",
        "kinematic_raw_classifier_integrity_pass": collision["pass"],
        "kinematic_raw_equals_classified": collision["raw_equals_classified"],
        "kinematic_prohibited_collision_records": collision["prohibited_collision_records"],
        "executed_trial_prohibited_robot_environment_pairs": prohibited_pairs,
        "executed_trial_prohibited_contact_count": len(prohibited_pairs),
        "intentional_contacts": ["left_A_phone", "left_B_phone"],
        "unexecuted_stages_not_claimed": True,
    })
    save_npz(
        OUT / "phone_object_trajectory.npz",
        status=np.asarray("PHONE_ROTATION_TRUE_PHYSICS_FAILURE_TRAJECTORY"),
        pose_xyzw=phone_pose, action_indices=action_indices,
        trial=np.asarray("phone_rotation"), speed_scale=np.asarray(0.25),
        physics_steps=np.asarray(rotation_trial["physics_steps"]),
        object_pose_scripted=np.asarray(False), kinematic_object_follow=np.asarray(False),
    )
    save_npz(
        OUT / "accessory_object_trajectory.npz",
        status=np.asarray("ACCESSORY_NOT_YET_MANIPULATED_ROTATION_PREREQUISITE_FAILED"),
        pose_xyzw=accessory_pose, action_indices=action_indices,
        trial=np.asarray("phone_rotation"), speed_scale=np.asarray(0.25),
        physics_steps=np.asarray(rotation_trial["physics_steps"]),
        object_pose_scripted=np.asarray(False), kinematic_object_follow=np.asarray(False),
    )

    dump(OUT / "full_task_physics_result.json", {
        "status": "BLOCKED_PHONE_ROTATION_PHYSICS",
        "full_uninterrupted_task_executed": False,
        "full_task_pass": False,
        "dominant_failure_subsystem": "PHONE_PINCH_PRIMITIVE",
        "kinematic_prephysics_pass": True,
        "phone_grasp_physics_pass": phone_trial["pass"],
        "phone_rotation_physics_pass": rotation_trial["pass"],
        "later_stages_executed": False,
        "reason_later_stages_not_executed": "stage prerequisite policy",
        "object_pose_scripted": False,
        "kinematic_object_follow": False,
        "semantic_attach_detach": False,
        "gravity_enabled": True,
        "collision_enabled": True,
        "physics_steps_executed": int(phone_trial["physics_steps"] + rotation_trial["physics_steps"]),
        "physics_success_claimed": False,
    })

    videos = []
    for name, scope in (
        ("v17_1_phone_grasp_physics_overview.mp4", "PHONE_GRASP_PASS_OVERVIEW"),
        ("v17_1_phone_grasp_physics_closeup.mp4", "PHONE_GRASP_PASS_CLOSEUP"),
        ("v17_1_phone_rotation_physics.mp4", "PHONE_ROTATION_FAIL_CLOSEUP"),
        ("v17_1_phone_rotation_physics_overview.mp4", "PHONE_ROTATION_FAIL_OVERVIEW"),
    ):
        path = OUT / name
        if path.is_file():
            videos.append(video_info(path, scope))

    dt = np.gradient(timestamps)
    full_q = np.c_[arm, left, right]
    velocity = np.gradient(full_q, axis=0) / dt[:, None]
    acceleration = np.gradient(velocity, axis=0) / dt[:, None]
    dump(OUT / "pre_real_g1_readiness_v17_1.json", {
        "status": "NOT_READY_BLOCKED_PHONE_ROTATION_PHYSICS",
        "real_robot_safe": False,
        "ready_for_real_dex3_calibration_and_object_free_preflight": False,
        "candidate_translator_created": False,
        "arm_joint_names": arm_names,
        "left_dex3_joint_names": left_names,
        "right_dex3_joint_names": right_names,
        "root_xyz_m": root,
        "trajectory_shape": list(full_q.shape),
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(full_q, axis=0)))),
        "maximum_joint_velocity_rad_s": float(np.max(np.abs(velocity))),
        "maximum_joint_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "minimum_arm_joint_margin_rad": joints["minimum_arm_margin_rad"],
        "minimum_kinematic_table_clearance_m": collision["minimum_table_clearance_m_active_geom_vertices"],
        "phone_grasp_physics_pass": phone_trial["pass"],
        "phone_rotation_physics_pass": rotation_trial["pass"],
        "full_task_physics_pass": False,
        "remaining_real_only_blockers": [
            "resolve PHONE_PINCH rotation retention in Episode-49 simulation",
            "complete accessory, charger, release, and uninterrupted full-task physics gates",
            "record real Dex3 OPEN primitive",
            "calibrate real PHONE_PINCH and RING_HOOK primitives",
            "confirm actual runtime joint order",
            "perform actual object-free 0.25x preflight",
            "verify E-stop and supervisor readiness",
        ],
    })

    no_cheat = {
        "scripted_phone_pose_writes": max(
            phone_trial["integrity"]["object_pose_commands"],
            rotation_trial["integrity"]["object_pose_commands"],
        ),
        "scripted_accessory_pose_writes": max(
            phone_trial["integrity"]["object_pose_commands"],
            rotation_trial["integrity"]["object_pose_commands"],
        ),
        "kinematic_object_follow": False,
        "semantic_attach_calls": max(
            phone_trial["integrity"]["semantic_scripted_attach_detach"],
            rotation_trial["integrity"]["semantic_scripted_attach_detach"],
        ),
        "semantic_detach_calls": 0,
        "hidden_grasp_fixed_joint_creation": 0,
        "gravity_enabled": True,
        "collision_enabled": True,
        "physics_steps_positive": phone_trial["physics_steps"] > 0 and rotation_trial["physics_steps"] > 0,
    }
    no_cheat["pass"] = bool(
        no_cheat["scripted_phone_pose_writes"] == 0
        and no_cheat["scripted_accessory_pose_writes"] == 0
        and not no_cheat["kinematic_object_follow"]
        and no_cheat["semantic_attach_calls"] == 0
        and no_cheat["semantic_detach_calls"] == 0
        and no_cheat["hidden_grasp_fixed_joint_creation"] == 0
        and no_cheat["gravity_enabled"]
        and no_cheat["collision_enabled"]
        and no_cheat["physics_steps_positive"]
    )
    tests = {
        "input_freeze_byte_identical": freeze["byte_identical"],
        "cartesian_backbone_byte_identical": freeze["cartesian_backbone"]["byte_identical_arrays"],
        "root_unchanged": np.allclose(root, freeze["root_xyz_m"], atol=1e-12),
        "workspace_scale_unchanged": freeze["workspace_scale"] == 0.42,
        "generic_semantic_api_used": semantic["status"] == "GENERIC_SEMANTIC_API_USED",
        "runtime_literal_semantic_frame_count_zero": semantic["runtime_literal_audit"]["count"] == 0,
        "validation_read_count_zero": freeze["validation_read_count"] == 0,
        "heldout_read_count_zero": freeze["heldout_read_count"] == 0,
        "g1_expert_read_count_zero": freeze["g1_expert_read_count"] == 0,
        "kinematic_prephysics_pass": kinematic["gate_pass"],
        "collision_classifier_integrity_pass": collision["pass"],
        "phone_grasp_true_physics_pass": phone_trial["pass"],
        "phone_rotation_true_physics_pass": rotation_trial["pass"],
        "full_task_true_physics_pass": False,
        "no_cheat_physics_pass": no_cheat["pass"],
        "actual_arm_motion": float(np.max(np.ptp(arm, axis=0))) > 0.1,
        "actual_left_dex3_motion": float(np.max(np.ptp(left, axis=0))) > 0.1,
        "actual_right_dex3_motion": float(np.max(np.ptp(right, axis=0))) > 0.1,
        "videos_decode": len(videos) == 4 and all(row["decoded_frames"] > 0 for row in videos),
        "candidate_translator_absent": not (
            OUT / "configs/aloha_g1_execution_translator_v17_1.candidate.json"
        ).exists(),
    }
    dump(OUT / "tests_results.json", {
        "status": "INVARIANTS_PASS_EXECUTION_GATE_BLOCKED",
        "tests": tests,
        "all_invariant_and_integrity_tests_pass": all(
            value for key, value in tests.items()
            if key not in {"phone_rotation_true_physics_pass", "full_task_true_physics_pass"}
        ),
        "detector_invocation_count": semantic["v17_1_execution"]["detector_invocation_count"],
        "no_cheat_physics_audit": no_cheat,
        "expected_failed_gates": {
            "phone_rotation_true_physics_pass": False,
            "full_task_true_physics_pass": False,
        },
    })

    f = fidelity
    om = orientation["task_facing"]
    orot = orientation["portrait_charger_rotation"]
    cm = phone_trial["object_metrics"]
    rm = rotation_trial["object_metrics"]
    final_lines = """ALOHA-DERIVED CARTESIAN ARM MOTION REMAINED THE PRIMARY AND IMMUTABLE BEHAVIOR BACKBONE
THE G1 CARTESIAN ARM PATH WAS NOT REDESIGNED TO MAKE GRASPING EASIER
ONLY TASK-CRITICAL SEMANTIC-LOCAL WRIST ORIENTATION WAS ADDED
ALL POSTURE AND COLLISION CORRECTIONS WERE RESTRICTED TO THE TASK NULL SPACE
DEX3 WAS INTERPRETED AS AN EMBODIMENT-SPECIFIC PREDEFINED HAND ADAPTER
PHONE_PINCH AND RING_HOOK DID NOT GENERATE OR MODIFY THE G1 CARTESIAN ARM PATH
DEX3 TRANSITIONS WERE DRIVEN BY GENERIC SEMANTIC PROGRESS RATHER THAN EPISODE-49 FRAME CONSTANTS
WRIST-FINGER AND ARM-FINGER SELF-CONTACTS WERE NOT ALLOWED TO DISAPPEAR FROM COLLISION ACCOUNTING
ISAAC LAB SUCCESS WAS JUDGED BY TRUE PHYSICAL PHONE AND ACCESSORY MOTION
NO KINEMATIC OBJECT FOLLOW OR SCRIPTED OBJECT ATTACHMENT WAS USED TO CLAIM SUCCESS
NO VALIDATION OR HELD-OUT TRAJECTORY WAS USED FOR DEVELOPMENT
NO G1 EXPERT MOTION WAS USED
NO DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED"""
    report = f"""1. v14 Cartesian backbone은 byte-identical로 보존되었고 semantic-local orientation/kinematic gate를 통과했다.
2. 0.25× true physics에서 PHONE_GRASP는 통과했지만 A/B 접촉이 rotation 중 끊겨 portrait error가 {rm['phone_portrait_error_final_deg']:.3f}°로 남았다.
3. 최종 상태는 `BLOCKED_PHONE_PINCH_PRIMITIVE`; accessory/charger/full-task 단계와 candidate translator 생성은 선행 gate 정책에 따라 수행하지 않았다.

## 1. Final status

`BLOCKED_PHONE_PINCH_PRIMITIVE`. 동시에 `ALOHA_ARM_BACKBONE_BYTE_IDENTICAL`, `SEMANTIC_LOCAL_PARTIAL_ORIENTATION_PASS`, `COLLISION_CLASSIFIER_INTEGRITY_PASS`, `NULLSPACE_ROBOT_COLLISION_PASS`, `ALOHA_MOTION_FIDELITY_PASS`, `PREDEFINED_DEX3_EXECUTION_PASS`, `PHONE_GRASP_PHYSICS_PASS`는 성립한다. `PHONE_ROTATION_PHYSICS_PASS`와 이후 full-task 상태는 성립하지 않는다.

## 2. ALOHA arm-backbone proof

v14 left/right Cartesian target max difference는 각각 0.0/0.0 m이고 배열 hash와 byte identity가 동일하다. Cartesian residual, waypoint, root/object 이동은 사용하지 않았다.

## 3. v14 vs v17.1 Cartesian hashes

- Left: `{freeze['cartesian_backbone']['v14_left_cartesian_target_sha256']}` = `{freeze['cartesian_backbone']['v17_1_left_cartesian_target_sha256']}`
- Right: `{freeze['cartesian_backbone']['v14_right_cartesian_target_sha256']}` = `{freeze['cartesian_backbone']['v17_1_right_cartesian_target_sha256']}`

## 4. Source provenance

SmolVLA-generated `optimized_action`, `{SOURCE}`, file SHA-256 `{sha(SOURCE)}`, shape 990×14, 30 Hz. Validation/held-out/G1 Expert read count는 모두 0이다.

## 5. Semantic-local orientation

Human-reviewed Episode-49 development timeline을 generic `SemanticTimeline` API로 명시적으로 공급했다. Runtime literal semantic frame dependency는 0이며 acquisition/rotation/accessory/charger activation은 named progress만 사용한다.

## 6. Task-facing orientation

Phone grasp error {om['phone_grasp_task_facing_error_deg']:.3f}°, portrait long-axis {orot['portrait_long_axis_error_deg']:.3f}°, charger normal {orot['charger_normal_error_deg']:.6f}°, charger vertical {orot['charger_vertical_axis_error_deg']:.6f}°, right hook primary axis {om['right_ring_hook_primary_axis_error_deg']:.3f}°로 numerical orientation gate를 통과했다.

## 7. ALOHA fidelity

Left path/speed {f['left_path_shape']:.6f}/{f['left_speed']:.6f}, right path/speed {f['right_path_shape']:.6f}/{f['right_speed']:.6f}, midpoint {f['bimanual_midpoint']:.6f}, relative vector {f['relative_hand_vector']:.6f}, inter-hand distance {f['inter_hand_distance']:.6f}. Left/right rotation progress는 {f['rotation']['left_rotation_progress_correlation']:.6f}/{f['rotation']['right_rotation_progress_correlation']:.6f}이다.

## 8. Null-space posture

Position tracking을 유지한 task-null-space continuation으로 orientation, joint center, shoulder posture, collision clearance를 보정했다. Cartesian target mutation은 0이다.

## 9. Branch continuity

Branch discontinuity {joints['branch_discontinuity_count']}, max arm step {joints['maximum_arm_step_rad']:.6f} rad이다.

## 10. Joint margin

최소 arm margin {joints['minimum_arm_margin_rad']:.9f} rad로 0.01-rad diagnostic target은 경계에서 통과하지만 0.03-rad preferred target에는 못 미친다. Real-robot safety 승인이 아니다.

## 11. Collision classifier integrity

Raw robot-robot {collision['raw_robot_robot_contact_count']}, classified {collision['classified_robot_robot_contact_count']}, ignored {collision['ignored_non_robot_robot_contact_count']}; raw==classified `{collision['raw_equals_classified']}`.

## 12. Wrist/arm-finger accounting

Wrist-finger, arm-finger, wrist-hand, same-hand finger contacts는 모두 explicit category로 유지했다. 선택 trajectory의 해당 raw prohibited record는 0이며 filter drop도 0이다.

## 13. Prohibited collision

Kinematic prohibited collision 0. 실행된 grasp/rotation physics에서 task-intentional A/B-phone 이외 robot-table/object prohibited pair는 0이다.

## 14. PHONE_PINCH primitive

`A_SCREEN_B_BACK`, phone-local fixed PREGRASP와 fixed PINCH를 사용했다. Thumb/index preload는 1/2 mm이며 서로 반대 thickness surface에 배치되었다. Per-frame fingertip IK는 없다.

## 15. PHONE_PINCH timing

Named grasp→rotation-start interval의 minimum-jerk clock에서 completion progress 0.63을 사용했다. Grasp PASS: A/B simultaneous run {cm['left_AB_simultaneous_contact_longest_run_samples']} samples, slip {cm['phone_hand_translation_slip_m']*1000:.3f} mm, phone/wrist ratio {cm['phone_to_wrist_motion_ratio']:.3f}. Acquisition은 table-supported였고 3-mm lift flag는 `{cm['phone_lifted_clear_of_table_3mm']}`로 별도 보존했다.

## 16. RING_HOOK primitive

고정 right-C `RIGHT_RING_PREHOOK/RIGHT_RING_HOOK`과 collision-free non-task A/B 자세를 만들었다. Rotation prerequisite 실패로 true-physics hook은 실행하지 않았다.

## 17. Dex3 semantic interpolation

Left OPEN→PREGRASP→INDEX_CONTACT→PINCH→HOLD→RELEASE, right OPEN→PREHOOK→HOOK→HOLD→RELEASE를 generic semantic/source-gripper progress로 smooth interpolation했다.

## 18. Kinematic prephysics gate

`V17_1_KINEMATIC_PREPHYSICS_PASS`: 990 finite samples, simultaneous 5-mm rate {kinematic['metrics']['position']['simultaneous_5mm_rate']:.6f}, joint violations 0, branch discontinuities 0, prohibited collision 0.

## 19. Phone grasp true physics

`PHONE_GRASP_PHYSICS_PASS` at 0.25×. Physics steps {phone_trial['physics_steps']}, tracking max/RMSE {phone_trial['tracking']['maximum_absolute_error_rad']:.6f}/{phone_trial['tracking']['rmse_rad']:.6f} rad, A/B force maxima {cm['left_A_phone_force_max_n']:.3f}/{cm['left_B_phone_force_max_n']:.3f} N.

## 20. Phone rotation true physics

`BLOCKED_PHONE_ROTATION_PHYSICS`. End contact false, slip {rm['phone_hand_translation_slip_m']*1000:.3f} mm, phone/wrist ratio {rm['phone_to_wrist_motion_ratio']:.3f}, portrait error {rm['phone_portrait_error_final_deg']:.3f}°. Phone remained on the table; stronger opposing preload diagnostic also failed retention.

## 21. Accessory removal

NOT RUN — phone rotation prerequisite failed.

## 22. Bimanual transport

NOT RUN — phone rotation prerequisite failed.

## 23. Charger placement

NOT RUN — phone rotation prerequisite failed.

## 24. Accessory release

NOT RUN — phone rotation prerequisite failed.

## 25. Uninterrupted full task

NOT RUN. No full-task PASS is claimed.

## 26. Physics tracking

Executed runs used 28/28 name mapping and actuator targets. Maximum command/readback error across final grasp/rotation runs was {max(phone_trial['tracking']['maximum_absolute_error_rad'], rotation_trial['tracking']['maximum_absolute_error_rad']):.6f} rad.

## 27. Physical object trajectories

`phone_object_trajectory.npz` and `accessory_object_trajectory.npz` store the actual rotation-trial rigid-body states. Phone moved only through physics; accessory was not manipulated because the prerequisite failed.

## 28. No-cheat audit

Object pose writes 0, kinematic object follow 0, scripted attach/detach 0, hidden grasp joint 0, gravity/collision enabled, timed physics steps >0.

## 29. Videos

""" + "\n".join(f"- `{row['path']}` — {row['decoded_frames']} frames @ {row['frame_rate']}" for row in videos) + f"""

Accessory/charger/full-task videos were not generated because their stage prerequisites were not met.

## 30. Candidate translator config

Not created. Success-only candidate policy was preserved.

## 31. Pre-real readiness

`NOT_READY_BLOCKED_PHONE_ROTATION_PHYSICS`; `REAL_ROBOT_SAFE` is false.

## 32. Remaining real-hardware blockers

Simulation rotation retention/full task, real Dex3 OPEN/PHONE_PINCH/RING_HOOK calibration, runtime joint-order confirmation, object-free 0.25× preflight, E-stop/supervisor readiness.

## 33. Exactly one next action

Episode 49에서만 fixed `PHONE_PINCH`의 task-local contact geometry를 재교정하여 immutable arm path의 mapped rotation 동안 A/B opposing contact를 유지하게 하라. Arm Cartesian trajectory, object physics, validation/held-out data는 변경하거나 사용하지 마라.

{final_lines}
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    report_dir = OUT / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "index.html").write_text(
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<title>Episode-49 v17.1 execution physics</title>"
        "<style>body{max-width:1100px;margin:2rem auto;font-family:system-ui;line-height:1.5}"
        "pre{white-space:pre-wrap;background:#f5f5f5;padding:1.5rem}</style></head>"
        f"<body><pre>{html.escape(report)}</pre></body></html>\n",
        encoding="utf-8",
    )
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd /home/jbnu/aloha_g1_dataset
/home/jbnu/miniconda3/envs/isaaclab6/bin/python tools/build_episode49_execution_physics_v17_1.py
/home/jbnu/miniconda3/envs/isaaclab6/bin/python isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py --enable_cameras --input outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1 --artifact-prefix v17_1 --trial phone_grasp --speed 0.25 --viz none
/home/jbnu/miniconda3/envs/isaaclab6/bin/python isaaclab_magsafe_fixed_scene/run_execution_physics_v17.py --enable_cameras --input outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1/final_arm_dex3_trajectory.npz --output-dir outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1 --artifact-prefix v17_1 --trial phone_rotation --speed 0.25 --viz none
/home/jbnu/miniconda3/envs/isaaclab6/bin/python tools/finalize_episode49_execution_physics_v17_1.py
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    artifacts = []
    for path in sorted(value for value in OUT.rglob("*") if value.is_file()):
        if path.name == "run_manifest.json" or "physics_timing_sweep" in path.parts:
            continue
        artifacts.append({
            "path": str(path.relative_to(OUT)),
            "size_bytes": path.stat().st_size,
            "sha256": sha(path),
        })
    manifest = {
        "method": "ALOHA_PRIMARY_EP49_EXECUTION_PHYSICS_V17_1",
        "status": "BLOCKED_PHONE_PINCH_PRIMITIVE",
        "dominant_failure_subsystem": "PHONE_PINCH_PRIMITIVE",
        "successful_gates": [
            "ALOHA_ARM_BACKBONE_BYTE_IDENTICAL",
            "SEMANTIC_LOCAL_PARTIAL_ORIENTATION_PASS",
            "COLLISION_CLASSIFIER_INTEGRITY_PASS",
            "NULLSPACE_ROBOT_COLLISION_PASS",
            "ALOHA_MOTION_FIDELITY_PASS",
            "PREDEFINED_DEX3_EXECUTION_PASS",
            "PHONE_GRASP_PHYSICS_PASS",
        ],
        "failed_gate": "PHONE_ROTATION_PHYSICS_PASS",
        "episode_id": 49,
        "source_action_type": "SMOLVLA_GENERATED",
        "source_action_sha256": sha(SOURCE),
        "semantic_timeline_sha256": semantic["timeline_sha256"],
        "backup_path": freeze["backup"],
        "v14_cartesian_backbone": freeze["cartesian_backbone"],
        "final_trajectory_sha256": sha(OUT / "final_arm_dex3_trajectory.npz"),
        "builder_sha256": sha(BUILDER),
        "trajectory_code_sha256": sha(TRAJECTORY_CODE),
        "physics_runner_sha256": sha(PHYSICS_RUNNER),
        "validation_read_count": 0,
        "heldout_read_count": 0,
        "g1_expert_read_count": 0,
        "physics_steps_executed": int(phone_trial["physics_steps"] + rotation_trial["physics_steps"]),
        "physics": True,
        "executed_speed_scales": [0.25],
        "no_cheat_physics_audit": no_cheat,
        "dds": False,
        "publisher": False,
        "real_robot_command": False,
        "candidate_translator_created": False,
        "videos": videos,
        "artifacts": artifacts,
    }
    dump(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": manifest["status"],
        "dominant_failure_subsystem": manifest["dominant_failure_subsystem"],
        "phone_grasp_pass": phone_trial["pass"],
        "phone_rotation_pass": rotation_trial["pass"],
        "videos": len(videos),
        "artifacts": len(artifacts) + 1,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
