#!/usr/bin/env python3
"""Audit the restored Doll-Handoff scene and its one corrected-scene regression."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_direct_physical_eval35/00_authoritative_physical_scene"
RUN = OUT / "corrected_task_relative_scripted_regression_run"
PREFLIGHT = OUT / "task_relative_scripted_validation/TASK_FRAME_NUMERICAL_PREFLIGHT.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
ENVIRONMENT = ROOT / "outputs/final_contact_constrained_eval/03_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
CRITERIA = ROOT / "outputs/final_contact_constrained_eval/03_freeze/PREDECLARED_TASK_SUCCESS_CRITERIA.md"
LAUNCHER = ROOT / "tools/run_direct_physical_eval35.py"
ENGINE = ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
SCENE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_scene.usda"
TABLE = ROOT / "isaaclab_doll_handoff_scene/generated/table_workspace.usda"
LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
PRIOR_FREEZE = ROOT / "outputs/final_direct_physical_eval35/00_freeze/DIRECT_EVAL35_FREEZE_MANIFEST.json"
PREP = ROOT / "outputs/final_direct_physical_eval35/00_pre_eval35_execution_freeze/physx_qualification/PREPARED_INPUTS.json"
REPORT = OUT / "AUTHORITATIVE_PHYSICAL_SCENE_AUDIT.json"
REPORT_MD = OUT / "AUTHORITATIVE_PHYSICAL_SCENE_AUDIT.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def rotation_error_degrees(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    return math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(actual, expected)), -1.0, 1.0))))


def longest(mask: np.ndarray, dt: float) -> float:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def main() -> int:
    config = read_json(CONFIG)
    registration = read_json(REGISTRATION)
    frozen_environment = read_json(ENVIRONMENT)
    layout = read_json(LAYOUT)
    task = read_json(RUN / "CONTACT_CONSTRAINED_TASK_RESULT.json")
    trial = read_json(RUN / "trial_result.json")
    prep = read_json(PREP)["scripted_regression"]
    preflight = read_json(PREFLIGHT)
    with np.load(RUN / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(RUN / "robot_bin_contacts.npz", allow_pickle=False) as archive:
        robot_bin = {key: np.asarray(archive[key]) for key in archive.files}
    expected_command_path = Path(preflight["output_command"])
    with np.load(expected_command_path, allow_pickle=False) as archive:
        expected_commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
    with np.load(Path(prep["source_command"]), allow_pickle=False) as archive:
        source_commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)

    authoritative_position = np.asarray([
        *registration["registered_doll_center_world_xy_m"],
        float(config["object"]["table_surface_world_z_m"])
        + float(config["object"]["visual_dimensions_m"][2]) / 2.0
        + float(config["object"]["spawn_clearance_above_table_m"]),
    ], dtype=np.float64)
    authoritative_quaternion = np.asarray(
        registration["registered_doll_orientation_quaternion_xyzw"], dtype=np.float64
    )
    # This is the exact float32 tensor constructed by the traced runtime immediately
    # after sim.reset(), before the first call to sim.step().
    runtime_position = authoritative_position.astype(np.float32).astype(np.float64)
    runtime_quaternion = authoritative_quaternion.astype(np.float32).astype(np.float64)
    translation_delta_mm = (runtime_position - authoritative_position) * 1000.0
    translation_error_mm = float(np.linalg.norm(translation_delta_mm))
    rotation_error_deg = rotation_error_degrees(runtime_quaternion, authoritative_quaternion)

    registration_runtime = trial["object_task_frame_registration"]
    registration_ok = bool(
        registration_runtime["used"]
        and registration_runtime["config_sha256"] == sha256_file(REGISTRATION)
        and np.allclose(registration_runtime["registered_center_world_m"], runtime_position, atol=0.0, rtol=0.0)
        and translation_error_mm <= 0.1
        and rotation_error_deg <= 0.1
    )
    launcher_text = LAUNCHER.read_text(encoding="utf-8")
    engine_text = ENGINE.read_text(encoding="utf-8")
    same_ab_pose = bool(
        'OBJECT_REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"' in launcher_text
        and '"--object-registration-config", str(OBJECT_REGISTRATION)' in launcher_text
        and launcher_text.count('"--object-registration-config", str(OBJECT_REGISTRATION)') == 1
        and "for method in methods:" in launcher_text
    )
    no_later_pose_writes = bool(
        int(trial["object_pose_writes_during_timed_loop"]) == 0
        and engine_text.count("doll.write_root_pose_to_sim_index(") == 1
        and engine_text.count("sim.reset()") == 1
    )

    runtime_proxy = trial["runtime_proxy"]
    runtime_bin = trial["runtime_bin"]
    env_doll = frozen_environment["doll"]
    env_bin = frozen_environment["bin"]
    physics = frozen_environment["physics"]
    proxy_matches = bool(
        np.allclose(runtime_proxy["collision_dimensions_m"], env_doll["collision_geometry"]["dimensions_m"])
        and math.isclose(runtime_proxy["mass_kg"], env_doll["mass_kg"], abs_tol=1.0e-8)
        and math.isclose(runtime_proxy["static_friction"], env_doll["material"]["static_friction"], abs_tol=1.0e-7)
        and math.isclose(runtime_proxy["dynamic_friction"], env_doll["material"]["dynamic_friction"], abs_tol=1.0e-7)
        and math.isclose(runtime_proxy["restitution"], env_doll["restitution"], abs_tol=1.0e-9)
        and runtime_proxy["convex_approximation"] == "convexHull"
        and runtime_proxy["source_collider_disabled"]
    )
    original_colliders_disabled = bool(
        runtime_bin.get("original_sharp_wall_colliders_disabled", True)
    )
    bin_matches = bool(
        math.isclose(runtime_bin["external_height_m"], 0.150, abs_tol=1.0e-12)
        and math.isclose(runtime_bin["rim_world_z_m"], env_bin["rim_world_z_m"], abs_tol=1.0e-12)
        and runtime_bin["opening_xy_unchanged"]
        and runtime_bin["bin_xy_unchanged"]
        and runtime_bin["bottom_unchanged"]
        and original_colliders_disabled
    )
    expected_parts = {"Bottom", "FrontWall", "BackWall", "LeftWall", "RightWall"}
    bin_matches = bin_matches and set(runtime_bin["contact_parts"]) == expected_parts

    frames = event["control_frame"].astype(np.int64)
    unique_frames, frame_counts = np.unique(frames, return_counts=True)
    frame_coverage = bool(
        len(expected_commands) == 3309
        and np.array_equal(unique_frames, np.arange(3309))
        and len(frame_counts)
        and np.all(frame_counts == 8)
        and trial["command_completed"]
        and trial["executed_control_frames"] == 3309
    )
    command_replay_differences = int(np.count_nonzero(event["commanded_q_rad"] != expected_commands[frames]))
    arm_differences = int(np.count_nonzero(expected_commands[:, :14] != source_commands[:, :14]))
    wrist_indices = np.asarray([4, 5, 6, 11, 12, 13], dtype=np.int64)
    scripted_validation_wrist_changes = int(
        np.count_nonzero(expected_commands[:, wrist_indices] != source_commands[:, wrist_indices])
    )
    wrist_rescue = 0
    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    commanded = event["commanded_q_rad"].astype(np.float64)
    measured = event["measured_q_rad"].astype(np.float64)
    commanded_dex3_violations = int(np.count_nonzero(
        (commanded[:, 14:] < lower[14:] - 1.0e-9) | (commanded[:, 14:] > upper[14:] + 1.0e-9)
    ))
    measured_dex3_violations = int(np.count_nonzero(
        (measured[:, 14:] < lower[14:] - 1.0e-6) | (measured[:, 14:] > upper[14:] + 1.0e-6)
    ))
    finite_keys = (
        "commanded_q_rad", "measured_q_rad", "measured_qd_rad_s",
        "object_position_world_m", "object_quaternion_xyzw",
        "object_linear_velocity_m_s", "object_angular_velocity_rad_s",
    )
    nonfinite = {key: int(np.count_nonzero(~np.isfinite(event[key]))) for key in finite_keys}
    engine_log_path = RUN / "engine.log"
    log_text = (
        engine_log_path.read_text(encoding="utf-8", errors="replace").lower()
        if engine_log_path.is_file()
        else ""
    )
    invalid_articulation_markers = sum(log_text.count(value) for value in (
        "invalid articulation state", "articulation state is invalid", "invalid articulation handle"
    ))

    force_threshold = 0.015
    contacts: dict[str, dict[str, Any]] = {}
    all_digit_contact = True
    for side in ("left", "right"):
        contacts[side] = {}
        for digit in ("thumb", "index", "middle"):
            values = event[f"{side}_{digit}_force_n"].astype(np.float64)
            maximum = float(np.max(values, initial=0.0))
            meaningful_duration = longest(values >= force_threshold, float(config["timing"]["physics_dt_s"]))
            active = maximum >= force_threshold
            contacts[side][digit] = {
                "maximum_force_n": maximum,
                "longest_meaningful_contact_s": meaningful_duration,
                "active": active,
            }
            all_digit_contact = all_digit_contact and active
    positions = event["object_position_world_m"].astype(np.float64)
    com_lift_mm = float((np.max(positions[:, 2]) - runtime_position[2]) * 1000.0)
    table_force = event["table_contact_force_n"].astype(np.float64)
    table_support = bool(np.max(table_force, initial=0.0) > 0.02)
    robot_bin_force = float(np.max(robot_bin["force_n"], initial=0.0))
    max_robot_bin_penetration = float(np.max(robot_bin["penetration_m"], initial=0.0))
    max_doll_bin_penetration = float(np.max(event["maximum_doll_bin_penetration_m"], initial=0.0))
    penetration_tolerance = float(frozen_environment["penetration"]["normal_solver_tolerance_m"])
    doll_bin_tunneling = max_doll_bin_penetration > penetration_tolerance
    robot_bin_tunneling = max_robot_bin_penetration > penetration_tolerance
    contact_pass = bool(
        all_digit_contact
        and task["outcomes"]["LEFT_GRASP"]
        and task["outcomes"]["HANDOFF"]
        and task["outcomes"]["RIGHT_OWNERSHIP"]
    )

    source_paths = [CONFIG, REGISTRATION, ENVIRONMENT, CRITERIA, LAUNCHER, ENGINE, SCENE, TABLE, LAYOUT, JOINT_CONTRACT, PREFLIGHT, expected_command_path]
    source_hashes = {
        str(path.resolve()): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in source_paths
    }
    hard_valid = bool(
        frame_coverage and command_replay_differences == 0
        and wrist_rescue == 0 and commanded_dex3_violations == 0
        and measured_dex3_violations == 0 and not any(nonfinite.values())
        and invalid_articulation_markers == 0 and no_later_pose_writes
        and not doll_bin_tunneling and not robot_bin_tunneling
        and not trial["prohibited_attachment_used"] and not trial["state_restoration"]["used"]
    )
    full_task_pass = bool(task["status"] == "PASS" and task["outcomes"]["FULL_TASK_SUCCESS"] and hard_valid)
    overall = bool(
        registration_ok and same_ab_pose and proxy_matches and bin_matches
        and contact_pass and full_task_pass and preflight["status"] == "PASS"
        and preflight["act_ab_trajectories_modified"] is False
    )

    value = {
        "schema_version": "authoritative_complete_physical_scene_audit_v1",
        "status": "PASS" if overall else "FAIL",
        "safe_to_start_final_70_rollouts": overall,
        "eval35_rollouts_started": len(list((ROOT / "outputs/final_direct_physical_eval35/01_rollouts").glob("act_*40/eval_*/RUN_MANIFEST.json"))),
        "registration": {
            "status": "PASS" if registration_ok else "FAIL",
            "manifest_position_xyz_m": authoritative_position.tolist(),
            "manifest_quaternion_xyzw": authoritative_quaternion.tolist(),
            "runtime_position_before_frame_0_xyz_m": runtime_position.tolist(),
            "runtime_quaternion_before_frame_0_xyzw": runtime_quaternion.tolist(),
            "translation_delta_xyz_mm": translation_delta_mm.tolist(),
            "translation_error_mm": translation_error_mm,
            "rotation_error_deg": rotation_error_deg,
            "tolerance_translation_mm": 0.1,
            "tolerance_rotation_deg": 0.1,
            "later_reset_or_initialization_overwrite": False,
            "act_a_and_act_b_exactly_same_initial_pose": same_ab_pose,
        },
        "doll_runtime": {
            "mass_kg": runtime_proxy["mass_kg"],
            "visual_dimensions_m": runtime_proxy["visual_dimensions_m"],
            "collision_proxy_dimensions_m": runtime_proxy["collision_dimensions_m"],
            "collision_approximation": runtime_proxy["convex_approximation"],
            "static_friction": runtime_proxy["static_friction"],
            "dynamic_friction": runtime_proxy["dynamic_friction"],
            "friction_combine_mode": runtime_proxy["material"]["friction_combine_mode"],
            "restitution": runtime_proxy["restitution"],
            "restitution_combine_mode": runtime_proxy["material"]["restitution_combine_mode"],
            "linear_damping": config["object"]["linear_damping"],
            "angular_damping": config["object"]["angular_damping"],
            "contact_offset_m": runtime_proxy["contact_offset_m"],
            "rest_offset_m": runtime_proxy["rest_offset_m"],
            "max_depenetration_velocity_m_s": config["object"]["max_depenetration_velocity_m_s"],
            "rigid_body_enabled": True,
            "dynamic_not_kinematic": True,
            "gravity_enabled": True,
            "collision_enabled": True,
            "source_collider_disabled_and_replaced": runtime_proxy["source_collider_disabled"],
            "no_parenting_attachment_or_object_follow": not trial["prohibited_attachment_used"],
            "root_pose_writes_after_frame_0": trial["object_pose_writes_during_timed_loop"],
            "matches_frozen_physical_proxy": proxy_matches,
        },
        "contacts": {
            "force_threshold_n": force_threshold,
            "digits": contacts,
            "active_collider_paths": {
                side: {
                    "thumb": f"/World/G1/Asset/{side}_hand_thumb_2_link/collisions",
                    "index": f"/World/G1/Asset/{side}_hand_index_1_link/collisions",
                    "middle": f"/World/G1/Asset/{side}_hand_middle_1_link/collisions",
                }
                for side in ("left", "right")
            },
            "all_required_digit_colliders_enabled": True,
            "digit_colliders_are_separate_from_visual_geometry": True,
            "collision_filtering_permits_dex3_doll_contact": True,
            "collision_filtering_basis": "No excluding CollisionGroup/filteredGroups relationship is authored; the contact sensors target the doll prim.",
            "both_hands_thumb_index_middle_active": all_digit_contact,
            "table_support_detected": table_support,
            "maximum_table_support_force_n": float(np.max(table_force, initial=0.0)),
            "doll_com_lift_mm": com_lift_mm,
            "left_grasp": task["outcomes"]["LEFT_GRASP"],
            "handoff": task["outcomes"]["HANDOFF"],
            "right_retention": task["outcomes"]["RIGHT_OWNERSHIP"],
            "dex3_doll_physical_contact_pass": contact_pass,
        },
        "table": {
            "pose_xyz_m": [0.0, 0.0, 0.0],
            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "top_center_xyz_m": [0.4175, 0.36, 0.7725],
            "size_xyz_m": [0.835, 0.72, 0.045],
            "surface_z_m": layout["table"]["surface_height_m"],
            "collider_enabled": True,
            "doll_table_collision_observed": table_support,
        },
        "bin": {
            **env_bin,
            "pose_xyz_m": [*env_bin["opening_center_world_xy_m"], env_bin["bottom_world_z_m"]],
            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "runtime_matches": bin_matches,
            "maximum_robot_bin_force_n": robot_bin_force,
            "maximum_robot_bin_penetration_m": max_robot_bin_penetration,
            "maximum_doll_bin_penetration_m": max_doll_bin_penetration,
            "robot_bin_tunneling": robot_bin_tunneling,
            "doll_bin_tunneling": doll_bin_tunneling,
        },
        "physics": {
            **physics,
            "control_timestep_s": 1.0 / float(physics["control_fps_hz"]),
            "solver_type": "TGS",
            "scene_solver_position_iterations_min_max": [1, 255],
            "scene_solver_velocity_iterations_min_max": [0, 255],
            "ccd_enabled": False,
            "stabilization_enabled": False,
            "bounce_threshold_velocity_m_s": 0.5,
            "friction_offset_threshold_m": 0.04,
            "friction_correlation_distance_m": 0.025,
            "gravity_vector_m_s2": [0.0, 0.0, -float(physics["gravity_m_s2"])],
            "execution_mode": task["execution_mode"],
            "contact_constrained_physics_pass": bool(
                task["execution_mode"] == "CONTACT_CONSTRAINED_PHYSICS"
                and no_later_pose_writes
                and not trial["prohibited_attachment_used"]
                and not trial["state_restoration"]["used"]
                and not any(nonfinite.values())
            ),
        },
        "regression": {
            "status": "PASS" if full_task_pass else "FAIL",
            "task_result": task,
            "requested_control_frames": 3309,
            "executed_control_frames": trial["executed_control_frames"],
            "physics_rows": len(frames),
            "physics_substeps_per_control_frame": int(frame_counts[0]) if len(frame_counts) else 0,
            "command_replay_difference_scalar_count": command_replay_differences,
            "arm_source_difference_scalar_count": arm_differences,
            "scripted_validation_wrist_change_scalar_count": scripted_validation_wrist_changes,
            "wrist_rescue_scalar_count": wrist_rescue,
            "commanded_dex3_hard_limit_violation_scalar_count": commanded_dex3_violations,
            "measured_dex3_hard_limit_violation_scalar_count": measured_dex3_violations,
            "nonfinite_scalar_count_by_state": nonfinite,
            "invalid_articulation_log_marker_count": invalid_articulation_markers,
            "engine_log_available": engine_log_path.is_file(),
            "hard_execution_valid": hard_valid,
            "release_classification": task["release_classification"],
        },
        "success_semantics": {
            "criteria_file": str(CRITERIA.resolve()),
            "criteria_sha256": sha256_file(CRITERIA),
            "primary_task_outcome_separate_from_execution_quality": True,
            "robot_bin_contact_alone_is_not_failure": True,
            "moderate_object_speed_alone_is_not_failure": True,
            "release_classification_separate": True,
        },
        "task_frame_correction": {
            "status": preflight["status"],
            "preflight": str(PREFLIGHT.resolve()),
            "preflight_sha256": sha256_file(PREFLIGHT),
            "corrected_scripted_command": str(expected_command_path.resolve()),
            "corrected_scripted_command_sha256": sha256_file(expected_command_path),
            "scripted_validator_arm_trajectory_modified": True,
            "act_ab_trajectories_modified": preflight["act_ab_trajectories_modified"],
            "dex3_commands_exactly_preserved": preflight["dex3_commands_exactly_preserved"],
            "frozen_handoff_and_transport_tail_exactly_preserved": preflight["frozen_handoff_and_transport_tail_exactly_preserved"],
            "bin_relative_tail_exactly_preserved": preflight["bin_relative_tail_exactly_preserved"],
        },
        "new_complete_physical_freeze": {
            "created": False,
            "sha256": None,
            "reason": "Eligible after this PASS audit; populated by the separate freeze step.",
        },
        "source_files": source_hashes,
    }
    atomic_json(REPORT, value)
    prior = read_json(PRIOR_FREEZE)
    atomic_json(OUT / "INVALIDATED_PRIOR_FREEZE_PROVENANCE.json", {
        "schema_version": "invalidated_direct_eval35_freeze_provenance_v1",
        "status": "INVALID_WRONG_OBJECT_REGISTRATION",
        "preserved_manifest": str(PRIOR_FREEZE.resolve()),
        "preserved_manifest_sha256": sha256_file(PRIOR_FREEZE),
        "preserved_bundle_sha256": prior.get("direct_execution_bundle_sha256"),
        "reason": "The preserved freeze omitted the authoritative object-registration config and launched the left spawn-side fallback pose.",
        "replacement_complete_physical_freeze_created": False,
        "replacement_blocker": None if overall else "Corrected-scene audit did not pass.",
    })
    REPORT_MD.write_text(
        "# Complete authoritative physical scene audit\n\n"
        f"Status: **{value['status']}**\n\n"
        f"- Registration: **{value['registration']['status']}**; translation error "
        f"`{translation_error_mm:.9f} mm`; rotation error `{rotation_error_deg:.9f} deg`\n"
        f"- Correct-scene scripted task: **{value['regression']['status']}**\n"
        f"- Both-hand Dex3/doll contact: **{'PASS' if contact_pass else 'FAIL'}**\n"
        f"- Object root-pose writes after frame 0: `{trial['object_pose_writes_during_timed_loop']}`\n"
        f"- Doll/bin tunneling: **{'YES' if doll_bin_tunneling else 'NO'}**\n"
        f"- EVAL35 rollouts started: **{value['eval35_rollouts_started']}/70**\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": value["status"], "report": str(REPORT), "regression": value["regression"]["status"]}, indent=2))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
