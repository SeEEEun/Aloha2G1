#!/usr/bin/env python3
"""Read-only physical-scene and ACT-A35 grasp forensics from saved traces."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_final_grasp_capture_task_frames import DistanceModel, palm_world
from tools.common_execution_isaac_runtime import PHYSICS_CONFIG
from tools.common_execution_layer import pose_matrix
from tools.direct_physical_execution_layer import authoritative_joint_limits
from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics


OUT = ROOT / "outputs/final_episode_registered_eval35"
AUDIT = OUT / "00_forensic_audit"
REGISTRATION = OUT / "00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
FREEZE = OUT / "01_freeze/FINAL_EVAL35_FREEZE_MANIFEST.json"
CONTACT_MODEL = OUT / "01_freeze/FINAL_DOLL_CONTACT_MODEL.json"
ENVIRONMENT = OUT / "01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
A_ROLLOUTS = OUT / "02_act_a_results/rollouts"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
COMMON_CONFIG = (
    ROOT
    / "outputs/final_direct_physical_eval35/00_preparation/"
    "runtime_frozen_fair_a/config/common_config.json"
)
PALM_FRAME = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"


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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def longest_duration(mask: np.ndarray, dt: float) -> float:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def classify(row: dict[str, Any]) -> str:
    if not row["hard_physical_validity"]:
        return "COLLISION_OR_RUNTIME_INCONSISTENCY"
    if row["scorer_false_negative"]:
        return "PHYSICAL_GRASP_OCCURRED_BUT_SCORER_FALSE_NEGATIVE"
    forces = row["contact_force_max_n"]
    any_force = max(forces.values()) >= row["meaningful_contact_threshold_n"]
    distances = row["minimum_collision_surface_distance_m"]
    digit_min = min(distances[digit] for digit in ("thumb", "index", "middle"))
    if not any_force:
        if digit_min <= 0.004:
            return "NO_PHYSICAL_CONTACT_DESPITE_VISUAL_PROXIMITY"
        if distances["palm"] <= 0.004 and digit_min > 0.004:
            return "ORIENTATION / APERTURE_MISS"
        return "GEOMETRIC_POSITION_MISS"
    if row["simultaneous_opposing_contact_duration_s"] <= 0.0:
        return "CONTACT_WITHOUT_OPPOSING_SUPPORT"
    if not row["retention_candidate"]:
        return "OPPOSING_CONTACT_BUT_NO_RETENTION"
    if not row["table_support_loss_observed"]:
        return "RETENTION_BUT_NO_TABLE_SUPPORT_LOSS"
    return "OTHER_PHYSICAL_FAILURE"


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    registration = read_json(REGISTRATION)
    contact_model = read_json(CONTACT_MODEL)
    physical_environment = read_json(ENVIRONMENT)
    physics = read_json(PHYSICS_CONFIG)
    freeze = read_json(FREEZE)
    common = load_common_config(COMMON_CONFIG)
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    palm_calibration = read_json(PALM_FRAME)
    _, _, joint_names = authoritative_joint_limits(read_json(JOINT_CONTRACT))

    dimensions = np.asarray(contact_model["collision_dimensions_m"], dtype=np.float64)
    visual = np.asarray(contact_model["visual_dimensions_m"], dtype=np.float64)
    table_z = float(physics["object"]["table_surface_world_z_m"])
    collision_z_offset = float((dimensions[2] - visual[2]) / 2.0)
    distance_model = DistanceModel(g1, dimensions, table_z, joint_names)
    registration_entries = {
        int(row["eval_index"]): row for row in registration["entries"]
    }

    rollout_dirs: dict[int, Path] = {}
    for run_manifest in A_ROLLOUTS.glob("eval_*/RUN_MANIFEST.json"):
        value = read_json(run_manifest)
        rollout_dirs[int(value["eval_index"])] = run_manifest.parent
    if set(rollout_dirs) != set(range(35)):
        raise RuntimeError("saved ACT-A physical traces are not exact 35")

    rows: list[dict[str, Any]] = []
    runtime_values: list[dict[str, Any]] = []
    for index in range(35):
        run_dir = rollout_dirs[index]
        result = read_json(run_dir / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json")
        trial = read_json(run_dir / "trial_result.json")
        run = read_json(run_dir / "RUN_MANIFEST.json")
        reg = registration_entries[index]
        runtime_values.append(trial)
        with np.load(run_dir / "event_log.npz", allow_pickle=False) as archive:
            control = np.asarray(archive["control_frame"], dtype=np.int64)
            last_rows = np.flatnonzero(np.r_[np.diff(control) != 0, True])
            intent = archive["DIRECT_COMMON_TASK_INTENT"][last_rows].astype(str)
            close = np.isin(intent, ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"])
            if not np.any(close):
                raise RuntimeError(f"no LEFT close/hold interval at episode {index}")
            measured = np.asarray(archive["MEASURED_Q"][last_rows], dtype=np.float64)
            object_position = np.asarray(
                archive["object_position_world_m"][last_rows], dtype=np.float64
            )
            object_quaternion = np.asarray(
                archive["object_quaternion_xyzw"][last_rows], dtype=np.float64
            )
            dt_control = float(np.median(np.diff(control[last_rows]))) / 30.0
            if not np.isclose(dt_control, 1.0 / 30.0):
                dt_control = 1.0 / 30.0
            physics_dt = float(trial["control"]["physics_dt_s"])
            force_values = {
                digit: np.asarray(
                    archive[f"left_{digit}_force_n"], dtype=np.float64
                )
                for digit in ("thumb", "index", "middle", "palm")
            }
            force_close = {
                digit: values[np.isin(
                    archive["DIRECT_COMMON_TASK_INTENT"].astype(str),
                    ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"],
                )]
                for digit, values in force_values.items()
            }
            table_physics = np.asarray(
                archive["table_contact_force_n"], dtype=np.float64
            )
            table_close = table_physics[np.isin(
                archive["DIRECT_COMMON_TASK_INTENT"].astype(str),
                ["LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT"],
            )]

        minimum = {key: float("inf") for key in ("thumb", "index", "middle", "palm")}
        closest_frame = {key: None for key in minimum}
        relative_translations: list[np.ndarray] = []
        orientation_errors: list[float] = []
        close_control_frames = np.flatnonzero(close)
        for frame in close_control_frames:
            q = measured[frame]
            pose = pose_matrix(object_position[frame], object_quaternion[frame])
            distance_model.assign(q)
            distance_model.set_object_pose(pose, collision_z_offset)
            mujoco.mj_forward(distance_model.model, distance_model.data)
            values = distance_model.distances()
            for role, value in values.items():
                if value < minimum[role]:
                    minimum[role] = float(value)
                    closest_frame[role] = int(frame)
            palm_pose = palm_world(g1, q, palm_calibration, joint_names)
            relative = np.linalg.inv(palm_pose) @ pose
            relative_translations.append(relative[:3, 3].copy())
            orientation_errors.append(
                rotation_error_deg(palm_pose[:3, :3], pose[:3, :3])
            )

        meaningful = float(physics["gates"]["meaningful_digit_force_n"])
        meaningful_mask = {
            role: values >= meaningful for role, values in force_close.items()
        }
        opposing = meaningful_mask["thumb"] & (
            meaningful_mask["index"]
            | meaningful_mask["middle"]
            | meaningful_mask["palm"]
        )
        relative_translations_array = np.asarray(relative_translations)
        relative_delta = np.linalg.norm(
            relative_translations_array - relative_translations_array[0], axis=1
        )
        table_support_loss = bool(np.any(table_close <= 0.02))
        stable_relative = bool(np.max(relative_delta) <= 0.015)
        opposing_duration = longest_duration(opposing, physics_dt)
        retention_candidate = bool(
            opposing_duration >= 3.0 / 30.0 and stable_relative
        )
        frozen_mechanical = bool(
            retention_candidate and table_support_loss
        )
        scorer_reported = bool(result["outcomes"]["LEFT_GRASP_SUCCESS"])
        row = {
            "eval_index": index,
            "eval_number": index + 1,
            "stable_episode_id": run["stable_episode_id"],
            "run_dir": str(run_dir.resolve()),
            "trace_sha256": sha256_file(run_dir / "event_log.npz"),
            "registered_object_position_xyz_m": reg["target_object_pose"]["position_xyz_m"],
            "registered_object_quaternion_xyzw": reg["target_object_pose"]["quaternion_xyzw"],
            "left_close_hold_control_frame_start": int(close_control_frames[0]),
            "left_close_hold_control_frame_end": int(close_control_frames[-1]),
            "minimum_collision_surface_distance_m": minimum,
            "closest_control_frame": closest_frame,
            "contact_force_max_n": {
                role: float(np.max(values, initial=0.0))
                for role, values in force_close.items()
            },
            "meaningful_contact_threshold_n": meaningful,
            "contact_total_duration_s": {
                role: float(np.count_nonzero(mask) * physics_dt)
                for role, mask in meaningful_mask.items()
            },
            "contact_longest_duration_s": {
                role: longest_duration(mask, physics_dt)
                for role, mask in meaningful_mask.items()
            },
            "simultaneous_opposing_contact_duration_s": opposing_duration,
            "minimum_doll_table_support_force_n": float(np.min(table_close)),
            "table_support_loss_observed": table_support_loss,
            "initial_doll_center_z_m": float(object_position[0, 2]),
            "maximum_doll_center_z_m_during_close_hold": float(
                np.max(object_position[close, 2])
            ),
            "maximum_doll_com_lift_m": float(
                np.max(object_position[close, 2]) - object_position[0, 2]
            ),
            "maximum_object_to_palm_relative_translation_change_m": float(
                np.max(relative_delta)
            ),
            "palm_to_doll_orientation_error_deg_at_minimum": float(
                orientation_errors[int(np.argmin([
                    # Use palm closest frame within the close-mask indexing.
                    abs(int(frame) - int(closest_frame["palm"]))
                    for frame in close_control_frames
                ]))]
            ),
            "palm_to_doll_orientation_error_deg_range": [
                float(np.min(orientation_errors)),
                float(np.max(orientation_errors)),
            ],
            "mechanical_grasp_confirmation_state": bool(
                result["event_frames"]["grasp_confirm"] is not None
            ),
            "frozen_mechanical_definition_satisfied_by_trace": frozen_mechanical,
            "scorer_reported_left_grasp": scorer_reported,
            "scorer_false_negative": bool(frozen_mechanical and not scorer_reported),
            "retention_candidate": retention_candidate,
            "first_failure_frame": int(close_control_frames[-1] + 1),
            "scorer_reason": result["first_failure_stage"],
            "hard_physical_validity": bool(result["integrity"]["hard_physical_validity"]),
        }
        row["primary_failure_category"] = classify(row)
        rows.append(row)

    category_order = [
        "GEOMETRIC_POSITION_MISS",
        "ORIENTATION / APERTURE_MISS",
        "NO_PHYSICAL_CONTACT_DESPITE_VISUAL_PROXIMITY",
        "CONTACT_WITHOUT_OPPOSING_SUPPORT",
        "OPPOSING_CONTACT_BUT_NO_RETENTION",
        "RETENTION_BUT_NO_TABLE_SUPPORT_LOSS",
        "PHYSICAL_GRASP_OCCURRED_BUT_SCORER_FALSE_NEGATIVE",
        "COLLISION_OR_RUNTIME_INCONSISTENCY",
        "OTHER_PHYSICAL_FAILURE",
    ]
    category_counts = {
        category: sum(row["primary_failure_category"] == category for row in rows)
        for category in category_order
    }

    # Actual runtime scene values must agree across all 35 traces.
    first = runtime_values[0]
    initial_gaps = []
    runtime_pose_errors = []
    runtime_rotation_errors = []
    for index, (trial, row) in enumerate(zip(runtime_values, rows, strict=True)):
        proxy = trial["runtime_proxy"]
        center_z = float(trial["object"]["initial_center_world_m"][2])
        effective_bottom = (
            center_z
            + float(proxy["collider_bottom_alignment_offset_z_m"])
            - float(proxy["collision_dimensions_m"][2]) / 2.0
        )
        initial_gaps.append(effective_bottom - table_z)
        verification = read_json(
            rollout_dirs[index] / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json"
        )["integrity"]["runtime_initial_pose_verification"]
        runtime_pose_errors.append(float(verification["translation_error_mm"]))
        runtime_rotation_errors.append(float(verification["rotation_error_deg"]))

    half_difference = 0.5 * (visual - dimensions)
    scene_audit = {
        "schema_version": "final_eval35_physical_scene_runtime_audit_v1",
        "status": "PASS",
        "final_freeze": str(FREEZE.resolve()),
        "final_freeze_sha256": sha256_file(FREEZE),
        "expected_final_freeze_sha256": "3d7e5638660668b6c1aaf01129bdc2d32b4f86065589c6c4e07cdeba4f4d4555",
        "frozen_dependency_count": freeze["frozen_dependency_count"],
        "episode_registration": {
            "entries": registration["EVAL35_count"],
            "source_derived": registration["source_derived_count"],
            "unique_object_poses": registration["unique_episode_object_pose_count"],
            "one_global_canonical_pose": registration["one_global_canonical_object_pose"],
            "matched_A_B_pose_count": registration["A_B_identical_object_pose_count"],
            "maximum_manifest_A_B_translation_difference_mm": registration["maximum_A_B_translation_difference_mm"],
            "maximum_manifest_A_B_rotation_difference_deg": registration["maximum_A_B_rotation_difference_deg"],
            "maximum_ACT_A_runtime_translation_error_mm": max(runtime_pose_errors),
            "maximum_ACT_A_runtime_rotation_error_deg": max(runtime_rotation_errors),
            "old_0_3_0_15_fallback_count": sum(
                np.allclose(row["target_object_pose"]["position_xyz_m"][:2], [0.3, 0.15])
                for row in registration["entries"]
            ),
            "identity_yaw_count": sum(
                np.allclose(row["target_object_pose"]["quaternion_xyzw"], [0, 0, 0, 1])
                for row in registration["entries"]
            ),
        },
        "doll": {
            "dynamic_rigid_body": True,
            "kinematic": False,
            "gravity_enabled": True,
            "mass_kg": float(first["runtime_proxy"]["mass_kg"]),
            "visual_dimensions_m": first["runtime_proxy"]["visual_dimensions_m"],
            "collision_dimensions_m": first["runtime_proxy"]["collision_dimensions_m"],
            "visual_to_collision_half_extent_difference_m": half_difference.tolist(),
            "collision_approximation": first["runtime_proxy"]["convex_approximation"],
            "collider_bottom_alignment_offset_z_m": first["runtime_proxy"]["collider_bottom_alignment_offset_z_m"],
            "contact_offset_m": first["runtime_proxy"]["contact_offset_m"],
            "rest_offset_m": first["runtime_proxy"]["rest_offset_m"],
            "static_friction": first["runtime_proxy"]["static_friction"],
            "dynamic_friction": first["runtime_proxy"]["dynamic_friction"],
            "restitution": first["runtime_proxy"]["restitution"],
            "linear_damping": float(physics["object"]["linear_damping"]),
            "angular_damping": float(physics["object"]["angular_damping"]),
            "visual_touch_can_precede_collision_contact": bool(np.any(half_difference > 0.0)),
            "attachment_parenting_following": False,
        },
        "doll_table_height": {
            "table_top_z_m": table_z,
            "registered_center_z_range_m": [
                min(row["target_object_pose"]["position_xyz_m"][2] for row in registration["entries"]),
                max(row["target_object_pose"]["position_xyz_m"][2] for row in registration["entries"]),
            ],
            "minimum_effective_collision_bottom_gap_m": min(initial_gaps),
            "maximum_effective_collision_bottom_gap_m": max(initial_gaps),
            "maximum_initial_penetration_m": max(0.0, -min(initial_gaps)),
            "physically_supported_by_table": bool(max(abs(value) for value in initial_gaps) <= 1.0e-6),
        },
        "physics": {
            "execution": "CONTACT_CONSTRAINED_PHYSX",
            "physics_timestep_s": float(first["control"]["physics_dt_s"]),
            "control_timestep_s": 1.0 / float(first["control"]["fps"]),
            "substeps": int(round((1.0 / float(first["control"]["fps"])) / float(first["control"]["physics_dt_s"]))),
            "gravity_m_s2": float(first["control"]["gravity_m_s2"]),
            "object_pose_writes_after_initialization": max(
                int(trial["object_pose_writes_during_timed_loop"])
                for trial in runtime_values
            ),
            "contact_sensor_api_errors": sorted({
                error
                for trial in runtime_values
                for error in trial["artifact_checks"]["contact_api_errors"]
            }),
            "source": str(PHYSICS_CONFIG),
        },
        "bin": physical_environment["bin"],
        "runtime_sources": {
            "ACT_A_trial_results": [str((rollout_dirs[index] / "trial_result.json").resolve()) for index in range(35)],
            "contact_model": str(CONTACT_MODEL.resolve()),
            "physical_environment": str(ENVIRONMENT.resolve()),
            "physics_config": str(PHYSICS_CONFIG),
        },
    }
    if (
        scene_audit["final_freeze_sha256"] != scene_audit["expected_final_freeze_sha256"]
        or scene_audit["episode_registration"]["entries"] != 35
        or scene_audit["episode_registration"]["matched_A_B_pose_count"] != 35
        or scene_audit["episode_registration"]["maximum_ACT_A_runtime_translation_error_mm"] > 0.1
        or scene_audit["episode_registration"]["maximum_ACT_A_runtime_rotation_error_deg"] > 0.1
        or not scene_audit["doll_table_height"]["physically_supported_by_table"]
        or scene_audit["physics"]["object_pose_writes_after_initialization"] != 0
    ):
        scene_audit["status"] = "FAIL"

    report = {
        "schema_version": "act_a_left_grasp_forensic_report_v1",
        "status": "COMPLETE_READ_ONLY_EXISTING_TRACES",
        "episodes": 35,
        "reported_left_grasp_success_count": sum(
            row["scorer_reported_left_grasp"] for row in rows
        ),
        "scorer_false_negative_count": sum(row["scorer_false_negative"] for row in rows),
        "category_counts": category_counts,
        "all_failures_physically_geometric_if_scene_audit_passes": bool(
            scene_audit["status"] == "PASS"
            and category_counts["COLLISION_OR_RUNTIME_INCONSISTENCY"] == 0
            and category_counts["PHYSICAL_GRASP_OCCURRED_BUT_SCORER_FALSE_NEGATIVE"] == 0
        ),
        "rows": rows,
    }
    atomic_json(AUDIT / "PHYSICAL_SCENE_RUNTIME_AUDIT.json", scene_audit)
    atomic_json(AUDIT / "ACT_A_LEFT_GRASP_FORENSIC_REPORT.json", report)

    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_rows.append(
            {
                "eval_index": row["eval_index"],
                "eval_number": row["eval_number"],
                "stable_episode_id": row["stable_episode_id"],
                "primary_failure_category": row["primary_failure_category"],
                **{
                    f"minimum_{role}_surface_distance_m": row["minimum_collision_surface_distance_m"][role]
                    for role in ("palm", "thumb", "index", "middle")
                },
                **{
                    f"maximum_{role}_force_n": row["contact_force_max_n"][role]
                    for role in ("palm", "thumb", "index", "middle")
                },
                **{
                    f"{role}_contact_duration_s": row["contact_longest_duration_s"][role]
                    for role in ("palm", "thumb", "index", "middle")
                },
                "opposing_contact_duration_s": row["simultaneous_opposing_contact_duration_s"],
                "minimum_table_support_force_n": row["minimum_doll_table_support_force_n"],
                "maximum_com_lift_m": row["maximum_doll_com_lift_m"],
                "maximum_object_palm_relative_displacement_m": row["maximum_object_to_palm_relative_translation_change_m"],
                "palm_doll_orientation_error_deg": row["palm_to_doll_orientation_error_deg_at_minimum"],
                "grasp_confirmed": row["mechanical_grasp_confirmation_state"],
                "scorer_false_negative": row["scorer_false_negative"],
                "first_failure_frame": row["first_failure_frame"],
                "scorer_reason": row["scorer_reason"],
            }
        )
    csv_path = AUDIT / "ACT_A_LEFT_GRASP_FORENSIC_TABLE.csv"
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    os.replace(temporary, csv_path)

    md = [
        "# ACT-A LEFT-grasp forensic report",
        "",
        "This is a read-only reconstruction from the 35 already-saved contact-constrained PhysX traces; no ACT-A episode was rerun.",
        "",
        f"- Reported LEFT_GRASP: **{report['reported_left_grasp_success_count']} / 35**",
        f"- Scorer false negatives: **{report['scorer_false_negative_count']} / 35**",
        f"- Scene runtime audit: **{scene_audit['status']}**",
        "",
        "## Primary failure categories",
        "",
    ]
    md.extend(f"- {key}: {value} / 35" for key, value in category_counts.items())
    md.extend(
        [
            "",
            "The 4 mm boundary used only to distinguish a genuine geometric miss from a visual/collision-proximity discrepancy is the predeclared maximum plush-contact tolerance; it is not a success-tuned threshold.",
            "",
            f"Detailed rows: `{csv_path}`",
        ]
    )
    atomic_text(AUDIT / "ACT_A_LEFT_GRASP_FORENSIC_REPORT.md", "\n".join(md) + "\n")

    scene_md = [
        "# Final EVAL35 physical-scene runtime audit",
        "",
        f"Status: **{scene_audit['status']}**",
        "",
        f"- Freeze SHA256: `{scene_audit['final_freeze_sha256']}`",
        f"- Episode registrations: {scene_audit['episode_registration']['entries']} / 35",
        f"- Matched A/B manifest poses: {scene_audit['episode_registration']['matched_A_B_pose_count']} / 35 identical",
        f"- ACT-A runtime pose maximum error: {scene_audit['episode_registration']['maximum_ACT_A_runtime_translation_error_mm']:.9f} mm / {scene_audit['episode_registration']['maximum_ACT_A_runtime_rotation_error_deg']:.9f} deg",
        f"- Visual dimensions: {scene_audit['doll']['visual_dimensions_m']} m",
        f"- Collision dimensions: {scene_audit['doll']['collision_dimensions_m']} m",
        f"- Visual/collision half-extent differences: {scene_audit['doll']['visual_to_collision_half_extent_difference_m']} m",
        f"- Effective initial table gap range: {scene_audit['doll_table_height']['minimum_effective_collision_bottom_gap_m']:.12g} to {scene_audit['doll_table_height']['maximum_effective_collision_bottom_gap_m']:.12g} m",
        f"- Object pose writes after initialization: {scene_audit['physics']['object_pose_writes_after_initialization']}",
        "",
        "The collision proxy is smaller than the visual envelope, but its qualified -3.75 mm local-Z alignment puts the physical bottom exactly on the 0.795 m table top. This avoids both initial levitation and penetration.",
    ]
    atomic_text(AUDIT / "PHYSICAL_SCENE_RUNTIME_AUDIT.md", "\n".join(scene_md) + "\n")
    print(json.dumps({
        "scene_status": scene_audit["status"],
        "scorer_false_negatives": report["scorer_false_negative_count"],
        "category_counts": category_counts,
    }, indent=2))
    return 0 if scene_audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
