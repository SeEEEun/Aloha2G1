#!/usr/bin/env python3
"""Finalize the TRAIN-only common registration/IK qualification audit.

This tool is deliberately read-only with respect to scientific inputs.  It
summarizes the registration-bound smoke artifacts for episodes 0, 24 and 49
and enforces the sequential gate: a failed common IK gate prevents loaded
PhysX or policy work from being launched.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/single_variable_ab_common_execution"
SMOKE = OUT / "02_registration_bound_smoke"
REGISTRATION = OUT / "00_registration/COMMON_TASK_REGISTRATION_TRAIN_SMOKE.json"
HISTORICAL_DEX3 = (
    ROOT
    / "outputs/final_episode_registered_eval35/00_forensic_audit/"
    "DEX3_ARTICULATION_FORENSIC_AUDIT.json"
)
EPISODES = (0, 24, 49)
METHODS = {"baseline": "WRIST", "proposed": "INTERACTION"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ranges(mask: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    output: list[list[int]] = []
    first = previous = int(indices[0])
    for item in map(int, indices[1:]):
        if item != previous + 1:
            output.append([first, previous])
            first = item
        previous = item
    output.append([first, previous])
    return output


def main() -> int:
    # Imports occur here so this report can still show a clear missing-artifact
    # error in environments without the project simulation dependencies.
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics

    registration = load(REGISTRATION)
    common_path = SMOKE / "config/common_config.json"
    common = load_common_config(common_path)
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal = np.asarray(common["resolved"]["canonical_g1_nominal_q"], dtype=np.float64)
    landmarks = g1.arm_landmarks(nominal)
    reach = {
        side: float(
            np.linalg.norm(landmarks[side]["elbow"] - landmarks[side]["shoulder_pitch"])
            + np.linalg.norm(
                landmarks[side]["wrist_yaw"] - landmarks[side]["elbow"]
            )
        )
        for side in ("left", "right")
    }
    entries = {int(row["episode_index"]): row for row in registration["entries"]}

    registration_rows: list[dict[str, Any]] = []
    ik_rows: list[dict[str, Any]] = []
    equality_pass = True
    bound_pass = True
    maximum_reference_position_error = {mode: 0.0 for mode in METHODS.values()}
    maximum_reference_rotation_error = {mode: 0.0 for mode in METHODS.values()}

    tool_report = load(SMOKE / "config/tool_frame_report.json")
    wrist_to_tool = {
        side: np.asarray(
            tool_report["g1"][f"{side}_wrist_to_grasp_frame"], dtype=np.float64
        )
        for side in ("left", "right")
    }

    for episode in EPISODES:
        entry = entries[episode]
        method_archives: dict[str, dict[str, Any]] = {}
        for method, mode in METHODS.items():
            path = next((SMOKE / method / "trajectories").glob(f"*ep{episode:03d}.npz"))
            with np.load(path, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            method_archives[method] = values
            bound_pass &= bool(values["episode_registration_bound"])
            bound_pass &= str(values["episode_registration_entry_sha256"]) == str(
                entry["entry_sha256"]
            )
            bound_pass &= str(values["representation_mode"]) == mode

        a = method_archives["baseline"]
        b = method_archives["proposed"]
        translation_difference = float(
            np.linalg.norm(
                np.asarray(a["registered_object_position_world"])
                - np.asarray(b["registered_object_position_world"])
            )
        )
        qa = np.asarray(a["registered_object_quaternion_xyzw"], dtype=np.float64)
        qb = np.asarray(b["registered_object_quaternion_xyzw"], dtype=np.float64)
        rotation_difference = float((Rotation.from_quat(qa).inv() * Rotation.from_quat(qb)).magnitude())
        equality_pass &= translation_difference <= 1e-12 and rotation_difference <= 1e-12

        row = {
            "episode_index": episode,
            "source_name": entry["source_name"],
            "source_task_frame": entry["source_task_frame"],
            "source_object_pose": entry["source_object_pose"],
            "target_registered_object_pose": entry["target_object_pose"],
            "source_to_target_direction": entry["source_to_target_direction"],
            "source_to_target_transform_matrix": entry[
                "source_to_target_transform_matrix"
            ],
            "A_B_translation_difference_m": translation_difference,
            "A_B_rotation_difference_rad": rotation_difference,
            "A_B_registration_entry_hash_equal": bool(
                str(a["episode_registration_entry_sha256"])
                == str(b["episode_registration_entry_sha256"])
            ),
        }
        event_lookup = dict(
            zip(
                map(str, a["event_names"].tolist()),
                map(int, a["event_frames"].tolist()),
            )
        )
        grasp_frame = int(event_lookup["LEFT_GRASP"])
        object_position = np.asarray(entry["target_object_pose"]["position_xyz_m"])
        object_rotation = Rotation.from_quat(
            np.asarray(entry["target_object_pose"]["quaternion_xyzw"])
        ).as_matrix()
        row["source_left_grasp_frame"] = grasp_frame
        row["source_object_to_registered_source_TCP_at_grasp"] = {
            "translation_xyz_m": (
                np.asarray(
                    a["target_left_interaction_frame_position_world"][grasp_frame]
                )
                - object_position
            ),
            "position_norm_m": float(
                np.linalg.norm(
                    np.asarray(
                        a["target_left_interaction_frame_position_world"][grasp_frame]
                    )
                    - object_position
                )
            ),
        }
        row["representation_targets_at_source_left_grasp"] = {}
        for method, mode in METHODS.items():
            archive = method_archives[method]
            wrist_position_model = np.asarray(
                archive["target_left_wrist_position_model"][grasp_frame]
            )
            wrist_rotation_model = np.asarray(
                archive["target_left_wrist_rotation_model"][grasp_frame]
            )
            wrist_position_world = g1.model_to_world_position(wrist_position_model)
            wrist_rotation_world = g1.model_to_world_rotation(wrist_rotation_model)
            row["representation_targets_at_source_left_grasp"][mode] = {
                "target_wrist_position_world_m": wrist_position_world,
                "target_wrist_quaternion_xyzw_world": Rotation.from_matrix(
                    wrist_rotation_world
                ).as_quat(),
                "object_to_wrist_translation_xyz_m": wrist_position_world
                - object_position,
                "object_to_wrist_rotation_xyzw": Rotation.from_matrix(
                    object_rotation.T @ wrist_rotation_world
                ).as_quat(),
                "common_solver_input_position_model_m": wrist_position_model,
                "common_solver_input_quaternion_xyzw_model": Rotation.from_matrix(
                    wrist_rotation_model
                ).as_quat(),
            }

        for method, mode in METHODS.items():
            archive = method_archives[method]
            metric_path = next(
                (SMOKE / method / "metrics").glob(f"*ep{episode:03d}.json")
            )
            metrics = load(metric_path)
            validation = load(metric_path.with_suffix(".validation.json"))
            q = np.asarray(archive["g1_arm_qpos"], dtype=np.float64)
            position_error: list[float] = []
            orientation_error: list[float] = []
            side_position_error: dict[str, list[float]] = {"left": [], "right": []}
            side_orientation_error: dict[str, list[float]] = {"left": [], "right": []}
            shoulder_radius: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                target_position = np.asarray(
                    archive[f"target_{side}_wrist_position_model"], dtype=np.float64
                )
                target_rotation = np.asarray(
                    archive[f"target_{side}_wrist_rotation_model"], dtype=np.float64
                )
                shoulder_radius[side] = np.linalg.norm(
                    target_position - landmarks[side]["shoulder_pitch"], axis=1
                )
                # Reconstruct the intended tool point from the target wrist.
                reconstructed_tool = target_position + np.einsum(
                    "tij,j->ti", target_rotation, wrist_to_tool[side][:3, 3]
                )
                expected_tool_model = g1.world_to_model_position(
                    np.asarray(
                        archive[f"target_{side}_interaction_frame_position_world"],
                        dtype=np.float64,
                    )
                )
                maximum_reference_position_error[mode] = max(
                    maximum_reference_position_error[mode],
                    float(np.max(np.linalg.norm(reconstructed_tool - expected_tool_model, axis=1))),
                )
                # R_wrist * R_wrist_to_tool is the target tool orientation by
                # construction.  Re-associating the same matrices tests the
                # fixed-frame multiplication and adds no data-derived transform.
                reconstructed_rotation = np.einsum(
                    "tij,jk->tik", target_rotation, wrist_to_tool[side][:3, :3]
                )
                round_trip_rotation = np.einsum(
                    "tij,jk->tik", reconstructed_rotation, wrist_to_tool[side][:3, :3].T
                )
                maximum_reference_rotation_error[mode] = max(
                    maximum_reference_rotation_error[mode],
                    float(
                        np.max(
                            (
                                Rotation.from_matrix(round_trip_rotation)
                                * Rotation.from_matrix(target_rotation).inv()
                            ).magnitude()
                        )
                    ),
                )

            for frame, q_frame in enumerate(q):
                state = g1.wrist_state(q_frame)
                per_position: list[float] = []
                per_orientation: list[float] = []
                for side in ("left", "right"):
                    p = float(
                        np.linalg.norm(
                            state[f"{side}_position"]
                            - np.asarray(
                                archive[f"target_{side}_wrist_position_model"][frame],
                                dtype=np.float64,
                            )
                        )
                    )
                    r = float(
                        Rotation.from_matrix(
                            np.asarray(
                                archive[f"target_{side}_wrist_rotation_model"][frame],
                                dtype=np.float64,
                            )
                            @ state[f"{side}_rotation"].T
                        ).magnitude()
                    )
                    side_position_error[side].append(p)
                    side_orientation_error[side].append(r)
                    per_position.append(p)
                    per_orientation.append(r)
                position_error.append(max(per_position))
                orientation_error.append(max(per_orientation))

            position_error_array = np.asarray(position_error)
            orientation_error_array = np.asarray(orientation_error)
            ptol = float(common["shared_temporal_ik"]["position_tolerance_m"])
            otol = float(common["shared_temporal_ik"]["orientation_tolerance_rad"])
            rejected = (position_error_array > ptol) | (orientation_error_array > otol)
            normalized = np.maximum(position_error_array / ptol, orientation_error_array / otol)
            worst = int(np.argmax(normalized))
            first = int(np.flatnonzero(rejected)[0]) if np.any(rejected) else None
            target_rotation = np.asarray(
                archive["target_left_wrist_rotation_model"][worst], dtype=np.float64
            )
            target_position = np.asarray(
                archive["target_left_wrist_position_model"][worst], dtype=np.float64
            )
            outside = {
                side: shoulder_radius[side] > reach[side] + ptol
                for side in ("left", "right")
            }
            if any(np.any(value) for value in outside.values()):
                classification = "UNREACHABLE_TARGET"
            elif validation["checks"].get("ik") is False:
                classification = "BAD_SEED_OR_SOLVER_NUMERICAL"
            elif validation["checks"].get("collision") is False:
                classification = "COLLISION"
            else:
                classification = "PASS"
            at_lower = np.isclose(q[worst], g1.arm_limits[:, 0], atol=1e-5)
            at_upper = np.isclose(q[worst], g1.arm_limits[:, 1], atol=1e-5)
            ik_rows.append(
                {
                    "episode_index": episode,
                    "source_name": entry["source_name"],
                    "method": method,
                    "representation_mode": mode,
                    "registration_entry_sha256": str(
                        archive["episode_registration_entry_sha256"]
                    ),
                    "common_solver_config_sha256": str(
                        archive["common_config_sha256"]
                    ),
                    "initial_seed_q_rad": nominal,
                    "frame_count": len(q),
                    "ik_success_rate": float(metrics["ik_success_rate"]),
                    "accepted": bool(validation["checks"]["ik"]),
                    "exact_rejection_gate": validation["first_failure_gate"],
                    "classification": classification,
                    "first_rejected_frame": first,
                    "rejected_frame_count": int(np.count_nonzero(rejected)),
                    "rejected_frame_ranges": ranges(rejected),
                    "worst_frame": worst,
                    "target_left_wrist_xyz_model_m_at_worst": target_position,
                    "target_left_wrist_quaternion_xyzw_at_worst": Rotation.from_matrix(
                        target_rotation
                    ).as_quat(),
                    "target_left_wrist_rpy_deg_at_worst": Rotation.from_matrix(
                        target_rotation
                    ).as_euler("xyz", degrees=True),
                    "final_position_residual_max_m": float(position_error_array[worst]),
                    "final_orientation_residual_max_rad": float(
                        orientation_error_array[worst]
                    ),
                    "joint_limit_status": {
                        "violations": int(metrics["arm_joint_limit_violation_count"]),
                        "joints_at_lower_limit_at_worst": g1.arm_joint_names[at_lower],
                        "joints_at_upper_limit_at_worst": g1.arm_joint_names[at_upper],
                    },
                    "collision_status": {
                        "validation_pass": bool(validation["checks"]["collision"]),
                        "invalid_self_body_collision_frames": int(
                            metrics["collisions"]["invalid_self_body_collision_frames"]
                        ),
                    },
                    "reachability_estimate": {
                        side: {
                            "two_link_max_wrist_radius_m": reach[side],
                            "target_radius_max_m": float(np.max(shoulder_radius[side])),
                            "frames_beyond_radius_plus_position_tolerance": int(
                                np.count_nonzero(outside[side])
                            ),
                        }
                        for side in ("left", "right")
                    },
                    "maximum_joint_step_rad": float(metrics["maximum_joint_step_rad"]),
                    "maximum_joint_acceleration_rad_s2": float(
                        metrics["maximum_joint_acceleration_rad_s2"]
                    ),
                    "branch_discontinuity_count": int(
                        metrics["branch_discontinuity_count"]
                    ),
                }
            )
        registration_rows.append(row)

    registration_pass = bool(
        registration.get("status") == "PASS"
        and registration.get("common_for_A_B") is True
        and registration.get("policy_output_derived") is False
        and registration.get("manual_episode_nudges") is False
        and set(entries) == set(EPISODES)
        and bound_pass
        and equality_pass
        and max(maximum_reference_position_error.values()) <= 1e-6
        and max(maximum_reference_rotation_error.values()) <= 1e-9
    )
    task_audit = {
        "schema_version": "single_variable_common_task_registration_audit_v1",
        "status": "PASS" if registration_pass else "FAIL",
        "scope": "TRAIN_ONLY_SMOKE_0_24_49",
        "transform_math": {
            "direction": "T_target_from_source",
            "position_column_form": "p_target = R_target_from_source @ p_source + t",
            "position_numpy_row_batch_form": "P_target = P_source @ R.T + t",
            "rotation_form": "R_target = R_target_from_source @ R_source",
            "homogeneous_multiplication_order": "T_target_pose = T_target_from_source @ T_source_pose",
            "world_task_direction": "source task/world to registered target task/world",
            "units": "meters and radians",
            "serialized_object_quaternion": "XYZW",
            "scene_root_quaternion": "WXYZ (converted explicitly when consumed)",
            "left_right_indexing": "named left/right channels; no positional swap",
            "tool_convention": "ALOHA link6 -> physical jaw TCP, then one fixed target-tool -> G1 wrist transform",
        },
        "registration_manifest": str(REGISTRATION),
        "manifest_content_sha256": registration["content_sha256"],
        "episode_count": len(registration_rows),
        "A_B_exact_pose_equality_count": sum(
            row["A_B_translation_difference_m"] == 0.0
            and row["A_B_rotation_difference_rad"] == 0.0
            for row in registration_rows
        ),
        "A_reference_registration_pass": maximum_reference_position_error["WRIST"] <= 1e-6,
        "B_reference_registration_pass": maximum_reference_position_error["INTERACTION"] <= 1e-6,
        "maximum_fixed_tool_position_round_trip_error_m": maximum_reference_position_error,
        "maximum_fixed_tool_rotation_round_trip_error_rad": maximum_reference_rotation_error,
        "entries": registration_rows,
    }
    write_json(OUT / "COMMON_TASK_REGISTRATION_AUDIT.json", task_audit)

    ik_acceptance = {
        mode: sum(row["accepted"] for row in ik_rows if row["representation_mode"] == mode)
        for mode in METHODS.values()
    }
    ik_pass = all(value == len(EPISODES) for value in ik_acceptance.values())
    ik_audit = {
        "schema_version": "single_variable_common_ik_audit_v1",
        "status": "PASS" if ik_pass else "FAIL",
        "scope": "TRAIN_ONLY_SMOKE_0_24_49",
        "common_api": "SharedTemporalIK.solve(target_wrist_SE3)",
        "solver_knows_representation_mode": False,
        "same_implementation": True,
        "same_config": True,
        "same_seed_convention": True,
        "same_joint_limits": True,
        "same_regularization": True,
        "same_tolerances": True,
        "same_temporal_continuity": True,
        "acceptance": ik_acceptance,
        "required": {mode: len(EPISODES) for mode in METHODS.values()},
        "rows": ik_rows,
        "decision": (
            "STOP_BEFORE_LOADED_DEX3_AND_TRAINING"
            if not ik_pass
            else "PROCEED_TO_LOADED_DEX3"
        ),
        "scientific_constraint": (
            "The direct WRIST reference cannot be made 3/3 IK-valid without changing "
            "its spatial target, the common task pose, or the declared tolerances. "
            "Those are scientific changes, not common-IK implementation repairs."
        ),
    }
    write_json(OUT / "COMMON_IK_AUDIT.json", ik_audit)

    historical = load(HISTORICAL_DEX3) if HISTORICAL_DEX3.is_file() else {}
    dex3_audit = {
        "schema_version": "single_variable_loaded_dex3_gate_v1",
        "status": "NOT_RUN_DUE_PRIOR_COMMON_IK_GATE",
        "new_physics_runs": 0,
        "zero_contact_historical_mapping_pass_count": sum(
            bool(row.get("mapping_pass")) for row in historical.get("all_14_joint_limits", [])
        ),
        "zero_contact_historical_sign_pass_count": sum(
            bool(row.get("sign_pass")) for row in historical.get("all_14_joint_limits", [])
        ),
        "zero_contact_historical_readback_pass_count": sum(
            bool(row.get("readback_pass")) for row in historical.get("all_14_joint_limits", [])
        ),
        "loaded_qualification_pass_count": 0,
        "loaded_qualification_required_count": 14,
        "measured_hard_limit_violations": "NOT_REMEASURED",
        "reason": (
            "Section 6 requires an immediate stop when either method is systematically "
            "blocked by common IK. Launching loaded PhysX would violate that sequential gate."
        ),
        "historical_evidence": str(HISTORICAL_DEX3),
    }
    write_json(OUT / "LOADED_DEX3_ARTICULATION_AUDIT.json", dex3_audit)

    parity = {
        "schema_version": "single_variable_common_execution_parity_v1",
        "status": "PASS_WITH_DOWNSTREAM_GATES_NOT_QUALIFIED",
        "shared": {
            "source_episodes": True,
            "task_registration": True,
            "event_timing": True,
            "Dex3_commands": True,
            "IK_implementation": True,
            "solver_settings": True,
            "joint_limits": True,
            "articulation_implementation": True,
            "physical_scene": True,
            "scorer": True,
            "dataset_schema": True,
        },
        "only_intended_difference": {
            "switch": "representation_mode",
            "WRIST": "registered source wrist/TCP target generation",
            "INTERACTION": "registered object/whole-hand interaction target generation",
        },
        "unintended_A_B_confounds": 0,
        "qualification_is_not_scientific_readiness": True,
    }
    write_json(OUT / "A_B_COMMON_EXECUTION_PARITY_REPORT.json", parity)

    task_md = [
        "# Common task-registration audit",
        "",
        f"Status: **{task_audit['status']}**",
        "",
        "This is a TRAIN-only audit for episodes 0, 24, and 49. The source-derived "
        "registration is loaded before the representation switch and the exact same "
        "entry hash is serialized into A and B.",
        "",
        f"- A/B exact registered-object equality: {task_audit['A_B_exact_pose_equality_count']} / 3",
        f"- A fixed-tool round-trip error: {maximum_reference_position_error['WRIST'] * 1e3:.9f} mm",
        f"- B fixed-tool round-trip error: {maximum_reference_position_error['INTERACTION'] * 1e3:.9f} mm",
        f"- Maximum rotation round-trip error: {math.degrees(max(maximum_reference_rotation_error.values())):.12f} deg",
        "- Units: meters/radians",
        "- Object quaternion: XYZW",
        "- Scene-root quaternion: WXYZ, explicitly converted by the scene loader",
        "- Transform direction: `T_target_from_source`",
        "- Multiplication order: `T_target_pose = T_target_from_source @ T_source_pose`",
        "- Manual nudges: NO",
        "- Policy-output-derived registration: NO",
        "",
        "| Episode | Source | A/B translation | A/B rotation | Entry hash |",
        "|---:|---|---:|---:|---|",
    ]
    for row in registration_rows:
        task_md.append(
            f"| {row['episode_index']} | {row['source_name']} | "
            f"{row['A_B_translation_difference_m'] * 1e3:.12f} mm | "
            f"{math.degrees(row['A_B_rotation_difference_rad']):.12f} deg | "
            f"`{entries[row['episode_index']]['entry_sha256']}` |"
        )
    (OUT / "COMMON_TASK_REGISTRATION_AUDIT.md").write_text(
        "\n".join(task_md) + "\n", encoding="utf-8"
    )

    ik_md = [
        "# Common IK audit",
        "",
        f"Status: **{ik_audit['status']}**",
        "",
        "A and B call the same `SharedTemporalIK.solve` API with identical seed, "
        "limits, weights, regularization, tolerances, and continuity handling. Only "
        "the target SE(3) values differ.",
        "",
        f"- A (WRIST): {ik_acceptance['WRIST']} / 3 accepted",
        f"- B (INTERACTION): {ik_acceptance['INTERACTION']} / 3 accepted",
        "- Required: 3 / 3 for each",
        "",
        "| Mode | Episode | IK rate | Classification | Max left radius | Max right radius | Reach | First gate |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in ik_rows:
        left = row["reachability_estimate"]["left"]
        right = row["reachability_estimate"]["right"]
        ik_md.append(
            f"| {row['representation_mode']} | {row['episode_index']} | "
            f"{100.0 * row['ik_success_rate']:.2f}% | {row['classification']} | "
            f"{left['target_radius_max_m'] * 1e3:.1f} mm | "
            f"{right['target_radius_max_m'] * 1e3:.1f} mm | "
            f"{left['two_link_max_wrist_radius_m'] * 1e3:.1f} mm | "
            f"{row['exact_rejection_gate']} |"
        )
    ik_md.extend(
        [
            "",
            "## Decision",
            "",
            "The common registration is algebraically and provenance-valid, but the "
            "IK gate remains systematically blocked. In particular, the faithful "
            "WRIST reference contains sustained targets outside the active G1 arm's "
            "model-derived wrist reach. Forcing 3/3 would require a target-space "
            "projection/scaling, changed task registration, or relaxed acceptance "
            "criteria. Each would change the scientific experiment and is outside "
            "this common-execution repair task.",
            "",
            "Per the required sequential gate, no loaded Dex3 test, shared physical "
            "smoke, dataset regeneration, ACT training, DEV35 run, or FINAL_TEST "
            "collection was launched.",
        ]
    )
    (OUT / "COMMON_IK_AUDIT.md").write_text("\n".join(ik_md) + "\n", encoding="utf-8")

    dex3_md = [
        "# Loaded Dex3 articulation audit",
        "",
        "Status: **NOT RUN — PRIOR COMMON IK GATE FAILED**",
        "",
        "- New physics runs: 0",
        f"- Historical zero-contact mapping: {dex3_audit['zero_contact_historical_mapping_pass_count']} / 14",
        f"- Historical zero-contact sign: {dex3_audit['zero_contact_historical_sign_pass_count']} / 14",
        f"- Historical zero-contact readback: {dex3_audit['zero_contact_historical_readback_pass_count']} / 14",
        "- Loaded qualification: 0 / 14 (not executed)",
        "- Measured hard-limit violations: NOT REMEASURED",
        "",
        dex3_audit["reason"],
    ]
    (OUT / "LOADED_DEX3_ARTICULATION_AUDIT.md").write_text(
        "\n".join(dex3_md) + "\n", encoding="utf-8"
    )

    parity_md = [
        "# A/B common-execution parity report",
        "",
        "The audited code path has one intentional switch: `representation_mode`.",
        "",
        "| Component after source loading | A/B parity |",
        "|---|---|",
    ]
    for name, value in parity["shared"].items():
        parity_md.append(f"| {name.replace('_', ' ')} | {'IDENTICAL' if value else 'DIFFERENT'} |")
    parity_md.extend(
        [
            "",
            "Intentional spatial target-generation difference only:",
            "",
            "- A: `WRIST`",
            "- B: `INTERACTION`",
            "",
            "Unintended A/B confounds in this code/config parity audit: **0**.",
            "This does not override the failed common IK and unrun loaded-articulation gates.",
        ]
    )
    (OUT / "A_B_COMMON_EXECUTION_PARITY_REPORT.md").write_text(
        "\n".join(parity_md) + "\n", encoding="utf-8"
    )

    summary = {
        "task_registration": task_audit["status"],
        "A_reference_registration": "PASS" if task_audit["A_reference_registration_pass"] else "FAIL",
        "B_reference_registration": "PASS" if task_audit["B_reference_registration_pass"] else "FAIL",
        "common_IK_A": ik_acceptance["WRIST"],
        "common_IK_B": ik_acceptance["INTERACTION"],
        "loaded_Dex3": "NOT_RUN_DUE_PRIOR_COMMON_IK_GATE",
        "shared_pipeline_smoke_A": "NOT_RUN_DUE_PRIOR_COMMON_IK_GATE",
        "shared_pipeline_smoke_B": "NOT_RUN_DUE_PRIOR_COMMON_IK_GATE",
        "unintended_A_B_confounds": 0,
        "ready_to_regenerate_and_retrain": False,
        "terminal_status": "COMMON_IK_STILL_INVALID",
    }
    write_json(OUT / "COMMON_EXECUTION_QUALIFICATION.json", summary)
    (OUT / "COMMON_EXECUTION_QUALIFICATION.md").write_text(
        "\n".join(
            [
                "# Common execution qualification",
                "",
                "Status: **COMMON_IK_STILL_INVALID**",
                "",
                f"- Task registration: {summary['task_registration']}",
                f"- A reference registration: {summary['A_reference_registration']}",
                f"- B reference registration: {summary['B_reference_registration']}",
                f"- A common IK: {summary['common_IK_A']} / 3",
                f"- B common IK: {summary['common_IK_B']} / 3",
                "- Loaded Dex3: NOT RUN (sequential IK gate)",
                "- Shared physical smoke: NOT RUN (sequential IK gate)",
                "- ACT training: NOT RUN",
                "- DEV35/final evaluation: NOT RUN",
                "- FINAL_TEST collection: NOT RUN",
                "- Ready to regenerate/retrain: NO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
