#!/usr/bin/env python3
"""Apply one frozen Dex3 grasp realization after the common whole-hand gate.

The arm/wrist trajectory is never changed.  A common OPEN/PRESHAPE posture is a
target-embodiment realization; FULL CLOSE activates only if the representation-
neutral object-relative whole-hand gate is reached.  The same code and constants
are applied to Fair-A and Proposed-B.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from audit_physical_evaluator_task_frame_alignment import (
    FrozenProxySurface,
    PALM_CONFIG,
    PHYSICS_CONFIG,
    geometry_state,
    load_common_config,
    load_scene,
    read_json,
    G1Kinematics,
)


ROOT = Path("/home/jbnu/aloha_g1_dataset")
BASE = ROOT / "outputs/final_contact_constrained_eval/09_reference_alignment_preflight"
MANIFEST = BASE / "REFERENCE_COMMAND_MANIFEST.json"
REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
GATE = ROOT / "configs/contact_eval_common_whole_hand_readiness_gate_v1.json"
GRASP = ROOT / "configs/contact_eval_common_dex3_grasp_realization_v1.json"
WRIST_PRIMITIVE = ROOT / "configs/contact_eval_common_object_relative_left_grasp_v1.json"
OUT = BASE / "common_object_relative_commands_v8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def main() -> int:
    manifest = read_json(MANIFEST)
    registration = read_json(REGISTRATION)
    gate = read_json(GATE)
    grasp = read_json(GRASP)
    wrist_primitive = read_json(WRIST_PRIMITIVE)
    physics = read_json(PHYSICS_CONFIG)
    palm = read_json(PALM_CONFIG)
    geometry = {
        row["name"]: row for row in physics["geometry_candidates"]
    }["FROZEN_COMPRESSED_SHORT_55"]
    surface = FrozenProxySurface(
        np.asarray(geometry["dimensions_m"]),
        np.asarray(physics["object"]["visual_dimensions_m"]),
    )
    object_pose = np.eye(4, dtype=np.float64)
    object_pose[:3, 3] = [
        *registration["registered_doll_center_world_xy_m"],
        float(physics["object"]["table_surface_world_z_m"])
        + float(physics["object"]["visual_dimensions_m"][2]) / 2.0
        + float(physics["object"]["spawn_clearance_above_table_m"]),
    ]
    object_pose[:3, :3] = Rotation.from_quat(
        registration["registered_doll_orientation_quaternion_xyzw"]
    ).as_matrix()
    g1 = G1Kinematics(load_common_config(), load_scene(load_common_config()))
    open_q = np.asarray(grasp["open_7d_rad"], dtype=np.float64)
    preshape_q = np.asarray(grasp["preshape_7d_rad"], dtype=np.float64)
    power_q = np.asarray(grasp["full_close_7d_rad"], dtype=np.float64)
    transition_frames = int(grasp["transition_frames_at_30hz"])
    object_from_wrist = np.eye(4, dtype=np.float64)
    object_from_wrist[:3, 3] = wrist_primitive[
        "object_from_left_wrist_position_m"
    ]
    object_from_wrist[:3, :3] = Rotation.from_quat(
        wrist_primitive["object_from_left_wrist_quaternion_xyzw"]
    ).as_matrix()
    target_wrist_world = object_pose @ object_from_wrist
    output_records: list[dict[str, object]] = []
    for record in manifest["records"]:
        source_command = Path(record["command"])
        reference = Path(record["source_reference"])
        with np.load(source_command, allow_pickle=False) as archive:
            raw = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
            stages = archive["stage"].astype(str)
            names = archive["joint_names"].astype(str)
        with np.load(reference, allow_pickle=False) as archive:
            left_phase = archive["left_hand_phase"].astype(str)
            owner = archive["ownership_state"].astype(str)
        if len(raw) != len(left_phase):
            raise RuntimeError("reference/command length mismatch")

        # The readiness decision is made on the frozen method-specific reference
        # geometry before any common finger realization is applied.
        candidates = np.flatnonzero(np.isin(left_phase, ["PRESHAPE", "GRASP", "HOLD"]))
        trigger: int | None = None
        trigger_geometry: dict[str, object] | None = None
        for frame in candidates:
            state = geometry_state(g1, surface, palm, raw[frame], object_pose, "left")
            gaps = state["pad_collider_to_doll_surface_sampled_signed_m"]
            ready = bool(
                state["whole_hand_to_doll_center_m"]
                <= float(gate["whole_hand_center_to_doll_center_max_m"])
                and max(float(value) for value in gaps.values())
                <= float(gate["each_digit_pad_sampled_positive_surface_gap_max_m"])
                and state["closing_axis_to_object_short_axis_unsigned_deg"]
                <= float(gate["closing_axis_to_object_short_axis_unsigned_max_deg"])
                and bool(state["object_between_thumb_and_index_middle"])
            )
            if ready:
                trigger = int(frame)
                trigger_geometry = {
                    "whole_hand_center_to_doll_m": float(
                        state["whole_hand_to_doll_center_m"]
                    ),
                    "pad_sampled_surface_gaps_m": {
                        key: float(value) for key, value in gaps.items()
                    },
                    "closing_axis_to_short_axis_deg": float(
                        state["closing_axis_to_object_short_axis_unsigned_deg"]
                    ),
                    "object_between_opposition": True,
                }
                break

        target_arm: np.ndarray | None = None
        target_arm_errors: dict[str, float] | None = None
        if trigger is not None:
            seed_q28 = raw[trigger].copy()
            seed_arm = seed_q28[:7].copy()
            lower = g1.arm_limits[:7, 0] + 1.0e-6
            upper = g1.arm_limits[:7, 1] - 1.0e-6

            def wrist_pose(arm: np.ndarray) -> np.ndarray:
                candidate = seed_q28.copy()
                candidate[:7] = arm
                from audit_physical_evaluator_task_frame_alignment import assign, world_pose_from_model

                assign(g1, candidate)
                return world_pose_from_model(g1, g1.wrist_pose("left"))

            def residual(arm: np.ndarray) -> np.ndarray:
                achieved = wrist_pose(arm)
                rotation_error = Rotation.from_matrix(
                    target_wrist_world[:3, :3] @ achieved[:3, :3].T
                ).as_rotvec()
                return np.r_[
                    40.0 * (achieved[:3, 3] - target_wrist_world[:3, 3]),
                    2.0 * rotation_error,
                    0.02 * (arm - seed_arm),
                ]

            solution = least_squares(
                residual,
                seed_arm,
                bounds=(lower, upper),
                max_nfev=300,
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
            target_arm = solution.x
            achieved = wrist_pose(target_arm)
            position_error = float(
                np.linalg.norm(achieved[:3, 3] - target_wrist_world[:3, 3])
            )
            orientation_error = float(
                np.rad2deg(
                    Rotation.from_matrix(
                        target_wrist_world[:3, :3] @ achieved[:3, :3].T
                    ).magnitude()
                )
            )
            if (
                position_error > float(wrist_primitive["ik_position_tolerance_m"])
                or orientation_error
                > float(wrist_primitive["ik_orientation_tolerance_deg"])
            ):
                raise RuntimeError("common object-relative wrist target IK failed")
            target_arm_errors = {
                "position_m": position_error,
                "orientation_deg": orientation_error,
                "maximum_joint_delta_rad": float(
                    np.max(np.abs(target_arm - seed_arm))
                ),
            }

        executed = raw.copy()
        override = np.zeros_like(executed, dtype=bool)
        first_preshape = int(candidates[0]) if len(candidates) else len(executed)
        # Common Dex3 OPEN is used only before the method declares pre-shape.
        executed[:first_preshape, 14:21] = open_q
        override[:first_preshape, 14:21] = np.abs(
            raw[:first_preshape, 14:21] - open_q
        ) > 1.0e-12
        if trigger is not None:
            if target_arm is None:
                raise RuntimeError("missing gated common wrist target")
            entry_end = min(
                trigger + int(wrist_primitive["entry_transition_frames_at_30hz"]),
                len(executed) - 1,
            )
            close_start = min(entry_end + 1, len(executed) - 1)
            thumb_preload_end = min(
                close_start + int(grasp["thumb_preload_frames_at_30hz"]),
                len(executed) - 1,
            )
            opposition_end = min(
                thumb_preload_end
                + int(grasp["opposition_side_close_frames_at_30hz"]),
                len(executed) - 1,
            )
            close_end = min(
                opposition_end + int(grasp["thumb_close_frames_at_30hz"]),
                len(executed) - 1,
            )
            dwell_end = min(
                close_end
                + int(wrist_primitive["stabilization_dwell_frames_at_30hz"]),
                len(executed) - 1,
            )
            return_end = min(
                dwell_end
                + int(wrist_primitive["return_to_method_specific_frames_at_30hz"]),
                len(executed) - 1,
            )
            if grasp["closure_mode"] == "COMMON_P14_FULL_CLOSE":
                # Keep the collision-safe common OPEN posture until the actual
                # whole-hand readiness instant.  This prevents a morphology-
                # specific PRESHAPE collider from displacing the free object
                # before the object-relative primitive owns the local motion.
                executed[first_preshape:close_start, 14:21] = open_q
                override[first_preshape:close_start, 14:21] = np.abs(
                    raw[first_preshape:close_start, 14:21] - open_q
                ) > 1.0e-12
                start_q = open_q.copy()
                thumb_preload_q = open_q[:3] + float(
                    grasp["thumb_preload_fraction"]
                ) * (power_q[:3] - open_q[:3])
                for frame in range(close_start, thumb_preload_end + 1):
                    alpha = (frame - close_start + 1) / max(
                        thumb_preload_end - close_start + 1, 1
                    )
                    weight = float(minimum_jerk(np.asarray(alpha)))
                    executed[frame, 14:17] = (
                        (1.0 - weight) * open_q[:3] + weight * thumb_preload_q
                    )
                    executed[frame, 17:21] = open_q[3:]
                for frame in range(thumb_preload_end + 1, opposition_end + 1):
                    alpha = (frame - thumb_preload_end) / max(
                        opposition_end - thumb_preload_end, 1
                    )
                    weight = float(minimum_jerk(np.asarray(alpha)))
                    executed[frame, 14:17] = thumb_preload_q
                    executed[frame, 17:21] = (
                        (1.0 - weight) * start_q[3:] + weight * power_q[3:]
                    )
                for frame in range(opposition_end + 1, close_end + 1):
                    alpha = (frame - opposition_end) / max(
                        close_end - opposition_end, 1
                    )
                    weight = float(minimum_jerk(np.asarray(alpha)))
                    executed[frame, 14:17] = (
                        (1.0 - weight) * thumb_preload_q + weight * power_q[:3]
                    )
                    executed[frame, 17:21] = power_q[3:]
                right_owned = np.flatnonzero(owner == "RIGHT_OWNED")
                hold_end = int(right_owned[0]) if len(right_owned) else len(executed)
                executed[close_end + 1 : hold_end, 14:21] = power_q
                override[first_preshape:hold_end, 14:21] = np.abs(
                    executed[first_preshape:hold_end, 14:21]
                    - raw[first_preshape:hold_end, 14:21]
                ) > 1.0e-12
            elif grasp["closure_mode"] != "PRESERVE_METHOD_SPECIFIC_AFTER_COMMON_OPEN":
                raise RuntimeError("unknown common grasp closure mode")
            arm_start = raw[trigger - 1, :7].copy() if trigger else raw[0, :7].copy()
            start_pose = wrist_pose(arm_start)
            object_center = object_pose[:3, 3]
            clearance = float(wrist_primitive["radial_clearance_m"])
            start_vector = start_pose[:3, 3] - object_center
            target_vector = target_wrist_world[:3, 3] - object_center
            start_clearance = object_center + start_vector * (
                1.0 + clearance / max(float(np.linalg.norm(start_vector)), 1.0e-9)
            )
            target_clearance = object_center + target_vector * (
                1.0 + clearance / max(float(np.linalg.norm(target_vector)), 1.0e-9)
            )
            phase_one, phase_two = [
                float(value) for value in wrist_primitive["entry_path_phase_fractions"]
            ]
            rotation_delta = Rotation.from_matrix(
                start_pose[:3, :3].T @ target_wrist_world[:3, :3]
            ).as_rotvec()
            previous_arm = arm_start.copy()
            for frame in range(trigger, entry_end + 1):
                alpha = (frame - trigger + 1) / max(entry_end - trigger + 1, 1)
                orientation_weight = float(minimum_jerk(np.asarray(alpha)))
                desired = np.eye(4, dtype=np.float64)
                if alpha <= phase_one:
                    local_alpha = alpha / phase_one
                    weight = float(minimum_jerk(np.asarray(local_alpha)))
                    desired[:3, 3] = (
                        (1.0 - weight) * start_pose[:3, 3]
                        + weight * start_clearance
                    )
                elif alpha <= phase_two:
                    local_alpha = (alpha - phase_one) / (phase_two - phase_one)
                    weight = float(minimum_jerk(np.asarray(local_alpha)))
                    desired[:3, 3] = (
                        (1.0 - weight) * start_clearance
                        + weight * target_clearance
                    )
                else:
                    local_alpha = (alpha - phase_two) / (1.0 - phase_two)
                    weight = float(minimum_jerk(np.asarray(local_alpha)))
                    desired[:3, 3] = (
                        (1.0 - weight) * target_clearance
                        + weight * target_wrist_world[:3, 3]
                    )
                desired[:3, :3] = start_pose[:3, :3] @ Rotation.from_rotvec(
                    orientation_weight * rotation_delta
                ).as_matrix()

                def path_residual(arm: np.ndarray) -> np.ndarray:
                    achieved_path = wrist_pose(arm)
                    rotation_error = Rotation.from_matrix(
                        desired[:3, :3] @ achieved_path[:3, :3].T
                    ).as_rotvec()
                    return np.r_[
                        40.0 * (achieved_path[:3, 3] - desired[:3, 3]),
                        2.0 * rotation_error,
                        0.02 * (arm - previous_arm),
                    ]

                step_solution = least_squares(
                    path_residual,
                    previous_arm,
                    bounds=(lower, upper),
                    max_nfev=100,
                    xtol=1.0e-10,
                    ftol=1.0e-10,
                    gtol=1.0e-10,
                )
                previous_arm = step_solution.x
                executed[frame, :7] = previous_arm
            executed[entry_end + 1 : dwell_end + 1, :7] = target_arm
            return_target = raw[return_end, :7].copy()
            for frame in range(dwell_end + 1, return_end + 1):
                alpha = (frame - dwell_end) / max(return_end - dwell_end, 1)
                weight = float(minimum_jerk(np.asarray(alpha)))
                executed[frame, :7] = (1.0 - weight) * target_arm + weight * return_target
            override[trigger : return_end + 1, :7] = np.abs(
                executed[trigger : return_end + 1, :7]
                - raw[trigger : return_end + 1, :7]
            ) > 1.0e-12

        # Reference preflight ends after the first handoff window.  No claim is
        # made about right acquisition or full task here.
        dual = np.flatnonzero(owner == "DUAL_CONTACT")
        preflight_end = min(len(executed), (int(dual[0]) + 30) if len(dual) else len(executed))
        executed = executed[:preflight_end]
        raw = raw[:preflight_end]
        stages = stages[:preflight_end]
        override = override[:preflight_end]
        destination = (
            OUT
            / str(record["method"]).lower().replace("-", "_")
            / Path(record["command"]).name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                commanded_q_rad=executed.astype(np.float32),
                stage=stages,
                joint_names=names,
                control_fps_hz=np.asarray(30.0),
                raw_reference_command=raw.astype(np.float32),
                executed_common_controller_command=executed.astype(np.float32),
                common_controller_override_mask=override,
                common_whole_hand_gate_triggered=np.asarray(trigger is not None),
                common_whole_hand_gate_frame=np.asarray(-1 if trigger is None else trigger),
                common_whole_hand_gate_config=np.asarray(str(GATE)),
                common_whole_hand_gate_config_sha256=np.asarray(sha256(GATE)),
                common_task_registration=np.asarray(str(REGISTRATION)),
                common_task_registration_sha256=np.asarray(sha256(REGISTRATION)),
                common_dex3_grasp_realization=np.asarray(str(GRASP)),
                common_dex3_grasp_realization_sha256=np.asarray(sha256(GRASP)),
                method=np.asarray(record["method"]),
                eval_index=np.asarray(int(record["eval_index"])),
                reference_level_preflight=np.asarray(True),
                runtime_right_three_digit_gate_required=np.asarray(False),
            )
        os.replace(temporary, destination)
        output_records.append(
            {
                **record,
                "common_command": str(destination.resolve()),
                "common_command_sha256": sha256(destination),
                "whole_hand_gate_triggered": trigger is not None,
                "whole_hand_gate_frame": trigger,
                "whole_hand_gate_geometry": trigger_geometry,
                "object_relative_wrist_target_ik": target_arm_errors,
                "arm_scalar_overrides": int(np.count_nonzero(override[:, :14])),
                "dex3_scalar_overrides": int(np.count_nonzero(override[:, 14:])),
                "preflight_frames": preflight_end,
            }
        )
    output_manifest = {
        "schema_version": "common_whole_hand_reference_preflight_v1",
        "status": "PREPARED",
        "task_registration": str(REGISTRATION),
        "task_registration_sha256": sha256(REGISTRATION),
        "readiness_gate": str(GATE),
        "readiness_gate_sha256": sha256(GATE),
        "grasp_realization": str(GRASP),
        "grasp_realization_sha256": sha256(GRASP),
        "object_relative_wrist_primitive": str(WRIST_PRIMITIVE),
        "object_relative_wrist_primitive_sha256": sha256(WRIST_PRIMITIVE),
        "base_physics_config": str(PHYSICS_CONFIG),
        "base_physics_config_sha256": sha256(PHYSICS_CONFIG),
        "arm_overrides_allowed": False,
        "same_logic_for_a_b": True,
        "records": output_records,
    }
    path = OUT.parent / "COMMON_OBJECT_RELATIVE_COMMAND_MANIFEST_V8.json"
    path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PREPARED", "manifest": str(path), "commands": len(output_records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
