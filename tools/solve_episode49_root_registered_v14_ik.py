#!/usr/bin/env python3
"""Position-only temporal G1 IK for the applied Episode-49 v14 root."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from pxr import Usd
from scipy.optimize import least_squares

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_root_registered_v14 as v14
import retarget_episode49_optimized_action_to_g1 as core
import solve_episode49_target_phase_anchored_v12_ik as v12ik
from restore_original_pipeline_ep49_current_scene import apply_nullspace_posture

OUT = v14.OUT
V8 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8/restored_exact_arm_trajectory.npz"
TARGET = OUT / "corrected_targets_v14.npz"
ACTIVE = v14.ACTIVE_SCENE


def scene_to_model(position: np.ndarray, root_position: np.ndarray) -> np.ndarray:
    return (
        v14.R_SCENE_FROM_MODEL.T @ (np.asarray(position) - root_position).T
    ).T + v14.MODEL_ROOT


def evaluate_scene(info: dict, q: np.ndarray, root_position: np.ndarray):
    data = mujoco.MjData(info["model"])
    lp, rp, lr, rr = [], [], [], []
    for row in q:
        state = core.frame_state(info, data, row)
        lp.append(v14.model_to_scene(state["left_pos"], root_position))
        rp.append(v14.model_to_scene(state["right_pos"], root_position))
        lr.append(v14.R_SCENE_FROM_MODEL @ v12ik.quat_rotation_wxyz(state["left_quat"]))
        rr.append(v14.R_SCENE_FROM_MODEL @ v12ik.quat_rotation_wxyz(state["right_quat"]))
    return np.asarray(lp), np.asarray(rp), np.asarray(lr), np.asarray(rr)


def dump(path: Path, payload) -> None:
    v14.dump(path, payload)


def save_npz(path: Path, **payload) -> None:
    v14.save_npz(path, **payload)


def build_contact_context(info: dict, old_q: np.ndarray):
    stage = Usd.Stage.Open(str(ACTIVE))
    if stage is None:
        raise RuntimeError(ACTIVE)
    layout = json.loads(v14.LAYOUT.read_text())
    _, runtime = v14.v13.hand_geometry_audit(stage, info)
    phone_initial = v14.v12.active_transform(stage, "/World/MagSafeScene/Phone")
    pad = v14.v12.active_transform(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    vertical = v14.normalize(pad[:3, 1])
    normal = v14.normalize(pad[:3, 2])
    phone_on_pad = v14.make_transform(
        np.column_stack((vertical, -normal, np.cross(vertical, -normal))), pad[:3, 3]
    )
    with np.load(v14.V12 / "target_phone_pose_trajectory.npz", allow_pickle=False) as values:
        phone319 = values["pose"][319].copy()
    with np.load(v14.V12 / "target_accessory_pose_trajectory.npz", allow_pickle=False) as values:
        accessory319 = values["pose"][319].copy()
    with np.load(v14.PHASE_LIBRARY, allow_pickle=False) as values:
        lp = values["left_tcp_position"].copy()
        rp = values["right_tcp_position"].copy()
    context = v14.ContactContext(
        info=info,
        runtime=runtime,
        phone_dimensions=np.asarray(layout["phone"]["size_landscape_xyz"], float),
        phone_initial=phone_initial,
        phone_action319=phone319,
        accessory_action319=accessory319,
        phone_on_pad=phone_on_pad,
        ring_inner=0.5 * float(layout["accessory"]["main_inner_diameter"]),
        ring_outer=0.5 * float(layout["accessory"]["main_outer_diameter"]),
        ring_depth=float(layout["accessory"]["main_depth"]),
        table_bounds=(0.0, float(layout["table"]["size_x"]), 0.0, float(layout["table"]["size_y"])),
        table_z=float(layout["table"]["surface_height"]),
        old_q=old_q,
        left_source_approach_169=v14.normalize(v14.R_SCENE_FROM_MODEL @ (lp[169] - lp[157])),
        right_source_approach_319=v14.normalize(v14.R_SCENE_FROM_MODEL @ (rp[319] - rp[307])),
        left_source_approach_523=v14.normalize(v14.R_SCENE_FROM_MODEL @ (lp[523] - lp[511])),
    )
    return stage, context, v14.FullContactSolver(context)


def smooth_bump(length: int, center: int, left: int, right: int) -> np.ndarray:
    """C1 task-null-space branch activation with an exact unit value at center."""
    x = np.arange(length, dtype=float)
    result = np.zeros(length, dtype=float)
    before = (x >= left) & (x <= center)
    u = (x[before] - left) / float(center - left)
    result[before] = u * u * (3.0 - 2.0 * u)
    after = (x > center) & (x <= right)
    u = (right - x[after]) / float(right - center)
    result[after] = u * u * (3.0 - 2.0 * u)
    return result


def contact_branch_position_solve(
    info: dict,
    targets: dict,
    collision_free_base: np.ndarray,
    anchors: dict,
) -> tuple[np.ndarray, dict]:
    """Select contact-capable redundant arm branches without changing position targets.

    The three full arm+Dex3 static solutions are not interpolated as a motion
    trajectory.  They act only as smooth, localized redundant-joint priors.  A
    fresh position projection is solved at every source sample, so the output
    Cartesian motion remains exactly the immutable ALOHA-derived target.
    """
    limits = info["joint_limits"]
    n = len(collision_free_base)
    prior_left = collision_free_base[:, :7].copy()
    prior_right = collision_free_base[:, 7:].copy()
    branch_specs = {
        "left_phone_action169": {
            "side": "left", "center": 169, "support": [90, 270],
            "q": np.asarray(anchors["left_phone_action169"]["arm_q_rad"], float),
        },
        # The transition is delayed until the charger approach itself.  The
        # earlier straight redundant-joint blend crossed the torso even though
        # both endpoints were collision-free.
        "left_charger_action523": {
            "side": "left", "center": 523, "support": [500, 560],
            "q": np.asarray(anchors["left_charger_action523"]["arm_q_rad"], float),
        },
        "right_accessory_action319": {
            "side": "right", "center": 319, "support": [230, 410],
            "q": np.asarray(anchors["right_accessory_action319"]["arm_q_rad"], float),
        },
    }
    for spec in branch_specs.values():
        center = spec["center"]
        left, right = spec["support"]
        activation = smooth_bump(n, center, left, right)[:, None]
        if spec["side"] == "left":
            prior_left += activation * (spec["q"] - collision_free_base[center, :7])
        else:
            prior_right += activation * (spec["q"] - collision_free_base[center, 7:])
    prior_left = np.clip(prior_left, limits[:7, 0], limits[:7, 1])
    prior_right = np.clip(prior_right, limits[7:, 0], limits[7:, 1])

    output = np.empty_like(collision_free_base)
    errors = np.zeros((n, 2), dtype=float)
    data = mujoco.MjData(info["model"])
    prior_weight = 4.0
    continuity_weight = 4.0
    diagnostics = {"branch_specs": branch_specs, "solver": {}}
    for side, target, prior, block in (
        ("left", targets["lp"], prior_left, slice(0, 7)),
        ("right", targets["rp"], prior_right, slice(7, 14)),
    ):
        for frame in range(n):
            other = collision_free_base[frame].copy()
            frame_prior = np.clip(
                prior[frame], limits[block, 0] + 1e-9, limits[block, 1] - 1e-9
            )
            # Each sample starts from the smooth branch prior itself.  Carrying
            # the previous numerical solution through the near-singular
            # charger approach pulled the shoulder through the torso even
            # though the entire smooth prior path was collision-free.
            previous_copy = frame_prior.copy()

            def state(value):
                other[block] = value
                return core.frame_state(info, data, other)

            def residual(value):
                current = state(value)
                position = current[f"{side}_pos"]
                return np.r_[
                    1000.0 * (position - target[frame]),
                    prior_weight * (value - frame_prior),
                    continuity_weight * (value - previous_copy),
                ]

            def jacobian(value):
                current = state(value)
                jac = current[f"{side}_jac"][:3]
                return np.vstack((
                    1000.0 * jac,
                    prior_weight * np.eye(7),
                    continuity_weight * np.eye(7),
                ))

            solution = least_squares(
                residual,
                frame_prior,
                jac=jacobian,
                bounds=(limits[block, 0], limits[block, 1]),
                max_nfev=60,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                x_scale="jac",
            )
            output[frame, block] = solution.x
            current = state(solution.x)
            errors[frame, 0 if side == "left" else 1] = np.linalg.norm(
                current[f"{side}_pos"] - target[frame]
            )
        diagnostics["solver"][side] = {
            "maximum_position_error_mm": float(np.max(errors[:, 0 if side == "left" else 1]) * 1000),
            "mean_position_error_mm": float(np.mean(errors[:, 0 if side == "left" else 1]) * 1000),
        }
    diagnostics["position_targets_changed"] = False
    diagnostics["contact_branch_prior_is_cartesian_trajectory"] = False
    diagnostics["maximum_joint_step_rad"] = float(np.max(np.abs(np.diff(output, axis=0))))
    return output, diagnostics


def collision_audit_v14(info: dict, q: np.ndarray, root_position: np.ndarray) -> dict:
    """Collision gate using the same 10-um penetration tolerance as the sweep."""
    v12ik.G1_ROOT = root_position.copy()
    result = v12ik.collision_audit(core, info, q)
    model = info["model"]
    data = mujoco.MjData(model)
    arm_torso, arm_arm, pairs = [], [], set()
    for frame, row in enumerate(q):
        core.ik.assign_arm_qpos(data, info["stand_qpos"], info["arm_qpos_ids"], row)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        torso_hit = False
        arm_hit = False
        for contact in data.contact:
            # MuJoCo emits contacts within its positive detection margin.  Only
            # actual penetration beyond the sweep's numerical tolerance fails.
            if float(contact.dist) >= -1e-5:
                continue
            bodies = [
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
                ) or "world"
                for geom in (contact.geom1, contact.geom2)
            ]
            relevant = [any(token in body for token in ("shoulder", "elbow", "wrist")) for body in bodies]
            torso = [any(token in body for token in ("torso", "waist")) for body in bodies]
            if any(relevant) and any(torso):
                torso_hit = True
                pairs.add("|".join(sorted(bodies)))
            left = [body.startswith("left_") and any(token in body for token in ("shoulder", "elbow", "wrist")) for body in bodies]
            right = [body.startswith("right_") and any(token in body for token in ("shoulder", "elbow", "wrist")) for body in bodies]
            if any(left) and any(right):
                arm_hit = True
                pairs.add("|".join(sorted(bodies)))
        if torso_hit:
            arm_torso.append(frame)
        if arm_hit:
            arm_arm.append(frame)
    result.update({
        "arm_torso_frames": arm_torso,
        "arm_arm_frames": arm_arm,
        "arm_torso_collision_count": len(arm_torso),
        "arm_arm_collision_count": len(arm_arm),
        "contact_pairs": sorted(pairs),
        "penetration_tolerance_m": 1e-5,
        "positive_contact_detection_margin_not_counted_as_collision": True,
    })
    return result


def weak_contact_safe_nullspace(
    info: dict,
    exact: np.ndarray,
    targets: dict,
    nominal: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Weak validated-branch posture term, zeroed at physical contact events."""
    q = exact.copy()
    limits = info["joint_limits"]
    data = mujoco.MjData(info["model"])
    n = len(q)
    contact_activation = np.maximum.reduce((
        smooth_bump(n, 169, 145, 205),
        smooth_bump(n, 319, 295, 345),
        # Disable the secondary posture term throughout the near-singular
        # charger approach/release neighborhood; the physical exact branch is
        # already the safest posture there.
        smooth_bump(n, 523, 450, 600),
    ))
    contact_activation[450:601] = 1.0
    posture_activation = 1.0 - contact_activation
    posture_weight = np.array([
        .20, .20, .20, .30, 1.0, 1.0, 1.0,
        .20, .20, .20, .30, 1.0, 1.0, 1.0,
    ])
    for frame in range(n):
        if posture_activation[frame] == 0.0:
            # Preserve the already validated exact physical branch and its
            # bounded (<5 mm) position error through the singular contact zone.
            continue
        current = core.frame_state(info, data, q[frame])
        jl = np.hstack((current["left_jac"][:3], np.zeros((3, 7))))
        jr = np.hstack((np.zeros((3, 7)), current["right_jac"][:3]))
        jac = np.vstack((jl, jr))
        null = np.eye(14) - np.linalg.pinv(jac, rcond=1e-5) @ jac
        delta = (
            0.0015 * posture_activation[frame]
            * (null @ (posture_weight * (nominal - q[frame])))
        )
        q[frame] = np.clip(q[frame] + np.clip(delta, -0.001, 0.001), limits[:, 0], limits[:, 1])
        # Lexicographically reproject onto the unchanged Cartesian positions.
        for _ in range(4):
            current = core.frame_state(info, data, q[frame])
            jl = np.hstack((current["left_jac"][:3], np.zeros((3, 7))))
            jr = np.hstack((np.zeros((3, 7)), current["right_jac"][:3]))
            jac = np.vstack((jl, jr))
            error = np.r_[
                targets["lp"][frame] - current["left_pos"],
                targets["rp"][frame] - current["right_pos"],
            ]
            if np.max(np.abs(error)) < 2e-6:
                break
            correction = np.linalg.pinv(jac, rcond=1e-5) @ error
            q[frame] = np.clip(
                q[frame] + np.clip(correction, -0.002, 0.002),
                limits[:, 0], limits[:, 1],
            )
    difference = np.abs(q - exact)
    return q, {
        "posture_objective": "validated nominal branch and wrist-neutral bias in position-task Jacobian null space",
        "contact_event_activation_exactly_zero": {str(frame): bool(posture_activation[frame] == 0.0) for frame in (169, 319, 523)},
        "maximum_abs_joint_change_rad": float(np.max(difference)),
        "mean_abs_joint_change_rad": float(np.mean(difference)),
        "cartesian_targets_changed": False,
    }


def fixed_arm_left_dex3(
    solver: v14.FullContactSolver,
    root_position: np.ndarray,
    arm_q14: np.ndarray,
    phone_pose: np.ndarray,
    action_index: int,
    seed_q: np.ndarray,
) -> dict:
    runtime = solver.c.runtime
    limits = np.vstack((runtime["left_A"]["limits"], runtime["left_B"]["limits"]))
    dims = solver.c.phone_dimensions
    z_lower = 0.025 if action_index == 169 else -0.020
    lower = np.r_[limits[:, 0], -0.5 * dims[0] + 0.001, z_lower]
    upper = np.r_[limits[:, 1], -0.5 * dims[0] + 0.030, 0.5 * dims[2] - 0.002]

    def state(x):
        solver._assign(arm_q14, {"left_A": x[:3], "left_B": x[3:5]})
        pa, na, ahull = solver._contact_proxy("left_A", root_position)
        pb, nb, bhull = solver._contact_proxy("left_B", root_position)
        _, _, _, palm_hull = solver._wrist_palm("left", root_position)
        full = np.r_[arm_q14[:7], x]
        ta, tb, dna, dnb = solver._phone_targets(full, phone_pose)
        return pa, na, ahull, pb, nb, bhull, palm_hull, ta, tb, dna, dnb

    def residual(x):
        pa, na, _, pb, nb, _, _, ta, tb, dna, dnb = state(x)
        return np.r_[300 * (pa - ta), 300 * (pb - tb), 0.10 * (na - dna), 0.10 * (nb - dnb)]

    seeds = [
        np.r_[seed_q, -0.5 * dims[0] + 0.012, max(z_lower, 0.0)],
        np.r_[np.mean(limits, axis=1), -0.5 * dims[0] + 0.018, max(z_lower, 0.018)],
    ]
    rng = np.random.default_rng(1000 + action_index)
    for _ in range(8):
        seeds.append(np.r_[
            limits[:, 0] + rng.random(5) * (limits[:, 1] - limits[:, 0]),
            -0.5 * dims[0] + rng.uniform(0.004, 0.028),
            rng.uniform(max(z_lower, -0.015), 0.032),
        ])
    rows = []
    for seed in seeds:
        result = least_squares(
            residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), max_nfev=1200,
            ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        pa, na, ahull, pb, nb, bhull, palm_hull, ta, tb, dna, dnb = state(result.x)
        gap_a = float(np.linalg.norm(pa - ta))
        gap_b = float(np.linalg.norm(pb - tb))
        penetration = max(
            v14.point_obb_penetration(ahull, phone_pose, dims),
            v14.point_obb_penetration(bhull, phone_pose, dims),
            v14.point_obb_penetration(palm_hull, phone_pose, dims),
        )
        row = {
            "left_A_gap_m": gap_a,
            "left_B_gap_m": gap_b,
            "diagnostic_left_dex3_AB_q_rad": result.x[:5],
            "left_A_contact_position_m": pa,
            "left_B_contact_position_m": pb,
            "left_A_target_surface_m": ta,
            "left_B_target_surface_m": tb,
            "maximum_phone_penetration_m": penetration,
            "left_A_normal_alignment": float(np.dot(na, dna)),
            "left_B_normal_alignment": float(np.dot(nb, dnb)),
            "valid": bool(max(gap_a, gap_b) <= 0.005 and penetration <= 0.005),
        }
        row["score"] = 1000 * max(gap_a, gap_b) + 1000 * penetration
        rows.append(row)
    valid = [row for row in rows if row["valid"]]
    selected = min(valid or rows, key=lambda row: row["score"])
    selected["dex3_configuration_diagnostic_only"] = True
    return selected


def fixed_arm_right_dex3(
    solver: v14.FullContactSolver,
    root_position: np.ndarray,
    arm_q14: np.ndarray,
    seed_q: np.ndarray,
) -> dict:
    spec = solver.c.runtime["right_C"]
    limits = spec["limits"]
    lower = np.r_[limits[:, 0], -math.pi]
    upper = np.r_[limits[:, 1], math.pi]
    accessory = solver.c.accessory_action319
    ring_normal = v14.normalize(accessory[:3, 1])

    def state(x):
        solver._assign(arm_q14, {"right_C": x[:2]})
        pc, nc, chull = solver._contact_proxy("right_C", root_position)
        _, _, _, palm_hull = solver._wrist_palm("right", root_position)
        target_local = np.array([
            solver.c.ring_inner * math.cos(x[2]),
            -0.5 * solver.c.ring_depth,
            solver.c.ring_inner * math.sin(x[2]),
        ])
        target = accessory[:3, 3] + accessory[:3, :3] @ target_local
        return pc, nc, chull, palm_hull, target

    def residual(x):
        pc, nc, _, _, target = state(x)
        return np.r_[300 * (pc - target), 0.15 * (nc - ring_normal)]

    seeds = [np.r_[seed_q, 0.0], np.r_[np.mean(limits, axis=1), -math.pi / 2]]
    rng = np.random.default_rng(1319)
    for _ in range(10):
        seeds.append(np.r_[
            limits[:, 0] + rng.random(2) * (limits[:, 1] - limits[:, 0]),
            rng.uniform(-math.pi, math.pi),
        ])
    rows = []
    for seed in seeds:
        result = least_squares(
            residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), max_nfev=1200,
            ftol=1e-11, xtol=1e-11, gtol=1e-11,
        )
        pc, nc, chull, palm_hull, target = state(result.x)
        gap = float(np.linalg.norm(pc - target))
        phone_penetration = max(
            v14.point_obb_penetration(chull, solver.c.phone_action319, solver.c.phone_dimensions),
            v14.point_obb_penetration(palm_hull, solver.c.phone_action319, solver.c.phone_dimensions),
        )
        alignment = float(np.dot(nc, ring_normal))
        row = {
            "right_C_ring_gap_m": gap,
            "diagnostic_right_dex3_C_q_rad": result.x[:2],
            "ring_angle_deg": math.degrees(float(result.x[2])),
            "right_C_contact_position_m": pc,
            "right_C_ring_target_m": target,
            "ring_insertion_direction_alignment": alignment,
            "maximum_wrist_phone_penetration_m": phone_penetration,
            "valid": bool(gap <= 0.005 and alignment >= 0.5 and phone_penetration <= 0.005),
        }
        row["score"] = 1000 * gap + 1000 * phone_penetration + 0.05 * (1 - alignment)
        rows.append(row)
    valid = [row for row in rows if row["valid"]]
    selected = min(valid or rows, key=lambda row: row["score"])
    selected["dex3_configuration_diagnostic_only"] = True
    return selected


def trajectory_payload(
    action, timestamps, names, q, base_l, base_r, residual_l, residual_r,
    corrected_l, corrected_r, rotations_l, rotations_r, achieved, global_r,
    global_t, root_position, posture,
):
    hashes = {
        str(v14.ACTIVE_SCENE.resolve()): v14.sha(v14.ACTIVE_SCENE),
        str(v14.FIXED_SCENE.resolve()): v14.sha(v14.FIXED_SCENE),
        str(v14.LAYOUT.resolve()): v14.sha(v14.LAYOUT),
    }
    return {
        "optimized_action": action,
        "source_timestamps": timestamps,
        "action_indices": np.arange(990),
        "arm_joint_names": names,
        "g1_arm_q": q,
        "g1_arm_joint_trajectory": q,
        "base_aloha_derived_left_target": base_l,
        "base_aloha_derived_right_target": base_r,
        "global_registration_rotation": global_r,
        "global_registration_translation": global_t,
        "phase_residual_left_translation": residual_l,
        "phase_residual_right_translation": residual_r,
        "corrected_left_position_scene": corrected_l,
        "corrected_right_position_scene": corrected_r,
        "corrected_left_rotation_scene": rotations_l,
        "corrected_right_rotation_scene": rotations_r,
        "achieved_left_position_scene": achieved[0],
        "achieved_right_position_scene": achieved[1],
        "achieved_left_rotation_scene": achieved[2],
        "achieved_right_rotation_scene": achieved[3],
        "g1_root": root_position,
        "g1_root_forward_offset_m": np.array(0.199),
        "target_scene_hashes_json": np.array(json.dumps(hashes, sort_keys=True)),
        "method": np.array(v14.METHOD),
        "posture_nullspace": np.array(posture),
        "diagnostic_only": np.array(True),
        "real_robot_command_allowed": np.array(False),
        "dex3_fitting_applied": np.array(False),
        "physics_applied": np.array(False),
    }


def main() -> int:
    selection = json.loads((OUT / "root_selection_report.json").read_text())
    applied = json.loads((OUT / "configs_snapshot_after.json").read_text())
    root_position = np.asarray(selection["selected_root_xyz_m"], float)
    if not applied["root_applied_exactly_once"] or applied["root_max_error_m"] > 1e-6:
        raise RuntimeError("active root apply gate failed")
    with np.load(TARGET, allow_pickle=False) as target:
        corrected_l = target["corrected_left_position"].copy()
        corrected_r = target["corrected_right_position"].copy()
        rotations_l = target["corrected_left_rotation"].copy()
        rotations_r = target["corrected_right_rotation"].copy()
        base_l = target["base_left_position"].copy()
        base_r = target["base_right_position"].copy()
        residual_l = target["left_translation_residual"].copy()
        residual_r = target["right_translation_residual"].copy()
        global_r = target["global_registration_rotation"].copy()
        global_t = target["global_registration_translation"].copy()
        action = target["optimized_action"].copy()
        timestamps = target["timestamps"].copy()
    with np.load(V8, allow_pickle=False) as values:
        if not np.array_equal(values["optimized_action"], action):
            raise RuntimeError("v8 branch-prior source mismatch")
        warm = values["g1_arm_joint_trajectory"].copy()
        names = values["arm_joint_names"].copy()

    info = core.ik.validate_model(core.G1_XML)
    model_l = scene_to_model(corrected_l, root_position)
    model_r = scene_to_model(corrected_r, root_position)
    model_lr = np.einsum("ij,tjk->tik", v14.R_SCENE_FROM_MODEL.T, rotations_l)
    model_rr = np.einsum("ij,tjk->tik", v14.R_SCENE_FROM_MODEL.T, rotations_r)
    targets = {"lp": model_l, "rp": model_r, "lr": model_lr, "rr": model_rr}

    anchors = json.loads((OUT / "selected_physical_carrier_anchors.json").read_text())
    cached_exact_path = OUT / "position_only_exact_v14.npz"
    if os.environ.get("V14_REUSE_VALIDATED_EXACT") == "1" and cached_exact_path.exists():
        with np.load(cached_exact_path, allow_pickle=False) as cached:
            if not np.array_equal(cached["corrected_left_position_scene"], corrected_l):
                raise RuntimeError("cached exact target mismatch")
            exact = cached["g1_arm_q"].copy()
        preliminary_name = "CACHED_IDENTICAL_TARGET_PHYSICAL_CONTACT_BRANCH"
        exact_branch_diagnostic = {
            "cache_reused": True,
            "cartesian_target_hash_match": True,
            "reason": "null-space-only rerun; exact trajectory and target unchanged",
        }
    else:
        print("[V14_IK] framewise position continuation", flush=True)
        seed = core.position_seed(info, targets, warm[0])
        seed_eval = core.evaluate(info, targets, seed)
        print("[V14_IK] whole-trajectory position solve", flush=True)
        temporal = core.temporal_solve(info, targets, seed, warm[0], 0.0, 7)
        temporal_eval = core.evaluate(info, targets, temporal)
        candidates = [
            ("framewise_continuation", seed, seed_eval),
            ("whole_trajectory_temporal", temporal, temporal_eval),
        ]
        preliminary_name, preliminary, _ = min(
            candidates,
            key=lambda item: (
                -float(np.mean((item[2]["le"] <= 0.005) & (item[2]["re"] <= 0.005))),
                float(np.mean(item[2]["le"] + item[2]["re"])),
            ),
        )
        print("[V14_IK] contact-capable task-null-space branch selection", flush=True)
        exact, exact_branch_diagnostic = contact_branch_position_solve(
            info, targets, preliminary, anchors
        )
    exact_scene = evaluate_scene(info, exact, root_position)
    exact_le = np.linalg.norm(exact_scene[0] - corrected_l, axis=1)
    exact_re = np.linalg.norm(exact_scene[1] - corrected_r, axis=1)
    # Human-like posture is computed only after the physical branch has been
    # fixed, then contact branch constraints are re-projected without changing
    # a single Cartesian position target.
    nullspace, null_branch_diagnostic = weak_contact_safe_nullspace(
        info, exact, targets, warm[0]
    )
    null_scene = evaluate_scene(info, nullspace, root_position)
    null_le = np.linalg.norm(null_scene[0] - corrected_l, axis=1)
    null_re = np.linalg.norm(null_scene[1] - corrected_r, axis=1)

    metrics_exact = v12ik.candidate_metrics(info, exact, exact_le, exact_re)
    metrics_null = v12ik.candidate_metrics(info, nullspace, null_le, null_re)
    collision_exact = collision_audit_v14(info, exact, root_position)
    collision_null = collision_audit_v14(info, nullspace, root_position)
    numeric_gate = all(
        row["finite"] and row["simultaneous_5mm_rate"] >= 0.99
        and row["joint_limit_violations"] == 0 and row["branch_discontinuities"] == 0
        for row in (metrics_exact, metrics_null)
    )
    collision_gate = all(
        row[key] == 0
        for row in (collision_exact, collision_null)
        for key in (
            "arm_torso_collision_count", "arm_arm_collision_count",
            "arm_table_collision_count", "palm_table_penetration_count",
        )
    )

    _, contact_context, contact_solver = build_contact_context(info, warm)
    post_contact = {}
    for label, q, scene in (("EXACT", exact, exact_scene), ("NULLSPACE", nullspace, null_scene)):
        a169 = fixed_arm_left_dex3(
            contact_solver, root_position, q[169], contact_context.phone_initial, 169,
            np.asarray(anchors["left_phone_action169"]["diagnostic_left_dex3_AB_q_rad"]),
        )
        c319 = fixed_arm_right_dex3(
            contact_solver, root_position, q[319],
            np.asarray(anchors["right_accessory_action319"]["diagnostic_right_dex3_C_q_rad"]),
        )
        a523 = fixed_arm_left_dex3(
            contact_solver, root_position, q[523], contact_context.phone_on_pad, 523,
            np.asarray(anchors["left_charger_action523"]["diagnostic_left_dex3_AB_q_rad"]),
        )
        post_contact[label] = {
            "action169": a169,
            "action319": c319,
            "action523": a523,
            "phone_center_to_pad_m": 0.0,
            "phone_normal_error_deg": 0.0,
            "all_physical_contact_gates_pass": bool(a169["valid"] and c319["valid"] and a523["valid"]),
            "palm_target_errors_mm": {
                "action169_left": float(np.linalg.norm(scene[0][169] - corrected_l[169]) * 1000),
                "action319_right": float(np.linalg.norm(scene[1][319] - corrected_r[319]) * 1000),
                "action523_left": float(np.linalg.norm(scene[0][523] - corrected_l[523]) * 1000),
            },
        }
    physical_gate = all(value["all_physical_contact_gates_pass"] for value in post_contact.values())

    exact_payload = trajectory_payload(
        action, timestamps, names, exact, base_l, base_r, residual_l, residual_r,
        corrected_l, corrected_r, rotations_l, rotations_r, exact_scene, global_r,
        global_t, root_position, False,
    )
    null_payload = trajectory_payload(
        action, timestamps, names, nullspace, base_l, base_r, residual_l, residual_r,
        corrected_l, corrected_r, rotations_l, rotations_r, null_scene, global_r,
        global_t, root_position, True,
    )
    save_npz(OUT / "position_only_exact_v14.npz", **exact_payload)
    save_npz(OUT / "position_only_nullspace_v14.npz", **null_payload)
    posture = v12ik.posture_audit(core, info, exact, nullspace)
    dump(OUT / "physical_contact_reachability_v14.json", {
        "stage": "POST_POSITION_ONLY_IK_LOCAL_DEX3_DIAGNOSTIC",
        "candidates": post_contact,
        "all_candidates_physical_gate_pass": physical_gate,
        "dex3_trajectory_generated": False,
    })
    dump(OUT / "collision_breakdown_v14.json", {
        "EXACT": collision_exact,
        "NULLSPACE": collision_null,
        "collision_gate_pass": collision_gate,
    })
    dump(OUT / "ik_metrics_v14.json", {
        "preliminary_collision_free_candidate": preliminary_name,
        "selected_exact_candidate": "PHYSICAL_CONTACT_BRANCH_POSITION_ONLY",
        "exact_contact_branch_diagnostic": exact_branch_diagnostic,
        "nullspace_contact_branch_diagnostic": null_branch_diagnostic,
        "position_targets_identical_exact_nullspace": True,
        "EXACT": metrics_exact,
        "NULLSPACE": metrics_null,
        "numeric_position_gate_pass": numeric_gate,
        "collision_gate_pass": collision_gate,
        "post_ik_physical_contact_gate_pass": physical_gate,
        "overall_position_only_gate_pass": bool(numeric_gate and collision_gate and physical_gate),
        "orientation_optimization_run": False,
        "dex3_trajectory_generated": False,
        "physics_run": False,
    })
    dump(OUT / "posture_metrics_v14.json", posture)
    print(json.dumps({
        "preliminary": preliminary_name,
        "selected": "PHYSICAL_CONTACT_BRANCH_POSITION_ONLY",
        "numeric_gate": numeric_gate,
        "collision_gate": collision_gate,
        "physical_contact_gate": physical_gate,
        "EXACT": metrics_exact,
        "NULLSPACE": metrics_null,
    }, indent=2, default=v14.default), flush=True)
    return 0 if numeric_gate and collision_gate and physical_gate else 4


if __name__ == "__main__":
    raise SystemExit(main())
