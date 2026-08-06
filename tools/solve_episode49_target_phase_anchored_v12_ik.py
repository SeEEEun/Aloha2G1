#!/usr/bin/env python3
"""Position-first arm-only temporal IK and staged task-axis audit for v12."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SRC = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
V8 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8/restored_exact_arm_trajectory.npz"
ACTIVE_STAGE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
METHOD = "ALOHA_PRIMARY_TARGET_SIDE_PHASE_ANCHORED_RETARGETING"
R_SCENE_FROM_MODEL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MODEL_ROOT = np.array([0.0, 0.0, 0.7922728583])
G1_ROOT = np.array([0.44514890950197095, -0.35257022755443246, 0.7922728583])
TABLE_Z = 0.795
TABLE_BOUNDS = (-0.02, 0.855, -0.02, 0.74)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--orientation-iterations", type=int, default=3)
    return p.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload) -> None:
    def default(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, Path): return str(value)
        raise TypeError(type(value).__name__)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(payload, indent=2, default=default) + "\n")
    os.replace(tmp, path)


def save_npz(path: Path, **payload) -> None:
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def scene_to_model(position: np.ndarray) -> np.ndarray:
    return (R_SCENE_FROM_MODEL.T @ (np.asarray(position) - G1_ROOT).T).T + MODEL_ROOT


def model_to_scene(position: np.ndarray) -> np.ndarray:
    return (R_SCENE_FROM_MODEL @ (np.asarray(position) - MODEL_ROOT).T).T + G1_ROOT


def quat_rotation_wxyz(q: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def branch_flags(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(np.diff(q, axis=0), axis=1)
    flags = np.zeros(len(q), dtype=bool)
    for i in range(1, len(q)):
        local = np.median(norm[max(0, i - 10) : min(len(norm), i + 9)])
        flags[i] = norm[i - 1] > max(0.15, 8.0 * max(float(local), 1e-5))
    return flags


def evaluate_scene(core, info, q):
    data = mujoco.MjData(info["model"])
    lp, rp, lr, rr = [], [], [], []
    for row in q:
        state = core.frame_state(info, data, row)
        lp.append(model_to_scene(state["left_pos"]))
        rp.append(model_to_scene(state["right_pos"]))
        lr.append(R_SCENE_FROM_MODEL @ quat_rotation_wxyz(state["left_quat"]))
        rr.append(R_SCENE_FROM_MODEL @ quat_rotation_wxyz(state["right_quat"]))
    return np.asarray(lp), np.asarray(rp), np.asarray(lr), np.asarray(rr)


def geom_world_vertices(model, data, geom_id):
    geom_type = int(model.geom_type[geom_id])
    center = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        return center + model.mesh_vert[start : start + count] @ rotation.T
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(model.geom_size[geom_id, 0])
        return center + np.asarray([[sx*r, sy*r, sz*r] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        size = model.geom_size[geom_id]
        local = np.asarray([[sx*size[0], sy*size[1], sz*size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return center + local @ rotation.T
    # Capsule/cylinder axis is local Z in MuJoCo; the support box is exact for
    # the table half-space test and less conservative than geom_rbound.
    radius = float(model.geom_size[geom_id, 0])
    half = float(model.geom_size[geom_id, 1]) if model.geom_size.shape[1] > 1 else 0.0
    local = np.asarray([[sx*radius, sy*radius, sz*(half+radius)] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return center + local @ rotation.T


def collision_audit(core, info, q):
    model = info["model"]
    data = mujoco.MjData(model)
    arm_torso, arm_arm, arm_table, palm_table = [], [], [], []
    pairs = set()
    for frame, row in enumerate(q):
        core.ik.assign_arm_qpos(data, info["stand_qpos"], info["arm_qpos_ids"], row)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        torso_hit = arm_hit = False
        for contact in data.contact:
            bodies = []
            for geom in (contact.geom1, contact.geom2):
                bodies.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or "world")
            relevant = [any(key in b for key in ("shoulder", "elbow", "wrist")) for b in bodies]
            torso = [any(key in b for key in ("torso", "waist")) for b in bodies]
            if any(relevant) and any(torso):
                torso_hit = True; pairs.add("|".join(sorted(bodies)))
            # The arm-only gate deliberately excludes neutral Dex3 placeholder
            # contacts.  Count a cross-side contact only when both colliding
            # bodies belong to the actuated shoulder/elbow/wrist arm chains.
            left_arm = [
                body.startswith("left_")
                and any(key in body for key in ("shoulder", "elbow", "wrist"))
                for body in bodies
            ]
            right_arm = [
                body.startswith("right_")
                and any(key in body for key in ("shoulder", "elbow", "wrist"))
                for body in bodies
            ]
            if any(left_arm) and any(right_arm):
                arm_hit = True; pairs.add("|".join(sorted(bodies)))
        if torso_hit: arm_torso.append(frame)
        if arm_hit: arm_arm.append(frame)
        table_hit = False
        for geom_id in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
            if not any(key in body for key in ("shoulder", "elbow", "wrist")):
                continue
            vertices_scene = model_to_scene(geom_world_vertices(model, data, geom_id))
            inside = ((vertices_scene[:, 0] >= TABLE_BOUNDS[0]) & (vertices_scene[:, 0] <= TABLE_BOUNDS[1]) &
                      (vertices_scene[:, 1] >= TABLE_BOUNDS[2]) & (vertices_scene[:, 1] <= TABLE_BOUNDS[3]))
            if np.any(inside & (vertices_scene[:, 2] < TABLE_Z - 1e-4)):
                table_hit = True
        if table_hit: arm_table.append(frame)
        state = core.frame_state(info, data, row)
        palm_scene_l = model_to_scene(state["left_pos"])
        palm_scene_r = model_to_scene(state["right_pos"])
        if min(palm_scene_l[2], palm_scene_r[2]) < TABLE_Z - 1e-4:
            palm_table.append(frame)
    return {
        "arm_torso_frames": arm_torso,
        "arm_arm_frames": arm_arm,
        "arm_table_frames": arm_table,
        "palm_table_penetration_frames": palm_table,
        "arm_torso_collision_count": len(arm_torso),
        "arm_arm_collision_count": len(arm_arm),
        "arm_table_collision_count": len(arm_table),
        "palm_table_penetration_count": len(palm_table),
        "contact_pairs": sorted(pairs),
        "dex3_placeholder_contacts_excluded": True,
        "table_test": "active arm geom vertices in authoritative scene table half-space",
    }


def candidate_metrics(info, q, left_error, right_error):
    limits = info["joint_limits"]
    violation = (q < limits[:, 0] - 1e-9) | (q > limits[:, 1] + 1e-9)
    branch = branch_flags(q)
    return {
        "frames": len(q),
        "finite": bool(np.isfinite(q).all()),
        "simultaneous_5mm_rate": float(np.mean((left_error <= 0.005) & (right_error <= 0.005))),
        "left_error_mean_mm": float(np.mean(left_error) * 1000),
        "left_error_max_mm": float(np.max(left_error) * 1000),
        "right_error_mean_mm": float(np.mean(right_error) * 1000),
        "right_error_max_mm": float(np.max(right_error) * 1000),
        "joint_limit_violations": int(np.count_nonzero(violation)),
        "minimum_joint_limit_margin_rad": float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
        "branch_discontinuities": int(np.count_nonzero(branch)),
        "branch_discontinuity_frames": np.flatnonzero(branch).tolist(),
        "max_joint_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
        "max_joint_velocity_rad_s": float(np.max(np.abs(np.diff(q, axis=0))) * 30.0),
        "max_joint_acceleration_rad_s2": float(np.max(np.abs(np.diff(q, n=2, axis=0))) * 900.0),
    }


def neutral_tip_proxy_local(core, info, side, finger):
    data = mujoco.MjData(info["model"])
    core.ik.assign_arm_qpos(data, info["stand_qpos"], info["arm_qpos_ids"], info["stand_arm_q"])
    mujoco.mj_forward(info["model"], data)
    wrist_name = f"{side}_wrist_yaw_link"
    body_name = f"{side}_hand_{finger}_" + ("2_link" if finger == "thumb" else "1_link")
    wrist = mujoco.mj_name2id(info["model"], mujoco.mjtObj.mjOBJ_BODY, wrist_name)
    body = mujoco.mj_name2id(info["model"], mujoco.mjtObj.mjOBJ_BODY, body_name)
    wrist_r = data.xmat[wrist].reshape(3, 3)
    offset = np.array([0.0415, 0.003 if side == "left" else -0.003, 0.0])
    palm = data.xpos[wrist] + wrist_r @ offset
    vertices = []
    for geom in range(info["model"].ngeom):
        if int(info["model"].geom_bodyid[geom]) == body:
            vertices.append(geom_world_vertices(info["model"], data, geom))
    if not vertices:
        raise RuntimeError(body_name)
    local = (np.vstack(vertices) - palm) @ wrist_r
    return local[np.argmax(local[:, 0])]


def posture_audit(core, info, exact, nullspace):
    ids = {name: i for i, name in enumerate(info["joint_names"].astype(str).tolist())}
    rows = {}
    for label, q in (("EXACT", exact), ("NULLSPACE", nullspace)):
        rows[label] = {
            "mean_abs_wrist_joint_rad": float(np.mean(np.abs(q[:, [ids["left_wrist_roll_joint"], ids["left_wrist_pitch_joint"], ids["left_wrist_yaw_joint"], ids["right_wrist_roll_joint"], ids["right_wrist_pitch_joint"], ids["right_wrist_yaw_joint"]]]))),
            "mean_elbow_flexion_rad": float(np.mean(q[:, [ids["left_elbow_joint"], ids["right_elbow_joint"]]])),
            "mean_abs_shoulder_roll_rad": float(np.mean(np.abs(q[:, [ids["left_shoulder_roll_joint"], ids["right_shoulder_roll_joint"]]]))),
        }
    diff = np.abs(exact - nullspace)
    return {"candidates": rows, "q_max_abs_difference_rad": float(np.max(diff)), "q_mean_abs_difference_rad": float(np.mean(diff)), "differing_frames": int(np.count_nonzero(np.any(diff > 1e-12, axis=1))), "differing_joints": np.flatnonzero(np.any(diff > 1e-12, axis=0)).tolist(), "posture_term": "validated branch/wrist-neutral objective projected through the 6D position-task Jacobian null space"}


def trajectory_payload(action, timestamps, names, q, base, residual, corrected, achieved, rotations, global_r, global_t, scene_hashes, posture):
    return {
        "optimized_action": action,
        "source_timestamps": timestamps,
        "action_indices": np.arange(990),
        "arm_joint_names": names,
        "g1_arm_q": q,
        "g1_arm_joint_trajectory": q,
        "base_aloha_derived_left_target": base[0],
        "base_aloha_derived_right_target": base[1],
        "global_registration_rotation": global_r,
        "global_registration_translation": global_t,
        "phase_residual_left_translation": residual[0],
        "phase_residual_right_translation": residual[1],
        "corrected_left_position_scene": corrected[0],
        "corrected_right_position_scene": corrected[1],
        "corrected_left_rotation_scene": rotations[0],
        "corrected_right_rotation_scene": rotations[1],
        "achieved_left_position_scene": achieved[0],
        "achieved_right_position_scene": achieved[1],
        "achieved_left_rotation_scene": achieved[2],
        "achieved_right_rotation_scene": achieved[3],
        "g1_root": G1_ROOT,
        "g1_root_forward_offset_m": np.array(0.15),
        "target_scene_hashes_json": np.array(json.dumps(scene_hashes, sort_keys=True)),
        "method": np.array(METHOD),
        "posture_nullspace": np.array(posture),
        "diagnostic_only": np.array(True),
        "real_robot_command_allowed": np.array(False),
        "dex3_fitting_applied": np.array(False),
        "physics_applied": np.array(False),
    }


def main() -> int:
    opt = args()
    import retarget_episode49_optimized_action_to_g1 as core
    from restore_original_pipeline_ep49_current_scene import apply_nullspace_posture

    source = np.load(SRC, allow_pickle=False)
    action, timestamps = source["optimized_action"].copy(), source["timestamp"].copy()
    source.close()
    target = np.load(OUT / "corrected_aloha_targets.npz", allow_pickle=False)
    if not np.array_equal(target["optimized_action"], action) or not np.array_equal(target["timestamps"], timestamps):
        raise RuntimeError("immutable source invariant failed before IK")
    base_l, base_r = target["base_left_position"].copy(), target["base_right_position"].copy()
    corrected_l, corrected_r = target["corrected_left_position"].copy(), target["corrected_right_position"].copy()
    mapped_lrot, mapped_rrot = target["corrected_left_rotation"].copy(), target["corrected_right_rotation"].copy()
    task_lrot, task_rrot = target["task_axis_left_rotation"].copy(), target["task_axis_right_rotation"].copy()
    residual_l, residual_r = target["left_translation_residual"].copy(), target["right_translation_residual"].copy()
    global_r, global_t = target["global_registration_rotation"].copy(), target["global_registration_translation"].copy()
    selected_candidate = str(target["selected_candidate"])
    target.close()
    with np.load(V8, allow_pickle=False) as z:
        if not np.array_equal(z["optimized_action"], action): raise RuntimeError("v8 warm source mismatch")
        warm, names = z["g1_arm_joint_trajectory"].copy(), z["arm_joint_names"].copy()

    info = core.ik.validate_model(core.G1_XML)
    model_l, model_r = scene_to_model(corrected_l), scene_to_model(corrected_r)
    model_mapped_lr = np.einsum("ij,tjk->tik", R_SCENE_FROM_MODEL.T, mapped_lrot)
    model_mapped_rr = np.einsum("ij,tjk->tik", R_SCENE_FROM_MODEL.T, mapped_rrot)
    tar_position = {"lp": model_l, "rp": model_r, "lr": model_mapped_lr, "rr": model_mapped_rr}

    print("[V12_IK] computing position continuation seed", flush=True)
    seed = core.position_seed(info, tar_position, warm[0])
    seed_eval = core.evaluate(info, tar_position, seed)
    print("[V12_IK] computing whole-trajectory position solve", flush=True)
    temporal = core.temporal_solve(info, tar_position, seed, warm[0], 0.0, opt.iterations)
    temporal_eval = core.evaluate(info, tar_position, temporal)
    warm_eval = core.evaluate(info, tar_position, warm)
    candidates = []
    for label, q, ev in (("v8_warm_start_diagnostic", warm, warm_eval), ("framewise_continuation", seed, seed_eval), ("whole_trajectory_temporal", temporal, temporal_eval)):
        candidates.append({"name": label, "simultaneous_5mm_rate": float(np.mean((ev["le"] <= .005) & (ev["re"] <= .005))), "mean_bimanual_error_mm": float(np.mean(ev["le"] + ev["re"]) * 1000), "max_error_mm": float(max(np.max(ev["le"]), np.max(ev["re"])) * 1000)})
    selected_label, exact, exact_ev = min(
        (("framewise_continuation", seed, seed_eval), ("whole_trajectory_temporal", temporal, temporal_eval)),
        key=lambda x: (
            -float(np.mean((x[2]["le"] <= .005) & (x[2]["re"] <= .005))),
            float(np.mean(x[2]["le"] + x[2]["re"])),
        ),
    )
    exact_scene = evaluate_scene(core, info, exact)
    exact_le = np.linalg.norm(exact_scene[0] - corrected_l, axis=1)
    exact_re = np.linalg.norm(exact_scene[1] - corrected_r, axis=1)

    nullspace = apply_nullspace_posture(exact, model_l, model_r, warm[0], info)
    null_scene = evaluate_scene(core, info, nullspace)
    null_le = np.linalg.norm(null_scene[0] - corrected_l, axis=1)
    null_re = np.linalg.norm(null_scene[1] - corrected_r, axis=1)

    metrics_exact = candidate_metrics(info, exact, exact_le, exact_re)
    metrics_null = candidate_metrics(info, nullspace, null_le, null_re)
    collision_exact = collision_audit(core, info, exact)
    collision_null = collision_audit(core, info, nullspace)
    position_numeric_gate = all(row["simultaneous_5mm_rate"] >= .99 and row["joint_limit_violations"] == 0 and row["branch_discontinuities"] == 0 and row["finite"] for row in (metrics_exact, metrics_null))
    collision_gate = all(row[key] == 0 for row in (collision_exact, collision_null) for key in ("arm_torso_collision_count", "arm_arm_collision_count", "arm_table_collision_count", "palm_table_penetration_count"))

    left_anchors = json.loads((OUT / "target_left_phase_anchors.json").read_text())["anchors"]
    right_anchors = json.loads((OUT / "target_right_phase_anchors.json").read_text())["anchors"]
    phone_npz = np.load(OUT / "target_phone_pose_trajectory.npz", allow_pickle=False)
    desired_phone = phone_npz["phone_on_charger_pose"].copy()
    phone_npz.close()
    anchor_metrics = {"gate_mm": 5.0, "future_finger_proxies_are_diagnostic_only": True, "candidates": {}}
    future_c_local = np.asarray(right_anchors["accessory_grasp"]["future_right_C_static_neutral_proxy_local_m"])
    future_c_target = np.asarray(right_anchors["accessory_grasp"]["future_right_C_proxy_target_m"])
    left_thumb_local = neutral_tip_proxy_local(core, info, "left", "thumb")
    left_index_local = neutral_tip_proxy_local(core, info, "left", "index")
    for label, scene_values in (("EXACT", exact_scene), ("NULLSPACE", null_scene)):
        lp, rp, lr, rr = scene_values
        phone_grasp = np.linalg.norm(lp[169] - np.asarray(left_anchors["phone_grasp"]["position_m"]))
        right_grasp = np.linalg.norm(rp[319] - np.asarray(right_anchors["accessory_grasp"]["position_m"]))
        charger_palm = np.linalg.norm(lp[523] - np.asarray(left_anchors["charger"]["position_m"]))
        intended_center = desired_phone[:3, 3] + (lp[523] - np.asarray(left_anchors["charger"]["position_m"]))
        actual_future_c = rp[319] + rr[319] @ future_c_local
        thumb = lp[169] + lr[169] @ left_thumb_local
        index = lp[169] + lr[169] @ left_index_local
        anchor_metrics["candidates"][label] = {
            "left_palm_phone_anchor_error_mm": float(phone_grasp * 1000),
            "right_palm_pregrasp_anchor_error_mm": float(right_grasp * 1000),
            "left_palm_charger_anchor_error_mm": float(charger_palm * 1000),
            "position_only_intended_carrier_phone_center_to_pad_mm": float(np.linalg.norm(intended_center - desired_phone[:3, 3]) * 1000),
            "future_right_C_static_proxy_to_ring_surface_mm": float(np.linalg.norm(actual_future_c - future_c_target) * 1000),
            "future_left_A_B_static_proxy_separation_mm": float(np.linalg.norm(thumb - index) * 1000),
            "future_left_A_thumb_position_m": thumb,
            "future_left_B_index_position_m": index,
            "future_right_C_proxy_position_m": actual_future_c,
        }

    anchor_gate = all(max(row["left_palm_phone_anchor_error_mm"], row["right_palm_pregrasp_anchor_error_mm"], row["position_only_intended_carrier_phone_center_to_pad_mm"]) <= 5.0 for row in anchor_metrics["candidates"].values())
    position_gate = position_numeric_gate and collision_gate and anchor_gate

    orientation_sweep = []
    selected_orientation = None
    if position_gate:
        stages = [
            ("O1_MAPPED_ALOHA_RELATIVE_ROTATION", mapped_lrot, mapped_rrot),
            ("O2_LEFT_PHONE_AND_CHARGER_TASK_AXES", task_lrot, mapped_rrot),
            ("O3_RIGHT_ACCESSORY_TASK_AXIS", task_lrot, task_rrot),
        ]
        current = exact.copy()
        for stage_name, left_rot, right_rot in stages:
            tar = {"lp": model_l, "rp": model_r, "lr": np.einsum("ij,tjk->tik", R_SCENE_FROM_MODEL.T, left_rot), "rr": np.einsum("ij,tjk->tik", R_SCENE_FROM_MODEL.T, right_rot)}
            best_stage = None
            # Start close to the validated position-only solution.  The prior
            # sweep began at 7.5e-4 and skipped the narrow feasible interval
            # where mapped source rotation improves without sacrificing the
            # 5 mm positional gate.
            for weight in (0.00005, 0.0001, 0.0002, 0.0004, 0.00075, 0.0015, 0.003):
                q_stage = core.temporal_solve(info, tar, current, warm[0], weight, opt.orientation_iterations)
                ev = core.evaluate(info, tar, q_stage)
                scene_values = evaluate_scene(core, info, q_stage)
                le = np.linalg.norm(scene_values[0] - corrected_l, axis=1); re = np.linalg.norm(scene_values[1] - corrected_r, axis=1)
                metric = candidate_metrics(info, q_stage, le, re); collision = collision_audit(core, info, q_stage)
                row = {"stage": stage_name, "orientation_weight": weight, "simultaneous_5mm_rate": metric["simultaneous_5mm_rate"], "left_orientation_mean_deg": float(np.degrees(np.mean(ev["lo"]))), "right_orientation_mean_deg": float(np.degrees(np.mean(ev["ro"]))), "joint_limit_violations": metric["joint_limit_violations"], "branch_discontinuities": metric["branch_discontinuities"], "collision_counts": {k: collision[k] for k in ("arm_torso_collision_count", "arm_arm_collision_count", "arm_table_collision_count", "palm_table_penetration_count")}}
                row["gate_pass"] = bool(row["simultaneous_5mm_rate"] >= .99 and row["joint_limit_violations"] == 0 and row["branch_discontinuities"] == 0 and all(v == 0 for v in row["collision_counts"].values()))
                orientation_sweep.append(row)
                if row["gate_pass"]:
                    best_stage = (q_stage, scene_values, left_rot, right_rot, row)
            if best_stage is None:
                break
            current = best_stage[0]
            selected_orientation = (stage_name, *best_stage)

    scene_hashes = {str(ACTIVE_STAGE.resolve()): sha(ACTIVE_STAGE), str(SCENE.resolve()): sha(SCENE)}
    common_args = (action, timestamps, names, (base_l, base_r), (residual_l, residual_r), (corrected_l, corrected_r), (mapped_lrot, mapped_rrot), global_r, global_t, scene_hashes)
    save_npz(OUT / "position_only_exact_arm_trajectory.npz", **trajectory_payload(common_args[0], common_args[1], common_args[2], exact, common_args[3], common_args[4], common_args[5], exact_scene, common_args[6], common_args[7], common_args[8], common_args[9], False))
    save_npz(OUT / "position_only_nullspace_arm_trajectory.npz", **trajectory_payload(common_args[0], common_args[1], common_args[2], nullspace, common_args[3], common_args[4], common_args[5], null_scene, common_args[6], common_args[7], common_args[8], common_args[9], True))
    if selected_orientation is not None:
        stage_name, q_stage, stage_scene, left_rot, right_rot, row = selected_orientation
        payload = trajectory_payload(action, timestamps, names, q_stage, (base_l, base_r), (residual_l, residual_r), (corrected_l, corrected_r), stage_scene, (left_rot, right_rot), global_r, global_t, scene_hashes, False)
        payload["orientation_stage"] = np.array(stage_name)
        save_npz(OUT / "task_axis_arm_trajectory.npz", **payload)

    posture = posture_audit(core, info, exact, nullspace)
    dump(OUT / "posture_metrics.json", posture)
    dump(OUT / "collision_breakdown.json", {"EXACT": collision_exact, "NULLSPACE": collision_null, "position_only_collision_gate_pass": collision_gate})
    dump(OUT / "anchor_metrics.json", {"constructed_target": json.loads((OUT / "anchor_metrics.json").read_text()), "achieved_position_only": anchor_metrics, "position_only_anchor_gate_pass": anchor_gate})
    status = "POSITION_ONLY_VALIDATED" if position_gate else ("BLOCKED_COLLISION" if position_numeric_gate and not collision_gate else "BLOCKED_POSITION_ONLY_IK")
    orientation_status = "NOT_RUN_POSITION_GATE_FAILED" if not position_gate else ("TASK_AXIS_ORIENTATION_VALIDATED" if selected_orientation is not None else "BLOCKED_TASK_AXIS_ORIENTATION")
    dump(OUT / "ik_metrics.json", {
        "status": status,
        "position_only_gate_pass": position_gate,
        "selected_position_candidate": selected_label,
        "position_candidate_audit": candidates,
        "cartesian_position_targets_array_identical_exact_nullspace": True,
        "EXACT": metrics_exact,
        "NULLSPACE": metrics_null,
        "orientation_status": orientation_status,
        "orientation_sweep": orientation_sweep,
        "selected_orientation_stage": selected_orientation[0] if selected_orientation else None,
        "selected_phase_residual_candidate": selected_candidate,
        "no_dex3": True, "no_physics": True, "no_hardware": True,
    })
    print(json.dumps({"status": status, "orientation_status": orientation_status, "EXACT": metrics_exact, "NULLSPACE": metrics_null, "collision_gate": collision_gate, "anchor_gate": anchor_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
