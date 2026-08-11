#!/usr/bin/env python3
"""Build Episode-49 v13 targets from active G1/Dex3 contact geometry.

This is a kinematic, arm-only target builder.  It never edits the immutable
ALOHA action/phase library or the authoritative Isaac scene.  Dex3 is sampled
only to prove that an arm carrier pose has a locally reachable physical contact;
the diagnostic finger configurations are never expanded into a trajectory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.optimize import differential_evolution, least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_target_phase_anchored_v12 as v12
import retarget_episode49_optimized_action_to_g1 as core

OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_physical_contact_anchored_v13"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
V12_RENDERFIX = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12_renderfix"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
ACTIVE_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
REGISTRATION = ROOT / "configs/magsafe_task_frame_registration.sim.json"
ACTIVE_G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)

METHOD = "ALOHA_PRIMARY_PHYSICAL_CONTACT_ANCHORED_ARM_RETARGETING"
SCALE = 0.42
LAG = 7
FPS = 30.0
KNOTS = v12.KNOTS.copy()
R_SCENE_FROM_MODEL = v12.R_SCENE_FROM_MODEL.copy()
MODEL_ROOT = v12.MODEL_ROOT.copy()
G1_ROOT = v12.G1_ROOT.copy()
PALM_OFFSET = {
    "left": np.array([0.0415, 0.003, 0.0]),
    "right": np.array([0.0415, -0.003, 0.0]),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(value, indent=2, default=default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz(path: Path, **value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **value)
    os.replace(tmp, path)


def make_transform(rotation=np.eye(3), position=np.zeros(3)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, float)
    result[:3, 3] = np.asarray(position, float)
    return result


def invert(value: np.ndarray) -> np.ndarray:
    r = value[:3, :3]
    return make_transform(r.T, -r.T @ value[:3, 3])


def normalize(value, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    value = np.asarray(value, float)
    n = float(np.linalg.norm(value))
    return value / n if n > 1e-12 else normalize(fallback)


def model_to_scene(position: np.ndarray) -> np.ndarray:
    return (R_SCENE_FROM_MODEL @ (np.asarray(position) - MODEL_ROOT).T).T + G1_ROOT


def body_id(model, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise RuntimeError(f"missing body {name}")
    return result


def joint_id(model, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if result < 0:
        raise RuntimeError(f"missing joint {name}")
    return result


def usd_mesh_points(stage: Usd.Stage, link: str, collision=True) -> np.ndarray:
    group = "collisions" if collision else "visuals"
    path = f"/World/G1/Asset/{link}/{group}/{link}/mesh"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing active USD mesh {path}")
    return np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)


def cap_contact_proxy(points: np.ndarray, axis=0, sign=1.0, band=0.001) -> np.ndarray:
    coordinate = sign * points[:, axis]
    selected = points[coordinate >= coordinate.max() - band]
    if len(selected) < 10:
        raise RuntimeError("active collision contact patch is empty")
    return selected.mean(axis=0)


def hull_points(points: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(points, 8), axis=0)
    return unique[ConvexHull(unique).vertices]


def active_joint_limit(stage: Usd.Stage, joint_name: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(f"/World/G1/Asset/joints/{joint_name}")
    if not prim.IsValid() or not prim.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError(f"missing active USD revolute joint {joint_name}")
    lo = float(prim.GetAttribute("physics:lowerLimit").Get())
    hi = float(prim.GetAttribute("physics:upperLimit").Get())
    return np.radians([lo, hi])


def hand_geometry_audit(stage: Usd.Stage, info: dict) -> tuple[dict, dict]:
    specifications = {
        "left_A": ("left", "thumb", "left_hand_thumb_2_link", [
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"
        ]),
        "left_B": ("left", "index", "left_hand_index_1_link", [
            "left_hand_index_0_joint", "left_hand_index_1_joint"
        ]),
        "right_C": ("right", "middle", "right_hand_middle_1_link", [
            "right_hand_middle_0_joint", "right_hand_middle_1_joint"
        ]),
    }
    records, runtime = {}, {}
    model = info["model"]
    contact_caps = {
        # The thumb distal mesh is longitudinal in local -Y; index/middle are
        # longitudinal in local +X.  These are actual collision-mesh end caps,
        # not inferred finger lengths or ALOHA offsets.
        "left_A": (1, -1.0),
        "left_B": (0, 1.0),
        "right_C": (0, 1.0),
    }
    for label, (side, finger, link, joints) in specifications.items():
        points = usd_mesh_points(stage, link, True)
        cap_axis, cap_sign = contact_caps[label]
        proxy = cap_contact_proxy(points, cap_axis, cap_sign)
        normal_local = np.zeros(3)
        normal_local[cap_axis] = cap_sign
        hull = hull_points(points)
        usd_limits = np.asarray([active_joint_limit(stage, name) for name in joints])
        mujoco_limits = np.asarray([model.jnt_range[joint_id(model, name)] for name in joints])
        cross = float(np.max(np.abs(usd_limits - mujoco_limits)))
        records[label] = {
            "side": side,
            "project_mapping": {"left_A": "thumb", "left_B": "index", "right_C": "middle"}[label],
            "distal_link": link,
            "joint_names": joints,
            "active_usd_joint_limits_rad": usd_limits,
            "mujoco_diagnostic_joint_limits_rad": mujoco_limits,
            "maximum_limit_crosscheck_difference_rad": cross,
            "contact_proxy_definition": f"centroid of active distal collision-mesh axis {cap_axis} sign {cap_sign:+.0f} cap within 1 mm of support maximum",
            "distal_link_local_contact_proxy_m": proxy,
            "distal_link_local_contact_normal": normal_local,
            "collision_mesh_extent_m": [points.min(axis=0), points.max(axis=0)],
            "collision_hull_vertex_count": len(hull),
            "workspace_scale_applied_to_geometry": False,
        }
        runtime[label] = {
            "side": side,
            "finger": finger,
            "link": link,
            "joints": joints,
            "qpos": np.asarray([model.jnt_qposadr[joint_id(model, name)] for name in joints], int),
            "limits": usd_limits,
            "proxy": proxy,
            "normal_local": normal_local,
            "hull": hull,
        }

    for side in ("left", "right"):
        link = f"{side}_hand_palm_link"
        visual = usd_mesh_points(stage, link, False)
        wrist_world = v12.active_transform(stage, f"/World/G1/Asset/{side}_wrist_yaw_link")
        palm_world = v12.active_transform(stage, f"/World/G1/Asset/{link}")
        palm_from_wrist = invert(wrist_world) @ palm_world
        records[f"{side}_physical_palm"] = {
            "link": link,
            "visual_mesh_extent_m": [visual.min(axis=0), visual.max(axis=0)],
            "visual_mesh_vertex_count": len(visual),
            "collision_mesh_present": False,
            "palm_link_from_wrist_zero_pose": palm_from_wrist,
            "note": "active USD has palm visual mesh but no separate palm collision mesh; table audit uses the physical visual extent conservatively",
        }
        runtime[f"{side}_palm_visual"] = hull_points(visual)
        runtime[f"{side}_palm_from_wrist"] = palm_from_wrist
    audit = {
        "status": "ACTIVE_G1_DEX3_CONTACT_FRAMES_AUDITED",
        "active_stage": str(ACTIVE_SCENE.resolve()),
        "active_g1_asset": str(ACTIVE_G1_USD),
        "active_g1_asset_sha256": sha(ACTIVE_G1_USD),
        "frames": records,
        "wrist_link_frames": {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"},
        "existing_palm_proxy_local_m": PALM_OFFSET,
        "dex3_trajectory_generated": False,
        "simulation_only": True,
    }
    return audit, runtime


def relative_hand_state(info: dict, runtime: dict, arm_q: np.ndarray, label: str, finger_q: np.ndarray):
    model = info["model"]
    data = mujoco.MjData(model)
    data.qpos[:] = info["stand_qpos"]
    data.qpos[info["arm_qpos_ids"]] = arm_q
    spec = runtime[label]
    data.qpos[spec["qpos"]] = finger_q
    mujoco.mj_forward(model, data)
    wrist = body_id(model, f"{spec['side']}_wrist_yaw_link")
    distal = body_id(model, spec["link"])
    wr = data.xmat[wrist].reshape(3, 3)
    dr = data.xmat[distal].reshape(3, 3)
    proxy_model = data.xpos[distal] + dr @ spec["proxy"]
    proxy_wrist = wr.T @ (proxy_model - data.xpos[wrist])
    normal_wrist = wr.T @ (dr @ spec["normal_local"])
    return proxy_wrist, normal_wrist, wr.T @ dr, wr.T @ (data.xpos[distal] - data.xpos[wrist])


def sample_envelopes(info: dict, runtime: dict) -> dict:
    arm = info["stand_arm_q"]
    left_limits = np.vstack((runtime["left_A"]["limits"], runtime["left_B"]["limits"]))
    left_samples = qmc.Sobol(5, scramble=False).random_base2(13)
    left_q = left_limits[:, 0] + left_samples * (left_limits[:, 1] - left_limits[:, 0])
    left_a, left_b = [], []
    for row in left_q:
        left_a.append(relative_hand_state(info, runtime, arm, "left_A", row[:3])[0])
        left_b.append(relative_hand_state(info, runtime, arm, "left_B", row[3:])[0])
    left_a, left_b = np.asarray(left_a), np.asarray(left_b)

    right_limits = runtime["right_C"]["limits"]
    right_samples = qmc.Sobol(2, scramble=False).random_base2(12)
    right_q = right_limits[:, 0] + right_samples * (right_limits[:, 1] - right_limits[:, 0])
    right_c = np.asarray([
        relative_hand_state(info, runtime, arm, "right_C", row)[0] for row in right_q
    ])
    save_npz(
        OUT / "left_ab_reach_envelope.npz",
        left_dex3_ab_q=left_q,
        left_A_contact_proxy_from_wrist=left_a,
        left_B_contact_proxy_from_wrist=left_b,
        wrist_frame=np.array("left_wrist_yaw_link"),
        source_model=np.array(str(ACTIVE_G1_USD)),
        diagnostic_only=np.array(True),
    )
    save_npz(
        OUT / "right_c_reach_envelope.npz",
        right_dex3_c_q=right_q,
        right_C_contact_proxy_from_wrist=right_c,
        wrist_frame=np.array("right_wrist_yaw_link"),
        source_model=np.array(str(ACTIVE_G1_USD)),
        diagnostic_only=np.array(True),
    )
    result = {
        "status": "STATIC_CONTACT_REACH_ENVELOPES_COMPUTED",
        "left_AB": {
            "sample_count": len(left_q),
            "A_wrist_local_bounds_m": [left_a.min(axis=0), left_a.max(axis=0)],
            "B_wrist_local_bounds_m": [left_b.min(axis=0), left_b.max(axis=0)],
            "A_B_aperture_range_m": [
                float(np.min(np.linalg.norm(left_a - left_b, axis=1))),
                float(np.max(np.linalg.norm(left_a - left_b, axis=1))),
            ],
        },
        "right_C": {
            "sample_count": len(right_q),
            "wrist_local_bounds_m": [right_c.min(axis=0), right_c.max(axis=0)],
            "reach_norm_range_m": [
                float(np.min(np.linalg.norm(right_c, axis=1))),
                float(np.max(np.linalg.norm(right_c, axis=1))),
            ],
        },
        "arm_pose_fixed": True,
        "active_joint_limits_used": True,
        "active_collision_contact_patches_used": True,
        "workspace_scale_applied_to_hand_geometry": False,
        "dex3_trajectory_generated": False,
    }
    dump(OUT / "dex3_static_contact_reach_envelope.json", result)
    return result


def scene_wrist_pose(info: dict, arm_q: np.ndarray, side: str):
    model, data = info["model"], mujoco.MjData(info["model"])
    data.qpos[:] = info["stand_qpos"]
    data.qpos[info["arm_qpos_ids"]] = arm_q
    mujoco.mj_forward(model, data)
    wrist = body_id(model, f"{side}_wrist_yaw_link")
    r = R_SCENE_FROM_MODEL @ data.xmat[wrist].reshape(3, 3)
    p = model_to_scene(data.xpos[wrist])
    palm = p + r @ PALM_OFFSET[side]
    return p, r, palm


def active_arm_max_palm_reach(info: dict, side: str) -> dict:
    """Numerically maximize shoulder-to-palm reach using active arm limits/FK."""
    model = info["model"]
    data = mujoco.MjData(model)
    offset = 0 if side == "left" else 7
    shoulder_id = body_id(model, f"{side}_shoulder_pitch_link")
    limits = info["joint_limits"][offset:offset + 7]

    def objective(side_q):
        q = info["stand_arm_q"].copy()
        q[offset:offset + 7] = side_q
        state = core.frame_state(info, data, q)
        palm = state[f"{side}_pos"]
        return -float(np.linalg.norm(palm - data.xpos[shoulder_id]))

    result = differential_evolution(
        objective, [tuple(row) for row in limits], seed=13 if side == "left" else 17,
        maxiter=120, popsize=12, tol=1e-9, polish=True, workers=1,
        updating="immediate",
    )
    q = info["stand_arm_q"].copy()
    q[offset:offset + 7] = result.x
    core.frame_state(info, data, q)
    shoulder_scene = model_to_scene(data.xpos[shoulder_id])
    return {
        "side": side,
        "maximum_shoulder_to_palm_reach_m": -float(result.fun),
        "maximizing_arm_q_rad": result.x,
        "active_shoulder_scene_position_m": shoulder_scene,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "joint_limits_used": limits,
        "active_model_fk_used": True,
    }


def nearest_box_surface(point: np.ndarray, pose: np.ndarray, dimensions: np.ndarray):
    """Return signed distance and the closest physical OBB surface point."""
    local = pose[:3, :3].T @ (np.asarray(point) - pose[:3, 3])
    half = 0.5 * np.asarray(dimensions)
    excess = np.abs(local) - half
    if np.any(excess > 0.0):
        nearest_local = np.clip(local, -half, half)
        signed_distance = float(np.linalg.norm(local - nearest_local))
    else:
        face_margin = half - np.abs(local)
        axis = int(np.argmin(face_margin))
        nearest_local = local.copy()
        nearest_local[axis] = math.copysign(half[axis], local[axis] if local[axis] else 1.0)
        signed_distance = -float(face_margin[axis])
    nearest_world = pose[:3, 3] + pose[:3, :3] @ nearest_local
    return signed_distance, nearest_world, local


def active_finger_phone_reach_audit(
    info: dict, runtime: dict, label: str, phone_pose: np.ndarray,
    phone_dimensions: np.ndarray,
) -> dict:
    """Global static FK search: can one active contact cap reach the phone box?"""
    side = runtime[label]["side"]
    offset = 0 if side == "left" else 7
    model = info["model"]
    data = mujoco.MjData(model)
    spec = runtime[label]
    distal_id = body_id(model, spec["link"])
    limits = np.vstack((info["joint_limits"][offset:offset + 7], spec["limits"]))

    def state(x):
        data.qpos[:] = info["stand_qpos"]
        arm_q = info["stand_arm_q"].copy()
        arm_q[offset:offset + 7] = x[:7]
        data.qpos[info["arm_qpos_ids"]] = arm_q
        data.qpos[spec["qpos"]] = x[7:]
        mujoco.mj_forward(model, data)
        distal_r = data.xmat[distal_id].reshape(3, 3)
        proxy_model = data.xpos[distal_id] + distal_r @ spec["proxy"]
        proxy_scene = model_to_scene(proxy_model)
        signed, nearest, local = nearest_box_surface(proxy_scene, phone_pose, phone_dimensions)
        return signed, nearest, local, proxy_scene, arm_q

    def objective(x):
        signed, _, _, _, _ = state(x)
        return signed * signed + 1e-10 * float(np.dot(x, x))

    result = differential_evolution(
        objective, [tuple(row) for row in limits],
        seed={"left_A": 41, "left_B": 43, "right_C": 47}.get(label, 53),
        maxiter=500, popsize=18, tol=1e-10, polish=True, workers=1,
        updating="immediate",
    )
    signed, nearest, local, proxy, arm_q = state(result.x)
    return {
        "contact_label": label,
        "minimum_absolute_surface_gap_m": abs(signed),
        "signed_surface_distance_m": signed,
        "contact_proxy_scene_position_m": proxy,
        "nearest_phone_surface_position_m": nearest,
        "distance_vector_to_surface_m": nearest - proxy,
        "phone_local_contact_proxy_m": local,
        "diagnostic_arm_q_rad": arm_q[offset:offset + 7],
        "diagnostic_finger_q_rad": result.x[7:],
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "active_arm_and_finger_joint_limits_used": True,
        "workspace_scale_applied_to_hand_geometry": False,
        "diagnostic_keyframe_only": True,
    }


def left_contact_candidate(
    info, runtime, phone_pose, phone_dims, old_anchor, approach,
    palm_y_upper=-0.055, expanded_rotation_seeds=False,
    direction_weight=2.0, reach_weight=30.0, arm_reach=None,
):
    """Static A+B carrier solve; diagnostic q is not a trajectory."""
    center, rotation = phone_pose[:3, 3], phone_pose[:3, :3]
    x_lo = -0.5 * phone_dims[0] + 0.001
    x_hi = x_lo + 0.030
    z_lo = -0.020
    z_hi = 0.5 * phone_dims[2] - 0.002
    y_back, y_front = 0.5 * phone_dims[1], -0.5 * phone_dims[1]
    limits = np.vstack((runtime["left_A"]["limits"], runtime["left_B"]["limits"]))
    if arm_reach is None:
        arm_reach = active_arm_max_palm_reach(info, "left")
    shoulder = np.asarray(arm_reach["active_shoulder_scene_position_m"], float)
    arm_reach_max = float(arm_reach["maximum_shoulder_to_palm_reach_m"])
    # Build a proper right-handed wrist basis.  The old construction placed
    # ``approach`` next to an unprojected world-up vector, so the two columns
    # were generally not orthogonal.  Feeding that matrix to the contact FK
    # silently sheared fingertip positions and corrupted the arm-reach test.
    basis_x = normalize(approach)
    basis_reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(basis_x, basis_reference))) > 0.95:
        basis_reference = np.array([0.0, 1.0, 0.0])
    basis_y = normalize(np.cross(basis_reference, basis_x))
    basis_z = normalize(np.cross(basis_x, basis_y))
    base_rotation = np.column_stack((basis_x, basis_y, basis_z))
    if not np.allclose(base_rotation.T @ base_rotation, np.eye(3), atol=1e-12):
        raise RuntimeError("left contact base rotation is not orthonormal")
    if not np.isclose(np.linalg.det(base_rotation), 1.0, atol=1e-12):
        raise RuntimeError("left contact base rotation is not a proper SO(3) rotation")
    candidates = []
    seeds = [
        np.r_[np.mean(limits, axis=1), np.zeros(3), old_anchor, x_lo + 0.006, 0.020],
        np.r_[limits[:, 0] * .25 + limits[:, 1] * .75, np.zeros(3), old_anchor, x_lo + 0.018, z_hi - .002],
        np.r_[limits[:, 0] * .75 + limits[:, 1] * .25, np.zeros(3), old_anchor, x_lo + 0.012, 0.0],
    ]
    if expanded_rotation_seeds:
        for rv in (
            [math.pi/2, 0, 0], [0, math.pi/2, 0], [0, 0, math.pi/2],
        ):
            seeds.append(np.r_[np.mean(limits, axis=1), rv, old_anchor, x_lo + .012, 0.0])
        # Seed the nonlinear solve with actual high-reach A/B configurations
        # from the active-model Sobol envelope.  Mean finger q folds the hand
        # and can create a false arm-reach blocker at the distant charger.
        envelope = np.load(OUT / "left_ab_reach_envelope.npz")
        envelope_q = envelope["left_dex3_ab_q"]
        envelope_a = envelope["left_A_contact_proxy_from_wrist"]
        envelope_b = envelope["left_B_contact_proxy_from_wrist"]
        envelope_aperture = np.linalg.norm(envelope_a - envelope_b, axis=1)
        envelope_mid_reach = np.linalg.norm(
            0.5 * (envelope_a + envelope_b) - PALM_OFFSET["left"], axis=1
        )
        palm_reach_seed = shoulder + normalize(center - shoulder) * .415
        for desired_aperture in (0.008, 0.025, 0.035, 0.050, 0.070):
            seed_score = 4.0 * np.abs(envelope_aperture - desired_aperture) - envelope_mid_reach
            q_seed = envelope_q[int(np.argmin(seed_score))]
            seeds.append(np.r_[q_seed, np.zeros(3), palm_reach_seed, x_lo + .012, 0.0])
    lower = np.r_[limits[:, 0], [-math.pi] * 3, [0.20, -0.18, 0.800], x_lo, z_lo]
    # The target phone is 0.423 m in front of the active shoulder.  A carrier
    # palm placed at the phone center is not an arm-reachable carrier even if a
    # folded diagnostic finger happens to touch it.  Require the palm to stay
    # on the robot side of the phone and explicitly penalize arm-chain excess.
    upper = np.r_[limits[:, 1], [math.pi] * 3, [0.62, palm_y_upper, 1.08], x_hi, z_hi]
    contact_modes = [
        ("BROAD_BACK_FRONT", 1.0),
        ("BROAD_FRONT_BACK", -1.0),
        ("A_EDGE_B_FRONT", -1.0),
        ("A_EDGE_B_BACK", 1.0),
        ("B_EDGE_A_FRONT", -1.0),
        ("B_EDGE_A_BACK", 1.0),
    ]
    for contact_mode, assignment in contact_modes:
        for seed_id, seed in enumerate(seeds):
            seed = np.clip(seed, lower + 1e-7, upper - 1e-7)

            def residual(x):
                q = x[:5]
                wrist_r = Rotation.from_rotvec(x[5:8]).as_matrix() @ base_rotation
                palm = x[8:11]
                wrist = palm - wrist_r @ PALM_OFFSET["left"]
                a, na, _, _ = relative_hand_state(info, runtime, info["stand_arm_q"], "left_A", q[:3])
                b, nb, _, _ = relative_hand_state(info, runtime, info["stand_arm_q"], "left_B", q[3:])
                pa, pb = wrist + wrist_r @ a, wrist + wrist_r @ b
                na, nb = wrist_r @ na, wrist_r @ nb
                if contact_mode.startswith("BROAD"):
                    target_a_local = np.array([x[11], assignment * y_back, x[12]])
                    target_b_local = np.array([x[11], assignment * y_front, x[12]])
                    desired_na = -assignment * rotation[:, 1]
                    desired_nb = assignment * rotation[:, 1]
                elif contact_mode.startswith("A_EDGE"):
                    target_a_local = np.array([-0.5 * phone_dims[0], 0.0, x[12]])
                    target_b_local = np.array([x[11], assignment * y_back, x[12]])
                    desired_na = rotation[:, 0]
                    desired_nb = -assignment * rotation[:, 1]
                else:
                    target_a_local = np.array([x[11], assignment * y_back, x[12]])
                    target_b_local = np.array([-0.5 * phone_dims[0], 0.0, x[12]])
                    desired_na = -assignment * rotation[:, 1]
                    desired_nb = rotation[:, 0]
                target_a = center + rotation @ target_a_local
                target_b = center + rotation @ target_b_local
                direction = normalize(0.5 * (pa + pb) - palm)
                reach_excess = max(0.0, float(np.linalg.norm(palm - shoulder)) - arm_reach_max)
                return np.r_[
                    180.0 * (pa - target_a), 180.0 * (pb - target_b),
                    .5 * (na - desired_na), .5 * (nb - desired_nb),
                    .45 * (palm - old_anchor), direction_weight * (direction - normalize(approach)),
                    [reach_weight * reach_excess],
                    .01 * (q - np.mean(limits, axis=1)),
                ]

            solution = least_squares(
                residual, seed, bounds=(lower, upper), max_nfev=300,
                ftol=1e-11, xtol=1e-11, gtol=1e-11,
            )
            x = solution.x
            q = x[:5]
            wrist_r = Rotation.from_rotvec(x[5:8]).as_matrix() @ base_rotation
            if not np.allclose(wrist_r.T @ wrist_r, np.eye(3), atol=1e-9):
                raise RuntimeError("optimized wrist rotation is not orthonormal")
            if not np.isclose(np.linalg.det(wrist_r), 1.0, atol=1e-9):
                raise RuntimeError("optimized wrist rotation is not a proper SO(3) rotation")
            palm = x[8:11]
            wrist = palm - wrist_r @ PALM_OFFSET["left"]
            a, na, _, _ = relative_hand_state(info, runtime, info["stand_arm_q"], "left_A", q[:3])
            b, nb, _, _ = relative_hand_state(info, runtime, info["stand_arm_q"], "left_B", q[3:])
            pa, pb = wrist + wrist_r @ a, wrist + wrist_r @ b
            if contact_mode.startswith("BROAD"):
                ta = center + rotation @ np.array([x[11], assignment * y_back, x[12]])
                tb = center + rotation @ np.array([x[11], assignment * y_front, x[12]])
            elif contact_mode.startswith("A_EDGE"):
                ta = center + rotation @ np.array([-0.5 * phone_dims[0], 0.0, x[12]])
                tb = center + rotation @ np.array([x[11], assignment * y_back, x[12]])
            else:
                ta = center + rotation @ np.array([x[11], assignment * y_back, x[12]])
                tb = center + rotation @ np.array([-0.5 * phone_dims[0], 0.0, x[12]])
            gaps = [float(np.linalg.norm(pa - ta)), float(np.linalg.norm(pb - tb))]
            # Conservative active palm visual table test.
            palm_link_r = runtime["left_palm_from_wrist"][:3, :3]
            palm_link_p = runtime["left_palm_from_wrist"][:3, 3]
            palm_vertices = wrist + (wrist_r @ (palm_link_p[:, None] + palm_link_r @ runtime["left_palm_visual"].T)).T
            palm_min_z = float(palm_vertices[:, 2].min())
            shoulder_distance = float(np.linalg.norm(palm - shoulder))
            valid = bool(max(gaps) <= .005 and palm_min_z >= .795 - 1e-5 and shoulder_distance <= arm_reach_max + 1e-6)
            score = float(max(gaps) * 1000 + np.linalg.norm(palm - old_anchor) + 3 * max(0, .795 - palm_min_z) + 10 * max(0, shoulder_distance - arm_reach_max))
            candidates.append({
                "candidate_id": f"{contact_mode}_seed_{seed_id}",
                "contact_mode": contact_mode,
                "optimizer_success": bool(solution.success),
                "optimizer_cost": float(solution.cost),
                "left_dex3_AB_diagnostic_q": q,
                "wrist_position_m": wrist,
                "wrist_rotation": wrist_r,
                "palm_position_m": palm,
                "A_contact_proxy_m": pa,
                "B_contact_proxy_m": pb,
                "A_target_surface_m": ta,
                "B_target_surface_m": tb,
                "A_surface_gap_m": gaps[0],
                "B_surface_gap_m": gaps[1],
                "physical_palm_min_z_m": palm_min_z,
                "table_penetration_m": max(0.0, .795 - palm_min_z),
                "palm_to_active_left_shoulder_m": shoulder_distance,
                "active_left_arm_max_palm_reach_m": arm_reach_max,
                "required_arm_translation_from_v12_anchor_m": palm - old_anchor,
                "required_arm_translation_norm_m": float(np.linalg.norm(palm - old_anchor)),
                "valid": valid,
                "selection_score": score,
                "diagnostic_q_only": True,
            })
    valid = [row for row in candidates if row["valid"]]
    selected = min(valid or candidates, key=lambda row: row["selection_score"])
    return candidates, selected


def right_local_carrier_candidates(info, runtime, ring_inner, ring_depth):
    """Build accessory-local right-C carrier candidates from actual C geometry."""
    limits = runtime["right_C"]["limits"]
    candidates = []
    # In accessory coordinates +Y is the ring normal and the G1 approaches
    # from the -Y side.  The contact lies on the measured inner annulus.
    for angle_deg in np.arange(0.0, 360.0, 30.0):
        theta = math.radians(float(angle_deg))
        target = np.array([ring_inner * math.cos(theta), -0.5 * ring_depth, ring_inner * math.sin(theta)])
        radial = normalize(np.array([math.cos(theta), 0.0, math.sin(theta)]))
        for fraction in (0.15, 0.5, 0.85):
            q = limits[:, 0] + fraction * (limits[:, 1] - limits[:, 0])
            c, nc, _, _ = relative_hand_state(info, runtime, info["stand_arm_q"], "right_C", q)
            # Align contact-link +X with ring normal while keeping the wrist's
            # local +X generally along the source approach (+Y).
            base = np.column_stack((np.array([0.0, 1.0, 0.0]), -radial, normalize(np.cross([0, 1, 0], -radial))))
            if np.linalg.det(base) < 0:
                base[:, 2] *= -1
            align = v12.minimal_vector_alignment(base @ nc, np.array([0.0, 1.0, 0.0]))
            wrist_r = align @ base
            wrist = target - wrist_r @ c
            palm = wrist + wrist_r @ PALM_OFFSET["right"]
            contact = wrist + wrist_r @ c
            gap = float(np.linalg.norm(contact - target))
            insertion_alignment = float(np.dot(normalize(wrist_r @ nc), np.array([0.0, 1.0, 0.0])))
            candidates.append({
                "candidate_id": f"inner_{angle_deg:05.1f}_q{fraction:.2f}",
                "ring_angle_deg": float(angle_deg),
                "right_dex3_C_diagnostic_q": q,
                "accessory_from_wrist_position_m": wrist,
                "accessory_from_wrist_rotation": wrist_r,
                "accessory_from_palm_position_m": palm,
                "C_contact_proxy_m": contact,
                "ring_surface_target_m": target,
                "C_to_ring_gap_m": gap,
                "insertion_direction_alignment": insertion_alignment,
                "valid": bool(gap <= .005 and insertion_alignment >= .90),
                "diagnostic_q_only": True,
            })
    return candidates


def point_box_surface_gap(point: np.ndarray, pose: np.ndarray, dimensions: np.ndarray) -> tuple[float, float]:
    local = pose[:3, :3].T @ (np.asarray(point) - pose[:3, 3])
    half = .5 * np.asarray(dimensions)
    outside = np.maximum(np.abs(local) - half, 0.0)
    if np.any(outside > 0):
        return float(np.linalg.norm(outside)), 0.0
    depth = float(np.min(half - np.abs(local)))
    return 0.0, depth


def ring_gap(point: np.ndarray, pose: np.ndarray, inner: float, outer: float, depth: float) -> float:
    local = pose[:3, :3].T @ (np.asarray(point) - pose[:3, 3])
    radial = float(np.linalg.norm(local[[0, 2]]))
    y = abs(float(local[1]))
    radial_gap = max(inner - radial, radial - outer, 0.0)
    depth_gap = max(y - .5 * depth, 0.0)
    if radial_gap or depth_gap:
        return float(math.hypot(radial_gap, depth_gap))
    return float(min(radial - inner, outer - radial, .5 * depth - y))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        SOURCE, TIMELINE, ALIGNMENT, LAYOUT, FIXED_SCENE, ACTIVE_SCENE, REGISTRATION,
        ACTIVE_G1_USD, V12 / "aloha_phase_motion_library.npz",
        V12 / "globally_registered_base_targets.npz",
        V12 / "corrected_aloha_targets.npz",
        V12 / "position_only_exact_arm_trajectory.npz",
        V12 / "position_only_nullspace_arm_trajectory.npz",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    scene_files = [LAYOUT, FIXED_SCENE, ACTIVE_SCENE, REGISTRATION]
    hashes_before = {str(path.resolve()): sha(path) for path in scene_files}
    immutable_hashes = {str(path.resolve()): sha(path) for path in required}
    with np.load(SOURCE, allow_pickle=False) as source:
        action = source["optimized_action"].copy()
        timestamps = source["timestamp"].copy()
        fps = float(source["fps"])
    with np.load(V12 / "aloha_phase_motion_library.npz", allow_pickle=False) as library:
        if not np.array_equal(library["optimized_action"], action):
            raise RuntimeError("immutable v12 phase library action mismatch")
        if not np.array_equal(library["timestamps"], timestamps):
            raise RuntimeError("immutable v12 phase library timestamp mismatch")
        left_source_p = library["left_tcp_position"].copy()
        right_source_p = library["right_tcp_position"].copy()
        left_source_r = library["left_tcp_rotation"].copy()
        right_source_r = library["right_tcp_rotation"].copy()
    if action.shape != (990, 14) or fps != FPS or not np.isfinite(action).all():
        raise RuntimeError("immutable optimized_action invariant failed")
    with np.load(V12 / "globally_registered_base_targets.npz", allow_pickle=False) as base:
        base_model_l = base["base_aloha_derived_left_position_model"].copy()
        base_model_r = base["base_aloha_derived_right_position_model"].copy()
        old_global_r = base["global_registration_rotation"].copy()
        base_model_lr = np.einsum("ji,tjk->tik", old_global_r, base["globally_registered_left_rotation"])
        base_model_rr = np.einsum("ji,tjk->tik", old_global_r, base["globally_registered_right_rotation"])
        arm_names = base["arm_joint_names"].copy()
    with np.load(V12 / "position_only_exact_arm_trajectory.npz", allow_pickle=False) as old_exact:
        old_exact_q = old_exact["g1_arm_q"].copy()
        old_exact_lp = old_exact["achieved_left_position_scene"].copy()
        old_exact_rp = old_exact["achieved_right_position_scene"].copy()
        old_exact_lr = old_exact["achieved_left_rotation_scene"].copy()
        old_exact_rr = old_exact["achieved_right_rotation_scene"].copy()
    with np.load(V12 / "position_only_nullspace_arm_trajectory.npz", allow_pickle=False) as old_null:
        old_null_q = old_null["g1_arm_q"].copy()
    old_left = json.loads((V12 / "target_left_phase_anchors.json").read_text())["anchors"]
    old_right = json.loads((V12 / "target_right_phase_anchors.json").read_text())["anchors"]

    timeline_hash = sha(TIMELINE)
    alignment_hash = sha(ALIGNMENT)
    dump(OUT / "input_hash_audit.json", {
        "immutable_input_sha256": immutable_hashes,
        "optimized_action_shape": action.shape,
        "optimized_action_unchanged": True,
        "timestamps_unchanged": True,
        "timeline_sha256": timeline_hash,
        "alignment_sha256": alignment_hash,
        "action_to_observation_lag_frames": LAG,
        "workspace_scale": SCALE,
        "v12_exact_q_max_peak_to_peak_rad": float(np.max(np.ptp(old_exact_q, axis=0))),
        "v12_nullspace_q_max_peak_to_peak_rad": float(np.max(np.ptp(old_null_q, axis=0))),
        "v12_phase_library_loaded_byte_identical": True,
        "renderfix_articulation_path": str((ROOT / "isaaclab_magsafe_fixed_scene/render_target_phase_anchored_v12_renderfix.py").resolve()),
    })

    stage = Usd.Stage.Open(str(ACTIVE_SCENE))
    if stage is None:
        raise RuntimeError(ACTIVE_SCENE)
    layout = json.loads(LAYOUT.read_text())
    phone_pose = v12.active_transform(stage, "/World/MagSafeScene/Phone")
    accessory_initial = v12.active_transform(stage, "/World/MagSafeScene/Accessory")
    pad_face = v12.active_transform(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    active_g1 = v12.active_transform(stage, "/World/G1")
    phone_dims = np.asarray(layout["phone"]["size_landscape_xyz"], float)
    inner = .5 * float(layout["accessory"]["main_inner_diameter"])
    outer = .5 * float(layout["accessory"]["main_outer_diameter"])
    ring_depth = float(layout["accessory"]["main_depth"])
    attachment = make_transform(np.eye(3), np.array([0.0, .006425, 0.0]))
    pad_center = pad_face[:3, 3]
    pad_vertical = normalize(pad_face[:3, 1])
    pad_normal = normalize(pad_face[:3, 2])
    desired_phone_r = np.column_stack((pad_vertical, -pad_normal, np.cross(pad_vertical, -pad_normal)))
    if np.linalg.det(desired_phone_r) < 0:
        desired_phone_r[:, 2] *= -1
    desired_phone = make_transform(desired_phone_r, pad_center)

    info = core.ik.validate_model(core.G1_XML)
    frame_audit, runtime = hand_geometry_audit(stage, info)
    dump(OUT / "dex3_contact_frame_audit.json", frame_audit)
    sample_envelopes(info, runtime)
    arm_reach = {
        "left": active_arm_max_palm_reach(info, "left"),
        "right": active_arm_max_palm_reach(info, "right"),
    }
    arm_reach["method"] = "active 7-DOF arm FK maximization inside active joint limits"
    arm_reach["v12_conservative_neutral_chain_value_m"] = 0.4261960949595319
    arm_reach["v12_value_rejected_as_physical_maximum"] = True
    dump(OUT / "g1_arm_static_reach_audit.json", arm_reach)

    # Preserve the user's visual failure as an explicit physical-gap audit.
    failure_rows = []
    for label, frame, point, object_pose, dims in (
        ("left_phone", 169, old_exact_lp[169], phone_pose, phone_dims),
        ("left_charger", 523, old_exact_lp[523], desired_phone, phone_dims),
    ):
        gap, penetration = point_box_surface_gap(point, object_pose, dims)
        failure_rows.append({
            "event": label, "action_index": frame, "old_palm_position_m": point,
            "object_center_m": object_pose[:3, 3],
            "palm_to_object_center_vector_m": point - object_pose[:3, 3],
            "palm_proxy_to_nearest_object_surface_m": gap,
            "palm_proxy_object_penetration_m": penetration,
        })
    old_anchor_metrics = json.loads((V12 / "anchor_metrics.json").read_text())
    dump(OUT / "v12_physical_anchor_failure_audit.json", {
        "status": ["POSITION_TARGET_OBJECT_RELEVANCE_FAIL", "BLOCKED_PHYSICAL_CONTACT_ANCHOR_GEOMETRY", "NUMERIC_IK_PASS_BUT_TASK_CONTACT_FAIL"],
        "target_achieved_error_is_not_physical_contact_proof": True,
        "reported_old_anchors": {"left": old_left, "right": old_right},
        "direct_geometry_rows": failure_rows,
        "v12_self_reported_future_contact_diagnostics": old_anchor_metrics["achieved_position_only"]["candidates"],
        "exact_right_C_ring_gap_mm": old_anchor_metrics["achieved_position_only"]["candidates"]["EXACT"]["future_right_C_static_proxy_to_ring_surface_mm"],
        "nullspace_right_C_ring_gap_mm": old_anchor_metrics["achieved_position_only"]["candidates"]["NULLSPACE"]["future_right_C_static_proxy_to_ring_surface_mm"],
        "decision": "v12 palm-target error is retained as numeric IK provenance but rejected as a physical contact gate",
    })

    approach_left = normalize(old_exact_lp[169] - old_exact_lp[max(0, 169 - 12)], [0, 1, 0])
    left_candidates, selected_left = left_contact_candidate(
        info, runtime, phone_pose, phone_dims,
        np.asarray(old_left["phone_grasp"]["position_m"]), approach_left,
        arm_reach=arm_reach["left"],
    )
    dump(OUT / "left_phone_physical_carrier_candidates.json", {
        "active_phone_pose": phone_pose,
        "active_phone_dimensions_m": phone_dims,
        "mapped_aloha_approach_direction": approach_left,
        "selection": selected_left,
        "candidates": left_candidates,
        "dex3_q_is_diagnostic_only": True,
    })
    if not selected_left["valid"]:
        raise RuntimeError("BLOCKED_LEFT_PHONE_PHYSICAL_CARRIER")

    # Action 169 is contact acquisition, not the stable carrier lock.  Build a
    # second, embodiment-specific A+B relation from the strongest geometric
    # anchor: the known phone-on-pad pose at action 523.  Only this relation is
    # propagated from action 216 onward.
    approach_left_charger = normalize(old_exact_lp[523] - old_exact_lp[511], [0, 1, 0])
    stable_candidates, selected_stable_left = left_contact_candidate(
        info, runtime, desired_phone, phone_dims,
        np.asarray(old_left["charger"]["position_m"]), approach_left_charger,
        palm_y_upper=0.08, expanded_rotation_seeds=True, direction_weight=0.1,
        reach_weight=1000.0, arm_reach=arm_reach["left"],
    )
    dump(OUT / "left_stable_phone_carrier_candidates.json", {
        "definition": "target-side stable A+B carrier calibrated at the authoritative action-523 phone-on-pad pose",
        "valid_from_action_index": 216,
        "charger_action_index": 523,
        "selection": selected_stable_left,
        "candidates": stable_candidates,
        "not_equal_to_action_169_contact_start_relation": True,
        "dex3_q_is_diagnostic_only": True,
    })
    charger_single_finger_audit = {
        label: active_finger_phone_reach_audit(
            info, runtime, label, desired_phone, phone_dims
        ) for label in ("left_A", "left_B")
    }
    charger_single_finger_audit["gate"] = {
        "threshold_mm": 5.0,
        "left_A_pass": charger_single_finger_audit["left_A"]["minimum_absolute_surface_gap_m"] <= .005,
        "left_B_pass": charger_single_finger_audit["left_B"]["minimum_absolute_surface_gap_m"] <= .005,
        "both_pass": all(
            charger_single_finger_audit[label]["minimum_absolute_surface_gap_m"] <= .005
            for label in ("left_A", "left_B")
        ),
    }
    charger_single_finger_audit["decision"] = (
        "PASS" if charger_single_finger_audit["gate"]["both_pass"]
        else "BLOCKED_CHARGER_PHYSICAL_CARRIER"
    )
    dump(OUT / "charger_active_fk_contact_reach_audit.json", charger_single_finger_audit)
    charger_gate_pass = bool(
        selected_stable_left["valid"] and charger_single_finger_audit["gate"]["both_pass"]
    )

    right_local_candidates = right_local_carrier_candidates(info, runtime, inner, ring_depth)
    dump(OUT / "right_accessory_physical_carrier_candidates.json", {
        "ring_inner_radius_m": inner,
        "ring_outer_radius_m": outer,
        "ring_depth_m": ring_depth,
        "candidates_in_accessory_local_frame": right_local_candidates,
        "selection_deferred_until_common_global_registration": True,
        "dex3_q_is_diagnostic_only": True,
    })

    phone_to_palm = invert(desired_phone) @ make_transform(
        selected_stable_left["wrist_rotation"], selected_stable_left["palm_position_m"]
    )
    charger_palm_pose = desired_phone @ phone_to_palm
    charger_gap_a = float(selected_stable_left["A_surface_gap_m"])
    charger_gap_b = float(selected_stable_left["B_surface_gap_m"])
    charger_candidates = [{
        "candidate_id": "SAME_PHYSICAL_AB_CARRIER_ON_PAD",
        "phone_pose": desired_phone,
        "palm_pose": charger_palm_pose,
        "left_dex3_AB_diagnostic_q": selected_stable_left["left_dex3_AB_diagnostic_q"],
        "reconstructed_phone_center_to_pad_m": 0.0,
        "phone_normal_error_deg": 0.0,
        "left_A_surface_gap_m": charger_gap_a,
        "left_B_surface_gap_m": charger_gap_b,
        "same_phone_to_palm_relation_as_phone_contact": True,
        "valid": charger_gate_pass,
        "active_arm_and_finger_fk_crosscheck": charger_single_finger_audit,
    }]
    dump(OUT / "charger_physical_carrier_candidates.json", {"candidates": charger_candidates, "selected": charger_candidates[0]})

    # Recompute the one common rigid registration.  Scale 0.42 is already in
    # base_model_{l,r}; it is never applied to physical object/hand geometry.
    mapped_phone_rotation = v12.C_LEFT.T @ (left_source_r[169].T @ left_source_r[216]) @ v12.C_LEFT
    raw_portrait_r = phone_pose[:3, :3] @ mapped_phone_rotation
    portrait_options = []
    for sign in (1.0, -1.0):
        correction = v12.minimal_vector_alignment(raw_portrait_r[:, 0], [0, 0, sign])
        result = correction @ raw_portrait_r
        score = Rotation.from_matrix(correction).magnitude() + (0 if np.dot(result[:, 1], [0, 1, 0]) >= 0 else math.pi)
        portrait_options.append((score, result, correction))
    _, portrait_r, portrait_axis_correction = min(portrait_options, key=lambda row: row[0])

    registration_rows = []
    best = None
    left_anchor169 = np.asarray(selected_left["palm_position_m"])
    source_transport = base_model_l[523] - base_model_l[169]
    required_transport = charger_palm_pose[:3, 3] - left_anchor169
    transport_alignment = v12.minimal_vector_alignment(source_transport, required_transport)
    required_axis = normalize(required_transport)
    # First satisfy the dominant left phone-to-charger displacement direction.
    # The remaining one degree of freedom is a common twist about that vector;
    # search it for the right accessory residual and reach.  This is a fixed
    # registration, not time-varying deformation.
    for twist_deg in np.arange(-180.0, 180.01, 2.0):
                global_r = Rotation.from_rotvec(math.radians(float(twist_deg)) * required_axis).as_matrix() @ transport_alignment
                global_t = left_anchor169 - global_r @ base_model_l[169]
                base_l = (global_r @ base_model_l.T).T + global_t
                base_r = (global_r @ base_model_r.T).T + global_t
                # Portrait object translation preserves the registered ALOHA
                # phase-relative displacement; no static-grasp-first path.
                phone216 = make_transform(portrait_r, phone_pose[:3, 3] + global_r @ (base_model_l[216] - base_model_l[169]))
                palm216 = phone216 @ phone_to_palm
                phone319_r = portrait_r @ (v12.C_LEFT.T @ (left_source_r[216].T @ left_source_r[319]) @ v12.C_LEFT)
                phone319 = make_transform(phone319_r, phone216[:3, 3] + global_r @ (base_model_l[319] - base_model_l[216]))
                accessory319 = phone319 @ attachment
                approach_r = normalize(global_r @ (base_model_r[319] - base_model_r[216]), [0, 1, 0])
                local_approach = accessory319[:3, :3].T @ approach_r
                choices = []
                for carrier in right_local_candidates:
                    palm_world = accessory319 @ np.r_[np.asarray(carrier["accessory_from_palm_position_m"]), 1.0]
                    palm_world = palm_world[:3]
                    local_reach = normalize(np.asarray(carrier["ring_surface_target_m"]) - np.asarray(carrier["accessory_from_palm_position_m"]))
                    alignment_score = float(np.dot(local_reach, normalize(local_approach)))
                    residual_norm = float(np.linalg.norm(palm_world - base_r[319]))
                    choices.append((residual_norm + .05 * (1 - alignment_score), carrier, palm_world, alignment_score))
                _, carrier, right_anchor319, approach_alignment = min(choices, key=lambda row: row[0])
                right_anchor334 = right_anchor319 + global_r @ (base_model_r[334] - base_model_r[319])
                residuals = {
                    "left_portrait": float(np.linalg.norm(palm216[:3, 3] - base_l[216])),
                    "left_charger": float(np.linalg.norm(charger_palm_pose[:3, 3] - base_l[523])),
                    "right_accessory": float(np.linalg.norm(right_anchor319 - base_r[319])),
                }
                left_shoulder = np.asarray(arm_reach["left"]["active_shoulder_scene_position_m"])
                right_shoulder = np.asarray(arm_reach["right"]["active_shoulder_scene_position_m"])
                left_reach_max = float(arm_reach["left"]["maximum_shoulder_to_palm_reach_m"])
                right_reach_max = float(arm_reach["right"]["maximum_shoulder_to_palm_reach_m"])
                reach_excess = max(0.0, np.linalg.norm(charger_palm_pose[:3, 3] - left_shoulder) - left_reach_max) ** 2
                reach_excess += max(0.0, np.linalg.norm(right_anchor319 - right_shoulder) - right_reach_max) ** 2
                objective = (
                    (residuals["left_charger"] / .15) ** 2
                    + (residuals["right_accessory"] / .15) ** 2
                    + 300 * reach_excess + .1 * (1 - approach_alignment)
                )
                row = {
                    "transport_aligned_common_twist_deg": twist_deg, "objective": objective,
                    "endpoint_residual_m": residuals,
                    "right_approach_alignment": approach_alignment,
                    "right_carrier_candidate_id": carrier["candidate_id"],
                    "reach_excess_energy": reach_excess,
                }
                registration_rows.append(row)
                candidate = (objective, global_r, global_t, base_l, base_r, phone216, palm216, phone319, accessory319, carrier, right_anchor319, right_anchor334, row)
                if best is None or objective < best[0]:
                    best = candidate

    if best is None:
        raise RuntimeError("BLOCKED_GLOBAL_REGISTRATION")
    (_, global_r, global_t, base_l, base_r, phone216, palm216, phone319,
     accessory319, selected_right, right_anchor319, right_anchor334, selected_registration) = best
    base_lr = np.einsum("ij,tjk->tik", global_r, base_model_lr)
    base_rr = np.einsum("ij,tjk->tik", global_r, base_model_rr)

    dump(OUT / "physical_global_task_registration.json", {
        "method": "one common rotation and translation after immutable scale 0.42",
        "workspace_scale": SCALE,
        "scale_changed": False,
        "global_rotation": global_r,
        "global_translation_m": global_t,
        "selected_candidate": selected_registration,
        "candidate_count": len(registration_rows),
        "candidate_results": registration_rows,
        "same_transform_both_hands_all_990_samples": True,
        "physical_hand_and_object_geometry_scaled": False,
    })

    weights = {
        "VERY_STRONG_ALOHA": {"magnitude": 14.0, "velocity": 220.0, "acceleration": 1800.0, "bimanual": 55.0},
        "STRONG_ALOHA": {"magnitude": 6.0, "velocity": 120.0, "acceleration": 850.0, "bimanual": 28.0},
        "BALANCED": {"magnitude": 2.0, "velocity": 55.0, "acceleration": 320.0, "bimanual": 10.0},
    }
    source_progress = v12.combined_source_progress(left_source_p, right_source_p, left_source_r, right_source_r)
    left_res169 = left_anchor169 - base_l[169]
    left_res216 = palm216[:3, 3] - base_l[216]
    left_res523 = charger_palm_pose[:3, 3] - base_l[523]
    right_res319 = right_anchor319 - base_r[319]
    right_res334 = right_anchor334 - base_r[334]
    # Portrait is evaluated but is not an additional hand-written positional
    # waypoint.  The immutable L1/L2 motion remains free between the physical
    # phone and charger carrier anchors.
    anchor_l = {0: np.zeros(3), 169: left_res169, 523: left_res523, 695: left_res523}
    anchor_r = {0: np.zeros(3), 319: right_res319, 322: right_res319, 334: right_res334, 639: right_res319, 695: right_res319}
    candidate_results, candidate_arrays = {}, {}
    for name, profile in weights.items():
        knots_l, knots_r = v12.solve_residual_knots(anchor_l, anchor_r, profile)
        residual_l = v12.blend_knot_values(knots_l, source_progress)
        residual_r = v12.blend_knot_values(knots_r, source_progress)
        corrected_l, corrected_r = base_l + residual_l, base_r + residual_r
        phases, bimanual, minimum = v12.phase_fidelity(base_l, base_r, corrected_l, corrected_r, base_lr, base_rr, base_lr, base_rr)
        anchor_error = max(
            np.linalg.norm(corrected_l[169] - left_anchor169),
            np.linalg.norm(corrected_l[523] - charger_palm_pose[:3, 3]),
            np.linalg.norm(corrected_r[319] - right_anchor319),
            np.linalg.norm(corrected_r[334] - right_anchor334),
        )
        valid = bool(anchor_error <= 1e-9)
        correction_energy = float(np.sum(np.diff(residual_l, axis=0) ** 2) + np.sum(np.diff(residual_r, axis=0) ** 2))
        candidate_results[name] = {
            "weights": profile, "anchor_error_m": anchor_error, "anchor_valid": valid,
            "correction_energy": correction_energy,
            "minimum_major_phase_fidelity": minimum, "phases": phases, "bimanual": bimanual,
        }
        candidate_arrays[name] = (knots_l, knots_r, residual_l, residual_r, corrected_l, corrected_r)
    valid_names = [name for name, row in candidate_results.items() if row["anchor_valid"]]
    selected_name = max(valid_names, key=lambda name: (
        min(candidate_results[name]["minimum_major_phase_fidelity"].values()),
        -candidate_results[name]["correction_energy"],
    ))
    knots_l, knots_r, residual_l, residual_r, corrected_l, corrected_r = candidate_arrays[selected_name]
    common = .5 * (residual_l + residual_r)
    left_specific, right_specific = residual_l - common, residual_r - common

    save_npz(
        OUT / "globally_registered_aloha_targets.npz",
        optimized_action=action, timestamps=timestamps,
        base_aloha_derived_left_position_model=base_model_l,
        base_aloha_derived_right_position_model=base_model_r,
        global_registration_rotation=global_r, global_registration_translation=global_t,
        global_workspace_scale=np.array(SCALE),
        globally_registered_left_position=base_l, globally_registered_right_position=base_r,
        globally_registered_left_rotation=base_lr, globally_registered_right_rotation=base_rr,
        arm_joint_names=arm_names, method=np.array(METHOD),
    )
    save_npz(
        OUT / "physical_phase_residual_components.npz",
        residual_knots=KNOTS, left_knot_values=knots_l, right_knot_values=knots_r,
        common_translation_residual=common, left_specific_translation_residual=left_specific,
        right_specific_translation_residual=right_specific,
        left_translation_residual=residual_l, right_translation_residual=residual_r,
        selected_candidate=np.array(selected_name),
    )
    save_npz(
        OUT / "physical_corrected_targets.npz",
        optimized_action=action, timestamps=timestamps, action_indices=np.arange(990),
        base_left_position=base_l, base_right_position=base_r,
        base_left_rotation=base_lr, base_right_rotation=base_rr,
        global_registration_rotation=global_r, global_registration_translation=global_t,
        common_translation_residual=common,
        left_specific_translation_residual=left_specific, right_specific_translation_residual=right_specific,
        left_translation_residual=residual_l, right_translation_residual=residual_r,
        corrected_left_position=corrected_l, corrected_right_position=corrected_r,
        corrected_left_rotation=base_lr, corrected_right_rotation=base_rr,
        task_axis_left_rotation=base_lr, task_axis_right_rotation=base_rr,
        residual_knots=KNOTS, selected_candidate=np.array(selected_name), method=np.array(METHOD),
        diagnostic_only=np.array(True), real_robot_command_allowed=np.array(False),
    )
    dump(OUT / "selected_physical_carrier_anchors.json", {
        "left_phone_contact_start": selected_left,
        "left_stable_carrier_from_action_216": selected_stable_left,
        "left_portrait": {"palm_pose": palm216, "phone_pose": phone216},
        "right_accessory": {
            "carrier_local": selected_right,
            "accessory_pose_action_319": accessory319,
            "palm_position_action_319": right_anchor319,
            "palm_position_action_334": right_anchor334,
        },
        "left_charger": charger_candidates[0],
        "all_dex3_q_values_diagnostic_only": True,
    })
    dump(OUT / "phase_residual_candidate_results.json", candidate_results)
    dump(OUT / "aloha_fidelity_metrics.json", {
        "selected_candidate": selected_name,
        "thresholds": {"path_shape": .90, "speed": .90, "rotation_progress": .90},
        "minimum_major_phase_fidelity": candidate_results[selected_name]["minimum_major_phase_fidelity"],
        "per_phase": candidate_results[selected_name]["phases"],
        "bimanual": candidate_results[selected_name]["bimanual"],
        "global_similarity_excluded_from_deformation": True,
    })

    # Target-side diagnostic object poses, rebuilt from physical carriers.
    phone_traj = np.repeat(phone_pose[None], 990, axis=0)
    for k in range(169, 217):
        u = (k - 169) / (216 - 169)
        w = 10*u**3 - 15*u**4 + 6*u**5
        rel = Rotation.from_matrix(portrait_r @ phone_pose[:3, :3].T).as_rotvec()
        rr = Rotation.from_rotvec(w * rel).as_matrix() @ phone_pose[:3, :3]
        pp = (1-w)*phone_pose[:3, 3] + w*phone216[:3, 3]
        phone_traj[k] = make_transform(rr, pp)
    phone_from_palm_lock = invert(make_transform(base_lr[216], corrected_l[216])) @ phone216
    for k in range(217, 523):
        phone_traj[k] = make_transform(base_lr[k], corrected_l[k]) @ phone_from_palm_lock
    phone_traj[523:] = desired_phone
    accessory_traj = np.asarray([pose @ attachment for pose in phone_traj])
    right_palm_pose319 = make_transform(accessory319[:3, :3] @ np.asarray(selected_right["accessory_from_wrist_rotation"]), right_anchor319)
    accessory_from_right_palm = invert(right_palm_pose319) @ accessory319
    for k in range(334, 639):
        accessory_traj[k] = make_transform(base_rr[k], corrected_r[k]) @ accessory_from_right_palm
    accessory_traj[639:] = accessory_traj[638]
    save_npz(OUT / "target_phone_pose_trajectory.npz", action_index=np.arange(990), pose=phone_traj, position=phone_traj[:, :3, 3], rotation=phone_traj[:, :3, :3], diagnostic_only=np.array(True), physics=np.array(False))
    save_npz(OUT / "target_accessory_pose_trajectory.npz", action_index=np.arange(990), pose=accessory_traj, position=accessory_traj[:, :3, 3], rotation=accessory_traj[:, :3, :3], diagnostic_only=np.array(True), physics=np.array(False))

    dump(OUT / "physical_contact_reachability_metrics.json", {
        "stage": "PRE_IK_PHYSICAL_CARRIER_GATE",
        "left_action_169_A_gap_mm": selected_left["A_surface_gap_m"] * 1000,
        "left_action_169_B_gap_mm": selected_left["B_surface_gap_m"] * 1000,
        "right_action_319_C_ring_gap_mm": selected_right["C_to_ring_gap_m"] * 1000,
        "left_action_523_A_gap_mm": charger_gap_a * 1000,
        "left_action_523_B_gap_mm": charger_gap_b * 1000,
        "phone_center_to_pad_mm": 0.0,
        "phone_normal_error_deg": 0.0,
        "pre_ik_gate_pass": bool(selected_left["valid"] and selected_right["valid"] and charger_gate_pass),
        "blocking_state": None if charger_gate_pass else "BLOCKED_CHARGER_PHYSICAL_CARRIER",
        "required_local_dex3_q_is_diagnostic_only": True,
    })
    dump(OUT / "environment_audit.json", {
        "active_g1_transform": active_g1,
        "g1_root_expected": G1_ROOT,
        "g1_root_forward_offset_m": .15,
        "phone_pose": phone_pose,
        "accessory_pose": accessory_initial,
        "charger_pad_pose": pad_face,
        "scene_hashes_before": hashes_before,
        "scene_hashes_after": {str(path.resolve()): sha(path) for path in scene_files},
        "scene_byte_identical": hashes_before == {str(path.resolve()): sha(path) for path in scene_files},
    })
    print(json.dumps({
        "status": (
            "PHYSICAL_CARRIER_ANCHORS_READY_FOR_POSITION_ONLY_IK"
            if charger_gate_pass else "BLOCKED_CHARGER_PHYSICAL_CARRIER"
        ),
        "left_AB_gap_mm": [selected_left["A_surface_gap_m"]*1000, selected_left["B_surface_gap_m"]*1000],
        "right_C_gap_mm": selected_right["C_to_ring_gap_m"]*1000,
        "selected_residual": selected_name,
        "minimum_fidelity": candidate_results[selected_name]["minimum_major_phase_fidelity"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
