#!/usr/bin/env python3
"""Build the Episode-49 v17 execution-oriented kinematic preflight.

The program is deliberately Episode-49 development-only.  All behavior phase
transitions are resolved through ``SemanticTimeline`` names/progress arrays;
numeric indices are output provenance, never runtime rules.  True physics is
executed by a separate Isaac Lab process only when this pre-physics gate passes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

from aloha_g1_v15.kinematics import ActiveG1Dex3, sha256_file  # noqa: E402
from aloha_g1_v15.semantic_input import (  # noqa: E402
    TASK_EVENTS,
    load_human_reviewed_development_timeline,
)
from aloha_g1_v15.translator import compose_pose, interpolate_rotation, normalize  # noqa: E402
from aloha_g1_v17.trajectory import (  # noqa: E402
    build_predefined_hand_trajectories,
    build_task_partial_orientation_targets,
    evaluate_kinematic_candidate,
    solve_partial_orientation_trajectory,
)
from retarget_aloha_trajectory_to_g1 import retarget_aloha_trajectory_to_g1  # noqa: E402
from v15_semantic_interface import readiness as semantic_readiness  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
V15 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_orientation_dex3_v15"
V16 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
V16_LEFT_CARRIER = V16 / "selected_left_pincher_carrier.json"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
PHASE_LIBRARY = V12 / "aloha_phase_motion_library.npz"
V14_TARGET = V14 / "corrected_targets_v14.npz"
V14_ARM = V14 / "position_only_nullspace_v14.npz"
V14_ANCHORS = V14 / "selected_physical_carrier_anchors.json"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
ROOT_CONFIG = ROOT / "configs/g1_root_forward_v14.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
PREVIEW_CONFIG = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
ACTIVE_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
MAGNETIC_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_magnetic_scene_v2.usda"
MAGNET_CONFIG = ROOT / "isaaclab_magsafe_fixed_scene/magnet_config_v2.json"
DYNAMIC_BUILDER = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_dynamic_scene_builder.py"
MAGNET_CONTROLLER = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_magnet_controller.py"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEX3_MAPPING = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
PALM_CONFIG = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"
OLD_PRIMITIVES = ROOT / "configs/dex3_magsafe_grasp_primitives.sim.json"
METHOD = "ALOHA_PRIMARY_EP49_EXECUTION_PHYSICS_V17"


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, default=default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def inventory(directory: Path, *, include_large: bool = False) -> dict[str, Any]:
    rows = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        relative = str(path.relative_to(directory))
        size = path.stat().st_size
        row = {"path": relative, "size_bytes": size}
        if include_large or size <= 32 * 1024 * 1024 or path.suffix not in (".mp4", ".npz"):
            row["sha256"] = sha256_file(path)
        rows.append(row)
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "directory": str(directory.resolve()),
        "files": rows,
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "note": "large immutable binaries use size identity; critical NPZ hashes are recorded separately",
    }


def usd_pose(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"active scene missing {prim_path}")
    return np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    ).T


def phone_on_pad_pose(pad: np.ndarray) -> np.ndarray:
    long_axis = normalize(pad[:3, 1], (0.0, 0.0, 1.0))
    back_axis = -normalize(pad[:3, 2], (0.0, -1.0, 0.0))
    short_axis = normalize(np.cross(long_axis, back_axis), (1.0, 0.0, 0.0))
    back_axis = normalize(np.cross(short_axis, long_axis), (0.0, -1.0, 0.0))
    return compose_pose(np.column_stack((long_axis, back_axis, short_axis)), pad[:3, 3])


def historical_seed(payload: dict[str, Any], action_index: int, field: str) -> np.ndarray:
    matches = [
        row for row in payload.values()
        if isinstance(row, dict) and row.get("action_index") == action_index and field in row
    ]
    if len(matches) != 1:
        raise RuntimeError(f"v14 diagnostic seed lookup {action_index}/{field}: {len(matches)} matches")
    return np.asarray(matches[0][field], dtype=np.float64)


def remap_old_primitive(
    payload: dict[str, Any], runtime: ActiveG1Dex3, side: str, primitive_name: str,
) -> np.ndarray:
    row = payload["primitives"][primitive_name]
    lookup = dict(zip(row["joint_names"], row["qpos"]))
    return np.asarray([lookup[name] for name in runtime.hand_joint_names[side]], dtype=np.float64)


def clip_margin(values: np.ndarray, limits: np.ndarray, margin: float = 0.02) -> np.ndarray:
    span = np.ptp(limits, axis=1)
    usable = np.minimum(margin, 0.20 * span)
    return np.clip(values, limits[:, 0] + usable, limits[:, 1] - usable)


def active_usd_cap_proxy(
    stage: Usd.Stage, link: str, axis: int, sign: float, band_m: float = 0.001,
) -> np.ndarray:
    """Centroid of the active distal collision-mesh cap used for contact."""
    path = f"/World/G1/Asset/{link}/collisions/{link}/mesh"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"active Dex3 collision mesh missing: {path}")
    points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    support = float(sign) * points[:, int(axis)]
    selected = points[support >= support.max() - float(band_m)]
    if len(selected) < 10:
        raise RuntimeError(f"active Dex3 collision cap is undersampled: {path}")
    return selected.mean(axis=0)


def calibrate_left_phone_pinch(
    runtime: ActiveG1Dex3,
    active_stage: Usd.Stage,
    arm_q: np.ndarray,
    left_seed: np.ndarray,
    right_reference: np.ndarray,
    phone_pose: np.ndarray,
    phone_dimensions: np.ndarray,
    *,
    contact_preload_m: float = 0.005,
    index_contact_preload_m: float = 0.002,
    thumb_contact_vertical_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate one fixed thumb/index SIM primitive at the grasp pose.

    Both contact digits are fitted once against the active USD distal-mesh
    caps.  The thumb is commanded through the back surface as an actuator
    preload, while the index targets the opposite/front surface at a fixed
    geometry-relative edge inset.  The resulting joint vector is reused for
    the whole semantic HOLD interval; there is no per-frame finger IK.

    ``contact_preload_m`` is an actuator target preload: PhysX contact prevents
    actual penetration during execution.
    """
    result = np.asarray(left_seed, dtype=np.float64).copy()
    spec = runtime.contacts["left_A"]
    index_spec = runtime.contacts["left_B"]
    cap_local = active_usd_cap_proxy(active_stage, spec.link, axis=1, sign=-1.0)
    index_cap_local = active_usd_cap_proxy(active_stage, index_spec.link, axis=0, sign=1.0)
    body = runtime.body_ids[spec.link]
    index_body = runtime.body_ids[index_spec.link]

    def cap_world(thumb_q: np.ndarray) -> np.ndarray:
        hand = result.copy()
        hand[: len(thumb_q)] = thumb_q
        runtime.assign(arm_q, hand, right_reference)
        rotation = runtime.data.xmat[body].reshape(3, 3)
        model_point = runtime.data.xpos[body] + rotation @ cap_local
        return runtime.model_to_scene_position(model_point)

    def index_cap_world(index_q: np.ndarray) -> np.ndarray:
        hand = result.copy()
        hand[3:5] = index_q
        runtime.assign(arm_q, hand, right_reference)
        rotation = runtime.data.xmat[index_body].reshape(3, 3)
        model_point = runtime.data.xpos[index_body] + rotation @ index_cap_local
        return runtime.model_to_scene_position(model_point)

    seed = result[:3].copy()
    limits = runtime.hand_limits["left"][:3]
    usable = np.minimum(0.02, 0.20 * np.ptp(limits, axis=1))
    lower, upper = limits[:, 0] + usable, limits[:, 1] - usable
    seed = np.clip(seed, lower, upper)
    seed_world = cap_world(seed)
    seed_local = phone_pose[:3, :3].T @ (seed_world - phone_pose[:3, 3])
    half = 0.5 * np.asarray(phone_dimensions, dtype=np.float64)
    target_local = seed_local.copy()
    target_local[0] = np.clip(target_local[0], -half[0] + 0.003, half[0] - 0.003)
    target_local[1] = -half[1] + float(contact_preload_m)
    if thumb_contact_vertical_fraction is None:
        target_local[2] = np.clip(target_local[2], -half[2] + 0.006, half[2] - 0.006)
    else:
        fraction = float(np.clip(thumb_contact_vertical_fraction, -0.80, 0.80))
        target_local[2] = fraction * half[2]
    target_world = phone_pose[:3, 3] + phone_pose[:3, :3] @ target_local

    def residual(thumb_q: np.ndarray) -> np.ndarray:
        return np.r_[80.0 * (cap_world(thumb_q) - target_world), 0.05 * (thumb_q - seed)]

    solved = least_squares(
        residual, seed, bounds=(lower, upper), max_nfev=1000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    result[:3] = solved.x
    final_world = cap_world(solved.x)
    index_seed = result[3:5].copy()
    index_limits = runtime.hand_limits["left"][3:5]
    index_usable = np.minimum(0.02, 0.20 * np.ptp(index_limits, axis=1))
    index_lower = index_limits[:, 0] + index_usable
    index_upper = index_limits[:, 1] - index_usable
    index_seed = np.clip(index_seed, index_lower, index_upper)
    index_seed_world = index_cap_world(index_seed)
    index_seed_local = phone_pose[:3, :3].T @ (index_seed_world - phone_pose[:3, 3])
    index_edge_inset_m = 0.002
    index_target_local = index_seed_local.copy()
    index_target_local[0] = -half[0] + index_edge_inset_m
    index_target_local[1] = half[1] - float(index_contact_preload_m)
    index_target_local[2] = np.clip(index_target_local[2], -half[2] + 0.003, half[2] - 0.003)
    index_target_world = phone_pose[:3, 3] + phone_pose[:3, :3] @ index_target_local

    def index_residual(index_q: np.ndarray) -> np.ndarray:
        cap_local_phone = phone_pose[:3, :3].T @ (
            index_cap_world(index_q) - phone_pose[:3, 3]
        )
        # The active index chain has two DoF.  Match the task-critical edge
        # inset and opposing phone face; its vertical coordinate is checked as
        # a surface-bound constraint instead of over-constraining the solve.
        return np.r_[
            100.0 * (cap_local_phone[:2] - index_target_local[:2]),
            0.02 * (index_q - index_seed),
        ]

    index_solved = least_squares(
        index_residual, index_seed, bounds=(index_lower, index_upper), max_nfev=1000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    result[3:5] = index_solved.x
    index_final_world = index_cap_world(result[3:5])
    index_final_local = phone_pose[:3, :3].T @ (index_final_world - phone_pose[:3, 3])
    runtime.assign(arm_q, result, right_reference)
    same_hand_collisions = []
    for row in runtime.penetrating_contacts():
        bodies = row["bodies"]
        if (
            len(bodies) == 2
            and all(str(name).startswith("left_") for name in bodies)
            and all("hand" in str(name) for name in bodies)
        ):
            same_hand_collisions.append(row)
    if same_hand_collisions:
        raise RuntimeError(
            "selected fixed A_SCREEN_B_BACK primitive has same-hand collision: "
            f"{same_hand_collisions}"
        )
    vertical_surface_bound_pass = bool(abs(index_final_local[2]) <= half[2] - 0.003 + 1e-9)
    if not vertical_surface_bound_pass:
        raise RuntimeError("selected index contact lies outside the phone vertical surface")
    return result, {
        "method": "one_fixed_task_level_primitive_from_active_USD_distal_cap",
        "contact_assignment": "A_SCREEN_B_BACK",
        "active_distal_link": spec.link,
        "active_cap_local_xyz_m": cap_local,
        "seed_thumb_q_rad": seed,
        "selected_thumb_q_rad": solved.x,
        "active_index_distal_link": index_spec.link,
        "active_index_cap_local_xyz_m": index_cap_local,
        "seed_index_q_rad": index_seed,
        "selected_index_q_rad": index_solved.x,
        "index_edge_inset_m": index_edge_inset_m,
        "contact_preload_m": contact_preload_m,
        "index_contact_preload_m": index_contact_preload_m,
        "thumb_contact_vertical_fraction": thumb_contact_vertical_fraction,
        "index_preload_sweep_mm": {
            "collision_free": [0.0, 1.0, 2.0],
            "same_hand_collision": [3.0, 4.0],
            "selection_rule": "largest tested collision-free preload to improve physical retention margin",
        },
        "seed_cap_world_xyz_m": seed_world,
        "target_cap_world_xyz_m": target_world,
        "selected_cap_world_xyz_m": final_world,
        "target_error_mm": float(np.linalg.norm(final_world - target_world) * 1000.0),
        "seed_index_cap_world_xyz_m": index_seed_world,
        "target_index_cap_world_xyz_m": index_target_world,
        "selected_index_cap_world_xyz_m": index_final_world,
        "selected_index_cap_phone_local_xyz_m": index_final_local,
        "index_surface_xy_error_mm": float(np.linalg.norm(
            index_final_local[:2] - index_target_local[:2]
        ) * 1000.0),
        "index_vertical_surface_bound_pass": vertical_surface_bound_pass,
        "same_hand_collision_count_at_calibration_pose": len(same_hand_collisions),
        "optimizer_success": bool(solved.success and index_solved.success),
        "per_frame_finger_ik": False,
    }


def relevant_penetrations(runtime: ActiveG1Dex3) -> list[dict[str, Any]]:
    result = []
    for row in runtime.penetrating_contacts():
        bodies = row["bodies"]
        left = any(value.startswith("left_") for value in bodies)
        right = any(value.startswith("right_") for value in bodies)
        torso = any(any(token in value for token in ("torso", "waist", "pelvis")) for value in bodies)
        if (left and right) or (torso and (left or right)):
            result.append(row)
    return result


def select_right_nontask_tuck(
    runtime: ActiveG1Dex3,
    timeline,
    arm_q: np.ndarray,
    left_open: np.ndarray,
    left_pinch: np.ndarray,
    right_open: np.ndarray,
    right_hook: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select one global right-A safe pose; no per-frame finger fitting."""
    left_grasp = int(timeline.event("left_phone_grasp_start").action_index)
    right_grasp = int(timeline.event("right_accessory_grasp_start").action_index)
    rows = []
    for fraction in np.linspace(0.25, 0.75, 11):
        candidate_open = right_open.copy()
        candidate_hook = right_hook.copy()
        q_a = runtime.hand_limits["right"][:2, 0] + fraction * np.ptp(
            runtime.hand_limits["right"][:2], axis=1
        )
        candidate_open[:2] = q_a
        candidate_hook[:2] = q_a
        collisions = []
        minimum_distance = np.inf
        for frame in range(len(arm_q)):
            lq = left_pinch if frame >= left_grasp else left_open
            rq = candidate_hook if frame >= right_grasp else candidate_open
            runtime.assign(arm_q[frame], lq, rq)
            for record in relevant_penetrations(runtime):
                collisions.append({"frame": frame, **record})
                minimum_distance = min(minimum_distance, float(record["distance_m"]))
        rows.append({
            "normalized_A_tuck_fraction": float(fraction),
            "right_A_q_rad": q_a,
            "collision_record_count": len(collisions),
            "collision_frames": sorted({row["frame"] for row in collisions}),
            "maximum_penetration_m": 0.0 if not collisions else -minimum_distance,
            "distance_from_recorded_placeholder_rad": float(np.linalg.norm(q_a - right_open[:2])),
            "pass": not collisions,
        })
    eligible = [row for row in rows if row["pass"]]
    if not eligible:
        raise RuntimeError("no globally fixed right-A non-task tuck is collision-free on v14")
    selected = min(eligible, key=lambda row: row["distance_from_recorded_placeholder_rad"])
    output_open, output_hook = right_open.copy(), right_hook.copy()
    output_open[:2] = selected["right_A_q_rad"]
    output_hook[:2] = selected["right_A_q_rad"]
    return output_open, output_hook, {
        "method": "global task-level right-A tuck sweep over fixed normalized joint-range fractions",
        "episode_literal_frame_dependency": False,
        "per_frame_finger_IK": False,
        "candidates": rows,
        "selected": selected,
    }


def scan_runtime_literals(paths: list[Path], forbidden: set[int]) -> dict[str, Any]:
    findings = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value in forbidden:
                findings.append({"path": str(path), "line": node.lineno, "value": node.value})
    return {
        "scanned_runtime_files": [str(path.resolve()) for path in paths],
        "literal_semantic_runtime_dependencies": findings,
        "count": len(findings),
        "pass": not findings,
    }


def scene_physics_audit() -> dict[str, Any]:
    stage = Usd.Stage.Open(str(MAGNETIC_SCENE))
    active = Usd.Stage.Open(str(FIXED_SCENE))
    if stage is None or active is None:
        raise RuntimeError("cannot open fixed/magnetic scene")
    magnetic_root = "/MagSafeScene"
    active_root = "/MagSafeScene"
    objects = {}
    for name in ("Phone", "Accessory"):
        path = f"{magnetic_root}/{name}"
        prim = stage.GetPrimAtPath(path)
        rigid = UsdPhysics.RigidBodyAPI.Get(stage, path)
        mass = UsdPhysics.MassAPI.Get(stage, path)
        objects[name.lower()] = {
            "prim_valid": prim.IsValid(),
            "rigid_body_enabled": rigid.GetRigidBodyEnabledAttr().Get() if rigid else None,
            "kinematic_enabled": rigid.GetKinematicEnabledAttr().Get() if rigid else None,
            "mass_kg": mass.GetMassAttr().Get() if mass else None,
            "disable_gravity": prim.GetAttribute("physxRigidBody:disableGravity").Get(),
            "ccd_enabled": prim.GetAttribute("physxRigidBody:enableCCD").Get(),
        }
    joint_path = f"{magnetic_root}/MagneticJoints/AccessoryPhone"
    joint = UsdPhysics.FixedJoint.Get(stage, joint_path)
    magnet = json.loads(MAGNET_CONFIG.read_text(encoding="utf-8"))
    support_rows = []
    for prim in stage.Traverse():
        if "/Accessory/Colliders/SupportRing/" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
            support_rows.append({
                "path": str(prim.GetPath()), "active": prim.IsActive(),
                "collision_enabled": collision.GetCollisionEnabledAttr().Get() if collision else None,
            })
    active_pose = {
        name.lower(): usd_pose(active, f"{active_root}/{name}")
        for name in ("Phone", "Accessory", "Charger")
    }
    magnetic_pose = {
        name.lower(): usd_pose(stage, f"{magnetic_root}/{name}")
        for name in ("Phone", "Accessory", "Charger")
    }
    pose_delta = {
        name: float(np.linalg.norm(magnetic_pose[name][:3, 3] - active_pose[name][:3, 3]))
        for name in active_pose
    }
    stale_layout = max(pose_delta.values()) > 1e-6
    structural_pass = bool(
        all(row["rigid_body_enabled"] and not row["kinematic_enabled"] and not row["disable_gravity"] for row in objects.values())
        and joint and joint.GetJointEnabledAttr().Get()
        and magnet["accessory_phone"]["attachment_mode"] == "breakable_joint"
        and magnet["phone_charger"]["attachment_mode"] == "force_soft_lock"
    )
    return {
        "status": "AUTHORITATIVE_PHYSICS_STRUCTURE_PASS_WITH_RUNTIME_COMPOSITION" if structural_pass else "BLOCKED_AUTHORITATIVE_PHYSICS_MODEL",
        "authoritative_scene": str(MAGNETIC_SCENE.resolve()),
        "authoritative_scene_sha256": sha256_file(MAGNETIC_SCENE),
        "objects": objects,
        "accessory_phone_joint": {
            "path": joint_path,
            "valid": bool(joint),
            "enabled": joint.GetJointEnabledAttr().Get() if joint else None,
            "break_force_n": joint.GetBreakForceAttr().Get() if joint else None,
            "break_torque_nm": joint.GetBreakTorqueAttr().Get() if joint else None,
            "collision_enabled": joint.GetCollisionEnabledAttr().Get() if joint else None,
        },
        "magnet_config": magnet,
        "support_ring_collision_segments": support_rows,
        "active_fixed_scene_object_pose": active_pose,
        "legacy_dynamic_scene_object_pose": magnetic_pose,
        "legacy_dynamic_scene_pose_delta_m": pose_delta,
        "legacy_dynamic_scene_stale_layout": stale_layout,
        "runtime_physics_scene_strategy": (
            "reference the current fixed authoritative scene; apply rigid-body/contact APIs in a new run-only layer "
            "without authoring object transforms; derive charger and accessory anchors from active USD frames"
        ),
        "object_transform_authored_in_runtime_layer": False,
        "support_ring_collision_caveat": (
            "support-ring segment colliders are inactive because the authored inherited pose penetrated the table; "
            "main ring and support-foot v2 proxies remain the physical accessory representation"
        ),
        "physics_parameter_integrity_warning": (
            "magnet_config_v2 metadata labels several values DEBUG_INITIAL_GUESS; this run does not alter them"
        ),
        "scene_physics_modified": False,
        "structural_pass": structural_pass,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        SOURCE, PHASE_LIBRARY, V14_TARGET, V14_ARM, V14_ANCHORS, V16_LEFT_CARRIER, TIMELINE,
        ALIGNMENT, ROOT_CONFIG, LAYOUT, PREVIEW_CONFIG, ACTIVE_SCENE, FIXED_SCENE,
        MAGNETIC_SCENE, MAGNET_CONFIG, DYNAMIC_BUILDER, MAGNET_CONTROLLER, MODEL,
        DEX3_MAPPING, PALM_CONFIG, OLD_PRIMITIVES,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    critical_hashes_before = {str(path.resolve()): sha256_file(path) for path in required}
    v15_inventory_before = inventory(V15)
    v16_inventory_before = inventory(V16)

    with np.load(SOURCE, allow_pickle=False) as archive:
        optimized_action = archive["optimized_action"].copy()
        timestamps = archive["timestamp"].copy()
        fps = float(archive["fps"])
    with np.load(PHASE_LIBRARY, allow_pickle=False) as archive:
        if not np.array_equal(optimized_action, archive["optimized_action"]):
            raise RuntimeError("source/phase-library action mismatch")
        if not np.array_equal(timestamps, archive["timestamps"]):
            raise RuntimeError("source/phase-library timestamp mismatch")
        source_left_position = archive["left_tcp_position"].copy()
        source_right_position = archive["right_tcp_position"].copy()
        source_left_rotation = archive["left_tcp_rotation"].copy()
        source_right_rotation = archive["right_tcp_rotation"].copy()
    with np.load(V14_TARGET, allow_pickle=False) as archive:
        corrected_target_left = archive["corrected_left_position"].copy()
        corrected_target_right = archive["corrected_right_position"].copy()
        workspace_scale = float(archive["workspace_scale"])
    with np.load(V14_ARM, allow_pickle=False) as archive:
        v14_arm_q = archive["g1_arm_q"].copy()
        arm_joint_names = archive["arm_joint_names"].astype(str)
        root_position = archive["g1_root"].copy()
        root_offset = float(archive["g1_root_forward_offset_m"])
        # The visible, user-approved v14 arm motion is the achieved palm path.
        # Use it as the execution backbone so partial orientation receives the
        # full 5-mm tolerance instead of inheriting v14's pre-existing target
        # residual a second time.
        target_left = archive["achieved_left_position_scene"].copy()
        target_right = archive["achieved_right_position_scene"].copy()
    if optimized_action.shape != (990, 14) or len(timestamps) != 990 or fps != 30.0:
        raise RuntimeError("Episode-49 source invariant failed")
    if not np.isfinite(optimized_action).all() or not np.isfinite(v14_arm_q).all():
        raise RuntimeError("source/backbone contains NaN/Inf")
    root_cfg = json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))
    if not np.allclose(root_position, root_cfg["new_exact_root_xyz_m"], atol=1e-10):
        raise RuntimeError("v14 root mismatch")
    if abs(root_offset - root_cfg["selected_total_forward_offset_m"]) > 1e-12 or workspace_scale != 0.42:
        raise RuntimeError("v14 registration/scale mismatch")

    timeline = load_human_reviewed_development_timeline(
        TIMELINE, ALIGNMENT, optimized_action, timestamps,
        source_left_position, source_right_position,
        source_left_rotation, source_right_rotation,
        trajectory_path=SOURCE, fk_model_path=MODEL, task_geometry_path=LAYOUT,
    )
    interface = semantic_readiness(timeline)
    dry_run = retarget_aloha_trajectory_to_g1(
        optimized_action, timestamps, timeline,
        {"method": METHOD}, {"active_scene": str(ACTIVE_SCENE)}, dry_run=True,
    )
    alignment_payload = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    forbidden = {
        int(row["aligned_action_index"])
        for row in alignment_payload["event_mapping"].values()
    }
    runtime_paths = [
        ROOT / "tools/aloha_g1_v17/trajectory.py",
        ROOT / "tools/aloha_g1_v15/semantic_input.py",
        ROOT / "tools/retarget_aloha_trajectory_to_g1.py",
    ]
    literal = scan_runtime_literals(runtime_paths, forbidden)
    if not literal["pass"]:
        raise RuntimeError(f"semantic runtime literal dependency: {literal['literal_semantic_runtime_dependencies']}")
    semantic_audit = {
        "status": "EP49_GENERIC_SEMANTIC_API_USED",
        "timeline_source": "HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE",
        "timeline_path": str(TIMELINE.resolve()),
        "timeline_sha256": sha256_file(TIMELINE),
        "semantic_runtime_interface": "GENERIC",
        "hardcoded_runtime_indices": False,
        "resolved_events_for_provenance_only": {
            name: {"action_index": timeline.event(name).action_index, "action_time_sec": timeline.event(name).action_time_sec}
            for name in TASK_EVENTS
        },
        "interface_readiness": interface,
        "generic_converter_dry_run": dry_run,
        "runtime_literal_audit": literal,
        "terminal_events_block_manipulation": False,
        "validation_read_count": 0,
        "heldout_read_count": 0,
        "g1_expert_read_count": 0,
    }
    dump(OUT / "semantic_runtime_audit.json", semantic_audit)

    physics_audit = scene_physics_audit()
    dump(OUT / "scene_physics_audit.json", physics_audit)

    active_stage = Usd.Stage.Open(str(ACTIVE_SCENE))
    phone_initial = usd_pose(active_stage, "/World/MagSafeScene/Phone")
    pad = usd_pose(active_stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    phone_charger = phone_on_pad_pose(pad)
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    table_height = float(layout["table"]["surface_height"])
    table_bounds = (0.0, float(layout["table"]["size_x"]), 0.0, float(layout["table"]["size_y"]))
    runtime = ActiveG1Dex3(MODEL, DEX3_MAPPING, PALM_CONFIG, root_position)

    old = json.loads(OLD_PRIMITIVES.read_text(encoding="utf-8"))
    left_open = clip_margin(remap_old_primitive(old, runtime, "left", "LEFT_PHONE_OPEN"), runtime.hand_limits["left"])
    right_open = clip_margin(remap_old_primitive(old, runtime, "right", "RIGHT_ACCESSORY_OPEN"), runtime.hand_limits["right"])
    anchors = json.loads(V14_ANCHORS.read_text(encoding="utf-8"))
    left_grasp = int(timeline.event("left_phone_grasp_start").action_index)
    right_grasp = int(timeline.event("right_accessory_grasp_start").action_index)
    left_pinch = left_open.copy()
    left_pinch[:5] = historical_seed(anchors, left_grasp, "diagnostic_left_dex3_AB_q_rad")
    left_pinch = clip_margin(left_pinch, runtime.hand_limits["left"])
    left_pinch_solver_seed = left_pinch.copy()
    right_hook = right_open.copy()
    right_hook[-2:] = historical_seed(anchors, right_grasp, "diagnostic_right_dex3_C_q_rad")
    right_hook = clip_margin(right_hook, runtime.hand_limits["right"])
    left_pinch, left_pinch_audit = calibrate_left_phone_pinch(
        runtime, active_stage, v14_arm_q[left_grasp], left_pinch, right_open,
        phone_initial, np.asarray(layout["phone"]["size_landscape_xyz"], dtype=np.float64),
    )
    right_open, right_hook, tuck_audit = select_right_nontask_tuck(
        runtime, timeline, v14_arm_q, left_open, left_pinch_solver_seed, right_open, right_hook,
    )
    # Fixed task-level pre-contact postures.  PREGRASP retracts the index
    # halfway from contact; INDEX_CONTACT then reaches the front surface while
    # the thumb closes continuously.  The small fixed sequence avoids both an
    # early unilateral push and transient thumb/index crossing.  Timing still
    # comes from normalized source/semantic progress.
    left_pre = left_open.copy()
    left_pre[3:5] = 0.5 * (left_open[3:5] + left_pinch[3:5])
    left_pre = clip_margin(left_pre, runtime.hand_limits["left"])
    left_index_contact = left_open.copy()
    left_index_contact[3:5] = left_pinch[3:5]
    left_index_contact = clip_margin(left_index_contact, runtime.hand_limits["left"])
    right_pre = clip_margin(0.20 * right_open + 0.80 * right_hook, runtime.hand_limits["right"])
    primitives = {
        "LEFT_OPEN": left_open,
        "LEFT_PHONE_PREGRASP": left_pre,
        "LEFT_PHONE_INDEX_CONTACT": left_index_contact,
        "LEFT_PHONE_PINCH": left_pinch,
        "RIGHT_OPEN": right_open,
        "RIGHT_RING_PREHOOK": right_pre,
        "RIGHT_RING_HOOK": right_hook,
    }
    # Arm orientation remains independent of contact preload.  The solver
    # sees the previously validated task-level seed; the final fixed SIM
    # primitive is attached only after arm selection and then audited.  This
    # prevents Dex3 contact calibration from redesigning the ALOHA arm path.
    solver_left_pre = clip_margin(
        0.15 * left_open + 0.85 * left_pinch_solver_seed, runtime.hand_limits["left"]
    )
    solver_left_pre[:3] = left_pinch_solver_seed[:3]
    solver_primitives = dict(primitives)
    solver_primitives["LEFT_PHONE_PREGRASP"] = solver_left_pre
    solver_primitives["LEFT_PHONE_INDEX_CONTACT"] = left_pinch_solver_seed
    solver_primitives["LEFT_PHONE_PINCH"] = left_pinch_solver_seed
    primitive_config = {
        "schema_version": 1,
        "status": "SIMULATION_ONLY_TASK_PRIMITIVES_CALIBRATED",
        "simulation_only": True,
        "not_real_robot_calibrated": True,
        "authoritative_for_real_robot": False,
        "real_robot_command_allowed": False,
        "per_frame_finger_ik": False,
        "semantic_progress_driven": True,
        "source_gripper_progress_driven": True,
        "active_model": str(MODEL.resolve()),
        "active_model_sha256": sha256_file(MODEL),
        "source_placeholder_audit": {
            "path": str(OLD_PRIMITIVES.resolve()),
            "sha256": sha256_file(OLD_PRIMITIVES),
            "source_label": old["source"],
            "blindly_reused": False,
        },
        "joint_names": {
            "left": runtime.hand_joint_names["left"],
            "right": runtime.hand_joint_names["right"],
        },
        "primitives": primitives,
        "calibration": {
            "left_phone_pinch": left_pinch_audit,
            "right_ring_hook": "v14 full-arm-plus-Dex3 physical ring calibration seed, clipped to fixed 0.02-rad SIM margin",
            "right_non_task_A_tuck": tuck_audit,
            "middle_left_C_non_task": "LEFT_OPEN retained",
            "right_A_B_non_task": "global fixed safe posture retained",
            "pregrasp_blend_fractions": {"left_phone_index": 0.50, "right_ring": 0.80},
            "left_index_contact_completion_fraction_of_acquisition": 0.35,
            "left_digit_staging": (
                "safe PREGRASP -> fixed INDEX_CONTACT while thumb closes continuously -> fixed PINCH; "
                "all interpolation uses source/semantic progress"
            ),
        },
    }
    dump(OUT / "dex3_magsafe_execution_primitives_v17.sim.json", primitive_config)
    left_hand_q, right_hand_q, semantic_progress = build_predefined_hand_trajectories(
        timeline, runtime, primitives, optimized_action[:, 6], optimized_action[:, 13]
    )
    solver_left_hand_q, solver_right_hand_q, _ = build_predefined_hand_trajectories(
        timeline, runtime, solver_primitives, optimized_action[:, 6], optimized_action[:, 13]
    )
    if np.max(np.ptp(left_hand_q, axis=0)) <= 0.05 or np.max(np.ptp(right_hand_q, axis=0)) <= 0.05:
        raise RuntimeError("predefined Dex3 trajectories are static")

    right_anchor_rows = [
        row for row in anchors.values()
        if isinstance(row, dict) and row.get("action_index") == right_grasp and "wrist_rotation" in row
    ]
    if len(right_anchor_rows) != 1:
        raise RuntimeError("right ring wrist rotation seed not uniquely resolved")
    charger_index = int(timeline.event("phone_charger_attachment_complete").action_index)
    charger_anchor_rows = [
        row for row in anchors.values()
        if isinstance(row, dict) and row.get("action_index") == charger_index and "wrist_rotation" in row
    ]
    if len(charger_anchor_rows) != 1:
        raise RuntimeError("left charger wrist rotation seed not uniquely resolved")
    runtime.assign(v14_arm_q[left_grasp], solver_left_hand_q[left_grasp], solver_right_hand_q[left_grasp])
    v14_grasp_wrist_rotation = runtime.wrist_pose("left")[:3, :3]
    v16_grasp_wrist_rotation = np.asarray(
        json.loads(V16_LEFT_CARRIER.read_text(encoding="utf-8"))["selected"]["initial_wrist"],
        dtype=np.float64,
    )[:3, :3]
    candidate_configs = [
        {
            "name": f"V14_PINCH_FACE_{int(round(fraction * 100)):03d}",
            "grasp_orientation_fraction": fraction,
            "orientation_gain": 40.0, "prior_gain": 0.09, "temporal_gain": 0.08,
            "collision_gain": 80000.0, "shoulder_prior_gain": 12.0,
            "max_deviation_rad": 1.20, "max_step_rad": 0.32,
        }
        for fraction in (0.0, 0.10, 0.33, 1.0)
    ]
    candidates = []
    for config in candidate_configs:
        print(f"[V17] solving {config['name']}", flush=True)
        grasp_rotation = interpolate_rotation(
            v14_grasp_wrist_rotation, v16_grasp_wrist_rotation,
            np.asarray([config["grasp_orientation_fraction"]], dtype=np.float64),
        )[0]
        orientation_targets = build_task_partial_orientation_targets(
            timeline, runtime, v14_arm_q, solver_left_hand_q, solver_right_hand_q,
            source_left_rotation, source_right_rotation,
            phone_initial, phone_charger, grasp_rotation,
            np.asarray(charger_anchor_rows[0]["wrist_rotation"], dtype=np.float64),
            np.asarray(right_anchor_rows[0]["wrist_rotation"], dtype=np.float64),
        )
        arm_q = solve_partial_orientation_trajectory(
            runtime, v14_arm_q, target_left, target_right,
            orientation_targets["left_rotation"], orientation_targets["right_rotation"],
            orientation_targets["left_axis_weight"], orientation_targets["right_axis_weight"],
            solver_left_hand_q, solver_right_hand_q,
            **{
                key: value for key, value in config.items()
                if key not in ("name", "grasp_orientation_fraction")
            },
        )
        metrics = evaluate_kinematic_candidate(
            timeline, runtime, arm_q, solver_left_hand_q, solver_right_hand_q,
            target_left, target_right, orientation_targets,
            source_left_rotation, source_right_rotation,
            phone_initial, phone_charger, table_height, table_bounds,
        )
        achieved = metrics.pop("achieved")
        achieved_grasp_wrist = achieved["left_wrist"][left_grasp, :3, :3]
        grasp_cosine = np.clip(
            (np.trace(achieved_grasp_wrist.T @ v16_grasp_wrist_rotation) - 1.0) / 2.0,
            -1.0, 1.0,
        )
        grasp_task_error = float(np.degrees(np.arccos(grasp_cosine)))
        metrics["phone_grasp_task_orientation"] = {
            "error_to_v16_verified_task_wrist_deg": grasp_task_error,
            "diagnostic_max_deg": 10.0,
            "pass": grasp_task_error <= 10.0,
        }
        metrics["gate_pass"] = bool(
            metrics["gate_pass"] and metrics["phone_grasp_task_orientation"]["pass"]
        )
        candidates.append({
            "config": config, "arm_q": arm_q, "metrics": metrics,
            "achieved": achieved, "orientation_targets": orientation_targets,
        })
        print(json.dumps({
            "name": config["name"], "gate_pass": metrics["gate_pass"],
            "position": metrics["position"], "orientation": metrics["orientation"],
            "phone_grasp_task_orientation": metrics["phone_grasp_task_orientation"],
            "fidelity": metrics["fidelity"], "joint": metrics["joint"],
            "collision_records": metrics["collision"]["prohibited_collision_records"],
        }, indent=2), flush=True)

    eligible = [row for row in candidates if row["metrics"]["gate_pass"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                -row["config"]["grasp_orientation_fraction"],
                row["metrics"]["orientation"]["portrait_long_axis_error_deg"]
                + row["metrics"]["orientation"]["charger_normal_error_deg"]
                + row["metrics"]["orientation"]["charger_vertical_axis_error_deg"],
                -row["metrics"]["fidelity"]["minimum_primary_metric"],
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["metrics"]["collision"]["pass"],
                row["metrics"]["position"]["simultaneous_5mm_rate"],
                row["metrics"]["fidelity"]["minimum_primary_metric"],
                -(
                    row["metrics"]["orientation"]["portrait_long_axis_error_deg"]
                    + row["metrics"]["orientation"]["charger_normal_error_deg"]
                    + row["metrics"]["orientation"]["charger_vertical_axis_error_deg"]
                ),
            ),
        )
    selected_arm_q = selected["arm_q"]
    orientation_targets = selected["orientation_targets"]
    selected["solver_adapter_metrics"] = selected["metrics"]

    # Dex3 is calibrated only after the arm candidate is selected.  It cannot
    # redesign the Cartesian arm path or influence candidate selection.
    left_pinch, left_pinch_audit = calibrate_left_phone_pinch(
        runtime, active_stage, selected_arm_q[left_grasp], left_pinch_solver_seed, right_open,
        phone_initial, np.asarray(layout["phone"]["size_landscape_xyz"], dtype=np.float64),
    )
    left_pre = left_open.copy()
    left_pre[3:5] = 0.5 * (left_open[3:5] + left_pinch[3:5])
    left_pre = clip_margin(left_pre, runtime.hand_limits["left"])
    left_index_contact = left_open.copy()
    left_index_contact[3:5] = left_pinch[3:5]
    left_index_contact = clip_margin(left_index_contact, runtime.hand_limits["left"])
    primitives.update({
        "LEFT_PHONE_PREGRASP": left_pre,
        "LEFT_PHONE_INDEX_CONTACT": left_index_contact,
        "LEFT_PHONE_PINCH": left_pinch,
    })
    primitive_config["primitives"] = primitives
    primitive_config["calibration"]["left_phone_pinch"] = left_pinch_audit
    primitive_config["calibration"]["calibrated_after_arm_candidate_selection"] = True
    primitive_config["calibration"]["dex3_influenced_arm_candidate_selection"] = False
    dump(OUT / "dex3_magsafe_execution_primitives_v17.sim.json", primitive_config)
    left_hand_q, right_hand_q, semantic_progress = build_predefined_hand_trajectories(
        timeline, runtime, primitives, optimized_action[:, 6], optimized_action[:, 13]
    )
    selected_metrics = evaluate_kinematic_candidate(
        timeline, runtime, selected_arm_q, left_hand_q, right_hand_q,
        target_left, target_right, orientation_targets,
        source_left_rotation, source_right_rotation,
        phone_initial, phone_charger, table_height, table_bounds,
    )
    achieved = selected_metrics.pop("achieved")
    achieved_grasp_wrist = achieved["left_wrist"][left_grasp, :3, :3]
    grasp_cosine = np.clip(
        (np.trace(achieved_grasp_wrist.T @ v16_grasp_wrist_rotation) - 1.0) / 2.0,
        -1.0, 1.0,
    )
    grasp_task_error = float(np.degrees(np.arccos(grasp_cosine)))
    selected_metrics["phone_grasp_task_orientation"] = {
        "error_to_v16_verified_task_wrist_deg": grasp_task_error,
        "diagnostic_max_deg": 10.0,
        "pass": grasp_task_error <= 10.0,
    }
    selected_metrics["gate_pass"] = bool(
        selected_metrics["gate_pass"]
        and selected_metrics["phone_grasp_task_orientation"]["pass"]
    )
    selected["metrics"] = selected_metrics
    selected["achieved"] = achieved
    print(json.dumps({
        "name": selected["config"]["name"],
        "final_fixed_primitive_gate_pass": selected_metrics["gate_pass"],
        "dex3_redesigned_arm_path": False,
        "position": selected_metrics["position"],
        "orientation": selected_metrics["orientation"],
        "phone_grasp_task_orientation": selected_metrics["phone_grasp_task_orientation"],
        "fidelity": selected_metrics["fidelity"],
        "joint": selected_metrics["joint"],
        "collision_records": selected_metrics["collision"]["prohibited_collision_records"],
    }, indent=2), flush=True)
    all_q = np.c_[selected_arm_q, left_hand_q, right_hand_q]

    common_npz = {
        "optimized_action": optimized_action,
        "source_timestamps": timestamps,
        "arm_joint_names": arm_joint_names,
        "left_dex3_joint_names": np.asarray(runtime.hand_joint_names["left"]),
        "right_dex3_joint_names": np.asarray(runtime.hand_joint_names["right"]),
        "v14_reference_arm_q": v14_arm_q,
        "g1_root": root_position,
        "workspace_scale": np.asarray(workspace_scale),
        "method": np.asarray(METHOD),
        "semantic_timeline_sha256": np.asarray(sha256_file(TIMELINE)),
        "physics_applied": np.asarray(False),
        "simulation_only": np.asarray(True),
        "real_robot_command_allowed": np.asarray(False),
    }
    save_npz(
        OUT / "final_kinematic_arm_trajectory.npz", **common_npz,
        g1_arm_q=selected_arm_q,
        v14_left_position_target=target_left,
        v14_right_position_target=target_right,
        v14_corrected_left_position_target=corrected_target_left,
        v14_corrected_right_position_target=corrected_target_right,
        achieved_left_position=achieved["left_position"],
        achieved_right_position=achieved["right_position"],
        achieved_left_rotation=achieved["left_rotation"],
        achieved_right_rotation=achieved["right_rotation"],
        selected_candidate=np.asarray(selected["config"]["name"]),
    )
    save_npz(
        OUT / "final_dex3_trajectory.npz", **common_npz,
        left_dex3_q=left_hand_q, right_dex3_q=right_hand_q,
        left_dex3_qpos=left_hand_q, right_dex3_qpos=right_hand_q,
        primitive_source=np.asarray("predefined_simulation_task_primitives_v17"),
        **{f"semantic_{key}_progress": value for key, value in semantic_progress.items()},
    )
    save_npz(
        OUT / "final_arm_dex3_trajectory.npz", **common_npz,
        arm_qpos=selected_arm_q, g1_arm_q=selected_arm_q,
        left_dex3_qpos=left_hand_q, right_dex3_qpos=right_hand_q,
        full_joint_q=all_q, fps=np.asarray(fps),
        primitive_source=np.asarray("predefined_simulation_task_primitives_v17"),
        authoritative_for_real_robot=np.asarray(False),
    )
    save_npz(
        OUT / "partial_orientation_targets.npz",
        left_rotation=orientation_targets["left_rotation"],
        right_rotation=orientation_targets["right_rotation"],
        left_axis_weight=orientation_targets["left_axis_weight"],
        right_axis_weight=orientation_targets["right_axis_weight"],
    )

    candidate_records = [
        {"config": row["config"], "metrics": row["metrics"], "selected": row is selected}
        for row in candidates
    ]
    dump(OUT / "partial_orientation_config.json", {
        "method": "source-relative rotation with task-critical axis weights; unconstrained twist omitted",
        "v14_position_path_is_primary": True,
        "candidate_sweep": candidate_records,
        "selected": selected["config"],
        "v16_reused_algorithmic_evidence": [
            "source-relative rotation-progress mapping", "task-axis endpoint registration",
            "previous-sample continuation and bounded joint steps",
        ],
        "v16_final_arm_q_reused": False,
    })
    dump(OUT / "nullspace_collision_config.json", {
        "primary_position_weight": 2600.0,
        "collision_penetration_tolerance_m": 1e-5,
        "table_test": "active geom vertices in authoritative table half-space",
        "selected_parameters": selected["config"],
        "per_frame_cartesian_correction": False,
        "hand_contact_redesigns_arm_path": False,
    })
    dump(OUT / "aloha_fidelity_metrics.json", selected_metrics["fidelity"] | {
        "rotation": selected_metrics["orientation"],
        "source_action_is_sole_behavior_source": True,
    })
    dump(OUT / "kinematic_collision_metrics.json", selected_metrics["collision"])
    dump(OUT / "kinematic_joint_metrics.json", selected_metrics["joint"] | {
        "finite": selected_metrics["finite"],
        "position": selected_metrics["position"],
    })
    dump(OUT / "kinematic_prephysics_result.json", {
        "status": "ALOHA_PRIMARY_ARM_KINEMATIC_PASS" if selected_metrics["gate_pass"] else "BLOCKED_KINEMATIC_PREPHYSICS",
        "selected_candidate": selected["config"]["name"],
        "gate_pass": selected_metrics["gate_pass"],
        "metrics": selected_metrics,
        "physics_authorized_by_gate": bool(selected_metrics["gate_pass"] and physics_audit["structural_pass"]),
    })
    dump(OUT / "v14_v16_reuse_audit.json", {
        "v14_reused": [
            "root registration", "workspace scale", "task registration",
            "left/right Cartesian position targets", "position-only nullspace q warm start",
            "ALOHA path/speed/bimanual behavior",
        ],
        "v16_reused_algorithmic_elements": [
            "source-relative orientation mapping", "task-critical endpoint axes",
            "branch-continuous temporal continuation strategy",
        ],
        "deliberately_discarded": [
            "v15 final arm q", "v16 final arm q", "v16 contact translation residual",
            "v16 rigid A/B carrier", "v16 exact continuous C carrier",
        ],
        "v14_arm_backbone_hash": array_sha(v14_arm_q),
        "selected_v17_arm_hash": array_sha(selected_arm_q),
    })
    dump(OUT / "physics_controller_config.json", {
        "status": "PENDING_TRUE_PHYSICS" if selected_metrics["gate_pass"] else "NOT_RUN_KINEMATIC_GATE_FAILED",
        "control": "Isaac Lab implicit actuator position targets",
        "arm": {"effort_limit_sim": 25.0, "velocity_limit_sim": 12.0, "stiffness": 1000.0, "damping": 40.0},
        "dex3": {"effort_limit_sim": 2.5, "velocity_limit_sim": 12.0, "stiffness": 100.0, "damping": 4.0},
        "support_posture": {
            "mode": "fixed_base nominal lower-body/waist hold",
            "target": "active USD default joint positions",
            "leg_stiffness": 200.0, "leg_damping": 10.0,
            "waist_stiffness": 1000.0, "waist_damping": 40.0,
            "behavior_source": False,
        },
        "allowed_speed_scales": [0.25, 0.5, 1.0],
        "direct_kinematic_joint_write_in_success_run": False,
        "scripted_object_pose": False,
        "scripted_attach_detach": False,
    })

    critical_hashes_after = {str(path.resolve()): sha256_file(path) for path in required}
    freeze_pass = critical_hashes_before == critical_hashes_after
    input_audit = {
        "status": "INPUT_FREEZE_PASS" if freeze_pass else "INPUT_FREEZE_FAIL",
        "source": {
            "path": str(SOURCE.resolve()), "sha256": sha256_file(SOURCE),
            "optimized_action_shape": list(optimized_action.shape),
            "optimized_action_array_sha256": array_sha(optimized_action),
            "timestamps_array_sha256": array_sha(timestamps), "fps": fps, "finite": True,
        },
        "v14": {
            "root_xyz_m": root_position, "forward_offset_m": root_offset,
            "workspace_scale": workspace_scale,
            "arm_q_array_sha256": array_sha(v14_arm_q),
            "left_achieved_backbone_array_sha256": array_sha(target_left),
            "right_achieved_backbone_array_sha256": array_sha(target_right),
            "left_corrected_target_array_sha256": array_sha(corrected_target_left),
            "right_corrected_target_array_sha256": array_sha(corrected_target_right),
        },
        "critical_hashes_before": critical_hashes_before,
        "critical_hashes_after": critical_hashes_after,
        "byte_identical": freeze_pass,
        "v15_inventory_before": v15_inventory_before,
        "v16_inventory_before": v16_inventory_before,
        "validation_read_count": 0, "heldout_read_count": 0, "g1_expert_read_count": 0,
    }
    dump(OUT / "input_freeze_audit.json", input_audit)
    if not freeze_pass:
        raise RuntimeError("immutable input changed during v17 build")

    physics_not_run = {
        "status": "PENDING_TRUE_PHYSICS" if selected_metrics["gate_pass"] else "NOT_RUN_BLOCKED_KINEMATIC_PREPHYSICS",
        "reason": None if selected_metrics["gate_pass"] else "kinematic pre-physics gate did not pass",
        "physics_steps": 0,
        "object_pose_scripted": False,
        "kinematic_object_follow": False,
    }
    for filename in (
        "physics_tracking_metrics.json", "physics_collision_metrics.json",
        "stage_phone_grasp.json", "stage_phone_rotation.json",
        "stage_accessory_removal.json", "stage_bimanual_transport.json",
        "stage_charger_placement.json", "stage_accessory_release.json",
        "full_task_physics_result.json",
    ):
        dump(OUT / filename, physics_not_run)
    save_npz(
        OUT / "phone_object_trajectory.npz",
        status=np.asarray(physics_not_run["status"]), physics_steps=np.asarray(0),
        object_pose_scripted=np.asarray(False),
    )
    save_npz(
        OUT / "accessory_object_trajectory.npz",
        status=np.asarray(physics_not_run["status"]), physics_steps=np.asarray(0),
        object_pose_scripted=np.asarray(False),
    )

    summary = {
        "status": "KINEMATIC_GATE_PASS_PHYSICS_PENDING" if selected_metrics["gate_pass"] else "BLOCKED_KINEMATIC_PREPHYSICS",
        "kinematic_gate_pass": selected_metrics["gate_pass"],
        "physics_structure_pass": physics_audit["structural_pass"],
        "physics_run_required": bool(selected_metrics["gate_pass"] and physics_audit["structural_pass"]),
        "selected_candidate": selected["config"]["name"],
        "dominant_failure_subsystem": None if selected_metrics["gate_pass"] else (
            "PARTIAL_ORIENTATION" if not selected_metrics["phone_grasp_task_orientation"]["pass"]
            else "ARM/HAND_TORSO_COLLISION" if not selected_metrics["collision"]["pass"]
            else "PARTIAL_ORIENTATION" if not selected_metrics["orientation"]["pass"]
            else "ARM_TRACKING"
        ),
    }
    dump(OUT / "build_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if selected_metrics["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
