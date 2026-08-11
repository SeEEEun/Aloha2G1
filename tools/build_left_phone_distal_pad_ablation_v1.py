#!/usr/bin/env python3
"""Build a hand-only photo-derived distal-pad Candidate B around frozen A.

The phone registration, arm/wrist pose, non-task third finger, physics, and all
scientific trajectories are read-only.  Candidate B is selected once from
active Dex3 mesh geometry before any PhysX retention result is inspected.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_pad_ablation_v1"
PRIMITIVE = CAL / "left_phone_fingertip_pinch_primitive.json"
V17_2 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"

sys.path.insert(0, str(ROOT / "tools"))
import calibrate_left_phone_pinch_photo_reference_v1 as calibration  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, default=convert, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def mesh_region(
    model: mujoco.MjModel,
    hand: calibration.HandGeometry,
    geom_id: int,
    seed_local: np.ndarray,
    desired_normal_wrist: np.ndarray,
    radius_m: float,
    normal_tolerance_deg: float,
) -> dict:
    body_id = int(model.geom_bodyid[geom_id])
    body_rotation = hand.data.xmat[body_id].reshape(3, 3)
    wrist_rotation = hand.data.xmat[hand.wrist_id].reshape(3, 3)
    desired_normal_body = body_rotation.T @ wrist_rotation @ desired_normal_wrist

    mesh_id = int(model.geom_dataid[geom_id])
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    normal_start = int(model.mesh_normaladr[mesh_id])
    vertices = model.mesh_vert[vertex_start:vertex_start + vertex_count]
    normals = model.mesh_normal[normal_start:normal_start + vertex_count]
    geom_rotation_flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(geom_rotation_flat, model.geom_quat[geom_id])
    geom_rotation = geom_rotation_flat.reshape(3, 3)
    vertices_body = model.geom_pos[geom_id] + vertices @ geom_rotation.T
    normals_body = normals @ geom_rotation.T
    mask = (
        (np.linalg.norm(vertices_body - seed_local, axis=1) <= radius_m)
        & ((normals_body @ desired_normal_body) >= np.cos(np.deg2rad(normal_tolerance_deg)))
    )
    points = vertices_body[mask]
    selected_normals = normals_body[mask]
    if len(points) < 10:
        raise RuntimeError(f"insufficient distal-pad vertices for geom {geom_id}: {len(points)}")
    center = np.mean(points, axis=0)
    normal = np.mean(selected_normals, axis=0)
    normal /= np.linalg.norm(normal)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    tangent_0 = vh[0] - normal * float(np.dot(vh[0], normal))
    tangent_0 /= np.linalg.norm(tangent_0)
    tangent_1 = np.cross(normal, tangent_0)
    tangent_1 /= np.linalg.norm(tangent_1)
    coordinates = np.column_stack([
        (points - center) @ tangent_0,
        (points - center) @ tangent_1,
    ])
    low, high = np.percentile(coordinates, [5.0, 95.0], axis=0)
    return {
        "source_geom_id": geom_id,
        "source_mesh_id": mesh_id,
        "parent_link": calibration.obj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "selection": {
            "seed_local_m": seed_local,
            "maximum_seed_distance_m": radius_m,
            "maximum_normal_angle_to_required_inward_direction_deg": normal_tolerance_deg,
            "selected_vertex_count": int(np.sum(mask)),
            "active_mesh_vertex_count": vertex_count,
        },
        "local_surface_center_m": center,
        "local_surface_normal": normal,
        "local_tangent_0": tangent_0,
        "local_tangent_1": tangent_1,
        "usable_surface_extent_5_95_percentile_m": high - low,
        "surface_extent_is_contact_area": False,
        "interpretation": "distal mesh region near the photo-approved tip whose active surface normals face the fixed phone",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    primitive_hash_before = sha256(PRIMITIVE)
    primitive = load(PRIMITIVE)
    q_a = np.asarray(primitive["selected_static_q_rad"], dtype=float)
    expected = np.asarray([
        -0.517737046, 0.747053166, 0.050425649,
        -0.661925094, -1.705330000, -0.100000000, -0.100000000,
    ])
    if not np.array_equal(np.round(q_a, 9), expected):
        raise RuntimeError("Candidate A q mismatch")

    model = mujoco.MjModel.from_xml_path(str(calibration.MODEL))
    tips = load(calibration.TIP_CONFIG)["fingertips"]
    hand = calibration.HandGeometry(model, tips)
    state_a = hand.evaluate(q_a)
    registration = load(CAL / "left_phone_contact_frames.json")["static_phone_registration"]
    phone_center = np.asarray(registration["phone_center_wrist_m"], dtype=float)
    phone_rotation = np.asarray(registration["phone_rotation_wrist_columns_long_thickness_short"], dtype=float)
    dimensions = np.asarray(registration["phone_dimensions_long_thickness_short_m"], dtype=float)
    thickness_axis = phone_rotation[:, 1]

    regions = {
        "THUMB_DISTAL_INNER_PAD_REGION": mesh_region(
            model, hand, 61,
            np.asarray(tips["left_A"]["local_position_xyz_m"], dtype=float),
            thickness_axis, 0.015, 10.0,
        ),
        "INDEX_DISTAL_INNER_PAD_REGION": mesh_region(
            model, hand, 69,
            np.asarray(tips["left_B"]["local_position_xyz_m"], dtype=float),
            -thickness_axis, 0.015, 15.0,
        ),
    }
    dump(OUT / "distal_pad_regions.json", {
        "status": "ACTIVE_DISTAL_INNER_PAD_REGIONS_DEFINED",
        "active_model": str(calibration.MODEL),
        "active_model_sha256": sha256(calibration.MODEL),
        "physical_identity": {
            "THUMB": "left_hand_thumb_2_link",
            "INDEX": "left_hand_index_1_link",
            "THIRD_NON_TASK": "left_hand_middle_1_link",
        },
        "regions": regions,
        "literal_contact_area_claimed": False,
    })

    body_ids = {
        "thumb": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_2_link"),
        "index": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_1_link"),
    }
    region_by_finger = {
        "thumb": regions["THUMB_DISTAL_INNER_PAD_REGION"],
        "index": regions["INDEX_DISTAL_INNER_PAD_REGION"],
    }

    def evaluate(q_task: np.ndarray) -> tuple[dict, dict]:
        q = np.r_[q_task, q_a[5:]]
        state = hand.evaluate(q)
        wrist_rotation = hand.data.xmat[hand.wrist_id].reshape(3, 3)
        wrist_position = hand.data.xpos[hand.wrist_id]
        pads = {}
        for finger in ("thumb", "index"):
            body_id = body_ids[finger]
            body_rotation = hand.data.xmat[body_id].reshape(3, 3)
            region = region_by_finger[finger]
            position_wrist = wrist_rotation.T @ (
                hand.data.xpos[body_id]
                + body_rotation @ np.asarray(region["local_surface_center_m"])
                - wrist_position
            )
            normal_wrist = wrist_rotation.T @ body_rotation @ np.asarray(region["local_surface_normal"])
            normal_wrist /= np.linalg.norm(normal_wrist)
            pads[finger] = {
                "position_wrist_m": position_wrist,
                "normal_wrist": normal_wrist,
                "position_phone_local_m": phone_rotation.T @ (position_wrist - phone_center),
                "normal_phone_local": phone_rotation.T @ normal_wrist,
            }
        return state, pads

    _, pads_a = evaluate(q_a[:5])
    mean_a = 0.5 * (
        pads_a["thumb"]["position_phone_local_m"]
        + pads_a["index"]["position_phone_local_m"]
    )
    half_thickness = 0.5 * dimensions[1]
    target_center_preload = 0.000175
    target_thickness_coordinate = half_thickness - target_center_preload
    joint_ranges = np.asarray(hand.bounds[:5], dtype=float)
    trust_radius = 0.15
    minimum_margin = 0.02
    lower = np.maximum(joint_ranges[:, 0] + minimum_margin, q_a[:5] - trust_radius)
    upper = np.minimum(joint_ranges[:, 1] - minimum_margin, q_a[:5] + trust_radius)

    def residual(q_task: np.ndarray) -> np.ndarray:
        state, pads = evaluate(q_task)
        thumb = pads["thumb"]
        index = pads["index"]
        thumb_local = thumb["position_phone_local_m"]
        index_local = index["position_phone_local_m"]
        mean_local = 0.5 * (thumb_local + index_local)
        collision_residual = np.full(
            len(state["prohibited_left_hand_contacts"]), 50.0, dtype=float
        )
        return np.r_[
            (thumb_local[1] + target_thickness_coordinate) / 0.0004,
            (index_local[1] - target_thickness_coordinate) / 0.0004,
            (thumb_local[[0, 2]] - index_local[[0, 2]]) / 0.001,
            (mean_local[[0, 2]] - mean_a[[0, 2]]) / 0.002,
            (thumb["normal_wrist"] - thickness_axis) / 0.08,
            (index["normal_wrist"] + thickness_axis) / 0.08,
            np.sqrt(0.1) * (q_task - q_a[:5]) / trust_radius,
            collision_residual,
        ]

    optimization = least_squares(
        residual, q_a[:5], bounds=(lower, upper), max_nfev=5000,
        xtol=1e-13, ftol=1e-13, gtol=1e-13,
    )
    q_b = np.r_[optimization.x, q_a[5:]]
    state_b, pads_b = evaluate(q_b[:5])
    if state_b["prohibited_left_hand_contacts"]:
        raise RuntimeError("Candidate B has prohibited hand self-contact")
    if not np.array_equal(q_b[5:], q_a[5:]):
        raise RuntimeError("Candidate B changed the non-task third finger")
    if float(np.max(np.abs(q_b[:5] - q_a[:5]))) > trust_radius + 1e-12:
        raise RuntimeError("Candidate B exceeded initial photo-pose trust region")

    def geometry_payload(label: str, q: np.ndarray, state: dict, pads: dict) -> dict:
        margins = np.minimum(q - np.asarray([value[0] for value in hand.bounds]),
                             np.asarray([value[1] for value in hand.bounds]) - q)
        thumb_local = pads["thumb"]["position_phone_local_m"]
        index_local = pads["index"]["position_phone_local_m"]
        thumb_angle = np.degrees(np.arccos(np.clip(
            float(np.dot(pads["thumb"]["normal_wrist"], thickness_axis)), -1.0, 1.0
        )))
        index_angle = np.degrees(np.arccos(np.clip(
            float(np.dot(pads["index"]["normal_wrist"], -thickness_axis)), -1.0, 1.0
        )))
        opposition = np.degrees(np.arccos(np.clip(
            float(np.dot(-pads["thumb"]["normal_wrist"], pads["index"]["normal_wrist"])), -1.0, 1.0
        )))
        return {
            "candidate": label,
            "q_rad": q,
            "distal_pad": pads,
            "thumb_pad_to_phone_normal_angle_deg": float(thumb_angle),
            "index_pad_to_phone_normal_angle_deg": float(index_angle),
            "pad_normal_opposition_error_deg": float(opposition),
            "contact_centroid_long_axis_offset_m": float(abs(thumb_local[0] - index_local[0])),
            "contact_centroid_short_axis_offset_m": float(abs(thumb_local[2] - index_local[2])),
            "thumb_pad_center_geometric_penetration_proxy_m": float(max(0.0, half_thickness - abs(thumb_local[1]))),
            "index_pad_center_geometric_penetration_proxy_m": float(max(0.0, half_thickness - abs(index_local[1]))),
            "geometric_penetration_proxy_is_physx_signed_distance": False,
            "minimum_joint_margin_rad": float(np.min(margins)),
            "limiting_joint": calibration.LEFT_NAMES[int(np.argmin(margins))],
            "joint_limit_violation_count": int(np.sum(margins < 0.0)),
            "prohibited_self_collision_count": len(state["prohibited_left_hand_contacts"]),
            "third_q_rad": q[5:],
            "arm_wrist_modified": False,
        }

    geometry_a = geometry_payload("A_PHOTO_FINGERTIP_PINCH", q_a, state_a, pads_a)
    geometry_b = geometry_payload("B_PHOTO_DERIVED_DISTAL_PAD_PINCH", q_b, state_b, pads_b)
    dump(OUT / "candidate_A_geometry.json", geometry_a)
    dump(OUT / "candidate_B_geometry.json", geometry_b)
    dump(OUT / "candidate_B_optimization.json", {
        "status": "CANDIDATE_B_STATIC_GEOMETRY_ACCEPTANCE_PASS",
        "method": "deterministic bounded nonlinear least squares on active distal mesh regions",
        "physics_results_used_for_candidate_selection": False,
        "variables": calibration.LEFT_NAMES[:5],
        "fixed_joints": {calibration.LEFT_NAMES[5]: float(q_a[5]), calibration.LEFT_NAMES[6]: float(q_a[6])},
        "arm_wrist_variables": 0,
        "trust_region_rad_per_task_joint": trust_radius,
        "trust_region_expanded_to_0_25_rad": False,
        "minimum_joint_margin_constraint_rad": minimum_margin,
        "target_distal_pad_center_preload_m": target_center_preload,
        "candidate_A_q_rad": q_a,
        "candidate_B_q_rad": q_b,
        "candidate_B_minus_A_rad": q_b - q_a,
        "maximum_absolute_task_joint_change_rad": float(np.max(np.abs(q_b[:5] - q_a[:5]))),
        "optimizer_success": bool(optimization.success),
        "optimizer_message": optimization.message,
        "optimizer_cost": float(optimization.cost),
        "function_evaluations": int(optimization.nfev),
        "selection_frozen_before_physx_ab_test": True,
        "static_acceptance": {
            "correct_physical_thumb_index": True,
            "third_non_task_and_unchanged": True,
            "distal_near_fingertips": True,
            "power_grasp_conversion": False,
            "prohibited_self_collision_zero": geometry_b["prohibited_self_collision_count"] == 0,
            "joint_limit_violation_zero": geometry_b["joint_limit_violation_count"] == 0,
            "arm_wrist_unchanged": True,
        },
    })
    dump(OUT / "candidate_B_primitive.json", {
        "schema_version": 1,
        "status": "PHOTO_DERIVED_DISTAL_PAD_PINCH_STATIC_CANDIDATE_FROZEN_FOR_AB_TEST",
        "name": "PHOTO_DERIVED_DISTAL_PAD_PINCH",
        "simulation_only": True,
        "selected_q_rad": q_b,
        "candidate_A_q_rad": q_a,
        "joint_names": calibration.LEFT_NAMES,
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger_role": "NON_TASK",
        "third_q_identical_to_candidate_A": True,
        "thumb_opposition_preserved": bool(q_b[0] < 0.0),
        "phone_pose_modified": False,
        "arm_wrist_modified": False,
        "physics_parameters_modified": False,
        "integrated_into_990_frame_trajectory": False,
        "provenance": "REAL_DEX3_PHOTO_REFERENCE_PLUS_ACTIVE_DISTAL_PAD_GEOMETRY",
    })

    # Renderer input: the phone frame is exactly Candidate A's frame while the
    # displayed pad markers use Candidate B's active region centers.
    third_wrist = np.asarray(
        calibration.geometry_metrics(state_b, dimensions[1])["third_pad_position_wrist_m"], dtype=float
    )
    with (OUT / "candidate_B_static_calibration.npz.incomplete").open("wb") as stream:
        np.savez_compressed(
            stream,
            left_dex3_joint_names=np.asarray(calibration.LEFT_NAMES),
            left_dex3_q=q_b,
            full_mujoco_qpos=state_b["full_qpos"],
            phone_center_wrist=phone_center,
            phone_rotation_wrist=phone_rotation,
            phone_dimensions=dimensions,
            thumb_pad_wrist=pads_b["thumb"]["position_wrist_m"],
            index_pad_wrist=pads_b["index"]["position_wrist_m"],
            third_pad_wrist=third_wrist,
        )
    os.replace(
        OUT / "candidate_B_static_calibration.npz.incomplete",
        OUT / "candidate_B_static_calibration.npz",
    )

    primitive_hash_after = sha256(PRIMITIVE)
    freeze_pass = primitive_hash_before == primitive_hash_after
    dump(OUT / "candidate_A_freeze_audit.json", {
        "status": "CANDIDATE_A_UNCHANGED" if freeze_pass else "BLOCKED_CANDIDATE_A_MUTATION",
        "candidate_A_primitive": str(PRIMITIVE),
        "primitive_sha256_before": primitive_hash_before,
        "primitive_sha256_after": primitive_hash_after,
        "q_raw_sha256_before": array_sha256(q_a),
        "q_raw_sha256_after": array_sha256(np.asarray(load(PRIMITIVE)["selected_static_q_rad"], dtype=float)),
        "q_byte_identical": bool(np.array_equal(q_a, np.asarray(load(PRIMITIVE)["selected_static_q_rad"], dtype=float))),
        "v17_2_sha256": sha256(V17_2),
        "v14_cartesian_backbone_sha256": sha256(V14),
        "authoritative_scene_sha256": sha256(SCENE),
        "candidate_A_overwritten": False,
    })
    if not freeze_pass:
        raise RuntimeError("Candidate A mutation detected")
    print(json.dumps(calibration.serial({
        "status": "CANDIDATE_B_STATIC_GEOMETRY_ACCEPTANCE_PASS",
        "q_A": q_a.tolist(),
        "q_B": q_b.tolist(),
        "dq": (q_b - q_a).tolist(),
        "geometry_A": geometry_a,
        "geometry_B": geometry_b,
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
