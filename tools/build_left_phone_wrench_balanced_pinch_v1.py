#!/usr/bin/env python3
"""Build Stage-A geometry for a hand-only wrench-balanced Dex3 pinch.

Candidate A and every whole-body/scenario input are read-only.  Only the five
physical thumb/index joints are variables.  The contact proxy is derived from
the phone-facing support surface of the active collision meshes, rather than
from link origins or one visually chosen tip point.
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
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
OUT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_wrench_balanced_pinch_v1"
FORCE_AUDIT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_retention_force_audit_v1"
PRIMITIVE = CAL / "left_phone_fingertip_pinch_primitive.json"
V17_2 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
SCENE_LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"

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


class ActiveSupportSurfaces:
    """Phone-facing smooth support-surface proxy from active collision meshes."""

    def __init__(self, model: mujoco.MjModel, hand: calibration.HandGeometry, registration: dict):
        self.model = model
        self.hand = hand
        self.phone_center = np.asarray(registration["phone_center_wrist_m"], dtype=float)
        self.phone_rotation = np.asarray(
            registration["phone_rotation_wrist_columns_long_thickness_short"], dtype=float
        )
        self.dimensions = np.asarray(
            registration["phone_dimensions_long_thickness_short_m"], dtype=float
        )
        self.half = 0.5 * self.dimensions
        self.surface_temperature_m = 0.00025
        self.records = {}
        for finger, geom_id, side in (("THUMB", 61, -1), ("INDEX", 69, 1)):
            body_id = int(model.geom_bodyid[geom_id])
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[start:start + count]
            geom_rotation_flat = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(geom_rotation_flat, model.geom_quat[geom_id])
            geom_rotation = geom_rotation_flat.reshape(3, 3)
            vertices_body = model.geom_pos[geom_id] + vertices @ geom_rotation.T
            self.records[finger] = {
                "geom_id": geom_id,
                "mesh_id": mesh_id,
                "body_id": body_id,
                "vertices_body": vertices_body,
                "side": side,
            }

    def evaluate(self, q: np.ndarray) -> tuple[dict, dict]:
        state = self.hand.evaluate(q)
        wrist_rotation = self.hand.data.xmat[self.hand.wrist_id].reshape(3, 3)
        wrist_position = self.hand.data.xpos[self.hand.wrist_id]
        surfaces = {}
        for finger, record in self.records.items():
            body_id = record["body_id"]
            body_rotation = self.hand.data.xmat[body_id].reshape(3, 3)
            body_position = self.hand.data.xpos[body_id]
            points_wrist = (
                wrist_rotation.T
                @ (
                    body_position
                    + record["vertices_body"] @ body_rotation.T
                    - wrist_position
                ).T
            ).T
            points_phone = (points_wrist - self.phone_center) @ self.phone_rotation
            inside_contact_extent = (
                (np.abs(points_phone[:, 0]) <= self.half[0] + 0.003)
                & (np.abs(points_phone[:, 2]) <= self.half[2] + 0.003)
            )
            points = points_phone[inside_contact_extent]
            # Thumb lies on negative thickness side and supports toward +y;
            # index lies on positive side and supports toward -y.
            score = -float(record["side"]) * points[:, 1]
            score -= np.max(score)
            weights = np.exp(score / self.surface_temperature_m)
            weights /= np.sum(weights)
            centroid = weights @ points
            penetration = self.half[1] - abs(float(centroid[1]))
            surfaces[finger] = {
                "support_surface_centroid_phone_local_m": centroid,
                "phone_facing_signed_penetration_proxy_m": penetration,
                "phone_face_normal_phone_local": np.asarray(
                    [0.0, -float(record["side"]), 0.0], dtype=float
                ),
                "active_collision_geom_id": record["geom_id"],
                "active_collision_mesh_id": record["mesh_id"],
                "support_vertex_count": len(points),
                "soft_support_temperature_m": self.surface_temperature_m,
            }
        return state, surfaces


def geometry_payload(
    label: str,
    q: np.ndarray,
    state: dict,
    surfaces: dict,
    phone_up_local: np.ndarray,
    bounds: np.ndarray,
) -> dict:
    thumb = surfaces["THUMB"]["support_surface_centroid_phone_local_m"]
    index = surfaces["INDEX"]["support_surface_centroid_phone_local_m"]
    delta = thumb - index
    midpoint = 0.5 * (thumb + index)
    height_mismatch = abs(float(np.dot(delta, phone_up_local)))
    normal = np.asarray([0.0, 1.0, 0.0])

    def line_distance(point: np.ndarray) -> float:
        return float(np.linalg.norm(point - normal * float(np.dot(point, normal))))

    margins = np.minimum(q - bounds[:, 0], bounds[:, 1] - q)
    return {
        "candidate": label,
        "q_rad": q,
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger_q_rad": q[5:],
        "third_finger_role": "NON_TASK",
        "support_surfaces": surfaces,
        "phone_up_world_expressed_in_phone_local": phone_up_local,
        "contact_height_mismatch_proxy_m": height_mismatch,
        "contact_long_axis_mismatch_m": abs(float(delta[0])),
        "contact_short_axis_mismatch_m": abs(float(delta[2])),
        "contact_midpoint_phone_local_m": midpoint,
        "thumb_normal_line_to_phone_com_distance_m": line_distance(thumb),
        "index_normal_line_to_phone_com_distance_m": line_distance(index),
        "normal_line_distance_mismatch_m": abs(line_distance(thumb) - line_distance(index)),
        "thumb_index_opposition_axis": (index - thumb) / np.linalg.norm(index - thumb),
        "minimum_joint_margin_rad": float(np.min(margins)),
        "limiting_joint": calibration.LEFT_NAMES[int(np.argmin(margins))],
        "joint_limit_violation_count": int(np.sum(margins < 0.0)),
        "prohibited_self_collision_count": len(state["prohibited_left_hand_contacts"]),
        "prohibited_self_collision_records": state["prohibited_left_hand_contacts"],
        "arm_wrist_modified": False,
        "geometry_is_prephysics_proxy": True,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    search_dir = OUT / "search_candidates"
    search_dir.mkdir(parents=True, exist_ok=True)

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
    registration = load(CAL / "left_phone_contact_frames.json")["static_phone_registration"]
    surfaces = ActiveSupportSurfaces(model, hand, registration)
    baseline_trace = np.load(
        FORCE_AUDIT / "baseline_true_physics_v2/static_physics_trace.npz"
    )
    initial_phone_rotation = Rotation.from_quat(
        baseline_trace["phone_pose_xyzw"][0, 3:7]
    )
    phone_up_local = initial_phone_rotation.inv().apply([0.0, 0.0, 1.0])
    phone_up_local /= np.linalg.norm(phone_up_local)
    bounds = np.asarray(hand.bounds, dtype=float)

    state_a, support_a = surfaces.evaluate(q_a)
    geometry_a = geometry_payload(
        "A_PHOTO_FINGERTIP_PINCH", q_a, state_a, support_a, phone_up_local, bounds
    )

    trust_radius = 0.10
    minimum_margin = 0.03
    lower = np.maximum(bounds[:5, 0] + minimum_margin, q_a[:5] - trust_radius)
    upper = np.minimum(bounds[:5, 1] - minimum_margin, q_a[:5] + trust_radius)
    target_penetration_m = 0.0010
    target_midpoint = geometry_a["contact_midpoint_phone_local_m"].copy()
    # Move the force lines modestly toward COM without changing the distal
    # photo topology.  Equality/balance terms have much higher priority.
    target_midpoint[0] = min(target_midpoint[0] + 0.006, -0.035)

    def residual(q_task: np.ndarray) -> np.ndarray:
        q = np.r_[q_task, q_a[5:]]
        state, support = surfaces.evaluate(q)
        thumb = support["THUMB"]["support_surface_centroid_phone_local_m"]
        index = support["INDEX"]["support_surface_centroid_phone_local_m"]
        delta = thumb - index
        midpoint = 0.5 * (thumb + index)
        height = float(np.dot(delta, phone_up_local))
        p_thumb = support["THUMB"]["phone_facing_signed_penetration_proxy_m"]
        p_index = support["INDEX"]["phone_facing_signed_penetration_proxy_m"]
        collision_penalty = np.full(
            len(state["prohibited_left_hand_contacts"]), 100.0, dtype=float
        )
        return np.r_[
            height / 0.00050,
            delta[0] / 0.00050,
            delta[2] / 0.00080,
            (p_thumb - target_penetration_m) / 0.00025,
            (p_index - target_penetration_m) / 0.00025,
            (p_thumb - p_index) / 0.00030,
            (midpoint[0] - target_midpoint[0]) / 0.010,
            (midpoint[2] - target_midpoint[2]) / 0.010,
            np.sqrt(0.20) * (q_task - q_a[:5]) / trust_radius,
            collision_penalty,
        ]

    optimization = least_squares(
        residual, q_a[:5], bounds=(lower, upper), max_nfev=5000,
        xtol=1e-13, ftol=1e-13, gtol=1e-13,
    )
    q_stage_a = np.r_[optimization.x, q_a[5:]]
    state_c, support_c = surfaces.evaluate(q_stage_a)
    geometry_c = geometry_payload(
        "C_STAGE_A_WRENCH_BALANCED_GEOMETRY", q_stage_a, state_c,
        support_c, phone_up_local, bounds,
    )

    # The initial 0.10-rad Stage-A candidate is validated before this expanded
    # candidate is used.  Its measured PhysX run reduced the static height
    # proxy but did not improve sustained wrench/retention, which is the
    # user-authorized condition for expanding the trust region to 0.20 rad.
    expanded_trust_radius = 0.20
    expanded_lower = np.maximum(
        bounds[:5, 0] + minimum_margin, q_a[:5] - expanded_trust_radius
    )
    expanded_upper = np.minimum(
        bounds[:5, 1] - minimum_margin, q_a[:5] + expanded_trust_radius
    )

    def expanded_residual(q_task: np.ndarray) -> np.ndarray:
        q = np.r_[q_task, q_a[5:]]
        state, support = surfaces.evaluate(q)
        thumb = support["THUMB"]["support_surface_centroid_phone_local_m"]
        index = support["INDEX"]["support_surface_centroid_phone_local_m"]
        delta = thumb - index
        midpoint = 0.5 * (thumb + index)
        p_thumb = support["THUMB"]["phone_facing_signed_penetration_proxy_m"]
        p_index = support["INDEX"]["phone_facing_signed_penetration_proxy_m"]
        collision_penalty = np.full(
            len(state["prohibited_left_hand_contacts"]), 100.0, dtype=float
        )
        return np.r_[
            float(np.dot(delta, phone_up_local)) / 0.00040,
            delta[0] / 0.00070,
            delta[2] / 0.00100,
            delta[1] / 0.00200,
            (p_thumb - 0.00120) / 0.00040,
            (p_index - 0.00120) / 0.00040,
            (p_thumb - p_index) / 0.00040,
            (midpoint[0] + 0.040) / 0.004,
            midpoint[2] / 0.020,
            np.sqrt(0.10) * (q_task - q_a[:5]) / expanded_trust_radius,
            collision_penalty,
        ]

    expanded_optimization = least_squares(
        expanded_residual, q_stage_a[:5],
        bounds=(expanded_lower, expanded_upper), max_nfev=5000,
        xtol=1e-13, ftol=1e-13, gtol=1e-13,
    )
    q_expanded = np.r_[expanded_optimization.x, q_a[5:]]
    state_expanded, support_expanded = surfaces.evaluate(q_expanded)
    geometry_expanded = geometry_payload(
        "C_EXPANDED_WRENCH_BALANCED_GEOMETRY", q_expanded,
        state_expanded, support_expanded, phone_up_local, bounds,
    )
    if float(np.max(np.abs(q_expanded[:5] - q_a[:5]))) > expanded_trust_radius + 1e-12:
        raise RuntimeError("expanded Candidate C exceeded 0.20-rad trust region")
    if geometry_expanded["prohibited_self_collision_count"] or geometry_expanded["joint_limit_violation_count"]:
        raise RuntimeError("expanded Candidate C violates geometry gates")
    dump(OUT / "candidate_C_expanded_contact_geometry.json", geometry_expanded)
    dump(search_dir / "candidate_C_expanded_primitive.json", {
        "schema_version": 1,
        "status": "CANDIDATE_C_EXPANDED_TRUST_PHYSICS_VALIDATION_PROPOSAL",
        "name": "WRENCH_BALANCED_THUMB_INDEX_PINCH_EXPANDED",
        "simulation_only": True,
        "selected_q_rad": q_expanded,
        "candidate_A_q_rad": q_a,
        "candidate_C_minus_A_rad": q_expanded - q_a,
        "joint_names": calibration.LEFT_NAMES,
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger_role": "NON_TASK",
        "third_q_identical_to_candidate_A": True,
        "thumb_opposition_preserved": bool(q_expanded[0] < 0.0),
        "arm_wrist_modified": False,
        "phone_pose_modified": False,
        "physics_parameters_modified": False,
        "integrated_into_990_frame_trajectory": False,
        "provenance": "EXPANDED_ONLY_AFTER_0P10_RAD_PHYSX_WRENCH_GATE_FAILED",
    })

    if not np.array_equal(q_stage_a[5:], q_a[5:]):
        raise RuntimeError("Stage A changed non-task third finger")
    if float(np.max(np.abs(q_stage_a[:5] - q_a[:5]))) > trust_radius + 1e-12:
        raise RuntimeError("Stage A exceeded initial trust region")
    if geometry_c["prohibited_self_collision_count"]:
        raise RuntimeError("Stage A has prohibited hand collision")
    if geometry_c["joint_limit_violation_count"]:
        raise RuntimeError("Stage A has joint limit violation")

    dump(OUT / "wrench_balance_objectives.json", {
        "schema_version": 1,
        "status": "WRENCH_BALANCE_OBJECTIVES_FROZEN",
        "primary": [
            "sustained thumb-index contact-height balance about phone COM",
            "balanced contact-force lines and reduced net phone moment",
            "gravity-opposing vertical support",
        ],
        "secondary": [
            "distal inner-pad contact stability",
            "low contact-centroid excursion",
        ],
        "forbidden_success_mechanisms": [
            "deeper penetration than Candidate A", "higher friction",
            "higher effort", "arm or wrist motion", "third-finger support",
        ],
        "variables": calibration.LEFT_NAMES[:5],
        "fixed_non_task_joints": dict(zip(calibration.LEFT_NAMES[5:], q_a[5:])),
        "initial_trust_region_rad_per_task_joint": trust_radius,
        "maximum_permitted_trust_region_rad_per_task_joint": 0.20,
        "physics_refinement_policy": "small local 5-DOF refinement scored over sustained hold, never one sample",
        "Candidate_A_force_audit_evidence": str(FORCE_AUDIT),
    })
    dump(OUT / "candidate_A_contact_geometry.json", geometry_a)
    dump(OUT / "candidate_C_stage_A_contact_geometry.json", geometry_c)
    dump(search_dir / "candidate_C_stage_A_primitive.json", {
        "schema_version": 1,
        "status": "CANDIDATE_C_STAGE_A_FROZEN_FOR_TRUE_PHYSX_VALIDATION",
        "name": "WRENCH_BALANCED_THUMB_INDEX_PINCH_STAGE_A",
        "simulation_only": True,
        "selected_q_rad": q_stage_a,
        "candidate_A_q_rad": q_a,
        "candidate_C_minus_A_rad": q_stage_a - q_a,
        "joint_names": calibration.LEFT_NAMES,
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger_role": "NON_TASK",
        "third_q_identical_to_candidate_A": True,
        "thumb_opposition_preserved": bool(q_stage_a[0] < 0.0),
        "arm_wrist_modified": False,
        "phone_pose_modified": False,
        "physics_parameters_modified": False,
        "integrated_into_990_frame_trajectory": False,
        "provenance": "REAL_DEX3_PHOTO_REFERENCE_PLUS_ACTIVE_COLLISION_SUPPORT_SURFACE_WRENCH_GEOMETRY",
    })
    q_b = np.asarray(load(
        ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_pad_ablation_v1/candidate_B_primitive.json"
    )["selected_q_rad"], dtype=float)
    local_refinement_candidates = []
    for fraction in (0.25, 0.50, 0.75):
        q_local = q_b + fraction * (q_stage_a - q_b)
        name = f"candidate_C_local_B_to_stageA_{int(round(100 * fraction)):02d}"
        local_refinement_candidates.append({
            "name": name,
            "blend_fraction_from_B_toward_stage_A": fraction,
            "q_rad": q_local,
            "maximum_absolute_change_from_A_rad": float(
                np.max(np.abs(q_local[:5] - q_a[:5]))
            ),
        })
        dump(search_dir / f"{name}_primitive.json", {
            "schema_version": 1,
            "status": "CANDIDATE_C_SMALL_LOCAL_PHYSICS_REFINEMENT_PROPOSAL",
            "name": name,
            "simulation_only": True,
            "selected_q_rad": q_local,
            "candidate_A_q_rad": q_a,
            "candidate_C_minus_A_rad": q_local - q_a,
            "joint_names": calibration.LEFT_NAMES,
            "physical_task_fingers": ["THUMB", "INDEX"],
            "third_finger_role": "NON_TASK",
            "third_q_identical_to_candidate_A": True,
            "thumb_opposition_preserved": bool(q_local[0] < 0.0),
            "arm_wrist_modified": False,
            "phone_pose_modified": False,
            "physics_parameters_modified": False,
            "integrated_into_990_frame_trajectory": False,
            "provenance": "SMALL_LOCAL_BLEND_OF_DISTAL_PAD_ABLATION_AND_STAGE_A_WRENCH_GEOMETRY",
        })
    for name, task_delta in (
        ("candidate_C_local_B_balanced_preload_1", np.asarray([0.0, 0.01, 0.01, -0.01, 0.01])),
        ("candidate_C_local_B_balanced_preload_2", np.asarray([0.0, 0.01, 0.01, -0.02, 0.01])),
    ):
        q_local = q_b.copy()
        q_local[:5] += task_delta
        local_refinement_candidates.append({
            "name": name,
            "physics_rationale": "retain B's low early wrench while restoring balanced normal authority without exceeding A penetration",
            "q_rad": q_local,
            "maximum_absolute_change_from_A_rad": float(
                np.max(np.abs(q_local[:5] - q_a[:5]))
            ),
        })
        dump(search_dir / f"{name}_primitive.json", {
            "schema_version": 1,
            "status": "CANDIDATE_C_SMALL_LOCAL_PHYSICS_REFINEMENT_PROPOSAL",
            "name": name,
            "simulation_only": True,
            "selected_q_rad": q_local,
            "candidate_A_q_rad": q_a,
            "candidate_C_minus_A_rad": q_local - q_a,
            "joint_names": calibration.LEFT_NAMES,
            "physical_task_fingers": ["THUMB", "INDEX"],
            "third_finger_role": "NON_TASK",
            "third_q_identical_to_candidate_A": True,
            "thumb_opposition_preserved": bool(q_local[0] < 0.0),
            "arm_wrist_modified": False,
            "phone_pose_modified": False,
            "physics_parameters_modified": False,
            "integrated_into_990_frame_trajectory": False,
            "provenance": "SMALL_LOCAL_WRENCH_REFINEMENT_AROUND_B_WITH_PENETRATION_CAPPED_BY_A",
        })
    one_axis_probes = (
        ("thumb0_minus", np.asarray([-0.03, 0.0, 0.0, 0.0, 0.0])),
        ("thumb0_plus", np.asarray([0.03, 0.0, 0.0, 0.0, 0.0])),
        ("thumb1_plus", np.asarray([0.0, 0.015, 0.0, 0.0, 0.0])),
        ("thumb2_plus", np.asarray([0.0, 0.0, 0.03, 0.0, 0.0])),
        ("index0_minus", np.asarray([0.0, 0.0, 0.0, -0.02, 0.0])),
        ("index0_plus", np.asarray([0.0, 0.0, 0.0, 0.02, 0.0])),
        ("index1_plus", np.asarray([0.0, 0.0, 0.0, 0.0, 0.02])),
    )
    for suffix, task_delta in one_axis_probes:
        name = f"candidate_C_local_probe_{suffix}"
        q_local = q_b.copy()
        q_local[:5] += task_delta
        dump(search_dir / f"{name}_primitive.json", {
            "schema_version": 1,
            "status": "CANDIDATE_C_ONE_AXIS_PHYSICS_SENSITIVITY_PROBE",
            "name": name,
            "simulation_only": True,
            "selected_q_rad": q_local,
            "candidate_A_q_rad": q_a,
            "candidate_C_minus_A_rad": q_local - q_a,
            "joint_names": calibration.LEFT_NAMES,
            "physical_task_fingers": ["THUMB", "INDEX"],
            "third_finger_role": "NON_TASK",
            "third_q_identical_to_candidate_A": True,
            "thumb_opposition_preserved": bool(q_local[0] < 0.0),
            "arm_wrist_modified": False,
            "phone_pose_modified": False,
            "physics_parameters_modified": False,
            "integrated_into_990_frame_trajectory": False,
            "provenance": "ONE_AXIS_LOCAL_PHYSICS_SENSITIVITY_AROUND_GEOMETRIC_B_SEED",
        })
    q_composite = q_b.copy()
    q_composite[:5] += np.asarray([-0.03, 0.01, 0.01, -0.01, 0.02])
    dump(search_dir / "candidate_C_local_composite_balance_primitive.json", {
        "schema_version": 1,
        "status": "CANDIDATE_C_SMALL_LOCAL_PHYSICS_REFINEMENT_PROPOSAL",
        "name": "WRENCH_BALANCED_THUMB_INDEX_PINCH_COMPOSITE_LOCAL",
        "simulation_only": True,
        "selected_q_rad": q_composite,
        "candidate_A_q_rad": q_a,
        "candidate_C_minus_A_rad": q_composite - q_a,
        "joint_names": calibration.LEFT_NAMES,
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger_role": "NON_TASK",
        "third_q_identical_to_candidate_A": True,
        "thumb_opposition_preserved": bool(q_composite[0] < 0.0),
        "arm_wrist_modified": False,
        "phone_pose_modified": False,
        "physics_parameters_modified": False,
        "integrated_into_990_frame_trajectory": False,
        "provenance": "MEASURED_ONE_AXIS_WRENCH_SENSITIVITY_COMPOSITE_WITH_A_CAPPED_PENETRATION",
    })
    state_composite, support_composite = surfaces.evaluate(q_composite)
    geometry_composite = geometry_payload(
        "C_WRENCH_BALANCED_THUMB_INDEX_PINCH", q_composite,
        state_composite, support_composite, phone_up_local, bounds,
    )
    dump(OUT / "candidate_C_contact_geometry.json", geometry_composite)
    dump(OUT / "candidate_C_primitive.json", {
        "schema_version": 1,
        "status": "WRENCH_BALANCED_THUMB_INDEX_PINCH_FROZEN_FOR_FINAL_AC_DIAGNOSTIC",
        "name": "WRENCH_BALANCED_THUMB_INDEX_PINCH",
        "simulation_only": True,
        "selected_q_rad": q_composite,
        "candidate_A_q_rad": q_a,
        "candidate_C_minus_A_rad": q_composite - q_a,
        "joint_names": calibration.LEFT_NAMES,
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger_role": "NON_TASK",
        "third_q_identical_to_candidate_A": True,
        "thumb_opposition_preserved": bool(q_composite[0] < 0.0),
        "arm_wrist_modified": False,
        "phone_pose_modified": False,
        "physics_parameters_modified": False,
        "integrated_into_990_frame_trajectory": False,
        "selection_note": "smallest tested composite that reduced sustained height mismatch and net moment together; final replacement remains gated by A/C retention",
        "provenance": "REAL_DEX3_PHOTO_REFERENCE_PLUS_MEASURED_STATIC_PHYSX_WRENCH_SENSITIVITY",
    })

    # Static Isaac renderer input.  It preserves Candidate A's phone frame and
    # fixed arm/wrist, changing only the five task-finger joints.
    contact_frames = load(CAL / "left_phone_contact_frames.json")["frames"]
    wrist_rotation = hand.data.xmat[hand.wrist_id].reshape(3, 3)
    wrist_position = hand.data.xpos[hand.wrist_id]
    marker_positions = {}
    for label, frame_key in (
        ("thumb", "LEFT_THUMB_PHONE_PAD"),
        ("index", "LEFT_INDEX_PHONE_PAD"),
    ):
        frame = contact_frames[frame_key]
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, frame["distal_link"]
        )
        body_rotation = hand.data.xmat[body_id].reshape(3, 3)
        local_position = np.asarray(
            frame["contact_frame_local_position_xyz_m"], dtype=float
        )
        marker_positions[label] = wrist_rotation.T @ (
            hand.data.xpos[body_id] + body_rotation @ local_position - wrist_position
        )
    third_wrist = np.asarray(
        calibration.geometry_metrics(
            state_composite,
            np.asarray(registration["phone_dimensions_long_thickness_short_m"])[1],
        )["third_pad_position_wrist_m"], dtype=float,
    )
    static_archive = OUT / "candidate_C_static_calibration.npz"
    temporary_archive = static_archive.with_suffix(".npz.incomplete")
    with temporary_archive.open("wb") as stream:
        np.savez_compressed(
            stream,
            left_dex3_joint_names=np.asarray(calibration.LEFT_NAMES),
            left_dex3_q=q_composite,
            full_mujoco_qpos=state_composite["full_qpos"],
            phone_center_wrist=np.asarray(registration["phone_center_wrist_m"]),
            phone_rotation_wrist=np.asarray(
                registration["phone_rotation_wrist_columns_long_thickness_short"]
            ),
            phone_dimensions=np.asarray(
                registration["phone_dimensions_long_thickness_short_m"]
            ),
            thumb_pad_wrist=marker_positions["thumb"],
            index_pad_wrist=marker_positions["index"],
            third_pad_wrist=third_wrist,
        )
    os.replace(temporary_archive, static_archive)
    dump(OUT / "candidate_C_search.json", {
        "schema_version": 1,
        "status": "STAGE_A_GEOMETRIC_CANDIDATE_READY_FOR_PHYSICS",
        "stage_A_method": "bounded nonlinear least squares on active collision support surfaces",
        "physics_results_used_for_stage_A_selection": False,
        "optimizer_success": bool(optimization.success),
        "optimizer_message": optimization.message,
        "optimizer_cost": float(optimization.cost),
        "function_evaluations": int(optimization.nfev),
        "candidate_A_q_rad": q_a,
        "candidate_C_stage_A_q_rad": q_stage_a,
        "candidate_C_stage_A_minus_A_rad": q_stage_a - q_a,
        "maximum_absolute_task_joint_change_rad": float(
            np.max(np.abs(q_stage_a[:5] - q_a[:5]))
        ),
        "initial_trust_region_rad": trust_radius,
        "trust_region_expanded": False,
        "stage_A_geometry_before": geometry_a,
        "stage_A_geometry_after": geometry_c,
        "expanded_trust_candidate": {
            "justification": "0.10-rad Stage-A true-PhysX run did not reduce sustained net moment or retention",
            "source_physics_evidence": str(OUT / "search_stage_A_physics/static_physics_trace.npz"),
            "trust_region_rad": expanded_trust_radius,
            "optimizer_success": bool(expanded_optimization.success),
            "optimizer_message": expanded_optimization.message,
            "optimizer_cost": float(expanded_optimization.cost),
            "q_rad": q_expanded,
            "q_minus_A_rad": q_expanded - q_a,
            "geometry": geometry_expanded,
        },
        "stage_B_small_local_refinement_proposals": local_refinement_candidates,
        "next_stage": "TRUE_PHYSX_SUSTAINED_WRENCH_VALIDATION_AND_OPTIONAL_SMALL_LOCAL_REFINEMENT",
    })

    primitive_hash_after = sha256(PRIMITIVE)
    q_after = np.asarray(load(PRIMITIVE)["selected_static_q_rad"], dtype=float)
    frozen_pass = primitive_hash_before == primitive_hash_after and np.array_equal(q_a, q_after)
    dump(OUT / "candidate_A_freeze_audit.json", {
        "status": "CANDIDATE_A_BYTE_IDENTICAL" if frozen_pass else "BLOCKED_CANDIDATE_A_MUTATION",
        "candidate_A_primitive": str(PRIMITIVE),
        "primitive_sha256_before": primitive_hash_before,
        "primitive_sha256_after": primitive_hash_after,
        "candidate_A_q_sha256_before": array_sha256(q_a),
        "candidate_A_q_sha256_after": array_sha256(q_after),
        "candidate_A_q_byte_identical": bool(np.array_equal(q_a, q_after)),
        "v17_2_trajectory_sha256": sha256(V17_2),
        "v14_cartesian_backbone_sha256": sha256(V14),
        "authoritative_scene_sha256": sha256(SCENE),
        "scene_layout_sha256": sha256(SCENE_LAYOUT),
        "candidate_A_overwritten": False,
    })
    if not frozen_pass:
        raise RuntimeError("Candidate A mutation detected")

    print(json.dumps({
        "status": "STAGE_A_GEOMETRIC_CANDIDATE_READY_FOR_PHYSICS",
        "q_A": q_a.tolist(),
        "q_C_stage_A": q_stage_a.tolist(),
        "dq": (q_stage_a - q_a).tolist(),
        "height_proxy_A_mm": 1000.0 * geometry_a["contact_height_mismatch_proxy_m"],
        "height_proxy_C_mm": 1000.0 * geometry_c["contact_height_mismatch_proxy_m"],
        "penetration_proxy_A_mm": [
            1000.0 * support_a[name]["phone_facing_signed_penetration_proxy_m"]
            for name in ("THUMB", "INDEX")
        ],
        "penetration_proxy_C_mm": [
            1000.0 * support_c[name]["phone_facing_signed_penetration_proxy_m"]
            for name in ("THUMB", "INDEX")
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
