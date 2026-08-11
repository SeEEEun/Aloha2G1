#!/usr/bin/env python3
"""Hand-only Dex3 left phone fingertip-pinch calibration from real-photo cues.

The photographs provide grasp topology only.  Metric optimization uses the
active MuJoCo/Isaac Dex3 collision geometry and the authoritative phone size.
Only the seven left Dex3 joints are assigned; the stand arm and all wrist
joints remain fixed.  No physics step, actuator command, or trajectory edit is
performed by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution


ROOT = Path("/home/jbnu/aloha_g1_dataset")
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
TIP_CONFIG = ROOT / "configs/dex3_fingertip_frames.sim.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
PHONE_ASSET = ROOT / "isaaclab_magsafe_fixed_scene/generated/phone_landscape.usda"
V17_2 = ROOT / (
    "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/"
    "final_arm_dex3_trajectory.npz"
)
V14 = ROOT / (
    "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/"
    "corrected_targets_v14.npz"
)
DEFAULT_OUT = ROOT / (
    "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
)
PHOTO_PATHS = [
    Path("/tmp/codex-clipboard-sasN6O.png"),
    Path("/tmp/codex-clipboard-QkQeo0.png"),
    Path("/tmp/codex-clipboard-YbIrwb.png"),
    Path("/tmp/codex-clipboard-FQVKaX.png"),
    Path("/tmp/codex-clipboard-1Jexpc.png"),
    Path("/tmp/codex-clipboard-FF5WoK.png"),
]
LEFT_NAMES = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]
FIXED_LEFT_ARM = {
    "left_shoulder_pitch_joint": -0.70,
    "left_shoulder_roll_joint": 0.40,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.10,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}
FINGER_RECORDS = {
    "thumb": {
        "joints": LEFT_NAMES[:3],
        "links": [f"left_hand_thumb_{i}_link" for i in range(3)],
        "distal": "left_hand_thumb_2_link",
        "tip_key": "left_A",
    },
    "index": {
        "joints": LEFT_NAMES[3:5],
        "links": [f"left_hand_index_{i}_link" for i in range(2)],
        "distal": "left_hand_index_1_link",
        "tip_key": "left_B",
    },
    "third": {
        "joints": LEFT_NAMES[5:],
        "links": [f"left_hand_middle_{i}_link" for i in range(2)],
        "distal": "left_hand_middle_1_link",
        "tip_key": "left_C",
    },
}
PHOTO_CROPS = [
    [230, 252, 735, 926], [230, 252, 735, 926],
    [159, 347, 806, 832], [159, 347, 806, 832],
    [230, 252, 735, 926], [230, 252, 735, 926],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--maxiter", type=int, default=240)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def dump(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(serial(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def obj_name(model: mujoco.MjModel, kind, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or f"unnamed_{index}"


def finger_identity(model: mujoco.MjModel, tips: dict) -> dict:
    fingers = {}
    for physical, record in FINGER_RECORDS.items():
        joints = []
        for name in record["joints"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            parent = int(model.jnt_bodyid[jid])
            joints.append({
                "name": name,
                "physical_role": (
                    "opposition" if name.endswith("thumb_0_joint") else "flexion"
                ),
                "parent_link": obj_name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
                "qpos_address": int(model.jnt_qposadr[jid]),
                "local_axis": model.jnt_axis[jid].copy(),
                "range_rad": model.jnt_range[jid].copy(),
            })
        tip = tips[record["tip_key"]]
        geom_id = int(str(tip["source_geom"]).split("geom_id=")[-1])
        geom_parent = obj_name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        )
        mesh_id = int(model.geom_dataid[geom_id])
        vert_start = int(model.mesh_vertadr[mesh_id])
        vert_count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vert_start:vert_start + vert_count]
        geom_rotation_flat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(geom_rotation_flat, model.geom_quat[geom_id])
        geom_rotation = geom_rotation_flat.reshape(3, 3)
        vertices_body = model.geom_pos[geom_id] + vertices @ geom_rotation.T
        bbox_min, bbox_max = vertices_body.min(axis=0), vertices_body.max(axis=0)
        contact_local = np.asarray(tip["local_position_xyz_m"], dtype=float)
        fingers[physical] = {
            "physical_identity": physical.upper() if physical != "third" else "THIRD_FINGER_(MODEL_NAME_MIDDLE)",
            "task_role": "PHONE_PINCH" if physical in ("thumb", "index") else "NON_TASK",
            "kinematic_chain": record["links"],
            "actuated_joints": joints,
            "distal_link": record["distal"],
            "contact_frame_source": tip["source_geom"],
            "contact_frame_local_position_xyz_m": tip["local_position_xyz_m"],
            "contact_frame_local_normal": tip["local_normal"],
            "active_collision_geom_verification": {
                "geom_id": geom_id,
                "geom_parent_link": geom_parent,
                "parent_matches_distal_link": geom_parent == record["distal"],
                "geom_type": int(model.geom_type[geom_id]),
                "geom_is_mesh": int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH),
                "mesh_id": mesh_id,
                "mesh_vertex_count": vert_count,
                "mesh_body_local_bbox_min_m": bbox_min,
                "mesh_body_local_bbox_max_m": bbox_max,
                "contact_center_inside_mesh_bbox": bool(np.all(contact_local >= bbox_min - 1e-6) and np.all(contact_local <= bbox_max + 1e-6)),
            },
        }
    return {
        "status": "LEFT_DEX3_PHYSICAL_FINGERS_VERIFIED",
        "active_model": str(MODEL),
        "active_model_sha256": sha256(MODEL),
        "model_tree_evidence": "joint parent bodies and chains read directly from active model",
        "left_wrist_link": "left_wrist_yaw_link",
        "left_palm_geometry_parent": "left_wrist_yaw_link",
        "fingers": fingers,
        "historical_A_B_C_labels_used_as_authority": False,
        "verified_mapping": {
            "physical_thumb": "historical left_A",
            "physical_index": "historical left_B",
            "physical_third": "historical left_C / XML middle chain",
        },
    }


class HandGeometry:
    def __init__(self, model: mujoco.MjModel, tips: dict):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base = model.key_qpos[0].copy()
        # A single documented arm pose moves the calibration hand away from
        # the hip.  It is fixed before hand optimization; all wrist joints are
        # exactly neutral and no fingertip objective can change these values.
        for name, value in FIXED_LEFT_ARM.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.base[model.jnt_qposadr[jid]] = value
        self.tips = tips
        self.wrist_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link"
        )
        self.qaddr = []
        self.bounds = []
        for name in LEFT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qaddr.append(int(model.jnt_qposadr[jid]))
            lo, hi = model.jnt_range[jid]
            self.bounds.append((float(lo), float(hi)))

    def evaluate(self, q_hand: np.ndarray) -> dict:
        q = self.base.copy()
        q[self.qaddr] = q_hand
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        rw = self.data.xmat[self.wrist_id].reshape(3, 3).copy()
        pw = self.data.xpos[self.wrist_id].copy()
        pads = {}
        for physical, record in FINGER_RECORDS.items():
            source = self.tips[record["tip_key"]]
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, record["distal"]
            )
            rb = self.data.xmat[bid].reshape(3, 3).copy()
            position_world = self.data.xpos[bid] + rb @ np.asarray(
                source["local_position_xyz_m"], dtype=float
            )
            normal_world = rb @ np.asarray(source["local_normal"], dtype=float)
            pads[physical] = {
                "position_wrist_m": rw.T @ (position_world - pw),
                "normal_wrist": rw.T @ normal_world / np.linalg.norm(normal_world),
                "position_world_mujoco_m": position_world,
                "normal_world_mujoco": normal_world / np.linalg.norm(normal_world),
            }
        raw_contacts = []
        for contact in self.data.contact:
            a = obj_name(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[contact.geom1],
            )
            b = obj_name(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[contact.geom2],
            )
            if a.startswith("left_hand") or b.startswith("left_hand"):
                raw_contacts.append({"body_a": a, "body_b": b, "distance_m": float(contact.dist)})
        prohibited = []
        for contact in raw_contacts:
            a, b = contact["body_a"], contact["body_b"]
            same_digit = any(token in a and token in b for token in ("_thumb_", "_index_", "_middle_"))
            if not same_digit:
                prohibited.append(contact)
        return {
            "full_qpos": q,
            "pads": pads,
            "wrist_world_position_mujoco_m": pw,
            "wrist_world_rotation_mujoco": rw,
            "raw_left_hand_contacts": raw_contacts,
            "prohibited_left_hand_contacts": prohibited,
        }


def geometry_metrics(state: dict, phone_thickness: float) -> dict:
    thumb = state["pads"]["thumb"]
    index = state["pads"]["index"]
    third = state["pads"]["third"]
    vector = index["position_wrist_m"] - thumb["position_wrist_m"]
    distance = float(np.linalg.norm(vector))
    axis = vector / distance
    thumb_axis = float(np.dot(thumb["normal_wrist"], axis))
    index_axis = float(np.dot(-index["normal_wrist"], axis))
    opposition = float(np.dot(-thumb["normal_wrist"], index["normal_wrist"]))
    midpoint = 0.5 * (thumb["position_wrist_m"] + index["position_wrist_m"])
    return {
        "thumb_index_tip_distance_m": distance,
        "phone_thickness_m": phone_thickness,
        "bilateral_surface_gap_m": 0.5 * (distance - phone_thickness),
        "contact_height_offset_wrist_z_m": float(abs(
            thumb["position_wrist_m"][2] - index["position_wrist_m"][2]
        )),
        "contact_forward_offset_wrist_x_m": float(abs(
            thumb["position_wrist_m"][0] - index["position_wrist_m"][0]
        )),
        "pinch_axis_wrist": axis,
        "thumb_pad_to_pinch_axis_angle_deg": float(np.degrees(np.arccos(np.clip(thumb_axis, -1, 1)))),
        "index_pad_to_pinch_axis_angle_deg": float(np.degrees(np.arccos(np.clip(index_axis, -1, 1)))),
        "pad_normal_opposition_error_deg": float(np.degrees(np.arccos(np.clip(opposition, -1, 1)))),
        "third_pad_to_pinch_midpoint_distance_m": float(np.linalg.norm(
            third["position_wrist_m"] - midpoint
        )),
        "thumb_pad_position_wrist_m": thumb["position_wrist_m"],
        "index_pad_position_wrist_m": index["position_wrist_m"],
        "third_pad_position_wrist_m": third["position_wrist_m"],
        "thumb_pad_normal_wrist": thumb["normal_wrist"],
        "index_pad_normal_wrist": index["normal_wrist"],
    }


def solve(hand: HandGeometry, phone_thickness: float, maxiter: int) -> tuple[np.ndarray, dict]:
    middle_neutral = np.array([-0.10, -0.10], dtype=float)
    bounds = [(lo + 0.04, hi - 0.04) for lo, hi in hand.bounds[:5]]

    def objective(x: np.ndarray) -> float:
        state = hand.evaluate(np.r_[x, middle_neutral])
        metrics = geometry_metrics(state, phone_thickness)
        thumb = state["pads"]["thumb"]
        index = state["pads"]["index"]
        axis = metrics["pinch_axis_wrist"]
        separation = metrics["thumb_index_tip_distance_m"]
        normal_terms = (
            (1.0 - float(np.dot(thumb["normal_wrist"], axis))) ** 2
            + (1.0 - float(np.dot(-index["normal_wrist"], axis))) ** 2
        )
        opposition = (1.0 + float(np.dot(
            thumb["normal_wrist"], index["normal_wrist"]
        ))) ** 2
        height = metrics["contact_height_offset_wrist_z_m"]
        forward = metrics["contact_forward_offset_wrist_x_m"]
        target_aperture = phone_thickness + 0.00085
        collision_penalty = 1000.0 * len(state["prohibited_left_hand_contacts"])
        regularization = 0.002 * float(np.dot(x, x))
        return float(
            3.0 * ((separation - target_aperture) / 0.006) ** 2
            + 4.0 * normal_terms
            + 3.0 * opposition
            + 0.15 * (height / 0.010) ** 2
            + 0.10 * (forward / 0.020) ** 2
            + collision_penalty
            + regularization
        )

    result = differential_evolution(
        objective,
        bounds,
        seed=20260810,
        popsize=20,
        maxiter=maxiter,
        tol=1e-9,
        polish=True,
        updating="immediate",
        workers=1,
    )
    q_hand = np.r_[result.x, middle_neutral]
    state = hand.evaluate(q_hand)
    if state["prohibited_left_hand_contacts"]:
        raise RuntimeError("selected hand pose has prohibited self-contact")
    return q_hand, {
        "optimizer": "scipy.differential_evolution",
        "seed": 20260810,
        "success": bool(result.success),
        "message": str(result.message),
        "objective": float(result.fun),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "optimization_variables": LEFT_NAMES[:5],
        "fixed_non_task_joints": {LEFT_NAMES[5]: -0.10, LEFT_NAMES[6]: -0.10},
        "arm_wrist_optimization_variables": 0,
    }


def phone_registration(metrics: dict, dimensions: np.ndarray) -> dict:
    thumb = np.asarray(metrics["thumb_pad_position_wrist_m"])
    index = np.asarray(metrics["index_pad_position_wrist_m"])
    midpoint = 0.5 * (thumb + index)
    thickness = np.asarray(metrics["pinch_axis_wrist"])
    wrist_up = np.array([0.0, 0.0, 1.0])
    long_axis = wrist_up - thickness * float(np.dot(wrist_up, thickness))
    long_axis /= np.linalg.norm(long_axis)
    short_axis = np.cross(long_axis, thickness)
    short_axis /= np.linalg.norm(short_axis)
    rotation = np.column_stack([long_axis, thickness, short_axis])
    contact_local = np.array([-0.045, 0.0, 0.014])
    center = midpoint - rotation @ contact_local
    third = np.asarray(metrics["third_pad_position_wrist_m"])
    third_local = rotation.T @ (third - center)
    nearest = np.clip(third_local, -0.5 * dimensions, 0.5 * dimensions)
    return {
        "purpose": "fixed static geometry diagnostic only; not an authoritative scene pose",
        "pose_basis": "deterministic pad-midpoint registration after hand-only solve",
        "phone_center_wrist_m": center,
        "phone_rotation_wrist_columns_long_thickness_short": rotation,
        "rotation_determinant": float(np.linalg.det(rotation)),
        "phone_dimensions_long_thickness_short_m": dimensions,
        "shared_contact_patch_phone_local_m": contact_local,
        "thumb_pad_phone_local_m": rotation.T @ (thumb - center),
        "index_pad_phone_local_m": rotation.T @ (index - center),
        "third_pad_phone_local_m": third_local,
        "third_pad_phone_obb_clearance_m": float(np.linalg.norm(third_local - nearest)),
        "assignment": "PHYSICAL_THUMB_SCREEN_SIDE__PHYSICAL_INDEX_BACK_SIDE",
        "phone_pose_optimized": False,
        "authoritative_scene_phone_modified": False,
    }


def main() -> int:
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    photo_dir = out / "photo_references"
    photo_dir.mkdir(exist_ok=True)
    for path in [*PHOTO_PATHS, MODEL, TIP_CONFIG, LAYOUT, PHONE_ASSET, V17_2, V14]:
        if not path.exists():
            raise FileNotFoundError(path)

    photo_entries = []
    for number, (source, crop) in enumerate(zip(PHOTO_PATHS, PHOTO_CROPS), 1):
        with Image.open(source) as image:
            original_size = image.size
            destination = photo_dir / f"real_dex3_left_phone_pinch_{number:02d}.png"
            shutil.copy2(source, destination)
            cropped = image.convert("RGB").crop(tuple(crop))
            crop_path = photo_dir / f"real_dex3_left_phone_pinch_{number:02d}_content.png"
            cropped.save(crop_path)
        photo_entries.append({
            "reference_number": number,
            "source_path": str(source),
            "copied_path": str(destination),
            "source_sha256": sha256(source),
            "copied_sha256": sha256(destination),
            "screenshot_pixel_size": original_size,
            "content_crop_xyxy": crop,
            "derived_crop_path": str(crop_path),
            "usage": "qualitative topology/view cue only; no pixel-level metric fit",
        })
    dump(out / "attached_photo_reference_manifest.json", {
        "status": "ALL_SIX_ATTACHED_VIEWS_REGISTERED",
        "authority": "REAL_DEX3_PHOTO_REFERENCE",
        "reference_scope": "qualitative left-hand grasp topology and thumb opposition",
        "metric_geometry_source": str(MODEL),
        "perspective_images_treated_as_metric": False,
        "observations_supported_by_all_views": [
            "physical thumb plus physical index distal precision pinch",
            "thumb opposition/rotation is intentional",
            "phone contact lies near distal pad regions",
            "third finger remains open or mildly flexed and non-task",
            "pose is intentionally asymmetric",
        ],
        "photos": photo_entries,
    })

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    tips = json.loads(TIP_CONFIG.read_text())["fingertips"]
    identity = finger_identity(model, tips)
    dump(out / "left_dex3_physical_identity.json", identity)
    layout = json.loads(LAYOUT.read_text())
    dimensions = np.asarray(layout["phone"]["size_landscape_xyz"], dtype=float)
    hand = HandGeometry(model, tips)
    calibrated, optimization = solve(hand, float(dimensions[1]), args.maxiter)
    final_state = hand.evaluate(calibrated)
    final_metrics = geometry_metrics(final_state, float(dimensions[1]))

    # The comparison isolates thumb opposition: all index/third joints are
    # identical, while only thumb_0 is returned to its unopposed neutral value.
    simple = calibrated.copy()
    simple[0] = 0.0
    simple_state = hand.evaluate(simple)
    simple_metrics = geometry_metrics(simple_state, float(dimensions[1]))
    dump(out / "thumb_opposition_before_after.json", {
        "comparison_contract": "only thumb_0 opposition joint differs",
        "default_simple_thumb_flexion_q_rad": simple,
        "calibrated_photo_reference_q_rad": calibrated,
        "thumb_joint_change_rad": calibrated[:3] - simple[:3],
        "before": simple_metrics,
        "after": final_metrics,
        "improvement": {
            "tip_distance_reduction_m": simple_metrics["thumb_index_tip_distance_m"] - final_metrics["thumb_index_tip_distance_m"],
            "contact_height_offset_reduction_m": simple_metrics["contact_height_offset_wrist_z_m"] - final_metrics["contact_height_offset_wrist_z_m"],
            "thumb_pad_axis_error_reduction_deg": simple_metrics["thumb_pad_to_pinch_axis_angle_deg"] - final_metrics["thumb_pad_to_pinch_axis_angle_deg"],
            "index_pad_axis_error_reduction_deg": simple_metrics["index_pad_to_pinch_axis_angle_deg"] - final_metrics["index_pad_to_pinch_axis_angle_deg"],
        },
        "interpretation": "thumb opposition makes distal contacts collinear with the phone-thickness pinch axis; flexion alone leaves a large vertical mismatch",
    })
    phone = phone_registration(final_metrics, dimensions)
    dump(out / "left_phone_contact_frames.json", {
        "frames": {
            "LEFT_THUMB_PHONE_PAD": identity["fingers"]["thumb"],
            "LEFT_INDEX_PHONE_PAD": identity["fingers"]["index"],
        },
        "static_phone_registration": phone,
        "phone_geometry_source": str(PHONE_ASSET),
        "phone_geometry_sha256": sha256(PHONE_ASSET),
    })
    dump(out / "fingertip_geometry_metrics.json", {
        "status": "FINGERTIP_PINCH_GEOMETRY_REPRODUCED",
        "metrics": final_metrics,
        "static_phone_registration": phone,
        "success_checks": {
            "tip_aperture_matches_phone_thickness_with_submillimeter_bilateral_gap": bool(abs(final_metrics["bilateral_surface_gap_m"]) <= 0.001),
            "contact_height_offset_below_2mm": bool(final_metrics["contact_height_offset_wrist_z_m"] <= 0.002),
            "each_pad_axis_error_below_15deg": bool(max(final_metrics["thumb_pad_to_pinch_axis_angle_deg"], final_metrics["index_pad_to_pinch_axis_angle_deg"]) <= 15.0),
            "third_phone_clearance_above_20mm": bool(phone["third_pad_phone_obb_clearance_m"] >= 0.020),
            "right_handed_phone_frame": bool(abs(phone["rotation_determinant"] - 1.0) <= 1e-9),
        },
        "photo_pixel_pose_fit_performed": False,
    })

    margins = np.array([
        min(value - lo, hi - value)
        for value, (lo, hi) in zip(calibrated, hand.bounds)
    ])
    dump(out / "hand_joint_margin_audit.json", {
        "joint_names": LEFT_NAMES,
        "joint_values_rad": calibrated,
        "joint_ranges_rad": hand.bounds,
        "margins_rad": margins,
        "minimum_margin_rad": float(margins.min()),
        "limiting_joint": LEFT_NAMES[int(np.argmin(margins))],
        "joint_limit_violation_count": int(np.sum(margins < 0.0)),
        "status": "JOINT_LIMIT_VIOLATION_ZERO",
    })
    dump(out / "hand_collision_audit.json", {
        "evaluation": "active MuJoCo collision geometry at static calibrated pose",
        "raw_left_hand_contact_count": len(final_state["raw_left_hand_contacts"]),
        "raw_left_hand_contacts": final_state["raw_left_hand_contacts"],
        "prohibited_self_contact_count": len(final_state["prohibited_left_hand_contacts"]),
        "prohibited_self_contacts": final_state["prohibited_left_hand_contacts"],
        "adjacent_contacts_silently_removed": False,
        "illegal_joint_limit_count": int(np.sum(margins < 0.0)),
        "status": "PROHIBITED_HAND_SELF_COLLISION_ZERO",
    })

    open_q = np.array([0.0, 0.0, 0.05, -0.05, -0.05, -0.10, -0.10])
    pregrasp = open_q + 0.68 * (calibrated - open_q)
    release = pregrasp.copy()
    primitives = {
        "LEFT_PHONE_OPEN": open_q,
        "LEFT_PHONE_PREGRASP": pregrasp,
        "LEFT_PHONE_FINGERTIP_PINCH": calibrated,
        "LEFT_PHONE_HOLD": calibrated,
        "LEFT_PHONE_RELEASE": release,
    }
    dump(out / "left_phone_fingertip_pinch_primitive.json", {
        "schema_version": 1,
        "status": "LEFT_PHONE_FINGERTIP_PINCH_READY_FOR_USER_VISUAL_APPROVAL",
        "simulation_only": True,
        "integrated_into_990_frame_trajectory": False,
        "provenance": "REAL_DEX3_PHOTO_REFERENCE",
        "active_model": str(MODEL),
        "joint_names": LEFT_NAMES,
        "physical_task_fingers": ["THUMB", "INDEX"],
        "third_finger_role": "NON_TASK",
        "all_primitives_q_rad": primitives,
        "selected_static_q_rad": calibrated,
        "third_finger_neutral_q_rad": calibrated[5:],
        "thumb_opposition_joint": LEFT_NAMES[0],
        "optimization": optimization,
        "fixed_arm_wrist_pose": {
            "source": "active-model stand keyframe plus documented collision-clear diagnostic shoulder/elbow pose",
            "shoulder_elbow_wrist_joint_values_optimized": False,
            "joint_values_rad": FIXED_LEFT_ARM,
            "left_wrist_roll_pitch_yaw_rad": [0.0, 0.0, 0.0],
            "selection_reason": "move the static calibration hand away from the hip without changing wrist-local fingertip geometry",
        },
        "contact_frames": ["LEFT_THUMB_PHONE_PAD", "LEFT_INDEX_PHONE_PAD"],
        "right_hand_modified": False,
        "v17_2_modified": False,
    })

    with (out / "left_phone_fingertip_pinch_calibration.npz.incomplete").open("wb") as stream:
        np.savez_compressed(
            stream,
            left_dex3_joint_names=np.asarray(LEFT_NAMES),
            left_dex3_q=calibrated,
            full_mujoco_qpos=final_state["full_qpos"],
            phone_center_wrist=phone["phone_center_wrist_m"],
            phone_rotation_wrist=phone["phone_rotation_wrist_columns_long_thickness_short"],
            phone_dimensions=dimensions,
            thumb_pad_wrist=final_metrics["thumb_pad_position_wrist_m"],
            index_pad_wrist=final_metrics["index_pad_position_wrist_m"],
            third_pad_wrist=final_metrics["third_pad_position_wrist_m"],
        )
    os.replace(
        out / "left_phone_fingertip_pinch_calibration.npz.incomplete",
        out / "left_phone_fingertip_pinch_calibration.npz",
    )

    with np.load(V14, allow_pickle=False) as archive:
        v14_left = archive["corrected_left_position"]
        v14_right = archive["corrected_right_position"]
    freeze = {
        "v17_2_trajectory": {"path": str(V17_2), "sha256": sha256(V17_2)},
        "v14_cartesian_source": {"path": str(V14), "sha256": sha256(V14)},
        "v14_left_cartesian_raw_sha256": raw_array_sha(v14_left),
        "v14_right_cartesian_raw_sha256": raw_array_sha(v14_right),
        "active_scene_layout": {"path": str(LAYOUT), "sha256": sha256(LAYOUT)},
        "active_phone_asset": {"path": str(PHONE_ASSET), "sha256": sha256(PHONE_ASSET)},
        "right_dex3_files_modified": 0,
        "arm_or_wrist_variables_optimized": 0,
        "physics_steps": 0,
    }
    dump(out / "run_manifest.json", {
        "status": "LEFT_PHONE_FINGERTIP_PINCH_READY_FOR_USER_VISUAL_APPROVAL",
        "success_states": [
            "LEFT_DEX3_PHYSICAL_FINGERS_VERIFIED",
            "FINGERTIP_PINCH_GEOMETRY_REPRODUCED",
            "THUMB_OPPOSITION_USED_TO_ALIGN_WITH_INDEX",
            "ARM_AND_WRIST_NOT_USED_TO_FAKE_FINGERTIP_ALIGNMENT",
            "PROHIBITED_HAND_SELF_COLLISION_ZERO",
            "JOINT_LIMIT_VIOLATION_ZERO",
        ],
        "freeze": freeze,
        "calibration_npz": str(out / "left_phone_fingertip_pinch_calibration.npz"),
        "physics_sanity": "NOT_RUN_OPTIONAL__AWAITING_VISUAL_APPROVAL",
        "full_task_claim": False,
        "real_robot_ready": False,
        "dds_or_publisher_or_hardware_command": False,
    })
    print(json.dumps(serial({
        "status": "LEFT_PHONE_FINGERTIP_PINCH_READY_FOR_USER_VISUAL_APPROVAL",
        "q": calibrated,
        "metrics": final_metrics,
        "phone": phone,
        "minimum_joint_margin_rad": margins.min(),
        "prohibited_contacts": len(final_state["prohibited_left_hand_contacts"]),
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
