#!/usr/bin/env python3
"""True-PhysX static sanity test for the user-approved left Dex3 phone pinch.

The scientific inputs are read-only.  The diagnostic phone transform is
authored once in the temporary USD before simulation starts.  During timed
physics only articulation position targets are written; no phone pose,
velocity, attachment, or hidden joint is ever authored or commanded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SCENE = ROOT / "isaaclab_magsafe_fixed_scene"
CAL = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
OUT_DEFAULT = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_static_physics_v1"
PRIMITIVE = CAL / "left_phone_fingertip_pinch_primitive.json"
CALIBRATION = CAL / "left_phone_fingertip_pinch_calibration.npz"
CAL_STAGE = CAL / "left_phone_pinch_static_review.usda"
V17_2 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2/final_arm_dex3_trajectory.npz"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz"
PHONE_USD = SCENE / "generated/phone_landscape.usda"
AUTHORITATIVE_SCENE_USD = SCENE / "generated/magsafe_fixed_scene.usda"
SCENE_LAYOUT = SCENE / "scene_layout.json"
ROOT_CONFIG = ROOT / "configs/g1_root_forward_v14.approved.json"
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--pause-at-end", action="store_true")
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--video-fps", type=float, default=30.0)
parser.add_argument("--review-slowdown", type=float, default=1.5)
parser.add_argument(
    "--trial", choices=("open_pregrasp", "closed_hold"), default="open_pregrasp",
    help="closed_hold begins timed PhysX with the tested hand q already applied",
)
parser.add_argument("--hold-duration", type=float, default=1.5)
parser.add_argument("--test-q-json", type=Path)
parser.add_argument("--artifact-prefix", default="left_phone_pinch_static_physics")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.gui:
    args.headless = False
    args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


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


def minimum_jerk(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def angle_deg(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray) -> float:
    return float(np.degrees((Rotation.from_quat(q0_xyzw).inv() * Rotation.from_quat(q1_xyzw)).magnitude()))


def main() -> int:
    import carb
    import cv2
    import omni.physx
    import omni.usd
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import (
        Gf, PhysxSchema, PhysicsSchemaTools, Sdf, Usd, UsdGeom, UsdLux,
        UsdPhysics, UsdShade,
    )
    import warp as wp
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    sys.path.insert(0, str(SCENE))
    from physical_contact_monitor import enable_contact_reporting
    from robot_model_preview_common import compose_stage

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    primitive_hash_before = sha256(PRIMITIVE)
    frozen_before = {
        "approved_primitive": primitive_hash_before,
        "calibration_npz": sha256(CALIBRATION),
        "calibration_stage": sha256(CAL_STAGE),
        "v17_2_trajectory": sha256(V17_2),
        "v14_cartesian_backbone": sha256(V14),
        "phone_asset": sha256(PHONE_USD),
        "authoritative_scene": sha256(AUTHORITATIVE_SCENE_USD),
        "scene_layout": sha256(SCENE_LAYOUT),
        "root_config": sha256(ROOT_CONFIG),
    }
    primitive = json.loads(PRIMITIVE.read_text())
    contact_frames = json.loads((CAL / "left_phone_contact_frames.json").read_text())
    identity = json.loads((CAL / "left_dex3_physical_identity.json").read_text())
    geometric_calibration = json.loads((CAL / "fingertip_geometry_metrics.json").read_text())
    joint_names = primitive["joint_names"]
    primitives = primitive["all_primitives_q_rad"]
    q_open = np.asarray(primitives["LEFT_PHONE_OPEN"], dtype=np.float64)
    q_pre = np.asarray(primitives["LEFT_PHONE_PREGRASP"], dtype=np.float64)
    q_pinch = np.asarray(primitives["LEFT_PHONE_FINGERTIP_PINCH"], dtype=np.float64)
    q_hold = np.asarray(primitives["LEFT_PHONE_HOLD"], dtype=np.float64)
    q_release = np.asarray(primitives["LEFT_PHONE_RELEASE"], dtype=np.float64)
    approved_six = np.asarray(
        [-0.517737, 0.747053, 0.050426, -0.661925, -1.705330, -0.100000, -0.100000]
    )
    if not np.array_equal(np.round(q_pinch, 6), approved_six):
        raise RuntimeError("approved six-decimal pinch q mismatch")
    if not np.array_equal(q_pinch, np.asarray(primitive["selected_static_q_rad"], dtype=float)):
        raise RuntimeError("approved primitive internal q mismatch")
    tested_q = q_pinch.copy()
    tested_q_source = str(PRIMITIVE)
    if args.test_q_json is not None:
        candidate_path = args.test_q_json.resolve()
        candidate = json.loads(candidate_path.read_text())
        tested_q = np.asarray(candidate["selected_q_rad"], dtype=np.float64)
        if tested_q.shape != (7,) or not np.all(np.isfinite(tested_q)):
            raise RuntimeError("invalid test candidate selected_q_rad")
        if not np.array_equal(tested_q[5:], q_pinch[5:]):
            raise RuntimeError("candidate modified the non-task third finger")
        tested_q_source = str(candidate_path)

    fixed_arm = primitive["fixed_arm_wrist_pose"]["joint_values_rad"]
    expected_identity = {
        "thumb": ("left_hand_thumb_2_link", ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"]),
        "index": ("left_hand_index_1_link", ["left_hand_index_0_joint", "left_hand_index_1_joint"]),
        "third": ("left_hand_middle_1_link", ["left_hand_middle_0_joint", "left_hand_middle_1_joint"]),
    }
    for finger, (distal, joints) in expected_identity.items():
        record = identity["fingers"][finger]
        if record["distal_link"] != distal:
            raise RuntimeError(f"physical identity mismatch {finger}")
        if [row["name"] for row in record["actuated_joints"]] != joints:
            raise RuntimeError(f"physical joint identity mismatch {finger}")

    # Read the exact approved diagnostic transform.  It is copied into the
    # temporary stage before SimulationContext is created and never written
    # again during timed physics.
    approved_stage = Usd.Stage.Open(str(CAL_STAGE))
    approved_phone_prim = approved_stage.GetPrimAtPath("/World/DiagnosticPhone")
    phone_matrix = UsdGeom.Xformable(approved_phone_prim).GetLocalTransformation()

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", True)
    settings.set_float_array(
        "/rtx/post/backgroundZeroAlpha/backgroundDefaultColor", [0.97, 0.97, 0.97, 1.0]
    )
    settings.set_float("/rtx/post/tonemap/filmIso", 160.0)
    settings.set_float("/rtx/post/tonemap/exposureTime", 1.0 / 60.0)
    settings.set_float("/rtx/post/tonemap/fNumber", 8.0)
    settings.set_bool("/rtx/post/histogram/enabled", False)

    stage_path = out / "left_phone_pinch_static_true_physics.usda"
    root_offset = float(json.loads(ROOT_CONFIG.read_text())["selected_total_forward_offset_m"])
    stage = compose_stage(stage_path, "G1", G1_USD, "g1", forward_offset_m=root_offset)
    phone_path = "/World/MagSafeScene/Phone"
    phone_prim = stage.GetPrimAtPath(phone_path)
    if not phone_prim.IsValid():
        raise RuntimeError(f"authoritative phone prim missing: {phone_path}")

    # The approved calibration was solved with /World/G1 at identity.  Preserve
    # that exact phone-to-hand registration while expressing it in the
    # authoritative v14 root frame.  The resulting world pose is written once
    # before timed physics; the referenced scene and its objects are untouched.
    root_world = np.asarray(
        UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            stage.GetPrimAtPath("/World/G1")
        ),
        dtype=np.float64,
    ).T
    calibration_phone_world = np.asarray(phone_matrix, dtype=np.float64).T
    approved_phone_transform = root_world @ calibration_phone_world

    rigid = UsdPhysics.RigidBodyAPI.Apply(phone_prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    UsdPhysics.MassAPI.Apply(phone_prim).CreateMassAttr(0.177)
    PhysxSchema.PhysxRigidBodyAPI.Apply(phone_prim).CreateDisableGravityAttr(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(phone_prim).CreateEnableCCDAttr(True)
    collider = stage.GetPrimAtPath(f"{phone_path}/Colliders/Main")
    UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("convexHull")
    # Contact reporting is observational.  It changes no collision/material property.
    enable_contact_reporting(stage, [
        "/World/G1/Asset", phone_path, "/World/MagSafeScene/Table",
    ])

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/PaperWhiteDome")
    dome.CreateIntensityAttr(1350.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/PaperWhiteKey")
    key.CreateIntensityAttr(3200.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.985, 0.965))
    key.CreateAngleAttr(3.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 24.0, 18.0))
    fill = UsdLux.SphereLight.Define(stage, "/World/Lights/PaperWhiteFill")
    fill.CreateIntensityAttr(900.0)
    fill.CreateRadiusAttr(0.35)
    fill.CreateColorAttr(Gf.Vec3f(0.94, 0.97, 1.0))
    UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(0.5, 0.15, 1.25))
    stage.GetRootLayer().Save()
    if not omni.usd.get_context().open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")

    sim = SimulationContext(SimulationCfg(device="cuda:0", use_fabric=True))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/G1/Asset/root_joint", spawn=None,
        actuators={
            # Values are copied unchanged from run_execution_physics_v17.py.
            "fixed_base_legs": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_(hip|knee|ankle)_.*_joint", r"(left|right)_knee_joint"],
                effort_limit_sim=139.0, velocity_limit_sim=32.0, stiffness=200.0, damping=10.0,
            ),
            "fixed_base_waist": ImplicitActuatorCfg(
                joint_names_expr=[r"waist_.*_joint"], effort_limit_sim=35.0,
                velocity_limit_sim=30.0, stiffness=1000.0, damping=40.0,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_(shoulder|wrist)_.*_joint", r"(left|right)_elbow_joint"],
                effort_limit_sim=25.0, velocity_limit_sim=12.0, stiffness=1000.0, damping=40.0,
            ),
            "dex3": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_hand_.*_joint"], effort_limit_sim=2.5,
                velocity_limit_sim=12.0, stiffness=100.0, damping=4.0,
            ),
        },
    ))
    phone_object = RigidObject(RigidObjectCfg(prim_path=phone_path, spawn=None))
    distal_links = {
        "THUMB": "left_hand_thumb_2_link",
        "INDEX": "left_hand_index_1_link",
        "THIRD": "left_hand_middle_1_link",
    }
    sensors = {
        label: ContactSensor(ContactSensorCfg(
            prim_path=f"/World/G1/Asset/{link}", update_period=0.0,
            filter_prim_paths_expr=[phone_path],
            track_contact_points=True, track_friction_forces=True,
            max_contact_data_count_per_prim=64, force_threshold=0.0,
        ))
        for label, link in distal_links.items()
    }
    phone_table_sensor = ContactSensor(ContactSensorCfg(
        prim_path=phone_path, update_period=0.0,
        filter_prim_paths_expr=["/World/MagSafeScene/Table"],
        track_contact_points=True, track_friction_forces=True,
        max_contact_data_count_per_prim=64, force_threshold=0.0,
    ))

    calibration_camera_poses = {
        "overview": ((0.88, -0.42, 0.54), (0.43, 0.18, 0.09)),
        "closeup": ((0.162, 0.005, 0.150), (0.426, 0.203, 0.067)),
        "side": ((0.575, 0.083, 0.401), (0.426, 0.203, 0.067)),
    }
    camera_poses = {
        name: (
            tuple((root_world[:3, :3] @ np.asarray(eye) + root_world[:3, 3]).tolist()),
            tuple((root_world[:3, :3] @ np.asarray(look) + root_world[:3, 3]).tolist()),
        )
        for name, (eye, look) in calibration_camera_poses.items()
    }
    cameras = {
        name: Camera(CameraCfg(
            prim_path=f"/World/StaticPinchCamera_{name}", update_period=0,
            height=args.height, width=args.width, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=50.0 if name != "overview" else 35.0,
                clipping_range=(0.02, 20.0),
            ),
        ))
        for name in camera_poses
    }

    # Raw PhysX callback preserves actual collider identity for self-collision
    # accounting and never changes contact filtering.
    raw_contact_records: list[dict] = []
    callback_error: list[str] = []
    current_step = {"value": -1}

    def path_from_encoded(value) -> str:
        return str(PhysicsSchemaTools.intToSdfPath(value))

    def contact_callback(headers, data) -> None:
        try:
            for header in headers:
                actor0 = path_from_encoded(header.actor0)
                actor1 = path_from_encoded(header.actor1)
                collider0 = path_from_encoded(header.collider0)
                collider1 = path_from_encoded(header.collider1)
                for offset in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                    item = data[offset]
                    point = np.asarray(item.position, dtype=float)
                    normal = np.asarray(item.normal, dtype=float)
                    impulse = np.asarray(item.impulse, dtype=float)
                    raw_contact_records.append({
                        "physics_step": current_step["value"],
                        "actor0": actor0, "actor1": actor1,
                        "collider0": collider0, "collider1": collider1,
                        "point_m": point, "normal": normal,
                        "normal_force_n": abs(float(np.dot(impulse, normal))) / max(float(sim.get_physics_dt()), 1e-12),
                        "separation_m": float(item.separation),
                    })
        except Exception as exc:
            callback_error.append(f"{type(exc).__name__}: {exc}")

    contact_subscription = (
        omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
    )

    sim.reset()
    runtime_names = list(robot.data.joint_names)
    missing = [name for name in list(fixed_arm) + joint_names if name not in runtime_names]
    if missing:
        raise RuntimeError(f"runtime mapping missing {missing}")
    arm_ids = [runtime_names.index(name) for name in fixed_arm]
    hand_ids = [runtime_names.index(name) for name in joint_names]

    def runtime_shape_materials(view) -> list[list[float]]:
        """Read PhysX shape material buffers without authoring any values.

        Columns are the PhysX tensor API's static friction, dynamic friction,
        and restitution.  This readback is deliberately observational.
        """
        values = wp.to_torch(view.get_material_properties()).detach().cpu().numpy()
        return np.asarray(values, dtype=float).reshape(-1, 3).tolist()

    def physics_material_bindings(parent_path: str) -> list[dict]:
        bindings = []
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            return bindings
        for prim in Usd.PrimRange(parent):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            material = None
            try:
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            except Exception:
                material = None
            material_prim = material.GetPrim() if material is not None else Usd.Prim()
            record = {
                "collision_prim": str(prim.GetPath()),
                "physics_material_path": (
                    str(material_prim.GetPath()) if material_prim.IsValid() else None
                ),
            }
            if material_prim.IsValid():
                api = UsdPhysics.MaterialAPI(material_prim)
                physx_api = PhysxSchema.PhysxMaterialAPI(material_prim)
                record.update({
                    "static_friction": api.GetStaticFrictionAttr().Get(),
                    "dynamic_friction": api.GetDynamicFrictionAttr().Get(),
                    "restitution": api.GetRestitutionAttr().Get(),
                    "friction_combine_mode": physx_api.GetFrictionCombineModeAttr().Get(),
                    "restitution_combine_mode": physx_api.GetRestitutionCombineModeAttr().Get(),
                })
            bindings.append(record)
        return bindings

    # Query the actual backend buffers after PhysX initialization.  Individual
    # distal-link views avoid conflating unrelated G1 collision shapes.
    runtime_material_probe = {
        "source": "ACTIVE_PHYSX_SHAPE_MATERIAL_BUFFER_AFTER_SIM_RESET",
        "read_only": True,
        "phone": {
            "shape_materials_static_dynamic_restitution": runtime_shape_materials(
                phone_object.root_view
            ),
            "usd_physics_bindings": physics_material_bindings(phone_path),
        },
        "task_fingers": {},
    }
    for label, link in distal_links.items():
        link_path = f"/World/G1/Asset/{link}"
        link_view = robot._physics_sim_view.create_rigid_body_view(link_path)
        runtime_material_probe["task_fingers"][label] = {
            "link": link,
            "shape_materials_static_dynamic_restitution": runtime_shape_materials(link_view),
            "usd_physics_bindings": physics_material_bindings(link_path),
        }
    dump(out / "runtime_material_probe.json", runtime_material_probe)

    runtime_drive_probe = {
        "joint_names": joint_names,
        "effort_limits_nm": robot.data.joint_effort_limits.torch[0, hand_ids].detach().cpu().numpy(),
        "stiffness_nm_per_rad": robot.data.joint_stiffness.torch[0, hand_ids].detach().cpu().numpy(),
        "damping_nm_s_per_rad": robot.data.joint_damping.torch[0, hand_ids].detach().cpu().numpy(),
        "velocity_limits_rad_s": robot.data.joint_velocity_limits.torch[0, hand_ids].detach().cpu().numpy(),
        "actuator_type": "IsaacLab ImplicitActuator / PhysX position drive",
        "applied_torque_readback_note": (
            "IsaacLab applied_torque is not a physical drive-effort readback for an implicit PhysX drive; "
            "model-requested torque Kp*(q_target-q)-Kd*qdot and PhysX projected joint force are logged separately."
        ),
    }
    dump(out / "runtime_drive_probe.json", runtime_drive_probe)
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    target[0, arm_ids] = torch.as_tensor(list(fixed_arm.values()), dtype=torch.float32, device=robot.device)
    initial_hand_q = tested_q if args.trial == "closed_hold" else q_open
    target[0, hand_ids] = torch.as_tensor(initial_hand_q, dtype=torch.float32, device=robot.device)
    # One initial robot state write; timed execution below is actuator-only.
    robot.write_joint_state_to_sim(target, zero)
    # SimulationContext.reset performs backend initialization steps.  Restore
    # the approved phone transform and zero velocity once, after reset but
    # before the timed loop.  This is the single user-authorized object initial
    # condition; there are no object writes after this point.
    approved_phone_pose = np.r_[
        approved_phone_transform[:3, 3],
        Rotation.from_matrix(approved_phone_transform[:3, :3]).as_quat(),
    ]
    phone_object.write_root_pose_to_sim_index(
        root_pose=torch.as_tensor(approved_phone_pose[None], dtype=torch.float32, device=phone_object.device)
    )
    phone_object.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=phone_object.device)
    )
    sim.forward()
    robot.update(sim.get_physics_dt())
    phone_object.update(sim.get_physics_dt())
    for sensor in sensors.values():
        sensor.update(sim.get_physics_dt(), force_recompute=True)
    for name, camera in cameras.items():
        eye, look = camera_poses[name]
        camera.set_world_poses_from_view(np.asarray([eye], np.float32), np.asarray([look], np.float32))
    if args.gui:
        sim.set_camera_view(*camera_poses["closeup"])

    # Capture the authorized diagnostic initial condition before the first
    # timed PhysX step.  This is render-only: the phone pose and robot state
    # were written exactly once above, and are never written again below.
    if args.trial == "closed_hold":
        sim.forward()
        sim.render()
        sim.render_context.reset_transform_cadence()
        cameras["closeup"].update(sim.get_physics_dt(), force_recompute=True)
        pre_step_rgb = (
            cameras["closeup"].data.output["rgb"].torch[0, ..., :3]
            .detach().cpu().numpy().copy()
        )
        Image.fromarray(pre_step_rgb).save(out / "approved_closed_pose_before_physics.png")

    body_names = list(robot.data.body_names)
    body_ids = {label: body_names.index(link) for label, link in distal_links.items()}
    wrist_id = body_names.index("left_wrist_yaw_link")
    local_contact = {
        "THUMB": np.asarray(contact_frames["frames"]["LEFT_THUMB_PHONE_PAD"]["contact_frame_local_position_xyz_m"], float),
        "INDEX": np.asarray(contact_frames["frames"]["LEFT_INDEX_PHONE_PAD"]["contact_frame_local_position_xyz_m"], float),
    }
    # Third uses the verified distal contact center from the identity artifact.
    local_contact["THIRD"] = np.asarray(
        identity["fingers"]["third"]["contact_frame_local_position_xyz_m"], float
    )

    dt = float(sim.get_physics_dt())
    durations = (
        {"CLOSED_HOLD": float(args.hold_duration)}
        if args.trial == "closed_hold"
        else {"OPEN": 0.5, "PREGRASP": 0.5, "PINCH": 0.75, "HOLD": 1.75, "RELEASE": 0.5}
    )
    boundaries = np.cumsum([0.0, *durations.values()])
    total_time = float(boundaries[-1])
    total_steps = int(round(total_time / dt))
    capture_stride = max(1, int(round((1.0 / args.video_fps) / dt)))

    def phase_and_q(t: float) -> tuple[str, np.ndarray]:
        if args.trial == "closed_hold":
            return "HOLD", tested_q
        if t < boundaries[1]:
            return "OPEN", q_open
        if t < boundaries[2]:
            u = minimum_jerk((t - boundaries[1]) / durations["PREGRASP"])
            return "PREGRASP", (1.0 - u) * q_open + u * q_pre
        if t < boundaries[3]:
            u = minimum_jerk((t - boundaries[2]) / durations["PINCH"])
            return "PINCH", (1.0 - u) * q_pre + u * q_pinch
        if t < boundaries[4]:
            return "HOLD", q_hold
        u = minimum_jerk((t - boundaries[4]) / durations["RELEASE"])
        return "RELEASE", (1.0 - u) * q_hold + u * q_release

    raw_video_paths = {
        name: out / (
            f"{args.artifact_prefix}_contact_closeup.mp4.incomplete.mp4"
            if name == "closeup" else f"{args.artifact_prefix}_{name}.mp4.incomplete.mp4"
        )
        for name in cameras
    }
    final_video_paths = {
        name: out / (
            f"{args.artifact_prefix}_contact_closeup.mp4"
            if name == "closeup" else f"{args.artifact_prefix}_{name}.mp4"
        )
        for name in cameras
    }
    writers = {
        name: cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps,
            (args.width, args.height),
        )
        for name, path in raw_video_paths.items()
    }
    if not all(writer.isOpened() for writer in writers.values()):
        raise RuntimeError("video writer initialization failed")

    def to_numpy(value) -> np.ndarray:
        if hasattr(value, "torch"):
            return value.torch.detach().cpu().numpy()
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        return np.asarray(value)

    def read_contact_sensor(label: str) -> list[dict]:
        sensor = sensors[label]
        view = sensor.contact_view
        forces, points, normals, separations, counts, starts = view.get_contact_data(dt)
        forces = to_numpy(forces).reshape(-1)
        points = to_numpy(points).reshape(-1, 3)
        normals = to_numpy(normals).reshape(-1, 3)
        separations = to_numpy(separations).reshape(-1)
        counts = to_numpy(counts).reshape(sensor.num_sensors, -1).astype(np.int64)
        starts = to_numpy(starts).reshape(sensor.num_sensors, -1).astype(np.int64)
        result = []
        for sensor_index in range(sensor.num_sensors):
            for filter_index in range(counts.shape[1]):
                start = int(starts[sensor_index, filter_index])
                count = int(counts[sensor_index, filter_index])
                for index in range(start, start + count):
                    result.append({
                        "force_n": abs(float(forces[index])),
                        "point_m": points[index].astype(float),
                        "normal": normals[index].astype(float),
                        "separation_m": float(separations[index]),
                    })
        return result

    trace = []
    contact_rows = []
    closeup_frames: list[np.ndarray] = []
    closeup_meta: list[dict] = []
    initial_phone_pose = phone_object.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
    initial_phone_velocity = phone_object.data.root_vel_w.torch[0].detach().cpu().numpy().copy()
    initial_arm_q = robot.data.joint_pos.torch[0, arm_ids].detach().cpu().numpy().copy()
    capture_count = 0
    for step in range(total_steps + 1):
        t = min(step * dt, total_time)
        phase, command = phase_and_q(t)
        target[0, arm_ids] = torch.as_tensor(list(fixed_arm.values()), dtype=torch.float32, device=robot.device)
        target[0, hand_ids] = torch.as_tensor(command, dtype=torch.float32, device=robot.device)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        current_step["value"] = step
        sim.step(render=False)
        robot.update(dt)
        phone_object.update(dt)
        for sensor in sensors.values():
            sensor.update(dt, force_recompute=True)
        phone_table_sensor.update(dt, force_recompute=True)

        actual_hand = robot.data.joint_pos.torch[0, hand_ids].detach().cpu().numpy().copy()
        actual_arm = robot.data.joint_pos.torch[0, arm_ids].detach().cpu().numpy().copy()
        phone_pose = phone_object.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
        phone_velocity = phone_object.data.root_vel_w.torch[0].detach().cpu().numpy().copy()
        pad_world = {}
        for label, body_id in body_ids.items():
            body_p = robot.data.body_pos_w.torch[0, body_id].detach().cpu().numpy().copy()
            body_q = robot.data.body_quat_w.torch[0, body_id].detach().cpu().numpy().copy()
            pad_world[label] = body_p + Rotation.from_quat(body_q).apply(local_contact[label])
        pinch_center = 0.5 * (pad_world["THUMB"] + pad_world["INDEX"])
        frame_contacts = {label: read_contact_sensor(label) for label in sensors}
        contact_force = {
            label: float(sum(row["force_n"] for row in rows))
            for label, rows in frame_contacts.items()
        }
        # ContactSensor vectors are forces on the sensor body.  Their negative
        # is the equal-and-opposite wrench applied to the phone.
        normal_force_on_phone_w = {}
        friction_force_on_phone_w = {}
        total_force_on_phone_w = {}
        contact_centroid_w = {}
        for label, sensor in sensors.items():
            normal_on_hand = sensor.data.force_matrix_w.torch[0, 0, 0].detach().cpu().numpy().copy()
            friction_on_hand = sensor.data.friction_forces_w.torch[0, 0, 0].detach().cpu().numpy().copy()
            normal_force_on_phone_w[label] = -normal_on_hand
            friction_force_on_phone_w[label] = -friction_on_hand
            total_force_on_phone_w[label] = -(normal_on_hand + friction_on_hand)
            points = frame_contacts[label]
            contact_centroid_w[label] = (
                np.mean(np.asarray([row["point_m"] for row in points], dtype=float), axis=0)
                if points else np.full(3, np.nan, dtype=float)
            )
        actual_hand_velocity = robot.data.joint_vel.torch[0, hand_ids].detach().cpu().numpy().copy()
        stiffness = robot.data.joint_stiffness.torch[0, hand_ids].detach().cpu().numpy().copy()
        damping = robot.data.joint_damping.torch[0, hand_ids].detach().cpu().numpy().copy()
        effort_limits = robot.data.joint_effort_limits.torch[0, hand_ids].detach().cpu().numpy().copy()
        model_requested_torque = stiffness * (command - actual_hand) - damping * actual_hand_velocity
        model_clipped_torque = np.clip(model_requested_torque, -effort_limits, effort_limits)
        projected_joint_force = wp.to_torch(
            robot.root_view.get_dof_projected_joint_forces()
        )[0, hand_ids].detach().cpu().numpy().copy()
        table_normal_force_on_phone_w = (
            phone_table_sensor.data.force_matrix_w.torch[0, 0, 0]
            .detach().cpu().numpy().copy()
        )
        table_friction_force_on_phone_w = (
            phone_table_sensor.data.friction_forces_w.torch[0, 0, 0]
            .detach().cpu().numpy().copy()
        )
        contact_rows.append({
            "step": step, "time_s": t, "phase": phase,
            "contacts": frame_contacts, "force_n": contact_force,
        })
        trace.append({
            "step": step, "time_s": t, "phase": phase,
            "commanded_q": command.copy(), "actual_q": actual_hand,
            "actual_arm_q": actual_arm, "phone_pose_xyzw": phone_pose,
            "phone_velocity": phone_velocity, "pinch_center_m": pinch_center,
            "pad_world_m": pad_world, "force_n": contact_force,
            "normal_force_on_phone_w": normal_force_on_phone_w,
            "friction_force_on_phone_w": friction_force_on_phone_w,
            "total_force_on_phone_w": total_force_on_phone_w,
            "contact_centroid_w": contact_centroid_w,
            "actual_hand_velocity": actual_hand_velocity,
            "model_requested_drive_torque_nm": model_requested_torque,
            "model_clipped_drive_torque_nm": model_clipped_torque,
            "projected_joint_force_nm": projected_joint_force,
            "table_normal_force_on_phone_w": table_normal_force_on_phone_w,
            "table_friction_force_on_phone_w": table_friction_force_on_phone_w,
        })

        if step % capture_stride == 0 or step == total_steps:
            # Repaired native post-step state path: PhysX -> readback -> Fabric -> RTX.
            sim.forward()
            sim.render()
            sim.render_context.reset_transform_cadence()
            for name, camera in cameras.items():
                camera.update(dt, force_recompute=True)
                rgb = camera.data.output["rgb"].torch[0, ..., :3].detach().cpu().numpy().copy()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr, (0, 0), (args.width, 77), (20, 20, 20), -1)
                cv2.putText(bgr, f"TRUE PHYSX STATIC PINCH | {phase} | t={t:.3f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .50, (90, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr, f"THUMB {contact_force['THUMB']:.3f}N | INDEX {contact_force['INDEX']:.3f}N | THIRD {contact_force['THIRD']:.3f}N", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .46, (120, 255, 150), 1, cv2.LINE_AA)
                cv2.putText(bgr, "ARM/WRIST FIXED | phone pose not written after start", (10, 67), cv2.FONT_HERSHEY_SIMPLEX, .43, (210, 210, 255), 1, cv2.LINE_AA)
                writers[name].write(bgr)
                if name == "closeup":
                    closeup_frames.append(rgb)
                    closeup_meta.append({
                        "step": step, "time_s": t, "phase": phase,
                        "force_n": contact_force.copy(),
                        "contacts": frame_contacts,
                        "camera_position": camera.data.pos_w.torch[0].detach().cpu().numpy().copy(),
                        "camera_quaternion_ros_xyzw": camera.data.quat_w_ros.torch[0].detach().cpu().numpy().copy(),
                        "intrinsics": camera.data.intrinsic_matrices.torch[0].detach().cpu().numpy().copy(),
                        "pad_world_m": {key: value.copy() for key, value in pad_world.items()},
                    })
            capture_count += 1
            if args.gui and args.review_slowdown > 1.0:
                time.sleep((capture_stride * dt) * (args.review_slowdown - 1.0))

    contact_subscription = None
    for name, writer in writers.items():
        writer.release()
        os.replace(raw_video_paths[name], final_video_paths[name])

    commanded = np.asarray([row["commanded_q"] for row in trace])
    actual = np.asarray([row["actual_q"] for row in trace])
    actual_arm = np.asarray([row["actual_arm_q"] for row in trace])
    phone_pose = np.asarray([row["phone_pose_xyzw"] for row in trace])
    phone_velocity = np.asarray([row["phone_velocity"] for row in trace])
    pinch_centers = np.asarray([row["pinch_center_m"] for row in trace])
    phases = np.asarray([row["phase"] for row in trace])
    force = {label: np.asarray([row["force_n"][label] for row in trace]) for label in sensors}
    active = {label: force[label] > 1e-3 for label in sensors}
    simultaneous = active["THUMB"] & active["INDEX"]
    hold = phases == "HOLD"
    hold_indices = np.flatnonzero(hold)
    hold_start = int(hold_indices[0]) if len(hold_indices) else len(trace) - 1
    hold_end = int(hold_indices[-1]) if len(hold_indices) else len(trace) - 1
    phone_relative = phone_pose[:, :3] - pinch_centers
    slip = float(np.linalg.norm(phone_relative[hold_end] - phone_relative[hold_start]))
    hold_drop = float(phone_pose[hold_end, 2] - phone_pose[hold_start, 2])
    hold_rotation = angle_deg(phone_pose[hold_start, 3:7], phone_pose[hold_end, 3:7])
    longest_simultaneous = longest_true_run(simultaneous)
    simultaneous_duration = longest_simultaneous * dt
    contact_pass = bool(np.any(active["THUMB"]) and np.any(active["INDEX"]) and longest_simultaneous > 0)
    third_primary = bool(
        np.sum(active["THIRD"] & hold) > max(2, 0.25 * max(1, np.sum(hold)))
        or np.max(force["THIRD"]) > max(np.max(force["THUMB"]), np.max(force["INDEX"]), 1e-9)
    )

    # Classify all raw robot self-contacts.  Intended phone contacts are kept
    # separate; no same-hand/wrist-finger contact is ignored.
    self_records = []
    phone_records = []
    for row in raw_contact_records:
        pair_text = f"{row['collider0']} {row['collider1']}"
        g1_0 = "/World/G1/Asset/" in row["collider0"]
        g1_1 = "/World/G1/Asset/" in row["collider1"]
        if g1_0 and g1_1:
            self_records.append(row)
        if phone_path in pair_text and (g1_0 or g1_1):
            phone_records.append(row)
    arm_drift = np.max(np.abs(actual_arm - initial_arm_q), axis=0)
    arm_fixed_pass = bool(float(np.max(arm_drift)) <= 1e-3)
    prohibited_self_collision = len(self_records) > 0
    retention_weak = bool(
        contact_pass and (
            not np.any(simultaneous[hold]) or slip > 0.03 or hold_drop < -0.03
        )
    )
    if not contact_pass:
        status = "LEFT_PHONE_STATIC_PHYSICS_CONTACT_FAIL"
    elif retention_weak:
        status = "LEFT_PHONE_STATIC_PHYSICS_CONTACT_PASS_RETENTION_WARNING"
    elif third_primary or prohibited_self_collision or not arm_fixed_pass:
        status = "LEFT_PHONE_STATIC_PHYSICS_CONTACT_FAIL"
    else:
        status = "LEFT_PHONE_STATIC_PHYSICS_SANITY_PASS"

    # Representative contact locations and normals, preserving physical identity.
    identity_metrics = {}
    for label in sensors:
        samples = [
            {"step": row["step"], "time_s": row["time_s"], "phase": row["phase"], **sample}
            for row in contact_rows for sample in row["contacts"][label]
        ]
        per_step_patch = []
        for sample_step in sorted({sample["step"] for sample in samples}):
            step_samples = [sample for sample in samples if sample["step"] == sample_step]
            points = np.asarray([sample["point_m"] for sample in step_samples], dtype=float)
            forces_step = np.asarray([sample["force_n"] for sample in step_samples], dtype=float)
            separations_step = np.asarray([sample["separation_m"] for sample in step_samples], dtype=float)
            centroid = np.mean(points, axis=0)
            pairwise_span = float(
                np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1))
            ) if len(points) > 1 else 0.0
            pose_step = phone_pose[sample_step]
            phone_local_points = Rotation.from_quat(pose_step[3:7]).inv().apply(
                points - pose_step[:3]
            )
            centroid_phone_local = np.mean(phone_local_points, axis=0)
            per_step_patch.append({
                "step": sample_step,
                "time_s": float(sample_step * dt),
                "contact_point_count": len(step_samples),
                "centroid_world_m": centroid,
                "centroid_phone_local_m": centroid_phone_local,
                "maximum_pairwise_spatial_span_m": pairwise_span,
                "projected_long_axis_span_m": float(np.ptp(phone_local_points[:, 0])),
                "projected_short_axis_span_m": float(np.ptp(phone_local_points[:, 2])),
                "minimum_signed_separation_m": float(np.min(separations_step)),
                "mean_signed_separation_m": float(np.mean(separations_step)),
                "maximum_signed_separation_m": float(np.max(separations_step)),
                "total_normal_force_proxy_n": float(np.sum(forces_step)),
                "maximum_point_force_proxy_n": float(np.max(forces_step)),
            })
        patch_centroids = np.asarray(
            [row["centroid_phone_local_m"] for row in per_step_patch], dtype=float
        ) if per_step_patch else np.zeros((0, 3), dtype=float)
        patch_spans = np.asarray(
            [row["maximum_pairwise_spatial_span_m"] for row in per_step_patch], dtype=float
        ) if per_step_patch else np.zeros(0, dtype=float)
        peak = max(samples, key=lambda value: value["force_n"]) if samples else None
        first = samples[0] if samples else None
        initial_samples = [sample for sample in samples if sample["step"] == 0]
        initial_separations = [sample["separation_m"] for sample in initial_samples]
        identity_metrics[label] = {
            "physical_link": distal_links[label],
            "contact_samples": int(np.sum(active[label])),
            "raw_contact_points": len(samples),
            "contact_duration_all_samples_s": float(np.sum(active[label]) * dt),
            "maximum_force_n": float(np.max(force[label])),
            "mean_force_when_active_n": float(np.mean(force[label][active[label]]) if np.any(active[label]) else 0.0),
            "first_contact": first,
            "peak_contact": peak,
            "initial_solver_contact_points": initial_samples,
            "initial_solver_minimum_signed_separation_m": (
                float(min(initial_separations)) if initial_separations else None
            ),
            "initial_solver_maximum_signed_separation_m": (
                float(max(initial_separations)) if initial_separations else None
            ),
            "initial_solver_contact_point_count": len(initial_samples),
            "contact_patch_proxy": {
                "literal_contact_area_reported": False,
                "proxy_definition": "PhysX contact-point count, pairwise spatial span, phone-local centroid stability, and separation distribution",
                "per_step": per_step_patch,
                "mean_contact_point_count_when_present": (
                    float(np.mean([row["contact_point_count"] for row in per_step_patch]))
                    if per_step_patch else 0.0
                ),
                "mean_pairwise_spatial_span_m": float(np.mean(patch_spans)) if len(patch_spans) else 0.0,
                "maximum_pairwise_spatial_span_m": float(np.max(patch_spans)) if len(patch_spans) else 0.0,
                "phone_local_centroid_maximum_excursion_m": (
                    float(np.max(np.linalg.norm(patch_centroids - patch_centroids[0], axis=1)))
                    if len(patch_centroids) else 0.0
                ),
                "phone_local_centroid_rms_excursion_m": (
                    float(np.sqrt(np.mean(np.sum((patch_centroids - patch_centroids[0]) ** 2, axis=1))))
                    if len(patch_centroids) else 0.0
                ),
            },
        }

    primitive_hash_after = sha256(PRIMITIVE)
    frozen_after = {
        "approved_primitive": primitive_hash_after,
        "calibration_npz": sha256(CALIBRATION),
        "calibration_stage": sha256(CAL_STAGE),
        "v17_2_trajectory": sha256(V17_2),
        "v14_cartesian_backbone": sha256(V14),
        "phone_asset": sha256(PHONE_USD),
        "authoritative_scene": sha256(AUTHORITATIVE_SCENE_USD),
        "scene_layout": sha256(SCENE_LAYOUT),
        "root_config": sha256(ROOT_CONFIG),
    }
    freeze_pass = frozen_before == frozen_after and primitive_hash_before == primitive_hash_after
    dump(out / "input_freeze_audit.json", {
        "status": "APPROVED_LEFT_PHONE_PINCH_UNCHANGED" if freeze_pass else "BLOCKED_INPUT_MUTATION",
        "hashes_before": frozen_before, "hashes_after": frozen_after,
        "all_frozen_inputs_byte_identical": freeze_pass,
        "approved_q_exact_from_artifact_rad": q_pinch,
        "approved_q_rounded_six_decimals_rad": np.round(q_pinch, 6),
        "matches_user_approved_six_decimal_values": bool(np.array_equal(np.round(q_pinch, 6), approved_six)),
        "trial": args.trial, "tested_q_rad": tested_q,
        "tested_q_source": tested_q_source,
        "physical_identity": {
            "THUMB": expected_identity["thumb"], "INDEX": expected_identity["index"],
            "THIRD_NON_TASK": expected_identity["third"],
        },
        "right_hand_modified": False, "v17_2_modified": False,
    })
    dump(out / "physics_setup_audit.json", {
        "status": "TRUE_PHYSX_STATIC_SETUP_PASS",
        "temporary_stage": stage_path,
        "authoritative_scene": AUTHORITATIVE_SCENE_USD,
        "authoritative_scene_composed": True,
        "phone_prim_path": phone_path,
        "phone_asset": PHONE_USD,
        "phone_initial_transform_source": f"{CAL_STAGE}:/World/DiagnosticPhone",
        "calibration_phone_transform_matrix": calibration_phone_world,
        "authoritative_v14_root_transform_matrix": root_world,
        "phone_initial_transform_matrix": approved_phone_transform,
        "authoritative_table_present": bool(stage.GetPrimAtPath("/World/MagSafeScene/Table").IsValid()),
        "phone_mass_kg": 0.177,
        "phone_mass_source": "authoritative phone customData and active v17 runner",
        "friction_authored_or_changed_by_test": False,
        "gravity_enabled": True, "collision_enabled": True,
        "physics_dt_s": dt, "physics_steps": total_steps + 1,
        "trial": args.trial, "tested_q_rad": tested_q,
        "sequence_duration_s": total_time, "phase_durations_s": durations,
        "smooth_interpolation": "minimum_jerk",
        "controller_source": str((SCENE / "run_execution_physics_v17.py").resolve()),
        "controller_values_changed": False,
        "arm": {"effort_limit": 25.0, "velocity_limit": 12.0, "stiffness": 1000.0, "damping": 40.0},
        "dex3": {"effort_limit": 2.5, "velocity_limit": 12.0, "stiffness": 100.0, "damping": 4.0},
        "fabric_enabled": True, "rtx_reads_fabric": True,
        "render_order": ["actuator_target", "write_data_to_sim", "physics_step", "actual_readback", "sim_forward_fabric", "render", "camera_update", "capture"],
    })
    per_joint_tracking = []
    for index, name in enumerate(joint_names):
        error = actual[:, index] - commanded[:, index]
        per_joint_tracking.append({
            "joint": name, "commanded_final_rad": float(commanded[-1, index]),
            "actual_final_rad": float(actual[-1, index]),
            "rmse_rad": float(np.sqrt(np.mean(error**2))),
            "maximum_absolute_error_rad": float(np.max(np.abs(error))),
            "pinch_target_rad": float(tested_q[index]),
        })
    closest_approved_error = np.max(np.abs(actual - tested_q), axis=1)
    closest_task_digit_error = np.max(np.abs(actual[:, :5] - tested_q[:5]), axis=1)
    approved_pose_tracking_pass = bool(float(np.min(closest_approved_error)) <= 0.05)
    dump(out / "dex3_tracking_metrics.json", {
        "status": "DEX3_ACTUATOR_TRACKING_PASS" if approved_pose_tracking_pass else "HAND_ACTUATOR_TRACKING_BLOCKER",
        "joint_names": joint_names, "approved_pinch_q_rad": q_pinch,
        "tested_q_rad": tested_q, "tested_q_source": tested_q_source,
        "per_joint": per_joint_tracking,
        "overall_rmse_rad": float(np.sqrt(np.mean((actual - commanded)**2))),
        "maximum_absolute_error_rad": float(np.max(np.abs(actual - commanded))),
        "closest_actual_to_approved_pinch_max_error_rad": float(np.min(closest_approved_error)),
        "closest_thumb_index_chain_to_approved_max_error_rad": float(np.min(closest_task_digit_error)),
        "approved_pose_tracking_tolerance_rad": 0.05,
        "approved_all_7dof_pose_reached": approved_pose_tracking_pass,
    })
    dump(out / "phone_contact_identity_metrics.json", {
        "status": "THUMB_INDEX_SIMULTANEOUS_CONTACT_PASS" if contact_pass else "THUMB_INDEX_SIMULTANEOUS_CONTACT_FAIL",
        "force_activity_threshold_n": 1e-3,
        "identity": identity_metrics,
        "simultaneous_thumb_index_samples": int(np.sum(simultaneous)),
        "simultaneous_thumb_index_longest_run_samples": longest_simultaneous,
        "simultaneous_thumb_index_longest_duration_s": simultaneous_duration,
        "first_simultaneous_time_s": float(np.flatnonzero(simultaneous)[0] * dt) if np.any(simultaneous) else None,
        "third_is_primary": third_primary,
        "all_raw_phone_contact_records": len(phone_records),
        "callback_errors": callback_error,
    })
    failure_diagnosis = {
        "dominant_subsystem": (
            None if contact_pass else (
                "CLOSED_POSE_NO_BILATERAL_CONTACT"
                if args.trial == "closed_hold"
                else "PHONE_INITIAL_CONDITION_UNSUPPORTED_DURING_OPEN_PREGRASP"
            )
        ),
        "phone_displacement_before_pinch_transition_m": float(
            0.0 if args.trial == "closed_hold" else
            np.linalg.norm(phone_pose[min(len(phone_pose) - 1, int(round(boundaries[2] / dt))), :3] - phone_pose[0, :3])
        ),
        "phone_vertical_displacement_before_pinch_transition_m": float(
            0.0 if args.trial == "closed_hold" else
            phone_pose[min(len(phone_pose) - 1, int(round(boundaries[2] / dt))), 2] - phone_pose[0, 2]
        ),
        "approved_static_bilateral_surface_gap_m": float(
            geometric_calibration["metrics"]["bilateral_surface_gap_m"]
        ),
        "actuator_reached_approved_pinch": approved_pose_tracking_pass,
        "closest_all_7dof_approved_pose_error_rad": float(np.min(closest_approved_error)),
        "closest_thumb_index_chain_approved_pose_error_rad": float(np.min(closest_task_digit_error)),
        "secondary_subsystems": ([] if approved_pose_tracking_pass else ["HAND_ACTUATOR_TRACKING_BLOCKER"]),
        "interpretation": (
            (
                "The approved closed pose did not produce measured bilateral PhysX contact; collision-surface gap diagnosis is required before any preload decision."
                if args.trial == "closed_hold"
                else "The unconstrained phone left the diagnostic grasp region under gravity during the required OPEN/PREGRASP interval before the hand could close. This run therefore does not establish a residual pinch-geometry or collision-mesh blocker."
            ) if not contact_pass else "Physical opposing contact formed."
        ),
    }
    dump(out / "phone_retention_metrics.json", {
        "status": (
            "NOT_EVALUABLE_CONTACT_NOT_FORMED" if not contact_pass
            else "RETENTION_WARNING" if retention_weak
            else "STATIC_RETENTION_SANITY_PASS"
        ),
        "hold_start_s": float(hold_start * dt), "hold_end_s": float(hold_end * dt),
        "phone_initial_pose_xyzw": initial_phone_pose,
        "phone_initial_velocity": initial_phone_velocity,
        "phone_final_pose_xyzw": phone_pose[-1],
        "phone_com_total_displacement_m": float(np.linalg.norm(phone_pose[-1, :3] - phone_pose[0, :3])),
        "phone_hold_com_displacement_m": float(np.linalg.norm(phone_pose[hold_end, :3] - phone_pose[hold_start, :3])),
        "phone_hold_vertical_displacement_m": hold_drop,
        "phone_relative_to_pinch_center_hold_slip_m": slip,
        "phone_hold_orientation_change_deg": hold_rotation,
        "phone_max_linear_speed_m_s": float(np.max(np.linalg.norm(phone_velocity[:, :3], axis=1))),
        "phone_max_angular_speed_rad_s": float(np.max(np.linalg.norm(phone_velocity[:, 3:], axis=1))),
        "phone_immediate_ejection": bool(np.linalg.norm(phone_pose[min(len(phone_pose)-1, int(round(0.25/dt))), :3] - phone_pose[0, :3]) > 0.08),
        "failure_diagnosis": failure_diagnosis,
    })
    dump(out / "collision_audit.json", {
        "status": "PROHIBITED_SELF_COLLISION_ZERO" if not prohibited_self_collision else "BLOCKED_PROHIBITED_SELF_COLLISION",
        "raw_physx_contact_callback_errors": callback_error,
        "raw_contact_records_total": len(raw_contact_records),
        "intended_robot_phone_records": len(phone_records),
        "prohibited_robot_self_contact_records": len(self_records),
        "prohibited_robot_self_contacts": self_records[:200],
        "wrist_finger_contacts_ignored": 0,
        "all_robot_self_contacts_classified_as_prohibited": True,
    })
    dump(out / "no_cheat_audit.json", {
        "status": "NO_CHEAT_TRUE_PHYSICS_PASS",
        "phone_initial_pose_write_before_timed_physics": 1,
        "phone_initial_velocity_zero_write_before_timed_physics": 1,
        "timed_phone_pose_writes": 0, "timed_phone_velocity_writes": 0,
        "object_follow": 0, "scripted_attach": 0, "scripted_detach": 0,
        "hidden_fixed_joint": 0, "direct_link_transform_writes": 0,
        "robot_initial_state_writes_before_timed_physics": 1,
        "robot_direct_state_writes_during_timed_physics": 0,
        "actuator_target_steps": total_steps + 1,
        "gravity_enabled": True, "collision_enabled": True,
        "actual_physx_steps": total_steps + 1,
        "physics_state_source": "ACTUAL_PHYSX_ARTICULATION_AND_RIGID_BODY_READBACK",
        "kinematic_playback_used_as_physics_evidence": False,
        "dds": False, "publisher": False, "hardware_command": False,
    })

    # Full machine-readable trace for reproducibility.
    np.savez_compressed(
        out / "static_physics_trace.npz",
        time_s=np.asarray([row["time_s"] for row in trace]),
        phase=phases, commanded_left_dex3_q=commanded, actual_left_dex3_q=actual,
        actual_fixed_arm_q=actual_arm, phone_pose_xyzw=phone_pose,
        phone_velocity=phone_velocity, pinch_center_m=pinch_centers,
        thumb_phone_force_n=force["THUMB"], index_phone_force_n=force["INDEX"],
        third_phone_force_n=force["THIRD"], simultaneous_thumb_index=simultaneous,
        normal_force_on_phone_w=np.asarray([
            [row["normal_force_on_phone_w"][label] for label in ("THUMB", "INDEX", "THIRD")]
            for row in trace
        ]),
        friction_force_on_phone_w=np.asarray([
            [row["friction_force_on_phone_w"][label] for label in ("THUMB", "INDEX", "THIRD")]
            for row in trace
        ]),
        total_force_on_phone_w=np.asarray([
            [row["total_force_on_phone_w"][label] for label in ("THUMB", "INDEX", "THIRD")]
            for row in trace
        ]),
        contact_centroid_w=np.asarray([
            [row["contact_centroid_w"][label] for label in ("THUMB", "INDEX", "THIRD")]
            for row in trace
        ]),
        actual_left_dex3_velocity=np.asarray([row["actual_hand_velocity"] for row in trace]),
        model_requested_drive_torque_nm=np.asarray([
            row["model_requested_drive_torque_nm"] for row in trace
        ]),
        model_clipped_drive_torque_nm=np.asarray([
            row["model_clipped_drive_torque_nm"] for row in trace
        ]),
        projected_joint_force_nm=np.asarray([row["projected_joint_force_nm"] for row in trace]),
        table_normal_force_on_phone_w=np.asarray([
            row["table_normal_force_on_phone_w"] for row in trace
        ]),
        table_friction_force_on_phone_w=np.asarray([
            row["table_friction_force_on_phone_w"] for row in trace
        ]),
        dex3_effort_limits_nm=np.asarray(runtime_drive_probe["effort_limits_nm"]),
        dex3_stiffness_nm_per_rad=np.asarray(runtime_drive_probe["stiffness_nm_per_rad"]),
        dex3_damping_nm_s_per_rad=np.asarray(runtime_drive_probe["damping_nm_s_per_rad"]),
    )
    with (out / "static_physics_contact_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "time_s", "phase", "thumb_force_n", "index_force_n", "third_force_n", "simultaneous_thumb_index"])
        for index, row in enumerate(trace):
            writer.writerow([index, row["time_s"], row["phase"], force["THUMB"][index], force["INDEX"][index], force["THIRD"][index], bool(simultaneous[index])])

    # Select images by state/contact evidence.  If simultaneous contact never
    # occurs the closest recorded state is explicitly labeled instead.
    capture_times = np.asarray([row["time_s"] for row in closeup_meta])
    capture_forces = {
        label: np.asarray([row["force_n"][label] for row in closeup_meta])
        for label in sensors
    }
    def capture_near_time(target_time: float) -> int:
        return int(np.argmin(np.abs(capture_times - target_time)))
    def first_capture_with(label: str) -> int | None:
        indices = np.flatnonzero(capture_forces[label] > 1e-3)
        return int(indices[0]) if len(indices) else None
    simultaneous_capture = np.flatnonzero(
        (capture_forces["THUMB"] > 1e-3) & (capture_forces["INDEX"] > 1e-3)
    )
    closest_capture = int(np.argmax(np.minimum(capture_forces["THUMB"], capture_forces["INDEX"])))
    selected = [
        (("CLOSED BEFORE/START" if args.trial == "closed_hold" else "OPEN"), capture_near_time(0.0 if args.trial == "closed_hold" else 0.20)),
        (("CLOSED HOLD" if args.trial == "closed_hold" else "PREGRASP"), capture_near_time(min(total_time, 0.50 if args.trial == "closed_hold" else 0.85))),
        ("FIRST THUMB CONTACT" if first_capture_with("THUMB") is not None else "THUMB CONTACT — NONE; CLOSEST STATE", first_capture_with("THUMB") if first_capture_with("THUMB") is not None else closest_capture),
        ("FIRST INDEX CONTACT" if first_capture_with("INDEX") is not None else "INDEX CONTACT — NONE; CLOSEST STATE", first_capture_with("INDEX") if first_capture_with("INDEX") is not None else closest_capture),
        ("SIMULTANEOUS CONTACT" if len(simultaneous_capture) else "CLOSEST STATE — NO SIMULTANEOUS CONTACT", int(simultaneous_capture[0]) if len(simultaneous_capture) else closest_capture),
        ("HOLD", capture_near_time(0.75 * total_time if args.trial == "closed_hold" else 2.50)),
        ("FINAL", len(closeup_frames) - 1),
    ]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    panel_w, panel_h = args.width, args.height + 52
    sheet = Image.new("RGB", (panel_w * 4, panel_h * 2), (247, 247, 247))
    draw = ImageDraw.Draw(sheet)
    for panel, (label, index) in enumerate(selected):
        image = Image.fromarray(closeup_frames[index])
        x = (panel % 4) * panel_w
        y = (panel // 4) * panel_h
        sheet.paste(image, (x, y + 52))
        meta = closeup_meta[index]
        draw.text((x + 10, y + 7), label, fill=(20, 20, 20), font=font)
        draw.text((x + 10, y + 29), f"t={meta['time_s']:.3f}s  T={meta['force_n']['THUMB']:.3f}N  I={meta['force_n']['INDEX']:.3f}N  3rd={meta['force_n']['THIRD']:.3f}N", fill=(40, 70, 40), font=font)
    sheet_path = out / "left_phone_pinch_static_physics_contact_sheet.png"
    sheet.save(sheet_path)

    identity_index = int(simultaneous_capture[0]) if len(simultaneous_capture) else closest_capture
    identity_image = Image.fromarray(closeup_frames[identity_index]).convert("RGB")
    identity_draw = ImageDraw.Draw(identity_image)
    meta = closeup_meta[identity_index]

    def project(point_world: np.ndarray) -> tuple[int, int] | None:
        # Build the same ROS optical basis requested through
        # set_world_poses_from_view: +Z forward, +X right, +Y down.
        eye = np.asarray(camera_poses["closeup"][0], dtype=float)
        look = np.asarray(camera_poses["closeup"][1], dtype=float)
        forward = look - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        delta = np.asarray(point_world) - eye
        point_camera = np.array([np.dot(right, delta), np.dot(down, delta), np.dot(forward, delta)])
        if point_camera[2] <= 1e-6:
            return None
        pixel = meta["intrinsics"] @ point_camera
        pixel = pixel[:2] / pixel[2]
        x, y = int(round(pixel[0])), int(round(pixel[1]))
        return (x, y) if 0 <= x < args.width and 0 <= y < args.height else None

    colors = {"THUMB": (0, 210, 255), "INDEX": (255, 40, 190), "THIRD": (255, 185, 0)}
    label_anchors = {
        "THUMB": (20, 108),
        "INDEX": (485, 108),
        "THIRD": (410, args.height - 42),
    }
    for label in ("THUMB", "INDEX", "THIRD"):
        contacts = meta["contacts"][label]
        point = contacts[0]["point_m"] if contacts else meta["pad_world_m"][label]
        pixel = project(point)
        suffix = "CONTACT" if contacts else "NO CONTACT"
        if label == "THIRD":
            suffix += " — NON-TASK"
        if pixel is not None:
            identity_draw.ellipse((pixel[0]-8, pixel[1]-8, pixel[0]+8, pixel[1]+8), outline=colors[label], width=4)
            anchor = label_anchors[label]
            label_text = f"{label}: {suffix}"
            bbox = identity_draw.textbbox(anchor, label_text, font=font)
            identity_draw.rectangle(
                (bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3),
                fill=(245, 245, 245), outline=colors[label], width=2,
            )
            identity_draw.line((pixel[0], pixel[1], anchor[0], anchor[1] + 9), fill=colors[label], width=3)
            identity_draw.text(anchor, label_text, fill=colors[label], font=font)
    identity_draw.rectangle((8, 8, args.width - 8, 86), fill=(245, 245, 245), outline=(30, 30, 30), width=2)
    identity_draw.text((20, 18), f"PHYSICAL CONTACT IDENTITY | t={meta['time_s']:.3f}s | {meta['phase']}", fill=(20, 20, 20), font=font)
    identity_draw.text((20, 47), f"THUMB={meta['force_n']['THUMB']:.3f}N  INDEX={meta['force_n']['INDEX']:.3f}N  THIRD={meta['force_n']['THIRD']:.3f}N", fill=(30, 80, 30), font=font)
    identity_path = out / "left_phone_pinch_physics_contact_identity.png"
    identity_image.save(identity_path)

    next_action = (
        "INTEGRATE THE APPROVED LEFT_PHONE_FINGERTIP_PINCH INTO THE FULL 990-FRAME SEMANTIC TRAJECTORY AND THEN ADDRESS V17.2 TEMPORAL JITTER."
        if status in ("LEFT_PHONE_STATIC_PHYSICS_SANITY_PASS", "LEFT_PHONE_STATIC_PHYSICS_CONTACT_PASS_RETENTION_WARNING")
        else "DIAGNOSE THE STATIC THUMB/INDEX CONTACT FAILURE BEFORE FULL-TRAJECTORY INTEGRATION."
    )
    result = {
        "status": status, "pass": status != "LEFT_PHONE_STATIC_PHYSICS_CONTACT_FAIL",
        "trial": args.trial, "tested_q_rad": tested_q,
        "approved_primitive_unchanged": freeze_pass,
        "thumb_contact": bool(np.any(active["THUMB"])),
        "index_contact": bool(np.any(active["INDEX"])),
        "simultaneous_thumb_index_contact": bool(np.any(simultaneous)),
        "simultaneous_contact_duration_s": simultaneous_duration,
        "third_remained_non_task": not third_primary,
        "arm_wrist_fixed": arm_fixed_pass,
        "arm_wrist_maximum_drift_rad": float(np.max(arm_drift)),
        "prohibited_self_collision_zero": not prohibited_self_collision,
        "retention_warning": retention_weak,
        "failure_diagnosis": failure_diagnosis,
        "physics_steps": total_steps + 1,
        "video_paths": {key: str(path.resolve()) for key, path in final_video_paths.items()},
        "contact_sheet": str(sheet_path.resolve()),
        "identity_overlay": str(identity_path.resolve()),
        "next_action": next_action,
    }
    dump(out / "static_physics_result.json", result)

    gui_command = f"""source /home/jbnu/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab6
cd /home/jbnu/aloha_g1_dataset
DISPLAY=:0 /home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir {out}/gui_review \\
  --gui --pause-at-end --enable_cameras
"""
    commands = f"""#!/usr/bin/env bash
set -euo pipefail

# Headless reproducible run
cd /home/jbnu/aloha_g1_dataset
/home/jbnu/miniconda3/envs/isaaclab6/bin/python \\
  isaaclab_magsafe_fixed_scene/run_left_phone_pinch_static_physics_v1.py \\
  --output-dir {out} --headless --enable_cameras

# Interactive true-physics review (actual PhysX -> Fabric -> RTX)
{gui_command}
"""
    (out / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(out / "commands.sh", 0o755)
    report = f"""# Dex3 left phone fingertip pinch — static true-physics sanity

Status: `{status}`

The user-approved seven-DOF left-hand primitive was loaded without modification.  The
physical thumb is `{distal_links['THUMB']}`, the physical index is
`{distal_links['INDEX']}`, and the model's middle chain is the non-task third finger.
The fixed calibration arm/wrist target was held throughout; its maximum actual drift
was {float(np.max(arm_drift)):.9f} rad.

## Physical result

- Thumb phone contact: {bool(np.any(active['THUMB']))} ({int(np.sum(active['THUMB']))} samples, max {float(np.max(force['THUMB'])):.6f} N)
- Index phone contact: {bool(np.any(active['INDEX']))} ({int(np.sum(active['INDEX']))} samples, max {float(np.max(force['INDEX'])):.6f} N)
- Simultaneous thumb+index: {bool(np.any(simultaneous))}, longest {simultaneous_duration:.6f} s
- Third-finger primary support: {third_primary} (max {float(np.max(force['THIRD'])):.6f} N)
- Hold slip relative to pinch center: {slip*1000.0:.3f} mm
- Hold vertical displacement: {hold_drop*1000.0:.3f} mm
- Prohibited robot self-contact records: {len(self_records)}

## Failure diagnosis

{failure_diagnosis['interpretation']}  Phone displacement before PINCH began was
{failure_diagnosis['phone_displacement_before_pinch_transition_m']:.6f} m.  The
all-7-DOF approved PINCH pose reached status was
{approved_pose_tracking_pass}; closest max-joint error was
{float(np.min(np.max(np.abs(actual - q_pinch), axis=1))):.9f} rad.  The prior
static metric still records a nonpenetrating bilateral gap of
{failure_diagnosis['approved_static_bilateral_surface_gap_m']*1000.0:.3f} mm per side,
but this run cannot isolate that gap because the free phone had already fallen away.

The phone transform was authored once in the temporary stage from the approved
calibration pose.  Timed phone pose/velocity writes, object-follow, scripted
attachments, and hidden grasp joints were all zero.  Gravity, collision, and actual
actuator-driven PhysX stepping remained enabled.

## GUI

```bash
{gui_command}```

## Next action

`{next_action}`
"""
    (out / "report.md").write_text(report, encoding="utf-8")

    # Keep the manifest last so every listed artifact is complete.
    artifacts = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "run_manifest.json" and not path.name.endswith(".usda"):
            artifacts[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    dump(out / "run_manifest.json", {
        "status": status, "generated_at_epoch_s": time.time(),
        "approved_primitive_sha256": primitive_hash_after,
        "fixed_arm_wrist_pose": fixed_arm,
        "physics_steps": total_steps + 1,
        "right_hand_modified": False, "v17_2_modified": False,
        "friction_or_controller_tuning": False,
        "artifacts": artifacts,
    })

    print(json.dumps(result, indent=2), flush=True)
    if args.gui and args.pause_at_end:
        print("[GUI] Final true-physics state held for orbit/pan/zoom.", flush=True)
        while simulation_app.is_running():
            sim.forward()
            sim.render()
            sim.render_context.reset_transform_cadence()
            simulation_app.update()
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
