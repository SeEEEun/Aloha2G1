"""Replay a recorded Stationary ALOHA episode in the Isaac Lab MagSafe scene."""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent
ALOHA_USD = Path("/home/jbnu/robot_assets/stationary_aloha/usd_imported/stationary_aloha_imported.usd")
DEFAULT_DATASET = Path("/home/jbnu/aloha_g1_dataset/raw_recordings/GoPark_20260724_154440")
OUTPUT = ROOT / "generated" / "magsafe_aloha_episode_replay.usda"
MATERIAL_LAYER = ROOT / "generated" / "magsafe_phone_tape_material_layer.usda"
BARE_PHONE_LAYER = ROOT / "generated" / "magsafe_bare_phone_split_material_layer.usda"
TAPE_SLEEVE_LAYER = ROOT / "generated" / "aloha_distal_20mm_tape_sleeve_layer.usda"
ACCESSORY_HOLLOW_RING_LAYER = (
    ROOT / "generated" / "magsafe_accessory_hollow_ring_geometry_layer.usda"
)
REPORT_ROOT = ROOT / "generated" / "reports"
JOINT_NAMES = [
    *(f"follower_left_joint_{i}" for i in range(6)),
    "follower_left_left_carriage_joint", "follower_left_right_carriage_joint",
    *(f"follower_right_joint_{i}" for i in range(6)),
    "follower_right_left_carriage_joint", "follower_right_right_carriage_joint",
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--source", choices=("observation.state", "action"), default=None, help=argparse.SUPPRESS)
parser.add_argument("--arm-source", choices=("observation.state",), default="observation.state")
parser.add_argument("--gripper-source", choices=("observation.state", "action"), default=None)
parser.add_argument("--speed", type=float, choices=(.25, .5, 1.0), default=1.0)
parser.add_argument("--loop", action="store_true")
parser.add_argument(
    "--scene-mode",
    choices=("kinematic", "contact", "physical-task", "inspection", "magnetic"),
    default="physical-task",
)
parser.add_argument("--magnet-config", type=Path, default=ROOT / "magnet_config_v2.json")
parser.add_argument("--camera", choices=("overview", "front", "side", "top"), default="overview")
parser.add_argument(
    "--friction-profile", choices=("current", "tape-phone", "bare-phone-split"), default="current"
)
parser.add_argument("--gripper-drive-scale", type=float, choices=(1.0, 1.25, 1.5), default=1.0)
parser.add_argument(
    "--gripper-stiffness",
    type=float,
    default=None,
    help="Replay-only absolute carriage stiffness; damping=10, effort limit=60 N, velocity limit=0.5 m/s.",
)
parser.add_argument("--contact-diagnostic-tag", type=str, default="")
parser.add_argument("--tape-sleeve-config", type=Path, default=None)
parser.add_argument("--show-tape-sleeve", action="store_true")
parser.add_argument("--show-accessory-ring-collider", action="store_true")
parser.add_argument(
    "--gripper-command-mode",
    choices=("recorded", "hardware-max-close"),
    default="recorded",
)
parser.add_argument(
    "--gripper-drive-mode",
    choices=("baseline", "hardware-max", "effort-cap"),
    default="baseline",
)
parser.add_argument(
    "--gripper-effort-cap",
    type=float,
    default=None,
    help="Per-carriage force cap in N for --gripper-drive-mode effort-cap.",
)
parser.add_argument("--show-gripper-effort", action="store_true")
parser.add_argument("--ring-affordance-controller", action="store_true")
parser.add_argument(
    "--ring-pose-provider",
    choices=("simulation_ground_truth", "perception_estimate", "fixed_calibration"),
    default="simulation_ground_truth",
)
parser.add_argument("--show-ring-frame", action="store_true")
parser.add_argument("--show-ring-insertion-target", action="store_true")
parser.add_argument("--show-ring-clearance", action="store_true")
parser.add_argument("--show-right-ik-target", action="store_true")
parser.add_argument(
    "--phone-accessory-joint-mode",
    choices=("existing", "corrected-fixed"),
    default="existing",
    help="Replay-layer correction of the breakable phone-accessory fixed-joint frames.",
)
parser.add_argument(
    "--phone-pose-audit",
    action="store_true",
    help="Log phone/visual/wrist/target transform provenance and draw pose-debug markers.",
)
parser.add_argument("--stability-test-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
parser.add_argument("--end-frame", type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--start-frame", type=int, default=0,
    help="Begin diagnostic logging; physics still advances from frame 0.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
REPORT_ROOT = Path("/home/jbnu/aloha_g1_dataset/converted_runs") / (
    f"magsafe_{args.dataset.name.removeprefix('GoPark_')}"
) / "isaac_replay"
# Complete all Arrow/protobuf work before Kit initializes its own gRPC stack.
_dataset_preload = args.dataset.expanduser().resolve()
_info_preload = json.loads((_dataset_preload / "meta" / "info.json").read_text())
_parquet_preload = (
    _dataset_preload
    / "data"
    / "chunk-000"
    / f"episode_{args.episode:06d}.parquet"
)
_table_preload = pq.read_table(_parquet_preload, columns=["observation.state", "action", "timestamp"])
_state_preload = np.asarray(_table_preload["observation.state"].to_pylist(), dtype=np.float32)
_action_preload = np.asarray(_table_preload["action"].to_pylist(), dtype=np.float32)
_timestamps_preload = np.asarray(_table_preload["timestamp"].to_pylist(), dtype=np.float64)
del _table_preload
launcher = AppLauncher(args)
simulation_app = launcher.app
print("[ALOHA] AppLauncher returned; importing replay modules.", flush=True)

from robot_model_preview_common import CAMERAS, compose_stage, suppress_stationary_aloha_fixture
print("[ALOHA] replay modules imported.", flush=True)


def parquet_path(dataset: Path, episode: int) -> Path:
    path = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def freeze_objects(stage) -> None:
    from pxr import UsdPhysics
    for prim in stage.Traverse():
        if any(word in prim.GetName().lower() for word in ("phone", "accessory", "charger")):
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if api:
                api.GetKinematicEnabledAttr().Set(True)
                api.GetRigidBodyEnabledAttr().Set(True)
    stage.GetRootLayer().Save()


def fix_aloha_root_to_world(stage) -> tuple[str, str]:
    """Author a world fixed joint in this replay layer; referenced assets stay untouched."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    root_path = Sdf.Path("/World/StationaryALOHA/Asset/Geometry/tabletop_link")
    joint_path = root_path.AppendChild("ReplayWorldFixedJoint")
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise RuntimeError(f"ALOHA articulation root not found: {root_path}")
    world = UsdGeom.XformCache().GetLocalToWorldTransform(root)
    transform = Gf.Transform(world)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([root_path])
    joint.CreateLocalPos0Attr().Set(transform.GetTranslation())
    q = transform.GetRotation().GetQuat()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(q.GetImaginary())))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    # Match the standard fixed-base articulation layout: the world fixed joint,
    # rather than the formerly free root rigid body, owns ArticulationRootAPI.
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
    physx = PhysxSchema.PhysxArticulationAPI.Apply(joint.GetPrim())
    physx.CreateSolverPositionIterationCountAttr().Set(64)
    physx.CreateSolverVelocityIterationCountAttr().Set(8)
    physx.CreateSleepThresholdAttr().Set(0.0)
    stage.GetRootLayer().Save()
    return str(root_path), str(joint_path)


def correct_phone_accessory_fixed_joint(stage) -> dict:
    """Make both fixed-joint frames coincide at the authored phone rear anchor."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    phone_path = Sdf.Path("/World/MagSafeScene/Phone")
    accessory_path = Sdf.Path("/World/MagSafeScene/Accessory")
    joint_path = Sdf.Path("/World/MagSafeScene/MagneticJoints/AccessoryPhone")
    joint = UsdPhysics.FixedJoint.Get(stage, joint_path)
    if not joint:
        raise RuntimeError(f"Missing PhysicsFixedJoint: {joint_path}")
    if joint.GetBody0Rel().GetTargets() != [phone_path] or joint.GetBody1Rel().GetTargets() != [accessory_path]:
        raise RuntimeError(
            f"Unexpected phone-accessory bodies: "
            f"{joint.GetBody0Rel().GetTargets()}/{joint.GetBody1Rel().GetTargets()}"
        )
    cache = UsdGeom.XformCache()
    phone_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(phone_path))
    accessory_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(accessory_path))
    old_pos0 = Gf.Vec3d(joint.GetLocalPos0Attr().Get())
    old_pos1 = Gf.Vec3d(joint.GetLocalPos1Attr().Get())
    old_anchor0_world = phone_world.Transform(old_pos0)
    old_anchor1_world = accessory_world.Transform(old_pos1)
    old_mismatch = float((old_anchor0_world - old_anchor1_world).GetLength())

    # Preserve the physical phone-rear anchor. Express that exact world frame
    # in each rigid body's local coordinates; do not infer axes from world.
    pos0 = Gf.Vec3f(old_pos0)
    pos1 = Gf.Vec3f(accessory_world.GetInverse().Transform(old_anchor0_world))
    phone_rotation = Gf.Transform(phone_world).GetRotation()
    accessory_rotation = Gf.Transform(accessory_world).GetRotation()
    relative_rotation = phone_rotation * accessory_rotation.GetInverse()
    qr = relative_rotation.GetQuat()
    rot0 = Gf.Quatf(1.0)
    rot1 = Gf.Quatf(float(qr.GetReal()), Gf.Vec3f(qr.GetImaginary()))
    joint.GetLocalPos0Attr().Set(pos0)
    joint.GetLocalRot0Attr().Set(rot0)
    joint.GetLocalPos1Attr().Set(pos1)
    joint.GetLocalRot1Attr().Set(rot1)
    joint.GetJointEnabledAttr().Set(True)
    joint.GetCollisionEnabledAttr().Set(False)
    return {
        "joint_path": str(joint_path),
        "body0": str(phone_path),
        "body1": str(accessory_path),
        "old_anchor_world_mismatch_m": old_mismatch,
        "localPos0": list(pos0),
        "localRot0_wxyz": [rot0.GetReal(), *rot0.GetImaginary()],
        "localPos1": list(pos1),
        "localRot1_wxyz": [rot1.GetReal(), *rot1.GetImaginary()],
        "break_force_n": joint.GetBreakForceAttr().Get(),
        "break_torque_nm": joint.GetBreakTorqueAttr().Get(),
        "collision_enabled": joint.GetCollisionEnabledAttr().Get(),
        "projection_api_supported": False,
    }


def apply_replay_material_profile(stage, profile: str) -> dict:
    """Author realistic contact materials only in a replay sublayer."""
    print(
        f"[MATERIAL_DEBUG] profile={profile} bare_layer_exists={BARE_PHONE_LAYER.exists()}",
        flush=True,
    )
    if profile == "current":
        return {
            "profile": "current",
            "layer": "NONE",
            "phone_surface_split": False,
            "note": "No authored physics material; referenced scene/PhysX defaults remain active.",
        }
    if profile == "bare-phone-split" and BARE_PHONE_LAYER.exists():
        return {
            "profile": profile, "layer": str(BARE_PHONE_LAYER),
            "phone_surface_split": True, "original_collider_enabled": False,
            "dimensions_local_xyz_m": [0.1496, 0.00795, 0.0715],
            "component_thickness_m": [0.00595, 0.001, 0.001],
            "non_overlapping": True, "combine_mode": "average",
            "tape_static_dynamic": [0.75, 0.60],
            "phone_matte_static_dynamic": [0.55, 0.45],
            "phone_glass_static_dynamic": [0.35, 0.25],
            "note": "Reused existing deterministic replay-only split material layer.",
        }
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    if profile == "bare-phone-split":
        material_stage = Usd.Stage.CreateNew(str(BARE_PHONE_LAYER))
        material_stage.OverridePrim("/World")
        material_stage.OverridePrim("/World/ReplayPhysicsMaterials")

        def _split_material(name: str, static: float, dynamic: float):
            material = UsdShade.Material.Define(
                material_stage, f"/World/ReplayPhysicsMaterials/{name}"
            )
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr().Set(static)
            api.CreateDynamicFrictionAttr().Set(dynamic)
            api.CreateRestitutionAttr().Set(0.0)
            physx_api = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            physx_api.CreateFrictionCombineModeAttr().Set("average")
            physx_api.CreateRestitutionCombineModeAttr().Set("average")
            return material

        tape = _split_material("InsulatingTape", 0.75, 0.60)
        matte = _split_material("BarePhoneRearMatte", 0.55, 0.45)
        glass = _split_material("BarePhoneFrontGlass", 0.35, 0.25)
        original = material_stage.OverridePrim("/World/MagSafeScene/Phone/Colliders/Main")
        UsdPhysics.CollisionAPI.Apply(original).CreateCollisionEnabledAttr().Set(False)
        material_stage.OverridePrim("/World/MagSafeScene/Phone/BarePhoneCompound")
        central_thickness = 0.00595
        surface_thickness = 0.001

        def _box(name: str, thickness: float, y: float, material):
            cube = UsdGeom.Cube.Define(
                material_stage, f"/World/MagSafeScene/Phone/BarePhoneCompound/{name}"
            )
            cube.CreateSizeAttr().Set(2.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(0.0, y, 0.0))
            cube.AddScaleOp().Set(Gf.Vec3f(0.1496 / 2.0, thickness / 2.0, 0.0715 / 2.0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateContactOffsetAttr().Set(0.0005)
            collision.CreateRestOffsetAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                material, materialPurpose="physics"
            )
            return str(cube.GetPath())

        central_path = _box("CentralBody", central_thickness, 0.0, matte)
        front_path = _box(
            "FrontGlass", surface_thickness,
            -(central_thickness + surface_thickness) / 2.0, glass,
        )
        rear_path = _box(
            "RearMatte", surface_thickness,
            +(central_thickness + surface_thickness) / 2.0, matte,
        )
        pad_paths = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            low = path.lower()
            if (
                "/world/stationaryaloha/" in low
                and "gripper_" in low
                and ("pad_upper" in low or "pad_lower" in low or "_tip" in low)
                and prim.HasAPI(UsdPhysics.CollisionAPI)
            ):
                over = material_stage.OverridePrim(path)
                UsdShade.MaterialBindingAPI.Apply(over).Bind(tape, materialPurpose="physics")
                pad_paths.append(path)
        material_stage.GetRootLayer().Save()
        stage.GetRootLayer().subLayerPaths.append(str(BARE_PHONE_LAYER))
        stage.GetRootLayer().Save()
        return {
            "profile": profile, "layer": str(BARE_PHONE_LAYER),
            "phone_surface_split": True, "original_collider_enabled": False,
            "central_path": central_path, "front_glass_path": front_path,
            "rear_matte_path": rear_path,
            "dimensions_local_xyz_m": [0.1496, 0.00795, 0.0715],
            "component_thickness_m": [central_thickness, surface_thickness, surface_thickness],
            "non_overlapping": True, "pad_paths": pad_paths, "combine_mode": "average",
            "tape_static_dynamic": [0.75, 0.60],
            "phone_matte_static_dynamic": [0.55, 0.45],
            "phone_glass_static_dynamic": [0.35, 0.25],
            "effective_tape_glass_static_dynamic": [0.55, 0.425],
            "effective_tape_matte_static_dynamic": [0.65, 0.525],
        }

    if MATERIAL_LAYER.exists():
        stage.GetRootLayer().subLayerPaths.append(str(MATERIAL_LAYER))
        stage.GetRootLayer().Save()
        return {
            "profile": profile,
            "layer": str(MATERIAL_LAYER),
            "phone_surface_split": False,
            "tape_static_dynamic": [0.75, 0.60],
            "phone_matte_static_dynamic": [0.55, 0.45],
            "phone_glass_static_dynamic_unbound": [0.35, 0.25],
            "note": "Reused the existing deterministic replay-only material layer.",
        }
    material_stage = Usd.Stage.CreateNew(str(MATERIAL_LAYER))
    material_stage.OverridePrim("/World")
    material_stage.OverridePrim("/World/ReplayPhysicsMaterials")

    def _material(name: str, static: float, dynamic: float, restitution: float = 0.0):
        material = UsdShade.Material.Define(material_stage, f"/World/ReplayPhysicsMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(static)
        api.CreateDynamicFrictionAttr().Set(dynamic)
        api.CreateRestitutionAttr().Set(restitution)
        return material

    tape = _material("InsulatingTape", 0.75, 0.60)
    matte = _material("PhoneMatteRearApproximation", 0.55, 0.45)
    _material("PhoneGlassUnbound", 0.35, 0.25)
    phone_collider = material_stage.OverridePrim("/World/MagSafeScene/Phone/Colliders/Main")
    UsdShade.MaterialBindingAPI.Apply(phone_collider).Bind(
        matte, materialPurpose="physics"
    )
    pad_paths: list[str] = []
    source_stage = stage
    for prim in source_stage.Traverse():
        path = str(prim.GetPath())
        low = path.lower()
        if (
            "/world/stationaryaloha/" in low
            and "gripper_" in low
            and ("pad_upper" in low or "pad_lower" in low or "_tip" in low)
        ):
            over = material_stage.OverridePrim(path)
            UsdShade.MaterialBindingAPI.Apply(over).Bind(tape, materialPurpose="physics")
            pad_paths.append(path)
    material_stage.GetRootLayer().Save()
    stage.GetRootLayer().subLayerPaths.append(str(MATERIAL_LAYER))
    stage.GetRootLayer().Save()
    return {
        "profile": profile,
        "layer": str(MATERIAL_LAYER),
        "phone_surface_split": False,
        "pad_paths": pad_paths,
        "tape_static_dynamic": [0.75, 0.60],
        "phone_matte_static_dynamic": [0.55, 0.45],
        "phone_glass_static_dynamic_unbound": [0.35, 0.25],
        "note": "The phone uses one convex collision mesh; front/rear per-face physics materials are not reliable.",
    }


def apply_tape_sleeve_geometry(stage, config_path: Path, show_visual: bool) -> dict:
    """Create four replay-only single-convex distal tape sleeves."""
    config = json.loads(config_path.expanduser().resolve().read_text())
    length = float(config["distal_length_m"])
    thickness_offset = float(config["effective_thickness_m"])
    chamfer = float(config["distal_chamfer_m"])
    source = config["geometry_source"]
    tip_x = float(source["distal_tip_x_m"])
    width_z = float(source["finger_width_z_m"]) + 2.0 * thickness_offset
    center_z = float(source["finger_center_z_m"])
    thickness_y = float(source["finger_thickness_y_m"]) + 2.0 * thickness_offset
    inner_face_abs_y = float(source["inner_face_abs_y_m"])
    outer_face_abs_y = float(source["outer_face_abs_y_m"])
    if not math.isclose(length, 0.020, abs_tol=1e-12):
        raise ValueError("Tape sleeve distal length must remain exactly 0.020 m")
    if not 0.0 <= thickness_offset <= 0.0003:
        raise ValueError("Tape sleeve effective thickness must be within [0, 0.0003] m")
    if chamfer <= 0.0 or chamfer >= min(length, width_z) / 2.0:
        raise ValueError("Tape sleeve chamfer is outside the physical envelope")

    if TAPE_SLEEVE_LAYER.exists():
        return {
            "layer": str(TAPE_SLEEVE_LAYER),
            "config": str(config_path.expanduser().resolve()),
            "distal_length_m": length,
            "effective_thickness_m": thickness_offset,
            "chamfer_m": chamfer,
            "finger_longitudinal_axis": "+X",
            "closing_axis": "local_Y_toward_gripper_center",
            "finger_width_z_m": width_z,
            "finger_thickness_y_m": thickness_y,
            "reused_existing_layer": True,
        }
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    sleeve_stage = Usd.Stage.CreateNew(str(TAPE_SLEEVE_LAYER))
    sleeve_stage.OverridePrim("/World")
    tape_path = "/World/ReplayPhysicsMaterials/InsulatingTape"
    tape = UsdShade.Material.Define(sleeve_stage, tape_path)
    material_api = UsdPhysics.MaterialAPI.Apply(tape.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(float(config["material"]["static_friction"]))
    material_api.CreateDynamicFrictionAttr().Set(float(config["material"]["dynamic_friction"]))
    material_api.CreateRestitutionAttr().Set(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(tape.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(
        str(config["material"]["friction_combine_mode"])
    )
    visual_tape = UsdShade.Material.Define(
        sleeve_stage, "/World/ReplayVisualMaterials/BlackInsulatingTape"
    )
    visual_shader = UsdShade.Shader.Define(
        sleeve_stage,
        "/World/ReplayVisualMaterials/BlackInsulatingTape/PreviewSurface",
    )
    visual_shader.CreateIdAttr("UsdPreviewSurface")
    visual_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.008, 0.008, 0.008)
    )
    visual_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.58)
    visual_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    visual_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    visual_tape.CreateSurfaceOutput().ConnectToSource(
        visual_shader.ConnectableAPI(), "surface"
    )

    x0, x1 = tip_x - length, tip_x
    z0 = center_z - width_z / 2.0
    z1 = center_z + width_z / 2.0
    disabled = []
    sleeves = []
    bounds = {}
    root_prefix = "/World/StationaryALOHA/Asset/Geometry/tabletop_link"
    for hand in ("left", "right"):
        chain = "/".join(
            [f"follower_{hand}_base_link"]
            + [f"follower_{hand}_link_{i}" for i in range(1, 7)]
        )
        for finger in ("left", "right"):
            carriage = f"{root_prefix}/{chain}/follower_{hand}_carriage_{finger}"
            inner_sign = -1.0 if finger == "left" else 1.0
            inner_y = inner_sign * (inner_face_abs_y + thickness_offset)
            outer_y = inner_sign * (outer_face_abs_y - thickness_offset)
            y0, y1 = sorted((inner_y, outer_y))
            profile = [
                (x0, z0), (x1 - chamfer, z0), (x1, z0 + chamfer),
                (x1, z1 - chamfer), (x1 - chamfer, z1), (x0, z1),
            ]
            points = [
                Gf.Vec3f(float(x), float(y), float(z))
                for y in (y0, y1) for x, z in profile
            ]
            counts = [6, 6] + [4] * 6
            indices = [0, 1, 2, 3, 4, 5, 11, 10, 9, 8, 7, 6]
            for i in range(6):
                j = (i + 1) % 6
                indices.extend([i, j, 6 + j, 6 + i])
            sleeve_path = (
                f"{carriage}/follower_{hand}_gripper_{finger}_tape_sleeve_20mm"
            )
            mesh = UsdGeom.Mesh.Define(sleeve_stage, sleeve_path)
            mesh.CreatePointsAttr().Set(points)
            mesh.CreateFaceVertexCountsAttr().Set(counts)
            mesh.CreateFaceVertexIndicesAttr().Set(indices)
            mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr().Set(False)
            mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.01, 0.01, 0.01)])
            mesh.CreateDisplayOpacityAttr().Set([1.0])
            mesh.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr().Set(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
                "convexHull"
            )
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            physx_collision.CreateContactOffsetAttr().Set(0.0005)
            physx_collision.CreateRestOffsetAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                tape, materialPurpose="physics"
            )
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(visual_tape)
            sleeves.append(sleeve_path)
            bounds[sleeve_path] = {
                "min_local_m": [x0, y0, z0],
                "max_local_m": [x1, y1, z1],
            }
            for suffix in ("pad_upper", "pad_lower", "tip"):
                old_path = f"{carriage}/follower_{hand}_gripper_{finger}_{suffix}"
                old = sleeve_stage.OverridePrim(old_path)
                UsdPhysics.CollisionAPI.Apply(old).CreateCollisionEnabledAttr().Set(False)
                disabled.append(old_path)

    sleeve_stage.GetRootLayer().Save()
    stage.GetRootLayer().subLayerPaths.append(str(TAPE_SLEEVE_LAYER))
    stage.GetRootLayer().Save()
    return {
        "layer": str(TAPE_SLEEVE_LAYER),
        "config": str(config_path.expanduser().resolve()),
        "sleeve_paths": sleeves,
        "disabled_distal_colliders": disabled,
        "bounds": bounds,
        "distal_length_m": length,
        "effective_thickness_m": thickness_offset,
        "chamfer_m": chamfer,
        "finger_longitudinal_axis": "+X",
        "closing_axis": "local_Y_toward_gripper_center",
        "finger_width_z_m": width_z,
        "finger_thickness_y_m": thickness_y,
        "material_path": tape_path,
        "static_friction": float(config["material"]["static_friction"]),
        "dynamic_friction": float(config["material"]["dynamic_friction"]),
        "combine_mode": str(config["material"]["friction_combine_mode"]),
        "show_visual": show_visual,
        "convex_approximation": "convexHull",
        "phone_specific_groove": False,
    }


def apply_uniform_phone_mass_properties(stage) -> dict:
    """Use measured mass and ideal-box inertia on the replay composition only."""
    from pxr import Gf, UsdPhysics

    mass = 0.190
    accessory_mass = 0.027
    length, thickness, width = 0.1496, 0.00795, 0.0715
    diagonal = Gf.Vec3f(
        mass * (thickness**2 + width**2) / 12.0,
        mass * (length**2 + width**2) / 12.0,
        mass * (length**2 + thickness**2) / 12.0,
    )
    prim = stage.GetPrimAtPath("/World/MagSafeScene/Phone")
    if not prim.IsValid():
        raise RuntimeError("Phone rigid body prim is missing")
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr().Set(mass)
    api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0))
    api.CreateDiagonalInertiaAttr().Set(diagonal)
    api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0))
    accessory_prim = stage.GetPrimAtPath("/World/MagSafeScene/Accessory")
    if not accessory_prim.IsValid():
        raise RuntimeError("Accessory rigid body prim is missing")
    accessory_api = UsdPhysics.MassAPI.Apply(accessory_prim)
    accessory_api.CreateMassAttr().Set(accessory_mass)
    accessory_inertia_attr = accessory_api.GetDiagonalInertiaAttr()
    accessory_inertia = (
        accessory_inertia_attr.Get()
        if accessory_inertia_attr.HasAuthoredValueOpinion()
        else None
    )
    stage.GetRootLayer().Save()
    return {
        "mass_kg": mass,
        "accessory_mass_kg": accessory_mass,
        "initial_assembly_mass_kg": mass + accessory_mass,
        "com_local_m": [0.0, 0.0, 0.0],
        "dimensions_local_xyz_m": [length, thickness, width],
        "diagonal_inertia_kg_m2": list(diagonal),
        "accessory_diagonal_inertia_kg_m2": (
            list(accessory_inertia) if accessory_inertia is not None else None
        ),
        "accessory_inertia_source": (
            "authored" if accessory_inertia is not None else "PhysX collider-derived"
        ),
        "visual_topology_affects_mass_properties": False,
        "authored_on": "/World/MagSafeScene/Phone",
        "scope": "replay_composition_only",
    }


def main() -> None:
    print("[ALOHA] main entered.", flush=True)
    import torch
    print("[ALOHA] torch imported.", flush=True)
    import omni.usd
    print("[ALOHA] omni.usd imported.", flush=True)
    from pxr import Gf, Usd, UsdGeom
    print("[ALOHA] pxr imported.", flush=True)
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    print("[ALOHA] assets imported.", flush=True)
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    print("[ALOHA] Isaac Lab simulation modules imported.", flush=True)

    dataset = _dataset_preload
    info = _info_preload
    if info["features"]["observation.state"]["shape"] != [14]:
        raise RuntimeError("ALOHA replay requires a 14-value state")
    state_trajectory = _state_preload
    action_trajectory = _action_preload
    timestamps = _timestamps_preload
    if state_trajectory.shape[1:] != (14,) or action_trajectory.shape != state_trajectory.shape:
        raise RuntimeError(f"Invalid state/action shapes {state_trajectory.shape}/{action_trajectory.shape}")
    if not np.isfinite(state_trajectory).all() or not np.isfinite(action_trajectory).all():
        raise RuntimeError("State/action contains NaN or Inf")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise RuntimeError("Timestamps are not strictly increasing")

    print("[ALOHA] dataset validation complete; composing replay stage.", flush=True)
    replay_sublayers = tuple(
        path for path in (
            BARE_PHONE_LAYER if args.friction_profile == "bare-phone-split" else None,
            ACCESSORY_HOLLOW_RING_LAYER,
            TAPE_SLEEVE_LAYER if args.tape_sleeve_config is not None else None,
        )
        if path is not None and path.exists()
    )
    stage = compose_stage(
        OUTPUT, "StationaryALOHA", ALOHA_USD, "stationary_aloha",
        sublayers=replay_sublayers,
    )
    print("[ALOHA] stage composed.", flush=True)
    support_foot_proxy_path = (
        "/World/MagSafeScene/Accessory/Colliders/SupportFootProxy"
    )
    support_foot_proxy = stage.GetPrimAtPath(support_foot_proxy_path)
    if not support_foot_proxy.IsValid() or support_foot_proxy.IsActive():
        raise RuntimeError(
            "SupportFootProxy must exist as an inactive composed prim: "
            f"path={support_foot_proxy_path} valid={support_foot_proxy.IsValid()} "
            f"active={support_foot_proxy.IsActive() if support_foot_proxy.IsValid() else None}"
        )
    print(
        "[SUPPORT_FOOT_PROXY]\n"
        f"path={support_foot_proxy_path}\n"
        "active=false\n"
        "collision_effectively_disabled=true\n"
        f"source_layer={ACCESSORY_HOLLOW_RING_LAYER}",
        flush=True,
    )
    suppress_stationary_aloha_fixture(stage)
    root_path, fixed_joint_path = fix_aloha_root_to_world(stage)
    print("[ALOHA] fixed-base layer authored.", flush=True)
    mode = {"inspection": "kinematic", "magnetic": "physical-task"}.get(args.scene_mode, args.scene_mode)
    gripper_source = args.gripper_source or (
        "observation.state" if mode == "kinematic" else "action"
    )
    if args.source is not None:
        # Backward-compatible legacy behavior. Explicit new options take priority.
        if args.gripper_source is None:
            gripper_source = args.source
    material_report = apply_replay_material_profile(stage, args.friction_profile)
    print("[REPLAY_LAYER_DEBUG] material profile ready", flush=True)
    tape_sleeve_report = None
    if args.tape_sleeve_config is not None:
        tape_sleeve_report = apply_tape_sleeve_geometry(
            stage, args.tape_sleeve_config, args.show_tape_sleeve
        )
    print("[REPLAY_LAYER_DEBUG] tape sleeve ready", flush=True)
    mass_property_report = apply_uniform_phone_mass_properties(stage)
    print("[REPLAY_LAYER_DEBUG] mass properties ready", flush=True)
    if args.show_accessory_ring_collider:
        from pxr import UsdGeom, UsdPhysics
        collider_root = "/World/MagSafeScene/Accessory/Colliders/MainRing"
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(collider_root) and prim.HasAPI(
                UsdPhysics.CollisionAPI
            ):
                UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited
                )
    phone_accessory_joint_report = None
    if args.phone_accessory_joint_mode == "corrected-fixed":
        phone_accessory_joint_report = correct_phone_accessory_fixed_joint(stage)
        from pxr import UsdPhysics
        corrected_accessory_joint = UsdPhysics.FixedJoint.Get(
            stage, "/World/MagSafeScene/MagneticJoints/AccessoryPhone"
        )
        # PhysX break thresholds are not body-selective. Keep the configured
        # thresholds below as the right-contact wrench criterion, while this
        # native guard prevents left-hand assembly loads from breaking the
        # constraint before the right hand touches the accessory.
        corrected_accessory_joint.GetBreakForceAttr().Set(1.0e6)
        corrected_accessory_joint.GetBreakTorqueAttr().Set(1.0e6)
        print(f"[PHONE_ACCESSORY_JOINT] {phone_accessory_joint_report}", flush=True)
    else:
        corrected_accessory_joint = None
    hardware_motor_torque_upper_Nm = 1.439
    hardware_position_per_motor_radian = (0.05800 - 0.01844) / (1.4910 - (-0.6213))
    hardware_effort_limit_per_carriage = (
        hardware_motor_torque_upper_Nm / (2.0 * hardware_position_per_motor_radian)
    )
    hardware_persistent_error_reference_m = 0.0039403
    hardware_stiffness = (
        hardware_effort_limit_per_carriage / hardware_persistent_error_reference_m
    )
    if args.gripper_drive_mode == "hardware-max":
        if args.gripper_stiffness is not None:
            raise ValueError(
                "--gripper-stiffness cannot be combined with --gripper-drive-mode hardware-max"
            )
        gripper_stiffness = hardware_stiffness
        gripper_effort_limit = hardware_effort_limit_per_carriage
        drive_tag = "hardware_max"
    elif args.gripper_drive_mode == "effort-cap":
        if args.gripper_stiffness is not None:
            raise ValueError(
                "--gripper-stiffness cannot be combined with --gripper-drive-mode effort-cap"
            )
        if args.gripper_effort_cap is None:
            raise ValueError("--gripper-drive-mode effort-cap requires --gripper-effort-cap")
        if not (0.0 < args.gripper_effort_cap <= hardware_effort_limit_per_carriage):
            raise ValueError(
                "--gripper-effort-cap must be positive and no greater than the "
                f"hardware upper bound {hardware_effort_limit_per_carriage:.9g} N/carriage"
            )
        gripper_effort_limit = float(args.gripper_effort_cap)
        # A 5% controller margin makes the cap observable despite the measured
        # contact-error variation; the force limit remains the physical authority.
        gripper_stiffness = (
            1.05 * gripper_effort_limit / hardware_persistent_error_reference_m
        )
        drive_tag = f"effort_cap_{gripper_effort_limit:.4f}".replace(".", "_")
    elif args.gripper_stiffness is not None:
        if args.gripper_stiffness <= 0.0:
            raise ValueError("--gripper-stiffness must be positive")
        gripper_stiffness = float(args.gripper_stiffness)
        gripper_effort_limit = 60.0
        drive_tag = f"stiffness_{gripper_stiffness:.2f}".replace(".", "_")
    else:
        gripper_stiffness = 100.0 * args.gripper_drive_scale
        gripper_effort_limit = 40.0 * args.gripper_drive_scale
        drive_tag = f"drive_{args.gripper_drive_scale:g}"
    friction_tag = (
        f"{args.friction_profile.replace('-', '_')}_{drive_tag}_uniform_box_shape_v1"
    )
    if args.contact_diagnostic_tag:
        safe_tag = "".join(c if c.isalnum() or c == "_" else "_" for c in args.contact_diagnostic_tag)
        friction_tag = f"{friction_tag}_{safe_tag}"
    print(f"[MATERIAL] {material_report}", flush=True)
    if tape_sleeve_report is not None:
        print(f"[TAPE_SLEEVE] {tape_sleeve_report}", flush=True)
    print(f"[PHONE_MASS] {mass_property_report}", flush=True)
    if mode == "kinematic":
        freeze_objects(stage)
    monitored_body_paths: list[str] = []
    if mode != "kinematic":
        from physical_contact_monitor import enable_contact_reporting, select_rigid_bodies

        print("[CONTACT] enabling replay-layer contact reports.", flush=True)
        enabled = enable_contact_reporting(
            stage,
            [
                "/World/StationaryALOHA/Asset",
                "/World/MagSafeScene/Phone",
                "/World/MagSafeScene/Accessory",
                "/World/MagSafeScene/Table",
            ],
        )
        monitored_body_paths = select_rigid_bodies(
            stage,
            ["/World/StationaryALOHA/Asset"],
            lambda path: (
                "carriage_" in path
                or any(f"link_{index}" in path for index in (4, 5, 6))
            ),
        )
        print(
            f"[CONTACT] report_bodies={len(enabled)} monitored_robot_bodies={len(monitored_body_paths)}",
            flush=True,
        )
    if not omni.usd.get_context().open_stage(str(OUTPUT)):
        raise RuntimeError(f"Cannot open {OUTPUT}")
    runtime_stage = omni.usd.get_context().get_stage()
    audit_overlay = {}
    if args.phone_pose_audit:
        debug_root = "/World/PhonePoseAuditOverlay"
        UsdGeom.Xform.Define(runtime_stage, debug_root)

        def _debug_sphere(name: str, color: tuple[float, float, float], radius: float):
            sphere = UsdGeom.Sphere.Define(runtime_stage, f"{debug_root}/{name}")
            sphere.CreateRadiusAttr(radius)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            sphere.CreatePurposeAttr().Set("guide")
            return sphere

        audit_overlay["phone_com"] = _debug_sphere("PhoneRigidBodyCOM", (1.0, 0.1, 0.1), 0.006)
        audit_overlay["visual_origin"] = _debug_sphere("PhoneVisualOrigin", (0.1, 1.0, 0.1), 0.004)
        audit_overlay["wrist"] = _debug_sphere("LeftWristLink6Origin", (0.1, 0.3, 1.0), 0.005)
        z_axis = UsdGeom.BasisCurves.Define(runtime_stage, f"{debug_root}/WorldZAxis")
        z_axis.CreateTypeAttr().Set("linear")
        z_axis.CreateCurveVertexCountsAttr().Set([2])
        z_axis.CreatePointsAttr().Set([Gf.Vec3f(0.0, 0.0, 0.79), Gf.Vec3f(0.0, 0.0, 1.04)])
        z_axis.CreateWidthsAttr().Set([0.003, 0.003])
        z_axis.CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 0.0)])
        z_axis.CreatePurposeAttr().Set("guide")
        print(f"[PHONE_POSE_AUDIT] overlay_root={debug_root}", flush=True)
    sim = SimulationContext(SimulationCfg(device="cpu"))
    robot = Articulation(ArticulationCfg(
        prim_path=fixed_joint_path,
        spawn=None,
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[r"follower_(left|right)_joint_[0-5]"],
                effort_limit_sim=27.0,
                velocity_limit_sim=8.0,
                stiffness=250.0,
                damping=20.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[r"follower_(left|right)_(left|right)_carriage_joint"],
                effort_limit_sim=gripper_effort_limit,
                velocity_limit_sim=0.5,
                stiffness=gripper_stiffness,
                damping=10.0,
            ),
        },
    ))
    phone = accessory = None
    if mode != "kinematic":
        phone = RigidObject(RigidObjectCfg(prim_path="/World/MagSafeScene/Phone", spawn=None))
        accessory = RigidObject(RigidObjectCfg(prim_path="/World/MagSafeScene/Accessory", spawn=None))
    contact_monitor = None
    phone_table_contact_monitor = None
    if mode != "kinematic":
        from physical_contact_monitor import PhysicalContactMonitor
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        def _relevant_pair(owner: str, other: str) -> bool:
            combined = (owner + " " + other).lower()
            return (
                ("stationaryaloha" in combined)
                and any(token in combined for token in ("phone", "accessory", "table"))
            )

        def _relative_contact_velocity(owner: str, other: str) -> float:
            body_name = owner.rsplit("/", 1)[-1]
            try:
                body_index = list(robot.data.body_names).index(body_name)
                robot_velocity = robot.data.body_lin_vel_w.torch[0, body_index]
            except (ValueError, AttributeError):
                return float("nan")
            if "Accessory" in other:
                other_velocity = accessory.data.root_lin_vel_w.torch[0]
            elif "Phone" in other:
                other_velocity = phone.data.root_lin_vel_w.torch[0]
            else:
                other_velocity = torch.zeros_like(robot_velocity)
            return float(torch.linalg.vector_norm(robot_velocity - other_velocity).item())

        contact_filters = [
            "/World/MagSafeScene/Accessory",
            "/World/MagSafeScene/Phone",
            "/World/MagSafeScene/Table/Colliders/Top",
        ]
        contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/StationaryALOHA/Asset/.*(link_[456]|carriage_(left|right))",
                update_period=0.0,
                filter_prim_paths_expr=contact_filters,
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
        contact_monitor = PhysicalContactMonitor(
            contact_sensor,
            contact_filters,
            REPORT_ROOT / (
                f"contact_event_log_{'action' if gripper_source == 'action' else 'observation'}"
                f"_gripper_{friction_tag}.csv"
            ),
            robot="ALOHA",
            physics_dt=sim.get_physics_dt(),
            relevant_pair=_relevant_pair,
            relative_velocity=_relative_contact_velocity,
        )
        phone_table_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/MagSafeScene/Phone",
                update_period=0.0,
                filter_prim_paths_expr=["/World/MagSafeScene/Table/Colliders/Top"],
                track_contact_points=True,
                track_friction_forces=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
        phone_table_contact_monitor = PhysicalContactMonitor(
            phone_table_sensor,
            ["/World/MagSafeScene/Table/Colliders/Top"],
            REPORT_ROOT / f"phone_table_contact_force_{friction_tag}.csv",
            robot="PHONE",
            physics_dt=sim.get_physics_dt(),
            relevant_pair=lambda owner, other: (
                "/Phone" in owner and "/Table/Colliders/Top" in other
            ),
            relative_velocity=None,
        )
    sim.reset()
    names = list(robot.data.joint_names)
    missing = [name for name in JOINT_NAMES if name not in names]
    if missing:
        raise RuntimeError(f"Missing imported ALOHA joints: {missing}; actual={names}")
    print(f"[ALOHA] write_joint_state_to_sim{inspect.signature(robot.write_joint_state_to_sim)}")
    print(
        f"[ALOHA] arm_source={args.arm_source} gripper_source={gripper_source} "
        f"scene_mode={mode} frames={len(state_trajectory)} fps={info['fps']}"
    )
    print(
        f"[ALOHA] gripper_stiffness={gripper_stiffness} damping=10.0 "
        f"effort_limit_per_carriage={gripper_effort_limit} velocity_limit=0.5"
    )
    print(
        f"[ALOHA] gripper_command_mode={args.gripper_command_mode} "
        f"gripper_drive_mode={args.gripper_drive_mode}"
    )
    print(
        f"[GRIPPER_CONFIG] command_mode={args.gripper_command_mode} "
        f"drive_mode={args.gripper_drive_mode} "
        f"effort_cap_N_per_carriage={gripper_effort_limit:.9g} "
        f"stiffness_N_per_m={gripper_stiffness:.12g} damping_Ns_per_m=10 "
        f"accessory_mass_kg={mass_property_report['accessory_mass_kg']:.9g}",
        flush=True,
    )
    print(f"[ALOHA] runtime_joint_order={names}")
    print(f"[ALOHA] fixed_root_body={root_path} fixed_joint={fixed_joint_path}")
    eye, target = CAMERAS[args.camera]
    sim.set_camera_view(eye, target)
    pos = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    vel = torch.zeros_like(pos)
    indices = {name: names.index(name) for name in JOINT_NAMES}
    fps = float(info["fps"])
    dt = sim.get_physics_dt()
    ring_controller = None
    ring_contact_state = None
    right_arm_joint_ids = [
        indices[f"follower_right_joint_{j}"] for j in range(6)
    ]
    if args.ring_affordance_controller:
        if mode != "physical-task":
            raise ValueError("--ring-affordance-controller requires --scene-mode physical-task")
        if args.phone_accessory_joint_mode != "corrected-fixed":
            raise ValueError(
                "--ring-affordance-controller requires --phone-accessory-joint-mode corrected-fixed"
            )
        from ring_affordance_controller import (
            ContactState as RingContactState,
            FingerState as RingFingerState,
            RingAffordanceController,
            quat_to_matrix as ring_quat_to_matrix,
        )
        ring_controller = RingAffordanceController(
            REPORT_ROOT, dt, args.ring_pose_provider
        )
        ring_contact_state = RingContactState()
        right_open_target = float(max(
            np.max(state_trajectory[:, 13]), np.max(action_trajectory[:, 13])
        ))
        print(
            f"[RING_AFFORDANCE] provider={args.ring_pose_provider} "
            f"right_open_target_m={right_open_target:.9g}",
            flush=True,
        )
    end_frame = (
        len(state_trajectory) - 1
        if args.end_frame is None
        else min(args.end_frame, len(state_trajectory) - 1)
    )
    shape_contact_logger = None
    if mode != "kinematic":
        from physx_shape_contact_logger import PhysxShapeContactLogger

        shape_contact_logger = PhysxShapeContactLogger(
            REPORT_ROOT / f"left_phone_exact_shape_contacts_{friction_tag}.csv",
            physics_dt=dt,
            start_frame=args.start_frame,
            end_frame=end_frame,
        )
    if args.stability_test_seconds > 0:
        # Establish the authored joint constraint and frame-0 joint state before
        # starting the measured three-second window.
        x = state_trajectory[0]
        for side, offset in (("left", 0), ("right", 7)):
            for j in range(6):
                pos[0, indices[f"follower_{side}_joint_{j}"]] = float(x[offset + j])
            g = max(0.0, float(x[offset + 6]))
            pos[0, indices[f"follower_{side}_left_carriage_joint"]] = g
            pos[0, indices[f"follower_{side}_right_carriage_joint"]] = g
        ring_step = None
        ring_geo = None
        ring_frame_current = None
        ring_finger_state = None
        if ring_controller is not None:
            body_names = list(robot.data.body_names)
            insertion_body_index = body_names.index("follower_right_carriage_left")
            opposing_body_index = body_names.index("follower_right_carriage_right")
            insertion_pose = robot.data.body_pose_w.torch[
                0, insertion_body_index
            ].detach().cpu().numpy()
            opposing_pose = robot.data.body_pose_w.torch[
                0, opposing_body_index
            ].detach().cpu().numpy()
            insertion_q_xyzw = insertion_pose[[4, 5, 6, 3]]
            opposing_q_xyzw = opposing_pose[[4, 5, 6, 3]]
            insertion_r = ring_quat_to_matrix(insertion_q_xyzw)
            opposing_r = ring_quat_to_matrix(opposing_q_xyzw)
            insertion_tip_offset = np.array([0.069561665, -0.020484785, -0.001431745])
            opposing_tip_offset = np.array([0.069561665, 0.020484785, -0.001431745])
            insertion_tip = insertion_pose[:3] + insertion_r @ insertion_tip_offset
            opposing_tip = opposing_pose[:3] + opposing_r @ opposing_tip_offset
            ring_finger_state = RingFingerState(
                center=insertion_tip,
                quaternion_xyzw=insertion_q_xyzw,
                longitudinal=insertion_r[:, 0],
                wide=insertion_r[:, 2],
                thin=-insertion_r[:, 1],
                opposing_center=opposing_tip,
            )
            accessory_pose_control = accessory.data.root_pose_w.torch[
                0
            ].detach().cpu().numpy()
            accessory_pose_control = accessory_pose_control.copy()
            accessory_pose_control[3:7] = accessory_pose_control[[4, 5, 6, 3]]
            recorded_close = float(gripper_x[13]) <= 0.001
            ring_step, ring_frame_current, ring_geo = ring_controller.compute_target(
                sim_time, frame, accessory_pose_control, ring_finger_state,
                ring_contact_state, recorded_close,
            )
            # Geometry-state gating: retain an open aperture until insertion is
            # complete even if the demonstration has already commanded close.
            if not ring_controller.close_enabled:
                pos[0, indices["follower_right_left_carriage_joint"]] = right_open_target
                pos[0, indices["follower_right_right_carriage_joint"]] = right_open_target
            if ring_step is not None:
                target_p, target_r, _ = ring_step
                jacobians = robot.root_physx_view.get_jacobians()
                if ring_controller.target_count == 0:
                    print(
                        f"[RING_IK_API] jacobians={type(jacobians)} "
                        f"torch_shape={tuple(jacobians.torch.shape)} "
                        f"limits={type(robot.data.soft_joint_pos_limits)}",
                        flush=True,
                    )
                jacobian_body_index = insertion_body_index - 1
                jac = jacobians.torch[
                    0, jacobian_body_index, :, right_arm_joint_ids
                ].detach().cpu().numpy()
                # Convert the carriage-origin spatial Jacobian to the distal
                # collision-center point Jacobian.
                point_offset = insertion_tip - insertion_pose[:3]
                skew = np.array([
                    [0.0, -point_offset[2], point_offset[1]],
                    [point_offset[2], 0.0, -point_offset[0]],
                    [-point_offset[1], point_offset[0], 0.0],
                ])
                jac[:3, :] = jac[:3, :] - skew @ jac[3:, :]
                q_now = robot.data.joint_pos.torch[
                    0, right_arm_joint_ids
                ].detach().cpu().numpy()
                limits = robot.data.soft_joint_pos_limits.torch[
                    0, right_arm_joint_ids
                ].detach().cpu().numpy()
                q_target, pe, re = ring_controller.solve_dls(
                    insertion_tip, insertion_r, target_p, target_r, jac,
                    q_now, limits[:, 0], limits[:, 1],
                    np.full(6, 8.0), np.full(6, 20.0),
                )
                ik_ok = q_target is not None
                if ik_ok:
                    for local_index, joint_id in enumerate(right_arm_joint_ids):
                        pos[0, joint_id] = float(q_target[local_index])
                ring_controller.log_target(
                    sim_time, frame, ring_frame_current, ring_geo,
                    ring_contact_state, (target_p, target_r),
                    ik_ok, pe, re, q_target if ik_ok else q_now,
                )
        if mode == "kinematic":
            robot.write_joint_state_to_sim(position=pos, velocity=vel)
        else:
            robot.set_joint_position_target(pos)
            robot.write_data_to_sim()
        for _ in range(2):
            sim.step(render=True)
            robot.update(dt)
        initial_pos = robot.data.root_pos_w.torch.clone()
        initial_quat = robot.data.root_quat_w.torch.clone()
        max_translation = max_rotation = max_linear = max_angular = 0.0
        steps = int(round(args.stability_test_seconds / dt))
        for _ in range(steps):
            sim.step(render=True)
            robot.update(dt)
            dp = torch.linalg.vector_norm(robot.data.root_pos_w.torch - initial_pos).item()
            dot = torch.abs(torch.sum(robot.data.root_quat_w.torch * initial_quat, dim=1)).clamp(0, 1)
            dr = (2 * torch.acos(dot)).max().item()
            max_translation = max(max_translation, dp)
            max_rotation = max(max_rotation, dr)
            max_linear = max(max_linear, torch.linalg.vector_norm(robot.data.root_lin_vel_w.torch, dim=1).max().item())
            max_angular = max(max_angular, torch.linalg.vector_norm(robot.data.root_ang_vel_w.torch, dim=1).max().item())
        # Demonstrate that internal arm/gripper joints remain movable after the fixed-base test.
        before = robot.data.joint_pos.torch.clone()
        pos[0, indices["follower_left_joint_5"]] += 0.05
        pos[0, indices["follower_left_left_carriage_joint"]] = min(
            float(pos[0, indices["follower_left_left_carriage_joint"]]) + .005, .044
        )
        pos[0, indices["follower_left_right_carriage_joint"]] = pos[
            0, indices["follower_left_left_carriage_joint"]
        ]
        if mode == "kinematic":
            robot.write_joint_state_to_sim(position=pos, velocity=vel)
        else:
            robot.set_joint_position_target(pos)
            robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(dt)
        moved = torch.abs(robot.data.joint_pos.torch - before)
        print(
            f"[STABILITY] duration_s={args.stability_test_seconds:g} "
            f"root_translation_max_m={max_translation:.12g} "
            f"root_rotation_max_deg={max_rotation * 180.0 / 3.141592653589793:.12g} "
            f"root_linear_velocity_max_m_s={max_linear:.12g} "
            f"root_angular_velocity_max_rad_s={max_angular:.12g}"
        )
        print(
            f"[STABILITY] arm_joint_test_delta_rad="
            f"{float(moved[0, indices['follower_left_joint_5']]):.12g} "
            f"gripper_test_delta_m="
            f"{float(moved[0, indices['follower_left_left_carriage_joint']]):.12g}"
        )
        print("[STABILITY] imported_usd_modified=False stationary_ai_xml_modified=False")
        return
    from physical_grasp_detector import GraspDetector, GraspState, TaskPhase, TaskPhaseDetector

    grasp_detector = (
        GraspDetector(
            REPORT_ROOT / (
                f"aloha_grasp_event_log_{'action' if gripper_source == 'action' else 'observation'}"
                f"_gripper_{friction_tag}.csv"
            ),
            robot="ALOHA",
            physics_dt=dt,
        )
        if mode != "kinematic"
        else None
    )
    phase_detector = TaskPhaseDetector()
    magnet_logger = None
    acc_pair = charger_pair = charger = None
    if mode == "physical-task":
        from magsafe_magnet_controller import BodyState, MagnetCsvLogger, MagnetState, MagneticPair, load_config

        cfg = load_config(args.magnet_config.resolve())

        def _body(obj: RigidObject) -> BodyState:
            pose = obj.data.root_pose_w.torch[0].detach().cpu().numpy()
            velocity = obj.data.root_vel_w.torch[0].detach().cpu().numpy()
            q_xyzw = pose[3:7]
            return BodyState(
                pose[:3].copy(),
                np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]),
                velocity[:3].copy(),
                velocity[3:].copy(),
            )

        def _static(position) -> BodyState:
            return BodyState(np.asarray(position, float), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))

        acc_pair = MagneticPair(
            "accessory_phone",
            cfg["accessory_phone"],
            np.array([0, -0.00175, 0.0]),
            np.array([0, -1, 0.0]),
            np.array([0, 0.004175, 0.0]),
            np.array([0, 1, 0.0]),
            0.027,
            cfg["safety"]["max_accessory_acceleration_mps2"],
            initial_state=MagnetState.ATTACHED,
        )
        target_clearance = float(cfg["charger_target"].get("surface_clearance_m", 0.0))
        charger_pair = MagneticPair(
            "phone_charger",
            cfg["phone_charger"],
            np.array([0, 0.004175, 0.0]),
            np.array([0, 1, 0.0]),
            np.array([0, 0.005846519, 0.13261811])
            + np.array([0, -math.cos(math.radians(15)), math.sin(math.radians(15))]) * target_clearance,
            np.array([0, -math.cos(math.radians(15)), math.sin(math.radians(15))]),
            0.190,
            cfg["safety"]["max_phone_acceleration_mps2"],
            initial_state=MagnetState.DETACHED,
            target_orientation_world=np.asarray(cfg["charger_target"]["target_rotation_wxyz"], float),
        )
        charger = _static((0.42, 0.52, 0.807))
        magnet_logger = MagnetCsvLogger(
            REPORT_ROOT
            / (
                f"magnetic_event_log_{'action' if gripper_source == 'action' else 'observation'}"
                f"_gripper_{friction_tag}.csv"
            )
        )

        def _wrench(obj: RigidObject, force: np.ndarray, torque: np.ndarray) -> None:
            force_tensor = torch.tensor(force.reshape(1, 1, 3), device=obj.device, dtype=torch.float32)
            torque_tensor = torch.tensor(torque.reshape(1, 1, 3), device=obj.device, dtype=torch.float32)
            obj.instantaneous_wrench_composer.set_forces_and_torques_index(
                force_tensor, torque_tensor, is_global=True
            )

    sim_time = 0.0
    initial_phone = phone.data.root_pos_w.torch.clone() if phone is not None else None
    initial_accessory = accessory.data.root_pos_w.torch.clone() if accessory is not None else None
    initial_phone_np = initial_phone[0].detach().cpu().numpy() if initial_phone is not None else None
    initial_accessory_np = initial_accessory[0].detach().cpu().numpy() if initial_accessory is not None else None
    initial_phone_pose = (
        phone.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
        if phone is not None
        else None
    )
    initial_accessory_pose = (
        accessory.data.root_pose_w.torch[0].detach().cpu().numpy().copy()
        if accessory is not None
        else None
    )

    def _quat_mul_xyzw_global(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz,
        ])

    def _quat_rotate_inverse_xyzw(q, vector):
        conjugate = np.array([-q[0], -q[1], -q[2], q[3]])
        pure = np.array([vector[0], vector[1], vector[2], 0.0])
        return _quat_mul_xyzw_global(
            _quat_mul_xyzw_global(conjugate, pure), q
        )[:3]

    def _relative_pose(phone_pose_value, accessory_pose_value):
        q_phone = phone_pose_value[3:7]
        q_accessory = accessory_pose_value[3:7]
        q_phone_conj = np.array([-q_phone[0], -q_phone[1], -q_phone[2], q_phone[3]])
        relative_q = _quat_mul_xyzw_global(q_phone_conj, q_accessory)
        if relative_q[3] < 0.0:
            relative_q = -relative_q
        relative_p = _quat_rotate_inverse_xyzw(
            q_phone, accessory_pose_value[:3] - phone_pose_value[:3]
        )
        return relative_p, relative_q

    initial_relative_position, initial_relative_quaternion = _relative_pose(
        initial_phone_pose, initial_accessory_pose
    )
    accessory_attached = mode == "physical-task"
    accessory_detach_frame = None
    magnetic_attach_frame = None
    first_contact: dict[str, int] = {}
    max_slip = 0.0
    last_object_positions: dict[str, np.ndarray] = {}
    gripper_log_file = None
    gripper_log_writer = None
    if mode != "kinematic":
        runtime_mass = phone.data.body_mass.torch[0, 0].item()
        runtime_inertia = phone.data.body_inertia.torch[0, 0].detach().cpu().numpy().reshape(3, 3)
        runtime_com_b = phone.data.body_com_pos_b.torch[0, 0].detach().cpu().numpy()
        theoretical_inertia = np.diag(
            [
                0.190 / 12.0 * (0.00795**2 + 0.0715**2),
                0.190 / 12.0 * (0.1496**2 + 0.0715**2),
                0.190 / 12.0 * (0.1496**2 + 0.00795**2),
            ]
        )
        mass_report_path = REPORT_ROOT / f"phone_mass_properties_{friction_tag}.txt"
        mass_report_path.write_text(
            "\n".join(
                [
                    f"friction_profile={args.friction_profile}",
                    "geometry=uniform_rectangular_box",
                    "dimensions_local_xyz_m=[0.1496,0.00795,0.0715]",
                    f"runtime_mass_kg={runtime_mass:.12g}",
                    f"runtime_com_local_m={runtime_com_b.tolist()}",
                    f"runtime_inertia_kg_m2={runtime_inertia.tolist()}",
                    f"theoretical_uniform_box_inertia_kg_m2={theoretical_inertia.tolist()}",
                    f"inertia_max_abs_error={float(np.max(np.abs(runtime_inertia-theoretical_inertia))):.12g}",
                    f"material_report={material_report}",
                    "phone_orientation_forced=False",
                    "phone_pose_snap_used=False",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        gripper_log_path = REPORT_ROOT / (
            f"gripper_action_physics_log_{friction_tag}.csv"
            if gripper_source == "action"
            else f"gripper_observation_physics_log_{friction_tag}.csv"
        )
        gripper_log_file = gripper_log_path.open("w", newline="", encoding="utf-8")
        gripper_log_writer = csv.DictWriter(
            gripper_log_file,
            fieldnames=[
                "sim_time", "source_frame", "left_action", "right_action",
                "left_applied_target", "right_applied_target",
                "left_observation", "right_observation",
                "left_sim_actual", "right_sim_actual",
                "left_command_error", "right_command_error",
                "left_target_observation_difference", "right_target_observation_difference",
                "left_drive_effort", "right_drive_effort",
                "left_commanded_drive_effort", "right_commanded_drive_effort",
                "gripper_effort_limit_per_carriage",
                "left_effort_saturated", "right_effort_saturated",
                "left_contact_force", "right_contact_force",
                "penetration", "grasp_state", "object_name", "object_slip",
                "phone_xyz", "accessory_xyz",
            ],
        )
        gripper_log_writer.writeheader()
    phone_motion_file = None
    phone_motion_writer = None
    attachment_file = None
    attachment_writer = None
    ring_audit_file = None
    ring_audit_writer = None
    pose_audit_file = None
    pose_audit_writer = None
    if mode != "kinematic":
        phone_motion_file = (
            REPORT_ROOT / f"phone_passive_rotation_{friction_tag}.csv"
        ).open("w", newline="", encoding="utf-8")
        phone_motion_writer = csv.DictWriter(
            phone_motion_file,
            fieldnames=[
                "sim_time", "source_frame", "phone_x", "phone_y", "phone_z",
                "quat_x", "quat_y", "quat_z", "quat_w",
                "omega_x", "omega_y", "omega_z", "angular_speed",
                "wrist_x", "wrist_y", "wrist_z",
                "wrist_quat_x", "wrist_quat_y", "wrist_quat_z", "wrist_quat_w",
                "pinch_axis_x", "pinch_axis_y", "pinch_axis_z",
                "relative_rotation_angle_deg", "long_axis_vertical_error_deg",
                "relative_axis_x", "relative_axis_y", "relative_axis_z",
                "contact_count", "grasp_point_x", "grasp_point_y", "grasp_point_z",
                "grasp_point_com_offset_x", "grasp_point_com_offset_y",
                "grasp_point_com_offset_z", "grasp_point_com_offset_norm",
                "phone_lift_m", "grasp_state",
            ],
        )
        phone_motion_writer.writeheader()
        attachment_file = (
            REPORT_ROOT / f"phone_accessory_relative_transform_raw_{friction_tag}.csv"
        ).open("w", newline="", encoding="utf-8")
        attachment_writer = csv.DictWriter(
            attachment_file,
            fieldnames=[
                "sim_time", "source_frame", "joint_state",
                "relative_x", "relative_y", "relative_z",
                "relative_qx", "relative_qy", "relative_qz", "relative_qw",
                "relative_roll_deg", "relative_pitch_deg", "relative_yaw_deg",
                "translation_drift_m", "rotation_drift_deg",
                "relative_linear_velocity_m_s", "relative_angular_velocity_rad_s",
                "phone_world_rotation_from_initial_deg",
                "accessory_world_rotation_from_initial_deg",
            ],
        )
        attachment_writer.writeheader()
        ring_audit_file = (
            REPORT_ROOT / f"right_ring_geometry_raw_{friction_tag}.csv"
        ).open("w", newline="", encoding="utf-8")
        ring_fields = ["sim_time", "source_frame"]
        for side in ("left", "right"):
            ring_fields += [
                f"{side}_tip_world_x", f"{side}_tip_world_y", f"{side}_tip_world_z",
                f"{side}_tip_ring_x", f"{side}_tip_ring_y", f"{side}_tip_ring_z",
                f"{side}_axis_ring_x", f"{side}_axis_ring_y", f"{side}_axis_ring_z",
                f"{side}_axis_normal_angle_deg", f"{side}_center_radial_offset_m",
                f"{side}_inner_edge_clearance_m",
            ]
        ring_fields += [
            "ring_world_x", "ring_world_y", "ring_world_z",
            "ring_outward_x", "ring_outward_y", "ring_outward_z",
            "gripper_center_ring_x", "gripper_center_ring_y", "gripper_center_ring_z",
        ]
        ring_audit_writer = csv.DictWriter(ring_audit_file, fieldnames=ring_fields)
        ring_audit_writer.writeheader()
        if args.phone_pose_audit:
            pose_audit_file = (
                REPORT_ROOT / f"phone_pose_transform_audit_{friction_tag}.csv"
            ).open("w", newline="", encoding="utf-8")
            pose_audit_writer = csv.DictWriter(
                pose_audit_file,
                fieldnames=[
                    "source_frame", "quantity", "prim_path", "source",
                    "world_transform", "parent_transform", "local_transform",
                    "world_position_xyz", "note",
                ],
            )
            pose_audit_writer.writeheader()
    last_print = -1
    effort_overlay_label = None
    effort_overlay_window = None
    if args.show_gripper_effort and not args.headless:
        import omni.ui as ui
        effort_overlay_window = ui.Window("ALOHA Gripper Effort", width=520, height=245)
        with effort_overlay_window.frame:
            effort_overlay_label = ui.Label(
                "Waiting for physics data...", alignment=ui.Alignment.LEFT_TOP
            )
    while simulation_app.is_running():
        source_frame_float = min(sim_time * fps * args.speed, end_frame)
        frame = int(source_frame_float)
        next_frame = min(frame + 1, end_frame)
        alpha = source_frame_float - frame
        state_x = (1.0 - alpha) * state_trajectory[frame] + alpha * state_trajectory[next_frame]
        action_x = (1.0 - alpha) * action_trajectory[frame] + alpha * action_trajectory[next_frame]
        gripper_x = action_x if gripper_source == "action" else state_x
        for side, offset in (("left", 0), ("right", 7)):
            for j in range(6):
                pos[0, indices[f"follower_{side}_joint_{j}"]] = float(state_x[offset + j])
            raw_g = float(gripper_x[offset + 6])
            if not -0.001 <= raw_g <= 0.044 + 1e-4:
                raise RuntimeError(f"frame {frame} {side} gripper command outside tolerance: {raw_g}")
            g = min(max(raw_g, 0.0), .044)
            if args.gripper_command_mode == "hardware-max-close" and raw_g <= 0.001:
                # Command-derived close state; no frame-based force or grasp event.
                g = 0.0
            pos[0, indices[f"follower_{side}_left_carriage_joint"]] = g
            pos[0, indices[f"follower_{side}_right_carriage_joint"]] = g
        if ring_controller is not None:
            body_names = list(robot.data.body_names)
            insertion_body_index = body_names.index("follower_right_carriage_left")
            opposing_body_index = body_names.index("follower_right_carriage_right")
            insertion_pose = robot.data.body_pose_w.torch[
                0, insertion_body_index
            ].detach().cpu().numpy()
            opposing_pose = robot.data.body_pose_w.torch[
                0, opposing_body_index
            ].detach().cpu().numpy()
            insertion_q_xyzw = insertion_pose[[4, 5, 6, 3]]
            opposing_q_xyzw = opposing_pose[[4, 5, 6, 3]]
            insertion_r = ring_quat_to_matrix(insertion_q_xyzw)
            opposing_r = ring_quat_to_matrix(opposing_q_xyzw)
            insertion_tip_offset = np.array([0.069561665, -0.020484785, -0.001431745])
            opposing_tip_offset = np.array([0.069561665, 0.020484785, -0.001431745])
            insertion_tip = insertion_pose[:3] + insertion_r @ insertion_tip_offset
            opposing_tip = opposing_pose[:3] + opposing_r @ opposing_tip_offset
            ring_finger_state = RingFingerState(
                center=insertion_tip,
                quaternion_xyzw=insertion_q_xyzw,
                longitudinal=insertion_r[:, 0],
                wide=insertion_r[:, 2],
                thin=-insertion_r[:, 1],
                opposing_center=opposing_tip,
            )
            accessory_pose_control = accessory.data.root_pose_w.torch[
                0
            ].detach().cpu().numpy()
            accessory_pose_control = accessory_pose_control.copy()
            accessory_pose_control[3:7] = accessory_pose_control[[4, 5, 6, 3]]
            recorded_close = float(gripper_x[13]) <= 0.001
            ring_step, ring_frame_current, ring_geo = ring_controller.compute_target(
                sim_time, frame, accessory_pose_control, ring_finger_state,
                ring_contact_state, recorded_close,
            )
            if not ring_controller.close_enabled:
                pos[0, indices["follower_right_left_carriage_joint"]] = right_open_target
                pos[0, indices["follower_right_right_carriage_joint"]] = right_open_target
            if ring_step is not None:
                target_p, target_r, _ = ring_step
                jacobians = robot.root_physx_view.get_jacobians()
                if ring_controller.target_count == 0:
                    print(
                        f"[RING_IK_API_LOOP] jacobians={type(jacobians)} "
                        f"shape={getattr(jacobians, 'shape', 'NONE')} "
                        f"limits={type(robot.data.soft_joint_pos_limits)}",
                        flush=True,
                    )
                jacobian_body_index = insertion_body_index - 1
                jac_tensor = (
                    jacobians if isinstance(jacobians, torch.Tensor)
                    else __import__("warp").to_torch(jacobians)
                )
                jac = jac_tensor[
                    0, jacobian_body_index, :, right_arm_joint_ids
                ].detach().cpu().numpy()
                point_offset = insertion_tip - insertion_pose[:3]
                skew = np.array([
                    [0.0, -point_offset[2], point_offset[1]],
                    [point_offset[2], 0.0, -point_offset[0]],
                    [-point_offset[1], point_offset[0], 0.0],
                ])
                jac[:3, :] = jac[:3, :] - skew @ jac[3:, :]
                q_now = robot.data.joint_pos.torch[
                    0, right_arm_joint_ids
                ].detach().cpu().numpy()
                limits = robot.data.soft_joint_pos_limits.torch[
                    0, right_arm_joint_ids
                ].detach().cpu().numpy()
                q_target, pe, re = ring_controller.solve_dls(
                    insertion_tip, insertion_r, target_p, target_r, jac,
                    q_now, limits[:, 0], limits[:, 1],
                    np.full(6, 8.0), np.full(6, 20.0),
                )
                ik_ok = q_target is not None
                if ik_ok:
                    for local_index, joint_id in enumerate(right_arm_joint_ids):
                        pos[0, joint_id] = float(q_target[local_index])
                ring_controller.log_target(
                    sim_time, frame, ring_frame_current, ring_geo,
                    ring_contact_state, (target_p, target_r),
                    ik_ok, pe, re, q_target if ik_ok else q_now,
                )
        if mode == "kinematic":
            robot.write_joint_state_to_sim(position=pos, velocity=vel)
        else:
            robot.set_joint_position_target(pos)
            robot.write_data_to_sim()
        if mode == "physical-task":
            from magsafe_magnet_controller import MagnetState, quat_rotate

            accessory_state = _body(accessory)
            phone_state = _body(phone)
            # A native breakable joint remains the source of truth. Full SE(3)
            # drift is used only to observe that PhysX has broken it; ordinary
            # assembly rotation must not be mistaken for detachment.
            current_phone_pose = phone.data.root_pose_w.torch[0].detach().cpu().numpy()
            current_accessory_pose = accessory.data.root_pose_w.torch[0].detach().cpu().numpy()
            relative_position, relative_quaternion = _relative_pose(
                current_phone_pose, current_accessory_pose
            )
            relative_translation_drift = float(
                np.linalg.norm(relative_position - initial_relative_position)
            )
            relative_q_delta = _quat_mul_xyzw_global(
                np.array([
                    -initial_relative_quaternion[0], -initial_relative_quaternion[1],
                    -initial_relative_quaternion[2], initial_relative_quaternion[3],
                ]),
                relative_quaternion,
            )
            relative_rotation_drift_deg = 2.0 * math.degrees(
                math.acos(float(np.clip(abs(relative_q_delta[3]), 0.0, 1.0)))
            )
            if accessory_attached and (
                relative_translation_drift > 0.003 or relative_rotation_drift_deg > 5.0
            ):
                accessory_attached = False
                accessory_detach_frame = frame
                acc_pair.state = MagnetState.COOLDOWN
                acc_pair.cooldown_until = sim_time + cfg["accessory_phone"]["cooldown_s"]
                phase_detector.advance(frame, TaskPhase.ACCESSORY_DETACH)
            ar = acc_pair.update(sim_time, accessory_state, phone_state)
            phone_normal = quat_rotate(phone_state.quaternion_wxyz, np.array([0, 1, 0.0]))
            correct_face = float(np.dot(phone_normal, -charger_pair.target_normal_local)) > 0.0
            pr = charger_pair.update(
                sim_time,
                phone_state,
                charger,
                penetration=contact_monitor.max_penetration if contact_monitor else 0.0,
                correct_face=correct_face,
            )
            accessory_force = ar.force if not accessory_attached else np.zeros(3)
            accessory_torque = ar.torque if not accessory_attached else np.zeros(3)
            _wrench(accessory, accessory_force, accessory_torque)
            _wrench(phone, pr.force - accessory_force, pr.torque - accessory_torque)
            if pr.attach_event:
                magnetic_attach_frame = frame
                phase_detector.advance(frame, TaskPhase.FINAL_ATTACHED)
            accessory.write_data_to_sim()
            phone.write_data_to_sim()
            magnet_logger.write(sim_time, "accessory_phone", ar, "native_joint" if accessory_attached else "broken")
            magnet_logger.write(sim_time, "phone_charger", pr, "force_soft_lock")
        if shape_contact_logger is not None:
            body_states = {}
            robot_body_names = list(robot.data.body_names)
            for body_path in monitored_body_paths:
                body_name = body_path.rsplit("/", 1)[-1]
                if body_name not in robot_body_names:
                    continue
                body_index = robot_body_names.index(body_name)
                body_states[body_path] = {
                    "position": robot.data.body_pos_w.torch[0, body_index].detach().cpu().numpy(),
                    "quaternion": robot.data.body_quat_w.torch[0, body_index].detach().cpu().numpy(),
                    "linear": robot.data.body_lin_vel_w.torch[0, body_index].detach().cpu().numpy(),
                    "angular": robot.data.body_ang_vel_w.torch[0, body_index].detach().cpu().numpy(),
                }
            if phone is not None:
                phone_pose_context = phone.data.root_pose_w.torch[0].detach().cpu().numpy()
                body_states["/World/MagSafeScene/Phone"] = {
                    "position": phone_pose_context[:3],
                    "quaternion": np.array([
                        phone_pose_context[6], phone_pose_context[3],
                        phone_pose_context[4], phone_pose_context[5],
                    ]),
                    "linear": phone.data.root_lin_vel_w.torch[0].detach().cpu().numpy(),
                    "angular": phone.data.root_ang_vel_w.torch[0].detach().cpu().numpy(),
                }
            shape_contact_logger.set_context(sim_time, frame, body_states)
        sim.step(render=True)
        robot.update(dt)
        if phone is not None:
            phone.update(dt)
            accessory.update(dt)
            phone_pose = phone.data.root_pose_w.torch[0].detach().cpu().numpy()
            accessory_pose = accessory.data.root_pose_w.torch[0].detach().cpu().numpy()
            relative_position_log, relative_quaternion_log = _relative_pose(
                phone_pose, accessory_pose
            )
            relative_q_delta_log = _quat_mul_xyzw_global(
                np.array([
                    -initial_relative_quaternion[0], -initial_relative_quaternion[1],
                    -initial_relative_quaternion[2], initial_relative_quaternion[3],
                ]),
                relative_quaternion_log,
            )
            translation_drift_log = float(
                np.linalg.norm(relative_position_log - initial_relative_position)
            )
            rotation_drift_log = 2.0 * math.degrees(
                math.acos(float(np.clip(abs(relative_q_delta_log[3]), 0.0, 1.0)))
            )
            qx, qy, qz, qw = relative_quaternion_log
            roll = math.degrees(math.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy)))
            pitch = math.degrees(math.asin(float(np.clip(2*(qw*qy-qz*qx), -1, 1))))
            yaw = math.degrees(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
            phone_velocity_log = phone.data.root_vel_w.torch[0].detach().cpu().numpy()
            accessory_velocity_log = accessory.data.root_vel_w.torch[0].detach().cpu().numpy()
            relative_linear_speed_log = float(
                np.linalg.norm(accessory_velocity_log[:3] - phone_velocity_log[:3])
            )
            relative_angular_speed_log = float(
                np.linalg.norm(accessory_velocity_log[3:] - phone_velocity_log[3:])
            )

            def _world_rotation_from_initial(current_q, initial_q):
                delta = _quat_mul_xyzw_global(
                    np.array([-initial_q[0], -initial_q[1], -initial_q[2], initial_q[3]]),
                    current_q,
                )
                return 2.0 * math.degrees(
                    math.acos(float(np.clip(abs(delta[3]), 0.0, 1.0)))
                )

            attachment_writer.writerow({
                "sim_time": f"{sim_time:.9f}", "source_frame": frame,
                "joint_state": "ATTACHED" if accessory_attached else "DETACHED",
                "relative_x": f"{relative_position_log[0]:.12g}",
                "relative_y": f"{relative_position_log[1]:.12g}",
                "relative_z": f"{relative_position_log[2]:.12g}",
                "relative_qx": f"{qx:.12g}", "relative_qy": f"{qy:.12g}",
                "relative_qz": f"{qz:.12g}", "relative_qw": f"{qw:.12g}",
                "relative_roll_deg": f"{roll:.12g}",
                "relative_pitch_deg": f"{pitch:.12g}",
                "relative_yaw_deg": f"{yaw:.12g}",
                "translation_drift_m": f"{translation_drift_log:.12g}",
                "rotation_drift_deg": f"{rotation_drift_log:.12g}",
                "relative_linear_velocity_m_s": f"{relative_linear_speed_log:.12g}",
                "relative_angular_velocity_rad_s": f"{relative_angular_speed_log:.12g}",
                "phone_world_rotation_from_initial_deg": f"{_world_rotation_from_initial(phone_pose[3:7], initial_phone_pose[3:7]):.12g}",
                "accessory_world_rotation_from_initial_deg": f"{_world_rotation_from_initial(accessory_pose[3:7], initial_accessory_pose[3:7]):.12g}",
            })
            ring_row = {"sim_time": f"{sim_time:.9f}", "source_frame": frame}
            ring_world = accessory_pose[:3]
            ring_outward = _quat_mul_xyzw_global(
                _quat_mul_xyzw_global(
                    accessory_pose[3:7], np.array([0.0, 1.0, 0.0, 0.0])
                ),
                np.array([
                    -accessory_pose[3], -accessory_pose[4], -accessory_pose[5],
                    accessory_pose[6],
                ]),
            )[:3]
            ring_outward /= max(float(np.linalg.norm(ring_outward)), 1.0e-12)
            tip_worlds = {}
            robot_body_names_for_ring = list(robot.data.body_names)
            for side, local_y in (("left", -0.020484785), ("right", 0.020484785)):
                body_index = robot_body_names_for_ring.index(
                    f"follower_right_carriage_{side}"
                )
                body_pose = robot.data.body_pose_w.torch[
                    0, body_index
                ].detach().cpu().numpy()
                local_tip = np.array([0.069561665, local_y, -0.001431745])
                local_axis = np.array([1.0, 0.0, 0.0])
                tip_world = body_pose[:3] + _quat_mul_xyzw_global(
                    _quat_mul_xyzw_global(
                        body_pose[3:7], np.array([*local_tip, 0.0])
                    ),
                    np.array([
                        -body_pose[3], -body_pose[4], -body_pose[5], body_pose[6]
                    ]),
                )[:3]
                axis_world = _quat_mul_xyzw_global(
                    _quat_mul_xyzw_global(
                        body_pose[3:7], np.array([*local_axis, 0.0])
                    ),
                    np.array([
                        -body_pose[3], -body_pose[4], -body_pose[5], body_pose[6]
                    ]),
                )[:3]
                tip_accessory = _quat_rotate_inverse_xyzw(
                    accessory_pose[3:7], tip_world-ring_world
                )
                axis_accessory = _quat_rotate_inverse_xyzw(
                    accessory_pose[3:7], axis_world
                )
                tip_ring = np.array([
                    tip_accessory[0], -tip_accessory[2], tip_accessory[1]
                ])
                axis_ring = np.array([
                    axis_accessory[0], -axis_accessory[2], axis_accessory[1]
                ])
                radial = float(np.linalg.norm(tip_ring[:2]))
                angle = math.degrees(math.acos(float(np.clip(
                    abs(np.dot(axis_world, ring_outward)), 0.0, 1.0
                ))))
                for axis_name, value in zip("xyz", tip_world):
                    ring_row[f"{side}_tip_world_{axis_name}"] = f"{value:.12g}"
                for axis_name, value in zip("xyz", tip_ring):
                    ring_row[f"{side}_tip_ring_{axis_name}"] = f"{value:.12g}"
                for axis_name, value in zip("xyz", axis_ring):
                    ring_row[f"{side}_axis_ring_{axis_name}"] = f"{value:.12g}"
                ring_row[f"{side}_axis_normal_angle_deg"] = f"{angle:.12g}"
                ring_row[f"{side}_center_radial_offset_m"] = f"{radial:.12g}"
                # Conservative clearance includes half the 19.656 mm sleeve width.
                ring_row[f"{side}_inner_edge_clearance_m"] = f"{0.0225-radial-0.009828095:.12g}"
                tip_worlds[side] = tip_world
            gripper_center = 0.5*(tip_worlds["left"]+tip_worlds["right"])
            center_accessory = _quat_rotate_inverse_xyzw(
                accessory_pose[3:7], gripper_center-ring_world
            )
            center_ring = np.array([
                center_accessory[0], -center_accessory[2], center_accessory[1]
            ])
            for axis_name, value in zip("xyz", ring_world):
                ring_row[f"ring_world_{axis_name}"] = f"{value:.12g}"
            for axis_name, value in zip("xyz", ring_outward):
                ring_row[f"ring_outward_{axis_name}"] = f"{value:.12g}"
            for axis_name, value in zip("xyz", center_ring):
                ring_row[f"gripper_center_ring_{axis_name}"] = f"{value:.12g}"
            ring_audit_writer.writerow(ring_row)
            if pose_audit_writer is not None:
                visual_path = "/World/MagSafeScene/Phone/Visuals/MetalFrame"
                phone_path = "/World/MagSafeScene/Phone"
                wrist_path = (
                    "/World/StationaryALOHA/Asset/Geometry/tabletop_link/"
                    "follower_left_base_link/follower_left_link_1/follower_left_link_2/"
                    "follower_left_link_3/follower_left_link_4/follower_left_link_5/"
                    "follower_left_link_6"
                )

                def _matrix_list(matrix):
                    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]

                def _pose_matrix(position, quaternion, quaternion_order: str):
                    if quaternion_order == "xyzw":
                        quaternion = [
                            quaternion[3], quaternion[0], quaternion[1], quaternion[2]
                        ]
                    matrix = Gf.Matrix4d(1.0)
                    matrix.SetRotate(
                        Gf.Quatd(
                            float(quaternion[0]),
                            Gf.Vec3d(*[float(x) for x in quaternion[1:4]]),
                        )
                    )
                    matrix.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in position]))
                    return matrix

                def _write_transform(quantity, path, source, world, parent, local, note=""):
                    position = (
                        list(world.ExtractTranslation()) if world is not None else [math.nan] * 3
                    )
                    pose_audit_writer.writerow(
                        {
                            "source_frame": frame,
                            "quantity": quantity,
                            "prim_path": path,
                            "source": source,
                            "world_transform": json.dumps(_matrix_list(world)) if world is not None else "N/A",
                            "parent_transform": json.dumps(_matrix_list(parent)) if parent is not None else "N/A",
                            "local_transform": json.dumps(_matrix_list(local)) if local is not None else "N/A",
                            "world_position_xyz": json.dumps(position),
                            "note": note,
                        }
                    )

                identity = Gf.Matrix4d(1.0)
                phone_world = _pose_matrix(phone_pose[:3], phone_pose[3:7], "xyzw")
                phone_parent = identity
                phone_local = phone_world
                # The visible MetalFrame has an identity local transform below the
                # rigid-body Xform. Its runtime world transform therefore follows
                # the PhysX tensor pose, not the stale USD/Fabric XformCache value.
                visual_world = phone_world
                visual_parent = phone_world
                visual_local = identity
                robot_body_names = list(robot.data.body_names)
                wrist_index = robot_body_names.index("follower_left_link_6")
                wrist_parent_index = robot_body_names.index("follower_left_link_5")
                wrist_position = robot.data.body_pos_w.torch[0, wrist_index].detach().cpu().numpy()
                wrist_quaternion = robot.data.body_quat_w.torch[0, wrist_index].detach().cpu().numpy()
                wrist_parent_position = (
                    robot.data.body_pos_w.torch[0, wrist_parent_index].detach().cpu().numpy()
                )
                wrist_parent_quaternion = (
                    robot.data.body_quat_w.torch[0, wrist_parent_index].detach().cpu().numpy()
                )
                wrist_world = _pose_matrix(wrist_position, wrist_quaternion, "wxyz")
                wrist_parent = _pose_matrix(
                    wrist_parent_position, wrist_parent_quaternion, "wxyz"
                )
                wrist_local = wrist_world * wrist_parent.GetInverse()
                phone_com = phone.data.body_com_pos_w.torch[0, 0].detach().cpu().numpy()
                com_world = Gf.Matrix4d(1.0)
                com_world.SetTranslate(Gf.Vec3d(*[float(x) for x in phone_com]))
                com_local = com_world * phone_parent.GetInverse()
                _write_transform(
                    "actual_phone_rigid_body_COM", phone_path,
                    "RigidObject.data.body_com_pos_w", com_world, phone_parent, com_local,
                    "MassAPI centerOfMass local=[0,0,0]",
                )
                _write_transform(
                    "phone_rigid_body_Xform", phone_path, "RigidObject.data.root_pose_w",
                    phone_world, phone_parent, phone_local,
                    "Runtime tensor used; USD XformCache is stale when Fabric synchronization is disabled",
                )
                _write_transform(
                    "phone_visual_mesh_origin", visual_path,
                    "RigidObject root_pose_w composed with authored visual local transform",
                    visual_world, visual_parent, visual_local,
                )
                _write_transform(
                    "left_wrist_link6_origin", wrist_path, "Articulation.data.body_pose_w",
                    wrist_world, wrist_parent, wrist_local,
                    "No separate TCP prim exists; link_6 rigid-body origin is reported",
                )
                _write_transform(
                    "EE_target", "NONE", "NONE", None, None, None,
                    "Replay has no Cartesian EE target or EE-target prim",
                )
                _write_transform(
                    "articulation_position_target", fixed_joint_path,
                    "robot.set_joint_position_target", None, None, None,
                    "Joint-space vector only; no world transform exists",
                )
                for key, position in (
                    ("phone_com", phone_com),
                    ("visual_origin", np.asarray(visual_world.ExtractTranslation(), dtype=float)),
                    ("wrist", wrist_position),
                ):
                    sphere = audit_overlay[key]
                    sphere.GetPrim().GetAttribute("xformOp:translate").Set(
                        Gf.Vec3d(*[float(x) for x in position])
                    ) if sphere.GetPrim().HasAttribute("xformOp:translate") else UsdGeom.Xformable(
                        sphere.GetPrim()
                    ).AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in position]))
            apertures = {
                "follower_left_": 2.0 * float(gripper_x[6]),
                "follower_right_": 2.0 * float(gripper_x[13]),
            }
            current_grasp_states = {
                "follower_left_": grasp_detector.states.get(("follower_left_", "Accessory"), GraspState.NONE).value,
                "follower_right_": grasp_detector.states.get(("follower_right_", "Accessory"), GraspState.NONE).value,
            }
            contacts = contact_monitor.sample(
                sim_time=sim_time,
                source_frame=frame,
                task_phase=phase_detector.phase.value,
                apertures=apertures,
                grasp_states=current_grasp_states,
                accessory_attached=accessory_attached,
                phone_charger_state=(pr.state.value if mode == "physical-task" else "DISABLED"),
                phone_position=phone_pose[:3],
                phone_orientation=phone_pose[3:7],
            )
            if ring_controller is not None:
                from ring_affordance_controller import ContactState as RingContactState
                accessory_pose_xyzw = accessory_pose.copy()
                accessory_pose_xyzw[3:7] = accessory_pose[[4, 5, 6, 3]]
                current_ring = ring_controller.provider.get(accessory_pose_xyzw)
                insertion_contacts = [
                    c for c in contacts
                    if "follower_right_carriage_left" in c.sensor_prim
                    and "accessory" in c.other_prim.lower() and c.normal_force > 0.0
                ]
                opposing_contacts = [
                    c for c in contacts
                    if "follower_right_carriage_right" in c.sensor_prim
                    and "accessory" in c.other_prim.lower() and c.normal_force > 0.0
                ]
                insertion_inner = any(
                    np.linalg.norm(current_ring.local(c.point)[:2])
                    <= ring_controller.INNER_RADIUS + 0.002
                    for c in insertion_contacts
                )
                opposing_outer = any(
                    np.linalg.norm(current_ring.local(c.point)[:2])
                    >= ring_controller.INNER_RADIUS - 0.002
                    for c in opposing_contacts
                )
                accessory_before_crossing = bool(
                    (insertion_contacts or opposing_contacts)
                    and not ring_controller.plane_crossed
                )
                phone_collision = any(
                    "follower_right_" in c.sensor_prim
                    and "phone" in c.other_prim.lower() and c.normal_force > 0.0
                    for c in contacts
                )
                ring_contact_state = RingContactState(
                    insertion_inner=insertion_inner,
                    opposing_outer=opposing_outer,
                    side_collision=accessory_before_crossing,
                    phone_collision=phone_collision,
                    max_penetration_m=max(
                        [c.penetration for c in insertion_contacts+opposing_contacts] or [0.0]
                    ),
                    pull_force_n=ring_contact_state.pull_force_n,
                )
            if phone_table_contact_monitor is not None:
                phone_table_contact_monitor.sample(
                    sim_time=sim_time,
                    source_frame=frame,
                    task_phase=phase_detector.phase.value,
                    apertures={},
                    grasp_states={},
                    accessory_attached=accessory_attached,
                    phone_charger_state=(pr.state.value if mode == "physical-task" else "DISABLED"),
                    phone_position=phone_pose[:3],
                    phone_orientation=phone_pose[3:7],
                )
            if (
                accessory_attached
                and corrected_accessory_joint is not None
                and corrected_accessory_joint.GetJointEnabledAttr().Get()
            ):
                right_accessory = [
                    c for c in contacts
                    if "follower_right_" in c.sensor_prim
                    and "accessory" in c.other_prim.lower()
                    and c.normal_force > 0.0
                ]
                # De-duplicate patch repetitions by sensor rigid body.
                unique_right_accessory = {}
                for c in right_accessory:
                    unique_right_accessory.setdefault(c.sensor_prim, c)
                right_accessory = list(unique_right_accessory.values())
                right_sides = {
                    "left" if "carriage_left" in c.sensor_prim else "right"
                    for c in right_accessory
                }
                pull_axis = accessory_pose[:3] - phone_pose[:3]
                pull_axis /= max(float(np.linalg.norm(pull_axis)), 1.0e-12)
                accessory_force = np.sum(
                    [-(c.normal*c.normal_force + c.friction_force) for c in right_accessory],
                    axis=0,
                ) if right_accessory else np.zeros(3)
                accessory_com = accessory.data.body_com_pos_w.torch[
                    0, 0
                ].detach().cpu().numpy()
                accessory_torque = np.sum(
                    [
                        np.cross(c.point-accessory_com, -(c.normal*c.normal_force+c.friction_force))
                        for c in right_accessory
                    ],
                    axis=0,
                ) if right_accessory else np.zeros(3)
                pull_force = float(np.dot(accessory_force, pull_axis))
                break_force = float(phone_accessory_joint_report["break_force_n"])
                break_torque = float(phone_accessory_joint_report["break_torque_nm"])
                if ring_controller is not None:
                    ring_contact_state.pull_force_n = max(0.0, pull_force)
                    physical_pull_gate = (
                        ring_controller.stage.value == "PHYSICAL_PULL"
                        and right_sides == {"left", "right"}
                        and pull_force >= break_force
                    )
                    ring_controller.pull_force_hold = (
                        ring_controller.pull_force_hold + dt
                        if physical_pull_gate else 0.0
                    )
                    should_break = ring_controller.pull_force_hold >= 0.08
                else:
                    should_break = right_sides == {"left", "right"} and (
                        pull_force >= break_force
                        or float(np.linalg.norm(accessory_torque)) >= break_torque
                    )
                if should_break:
                    corrected_accessory_joint.GetJointEnabledAttr().Set(False)
                    accessory_attached = False
                    accessory_detach_frame = frame
                    acc_pair.state = MagnetState.COOLDOWN
                    acc_pair.cooldown_until = sim_time + cfg["accessory_phone"]["cooldown_s"]
                    phase_detector.advance(frame, TaskPhase.ACCESSORY_DETACH)
                    if ring_controller is not None:
                        ring_controller.note_detach(frame)
                    print(
                        f"[ACCESSORY_PHYSICAL_BREAK] frame={frame} "
                        f"right_pull_force_n={pull_force:.9g} "
                        f"right_torque_nm={np.linalg.norm(accessory_torque):.9g}",
                        flush=True,
                    )
            for object_name in ("Accessory", "Phone"):
                if any(
                    object_name.lower() in c.other_prim.lower() and c.normal_force >= 0.05
                    for c in contacts
                ):
                    first_contact.setdefault(object_name, frame)
                    phase_detector.advance(
                        frame,
                        TaskPhase.ACCESSORY_CONTACT if object_name == "Accessory" else TaskPhase.PHONE_CONTACT,
                    )
            for gripper, aperture_key in (("follower_left_", 6), ("follower_right_", 13)):
                carriage_indices = [
                    body_index
                    for body_index, body_name in enumerate(robot.data.body_names)
                    if body_name.startswith(gripper) and "carriage_" in body_name
                ]
                gripper_velocity = (
                    torch.mean(
                        robot.data.body_lin_vel_w.torch[0, carriage_indices],
                        dim=0,
                    ).detach().cpu().numpy()
                    if carriage_indices
                    else np.zeros(3)
                )
                for object_name, obj_pose in (("Accessory", accessory_pose), ("Phone", phone_pose)):
                    obj = accessory if object_name == "Accessory" else phone
                    object_velocity = obj.data.root_lin_vel_w.torch[0].detach().cpu().numpy()
                    relative_speed = float(np.linalg.norm(object_velocity - gripper_velocity))
                    observation = grasp_detector.update(
                        sim_time=sim_time,
                        frame=frame,
                        gripper=gripper,
                        object_name=object_name,
                        contacts=contacts,
                        aperture=2.0 * float(gripper_x[aperture_key]),
                        relative_speed=relative_speed,
                    )
                    max_slip = max(max_slip, observation.slip)
                    if observation.state == GraspState.STABLE_GRASP:
                        phase_detector.advance(
                            frame,
                            TaskPhase.ACCESSORY_GRASP if object_name == "Accessory" else TaskPhase.PHONE_GRASP,
                        )
            last_object_positions["Accessory"] = accessory_pose[:3].copy()
            last_object_positions["Phone"] = phone_pose[:3].copy()
            left_joint_ids = [
                indices["follower_left_left_carriage_joint"],
                indices["follower_left_right_carriage_joint"],
            ]
            right_joint_ids = [
                indices["follower_right_left_carriage_joint"],
                indices["follower_right_right_carriage_joint"],
            ]
            left_actual = float(torch.mean(robot.data.joint_pos.torch[0, left_joint_ids]).item())
            right_actual = float(torch.mean(robot.data.joint_pos.torch[0, right_joint_ids]).item())
            left_effort = float(torch.sum(torch.abs(robot.data.applied_torque.torch[0, left_joint_ids])).item())
            right_effort = float(torch.sum(torch.abs(robot.data.applied_torque.torch[0, right_joint_ids])).item())
            left_target = float(pos[0, left_joint_ids[0]].item())
            right_target = float(pos[0, right_joint_ids[0]].item())
            left_velocity = robot.data.joint_vel.torch[0, left_joint_ids]
            right_velocity = robot.data.joint_vel.torch[0, right_joint_ids]
            left_errors = pos[0, left_joint_ids] - robot.data.joint_pos.torch[0, left_joint_ids]
            right_errors = pos[0, right_joint_ids] - robot.data.joint_pos.torch[0, right_joint_ids]
            left_commanded_per_joint = gripper_stiffness * left_errors - 10.0 * left_velocity
            right_commanded_per_joint = gripper_stiffness * right_errors - 10.0 * right_velocity
            left_commanded = float(torch.sum(torch.abs(left_commanded_per_joint)).item())
            right_commanded = float(torch.sum(torch.abs(right_commanded_per_joint)).item())
            left_saturated = bool(
                torch.any(torch.abs(left_commanded_per_joint) >= gripper_effort_limit * 0.999)
            )
            right_saturated = bool(
                torch.any(torch.abs(right_commanded_per_joint) >= gripper_effort_limit * 0.999)
            )
            left_force = sum(c.normal_force for c in contacts if "follower_left_" in c.sensor_prim)
            right_force = sum(c.normal_force for c in contacts if "follower_right_" in c.sensor_prim)
            contacted_objects = sorted(
                {
                    name
                    for name in ("Accessory", "Phone", "Table")
                    if any(name.lower() in c.other_prim.lower() and c.normal_force >= 0.05 for c in contacts)
                }
            )
            grasp_values = [state.value for state in grasp_detector.states.values()]
            gripper_log_writer.writerow(
                {
                    "sim_time": f"{sim_time:.9f}", "source_frame": frame,
                    "left_action": f"{action_x[6]:.9g}", "right_action": f"{action_x[13]:.9g}",
                    "left_applied_target": f"{left_target:.9g}",
                    "right_applied_target": f"{right_target:.9g}",
                    "left_observation": f"{state_x[6]:.9g}", "right_observation": f"{state_x[13]:.9g}",
                    "left_sim_actual": f"{left_actual:.9g}", "right_sim_actual": f"{right_actual:.9g}",
                    "left_command_error": f"{left_target - left_actual:.9g}",
                    "right_command_error": f"{right_target - right_actual:.9g}",
                    "left_target_observation_difference": f"{float(gripper_x[6] - state_x[6]):.9g}",
                    "right_target_observation_difference": f"{float(gripper_x[13] - state_x[13]):.9g}",
                    "left_drive_effort": f"{left_effort:.9g}", "right_drive_effort": f"{right_effort:.9g}",
                    "left_commanded_drive_effort": f"{left_commanded:.9g}",
                    "right_commanded_drive_effort": f"{right_commanded:.9g}",
                    "gripper_effort_limit_per_carriage": f"{gripper_effort_limit:.9g}",
                    "left_effort_saturated": int(left_saturated),
                    "right_effort_saturated": int(right_saturated),
                    "left_contact_force": f"{left_force:.9g}", "right_contact_force": f"{right_force:.9g}",
                    "penetration": f"{max((c.penetration for c in contacts), default=0.0):.9g}",
                    "grasp_state": "|".join(sorted(set(grasp_values))),
                    "object_name": "|".join(contacted_objects) or "NONE",
                    "object_slip": f"{max_slip:.9g}",
                    "phone_xyz": " ".join(f"{v:.9g}" for v in phone_pose[:3]),
                    "accessory_xyz": " ".join(f"{v:.9g}" for v in accessory_pose[:3]),
                }
            )
            phone_contacts = [
                c for c in contacts
                if "phone" in c.other_prim.lower() and c.normal_force >= 0.05
            ]
            phone_com = phone.data.body_com_pos_w.torch[0, 0].detach().cpu().numpy()
            if phone_contacts:
                grasp_point = np.mean([c.point for c in phone_contacts], axis=0)
                grasp_offset = grasp_point - phone_com
            else:
                grasp_point = np.full(3, np.nan)
                grasp_offset = np.full(3, np.nan)
            q = phone_pose[3:7]
            q0 = initial_phone_pose[3:7]

            def _quat_mul_xyzw(a, b):
                ax, ay, az, aw = a
                bx, by, bz, bw = b
                return np.array(
                    [
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz,
                    ]
                )

            q_rel = _quat_mul_xyzw(q, np.array([-q0[0], -q0[1], -q0[2], q0[3]]))
            if q_rel[3] < 0.0:
                q_rel = -q_rel
            relative_angle = 2.0 * math.degrees(
                math.acos(float(np.clip(q_rel[3], 0.0, 1.0)))
            )
            axis_norm = float(np.linalg.norm(q_rel[:3]))
            relative_axis = q_rel[:3] / axis_norm if axis_norm > 1.0e-9 else np.zeros(3)
            q_wxyz = np.array([q[3], q[0], q[1], q[2]])
            from magsafe_magnet_controller import quat_rotate
            long_axis = quat_rotate(q_wxyz, np.array([1.0, 0.0, 0.0]))
            vertical_error = math.degrees(
                math.acos(float(np.clip(abs(long_axis[2]), 0.0, 1.0)))
            )
            omega = phone.data.root_ang_vel_w.torch[0].detach().cpu().numpy()
            wrist_index = list(robot.data.body_names).index("follower_left_link_6")
            wrist_pose = robot.data.body_pose_w.torch[0, wrist_index].detach().cpu().numpy()
            wx, wy, wz, ww = wrist_pose[3:7]
            wrist_rotation = np.array(
                [
                    [1 - 2 * (wy * wy + wz * wz), 2 * (wx * wy - wz * ww), 2 * (wx * wz + wy * ww)],
                    [2 * (wx * wy + wz * ww), 1 - 2 * (wx * wx + wz * wz), 2 * (wy * wz - wx * ww)],
                    [2 * (wx * wz - wy * ww), 2 * (wy * wz + wx * ww), 1 - 2 * (wx * wx + wy * wy)],
                ]
            )
            # Imported Stationary ALOHA closes along link-6 local +Y/-Y.
            pinch_axis = wrist_rotation[:, 1]
            phone_grasp_states = [
                state.value
                for (gripper_name, object_name), state in grasp_detector.states.items()
                if object_name == "Phone"
            ]
            phone_motion_writer.writerow(
                {
                    "sim_time": f"{sim_time:.9f}", "source_frame": frame,
                    "phone_x": f"{phone_pose[0]:.9g}", "phone_y": f"{phone_pose[1]:.9g}",
                    "phone_z": f"{phone_pose[2]:.9g}",
                    "quat_x": f"{q[0]:.9g}", "quat_y": f"{q[1]:.9g}",
                    "quat_z": f"{q[2]:.9g}", "quat_w": f"{q[3]:.9g}",
                    "omega_x": f"{omega[0]:.9g}", "omega_y": f"{omega[1]:.9g}",
                    "omega_z": f"{omega[2]:.9g}",
                    "angular_speed": f"{float(np.linalg.norm(omega)):.9g}",
                    "wrist_x": f"{wrist_pose[0]:.9g}",
                    "wrist_y": f"{wrist_pose[1]:.9g}",
                    "wrist_z": f"{wrist_pose[2]:.9g}",
                    "wrist_quat_x": f"{wx:.9g}",
                    "wrist_quat_y": f"{wy:.9g}",
                    "wrist_quat_z": f"{wz:.9g}",
                    "wrist_quat_w": f"{ww:.9g}",
                    "pinch_axis_x": f"{pinch_axis[0]:.9g}",
                    "pinch_axis_y": f"{pinch_axis[1]:.9g}",
                    "pinch_axis_z": f"{pinch_axis[2]:.9g}",
                    "relative_rotation_angle_deg": f"{relative_angle:.9g}",
                    "long_axis_vertical_error_deg": f"{vertical_error:.9g}",
                    "relative_axis_x": f"{relative_axis[0]:.9g}",
                    "relative_axis_y": f"{relative_axis[1]:.9g}",
                    "relative_axis_z": f"{relative_axis[2]:.9g}",
                    "contact_count": len(phone_contacts),
                    "grasp_point_x": f"{grasp_point[0]:.9g}",
                    "grasp_point_y": f"{grasp_point[1]:.9g}",
                    "grasp_point_z": f"{grasp_point[2]:.9g}",
                    "grasp_point_com_offset_x": f"{grasp_offset[0]:.9g}",
                    "grasp_point_com_offset_y": f"{grasp_offset[1]:.9g}",
                    "grasp_point_com_offset_z": f"{grasp_offset[2]:.9g}",
                    "grasp_point_com_offset_norm": f"{float(np.linalg.norm(grasp_offset)):.9g}",
                    "phone_lift_m": f"{phone_pose[2] - initial_phone_pose[2]:.9g}",
                    "grasp_state": "|".join(sorted(set(phone_grasp_states))) or "NONE",
                }
            )
            if effort_overlay_label is not None:
                x, y, z, w = q
                rotation = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
                ])
                support_z = float(
                    np.abs(rotation[2])
                    @ (np.array([0.1496, 0.00795, 0.0715]) / 2.0)
                )
                clearance = float(phone_com[2] - 0.795 - support_z)
                phone_force_z = sum(
                    float(-(c.normal_force * c.normal[2] + c.friction_force[2]))
                    for c in contacts
                    if "follower_left_" in c.sensor_prim and "Phone" in c.other_prim
                )
                effort_overlay_label.text = (
                    f"target: {left_target:.6f} m\n"
                    f"actual: {left_actual:.6f} m\n"
                    f"error: {left_target-left_actual:.6f} m\n"
                    f"commanded effort sum: {left_commanded:.3f} N\n"
                    f"applied effort sum: {left_effort:.3f} N\n"
                    f"limit: {gripper_effort_limit:.3f} N / carriage\n"
                    f"saturated: {left_saturated}\n"
                    f"gripper force Z on phone: {phone_force_z:.3f} N\n"
                    f"assembly weight: 2.129 N\n"
                    f"true table clearance: {clearance*1000:.3f} mm"
                )
        sim_time += dt
        if frame != last_print and frame % 15 == 0:
            error = torch.max(torch.abs(robot.data.joint_pos.torch - pos)).item()
            speed_now = torch.max(torch.abs(robot.data.joint_vel.torch)).item()
            effort = torch.max(torch.abs(robot.data.applied_torque.torch)).item()
            print(
                f"[ALOHA] frame={frame}/{len(state_trajectory)-1} arm_source={args.arm_source} "
                f"gripper_source={gripper_source} mode={mode} "
                f"tracking_max_rad={error:.6g} velocity_max={speed_now:.6g} effort_max={effort:.6g}"
            )
            print(
                f"[GRIPPER_RUNTIME] frame={frame} applied_effort_max_N={effort:.9g} "
                f"effort_cap_N_per_carriage={gripper_effort_limit:.9g} "
                f"accessory_mass_kg={mass_property_report['accessory_mass_kg']:.9g}",
                flush=True,
            )
            if phone is not None:
                pd = torch.linalg.vector_norm(phone.data.root_pos_w.torch - initial_phone).item()
                ad = torch.linalg.vector_norm(accessory.data.root_pos_w.torch - initial_accessory).item()
                print(f"[OBJECT] phone_displacement_m={pd:.6g} accessory_displacement_m={ad:.6g}")
            last_print = frame
        if frame == end_frame:
            if args.loop:
                sim_time = 0.0
            else:
                if args.headless:
                    break
                sim.pause()
                while simulation_app.is_running():
                    simulation_app.update()
                break
    if contact_monitor is not None:
        contact_monitor.close()
    if phone_table_contact_monitor is not None:
        phone_table_contact_monitor.close()
    if shape_contact_logger is not None:
        print(
            f"[SHAPE_CONTACT] api_error={shape_contact_logger.api_error or 'NONE'}",
            flush=True,
        )
        shape_contact_logger.close()
    if grasp_detector is not None:
        grasp_detector.close()
    if magnet_logger is not None:
        magnet_logger.close()
    if gripper_log_file is not None:
        gripper_log_file.flush()
        gripper_log_file.close()
    if phone_motion_file is not None:
        phone_motion_file.flush()
        phone_motion_file.close()
    if attachment_file is not None:
        attachment_file.flush()
        attachment_file.close()
    if ring_audit_file is not None:
        ring_audit_file.flush()
        ring_audit_file.close()
    if pose_audit_file is not None:
        pose_audit_file.flush()
        pose_audit_file.close()
    ring_result = None
    if ring_controller is not None:
        left_hold = (
            phone is not None
            and float(torch.linalg.vector_norm(
                phone.data.root_pos_w.torch - initial_phone
            ).item()) < 0.08
        )
        ring_result = ring_controller.close(left_hold)
        print(f"[RING_AFFORDANCE_SUMMARY] {ring_result}", flush=True)
    if mode != "kinematic":
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        phase_lines = [f"{frame},{phase}" for frame, phase in phase_detector.transitions]
        phase_report_path = REPORT_ROOT / (
            f"aloha_task_phase_action_gripper_{friction_tag}_report.txt"
            if gripper_source == "action"
            else f"aloha_task_phase_observation_gripper_{friction_tag}_report.txt"
        )
        phase_report_path.write_text(
            "frame,phase\n" + "\n".join(phase_lines) + "\n", encoding="utf-8"
        )
        phone_distance = float(
            torch.linalg.vector_norm(phone.data.root_pos_w.torch - initial_phone).item()
        )
        accessory_distance = float(
            torch.linalg.vector_norm(accessory.data.root_pos_w.torch - initial_accessory).item()
        )
        first_failure = next(
            (
                name
                for name, present in (
                    ("ACCESSORY_CONTACT", "Accessory" in first_contact),
                    (
                        "ACCESSORY_GRASP",
                        any(
                            key.endswith(":Accessory:STABLE_GRASP")
                            for key in grasp_detector.events
                        ),
                    ),
                    ("ACCESSORY_DETACH", accessory_detach_frame is not None),
                    ("PHONE_CONTACT", "Phone" in first_contact),
                    (
                        "PHONE_GRASP",
                        any(key.endswith(":Phone:STABLE_GRASP") for key in grasp_detector.events),
                    ),
                    ("MAGNETIC_CAPTURE", magnetic_attach_frame is not None),
                )
                if not present
            ),
            "NONE",
        )
        contact_pass = bool(contact_monitor.contact_count) and contact_monitor.max_penetration <= 0.001 and frame == end_frame
        physical_pass = contact_pass and first_failure == "NONE"
        failure_classification = (
            "OBJECT_INITIAL_POSE_MISMATCH"
            if first_failure == "ACCESSORY_CONTACT" and "Phone" in first_contact
            else "NONE" if physical_pass else "TRAJECTORY_DOES_NOT_EXECUTE_TASK_PHYSICALLY"
        )
        physical_report_path = REPORT_ROOT / (
            f"aloha_physical_task_action_gripper_{friction_tag}_report.txt"
            if gripper_source == "action"
            else f"aloha_physical_task_observation_gripper_{friction_tag}_report.txt"
        )
        physical_report_path.write_text(
            "\n".join(
                [
                    f"mode={mode}",
                    "joint_replay=position_drive",
                    f"trajectory_completed={frame == end_frame}",
                    f"first_accessory_contact_frame={first_contact.get('Accessory', 'NONE')}",
                    f"first_phone_contact_frame={first_contact.get('Phone', 'NONE')}",
                    f"accessory_detach_frame={accessory_detach_frame if accessory_detach_frame is not None else 'NONE'}",
                    f"magnetic_attach_frame={magnetic_attach_frame if magnetic_attach_frame is not None else 'NONE'}",
                    f"maximum_penetration_m={contact_monitor.max_penetration:.9g}",
                    f"maximum_normal_force_n={contact_monitor.max_normal_force:.9g}",
                    f"maximum_grasp_slip_m={max_slip:.9g}",
                    f"phone_movement_m={phone_distance:.9g}",
                    f"accessory_movement_m={accessory_distance:.9g}",
                    f"contact_api_error={contact_monitor.api_error or 'NONE'}",
                    f"tangential_force_method={contact_monitor.tangential_force_method}",
                    f"FIRST_FAILURE_PHASE={first_failure}",
                    f"failure_classification={failure_classification}",
                    f"ALOHA_CONTACT_PASS={contact_pass}",
                    f"ALOHA_PHYSICAL_TASK_PASS={physical_pass}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"[PHYSICAL_SUMMARY] accessory_contact={first_contact.get('Accessory')} "
            f"phone_contact={first_contact.get('Phone')} detach={accessory_detach_frame} "
            f"magnetic_attach={magnetic_attach_frame} penetration_max_m={contact_monitor.max_penetration:.6g} "
            f"force_max_n={contact_monitor.max_normal_force:.6g} slip_max_m={max_slip:.6g} "
            f"FIRST_FAILURE_PHASE={first_failure}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
