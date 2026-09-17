#!/usr/bin/env python3
"""Policy-free bilateral calibration for the graspable doll proxy v2.

The runner creates one triaxial convex ellipsoid in the anonymous USD session
layer, reuses the already validated fixed arm placement/lift path, and changes
only a predeclared mirrored Dex3 OPEN/PRESHAPE/POWER_GRASP command.  It has no
dataset, policy, checkpoint, DDS, or real-robot interface.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--side", choices=("left", "right"), required=True)
parser.add_argument("--geometry", required=True)
parser.add_argument("--profile", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--scripted-command-path",
    type=Path,
    help="Replay an externally built policy-free 28D command instead of constructing a single-hand trial.",
)
parser.add_argument(
    "--object-spawn-side",
    choices=("left", "right"),
    help="Object reset location for an external scripted command; defaults to --side.",
)
parser.add_argument(
    "--object-registration-config",
    type=Path,
    help=(
        "Optional common, method-independent task-frame registration for the doll. "
        "This changes only the initial doll pose and is intended for evaluator-alignment "
        "preflight; the base physics/grasp configuration remains unchanged."
    ),
)
parser.add_argument(
    "--episode-registration-manifest",
    type=Path,
    help="Source-derived EVAL35 per-episode doll registration manifest.",
)
parser.add_argument(
    "--episode-stable-id",
    help="Stable source episode ID to select from --episode-registration-manifest.",
)
parser.add_argument(
    "--restore-keyframe",
    type=Path,
    help=(
        "Debug-only restoration of a verified robot/object physics keyframe. "
        "Never permitted for final continuous validation."
    ),
)
parser.add_argument(
    "--audit-robot-bin",
    action="store_true",
    help=(
        "Record exact robot-link versus each bin-collider contacts while preserving "
        "the normal command/physics execution unchanged."
    ),
)
parser.add_argument(
    "--full-task-audit",
    action="store_true",
    help=(
        "Record bilateral hand-object and doll-bin diagnostics for a complete "
        "scripted task. This adds read-only contact sensors and does not change "
        "the command, controller, or physics settings."
    ),
)
parser.add_argument(
    "--bin-height-m",
    type=float,
    help=(
        "Session-layer-only external bin height. The bottom, XY footprint, opening, "
        "wall thickness, material, and bin XY pose remain unchanged; the four wall "
        "Cube prims are resized and bottom-aligned in both visual and collision geometry."
    ),
)
parser.add_argument(
    "--bin-rim-bevel-m",
    type=float,
    default=0.0,
    help=(
        "Optional documented top-inner-edge bevel for the four bin walls. "
        "Permitted only after a rim-corner snag has been confirmed."
    ),
)
parser.add_argument(
    "--dex3-hard-limit-contract",
    type=Path,
    help=(
        "Optional authoritative 28-D joint contract. When supplied, all 14 Dex3 "
        "PhysX revolute-joint stops are narrowed inside the authoritative measured "
        "limits in the anonymous session layer before the articulation is created."
    ),
)
parser.add_argument(
    "--dex3-hard-limit-inset-rad",
    type=float,
    default=0.0,
    help="Common symmetric safety inset used with --dex3-hard-limit-contract.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import omni.usd
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.evaluation.contracts import authoritative_joint_ranges
from tools.policy_b_isaac_control_contract import CONTROLLER_CONTRACT


DOLL = "/World/DollHandoffEnvironment/Doll"
SOURCE_BODY = f"{DOLL}/Body"
PROXY = f"{DOLL}/MeasuredProxyV3Collider"
VISUAL = f"{DOLL}/MeasuredProxyV3Visual"
G1 = "/World/G1/Asset"
TABLE = "/World/DollHandoffEnvironment/Table/Colliders/Top"
BIN_PARTS = {
    name: f"/World/DollHandoffEnvironment/TrashBin/{name}"
    for name in ("Bottom", "FrontWall", "BackWall", "LeftWall", "RightWall")
}


def _define_beveled_wall_mesh(
    stage: Usd.Stage,
    source_prim: Usd.Prim,
    name: str,
    external_height_m: float,
    bevel_m: float,
) -> str:
    """Replace one sharp Cube wall with a same-footprint convex beveled prism."""
    bottom_z = 0.006
    top_z = float(external_height_m)
    shoulder_z = top_z - bevel_m
    outer_x, outer_y = 0.095, 0.0825
    inner_x, inner_y = 0.089, 0.0765
    # Replace the sharp rectangular cap with one symmetric 3 mm roof.  The
    # authored wall footprint is unchanged through ``shoulder_z``; both the
    # inner and outer top corners are beveled into a single centered crest.
    # This matters because the confirmed snag trace spans both sides of the
    # old horizontal cap rather than only its inner edge.
    if name == "FrontWall":
        polygon = [
            (-outer_y, bottom_z),
            (-inner_y, bottom_z),
            (-inner_y, shoulder_z),
            (-0.5 * (inner_y + outer_y), top_z),
            (-outer_y, shoulder_z),
        ]
        points = [(x, y, z) for x in (-outer_x, outer_x) for y, z in polygon]
    elif name == "BackWall":
        polygon = [
            (inner_y, bottom_z),
            (outer_y, bottom_z),
            (outer_y, shoulder_z),
            (0.5 * (inner_y + outer_y), top_z),
            (inner_y, shoulder_z),
        ]
        points = [(x, y, z) for x in (-outer_x, outer_x) for y, z in polygon]
    elif name == "LeftWall":
        polygon = [
            (-outer_x, bottom_z),
            (-inner_x, bottom_z),
            (-inner_x, shoulder_z),
            (-0.5 * (inner_x + outer_x), top_z),
            (-outer_x, shoulder_z),
        ]
        points = [(x, y, z) for y in (-inner_y, inner_y) for x, z in polygon]
    elif name == "RightWall":
        polygon = [
            (inner_x, bottom_z),
            (outer_x, bottom_z),
            (outer_x, shoulder_z),
            (0.5 * (inner_x + outer_x), top_z),
            (inner_x, shoulder_z),
        ]
        points = [(x, y, z) for y in (-inner_y, inner_y) for x, z in polygon]
    else:
        raise ValueError(name)
    count = len(polygon)
    faces: list[list[int]] = [
        list(reversed(range(count))),
        list(range(count, 2 * count)),
    ]
    for index in range(count):
        following = (index + 1) % count
        faces.append([index, following, count + following, count + index])
    path = f"/World/DollHandoffEnvironment/TrashBin/{name}Beveled"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([value for face in faces for value in face])
    mesh.CreateSubdivisionSchemeAttr("none")
    collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    binding_targets = (
        UsdShade.MaterialBindingAPI(source_prim).GetDirectBindingRel().GetTargets()
    )
    if binding_targets:
        mesh.GetPrim().CreateRelationship("material:binding").SetTargets(binding_targets)
    UsdPhysics.CollisionAPI(source_prim).GetCollisionEnabledAttr().Set(False)
    UsdGeom.Imageable(source_prim).MakeInvisible()
    return path


def apply_bin_height(
    stage: Usd.Stage, external_height_m: float, rim_bevel_m: float = 0.0
) -> dict[str, Any]:
    """Bottom-align all four authored wall cubes at one common external height."""
    original_external_height_m = 0.190
    bottom_thickness_m = 0.006
    minimum_height_m = 0.100
    height = float(external_height_m)
    if not minimum_height_m <= height <= original_external_height_m:
        raise ValueError(
            f"bin height must be within [{minimum_height_m}, {original_external_height_m}] m"
        )
    wall_height = height - bottom_thickness_m
    wall_center_z = bottom_thickness_m + 0.5 * wall_height
    changes: dict[str, Any] = {}
    contact_parts = dict(BIN_PARTS)
    for name in ("FrontWall", "BackWall", "LeftWall", "RightWall"):
        prim = stage.GetPrimAtPath(BIN_PARTS[name])
        if not prim.IsValid() or not prim.IsA(UsdGeom.Cube):
            raise RuntimeError(f"authored bin wall is missing or not a Cube: {BIN_PARTS[name]}")
        xformable = UsdGeom.Xformable(prim)
        scale_op = next(
            (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        translate_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        if scale_op is None or translate_op is None:
            raise RuntimeError(f"authored bin wall lacks scale/translate xform ops: {prim.GetPath()}")
        old_scale = np.asarray(scale_op.Get(), dtype=np.float64)
        old_translate = np.asarray(translate_op.Get(), dtype=np.float64)
        new_scale = old_scale.copy()
        new_translate = old_translate.copy()
        new_scale[2] = wall_height
        new_translate[2] = wall_center_z
        scale_op.Set(Gf.Vec3f(*map(float, new_scale)))
        translate_op.Set(Gf.Vec3d(*map(float, new_translate)))
        changes[name] = {
            "prim": str(prim.GetPath()),
            "old_scale_xyz_m": old_scale,
            "new_scale_xyz_m": new_scale,
            "old_translate_xyz_m": old_translate,
            "new_translate_xyz_m": new_translate,
        }
    bevel = float(rim_bevel_m)
    if bevel:
        if not 0.0 < bevel <= 0.003:
            raise ValueError("bounded rim bevel must be within (0, 0.003] m")
        for name in ("FrontWall", "BackWall", "LeftWall", "RightWall"):
            source_prim = stage.GetPrimAtPath(BIN_PARTS[name])
            contact_parts[name] = _define_beveled_wall_mesh(
                stage, source_prim, name, height, bevel
            )
    return {
        "edit_layer": "anonymous USD session layer",
        "external_height_m": height,
        "rim_world_z_m": 0.795 + height,
        "bottom_thickness_m": bottom_thickness_m,
        "wall_height_m": wall_height,
        "opening_xy_unchanged": True,
        "bin_xy_unchanged": True,
        "bottom_unchanged": True,
        "material_unchanged": True,
        "visual_collision_consistency": (
            "same four beveled convex Mesh prims provide visual and collision geometry"
            if bevel
            else "same four wall Cube prims provide visual and collision geometry"
        ),
        "rim_bevel_m": bevel,
        "rim_bevel_basis": (
            "RIM_CORNER_SNAG_CONFIRMED; symmetric inner/outer top-edge bevel; "
            "same convex mesh is visual and collision geometry"
            if bevel
            else None
        ),
        "contact_parts": contact_parts,
        "changes": changes,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direct_freeze_pins_exactly(*paths: Path) -> bool:
    """Return true only when the active final freeze pins every supplied file.

    The source-derived registration manifest predates the common articulation
    solver correction and consequently records the hash of the predecessor
    all-in-one physics config.  Registration itself is immutable.  For a final
    run, the active freeze is the authoritative binding between that unchanged
    registration artifact and the corrected current physics config.
    """
    freeze_value = os.environ.get("DIRECT_EVAL35_FREEZE_MANIFEST")
    if not freeze_value:
        return False
    freeze_path = Path(freeze_value).resolve()
    if not freeze_path.is_file():
        return False
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN_BEFORE_EVAL35":
        return False
    pinned = {
        Path(row["path"]).resolve(): row.get("sha256")
        for row in freeze.get("files", [])
    }
    return all(
        path.resolve() in pinned
        and pinned[path.resolve()] == sha256_file(path.resolve())
        for path in paths
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def minimum_jerk(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, max(2, int(count)), dtype=np.float64)
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start[None] + blend[:, None] * (end - start)[None]


def longest_duration(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def rounded_oval_mesh(dimensions: np.ndarray) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    """Convex ellipsoidal oval with a small, explicit table-support patch.

    A pure triangulated ellipsoid rests on its single south-pole vertex.  For a
    20 g free rigid body that makes the reset pose numerically prone to rolling
    before the hand arrives.  Making only the lowest latitude ring coplanar
    with the exact lower Z extent creates a small flat support patch while
    retaining a single convex, rounded oval and the declared outer dimensions.
    """
    radii = np.asarray(dimensions, dtype=np.float64) / 2.0
    latitudes, longitudes = 12, 24
    points: list[np.ndarray] = [np.asarray([0.0, 0.0, radii[2]])]
    for latitude in range(1, latitudes):
        theta = math.pi * latitude / latitudes
        for longitude in range(longitudes):
            phi = 2.0 * math.pi * longitude / longitudes
            z = radii[2] * math.cos(theta)
            if latitude >= latitudes - 2:
                z = -radii[2]
            points.append(
                np.asarray(
                    [
                        radii[0] * math.sin(theta) * math.cos(phi),
                        radii[1] * math.sin(theta) * math.sin(phi),
                        z,
                    ]
                )
            )
    points.append(np.asarray([0.0, 0.0, -radii[2]]))
    north, south = 0, len(points) - 1
    faces: list[list[int]] = []
    for longitude in range(longitudes):
        faces.append([north, 1 + longitude, 1 + (longitude + 1) % longitudes])
    for latitude in range(latitudes - 2):
        first = 1 + latitude * longitudes
        second = first + longitudes
        for longitude in range(longitudes):
            following = (longitude + 1) % longitudes
            faces.append(
                [first + longitude, second + longitude, second + following, first + following]
            )
    last = 1 + (latitudes - 2) * longitudes
    for longitude in range(longitudes):
        faces.append([last + longitude, south, last + (longitude + 1) % longitudes])
    return (
        [Gf.Vec3f(*map(float, point)) for point in points],
        [len(face) for face in faces],
        [index for face in faces for index in face],
    )


def apply_proxy(
    stage: Usd.Stage, config: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    source = stage.GetPrimAtPath(SOURCE_BODY)
    if not source.IsValid():
        raise RuntimeError("source doll collider is missing")
    UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Set(False)
    for child in stage.GetPrimAtPath(DOLL).GetChildren():
        if child.GetPath() in {Sdf.Path(PROXY), Sdf.Path(VISUAL)}:
            continue
        if child.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(child).MakeInvisible()
    visual_dimensions = np.asarray(config["object"]["visual_dimensions_m"], dtype=np.float64)
    collision_dimensions = np.asarray(geometry["dimensions_m"], dtype=np.float64)

    visual_mesh = UsdGeom.Mesh.Define(stage, VISUAL)
    visual_points, visual_counts, visual_indices = rounded_oval_mesh(visual_dimensions)
    visual_mesh.CreatePointsAttr(visual_points)
    visual_mesh.CreateFaceVertexCountsAttr(visual_counts)
    visual_mesh.CreateFaceVertexIndicesAttr(visual_indices)
    visual_mesh.CreateSubdivisionSchemeAttr("none")
    visual_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.72, 0.34)])

    mesh = UsdGeom.Mesh.Define(stage, PROXY)
    points, counts, indices = rounded_oval_mesh(collision_dimensions)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    # Preserve the measured visible extent while bottom-aligning the uniformly
    # inset collider.  This approximates compression without a hidden fixture
    # and prevents a smaller collider from floating above the table.
    collider_z_offset = float((collision_dimensions[2] - visual_dimensions[2]) / 2.0)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, collider_z_offset))
    UsdGeom.Imageable(mesh).MakeInvisible()
    prim = mesh.GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateContactOffsetAttr(float(config["object"]["contact_offset_m"]))
    physx_collision.CreateRestOffsetAttr(float(config["object"]["rest_offset_m"]))
    material = UsdShade.Material.Define(stage, "/World/DollGraspableProxyV2Material")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_spec = copy.deepcopy(config["material"])
    material_spec.update(geometry.get("material_override", {}))
    material_api.CreateStaticFrictionAttr(float(material_spec["static_friction"]))
    material_api.CreateDynamicFrictionAttr(float(material_spec["dynamic_friction"]))
    material_api.CreateRestitutionAttr(float(config["object"]["restitution"]))
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(
        str(material_spec["friction_combine_mode"])
    )
    physx_material.CreateRestitutionCombineModeAttr().Set(
        str(material_spec["restitution_combine_mode"])
    )
    compliant_stiffness = material_spec.get("compliant_contact_stiffness_n_m")
    compliant_damping = material_spec.get("compliant_contact_damping_n_s_m")
    if compliant_stiffness is not None:
        physx_material.CreateCompliantContactStiffnessAttr().Set(
            float(compliant_stiffness)
        )
    if compliant_damping is not None:
        physx_material.CreateCompliantContactDampingAttr().Set(float(compliant_damping))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    root = stage.GetPrimAtPath(DOLL)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(config["object"]["mass_kg"]))
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rigid.CreateLinearDampingAttr(float(config["object"]["linear_damping"]))
    rigid.CreateAngularDampingAttr(float(config["object"]["angular_damping"]))
    rigid.CreateMaxDepenetrationVelocityAttr(
        float(config["object"]["max_depenetration_velocity_m_s"])
    )
    return {
        "source_collider_disabled": True,
        "proxy_prim": PROXY,
        "proxy_type": prim.GetTypeName(),
        "convex_approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
        "visual_prim": VISUAL,
        "collision_prim": PROXY,
        "visual_dimensions_m": visual_dimensions,
        "collision_dimensions_m": collision_dimensions,
        "collision_scale_percent": int(geometry["collision_scale_percent"]),
        "collider_bottom_alignment_offset_z_m": collider_z_offset,
        "visual_and_collision_same_prim": False,
        "local_transform": np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()),
        "mass_kg": float(UsdPhysics.MassAPI(root).GetMassAttr().Get()),
        "static_friction": float(material_api.GetStaticFrictionAttr().Get()),
        "dynamic_friction": float(material_api.GetDynamicFrictionAttr().Get()),
        "restitution": float(material_api.GetRestitutionAttr().Get()),
        "material": material_spec,
        "compliant_contact_stiffness_n_m": (
            float(physx_material.GetCompliantContactStiffnessAttr().Get())
            if compliant_stiffness is not None
            else None
        ),
        "compliant_contact_damping_n_s_m": (
            float(physx_material.GetCompliantContactDampingAttr().Get())
            if compliant_damping is not None
            else None
        ),
        "contact_offset_m": float(physx_collision.GetContactOffsetAttr().Get()),
        "rest_offset_m": float(physx_collision.GetRestOffsetAttr().Get()),
        "session_layer_only": True,
        "source_scene_saved_or_modified": False,
    }


def build_actuators(config: dict[str, Any]) -> dict[str, Any]:
    values = copy.deepcopy(CONTROLLER_CONTRACT["actuators"])
    drive = config["finger_drive"]
    values["dex3"].update(
        {
            "effort_limit_sim": float(drive["effort_limit_sim"]),
            "velocity_limit_sim": float(drive["velocity_limit_sim"]),
            "stiffness": float(drive["kp"]),
            "damping": float(drive["kd"]),
        }
    )
    return {name: ImplicitActuatorCfg(**spec) for name, spec in values.items()}


def apply_authoritative_dex3_joint_stops(
    stage: Usd.Stage, contract_path: Path | None, inset_rad: float
) -> dict[str, Any]:
    """Narrow simulator stops inside measured Dex3 limits without editing source USD."""

    if contract_path is None:
        if not np.isclose(inset_rad, 0.0):
            raise RuntimeError("Dex3 limit inset requires an authoritative contract")
        return {"enabled": False}
    contract_path = contract_path.resolve()
    contract = read_json(contract_path)
    names = [str(name) for name in contract.get("joint_names", [])]
    specs = contract.get("joint_specs", [])
    if len(names) != 28 or len(specs) != 28:
        raise RuntimeError("authoritative joint contract is malformed")
    by_index = {int(row["index"]): row for row in specs}
    expected = names[14:28]
    for index, name in zip(range(14, 28), expected, strict=True):
        if str(by_index[index].get("joint_name")) != name:
            raise RuntimeError("authoritative Dex3 ordering mismatch")
    inset = float(inset_rad)
    if not np.isfinite(inset) or inset < 0.0:
        raise RuntimeError("invalid Dex3 joint-stop inset")
    prim_by_name: dict[str, Usd.Prim] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in expected:
            if prim.GetName() in prim_by_name:
                raise RuntimeError(f"duplicate Dex3 joint prim: {prim.GetName()}")
            prim_by_name[prim.GetName()] = prim
    missing = sorted(set(expected) - set(prim_by_name))
    if missing:
        raise RuntimeError(f"Dex3 joint prims missing from stage: {missing}")
    rows: list[dict[str, Any]] = []
    for index, name in zip(range(14, 28), expected, strict=True):
        row = by_index[index]
        joint = UsdPhysics.RevoluteJoint(prim_by_name[name])
        authored_lower_rad = math.radians(float(joint.GetLowerLimitAttr().Get()))
        authored_upper_rad = math.radians(float(joint.GetUpperLimitAttr().Get()))
        authoritative_lower_rad = float(row["minimum"])
        authoritative_upper_rad = float(row["maximum"])
        effective_lower_rad = max(authored_lower_rad, authoritative_lower_rad) + inset
        effective_upper_rad = min(authored_upper_rad, authoritative_upper_rad) - inset
        if effective_lower_rad >= effective_upper_rad:
            raise RuntimeError(f"Dex3 inset collapses joint interval: {name}")
        joint.GetLowerLimitAttr().Set(math.degrees(effective_lower_rad))
        joint.GetUpperLimitAttr().Set(math.degrees(effective_upper_rad))
        rows.append(
            {
                "index_28d": index,
                "joint_name": name,
                "authoritative_lower_rad": authoritative_lower_rad,
                "authoritative_upper_rad": authoritative_upper_rad,
                "source_usd_lower_rad": authored_lower_rad,
                "source_usd_upper_rad": authored_upper_rad,
                "runtime_lower_rad": effective_lower_rad,
                "runtime_upper_rad": effective_upper_rad,
                "articulation_default_rad": float(
                    0.5 * (effective_lower_rad + effective_upper_rad)
                ),
                "session_layer_only": True,
            }
        )
    return {
        "enabled": True,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "units": "radian",
        "common_symmetric_inset_rad": inset,
        "source_usd_saved_or_modified": False,
        "joint_order": expected,
        "joints": rows,
    }


def apply_authoritative_articulation_solver(
    stage: Usd.Stage, config: dict[str, Any]
) -> dict[str, Any]:
    """Apply the common contact-robust G1 solver contract in the session layer."""

    values = config.get("articulation_solver")
    if not isinstance(values, dict):
        raise RuntimeError("common articulation solver contract is missing")
    position = int(values["position_iterations"])
    velocity = int(values["velocity_iterations"])
    if position < 1 or velocity < 0:
        raise RuntimeError("invalid articulation solver iteration contract")
    prim = stage.GetPrimAtPath(f"{G1}/root_joint")
    if not prim.IsValid():
        raise RuntimeError("G1 articulation prim is missing")
    position_attr = prim.GetAttribute(
        "physxArticulation:solverPositionIterationCount"
    )
    velocity_attr = prim.GetAttribute(
        "physxArticulation:solverVelocityIterationCount"
    )
    source_position = int(position_attr.Get())
    source_velocity = int(velocity_attr.Get())
    position_attr.Set(position)
    velocity_attr.Set(velocity)
    return {
        "prim": str(prim.GetPath()),
        "source_position_iterations": source_position,
        "source_velocity_iterations": source_velocity,
        "runtime_position_iterations": int(position_attr.Get()),
        "runtime_velocity_iterations": int(velocity_attr.Get()),
        "scope": "common G1 articulation; identical ACT-A/ACT-B",
        "selection_basis": values["selection_basis"],
        "session_layer_only": True,
        "source_scene_saved_or_modified": False,
    }


def build_commands(
    config: dict[str, Any], side: str, profile: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    primitive_path = Path(config["source_arm_primitives"][side]).resolve()
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    commands = np.asarray(primitive["commanded_q_rad"], dtype=np.float64)
    if not {
        "open_arm_q_rad",
        "approach_arm_q_rad",
        "lift_arm_q_rad",
    }.issubset(primitive):
        raise RuntimeError("v2 arm primitive lacks the fixed power-approach contract")
    approach_arms = np.asarray(primitive["approach_arm_q_rad"], dtype=np.float64)
    lift_path = np.asarray(primitive["lift_arm_q_rad"], dtype=np.float64)
    base_arms = approach_arms[-1]
    # The primitive builder provides a mirrored, policy-independent outward
    # pregrasp -> centered-palm approach.  Preserve it during PRESHAPE so the
    # OPEN hand never has to spawn inside the object.  The arm remains fixed at
    # the declared centered placement throughout POWER_GRASP and retention.
    open_arms = approach_arms[0].copy()
    exact_q5_arm = lift_path[-1]
    lift_arms = lift_path[1:]
    states = config["hand_states"]
    active_slice = slice(14, 21) if side == "left" else slice(21, 28)
    inactive_slice = slice(21, 28) if side == "left" else slice(14, 21)
    inactive_side = "right" if side == "left" else "left"
    open_q = np.asarray(states[side]["OPEN"], dtype=np.float64)
    preshape_key = f"PRESHAPE_{profile}"
    preshape_value = (
        states[side][preshape_key]
        if preshape_key in states[side]
        else states[side]["PRESHAPE"]
    )
    preshape_q = np.asarray(preshape_value, dtype=np.float64)
    power_q = np.asarray(states[side][f"POWER_GRASP_{profile}"], dtype=np.float64)
    inactive_open = np.asarray(states[inactive_side]["OPEN"], dtype=np.float64)
    fps = float(config["timing"]["control_fps_hz"])

    rows: list[np.ndarray] = []
    labels: list[str] = []

    def append(arms: np.ndarray, hand: np.ndarray, label: str) -> None:
        row = np.zeros(28, dtype=np.float64)
        row[:14] = arms
        row[active_slice] = hand
        row[inactive_slice] = inactive_open
        rows.append(row)
        labels.append(label)

    for _ in range(max(1, round(config["timing"]["open_hold_s"] * fps))):
        append(open_arms, open_q, "OPEN")
    preshape_hands = minimum_jerk(
        open_q, preshape_q, round(config["timing"]["preshape_transition_s"] * fps)
    )
    preshape_arms = approach_arms
    if len(preshape_arms) != len(preshape_hands):
        source_u = np.linspace(0.0, 1.0, len(preshape_arms))
        target_u = np.linspace(0.0, 1.0, len(preshape_hands))
        preshape_arms = np.column_stack(
            [np.interp(target_u, source_u, preshape_arms[:, joint]) for joint in range(14)]
        )
    for arm, hand in zip(preshape_arms[1:], preshape_hands[1:], strict=True):
        append(arm, hand, "PRESHAPE")
    power_hands = minimum_jerk(
        preshape_q, power_q, round(config["timing"]["power_close_s"] * fps)
    )
    for hand in power_hands[1:]:
        append(base_arms, hand, "POWER_GRASP")
    for _ in range(max(1, round(config["timing"]["gravity_retention_s"] * fps))):
        append(base_arms, power_q, "GRAVITY_RETENTION")
    for arm in lift_arms:
        append(arm, power_q, "LIFT_5CM")
    for _ in range(max(1, round(config["timing"]["elevated_hold_s"] * fps))):
        append(exact_q5_arm, power_q, "HOLD_ELEVATED")
    if not bool(config["timing"].get("end_after_elevated_hold", False)):
        for arm in lift_arms[-2::-1]:
            append(arm, power_q, "LOWER")
        append(base_arms, power_q, "LOWER")
        for hand in minimum_jerk(
            power_q, open_q, round(config["timing"]["release_transition_s"] * fps)
        )[1:]:
            append(base_arms, hand, "RELEASE")
        for _ in range(max(1, round(config["timing"]["post_release_s"] * fps))):
            append(base_arms, open_q, "POST_RELEASE")
    return np.asarray(rows), np.asarray(labels), power_q, primitive_path


def filtered_force(sensor: ContactSensor) -> float:
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        return 0.0
    values = numpy(matrix)
    return float(np.max(np.linalg.norm(values.reshape(-1, 3), axis=-1))) if values.size else 0.0


def contact_rows(
    sensor: ContactSensor, dt: float
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        forces, points, normals, separations, counts, starts = sensor.contact_view.get_contact_data(dt)
        forces_np = numpy(forces).reshape(-1)
        points_np = numpy(points).reshape(-1, 3)
        normals_np = numpy(normals).reshape(-1, 3)
        separations_np = numpy(separations).reshape(-1)
        owners = list(sensor.body_physx_view.prim_paths[: sensor.num_sensors])
        counts_np = numpy(counts).reshape(len(owners), -1).astype(np.int64)
        starts_np = numpy(starts).reshape(len(owners), -1).astype(np.int64)
        rows: list[dict[str, Any]] = []
        for owner_index in range(len(owners)):
            for filter_index in range(counts_np.shape[1]):
                start, count = starts_np[owner_index, filter_index], counts_np[owner_index, filter_index]
                for index in range(int(start), int(start + count)):
                    rows.append(
                        {
                            "owner": owners[owner_index],
                            "force_n": float(abs(forces_np[index])),
                            "point": points_np[index].astype(np.float64),
                            "normal": normals_np[index].astype(np.float64),
                            "separation_m": float(separations_np[index]),
                        }
                    )
        return rows, None
    except Exception as error:
        return [], f"{type(error).__name__}: {error}"


def main() -> int:
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    object_registration_path: Path | None = None
    object_registration: dict[str, Any] | None = None
    episode_registration_entry: dict[str, Any] | None = None
    episode_registration_binding = "NOT_APPLICABLE"
    if (args.episode_registration_manifest is None) != (args.episode_stable_id is None):
        raise RuntimeError("episode registration manifest and stable ID are required together")
    if args.object_registration_config is not None and args.episode_registration_manifest is not None:
        raise RuntimeError("global and episode-conditioned object registration are mutually exclusive")
    if args.object_registration_config is not None:
        object_registration_path = args.object_registration_config.resolve()
        object_registration = read_json(object_registration_path)
        if object_registration.get("schema_version") != "common_task_frame_registration_v1":
            raise RuntimeError("unexpected object-registration schema")
        if not bool(object_registration.get("method_independent", False)):
            raise RuntimeError("object registration must be method-independent")
        if not bool(object_registration.get("episode_independent", False)):
            raise RuntimeError("object registration must be episode-independent")
        if bool(object_registration.get("changes_object_physics", True)):
            raise RuntimeError("object registration must not change object physics")
        declared_base = Path(object_registration["base_physics_config"]).resolve()
        if declared_base != config_path:
            raise RuntimeError("object registration base-config identity mismatch")
        if sha256_file(config_path) != object_registration["base_physics_config_sha256"]:
            raise RuntimeError("object registration base-config hash mismatch")
    elif args.episode_registration_manifest is not None:
        object_registration_path = args.episode_registration_manifest.resolve()
        manifest = read_json(object_registration_path)
        if manifest.get("schema_version") != "eval35_episode_source_derived_object_registration_v1":
            raise RuntimeError("unexpected episode-registration schema")
        if manifest.get("status") != "PASS" or manifest.get("physical_eval35_outcomes_read") is not False:
            raise RuntimeError("episode registration is not source-only qualified")
        if manifest.get("one_global_canonical_object_pose") is not False:
            raise RuntimeError("episode-conditioned runner refuses a global canonical pose")
        matches = [
            row for row in manifest.get("entries", [])
            if str(row.get("stable_episode_id")) == str(args.episode_stable_id)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"episode registration lookup is not unique: {args.episode_stable_id}")
        episode_registration_entry = matches[0]
        if episode_registration_entry.get("A_B_identical_object_pose") is not True:
            raise RuntimeError("episode registration is not identical for matched A/B")
        if episode_registration_entry.get("runtime_initialization_rule", {}).get("bin_pose_fixed") is not True:
            raise RuntimeError("episode registration attempts to move the frozen bin")
        config_hash = manifest.get("authoritative_inputs", {}).get(str(config_path))
        if config_hash != sha256_file(config_path):
            if not direct_freeze_pins_exactly(config_path, object_registration_path):
                raise RuntimeError("episode registration base physics config drift")
            episode_registration_binding = (
                "ACTIVE_FINAL_FREEZE_PINS_UNCHANGED_REGISTRATION_AND_CURRENT_PHYSICS_CONFIG"
            )
        else:
            episode_registration_binding = "ORIGINAL_REGISTRATION_CONFIG_HASH_MATCH"
        pose = episode_registration_entry["target_object_pose"]
        object_registration = {
            "registered_doll_center_world_xy_m": pose["position_xyz_m"][:2],
            "registered_doll_center_world_z_m": pose["position_xyz_m"][2],
            "registered_doll_orientation_quaternion_xyzw": pose["quaternion_xyzw"],
            "method_independent": True,
            "episode_independent": False,
            "changes_object_physics": False,
        }
    schema = config.get("schema_version")
    if schema not in {
        "doll_handoff_graspable_proxy_v2",
        "doll_handoff_measured_proxy_v3",
        "dex3_measured_doll_graspability_sanity_v1",
        "dex3_simple_graspable_doll_proxy_v1",
    }:
        raise RuntimeError("unexpected graspable-proxy config")
    if config.get("learned_policy_used_for_calibration") or config.get("real_robot_allowed"):
        raise RuntimeError("calibration must remain policy-free and simulation-only")
    geometries = {row["name"]: row for row in config["geometry_candidates"]}
    if args.geometry not in geometries:
        raise RuntimeError("geometry was not predeclared")
    names, _ = authoritative_joint_ranges()
    geometry = geometries[args.geometry]
    runtime_gate_required = False
    runtime_gate_minimum_s = 0.5
    restored_keyframe: dict[str, np.ndarray] | None = None
    restored_keyframe_path: Path | None = None
    if args.restore_keyframe is not None:
        if args.scripted_command_path is None:
            raise RuntimeError("keyframe restoration requires an external tail command")
        restored_keyframe_path = args.restore_keyframe.resolve()
        with np.load(restored_keyframe_path, allow_pickle=False) as archive:
            restored_keyframe = {
                key: np.asarray(archive[key]) for key in archive.files
            }
        if (
            str(np.asarray(restored_keyframe.get("schema_version", "")).item())
            != "post_handoff_keyframe_v1"
            or not bool(
                np.asarray(
                    restored_keyframe.get("state_restoration_debug_only", False)
                ).item()
            )
            or bool(
                np.asarray(
                    restored_keyframe.get(
                        "permitted_in_final_continuous_validation", True
                    )
                ).item()
            )
        ):
            raise RuntimeError("unverified or final-validation-ineligible keyframe")
    if args.scripted_command_path is not None:
        command_path = args.scripted_command_path.resolve()
        with np.load(command_path, allow_pickle=False) as archive:
            commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
            stages = np.asarray(archive["stage"]).astype(str)
            external_names = archive["joint_names"].astype(str).tolist()
            external_fps = float(np.asarray(archive["control_fps_hz"]).item())
            if "runtime_right_three_digit_gate_required" in archive.files:
                runtime_gate_required = bool(
                    np.asarray(archive["runtime_right_three_digit_gate_required"]).item()
                )
            if "right_three_digit_gate_minimum_s" in archive.files:
                runtime_gate_minimum_s = float(
                    np.asarray(archive["right_three_digit_gate_minimum_s"]).item()
                )
        if external_names != names:
            raise RuntimeError("external scripted command named joint order mismatch")
        if not np.isclose(external_fps, float(config["timing"]["control_fps_hz"])):
            raise RuntimeError("external scripted command control rate mismatch")
        power_q = np.asarray(
            config["hand_states"][args.side][f"POWER_GRASP_{args.profile}"],
            dtype=np.float64,
        )
        primitive_path = Path(config["source_arm_primitives"][args.side]).resolve()
    else:
        commands, stages, power_q, primitive_path = build_commands(
            config, args.side, args.profile
        )
        command_path = output_dir / "scripted_command.npz"
        temporary = command_path.with_suffix(".npz.incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                commanded_q_rad=commands.astype(np.float32),
                stage=stages,
                joint_names=np.asarray(names),
                control_fps_hz=np.asarray(config["timing"]["control_fps_hz"]),
                side=np.asarray(args.side),
                geometry=np.asarray(args.geometry),
                profile=np.asarray(args.profile),
                policy_independent=np.asarray(True),
            )
        os.replace(temporary, command_path)
    with np.load(primitive_path, allow_pickle=False) as source:
        if source["joint_names"].astype(str).tolist() != names:
            raise RuntimeError("source primitive named joint order mismatch")
    if commands.shape[1] != 28 or not np.isfinite(commands).all():
        raise RuntimeError("invalid scripted commands")
    if not omni.usd.get_context().open_stage(config["source_scene"]):
        raise RuntimeError("failed to open source scene")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    runtime_proxy = apply_proxy(stage, config, geometry)
    runtime_bin = (
        apply_bin_height(stage, args.bin_height_m, args.bin_rim_bevel_m)
        if args.bin_height_m is not None
        else {
            "edit_layer": None,
            "external_height_m": 0.190,
            "rim_world_z_m": 0.985,
            "opening_xy_unchanged": True,
            "bin_xy_unchanged": True,
            "bottom_unchanged": True,
            "material_unchanged": True,
            "rim_bevel_m": 0.0,
            "contact_parts": dict(BIN_PARTS),
        }
    )
    runtime_dex3_joint_stops = apply_authoritative_dex3_joint_stops(
        stage,
        args.dex3_hard_limit_contract,
        args.dex3_hard_limit_inset_rad,
    )
    runtime_articulation_solver = apply_authoritative_articulation_solver(stage, config)
    dt = float(config["timing"]["physics_dt_s"])
    substeps = int(config["timing"]["physics_substeps_per_control_frame"])
    fps = float(config["timing"]["control_fps_hz"])
    if not np.isclose(dt * substeps, 1.0 / fps):
        raise RuntimeError("physics/control timing mismatch")
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=str(config["simulation"]["device"]),
            gravity=(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"])),
            use_fabric=bool(config["simulation"]["use_fabric"]),
        )
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path=f"{G1}/root_joint",
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "^(?!.*_hand_).*$": 0.0,
                    **{
                        row["joint_name"]: row["articulation_default_rad"]
                        for row in runtime_dex3_joint_stops.get("joints", [])
                    },
                }
            ),
            actuators=build_actuators(config),
        )
    )
    doll = RigidObject(RigidObjectCfg(prim_path=DOLL, spawn=None))
    whole_hand = read_json(Path(config["whole_hand_geometry"]))
    role_to_digit = {
        role: whole_hand[args.side][role]["digit_chain"] for role in ("A", "B", "C")
    }
    role_to_link = {
        role: whole_hand[args.side][role]["distal_link"] for role in ("A", "B", "C")
    }
    digit_sensor_paths = {
        role: (
            f"{G1}/{args.side}_hand_{role_to_digit[role]}_.*_link"
            if schema in {
                "doll_handoff_measured_proxy_v3",
                "dex3_measured_doll_graspability_sanity_v1",
                "dex3_simple_graspable_doll_proxy_v1",
            }
            else f"{G1}/{link}"
        )
        for role, link in role_to_link.items()
    }
    digit_sensors = {
        role: ContactSensor(
            ContactSensorCfg(
                prim_path=digit_sensor_paths[role],
                update_period=0.0,
                filter_prim_paths_expr=[DOLL],
                track_contact_points=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
        for role in role_to_link
    }
    # The legacy trial result is intentionally keyed to ``--side``.  A final
    # full-task audit additionally observes the opposite hand without changing
    # that contract or instantiating a duplicate sensor for the selected hand.
    opposite_side = "left" if args.side == "right" else "right"
    opposite_role_to_digit = {
        role: whole_hand[opposite_side][role]["digit_chain"]
        for role in ("A", "B", "C")
    }
    opposite_digit_sensors = (
        {
            role: ContactSensor(
                ContactSensorCfg(
                    prim_path=f"{G1}/{opposite_side}_hand_{opposite_role_to_digit[role]}_.*_link",
                    update_period=0.0,
                    filter_prim_paths_expr=[DOLL],
                    track_contact_points=True,
                    max_contact_data_count_per_prim=64,
                    force_threshold=0.0,
                )
            )
            for role in opposite_role_to_digit
        }
        if args.full_task_audit
        else {}
    )
    # Read-only palm diagnostics for the final physical traces.  In the G1 USD
    # the palm collision mesh is owned by the wrist-yaw rigid body, so these
    # sensors observe that body against the doll.  They do not feed the common
    # controller and therefore cannot change grasp, arm, or wrist commands.
    palm_sensors = {
        side: ContactSensor(
            ContactSensorCfg(
                prim_path=f"{G1}/{side}_wrist_yaw_link",
                update_period=0.0,
                filter_prim_paths_expr=[DOLL],
                track_contact_points=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
        for side in ("left", "right")
    }
    table_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=DOLL,
            update_period=0.0,
            filter_prim_paths_expr=[TABLE],
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            force_threshold=0.0,
        )
    )
    bin_sensors = (
        {
            name: ContactSensor(
                ContactSensorCfg(
                    # The bounded bin audit concerns the transporting RIGHT arm.
                    # Restricting ownership to right-side rigid links preserves
                    # all relevant contact pairs and avoids instantiating sensors
                    # on unrelated legs, torso, and LEFT-hand visual descendants.
                    prim_path=f"{G1}/right_.*_link",
                    update_period=0.0,
                    filter_prim_paths_expr=[path],
                    track_contact_points=True,
                    max_contact_data_count_per_prim=64,
                    force_threshold=0.0,
                )
            )
            for name, path in runtime_bin["contact_parts"].items()
        }
        if args.audit_robot_bin
        else {}
    )
    doll_bin_sensors = (
        {
            name: ContactSensor(
                ContactSensorCfg(
                    prim_path=DOLL,
                    update_period=0.0,
                    filter_prim_paths_expr=[path],
                    track_contact_points=True,
                    max_contact_data_count_per_prim=64,
                    force_threshold=0.0,
                )
            )
            for name, path in runtime_bin["contact_parts"].items()
        }
        if args.full_task_audit
        else {}
    )
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    joint_ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(joint_ids) != 28 or len(set(joint_ids)) != 28:
        raise RuntimeError(f"named joint mapping failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    target[0, joint_ids] = torch.as_tensor(commands[0], device=robot.device, dtype=torch.float32)
    initial_position = target.clone()
    initial_velocity = torch.zeros_like(target)
    if restored_keyframe is not None:
        if restored_keyframe["joint_names"].astype(str).tolist() != names:
            raise RuntimeError("restored keyframe named joint order mismatch")
        initial_position[0, joint_ids] = torch.as_tensor(
            restored_keyframe["measured_q_rad"],
            device=robot.device,
            dtype=torch.float32,
        )
        initial_velocity[0, joint_ids] = torch.as_tensor(
            restored_keyframe["measured_qd_rad_s"],
            device=robot.device,
            dtype=torch.float32,
        )
    robot.write_joint_state_to_sim(initial_position, initial_velocity)
    dimensions = np.asarray(geometry["dimensions_m"], dtype=np.float32)
    visual_dimensions = np.asarray(config["object"]["visual_dimensions_m"], dtype=np.float32)
    spawn_side = args.object_spawn_side or args.side
    center_xy = config["object"].get("center_world_xy_m_by_side", {}).get(
        spawn_side, config["object"].get("center_world_xy_m")
    )
    if object_registration is not None:
        center_xy = object_registration["registered_doll_center_world_xy_m"]
    if center_xy is None:
        raise RuntimeError("object center is not declared for the selected side")
    center = np.asarray(
        [
            *center_xy,
            float(config["object"]["table_surface_world_z_m"])
            + float(visual_dimensions[2]) / 2.0
            + float(config["object"]["spawn_clearance_above_table_m"]),
        ],
        dtype=np.float32,
    )
    if (
        object_registration is not None
        and episode_registration_entry is None
        and "registered_doll_center_world_z_m" in object_registration
    ):
        center[2] = float(object_registration["registered_doll_center_world_z_m"])
    if episode_registration_entry is not None:
        center = np.asarray(
            episode_registration_entry["target_object_pose"]["position_xyz_m"],
            dtype=np.float32,
        )
    orientation_value = config["object"].get(
        "orientation_quaternion_xyzw_by_side", {}
    ).get(
        args.side,
        config["object"].get("orientation_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
    )
    orientation_xyzw = np.asarray(orientation_value, dtype=np.float32)
    if object_registration is not None:
        orientation_xyzw = np.asarray(
            object_registration["registered_doll_orientation_quaternion_xyzw"],
            dtype=np.float32,
        )
    root_velocity_value = np.zeros(6, dtype=np.float32)
    if restored_keyframe is not None:
        center = np.asarray(
            restored_keyframe["object_position_world_m"], dtype=np.float32
        )
        orientation_xyzw = np.asarray(
            restored_keyframe["object_quaternion_xyzw"], dtype=np.float32
        )
        root_velocity_value = np.r_[
            restored_keyframe["object_linear_velocity_m_s"],
            restored_keyframe["object_angular_velocity_rad_s"],
        ].astype(np.float32)
    root_pose = torch.as_tensor(
        np.r_[center, orientation_xyzw][None], device=doll.device, dtype=torch.float32
    )
    doll.write_root_pose_to_sim_index(root_pose=root_pose)
    doll.write_root_velocity_to_sim_index(
        root_velocity=torch.as_tensor(
            root_velocity_value[None], device=doll.device, dtype=torch.float32
        )
    )
    actual_initial_pose = numpy(doll.data.root_pose_w)[0].astype(np.float64)
    requested_initial_pose = np.r_[center, orientation_xyzw].astype(np.float64)
    initial_translation_error_mm = 1000.0 * float(
        np.linalg.norm(actual_initial_pose[:3] - requested_initial_pose[:3])
    )
    requested_quaternion = requested_initial_pose[3:7] / np.linalg.norm(
        requested_initial_pose[3:7]
    )
    actual_quaternion = actual_initial_pose[3:7] / np.linalg.norm(actual_initial_pose[3:7])
    initial_rotation_error_deg = float(
        np.degrees(2.0 * np.arccos(np.clip(abs(float(requested_quaternion @ actual_quaternion)), 0.0, 1.0)))
    )
    if initial_translation_error_mm > 0.1 or initial_rotation_error_deg > 0.1:
        raise RuntimeError(
            "runtime doll initialization differs from registered pose: "
            f"{initial_translation_error_mm} mm / {initial_rotation_error_deg} deg"
        )
    runtime_initial_pose_verification = {
        "stable_episode_id": str(args.episode_stable_id) if episode_registration_entry is not None else None,
        "requested_position_xyz_m": requested_initial_pose[:3].tolist(),
        "requested_quaternion_xyzw": requested_initial_pose[3:7].tolist(),
        "actual_position_xyz_m_before_frame_0": actual_initial_pose[:3].tolist(),
        "actual_quaternion_xyzw_before_frame_0": actual_initial_pose[3:7].tolist(),
        "translation_error_mm": initial_translation_error_mm,
        "rotation_error_deg": initial_rotation_error_deg,
        "within_0_1_mm_and_0_1_deg": True,
        "object_root_pose_writes_after_initialization": 0,
    }
    body_names = list(robot.data.body_names)
    body_ids = {name: body_names.index(name) for name in body_names}
    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "physics_step",
            "control_frame",
            "stage",
            "commanded_q_rad",
            "measured_q_rad",
            "measured_qd_rad_s",
            "object_position_world_m",
            "object_quaternion_xyzw",
            "object_linear_velocity_m_s",
            "object_angular_velocity_rad_s",
            "thumb_force_n",
            "index_force_n",
            "middle_force_n",
            "thumb_contact_count",
            "index_contact_count",
            "middle_contact_count",
            "thumb_contact_normal_world",
            "index_contact_normal_world",
            "middle_contact_normal_world",
            "thumb_contact_point_world_m",
            "index_contact_point_world_m",
            "middle_contact_point_world_m",
            "table_contact_force_n",
            "maximum_digit_penetration_m",
            "maximum_table_penetration_m",
            "maximum_contact_tangential_velocity_m_s",
            "finger_joint_position_error_to_power_rad",
            "finger_computed_torque_nm",
            "finger_applied_torque_nm",
            "left_thumb_force_n",
            "left_index_force_n",
            "left_middle_force_n",
            "right_thumb_force_n",
            "right_index_force_n",
            "right_middle_force_n",
            "left_palm_force_n",
            "right_palm_force_n",
            "doll_bin_contact_force_n",
            "maximum_doll_bin_penetration_m",
        )
    }
    bin_contact_records: dict[str, list[Any]] = {
        key: []
        for key in (
            "physics_step",
            "control_frame",
            "stage",
            "robot_link",
            "bin_collider",
            "force_n",
            "point_world_m",
            "normal_world",
            "separation_m",
            "penetration_m",
        )
    }
    physics_step = 0
    api_errors: set[str] = set()
    active_links_seen = {digit: set() for digit in ("thumb", "index", "middle")}
    hand_slice = slice(14, 21) if args.side == "left" else slice(21, 28)

    def update() -> None:
        robot.update(dt)
        doll.update(dt)
        for sensor in digit_sensors.values():
            sensor.update(dt, force_recompute=True)
        for sensor in opposite_digit_sensors.values():
            sensor.update(dt, force_recompute=True)
        for sensor in palm_sensors.values():
            sensor.update(dt, force_recompute=True)
        table_sensor.update(dt, force_recompute=True)
        for sensor in bin_sensors.values():
            sensor.update(dt, force_recompute=True)
        for sensor in doll_bin_sensors.values():
            sensor.update(dt, force_recompute=True)

    runtime_gate_result: dict[str, Any] = {
        "required": runtime_gate_required,
        "minimum_simultaneous_support_s": runtime_gate_minimum_s,
        "status": "NOT_REQUIRED" if not runtime_gate_required else "NOT_EVALUATED",
        "left_thumb_release_applied": False,
    }
    executed_control_frames = 0
    for control_frame, (command, label) in enumerate(zip(commands, stages, strict=True)):
        if (
            runtime_gate_required
            and label == "LEFT_THUMB_RELEASE"
            and runtime_gate_result["status"] == "NOT_EVALUATED"
        ):
            verification = np.asarray(records["stage"]).astype(str) == (
                "RIGHT_THREE_DIGIT_VERIFICATION"
            )
            force_threshold = float(config["gates"]["meaningful_digit_force_n"])
            force_values = {
                digit: np.asarray(records[f"{digit}_force_n"], dtype=np.float64)
                for digit in ("thumb", "index", "middle")
            }
            meaningful = {
                digit: force_values[digit] >= force_threshold
                for digit in ("thumb", "index", "middle")
            }
            simultaneous = verification.copy()
            for digit in ("thumb", "index", "middle"):
                simultaneous &= meaningful[digit]
            simultaneous_s = longest_duration(simultaneous, dt)
            individual_s = {
                digit: longest_duration(verification & meaningful[digit], dt)
                for digit in ("thumb", "index", "middle")
            }
            table_force = np.asarray(records["table_contact_force_n"], dtype=np.float64)
            maximum_table_force = (
                float(np.max(table_force[verification], initial=0.0))
                if len(table_force)
                else float("inf")
            )
            gate_pass = bool(
                args.side == "right"
                and np.any(verification)
                and simultaneous_s >= runtime_gate_minimum_s
                and maximum_table_force
                <= float(config["gates"]["maximum_table_force_for_elevated_n"])
            )
            runtime_gate_result.update(
                {
                    "status": "PASS" if gate_pass else "FAIL",
                    "sensor_side": args.side,
                    "force_threshold_n": force_threshold,
                    "individual_support_s": individual_s,
                    "simultaneous_support_s": simultaneous_s,
                    "maximum_table_force_n": maximum_table_force,
                    "evaluated_before_control_frame": control_frame,
                }
            )
            if not gate_pass:
                break
            runtime_gate_result["left_thumb_release_applied"] = True
        target[0, joint_ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        for _ in range(substeps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            update()
            pose = numpy(doll.data.root_pose_w)[0].astype(np.float64)
            object_velocity = numpy(doll.data.root_vel_w)[0].astype(np.float64)
            measured = numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64)
            measured_qd = numpy(robot.data.joint_vel)[0, joint_ids].astype(np.float64)
            computed = numpy(robot.data.computed_torque)[0, joint_ids][hand_slice].astype(np.float64)
            applied = numpy(robot.data.applied_torque)[0, joint_ids][hand_slice].astype(np.float64)
            digit_force = {digit: 0.0 for digit in ("thumb", "index", "middle")}
            digit_count = {digit: 0 for digit in ("thumb", "index", "middle")}
            digit_normal = {
                digit: np.zeros(3, dtype=np.float64)
                for digit in ("thumb", "index", "middle")
            }
            digit_point = {
                digit: np.full(3, np.nan, dtype=np.float64)
                for digit in ("thumb", "index", "middle")
            }
            penetrations: list[float] = []
            tangential_speeds: list[float] = []
            body_lin = numpy(robot.data.body_com_lin_vel_w)[0].astype(np.float64)
            body_ang = numpy(robot.data.body_com_ang_vel_w)[0].astype(np.float64)
            body_pos = numpy(robot.data.body_com_pos_w)[0].astype(np.float64)
            object_lin = numpy(doll.data.body_com_lin_vel_w)[0, 0].astype(np.float64)
            object_ang = numpy(doll.data.body_com_ang_vel_w)[0, 0].astype(np.float64)
            object_pos = numpy(doll.data.body_com_pos_w)[0, 0].astype(np.float64)
            for role, sensor in digit_sensors.items():
                digit = role_to_digit[role]
                rows, error = contact_rows(sensor, dt)
                if error:
                    api_errors.add(error)
                # With a regex sensor spanning a complete digit chain, Isaac's
                # filtered force matrix can remain zero for a contact owned by
                # one of the matched child bodies even though the low-level
                # contact stream reports the pair.  The contact stream is the
                # authoritative per-pair measurement here; summing its scalar
                # normal forces also gives the requested total digit load.
                digit_force[digit] = float(
                    np.sum([abs(float(row["force_n"])) for row in rows])
                )
                digit_count[digit] = len(rows)
                if rows:
                    weights = np.asarray(
                        [max(float(row["force_n"]), 1.0e-12) for row in rows]
                    )
                    normal_sum = np.sum(
                        np.stack([row["normal"] for row in rows]) * weights[:, None],
                        axis=0,
                    )
                    digit_normal[digit] = normal_sum / max(
                        np.linalg.norm(normal_sum), 1.0e-12
                    )
                    digit_point[digit] = np.average(
                        np.stack([row["point"] for row in rows]), axis=0, weights=weights
                    )
                for row in rows:
                    owner_link = str(row["owner"]).rsplit("/", 1)[-1]
                    active_links_seen[digit].add(owner_link)
                    penetrations.append(max(0.0, -row["separation_m"]))
                    point = row["point"]
                    body_id = body_ids.get(owner_link, body_ids[role_to_link[role]])
                    finger_velocity = body_lin[body_id] + np.cross(
                        body_ang[body_id], point - body_pos[body_id]
                    )
                    object_point_velocity = object_lin + np.cross(
                        object_ang, point - object_pos
                    )
                    relative = finger_velocity - object_point_velocity
                    normal = row["normal"] / max(np.linalg.norm(row["normal"]), 1.0e-12)
                    tangent = relative - float(relative @ normal) * normal
                    tangential_speeds.append(float(np.linalg.norm(tangent)))
            bilateral_digit_force = {
                "left": {digit: 0.0 for digit in ("thumb", "index", "middle")},
                "right": {digit: 0.0 for digit in ("thumb", "index", "middle")},
            }
            bilateral_digit_force[args.side].update(digit_force)
            for role, sensor in opposite_digit_sensors.items():
                digit = opposite_role_to_digit[role]
                opposite_rows, opposite_error = contact_rows(sensor, dt)
                if opposite_error:
                    api_errors.add(opposite_error)
                bilateral_digit_force[opposite_side][digit] = float(
                    np.sum([abs(float(row["force_n"])) for row in opposite_rows])
                )
            palm_force = {"left": 0.0, "right": 0.0}
            for palm_side, sensor in palm_sensors.items():
                palm_rows, palm_error = contact_rows(sensor, dt)
                if palm_error:
                    api_errors.add(palm_error)
                palm_force[palm_side] = float(
                    np.sum([abs(float(row["force_n"])) for row in palm_rows])
                )
            table_rows, table_error = contact_rows(table_sensor, dt)
            if table_error:
                api_errors.add(table_error)
            doll_bin_force = 0.0
            doll_bin_penetration = 0.0
            for sensor in doll_bin_sensors.values():
                doll_bin_rows, doll_bin_error = contact_rows(sensor, dt)
                if doll_bin_error:
                    api_errors.add(doll_bin_error)
                doll_bin_force = max(
                    doll_bin_force,
                    float(np.sum([abs(float(row["force_n"])) for row in doll_bin_rows])),
                )
                doll_bin_penetration = max(
                    doll_bin_penetration,
                    max(
                        [max(0.0, -float(row["separation_m"])) for row in doll_bin_rows],
                        default=0.0,
                    ),
                )
            for bin_name, sensor in bin_sensors.items():
                bin_rows, bin_error = contact_rows(sensor, dt)
                if bin_error:
                    api_errors.add(bin_error)
                for row in bin_rows:
                    bin_contact_records["physics_step"].append(physics_step)
                    bin_contact_records["control_frame"].append(control_frame)
                    bin_contact_records["stage"].append(str(label))
                    bin_contact_records["robot_link"].append(
                        str(row["owner"]).rsplit("/", 1)[-1]
                    )
                    bin_contact_records["bin_collider"].append(
                        runtime_bin["contact_parts"][bin_name]
                    )
                    bin_contact_records["force_n"].append(float(row["force_n"]))
                    bin_contact_records["point_world_m"].append(row["point"])
                    bin_contact_records["normal_world"].append(row["normal"])
                    bin_contact_records["separation_m"].append(float(row["separation_m"]))
                    bin_contact_records["penetration_m"].append(
                        max(0.0, -float(row["separation_m"]))
                    )
            values = {
                "physics_step": physics_step,
                "control_frame": control_frame,
                "stage": str(label),
                "commanded_q_rad": command,
                "measured_q_rad": measured,
                "measured_qd_rad_s": measured_qd,
                "object_position_world_m": pose[:3],
                "object_quaternion_xyzw": pose[3:7],
                "object_linear_velocity_m_s": object_velocity[:3],
                "object_angular_velocity_rad_s": object_velocity[3:],
                "thumb_force_n": digit_force["thumb"],
                "index_force_n": digit_force["index"],
                "middle_force_n": digit_force["middle"],
                "thumb_contact_count": digit_count["thumb"],
                "index_contact_count": digit_count["index"],
                "middle_contact_count": digit_count["middle"],
                "thumb_contact_normal_world": digit_normal["thumb"],
                "index_contact_normal_world": digit_normal["index"],
                "middle_contact_normal_world": digit_normal["middle"],
                "thumb_contact_point_world_m": digit_point["thumb"],
                "index_contact_point_world_m": digit_point["index"],
                "middle_contact_point_world_m": digit_point["middle"],
                "table_contact_force_n": filtered_force(table_sensor),
                "maximum_digit_penetration_m": max(penetrations, default=0.0),
                "maximum_table_penetration_m": max(
                    [max(0.0, -row["separation_m"]) for row in table_rows], default=0.0
                ),
                "maximum_contact_tangential_velocity_m_s": max(tangential_speeds, default=0.0),
                "finger_joint_position_error_to_power_rad": power_q - measured[hand_slice],
                "finger_computed_torque_nm": computed,
                "finger_applied_torque_nm": applied,
                "left_thumb_force_n": bilateral_digit_force["left"]["thumb"],
                "left_index_force_n": bilateral_digit_force["left"]["index"],
                "left_middle_force_n": bilateral_digit_force["left"]["middle"],
                "right_thumb_force_n": bilateral_digit_force["right"]["thumb"],
                "right_index_force_n": bilateral_digit_force["right"]["index"],
                "right_middle_force_n": bilateral_digit_force["right"]["middle"],
                "left_palm_force_n": palm_force["left"],
                "right_palm_force_n": palm_force["right"],
                "doll_bin_contact_force_n": doll_bin_force,
                "maximum_doll_bin_penetration_m": doll_bin_penetration,
            }
            for key, value in values.items():
                records[key].append(value)
            physics_step += 1
        executed_control_frames = control_frame + 1
    command_completed = executed_control_frames == len(commands)
    if runtime_gate_required and runtime_gate_result["status"] == "NOT_EVALUATED":
        runtime_gate_result["status"] = "FAIL_MISSING_RELEASE_TRANSITION"
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["timestamp_s"] = arrays["physics_step"].astype(np.float64) * dt
    arrays["joint_names"] = np.asarray(names)
    log_path = output_dir / "event_log.npz"
    temporary = log_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, log_path)
    bin_contact_path: Path | None = None
    if args.audit_robot_bin:
        bin_contact_path = output_dir / "robot_bin_contacts.npz"
        bin_arrays = {key: np.asarray(value) for key, value in bin_contact_records.items()}
        temporary_bin = bin_contact_path.with_suffix(".npz.incomplete")
        with temporary_bin.open("wb") as stream:
            np.savez_compressed(stream, **bin_arrays)
        os.replace(temporary_bin, bin_contact_path)
    labels = arrays["stage"].astype(str)
    gates = config["gates"]
    digit_results: dict[str, Any] = {}
    retention_mask = labels == "GRAVITY_RETENTION"
    elevated_mask = labels == "HOLD_ELEVATED"
    release_mask = labels == "POST_RELEASE"
    contact_any = np.zeros(len(labels), dtype=bool)
    for digit in ("thumb", "index", "middle"):
        force = arrays[f"{digit}_force_n"]
        meaningful = force >= float(gates["meaningful_digit_force_n"])
        contact_any |= meaningful
        retention_duration = longest_duration(meaningful & retention_mask, dt)
        elevated_duration = longest_duration(meaningful & elevated_mask, dt)
        retention_impulse = float(np.sum(force[retention_mask]) * dt)
        full_scope_impulse = float(np.sum(force[retention_mask | elevated_mask]) * dt)
        normals = arrays[f"{digit}_contact_normal_world"]
        points = arrays[f"{digit}_contact_point_world_m"]
        retained = retention_mask & (arrays[f"{digit}_contact_count"] > 0)
        digit_results[digit] = {
            "active_link": next(
                role_to_link[role] for role in role_to_link if role_to_digit[role] == digit
            ),
            "active_contact_links": sorted(active_links_seen[digit]),
            "maximum_force_n": float(np.max(force, initial=0.0)),
            "mean_force_during_retention_n": (
                float(np.mean(force[retention_mask])) if np.any(retention_mask) else 0.0
            ),
            "meaningful_contact_duration_s": retention_duration,
            "elevated_meaningful_contact_duration_s": elevated_duration,
            "normal_impulse_retention_ns": retention_impulse,
            "normal_impulse_retention_and_elevated_ns": full_scope_impulse,
            "mean_retention_contact_normal_world": (
                np.nanmean(normals[retained], axis=0)
                if np.any(retained)
                else np.zeros(3, dtype=np.float64)
            ),
            "mean_retention_contact_point_world_m": (
                np.nanmean(points[retained], axis=0)
                if np.any(retained)
                else None
            ),
            "meaningful": bool(
                retention_duration
                >= float(gates["minimum_meaningful_digit_contact_s"])
                and retention_impulse >= float(gates["minimum_digit_normal_impulse_ns"])
            ),
        }
    all_three_meaningful = all(row["meaningful"] for row in digit_results.values())
    positions = arrays["object_position_world_m"]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    # Initial overlap is a reset-state property, not contact that may develop
    # while the object settles over the full 0.5 s OPEN stage.
    open_mask = arrays["control_frame"] == 0
    initial_penetration = float(
        np.max(arrays["maximum_digit_penetration_m"][open_mask], initial=0.0)
    )
    max_penetration = float(np.max(arrays["maximum_digit_penetration_m"], initial=0.0))
    retention_start = (
        positions[np.flatnonzero(retention_mask)[0]]
        if np.any(retention_mask)
        else positions[0]
    )
    retention_displacement = np.linalg.norm((positions - retention_start)[..., :2], axis=1)
    retention_contact_duration = longest_duration(contact_any & retention_mask, dt)
    retention_pass = bool(
        retention_contact_duration >= float(gates["minimum_retention_contact_s"])
        and float(np.max(retention_displacement[retention_mask], initial=0.0))
        <= float(gates["maximum_retention_horizontal_displacement_m"])
    )
    pre_lift_z = (
        float(
            np.median(
                positions[retention_mask][
                    -min(30, np.count_nonzero(retention_mask)) :, 2
                ]
            )
        )
        if np.any(retention_mask)
        else float(positions[0, 2])
    )
    measured_lift = float(np.max(positions[:, 2]) - pre_lift_z)
    elevated_contact = contact_any & elevated_mask
    elevated_no_table = arrays["table_contact_force_n"] <= float(
        gates["maximum_table_force_for_elevated_n"]
    )
    elevated_height = positions[:, 2] - pre_lift_z >= float(gates["minimum_measured_lift_m"])
    elevated_hold_duration = longest_duration(
        elevated_contact & elevated_no_table & elevated_height & elevated_mask, dt
    )
    release_no_hand = ~contact_any & release_mask
    release_on_table = arrays["table_contact_force_n"] > 0.0
    release_no_hand_duration = longest_duration(release_no_hand, dt)
    release_table_duration = longest_duration(release_on_table & release_mask, dt)
    release_required = not bool(config["timing"].get("end_after_elevated_hold", False))
    release_pass = bool(
        not release_required
        or (
            release_no_hand_duration >= float(gates["minimum_release_no_hand_contact_s"])
            and release_table_duration >= float(gates["minimum_release_table_contact_s"])
        )
    )
    max_linear = float(
        np.max(np.linalg.norm(arrays["object_linear_velocity_m_s"], axis=1), initial=0.0)
    )
    max_angular = float(
        np.max(np.linalg.norm(arrays["object_angular_velocity_rad_s"], axis=1), initial=0.0)
    )
    artifact_free = bool(
        not api_errors
        and initial_penetration <= float(gates["maximum_initial_penetration_m"])
        and max_penetration <= float(gates["maximum_runtime_penetration_m"])
        and max_linear <= float(gates["maximum_object_linear_speed_m_s"])
        and max_angular <= float(gates["maximum_object_angular_speed_rad_s"])
        and float(np.max(steps, initial=0.0)) <= float(gates["maximum_object_com_step_m"])
    )
    lift_pass = bool(
        measured_lift >= float(gates["minimum_measured_lift_m"])
        and elevated_hold_duration >= float(gates["minimum_elevated_hold_s"])
    )
    passed = bool(
        artifact_free
        and all_three_meaningful
        and retention_pass
        and lift_pass
        and release_pass
    )
    result = {
        "schema_version": f"{schema}_trial",
        "status": "PASS" if passed else "FAIL",
        "side": args.side,
        "geometry": args.geometry,
        "dimensions_m": dimensions,
        "visual_dimensions_m": visual_dimensions,
        "collision_scale_percent": int(geometry["collision_scale_percent"]),
        "profile": args.profile,
        "runtime_bin": runtime_bin,
        "runtime_dex3_joint_stops": runtime_dex3_joint_stops,
        "runtime_articulation_solver": runtime_articulation_solver,
        "active_power_grasp_7d_rad": power_q,
        "three_meaningful_digit_contacts": all_three_meaningful,
        "digits": digit_results,
        "retention": {
            "status": "PASS" if retention_pass else "FAIL",
            "required_s": gates["minimum_retention_contact_s"],
            "measured_contact_s": retention_contact_duration,
            "maximum_horizontal_displacement_m": float(
                np.max(retention_displacement[retention_mask], initial=0.0)
            ),
        },
        "lift": {
            "status": "PASS" if lift_pass else "FAIL",
            "requested_m": config["timing"]["requested_lift_m"],
            "measured_m": measured_lift,
            "elevated_hold_s": elevated_hold_duration,
        },
        "release": {
            "status": "NOT_REQUESTED" if not release_required else "PASS" if release_pass else "FAIL",
            "no_hand_contact_s": release_no_hand_duration,
            "table_contact_s": release_table_duration,
        },
        "artifact_checks": {
            "status": "PASS" if artifact_free else "FAIL",
            "initial_penetration_m": initial_penetration,
            "maximum_runtime_penetration_m": max_penetration,
            "maximum_object_linear_speed_m_s": max_linear,
            "maximum_object_angular_speed_rad_s": max_angular,
            "maximum_object_com_step_m": float(np.max(steps, initial=0.0)),
            "contact_api_errors": sorted(api_errors),
            "explosive_motion": bool(
                max_linear > float(gates["maximum_object_linear_speed_m_s"])
                or max_angular > float(gates["maximum_object_angular_speed_rad_s"])
            ),
        },
        "runtime_proxy": runtime_proxy,
        "collision_contact_pairs": [
            f"{G1}/{link}/collisions <-> {PROXY}"
            for digit in ("thumb", "index", "middle")
            for link in sorted(active_links_seen[digit])
        ],
        "object": {
            "mass_kg": config["object"]["mass_kg"],
            "gravitational_load_n": float(
                config["object"]["mass_kg"] * config["simulation"]["gravity_m_s2"]
            ),
            "initial_center_world_m": center,
            "maximum_contact_tangential_velocity_m_s": float(
                np.max(arrays["maximum_contact_tangential_velocity_m_s"], initial=0.0)
            ),
        },
        "material": runtime_proxy["material"],
        "finger_drive": config["finger_drive"],
        "control": {
            "fps": fps,
            "physics_dt_s": dt,
            "gravity_m_s2": config["simulation"]["gravity_m_s2"],
            "arm_placement_changed_during_hand_close": False,
            "power_close_arm_motion": "none; one fixed policy-independent palm center is held through OPEN, PRESHAPE, POWER_GRASP, and gravity retention",
            "lift_arm_source": str(primitive_path),
        },
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "object_task_frame_registration": {
            "used": object_registration is not None,
            "config": str(object_registration_path)
            if object_registration_path is not None
            else None,
            "config_sha256": sha256_file(object_registration_path)
            if object_registration_path is not None
            else None,
            "registered_center_world_m": center,
            "method_independent": bool(
                object_registration.get("method_independent", False)
            )
            if object_registration is not None
            else None,
            "episode_independent": bool(
                object_registration.get("episode_independent", False)
            )
            if object_registration is not None
            else None,
            "changes_object_physics": bool(
                object_registration.get("changes_object_physics", True)
            )
            if object_registration is not None
            else None,
            "episode_conditioned_source_derived": episode_registration_entry is not None,
            "registration_physics_binding": episode_registration_binding,
            "stable_episode_id": str(args.episode_stable_id)
            if episode_registration_entry is not None
            else None,
            "runtime_initial_pose_verification": runtime_initial_pose_verification,
            "bin_pose_fixed": True,
            "table_pose_fixed": True,
            "G1_root_fixed": True,
        },
        "source_scene_sha256": sha256_file(Path(config["source_scene"])),
        "source_arm_primitive": str(primitive_path),
        "source_arm_primitive_sha256": sha256_file(primitive_path),
        "scripted_command": str(command_path),
        "scripted_command_sha256": sha256_file(command_path),
        "event_log": str(log_path),
        "event_log_sha256": sha256_file(log_path),
        "robot_bin_contact_audit": {
            "enabled": bool(args.audit_robot_bin),
            "contact_log": str(bin_contact_path) if bin_contact_path is not None else None,
            "contact_log_sha256": (
                sha256_file(bin_contact_path) if bin_contact_path is not None else None
            ),
            "contact_row_count": len(bin_contact_records["physics_step"]),
            "bin_colliders": runtime_bin["contact_parts"],
            "command_or_physics_modified": False,
        },
        "full_task_contact_audit": {
            "enabled": bool(args.full_task_audit),
            "bilateral_hand_object_sensors": bool(args.full_task_audit),
            "doll_bin_sensors": bool(args.full_task_audit),
            "command_or_physics_modified": False,
        },
        "policy_or_checkpoint_used": False,
        "prohibited_attachment_used": False,
        "object_pose_writes_during_timed_loop": 0,
        "state_restoration": {
            "used": restored_keyframe is not None,
            "debug_only": restored_keyframe is not None,
            "permitted_in_final_continuous_validation": False
            if restored_keyframe is not None
            else None,
            "keyframe": str(restored_keyframe_path)
            if restored_keyframe_path is not None
            else None,
            "keyframe_sha256": sha256_file(restored_keyframe_path)
            if restored_keyframe_path is not None
            else None,
        },
        "runtime_right_three_digit_gate": runtime_gate_result,
        "executed_control_frames": executed_control_frames,
        "requested_control_frames": len(commands),
        "command_completed": command_completed,
        "real_robot": False,
    }
    result_path = output_dir / "trial_result.json"
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0 if passed else 2


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
