#!/usr/bin/env python3
"""Autorun the strongest persisted frame-zero Doll-Handoff physics trace.

This is deliberately a read-only visual diagnostic.  It exposes no robot
controls, does not construct or edit a trajectory, and replays the persisted
measured robot/object trace after physical task failure.  Closing the Isaac
Sim window is the only user interaction handled by this tool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gui", action="store_true", help="Launch the visible Isaac Kit GUI.")
parser.add_argument(
    "--latest-bin-calibration",
    action="store_true",
    help=(
        "Replay the hash-locked terminal 100 mm symmetric-beveled bin calibration "
        "trace instead of the historical 190 mm-bin trace."
    ),
)
parser.add_argument(
    "--bin-150mm-audit",
    action="store_true",
    help=(
        "Replay the hash-locked measured-state trace from the exact 150 mm "
        "bin physics audit. No physics result is recomputed."
    ),
)
parser.add_argument(
    "--contact-constrained-final",
    action="store_true",
    help=(
        "Replay the measured-state trace from the frozen 3/3 contact-constrained "
        "150 mm-bin scripted validation. No physics result is recomputed."
    ),
)
parser.add_argument(
    "--playback-speed",
    type=float,
    default=1.0,
    help=(
        "Display-only wall-clock speed multiplier. Every persisted command/event "
        "frame is still applied in order; rendering is decimated proportionally."
    ),
)
parser.add_argument(
    "--validation-exit-after-frames",
    type=int,
    default=0,
    help=argparse.SUPPRESS,
)
parser.add_argument("--validation-exit-after-replay", action="store_true", help=argparse.SUPPRESS)
parser.add_argument(
    "--capture-keyframes-dir",
    type=Path,
    default=None,
    help=argparse.SUPPRESS,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    parser.error("this visual replay requires --gui")
if not np.isfinite(args.playback_speed) or args.playback_speed <= 0.0:
    parser.error("--playback-speed must be a finite positive value")
if sum(
    map(
        bool,
        (
            args.latest_bin_calibration,
            args.bin_150mm_audit,
            args.contact_constrained_final,
        ),
    )
) > 1:
    parser.error("select only one recorded bin-audit trace")
if getattr(args, "headless_explicit", False):
    parser.error("--gui cannot be combined with --headless")
args.visualizer = ["kit"]
args.visualizer_explicit = True
args.visualizer_disable_all = False
args.headless = False
args.headless_explicit = False
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

import carb.settings
import omni.usd
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.evaluation.contracts import authoritative_joint_ranges
from tools.policy_b_isaac_control_contract import CONTROLLER_CONTRACT


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"

# This is the latest persisted, frame-zero command which includes every
# downstream stage through RIGHT transport, controlled bin descent, release,
# and bin settle.  It is intentionally replayed despite its recorded physical
# failure.  Newer G04 bridge-only traces end at high stabilization and therefore
# are not complete full-sequence visual diagnostics.
COMMAND = (
    ROOT
    / "outputs/final_task_completion_v1/08_r26_retiming/variants/R26_T4_4P0X"
    / "retimed_r26_full_command.npz"
)
EVENT = COMMAND.parent / "physics_full/event_log.npz"
OFFLINE = COMMAND.parent / "offline_report.json"

EXPECTED_SHA256 = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    COMMAND: "cb680b12d78cff61c53e6e466d605b3732508a365f571b1ceee62aba54d4f52c",
    EVENT: "94969897401da32bf1906905991acef2bf6939f514ea0ce7dfaccd3fe1696b17",
}
EXPECTED_SCENE_SHA256 = "770f574db096819dd661b86d2bafdbd99964de7b77718e2013efeb95d1840aaf"

LATEST_COMMAND = (
    ROOT
    / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm"
    / "bin_calibrated_full_command.npz"
)
LATEST_EVENT = (
    ROOT
    / "outputs/final_bin_calibrated_completion/01_selected_bin/height_100mm"
    / "symmetric_beveled_rim_physics_audit/event_log.npz"
)
LATEST_CONTACT = LATEST_EVENT.parent / "robot_bin_contacts.npz"
LATEST_TRIAL = LATEST_EVENT.parent / "trial_result.json"
LATEST_EXPECTED_SHA256 = {
    LATEST_COMMAND: "fc8b81b001f031c141843c55e78e606ae478c534b12ec86bfe9728a4a93bf724",
    LATEST_EVENT: "580401d77be9c95725d04754bfef615d103ce7dc949cabf1a5d35134f9d1c1b8",
    LATEST_CONTACT: "24d766195e68ac17d1ef23fef3c8a6319e5da64f2f11c08527453f7186426d9c",
    LATEST_TRIAL: "3bf6ad2620ba4348380de2db4c8e8df260aecbb1fcc1b90d669028f320dc6851",
}
LATEST_BIN_HEIGHT_M = 0.100
LATEST_BIN_RIM_BEVEL_M = 0.003

BIN_150_COMMAND = LATEST_COMMAND
BIN_150_EVENT = (
    ROOT
    / "outputs/final_150mm_bin_completion/01_exact_current_trajectory_physics"
    / "event_log.npz"
)
BIN_150_CONTACT = BIN_150_EVENT.parent / "robot_bin_contacts.npz"
BIN_150_TRIAL = BIN_150_EVENT.parent / "trial_result.json"
BIN_150_EXPECTED_SHA256 = {
    BIN_150_COMMAND: "fc8b81b001f031c141843c55e78e606ae478c534b12ec86bfe9728a4a93bf724",
    BIN_150_EVENT: "5ef5981d9b0e29d493451dbbb361c818ce6c9c9487a80cb4cf769983d84f353f",
    BIN_150_CONTACT: "50649af6a5c196703ab2873e978c424403ab5671e4dc19bb23b4b887c4db7ebf",
    BIN_150_TRIAL: "f4ec930b9b4a1718b48b96a592a0e96dcbee775044ac3d8c926e9c628b8fc0c9",
}
BIN_150_HEIGHT_M = 0.150
BIN_150_RIM_BEVEL_M = 0.003

CONTACT_FINAL_COMMAND = LATEST_COMMAND
CONTACT_FINAL_EVENT = (
    ROOT
    / "outputs/final_contact_constrained_eval/02_scripted_validation/run_01"
    / "event_log.npz"
)
CONTACT_FINAL_CONTACT = CONTACT_FINAL_EVENT.parent / "robot_bin_contacts.npz"
CONTACT_FINAL_TRIAL = CONTACT_FINAL_EVENT.parent / "trial_result.json"
CONTACT_FINAL_RESULT = CONTACT_FINAL_EVENT.parent / "CONTACT_CONSTRAINED_TASK_RESULT.json"
CONTACT_FINAL_EXPECTED_SHA256 = {
    CONTACT_FINAL_COMMAND: "fc8b81b001f031c141843c55e78e606ae478c534b12ec86bfe9728a4a93bf724",
    CONTACT_FINAL_EVENT: "ea5ccd9a1cb864fec1f498a23425f6c5fb456880447df993b400bc6e90891e6e",
    CONTACT_FINAL_CONTACT: "50649af6a5c196703ab2873e978c424403ab5671e4dc19bb23b4b887c4db7ebf",
    CONTACT_FINAL_TRIAL: "e6fd1dcfc619bcf86f8ebb3767282ae5c7e18908c5b7730ff4251f6fd884f34d",
    CONTACT_FINAL_RESULT: "d1b9cee9e656d3cc4eb472c2f86495ca7e8cd6bae34f5bff696d9aa93e69ecc8",
}

DOLL = "/World/DollHandoffEnvironment/Doll"
SOURCE_BODY = f"{DOLL}/Body"
PROXY = f"{DOLL}/MeasuredProxyV3Collider"
VISUAL = f"{DOLL}/MeasuredProxyV3Visual"
G1 = "/World/G1/Asset"
BIN_PARTS = {
    name: f"/World/DollHandoffEnvironment/TrashBin/{name}"
    for name in ("Bottom", "FrontWall", "BackWall", "LeftWall", "RightWall")
}
CONTROL_FPS = 30.0
# RTX/Kit rendering on this workstation is slower than 30 complete viewport
# updates per second.  Rendering every third *recorded control* frame preserves
# the exact frozen 30-Hz physics trace and presents its normal wall-clock
# duration at approximately 10 viewport frames per second.
VIEWPORT_RENDER_STRIDE = 3


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"immutable replay dependency changed: {path}: {actual}")


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def rounded_oval_mesh(dimensions: np.ndarray) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    """Match the authoritative compressed-plush proxy construction exactly."""

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


def apply_frozen_proxy(stage: Usd.Stage, config: dict[str, Any]) -> None:
    """Reapply the frozen session-layer proxy without saving the USD."""

    geometry = config["geometry_candidates"][0]
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
    for path, dimensions, visible in (
        (VISUAL, visual_dimensions, True),
        (PROXY, collision_dimensions, False),
    ):
        mesh = UsdGeom.Mesh.Define(stage, path)
        points, counts, indices = rounded_oval_mesh(dimensions)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        if visible:
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.72, 0.34)])
        else:
            collider_z_offset = float((collision_dimensions[2] - visual_dimensions[2]) / 2.0)
            UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, collider_z_offset))
            UsdGeom.Imageable(mesh).MakeInvisible()

    proxy = stage.GetPrimAtPath(PROXY)
    UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(proxy).CreateApproximationAttr("convexHull")
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(proxy)
    physx_collision.CreateContactOffsetAttr(float(config["object"]["contact_offset_m"]))
    physx_collision.CreateRestOffsetAttr(float(config["object"]["rest_offset_m"]))

    material = UsdShade.Material.Define(stage, "/World/DollGraspableProxyV2Material")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(float(config["material"]["static_friction"]))
    material_api.CreateDynamicFrictionAttr(float(config["material"]["dynamic_friction"]))
    material_api.CreateRestitutionAttr(float(config["object"]["restitution"]))
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(
        str(config["material"]["friction_combine_mode"])
    )
    physx_material.CreateRestitutionCombineModeAttr().Set(
        str(config["material"]["restitution_combine_mode"])
    )
    UsdShade.MaterialBindingAPI.Apply(proxy).Bind(material, materialPurpose="physics")

    root = stage.GetPrimAtPath(DOLL)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(config["object"]["mass_kg"]))
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rigid.CreateLinearDampingAttr(float(config["object"]["linear_damping"]))
    rigid.CreateAngularDampingAttr(float(config["object"]["angular_damping"]))
    rigid.CreateMaxDepenetrationVelocityAttr(
        float(config["object"]["max_depenetration_velocity_m_s"])
    )


def define_symmetric_beveled_wall(
    stage: Usd.Stage,
    source_prim: Usd.Prim,
    name: str,
    external_height_m: float,
    bevel_m: float,
) -> str:
    """Recreate the exact symmetric rim mesh used by the terminal audit."""

    bottom_z = 0.006
    top_z = float(external_height_m)
    shoulder_z = top_z - float(bevel_m)
    outer_x, outer_y = 0.095, 0.0825
    inner_x, inner_y = 0.089, 0.0765
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
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    binding_targets = (
        UsdShade.MaterialBindingAPI(source_prim).GetDirectBindingRel().GetTargets()
    )
    if binding_targets:
        mesh.GetPrim().CreateRelationship("material:binding").SetTargets(binding_targets)
    UsdPhysics.CollisionAPI(source_prim).GetCollisionEnabledAttr().Set(False)
    UsdGeom.Imageable(source_prim).MakeInvisible()
    return path


def apply_and_verify_trace_bin(
    stage: Usd.Stage, height_m: float, bevel_m: float
) -> dict[str, Any]:
    """Apply and verify one recorded run's bottom-aligned runtime bin."""

    height = float(height_m)
    bevel = float(bevel_m)
    bottom_thickness = 0.006
    wall_height = height - bottom_thickness
    wall_center_z = bottom_thickness + 0.5 * wall_height
    mesh_paths: dict[str, str] = {}
    for name in ("FrontWall", "BackWall", "LeftWall", "RightWall"):
        source = stage.GetPrimAtPath(BIN_PARTS[name])
        if not source.IsValid() or not source.IsA(UsdGeom.Cube):
            raise RuntimeError(f"authored sharp bin wall missing: {BIN_PARTS[name]}")
        xformable = UsdGeom.Xformable(source)
        scale_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
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
            raise RuntimeError(f"authored bin wall xform incomplete: {source.GetPath()}")
        scale = np.asarray(scale_op.Get(), dtype=np.float64)
        translate = np.asarray(translate_op.Get(), dtype=np.float64)
        scale[2] = wall_height
        translate[2] = wall_center_z
        scale_op.Set(Gf.Vec3f(*map(float, scale)))
        translate_op.Set(Gf.Vec3d(*map(float, translate)))
        mesh_paths[name] = define_symmetric_beveled_wall(
            stage, source, name, height, bevel
        )

    # Read the actual running stage, not only the requested constants.
    for name, path in mesh_paths.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"runtime beveled wall missing: {path}")
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        if not np.isclose(points[:, 2].max(), height, atol=1.0e-9):
            raise RuntimeError(f"runtime bin height verification failed: {name}")
        shoulder = np.sort(np.unique(np.round(points[:, 2], 9)))[-2]
        if not np.isclose(height - shoulder, bevel, atol=1.0e-9):
            raise RuntimeError(f"runtime rim bevel verification failed: {name}")
        source = stage.GetPrimAtPath(BIN_PARTS[name])
        if UsdGeom.Imageable(source).ComputeVisibility() != UsdGeom.Tokens.invisible:
            raise RuntimeError(f"original 190 mm sharp wall remains displayed: {name}")
        if UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Get() is not False:
            raise RuntimeError(f"original sharp wall collision remains enabled: {name}")
    return {
        "height_m": height,
        "rim_world_z_m": 0.795 + height,
        "bevel_m": bevel,
        "mesh_paths": mesh_paths,
        "original_190mm_sharp_walls_visible": False,
        "original_190mm_sharp_walls_collision_enabled": False,
    }


def build_actuators(config: dict[str, Any]) -> dict[str, ImplicitActuatorCfg]:
    values = copy.deepcopy(CONTROLLER_CONTRACT["actuators"])
    drive = config["finger_drive"]
    values["dex3"].update(
        effort_limit_sim=float(drive["effort_limit_sim"]),
        velocity_limit_sim=float(drive["velocity_limit_sim"]),
        stiffness=float(drive["kp"]),
        damping=float(drive["kd"]),
    )
    return {name: ImplicitActuatorCfg(**spec) for name, spec in values.items()}


def persisted_first_failure(event: dict[str, np.ndarray], speed_gate: float) -> tuple[int, str]:
    """Return the first hard physical failure recorded in the authoritative trace."""

    speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    failed = np.flatnonzero(speed > speed_gate)
    if not failed.size:
        raise RuntimeError("the selected failed full sequence has no recorded hard failure")
    row = int(failed[0])
    return int(event["control_frame"][row]), str(event["stage"][row])


def main() -> int:
    latest = bool(args.latest_bin_calibration)
    bin_150 = bool(args.bin_150mm_audit)
    contact_final = bool(args.contact_constrained_final)
    recorded_trace = latest or bin_150 or contact_final
    playback_speed = float(args.playback_speed)
    display_source_stride = max(1, int(round(playback_speed)))
    if recorded_trace and not np.isclose(playback_speed, display_source_stride):
        raise RuntimeError(
            "recorded trace viewer requires an integer --playback-speed so every "
            "displayed state is an exact persisted control frame"
        )
    if contact_final:
        command_path = CONTACT_FINAL_COMMAND
        event_path = CONTACT_FINAL_EVENT
        contact_path = CONTACT_FINAL_CONTACT
        trial_path = CONTACT_FINAL_TRIAL
        bin_height_m = BIN_150_HEIGHT_M
        bin_bevel_m = BIN_150_RIM_BEVEL_M
        immutable = {CONFIG: EXPECTED_SHA256[CONFIG], **CONTACT_FINAL_EXPECTED_SHA256}
        final_result = read_json(CONTACT_FINAL_RESULT)
        if not final_result.get("outcomes", {}).get("FULL_TASK_SUCCESS"):
            raise RuntimeError("frozen contact-constrained final trace is not a full-task PASS")
    elif bin_150:
        command_path = BIN_150_COMMAND
        event_path = BIN_150_EVENT
        contact_path = BIN_150_CONTACT
        trial_path = BIN_150_TRIAL
        bin_height_m = BIN_150_HEIGHT_M
        bin_bevel_m = BIN_150_RIM_BEVEL_M
        immutable = {CONFIG: EXPECTED_SHA256[CONFIG], **BIN_150_EXPECTED_SHA256}
    elif latest:
        command_path = LATEST_COMMAND
        event_path = LATEST_EVENT
        contact_path = LATEST_CONTACT
        trial_path = LATEST_TRIAL
        bin_height_m = LATEST_BIN_HEIGHT_M
        bin_bevel_m = LATEST_BIN_RIM_BEVEL_M
        immutable = {CONFIG: EXPECTED_SHA256[CONFIG], **LATEST_EXPECTED_SHA256}
    else:
        command_path = COMMAND
        event_path = EVENT
        contact_path = None
        trial_path = None
        bin_height_m = None
        bin_bevel_m = None
        immutable = EXPECTED_SHA256
    for path, digest in immutable.items():
        require_sha(path, digest)
    config = read_json(CONFIG)
    scene = Path(config["source_scene"])
    require_sha(scene, EXPECTED_SCENE_SHA256)
    if recorded_trace:
        assert trial_path is not None and bin_height_m is not None and bin_bevel_m is not None
        trial = read_json(trial_path)
        runtime_bin = trial.get("runtime_bin", {})
        if not np.isclose(runtime_bin.get("external_height_m", np.nan), bin_height_m):
            raise RuntimeError("recorded trace provenance has a different bin height")
        if not np.isclose(runtime_bin.get("rim_bevel_m", np.nan), bin_bevel_m):
            raise RuntimeError("recorded trace provenance has a different rim bevel")
    else:
        offline = read_json(OFFLINE)
        if offline.get("status") != "OFFLINE_PASS":
            raise RuntimeError("persisted full command no longer has an offline safety PASS")

    with np.load(command_path, allow_pickle=False) as archive:
        command = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(event_path, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    commands = np.asarray(command["commanded_q_rad"], dtype=np.float64)
    stages = command["stage"].astype(str)
    names, _ranges = authoritative_joint_ranges()
    expected_frames = 3309 if recorded_trace else 2635
    if commands.shape != (expected_frames, 28) or command["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("persisted full command shape or named joint order changed")
    if not np.isfinite(commands).all() or not np.isclose(
        float(command["control_fps_hz"]), CONTROL_FPS
    ):
        raise RuntimeError("persisted full command is malformed")
    event_frames = event["control_frame"].astype(np.int64)
    if not np.array_equal(np.unique(event_frames), np.arange(len(commands))):
        raise RuntimeError("persisted physics trace does not cover every command frame")
    # One stored row per 240-Hz physics substep; select the final measured state
    # in each 30-Hz control frame for exact, non-interpolated visual playback.
    final_event_rows = np.r_[np.flatnonzero(np.diff(event_frames) != 0), len(event_frames) - 1]
    if len(final_event_rows) != len(commands):
        raise RuntimeError("persisted physics trace/control-frame cardinality changed")
    if not np.allclose(
        event["commanded_q_rad"][final_event_rows], commands, atol=1.0e-7, rtol=0.0
    ):
        raise RuntimeError("persisted physics trace no longer matches the selected command")
    required_stages = {
        "LEFT_POWER_GRASP",
        "LEFT_LIFT_5CM",
        "LEFT_TRANSPORT",
        "RIGHT_THREE_DIGIT_VERIFICATION",
        "LEFT_THUMB_RELEASE",
        "RIGHT_POST_RELEASE_RETENTION",
        "RIGHT_TRANSPORT_TO_BIN",
        "RIGHT_RIM_RELATIVE_CONTROLLED_DESCENT"
        if recorded_trace
        else "RIGHT_CONTROLLED_BIN_DESCENT",
        "RIGHT_RELEASE",
        "BIN_SETTLE",
    }
    if not required_stages.issubset(set(stages)):
        raise RuntimeError("selected persisted command is not a complete visual sequence")
    if recorded_trace:
        assert contact_path is not None
        if contact_final:
            failure_frame = len(commands) + 1
            failure_stage = "NONE_SCRIPTED_FULL_TASK_PASS"
        else:
            with np.load(contact_path, allow_pickle=False) as archive:
                contact_force = np.asarray(archive["force_n"], dtype=np.float64)
                positive = np.flatnonzero(contact_force > 1.0e-6)
                if not positive.size:
                    raise RuntimeError("recorded calibration contact provenance is missing")
                first = int(positive[0])
                failure_frame = int(archive["control_frame"][first])
                failure_stage = str(archive["stage"][first])
    else:
        failure_frame, failure_stage = persisted_first_failure(
            event, float(config["gates"]["maximum_object_linear_speed_m_s"])
        )

    settings = carb.settings.get_settings()
    settings.set_string("/isaaclab/visualizer/types", "")
    settings.set_bool("/isaaclab/visualizer/explicit", True)
    settings.set_bool("/isaaclab/visualizer/disable_all", True)
    settings.set_bool(
        "/rtx/hydra/readTransformsFromFabricInRenderDelegate", not recorded_trace
    )
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError("failed to open frozen Doll-Handoff scene")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    apply_frozen_proxy(stage, config)
    runtime_bin_verification = (
        apply_and_verify_trace_bin(stage, bin_height_m, bin_bevel_m)
        if recorded_trace
        else None
    )

    dt = float(config["timing"]["physics_dt_s"])
    substeps = int(config["timing"]["physics_substeps_per_control_frame"])
    if not np.isclose(dt * substeps, 1.0 / CONTROL_FPS):
        raise RuntimeError("frozen physics/control timing changed")
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=str(config["simulation"]["device"]),
            gravity=(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"])),
            # Latest mode is a USD-backed recorded-state viewer, not a physics
            # rerun.  Disabling Fabric here affects only transform delivery to
            # the viewport; the persisted scientific trace is unchanged.
            use_fabric=False
            if recorded_trace
            else bool(config["simulation"]["use_fabric"]),
        )
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path=f"{G1}/root_joint",
            spawn=None,
            actuators=build_actuators(config),
        )
    )
    doll = RigidObject(RigidObjectCfg(prim_path=DOLL, spawn=None))
    sim.reset()
    if recorded_trace:
        assert runtime_bin_verification is not None
        print(
            (
                "CONTACT_CONSTRAINED_150MM_STAGE_VERIFIED "
                if contact_final
                else "BIN_150MM_STAGE_VERIFIED "
                if bin_150
                else "LATEST_BIN_STAGE_VERIFIED "
            )
            +
            f"height_m={runtime_bin_verification['height_m']:.3f} "
            f"bevel_m={runtime_bin_verification['bevel_m']:.3f} "
            "original_190mm_sharp_bin=false",
            flush=True,
        )

    isaac_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    missing = [name for name in names if name not in isaac_names]
    joint_ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(joint_ids) != 28 or len(set(joint_ids)) != 28:
        raise RuntimeError(f"Isaac named joint mapping failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero_velocity = torch.zeros_like(target)

    asset_xform = UsdGeom.Xformable(stage.GetPrimAtPath(G1))
    world_from_asset = np.asarray(
        asset_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    ).T
    asset_from_world = np.linalg.inv(world_from_asset)
    body_visual_ops: dict[str, UsdGeom.XformOp] = {}
    for body_name in body_names:
        body_prim = stage.GetPrimAtPath(f"{G1}/{body_name}")
        if not body_prim.IsValid():
            raise RuntimeError(f"G1 visual body prim is missing: {body_name}")
        body_visual_ops[body_name] = UsdGeom.Xformable(body_prim).MakeMatrixXform()
    doll_prim = stage.GetPrimAtPath(DOLL)
    doll_parent = doll_prim.GetParent()
    world_from_doll_parent = np.asarray(
        UsdGeom.Xformable(doll_parent).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ),
        dtype=np.float64,
    ).T
    doll_parent_from_world = np.linalg.inv(world_from_doll_parent)
    doll_visual_op = UsdGeom.Xformable(doll_prim).MakeMatrixXform()
    physics_view = sim.physics_manager.get_physics_sim_view()
    if physics_view is None:
        raise RuntimeError("PhysX simulation view unavailable for recorded-state FK")

    visual_dimensions = np.asarray(config["object"]["visual_dimensions_m"], dtype=np.float32)
    center_xy = config["object"]["center_world_xy_m_by_side"]["left"]
    center = np.asarray(
        [
            *center_xy,
            float(config["object"]["table_surface_world_z_m"])
            + float(visual_dimensions[2]) / 2.0
            + float(config["object"]["spawn_clearance_above_table_m"]),
        ],
        dtype=np.float32,
    )
    orientation = np.asarray(
        config["object"].get("orientation_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
        dtype=np.float32,
    )

    # The source scene camera is set once.  No subsequent camera write occurs,
    # leaving ordinary viewport orbit/pan/zoom fully available throughout and
    # after replay.
    camera = read_json(ROOT / "isaaclab_doll_handoff_scene/scene_layout.json")["camera"][
        "presets"
    ]["overview"]
    sim.set_camera_view(camera["eye_world_xyz_m"], camera["target_world_xyz_m"])
    from isaacsim.core.rendering_manager import ViewportManager

    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=camera["eye_world_xyz_m"],
        target=camera["target_world_xyz_m"],
    )

    capture_targets = {
        80: "01_left_grasp.png",
        160: "02_left_lift.png",
        400: "03_handoff.png",
        800: "04_right_acquisition.png",
        1640: "05_left_release.png",
        1740: "06_right_ownership.png",
        2250: "07_right_transport.png",
        2500: "08_bin_approach.png",
        2800: "09_release.png",
        3250: "10_doll_settled.png",
    }
    capture_helpers: list[Any] = []
    captured_targets: set[int] = set()
    if args.capture_keyframes_dir is not None:
        if not contact_final:
            raise RuntimeError("keyframe capture is restricted to --contact-constrained-final")
        args.capture_keyframes_dir.mkdir(parents=True, exist_ok=True)

    def render_and_pump() -> None:
        sim.render()
        sim.render_context.reset_transform_cadence()
        prior = settings.get("/app/player/playSimulations")
        settings.set_bool("/app/player/playSimulations", False)
        try:
            simulation_app.update()
        finally:
            settings.set_bool(
                "/app/player/playSimulations", True if prior is None else bool(prior)
            )

    def propagate_recorded_state_to_viewport(
        doll_position: np.ndarray, doll_quaternion_xyzw: np.ndarray
    ) -> None:
        """Run FK and author transient body matrices without advancing physics."""

        physics_view.update_articulations_kinematic()
        sim.forward()
        robot.update(sim.get_physics_dt())
        doll.update(sim.get_physics_dt())
        body_positions = numpy(robot.data.body_pos_w)[0]
        body_quaternions_xyzw = numpy(robot.data.body_quat_w)[0]
        for body_id, body_name in enumerate(body_names):
            world_from_body = np.eye(4, dtype=np.float64)
            world_from_body[:3, :3] = Rotation.from_quat(
                body_quaternions_xyzw[body_id]
            ).as_matrix()
            world_from_body[:3, 3] = body_positions[body_id]
            asset_from_body = asset_from_world @ world_from_body
            body_visual_ops[body_name].Set(
                Gf.Matrix4d(*asset_from_body.T.reshape(-1).tolist())
            )
        world_from_doll = np.eye(4, dtype=np.float64)
        world_from_doll[:3, :3] = Rotation.from_quat(
            np.asarray(doll_quaternion_xyzw, dtype=np.float64)
        ).as_matrix()
        world_from_doll[:3, 3] = np.asarray(doll_position, dtype=np.float64)
        doll_parent_from_doll = doll_parent_from_world @ world_from_doll
        doll_visual_op.Set(Gf.Matrix4d(*doll_parent_from_doll.T.reshape(-1).tolist()))

    # Exact frame-zero reset used by the authoritative policy-free runner.
    sim.reset()
    target[0, joint_ids] = torch.as_tensor(
        commands[0], device=robot.device, dtype=torch.float32
    )
    robot.write_joint_state_to_sim(target, zero_velocity)
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    doll.write_root_pose_to_sim_index(
        root_pose=torch.as_tensor(
            np.r_[center, orientation][None], device=doll.device, dtype=torch.float32
        )
    )
    doll.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=doll.device, dtype=torch.float32)
    )
    robot.update(0.0)
    doll.update(0.0)
    propagate_recorded_state_to_viewport(center, orientation)
    for _ in range(8):
        render_and_pump()

    initial_measured = numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64)
    initial_object_position = center.astype(np.float64)
    display_link = stage.GetPrimAtPath(f"{G1}/left_hand_palm_link")
    if not display_link.IsValid():
        raise RuntimeError("viewport motion audit link is missing")
    xform_cache = UsdGeom.XformCache()
    initial_link_position = np.asarray(
        xform_cache.GetLocalToWorldTransform(display_link).ExtractTranslation(),
        dtype=np.float64,
    )
    maximum_measured_motion = 0.0
    maximum_object_motion = 0.0
    maximum_display_link_motion = 0.0
    failure_printed = False
    print("FULL_REPLAY_START", flush=True)
    replay_started = time.monotonic()
    if recorded_trace:
        source_frames = np.arange(0, len(commands), display_source_stride, dtype=np.int64)
        if source_frames[-1] != len(commands) - 1:
            source_frames = np.r_[source_frames, len(commands) - 1]
    else:
        source_frames = np.arange(len(commands), dtype=np.int64)
    for displayed_frame, frame_value in enumerate(source_frames):
        frame = int(frame_value)
        event_row = int(final_event_rows[frame])
        if not simulation_app.is_running():
            return 0
        measured_q = event["measured_q_rad"][event_row].astype(np.float64)
        measured_qd = event["measured_qd_rad_s"][event_row].astype(np.float64)
        target[0, joint_ids] = torch.as_tensor(
            measured_q, device=robot.device, dtype=torch.float32
        )
        replay_velocity = zero_velocity.clone()
        replay_velocity[0, joint_ids] = torch.as_tensor(
            measured_qd, device=robot.device, dtype=torch.float32
        )
        robot.write_joint_state_to_sim(target, replay_velocity)
        object_pose = np.r_[
            event["object_position_world_m"][event_row],
            event["object_quaternion_xyzw"][event_row],
        ]
        object_velocity = np.r_[
            event["object_linear_velocity_m_s"][event_row],
            event["object_angular_velocity_rad_s"][event_row],
        ]
        doll.write_root_pose_to_sim_index(
            root_pose=torch.as_tensor(
                object_pose[None], device=doll.device, dtype=torch.float32
            )
        )
        doll.write_root_velocity_to_sim_index(
            root_velocity=torch.as_tensor(
                object_velocity[None], device=doll.device, dtype=torch.float32
            )
        )
        robot.update(0.0)
        doll.update(0.0)
        propagate_recorded_state_to_viewport(object_pose[:3], object_pose[3:7])
        if args.capture_keyframes_dir is not None:
            pending = [key for key in capture_targets if key <= frame and key not in captured_targets]
            for key in pending:
                from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("active viewport unavailable for physical keyframe capture")
                output = args.capture_keyframes_dir / capture_targets[key]
                capture_helpers.append(capture_viewport_to_file(viewport, file_path=str(output)))
                captured_targets.add(key)
        # Every selected state is a displayed frame and is explicitly rendered.
        # A scheduled capture above therefore records this exact persisted state.
        render_and_pump()
        measured = numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64)
        maximum_measured_motion = max(
            maximum_measured_motion, float(np.max(np.abs(measured - initial_measured)))
        )
        maximum_object_motion = max(
            maximum_object_motion,
            float(np.linalg.norm(object_pose[:3] - initial_object_position)),
        )
        xform_cache.Clear()
        displayed_link_position = np.asarray(
            xform_cache.GetLocalToWorldTransform(display_link).ExtractTranslation(),
            dtype=np.float64,
        )
        maximum_display_link_motion = max(
            maximum_display_link_motion,
            float(np.linalg.norm(displayed_link_position - initial_link_position)),
        )
        if frame >= failure_frame and not failure_printed:
            print(
                f"FIRST_PHYSICAL_FAILURE frame={failure_frame} stage={failure_stage}",
                flush=True,
            )
            failure_printed = True
        if (
            args.validation_exit_after_frames
            and displayed_frame + 1 >= args.validation_exit_after_frames
        ):
            if maximum_measured_motion <= 1.0e-3:
                raise RuntimeError("viewport replay validation found no measured robot motion")
            if maximum_object_motion <= 1.0e-3:
                raise RuntimeError("viewport replay validation found no recorded doll motion")
            if maximum_display_link_motion <= 1.0e-3:
                raise RuntimeError("viewport USD transforms did not visibly update")
            return 0
        scheduled_elapsed = (displayed_frame + 1) / CONTROL_FPS
        remaining = scheduled_elapsed - (time.monotonic() - replay_started)
        if remaining > 0.0:
            time.sleep(remaining)

    if maximum_measured_motion <= 0.1:
        raise RuntimeError("complete replay did not produce visible-scale measured robot motion")
    if maximum_object_motion <= 0.05:
        raise RuntimeError("complete replay did not display the recorded doll motion")
    if maximum_display_link_motion <= 0.05:
        raise RuntimeError("complete replay did not update viewport robot transforms")
    if not failure_printed and not contact_final:
        print(
            f"FIRST_PHYSICAL_FAILURE frame={failure_frame} stage={failure_stage}",
            flush=True,
        )
    print("FULL_REPLAY_DONE", flush=True)

    if args.capture_keyframes_dir is not None:
        # Scheduled viewport captures finish on subsequent Kit updates.  The
        # trace state remains fixed while these rendering-only updates run.
        for _ in range(90):
            simulation_app.update()
        import importlib

        renderer_capture = importlib.import_module("omni.kit.renderer_capture")
        renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        missing = [
            str(args.capture_keyframes_dir / capture_targets[key])
            for key in capture_targets
            if not (args.capture_keyframes_dir / capture_targets[key]).is_file()
        ]
        if missing:
            raise RuntimeError(f"viewport keyframe capture incomplete: {missing}")
        print(f"PHYSICAL_KEYFRAMES_READY dir={args.capture_keyframes_dir}", flush=True)

    if args.validation_exit_after_replay:
        return 0

    # Preserve the final simulated pose.  Only pump Kit; physics and commands
    # remain stopped while the user freely inspects the viewport.
    while simulation_app.is_running():
        simulation_app.update()
        time.sleep(1.0 / 60.0)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
