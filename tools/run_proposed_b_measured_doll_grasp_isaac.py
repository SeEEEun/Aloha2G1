#!/usr/bin/env python3
"""Isaac diagnostic of the frozen Proposed-B LEFT_GRASP on the measured doll.

No policy, retargeting, IK, pose search, or parameter tuning is performed.  The
runner consumes the provenance-checked command prepared by
``prepare_proposed_b_measured_doll_grasp_diagnostic.py`` and executes only the
frozen B LEFT_GRASP -> HOLD -> first 50-mm lift prefix.  The measured 120 x 90
x 85 mm, 20 g proxy is bottom-registered at the original scene doll footprint.
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
import time
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
parser.add_argument(
    "--recovery",
    type=Path,
    default=ROOT / "outputs/proposed_b_measured_doll_left_grasp_diagnostic/FROZEN_TARGET_RECOVERY.json",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=ROOT / "outputs/proposed_b_measured_doll_left_grasp_diagnostic/isaac",
)
parser.add_argument("--gui", action="store_true", help="Launch visible Isaac GUI.")
parser.add_argument(
    "--keep-open",
    action="store_true",
    help="Keep the GUI at the final/abort pose for free-camera inspection.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.gui:
    args.headless = False
    args.headless_explicit = False
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
PROXY = f"{DOLL}/ProposedBMeasuredDiagnosticCollider"
VISUAL = f"{DOLL}/ProposedBMeasuredDiagnosticVisual"
G1 = "/World/G1/Asset"
TABLE = "/World/DollHandoffEnvironment/Table/Colliders/Top"
DIAGNOSTICS = "/World/ProposedBMeasuredDollDiagnostic"
DIGITS = ("thumb", "index", "middle")
COLORS = {
    "thumb": Gf.Vec3f(1.0, 0.2, 0.15),
    "index": Gf.Vec3f(0.1, 0.45, 1.0),
    "middle": Gf.Vec3f(1.0, 0.75, 0.05),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def longest_duration(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def rounded_oval_mesh(dimensions: np.ndarray) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
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


def apply_measured_proxy(stage: Usd.Stage, config: dict[str, Any]) -> dict[str, Any]:
    source = stage.GetPrimAtPath(SOURCE_BODY)
    if not source.IsValid():
        raise RuntimeError("source doll collider is missing")
    UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Set(False)
    for child in stage.GetPrimAtPath(DOLL).GetChildren():
        if child.GetPath() in {Sdf.Path(PROXY), Sdf.Path(VISUAL)}:
            continue
        if child.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(child).MakeInvisible()
    dimensions = np.asarray(config["object"]["visual_dimensions_m"], dtype=np.float64)
    points, counts, indices = rounded_oval_mesh(dimensions)
    visual = UsdGeom.Mesh.Define(stage, VISUAL)
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr(counts)
    visual.CreateFaceVertexIndicesAttr(indices)
    visual.CreateSubdivisionSchemeAttr("none")
    visual.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.72, 0.34)])
    collider = UsdGeom.Mesh.Define(stage, PROXY)
    collider.CreatePointsAttr(points)
    collider.CreateFaceVertexCountsAttr(counts)
    collider.CreateFaceVertexIndicesAttr(indices)
    collider.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Imageable(collider).MakeInvisible()
    prim = collider.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateContactOffsetAttr(float(config["object"]["contact_offset_m"]))
    physx_collision.CreateRestOffsetAttr(float(config["object"]["rest_offset_m"]))
    material = UsdShade.Material.Define(stage, f"{DIAGNOSTICS}/MeasuredMaterial")
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
        "visual_dimensions_m": dimensions,
        "collision_dimensions_m": dimensions,
        "collision_scale_percent": 100,
        "mass_kg": float(UsdPhysics.MassAPI(root).GetMassAttr().Get()),
        "static_friction": float(material_api.GetStaticFrictionAttr().Get()),
        "dynamic_friction": float(material_api.GetDynamicFrictionAttr().Get()),
        "contact_offset_m": float(physx_collision.GetContactOffsetAttr().Get()),
        "rest_offset_m": float(physx_collision.GetRestOffsetAttr().Get()),
        "source_collider_disabled": True,
        "session_layer_only": True,
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


def contact_rows(sensor: ContactSensor, dt: float) -> tuple[list[dict[str, Any]], str | None]:
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
        for owner_index, owner in enumerate(owners):
            for filter_index in range(counts_np.shape[1]):
                start = int(starts_np[owner_index, filter_index])
                count = int(counts_np[owner_index, filter_index])
                for index in range(start, start + count):
                    rows.append(
                        {
                            "owner": owner,
                            "force_n": float(abs(forces_np[index])),
                            "point": points_np[index].astype(np.float64),
                            "normal": normals_np[index].astype(np.float64),
                            "separation_m": float(separations_np[index]),
                        }
                    )
        return rows, None
    except Exception as error:
        return [], f"{type(error).__name__}: {error}"


def define_sphere(
    stage: Usd.Stage, path: str, radius: float, color: Gf.Vec3f, position: np.ndarray
) -> Any:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([color])
    return UsdGeom.Xformable(sphere).AddTranslateOp().Set(
        Gf.Vec3d(*map(float, position))
    ) or UsdGeom.Xformable(sphere)


def sphere_translate_op(stage: Usd.Stage, path: str) -> Any:
    prim = stage.GetPrimAtPath(path)
    operations = UsdGeom.Xformable(prim).GetOrderedXformOps()
    if not operations:
        return UsdGeom.Xformable(prim).AddTranslateOp()
    return operations[0]


def define_line(
    stage: Usd.Stage,
    path: str,
    start: np.ndarray,
    end: np.ndarray,
    color: Gf.Vec3f,
    width: float = 0.002,
) -> UsdGeom.BasisCurves:
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr(
        [Gf.Vec3f(*map(float, start)), Gf.Vec3f(*map(float, end))]
    )
    curve.CreateWidthsAttr([float(width), float(width)])
    curve.CreateDisplayColorAttr([color])
    return curve


def update_line(curve: UsdGeom.BasisCurves, start: np.ndarray, end: np.ndarray) -> None:
    curve.GetPointsAttr().Set(
        [Gf.Vec3f(*map(float, start)), Gf.Vec3f(*map(float, end))]
    )


def add_diagnostic_geometry(stage: Usd.Stage, recovery: dict[str, Any]) -> dict[str, Any]:
    UsdGeom.Xform.Define(stage, DIAGNOSTICS)
    frame = np.asarray(
        recovery["proposed_b"]["target_whole_hand_grasp_frame_pose_world"],
        dtype=np.float64,
    )
    origin = frame[:3, 3]
    for axis, color, index in (
        ("X", Gf.Vec3f(1.0, 0.0, 0.0), 0),
        ("Y", Gf.Vec3f(0.0, 1.0, 0.0), 1),
        ("Z", Gf.Vec3f(0.0, 0.0, 1.0), 2),
    ):
        define_line(
            stage,
            f"{DIAGNOSTICS}/ProposedBGraspFrame{axis}",
            origin,
            origin + 0.055 * frame[:3, index],
            color,
            0.0025,
        )
    handles: dict[str, Any] = {"contact_point": {}, "contact_normal": {}}
    for digit in DIGITS:
        row = recovery["digits"][digit]
        target = np.asarray(row["canonical_target_pad_position_world_m"], dtype=np.float64)
        surface = np.asarray(row["nearest_surface_position_world_m"], dtype=np.float64)
        normal = np.asarray(row["surface_outward_normal_world"], dtype=np.float64)
        define_sphere(stage, f"{DIAGNOSTICS}/{digit.title()}TargetPad", 0.0045, COLORS[digit], target)
        define_sphere(
            stage,
            f"{DIAGNOSTICS}/{digit.title()}TargetSurfacePoint",
            0.0025,
            Gf.Vec3f(0.95, 0.95, 0.95),
            surface,
        )
        define_line(
            stage,
            f"{DIAGNOSTICS}/{digit.title()}TargetSurfaceNormal",
            surface,
            surface + 0.025 * normal,
            COLORS[digit],
            0.0015,
        )
        contact_path = f"{DIAGNOSTICS}/{digit.title()}ActualContact"
        define_sphere(
            stage,
            contact_path,
            0.0035,
            Gf.Vec3f(1.0, 1.0, 1.0),
            np.asarray([0.0, 0.0, -10.0]),
        )
        handles["contact_point"][digit] = sphere_translate_op(stage, contact_path)
        handles["contact_normal"][digit] = define_line(
            stage,
            f"{DIAGNOSTICS}/{digit.title()}ActualContactNormal",
            np.asarray([0.0, 0.0, -10.0]),
            np.asarray([0.0, 0.0, -10.0]),
            COLORS[digit],
            0.002,
        )
    return handles


def main() -> int:
    recovery_path = args.recovery.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery = read_json(recovery_path)
    if recovery.get("status") != "PASS" or recovery.get("pose_search_or_tuning_used"):
        raise RuntimeError("invalid frozen target recovery")
    command_path = Path(recovery["command"]["path"]).resolve()
    if sha256_file(command_path) != recovery["command"]["sha256"]:
        raise RuntimeError("frozen diagnostic command hash mismatch")
    measured_path = ROOT / "configs/doll_handoff_measured_proxy_v3.json"
    measured = read_json(measured_path)
    with np.load(command_path, allow_pickle=False) as archive:
        command_archive = {key: np.asarray(archive[key]) for key in archive.files}
    commands = command_archive["commanded_q_rad"].astype(np.float64)
    command_names = command_archive["joint_names"].astype(str).tolist()
    source_frames = command_archive["source_frame_index"].astype(np.int64)
    labels = command_archive["stage"].astype(str)
    target_pads = command_archive["target_pad_position_world_m"].astype(np.float64)
    if commands.shape != (len(labels), 28) or not np.isfinite(commands).all():
        raise RuntimeError("invalid frozen B command")

    if not omni.usd.get_context().open_stage(measured["source_scene"]):
        raise RuntimeError("failed to open Doll-Handoff source scene")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    runtime_proxy = apply_measured_proxy(stage, measured)
    marker_handles = add_diagnostic_geometry(stage, recovery)

    dt = float(measured["timing"]["physics_dt_s"])
    substeps = int(measured["timing"]["physics_substeps_per_control_frame"])
    fps = float(measured["timing"]["control_fps_hz"])
    if not np.isclose(dt * substeps, 1.0 / fps):
        raise RuntimeError("physics/control timing mismatch")
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=str(measured["simulation"]["device"]),
            gravity=(0.0, 0.0, -float(measured["simulation"]["gravity_m_s2"])),
            use_fabric=bool(measured["simulation"]["use_fabric"]),
        )
    )
    if args.gui:
        sim.set_camera_view(
            eye=np.asarray([0.42, -0.72, 1.20]),
            target=np.asarray([0.11, 0.06, 0.84]),
        )
    robot = Articulation(
        ArticulationCfg(
            prim_path=f"{G1}/root_joint",
            spawn=None,
            actuators=build_actuators(measured),
        )
    )
    doll = RigidObject(RigidObjectCfg(prim_path=DOLL, spawn=None))
    whole_hand = read_json(Path(measured["whole_hand_geometry"]))
    distal_links = {
        row["digit_chain"]: row["distal_link"]
        for role in ("A", "B", "C")
        for row in (whole_hand["left"][role],)
    }
    digit_sensors = {
        digit: ContactSensor(
            ContactSensorCfg(
                prim_path=f"{G1}/{distal_links[digit]}",
                update_period=0.0,
                filter_prim_paths_expr=[DOLL],
                track_contact_points=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
        for digit in DIGITS
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
    sim.reset()

    isaac_names = list(robot.data.joint_names)
    missing = [name for name in command_names if name not in isaac_names]
    if missing:
        raise RuntimeError(f"Isaac named joint mapping failed: {missing}")
    joint_ids = [isaac_names.index(name) for name in command_names]
    if len(joint_ids) != 28 or len(set(joint_ids)) != 28:
        raise RuntimeError("Isaac 28D named joint mapping is not bijective")
    authoritative_names, _ = authoritative_joint_ranges()
    if set(authoritative_names) != set(command_names):
        raise RuntimeError("frozen B command differs from authoritative 28D joint set")

    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    target[0, joint_ids] = torch.as_tensor(commands[0], device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    center = np.asarray(
        recovery["measured_object_registration"]["object_center_world_m"], dtype=np.float32
    )
    root_pose = torch.as_tensor(
        np.r_[center, [0.0, 0.0, 0.0, 1.0]][None],
        device=doll.device,
        dtype=torch.float32,
    )
    doll.write_root_pose_to_sim_index(root_pose=root_pose)
    doll.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), device=doll.device, dtype=torch.float32)
    )
    # Read registration before the first physics/contact impulse.  The first
    # integration step is intentionally allowed to expose target-induced
    # penetration/ejection and must not be mislabeled as a spawn transform error.
    doll.update(dt)
    prephysics_object_pose = numpy(doll.data.root_pose_w)[0].astype(np.float64)
    prephysics_pose_error = float(np.linalg.norm(prephysics_object_pose[:3] - center))

    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "physics_step",
            "control_frame",
            "source_frame",
            "stage",
            "commanded_q_rad",
            "measured_q_rad",
            "object_position_world_m",
            "object_linear_velocity_m_s",
            "object_angular_velocity_rad_s",
            "table_contact_count",
            "maximum_digit_penetration_m",
        )
    }
    for digit in DIGITS:
        for suffix in (
            "force_n",
            "contact_count",
            "contact_point_world_m",
            "contact_normal_world",
            "contact_position_error_to_target_pad_m",
        ):
            records[f"{digit}_{suffix}"] = []
    api_errors: set[str] = set()
    contact_pairs: set[str] = set()
    physics_step = 0

    def update() -> None:
        robot.update(dt)
        doll.update(dt)
        for sensor in digit_sensors.values():
            sensor.update(dt, force_recompute=True)
        table_sensor.update(dt, force_recompute=True)

    for control_frame, (command, source_frame, label) in enumerate(
        zip(commands, source_frames, labels, strict=True)
    ):
        target[0, joint_ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        for _ in range(substeps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=bool(args.gui))
            update()
            object_pose = numpy(doll.data.root_pose_w)[0].astype(np.float64)
            object_velocity = numpy(doll.data.root_vel_w)[0].astype(np.float64)
            measured_q = numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64)
            digit_values: dict[str, dict[str, Any]] = {}
            penetration: list[float] = []
            for digit_index, digit in enumerate(DIGITS):
                rows, error = contact_rows(digit_sensors[digit], dt)
                if error:
                    api_errors.add(error)
                for row in rows:
                    contact_pairs.add(f"{row['owner']} <-> {DOLL}")
                    penetration.append(max(0.0, -float(row["separation_m"])))
                force = float(sum(abs(float(row["force_n"])) for row in rows))
                if rows:
                    weights = np.asarray(
                        [max(abs(float(row["force_n"])), 1.0e-12) for row in rows]
                    )
                    point = np.average(
                        np.stack([row["point"] for row in rows]), axis=0, weights=weights
                    )
                    normal_sum = np.sum(
                        np.stack([row["normal"] for row in rows]) * weights[:, None], axis=0
                    )
                    normal = normal_sum / max(float(np.linalg.norm(normal_sum)), 1.0e-12)
                    error_to_target = float(
                        np.linalg.norm(point - target_pads[control_frame, digit_index])
                    )
                    marker_handles["contact_point"][digit].Set(
                        Gf.Vec3d(*map(float, point))
                    )
                    update_line(
                        marker_handles["contact_normal"][digit],
                        point,
                        point + 0.025 * normal,
                    )
                else:
                    point = np.full(3, np.nan)
                    normal = np.zeros(3)
                    error_to_target = float("nan")
                    marker_handles["contact_point"][digit].Set(Gf.Vec3d(0.0, 0.0, -10.0))
                    update_line(
                        marker_handles["contact_normal"][digit],
                        np.asarray([0.0, 0.0, -10.0]),
                        np.asarray([0.0, 0.0, -10.0]),
                    )
                digit_values[digit] = {
                    "force_n": force,
                    "contact_count": len(rows),
                    "contact_point_world_m": point,
                    "contact_normal_world": normal,
                    "contact_position_error_to_target_pad_m": error_to_target,
                }
            table_rows, table_error = contact_rows(table_sensor, dt)
            if table_error:
                api_errors.add(table_error)
            values = {
                "physics_step": physics_step,
                "control_frame": control_frame,
                "source_frame": source_frame,
                "stage": str(label),
                "commanded_q_rad": command,
                "measured_q_rad": measured_q,
                "object_position_world_m": object_pose[:3],
                "object_linear_velocity_m_s": object_velocity[:3],
                "object_angular_velocity_rad_s": object_velocity[3:],
                "table_contact_count": len(table_rows),
                "maximum_digit_penetration_m": max(penetration, default=0.0),
            }
            for digit in DIGITS:
                for suffix, value in digit_values[digit].items():
                    values[f"{digit}_{suffix}"] = value
            for key, value in values.items():
                records[key].append(value)
            physics_step += 1

    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["timestamp_s"] = arrays["physics_step"].astype(np.float64) * dt
    arrays["joint_names"] = np.asarray(command_names)
    event_log = output_dir / "event_log.npz"
    atomic_npz(event_log, **arrays)

    meaningful_threshold = float(measured["gates"]["meaningful_digit_force_n"])
    digits: dict[str, Any] = {}
    all_meaningful = True
    for digit in DIGITS:
        counts = arrays[f"{digit}_contact_count"] > 0
        forces = arrays[f"{digit}_force_n"]
        meaningful = forces >= meaningful_threshold
        errors = arrays[f"{digit}_contact_position_error_to_target_pad_m"]
        finite_error = np.isfinite(errors) & counts
        force_weight = np.where(finite_error, np.maximum(forces, 1.0e-12), 0.0)
        position_error = (
            float(np.sum(np.where(finite_error, errors, 0.0) * force_weight) / np.sum(force_weight))
            if np.sum(force_weight) > 0.0
            else None
        )
        duration = longest_duration(counts, dt)
        meaningful_duration = longest_duration(meaningful, dt)
        is_meaningful = bool(meaningful_duration >= 0.10)
        all_meaningful &= is_meaningful
        contact_points = arrays[f"{digit}_contact_point_world_m"]
        contact_normals = arrays[f"{digit}_contact_normal_world"]
        digits[digit] = {
            "target_pad_to_surface_signed_distance_m": recovery["digits"][digit][
                "target_pad_to_surface_signed_distance_m"
            ],
            "frozen_q_pad_to_surface_signed_distance_m": recovery["digits"][digit][
                "frozen_q_pad_to_surface_signed_distance_m"
            ],
            "whole_hand_pad_realization_error_before_physics_m": recovery["digits"][digit][
                "whole_hand_pad_realization_error_m"
            ],
            "actual_contact": bool(np.any(counts)),
            "meaningful_contact": is_meaningful,
            "contact_position_error_m": position_error,
            "maximum_normal_force_n": float(np.max(forces, initial=0.0)),
            "mean_normal_force_while_contact_n": (
                float(np.mean(forces[counts])) if np.any(counts) else 0.0
            ),
            "contact_duration_s": duration,
            "meaningful_contact_duration_s": meaningful_duration,
            "mean_contact_position_world_m": (
                np.nanmean(contact_points[counts], axis=0) if np.any(counts) else None
            ),
            "mean_surface_contact_normal_world": (
                np.mean(contact_normals[counts], axis=0) if np.any(counts) else None
            ),
        }

    positions = arrays["object_position_world_m"]
    first_step_displacement = float(np.linalg.norm(positions[0] - center))
    maximum_rise = float(np.max(positions[:, 2]) - positions[0, 2])
    final_rise = float(positions[-1, 2] - positions[0, 2])
    max_speed = float(
        np.max(np.linalg.norm(arrays["object_linear_velocity_m_s"], axis=1), initial=0.0)
    )
    max_penetration = float(np.max(arrays["maximum_digit_penetration_m"], initial=0.0))
    final_mask = arrays["stage"].astype(str) == "ELEVATED_HOLD"
    final_contact = np.zeros(len(arrays["stage"]), dtype=bool)
    for digit in DIGITS:
        final_contact |= arrays[f"{digit}_contact_count"] > 0
    elevated_retention = bool(
        maximum_rise >= 0.045
        and longest_duration(final_mask & final_contact, dt) >= 0.50
        and final_rise >= 0.040
    )
    registration_error = abs(
        float(
            recovery["measured_object_registration"][
                "source_interaction_origin_signed_distance_to_surface_m"
            ]
        )
    )
    registration_pass = bool(registration_error <= 0.005 and prephysics_pose_error <= 0.0005)
    if not registration_pass:
        classification = "OBJECT_FRAME_REGISTRATION_ERROR"
    elif not all_meaningful:
        classification = "WHOLE_HAND_GRASP_REALIZATION_ERROR"
    elif not elevated_retention:
        classification = "CONTACT_PHYSICS_RETENTION_ERROR"
    else:
        classification = "B_GRASP_PHYSICALLY_VALID"

    report = {
        "schema_version": "proposed_b_measured_doll_left_grasp_isaac_diagnostic_v1",
        "status": "COMPLETE",
        "classification": classification,
        "episode": recovery["representative_selection"],
        "execution": "B LEFT_GRASP target -> frozen stable HOLD -> frozen prefix reaching 50 mm -> elevated HOLD",
        "pose_search_or_tuning_used": False,
        "retargeting_or_dataset_modified": False,
        "policy_used": False,
        "measured_doll": {
            "dimensions_m": runtime_proxy["visual_dimensions_m"],
            "mass_kg": runtime_proxy["mass_kg"],
            "collision_scale_percent": 100,
            "object_center_world_m": center,
            "registration": recovery["measured_object_registration"],
            "prephysics_sim_pose_error_m": prephysics_pose_error,
            "first_physics_step_displacement_m": first_step_displacement,
            "registration_gate": "PASS" if registration_pass else "FAIL",
        },
        "digits": digits,
        "lift": {
            "maximum_object_rise_m": maximum_rise,
            "final_object_rise_m": final_rise,
            "elevated_retention": elevated_retention,
            "commanded_static_whole_hand_rise_m": float(
                np.max(
                    command_archive["frozen_q_pad_position_world_m"][:, :, 2]
                    - command_archive["frozen_q_pad_position_world_m"][0, :, 2],
                    initial=0.0,
                )
            ),
        },
        "physics": {
            "material": measured["material"],
            "finger_drive": measured["finger_drive"],
            "gravity_m_s2": measured["simulation"]["gravity_m_s2"],
            "physics_dt_s": dt,
            "control_fps_hz": fps,
            "maximum_object_linear_speed_m_s": max_speed,
            "maximum_digit_penetration_m": max_penetration,
            "contact_api_errors": sorted(api_errors),
            "collision_contact_pairs": sorted(contact_pairs),
            "attachments_or_constraints_used": False,
        },
        "artifacts": {
            "recovery": str(recovery_path),
            "recovery_sha256": sha256_file(recovery_path),
            "command": str(command_path),
            "command_sha256": sha256_file(command_path),
            "event_log": str(event_log),
            "event_log_sha256": sha256_file(event_log),
            "source_scene": measured["source_scene"],
            "source_scene_sha256": sha256_file(Path(measured["source_scene"])),
            "measured_config": str(measured_path),
            "measured_config_sha256": sha256_file(measured_path),
        },
        "gui": {
            "target_grasp_frame_visible": True,
            "target_pad_positions_visible": True,
            "target_surface_normals_visible": True,
            "actual_contact_positions_visible": True,
            "actual_contact_normals_visible": True,
            "keep_open_requested": bool(args.keep_open),
        },
        "real_robot": False,
    }
    report_path = output_dir / "DIAGNOSTIC_RESULT.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    print(f"FAILURE_CLASS={classification}")
    if args.gui and args.keep_open:
        print("GUI_FINAL_POSE_READY_FOR_FREE_CAMERA_INSPECTION")
        while simulation_app.is_running():
            sim.render()
            time.sleep(0.01)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
