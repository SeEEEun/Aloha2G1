#!/usr/bin/env python3
"""Run one policy-independent Dex3 rigid-proxy contact stage in Isaac.

Every invocation starts from the same composed G1 scene, authors overrides only
in the anonymous USD session layer, and logs at the 120 Hz physics rate.  There
is deliberately no policy, dataset, checkpoint, DDS, or real-robot interface.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
parser.add_argument("--shape", required=True)
parser.add_argument("--side", choices=("left", "right"), required=True)
parser.add_argument(
    "--stage",
    choices=(
        "SINGLE_FINGERTIP_NO_GRAVITY",
        "SINGLE_FINGERTIP_PUSH_GRAVITY",
        "FIXED_ARM_THREE_FINGER_CLOSE",
        "STATIC_HOLD",
        "VERTICAL_LIFT_5CM",
        "CLOSURE_TO_GRAVITY_RETENTION",
        "CLOSURE_TO_GRAVITY_RETENTION_AND_LIFT_5CM",
    ),
    required=True,
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--retention-config",
    type=Path,
    default=None,
    help="Optional frozen capsule retention contract containing bounded conditions.",
)
parser.add_argument(
    "--condition",
    default="baseline",
    help="Named condition from --retention-config (baseline by default).",
)
parser.add_argument(
    "--initial-state-from",
    type=Path,
    default=None,
    help="Continue from the final physics state of a passed lower-level stage log.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import carb
import omni.physx
import omni.usd
import torch
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.evaluation.contracts import authoritative_joint_ranges
from tools.policy_b_isaac_control_contract import CONTROLLER_CONTRACT, build_implicit_actuators


DOLL = "/World/DollHandoffEnvironment/Doll"
SOURCE_BODY = f"{DOLL}/Body"
PROXY = f"{DOLL}/DiagnosticProxy"
TABLE = "/World/DollHandoffEnvironment/Table/Colliders/Top"
G1 = "/World/G1/Asset"
CONTACT_FIELDS = [
    "physics_step",
    "sim_time_s",
    "control_frame",
    "stage",
    "role",
    "event_type",
    "actor0",
    "actor1",
    "collider0",
    "collider1",
    "contact_x",
    "contact_y",
    "contact_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "impulse_x",
    "impulse_y",
    "impulse_z",
    "normal_force_n",
    "normal_impulse_ns",
    "relative_normal_velocity_m_s",
    "relative_tangential_velocity_m_s",
    "separation_m",
    "penetration_m",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
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


def normalized(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def candidate_support(candidate: dict[str, Any]) -> float:
    if candidate["type"] == "sphere":
        return float(candidate["radius_m"])
    axis = normalized(candidate["major_axis_world_xyz"])
    if candidate["type"] == "ellipsoid":
        major = float(candidate["major_semi_axis_m"])
        minor = float(candidate["minor_semi_axis_m"])
        return math.sqrt((major * axis[2]) ** 2 + minor**2 * (1.0 - axis[2] ** 2))
    if candidate["type"] == "capsule":
        return float(candidate["radius_m"]) + 0.5 * float(
            candidate["cylinder_height_m"]
        ) * abs(float(axis[2]))
    raise ValueError(candidate["type"])


def quaternion_from_z(axis: np.ndarray) -> Gf.Quatf:
    target = normalized(axis)
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot < -0.999999:
        return Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0))
    cross = np.cross(source, target)
    w = math.sqrt(0.5 * (1.0 + dot))
    xyz = cross / (2.0 * w)
    return Gf.Quatf(float(w), Gf.Vec3f(*map(float, xyz)))


def ellipsoid_mesh(candidate: dict[str, Any]) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    major = float(candidate["major_semi_axis_m"])
    minor = float(candidate["minor_semi_axis_m"])
    axis0 = normalized(candidate["major_axis_world_xyz"])
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(helper @ axis0)) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    axis1 = normalized(np.cross(axis0, helper))
    axis2 = normalized(np.cross(axis0, axis1))
    latitudes, longitudes = 8, 16
    points: list[np.ndarray] = [major * axis0]
    for latitude in range(1, latitudes):
        theta = math.pi * latitude / latitudes
        for longitude in range(longitudes):
            phi = 2.0 * math.pi * longitude / longitudes
            point = (
                major * math.cos(theta) * axis0
                + minor * math.sin(theta) * math.cos(phi) * axis1
                + minor * math.sin(theta) * math.sin(phi) * axis2
            )
            points.append(point)
    points.append(-major * axis0)
    north, south = 0, len(points) - 1
    faces: list[list[int]] = []
    first_ring = 1
    for longitude in range(longitudes):
        faces.append([north, first_ring + longitude, first_ring + (longitude + 1) % longitudes])
    for latitude in range(latitudes - 2):
        first = 1 + latitude * longitudes
        second = first + longitudes
        for longitude in range(longitudes):
            following = (longitude + 1) % longitudes
            faces.append(
                [first + longitude, second + longitude, second + following, first + following]
            )
    last_ring = 1 + (latitudes - 2) * longitudes
    for longitude in range(longitudes):
        faces.append([last_ring + longitude, south, last_ring + (longitude + 1) % longitudes])
    return (
        [Gf.Vec3f(*map(float, point)) for point in points],
        [len(face) for face in faces],
        [index for face in faces for index in face],
    )


def apply_proxy(stage: Usd.Stage, config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    source = stage.GetPrimAtPath(SOURCE_BODY)
    if not source.IsValid():
        raise RuntimeError("source doll Body is missing")
    UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Set(False)
    for child in stage.GetPrimAtPath(DOLL).GetChildren():
        if child.GetPath() == Sdf.Path(PROXY):
            continue
        if child.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(child).MakeInvisible()
    kind = candidate["type"]
    if kind == "sphere":
        shape = UsdGeom.Sphere.Define(stage, PROXY)
        shape.CreateRadiusAttr(float(candidate["radius_m"]))
        prim = shape.GetPrim()
    elif kind == "ellipsoid":
        shape = UsdGeom.Mesh.Define(stage, PROXY)
        points, counts, indices = ellipsoid_mesh(candidate)
        shape.CreatePointsAttr(points)
        shape.CreateFaceVertexCountsAttr(counts)
        shape.CreateFaceVertexIndicesAttr(indices)
        shape.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
        prim = shape.GetPrim()
    elif kind == "capsule":
        shape = UsdGeom.Capsule.Define(stage, PROXY)
        shape.CreateRadiusAttr(float(candidate["radius_m"]))
        shape.CreateHeightAttr(float(candidate["cylinder_height_m"]))
        shape.CreateAxisAttr("Z")
        UsdGeom.Xformable(shape.GetPrim()).AddOrientOp().Set(
            quaternion_from_z(np.asarray(candidate["major_axis_world_xyz"], dtype=np.float64))
        )
        prim = shape.GetPrim()
    else:
        raise ValueError(kind)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.72, 0.28)])
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(True)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    physx_collision.CreateContactOffsetAttr(float(config["object"]["contact_offset_m"]))
    physx_collision.CreateRestOffsetAttr(float(config["object"]["rest_offset_m"]))
    material = UsdShade.Material.Define(stage, "/World/Dex3ContactDiagnosticMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    fixed = config["fixed_material"]
    material_api.CreateStaticFrictionAttr(float(fixed["static_friction"]))
    material_api.CreateDynamicFrictionAttr(float(fixed["dynamic_friction"]))
    material_api.CreateRestitutionAttr(float(config["object"]["restitution"]))
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(str(fixed["friction_combine_mode"]))
    physx_material.CreateRestitutionCombineModeAttr().Set(
        str(fixed["restitution_combine_mode"])
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
        "source_body_collision_disabled": True,
        "diagnostic_proxy_path": PROXY,
        "diagnostic_proxy_type": prim.GetTypeName(),
        "diagnostic_proxy_local_transform": np.asarray(
            UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=np.float64
        ),
        "visual_and_collision_are_same_prim": True,
        "static_friction": float(fixed["static_friction"]),
        "dynamic_friction": float(fixed["dynamic_friction"]),
        "contact_offset_m": float(physx_collision.GetContactOffsetAttr().Get()),
        "rest_offset_m": float(physx_collision.GetRestOffsetAttr().Get()),
        "friction_tuned": str(fixed.get("name", "")) not in (
            "MEDIUM_PREDECLARED_NOT_TUNED",
            "BASELINE_055_045",
        ),
    }


def apply_collision_isolation(stage: Usd.Stage, side: str, stage_name: str, link: str) -> dict[str, Any]:
    colliders: list[str] = []
    disabled: list[str] = []
    selected = f"{G1}/{link}/collisions"
    isolate = stage_name.startswith("SINGLE_FINGERTIP")
    for prim in Usd.PrimRange(stage.GetPrimAtPath(G1)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        colliders.append(path)
        if isolate and path != selected:
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            disabled.append(path)
    table_prim = stage.GetPrimAtPath(TABLE)
    table_enabled = stage_name != "SINGLE_FINGERTIP_NO_GRAVITY"
    UsdPhysics.CollisionAPI(table_prim).GetCollisionEnabledAttr().Set(table_enabled)
    if selected not in colliders:
        raise RuntimeError(f"selected fingertip collider missing: {selected}")
    return {
        "isolated_single_fingertip": isolate,
        "selected_fingertip_collider": selected,
        "robot_collider_count": len(colliders),
        "disabled_robot_collider_count": len(disabled),
        "disabled_robot_colliders": disabled,
        "table_collision_enabled": table_enabled,
        "full_robot_colliders_enabled_for_three_finger_stages": not isolate,
    }


def apply_contact_reporting(stage: Usd.Stage) -> list[str]:
    paths: list[str] = []
    for root_path in (G1, DOLL, "/World/DollHandoffEnvironment/Table"):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
            paths.append(str(prim.GetPath()))
    return sorted(set(paths))


class ExactContactLogger:
    def __init__(self, path: Path, physics_dt: float):
        self.dt = float(physics_dt)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=CONTACT_FIELDS)
        self.writer.writeheader()
        self.physics_step = 0
        self.sim_time = 0.0
        self.control_frame = 0
        self.stage = "INIT"
        self.latest: list[dict[str, Any]] = []
        self.api_error: str | None = None
        self.header_count = 0
        self.header_preview: list[dict[str, Any]] = []
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
            self._callback
        )

    @staticmethod
    def path(encoded: Any) -> str:
        try:
            return str(PhysicsSchemaTools.intToSdfPath(encoded))
        except Exception:
            return str(encoded)

    def set_context(self, physics_step: int, control_frame: int, stage: str) -> None:
        self.physics_step = int(physics_step)
        self.sim_time = self.physics_step * self.dt
        self.control_frame = int(control_frame)
        self.stage = str(stage)
        self.latest = []

    def _callback(self, headers: Any, data: Any) -> None:
        try:
            for header in headers:
                actor0, actor1 = self.path(header.actor0), self.path(header.actor1)
                collider0, collider1 = self.path(header.collider0), self.path(header.collider1)
                self.header_count += 1
                if len(self.header_preview) < 32:
                    self.header_preview.append(
                        {
                            "actor0": actor0,
                            "actor1": actor1,
                            "collider0": collider0,
                            "collider1": collider1,
                            "event_type": str(header.type),
                            "num_contact_data": int(header.num_contact_data),
                        }
                    )
                combined = " ".join((actor0, actor1, collider0, collider1))
                if DOLL not in combined:
                    continue
                if G1 not in combined and "/Table" not in combined:
                    continue
                for index in range(
                    int(header.contact_data_offset),
                    int(header.contact_data_offset + header.num_contact_data),
                ):
                    item = data[index]
                    point = np.asarray(item.position, dtype=np.float64)
                    normal = np.asarray(item.normal, dtype=np.float64)
                    impulse = np.asarray(item.impulse, dtype=np.float64)
                    separation = float(item.separation)
                    row = {
                        "physics_step": self.physics_step,
                        "sim_time_s": f"{self.sim_time:.9f}",
                        "control_frame": self.control_frame,
                        "stage": self.stage,
                        "event_type": str(header.type),
                        "actor0": actor0,
                        "actor1": actor1,
                        "collider0": collider0,
                        "collider1": collider1,
                        "contact_x": f"{point[0]:.9g}",
                        "contact_y": f"{point[1]:.9g}",
                        "contact_z": f"{point[2]:.9g}",
                        "normal_x": f"{normal[0]:.9g}",
                        "normal_y": f"{normal[1]:.9g}",
                        "normal_z": f"{normal[2]:.9g}",
                        "impulse_x": f"{impulse[0]:.9g}",
                        "impulse_y": f"{impulse[1]:.9g}",
                        "impulse_z": f"{impulse[2]:.9g}",
                        "normal_force_n": f"{abs(float(impulse @ normal)) / self.dt:.9g}",
                        "separation_m": f"{separation:.9g}",
                        "penetration_m": f"{max(0.0, -separation):.9g}",
                    }
                    self.writer.writerow(row)
                    self.latest.append(row)
        except Exception as error:
            self.api_error = f"{type(error).__name__}: {error}"

    def close(self) -> None:
        self.subscription = None
        self.file.flush()
        self.file.close()


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def filtered_force(sensor: ContactSensor) -> float:
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        return 0.0
    values = numpy(matrix)
    return float(np.max(np.linalg.norm(values.reshape(-1, 3), axis=-1))) if values.size else 0.0


def tensor_contact_rows(
    sensor: ContactSensor,
    *,
    physics_dt: float,
    owner_fallback: str,
    other_path: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Read pair-resolved contact points from the IsaacLab tensor view."""
    try:
        forces, points, normals, separations, counts, starts = sensor.contact_view.get_contact_data(
            physics_dt
        )
        forces_np = numpy(forces).reshape(-1)
        points_np = numpy(points).reshape(-1, 3)
        normals_np = numpy(normals).reshape(-1, 3)
        separations_np = numpy(separations).reshape(-1)
        owners = list(sensor.body_physx_view.prim_paths[: sensor.num_sensors])
        if not owners:
            owners = [owner_fallback]
        counts_np = numpy(counts).reshape(len(owners), -1).astype(np.int64)
        starts_np = numpy(starts).reshape(len(owners), -1).astype(np.int64)
        rows: list[dict[str, Any]] = []
        for owner_index, owner in enumerate(owners):
            for filter_index in range(counts_np.shape[1]):
                start = int(starts_np[owner_index, filter_index])
                count = int(counts_np[owner_index, filter_index])
                for contact_index in range(start, start + count):
                    rows.append(
                        {
                            "owner": str(owner),
                            "other": other_path,
                            "force_n": float(abs(forces_np[contact_index])),
                            "point": points_np[contact_index].astype(np.float64),
                            "normal": normals_np[contact_index].astype(np.float64),
                            "separation_m": float(separations_np[contact_index]),
                        }
                    )
        return rows, None
    except Exception as error:
        return [], f"{type(error).__name__}: {error}"


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    # IsaacLab 3.x exposes native PhysX/Warp XYZW quaternions.
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def point_proxy_signed_distance(
    point_world: np.ndarray,
    object_position: np.ndarray,
    object_quaternion_xyzw: np.ndarray,
    candidate: dict[str, Any],
) -> float:
    local = rotation_xyzw(object_quaternion_xyzw).T @ (point_world - object_position)
    kind = candidate["type"]
    if kind == "sphere":
        return float(np.linalg.norm(local) - float(candidate["radius_m"]))
    axis = normalized(candidate["major_axis_world_xyz"])
    if kind == "ellipsoid":
        major = float(candidate["major_semi_axis_m"])
        minor = float(candidate["minor_semi_axis_m"])
        parallel = float(local @ axis)
        perpendicular = local - parallel * axis
        direction_norm = float(np.linalg.norm(local))
        if direction_norm <= 1.0e-12:
            return -minor
        direction = local / direction_norm
        d_parallel = float(direction @ axis)
        radial = 1.0 / math.sqrt(
            d_parallel**2 / major**2 + (1.0 - d_parallel**2) / minor**2
        )
        return direction_norm - radial
    if kind == "capsule":
        half = 0.5 * float(candidate["cylinder_height_m"])
        closest = np.clip(float(local @ axis), -half, half) * axis
        return float(np.linalg.norm(local - closest) - float(candidate["radius_m"]))
    raise ValueError(kind)


def longest_duration(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * dt


def build_condition_actuators(config: dict[str, Any]) -> dict[str, Any]:
    """Build the common controller with only the predeclared Dex3 drive override."""
    actuator_values = copy.deepcopy(CONTROLLER_CONTRACT["actuators"])
    drive = config["finger_drive"]
    actuator_values["dex3"].update(
        {
            "effort_limit_sim": float(drive["effort_limit_sim"]),
            "velocity_limit_sim": float(drive["velocity_limit_sim"]),
            "stiffness": float(drive["kp"]),
            "damping": float(drive["kd"]),
        }
    )
    return {name: ImplicitActuatorCfg(**values) for name, values in actuator_values.items()}


def build_commands(
    primitive: dict[str, np.ndarray],
    stage_name: str,
    retention_contract: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    commands = np.asarray(primitive["commanded_q_rad"], dtype=np.float64)
    stages = primitive["stage"].astype(str)
    close_end = int(np.flatnonzero(stages == "CLOSE_GRASP")[-1])
    hold_end = int(np.flatnonzero(stages == "HOLD_ON_TABLE")[-1])
    fps = float(primitive["fps"].reshape(()))
    if stage_name in ("SINGLE_FINGERTIP_NO_GRAVITY", "SINGLE_FINGERTIP_PUSH_GRAVITY"):
        end = hold_end
        return commands[: end + 1], stages[: end + 1]
    if stage_name == "FIXED_ARM_THREE_FINGER_CLOSE":
        count = max(1, int(round(0.25 * fps)))
        close = np.flatnonzero(stages == "CLOSE_GRASP")
        return (
            np.concatenate(
            ((commands[0][None], commands[close], np.repeat(commands[close_end][None], count, axis=0)))
            ),
            np.concatenate(
                ((np.asarray(["RESET_OPEN"]), stages[close], np.repeat("POST_CLOSE", count)))
            ),
        )
    if stage_name == "STATIC_HOLD":
        count = max(1, int(round(1.0 * fps)))
        return (
            np.repeat(commands[hold_end][None], count, axis=0),
            np.repeat("STATIC_HOLD", count),
        )
    if stage_name == "VERTICAL_LIFT_5CM":
        target = np.asarray(primitive["target_whole_hand_position_world_m"], dtype=np.float64)
        lift = np.flatnonzero(stages == "LIFT")
        baseline = float(target[hold_end, 2])
        requested = 0.05
        upper_candidates = lift[target[lift, 2] - baseline >= requested]
        if not len(upper_candidates):
            raise RuntimeError("source primitive never reaches 5 cm")
        upper = int(upper_candidates[0])
        lower = upper - 1
        low_delta = float(target[lower, 2] - baseline)
        high_delta = float(target[upper, 2] - baseline)
        alpha = (requested - low_delta) / (high_delta - low_delta)
        q5 = commands[lower] + alpha * (commands[upper] - commands[lower])
        prehold_count = max(1, int(round(0.5 * fps)))
        lift_commands = np.concatenate(
            (
                np.repeat(commands[hold_end][None], prehold_count, axis=0),
                commands[hold_end + 1 : lower + 1],
                q5[None],
            ),
            axis=0,
        )
        lift_labels = np.concatenate(
            (
                np.repeat("PRE_LIFT_STATIC_HOLD", prehold_count),
                stages[hold_end + 1 : lower + 1],
                np.asarray(["LIFT_5CM"]),
            )
        )
        hold_count = max(1, int(round(0.75 * fps)))
        return (
            np.concatenate((lift_commands, np.repeat(q5[None], hold_count, axis=0))),
            np.concatenate((lift_labels, np.repeat("HOLD_ELEVATED_5CM", hold_count))),
        )
    if stage_name in (
        "CLOSURE_TO_GRAVITY_RETENTION",
        "CLOSURE_TO_GRAVITY_RETENTION_AND_LIFT_5CM",
    ):
        if retention_contract is None:
            raise RuntimeError(f"{stage_name} requires --retention-config")
        execution = retention_contract["grasp_execution"]
        close = np.flatnonzero(stages == "CLOSE_GRASP")
        post_count = max(1, int(round(float(execution["no_gravity_post_close_s"]) * fps)))
        retention_count = max(1, int(round(float(execution["gravity_retention_s"]) * fps)))
        retention_commands = np.concatenate(
            (
                commands[0][None],
                commands[close],
                np.repeat(commands[close_end][None], post_count, axis=0),
                np.repeat(commands[hold_end][None], retention_count, axis=0),
            ),
            axis=0,
        )
        retention_labels = np.concatenate(
            (
                np.asarray(["RESET_OPEN"]),
                stages[close],
                np.repeat("POST_CLOSE_NO_GRAVITY", post_count),
                np.repeat("GRAVITY_RETENTION", retention_count),
            )
        )
        if stage_name == "CLOSURE_TO_GRAVITY_RETENTION":
            return retention_commands, retention_labels
        target = np.asarray(primitive["target_whole_hand_position_world_m"], dtype=np.float64)
        lift = np.flatnonzero(stages == "LIFT")
        baseline = float(target[hold_end, 2])
        requested = float(execution["lift_height_m"])
        upper_candidates = lift[target[lift, 2] - baseline >= requested]
        if not len(upper_candidates):
            raise RuntimeError("source primitive never reaches the contracted 5 cm lift")
        upper = int(upper_candidates[0])
        lower = upper - 1
        low_delta = float(target[lower, 2] - baseline)
        high_delta = float(target[upper, 2] - baseline)
        alpha = (requested - low_delta) / (high_delta - low_delta)
        q5 = commands[lower] + alpha * (commands[upper] - commands[lower])
        hold_count = max(1, int(round(float(execution["lift_hold_s"]) * fps)))
        lift_commands = np.concatenate(
            (commands[hold_end + 1 : lower + 1], q5[None], np.repeat(q5[None], hold_count, axis=0)),
            axis=0,
        )
        lift_labels = np.concatenate(
            (
                stages[hold_end + 1 : lower + 1],
                np.asarray(["LIFT_5CM"]),
                np.repeat("HOLD_ELEVATED_5CM", hold_count),
            )
        )
        return (
            np.concatenate((retention_commands, lift_commands), axis=0),
            np.concatenate((retention_labels, lift_labels)),
        )
    raise ValueError(stage_name)


def main() -> int:
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    if config.get("schema_version") != "dex3_rigid_proxy_contact_diagnostic_v1":
        raise RuntimeError("unexpected diagnostic config")
    if config["policy_evaluation_allowed"] or config["learned_policy_used"]:
        raise RuntimeError("policy use is prohibited in contact diagnosis")
    retention_contract: dict[str, Any] | None = None
    retention_config_path: Path | None = None
    execution_condition: dict[str, Any] | None = None
    if args.retention_config is not None:
        retention_config_path = args.retention_config.resolve()
        retention_contract = read_json(retention_config_path)
        if retention_contract.get("schema_version") != "dex3_capsule_retention_bounded_repair_v1":
            raise RuntimeError("unexpected capsule retention config")
        if (
            retention_contract.get("policy_evaluation_allowed")
            or retention_contract.get("learned_policy_used")
            or retention_contract.get("real_robot_allowed")
        ):
            raise RuntimeError("policy/real-robot use is prohibited in retention diagnosis")
        if Path(retention_contract["source_contact_config"]).resolve() != config_path:
            raise RuntimeError("retention contract points to a different base contact config")
        if retention_contract["source_contact_config_sha256"] != sha256_file(config_path):
            raise RuntimeError("base contact config hash changed after retention predeclaration")
        if args.condition not in retention_contract["conditions"]:
            raise RuntimeError(f"condition not predeclared: {args.condition}")
        execution_condition = copy.deepcopy(retention_contract["conditions"][args.condition])
        config = copy.deepcopy(config)
        config["fixed_material"] = {
            **execution_condition["material"],
            "rationale": f"Bounded retention condition {args.condition}",
        }
        config["finger_drive"] = copy.deepcopy(execution_condition["finger_drive"])
    elif args.condition != "baseline":
        raise RuntimeError("non-baseline --condition requires --retention-config")
    candidates = {candidate["name"]: candidate for candidate in config["shape_candidates"]}
    if args.shape not in candidates:
        raise RuntimeError(f"shape not predeclared: {args.shape}")
    candidate = candidates[args.shape]
    if retention_contract is not None:
        if args.shape != retention_contract["frozen_geometry"]["name"]:
            raise RuntimeError("retention task is locked to the frozen capsule geometry")
        geometry_keys = ("name", "type", "radius_m", "cylinder_height_m", "major_axis_world_xyz")
        source_geometry = {key: candidate[key] for key in geometry_keys}
        frozen_geometry = {key: retention_contract["frozen_geometry"][key] for key in geometry_keys}
        if source_geometry != frozen_geometry:
            raise RuntimeError("frozen capsule geometry differs from the completed contact diagnostic")
    primitive_path = Path(config[f"{args.side}_primitive"])
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    commands, command_stages = build_commands(primitive, args.stage, retention_contract)
    carryover: dict[str, Any] | None = None
    if args.initial_state_from is not None:
        carry_path = args.initial_state_from.resolve()
        source_result_path = carry_path.parent / "stage_result.json"
        if not carry_path.is_file() or not source_result_path.is_file():
            raise RuntimeError("carryover stage log/result is missing")
        source_result = read_json(source_result_path)
        if (
            source_result.get("status") != "PASS"
            or source_result.get("shape") != args.shape
            or source_result.get("side") != args.side
            or source_result.get("learned_policy_used")
        ):
            raise RuntimeError("carryover must be a passed policy-free stage for the same side/shape")
        required_source = {
            "STATIC_HOLD": "FIXED_ARM_THREE_FINGER_CLOSE",
            "VERTICAL_LIFT_5CM": "STATIC_HOLD",
        }.get(args.stage)
        if required_source is not None and source_result.get("stage") != required_source:
            raise RuntimeError(
                f"{args.stage} requires carryover from {required_source}, got {source_result.get('stage')}"
            )
        with np.load(carry_path, allow_pickle=False) as source:
            carryover = {
                "path": str(carry_path),
                "sha256": sha256_file(carry_path),
                "source_result": str(source_result_path),
                "source_result_sha256": sha256_file(source_result_path),
                "source_stage": source_result["stage"],
                "object_position_world_m": np.asarray(source["object_position_world_m"][-1]),
                "object_quaternion_xyzw": np.asarray(source["object_quaternion_xyzw"][-1]),
                "object_linear_velocity_m_s": np.asarray(source["object_linear_velocity_m_s"][-1]),
                "object_angular_velocity_rad_s": np.asarray(source["object_angular_velocity_rad_s"][-1]),
                "measured_q_rad": np.asarray(source["measured_q_rad"][-1]),
            }
    names, _ = authoritative_joint_ranges()
    if primitive["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("primitive named joint order mismatch")
    dt = float(config["simulation"]["physics_dt_s"])
    fps = float(config["simulation"]["control_fps_hz"])
    substeps = int(config["simulation"]["physics_substeps_per_control_frame"])
    if not np.isclose(dt * substeps, 1.0 / fps):
        raise RuntimeError("physics/control timing mismatch")
    if not omni.usd.get_context().open_stage(config["scene"]):
        raise RuntimeError("failed to open diagnostic scene")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    runtime_proxy = apply_proxy(stage, config, candidate)
    selected_link = config["single_fingertip"][f"{args.side}_link"]
    collision_setup = apply_collision_isolation(stage, args.side, args.stage, selected_link)
    report_bodies = apply_contact_reporting(stage)
    gravity = 0.0 if args.stage in (
        "SINGLE_FINGERTIP_NO_GRAVITY",
        "FIXED_ARM_THREE_FINGER_CLOSE",
        "CLOSURE_TO_GRAVITY_RETENTION",
        "CLOSURE_TO_GRAVITY_RETENTION_AND_LIFT_5CM",
    ) else float(
        config["simulation"]["gravity_m_s2"]
    )
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=str(config["simulation"]["device"]),
            gravity=(0.0, 0.0, -gravity),
            use_fabric=bool(config["simulation"]["use_fabric"]),
        )
    )
    actuators = (
        build_implicit_actuators(ImplicitActuatorCfg)
        if retention_contract is None
        else build_condition_actuators(config)
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path=f"{G1}/root_joint",
            spawn=None,
            actuators=actuators,
        )
    )
    doll = RigidObject(RigidObjectCfg(prim_path=DOLL, spawn=None))
    whole_hand = read_json(Path(config["whole_hand_geometry"]))
    distal = {
        role: whole_hand[args.side][role]["distal_link"] for role in ("A", "B", "C")
    }
    digit_sensors = {
        role: ContactSensor(
            ContactSensorCfg(
                prim_path=f"{G1}/{link}",
                update_period=0.0,
                filter_prim_paths_expr=[DOLL],
                track_contact_points=True,
                max_contact_data_count_per_prim=64,
                force_threshold=0.0,
            )
        )
        for role, link in distal.items()
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
    callback_log_path = output_dir / "contact_report_callback.csv"
    exact_logger = ExactContactLogger(callback_log_path, dt)
    tensor_log_path = output_dir / "contact_pairs.csv"
    tensor_log_file = tensor_log_path.open("w", newline="", encoding="utf-8")
    tensor_log_writer = csv.DictWriter(tensor_log_file, fieldnames=CONTACT_FIELDS)
    tensor_log_writer.writeheader()
    tensor_api_errors: set[str] = set()
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    joint_ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(joint_ids) != 28 or len(set(joint_ids)) != 28:
        raise RuntimeError(f"named Isaac mapping failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    velocity = torch.zeros_like(target)
    initial_q = commands[0] if carryover is None else carryover["measured_q_rad"]
    target[0, joint_ids] = torch.as_tensor(initial_q, device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(target, velocity)
    support = candidate_support(candidate)
    center = np.asarray(
        [
            *config["object"]["center_world_xy_m"],
            float(config["object"]["table_surface_world_z_m"])
            + support
            + float(config["object"]["spawn_clearance_above_table_m"]),
        ],
        dtype=np.float32,
    )
    initial_object_position = center if carryover is None else carryover["object_position_world_m"]
    initial_object_quaternion = (
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if carryover is None
        else carryover["object_quaternion_xyzw"]
    )
    root_pose = torch.as_tensor(
        np.r_[initial_object_position, initial_object_quaternion][None],
        device=doll.device,
        dtype=torch.float32,
    )
    doll.write_root_pose_to_sim_index(root_pose=root_pose)
    initial_object_velocity = (
        np.zeros(6, dtype=np.float32)
        if carryover is None
        else np.r_[
            carryover["object_linear_velocity_m_s"],
            carryover["object_angular_velocity_rad_s"],
        ]
    )
    doll.write_root_velocity_to_sim_index(
        root_velocity=torch.as_tensor(initial_object_velocity[None], dtype=torch.float32, device=doll.device)
    )
    body_names = list(robot.data.body_names)
    distal_body_ids = {role: body_names.index(link) for role, link in distal.items()}
    hand_slice = slice(14, 21) if args.side == "left" else slice(21, 28)
    primitive_stages = primitive["stage"].astype(str)
    hold_end = int(np.flatnonzero(primitive_stages == "HOLD_ON_TABLE")[-1])
    grasp_target_q = np.asarray(primitive["commanded_q_rad"][hold_end, hand_slice], dtype=np.float64)
    geometry_role = next(
        role
        for role in ("A", "B", "C")
        if whole_hand[args.side][role]["digit_chain"] == "index"
    )
    selected_body_id = distal_body_ids[geometry_role]
    selected_spec = whole_hand[args.side][geometry_role]
    selected_collision_prim = stage.GetPrimAtPath(collision_setup["selected_fingertip_collider"])
    finger_physx = PhysxSchema.PhysxCollisionAPI(selected_collision_prim)
    finger_offsets = {
        "contact_offset_authored": bool(finger_physx.GetContactOffsetAttr().HasAuthoredValueOpinion()),
        "contact_offset_schema_value": finger_physx.GetContactOffsetAttr().Get(),
        "rest_offset_authored": bool(finger_physx.GetRestOffsetAttr().HasAuthoredValueOpinion()),
        "rest_offset_schema_value": finger_physx.GetRestOffsetAttr().Get(),
        "source": "composed G1 fingertip collider; no diagnostic override",
    }
    try:
        finger_offsets["effective_contact_offsets_m"] = numpy(
            digit_sensors[geometry_role].body_physx_view.get_contact_offsets()
        ).reshape(-1).astype(np.float64).tolist()
        finger_offsets["effective_rest_offsets_m"] = numpy(
            digit_sensors[geometry_role].body_physx_view.get_rest_offsets()
        ).reshape(-1).astype(np.float64).tolist()
        finger_offsets["effective_readback_error"] = None
    except Exception as error:
        finger_offsets["effective_contact_offsets_m"] = []
        finger_offsets["effective_rest_offsets_m"] = []
        finger_offsets["effective_readback_error"] = f"{type(error).__name__}: {error}"
    try:
        runtime_proxy["effective_contact_offsets_m"] = numpy(
            doll.root_physx_view.get_contact_offsets()
        ).reshape(-1).astype(np.float64).tolist()
        runtime_proxy["effective_rest_offsets_m"] = numpy(
            doll.root_physx_view.get_rest_offsets()
        ).reshape(-1).astype(np.float64).tolist()
        runtime_proxy["effective_offset_readback_error"] = None
    except Exception as error:
        runtime_proxy["effective_contact_offsets_m"] = []
        runtime_proxy["effective_rest_offsets_m"] = []
        runtime_proxy["effective_offset_readback_error"] = f"{type(error).__name__}: {error}"

    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "physics_step",
            "control_frame",
            "stage",
            "object_position_world_m",
            "object_quaternion_xyzw",
            "object_linear_velocity_m_s",
            "object_angular_velocity_rad_s",
            "selected_fingertip_pad_world_m",
            "selected_fingertip_point_proxy_signed_distance_m",
            "role_A_fingertip_point_proxy_signed_distance_m",
            "role_B_fingertip_point_proxy_signed_distance_m",
            "role_C_fingertip_point_proxy_signed_distance_m",
            "role_A_contact_force_n",
            "role_B_contact_force_n",
            "role_C_contact_force_n",
            "role_A_contact_count",
            "role_B_contact_count",
            "role_C_contact_count",
            "total_hand_object_normal_force_n",
            "total_hand_object_normal_impulse_ns",
            "maximum_contact_tangential_velocity_m_s",
            "mean_contact_tangential_velocity_m_s",
            "active_fingertip_links",
            "table_contact_force_n",
            "exact_dex3_object_contact",
            "exact_table_object_contact",
            "maximum_step_penetration_m",
            "gravity_enabled",
            "measured_q_rad",
            "commanded_q_rad",
            "measured_qd_rad_s",
            "finger_grasp_target_q_rad",
            "finger_joint_position_error_to_grasp_rad",
            "finger_estimated_implicit_drive_torque_nm",
            "finger_computed_torque_nm",
            "finger_applied_torque_nm",
            "finger_projected_joint_force_n_or_nm",
        )
    }
    physics_step = 0
    gravity_enabled = gravity > 0.0
    projected_force_error: str | None = None

    def update() -> None:
        robot.update(dt)
        doll.update(dt)
        for sensor in digit_sensors.values():
            sensor.update(dt, force_recompute=True)
        table_sensor.update(dt, force_recompute=True)

    for control_frame, (command, label) in enumerate(zip(commands, command_stages, strict=True)):
        if str(label) == "GRAVITY_RETENTION" and not gravity_enabled:
            sim.physics_sim_view.set_gravity(
                carb.Float3(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"]))
            )
            gravity_enabled = True
        target[0, joint_ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        for _ in range(substeps):
            exact_logger.set_context(physics_step, control_frame, str(label))
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            update()
            pose = doll.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
            object_velocity = doll.data.root_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
            positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
            quaternions = robot.data.body_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
            body_com_positions = numpy(robot.data.body_com_pos_w)[0].astype(np.float64)
            body_com_linear_velocities = numpy(robot.data.body_com_lin_vel_w)[0].astype(np.float64)
            body_com_angular_velocities = numpy(robot.data.body_com_ang_vel_w)[0].astype(np.float64)
            object_com_position = numpy(doll.data.body_com_pos_w)[0, 0].astype(np.float64)
            object_com_linear_velocity = numpy(doll.data.body_com_lin_vel_w)[0, 0].astype(np.float64)
            object_com_angular_velocity = numpy(doll.data.body_com_ang_vel_w)[0, 0].astype(np.float64)
            pads: dict[str, np.ndarray] = {}
            for role, body_id in distal_body_ids.items():
                role_rotation = rotation_xyzw(quaternions[body_id])
                pads[role] = positions[body_id] + role_rotation @ np.asarray(
                    whole_hand[args.side][role]["local_position_xyz_m"], dtype=np.float64
                )
            pad = pads[geometry_role]
            callback_combined = [
                " ".join((row["actor0"], row["actor1"], row["collider0"], row["collider1"]))
                for row in exact_logger.latest
            ]
            tensor_digit_rows: dict[str, list[dict[str, Any]]] = {}
            for role, sensor in digit_sensors.items():
                rows, error = tensor_contact_rows(
                    sensor,
                    physics_dt=dt,
                    owner_fallback=f"{G1}/{distal[role]}",
                    other_path=PROXY,
                )
                tensor_digit_rows[role] = rows
                if error:
                    tensor_api_errors.add(error)
                body_id = distal_body_ids[role]
                for row in rows:
                    point = row["point"]
                    finger_point_velocity = body_com_linear_velocities[body_id] + np.cross(
                        body_com_angular_velocities[body_id], point - body_com_positions[body_id]
                    )
                    object_point_velocity = object_com_linear_velocity + np.cross(
                        object_com_angular_velocity, point - object_com_position
                    )
                    relative_velocity = finger_point_velocity - object_point_velocity
                    normal = normalized(row["normal"])
                    relative_normal = float(relative_velocity @ normal)
                    tangential = relative_velocity - relative_normal * normal
                    row["role"] = role
                    row["normal_impulse_ns"] = float(row["force_n"] * dt)
                    row["relative_normal_velocity_m_s"] = relative_normal
                    row["relative_tangential_velocity_m_s"] = float(np.linalg.norm(tangential))
            tensor_table_rows, table_error = tensor_contact_rows(
                table_sensor,
                physics_dt=dt,
                owner_fallback=DOLL,
                other_path=TABLE,
            )
            if table_error:
                tensor_api_errors.add(table_error)
            tensor_rows = [row for rows in tensor_digit_rows.values() for row in rows]
            tensor_rows.extend(tensor_table_rows)
            for row in tensor_rows:
                point, normal = row["point"], row["normal"]
                owner, other = row["owner"], row["other"]
                proxy_pair = DOLL in owner or DOLL in other or PROXY in owner or PROXY in other
                tensor_log_writer.writerow(
                    {
                        "physics_step": physics_step,
                        "sim_time_s": f"{physics_step * dt:.9f}",
                        "control_frame": control_frame,
                        "stage": str(label),
                        "role": row.get("role", "TABLE"),
                        "event_type": "TENSOR_CONTACT_VIEW",
                        "actor0": owner,
                        "actor1": other,
                        "collider0": (
                            f"{owner}/collisions" if owner.startswith(G1) else PROXY
                        ),
                        "collider1": (PROXY if proxy_pair and owner.startswith(G1) else other),
                        "contact_x": f"{point[0]:.9g}",
                        "contact_y": f"{point[1]:.9g}",
                        "contact_z": f"{point[2]:.9g}",
                        "normal_x": f"{normal[0]:.9g}",
                        "normal_y": f"{normal[1]:.9g}",
                        "normal_z": f"{normal[2]:.9g}",
                        "impulse_x": f"{normal[0] * row['force_n'] * dt:.9g}",
                        "impulse_y": f"{normal[1] * row['force_n'] * dt:.9g}",
                        "impulse_z": f"{normal[2] * row['force_n'] * dt:.9g}",
                        "normal_force_n": f"{row['force_n']:.9g}",
                        "normal_impulse_ns": f"{row['force_n'] * dt:.9g}",
                        "relative_normal_velocity_m_s": (
                            f"{row['relative_normal_velocity_m_s']:.9g}"
                            if "relative_normal_velocity_m_s" in row
                            else ""
                        ),
                        "relative_tangential_velocity_m_s": (
                            f"{row['relative_tangential_velocity_m_s']:.9g}"
                            if "relative_tangential_velocity_m_s" in row
                            else ""
                        ),
                        "separation_m": f"{row['separation_m']:.9g}",
                        "penetration_m": f"{max(0.0, -row['separation_m']):.9g}",
                    }
                )
            penetrations = [float(row["penetration_m"]) for row in exact_logger.latest]
            penetrations.extend(max(0.0, -row["separation_m"]) for row in tensor_rows)
            measured = robot.data.joint_pos.torch[0, joint_ids].detach().cpu().numpy().astype(np.float64)
            measured_qd = robot.data.joint_vel.torch[0, joint_ids].detach().cpu().numpy().astype(np.float64)
            finger_error = grasp_target_q - measured[hand_slice]
            drive = config["finger_drive"]
            estimated_drive_torque = np.clip(
                float(drive["kp"]) * finger_error - float(drive["kd"]) * measured_qd[hand_slice],
                -float(drive["effort_limit_sim"]),
                float(drive["effort_limit_sim"]),
            )
            computed_torque = numpy(robot.data.computed_torque)[0, joint_ids][hand_slice].astype(np.float64)
            applied_torque = numpy(robot.data.applied_torque)[0, joint_ids][hand_slice].astype(np.float64)
            try:
                projected_all = numpy(robot.root_physx_view.get_dof_projected_joint_forces()).reshape(1, -1)
                projected_force = projected_all[0, joint_ids][hand_slice].astype(np.float64)
            except Exception as error:
                if projected_force_error is None:
                    projected_force_error = f"{type(error).__name__}: {error}"
                projected_force = np.full(7, np.nan, dtype=np.float64)
            digit_rows = [row for rows in tensor_digit_rows.values() for row in rows]
            total_normal_force = float(sum(row["force_n"] for row in digit_rows))
            tangential_speeds = [row["relative_tangential_velocity_m_s"] for row in digit_rows]
            active_roles = [role for role in ("A", "B", "C") if tensor_digit_rows[role]]
            values = {
                "physics_step": physics_step,
                "control_frame": control_frame,
                "stage": str(label),
                "object_position_world_m": pose[:3],
                "object_quaternion_xyzw": pose[3:7],
                "object_linear_velocity_m_s": object_velocity[:3],
                "object_angular_velocity_rad_s": object_velocity[3:],
                "selected_fingertip_pad_world_m": pad,
                "selected_fingertip_point_proxy_signed_distance_m": point_proxy_signed_distance(
                    pad, pose[:3], pose[3:7], candidate
                ),
                "role_A_fingertip_point_proxy_signed_distance_m": point_proxy_signed_distance(
                    pads["A"], pose[:3], pose[3:7], candidate
                ),
                "role_B_fingertip_point_proxy_signed_distance_m": point_proxy_signed_distance(
                    pads["B"], pose[:3], pose[3:7], candidate
                ),
                "role_C_fingertip_point_proxy_signed_distance_m": point_proxy_signed_distance(
                    pads["C"], pose[:3], pose[3:7], candidate
                ),
                "role_A_contact_force_n": filtered_force(digit_sensors["A"]),
                "role_B_contact_force_n": filtered_force(digit_sensors["B"]),
                "role_C_contact_force_n": filtered_force(digit_sensors["C"]),
                "role_A_contact_count": len(tensor_digit_rows["A"]),
                "role_B_contact_count": len(tensor_digit_rows["B"]),
                "role_C_contact_count": len(tensor_digit_rows["C"]),
                "total_hand_object_normal_force_n": total_normal_force,
                "total_hand_object_normal_impulse_ns": total_normal_force * dt,
                "maximum_contact_tangential_velocity_m_s": max(tangential_speeds, default=0.0),
                "mean_contact_tangential_velocity_m_s": (
                    float(np.mean(tangential_speeds)) if tangential_speeds else 0.0
                ),
                "active_fingertip_links": ",".join(distal[role] for role in active_roles),
                "table_contact_force_n": filtered_force(table_sensor),
                "exact_dex3_object_contact": bool(tensor_rows[:-len(tensor_table_rows) or None])
                or any(G1 in row and DOLL in row for row in callback_combined),
                "exact_table_object_contact": bool(tensor_table_rows)
                or any("/Table" in row and DOLL in row for row in callback_combined),
                "maximum_step_penetration_m": max(penetrations, default=0.0),
                "gravity_enabled": gravity_enabled,
                "measured_q_rad": measured,
                "commanded_q_rad": command,
                "measured_qd_rad_s": measured_qd,
                "finger_grasp_target_q_rad": grasp_target_q,
                "finger_joint_position_error_to_grasp_rad": finger_error,
                "finger_estimated_implicit_drive_torque_nm": estimated_drive_torque,
                "finger_computed_torque_nm": computed_torque,
                "finger_applied_torque_nm": applied_torque,
                "finger_projected_joint_force_n_or_nm": projected_force,
            }
            for key, value in values.items():
                records[key].append(value)
            physics_step += 1
    exact_logger.close()
    tensor_log_file.flush()
    tensor_log_file.close()
    arrays = {key: np.asarray(value) for key, value in records.items()}
    arrays["timestamp_s"] = arrays["physics_step"].astype(np.float64) * dt
    arrays["joint_names"] = np.asarray(names)
    log_path = output_dir / "stage_log.npz"
    temporary = log_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, log_path)
    force_threshold = float(config["gates"]["minimum_contact_force_n"])
    role_masks = {
        role: (
            (arrays[f"role_{role}_contact_force_n"] >= force_threshold)
            | (arrays[f"role_{role}_contact_count"] > 0)
        )
        for role in ("A", "B", "C")
    }
    any_force = np.logical_or.reduce(list(role_masks.values()))
    exact_contact = arrays["exact_dex3_object_contact"].astype(bool)
    contact_mask = any_force | exact_contact
    initial_window = min(substeps, len(contact_mask))
    initial_contact_penetration = bool(
        np.max(arrays["maximum_step_penetration_m"][:initial_window], initial=0.0)
        > float(config["gates"]["maximum_initial_hand_proxy_penetration_m"])
    )
    initial_overlap = bool(initial_contact_penetration and carryover is None)
    velocities_linear = np.linalg.norm(arrays["object_linear_velocity_m_s"], axis=1)
    velocities_angular = np.linalg.norm(arrays["object_angular_velocity_rad_s"], axis=1)
    steps = np.linalg.norm(np.diff(arrays["object_position_world_m"], axis=0), axis=1)
    max_penetration = float(np.max(arrays["maximum_step_penetration_m"], initial=0.0))
    max_linear = float(np.max(velocities_linear, initial=0.0))
    max_angular = float(np.max(velocities_angular, initial=0.0))
    max_step = float(np.max(steps, initial=0.0))
    contact_duration = longest_duration(contact_mask, dt)
    role_durations = {role: longest_duration(mask, dt) for role, mask in role_masks.items()}
    contact_roles = [role for role, mask in role_masks.items() if bool(np.any(mask))]
    distinct_pairs: set[str] = set()
    with tensor_log_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            distinct_pairs.add(
                f"{row['collider0']} <-> {row['collider1']}"
            )
    common_safe = bool(
        not initial_overlap
        and exact_logger.api_error is None
        and max_penetration <= float(config["gates"]["maximum_contact_penetration_m"])
        and max_linear <= float(config["gates"]["maximum_object_linear_speed_m_s"])
        and max_angular <= float(config["gates"]["maximum_object_angular_speed_rad_s"])
        and max_step <= float(config["gates"]["maximum_object_com_step_m"])
    )
    initial_position = arrays["object_position_world_m"][0]
    final_position = arrays["object_position_world_m"][-1]
    horizontal_displacement = float(np.linalg.norm((final_position - initial_position)[:2]))
    hold_mask = np.isin(
        arrays["stage"].astype(str), ["STATIC_HOLD", "POST_CLOSE", "GRAVITY_RETENTION"]
    )
    hold_contact_duration = longest_duration(contact_mask & hold_mask, dt)
    stage_labels = arrays["stage"].astype(str)
    gravity_mask = stage_labels == "GRAVITY_RETENTION"
    table_supported = arrays["table_contact_force_n"] >= force_threshold
    gravity_indices = np.flatnonzero(gravity_mask)
    retention_reference_z = (
        float(arrays["object_position_world_m"][gravity_indices[0], 2])
        if len(gravity_indices)
        else float(arrays["object_position_world_m"][0, 2])
    )
    maximum_retained_drop = (
        float(retention_contract["gates"]["maximum_retained_com_drop_m"])
        if retention_contract is not None
        else 0.005
    )
    retention_mask = (
        gravity_mask
        & contact_mask
        & ~table_supported
        & (
            arrays["object_position_world_m"][:, 2]
            >= retention_reference_z - maximum_retained_drop
        )
    )
    gravity_retention_duration = longest_duration(retention_mask, dt)
    minimum_retention = (
        float(retention_contract["gates"]["minimum_gravity_retention_s"])
        if retention_contract is not None
        else 1.0
    )
    retention_pass = bool(gravity_retention_duration >= minimum_retention)
    lift_mask = arrays["stage"].astype(str) == "LIFT_5CM"
    elevated_mask = arrays["stage"].astype(str) == "HOLD_ELEVATED_5CM"
    pre_lift = np.flatnonzero(arrays["stage"].astype(str) == "PRE_LIFT_STATIC_HOLD")
    if len(pre_lift):
        baseline_z = float(
            np.median(
                arrays["object_position_world_m"][
                    pre_lift[-min(len(pre_lift), int(round(0.25 / dt))) :], 2
                ]
            )
        )
    elif len(gravity_indices):
        baseline_window = gravity_indices[-min(len(gravity_indices), int(round(0.25 / dt))) :]
        baseline_z = float(np.median(arrays["object_position_world_m"][baseline_window, 2]))
    else:
        baseline_z = float(initial_position[2])
    lift_height = float(np.max(arrays["object_position_world_m"][:, 2]) - baseline_z)
    elevated_duration = longest_duration(
        elevated_mask
        & contact_mask
        & ~table_supported
        & (arrays["object_position_world_m"][:, 2] - baseline_z >= float(config["gates"]["minimum_lift_m"])),
        dt,
    )
    if args.stage == "SINGLE_FINGERTIP_NO_GRAVITY":
        passed = common_safe and contact_duration >= float(
            config["gates"]["minimum_single_contact_duration_s"]
        )
    elif args.stage == "SINGLE_FINGERTIP_PUSH_GRAVITY":
        passed = (
            common_safe
            and contact_duration >= float(config["gates"]["minimum_gravity_push_contact_duration_s"])
            and horizontal_displacement
            >= float(config["gates"]["minimum_gravity_push_displacement_m"])
        )
    elif args.stage == "FIXED_ARM_THREE_FINGER_CLOSE":
        passed = common_safe and len(contact_roles) == 3
    elif args.stage == "STATIC_HOLD":
        passed = (
            common_safe
            and len(contact_roles) == 3
            and hold_contact_duration >= float(config["gates"]["minimum_static_hold_duration_s"])
        )
    elif args.stage == "VERTICAL_LIFT_5CM":
        passed = (
            common_safe
            and len(contact_roles) == 3
            and lift_height >= float(config["gates"]["minimum_lift_m"])
            and elevated_duration >= float(config["gates"]["minimum_elevated_hold_duration_s"])
        )
    elif args.stage == "CLOSURE_TO_GRAVITY_RETENTION":
        passed = common_safe and len(contact_roles) == 3 and retention_pass
    elif args.stage == "CLOSURE_TO_GRAVITY_RETENTION_AND_LIFT_5CM":
        passed = (
            common_safe
            and len(contact_roles) == 3
            and retention_pass
            and lift_height >= float(retention_contract["gates"]["minimum_lift_m"])
            and elevated_duration
            >= float(retention_contract["gates"]["minimum_elevated_hold_s"])
        )
    else:
        raise ValueError(args.stage)
    result = {
        "schema_version": "dex3_contact_stage_result_v1",
        "status": "PASS" if passed else "FAIL",
        "stage": args.stage,
        "side": args.side,
        "shape": args.shape,
        "duration_s": len(arrays["physics_step"]) * dt,
        "initial_gravity_m_s2": gravity,
        "gravity_enabled_during_stage": bool(np.any(arrays["gravity_enabled"])),
        "gravity_m_s2_after_switch": (
            float(config["simulation"]["gravity_m_s2"])
            if bool(np.any(arrays["gravity_enabled"]))
            else gravity
        ),
        "collision_contact_pairs": sorted(distinct_pairs),
        "contact_duration_s": contact_duration,
        "per_role_contact_duration_s": role_durations,
        "contact_roles": contact_roles,
        "object_initial_position_world_m": initial_position,
        "object_final_position_world_m": final_position,
        "object_horizontal_displacement_m": horizontal_displacement,
        "object_max_linear_velocity_m_s": max_linear,
        "object_max_angular_velocity_rad_s": max_angular,
        "maximum_object_com_step_m": max_step,
        "maximum_penetration_m": max_penetration,
        "minimum_fingertip_point_to_proxy_signed_distance_m": float(
            np.min(arrays["selected_fingertip_point_proxy_signed_distance_m"])
        ),
        "initial_overlap": initial_overlap,
        "initial_contact_penetration_above_spawn_gate": initial_contact_penetration,
        "continued_from_passed_lower_stage": carryover is not None,
        "carryover": carryover,
        "object_contact_offset_m": runtime_proxy["contact_offset_m"],
        "object_rest_offset_m": runtime_proxy["rest_offset_m"],
        "selected_fingertip_offsets": finger_offsets,
        "finger_drive": config["finger_drive"],
        "collision_filtering": collision_setup,
        "contact_report_bodies": report_bodies,
        "contact_report_api_error": exact_logger.api_error,
        "contact_report_header_count": exact_logger.header_count,
        "contact_report_header_preview": exact_logger.header_preview,
        "tensor_contact_api_errors": sorted(tensor_api_errors),
        "runtime_proxy": runtime_proxy,
        "lift": {
            "requested_vertical_lift_m": float(config["gates"]["requested_lift_m"]),
            "measured_object_lift_m": lift_height,
            "elevated_hold_duration_s": elevated_duration,
            "baseline_object_z_m": baseline_z,
        },
        "static_hold_contact_duration_s": hold_contact_duration,
        "gravity_retention": {
            "status": "PASS" if retention_pass else "FAIL",
            "required_duration_s": minimum_retention,
            "measured_duration_s": gravity_retention_duration,
            "reference_com_z_m": retention_reference_z,
            "maximum_allowed_com_drop_m": maximum_retained_drop,
            "minimum_com_z_during_gravity_m": (
                float(np.min(arrays["object_position_world_m"][gravity_mask, 2]))
                if bool(np.any(gravity_mask))
                else None
            ),
            "maximum_total_normal_force_n": float(
                np.max(arrays["total_hand_object_normal_force_n"], initial=0.0)
            ),
            "median_total_normal_force_n_during_contact": (
                float(np.median(arrays["total_hand_object_normal_force_n"][contact_mask]))
                if bool(np.any(contact_mask))
                else 0.0
            ),
            "maximum_contact_tangential_velocity_m_s": float(
                np.max(arrays["maximum_contact_tangential_velocity_m_s"], initial=0.0)
            ),
            "gravity_load_n": (
                float(retention_contract["object"]["gravitational_load_n"])
                if retention_contract is not None
                else float(config["object"]["mass_kg"] * config["simulation"]["gravity_m_s2"])
            ),
        },
        "maximum_arm_command_change_during_stage_rad": float(
            np.max(np.abs(commands[:, :14] - commands[0, :14]))
        ),
        "fixed_material": config["fixed_material"],
        "execution_condition_name": args.condition,
        "execution_condition": execution_condition,
        "friction_changed_from_baseline": bool(
            execution_condition is not None
            and execution_condition["category"] in ("FRICTION_ONLY", "COMBINED_SINGLE_CANDIDATE")
        ),
        "drive_changed_from_baseline": bool(
            execution_condition is not None
            and execution_condition["category"] in ("DRIVE_ONLY", "COMBINED_SINGLE_CANDIDATE")
        ),
        "friction_sweep_or_tuning_performed": False,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "retention_config": str(retention_config_path) if retention_config_path else None,
        "retention_config_sha256": (
            sha256_file(retention_config_path) if retention_config_path else None
        ),
        "primitive": str(primitive_path),
        "primitive_sha256": sha256_file(primitive_path),
        "stage_log": str(log_path),
        "stage_log_sha256": sha256_file(log_path),
        "contact_pair_log": str(tensor_log_path),
        "contact_pair_log_sha256": sha256_file(tensor_log_path),
        "contact_report_callback_log": str(callback_log_path),
        "contact_report_callback_log_sha256": sha256_file(callback_log_path),
        "learned_policy_used": False,
        "policy_checkpoint_read": False,
        "real_robot": False,
        "prohibited_attachment_used": False,
        "implicit_drive_torque_observability": {
            "computed_torque_all_zero": bool(
                np.allclose(arrays["finger_computed_torque_nm"], 0.0)
            ),
            "applied_torque_all_zero": bool(
                np.allclose(arrays["finger_applied_torque_nm"], 0.0)
            ),
            "projected_joint_force_readback_error": projected_force_error,
            "estimated_drive_torque_definition": "clip(kp*(GRASP_target-q)-kd*qd, +/-effort_limit); an estimate for the implicit PhysX drive, not a direct actuator sensor",
        },
    }
    atomic_json(output_dir / "stage_result.json", result)
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
