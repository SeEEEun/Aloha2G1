#!/usr/bin/env python3
"""Policy-free Dex3 grasp test for one native PhysX volume-deformable plush proxy.

The measured 120 x 90 x 85 mm, 20 g object is represented by one tetrahedral
FEM body.  No dataset, learned policy, target-point solve, attachment, object
pose write during execution, DDS, or real-robot interface is present.
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
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--side", choices=("left", "right"), required=True)
parser.add_argument("--material", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--probe-only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import omni.usd
from omni.physx import get_physx_simulation_interface
import torch
import warp as wp
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics
from omni.physx.scripts import deformableMeshUtils

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxDeformableBodyMaterialCfg

from tools.evaluation.contracts import authoritative_joint_ranges
from tools.policy_b_isaac_control_contract import CONTROLLER_CONTRACT


OLD_DOLL = "/World/DollHandoffEnvironment/Doll"
OLD_DOLL_BODY = f"{OLD_DOLL}/Body"
SOFT_DOLL = "/World/SoftPlushDollV1"
SOFT_COLLISION = f"{SOFT_DOLL}/sim_mesh"
G1 = "/World/G1/Asset"


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


def numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
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


def build_actuators(config: dict[str, Any]) -> dict[str, ImplicitActuatorCfg]:
    values = copy.deepcopy(CONTROLLER_CONTRACT["actuators"])
    drive = config["finger_drive"]
    values["dex3"].update(
        {
            "effort_limit_sim": float(drive["effort_limit_sim_nm"]),
            "velocity_limit_sim": float(drive["velocity_limit_sim_rad_s"]),
            "stiffness": float(drive["kp"]),
            "damping": float(drive["kd"]),
        }
    )
    return {name: ImplicitActuatorCfg(**spec) for name, spec in values.items()}


def build_commands(
    config: dict[str, Any], side: str
) -> tuple[np.ndarray, np.ndarray, Path]:
    primitive_path = Path(config["source_arm_primitives"][side]).resolve()
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    names, _ = authoritative_joint_ranges()
    if primitive["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("source arm primitive named-joint order mismatch")
    approach_arms = np.asarray(primitive["approach_arm_q_rad"], dtype=np.float64)
    lift_arms = np.asarray(primitive["lift_arm_q_rad"], dtype=np.float64)
    base_arms = approach_arms[-1]
    lifted_arms = lift_arms[-1]
    active_slice = slice(14, 21) if side == "left" else slice(21, 28)
    inactive_slice = slice(21, 28) if side == "left" else slice(14, 21)
    inactive_side = "right" if side == "left" else "left"
    states = config["hand_states"]
    open_q = np.asarray(states[side]["OPEN"], dtype=np.float64)
    preshape_q = np.asarray(states[side]["PRESHAPE"], dtype=np.float64)
    full_q = np.asarray(states[side]["FULL_CLOSE"], dtype=np.float64)
    inactive_open = np.asarray(states[inactive_side]["OPEN"], dtype=np.float64)
    timing = config["timing"]
    fps = float(timing["control_fps_hz"])
    rows: list[np.ndarray] = []
    labels: list[str] = []

    def append(arms: np.ndarray, hand: np.ndarray, label: str) -> None:
        row = np.zeros(28, dtype=np.float64)
        row[:14] = arms
        row[active_slice] = hand
        row[inactive_slice] = inactive_open
        rows.append(row)
        labels.append(label)

    for _ in range(round(float(timing["gravity_settle_s"]) * fps)):
        append(base_arms, open_q, "GRAVITY_SETTLE")
    for _ in range(round(float(timing["open_hold_s"]) * fps)):
        append(base_arms, open_q, "OPEN")
    for hand in minimum_jerk(
        open_q, preshape_q, round(float(timing["preshape_transition_s"]) * fps)
    )[1:]:
        append(base_arms, hand, "PRESHAPE")
    for hand in minimum_jerk(
        preshape_q, full_q, round(float(timing["full_close_transition_s"]) * fps)
    )[1:]:
        append(base_arms, hand, "FULL_CLOSE")
    for _ in range(round(float(timing["squeeze_s"]) * fps)):
        append(base_arms, full_q, "SQUEEZE")
    for arm in lift_arms[1:]:
        append(arm, full_q, "LIFT_50MM")
    for _ in range(round(float(timing["elevated_hold_s"]) * fps)):
        append(lifted_arms, full_q, "HOLD_ELEVATED")
    for hand in minimum_jerk(
        full_q, open_q, round(float(timing["release_transition_s"]) * fps)
    )[1:]:
        append(lifted_arms, hand, "RELEASE")
    for _ in range(round(float(timing["post_release_recovery_s"]) * fps)):
        append(lifted_arms, open_q, "RECOVERY")
    commands = np.asarray(rows, dtype=np.float64)
    if commands.shape[1] != 28 or not np.isfinite(commands).all():
        raise RuntimeError("invalid scripted command")
    return commands, np.asarray(labels), primitive_path


def disable_old_doll(stage) -> dict[str, Any]:
    body = stage.GetPrimAtPath(OLD_DOLL_BODY)
    if body.IsValid() and body.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(body).GetCollisionEnabledAttr().Set(False)
    root = stage.GetPrimAtPath(OLD_DOLL)
    if root.IsValid() and root.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(root).GetRigidBodyEnabledAttr().Set(False)
    hidden: list[str] = []
    if root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim == root:
                continue
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden.append(prim.GetPath().pathString)
    return {
        "old_rigid_doll_collision_disabled": bool(body.IsValid()),
        "old_rigid_doll_body_disabled": bool(root.IsValid()),
        "old_visual_prims_hidden": hidden,
    }


def measured_oval_tet_mesh(
    dimensions: np.ndarray, voxel_resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a pre-tetrahedralized oval with exact measured rest extents.

    Isaac Lab's automatic ``pytetwild`` cooking changed a 120 x 90 x 85 mm
    input surface to a 127.4 x 92.3 x 74.6 mm FEM rest mesh in the first
    instrumented probe.  A pre-tetrahedralized PhysX voxel sphere avoids that
    approximation step.  Positive axis scaling converts the one connected
    sphere volume into an ellipsoid without changing tet topology.
    """

    raw_points, raw_indices = deformableMeshUtils.createTetraVoxelSphere(
        int(voxel_resolution)
    )
    points = np.asarray(
        [[float(point[0]), float(point[1]), float(point[2])] for point in raw_points],
        dtype=np.float64,
    )
    indices = np.asarray(raw_indices, dtype=np.int32).reshape(-1, 4)
    points -= (points.max(axis=0) + points.min(axis=0)) / 2.0
    source_extents = points.max(axis=0) - points.min(axis=0)
    if np.any(source_extents <= 0.0):
        raise RuntimeError("invalid voxel-sphere tetrahedral mesh extents")
    points *= dimensions / source_extents

    # UsdGeom.TetMesh requires positive signed volume.  The official helper is
    # normally consistently wound, but validate and repair it explicitly after
    # anisotropic scaling so the authored rest topology is self-auditing.
    p0 = points[indices[:, 0]]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        points[indices[:, 1]] - p0,
        np.cross(points[indices[:, 2]] - p0, points[indices[:, 3]] - p0),
    )
    negative = signed_six_volume < 0.0
    if np.any(negative):
        flipped = indices[negative].copy()
        flipped[:, [2, 3]] = flipped[:, [3, 2]]
        indices[negative] = flipped
    if np.any(np.abs(signed_six_volume) < 1.0e-15):
        raise RuntimeError("degenerate tetrahedron in measured plush mesh")
    authored_extents = points.max(axis=0) - points.min(axis=0)
    if not np.allclose(authored_extents, dimensions, atol=1.0e-7, rtol=0.0):
        raise RuntimeError("pre-tetrahedralized plush does not match measured extents")
    return points.astype(np.float32), indices


def spawn_soft_doll(
    stage, config: dict[str, Any], material: dict[str, Any], center: np.ndarray
) -> dict[str, Any]:
    dimensions = np.asarray(config["object"]["unloaded_dimensions_m"], dtype=np.float64)
    body_cfg = config["deformable_body"]
    deformable_props = PhysxDeformableBodyPropertiesCfg(
        deformable_body_enabled=True,
        mass=float(config["object"]["mass_kg"]),
        solver_position_iteration_count=int(
            material.get(
                "solver_position_iteration_count",
                body_cfg["solver_position_iteration_count"],
            )
        ),
        linear_damping=float(
            material.get("linear_damping_per_s", body_cfg["linear_damping_per_s"])
        ),
        settling_damping=float(
            material.get("settling_damping_per_s", body_cfg["settling_damping_per_s"])
        ),
        settling_threshold=float(
            material.get("settling_threshold_m_s", body_cfg["settling_threshold_m_s"])
        ),
        sleep_threshold=float(
            material.get("sleep_threshold_m_s", body_cfg["sleep_threshold_m_s"])
        ),
        max_linear_velocity=float(
            material.get("max_linear_velocity_m_s", body_cfg["max_linear_velocity_m_s"])
        ),
        max_depenetration_velocity=float(
            material.get(
                "max_depenetration_velocity_m_s",
                body_cfg["max_depenetration_velocity_m_s"],
            )
        ),
        self_collision=bool(material.get("self_collision", body_cfg["self_collision"])),
        self_collision_filter_distance=float(body_cfg["self_collision_filter_distance_m"]),
        enable_speculative_c_c_d=bool(
            material.get("enable_speculative_ccd", body_cfg["enable_speculative_ccd"])
        ),
        contact_offset=float(
            material.get("contact_offset_m", body_cfg["contact_offset_m"])
        ),
        rest_offset=float(material.get("rest_offset_m", body_cfg["rest_offset_m"])),
    )
    volume = 4.0 / 3.0 * np.pi * np.prod(dimensions / 2.0)
    density = float(config["object"]["mass_kg"] / volume)
    material_cfg = PhysxDeformableBodyMaterialCfg(
        density=density,
        static_friction=float(material["static_friction"]),
        dynamic_friction=float(material["dynamic_friction"]),
        youngs_modulus=float(material["youngs_modulus_pa"]),
        poissons_ratio=float(material["poissons_ratio"]),
        elasticity_damping=float(material["elasticity_damping"]),
    )
    points, tet_indices = measured_oval_tet_mesh(
        dimensions, int(config["object"]["mesh"]["voxel_resolution"])
    )
    sim_utils.create_prim(
        SOFT_DOLL,
        prim_type="Xform",
        translation=tuple(map(float, center)),
        orientation=tuple(config["object"]["orientation_quaternion_xyzw"]),
        stage=stage,
    )
    sim_utils.create_prim(
        SOFT_COLLISION,
        prim_type="TetMesh",
        attributes={"points": points, "tetVertexIndices": tet_indices},
        stage=stage,
    )
    sim_utils.define_deformable_body_properties(
        SOFT_DOLL, deformable_props, stage=stage, deformable_type="volume"
    )
    material_path = f"{SOFT_DOLL}/material"
    material_cfg.func(material_path, material_cfg)
    # ``bind_physics_material`` is decorated with ``apply_nested`` and therefore
    # intentionally returns None even when the binding succeeds.
    sim_utils.bind_physics_material(SOFT_DOLL, material_path, stage=stage)
    visual_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{SOFT_DOLL}/vis_mesh"))
    visual_mesh.CreateDisplayColorAttr([(0.18, 0.72, 0.34)])
    surface_faces = UsdGeom.TetMesh.ComputeSurfaceFaces(
        UsdGeom.TetMesh(stage.GetPrimAtPath(SOFT_COLLISION)), Usd.TimeCode.Default()
    )
    root = stage.GetPrimAtPath(SOFT_DOLL)
    return {
        "root_prim": SOFT_DOLL,
        "collision_prim": SOFT_COLLISION,
        "root_schemas": list(root.GetAppliedSchemas()),
        "simulation_vertices": int(len(points)),
        "simulation_tetrahedra": int(len(tet_indices)),
        "surface_triangles": int(len(surface_faces)),
        "analytic_ellipsoid_volume_m3": volume,
        "material_density_kg_m3": density,
        "physics_material_prim": material_path,
        "unloaded_surface_extents_m": points.max(axis=0) - points.min(axis=0),
        "tetrahedralization": config["object"]["mesh"]["type"],
        "single_deformable_root": True,
        "rigid_collision_shell": False,
    }


def apply_digit_contact_reporting(stage, side: str) -> list[str]:
    """Enable exact PhysX contact reports on every rigid link of one Dex3 hand."""

    paths: list[str] = []
    robot_root = stage.GetPrimAtPath(G1)
    for prim in Usd.PrimRange(robot_root):
        path = prim.GetPath().pathString
        if f"/{side}_hand_" not in path or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        paths.append(path)
    if not paths:
        raise RuntimeError(f"no rigid Dex3 links found for contact reporting: {side}")
    return sorted(paths)


def kabsch_dimensions(
    rest_nodes: np.ndarray, current_nodes: np.ndarray
) -> tuple[np.ndarray, float]:
    rest_centered = rest_nodes - rest_nodes.mean(axis=0)
    current_centered = current_nodes - current_nodes.mean(axis=0)
    covariance = rest_centered.T @ current_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    local = current_centered @ rotation
    dimensions = local.max(axis=0) - local.min(axis=0)
    deformation = float(np.max(np.linalg.norm(local - rest_centered, axis=1)))
    return dimensions, deformation


def kabsch_material_state(
    rest_nodes: np.ndarray, current_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove rigid translation/rotation and return the current material frame."""

    rest_centered = rest_nodes - rest_nodes.mean(axis=0)
    current_center = current_nodes.mean(axis=0)
    current_centered = current_nodes - current_center
    covariance = rest_centered.T @ current_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation_current_to_rest = right_t.T @ left.T
    if np.linalg.det(rotation_current_to_rest) < 0.0:
        right_t[-1] *= -1.0
        rotation_current_to_rest = right_t.T @ left.T
    current_local = current_centered @ rotation_current_to_rest
    dimensions = current_local.max(axis=0) - current_local.min(axis=0)
    return current_local, rest_centered, rotation_current_to_rest, current_center


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Convert IsaacLab 3.x native PhysX/Warp XYZW quaternions to rotation matrices."""

    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        raise RuntimeError("invalid zero-norm body quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def point_to_oriented_box_signed_distance(
    points_world: np.ndarray,
    center_world: np.ndarray,
    rotation_local_to_world: np.ndarray,
    half_extent_m: np.ndarray,
) -> np.ndarray:
    """Vectorized exact signed distance from points to a named pad OBB proxy."""

    local = (points_world - center_world) @ rotation_local_to_world
    q = np.abs(local) - half_extent_m
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def main() -> int:
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    if config.get("schema_version") not in {
        "dex3_soft_plush_graspability_v1",
        "dex3_marshmallow_plush_graspability_v1",
    }:
        raise RuntimeError("unexpected soft-plush config schema")
    if config.get("learned_policy_used") or config.get("real_robot_allowed"):
        raise RuntimeError("this tool must remain policy-free and simulation-only")
    materials = {row["name"]: row for row in config["material_candidates"]}
    if args.material not in materials:
        raise RuntimeError("material candidate was not predeclared")
    material = materials[args.material]
    atomic_json(output_dir / "resolved_config.json", config)
    commands, stages, primitive_path = build_commands(config, args.side)
    if args.probe_only:
        keep = np.isin(stages, ["GRAVITY_SETTLE", "OPEN"])
        commands, stages = commands[keep], stages[keep]
    command_path = output_dir / "scripted_command.npz"
    temporary = command_path.with_suffix(".npz.incomplete")
    names, _ = authoritative_joint_ranges()
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=commands.astype(np.float32),
            stage=stages,
            joint_names=np.asarray(names),
            side=np.asarray(args.side),
            material=np.asarray(args.material),
            policy_independent=np.asarray(True),
        )
    os.replace(temporary, command_path)

    stage = sim_utils.open_stage(config["source_scene"])
    if stage is None:
        raise RuntimeError("failed to open source Doll-Handoff scene")
    stage.SetEditTarget(stage.GetSessionLayer())
    old_doll = disable_old_doll(stage)
    dimensions = np.asarray(config["object"]["unloaded_dimensions_m"], dtype=np.float64)
    center_xy = config["object"]["center_world_xy_m_by_side"][args.side]
    center = np.asarray(
        [
            *center_xy,
            float(config["object"]["table_surface_world_z_m"])
            + float(config["object"]["initial_table_clearance_m"])
            + dimensions[2] / 2.0,
        ],
        dtype=np.float64,
    )
    spawn_audit = spawn_soft_doll(stage, config, material, center)
    contact_report_links = apply_digit_contact_reporting(stage, args.side)
    timing = config["timing"]
    dt = float(timing["physics_dt_s"])
    substeps = int(timing["physics_substeps_per_control_frame"])
    fps = float(timing["control_fps_hz"])
    if not np.isclose(dt * substeps, 1.0 / fps):
        raise RuntimeError("physics/control timing mismatch")
    physics = PhysxCfg(
        solver_type=1,
        solve_articulation_contact_last=bool(
            config["simulation"]["solve_articulation_contact_last"]
        ),
        enable_external_forces_every_iteration=bool(
            config["simulation"]["enable_external_forces_every_iteration"]
        ),
        enable_enhanced_determinism=True,
    )
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=str(config["simulation"]["device"]),
            gravity=(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"])),
            use_fabric=bool(config["simulation"]["use_fabric"]),
            physics=physics,
        )
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path=f"{G1}/root_joint", spawn=None, actuators=build_actuators(config)
        )
    )
    soft = DeformableObject(DeformableObjectCfg(prim_path=SOFT_DOLL, spawn=None))
    # Filtered rigid-contact views reject deformable colliders in this PhysX
    # build, but Isaac Sim 6.1 exposes an unfiltered raw-contact stream that
    # includes the other-actor ID.  One view per digit lets us attribute the
    # deformable reaction without inventing geometric proximity contacts.
    digit_sensors = {
        digit: ContactSensor(
            ContactSensorCfg(
                prim_path=f"{G1}/{args.side}_hand_{digit}_.*_link",
                update_period=0.0,
                filter_prim_paths_expr=[],
                track_contact_points=False,
                max_contact_data_count_per_prim=128,
                force_threshold=0.0,
            )
        )
        for digit in ("thumb", "index", "middle")
    }
    sim.reset()
    if not soft.is_initialized:
        raise RuntimeError("native PhysX deformable object failed to initialize")
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    joint_ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(joint_ids) != 28 or len(set(joint_ids)) != 28:
        raise RuntimeError(f"named 28D joint mapping failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    target[0, joint_ids] = torch.as_tensor(
        commands[0], device=robot.device, dtype=target.dtype
    )
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    # Reset the one FEM body exactly once before timed execution.  No nodal or
    # object-pose write occurs inside the control/physics loop.
    default_state = soft.data.default_nodal_state_w.torch.clone()
    authored_local_nodes = np.asarray(
        [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in UsdGeom.TetMesh(
                stage.GetPrimAtPath(SOFT_COLLISION)
            ).GetPointsAttr().Get()
        ],
        dtype=np.float64,
    )
    orientation = np.asarray(
        config["object"]["orientation_quaternion_xyzw"], dtype=np.float64
    )
    if not np.allclose(orientation, [0.0, 0.0, 0.0, 1.0], atol=1.0e-12):
        raise RuntimeError("the v1 exact-node initializer currently requires identity object orientation")
    authored_world_nodes = authored_local_nodes + center
    if (
        int(default_state.shape[1]) != int(authored_world_nodes.shape[0])
        or int(authored_world_nodes.shape[1]) != 3
        or int(default_state.shape[2]) < 6
    ):
        raise RuntimeError(
            "PhysX tensor node order/count does not match the preauthored TetMesh"
        )
    default_state[0, :, :3] = torch.as_tensor(
        authored_world_nodes, device=soft.device, dtype=default_state.dtype
    )
    default_state[0, :, 3:] = 0.0
    soft.write_nodal_state_to_sim_index(default_state)
    soft.reset()
    # Rest/deformation measurements use the authored measured-volume nodes, not
    # an already-contacted state sampled during simulator initialization.
    rest_nodes = authored_world_nodes
    rest_dimensions, _ = kabsch_dimensions(rest_nodes, rest_nodes)
    soft_node_count = int(rest_nodes.shape[0])

    indentation_cfg = config.get("indentation_diagnostics")
    indentation_enabled = indentation_cfg is not None
    digit_pad_specs: dict[str, dict[str, Any]] = {}
    distal_body_ids: dict[str, int] = {}
    surface_node_ids = np.empty(0, dtype=np.int64)
    surface_rest_normals = np.empty((0, 3), dtype=np.float64)
    surface_node_area_m2 = np.empty(0, dtype=np.float64)
    if indentation_enabled:
        whole_hand_path = Path(config["whole_hand_geometry"]).resolve()
        whole_hand = read_json(whole_hand_path)
        digit_pad_specs = {
            spec["digit_chain"]: spec
            for spec in whole_hand[args.side].values()
            if spec["digit_chain"] in {"thumb", "index", "middle"}
        }
        if set(digit_pad_specs) != {"thumb", "index", "middle"}:
            raise RuntimeError("named whole-hand geometry is missing a Dex3 digit")
        body_names = list(robot.data.body_names)
        distal_body_ids = {
            digit: body_names.index(spec["distal_link"])
            for digit, spec in digit_pad_specs.items()
        }
        surface_faces = np.asarray(
            UsdGeom.TetMesh.ComputeSurfaceFaces(
                UsdGeom.TetMesh(stage.GetPrimAtPath(SOFT_COLLISION)),
                Usd.TimeCode.Default(),
            ),
            dtype=np.int64,
        ).reshape(-1, 3)
        surface_node_ids = np.unique(surface_faces.reshape(-1))
        surface_lookup = {int(node): index for index, node in enumerate(surface_node_ids)}
        surface_node_area_m2 = np.zeros(len(surface_node_ids), dtype=np.float64)
        triangles = rest_nodes[surface_faces]
        triangle_areas = 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )
        for face, area in zip(surface_faces, triangle_areas, strict=True):
            for node in face:
                surface_node_area_m2[surface_lookup[int(node)]] += float(area / 3.0)
        rest_centered = rest_nodes - rest_nodes.mean(axis=0)
        surface_rest = rest_centered[surface_node_ids]
        semi_axes = dimensions / 2.0
        surface_rest_normals = surface_rest / np.square(semi_axes)[None, :]
        surface_rest_normals /= np.maximum(
            np.linalg.norm(surface_rest_normals, axis=1, keepdims=True), 1.0e-12
        )

    records: dict[str, list[Any]] = {
        key: []
        for key in (
            "physics_step",
            "control_frame",
            "stage",
            "commanded_q_rad",
            "measured_q_rad",
            "finger_joint_error_to_full_close_rad",
            "finger_applied_torque_nm",
            "com_world_m",
            "root_velocity_world_m_s",
            "dimensions_material_frame_m",
            "short_axis_width_m",
            "maximum_nodal_deformation_m",
            "minimum_node_world_z_m",
            "maximum_node_speed_m_s",
            "thumb_force_n",
            "index_force_n",
            "middle_force_n",
            "thumb_net_force_n",
            "index_net_force_n",
            "middle_net_force_n",
            "thumb_contact_count",
            "index_contact_count",
            "middle_contact_count",
        )
    }
    if indentation_enabled:
        for digit in ("thumb", "index", "middle"):
            for suffix in (
                "pad_to_surface_signed_distance_m",
                "indentation_depth_m",
                "contact_patch_area_m2",
                "geometric_contact",
            ):
                records[f"{digit}_{suffix}"] = []
    active_slice = slice(14, 21) if args.side == "left" else slice(21, 28)
    full_q = np.asarray(config["hand_states"][args.side]["FULL_CLOSE"], dtype=np.float64)
    # Rigid-contact tensor filters reject deformable colliders in this PhysX
    # version.  The event callback retains exact collider-pair provenance and
    # provides the per-contact impulse vector for force reconstruction.
    callback_step = {
        digit: {"impulse_ns": 0.0, "count": 0}
        for digit in ("thumb", "index", "middle")
    }
    callback_diagnostics: dict[str, Any] = {
        "all_header_count": 0,
        "soft_pair_header_count": 0,
        "path_preview": [],
    }
    raw_actor_path_cache: dict[int, str] = {}

    def raw_deformable_contacts(sensor: ContactSensor) -> tuple[float, int]:
        view = sensor.contact_view
        forces, _, _, _, counts, starts, actor_ids = view.get_raw_contact_data(dt)
        force_values = numpy(forces).reshape(-1)
        count_values = numpy(counts).reshape(-1).astype(np.int64)
        start_values = numpy(starts).reshape(-1).astype(np.int64)
        id_values = numpy(actor_ids).reshape(-1).astype(np.uint64)
        total_force = 0.0
        total_count = 0
        for count, start in zip(count_values, start_values, strict=True):
            for index in range(int(start), int(start + count)):
                actor_id = int(id_values[index])
                if actor_id not in raw_actor_path_cache:
                    cpu_id = wp.array(
                        np.asarray([actor_id], dtype=np.uint64),
                        dtype=wp.uint64,
                        device="cpu",
                    )
                    resolved = view.get_other_actor_paths_from_ids(cpu_id)
                    raw_actor_path_cache[actor_id] = str(resolved[0]) if resolved else ""
                if raw_actor_path_cache[actor_id].startswith(SOFT_DOLL):
                    total_force += float(abs(force_values[index]))
                    total_count += 1
        return total_force, total_count

    def on_contact_report(contact_headers, contact_data) -> None:
        for header in contact_headers:
            callback_diagnostics["all_header_count"] += 1
            paths = [
                str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            ]
            if not any(path.startswith(SOFT_DOLL) for path in paths):
                continue
            callback_diagnostics["soft_pair_header_count"] += 1
            if len(callback_diagnostics["path_preview"]) < 24:
                callback_diagnostics["path_preview"].append(paths)
            digit = next(
                (
                    name
                    for name in ("thumb", "index", "middle")
                    if any(f"{args.side}_hand_{name}_" in path for path in paths)
                ),
                None,
            )
            if digit is None:
                continue
            start = int(header.contact_data_offset)
            stop = start + int(header.num_contact_data)
            for index in range(start, stop):
                impulse = np.asarray(contact_data[index].impulse, dtype=np.float64)
                callback_step[digit]["impulse_ns"] += float(np.linalg.norm(impulse))
                callback_step[digit]["count"] += 1

    contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
        on_contact_report
    )
    physics_step = 0
    for control_frame, (command, label) in enumerate(zip(commands, stages, strict=True)):
        target[0, joint_ids] = torch.as_tensor(
            command, device=robot.device, dtype=target.dtype
        )
        for _ in range(substeps):
            for digit in callback_step:
                callback_step[digit]["impulse_ns"] = 0.0
                callback_step[digit]["count"] = 0
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            soft.write_data_to_sim()
            sim.step(render=not args.headless)
            robot.update(dt)
            soft.update(dt)
            raw_contacts = {
                digit: raw_deformable_contacts(sensor)
                for digit, sensor in digit_sensors.items()
            }
            nodes = numpy(soft.data.nodal_pos_w)[0].astype(np.float64)
            velocities = numpy(soft.data.nodal_vel_w)[0].astype(np.float64)
            material_dimensions, maximum_deformation = kabsch_dimensions(rest_nodes, nodes)
            measured = numpy(robot.data.joint_pos)[0, joint_ids].astype(np.float64)
            applied = numpy(robot.data.applied_torque)[0, joint_ids][active_slice].astype(np.float64)
            root_velocity = numpy(soft.data.root_vel_w)[0].astype(np.float64)
            callback_forces = {
                digit: float(callback_step[digit]["impulse_ns"] / dt)
                for digit in callback_step
            }
            # The callback stream is retained for capability auditing, while
            # the unfiltered raw tensor stream is authoritative on 6.1.
            raw_forces = {digit: raw_contacts[digit][0] for digit in raw_contacts}
            raw_counts = {digit: raw_contacts[digit][1] for digit in raw_contacts}
            geometric_values: dict[str, Any] = {}
            if indentation_enabled:
                current_local, rest_centered, _, _ = kabsch_material_state(rest_nodes, nodes)
                surface_current_local = current_local[surface_node_ids]
                surface_inward_displacement = np.einsum(
                    "ij,ij->i",
                    rest_centered[surface_node_ids] - surface_current_local,
                    surface_rest_normals,
                )
                body_positions = numpy(robot.data.body_pos_w)[0].astype(np.float64)
                body_quaternions = numpy(robot.data.body_quat_w)[0].astype(np.float64)
                surface_nodes_world = nodes[surface_node_ids]
                contact_tolerance = float(
                    indentation_cfg["geometric_contact_tolerance_m"]
                )
                for digit in ("thumb", "index", "middle"):
                    spec = digit_pad_specs[digit]
                    body_id = distal_body_ids[digit]
                    body_rotation = rotation_xyzw(body_quaternions[body_id])
                    pad_center = body_positions[body_id] + body_rotation @ np.asarray(
                        spec["local_position_xyz_m"], dtype=np.float64
                    )
                    signed_distance = point_to_oriented_box_signed_distance(
                        surface_nodes_world,
                        pad_center,
                        body_rotation,
                        np.asarray(spec["pad_half_extent_m"], dtype=np.float64),
                    )
                    proximity = signed_distance <= contact_tolerance
                    positive_indentation = np.maximum(surface_inward_displacement, 0.0)
                    geometric_values[f"{digit}_pad_to_surface_signed_distance_m"] = float(
                        np.min(signed_distance)
                    )
                    geometric_values[f"{digit}_indentation_depth_m"] = float(
                        np.max(positive_indentation[proximity], initial=0.0)
                    )
                    geometric_values[f"{digit}_contact_patch_area_m2"] = float(
                        np.sum(surface_node_area_m2[proximity])
                    )
                    geometric_values[f"{digit}_geometric_contact"] = bool(
                        np.any(proximity)
                    )
            values = {
                "physics_step": physics_step,
                "control_frame": control_frame,
                "stage": str(label),
                "commanded_q_rad": command,
                "measured_q_rad": measured,
                "finger_joint_error_to_full_close_rad": measured[active_slice] - full_q,
                "finger_applied_torque_nm": applied,
                "com_world_m": nodes.mean(axis=0),
                "root_velocity_world_m_s": root_velocity,
                "dimensions_material_frame_m": material_dimensions,
                "short_axis_width_m": material_dimensions[1],
                "maximum_nodal_deformation_m": maximum_deformation,
                "minimum_node_world_z_m": float(nodes[:, 2].min()),
                "maximum_node_speed_m_s": float(np.linalg.norm(velocities, axis=1).max()),
                "thumb_force_n": raw_forces["thumb"],
                "index_force_n": raw_forces["index"],
                "middle_force_n": raw_forces["middle"],
                "thumb_net_force_n": callback_forces["thumb"],
                "index_net_force_n": callback_forces["index"],
                "middle_net_force_n": callback_forces["middle"],
                "thumb_contact_count": raw_counts["thumb"],
                "index_contact_count": raw_counts["index"],
                "middle_contact_count": raw_counts["middle"],
                **geometric_values,
            }
            for key, value in values.items():
                records[key].append(value)
            physics_step += 1

    arrays = {key: np.asarray(value) for key, value in records.items()}
    log_path = output_dir / "deformation_contact_log.npz"
    temporary = log_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, joint_names=np.asarray(names))
    os.replace(temporary, log_path)

    stage_values = arrays["stage"].astype(str)
    gates = config["gates"]
    settle_mask = stage_values == "GRAVITY_SETTLE"
    finite = bool(
        all(
            np.isfinite(value).all()
            for value in arrays.values()
            if value.dtype.kind not in "USO"
        )
    )
    stable = bool(
        finite
        and np.max(arrays["maximum_node_speed_m_s"], initial=0.0)
        <= float(gates["maximum_node_speed_m_s"])
        and np.min(arrays["dimensions_material_frame_m"])
        >= float(gates["minimum_noncollapsed_axis_m"])
        and np.max(arrays["dimensions_material_frame_m"] / rest_dimensions)
        <= float(gates["maximum_axis_expansion_ratio"])
    )
    unloaded_width = float(rest_dimensions[1])
    settled_dimensions = np.median(
        arrays["dimensions_material_frame_m"][settle_mask][-substeps:], axis=0
    )
    settled_width = float(settled_dimensions[1])
    settled_height = float(settled_dimensions[2])
    initial_dimension_pass = bool(
        np.all(
            np.abs(rest_dimensions - dimensions)
            <= float(gates["unloaded_dimension_relative_tolerance"]) * dimensions
        )
    )
    gravity_stable = bool(
        settled_width
        >= float(gates["minimum_gravity_settled_short_width_ratio"]) * unloaded_width
        and settled_height
        >= float(gates.get("minimum_gravity_settled_height_ratio", 0.0))
        * float(rest_dimensions[2])
    )
    if args.probe_only:
        probe_pass = bool(initial_dimension_pass and gravity_stable and stable)
        result = {
            "schema_version": f"{config['schema_version']}_native_probe",
            "status": "PASS" if probe_pass else "FAIL",
            "side": args.side,
            "material": material,
            "method": config["object"]["method"],
            "single_deformable_object": True,
            "single_continuous_body": True,
            "unloaded_dimensions_m": dimensions,
            "mass_kg": config["object"]["mass_kg"],
            "soft_node_count": soft_node_count,
            "spawn_audit": spawn_audit,
            "old_doll_audit": old_doll,
            "rest_dimensions_m": rest_dimensions,
            "gravity_settled_short_width_m": settled_width,
            "gravity_settled_short_width_ratio": float(settled_width / unloaded_width),
            "gravity_settled_height_m": settled_height,
            "gravity_settled_height_ratio": float(settled_height / rest_dimensions[2]),
            "initial_dimensions_pass": initial_dimension_pass,
            "gravity_stable": gravity_stable,
            "stability": {
                "status": "PASS" if stable else "FAIL",
                "finite": finite,
                "maximum_node_speed_m_s": float(
                    np.max(arrays["maximum_node_speed_m_s"], initial=0.0)
                ),
                "minimum_axis_extent_m": float(
                    np.min(arrays["dimensions_material_frame_m"])
                ),
                "maximum_axis_expansion_ratio": float(
                    np.max(arrays["dimensions_material_frame_m"] / rest_dimensions)
                ),
            },
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_scene_sha256": sha256_file(Path(config["source_scene"])),
            "scripted_command": str(command_path),
            "scripted_command_sha256": sha256_file(command_path),
            "deformation_contact_log": str(log_path),
            "deformation_contact_log_sha256": sha256_file(log_path),
            "object_pose_writes_during_timed_loop": 0,
            "attachment_used": False,
            "real_robot": False,
            "learned_policy": False,
        }
        result_path = output_dir / "trial_result.json"
        atomic_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
        return 0 if probe_pass else 2

    squeeze_mask = np.isin(stage_values, ["FULL_CLOSE", "SQUEEZE"])
    retained_mask = np.isin(stage_values, ["SQUEEZE", "LIFT_50MM", "HOLD_ELEVATED"])
    elevated_mask = stage_values == "HOLD_ELEVATED"
    recovery_mask = stage_values == "RECOVERY"
    prelift_com = np.mean(arrays["com_world_m"][stage_values == "SQUEEZE"][-substeps:], axis=0)
    elevated_com = np.mean(arrays["com_world_m"][elevated_mask][-substeps:], axis=0)
    com_rise = float(elevated_com[2] - prelift_com[2])
    compressed_width = float(np.min(arrays["short_axis_width_m"][squeeze_mask]))
    recovered_width = float(np.median(arrays["short_axis_width_m"][recovery_mask][-substeps:]))
    reference_com = np.mean(
        arrays["com_world_m"][stage_values == "OPEN"][-substeps:], axis=0
    )
    pre_release_escape_mask = np.isin(
        stage_values,
        ["PRESHAPE", "FULL_CLOSE", "SQUEEZE", "LIFT_50MM", "HOLD_ELEVATED"],
    )
    maximum_horizontal_displacement = float(
        np.max(
            np.linalg.norm(
                arrays["com_world_m"][pre_release_escape_mask, :2]
                - reference_com[None, :2],
                axis=1,
            )
        )
    )
    digit_results: dict[str, Any] = {}
    digit_contact_masks: list[np.ndarray] = []
    for digit in ("thumb", "index", "middle"):
        force = arrays[f"{digit}_force_n"]
        if indentation_enabled:
            geometric_contact = arrays[f"{digit}_geometric_contact"].astype(bool)
            indentation = arrays[f"{digit}_indentation_depth_m"]
            meaningful = geometric_contact & (
                indentation >= float(gates["minimum_digit_indentation_m"])
            )
            maximum_indentation = float(np.max(indentation, initial=0.0))
            maximum_patch_area = float(
                np.max(arrays[f"{digit}_contact_patch_area_m2"], initial=0.0)
            )
            preshape_patch_area = float(
                np.median(
                    arrays[f"{digit}_contact_patch_area_m2"][stage_values == "PRESHAPE"]
                )
            )
            squeeze_patch_area = float(
                np.median(
                    arrays[f"{digit}_contact_patch_area_m2"][stage_values == "SQUEEZE"]
                )
            )
            minimum_pad_distance = float(
                np.min(arrays[f"{digit}_pad_to_surface_signed_distance_m"])
            )
        else:
            geometric_contact = np.zeros_like(force, dtype=bool)
            meaningful = force >= float(gates["meaningful_digit_force_n"])
            maximum_indentation = 0.0
            maximum_patch_area = 0.0
            preshape_patch_area = 0.0
            squeeze_patch_area = 0.0
            minimum_pad_distance = None
        digit_contact_masks.append(meaningful)
        digit_results[digit] = {
            "contact": bool(np.any(meaningful)),
            "geometric_contact_detected": bool(np.any(geometric_contact)),
            "maximum_force_n": float(np.max(force, initial=0.0)),
            "mean_force_during_retention_n": float(
                np.mean(force[retained_mask]) if np.any(retained_mask) else 0.0
            ),
            "maximum_indentation_depth_m": maximum_indentation,
            "minimum_pad_to_surface_signed_distance_m": minimum_pad_distance,
            "maximum_contact_patch_area_m2": maximum_patch_area,
            "median_preshape_contact_patch_area_m2": preshape_patch_area,
            "median_squeeze_contact_patch_area_m2": squeeze_patch_area,
            "retained_contact_duration_s": longest_duration(
                meaningful & retained_mask, dt
            ),
            "elevated_contact_duration_s": longest_duration(
                meaningful & elevated_mask, dt
            ),
        }
    all_digit_contact = np.logical_and.reduce(digit_contact_masks)
    table_clear = (
        arrays["minimum_node_world_z_m"]
        > float(config["object"]["table_surface_world_z_m"])
        + float(gates["minimum_table_clearance_m"])
    )
    elevated_retention = longest_duration(all_digit_contact & table_clear & elevated_mask, dt)
    visibly_compressed = bool(
        compressed_width
        <= float(gates["maximum_squeezed_short_width_ratio"]) * unloaded_width
    )
    recovered = bool(
        recovered_width
        >= float(gates["minimum_recovered_short_width_ratio"]) * unloaded_width
    )
    contact_pass = all(
        row["retained_contact_duration_s"]
        >= float(
            indentation_cfg["minimum_sustained_contact_s"]
            if indentation_enabled
            else gates["minimum_digit_contact_s"]
        )
        for row in digit_results.values()
    )
    lateral_escape_pass = bool(
        maximum_horizontal_displacement
        <= float(gates.get("maximum_horizontal_escape_m", float("inf")))
    )
    lift_pass = bool(
        com_rise >= float(gates["minimum_com_rise_m"])
        and elevated_retention >= float(gates["minimum_elevated_retention_s"])
    )
    passed = bool(
        not args.probe_only
        and initial_dimension_pass
        and gravity_stable
        and stable
        and visibly_compressed
        and recovered
        and contact_pass
        and lift_pass
        and lateral_escape_pass
    )
    failure_mechanism = (
        "NONE"
        if passed
        else "OBJECT_EXPELLED_BEFORE_SUSTAINED_THREE_DIGIT_ENCLOSURE"
        if not lateral_escape_pass
        else "THREE_DIGIT_CONTACT_NOT_SUSTAINED"
        if not contact_pass
        else "INSUFFICIENT_COM_RISE_OR_ELEVATED_RETENTION"
        if not lift_pass
        else "INSUFFICIENT_COMPRESSION"
        if not visibly_compressed
        else "INSUFFICIENT_POST_RELEASE_RECOVERY"
        if not recovered
        else "NUMERICAL_STABILITY_OR_UNLOADED_SHAPE_GATE"
    )
    result = {
        "schema_version": f"{config['schema_version']}_trial",
        "status": "PASS" if passed else "PROBE_PASS" if args.probe_only and stable else "FAIL",
        "side": args.side,
        "material": material,
        "method": config["object"]["method"],
        "single_deformable_object": True,
        "single_continuous_body": True,
        "unloaded_dimensions_m": dimensions,
        "mass_kg": config["object"]["mass_kg"],
        "soft_node_count": soft_node_count,
        "spawn_audit": spawn_audit,
        "old_doll_audit": old_doll,
        "deformation": {
            "rest_dimensions_m": rest_dimensions,
            "gravity_settled_short_width_m": settled_width,
            "gravity_settled_height_m": settled_height,
            "maximum_squeeze_short_width_m": compressed_width,
            "short_width_deformation_ratio": float(1.0 - compressed_width / unloaded_width),
            "recovered_short_width_m": recovered_width,
            "recovery_ratio": float(recovered_width / unloaded_width),
            "maximum_nodal_deformation_m": float(
                np.max(arrays["maximum_nodal_deformation_m"], initial=0.0)
            ),
            "initial_dimensions_pass": initial_dimension_pass,
            "gravity_stable": gravity_stable,
            "visibly_compressed": visibly_compressed,
            "recovered": recovered,
        },
        "object_motion": {
            "maximum_horizontal_displacement_from_open_reference_m": maximum_horizontal_displacement,
            "lateral_escape_gate_pass": lateral_escape_pass,
        },
        "digits": digit_results,
        "lift": {
            "requested_m": timing["requested_lift_m"],
            "measured_com_rise_m": com_rise,
            "minimum_node_z_during_final_elevated_frame_m": float(
                np.min(arrays["minimum_node_world_z_m"][elevated_mask][-substeps:])
            ),
            "table_unsupported_duration_s": longest_duration(table_clear & elevated_mask, dt),
            "all_three_digit_elevated_retention_s": elevated_retention,
            "status": "PASS" if lift_pass else "FAIL",
        },
        "stability": {
            "status": "PASS" if stable else "FAIL",
            "finite": finite,
            "maximum_node_speed_m_s": float(
                np.max(arrays["maximum_node_speed_m_s"], initial=0.0)
            ),
            "minimum_axis_extent_m": float(np.min(arrays["dimensions_material_frame_m"])),
            "maximum_axis_expansion_ratio": float(
                np.max(arrays["dimensions_material_frame_m"] / rest_dimensions)
            ),
        },
        "contact_sensor": {
            "force_source": "PhysX 6.1 unfiltered rigid raw-contact force selected by deformable other-actor ID",
            "pair_match_prefix": SOFT_DOLL,
            "unfiltered_raw_tensor_used": True,
            "reason_filtered_matrix_not_used": "PhysX GPU tensor contact filters reject deformable colliders in Isaac Sim 6.1",
            "gpu_deformable_filter_used": False,
            "contact_report_rigid_links": contact_report_links,
            "raw_other_actor_paths": sorted(set(raw_actor_path_cache.values())),
            "callback_diagnostics": callback_diagnostics,
            "geometric_indentation_diagnostics": (
                {
                    "enabled": True,
                    "method": indentation_cfg["method"],
                    "whole_hand_geometry": str(Path(config["whole_hand_geometry"]).resolve()),
                    "surface_node_count": int(len(surface_node_ids)),
                    "contact_tolerance_m": indentation_cfg[
                        "geometric_contact_tolerance_m"
                    ],
                    "minimum_meaningful_indentation_m": gates[
                        "minimum_digit_indentation_m"
                    ],
                }
                if indentation_enabled
                else {"enabled": False}
            ),
        },
        "controller": {
            "full_close_target_held": True,
            "maximum_applied_finger_torque_nm": float(
                np.max(np.abs(arrays["finger_applied_torque_nm"]), initial=0.0)
            ),
            "mean_absolute_full_close_error_during_squeeze_rad": float(
                np.mean(
                    np.abs(arrays["finger_joint_error_to_full_close_rad"][
                        stage_values == "SQUEEZE"
                    ])
                )
            ),
        },
        "prohibited_mechanisms": {
            "internal_spring_lobes": False,
            "attachment": False,
            "magnet": False,
            "weld": False,
            "object_follow": False,
            "object_pose_writes_during_timed_loop": 0,
            "fingertip_target_points": False,
            "proposed_b_values": False,
        },
        "probe_only": bool(args.probe_only),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_scene_sha256": sha256_file(Path(config["source_scene"])),
        "source_arm_primitive": str(primitive_path),
        "source_arm_primitive_sha256": sha256_file(primitive_path),
        "scripted_command": str(command_path),
        "scripted_command_sha256": sha256_file(command_path),
        "deformation_contact_log": str(log_path),
        "deformation_contact_log_sha256": sha256_file(log_path),
        "real_robot": False,
        "learned_policy": False,
        "failure_mechanism": failure_mechanism,
    }
    result_path = output_dir / "trial_result.json"
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
    return 0 if (passed or (args.probe_only and stable)) else 2


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
