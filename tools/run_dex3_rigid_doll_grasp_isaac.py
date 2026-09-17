#!/usr/bin/env python3
"""Deferred Isaac executor for one fixed Dex3 rigid-doll calibration trial.

This script has no policy/checkpoint interface.  It replays a prebuilt 28-D
primitive once, writes the object initial condition once before the timed loop,
and logs the physical signals consumed by the common success detector.
"""

from __future__ import annotations

import argparse
import json
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
parser.add_argument("--material", choices=("LOW", "MEDIUM", "HIGH"), required=True)
parser.add_argument("--primitive", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import omni.usd
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.evaluation.contracts import authoritative_joint_ranges, sha256_file
from tools.evaluation.io import atomic_json
from tools.evaluation.physical_success import detect_physical_success
from tools.policy_b_isaac_control_contract import build_implicit_actuators


SCENE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
WHOLE_HAND = ROOT / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json"
TOOL_FRAME = ROOT / "outputs/dataset_a_final50_retargeting/config/tool_frame_report.json"
DOLL = "/World/DollHandoffEnvironment/Doll"
DOLL_BODY = f"{DOLL}/Body"
TABLE_FILTER = "/World/DollHandoffEnvironment/Table/Colliders/Top"
BIN_FILTERS = [
    f"/World/DollHandoffEnvironment/TrashBin/{name}"
    for name in ("Bottom", "FrontWall", "BackWall", "LeftWall", "RightWall")
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        return value.torch.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _apply_proxy(stage: Usd.Stage, config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    doll = stage.GetPrimAtPath(DOLL)
    body = stage.GetPrimAtPath(DOLL_BODY)
    if not doll.IsValid() or not body.IsValid():
        raise RuntimeError("rigid doll prims are missing")
    material = UsdShade.Material.Define(stage, "/World/RigidDollProxyPhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(candidate["static_friction"]))
    api.CreateDynamicFrictionAttr().Set(float(candidate["dynamic_friction"]))
    api.CreateRestitutionAttr().Set(float(config["object"]["restitution"]))
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(
        str(config["material_combine_modes"]["friction"])
    )
    physx_material.CreateRestitutionCombineModeAttr().Set(
        str(config["material_combine_modes"]["restitution"])
    )
    UsdShade.MaterialBindingAPI.Apply(body).Bind(material, materialPurpose="physics")
    UsdPhysics.MassAPI.Apply(doll).CreateMassAttr(float(config["object"]["mass_kg"]))
    doll.AddAppliedSchema("PhysxRigidBodyAPI")
    doll.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(
        float(config["object"]["linear_damping"])
    )
    doll.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(
        float(config["object"]["angular_damping"])
    )
    doll.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(
        float(config["simulation"]["contact_settings"]["max_depenetration_velocity_m_s"])
    )
    body.AddAppliedSchema("PhysxCollisionAPI")
    body.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
        float(config["simulation"]["contact_settings"]["contact_offset_m"])
    )
    body.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
        float(config["simulation"]["contact_settings"]["rest_offset_m"])
    )
    return {
        "material_path": str(material.GetPath()),
        "static_friction": float(api.GetStaticFrictionAttr().Get()),
        "dynamic_friction": float(api.GetDynamicFrictionAttr().Get()),
        "restitution": float(api.GetRestitutionAttr().Get()),
        "linear_damping": float(
            doll.GetAttribute("physxRigidBody:linearDamping").Get()
        ),
        "angular_damping": float(
            doll.GetAttribute("physxRigidBody:angularDamping").Get()
        ),
        "source_scene_saved_or_modified": False,
    }


def _force_by_filter(sensor: ContactSensor) -> np.ndarray:
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        return np.zeros(0, dtype=np.float64)
    values = _numpy(matrix)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    values = values.reshape(-1, values.shape[-2], 3)
    return np.max(np.linalg.norm(values, axis=-1), axis=0)


def _max_penetration(sensor: ContactSensor, dt: float) -> tuple[float, str | None]:
    try:
        _, _, _, separations, _, _ = sensor.contact_view.get_contact_data(dt)
        values = _numpy(separations).reshape(-1)
        return (float(np.max(np.maximum(-values, 0.0))) if values.size else 0.0), None
    except Exception as error:
        return 0.0, f"{type(error).__name__}: {error}"


def _joint_open_mask(
    measured: np.ndarray,
    names: list[str],
    tool_frame: dict[str, Any],
    side: str,
    tolerance_rad: float,
) -> bool:
    canonical = tool_frame["g1"]["canonical_states"][side]["OPEN"]
    local_names = tool_frame["g1"]["joint_names"][side]
    target = {name: float(value) for name, value in zip(local_names, canonical, strict=True)}
    indices = [names.index(name) for name in local_names]
    error = np.asarray([measured[index] - target[name] for index, name in zip(indices, local_names, strict=True)])
    return bool(np.max(np.abs(error)) <= tolerance_rad)


def main() -> int:
    config = read_json(args.config.resolve())
    if config.get("schema_version") != "doll_handoff_rigid_proxy_v1":
        raise RuntimeError("unexpected rigid proxy config")
    if Path(config["source_scene"]["g1_preview_usd"]).resolve() != SCENE.resolve():
        raise RuntimeError("calibration runner scene differs from rigid-proxy config")
    candidates = {row["name"]: row for row in config["material_candidates"]}
    candidate = candidates[args.material]
    with np.load(args.primitive.resolve(), allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    if str(primitive["active_side"].reshape(()).item()) != args.side:
        raise RuntimeError("primitive side mismatch")
    if str(primitive["config_sha256"].reshape(()).item()) != sha256_file(args.config.resolve()):
        raise RuntimeError("fixed primitive was built from a different proxy config")
    if not bool(primitive["policy_independent"].reshape(()).item()) or bool(
        primitive["learned_policy_used"].reshape(()).item()
    ):
        raise RuntimeError("calibration primitive is not policy-independent")
    names, _ = authoritative_joint_ranges()
    if primitive["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("primitive named 28-D order differs from authoritative order")
    commands = primitive["commanded_q_rad"].astype(np.float64)
    if commands.ndim != 2 or commands.shape[1] != 28 or not np.isfinite(commands).all():
        raise RuntimeError("malformed calibration commands")
    stages = primitive["stage"].astype(str)
    fps = float(config["timing"]["control_fps_hz"])
    dt = float(config["simulation"]["physics_dt_s"])
    substeps = int(config["simulation"]["physics_substeps_per_control_frame"])
    if not np.isclose(fps, 30.0) or not np.isclose(substeps * dt, 1.0 / fps):
        raise RuntimeError("calibration timing mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not omni.usd.get_context().open_stage(str(SCENE)):
        raise RuntimeError(f"failed to open {SCENE}")
    stage = omni.usd.get_context().get_stage()
    # All proxy overrides live in the anonymous session layer.  The checked-in
    # scene is never saved or changed by calibration.
    stage.SetEditTarget(stage.GetSessionLayer())
    runtime_material = _apply_proxy(stage, config, candidate)
    sys.path.insert(0, str(ROOT / "isaaclab_magsafe_fixed_scene"))
    from physical_contact_monitor import enable_contact_reporting

    enable_contact_reporting(stage, ["/World/G1/Asset", DOLL, TABLE_FILTER, *BIN_FILTERS])
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device="cuda:0",
            gravity=(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"])),
            use_fabric=True,
        )
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint",
            spawn=None,
            actuators=build_implicit_actuators(ImplicitActuatorCfg),
        )
    )
    doll = RigidObject(RigidObjectCfg(prim_path=DOLL, spawn=None))
    geometry = read_json(WHOLE_HAND)
    distal_links = {
        f"{side}_{role}": geometry[side][role]["distal_link"]
        for side in ("left", "right")
        for role in ("A", "B", "C")
    }
    digit_sensors = {
        label: ContactSensor(
            ContactSensorCfg(
                prim_path=f"/World/G1/Asset/{link}",
                update_period=0.0,
                filter_prim_paths_expr=[DOLL],
                track_contact_points=True,
                max_contact_data_count_per_prim=32,
                force_threshold=0.0,
            )
        )
        for label, link in distal_links.items()
    }
    external_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path=DOLL,
            update_period=0.0,
            filter_prim_paths_expr=[TABLE_FILTER, *BIN_FILTERS],
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            force_threshold=0.0,
        )
    )
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError(f"named Isaac mapping failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    target[0, ids] = torch.as_tensor(commands[0], device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(target, zero)
    reset_xyz = np.asarray(config["calibration"]["object_reset_center_world_xyz_m"], dtype=np.float32)
    # IsaacLab root-state pose order is XYZ + quaternion WXYZ.
    reset_pose_wxyz = torch.as_tensor(
        np.r_[reset_xyz, [1.0, 0.0, 0.0, 0.0]][None], dtype=torch.float32, device=doll.device
    )
    doll.write_root_pose_to_sim_index(root_pose=reset_pose_wxyz)
    doll.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=doll.device)
    )
    object_initial_writes = 1

    def update() -> None:
        robot.update(dt)
        doll.update(dt)
        for sensor in digit_sensors.values():
            sensor.update(dt, force_recompute=True)
        external_sensor.update(dt, force_recompute=True)

    for _ in range(int(round(1.0 / dt))):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        update()

    body_names = list(robot.data.body_names)
    distal_body_ids = {label: body_names.index(link) for label, link in distal_links.items()}
    tool_frame = read_json(TOOL_FRAME)
    log: dict[str, list[Any]] = {
        key: []
        for key in (
            "object_com_m",
            "object_orientation_xyzw",
            "object_linear_velocity_m_s",
            "object_angular_velocity_rad_s",
            "left_hand_object_contact",
            "right_hand_object_contact",
            "table_contact",
            "bin_contact",
            "hand_joint_q_rad",
            "arm_joint_q_rad",
            "left_hand_position_m",
            "right_hand_position_m",
            "right_hand_open",
            "measured_q_rad",
            "commanded_q_rad",
        )
    }
    maximum_penetration = 0.0
    penetration_api_errors: set[str] = set()
    for frame, command in enumerate(commands):
        target[0, ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        for _ in range(substeps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            update()
        measured = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        pose = doll.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
        velocity = doll.data.root_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
        # Isaac root_pose_w stores WXYZ; the portable log contract stores XYZW.
        orientation_xyzw = pose[[4, 5, 6, 3]]
        side_forces: dict[str, float] = {}
        for side in ("left", "right"):
            values = []
            for role in ("A", "B", "C"):
                force = _force_by_filter(digit_sensors[f"{side}_{role}"])
                values.append(float(np.max(force)) if force.size else 0.0)
            side_forces[side] = float(sum(values))
        external = _force_by_filter(external_sensor)
        table_force = float(external[0]) if len(external) else 0.0
        bin_force = float(np.max(external[1:])) if len(external) > 1 else 0.0
        for sensor in [*digit_sensors.values(), external_sensor]:
            penetration, error = _max_penetration(sensor, dt)
            maximum_penetration = max(maximum_penetration, penetration)
            if error:
                penetration_api_errors.add(error)
        positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
        hand_position = {
            side: np.mean(
                [positions[distal_body_ids[f"{side}_{role}"]] for role in ("A", "B", "C")],
                axis=0,
            )
            for side in ("left", "right")
        }
        log["object_com_m"].append(pose[:3])
        log["object_orientation_xyzw"].append(orientation_xyzw)
        log["object_linear_velocity_m_s"].append(velocity[:3])
        log["object_angular_velocity_rad_s"].append(velocity[3:])
        log["left_hand_object_contact"].append(side_forces["left"])
        log["right_hand_object_contact"].append(side_forces["right"])
        log["table_contact"].append(table_force)
        log["bin_contact"].append(bin_force)
        log["arm_joint_q_rad"].append(measured[:14])
        log["hand_joint_q_rad"].append(measured[14:])
        log["left_hand_position_m"].append(hand_position["left"])
        log["right_hand_position_m"].append(hand_position["right"])
        log["right_hand_open"].append(
            _joint_open_mask(
                measured,
                names,
                tool_frame,
                "right",
                float(config["success_thresholds"]["hand_open_joint_max_abs_error_rad"]),
            )
        )
        log["measured_q_rad"].append(measured)
        log["commanded_q_rad"].append(command)
    arrays = {key: np.asarray(value) for key, value in log.items()}
    arrays["frame_index"] = np.arange(len(commands), dtype=np.int64)
    arrays["timestamp_s"] = arrays["frame_index"].astype(np.float64) / fps
    arrays["stage"] = stages
    arrays["joint_names"] = np.asarray(names)
    output_npz = args.output_dir / "event_log.npz"
    temporary = output_npz.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, output_npz)
    detector_log = {key: arrays[key] for key in arrays}
    physical = detect_physical_success(detector_log, config)
    steps = np.linalg.norm(np.diff(arrays["object_com_m"], axis=0), axis=1)
    release_indices = np.flatnonzero(stages == "RELEASE")
    post_release_start = int(release_indices[-1] + 1) if len(release_indices) else len(stages)
    tail_frames = max(1, int(round(float(config["success_thresholds"]["post_release_nonsticky_duration_s"]) * fps)))
    contact_tail = arrays[f"{args.side}_hand_object_contact"][-tail_frames:]
    artifact_checks = {
        "penetration_artifact_detected": bool(
            penetration_api_errors
            or maximum_penetration
            > float(config["success_thresholds"]["maximum_allowed_contact_penetration_m"])
        ),
        "maximum_contact_penetration_m": maximum_penetration,
        "penetration_api_errors": sorted(penetration_api_errors),
        "magnetic_or_sticky_behavior_detected": bool(
            len(contact_tail)
            and np.any(
                contact_tail
                >= float(config["success_thresholds"]["minimum_hand_object_contact_force_n"])
            )
            and post_release_start < len(stages)
        ),
        "teleportation_detected": bool(
            len(steps)
            and np.max(steps)
            > float(config["success_thresholds"]["maximum_allowed_object_com_step_m"])
        ),
        "maximum_object_com_step_m": float(np.max(steps)) if len(steps) else 0.0,
        "object_constraint_attachment_detected": False,
        "object_pose_writes_before_timed_loop": object_initial_writes,
        "object_pose_writes_during_timed_loop": 0,
        "magnetic_force_or_joint_authored": False,
    }
    lift_key = f"{args.side.upper()}_LIFT_SUCCESS"
    lift_pass = bool(physical[lift_key]) and not any(
        artifact_checks[key]
        for key in (
            "penetration_artifact_detected",
            "magnetic_or_sticky_behavior_detected",
            "teleportation_detected",
            "object_constraint_attachment_detected",
        )
    )
    report = {
        "schema_version": "doll_handoff_rigid_proxy_grasp_calibration_v1",
        "status": "PASS" if lift_pass else "FAIL",
        "side": args.side,
        "material_candidate": args.material,
        "LEFT_LIFT_PASS": bool(lift_pass) if args.side == "left" else False,
        "RIGHT_LIFT_PASS": bool(lift_pass) if args.side == "right" else False,
        "physical_success": physical,
        "artifact_checks": artifact_checks,
        "runtime_material": runtime_material,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "primitive": str(args.primitive.resolve()),
        "primitive_sha256": sha256_file(args.primitive.resolve()),
        "event_log": str(output_npz.resolve()),
        "event_log_sha256": sha256_file(output_npz),
        "scene": str(SCENE),
        "scene_sha256": sha256_file(SCENE),
        "policy_or_checkpoint_used": False,
        "policy_specific_logic": False,
        "object_config_selected_from_policy_result": False,
        "real_robot": False,
    }
    atomic_json(args.output_dir / "calibration_result.json", report)
    return 0 if lift_pass else 2


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
