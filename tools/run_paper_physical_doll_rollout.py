#!/usr/bin/env python3
"""Deferred, common ACT-A/B physical rigid-doll rollout in Isaac.

This runner is intentionally separate from the current paper-core kinematic
source-conditioned experiment.  It will run only with a hash-verified frozen
rigid-proxy config and uses identical physics, initial state, source clock,
deployment projection, safety gates, logging, and success thresholds for A/B.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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

from paper_core_source_rollout_common import (
    ACTE1Bridge,
    COMMON_PROJECTION_FREEZE,
    COMMON_PROJECTION_SHA256,
    EXECUTION_CONFIG,
    EXECUTION_CONFIG_SHA256,
    INITIAL_CONTRACT_SHA256,
    SafetyAudit,
    SourceVideo,
    atomic_json,
    atomic_npz,
    common_initial_condition,
    frozen_interfaces,
    read_json,
    rollout_dynamics,
    sha256_array,
    sha256_file,
)
from policy_b_isaac_control_contract import CONTROL_FPS, PHYSICS_DT, build_implicit_actuators


SCENE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EXPERIMENT2 = ROOT / "outputs/paper_core_ab/offline_heldout8/experiment2_result.json"
DEFAULT_CONFIG = ROOT / "configs/doll_handoff_rigid_proxy_v1.json"
DEFAULT_CONFIG_MANIFEST = ROOT / "configs/doll_handoff_rigid_proxy_v1.sha256.json"
GEOMETRY = ROOT / "configs/doll_handoff_retargeting/dex3_whole_hand.sim.json"
TOOL_FRAME = ROOT / "outputs/dataset_a_final50_retargeting/config/tool_frame_report.json"
DOLL = "/World/DollHandoffEnvironment/Doll"
DOLL_BODY = f"{DOLL}/Body"
TABLE = "/World/DollHandoffEnvironment/Table/Colliders/Top"
BIN_PARTS = [
    f"/World/DollHandoffEnvironment/TrashBin/{name}"
    for name in ("Bottom", "FrontWall", "BackWall", "LeftWall", "RightWall")
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--method", choices=("a", "b"), required=True)
parser.add_argument("--heldout-episode", type=int, choices=range(8), required=True)
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
parser.add_argument("--config-manifest", type=Path, default=DEFAULT_CONFIG_MANIFEST)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--settle-seconds", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import omni.usd
import torch
from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from tools.evaluation.contracts import PHYSICAL_SUCCESS
from tools.evaluation.physical_success import detect_physical_success


def _method_record() -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    result = read_json(EXPERIMENT2)
    if result.get("status") != "PASS":
        raise RuntimeError("selected held-out ACT checkpoint result is not PASS")
    method = result["methods"][args.method]
    selection = method["checkpoint_selection"]
    checkpoint = Path(selection["selected_checkpoint"]).resolve()
    model_sha = str(selection["selected_model_sha256"])
    if sha256_file(checkpoint / "model.safetensors") != model_sha:
        raise RuntimeError("selected ACT checkpoint hash changed")
    manifest = read_json(HELDOUT)
    if manifest.get("status") != "PASS" or int(manifest["episode_count"]) != 8:
        raise RuntimeError("HELDOUT8 manifest is not frozen and valid")
    return method, manifest["entries"][args.heldout_episode], checkpoint, model_sha


def _frozen_proxy() -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = args.config.resolve()
    manifest_path = args.config_manifest.resolve()
    config = read_json(config_path)
    manifest = read_json(manifest_path)
    if (
        config.get("schema_version") != "doll_handoff_rigid_proxy_v1"
        or not bool(config.get("freeze", {}).get("frozen"))
        or not bool(config.get("policy_evaluation_allowed"))
        or bool(config.get("policy_specific_logic"))
    ):
        raise RuntimeError("physical policy rollout requires the frozen policy-independent proxy")
    if manifest.get("status") != "FROZEN" or manifest.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("rigid-proxy SHA256 manifest does not match the config")
    for record in manifest.get("source_scene_files", {}).values():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"frozen rigid-proxy scene dependency changed: {path}")
    selected = config.get("selected_material_candidate")
    candidates = {row["name"]: row for row in config["material_candidates"]}
    if selected not in candidates or manifest.get("selected_material_candidate") != selected:
        raise RuntimeError("frozen material selection is missing or inconsistent")
    return config, candidates[selected]


def _apply_proxy(stage: Any, config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    doll = stage.GetPrimAtPath(DOLL)
    body = stage.GetPrimAtPath(DOLL_BODY)
    if not doll.IsValid() or not body.IsValid():
        raise RuntimeError("rigid doll prims are missing")
    rigid = UsdPhysics.RigidBodyAPI.Apply(doll)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    collision = UsdPhysics.CollisionAPI.Apply(body)
    collision.CreateCollisionEnabledAttr(True)
    material = UsdShade.Material.Define(stage, "/World/RigidDollProxyPhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(candidate["static_friction"]))
    api.CreateDynamicFrictionAttr().Set(float(candidate["dynamic_friction"]))
    api.CreateRestitutionAttr().Set(float(config["object"]["restitution"]))
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set(str(config["material_combine_modes"]["friction"]))
    physx_material.CreateRestitutionCombineModeAttr().Set(str(config["material_combine_modes"]["restitution"]))
    UsdShade.MaterialBindingAPI.Apply(body).Bind(material, materialPurpose="physics")
    UsdPhysics.MassAPI.Apply(doll).CreateMassAttr(float(config["object"]["mass_kg"]))
    doll.AddAppliedSchema("PhysxRigidBodyAPI")
    for name, value in (
        ("linearDamping", config["object"]["linear_damping"]),
        ("angularDamping", config["object"]["angular_damping"]),
        (
            "maxDepenetrationVelocity",
            config["simulation"]["contact_settings"]["max_depenetration_velocity_m_s"],
        ),
    ):
        doll.CreateAttribute(f"physxRigidBody:{name}", Sdf.ValueTypeNames.Float).Set(float(value))
    body.AddAppliedSchema("PhysxCollisionAPI")
    body.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
        float(config["simulation"]["contact_settings"]["contact_offset_m"])
    )
    body.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
        float(config["simulation"]["contact_settings"]["rest_offset_m"])
    )
    return {
        **candidate,
        "mass_kg": float(config["object"]["mass_kg"]),
        "linear_damping": float(config["object"]["linear_damping"]),
        "angular_damping": float(config["object"]["angular_damping"]),
        "source_scene_saved_or_modified": False,
    }


def _numpy(value: Any) -> np.ndarray:
    return value.torch.detach().cpu().numpy() if hasattr(value, "torch") else np.asarray(value)


def _force_by_filter(sensor: ContactSensor) -> np.ndarray:
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        return np.zeros(0, dtype=np.float64)
    value = _numpy(matrix)
    if not value.size:
        return np.zeros(0, dtype=np.float64)
    value = value.reshape(-1, value.shape[-2], 3)
    return np.max(np.linalg.norm(value, axis=-1), axis=0)


def _maximum_penetration(sensor: ContactSensor, dt: float) -> tuple[float, str | None]:
    try:
        _, _, _, separations, _, _ = sensor.contact_view.get_contact_data(dt)
        value = _numpy(separations).reshape(-1)
        return (float(np.max(np.maximum(-value, 0.0))) if value.size else 0.0), None
    except Exception as error:
        return 0.0, f"{type(error).__name__}: {error}"


def _rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    xyz = quaternion[1:]
    value = np.asarray(vector, dtype=np.float64)
    return value + 2.0 * quaternion[0] * np.cross(xyz, value) + 2.0 * np.cross(xyz, np.cross(xyz, value))


def _circumcenter(points: np.ndarray) -> np.ndarray:
    first, second, third = np.asarray(points, dtype=np.float64)
    u = second - first
    v = third - first
    normal = np.cross(u, v)
    denominator = float(normal @ normal)
    if denominator <= 1e-16:
        raise RuntimeError("physical whole-hand pad centers are collinear")
    return first + (
        np.cross(normal, u) * float(v @ v) + np.cross(v, normal) * float(u @ u)
    ) / (2.0 * denominator)


def _is_open(
    measured: np.ndarray,
    names: list[str],
    definition: dict[str, Any],
    side: str,
    tolerance_rad: float,
) -> bool:
    local_names = definition["g1"]["joint_names"][side]
    target = definition["g1"]["canonical_states"][side]["OPEN"]
    error = [measured[names.index(name)] - float(value) for name, value in zip(local_names, target, strict=True)]
    return bool(np.max(np.abs(error)) <= tolerance_rad)


def run() -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite physical rollout: {output}")
    output.mkdir(parents=True)
    method_record, entry, checkpoint, model_sha = _method_record()
    config, material = _frozen_proxy()
    if Path(config["source_scene"]["g1_preview_usd"]).resolve() != SCENE.resolve():
        raise RuntimeError("physical runner scene differs from frozen proxy config")
    expected_frames = int(entry["frames"])
    source_video_path = Path(entry["source_rgb_identity"]["canonical_video_path"])
    if sha256_file(source_video_path) != entry["source_rgb_identity"]["canonical_video_sha256"]:
        raise RuntimeError("frozen source video hash changed")
    reference_path = Path(entry["b_trajectory_path"])
    if sha256_file(reference_path) != entry["b_trajectory_sha256"]:
        raise RuntimeError("authoritative interaction-reference trajectory hash changed")
    with np.load(reference_path, allow_pickle=False) as archive:
        reference_left = np.asarray(archive["target_left_interaction_frame_position_world"])
        reference_right = np.asarray(archive["target_right_interaction_frame_position_world"])
    if reference_left.shape != (expected_frames, 3) or reference_right.shape != (expected_frames, 3):
        raise RuntimeError("authoritative physical evaluation reference has the wrong shape")

    names, lower, upper, projector = frozen_interfaces()
    init_q, initial_record = common_initial_condition(names, args.settle_seconds)
    safety = SafetyAudit(names, lower, upper)
    if safety.collision(init_q[None])["invalid_hard_self_collision_incidence"]:
        raise RuntimeError("common initial state has a hard self-collision")
    if not omni.usd.get_context().open_stage(str(SCENE)):
        raise RuntimeError(f"failed to open {SCENE}")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    runtime_material = _apply_proxy(stage, config, material)
    sys.path.insert(0, str(ROOT / "isaaclab_magsafe_fixed_scene"))
    from physical_contact_monitor import enable_contact_reporting

    enable_contact_reporting(stage, ["/World/G1/Asset", DOLL, TABLE, *BIN_PARTS])
    sim = SimulationContext(
        SimulationCfg(
            dt=PHYSICS_DT,
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
    geometry = read_json(GEOMETRY)
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
            filter_prim_paths_expr=[TABLE, *BIN_PARTS],
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            force_threshold=0.0,
        )
    )
    robot_table_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/G1/Asset/.*_link",
            update_period=0.0,
            filter_prim_paths_expr=[TABLE],
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
        raise RuntimeError(f"Isaac named joint mapping failed: {missing}")
    body_names = list(robot.data.body_names)
    body_ids = {label: body_names.index(link) for label, link in distal_links.items()}
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    target[0, ids] = torch.as_tensor(init_q, device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(target, zero)
    object_center = np.asarray(config["object"]["initial_center_world_xyz_m"], dtype=np.float32)
    object_pose_wxyz = torch.as_tensor(
        np.r_[object_center, [1.0, 0.0, 0.0, 0.0]][None],
        dtype=torch.float32,
        device=doll.device,
    )
    doll.write_root_pose_to_sim_index(root_pose=object_pose_wxyz)
    doll.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=doll.device)
    )
    object_pose_writes_before_timed_loop = 1

    def update() -> None:
        robot.update(PHYSICS_DT)
        doll.update(PHYSICS_DT)
        for sensor in digit_sensors.values():
            sensor.update(PHYSICS_DT, force_recompute=True)
        external_sensor.update(PHYSICS_DT, force_recompute=True)
        robot_table_sensor.update(PHYSICS_DT, force_recompute=True)

    for _ in range(max(1, int(round(args.settle_seconds / PHYSICS_DT)))):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        update()
    immediate = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
    tolerance = float(initial_record["tolerances"]["post_settle_max_abs_from_nominal_rad"])
    if float(np.max(np.abs(immediate - init_q))) > tolerance:
        raise RuntimeError("common A/B post-settle initial-state tolerance failed")

    tool_frame = read_json(TOOL_FRAME)
    source = SourceVideo(source_video_path, expected_frames)
    bridge: ACTE1Bridge | None = None
    commands: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    measured: list[np.ndarray] = [immediate]
    velocities: list[np.ndarray] = [
        robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)
    ]
    source_hashes: list[str] = []
    safety_records: list[dict[str, Any]] = []
    projection_records: list[dict[str, Any]] = []
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
        )
    }
    maximum_penetration = 0.0
    penetration_errors: set[str] = set()
    safety_abort: dict[str, Any] | None = None
    steps_per_control = int(round((1.0 / CONTROL_FPS) / PHYSICS_DT))
    started = time.monotonic()
    try:
        bridge = ACTE1Bridge(checkpoint, model_sha, args.method, output)
        for frame in range(expected_frames):
            rgb = source.read()
            source_hashes.append(sha256_array(rgb))
            inference = bridge.infer(rgb, measured[-1])
            raw = inference["raw_ensembled_action"]
            projection = projector.project(raw.astype(np.float32)[None], inference_index=frame)
            command = projection.deployment_safe_action[0].astype(np.float64)
            projection_records.extend(projection.records)
            audit = safety.command(commands, measured[-1], velocities[-1], command)
            if audit["status"] != "PASS":
                safety_abort = {"frame": frame, "when": "before_command", "checks": audit["checks"]}
                safety_records.append({"frame": frame, "command_audit": audit})
                break
            target[0, ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
            for _ in range(steps_per_control):
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
                sim.step(render=False)
                update()
            q = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
            qdot = robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)
            commands.append(command.copy())
            raw_actions.append(raw.copy())
            raw_chunks.append(inference["raw_chunk"].copy())
            measured.append(q)
            velocities.append(qdot)
            measured_audit = safety.measured(measured, velocities)
            robot_table_force = _force_by_filter(robot_table_sensor)
            robot_table_max = float(np.max(robot_table_force)) if robot_table_force.size else 0.0
            safety_records.append(
                {
                    "frame": frame,
                    "command_audit": audit,
                    "measured_audit": measured_audit,
                    "robot_table_contact_force_n": robot_table_max,
                }
            )
            pose = doll.data.root_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
            object_velocity = doll.data.root_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
            side_forces: dict[str, float] = {}
            for side in ("left", "right"):
                side_forces[side] = float(
                    sum(
                        float(np.max(force)) if force.size else 0.0
                        for force in (
                            _force_by_filter(digit_sensors[f"{side}_{role}"])
                            for role in ("A", "B", "C")
                        )
                    )
                )
            external = _force_by_filter(external_sensor)
            table_force = float(external[0]) if len(external) else 0.0
            bin_force = float(np.max(external[1:])) if len(external) > 1 else 0.0
            positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
            quaternions = robot.data.body_quat_w.torch[0].detach().cpu().numpy()
            hand_positions: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                points = []
                for role in ("A", "B", "C"):
                    body_id = body_ids[f"{side}_{role}"]
                    local = np.asarray(geometry[side][role]["local_position_xyz_m"])
                    points.append(positions[body_id] + _rotate_wxyz(quaternions[body_id], local))
                hand_positions[side] = _circumcenter(np.asarray(points))
            for sensor in [*digit_sensors.values(), external_sensor]:
                penetration, error = _maximum_penetration(sensor, PHYSICS_DT)
                maximum_penetration = max(maximum_penetration, penetration)
                if error:
                    penetration_errors.add(error)
            log["object_com_m"].append(pose[:3])
            log["object_orientation_xyzw"].append(pose[[4, 5, 6, 3]])
            log["object_linear_velocity_m_s"].append(object_velocity[:3])
            log["object_angular_velocity_rad_s"].append(object_velocity[3:])
            log["left_hand_object_contact"].append(side_forces["left"])
            log["right_hand_object_contact"].append(side_forces["right"])
            log["table_contact"].append(table_force)
            log["bin_contact"].append(bin_force)
            log["arm_joint_q_rad"].append(q[:14])
            log["hand_joint_q_rad"].append(q[14:])
            log["left_hand_position_m"].append(hand_positions["left"])
            log["right_hand_position_m"].append(hand_positions["right"])
            log["right_hand_open"].append(
                _is_open(
                    q,
                    names,
                    tool_frame,
                    "right",
                    float(config["success_thresholds"]["hand_open_joint_max_abs_error_rad"]),
                )
            )
            robot_table_threshold = float(
                config["success_thresholds"]["maximum_force_for_table_unsupported_n"]
            )
            if measured_audit["status"] != "PASS" or robot_table_max > robot_table_threshold:
                safety_abort = {
                    "frame": frame,
                    "when": "after_command",
                    "measured_checks": measured_audit["checks"],
                    "robot_table_contact": robot_table_max > robot_table_threshold,
                }
                break
            if frame % 50 == 0:
                print(f"physical ACT-{args.method.upper()} frame {frame}/{expected_frames}", flush=True)
    finally:
        if bridge is not None:
            bridge.close()
        source.close()

    arrays = {key: np.asarray(value) for key, value in log.items()}
    executed = len(commands)
    event_log_path = output / "physical_event_log.npz"
    if executed:
        steps = np.linalg.norm(np.diff(arrays["object_com_m"], axis=0), axis=1)
        artifacts = {
            "penetration_artifact_detected": bool(
                penetration_errors
                or maximum_penetration
                > float(config["success_thresholds"]["maximum_allowed_contact_penetration_m"])
            ),
            "maximum_contact_penetration_m": maximum_penetration,
            "penetration_api_errors": sorted(penetration_errors),
            "teleportation_detected": bool(
                len(steps)
                and float(np.max(steps))
                > float(config["success_thresholds"]["maximum_allowed_object_com_step_m"])
            ),
            "maximum_object_com_step_m": float(np.max(steps)) if len(steps) else 0.0,
            "object_constraint_attachment_detected": False,
            "magnetic_or_sticky_behavior_authored": False,
            "object_pose_writes_before_timed_loop": object_pose_writes_before_timed_loop,
            "object_pose_writes_during_timed_loop": 0,
        }
        detected_physical = detect_physical_success(arrays, config)
        physics_trace_valid = not (
            artifacts["penetration_artifact_detected"]
            or artifacts["teleportation_detected"]
            or artifacts["object_constraint_attachment_detected"]
            or artifacts["magnetic_or_sticky_behavior_authored"]
        )
        physical = deepcopy(detected_physical)
        physical["physics_trace_valid"] = physics_trace_valid
        physical["detected_outcomes_before_artifact_gate"] = detected_physical["outcomes"]
        if not physics_trace_valid:
            physical["outcomes"] = {key: 0 for key in detected_physical["outcomes"]}
            for key in physical["outcomes"]:
                physical[key] = 0
            physical["task_sequence"]["task_sequence_success"] = 0
            physical["task_sequence"]["PHYSICAL_TASK_SUCCESS"] = 0
            physical["task_sequence"]["completed_ordered_phases"] = 0
            physical["task_sequence"]["phase_completion_score"] = 0.0
            physical["task_sequence"]["last_successfully_completed_phase"] = None
        arrays.update(
            {
                "frame_index": np.arange(executed, dtype=np.int64),
                "timestamp_s": np.arange(executed, dtype=np.float64) / CONTROL_FPS,
                "joint_names": np.asarray(names),
                "candidate_q_rad": np.asarray(measured[1:], dtype=np.float32),
                "candidate_left_whole_hand_position_m": arrays["left_hand_position_m"],
                "candidate_right_whole_hand_position_m": arrays["right_hand_position_m"],
                "reference_left_whole_hand_position_m": reference_left[:executed],
                "reference_right_whole_hand_position_m": reference_right[:executed],
                "commanded_action_rad": np.asarray(commands, dtype=np.float32),
                "raw_act_action_rad": np.asarray(raw_actions, dtype=np.float32),
                "raw_act_query_chunk_rad": np.asarray(raw_chunks, dtype=np.float32),
                "source_rgb_sha256": np.asarray(source_hashes[:executed]),
            }
        )
        atomic_npz(event_log_path, **arrays)
        feasibility = {
            "hard_ik_failure_count": 0,
            "hard_collision_count": int(
                any(
                    row.get("command_audit", {}).get("checks", {}).get(
                        "executed_prefix_self_collision"
                    )
                    is False
                    or row.get("measured_audit", {}).get("checks", {}).get(
                        "measured_self_collision"
                    )
                    is False
                    or float(row.get("robot_table_contact_force_n", 0.0))
                    > float(config["success_thresholds"]["maximum_force_for_table_unsupported_n"])
                    for row in safety_records
                )
            ),
            "joint_limit_failure_count": int(
                any(
                    row.get("command_audit", {}).get("checks", {}).get("command_hard_limits")
                    is False
                    or row.get("measured_audit", {}).get("checks", {}).get(
                        "measured_arm_hard_limits"
                    )
                    is False
                    or row.get("measured_audit", {}).get("checks", {}).get(
                        "measured_dex3_hard_limits_or_precharacterized_microscopic_excursion"
                    )
                    is False
                    for row in safety_records
                )
            ),
            "branch_discontinuity_count": int(
                any(
                    row.get("command_audit", {}).get("checks", {}).get("branch") is False
                    for row in safety_records
                )
            ),
            "minimum_clearance_m": None,
            "provenance": "unchanged common rollout SafetyAudit plus measured robot/table contact",
        }
        bundle = {
            "schema_version": "paper_evaluation_bundle_v1",
            "evaluation_mode": "physical",
            "success_kind": PHYSICAL_SUCCESS,
            "fps": CONTROL_FPS,
            "provenance": {
                "whole_hand_frame_authoritative": True,
                "whole_hand_frame_definition": str(TOOL_FRAME),
                "rigid_proxy_config_sha256": sha256_file(args.config.resolve()),
            },
            "episodes": [
                {
                    "source_episode_id": entry["stable_episode_id"],
                    "arrays_path": str(event_log_path),
                    "annotations": {
                        "candidate_phase_events_frame": (
                            physical["canonical_phase_events_frame"]
                            if physics_trace_valid
                            else {}
                        ),
                        "phase_events_authoritative": True,
                        "right_acquire_frame": physical["physical_event_frames"]["RIGHT_CONTACT"],
                        "left_release_frame": physical["physical_event_frames"]["LEFT_RELEASE"],
                        "handoff_events_authoritative": True,
                        "physical_success": physical["outcomes"],
                    },
                    "feasibility": feasibility,
                }
            ],
        }
        atomic_json(output / "evaluation_bundle.json", bundle)
    else:
        artifacts = {"status": "NA", "reason": "no physical frames executed"}
        physical = {"status": "NA", "reason": "no physical frames executed"}
        physics_trace_valid = False

    commands_array = np.asarray(commands, dtype=np.float32).reshape(-1, 28)
    atomic_json(output / "safety_records.json", safety_records)
    atomic_json(output / "projection_records.json", projection_records)
    status = (
        "PASS"
        if safety_abort is None
        and executed == expected_frames
        and physics_trace_valid
        else "SAFETY_ABORT_OR_INVALID_PHYSICS"
    )
    report = {
        "schema_version": "paper_physical_rigid_doll_rollout_v1",
        "status": status,
        "method": f"ACT-{args.method.upper()}40",
        "stable_episode_id": entry["stable_episode_id"],
        "heldout_output_episode": args.heldout_episode,
        "requested_frames": expected_frames,
        "executed_frames": executed,
        "source_video": str(source_video_path),
        "source_video_sha256": sha256_file(source_video_path),
        "source_clock": "original frame t at t/30; no policy-progress alignment",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": model_sha,
        "checkpoint_step": method_record["checkpoint_selection"]["selected_step"],
        "official_act_execution": {
            "execution_config": str(EXECUTION_CONFIG),
            "execution_config_sha256": EXECUTION_CONFIG_SHA256,
            "temporal_ensemble_coefficient": 0.01,
            "custom_smoothing": False,
        },
        "common_initial_state_sha256": INITIAL_CONTRACT_SHA256,
        "common_projection": str(COMMON_PROJECTION_FREEZE),
        "common_projection_sha256": COMMON_PROJECTION_SHA256,
        "rigid_proxy_config": str(args.config.resolve()),
        "rigid_proxy_config_sha256": sha256_file(args.config.resolve()),
        "runtime_material": runtime_material,
        "same_physics_and_threshold_contract_for_a_b": True,
        "object_dynamic": True,
        "object_constraint_attachment": False,
        "magnetic_grasp": False,
        "object_pose_writes_during_timed_loop": 0,
        "physics_trace_valid": physics_trace_valid,
        "artifact_checks": artifacts,
        "physical_success": physical,
        "physical_success_counted_only_when_trace_valid": True,
        "safety_abort": safety_abort,
        "command_dynamics": rollout_dynamics(commands_array),
        "wall_seconds": time.monotonic() - started,
        "event_log": str(event_log_path) if event_log_path.is_file() else None,
        "evaluation_bundle": str(output / "evaluation_bundle.json") if executed else None,
        "policy_specific_physics_logic": False,
        "real_robot": False,
    }
    atomic_json(output / "rollout_report.json", report)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    code = 1
    try:
        code = run()
    except Exception:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(code)
