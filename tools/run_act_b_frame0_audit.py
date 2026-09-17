#!/usr/bin/env python3
"""Named-joint ACT-B frame-0 audit in Isaac, with no commands executed.

This script is deliberately diagnostic-only.  It captures the reset and settled
articulation state, runs official ACT E0 and E1 inference, preserves native raw
chunks, and applies the already-frozen policy-independent deployment projection
to separate arrays.  It contains no real-hardware transport and never steps an
ACT command into the simulator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import cv2
import numpy as np

from isaaclab.app import AppLauncher

from common_deployment_safety_projection import NamedJointDeploymentSafetyProjector
from deployment_camera_config import apply_configured_distortion, load_camera_config
from policy_b_isaac_control_contract import PHYSICS_DT, build_implicit_actuators


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SCENE_STAGE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
ACTION_FREEZE = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
PRIOR_INITIAL = ROOT / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout/initial_condition.json"
PROJECTOR_FREEZE = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
COMMON_CONFIG = ROOT / "configs/doll_handoff_retargeting/common_config.template.json"
FEASIBILITY_CONFIG = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
POLICY_PYTHON = Path("/home/jbnu/miniconda3/envs/lerobot-smolvla/bin/python")
POLICY_WORKER = ROOT / "tools/act_b_inference_worker.py"
DEFAULT_CAMERA = ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"
AUTHKEY = b"act-b-isaac-local-v1"
CONTROL_FPS = 30.0
REPRESENTATIVE_EPISODE = 24


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--checkpoint-sha256", required=True)
parser.add_argument("--scene-mode", choices=("legacy-hidden", "common-visual-object-free"), required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--resnap-after-settle", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
DEPLOYMENT_CAMERA = load_camera_config(args.camera_config, purpose="ACT-B exact frame-0 audit")
launcher = AppLauncher(args)
simulation_app = launcher.app


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def signed_clearance(value: float, lower: float, upper: float) -> float:
    return float(min(value - lower, upper - value))


def frozen_interface() -> tuple[list[str], np.ndarray, np.ndarray]:
    freeze = read_json(ACTION_FREEZE)
    names = list(freeze["joint_names"])
    rows = list(freeze["joint_specs"])
    if len(names) != 28 or [row["joint_name"] for row in rows] != names:
        raise RuntimeError("authoritative Dataset-B named interface changed")
    lower = np.asarray([row["minimum"] for row in rows], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in rows], dtype=np.float64)
    return names, lower, upper


def initial_values(names: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = read_json(SOURCE_MANIFEST)
    initial_rows: list[np.ndarray] = []
    representative: np.ndarray | None = None
    representative_path: Path | None = None
    for row in manifest["episodes"]:
        path = Path(row["retargeted_trajectory_path"])
        if sha256_file(path) != row["retargeted_trajectory_sha256"]:
            raise RuntimeError(f"frozen trajectory hash changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            archive_names = archive["replay_joint_names"].astype(str).tolist()
            q0 = archive["replay_named_joint_qpos"][0].astype(np.float64)
        ordered = q0[[archive_names.index(name) for name in names]]
        initial_rows.append(ordered)
        if int(row["final_dataset_index"]) == REPRESENTATIVE_EPISODE:
            representative = ordered
            representative_path = path
    if representative is None or representative_path is None:
        raise RuntimeError("representative episode 24 is missing")
    median = np.median(np.stack(initial_rows), axis=0)
    prior = read_json(PRIOR_INITIAL)
    prior_source = np.asarray(prior["q_rad_policy_source_before_runtime_safety"], dtype=np.float64)
    prior_deployment = np.asarray(prior["q_rad"], dtype=np.float64)
    if prior["joint_names"] != names or not np.allclose(prior_source, median, rtol=0.0, atol=1e-12):
        raise RuntimeError("prior common initial-pose provenance changed")
    return representative, prior_deployment, {
        "representative_dataset_episode": REPRESENTATIVE_EPISODE,
        "representative_dataset_trajectory": representative_path,
        "dataset_initial_definition": "episode 24 frame-0 observation.state = q_target[0]",
        "nominal_definition": "dataset-wide median of 50 frozen episode first configurations followed by frozen common deployment projection",
        "source_episode_count": len(initial_rows),
        "prior_initial_condition": PRIOR_INITIAL,
        "prior_initial_condition_sha256": sha256_file(PRIOR_INITIAL),
        "dataset_median_before_projection": median,
    }


class ACTBridge:
    def __init__(self, checkpoint: Path, model_sha: str, execution: str, output: Path):
        bridge_dir = output / f"inference_bridge_{execution}"
        bridge_dir.mkdir(parents=True)
        token = hashlib.sha256(f"{output.resolve()}:{execution}".encode()).hexdigest()[:10]
        self.socket = Path(f"/tmp/actb_audit_{os.getpid()}_{token}.sock")
        self.ready_path = bridge_dir / "ready.json"
        self.log_path = bridge_dir / "worker.log"
        for path in (self.socket, self.ready_path):
            if path.exists() or path.is_socket():
                path.unlink()
        self.log_stream = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(POLICY_PYTHON), str(POLICY_WORKER),
                "--checkpoint", str(checkpoint),
                "--checkpoint-sha256", model_sha,
                "--socket", str(self.socket),
                "--ready", str(self.ready_path),
                "--execution", execution,
            ],
            cwd=ROOT,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 120.0
        while not self.ready_path.is_file():
            if self.process.poll() is not None:
                self.log_stream.flush()
                raise RuntimeError(f"ACT {execution} worker exited; inspect {self.log_path}")
            if time.monotonic() >= deadline:
                self.process.terminate()
                raise TimeoutError("ACT worker was not ready in 120 seconds")
            time.sleep(0.1)
        self.ready = read_json(self.ready_path)
        self.connection = Client(str(self.socket), family="AF_UNIX", authkey=AUTHKEY)

    def infer(self, rgb: np.ndarray, state: np.ndarray) -> dict[str, Any]:
        self.connection.send({"command": "infer", "rgb": np.ascontiguousarray(rgb, dtype=np.uint8), "state": np.ascontiguousarray(state, dtype=np.float32)})
        response = self.connection.recv()
        if response.get("status") != "PASS":
            raise RuntimeError(f"ACT worker inference failed: {response}")
        return response

    def close(self) -> None:
        try:
            self.connection.send({"command": "shutdown"})
            self.connection.recv()
            self.connection.close()
        finally:
            if self.process.poll() is None:
                self.process.wait(timeout=20.0)
            self.log_stream.close()


def look_at_ros_camera_quaternion_xyzw(eye_world_xyz: np.ndarray, target_world_xyz: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye_world_xyz, dtype=np.float64)
    target = np.asarray(target_world_xyz, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack((right, down, forward))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray([(rotation[2, 1] - rotation[1, 2]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            q = np.asarray([0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale])
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            q = np.asarray([(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            q = np.asarray([(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale])
    return q / np.linalg.norm(q)


def collision_audit(names: list[str], values: np.ndarray) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "tools"))
    from doll_handoff_retargeting.models import G1Kinematics

    g1 = G1Kinematics(read_json(COMMON_CONFIG), read_json(SCENE_LAYOUT))
    left = [names.index(name) for name in g1.hand_joint_names["left"]]
    right = [names.index(name) for name in g1.hand_joint_names["right"]]
    tolerance = float(read_json(FEASIBILITY_CONFIG)["unchanged_acceptance"]["collision_penetration_tolerance_m"])
    geometry = g1.trajectory_geometry(values[:, :14], values[:, left], values[:, right], tolerance)
    counts = {key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()}
    invalid = sum(counts.get(key, 0) for key in ("ARM_TORSO", "CROSS_ARM", "WRIST_OR_PALM_TORSO", "OTHER"))
    return {
        "frame_counts": counts,
        "invalid_hard_self_collision_incidence": invalid,
        "arm_torso_collision_incidence": counts.get("ARM_TORSO", 0) + counts.get("WRIST_OR_PALM_TORSO", 0),
        "distal_hand_hand_reported_separately": counts.get("DISTAL_HAND_HAND", 0),
    }


def run() -> None:
    import carb
    import omni.usd
    import torch
    from pxr import UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frame-0 audit: {output}")
    output.mkdir(parents=True)
    checkpoint = args.checkpoint.resolve()
    model_sha = sha256_file(checkpoint / "model.safetensors")
    if model_sha != args.checkpoint_sha256:
        raise RuntimeError("ACT checkpoint hash changed")

    names, dataset_envelope_lower, dataset_envelope_upper = frozen_interface()
    dataset_initial, nominal, initial_provenance = initial_values(names)
    projector = NamedJointDeploymentSafetyProjector.from_path(PROJECTOR_FREEZE)
    if projector.names != names:
        raise RuntimeError("common projection named order differs from Dataset B")
    hard_lower = projector.hard_lower
    hard_upper = projector.hard_upper
    nominal_projection = projector.project(nominal[None])
    if not np.array_equal(nominal_projection.deployment_safe_action[0], nominal):
        raise RuntimeError("frozen nominal is not already deployment safe")

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"failed to open {SCENE_STAGE}")
    stage = omni.usd.get_context().get_stage()
    if args.scene_mode == "legacy-hidden":
        for object_path in ("/World/DollHandoffEnvironment/Doll", "/World/DollHandoffEnvironment/TrashBin"):
            prim = stage.GetPrimAtPath(object_path)
            if not prim.IsValid():
                raise RuntimeError(f"missing scene object {object_path}")
            prim.SetActive(False)
    else:
        doll = stage.GetPrimAtPath("/World/DollHandoffEnvironment/Doll")
        rigid = UsdPhysics.RigidBodyAPI.Get(stage, doll.GetPath())
        rigid.CreateKinematicEnabledAttr(True)
        rigid.CreateRigidBodyEnabledAttr(True)
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith("/World/DollHandoffEnvironment/Doll") or path.startswith("/World/DollHandoffEnvironment/TrashBin"):
                collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
                if collision:
                    collision.CreateCollisionEnabledAttr(False)

    sim = SimulationContext(SimulationCfg(dt=PHYSICS_DT, device="cuda:0", use_fabric=True))
    robot = Articulation(ArticulationCfg(prim_path="/World/G1/Asset/root_joint", spawn=None, actuators=build_implicit_actuators(ImplicitActuatorCfg)))

    camera = Camera(
        CameraCfg(
            prim_path="/World/ACTFrame0PolicyCamera",
            update_period=0.0,
            width=DEPLOYMENT_CAMERA.width,
            height=DEPLOYMENT_CAMERA.height,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                DEPLOYMENT_CAMERA.intrinsic_matrix.reshape(-1).tolist(),
                width=DEPLOYMENT_CAMERA.width,
                height=DEPLOYMENT_CAMERA.height,
                clipping_range=DEPLOYMENT_CAMERA.clipping_range_m,
                lock_camera=True,
            ),
        )
    )
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    ids = [isaac_names.index(name) for name in names]
    if len(set(ids)) != 28:
        raise RuntimeError("Isaac named mapping is not one-to-one")
    camera.set_world_poses(
        DEPLOYMENT_CAMERA.position_world_xyz_m.astype(np.float32)[None],
        DEPLOYMENT_CAMERA.orientation_world_xyzw_ros_camera.astype(np.float32)[None],
        convention="ros",
    )
    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    target[0, ids] = torch.as_tensor(nominal, device=robot.device, dtype=torch.float32)

    def measured_state() -> np.ndarray:
        return robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)

    def measured_velocity() -> np.ndarray:
        return robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)

    robot.write_joint_state_to_sim(target, zero)
    sim.forward()
    robot.update(0.0)
    reset_state = measured_state()
    reset_velocity = measured_velocity()
    for _ in range(max(1, int(round(args.settle_seconds / PHYSICS_DT)))):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT)
    settled_state = measured_state()
    settled_velocity = measured_velocity()

    if args.resnap_after_settle:
        robot.write_joint_state_to_sim(target, zero)
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.forward()
        robot.update(0.0)
    inference_state = measured_state()
    inference_velocity = measured_velocity()

    sim.forward()
    sim.render()
    sim.render_context.reset_transform_cadence()
    camera.update(PHYSICS_DT, force_recompute=True)
    rgb = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = apply_configured_distortion(rgb.copy(), DEPLOYMENT_CAMERA)
    cv2.imwrite(str(output / "policy_input_frame0.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    responses: dict[str, dict[str, Any]] = {}
    ready: dict[str, Any] = {}
    for execution in ("e0", "e1"):
        bridge = ACTBridge(checkpoint, model_sha, execution, output)
        try:
            ready[execution] = bridge.ready
            responses[execution] = bridge.infer(rgb, inference_state)
        finally:
            bridge.close()

    e0_raw = np.asarray(responses["e0"]["action"], dtype=np.float32)
    e1_action = np.asarray(responses["e1"]["action"], dtype=np.float32)
    e0_norm_chunk = np.asarray(responses["e0"]["normalized_chunk"], dtype=np.float32)
    e1_norm_chunk = np.asarray(responses["e1"]["normalized_chunk"], dtype=np.float32)
    e0_raw_chunk = np.asarray(responses["e0"]["physical_chunk"], dtype=np.float32)
    e1_raw_chunk = np.asarray(responses["e1"]["physical_chunk"], dtype=np.float32)
    if e0_raw_chunk.shape != (50, 28) or e1_raw_chunk.shape != (50, 28):
        raise RuntimeError("ACT native chunk shape is not [50,28]")
    e0_projection = projector.project(e0_raw_chunk, inference_index=0)
    e1_projection = projector.project(e1_raw_chunk, inference_index=0)
    commanded = e0_projection.deployment_safe_action[0]

    thresholds = read_json(FEASIBILITY_CONFIG)["unchanged_acceptance"]
    branch_threshold = float(thresholds["branch_absolute_step_norm_rad"])
    branch_delta = e0_raw.astype(np.float64)[:14] - inference_state[:14]
    branch_norm = float(np.linalg.norm(branch_delta))
    projected_branch_norm = float(np.linalg.norm(commanded[:14] - inference_state[:14]))
    if branch_norm != projected_branch_norm:
        raise RuntimeError("Dex3-only projector unexpectedly altered arm branch metric")
    sequence = np.vstack((inference_state, e0_projection.deployment_safe_action.astype(np.float64)))
    qdot = np.diff(sequence, axis=0) * CONTROL_FPS
    qddot = np.vstack(((qdot[0] - inference_velocity) * CONTROL_FPS, np.diff(qdot, axis=0) * CONTROL_FPS))
    collisions = collision_audit(names, e0_projection.deployment_safe_action.astype(np.float64))
    hard_after = (e0_projection.deployment_safe_action < hard_lower[None]) | (e0_projection.deployment_safe_action > hard_upper[None])
    dry_checks = {
        "native_chunk_shape_50x28": e0_raw_chunk.shape == (50, 28),
        "native_chunk_finite": bool(np.isfinite(e0_raw_chunk).all()),
        "raw_output_preserved": bool(np.array_equal(e0_projection.policy_raw_action, e0_raw_chunk)),
        "command_hard_limit_violations_after_common_adapter_zero": int(np.count_nonzero(hard_after)) == 0,
        "branch_gate": branch_norm <= branch_threshold,
        "self_collision_zero": collisions["invalid_hard_self_collision_incidence"] == 0,
        "arm_torso_collision_zero": collisions["arm_torso_collision_incidence"] == 0,
        "velocity_preview_finite": bool(np.isfinite(qdot).all()),
        "acceleration_preview_finite": bool(np.isfinite(qddot).all()),
    }

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        row = {
            "policy_index": index,
            "isaac_index": ids[index],
            "joint_name": name,
            "group": "arm" if index < 14 else ("left_dex3" if index < 21 else "right_dex3"),
            "dataset_initial_state_rad": dataset_initial[index],
            "nominal_common_initial_rad": nominal[index],
            "isaac_reset_state_rad": reset_state[index],
            "isaac_settled_state_rad": settled_state[index],
            "isaac_inference_state_rad": inference_state[index],
            "act_e0_raw_first_action_rad": e0_raw[index],
            "act_e1_temporal_ensemble_first_action_rad": e1_action[index],
            "hard_lower_rad": hard_lower[index],
            "hard_upper_rad": hard_upper[index],
            "dataset_retargeting_envelope_lower_rad": dataset_envelope_lower[index],
            "dataset_retargeting_envelope_upper_rad": dataset_envelope_upper[index],
            "dataset_initial_signed_clearance_rad": signed_clearance(dataset_initial[index], hard_lower[index], hard_upper[index]),
            "reset_signed_clearance_rad": signed_clearance(reset_state[index], hard_lower[index], hard_upper[index]),
            "settled_signed_clearance_rad": signed_clearance(settled_state[index], hard_lower[index], hard_upper[index]),
            "act_raw_signed_clearance_rad": signed_clearance(e0_raw[index], hard_lower[index], hard_upper[index]),
            "hard_limit_projected_action_rad": e0_projection.hard_limit_projected_action[0, index],
            "deployment_projected_action_rad": commanded[index],
            "commanded_action_rad": commanded[index],
            "commanded_signed_clearance_rad": signed_clearance(commanded[index], hard_lower[index], hard_upper[index]),
            "act_raw_hard_limit_violation": bool(e0_raw[index] < hard_lower[index] or e0_raw[index] > hard_upper[index]),
            "common_projection_changed": bool(commanded[index] != e0_raw[index]),
            "branch_signed_delta_rad": branch_delta[index] if index < 14 else 0.0,
            "branch_squared_contribution_rad2": branch_delta[index] ** 2 if index < 14 else 0.0,
        }
        rows.append(row)
    with (output / "frame0_named_joint_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / "frame0_named_joint_audit.json", rows)
    atomic_npz(
        output / "frame0_arrays.npz",
        joint_names=np.asarray(names),
        dataset_initial_state=dataset_initial,
        nominal_common_initial=nominal,
        isaac_reset_state=reset_state,
        isaac_settled_state=settled_state,
        isaac_inference_state=inference_state,
        isaac_settled_velocity=settled_velocity,
        act_e0_normalized_chunk=e0_norm_chunk,
        act_e0_raw_chunk=e0_raw_chunk,
        act_e0_hard_limit_projected_chunk=e0_projection.hard_limit_projected_action,
        act_e0_deployment_projected_chunk=e0_projection.deployment_safe_action,
        act_e1_normalized_chunk=e1_norm_chunk,
        act_e1_raw_chunk=e1_raw_chunk,
        act_e1_first_action=e1_action,
    )

    violations = [row["joint_name"] for row in rows if row["act_raw_hard_limit_violation"]]
    reset_violations = [row["joint_name"] for row in rows if row["reset_signed_clearance_rad"] < 0.0]
    settled_violations = [row["joint_name"] for row in rows if row["settled_signed_clearance_rad"] < 0.0]
    report = {
        "schema_version": "act_b_exact_frame0_audit_v1",
        "status": "PASS" if all(dry_checks.values()) else "DRY_RUN_FAIL",
        "commands_executed": 0,
        "real_hardware_transport": False,
        "scene_mode": args.scene_mode,
        "scene_semantics": "legacy failed ACT runner hid Doll and TrashBin" if args.scene_mode == "legacy-hidden" else "policy-independent object-free safety: Doll and bin remain visible; Doll is kinematic and Doll/bin collisions are disabled",
        "settle_seconds": args.settle_seconds,
        "resnap_after_settle": args.resnap_after_settle,
        "checkpoint": checkpoint,
        "checkpoint_model_sha256": model_sha,
        "checkpoint_unchanged": model_sha == args.checkpoint_sha256,
        "dataset_b_unchanged_declaration": {"episodes": 50, "frames": 34478, "action_arrays_unchanged": "YES", "state_arrays_unchanged": "YES"},
        "initial_provenance": initial_provenance,
        "mapping": {
            "status": "PASS",
            "method": "name lookup only; no positional inference",
            "dataset_joint_names": names,
            "isaac_joint_ids_in_dataset_order": ids,
            "one_to_one": len(set(ids)) == 28,
            "no_left_right_swap": True,
            "no_sign_inversion": True,
            "native_logical_dimension": 28,
            "smolvla_32d_padding_path_present": False,
        },
        "normalization": {
            "status": "PASS" if responses["e0"]["normalization_round_trip_max_abs_rad"] <= 2e-6 else "FAIL",
            "authoritative_pipeline": "checkpoint ACT preprocessor NormalizerProcessorStep and postprocessor UnnormalizerProcessorStep",
            "normalized_output_shape": list(e0_norm_chunk.shape),
            "denormalized_output_shape": list(e0_raw_chunk.shape),
            "round_trip_max_abs_rad": responses["e0"]["normalization_round_trip_max_abs_rad"],
            "normalized_reconstruction_max_abs": responses["e0"]["normalization_reconstruction_max_abs"],
            "double_normalization": False,
        },
        "worker_ready": ready,
        "policy_input_rgb_sha256": sha256_array(rgb),
        "policy_input_state_sha256": sha256_array(inference_state.astype(np.float32)),
        "e0_e1_raw_chunk_max_abs_difference_rad": float(np.max(np.abs(e0_raw_chunk - e1_raw_chunk))),
        "e0_e1_first_action_max_abs_difference_rad": float(np.max(np.abs(e0_raw - e1_action))),
        "states": {
            "reset_max_abs_from_nominal_rad": float(np.max(np.abs(reset_state - nominal))),
            "settled_max_abs_from_nominal_rad": float(np.max(np.abs(settled_state - nominal))),
            "settled_arm_l2_from_nominal_rad": float(np.linalg.norm(settled_state[:14] - nominal[:14])),
            "inference_max_abs_from_nominal_rad": float(np.max(np.abs(inference_state - nominal))),
            "reset_hard_limit_violation_joints": reset_violations,
            "settled_hard_limit_violation_joints": settled_violations,
        },
        "right_dex3_raw_hard_limit_violation_joints": [name for name in violations if name.startswith("right_hand_")],
        "all_raw_hard_limit_violation_joints": violations,
        "raw_hard_limit_violation_count": len(violations),
        "common_projection": {
            "freeze_manifest": PROJECTOR_FREEZE,
            "freeze_manifest_sha256": sha256_file(PROJECTOR_FREEZE),
            "implementation_sha256": sha256_file(ROOT / "tools/common_deployment_safety_projection.py"),
            "policy_independent": True,
            "act_specific_clamp": False,
            "authoritative_hard_limit_source": "frozen common deployment adapter derived from exact URDF/controller limits",
            "dataset_retargeting_envelope_retained_separately": True,
            "maximum_abs_difference_from_rounded_dataset_retargeting_envelope_rad": float(
                np.max(
                    np.abs(
                        np.column_stack((hard_lower, hard_upper))
                        - np.column_stack((dataset_envelope_lower, dataset_envelope_upper))
                    )
                )
            ),
            "raw_action_preserved": True,
            "arms_bitwise_preserved": e0_projection.summary["arm_outputs_bitwise_preserved"],
            "summary": e0_projection.summary,
            "hard_limit_records": e0_projection.hard_limit_records,
            "deployment_margin_records": e0_projection.deployment_margin_records,
        },
        "branch_metric": {
            "definition": "L2 norm over 14 arm joints between proposed first commanded action and current measured Isaac configuration",
            "reference_configuration": "current measured Isaac state immediately before inference",
            "reference_state_rad": inference_state[:14],
            "act_first_action_rad": e0_raw[:14],
            "per_joint_signed_delta_rad": branch_delta,
            "per_joint_squared_contribution_rad2": np.square(branch_delta),
            "arm_branch_norm_rad": branch_norm,
            "gate_rad": branch_threshold,
            "gate_pass": branch_norm <= branch_threshold,
            "arm_norm_vs_nominal_rad": float(np.linalg.norm(e0_raw[:14] - nominal[:14])),
            "arm_norm_vs_reset_rad": float(np.linalg.norm(e0_raw[:14] - reset_state[:14])),
            "arm_norm_vs_settled_rad": float(np.linalg.norm(e0_raw[:14] - settled_state[:14])),
            "metric_recomputed_from_named_arrays": True,
        },
        "preview": {
            "maximum_joint_step_rad": float(np.max(np.abs(np.diff(sequence, axis=0)))),
            "maximum_velocity_rad_s": float(np.max(np.abs(qdot))),
            "maximum_acceleration_rad_s2": float(np.max(np.abs(qddot))),
            "finite_velocity": bool(np.isfinite(qdot).all()),
            "finite_acceleration": bool(np.isfinite(qddot).all()),
            "collision": collisions,
        },
        "dry_run_checks": dry_checks,
        "frame0_after_common_adapter": "PASS" if all(dry_checks.values()) else "FAIL",
    }
    atomic_json(output / "frame0_audit_report.json", report)
    print(json.dumps(report, indent=2, default=json_default))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
