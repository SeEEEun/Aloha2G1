#!/usr/bin/env python3
"""Render the four matched Fair-A/Proposed-B still pairs used in JKROS Fig. 6.

This script is intentionally narrow: it accepts only the frozen representative
episode 23, advances no simulation time, writes no video, and performs no IK,
retargeting, policy inference, or physics-based task evaluation.  The same
source-derived semantic frames, scene, camera, and frozen qualitative object
reconstruction are used for both retargeting methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SCENE_STAGE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
A_MANIFEST = ROOT / "outputs/fair_a_full50_hard_fail_audit/after/fair_a_repair_manifest.json"
B_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
SOURCE_MANIFEST = B_MANIFEST
PLAN = ROOT / "outputs/policy_b_g1visual/dataset_render_full/render_plans/episode_000023.npz"
PLAN_MANIFEST = ROOT / "outputs/policy_b_g1visual/dataset_render_full/render_plans/render_plan_manifest.json"
SNAPSHOT_REPORT = ROOT / "outputs/policy_b_g1visual/dataset_render_full/episode_reports/episode_000023.json"
DEFAULT_CAMERA = ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"
DEFAULT_OUTPUT = ROOT / "outputs/paper_final_figures/Fig06_Representative_Motion/render_assets"
CAMERA_KEY = "observation.images.cam_high"
REPRESENTATIVE_EPISODE = 23
SEMANTIC_STATES = (
    ("01_left_grasp", "Left grasp", "LEFT_OWNED"),
    ("02_handoff", "Handoff / dual contact", "DUAL_CONTACT"),
    ("03_right_owned", "Right owned", "RIGHT_OWNED"),
    ("04_release", "Release", "release"),
)
SCRIPT_STARTED = time.monotonic()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=convert) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_preflight() -> dict[str, Any]:
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    processes = [line.strip() for line in compute.splitlines() if line.strip()]
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processes:
        raise SystemExit(
            "GPU_BUSY: bounded JKROS still rendering was not started because "
            f"compute processes are active: {processes}"
        )
    return {"compute_processes": processes, "gpu_query": gpu, "checked_before_isaac": True}


GPU_PREFLIGHT = gpu_preflight()

from isaaclab.app import AppLauncher  # noqa: E402
from deployment_camera_config import (  # noqa: E402
    apply_configured_distortion,
    camera_manifest_record,
    load_camera_config,
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episode", type=int, default=REPRESENTATIVE_EPISODE)
parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episode != REPRESENTATIVE_EPISODE:
    raise SystemExit("This final-paper renderer is frozen to representative episode 23.")
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def resolve_frozen_trajectories() -> dict[str, dict[str, Any]]:
    a_manifest = read_json(A_MANIFEST)
    b_manifest = read_json(B_MANIFEST)
    a_row = next(row for row in a_manifest["trajectories"] if int(row["episode_index"]) == args.episode)
    b_row = next(row for row in b_manifest["episodes"] if int(row["final_dataset_index"]) == args.episode)
    result = {
        "A": {
            "path": Path(a_row["trajectory_path"]).resolve(),
            "recorded_sha256": a_row["trajectory_sha256"],
            "label": "Trajectory-Centric A",
        },
        "B": {
            "path": Path(b_row["retargeted_trajectory_path"]).resolve(),
            "recorded_sha256": b_row["retargeted_trajectory_sha256"],
            "label": "Interaction-Centric B",
        },
    }
    for method, entry in result.items():
        actual = sha256_file(entry["path"])
        if actual != entry["recorded_sha256"]:
            raise RuntimeError(f"{method} frozen trajectory hash mismatch: {actual}")
        entry["sha256"] = actual
    return result


def main() -> int:
    import carb
    import omni.usd
    import torch
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not SCENE_STAGE.is_file():
        raise FileNotFoundError(SCENE_STAGE)
    if read_json(PLAN_MANIFEST).get("status") != "READY_FOR_ISAAC_KINEMATIC_RENDERING":
        raise RuntimeError("Frozen render-plan identity gate is not ready")

    trajectories = resolve_frozen_trajectories()
    snapshot_report = read_json(SNAPSHOT_REPORT)
    snapshot_frames = {key: int(snapshot_report["snapshot_frames"][key]) for _, _, key in SEMANTIC_STATES}
    expected_frames = {"LEFT_OWNED": 182, "DUAL_CONTACT": 299, "RIGHT_OWNED": 320, "release": 402}
    if snapshot_frames != expected_frames:
        raise RuntimeError(f"episode-23 frozen semantic frames changed: {snapshot_frames}")

    plan = np.load(PLAN, allow_pickle=False)
    source_manifest = read_json(SOURCE_MANIFEST)
    source_record = source_manifest["episodes"][args.episode]
    if str(plan["source_raw_episode"].item()) != source_record["raw_directory"]:
        raise RuntimeError("source/render-plan identity mismatch")
    if len(plan["timestamp"]) != int(source_record["source_frame_count"]):
        raise RuntimeError("source/render-plan frame-count mismatch")

    loaded: dict[str, dict[str, Any]] = {}
    for method, entry in trajectories.items():
        with np.load(entry["path"], allow_pickle=False) as trajectory:
            if len(trajectory["timestamp"]) != len(plan["timestamp"]):
                raise RuntimeError(f"{method} frame-count mismatch")
            if str(trajectory["source_episode_id"].item()) != source_record["stable_episode_id"]:
                raise RuntimeError(f"{method} source identity mismatch")
            loaded[method] = {
                "joint_names": [str(value) for value in trajectory["replay_joint_names"]],
                "qpos": np.asarray(trajectory["replay_named_joint_qpos"], dtype=np.float32),
                "method_artifact_label": str(trajectory["method"].item()),
            }
    if loaded["A"]["joint_names"] != loaded["B"]["joint_names"]:
        raise RuntimeError("A/B frozen replay joint-name order differs")

    deployment_camera = load_camera_config(
        args.camera_config.resolve(), purpose="JKROS final matched A/B still rendering"
    )
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"failed to open {SCENE_STAGE}")
    stage = omni.usd.get_context().get_stage()
    doll_prim = stage.GetPrimAtPath("/World/DollHandoffEnvironment/Doll")
    rigid = UsdPhysics.RigidBodyAPI.Get(stage, doll_prim.GetPath())
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateRigidBodyEnabledAttr(True)

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0", use_fabric=False))
    robot = Articulation(ArticulationCfg(prim_path="/World/G1/Asset/root_joint", spawn=None, actuators={}))
    doll = RigidObject(RigidObjectCfg(prim_path="/World/DollHandoffEnvironment/Doll", spawn=None))
    camera_spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        deployment_camera.intrinsic_matrix.reshape(-1).tolist(),
        width=deployment_camera.width,
        height=deployment_camera.height,
        clipping_range=deployment_camera.clipping_range_m,
        lock_camera=True,
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/JKROSFinalCamera",
            update_period=0.0,
            width=deployment_camera.width,
            height=deployment_camera.height,
            data_types=["rgb"],
            spawn=camera_spawn,
        )
    )
    sim.reset()

    isaac_joint_names = list(robot.data.joint_names)
    requested_names = loaded["A"]["joint_names"]
    missing = [name for name in requested_names if name not in isaac_joint_names]
    joint_ids = [isaac_joint_names.index(name) for name in requested_names if name in isaac_joint_names]
    if missing or len(joint_ids) != 28 or len(set(joint_ids)) != 28:
        raise RuntimeError(f"named joint mapping failed: {missing}")

    body_names = list(robot.data.body_names)
    asset_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/G1/Asset"))
    world_from_asset = np.asarray(
        asset_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64
    ).T
    asset_from_world = np.linalg.inv(world_from_asset)
    body_visual_ops = {}
    for body_name in body_names:
        body_prim = stage.GetPrimAtPath(f"/World/G1/Asset/{body_name}")
        if not body_prim.IsValid():
            raise RuntimeError(f"G1 visual link prim missing: {body_name}")
        body_visual_ops[body_name] = UsdGeom.Xformable(body_prim).MakeMatrixXform()

    doll_parent = doll_prim.GetParent()
    world_from_doll_parent = np.asarray(
        UsdGeom.Xformable(doll_parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    ).T
    doll_parent_from_world = np.linalg.inv(world_from_doll_parent)
    doll_visual_op = UsdGeom.Xformable(doll_prim).MakeMatrixXform()
    camera.set_world_poses(
        deployment_camera.position_world_xyz_m.astype(np.float32)[None],
        deployment_camera.orientation_world_xyzw_ros_camera.astype(np.float32)[None],
        convention="ros",
    )

    default_q = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero_qd = torch.zeros_like(default_q)

    def propagate_and_sync_visuals(doll_position: np.ndarray, doll_wxyz: np.ndarray) -> None:
        physics_view = sim.physics_manager.get_physics_sim_view()
        if physics_view is None:
            raise RuntimeError("PhysX simulation view unavailable for kinematic propagation")
        physics_view.update_articulations_kinematic()
        sim.forward()
        robot.update(sim.get_physics_dt())
        doll.update(sim.get_physics_dt())
        body_positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
        body_quaternions_xyzw = robot.data.body_quat_w.torch[0].detach().cpu().numpy()
        for body_id, body_name in enumerate(body_names):
            world_from_body = np.eye(4, dtype=np.float64)
            world_from_body[:3, :3] = Rotation.from_quat(body_quaternions_xyzw[body_id]).as_matrix()
            world_from_body[:3, 3] = body_positions[body_id]
            asset_from_body = asset_from_world @ world_from_body
            body_visual_ops[body_name].Set(Gf.Matrix4d(*asset_from_body.T.reshape(-1).tolist()))
        world_from_doll = np.eye(4, dtype=np.float64)
        world_from_doll[:3, :3] = Rotation.from_quat(doll_wxyz[[1, 2, 3, 0]]).as_matrix()
        world_from_doll[:3, 3] = doll_position
        doll_parent_from_doll = doll_parent_from_world @ world_from_doll
        doll_visual_op.Set(Gf.Matrix4d(*doll_parent_from_doll.T.reshape(-1).tolist()))

    def capture() -> np.ndarray:
        sim.render()
        sim.render_context.reset_transform_cadence()
        camera.update(1.0 / 30.0, force_recompute=True)
        image = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return apply_configured_distortion(np.ascontiguousarray(image), deployment_camera)

    source_dir = output / "source"
    for directory in (source_dir, output / "A", output / "B"):
        directory.mkdir(parents=True, exist_ok=True)
    source_assets: dict[str, dict[str, Any]] = {}
    for stem, label, key in SEMANTIC_STATES:
        frame = snapshot_frames[key]
        source_path = (
            Path(source_record["raw_directory_path"])
            / "images/observation.images.cam_high/episode_000000"
            / f"frame_{frame:06d}.png"
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = source_dir / f"{stem}.png"
        shutil.copy2(source_path, destination)
        source_assets[key] = {
            "label": label,
            "frame": frame,
            "source_path": str(source_path),
            "output_path": str(destination),
            "sha256": sha256_file(destination),
        }

    warmup_captures_per_pose = 16
    maximum_q_error = {"A": 0.0, "B": 0.0}
    maximum_doll_error = {"A": 0.0, "B": 0.0}
    rendered_assets: list[dict[str, Any]] = []
    rendering_started = time.monotonic()
    for method in ("A", "B"):
        for stem, label, key in SEMANTIC_STATES:
            frame = snapshot_frames[key]
            requested_q = loaded[method]["qpos"][frame]
            q = default_q.clone()
            q[0, joint_ids] = torch.as_tensor(requested_q, dtype=torch.float32, device=robot.device)
            robot.write_joint_state_to_sim(q, zero_qd)

            doll_position = np.asarray(plan["doll_position_world"][frame], dtype=np.float32)
            doll_wxyz = np.asarray(plan["doll_orientation_world_wxyz"][frame], dtype=np.float32)
            doll_pose = torch.as_tensor(
                np.r_[doll_position, doll_wxyz[[1, 2, 3, 0]]][None],
                dtype=torch.float32,
                device=doll.device,
            )
            doll.write_root_pose_to_sim_index(root_pose=doll_pose)
            doll.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=doll.device)
            )
            propagate_and_sync_visuals(doll_position, doll_wxyz)
            omni.usd.get_context().reset_renderer_accumulation()
            for _ in range(warmup_captures_per_pose):
                capture()
            image = capture()

            measured_q = robot.data.joint_pos.torch[0, joint_ids].detach().cpu().numpy()
            q_error = float(np.max(np.abs(measured_q - requested_q)))
            doll_readback = doll.data.root_pose_w.torch[0].detach().cpu().numpy()[:3]
            doll_error = float(np.max(np.abs(doll_readback - doll_position)))
            maximum_q_error[method] = max(maximum_q_error[method], q_error)
            maximum_doll_error[method] = max(maximum_doll_error[method], doll_error)
            destination = output / method / f"{stem}.png"
            if not cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"failed to write {destination}")
            rendered_assets.append(
                {
                    "method": method,
                    "method_label": trajectories[method]["label"],
                    "semantic_state": label,
                    "snapshot_key": key,
                    "source_frame": frame,
                    "timestamp_s": float(plan["timestamp"][frame]),
                    "frozen_qpos_index": frame,
                    "frozen_qpos_sha256": hashlib.sha256(requested_q.tobytes()).hexdigest(),
                    "joint_readback_max_abs_error_rad": q_error,
                    "doll_position_readback_max_abs_error_m": doll_error,
                    "output_path": str(destination),
                    "sha256": sha256_file(destination),
                }
            )

    if max(maximum_q_error.values()) > 2.0e-6:
        raise RuntimeError(f"rendered named-joint readback error too large: {maximum_q_error}")
    if max(maximum_doll_error.values()) > 2.0e-6:
        raise RuntimeError(f"rendered doll readback error too large: {maximum_doll_error}")

    render_wall_time = time.monotonic() - rendering_started
    report = {
        "schema_version": "jkros_final_matched_stills_v1",
        "status": "MATCHED_STILLS_COMPLETE",
        "episode_index": args.episode,
        "stable_episode_id": source_record["stable_episode_id"],
        "source_raw_episode": source_record["raw_directory"],
        "selection_rule": "episode closest to the median Interaction-Centric-B whole-hand error within the common feasible set; lower episode index breaks an exact median tie",
        "selection_verified_by_figure_generator": True,
        "semantic_states": [
            {"label": label, "snapshot_key": key, "source_frame": snapshot_frames[key]}
            for _, label, key in SEMANTIC_STATES
        ],
        "rendered_robot_q": "frozen replay_named_joint_qpos[source semantic frame]",
        "trajectory_sources": {
            method: {
                "label": trajectories[method]["label"],
                "path": str(trajectories[method]["path"]),
                "sha256": trajectories[method]["sha256"],
                "artifact_method_label": loaded[method]["method_artifact_label"],
            }
            for method in ("A", "B")
        },
        "scene": {"path": str(SCENE_STAGE), "sha256": sha256_file(SCENE_STAGE)},
        "camera": camera_manifest_record(deployment_camera),
        "camera_config_path": str(args.camera_config.resolve()),
        "camera_config_sha256": sha256_file(args.camera_config.resolve()),
        "same_scene_camera_root_pose_lighting_resolution_for_A_and_B": True,
        "object_visualization": {
            "method": str(plan["object_visualization_method"].item()),
            "plan_path": str(PLAN),
            "plan_sha256": sha256_file(PLAN),
            "identical_pose_for_A_and_B_at_each_source_frame": True,
            "qualitative_only": True,
            "physical_grasp_success_claimed": False,
        },
        "source_assets": source_assets,
        "rendered_assets": rendered_assets,
        "warmup_captures_per_pose": warmup_captures_per_pose,
        "maximum_named_joint_qpos_readback_error_rad": maximum_q_error,
        "maximum_doll_position_readback_error_m": maximum_doll_error,
        "simulation_time_advanced": False,
        "physics_task_evaluation_run": False,
        "videos_written": False,
        "training_or_policy_inference_run": False,
        "gpu_preflight": GPU_PREFLIGHT,
        "gpu_render_wall_time_s": render_wall_time,
        "total_script_wall_time_s_before_shutdown": time.monotonic() - SCRIPT_STARTED,
        "generation_command": (
            "/home/jbnu/miniconda3/bin/conda run -n isaaclab6 --no-capture-output "
            "/home/jbnu/IsaacLab-3-beta/isaaclab.sh -p "
            "tools/render_jkros_final_matched_stills.py --episode 23 --headless"
        ),
    }
    atomic_json(output / "matched_render_report.json", report)
    (output / "generation_command.txt").write_text(report["generation_command"] + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "episode": args.episode,
        "semantic_frames": snapshot_frames,
        "rendered_stills": len(rendered_assets),
        "gpu_render_wall_time_s": render_wall_time,
        "report": str(output / "matched_render_report.json"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
