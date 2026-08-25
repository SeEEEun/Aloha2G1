#!/usr/bin/env python3
"""Render frozen Dataset-A or Dataset-B poses from one explicit camera config.

SOURCE_LIKE_CAM_HIGH remains an opt-in diagnostic preset.  An unresolved
HELMET_D455_FINAL_PENDING config is rejected before any output is rendered.
No retargeting, controller tracking, policy inference, or label mutation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab.app import AppLauncher
from deployment_camera_config import (
    apply_configured_distortion,
    camera_manifest_record,
    load_camera_config,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE_STAGE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
SCENE_LAYOUT = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
SOURCE_MANIFEST = ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json"
DEFAULT_PLANS = ROOT / "outputs/policy_b_g1visual/dataset_render_full/render_plans"
DEFAULT_OUTPUT = ROOT / "outputs/policy_b_g1visual/dataset_render_full"
CAMERA_KEY = "observation.images.cam_high"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--camera-config", type=Path, required=True)
parser.add_argument("--dataset-variant", choices=("A", "B"), required=True)
parser.add_argument("--episodes", default="all", help="all or comma-separated final episode indices")
parser.add_argument("--plans", type=Path, default=DEFAULT_PLANS)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--overwrite-incomplete", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
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
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class H264Writer:
    def __init__(self, path: Path, width: int = 640, height: int = 480, fps: int = 30):
        self.path = path
        self.width = width
        self.height = height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_name(path.stem + ".incomplete" + path.suffix)
        self.frames = 0
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-g", "2", "-movflags", "+faststart",
                str(self.temporary),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray) -> None:
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"malformed rendered RGB: {rgb.shape} {rgb.dtype}")
        assert self.process.stdin is not None
        self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.frames += 1

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {stderr[-4000:]}")
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


def parse_episodes(value: str) -> list[int]:
    if value == "all":
        return list(range(50))
    result = sorted({int(item) for item in value.split(",") if item.strip()})
    if not result or result[0] < 0 or result[-1] >= 50:
        raise ValueError("--episodes must select indices in 0..49")
    return result


def local_camera_rotation(forward: np.ndarray) -> np.ndarray:
    """ROS camera rotation: columns are right, down, optical-forward."""

    optical = np.asarray(forward, dtype=np.float64)
    optical /= np.linalg.norm(optical)
    parent_up = np.asarray([0.0, 0.0, 1.0])
    right = np.cross(optical, parent_up)
    right /= np.linalg.norm(right)
    down = np.cross(optical, right)
    down /= np.linalg.norm(down)
    return np.column_stack([right, down, optical])


def semantic_snapshot_frames(plan: Any) -> dict[str, int]:
    ownership = np.asarray(plan["ownership_state"]).astype(str)
    events = dict(
        zip(
            np.asarray(plan["event_names"]).astype(str).tolist(),
            np.asarray(plan["event_render_frames"], dtype=int).tolist(),
        )
    )

    def first(label: str) -> int:
        values = np.flatnonzero(ownership == label)
        if not len(values):
            raise RuntimeError(f"missing ownership phase {label}")
        return int(values[0])

    def midpoint(start: int, end: int) -> int:
        return int((start + end) // 2)

    left_owned = first("LEFT_OWNED")
    handoff = first("HANDOFF_APPROACH")
    dual = first("DUAL_CONTACT")
    right_owned = first("RIGHT_OWNED")
    right_transport = first("RIGHT_TRANSPORT")
    release = first("RELEASED")
    return {
        "initial": 0,
        "pre_grasp": max(0, int(events["LEFT_GRASP"]) - 8),
        "LEFT_OWNED": min(len(ownership) - 1, left_owned + 6),
        "left_transport": midpoint(left_owned + 6, handoff - 1),
        "handoff_approach": midpoint(handoff, dual - 1),
        "DUAL_CONTACT": midpoint(dual, right_owned - 1),
        "RIGHT_OWNED": min(len(ownership) - 1, right_owned + 5),
        "right_transport": midpoint(right_transport, release - 1),
        "release": min(len(ownership) - 1, release + 3),
    }


def main() -> int:
    import carb
    import omni.usd
    import torch
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    import isaaclab.sim as sim_utils

    episodes = parse_episodes(args.episodes)
    plans = args.plans.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    deployment_camera = load_camera_config(
        args.camera_config,
        purpose=f"Dataset {args.dataset_variant} G1-visual rendering",
    )
    camera_cfg = deployment_camera.raw
    if len(episodes) == 50 and deployment_camera.is_final_helmet:
        if camera_cfg.get("rendering", {}).get("final_50_episode_render_allowed") is not True:
            raise RuntimeError("final 50-episode rendering is not authorized by the frozen camera config")
    source_manifest = read_json(SOURCE_MANIFEST)
    render_plan_manifest = read_json(plans / "render_plan_manifest.json")
    if render_plan_manifest.get("status") != "READY_FOR_ISAAC_KINEMATIC_RENDERING":
        raise RuntimeError("render-plan identity gate has not passed")
    if not SCENE_STAGE.is_file():
        raise FileNotFoundError(SCENE_STAGE)

    settings = carb.settings.get_settings()
    # Zero-step articulation writes do not advance Fabric's render transform
    # cadence in this installed Isaac build.  The kinematic propagation and
    # explicit live-stage visual sync below are therefore the authoritative
    # renderer path; the source USD asset is never edited.
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"failed to open {SCENE_STAGE}")
    stage = omni.usd.get_context().get_stage()
    doll_prim = stage.GetPrimAtPath("/World/DollHandoffEnvironment/Doll")
    rigid = UsdPhysics.RigidBodyAPI.Get(stage, doll_prim.GetPath())
    rigid.CreateKinematicEnabledAttr(True)
    rigid.CreateRigidBodyEnabledAttr(True)

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0", use_fabric=False))
    robot = Articulation(
        ArticulationCfg(prim_path="/World/G1/Asset/root_joint", spawn=None, actuators={})
    )
    doll = RigidObject(
        RigidObjectCfg(prim_path="/World/DollHandoffEnvironment/Doll", spawn=None)
    )
    shared_spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        deployment_camera.intrinsic_matrix.reshape(-1).tolist(),
        width=deployment_camera.width,
        height=deployment_camera.height,
        clipping_range=deployment_camera.clipping_range_m,
        lock_camera=True,
    )
    cameras = {
        CAMERA_KEY: Camera(
            CameraCfg(
                prim_path="/World/G1VisualCamera_cam_high",
                update_period=0.0,
                width=deployment_camera.width,
                height=deployment_camera.height,
                data_types=["rgb"],
                spawn=shared_spawn,
            )
        )
    }
    sim.reset()
    isaac_joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    asset_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/G1/Asset"))
    world_from_asset = np.asarray(
        asset_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64
    ).T
    asset_from_world = np.linalg.inv(world_from_asset)
    body_visual_ops = {}
    missing_visual_prims = []
    for body_name in body_names:
        body_prim = stage.GetPrimAtPath(f"/World/G1/Asset/{body_name}")
        if not body_prim.IsValid():
            missing_visual_prims.append(body_name)
            continue
        body_visual_ops[body_name] = UsdGeom.Xformable(body_prim).MakeMatrixXform()
    if missing_visual_prims:
        raise RuntimeError(f"G1 visual link prims missing: {missing_visual_prims}")
    doll_parent = doll_prim.GetParent()
    world_from_doll_parent = np.asarray(
        UsdGeom.Xformable(doll_parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=np.float64,
    ).T
    doll_parent_from_world = np.linalg.inv(world_from_doll_parent)
    doll_visual_op = UsdGeom.Xformable(doll_prim).MakeMatrixXform()
    camera_position = deployment_camera.position_world_xyz_m.astype(np.float32)
    camera_quaternion = deployment_camera.orientation_world_xyzw_ros_camera.astype(np.float32)
    cameras[CAMERA_KEY].set_world_poses(
        camera_position[None], camera_quaternion[None], convention="ros"
    )

    default_q = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero_qd = torch.zeros_like(default_q)
    reports: list[dict[str, Any]] = []

    def capture() -> dict[str, np.ndarray]:
        sim.render()
        sim.render_context.reset_transform_cadence()
        images: dict[str, np.ndarray] = {}
        for key, camera in cameras.items():
            camera.update(1.0 / 30.0, force_recompute=True)
            image = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            images[key] = apply_configured_distortion(
                np.ascontiguousarray(image), deployment_camera
            )
        return images

    def propagate_and_sync_visuals(
        doll_position: np.ndarray, doll_quaternion_wxyz: np.ndarray
    ) -> None:
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
            np.asarray(doll_quaternion_wxyz)[[1, 2, 3, 0]]
        ).as_matrix()
        world_from_doll[:3, 3] = doll_position
        doll_parent_from_doll = doll_parent_from_world @ world_from_doll
        doll_visual_op.Set(Gf.Matrix4d(*doll_parent_from_doll.T.reshape(-1).tolist()))

    try:
        for episode in episodes:
            plan_path = plans / f"episode_{episode:06d}.npz"
            plan = np.load(plan_path, allow_pickle=False)
            states = np.asarray(plan["observation_state"], dtype=np.float32)
            timestamps = np.asarray(plan["timestamp"], dtype=np.float32)
            names = [str(value) for value in plan["policy_joint_names"]]
            missing = [name for name in names if name not in isaac_joint_names]
            ids = [isaac_joint_names.index(name) for name in names if name in isaac_joint_names]
            if missing or len(ids) != 28 or len(set(ids)) != 28:
                raise RuntimeError(f"episode {episode}: named joint mapping failed: {missing}")
            output_paths = {
                key: output / "videos" / key / "chunk-000" / f"file-{episode:03d}.mp4"
                for key in (CAMERA_KEY,)
            }
            existing = [path.exists() for path in output_paths.values()]
            if any(existing):
                if all(existing):
                    print(f"[G1VisualRender] episode {episode:02d} already complete; skipping", flush=True)
                    continue
                if not args.overwrite_incomplete:
                    raise RuntimeError(f"episode {episode}: incomplete mixed video outputs exist")
            writers = {
                key: H264Writer(path, deployment_camera.width, deployment_camera.height)
                for key, path in output_paths.items()
            }
            snapshot_frames = semantic_snapshot_frames(plan)
            snapshot_by_frame: dict[int, list[str]] = {}
            for label, frame in snapshot_frames.items():
                snapshot_by_frame.setdefault(frame, []).append(label)
            maximum_q_error = 0.0
            maximum_doll_position_error = 0.0
            # The RTX camera path uses temporal history.  A large kinematic
            # pose change between episodes can otherwise leave a one-frame
            # ghost from the previous episode.  Re-render the exact first pose
            # sixteen times without writing output; no simulation time
            # advances.  Four captures left a faint DLSS history trace in the
            # episode-49 preview, while sixteen is still negligible beside a
            # ~690-frame render and provides a conservative history flush.
            episode_start_render_warmup_captures = 16
            started = time.monotonic()
            try:
                for frame in range(len(states)):
                    q = default_q.clone()
                    q[0, ids] = torch.as_tensor(states[frame], device=robot.device)
                    robot.write_joint_state_to_sim(q, zero_qd)
                    doll_position = np.asarray(plan["doll_position_world"][frame], dtype=np.float32)
                    doll_wxyz = np.asarray(plan["doll_orientation_world_wxyz"][frame], dtype=np.float32)
                    doll_xyzw = doll_wxyz[[1, 2, 3, 0]]
                    doll_pose = torch.as_tensor(
                        np.r_[doll_position, doll_xyzw][None], dtype=torch.float32, device=doll.device
                    )
                    doll.write_root_pose_to_sim_index(root_pose=doll_pose)
                    doll.write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=doll.device)
                    )
                    propagate_and_sync_visuals(doll_position, doll_wxyz)
                    if frame == 0:
                        # Explicitly invalidate RTX/DLSS accumulation before
                        # the discarded warm-up renders.  This is a renderer
                        # history reset only; articulation/object state and the
                        # episode clock remain untouched.
                        omni.usd.get_context().reset_renderer_accumulation()
                        for _ in range(episode_start_render_warmup_captures):
                            capture()
                    images = capture()
                    measured = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy()
                    maximum_q_error = max(maximum_q_error, float(np.max(np.abs(measured - states[frame]))))
                    doll_readback = doll.data.root_pose_w.torch[0].detach().cpu().numpy()[:3]
                    maximum_doll_position_error = max(
                        maximum_doll_position_error,
                        float(np.max(np.abs(doll_readback - doll_position))),
                    )
                    for key, writer in writers.items():
                        writer.write(images[key])
                    if frame in snapshot_by_frame:
                        source = source_manifest["episodes"][episode]
                        source_path = (
                            Path(source["raw_directory_path"])
                            / "images/observation.images.cam_high/episode_000000"
                            / f"frame_{frame:06d}.png"
                        )
                        source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                        if source_bgr is None:
                            raise RuntimeError(f"cannot read matched source image {source_path}")
                        for label in snapshot_by_frame[frame]:
                            snapshot_dir = output / "snapshots" / f"episode_{episode:06d}"
                            snapshot_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(snapshot_dir / f"{label}_source_aloha.png"), source_bgr)
                            for key, image in images.items():
                                camera_name = key.split(".")[-1]
                                cv2.imwrite(
                                    str(snapshot_dir / f"{label}_{camera_name}.png"),
                                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                )
                for writer in writers.values():
                    writer.close()
            except BaseException:
                for writer in writers.values():
                    writer.abort()
                raise

            if maximum_q_error > 2.0e-6:
                raise RuntimeError(f"episode {episode}: rendered q readback error {maximum_q_error}")
            if maximum_doll_position_error > 2.0e-6:
                raise RuntimeError(
                    f"episode {episode}: rendered doll readback error {maximum_doll_position_error}"
                )
            video_entries = {
                key: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "frames": len(states),
                }
                for key, path in output_paths.items()
            }
            report = {
                "episode_index": episode,
                "source_raw_episode": str(plan["source_raw_episode"]),
                "frames": len(states),
                "fps": 30.0,
                "duration_s": float(len(states) / 30.0),
                "timestamp_start_s": float(timestamps[0]),
                "timestamp_end_s": float(timestamps[-1]),
                "rendered_robot_q": "observation.state[t]",
                "state_label": "RETARGETED_G1_STATE_SURROGATE",
                "controller_tracking_used": False,
                "maximum_named_joint_qpos_readback_error_rad": maximum_q_error,
                "maximum_doll_position_readback_error_m": maximum_doll_position_error,
                "episode_start_render_warmup_captures": episode_start_render_warmup_captures,
                "object_visualization_method": str(plan["object_visualization_method"]),
                "snapshot_frames": snapshot_frames,
                "camera_config": camera_manifest_record(deployment_camera),
                "videos": video_entries,
                "render_wall_time_s": time.monotonic() - started,
            }
            atomic_json(output / "episode_reports" / f"episode_{episode:06d}.json", report)
            reports.append(report)
            print(
                f"[G1VisualRender] episode {episode:02d}: {len(states)} frames, "
                f"qerr={maximum_q_error:.3e}, {report['render_wall_time_s']:.1f}s",
                flush=True,
            )

        camera_manifest = {
            "schema_version": "g1visual_camera_config_driven_v2",
            "dataset_variant": args.dataset_variant,
            "primary_training_camera": CAMERA_KEY,
            "camera": camera_manifest_record(deployment_camera),
            "all_render_geometry_from_camera_config": True,
            "runtime_articulation_body_names": body_names,
        }
        atomic_json(output / "camera_family.json", camera_manifest)
        completed_reports = []
        for path in sorted((output / "episode_reports").glob("episode_*.json")):
            completed_reports.append(read_json(path))
        summary = {
            "schema_version": "doll_handoff_g1visual_render_v1",
            "dataset_variant": args.dataset_variant,
            "status": "RENDER_COMPLETE" if len(completed_reports) == 50 else "PREVIEW_COMPLETE",
            "selected_episodes_this_invocation": episodes,
            "completed_episode_count": len(completed_reports),
            "completed_frame_count": sum(int(row["frames"]) for row in completed_reports),
            "render_plan_manifest": str(plans / "render_plan_manifest.json"),
            "render_plan_manifest_sha256": sha256_file(plans / "render_plan_manifest.json"),
            "camera_family": str(output / "camera_family.json"),
            "camera_family_sha256": sha256_file(output / "camera_family.json"),
            "episodes": completed_reports,
        }
        atomic_json(output / "render_manifest.json", summary)
        print(json.dumps({
            "status": summary["status"],
            "completed_episodes": len(completed_reports),
            "completed_frames": summary["completed_frame_count"],
            "manifest": str(output / "render_manifest.json"),
        }, indent=2), flush=True)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
