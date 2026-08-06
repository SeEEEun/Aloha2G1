#!/usr/bin/env python3
"""Render and validate v12 arm-only trajectories in the active Isaac Lab scene.

The script performs kinematic joint-state replay only.  It never advances
physics, fits Dex3, publishes commands, or edits an authoritative scene file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path("/home/jbnu/aloha_g1_dataset")
SCENE_DIR = ROOT / "isaaclab_magsafe_fixed_scene"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
FIXED_SCENE = SCENE_DIR / "generated/magsafe_fixed_scene.usda"
ACTIVE_STAGE = SCENE_DIR / "generated/magsafe_g1_model_preview.usda"
LAYOUT = SCENE_DIR / "scene_layout.json"
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
METHOD = "ALOHA_PRIMARY_TARGET_SIDE_PHASE_ANCHORED_RETARGETING"
TRAJECTORIES = {
    "exact": OUT / "position_only_exact_arm_trajectory.npz",
    "nullspace": OUT / "position_only_nullspace_arm_trajectory.npz",
    "task_axis": OUT / "task_axis_arm_trajectory.npz",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory", choices=tuple(TRAJECTORIES), default="nullspace")
parser.add_argument("--mode", choices=("fixed", "object-follow"), default="fixed")
parser.add_argument("--cameras", nargs="+", choices=("overview", "side", "top"), default=["overview"])
parser.add_argument("--max-frames", type=int)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--speed", type=float, default=0.25)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

NPZ = TRAJECTORIES[args.trajectory]
if not NPZ.is_file():
    raise FileNotFoundError(NPZ)

with np.load(NPZ, allow_pickle=False) as payload:
    q = payload["g1_arm_joint_trajectory"].astype(np.float32)
    joint_names = payload["arm_joint_names"].astype(str).tolist()
    base_left = payload["base_aloha_derived_left_target"].astype(float)
    base_right = payload["base_aloha_derived_right_target"].astype(float)
    target_left = payload["corrected_left_position_scene"].astype(float)
    target_right = payload["corrected_right_position_scene"].astype(float)
    achieved_left_reference = payload["achieved_left_position_scene"].astype(float)
    achieved_right_reference = payload["achieved_right_position_scene"].astype(float)
    method = str(payload["method"])
if q.shape != (990, 14) or method != METHOD:
    raise RuntimeError("v12 trajectory invariant failed")

with np.load(OUT / "target_phone_pose_trajectory.npz", allow_pickle=False) as payload:
    phone_pose = payload["pose"].astype(float)
with np.load(OUT / "target_accessory_pose_trajectory.npz", allow_pickle=False) as payload:
    accessory_pose = payload["pose"].astype(float)

timeline = json.loads((ROOT / "configs/episode49_task_timeline.approved.json").read_text())["events"]
events = sorted(
    [(int(row["frame"]) - 7, int(row["frame"]), row["event"]) for row in timeline],
    key=lambda row: (row[0], row[2]),
)
selected_candidate = json.loads((OUT / "selected_phase_residual.json").read_text())["selected"]
launcher = AppLauncher(args)
simulation_app = launcher.app


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_event(action_index: int) -> str:
    name = "pre_task"
    for aligned, _, event_name in events:
        if aligned <= action_index:
            name = event_name
    return name


def video_name(camera: str) -> str:
    if args.mode == "object-follow":
        return f"isaaclab_object_follow_{camera}.mp4"
    if args.trajectory == "exact":
        return f"isaaclab_position_only_exact_{camera}.mp4"
    if args.trajectory == "nullspace":
        return f"isaaclab_position_only_nullspace_{camera}.mp4"
    return f"isaaclab_task_axis_{camera}.mp4"


def add_metadata(raw: Path, output: Path, camera: str, frame_count: int, render_stage: Path) -> None:
    metadata = {
        "trajectory_path": str(NPZ.resolve()),
        "trajectory_sha256": sha256(NPZ),
        "source_action_path": str(SOURCE.resolve()),
        "source_action_sha256": sha256(SOURCE),
        "authoritative_scene_usd": str(ACTIVE_STAGE.resolve()),
        "authoritative_scene_usd_sha256": sha256(ACTIVE_STAGE),
        "fixed_scene_usd_sha256": sha256(FIXED_SCENE),
        "composed_render_stage": str(render_stage.resolve()),
        "composed_render_stage_sha256": sha256(render_stage),
        "root_forward_offset_m": 0.15,
        "frame_count": frame_count,
        "fps": 7.5,
        "candidate_name": selected_candidate,
        "trajectory_kind": args.trajectory,
        "mode": args.mode,
        "camera": camera,
        "method": METHOD,
        "dex3_applied": False,
        "physics": False,
        "real_robot_command_allowed": False,
    }
    temporary = output.with_name(output.stem + ".metadata.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-map", "0",
            "-c", "copy", "-metadata", f"title=Isaac Lab v12 {args.trajectory} {args.mode} {camera}",
            "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
            "-movflags", "+faststart", str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, output)
    raw.unlink()


def main() -> None:
    import cv2
    import omni.usd
    import torch
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
    from scipy.spatial.transform import Rotation

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from robot_model_preview_common import CAMERAS, compose_stage

    scene_hashes_before = {str(path.resolve()): sha256(path) for path in (LAYOUT, FIXED_SCENE, ACTIVE_STAGE)}
    render_stage = OUT / f"isaaclab_v12_{args.trajectory}_{args.mode}.usda"
    stage = compose_stage(render_stage, "G1", G1_USD, "g1", forward_offset_m=0.15)

    dome = UsdLux.DomeLight.Define(stage, "/World/V12RenderLights/Dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.93))
    key = UsdLux.DistantLight.Define(stage, "/World/V12RenderLights/Key")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(2.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, 15.0))

    # The object interaction is always kinematic.  Disable rigid-body state on
    # the composed stage only; referenced authoritative files remain untouched.
    for prim in stage.Traverse():
        if any(token in prim.GetName().lower() for token in ("phone", "accessory", "charger")):
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if api:
                api.GetKinematicEnabledAttr().Set(True)
                api.GetRigidBodyEnabledAttr().Set(False)

    marker_specs = {
        "BaseL": ((1.0, 0.1, 1.0), 0.008),
        "BaseR": ((0.1, 1.0, 1.0), 0.008),
        "TargetL": ((1.0, 0.1, 0.1), 0.012),
        "TargetR": ((0.1, 0.3, 1.0), 0.012),
        "AchievedL": ((1.0, 0.8, 0.1), 0.009),
        "AchievedR": ((0.1, 1.0, 0.5), 0.009),
    }
    for marker_name, (color, radius) in marker_specs.items():
        sphere = UsdGeom.Sphere.Define(stage, f"/World/V12Diagnostics/{marker_name}")
        sphere.CreateRadiusAttr(radius)
        sphere.CreateDisplayColorAttr([color])
        sphere.AddTranslateOp()
    for line_name, color in (("ErrorL", (1.0, 0.15, 0.05)), ("ErrorR", (0.05, 0.25, 1.0))):
        curve = UsdGeom.BasisCurves.Define(stage, f"/World/V12Diagnostics/{line_name}")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreateWidthsAttr([0.004])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreatePointsAttr([Gf.Vec3f(0.0), Gf.Vec3f(0.0)])
        curve.CreateDisplayColorAttr([color])

    frame_paths = {"Phone": phone_pose, "Accessory": accessory_pose}
    object_ops = {}
    if args.mode == "object-follow":
        for object_name in frame_paths:
            path = f"/World/MagSafeScene/{object_name}"
            xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "v12Follow")
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "v12Follow")

    # Dynamic object-frame axes are useful in both videos and contact sheets.
    for object_name in ("Phone", "Accessory", "ChargerPad"):
        for axis, color in ((0, (1.0, 0.0, 0.0)), (1, (0.0, 1.0, 0.0)), (2, (0.0, 0.3, 1.0))):
            curve = UsdGeom.BasisCurves.Define(stage, f"/World/V12Diagnostics/Frames/{object_name}_{axis}")
            curve.CreateTypeAttr("linear")
            curve.CreateCurveVertexCountsAttr([2])
            curve.CreateWidthsAttr([0.003])
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            curve.CreatePointsAttr([Gf.Vec3f(0.0), Gf.Vec3f(0.0)])
            curve.CreateDisplayColorAttr([color])

    stage.GetRootLayer().Save()
    if not omni.usd.get_context().open_stage(str(render_stage)):
        raise RuntimeError(f"could not open {render_stage}")
    live_stage = omni.usd.get_context().get_stage()

    marker_ops = {
        name: UsdGeom.Xformable(live_stage.GetPrimAtPath(f"/World/V12Diagnostics/{name}"))
        .GetOrderedXformOps()[0]
        for name in marker_specs
    }
    error_attributes = {
        name: UsdGeom.BasisCurves(live_stage.GetPrimAtPath(f"/World/V12Diagnostics/{name}"))
        .GetPointsAttr()
        for name in ("ErrorL", "ErrorR")
    }
    axis_attributes = {
        (object_name, axis): UsdGeom.BasisCurves(
            live_stage.GetPrimAtPath(f"/World/V12Diagnostics/Frames/{object_name}_{axis}")
        ).GetPointsAttr()
        for object_name in ("Phone", "Accessory", "ChargerPad")
        for axis in range(3)
    }
    if args.mode == "object-follow":
        for object_name in frame_paths:
            object_ops[object_name] = UsdGeom.Xformable(
                live_stage.GetPrimAtPath(f"/World/MagSafeScene/{object_name}")
            ).GetOrderedXformOps()

    simulation = SimulationContext(SimulationCfg(device="cuda:0"))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint",
            spawn=None,
            actuators={
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[r"(left|right)_(shoulder|wrist)_.*_joint", r"(left|right)_elbow_joint"],
                    effort_limit_sim=25.0,
                    velocity_limit_sim=12.0,
                    stiffness=100.0,
                    damping=5.0,
                )
            },
        )
    )
    cameras = {}
    if not args.gui:
        for camera_name in args.cameras:
            cameras[camera_name] = Camera(
                CameraCfg(
                    prim_path=f"/World/V12Camera_{camera_name}",
                    update_period=0,
                    height=args.height,
                    width=args.width,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20.0)),
                )
            )
    simulation.reset()
    runtime_names = list(robot.data.joint_names)
    missing = [name for name in joint_names if name not in runtime_names]
    if missing:
        raise RuntimeError(f"active Isaac joint mapping missing {missing}")
    joint_ids = [runtime_names.index(name) for name in joint_names]
    joint_position = robot.data.default_joint_pos.clone().to(robot.device, dtype=torch.float32)
    joint_velocity = torch.zeros_like(joint_position)
    dt = simulation.get_physics_dt()
    for camera_name, camera in cameras.items():
        eye, target = CAMERAS[camera_name]
        camera.set_world_poses_from_view(np.asarray([eye], np.float32), np.asarray([target], np.float32))

    if args.gui:
        eye, target = CAMERAS[args.cameras[0]]
        simulation.set_camera_view(eye, target)
        start = time.monotonic()
        while simulation_app.is_running():
            action_index = min(int((time.monotonic() - start) * 30.0 * args.speed), 989)
            joint_position[0, joint_ids] = torch.as_tensor(q[action_index], device=robot.device)
            robot.write_joint_state_to_sim(joint_position, joint_velocity)
            simulation.forward()
            simulation.render()
            if action_index == 989:
                start = time.monotonic()
        return

    total = 990 if args.max_frames is None else min(int(args.max_frames), 990)
    writers = {}
    video_paths = {}
    for camera_name in cameras:
        output = OUT / video_name(camera_name)
        raw = OUT / f".{output.stem}.raw.mp4"
        writer = cv2.VideoWriter(
            str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (args.width, args.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open writer {raw}")
        writers[camera_name] = writer
        video_paths[camera_name] = (raw, output)

    body_names = list(robot.data.body_names)
    left_wrist_id = body_names.index("left_wrist_yaw_link")
    right_wrist_id = body_names.index("right_wrist_yaw_link")
    palm_offsets = {
        "left": torch.tensor([0.0415, 0.003, 0.0], device=robot.device, dtype=torch.float32),
        "right": torch.tensor([0.0415, -0.003, 0.0], device=robot.device, dtype=torch.float32),
    }

    def quat_apply(quaternion, vector):
        # Isaac Lab 3 beta's body_quat_w tensor in this installed build is
        # verified as xyzw against the active USD palm-link transform.
        xyz = quaternion[:3]
        cross = 2.0 * torch.cross(xyz, vector, dim=0)
        return vector + quaternion[3] * cross + torch.cross(xyz, cross, dim=0)

    def set_axis(name: str, transform: np.ndarray) -> None:
        origin = transform[:3, 3]
        for axis in range(3):
            endpoint = origin + 0.055 * transform[:3, axis]
            axis_attributes[(name, axis)].Set(
                [Gf.Vec3f(*map(float, origin)), Gf.Vec3f(*map(float, endpoint))]
            )

    charger_transform = np.asarray(
        json.loads((OUT / "environment_audit.json").read_text())["active_transforms"]["charger_pad_face_asset"],
        dtype=float,
    )
    maximum_joint_mapping_error = 0.0
    isaac_left, isaac_right = [], []
    for action_index in range(total):
        joint_position[0, joint_ids] = torch.as_tensor(q[action_index], device=robot.device)
        robot.write_joint_state_to_sim(joint_position, joint_velocity)

        marker_values = {
            "BaseL": base_left[action_index], "BaseR": base_right[action_index],
            "TargetL": target_left[action_index], "TargetR": target_right[action_index],
            "AchievedL": achieved_left_reference[action_index],
            "AchievedR": achieved_right_reference[action_index],
        }
        for marker_name, value in marker_values.items():
            marker_ops[marker_name].Set(Gf.Vec3d(*map(float, value)))
        error_attributes["ErrorL"].Set(
            [Gf.Vec3f(*map(float, target_left[action_index])), Gf.Vec3f(*map(float, achieved_left_reference[action_index]))]
        )
        error_attributes["ErrorR"].Set(
            [Gf.Vec3f(*map(float, target_right[action_index])), Gf.Vec3f(*map(float, achieved_right_reference[action_index]))]
        )

        if args.mode == "object-follow":
            for object_name, trajectory in frame_paths.items():
                transform = trajectory[action_index]
                quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
                translate_op, orient_op = object_ops[object_name]
                translate_op.Set(Gf.Vec3d(*map(float, transform[:3, 3])))
                orient_op.Set(Gf.Quatd(float(quaternion[3]), Gf.Vec3d(*map(float, quaternion[:3]))))
        set_axis("Phone", phone_pose[action_index] if args.mode == "object-follow" else phone_pose[0])
        set_axis("Accessory", accessory_pose[action_index] if args.mode == "object-follow" else accessory_pose[0])
        set_axis("ChargerPad", charger_transform)

        simulation.forward()
        robot.update(dt)
        simulation.render()
        for camera in cameras.values():
            camera.update(dt)

        actual_q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
        maximum_joint_mapping_error = max(
            maximum_joint_mapping_error, float(np.max(np.abs(actual_q - q[action_index])))
        )
        left_pos = robot.data.body_pos_w[0, left_wrist_id]
        right_pos = robot.data.body_pos_w[0, right_wrist_id]
        left_quat = robot.data.body_quat_w[0, left_wrist_id]
        right_quat = robot.data.body_quat_w[0, right_wrist_id]
        left_palm_world = left_pos + quat_apply(left_quat, palm_offsets["left"])
        right_palm_world = right_pos + quat_apply(right_quat, palm_offsets["right"])
        isaac_left.append(left_palm_world.detach().cpu().numpy())
        isaac_right.append(right_palm_world.detach().cpu().numpy())
        if action_index == 0:
            print(json.dumps({
                "isaac_fk_frame0": {
                    "left_wrist_position": left_pos.detach().cpu().numpy().tolist(),
                    "left_wrist_quaternion": left_quat.detach().cpu().numpy().tolist(),
                    "left_palm_proxy": left_palm_world.detach().cpu().numpy().tolist(),
                    "left_numerical_reference": achieved_left_reference[0].tolist(),
                    "right_wrist_position": right_pos.detach().cpu().numpy().tolist(),
                    "right_wrist_quaternion": right_quat.detach().cpu().numpy().tolist(),
                    "right_palm_proxy": right_palm_world.detach().cpu().numpy().tolist(),
                    "right_numerical_reference": achieved_right_reference[0].tolist(),
                }
            }), flush=True)

        observed = action_index + 7 if action_index <= 982 else None
        terminal = action_index >= 983
        target_error_mm = 1000.0 * max(
            np.linalg.norm(achieved_left_reference[action_index] - target_left[action_index]),
            np.linalg.norm(achieved_right_reference[action_index] - target_right[action_index]),
        )
        for camera_name, camera in cameras.items():
            image = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (0, 0), (args.width, 142), (10, 10, 10), -1)
            kind = "TASK_AXIS" if args.trajectory == "task_axis" else f"POSITION_ONLY {args.trajectory.upper()}"
            lines = [
                f"ALOHA-PRIMARY TARGET-SIDE RETARGETING | {kind}",
                f"action {action_index:03d}/989 | observed {observed if observed is not None else 'N/A'} | {current_event(action_index)}",
                f"max palm target error {target_error_mm:.3f} mm | candidate {selected_candidate}",
                "DEX3 NOT YET APPLIED | KINEMATIC ONLY | NO PHYSICS",
            ]
            if args.mode == "object-follow":
                lines.append("KINEMATIC OBJECT FOLLOW | NOT PHYSICS GRASP | DIAGNOSTIC ONLY")
            else:
                lines.append("AUTHORITATIVE FIXED OBJECTS | TARGET/ACHIEVED PALM MARKERS")
            if terminal:
                lines.append("POST-OBSERVATION TERMINAL COMMAND SAMPLE")
            for line_index, line in enumerate(lines):
                color = (40, 220, 255) if line_index else (90, 255, 120)
                cv2.putText(
                    image, line, (14, 23 + 22 * line_index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, color, 1, cv2.LINE_AA,
                )
            writers[camera_name].write(image)

        if action_index % 100 == 0:
            print(f"[V12_RENDER] {args.trajectory} {args.mode} frame {action_index}/{total - 1}", flush=True)

    for writer in writers.values():
        writer.release()
    for camera_name, (raw, output) in video_paths.items():
        add_metadata(raw, output, camera_name, total, render_stage)

    isaac_left = np.asarray(isaac_left)
    isaac_right = np.asarray(isaac_right)
    left_reference_error = np.linalg.norm(isaac_left - achieved_left_reference[:total], axis=1)
    right_reference_error = np.linalg.norm(isaac_right - achieved_right_reference[:total], axis=1)
    scene_hashes_after = {str(path.resolve()): sha256(path) for path in (LAYOUT, FIXED_SCENE, ACTIVE_STAGE)}
    result = {
        "status": "ISAACLAB_KINEMATIC_REPLAY_COMPLETE",
        "trajectory": args.trajectory,
        "mode": args.mode,
        "frames": total,
        "runtime_joint_mapping": "NAME_BASED",
        "runtime_joint_count": len(runtime_names),
        "missing_arm_joints": missing,
        "max_mapped_joint_error_rad": maximum_joint_mapping_error,
        "active_body_quaternion_storage": "xyzw (runtime-verified)",
        "isaac_vs_numerical_palm_fk": {
            "left_mean_mm": float(np.mean(left_reference_error) * 1000.0),
            "left_max_mm": float(np.max(left_reference_error) * 1000.0),
            "right_mean_mm": float(np.mean(right_reference_error) * 1000.0),
            "right_max_mm": float(np.max(right_reference_error) * 1000.0),
        },
        "scene_hashes_before": scene_hashes_before,
        "scene_hashes_after": scene_hashes_after,
        "authoritative_scene_unchanged": scene_hashes_before == scene_hashes_after,
        "physics_steps": 0,
        "physics": False,
        "dex3_fitting": False,
        "dds_initialized": False,
        "hardware_commands_sent": False,
        "videos": {name: str(paths[1].resolve()) for name, paths in video_paths.items()},
    }
    output_json = OUT / f"isaaclab_{args.trajectory}_{args.mode}_headless.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exception:
        failure = {
            "status": "BLOCKED_ISAACLAB_REPLAY",
            "trajectory": args.trajectory,
            "mode": args.mode,
            "exception_type": type(exception).__name__,
            "exception": str(exception),
            "traceback": traceback.format_exc(),
        }
        (OUT / f"isaaclab_{args.trajectory}_{args.mode}_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n"
        )
        print("[V12_RENDER_FAILURE] " + json.dumps(failure), flush=True)
        raise
    finally:
        simulation_app.close()
