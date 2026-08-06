#!/usr/bin/env python3
"""Kinematic Isaac Lab renderer proving v12 G1 articulation mesh motion.

The immutable v12 NPZ files are read directly.  The critical fix is resetting
the camera RenderContext transform cadence after every zero-physics-step
kinematic write; otherwise all frames share physics_step_count == 0 and RTX may
reuse the first transform snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12_renderfix"
SCENE_DIR = ROOT / "isaaclab_magsafe_fixed_scene"
FIXED_SCENE = SCENE_DIR / "generated/magsafe_fixed_scene.usda"
ACTIVE_SCENE = SCENE_DIR / "generated/magsafe_g1_model_preview.usda"
LAYOUT = SCENE_DIR / "scene_layout.json"
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
TRAJECTORIES = {
    "exact": V12 / "position_only_exact_arm_trajectory.npz",
    "nullspace": V12 / "position_only_nullspace_arm_trajectory.npz",
}
KEY_FRAMES = [0, 169, 216, 319, 334, 523, 695, 989]
METHOD = "ALOHA_PRIMARY_TARGET_SIDE_PHASE_ANCHORED_RETARGETING"
PROOF_CAMERA = ((1.15, -1.22, 1.38), (0.49, -0.01, 1.00))


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trajectory", choices=tuple(TRAJECTORIES), default="exact")
parser.add_argument("--mode", choices=("keyframes", "robot-only", "review"), default="keyframes")
parser.add_argument("--cameras", nargs="+", choices=("proof", "overview", "side", "top"), default=["proof"])
parser.add_argument("--gui", action="store_true")
parser.add_argument("--max-frames", type=int)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def dump(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


NPZ = TRAJECTORIES[args.trajectory]
with np.load(NPZ, allow_pickle=False) as payload:
    if "g1_arm_q" not in payload.files:
        raise RuntimeError(f"g1_arm_q missing from {NPZ}")
    q = payload["g1_arm_q"].astype(np.float32)
    q_alias = payload["g1_arm_joint_trajectory"].astype(np.float32)
    joint_names = payload["arm_joint_names"].astype(str).tolist()
    numerical_left_palm = payload["achieved_left_position_scene"].astype(float)
    numerical_right_palm = payload["achieved_right_position_scene"].astype(float)
    numerical_left_rotation = payload["achieved_left_rotation_scene"].astype(float)
    numerical_right_rotation = payload["achieved_right_rotation_scene"].astype(float)
    target_left = payload["corrected_left_position_scene"].astype(float)
    target_right = payload["corrected_right_position_scene"].astype(float)
if q.shape != (990, 14) or not np.array_equal(q, q_alias) or not np.isfinite(q).all():
    raise RuntimeError("immutable v12 q schema failed")


def output_video_name(camera: str) -> str:
    if args.mode == "robot-only":
        return f"g1_{args.trajectory}_robot_only_motion_proof.mp4"
    return f"isaaclab_position_only_{args.trajectory}_{camera}_RENDERFIX.mp4"


def add_metadata(raw: Path, output: Path, metadata: dict) -> None:
    temporary = output.with_name(output.stem + ".metadata.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-map", "0", "-c", "copy",
            "-metadata", f"title=G1 v12 renderfix {args.trajectory} {args.mode}",
            "-metadata", "comment=" + json.dumps(metadata, separators=(",", ":")),
            "-movflags", "+faststart", str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, output)
    raw.unlink()


def main() -> int:
    import carb
    import cv2
    import omni.usd
    import torch
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    from scipy.spatial.transform import Rotation

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from robot_model_preview_common import CAMERAS, compose_stage

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)
    # This installed RTX build warns that geometry streaming and reading link
    # transforms from Fabric together can leave dynamic meshes stale.  This
    # renderer is deliberately zero-step and uses the USD synchronization path.
    carb.settings.get_settings().set_bool(
        "/rtx/hydra/readTransformsFromFabricInRenderDelegate", False
    )
    scene_hashes_before = {str(path.resolve()): sha256(path) for path in (LAYOUT, FIXED_SCENE, ACTIVE_SCENE)}
    stage_path = OUT / f"renderfix_{args.trajectory}_{args.mode}.usda"
    stage = compose_stage(stage_path, "G1", G1_USD, "g1", forward_offset_m=0.15)

    dome = UsdLux.DomeLight.Define(stage, "/World/RenderfixLights/Dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.93))
    key = UsdLux.DistantLight.Define(stage, "/World/RenderfixLights/Key")
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(2.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, 15.0))
    for prim in stage.Traverse():
        if any(token in prim.GetName().lower() for token in ("phone", "accessory", "charger")):
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if api:
                api.GetKinematicEnabledAttr().Set(True)
                api.GetRigidBodyEnabledAttr().Set(False)

    marker_specs = {}
    if args.mode == "review":
        marker_specs = {
            "TargetL": ((1.0, 0.05, 0.05), 0.011),
            "TargetR": ((0.05, 0.25, 1.0), 0.011),
            "AchievedL": ((1.0, 0.8, 0.05), 0.008),
            "AchievedR": ((0.05, 1.0, 0.4), 0.008),
        }
        for name, (color, radius) in marker_specs.items():
            sphere = UsdGeom.Sphere.Define(stage, f"/World/RenderfixDiagnostics/{name}")
            sphere.CreateRadiusAttr(radius)
            sphere.CreateDisplayColorAttr([color])
            sphere.AddTranslateOp()
    stage.GetRootLayer().Save()
    if not omni.usd.get_context().open_stage(str(stage_path)):
        raise RuntimeError(stage_path)
    live_stage = omni.usd.get_context().get_stage()
    marker_ops = {
        name: UsdGeom.Xformable(live_stage.GetPrimAtPath(f"/World/RenderfixDiagnostics/{name}"))
        .GetOrderedXformOps()[0]
        for name in marker_specs
    }

    # RTX geometry streaming in this installed Isaac Sim build does not consume
    # kinematic articulation changes written only to Fabric when the physics
    # step counter stays at zero.  Disable Fabric for this proof renderer so
    # PhysX's updateToUsd path is authoritative for the rendered link meshes.
    # This remains a zero-step kinematic replay; it does not integrate physics.
    simulation = SimulationContext(SimulationCfg(device="cuda:0", use_fabric=False))
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
            cfg = CameraCfg(
                prim_path=f"/World/RenderfixCamera_{camera_name}",
                update_period=0,
                height=args.height,
                width=args.width,
                data_types=["rgb", "instance_id_segmentation_fast"],
                colorize_instance_id_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20.0)),
            )
            cameras[camera_name] = Camera(cfg)

    simulation.reset()
    runtime_names = list(robot.data.joint_names)
    missing = [name for name in joint_names if name not in runtime_names]
    duplicates = [name for name in joint_names if runtime_names.count(name) != 1]
    if missing or duplicates:
        raise RuntimeError(f"joint mapping failed missing={missing}, duplicates={duplicates}")
    joint_ids = [runtime_names.index(name) for name in joint_names]
    mapping_payload = {name: index for name, index in zip(joint_names, joint_ids)}
    mapping_hash = hashlib.sha256(json.dumps(mapping_payload, sort_keys=True).encode()).hexdigest()
    dt = simulation.get_physics_dt()
    zero_velocity = torch.zeros((1, 14), dtype=torch.float32, device=robot.device)
    body_names = list(robot.data.body_names)
    left_wrist_id = body_names.index("left_wrist_yaw_link")
    right_wrist_id = body_names.index("right_wrist_yaw_link")
    offsets = {
        "left": np.array([0.0415, 0.003, 0.0]),
        "right": np.array([0.0415, -0.003, 0.0]),
    }
    asset_xform = UsdGeom.Xformable(live_stage.GetPrimAtPath("/World/G1/Asset"))
    world_from_asset = np.asarray(
        asset_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float
    ).T
    asset_from_world = np.linalg.inv(world_from_asset)
    body_visual_ops = {}
    missing_body_prims = []
    for body_name in body_names:
        body_prim = live_stage.GetPrimAtPath(f"/World/G1/Asset/{body_name}")
        if not body_prim.IsValid():
            missing_body_prims.append(body_name)
            continue
        body_visual_ops[body_name] = UsdGeom.Xformable(body_prim).MakeMatrixXform()
    if missing_body_prims:
        raise RuntimeError(f"missing visual body prims: {missing_body_prims}")
    camera_poses = {**CAMERAS, "proof": PROOF_CAMERA}
    for name, camera in cameras.items():
        eye, target = camera_poses[name]
        camera.set_world_poses_from_view(np.asarray([eye], np.float32), np.asarray([target], np.float32))

    def read_joint_q() -> np.ndarray:
        return tensor(robot.data.joint_pos)[0, joint_ids].detach().cpu().numpy().astype(float)

    def wrist_state() -> dict:
        body_pos = tensor(robot.data.body_pos_w)[0]
        body_quat = tensor(robot.data.body_quat_w)[0]
        result = {}
        for side, body_id in (("left", left_wrist_id), ("right", right_wrist_id)):
            position = body_pos[body_id].detach().cpu().numpy().astype(float)
            quaternion_xyzw = body_quat[body_id].detach().cpu().numpy().astype(float)
            rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
            palm = position + rotation @ offsets[side]
            result[side] = {
                "wrist_position_m": position,
                "wrist_quaternion_xyzw": quaternion_xyzw,
                "wrist_rotation": rotation,
                "palm_proxy_position_m": palm,
            }
        return result

    def sync_articulation_links_to_usd() -> dict:
        """Mirror Isaac articulation body poses to the actual G1 link prims.

        The source poses are the post-write PhysX articulation body states.  We
        author only the temporary composed render stage, never the source G1 USD
        or authoritative fixed-scene assets.
        """
        body_pos = tensor(robot.data.body_pos_w)[0].detach().cpu().numpy().astype(float)
        body_quat = tensor(robot.data.body_quat_w)[0].detach().cpu().numpy().astype(float)
        for body_id, body_name in enumerate(body_names):
            world_from_body = np.eye(4, dtype=float)
            world_from_body[:3, :3] = Rotation.from_quat(body_quat[body_id]).as_matrix()
            world_from_body[:3, 3] = body_pos[body_id]
            asset_from_body = asset_from_world @ world_from_body
            body_visual_ops[body_name].Set(Gf.Matrix4d(*asset_from_body.T.reshape(-1).tolist()))

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        usd_wrist = {}
        for side in ("left", "right"):
            matrix = np.asarray(
                cache.GetLocalToWorldTransform(
                    live_stage.GetPrimAtPath(f"/World/G1/Asset/{side}_wrist_yaw_link")
                ),
                dtype=float,
            ).T
            usd_wrist[side] = {
                "position_m": matrix[:3, 3],
                "rotation": matrix[:3, :3],
            }
        return usd_wrist

    def write_frame(frame: int) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
        requested = torch.as_tensor(q[frame][None, :], dtype=torch.float32, device=robot.device)
        robot.write_joint_position_to_sim_index(position=requested, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim_index(velocity=zero_velocity, joint_ids=joint_ids)
        torch.cuda.synchronize()
        immediate = read_joint_q()
        # With use_fabric=False PhysxManager.forward() deliberately has no
        # Fabric work to perform, so invoke the installed tensor API's explicit
        # GPU articulation-kinematic propagation before the USD render sync.
        physics_view = simulation.physics_manager.get_physics_sim_view()
        if physics_view is None:
            raise RuntimeError("PhysX simulation view unavailable")
        physics_view.update_articulations_kinematic()
        simulation.forward()
        robot.update(dt)
        torch.cuda.synchronize()
        after_update = read_joint_q()
        state_after_update = wrist_state()
        usd_wrist_after_explicit_link_sync = sync_articulation_links_to_usd()
        for name, op in marker_ops.items():
            values = {
                "TargetL": target_left[frame], "TargetR": target_right[frame],
                "AchievedL": numerical_left_palm[frame], "AchievedR": numerical_right_palm[frame],
            }
            op.Set(Gf.Vec3d(*map(float, values[name])))
        simulation.render()

        # CRITICAL RENDERFIX: kinematic replay never increments physics_step_count.
        # RenderContext otherwise de-duplicates transform updates forever at step 0.
        simulation.render_context.reset_transform_cadence()
        images = {}
        segmentations = {}
        segmentation_info = {}
        for name, camera in cameras.items():
            camera.update(dt, force_recompute=True)
            images[name] = tensor(camera.data.output["rgb"])[0].detach().cpu().numpy()[..., :3].copy()
            segmentations[name] = (
                tensor(camera.data.output["instance_id_segmentation_fast"])[0]
                .detach().cpu().numpy().squeeze().astype(np.int32).copy()
            )
            segmentation_info[name] = camera.data.info.get("instance_id_segmentation_fast", {})
        torch.cuda.synchronize()
        after_render = read_joint_q()
        state_after_render = wrist_state()
        audit = {
            "frame": frame,
            "requested_q": q[frame].astype(float),
            "articulation_write_array": requested.detach().cpu().numpy()[0].astype(float),
            "readback_immediate": immediate,
            "readback_after_scene_update": after_update,
            "readback_after_render": after_render,
            "requested_immediate_max_error_rad": float(np.max(np.abs(immediate - q[frame]))),
            "requested_after_update_max_error_rad": float(np.max(np.abs(after_update - q[frame]))),
            "requested_after_render_max_error_rad": float(np.max(np.abs(after_render - q[frame]))),
            "wrist_state_after_update": state_after_update,
            "wrist_state_after_render": state_after_render,
            "usd_wrist_after_explicit_link_sync": usd_wrist_after_explicit_link_sync,
            "physics_step_count": int(simulation.get_physics_step_count()),
            "render_generation": int(simulation.render_generation),
            "render_context_transform_cadence_reset": True,
            "use_fabric": False,
            "explicit_update_articulations_kinematic": True,
            "explicit_actual_g1_link_usd_visual_sync": True,
            "rtx_read_transforms_from_fabric": False,
            "segmentation_info": segmentation_info,
        }
        return audit, images, segmentations

    if args.gui:
        import omni.ui as ui

        eye, target = camera_poses[args.cameras[0]]
        simulation.set_camera_view(eye, target)
        window = ui.Window("V12 RENDERFIX Articulation Parity", width=570, height=160)
        frame_model = ui.SimpleIntModel(0)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Frame slider (immutable NPZ action index)")
                ui.IntSlider(frame_model, min=0, max=989)
                status_label = ui.Label("")
        previous = None
        while simulation_app.is_running():
            frame = max(0, min(989, frame_model.as_int))
            if frame != previous:
                audit, _, _ = write_frame(frame)
                state = audit["wrist_state_after_render"]
                status_label.text = (
                    f"frame={frame} | readback max={audit['requested_after_render_max_error_rad']:.3e} rad\n"
                    f"L wrist={np.round(state['left']['wrist_position_m'], 4)} | "
                    f"R wrist={np.round(state['right']['wrist_position_m'], 4)}"
                )
                previous = frame
            simulation.render()
            time.sleep(0.005)
        return 0

    if args.mode == "keyframes":
        frames_dir = OUT / "keyframe_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        first_state = None
        for frame in KEY_FRAMES:
            audit, images, segmentations = write_frame(frame)
            state = audit["wrist_state_after_render"]
            if first_state is None:
                first_state = state
            for side, numerical_palm, numerical_rotation in (
                ("left", numerical_left_palm, numerical_left_rotation),
                ("right", numerical_right_palm, numerical_right_rotation),
            ):
                active_palm = np.asarray(state[side]["palm_proxy_position_m"])
                active_rotation = np.asarray(state[side]["wrist_rotation"])
                audit[f"{side}_palm_vs_numerical_error_m"] = float(
                    np.linalg.norm(active_palm - numerical_palm[frame])
                )
                audit[f"{side}_rotation_vs_numerical_error_deg"] = float(
                    np.degrees(Rotation.from_matrix(active_rotation.T @ numerical_rotation[frame]).magnitude())
                )
                audit[f"{side}_wrist_displacement_from_frame_0_m"] = float(
                    np.linalg.norm(
                        np.asarray(state[side]["wrist_position_m"])
                        - np.asarray(first_state[side]["wrist_position_m"])
                    )
                )
            audit["rgb_sha256"] = {}
            audit["segmentation_unique_ids"] = {}
            for camera_name, image_rgb in images.items():
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                audit["rgb_sha256"][camera_name] = hashlib.sha256(image_rgb.tobytes()).hexdigest()
                audit["segmentation_unique_ids"][camera_name] = np.unique(segmentations[camera_name]).tolist()
                output = frames_dir / f"{args.trajectory}_{camera_name}_{frame:03d}.png"
                cv2.rectangle(image_bgr, (0, 0), (args.width, 32), (0, 0, 0), -1)
                cv2.putText(
                    image_bgr, f"G1 ROBOT-ONLY | {args.trajectory.upper()} | frame {frame:03d}",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 120), 1, cv2.LINE_AA,
                )
                cv2.imwrite(str(output), image_bgr)
                np.save(frames_dir / f"{args.trajectory}_{camera_name}_{frame:03d}_instance.npy", segmentations[camera_name])
            rows.append(audit)
            print(f"[RENDERFIX_KEYFRAME] {args.trajectory} {frame}", flush=True)
        result = {
            "status": "ISAAC_KEYFRAME_RUNTIME_CAPTURE_COMPLETE",
            "trajectory": args.trajectory,
            "trajectory_path": str(NPZ.resolve()),
            "trajectory_sha256": sha256(NPZ),
            "q_key": "g1_arm_q",
            "joint_names": joint_names,
            "runtime_joint_names": runtime_names,
            "joint_mapping": mapping_payload,
            "joint_mapping_sha256": mapping_hash,
            "mapped_joints": len(joint_ids),
            "missing": missing,
            "duplicates": duplicates,
            "left_right_swap": False,
            "arm_order_mismatch": False,
            "physics_steps": int(simulation.get_physics_step_count()),
            "renderfix": (
                "zero-step PhysX body poses explicitly synchronized to actual G1 USD link prims; "
                "RTX Fabric transform read disabled for geometry-streaming compatibility"
            ),
            "use_fabric": False,
            "keyframes": rows,
            "scene_hashes_before": scene_hashes_before,
            "scene_hashes_after": {str(path.resolve()): sha256(path) for path in (LAYOUT, FIXED_SCENE, ACTIVE_SCENE)},
        }
        dump(OUT / f"keyframe_runtime_{args.trajectory}.json", result)
    else:
        total = 990 if args.max_frames is None else min(990, int(args.max_frames))
        writer_rows = {}
        for camera_name in cameras:
            output = OUT / output_video_name(camera_name)
            raw = OUT / f".{output.stem}.raw.mp4"
            writer = cv2.VideoWriter(
                str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (args.width, args.height)
            )
            if not writer.isOpened():
                raise RuntimeError(raw)
            writer_rows[camera_name] = (writer, raw, output)
        first_state = None
        keyframe_rows = []
        keyframe_images = {}
        keyframe_segments = {}
        maximum_readback_error = 0.0
        for frame in range(total):
            audit, images, segmentations = write_frame(frame)
            maximum_readback_error = max(maximum_readback_error, audit["requested_after_render_max_error_rad"])
            state = audit["wrist_state_after_render"]
            if first_state is None:
                first_state = state
            left_disp = float(
                np.linalg.norm(np.asarray(state["left"]["wrist_position_m"]) - np.asarray(first_state["left"]["wrist_position_m"]))
            )
            right_disp = float(
                np.linalg.norm(np.asarray(state["right"]["wrist_position_m"]) - np.asarray(first_state["right"]["wrist_position_m"]))
            )
            q_norm = float(np.linalg.norm(q[frame] - q[0]))
            for camera_name, image_rgb in images.items():
                image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                if args.mode == "robot-only":
                    cv2.rectangle(image, (0, 0), (args.width, 36), (0, 0, 0), -1)
                    cv2.putText(
                        image, f"frame {frame:03d}", (14, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (80, 255, 120), 1, cv2.LINE_AA,
                    )
                    if frame in KEY_FRAMES:
                        cv2.rectangle(image, (0, args.height - 31), (args.width, args.height), (0, 0, 0), -1)
                        cv2.putText(
                            image,
                            f"q norm {q_norm:.3f} rad | L wrist {left_disp*1000:.1f} mm | R wrist {right_disp*1000:.1f} mm",
                            (12, args.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 220, 255), 1, cv2.LINE_AA,
                        )
                else:
                    cv2.rectangle(image, (0, 0), (args.width, 76), (0, 0, 0), -1)
                    cv2.putText(
                        image, f"V12 RENDERFIX | {args.trajectory.upper()} | action {frame:03d}/989",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 120), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        image, f"NPZ q -> Isaac readback {audit['requested_after_render_max_error_rad']:.2e} rad | L/R wrist {left_disp*1000:.1f}/{right_disp*1000:.1f} mm",
                        (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 220, 255), 1, cv2.LINE_AA,
                    )
                    cv2.putText(
                        image, "ACTUAL ARTICULATION MESH | KINEMATIC ONLY | DEX3 NOT APPLIED",
                        (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 220, 255), 1, cv2.LINE_AA,
                    )
                writer_rows[camera_name][0].write(image)
                if frame in KEY_FRAMES:
                    keyframe_images[(camera_name, frame)] = image_rgb.copy()
                    keyframe_segments[(camera_name, frame)] = segmentations[camera_name].copy()
            if frame in KEY_FRAMES:
                audit["left_wrist_displacement_from_frame_0_m"] = left_disp
                audit["right_wrist_displacement_from_frame_0_m"] = right_disp
                keyframe_rows.append(audit)
            if frame % 100 == 0:
                print(f"[RENDERFIX_VIDEO] {args.trajectory} {args.mode} {frame}/{total-1}", flush=True)
        for camera_name, (writer, raw, output) in writer_rows.items():
            writer.release()
            metadata = {
                "trajectory_path": str(NPZ.resolve()),
                "trajectory_sha256": sha256(NPZ),
                "renderer_path": str(Path(__file__).resolve()),
                "renderer_sha256": sha256(Path(__file__)),
                "active_scene_path": str(ACTIVE_SCENE.resolve()),
                "active_scene_sha256": sha256(ACTIVE_SCENE),
                "fixed_scene_sha256": sha256(FIXED_SCENE),
                "q_key": "g1_arm_q",
                "joint_mapping_sha256": mapping_hash,
                "joint_names": joint_names,
                "root_forward_offset_m": 0.15,
                "frame_count": total,
                "fps": 7.5,
                "trajectory_kind": args.trajectory,
                "mode": args.mode,
                "camera": camera_name,
                "render_context_transform_cadence_reset_per_frame": True,
                "use_fabric": False,
                "rtx_read_transforms_from_fabric": False,
                "explicit_actual_g1_link_usd_visual_sync": True,
                "physics_steps": 0,
                "dex3_applied": False,
                "real_robot_command_allowed": False,
            }
            add_metadata(raw, output, metadata)
        np.savez_compressed(
            OUT / f"rendered_keyframes_{args.trajectory}_{args.mode}.npz",
            key_frames=np.asarray(KEY_FRAMES),
            **{
                f"rgb_{camera}_{frame}": image
                for (camera, frame), image in keyframe_images.items()
            },
            **{
                f"instance_{camera}_{frame}": mask
                for (camera, frame), mask in keyframe_segments.items()
            },
        )
        result = {
            "status": "RENDERFIX_KINEMATIC_VIDEO_COMPLETE",
            "trajectory": args.trajectory,
            "mode": args.mode,
            "frames": total,
            "trajectory_path": str(NPZ.resolve()),
            "trajectory_sha256": sha256(NPZ),
            "q_key": "g1_arm_q",
            "joint_mapping": mapping_payload,
            "joint_mapping_sha256": mapping_hash,
            "maximum_requested_readback_error_rad": maximum_readback_error,
            "physics_steps": int(simulation.get_physics_step_count()),
            "use_fabric": False,
            "rtx_read_transforms_from_fabric": False,
            "explicit_actual_g1_link_usd_visual_sync": True,
            "keyframes": keyframe_rows,
            "videos": {name: str(row[2].resolve()) for name, row in writer_rows.items()},
            "scene_hashes_before": scene_hashes_before,
            "scene_hashes_after": {str(path.resolve()): sha256(path) for path in (LAYOUT, FIXED_SCENE, ACTIVE_SCENE)},
        }
        dump(OUT / f"runtime_{args.trajectory}_{args.mode}.json", result)

    if int(simulation.get_physics_step_count()) != 0:
        raise RuntimeError("physics step was used in kinematic replay")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
