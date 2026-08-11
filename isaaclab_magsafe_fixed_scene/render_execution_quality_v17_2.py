#!/usr/bin/env python3
"""PAPER_WHITE zero-step Isaac review of the v17.2 whole-motion candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SCENE = ROOT / "isaaclab_magsafe_fixed_scene"
DEFAULT_OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_quality_v17_2"
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=DEFAULT_OUT / "final_arm_dex3_trajectory.npz")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
parser.add_argument("--artifact-version", default="v17_2")
parser.add_argument("--video-prefix", default="v17_2_KINEMATIC_FULL_RENDERFIX")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--interactive-review", action="store_true")
parser.add_argument("--camera", choices=("overview", "side", "top", "robot_only"), default="overview")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--pause-at-end", action="store_true")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--max-frames", type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.interactive_review:
    args.gui = True
if args.gui:
    args.headless = False
    args.enable_cameras = True
if args.loop and args.pause_at_end:
    parser.error("choose either --loop or --pause-at-end")

launcher = AppLauncher(args)
simulation_app = launcher.app


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload: dict) -> None:
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=default, allow_nan=False) + "\n")
    os.replace(temporary, path)


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
    from robot_model_preview_common import compose_stage

    sys.path[:0] = [str(ROOT / "tools"), str(SCENE)]
    from aloha_g1_v15.kinematics import ActiveG1Dex3
    import build_episode49_execution_physics_v17 as v17

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.input.resolve(), allow_pickle=False) as archive:
        arm = archive["arm_qpos"].astype(np.float32)
        left = archive["left_dex3_qpos"].astype(np.float32)
        right = archive["right_dex3_qpos"].astype(np.float32)
        joint_names = np.r_[
            archive["arm_joint_names"].astype(str),
            archive["left_dex3_joint_names"].astype(str),
            archive["right_dex3_joint_names"].astype(str),
        ].tolist()
    q = np.c_[arm, left, right]
    total = len(q) if args.max_frames is None else min(len(q), int(args.max_frames))
    if q.shape != (990, 28) or not np.isfinite(q).all():
        raise RuntimeError("v17.2 kinematic input contract failed")

    settings = carb.settings.get_settings()
    # This is a zero-physics-step review.  In the installed RTX build, geometry
    # streaming may leave link meshes stale when the render delegate reads only
    # Fabric transforms at physics step zero.  The known-good v12 renderfix uses
    # actual articulation body poses and mirrors them to the temporary review
    # stage before capture.  No target-derived link pose is authored.
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", True)
    settings.set_float_array(
        "/rtx/post/backgroundZeroAlpha/backgroundDefaultColor", [0.97, 0.97, 0.97, 1.0]
    )
    settings.set_float("/rtx/post/tonemap/filmIso", 160.0)
    settings.set_float("/rtx/post/tonemap/exposureTime", 1.0 / 60.0)
    settings.set_float("/rtx/post/tonemap/fNumber", 8.0)
    settings.set_bool("/rtx/post/histogram/enabled", False)

    root_config = json.loads((ROOT / "configs/g1_root_forward_v14.approved.json").read_text())
    stage_path = out / f"{args.artifact_version}_kinematic_review.usda"
    stage = compose_stage(
        stage_path, "G1", G1_USD, "g1",
        forward_offset_m=float(root_config["selected_total_forward_offset_m"]),
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/ExecutionReviewLights/PaperWhiteDome")
    dome.CreateIntensityAttr(1350.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/ExecutionReviewLights/PaperWhiteKey")
    key.CreateIntensityAttr(3200.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.985, 0.965))
    key.CreateAngleAttr(3.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-42.0, 24.0, 18.0))
    fill = UsdLux.SphereLight.Define(stage, "/World/ExecutionReviewLights/PaperWhiteFill")
    fill.CreateIntensityAttr(850.0)
    fill.CreateColorAttr(Gf.Vec3f(0.94, 0.97, 1.0))
    fill.CreateRadiusAttr(0.45)
    UsdGeom.Xformable(fill).AddTranslateOp().Set(Gf.Vec3d(0.25, -0.10, 1.65))
    # Derived review stage only: freeze objects visually and disable dynamics.
    for path in ("/World/MagSafeScene/Phone", "/World/MagSafeScene/Accessory"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if api:
                api.GetRigidBodyEnabledAttr().Set(False)
                api.GetKinematicEnabledAttr().Set(True)
    stage.GetRootLayer().Save()
    if not omni.usd.get_context().open_stage(str(stage_path)):
        raise RuntimeError(stage_path)
    live_stage = omni.usd.get_context().get_stage()

    sim = SimulationContext(SimulationCfg(device="cuda:0", use_fabric=False))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/G1/Asset/root_joint", spawn=None,
        actuators={
            "review": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"(left|right)_(shoulder|wrist)_.*_joint",
                    r"(left|right)_elbow_joint",
                    r"(left|right)_hand_(thumb|index|middle)_.*_joint",
                ],
                effort_limit_sim=25.0, velocity_limit_sim=12.0,
                stiffness=100.0, damping=5.0,
            )
        },
    ))
    camera_poses = {
        "overview": ((1.15, -1.22, 1.38), (0.49, 0.02, 0.98)),
        "side": ((1.62, -0.10, 1.34), (0.45, 0.00, 1.01)),
        "top": ((0.52, 0.05, 1.72), (0.47, 0.08, 0.87)),
        "robot_only": ((1.08, -1.05, 1.43), (0.45, -0.02, 1.04)),
    }
    cameras = {
        name: Camera(CameraCfg(
            prim_path=f"/World/ExecutionReviewCamera_{name}", update_period=0,
            height=args.height, width=args.width,
            data_types=["rgb", "instance_id_segmentation_fast"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.05, 20.0)),
        )) for name in camera_poses
    }
    sim.reset()
    runtime_names = list(robot.data.joint_names)
    missing = [name for name in joint_names if name not in runtime_names]
    ids = [runtime_names.index(name) for name in joint_names if name in runtime_names]
    if missing or len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError(f"joint mapping failed: {missing}")
    for name, camera in cameras.items():
        eye, target = camera_poses[name]
        camera.set_world_poses_from_view(np.asarray([eye], np.float32), np.asarray([target], np.float32))
    if args.gui:
        sim.set_camera_view(*camera_poses[args.camera])
    zero = torch.zeros((1, len(ids)), dtype=torch.float32, device=robot.device)
    dt = sim.get_physics_dt()

    body_names = list(robot.data.body_names)
    mapping = json.loads((ROOT / "configs/dex3_abc_finger_mapping.sim.json").read_text())
    parity_links = {
        "left_shoulder": "left_shoulder_pitch_link",
        "left_elbow": "left_elbow_link",
        "left_wrist": "left_wrist_yaw_link",
        "left_palm": "left_hand_palm_link",
        "left_thumb_distal": mapping["left"]["A"]["distal_link"],
        "left_index_distal": mapping["left"]["B"]["distal_link"],
        "right_shoulder": "right_shoulder_pitch_link",
        "right_elbow": "right_elbow_link",
        "right_wrist": "right_wrist_yaw_link",
        "right_palm": "right_hand_palm_link",
        "right_thumb_distal": mapping["right"]["B"]["distal_link"],
        "right_index_distal": mapping["right"]["A"]["distal_link"],
        "right_middle_C_distal": mapping["right"]["C"]["distal_link"],
    }
    missing_bodies = [name for name in parity_links.values() if name not in body_names]
    if missing_bodies:
        raise RuntimeError(f"parity body mapping failed: {missing_bodies}")
    parity_body_ids = {label: body_names.index(name) for label, name in parity_links.items()}

    numerical = ActiveG1Dex3(
        v17.MODEL, v17.DEX3_MAPPING, v17.PALM_CONFIG,
        np.asarray(json.loads((ROOT / "configs/g1_root_forward_v14.approved.json").read_text())["new_exact_root_xyz_m"], dtype=np.float64),
    )
    numerical_body_ids = {
        label: None if label in {"left_palm", "right_palm"} else numerical.body_id(name)
        for label, name in parity_links.items()
    }

    asset_xform = UsdGeom.Xformable(live_stage.GetPrimAtPath("/World/G1/Asset"))
    world_from_asset = np.asarray(
        asset_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64
    ).T
    asset_from_world = np.linalg.inv(world_from_asset)
    body_visual_ops = {}
    missing_body_prims = []
    for body_name in body_names:
        prim = live_stage.GetPrimAtPath(f"/World/G1/Asset/{body_name}")
        if not prim.IsValid():
            missing_body_prims.append(body_name)
            continue
        body_visual_ops[body_name] = UsdGeom.Xformable(prim).MakeMatrixXform()
    if missing_body_prims:
        raise RuntimeError(f"temporary review stage missing body prims: {missing_body_prims}")

    def serialize_states(states: dict) -> dict:
        return {
            label: {key: np.asarray(value).tolist() for key, value in row.items()}
            for label, row in states.items()
        }

    def articulation_states() -> dict:
        positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        quaternions = robot.data.body_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
        return {
            label: {
                "position_m": positions[body_id].copy(),
                "quaternion_xyzw": quaternions[body_id].copy(),
            }
            for label, body_id in parity_body_ids.items()
        }

    def numerical_states(actual_q: np.ndarray) -> dict:
        numerical.assign(actual_q[:14], actual_q[14:21], actual_q[21:28])
        result = {}
        for label, body_id in numerical_body_ids.items():
            if body_id is None:
                pose = numerical.palm_pose(label.split("_", 1)[0])
                result[label] = {"position_m": pose[:3, 3], "rotation": pose[:3, :3]}
            else:
                result[label] = {
                    "position_m": numerical.model_to_scene_position(numerical.data.xpos[body_id]),
                    "rotation": numerical.model_to_scene_rotation(
                        numerical.data.xmat[body_id].reshape(3, 3)
                    ),
                }
        return result

    def sync_actual_articulation_to_review_stage() -> dict:
        """Synchronize actual articulation body poses to the temporary USD stage."""
        positions = robot.data.body_pos_w.torch[0].detach().cpu().numpy().astype(np.float64)
        quaternions = robot.data.body_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
        for body_id, body_name in enumerate(body_names):
            world_from_body = np.eye(4, dtype=np.float64)
            world_from_body[:3, :3] = Rotation.from_quat(quaternions[body_id]).as_matrix()
            world_from_body[:3, 3] = positions[body_id]
            asset_from_body = asset_from_world @ world_from_body
            body_visual_ops[body_name].Set(
                Gf.Matrix4d(*asset_from_body.T.reshape(-1).tolist())
            )
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        return {
            label: {
                "position_m": np.asarray(
                    cache.GetLocalToWorldTransform(
                        live_stage.GetPrimAtPath(f"/World/G1/Asset/{name}")
                    ), dtype=np.float64,
                ).T[:3, 3]
            }
            for label, name in parity_links.items()
        }

    def write_frame(index: int) -> tuple[float, np.ndarray, dict, dict, dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
        requested = torch.as_tensor(q[index][None], dtype=torch.float32, device=robot.device)
        robot.write_joint_position_to_sim_index(position=requested, joint_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=zero, joint_ids=ids)
        view = sim.physics_manager.get_physics_sim_view()
        view.update_articulations_kinematic()
        sim.forward()
        robot.update(dt)
        torch.cuda.synchronize()
        actual = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().copy()
        isaac_states = articulation_states()
        numerical_link_states = numerical_states(actual)
        usd_states = sync_actual_articulation_to_review_stage()
        sim.render()
        sim.render_context.reset_transform_cadence()
        images = {}
        masks = {}
        for name, camera in cameras.items():
            camera.update(dt, force_recompute=True)
            images[name] = camera.data.output["rgb"].torch[0, ..., :3].detach().cpu().numpy().copy()
            segmentation = (
                camera.data.output["instance_id_segmentation_fast"].torch[0]
                .detach().cpu().numpy().squeeze().astype(np.int32).copy()
            )
            if segmentation.ndim == 3:
                segmentation = segmentation[..., 0]
            labels = camera.data.info.get("instance_id_segmentation_fast", {}).get("idToLabels", {})
            present_ids = set(np.unique(segmentation).astype(int).tolist())
            robot_ids = []
            for identifier, label in labels.items():
                if "/World/G1/" not in str(label):
                    continue
                candidates = identifier if isinstance(identifier, tuple) else (identifier,)
                robot_ids.extend(
                    value for value in (int(candidate) for candidate in candidates)
                    if value in present_ids and value != 0
                )
            robot_ids = sorted(set(robot_ids))
            masks[name] = np.isin(segmentation, robot_ids)
        return (
            float(np.max(np.abs(actual - q[index]))), actual, isaac_states,
            numerical_link_states, usd_states, images, masks,
        )

    if args.gui:
        index = 0
        while simulation_app.is_running():
            write_frame(index)
            if index < total - 1:
                index += 1
            elif args.loop:
                index = 0
            elif args.pause_at_end:
                sim.render()
                time.sleep(0.005)
            else:
                break
        simulation_app.close()
        return 0

    paths = {
        "overview": out / f"{args.video_prefix}_overview.mp4",
        "side": out / f"{args.video_prefix}_side.mp4",
        "top": out / f"{args.video_prefix}_top.mp4",
        "robot_only": out / f"{args.video_prefix}_robot_only.mp4",
    }
    writers = {
        name: cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (args.width, args.height))
        for name, path in paths.items()
    }
    maximum_readback = 0.0
    parity_frames = sorted(set(int(round(value * (total - 1))) for value in (0.0, .1, .25, .5, .75, .9, 1.0)))
    parity_samples = []
    trace_rows = []
    first_masks = {}
    first_images = {}
    rendered_motion = {name: [] for name in cameras}
    parity_rgb = {}
    parity_masks = {}
    actual_q_rows = []
    previous_actual = None
    for index in range(total):
        readback, actual, isaac_states, numerical_link_states, usd_states, images, masks = write_frame(index)
        actual_q_rows.append(actual.copy())
        maximum_readback = max(maximum_readback, readback)
        for name, image in images.items():
            mask = masks[name]
            if name not in first_masks:
                first_masks[name] = mask.copy()
                first_images[name] = image.copy()
            union = mask | first_masks[name]
            centroid = np.mean(np.argwhere(mask), axis=0).tolist() if np.any(mask) else [None, None]
            first_centroid = np.mean(np.argwhere(first_masks[name]), axis=0) if np.any(first_masks[name]) else np.asarray([np.nan, np.nan])
            centroid_delta = float(np.linalg.norm(np.asarray(centroid, dtype=float) - first_centroid)) if centroid[0] is not None else None
            coordinates = np.argwhere(mask)
            bbox = (
                [int(coordinates[:, 1].min()), int(coordinates[:, 0].min()), int(coordinates[:, 1].max()), int(coordinates[:, 0].max())]
                if len(coordinates) else None
            )
            rgb_delta = np.abs(image.astype(np.float32) - first_images[name].astype(np.float32))
            row = {
                "frame": index,
                "robot_pixel_count": int(np.count_nonzero(mask)),
                "mask_xor_pixels_from_frame0": int(np.count_nonzero(mask ^ first_masks[name])),
                "mask_identical_to_frame0": bool(np.array_equal(mask, first_masks[name])),
                "robot_mask_centroid_yx": centroid,
                "robot_mask_centroid_displacement_px": centroid_delta,
                "robot_mask_bbox_xyxy": bbox,
                "mean_rgb_difference_in_robot_union": float(np.mean(rgb_delta[union]) if np.any(union) else 0.0),
                "robot_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "raw_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
            rendered_motion[name].append(row)
            if index in parity_frames:
                parity_rgb[f"rgb_{name}_{index}"] = image.copy()
                parity_masks[f"mask_{name}_{index}"] = mask.copy()
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (0, 0), (args.width, 48), (0, 0, 0), -1)
            cv2.putText(bgr, f"{args.artifact_version.upper()} KINEMATIC WHOLE MOTION | action {index}/{total-1}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (70, 235, 255), 1, cv2.LINE_AA)
            cv2.putText(bgr, "ACTUAL G1+DEX3 MESH | PHYSICS STEPS 0 | CARTESIAN BACKBONE FROZEN", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, .34, (80, 235, 120), 1, cv2.LINE_AA)
            writers[name].write(bgr)
        trace_rows.append({
            "rendered_frame": index,
            "action_index": index,
            "physics_step": int(sim.get_physics_step_count()),
            "target_q_checksum": hashlib.sha256(np.asarray(q[index], dtype=np.float64).tobytes()).hexdigest(),
            "actual_q_checksum": hashlib.sha256(np.asarray(actual, dtype=np.float64).tobytes()).hexdigest(),
            "actual_q_delta_max_rad": 0.0 if previous_actual is None else float(np.max(np.abs(actual - previous_actual))),
            "left_wrist_xyz": json.dumps(isaac_states["left_wrist"]["position_m"].tolist()),
            "right_wrist_xyz": json.dumps(isaac_states["right_wrist"]["position_m"].tolist()),
            "left_thumb_xyz": json.dumps(isaac_states["left_thumb_distal"]["position_m"].tolist()),
            "left_index_xyz": json.dumps(isaac_states["left_index_distal"]["position_m"].tolist()),
            "right_c_xyz": json.dumps(isaac_states["right_middle_C_distal"]["position_m"].tolist()),
            "camera_frame_checksum": rendered_motion["overview"][-1]["raw_rgb_sha256"],
            "robot_mask_centroid_x": rendered_motion["overview"][-1]["robot_mask_centroid_yx"][1],
            "robot_mask_centroid_y": rendered_motion["overview"][-1]["robot_mask_centroid_yx"][0],
        })
        if index in parity_frames:
            link_parity = {}
            for label in parity_links:
                isaac_position = np.asarray(isaac_states[label]["position_m"])
                numerical_position = np.asarray(numerical_link_states[label]["position_m"])
                usd_position = np.asarray(usd_states[label]["position_m"])
                link_parity[label] = {
                    "numerical_actual_q_vs_isaac_position_error_m": float(np.linalg.norm(numerical_position - isaac_position)),
                    "isaac_vs_render_stage_position_error_m": float(np.linalg.norm(isaac_position - usd_position)),
                }
            parity_samples.append({
                "trajectory_index": index,
                "requested_q": q[index].copy(),
                "articulation_readback_q": actual.copy(),
                "isaac_link_states": serialize_states(isaac_states),
                "numerical_link_states_from_readback_q": serialize_states(numerical_link_states),
                "render_stage_link_states_from_actual_articulation": serialize_states(usd_states),
                "link_position_parity": link_parity,
                "rendered_motion": {name: rendered_motion[name][-1] for name in cameras},
            })
        previous_actual = actual.copy()
        if index % 100 == 0:
            print(f"[{args.artifact_version.upper()} KINEMATIC] {index}/{total-1}", flush=True)
    for writer in writers.values():
        writer.release()
    with (out / "kinematic_frame_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
        writer.writeheader(); writer.writerows(trace_rows)
    with (out / "kinematic_render_parity_keyframes.npz").open("wb") as stream:
        np.savez_compressed(
            stream, parity_frames=np.asarray(parity_frames, dtype=np.int64),
            **parity_rgb, **parity_masks,
        )

    camera_audit = {}
    for name, rows in rendered_motion.items():
        key_rows = [rows[index] for index in parity_frames]
        camera_audit[name] = {
            "keyframes": key_rows,
            "robot_masks_identical_at_all_keyframes": bool(all(row["mask_identical_to_frame0"] for row in key_rows)),
            "maximum_keyframe_mask_xor_pixels": int(max(row["mask_xor_pixels_from_frame0"] for row in key_rows)),
            "maximum_keyframe_centroid_displacement_px": float(max(row["robot_mask_centroid_displacement_px"] or 0.0 for row in key_rows)),
            "robot_mask_nonempty_all_frames": bool(all(row["robot_pixel_count"] > 0 for row in rows)),
        }
    link_displacements = {}
    for label in parity_links:
        values = np.asarray([sample["isaac_link_states"][label]["position_m"] for sample in parity_samples])
        link_displacements[label] = {
            "maximum_displacement_from_start_m": float(np.max(np.linalg.norm(values - values[0], axis=1))),
            "start_position_m": values[0], "end_position_m": values[-1],
        }
    numerical_errors = [
        metric["numerical_actual_q_vs_isaac_position_error_m"]
        for sample in parity_samples for metric in sample["link_position_parity"].values()
    ]
    usd_errors = [
        metric["isaac_vs_render_stage_position_error_m"]
        for sample in parity_samples for metric in sample["link_position_parity"].values()
    ]
    rendered_pass = bool(all(
        not row["robot_masks_identical_at_all_keyframes"]
        and row["maximum_keyframe_mask_xor_pixels"] > 100
        and row["robot_mask_nonempty_all_frames"]
        for row in camera_audit.values()
    ))
    parity_result = {
        "status": f"{args.artifact_version.upper()}_KINEMATIC_COMMAND_ACTUAL_RENDER_PARITY_PASS" if rendered_pass else "BLOCKED_RENDERED_ROBOT_STATIC_DESPITE_MOVING_Q",
        "requested_q_motion_max_peak_to_peak_rad": float(np.max(np.ptp(q[:total], axis=0))),
        "articulation_q_motion_max_peak_to_peak_rad": float(np.max(np.ptp(np.asarray(actual_q_rows), axis=0))),
        "maximum_joint_readback_error_rad": maximum_readback,
        "parity_frames": parity_frames,
        "samples": parity_samples,
        "link_displacements": link_displacements,
        "maximum_numerical_fk_vs_isaac_position_error_m": float(max(numerical_errors)),
        "maximum_isaac_vs_render_stage_position_error_m": float(max(usd_errors)),
        "rendered_motion": camera_audit,
        "physics_steps": int(sim.get_physics_step_count()),
        "render_sync": {
            "state_source": "ACTUAL_ARTICULATION_READBACK",
            "use_fabric": False,
            "rtx_read_transforms_from_fabric": False,
            "explicit_update_articulations_kinematic": True,
            "explicit_actual_articulation_body_pose_to_temporary_review_stage": True,
            "target_based_link_transform_write": False,
            "render_context_transform_cadence_reset_before_camera_update": True,
        },
    }
    dump(out / "kinematic_execution_render_parity.json", parity_result)
    dump(out / "kinematic_rendered_motion_audit.json", {"status": f"{args.artifact_version.upper()}_KINEMATIC_RENDERED_MESH_MOTION_PASS" if rendered_pass else "BLOCKED_RENDERED_ROBOT_STATIC_DESPITE_MOVING_Q", "cameras": camera_audit})
    dump(out / "numerical_isaac_fk_parity_kinematic.json", {
        "maximum_position_error_m": float(max(numerical_errors)),
        "mean_position_error_m": float(np.mean(numerical_errors)),
        "samples": [{"trajectory_index": row["trajectory_index"], "link_position_parity": row["link_position_parity"]} for row in parity_samples],
    })
    result = {
        "status": "FULL_990_FRAME_KINEMATIC_RENDERFIX_PASS" if total == 990 and maximum_readback <= 1e-6 and rendered_pass else "BLOCKED_KINEMATIC_REVIEW",
        "trajectory_path": str(args.input.resolve()),
        "trajectory_sha256": sha256(args.input.resolve()),
        "frames": total,
        "fps": 7.5,
        "physics_steps": int(sim.get_physics_step_count()),
        "maximum_joint_readback_error_rad": maximum_readback,
        "joint_mapping_complete": not missing,
        "paper_white": True,
        "fabric_kinematic_forward": False,
        "actual_articulation_to_temporary_usd_render_sync": True,
        "cartesian_target_modified": False,
        "videos": {name: str(path.resolve()) for name, path in paths.items()},
    }
    dump(out / "isaaclab_kinematic_renderfix.json", result)
    if result["physics_steps"] != 0:
        raise RuntimeError("kinematic review stepped physics")
    simulation_app.close()
    print(json.dumps(result, indent=2))
    return 0 if result["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
