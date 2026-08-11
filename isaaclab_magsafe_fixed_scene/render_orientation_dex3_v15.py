#!/usr/bin/env python3
"""Zero-physics-step Isaac Lab replay for a semantic v15 28-joint candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_orientation_dex3_v15"
NPZ = OUT / "full_arm_dex3_trajectory.npz"
LEFT_NPZ = OUT / "left_dex3_trajectory.npz"
RIGHT_NPZ = OUT / "right_dex3_trajectory.npz"
NUMERIC_GATES = OUT / "numeric_gate_summary.json"
SCENE_DIR = ROOT / "isaaclab_magsafe_fixed_scene"
FIXED_SCENE = SCENE_DIR / "generated/magsafe_fixed_scene.usda"
ACTIVE_SCENE = SCENE_DIR / "generated/magsafe_g1_model_preview.usda"
LAYOUT = SCENE_DIR / "scene_layout.json"
ROOT_CONFIG = ROOT / "configs/g1_root_forward_v14.approved.json"
CONTACT_CONFIG = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
G1_USD = Path(
    "/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/"
    "g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd"
)
METHOD = "ALOHA_PRIMARY_EP49_ORIENTATION_DEX3_V15"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gui", action="store_true")
parser.add_argument(
    "--gui-camera",
    choices=("overview", "side", "left_close", "right_close", "charger_close"),
    default="overview",
    help="Static GUI viewpoint; it does not alter replay state or semantic timing.",
)
parser.add_argument("--max-frames", type=int)
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--height", type=int, default=405)
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
    return str(value)


def dump(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


with np.load(NPZ, allow_pickle=False) as payload:
    q = payload["controlled_q"].astype(np.float32)
    joint_names = payload["controlled_joint_names"].astype(str).tolist()
    numerical_left = payload["achieved_left_position"].astype(float)
    numerical_right = payload["achieved_right_position"].astype(float)
    phone_pose = payload["target_phone_pose"].astype(float)
    accessory_pose = payload["target_accessory_pose"].astype(float)
    semantic_names = payload["semantic_event_names"].astype(str).tolist()
    semantic_indices = payload["semantic_event_indices"].astype(int).tolist()
    root_position = payload["g1_root"].astype(float)
with np.load(LEFT_NPZ, allow_pickle=False) as payload:
    numerical_contacts = {
        "left_A": payload["contact_A"].astype(float),
        "left_B": payload["contact_B"].astype(float),
        "left_C": payload["contact_C"].astype(float),
    }
with np.load(RIGHT_NPZ, allow_pickle=False) as payload:
    numerical_contacts.update({
        "right_A": payload["contact_A"].astype(float),
        "right_B": payload["contact_B"].astype(float),
        "right_C": payload["contact_C"].astype(float),
    })
if q.ndim != 2 or q.shape[1] != len(joint_names) or len(q) != len(phone_pose):
    raise RuntimeError("v15 controlled trajectory schema mismatch")
if len(joint_names) != 28 or not np.isfinite(q).all():
    raise RuntimeError("v15 requires finite 28-joint control")
event_by_index: dict[int, list[str]] = {}
for name, index in zip(semantic_names, semantic_indices):
    event_by_index.setdefault(index, []).append(name)
key_frames = sorted(set([0, len(q) - 1, *semantic_indices]))
numeric_gates = json.loads(NUMERIC_GATES.read_text(encoding="utf-8"))
object_follow_enabled = bool(numeric_gates["numeric_pass"])


def add_metadata(raw: Path, output: Path, camera: str) -> None:
    metadata = {
        "trajectory_path": str(NPZ.resolve()),
        "trajectory_sha256": sha256(NPZ),
        "renderer_path": str(Path(__file__).resolve()),
        "renderer_sha256": sha256(Path(__file__)),
        "active_scene_sha256": sha256(ACTIVE_SCENE),
        "fixed_scene_sha256": sha256(FIXED_SCENE),
        "controlled_q_key": "controlled_q",
        "controlled_joint_count": len(joint_names),
        "root_position_m": root_position.tolist(),
        "frame_count": len(q),
        "fps": 7.5,
        "camera": camera,
        "method": METHOD,
        "physics_steps": 0,
        "actual_Dex3_applied": True,
        "object_follow_enabled": object_follow_enabled,
        "diagnostic_only": True,
    }
    temporary = output.with_name(output.stem + ".metadata.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-map", "0", "-c", "copy",
            "-metadata", f"title=Episode49 v15 G1 Dex3 {camera}",
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
    from robot_model_preview_common import compose_stage

    OUT.mkdir(parents=True, exist_ok=True)
    carb.settings.get_settings().set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
    scene_hashes_before = {str(path.resolve()): sha256(path) for path in (LAYOUT, FIXED_SCENE, ACTIVE_SCENE)}
    root_config = json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))
    stage_path = OUT / "v15_isaaclab_kinematic_replay.usda"
    stage = compose_stage(
        stage_path,
        "G1",
        G1_USD,
        "g1",
        forward_offset_m=float(root_config["selected_total_forward_offset_m"]),
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/V15Lights/Dome")
    dome.CreateIntensityAttr(900.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.93))
    key = UsdLux.DistantLight.Define(stage, "/World/V15Lights/Key")
    key.CreateIntensityAttr(2600.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, 15.0))
    for prim in stage.Traverse():
        if any(token in prim.GetName().lower() for token in ("phone", "accessory", "charger")):
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if api:
                api.GetKinematicEnabledAttr().Set(True)
                api.GetRigidBodyEnabledAttr().Set(False)
    stage.GetRootLayer().Save()
    if not omni.usd.get_context().open_stage(str(stage_path)):
        raise RuntimeError(stage_path)
    live_stage = omni.usd.get_context().get_stage()

    simulation = SimulationContext(SimulationCfg(device="cuda:0", use_fabric=False))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint",
            spawn=None,
            actuators={
                "controlled": ImplicitActuatorCfg(
                    joint_names_expr=[
                        r"(left|right)_(shoulder|wrist)_.*_joint",
                        r"(left|right)_elbow_joint",
                        r"(left|right)_hand_(thumb|index|middle)_.*_joint",
                    ],
                    effort_limit_sim=25.0,
                    velocity_limit_sim=12.0,
                    stiffness=100.0,
                    damping=5.0,
                )
            },
        )
    )
    camera_poses = {
        "overview": ((1.15, -1.22, 1.38), (0.49, -0.01, 1.00)),
        "side": ((0.93, -0.28, 1.16), (0.45, 0.09, 0.95)),
        "left_close": ((0.76, -0.34, 1.08), (0.46, 0.09, 0.90)),
        "right_close": ((0.78, -0.29, 1.20), (0.48, 0.02, 1.02)),
        "charger_close": ((0.72, -0.06, 1.15), (0.42, 0.21, 0.94)),
    }
    cameras = {}
    if not args.gui:
        for name in camera_poses:
            cameras[name] = Camera(CameraCfg(
                prim_path=f"/World/V15Camera_{name}",
                update_period=0,
                height=args.height,
                width=args.width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.05, 20.0)),
            ))
    simulation.reset()
    runtime_names = list(robot.data.joint_names)
    missing = [name for name in joint_names if name not in runtime_names]
    duplicates = [name for name in joint_names if runtime_names.count(name) != 1]
    if missing or duplicates:
        raise RuntimeError(f"joint mapping failed missing={missing}, duplicate={duplicates}")
    joint_ids = [runtime_names.index(name) for name in joint_names]
    mapping = {name: index for name, index in zip(joint_names, joint_ids)}
    mapping_hash = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    zero_velocity = torch.zeros((1, len(joint_names)), dtype=torch.float32, device=robot.device)
    dt = simulation.get_physics_dt()
    body_names = list(robot.data.body_names)
    asset_xform = UsdGeom.Xformable(live_stage.GetPrimAtPath("/World/G1/Asset"))
    world_from_asset = np.asarray(
        asset_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float
    ).T
    asset_from_world = np.linalg.inv(world_from_asset)
    body_ops = {}
    for body_name in body_names:
        prim = live_stage.GetPrimAtPath(f"/World/G1/Asset/{body_name}")
        if not prim.IsValid():
            raise RuntimeError(f"missing visual body prim {body_name}")
        body_ops[body_name] = UsdGeom.Xformable(prim).MakeMatrixXform()
    for name, camera in cameras.items():
        eye, target = camera_poses[name]
        camera.set_world_poses_from_view(np.asarray([eye], np.float32), np.asarray([target], np.float32))
    contact_config = json.loads(CONTACT_CONFIG.read_text(encoding="utf-8"))
    contact_specs = {
        f"{side}_{role}": contact_config[side][role]
        for side in ("left", "right") for role in ("A", "B", "C")
    }
    body_ids = {name: body_names.index(spec["distal_link"]) for name, spec in contact_specs.items()}
    wrist_body_ids = {
        side: body_names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")
    }

    def read_q() -> np.ndarray:
        return tensor(robot.data.joint_pos)[0, joint_ids].detach().cpu().numpy().astype(float)

    def sync_links() -> None:
        positions = tensor(robot.data.body_pos_w)[0].detach().cpu().numpy().astype(float)
        quaternions = tensor(robot.data.body_quat_w)[0].detach().cpu().numpy().astype(float)
        for body_id, body_name in enumerate(body_names):
            world = np.eye(4)
            world[:3, :3] = Rotation.from_quat(quaternions[body_id]).as_matrix()
            world[:3, 3] = positions[body_id]
            local = asset_from_world @ world
            body_ops[body_name].Set(Gf.Matrix4d(*local.T.reshape(-1).tolist()))

    def link_contact_positions() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        positions = tensor(robot.data.body_pos_w)[0].detach().cpu().numpy().astype(float)
        quaternions = tensor(robot.data.body_quat_w)[0].detach().cpu().numpy().astype(float)
        result = {}
        wrist_local = {}
        for name, spec in contact_specs.items():
            body_id = body_ids[name]
            result[name] = positions[body_id] + Rotation.from_quat(quaternions[body_id]).as_matrix() @ np.asarray(
                spec["local_position_xyz_m"], dtype=float
            )
            side = name.split("_", 1)[0]
            wrist_id = wrist_body_ids[side]
            wrist_rotation = Rotation.from_quat(quaternions[wrist_id]).as_matrix()
            wrist_local[name] = wrist_rotation.T @ (result[name] - positions[wrist_id])
        return result, wrist_local

    def write_frame(frame: int) -> tuple[dict, dict[str, np.ndarray]]:
        requested = torch.as_tensor(q[frame][None], dtype=torch.float32, device=robot.device)
        robot.write_joint_position_to_sim_index(position=requested, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim_index(velocity=zero_velocity, joint_ids=joint_ids)
        torch.cuda.synchronize()
        immediate = read_q()
        view = simulation.physics_manager.get_physics_sim_view()
        if view is None:
            raise RuntimeError("PhysX simulation view unavailable")
        view.update_articulations_kinematic()
        simulation.forward()
        robot.update(dt)
        torch.cuda.synchronize()
        after_update = read_q()
        sync_links()
        simulation.render()
        simulation.render_context.reset_transform_cadence()
        images = {}
        for name, camera in cameras.items():
            camera.update(dt, force_recompute=True)
            images[name] = tensor(camera.data.output["rgb"])[0].detach().cpu().numpy()[..., :3].copy()
        torch.cuda.synchronize()
        after_render = read_q()
        contacts, contacts_wrist_local = link_contact_positions()
        return {
            "frame": frame,
            "requested_readback_immediate_max_error_rad": float(np.max(np.abs(immediate - q[frame]))),
            "requested_readback_after_update_max_error_rad": float(np.max(np.abs(after_update - q[frame]))),
            "requested_readback_after_render_max_error_rad": float(np.max(np.abs(after_render - q[frame]))),
            "contact_positions": contacts,
            "contact_positions_wrist_local": contacts_wrist_local,
            "contact_proxy_parity_m": {
                name: float(np.linalg.norm(contacts[name] - numerical_contacts[name][frame]))
                for name in contacts
            },
            "physics_step_count": int(simulation.get_physics_step_count()),
        }, images

    if args.gui:
        import omni.ui as ui

        simulation.set_camera_view(*camera_poses[args.gui_camera])
        window = ui.Window("Episode49 v15 G1 + Dex3 kinematic replay", width=600, height=180)
        model = ui.SimpleIntModel(0)
        with window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Action-index slider (semantic labels resolved from timeline)")
                ui.IntSlider(model, min=0, max=len(q) - 1)
                status = ui.Label("")
        previous = None
        while simulation_app.is_running():
            frame = max(0, min(len(q) - 1, model.as_int))
            if frame != previous:
                audit, _ = write_frame(frame)
                status.text = (
                    f"action={frame} events={','.join(event_by_index.get(frame, [])) or '-'}\n"
                    f"readback={audit['requested_readback_after_render_max_error_rad']:.3e} rad | "
                    f"physics_steps={audit['physics_step_count']}"
                )
                previous = frame
            simulation.render()
            time.sleep(0.005)
        simulation_app.close()
        return 0

    video_specs = {
        "overview_robot": ("overview", OUT / "v15_g1_dex3_robot_only_overview.mp4"),
        "side_robot": ("side", OUT / "v15_g1_dex3_robot_only_side.mp4"),
        "left_close": ("left_close", OUT / "v15_left_phone_grasp_closeup.mp4"),
        "right_close": ("right_close", OUT / "v15_right_accessory_hook_closeup.mp4"),
        "charger_close": ("charger_close", OUT / "v15_charger_placement_closeup.mp4"),
        "overview_object": ("overview", OUT / "v15_object_follow_overview.mp4"),
        "side_object": ("side", OUT / "v15_object_follow_side.mp4"),
    }
    writers = {}
    for label, (_, output) in video_specs.items():
        raw = OUT / f".{output.stem}.raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(raw)
        writers[label] = (writer, raw, output)
    key_images: dict[tuple[str, int], np.ndarray] = {}
    key_audits = []
    maximum_readback = 0.0
    maximum_proxy_error = 0.0
    first_contacts = None
    first_contacts_wrist_local = None
    maximum_finger_displacement = {name: 0.0 for name in contact_specs}
    maximum_finger_wrist_local_displacement = {name: 0.0 for name in contact_specs}
    total = len(q) if args.max_frames is None else min(len(q), int(args.max_frames))
    for frame in range(total):
        audit, images = write_frame(frame)
        maximum_readback = max(maximum_readback, audit["requested_readback_after_render_max_error_rad"])
        maximum_proxy_error = max(maximum_proxy_error, max(audit["contact_proxy_parity_m"].values()))
        if first_contacts is None:
            first_contacts = audit["contact_positions"]
            first_contacts_wrist_local = audit["contact_positions_wrist_local"]
        for name, value in audit["contact_positions"].items():
            maximum_finger_displacement[name] = max(
                maximum_finger_displacement[name], float(np.linalg.norm(value - first_contacts[name]))
            )
        for name, value in audit["contact_positions_wrist_local"].items():
            maximum_finger_wrist_local_displacement[name] = max(
                maximum_finger_wrist_local_displacement[name],
                float(np.linalg.norm(value - first_contacts_wrist_local[name])),
            )
        event_text = ",".join(event_by_index.get(frame, [])) or "-"
        for label, (camera_name, _) in video_specs.items():
            image = cv2.cvtColor(images[camera_name], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (0, 0), (args.width, 82), (0, 0, 0), -1)
            cv2.putText(image, f"EP49 v15 | action {frame}/{len(q)-1} | {event_text[:64]}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (80, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(image, "ACTUAL G1 + DEX3 MESH | KINEMATIC | PHYSICS STEPS 0", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .44, (40, 220, 255), 1, cv2.LINE_AA)
            if "object" in label:
                text = "KINEMATIC OBJECT FOLLOW" if object_follow_enabled else "OBJECT FOLLOW DISABLED: CONTACT GATE FAIL"
                cv2.putText(image, text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, .44, (40, 120, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, "FAILED DIAGNOSTIC CANDIDATE | NOT APPROVED", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, .44, (40, 120, 255), 1, cv2.LINE_AA)
            writers[label][0].write(image)
        if frame in key_frames:
            key_audits.append(audit)
            for camera_name, image in images.items():
                key_images[(camera_name, frame)] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if frame % 100 == 0:
            print(f"[V15_ISAAC] {frame}/{total-1}", flush=True)
    for label, (writer, raw, output) in writers.items():
        writer.release()
        add_metadata(raw, output, label)

    def contact_sheet(camera_name: str, output: Path, title: str) -> None:
        thumbs = []
        for frame in key_frames:
            if (camera_name, frame) not in key_images:
                continue
            image = key_images[(camera_name, frame)].copy()
            cv2.rectangle(image, (0, 0), (args.width, 28), (0, 0, 0), -1)
            labels = ",".join(event_by_index.get(frame, [])) or ("episode_start" if frame == 0 else "trajectory_end")
            cv2.putText(image, f"{labels} | action {frame}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (80, 255, 120), 1, cv2.LINE_AA)
            thumbs.append(cv2.resize(image, (360, 203)))
        columns = 3
        rows = int(np.ceil(len(thumbs) / columns))
        canvas = np.zeros((rows * 203 + 36, columns * 360, 3), dtype=np.uint8)
        cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 255, 120), 1, cv2.LINE_AA)
        for index, image in enumerate(thumbs):
            row, column = divmod(index, columns)
            canvas[36 + row * 203 : 36 + (row + 1) * 203, column * 360 : (column + 1) * 360] = image
        cv2.imwrite(str(output), canvas)

    contact_sheet("overview", OUT / "v15_semantic_keyframe_overview.png", "V15 SEMANTIC KEYFRAMES - FAILED DIAGNOSTIC")
    contact_sheet("left_close", OUT / "v15_left_ab_contact_closeup.png", "LEFT A+B CONTACT - FAILED DIAGNOSTIC")
    contact_sheet("right_close", OUT / "v15_right_c_contact_closeup.png", "RIGHT C CONTACT - FAILED DIAGNOSTIC")
    contact_sheet("charger_close", OUT / "v15_charger_alignment_closeup.png", "CHARGER ALIGNMENT - FAILED DIAGNOSTIC")
    contact_sheet("overview", OUT / "v15_orientation_axes_contact_sheet.png", "SOURCE-RELATIVE ORIENTATION STAGES")

    result = {
        "status": "BLOCKED_ISAACLAB_DEX3_REPLAY" if maximum_readback > 1e-6 else "ISAACLAB_ACTUAL_DEX3_MESH_REPLAY_PASS",
        "numeric_task_gates_pass": numeric_gates["numeric_pass"],
        "diagnostic_candidate_only": not numeric_gates["numeric_pass"],
        "trajectory_path": str(NPZ.resolve()),
        "trajectory_sha256": sha256(NPZ),
        "controlled_joint_count": len(joint_names),
        "joint_mapping": mapping,
        "joint_mapping_sha256": mapping_hash,
        "missing_joints": missing,
        "duplicate_joints": duplicates,
        "maximum_requested_readback_error_rad": maximum_readback,
        "maximum_numerical_contact_proxy_vs_Isaac_error_m": maximum_proxy_error,
        "maximum_finger_contact_displacement_from_start_m": maximum_finger_displacement,
        "maximum_finger_contact_wrist_local_displacement_from_start_m": maximum_finger_wrist_local_displacement,
        "task_finger_joint_peak_to_peak_rad": {
            name: {
                joint: float(np.ptp(q[:, joint_names.index(joint)]))
                for joint in contact_specs[name]["joint_names"]
            }
            for name in ("left_A", "left_B", "right_C")
        },
        # Wrist-local contact motion rules out apparent finger motion caused
        # solely by the arm moving through the world.
        "actual_task_finger_links_move": all(
            maximum_finger_wrist_local_displacement[name] > 0.0001
            and any(
                np.ptp(q[:, joint_names.index(joint)]) > 0.001
                for joint in contact_specs[name]["joint_names"]
            )
            for name in ("left_A", "left_B", "right_C")
        ),
        "non_task_finger_link_displacement_m": {
            name: maximum_finger_displacement[name]
            for name in ("left_C", "right_A", "right_B")
        },
        "physics_steps": int(simulation.get_physics_step_count()),
        "renderfix_transform_cadence_reset": True,
        "explicit_actual_G1_link_USD_sync": True,
        "object_follow_enabled": object_follow_enabled,
        "keyframes": key_audits,
        "videos": {label: str(value[2].resolve()) for label, value in writers.items()},
        "scene_hashes_before": scene_hashes_before,
        "scene_hashes_after": {str(path.resolve()): sha256(path) for path in (LAYOUT, FIXED_SCENE, ACTIVE_SCENE)},
        "real_robot_command_allowed": False,
    }
    dump(OUT / "isaaclab_kinematic_validation.json", result)
    if int(simulation.get_physics_step_count()) != 0:
        raise RuntimeError("physics step was used")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
