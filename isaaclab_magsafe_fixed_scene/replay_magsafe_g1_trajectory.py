"""Replay a G1 arm and bilateral Dex3 NPZ in the Isaac Lab MagSafe scene."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent
G1_USD = Path("/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd")
DEFAULT_ARM = Path("/home/jbnu/aloha_g1_dataset/converted_runs/smolvla_20k_episode49_consensus_relative_g1/g1_episode49_consensus_relative_trajectory.npz")
DEFAULT_FULL = Path("/home/jbnu/aloha_g1_dataset/outputs/g1_magsafe_arm_dex3_full_trajectory.npz")
RESULTS = Path("/home/jbnu/aloha_g1_dataset/outputs/g1_in_existing_magsafe_scene")
OUTPUT = ROOT / "generated" / "magsafe_g1_trajectory_replay.usda"

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=None, help="Compatibility alias for --full-input")
parser.add_argument("--arm-input", type=Path, default=DEFAULT_ARM)
parser.add_argument("--full-input", type=Path, default=DEFAULT_FULL)
parser.add_argument("--mode", choices=("kinematic-arm", "kinematic-full"), default="kinematic-arm")
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--settle-seconds", type=float, default=0.75)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--speed", type=float, choices=(.25, .5, 1.0), default=1.0)
parser.add_argument("--loop", action="store_true")
parser.add_argument(
    "--scene-mode",
    choices=("kinematic", "contact", "physical-task", "inspection", "magnetic"),
    default="physical-task",
)
parser.add_argument("--magnet-config", type=Path, default=ROOT / "magnet_config_v2.json")
parser.add_argument("--camera", choices=("overview", "front", "side", "top"), default="overview")
parser.add_argument("--root-forward-offset-m", type=float, default=0.0)
parser.add_argument("--root-lateral-offset-m", type=float, default=0.0)
parser.add_argument("--root-z-offset-m", type=float, default=0.0)
parser.add_argument("--root-yaw-offset-deg", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.input is not None:
    args.full_input = args.input
launcher = AppLauncher(args)
simulation_app = launcher.app

from robot_model_preview_common import CAMERAS, MAGSAFE_USD, POSE_CONFIG, compose_stage, load_pose


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def print_scene_identity(stage) -> dict:
    """Report immutable preview/replay inputs from the shared helper."""
    from pxr import UsdGeom
    original_root_position, root_quaternion = load_pose("g1")
    cache = UsdGeom.XformCache()
    applied = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/G1")).ExtractTranslation()
    root_position = [float(applied[0]), float(applied[1]), float(applied[2])]
    object_rows = {}
    for path in ("/World/MagSafeScene/Table", "/World/MagSafeScene/Phone",
                 "/World/MagSafeScene/Accessory", "/World/MagSafeScene/Charger"):
        prim = stage.GetPrimAtPath(path)
        matrix = cache.GetLocalToWorldTransform(prim)
        object_rows[path] = [list(row) for row in matrix]
    report = {
        "status": "PREVIEW_REPLAY_SHARED_AUTHORITATIVE_SCENE",
        "scene_usd": str(MAGSAFE_USD.resolve()), "scene_usd_sha256": sha256(MAGSAFE_USD),
        "scene_layout": str((ROOT / "scene_layout.json").resolve()),
        "scene_layout_sha256": sha256(ROOT / "scene_layout.json"),
        "pose_config": str(POSE_CONFIG.resolve()), "pose_config_sha256": sha256(POSE_CONFIG),
        "g1_prim_path": "/World/G1", "g1_articulation_path": "/World/G1/Asset/root_joint",
        "g1_asset": str(G1_USD), "g1_original_root_position": original_root_position,
        "g1_root_position": root_position,
        "g1_root_quaternion_wxyz": root_quaternion,
        "overview_camera_eye": list(CAMERAS["overview"][0]),
        "overview_camera_target": list(CAMERAS["overview"][1]),
        "aloha_prim_present": bool(stage.GetPrimAtPath("/World/StationaryALOHA")),
        "g1_prim_count": sum(str(p.GetPath()) == "/World/G1" for p in stage.Traverse()),
        "magsafe_scene_prim_count": sum(str(p.GetPath()) == "/World/MagSafeScene" for p in stage.Traverse()),
        "object_world_transforms": object_rows,
    }
    print("[SCENE_IDENTITY] " + json.dumps(report, sort_keys=True), flush=True)
    return report


def load_current_trajectory() -> dict:
    """Load current canonical files; never infer a mapping by index."""
    import numpy as np
    with np.load(args.arm_input.resolve(), allow_pickle=False) as z:
        if "g1_arm_joint_trajectory" not in z or "arm_joint_names" not in z:
            raise RuntimeError("arm input missing authoritative keys")
        arm = z["g1_arm_joint_trajectory"].astype(np.float32)
        arm_names = z["arm_joint_names"].astype(str).tolist()
        fps = float(z["fps"])
    with np.load(args.full_input.resolve(), allow_pickle=False) as z:
        required = ("arm_qpos", "left_dex3_qpos", "right_dex3_qpos", "arm_joint_names",
                    "left_dex3_joint_names", "right_dex3_joint_names", "primitive_source",
                    "authoritative_for_real_robot", "real_robot_command_allowed")
        missing = [key for key in required if key not in z.files]
        if missing:
            raise RuntimeError(f"full trajectory missing keys: {missing}")
        if bool(z["authoritative_for_real_robot"]) or bool(z["real_robot_command_allowed"]):
            raise RuntimeError("simulation safety metadata failed")
        left = z["left_dex3_qpos"].astype(np.float32)
        right = z["right_dex3_qpos"].astype(np.float32)
        left_names = z["left_dex3_joint_names"].astype(str).tolist()
        right_names = z["right_dex3_joint_names"].astype(str).tolist()
        if not np.array_equal(z["arm_joint_names"].astype(str), np.asarray(arm_names)):
            raise RuntimeError("arm joint-name order differs between authoritative inputs")
        if not np.array_equal(z["arm_qpos"].astype(np.float32), arm):
            raise RuntimeError("full trajectory does not preserve authoritative arm qpos exactly")
        primitive_source = str(z["primitive_source"])
    if arm.shape != (990, 14) or left.shape != (990, 7) or right.shape != (990, 7):
        raise RuntimeError(f"unexpected shapes arm={arm.shape}, left={left.shape}, right={right.shape}")
    if len(set(arm_names + left_names + right_names)) != 28:
        raise RuntimeError("duplicate trajectory joint names")
    if not all(np.isfinite(x).all() for x in (arm, left, right)):
        raise RuntimeError("trajectory contains NaN/Inf")
    if args.mode == "kinematic-arm":
        # Hold both hands at the first simulation-placeholder pose. This is
        # diagnostic neutral/open-state reuse, never a real-hand calibration.
        left = np.repeat(left[:1], len(arm), axis=0)
        right = np.repeat(right[:1], len(arm), axis=0)
    if args.max_frames:
        arm, left, right = arm[:args.max_frames], left[:args.max_frames], right[:args.max_frames]
    return dict(arm=arm, left=left, right=right, arm_names=arm_names,
                left_names=left_names, right_names=right_names, fps=fps,
                primitive_source=primitive_source)


def freeze_objects(stage) -> None:
    from pxr import UsdPhysics
    for prim in stage.Traverse():
        if any(word in prim.GetName().lower() for word in ("phone", "accessory", "charger")):
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if api:
                api.GetKinematicEnabledAttr().Set(True)
    stage.GetRootLayer().Save()


def main() -> None:
    import numpy as np
    import torch
    import omni.usd
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    trajectory = load_current_trajectory()
    arm, left, right = trajectory["arm"], trajectory["left"], trajectory["right"]
    arm_names, left_names, right_names = trajectory["arm_names"], trajectory["left_names"], trajectory["right_names"]
    fps = trajectory["fps"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        report = {"status":"DRY_RUN_PASS", "mode":args.mode, "frames":len(arm), "fps":fps,
                  "scene_source":str(MAGSAFE_USD.resolve()),
                  "g1_pose_source":str((ROOT/'magsafe_robot_preview_config.json').resolve()),
                  "g1_usd":str(G1_USD), "joint_mapping":"NAME_BASED_RUNTIME_CHECK_PENDING",
                  "arm_joint_names":arm_names, "left_dex3_joint_names":left_names,
                  "right_dex3_joint_names":right_names, "primitive_source":trajectory["primitive_source"],
                  "objects_moved_by_replay":False, "physics_grasp":False,
                  "hardware_commands_sent":False, "dds_initialized":False}
        (RESULTS/'dry_run.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2)); return
    stage = compose_stage(
        OUTPUT, "G1", G1_USD, "g1",
        forward_offset_m=args.root_forward_offset_m,
        lateral_offset_m=args.root_lateral_offset_m,
        z_offset_m=args.root_z_offset_m,
        yaw_offset_deg=args.root_yaw_offset_deg,
    )
    scene_identity = print_scene_identity(stage)
    if scene_identity["aloha_prim_present"] or scene_identity["g1_prim_count"] != 1 or scene_identity["magsafe_scene_prim_count"] != 1:
        raise RuntimeError(f"authoritative scene composition invariant failed: {scene_identity}")
    mode = "kinematic"
    freeze_objects(stage)
    if not omni.usd.get_context().open_stage(str(OUTPUT)):
        raise RuntimeError(f"Cannot open {OUTPUT}")
    sim = SimulationContext(SimulationCfg(device="cpu"))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/G1/Asset/root_joint",
        spawn=None,
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_(shoulder|wrist)_.*_joint", r"(left|right)_elbow_joint"],
                effort_limit_sim=25.0,
                velocity_limit_sim=12.0,
                stiffness=100.0,
                damping=5.0,
            ),
            "dex3": ImplicitActuatorCfg(
                joint_names_expr=[r"(left|right)_hand_.*_joint"],
                effort_limit_sim=2.5,
                velocity_limit_sim=12.0,
                stiffness=50.0,
                damping=2.0,
            ),
        },
    ))
    sim.reset()
    names = list(robot.data.joint_names)
    wanted = arm_names + left_names + right_names
    missing = [name for name in wanted if name not in names]
    duplicate_trajectory = len(wanted) - len(set(wanted))
    mapped_indices = [names.index(name) for name in wanted if name in names]
    duplicate_mapping = len(mapped_indices) - len(set(mapped_indices))
    unused_trajectory = [name for name in wanted if name not in names]
    unused_g1 = [name for name in names if name not in wanted]
    if missing or duplicate_trajectory or duplicate_mapping:
        raise RuntimeError(f"Ambiguous NPZ-to-USD mapping: missing={missing}, unique={len(set(wanted))}")
    (RESULTS / "isaac_runtime_joint_mapping.json").write_text(json.dumps({
        "status": "PASS", "mapping": "NAME_BASED", "runtime_joint_order": names,
        "requested_joint_names": wanted, "mapped_indices": mapped_indices,
        "arm_mapping": f"{sum(x in names for x in arm_names)}/14",
        "left_dex3_mapping": f"{sum(x in names for x in left_names)}/7",
        "right_dex3_mapping": f"{sum(x in names for x in right_names)}/7",
        "missing": missing, "duplicate_trajectory_joints": duplicate_trajectory,
        "duplicate_mapping_indices": duplicate_mapping,
        "unused_trajectory_joints": unused_trajectory, "unused_g1_joints": unused_g1,
        "mode": args.mode, "frames_requested": len(arm), "hardware_commands_sent": False,
        "dds_initialized": False, "scene_identity": scene_identity,
    }, indent=2) + "\n")
    print(f"[G1] set_joint_position_target{inspect.signature(robot.set_joint_position_target)}")
    print(f"[G1] runtime_joint_order={names}")
    print(f"[G1] arm={arm_names}; left_dex3={left_names}; right_dex3={right_names}")
    pos = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    indices = [names.index(name) for name in wanted]
    # Apply authored frame 0 before timed replay. Only the 28 named arm/hand
    # joints are touched; root, legs and waist stay at the preview defaults.
    first_values = np.r_[arm[0], left[0], right[0]]
    pos[0, indices] = torch.as_tensor(first_values, device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
    sim.forward()
    sim.render()
    robot.update(sim.get_physics_dt())
    print(f"[G1] frame0_applied=True settle_seconds={args.settle_seconds} lower_body_waist_source=PREVIEW_DEFAULT", flush=True)
    settle_deadline = time.monotonic() + max(args.settle_seconds, 0.0)
    while simulation_app.is_running() and time.monotonic() < settle_deadline:
        simulation_app.update()
    eye, target = CAMERAS[args.camera]
    sim.set_camera_view(eye, target)
    start = time.monotonic()
    last = -1
    frames_seen = set()
    max_mapped_error = 0.0
    max_preview_default_error = 0.0
    unused_indices = [names.index(name) for name in unused_g1]
    physics_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        source_time = (time.monotonic() - start) * args.speed
        f = min(source_time * fps, len(arm) - 1)
        i0, i1 = int(f), min(int(f) + 1, len(arm) - 1)
        alpha = float(f - i0)
        values = np.r_[(1-alpha)*arm[i0]+alpha*arm[i1],
                       (1-alpha)*left[i0]+alpha*left[i1],
                       (1-alpha)*right[i0]+alpha*right[i1]]
        pos[0, indices] = torch.as_tensor(values, device=robot.device, dtype=torch.float32)
        # Kinematic validation: write the authored state directly. No controller
        # target, physics grasp, or object attachment is used in either mode.
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        sim.forward()
        sim.render()
        robot.update(physics_dt)
        source_frame = int(f)
        frames_seen.add(source_frame)
        mapped_error = torch.max(torch.abs(robot.data.joint_pos.torch[0, indices] - pos[0, indices])).item()
        preview_error = torch.max(torch.abs(robot.data.joint_pos.torch[0, unused_indices] - pos[0, unused_indices])).item() if unused_indices else 0.0
        max_mapped_error = max(max_mapped_error, mapped_error)
        max_preview_default_error = max(max_preview_default_error, preview_error)
        if source_frame != last and source_frame % 15 == 0:
            speed_now = torch.max(torch.abs(robot.data.joint_vel.torch)).item()
            effort = torch.max(torch.abs(robot.data.applied_torque.torch)).item()
            print(
                f"[G1] frame={source_frame}/{len(arm)-1} mode={args.mode}:{mode} "
                f"mapped_error_rad={mapped_error:.6g} preview_default_error_rad={preview_error:.6g} "
                f"velocity_max={speed_now:.6g} effort_max={effort:.6g}"
            )
            last = source_frame
        if i0 == len(arm) - 1:
            if args.loop:
                start = time.monotonic()
            else:
                break
    completion = {
        "status": "KINEMATIC_REPLAY_COMPLETED", "source_frames": len(arm),
        "last_source_frame": max(frames_seen, default=-1), "fps": fps,
        "max_mapped_joint_error_rad": max_mapped_error,
        "max_unused_preview_joint_deviation_rad": max_preview_default_error,
        "frame0_preapplied": True, "lower_body_waist_trajectory_authored": False,
        "object_attachment": False, "physics_grasp": False,
        "hardware_commands_sent": False, "dds_initialized": False,
    }
    (RESULTS / "kinematic_replay_completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print("[G1_COMPLETION] " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
