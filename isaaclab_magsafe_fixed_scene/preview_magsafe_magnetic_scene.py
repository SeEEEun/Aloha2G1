"""GUI/single-environment MagSafe magnetic physics preview (no robot)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Preview robot-free MagSafe magnetic physics.")
parser.add_argument("--config", type=Path, default=ROOT / "magnet_config.json")
parser.add_argument("--test", choices=["initial_stability", "accessory_detach", "charger_snap", "outside_capture", "wrong_orientation"], default="initial_stability")
parser.add_argument("--duration", type=float, default=None)
parser.add_argument("--rebuild", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from magsafe_dynamic_scene_builder import build_magnetic_scene
from magsafe_magnet_controller import BodyState, MagnetCsvLogger, MagnetState, MagneticPair, load_config


def _body(obj: RigidObject) -> BodyState:
    pose = obj.data.root_pose_w.torch[0].detach().cpu().numpy()
    vel = obj.data.root_vel_w.torch[0].detach().cpu().numpy()
    # Isaac Lab 3.0 Beta tensor poses are XYZW; controller math uses WXYZ.
    q_xyzw = pose[3:7]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return BodyState(pose[:3], q_wxyz, vel[:3], vel[3:])


def _static(position, quat=(1.0, 0.0, 0.0, 0.0)) -> BodyState:
    return BodyState(np.array(position, float), np.array(quat, float), np.zeros(3), np.zeros(3))


def _set_pose(obj: RigidObject, pos, quat):
    q_wxyz = list(quat)
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    pose = torch.tensor([list(pos) + q_xyzw], device=obj.device, dtype=torch.float32)
    obj.write_root_pose_to_sim_index(root_pose=pose)
    obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=obj.device))


def _wrench(obj: RigidObject, force: np.ndarray, torque: np.ndarray):
    f = torch.tensor(force.reshape(1, 1, 3), device=obj.device, dtype=torch.float32)
    t = torch.tensor(torque.reshape(1, 1, 3), device=obj.device, dtype=torch.float32)
    obj.instantaneous_wrench_composer.set_forces_and_torques_index(f, t, is_global=True)


def main() -> dict:
    config_path = args_cli.config.expanduser().resolve()
    cfg = load_config(config_path)
    is_v2 = int(cfg.get("metadata", {}).get("version", 1)) >= 2
    scene_path = ROOT / "generated" / ("magsafe_magnetic_scene_v2.usda" if is_v2 else "magsafe_magnetic_scene.usda")
    if args_cli.rebuild or not scene_path.exists():
        build_magnetic_scene(output_path=scene_path, config_path=config_path)
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, render_interval=2, device=args_cli.device))
    sim_utils.GroundPlaneCfg(size=(6.0, 6.0)).func("/World/Ground", sim_utils.GroundPlaneCfg(size=(6.0, 6.0)))
    sim_utils.DomeLightCfg(intensity=1800.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1800.0))
    sim_utils.UsdFileCfg(usd_path=str(scene_path)).func("/World/MagSafeScene", sim_utils.UsdFileCfg(usd_path=str(scene_path)))
    phone = RigidObject(RigidObjectCfg(prim_path="/World/MagSafeScene/Phone", spawn=None))
    accessory = RigidObject(RigidObjectCfg(prim_path="/World/MagSafeScene/Accessory", spawn=None))
    if args_cli.test in ("charger_snap", "outside_capture", "wrong_orientation"):
        # Joint topology changes are made before the physics view is built.
        # Runtime joint deletion is deliberately avoided in this first GUI tool.
        joint = sim_utils.get_current_stage().GetPrimAtPath("/World/MagSafeScene/MagneticJoints/AccessoryPhone")
        if joint:
            joint.SetActive(False)
        print("[INFO] Charger test: accessory-phone joint disabled before reset.", flush=True)
    sim.set_camera_view((1.10, -0.82, 1.34), (0.42, 0.36, 0.86))
    sim.reset()
    print("[INFO] Physics views initialized.", flush=True)
    stage = sim_utils.get_current_stage()
    for label, path in (
        ("PadCenter", "/World/MagSafeScene/Charger/Frames/PadCenter"),
        ("PhoneTargetCenter", "/World/MagSafeScene/Charger/Frames/PhoneTargetCenter"),
        ("ChargerBaseCenter", "/World/MagSafeScene/Charger/Frames/BaseCenter"),
    ):
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0)
        print(f"[FRAME] {label} path={path} world_position={tuple(matrix.ExtractTranslation())}", flush=True)
    dt = sim.get_physics_dt()

    acc_pair = MagneticPair(
        "accessory_phone", cfg["accessory_phone"], np.array([0, -0.00175, 0.0]), np.array([0, -1, 0.0]),
        np.array([0, 0.004175, 0.0]), np.array([0, 1, 0.0]), 0.025,
        cfg["safety"]["max_accessory_acceleration_mps2"],
    )
    target_q = np.array(cfg.get("charger_target", {}).get("target_rotation_wxyz"), float) if is_v2 else None
    target_clearance = float(cfg.get("charger_target", {}).get("surface_clearance_m", 0.0))
    charger_pair = MagneticPair(
        "phone_charger", cfg["phone_charger"], np.array([0, 0.004175, 0.0]), np.array([0, 1, 0.0]),
        np.array([0, 0.005846519, 0.13261811]) + np.array([0, -math.cos(math.radians(15)), math.sin(math.radians(15))]) * target_clearance,
        np.array([0, -math.cos(math.radians(15)), math.sin(math.radians(15))]),
        0.177, cfg["safety"]["max_phone_acceleration_mps2"],
        target_orientation_world=target_q,
    )
    charger = _static((0.42, 0.52, 0.807))

    n = np.array([0, -math.cos(math.radians(15)), math.sin(math.radians(15))])
    target = charger.position + charger_pair.target_frame_local
    if is_v2:
        from magsafe_magnet_controller import quat_multiply
        perturb = np.array([math.cos(math.radians(3.5)), math.sin(math.radians(3.5)), 0.0, 0.0])
        good_q = quat_multiply(perturb, target_q)
    else:
        good_q = np.array([math.cos(math.radians(7.5)), -math.sin(math.radians(7.5)), 0, 0])
    if args_cli.test in ("charger_snap", "outside_capture", "wrong_orientation"):
        d = 0.025 if args_cli.test != "outside_capture" else 0.060
        q = good_q if args_cli.test != "wrong_orientation" else np.array([0.0, 0.0, 0.0, 1.0])
        offset = np.array([0, 0.004175, 0.0])
        # q rotates +Y; compute offset explicitly with a small local helper.
        from magsafe_magnet_controller import quat_rotate
        _set_pose(phone, target + n * d - quat_rotate(q, offset), q)
        print(f"[INFO] Phone test pose initialized for {args_cli.test}.", flush=True)
        # These tests isolate charger behavior from the initial accessory joint.

    duration = args_cli.duration or (4.0 if args_cli.test == "initial_stability" else 3.0)
    logger = MagnetCsvLogger(ROOT / "generated" / ("magnet_test_log_v2.csv" if is_v2 else "magnet_test_log.csv"))
    transitions = []
    max_force = max_torque = 0.0
    invalid_count = 0
    steps = int(duration / dt)
    for i in range(steps):
        now = i * dt
        pull = np.zeros(3)
        pull_norm = 0.0
        if args_cli.test == "accessory_detach":
            pull_norm = min(7.0, 7.0 * now / max(duration, 1e-6))
            pull = np.array([0.0, pull_norm, 0.0])
        ar = acc_pair.update(now, _body(accessory), _body(phone), external_force_norm=pull_norm)
        pr = charger_pair.update(
            now, _body(phone), charger,
            correct_face=float(np.dot(
                __import__("magsafe_magnet_controller").quat_rotate(_body(phone).quaternion_wxyz, np.array([0, 1, 0.])),
                -charger_pair.target_normal_local,
            )) > 0.0,
        )
        phone_now = _body(phone)
        from magsafe_magnet_controller import quat_rotate
        pr.long_axis = quat_rotate(phone_now.quaternion_wxyz, np.array([1.0, 0.0, 0.0]))
        pr.base_distance = float(np.linalg.norm(phone_now.position - charger.position))
        if ar.attach_event or ar.detach_event:
            transitions.append(("accessory_phone", ar.state.value, now))
        if pr.attach_event or pr.detach_event or (i and pr.state.value != prev_phone_state):
            transitions.append(("phone_charger", pr.state.value, now))
        prev_phone_state = pr.state.value
        # Native fixed joint holds the initial accessory. Magnetic force is only
        # needed when attracting/soft-locking; test pull is applied independently.
        _wrench(accessory, pull + (ar.force if ar.state != MagnetState.ATTACHED else 0.0), ar.torque)
        _wrench(phone, pr.force - (ar.force if ar.state != MagnetState.ATTACHED else 0.0), pr.torque - ar.torque)
        accessory.write_data_to_sim()
        phone.write_data_to_sim()
        logger.write(now, "accessory_phone", ar, "enabled" if ar.state == MagnetState.ATTACHED else "broken_or_disabled")
        logger.write(now, "phone_charger", pr, "force_soft_lock")
        max_force = max(max_force, float(np.linalg.norm(ar.force)), float(np.linalg.norm(pr.force)), pull_norm)
        max_torque = max(max_torque, float(np.linalg.norm(ar.torque)), float(np.linalg.norm(pr.torque)))
        invalid_count += int(ar.invalid) + int(pr.invalid)
        if i % 30 == 0:
            direction = pr.force / max(np.linalg.norm(pr.force), 1e-12)
            print(f"[MAGNET] t={now:5.2f} accessory={ar.state.value:10s} d={ar.distance:.4f}m | charger={pr.state.value:10s} d={pr.distance:.4f}m angle={pr.angle_deg:.1f}deg F={np.linalg.norm(pr.force):.2f}N direction={direction} target={target}")
        sim.step()
        phone.update(dt)
        accessory.update(dt)
        if not simulation_app.is_running():
            break
    logger.close()
    result = {"test": args_cli.test, "transitions": transitions, "max_force_n": max_force, "max_torque_nm": max_torque, "nan_inf": invalid_count, "final_accessory_state": ar.state.value, "final_charger_state": pr.state.value, "final_charger_distance_m": pr.distance, "final_charger_angle_deg": pr.angle_deg}
    print("[RESULT]", result)
    return result


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
