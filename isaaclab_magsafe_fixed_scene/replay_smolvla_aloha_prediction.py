#!/usr/bin/env python3
"""Physics replay of a teacher-forced ALOHA action trajectory in Isaac Lab.

This script controls only the simulated Stationary ALOHA articulation. It does
not import robot hardware, teleoperation, G1, LeRobot, or MuJoCo modules.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ALOHA_USD = Path("/home/jbnu/robot_assets/stationary_aloha/usd_imported/stationary_aloha_imported.usd")
OUTPUT_ROOT = PROJECT / "evaluation/smolvla_20k_isaaclab_aloha"
COMPOSED_STAGE = ROOT / "generated/smolvla_prediction_replay.usda"
ACTION_JOINT_NAMES = [
    *(f"follower_left_joint_{i}" for i in range(6)),
    "follower_left_left_carriage_joint",
    *(f"follower_right_joint_{i}" for i in range(6)),
    "follower_right_left_carriage_joint",
]
FULL_JOINT_NAMES = [
    *(f"follower_left_joint_{i}" for i in range(6)),
    "follower_left_left_carriage_joint", "follower_left_right_carriage_joint",
    *(f"follower_right_joint_{i}" for i in range(6)),
    "follower_right_left_carriage_joint", "follower_right_right_carriage_joint",
]
OBJECT_PATHS = {
    "phone": "/World/MagSafeScene/Phone",
    "accessory": "/World/MagSafeScene/Accessory",
    "charger": "/World/MagSafeScene/Charger",
}
LOG = logging.getLogger("isaaclab_smolvla_replay")

parser = argparse.ArgumentParser(description="Replay SmolVLA ALOHA joint targets in Isaac Lab")
parser.add_argument("--prediction-npz", type=Path, required=True)
parser.add_argument("--action-key", default="teacher_forced_predicted_action")
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--settle-seconds", type=float, default=1.0)
parser.add_argument("--record-video", action="store_true")
parser.add_argument("--video-output", type=Path, default=None)
parser.add_argument("--report-output", type=Path, default=None)
parser.add_argument("--max-frames", type=int, default=None)
parser.add_argument("--dry-run", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.speed <= 0 or args.settle_seconds < 0:
    parser.error("--speed must be positive and --settle-seconds non-negative")
if args.dry_run and (args.max_frames is None or args.max_frames > 100):
    parser.error("--dry-run requires --max-frames <= 100")
launcher = AppLauncher(args)
simulation_app = launcher.app


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete = path.with_name(path.name + ".incomplete")
    incomplete.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(incomplete, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete = path.with_name(path.name + ".incomplete")
    fields = list(rows[0]) if rows else ["frame_index"]
    with incomplete.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(incomplete, path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incomplete = path.with_name(path.name + ".incomplete")
    with incomplete.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(incomplete, path)


def refresh_global_outputs() -> None:
    reports = []
    for path in sorted(OUTPUT_ROOT.glob("episode_*/simulation_report.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    if not reports:
        return
    rows = []
    for report in reports:
        safety = report["safety_preflight"]
        rows.append({
            "episode_index": report["episode_index"],
            "status": report["status"],
            "input_frames": report["input_shape"][0],
            "physics_steps_executed": report.get("physics_steps_executed", 0),
            "joint_limit_violation_count": safety["joint_limit_violation_count"],
            "jump_threshold_exceeded_joint_count": len(safety["jump_threshold_exceeded_joints"]),
            "target_actual_rmse": report.get("target_actual_rmse"),
            "target_actual_max_error": report.get("target_actual_max_error"),
            "task_metric_status": report["task_metric_status"],
            "video": report.get("video"),
        })
    atomic_csv(OUTPUT_ROOT / "isaaclab_summary.csv", rows)
    atomic_json(OUTPUT_ROOT / "isaaclab_evaluation_report.json", {
        "evaluation_scope": "Isaac Lab Stationary ALOHA safety preflight and physics replay",
        "episodes": [r["episode_index"] for r in reports],
        "all_headless_replays_successful": all(r["status"] == "SUCCESS" for r in reports),
        "automatic_replay_blocked": any(r["status"] == "SAFETY_BLOCKED" for r in reports),
        "results": reports,
        "note": "No prediction was clipped, smoothed, resampled, scaled, or otherwise modified.",
    })


def load_prediction(path: Path, key: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as src:
        required = {"episode_index", "frame_index", "timestamp", "observation_state", key,
                    "expert_action", "fps", "checkpoint"}
        missing = sorted(required - set(src.files))
        if missing:
            raise RuntimeError(f"Prediction NPZ missing keys: {missing}")
        data = {name: src[name].copy() for name in src.files}
    action = np.asarray(data[key], dtype=np.float32)
    state = np.asarray(data["observation_state"], dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != 14 or state.shape != action.shape:
        raise RuntimeError(f"Expected state/action [T,14], got {state.shape}/{action.shape}")
    if len(data["frame_index"]) != len(action) or len(data["timestamp"]) != len(action):
        raise RuntimeError("Episode frame count mismatch in NPZ")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise RuntimeError("Prediction/state contains NaN or inf")
    return data


def fix_root(stage) -> tuple[str, str]:
    """Reuse the fixed-base layout authored by the existing ALOHA replay."""
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    root_path = Sdf.Path("/World/StationaryALOHA/Asset/Geometry/tabletop_link")
    joint_path = root_path.AppendChild("SmolVLAReplayWorldFixedJoint")
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise RuntimeError(f"ALOHA articulation root missing: {root_path}")
    transform = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(root))
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody1Rel().SetTargets([root_path])
    joint.CreateLocalPos0Attr().Set(transform.GetTranslation())
    q = transform.GetRotation().GetQuat()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(q.GetImaginary())))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
    physx = PhysxSchema.PhysxArticulationAPI.Apply(joint.GetPrim())
    physx.CreateSolverPositionIterationCountAttr().Set(64)
    physx.CreateSolverVelocityIterationCountAttr().Set(8)
    physx.CreateSleepThresholdAttr().Set(0.0)
    stage.GetRootLayer().Save()
    return str(root_path), str(joint_path)


def pose_to_array(pose_tensor) -> np.ndarray:
    return pose_tensor.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import omni.usd
    import torch
    from pxr import UsdGeom
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from robot_model_preview_common import CAMERAS, compose_stage, suppress_stationary_aloha_fixture

    source = load_prediction(args.prediction_npz, args.action_key)
    episode = int(source["episode_index"])
    episode_dir = OUTPUT_ROOT / f"episode_{episode:06d}"
    report_path = args.report_output or episode_dir / "simulation_report.json"
    action = np.asarray(source[args.action_key], dtype=np.float32)
    state = np.asarray(source["observation_state"], dtype=np.float32)
    expert = np.asarray(source["expert_action"], dtype=np.float32)
    requested_frames = min(len(action), args.max_frames or len(action))

    stage = compose_stage(COMPOSED_STAGE, "StationaryALOHA", ALOHA_USD, "stationary_aloha")
    suppress_stationary_aloha_fixture(stage)
    root_path, fixed_joint_path = fix_root(stage)
    object_prim_status = {k: bool(stage.GetPrimAtPath(v)) for k, v in OBJECT_PATHS.items()}
    if not omni.usd.get_context().open_stage(str(COMPOSED_STAGE)):
        raise RuntimeError(f"Failed to open composed stage: {COMPOSED_STAGE}")
    sim = SimulationContext(SimulationCfg(device="cpu"))
    robot = Articulation(ArticulationCfg(
        prim_path=fixed_joint_path,
        spawn=None,
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[r"follower_(left|right)_joint_[0-5]"],
                effort_limit_sim=27.0, velocity_limit_sim=8.0, stiffness=250.0, damping=20.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=[r"follower_(left|right)_(left|right)_carriage_joint"],
                effort_limit_sim=40.0, velocity_limit_sim=0.5, stiffness=100.0, damping=10.0,
            ),
        },
    ))
    phone = RigidObject(RigidObjectCfg(prim_path=OBJECT_PATHS["phone"], spawn=None))
    accessory = RigidObject(RigidObjectCfg(prim_path=OBJECT_PATHS["accessory"], spawn=None))
    sim.reset()
    names = list(robot.data.joint_names)
    missing = [name for name in FULL_JOINT_NAMES if name not in names]
    if missing:
        raise RuntimeError(f"Joint mapping mismatch; missing={missing}; actual={names}")
    ids = {name: names.index(name) for name in FULL_JOINT_NAMES}
    limits16 = robot.data.soft_joint_pos_limits.torch[0].detach().cpu().numpy()
    limits14 = np.empty((14, 2), dtype=np.float32)
    for side, offset in (("left", 0), ("right", 7)):
        for j in range(6):
            limits14[offset + j] = limits16[ids[f"follower_{side}_joint_{j}"]]
        left_lim = limits16[ids[f"follower_{side}_left_carriage_joint"]]
        right_lim = limits16[ids[f"follower_{side}_right_carriage_joint"]]
        limits14[offset + 6] = [max(left_lim[0], right_lim[0]), min(left_lim[1], right_lim[1])]

    expert_delta_p999 = np.percentile(np.abs(np.diff(expert, axis=0)), 99.9, axis=0)
    predicted_jump = np.max(np.abs(np.diff(action, axis=0)), axis=0)
    action_state_p999 = np.percentile(np.abs(expert - state), 99.9, axis=0)
    first_delta = np.abs(action[0] - state[0])
    limit_mask = (action < limits14[:, 0]) | (action > limits14[:, 1])
    # The specified threshold hierarchy: exact asset limits first, then this
    # episode's expert delta/action-state distributions. No arbitrary constant.
    jump_mask = predicted_jump > expert_delta_p999
    first_mask = first_delta > action_state_p999
    blockers = []
    if np.any(limit_mask):
        blockers.append("ACTUAL_ASSET_JOINT_LIMIT_VIOLATION")
    if np.any(first_mask):
        blockers.append("FIRST_TARGET_EXCEEDS_EXPERT_ACTION_STATE_P99_9")
    if np.any(jump_mask):
        blockers.append("PREDICTED_JUMP_EXCEEDS_EXPERT_DELTA_P99_9")
    safety = {
        "threshold_priority": [
            "actual Isaac Lab articulation soft_joint_pos_limits",
            "episode expert trajectory distributions",
        ],
        "joint_limits_14d": limits14.tolist(),
        "joint_limit_violation_count": int(limit_mask.sum()),
        "joint_limit_violation_count_per_joint": limit_mask.sum(axis=0).astype(int).tolist(),
        "predicted_min_per_joint": action.min(axis=0).tolist(),
        "predicted_max_per_joint": action.max(axis=0).tolist(),
        "expert_delta_p99_9": expert_delta_p999.tolist(),
        "predicted_max_jump_per_joint": predicted_jump.tolist(),
        "jump_threshold_exceeded_joints": np.flatnonzero(jump_mask).astype(int).tolist(),
        "expert_action_minus_state_p99_9": action_state_p999.tolist(),
        "first_target_minus_initial_state": first_delta.tolist(),
        "first_target_threshold_exceeded_joints": np.flatnonzero(first_mask).astype(int).tolist(),
        "blockers": blockers,
    }
    base_report: dict[str, Any] = {
        "episode_index": episode,
        "input": str(args.prediction_npz),
        "action_key": args.action_key,
        "input_shape": list(action.shape),
        "requested_frames": requested_frames,
        "dry_run": bool(args.dry_run),
        "scene": {
            "composed_stage": str(COMPOSED_STAGE),
            "aloha_usd": str(ALOHA_USD),
            "articulation_prim": fixed_joint_path,
            "root_body": root_path,
            "joint_names": names,
            "object_prims": object_prim_status,
        },
        "safety_preflight": safety,
        "task_metric_status": (
            "AVAILABLE_FOR_OBSERVATION_NOT_SUCCESS_CLASSIFICATION"
            if all(object_prim_status.values()) else "TASK_METRIC_NOT_AVAILABLE"
        ),
        "errors": [],
        "warnings": [],
    }
    if blockers:
        base_report["status"] = "SAFETY_BLOCKED"
        base_report["physics_steps_executed"] = 0
        base_report["task_metric_status"] = "TASK_METRIC_NOT_AVAILABLE"
        base_report["warnings"].append(
            "Physics replay was not automatically executed; raw predictions were not modified."
        )
        atomic_json(report_path, base_report)
        (episode_dir / "execution_log.txt").write_text(
            "status=SAFETY_BLOCKED\nphysics_steps_executed=0\n"
            f"blockers={','.join(blockers)}\n", encoding="utf-8"
        )
        refresh_global_outputs()
        LOG.error("Safety preflight blocked replay: %s", blockers)
        simulation_app.close()
        return

    pos = robot.data.default_joint_pos.torch.clone().to(dtype=torch.float32)
    vel = torch.zeros_like(pos)

    def fill_target(vector14: np.ndarray) -> None:
        for side, offset in (("left", 0), ("right", 7)):
            for j in range(6):
                pos[0, ids[f"follower_{side}_joint_{j}"]] = float(vector14[offset + j])
            g = float(vector14[offset + 6])
            pos[0, ids[f"follower_{side}_left_carriage_joint"]] = g
            pos[0, ids[f"follower_{side}_right_carriage_joint"]] = g

    # The only teleport: initialize once from recorded observation_state[0].
    fill_target(state[0])
    robot.write_joint_state_to_sim(position=pos, velocity=vel)
    robot.set_joint_position_target(pos)
    robot.write_data_to_sim()
    dt = float(sim.get_physics_dt())
    for _ in range(int(round(args.settle_seconds / dt))):
        sim.step(render=False)
        robot.update(dt)

    eye, target = CAMERAS["overview"]
    sim.set_camera_view(eye, target)
    physics_steps_per_action = max(1, int(round(1.0 / (float(source["fps"]) * args.speed * dt))))
    target_rows: list[dict[str, Any]] = []
    actual14_all, target14_all, ee_all, object_all = [], [], [], []
    left_body = list(robot.data.body_names).index("follower_left_link_6")
    right_body = list(robot.data.body_names).index("follower_right_link_6")
    reset_count = nan_count = 0
    initial_phone = pose_to_array(phone.data.root_pose_w.torch[0])
    initial_accessory = pose_to_array(accessory.data.root_pose_w.torch[0])
    runtime_stage = omni.usd.get_context().get_stage()
    charger_xform = UsdGeom.XformCache().GetLocalToWorldTransform(
        runtime_stage.GetPrimAtPath(OBJECT_PATHS["charger"])
    )
    charger_pos = np.asarray(charger_xform.ExtractTranslation(), dtype=np.float32)
    for frame in range(requested_frames):
        fill_target(action[frame])
        robot.set_joint_position_target(pos)
        robot.write_data_to_sim()
        for _ in range(physics_steps_per_action):
            sim.step(render=not args.headless)
            robot.update(dt)
            phone.update(dt)
            accessory.update(dt)
        actual16 = robot.data.joint_pos.torch[0].detach().cpu().numpy()
        actual14 = np.empty(14, np.float32)
        for side, offset in (("left", 0), ("right", 7)):
            for j in range(6):
                actual14[offset + j] = actual16[ids[f"follower_{side}_joint_{j}"]]
            actual14[offset + 6] = .5 * (
                actual16[ids[f"follower_{side}_left_carriage_joint"]]
                + actual16[ids[f"follower_{side}_right_carriage_joint"]]
            )
        if not np.isfinite(actual14).all():
            nan_count += int((~np.isfinite(actual14)).sum())
            raise RuntimeError(f"Articulation divergence at frame {frame}: {actual14}")
        actual14_all.append(actual14)
        target14_all.append(action[frame])
        ee_all.append(np.concatenate([
            pose_to_array(robot.data.body_pose_w.torch[0, left_body]),
            pose_to_array(robot.data.body_pose_w.torch[0, right_body]),
        ]))
        phone_pose = pose_to_array(phone.data.root_pose_w.torch[0])
        accessory_pose = pose_to_array(accessory.data.root_pose_w.torch[0])
        object_all.append(np.concatenate([phone_pose, accessory_pose, charger_pos]))
        row = {"frame_index": frame}
        for j in range(14):
            row[f"target_{j}"] = float(action[frame, j])
            row[f"actual_{j}"] = float(actual14[j])
            row[f"error_{j}"] = float(actual14[j] - action[frame, j])
        target_rows.append(row)

    actual_arr = np.asarray(actual14_all)
    target_arr = np.asarray(target14_all)
    error = actual_arr - target_arr
    velocity = np.diff(actual_arr, axis=0) * float(source["fps"])
    acceleration = np.diff(velocity, axis=0) * float(source["fps"])
    final_phone = pose_to_array(phone.data.root_pose_w.torch[0])
    final_accessory = pose_to_array(accessory.data.root_pose_w.torch[0])
    base_report.update({
        "status": "SUCCESS",
        "physics_steps_executed": requested_frames * physics_steps_per_action,
        "physics_dt": dt,
        "physics_steps_per_action": physics_steps_per_action,
        "target_actual_rmse": float(np.sqrt(np.mean(error**2))),
        "target_actual_max_error": float(np.max(np.abs(error))),
        "per_joint_tracking_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "joint_limit_violation_count": 0,
        "nan_inf_count": nan_count,
        "articulation_reset_count": reset_count,
        "max_joint_velocity": float(np.max(np.abs(velocity))),
        "max_joint_acceleration": float(np.max(np.abs(acceleration))) if len(acceleration) else 0.0,
        "self_collision_contact_warning": "TASK_METRIC_NOT_AVAILABLE",
        "object_metrics": {
            "phone_translation_m": float(np.linalg.norm(final_phone[:3] - initial_phone[:3])),
            "accessory_translation_m": float(np.linalg.norm(final_accessory[:3] - initial_accessory[:3])),
            "final_phone_to_charger_distance_m": float(np.linalg.norm(final_phone[:3] - charger_pos)),
            "task_success": "NOT_CLASSIFIED_FIXED_SCENE_ALIGNMENT_NOT_GUARANTEED",
        },
        "video": str(args.video_output) if args.record_video else None,
    })
    atomic_csv(episode_dir / "target_vs_actual.csv", target_rows)
    atomic_npz(episode_dir / "end_effector_trajectory.npz", {
        "frame_index": np.arange(requested_frames, dtype=np.int64),
        "left_pose_w_xyz_wxyz": np.asarray(ee_all)[:, :7],
        "right_pose_w_xyz_wxyz": np.asarray(ee_all)[:, 7:],
    })
    atomic_npz(episode_dir / "object_pose_trajectory.npz", {
        "frame_index": np.arange(requested_frames, dtype=np.int64),
        "phone_pose_w_xyz_wxyz": np.asarray(object_all)[:, :7],
        "accessory_pose_w_xyz_wxyz": np.asarray(object_all)[:, 7:14],
        "charger_position_w_xyz": np.asarray(object_all)[:, 14:],
    })
    if args.record_video:
        base_report["warnings"].append(
            "Video recording was requested but no unverified capture backend was invoked."
        )
        base_report["video"] = "TASK_METRIC_NOT_AVAILABLE"
    atomic_json(report_path, base_report)
    (episode_dir / "execution_log.txt").write_text(
        f"status=SUCCESS\nphysics_steps_executed={base_report['physics_steps_executed']}\n",
        encoding="utf-8",
    )
    refresh_global_outputs()
    simulation_app.close()


if __name__ == "__main__":
    main()
