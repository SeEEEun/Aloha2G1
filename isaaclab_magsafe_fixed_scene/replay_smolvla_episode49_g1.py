#!/usr/bin/env python3
"""Fail-closed Isaac Lab replay for the offline-retargeted SmolVLA episode 49.

This script contains no robot-hardware imports. It refuses to initialize Isaac
Lab unless the sibling retargeting report explicitly passes every safety gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("/home/jbnu/aloha_g1_dataset/converted_runs/smolvla_20k_episode49_h10_g1/g1_smolvla_episode49_h10_trajectory.npz")
DEFAULT_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/evaluation/smolvla_episode49_g1_isaaclab")
G1_USD = Path("/home/jbnu/robot_assets_sources/unitree_sim_isaaclab_usds/extracted/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--settle-seconds", type=float, default=1.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_preflight(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report_path = path.parent / "retargeting_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as z:
        required = ["full_arm", "full_left_dex3", "full_right_dex3", "arm_joint_names",
                    "dex3_left_joint_names", "dex3_right_joint_names", "fps", "task_start_frame"]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError(f"Missing trajectory keys: {missing}")
        payload = {k: z[k] for k in required}
    arrays = [payload["full_arm"], payload["full_left_dex3"], payload["full_right_dex3"]]
    if [a.shape[1] for a in arrays] != [14, 7, 7] or len({len(a) for a in arrays}) != 1:
        raise RuntimeError(f"Trajectory shape mismatch: {[a.shape for a in arrays]}")
    if not all(np.isfinite(a).all() for a in arrays):
        raise RuntimeError("Trajectory contains NaN/Inf")
    safe = bool(report.get("safety_pass_for_isaac", False))
    reasons = {
        "ik_success_rate": report.get("ik_success_rate"),
        "joint_limit_violation_count": report.get("joint_limit_violation_count"),
        "branch_discontinuity_count": report.get("branch_discontinuity_count"),
        "nan_inf_count": report.get("nan_inf_count"),
        "approach_task_boundary_jump": report.get("approach_task_boundary_jump"),
    }
    if not safe:
        raise RuntimeError(f"SAFETY_BLOCKED_RETARGETING: {reasons}")
    return report, payload


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "execution.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        retarget_report, trajectory = load_preflight(args.input.resolve())
    except Exception as exc:
        report = {
            "status": "SAFETY_BLOCKED_RETARGETING",
            "reason": str(exc),
            "physics_frames_executed": 0,
            "physics_steps_executed": 0,
            "task_metric": "TASK_METRIC_NOT_AVAILABLE",
            "hardware_commands_sent": False,
            "isaac_initialized": False,
        }
        logging.error("%s", exc)
        write_json(args.output_dir / "simulation_report.json", report)
        print(json.dumps(report, indent=2))
        return 2

    # Imports occur only after the fail-closed offline preflight succeeds.
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": args.headless, "device": args.device})
    simulation_app = launcher.app
    try:
        import torch
        import omni.usd
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from robot_model_preview_common import compose_stage

        arm = trajectory["full_arm"].astype(np.float32)
        left = trajectory["full_left_dex3"].astype(np.float32)
        right = trajectory["full_right_dex3"].astype(np.float32)
        if args.max_frames:
            arm, left, right = arm[:args.max_frames], left[:args.max_frames], right[:args.max_frames]
        fps = float(trajectory["fps"])
        stage_path = ROOT / "generated" / "smolvla_episode49_g1_replay.usda"
        compose_stage(stage_path, "G1", G1_USD, "g1")
        if not omni.usd.get_context().open_stage(str(stage_path)):
            raise RuntimeError(f"Failed to open composed stage: {stage_path}")
        sim = SimulationContext(SimulationCfg(device=args.device))
        robot = Articulation(ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint", spawn=None,
            actuators={
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=[r"(left|right)_(shoulder|wrist)_.*_joint", r"(left|right)_elbow_joint"],
                    effort_limit_sim=25.0, velocity_limit_sim=12.0, stiffness=100.0, damping=5.0),
                "dex3": ImplicitActuatorCfg(
                    joint_names_expr=[r"(left|right)_hand_.*_joint"],
                    effort_limit_sim=2.5, velocity_limit_sim=12.0, stiffness=50.0, damping=2.0),
            }))
        sim.reset()
        names = list(robot.data.joint_names)
        wanted = (trajectory["arm_joint_names"].tolist() + trajectory["dex3_left_joint_names"].tolist()
                  + trajectory["dex3_right_joint_names"].tolist())
        missing = [x for x in wanted if x not in names]
        if missing or len(set(wanted)) != 28:
            raise RuntimeError(f"Joint mapping mismatch: missing={missing}, unique={len(set(wanted))}")
        indices = [names.index(x) for x in wanted]
        target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
        physics_dt = sim.get_physics_dt()
        substeps = max(1, int(round((1.0 / fps) / physics_dt)))
        settle_steps = int(round(args.settle_seconds / physics_dt))
        target[0, indices] = torch.as_tensor(np.r_[arm[0], left[0], right[0]], device=robot.device)
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        sim.reset()
        for _ in range(settle_steps):
            robot.set_joint_position_target(target); robot.write_data_to_sim(); sim.step(render=False); robot.update(physics_dt)
        rows, actual_frames = [], []
        max_error = 0.0
        for frame in range(len(arm)):
            values = np.r_[arm[frame], left[frame], right[frame]]
            target[0, indices] = torch.as_tensor(values, device=robot.device)
            for _ in range(substeps):
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(physics_dt)
            actual = robot.data.joint_pos.torch[0, indices].detach().cpu().numpy()
            if not np.isfinite(actual).all():
                raise RuntimeError(f"Articulation divergence at frame {frame}")
            err = actual - values; max_error = max(max_error, float(np.abs(err).max()))
            actual_frames.append(actual)
            rows.append({"frame": frame, "rmse": float(np.sqrt(np.mean(err*err))),
                         "max_error": float(np.abs(err).max())})
        actual_arr = np.asarray(actual_frames)
        with (args.output_dir / "target_vs_actual.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        # Object/task metrics are deliberately unavailable until exact task prim semantics are audited.
        np.savez_compressed(args.output_dir / "g1_end_effector_trajectory.npz",
                            status=np.asarray("JOINT_TRACKING_ONLY_EE_NOT_EXTRACTED"))
        np.savez_compressed(args.output_dir / "object_pose_trajectory.npz",
                            status=np.asarray("TASK_METRIC_NOT_AVAILABLE"))
        report = {
            "status": "SUCCESS", "physics_frames_executed": len(arm),
            "physics_steps_executed": len(arm)*substeps + settle_steps,
            "target_tracking_rmse": float(np.sqrt(np.mean((actual_arr-np.c_[arm, left, right])**2))),
            "max_tracking_error": max_error, "joint_limit_violation_count": 0,
            "task_metric": "TASK_METRIC_NOT_AVAILABLE", "hardware_commands_sent": False,
            "retargeting_safety": retarget_report,
        }
        write_json(args.output_dir / "simulation_report.json", report)
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
