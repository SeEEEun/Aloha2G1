#!/usr/bin/env python3
"""Dry-reset the frozen symmetric paper A/B initial pose in Isaac only."""

from __future__ import annotations

import argparse
import traceback

import numpy as np

from isaaclab.app import AppLauncher

from paper_core_source_rollout_common import (
    INITIAL_CONTRACT,
    INITIAL_CONTRACT_SHA256,
    ROOT,
    SafetyAudit,
    atomic_json,
    common_initial_condition,
    frozen_interfaces,
)
from policy_b_isaac_control_contract import PHYSICS_DT, build_implicit_actuators


SCENE_STAGE = (
    ROOT
    / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
)
OUTPUT = ROOT / "outputs/paper_core_ab/common_initial_state_isaac_audit.json"


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app


def run() -> None:
    import omni.usd
    import torch
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite Isaac initial-state audit: {OUTPUT}")
    if not SCENE_STAGE.is_file():
        raise FileNotFoundError(SCENE_STAGE)
    names, lower, upper, _ = frozen_interfaces()
    q, record = common_initial_condition(names, 1.0)
    safety = SafetyAudit(names, lower, upper)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"could not open Isaac stage: {SCENE_STAGE}")
    sim = SimulationContext(
        SimulationCfg(dt=PHYSICS_DT, device="cuda:0", use_fabric=True)
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint",
            spawn=None,
            actuators=build_implicit_actuators(ImplicitActuatorCfg),
        )
    )
    sim.reset()
    isaac_names = list(robot.data.joint_names)
    missing = [name for name in names if name not in isaac_names]
    ids = [isaac_names.index(name) for name in names if name in isaac_names]
    if missing or len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError(f"Isaac named joint mapping failed: {missing}")
    target = robot.data.default_joint_pos.torch.clone().to(
        robot.device, dtype=torch.float32
    )
    zero = torch.zeros_like(target)
    target[0, ids] = torch.as_tensor(q, device=robot.device, dtype=torch.float32)
    robot.write_joint_state_to_sim(target, zero)
    sim.forward()
    robot.update(0.0)
    immediate = (
        robot.data.joint_pos.torch[0, ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    for _ in range(max(1, int(round(1.0 / PHYSICS_DT)))):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(PHYSICS_DT)
    settled = (
        robot.data.joint_pos.torch[0, ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    velocity = (
        robot.data.joint_vel.torch[0, ids]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    immediate_error = np.abs(immediate - q)
    settled_error = np.abs(settled - q)
    immediate_tolerance = float(
        record["tolerances"]["immediate_reset_max_abs_from_nominal_rad"]
    )
    settle_tolerance = float(
        record["tolerances"]["post_settle_max_abs_from_nominal_rad"]
    )
    hard_violation = (settled < lower - 1e-9) | (settled > upper + 1e-9)
    collision = safety.collision(settled[None])
    checks = {
        "named_joint_mapping": True,
        "finite": bool(
            np.isfinite(immediate).all()
            and np.isfinite(settled).all()
            and np.isfinite(velocity).all()
        ),
        "immediate_reset_tolerance": float(np.max(immediate_error))
        <= immediate_tolerance,
        "post_settle_tolerance": float(np.max(settled_error)) <= settle_tolerance,
        "settled_hard_limits": int(np.count_nonzero(hard_violation)) == 0,
        "settled_hard_self_collision": collision[
            "invalid_hard_self_collision_incidence"
        ]
        == 0,
    }
    result = {
        "schema_version": "paper_core_common_initial_state_isaac_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "contract": str(INITIAL_CONTRACT),
        "contract_sha256": INITIAL_CONTRACT_SHA256,
        "joint_names": names,
        "nominal_q_rad": q,
        "immediate_q_rad": immediate,
        "settled_q_rad": settled,
        "settled_qdot_rad_s": velocity,
        "immediate_max_abs_error_rad": float(np.max(immediate_error)),
        "immediate_tolerance_rad": immediate_tolerance,
        "post_settle_max_abs_error_rad": float(np.max(settled_error)),
        "post_settle_arm_l2_error_rad": float(
            np.linalg.norm(settled[:14] - q[:14])
        ),
        "post_settle_tolerance_rad": settle_tolerance,
        "settled_max_abs_velocity_rad_s": float(np.max(np.abs(velocity))),
        "hard_limit_violation_count": int(np.count_nonzero(hard_violation)),
        "hard_limit_violation_joints": [
            names[index] for index in np.flatnonzero(hard_violation)
        ],
        "collision": collision,
        "same_pose_for_act_a40_and_act_b40": True,
        "policy_inference_performed": False,
        "commands_sent_after_settle": 0,
        "real_hardware": False,
    }
    atomic_json(OUTPUT, result)
    print(result)
    if result["status"] != "PASS":
        raise RuntimeError("symmetric common A/B initial pose failed Isaac dry reset")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
