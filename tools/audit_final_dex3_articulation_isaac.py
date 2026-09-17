#!/usr/bin/env python3
"""Collision-isolated audit of the exact final-EVAL35 G1/Dex3 articulation.

This is a diagnostic, not a controller.  It applies the same anonymous-session
Dex3 stops, actuator contract, simulation time step, and named articulation
mapping as the final physical runner.  Task-object rigid bodies and colliders
are disabled before the articulation is created.  Each Dex3 DOF is then swept
independently through three interior targets.  An optional replay consumes the
already-saved B01 EXECUTED_COMMAND trace to test whether the excursion remains
when all task-environment collision is isolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/final_episode_registered_eval35/00_forensic_audit/dex3_zero_contact"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--replay-event-log", type=Path)
parser.add_argument("--replay-frames", type=int, default=230)
parser.add_argument("--replay-only", action="store_true")
parser.add_argument("--solver-position-iterations", type=int)
parser.add_argument("--solver-velocity-iterations", type=int)
parser.add_argument(
    "--isolate-all-task-environment",
    action="store_true",
    help="Disable every collision/rigid body under DollHandoffEnvironment, not only the doll.",
)
parser.add_argument(
    "--also-isolate-table",
    action="store_true",
    help="In addition to the doll, collision-isolate only the table hierarchy.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app
print("[Dex3Audit] application ready", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    print("[Dex3Audit] importing runtime modules", flush=True)
    import omni.usd
    import torch
    from pxr import Usd, UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from tools.policy_b_isaac_control_contract import CONTROLLER_CONTRACT
    print("[Dex3Audit] runtime modules imported", flush=True)

    config = read_json(CONFIG)
    contract = read_json(CONTRACT)
    names = [str(name) for name in contract["joint_names"]]
    specs = {int(row["index"]): row for row in contract["joint_specs"]}
    dex_names = names[14:28]
    inset = 0.005
    source_scene = Path(config["source_scene"]).resolve()
    print(f"[Dex3Audit] opening {source_scene}", flush=True)
    if not omni.usd.get_context().open_stage(str(source_scene)):
        raise RuntimeError(f"failed to open stage: {source_scene}")
    print("[Dex3Audit] stage opened", flush=True)
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    articulation_prim = stage.GetPrimAtPath("/World/G1/Asset/root_joint")
    original_solver_position_iterations = int(
        articulation_prim.GetAttribute(
            "physxArticulation:solverPositionIterationCount"
        ).Get()
    )
    original_solver_velocity_iterations = int(
        articulation_prim.GetAttribute(
            "physxArticulation:solverVelocityIterationCount"
        ).Get()
    )
    if args.solver_position_iterations is not None:
        articulation_prim.GetAttribute(
            "physxArticulation:solverPositionIterationCount"
        ).Set(int(args.solver_position_iterations))
    if args.solver_velocity_iterations is not None:
        articulation_prim.GetAttribute(
            "physxArticulation:solverVelocityIterationCount"
        ).Set(int(args.solver_velocity_iterations))

    # The B01 reproduction has no doll contact.  The primary audit nevertheless
    # removes the doll mechanically so that no object impulse can be involved.
    disabled_colliders: list[str] = []
    disabled_rigid_bodies: list[str] = []
    isolation_roots = ["/World/DollHandoffEnvironment/Doll"]
    if args.isolate_all_task_environment:
        isolation_roots = ["/World/DollHandoffEnvironment"]
    elif args.also_isolate_table:
        isolation_roots.append("/World/DollHandoffEnvironment/Table")
    for prim in stage.Traverse():
        if not any(str(prim.GetPath()).startswith(root) for root in isolation_roots):
            continue
        collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        if collision:
            collision.CreateCollisionEnabledAttr(False)
            disabled_colliders.append(str(prim.GetPath()))
        rigid = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        if rigid:
            rigid.CreateRigidBodyEnabledAttr(False)
            disabled_rigid_bodies.append(str(prim.GetPath()))
    print(f"[Dex3Audit] isolated {len(disabled_colliders)} colliders", flush=True)

    prim_by_name: dict[str, Usd.Prim] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in dex_names:
            if prim.GetName() in prim_by_name:
                raise RuntimeError(f"duplicate Dex3 joint: {prim.GetName()}")
            prim_by_name[prim.GetName()] = prim
    if set(prim_by_name) != set(dex_names):
        raise RuntimeError("Dex3 USD joint set mismatch")
    print("[Dex3Audit] found 14 USD joints", flush=True)

    mapping_rows: list[dict[str, Any]] = []
    runtime_lower = np.zeros(14, dtype=np.float64)
    runtime_upper = np.zeros(14, dtype=np.float64)
    for local_index, (policy_index, name) in enumerate(
        zip(range(14, 28), dex_names, strict=True)
    ):
        row = specs[policy_index]
        prim = prim_by_name[name]
        joint = UsdPhysics.RevoluteJoint(prim)
        source_lower = math.radians(float(joint.GetLowerLimitAttr().Get()))
        source_upper = math.radians(float(joint.GetUpperLimitAttr().Get()))
        project_lower = float(row["minimum"])
        project_upper = float(row["maximum"])
        lower = max(source_lower, project_lower) + inset
        upper = min(source_upper, project_upper) - inset
        joint.GetLowerLimitAttr().Set(math.degrees(lower))
        joint.GetUpperLimitAttr().Set(math.degrees(upper))
        runtime_lower[local_index] = lower
        runtime_upper[local_index] = upper
        mapping_rows.append(
            {
                "policy_action_index": policy_index,
                "prepared_archive_index": policy_index,
                "execution_layer_index": policy_index,
                "project_joint_name": name,
                "usd_joint_prim_path": str(prim.GetPath()),
                "usd_body0": [str(path) for path in joint.GetBody0Rel().GetTargets()],
                "usd_body1": [str(path) for path in joint.GetBody1Rel().GetTargets()],
                "usd_drive_axis": str(joint.GetAxisAttr().Get()),
                "usd_local_rot0": str(joint.GetLocalRot0Attr().Get()),
                "usd_local_rot1": str(joint.GetLocalRot1Attr().Get()),
                "source_usd_lower_rad": source_lower,
                "source_usd_upper_rad": source_upper,
                "project_lower_rad": project_lower,
                "project_upper_rad": project_upper,
                "safety_lower_rad": lower,
                "safety_upper_rad": upper,
            }
        )
    print("[Dex3Audit] applied 14 session-layer stops", flush=True)

    drive = config["finger_drive"]
    actuators = json.loads(json.dumps(CONTROLLER_CONTRACT["actuators"]))
    actuators["dex3"].update(
        {
            "effort_limit_sim": float(drive["effort_limit_sim"]),
            "velocity_limit_sim": float(drive["velocity_limit_sim"]),
            "stiffness": float(drive["kp"]),
            "damping": float(drive["kd"]),
        }
    )
    actuator_cfg = {
        name: ImplicitActuatorCfg(**value) for name, value in actuators.items()
    }
    dt = float(config["timing"]["physics_dt_s"])
    substeps = int(config["timing"]["physics_substeps_per_control_frame"])
    sim = SimulationContext(
        SimulationCfg(
            dt=dt,
            device=str(config["simulation"]["device"]),
            gravity=(0.0, 0.0, -float(config["simulation"]["gravity_m_s2"])),
            use_fabric=bool(config["simulation"]["use_fabric"]),
        )
    )
    print("[Dex3Audit] simulation context created", flush=True)
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/G1/Asset/root_joint",
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "^(?!.*_hand_).*$": 0.0,
                    **{
                        name: float(0.5 * (runtime_lower[i] + runtime_upper[i]))
                        for i, name in enumerate(dex_names)
                    },
                }
            ),
            actuators=actuator_cfg,
        )
    )
    print("[Dex3Audit] articulation created", flush=True)
    sim.reset()
    print("[Dex3Audit] simulation reset", flush=True)
    isaac_names = list(robot.data.joint_names)
    ids = [isaac_names.index(name) for name in names]
    if len(ids) != 28 or len(set(ids)) != 28:
        raise RuntimeError("named articulation mapping is not one-to-one")
    for row in mapping_rows:
        row["physx_articulation_dof_index"] = int(
            isaac_names.index(row["project_joint_name"])
        )
        row["measured_state_readback_index"] = row["policy_action_index"]
        row["reported_state_label"] = row["project_joint_name"]

    # Read the PhysX view's active limits to prove that the session edits
    # reached the instantiated articulation.
    view_limits = robot.root_physx_view.get_dof_limits()
    if hasattr(view_limits, "detach"):
        view_limits = view_limits.detach().cpu().numpy()
    view_limits = np.asarray(view_limits, dtype=np.float64)
    if view_limits.ndim == 3:
        view_limits = view_limits[0]
    for local_index, row in enumerate(mapping_rows):
        dof = row["physx_articulation_dof_index"]
        row["physx_runtime_lower_rad"] = float(view_limits[dof, 0])
        row["physx_runtime_upper_rad"] = float(view_limits[dof, 1])

    target = robot.data.default_joint_pos.torch.clone().to(
        robot.device, dtype=torch.float32
    )
    velocity = torch.zeros_like(target)
    midpoint = np.zeros(28, dtype=np.float64)
    midpoint[:14] = 0.0
    midpoint[14:] = 0.5 * (runtime_lower + runtime_upper)

    def write_state(command: np.ndarray) -> None:
        target[0, ids] = torch.as_tensor(
            command, device=robot.device, dtype=torch.float32
        )
        robot.write_joint_state_to_sim(target, velocity)
        sim.forward()
        robot.update(0.0)

    def step(command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target[0, ids] = torch.as_tensor(
            command, device=robot.device, dtype=torch.float32
        )
        for _ in range(substeps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
        measured = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        measured_qd = robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        return measured, measured_qd

    sweep_rows: list[dict[str, Any]] = []
    per_joint: list[dict[str, Any]] = []
    sweep_items = [] if args.replay_only else list(enumerate(dex_names))
    for local_index, name in sweep_items:
        policy_index = local_index + 14
        command = midpoint.copy()
        write_state(command)
        values = np.linspace(runtime_lower[local_index], runtime_upper[local_index], 7)[[1, 3, 5, 3]]
        joint_samples: list[dict[str, Any]] = []
        for target_value in values:
            start = float(command[policy_index])
            for phase, scalar in enumerate(np.linspace(start, target_value, 21)[1:]):
                command[policy_index] = float(scalar)
                measured, measured_qd = step(command)
                sample = {
                    "joint": name,
                    "policy_index": policy_index,
                    "phase": "RAMP",
                    "phase_index": phase,
                    "command_rad": float(command[policy_index]),
                    "drive_target_rad": float(command[policy_index]),
                    "measured_rad": float(measured[policy_index]),
                    "measured_velocity_rad_s": float(measured_qd[policy_index]),
                    "error_rad": float(measured[policy_index] - command[policy_index]),
                }
                sweep_rows.append(sample)
                joint_samples.append(sample)
            for phase in range(20):
                measured, measured_qd = step(command)
                sample = {
                    "joint": name,
                    "policy_index": policy_index,
                    "phase": "HOLD",
                    "phase_index": phase,
                    "command_rad": float(command[policy_index]),
                    "drive_target_rad": float(command[policy_index]),
                    "measured_rad": float(measured[policy_index]),
                    "measured_velocity_rad_s": float(measured_qd[policy_index]),
                    "error_rad": float(measured[policy_index] - command[policy_index]),
                }
                sweep_rows.append(sample)
                joint_samples.append(sample)
        commanded = np.asarray([row["command_rad"] for row in joint_samples])
        measured = np.asarray([row["measured_rad"] for row in joint_samples])
        hold_error = np.asarray(
            [row["error_rad"] for row in joint_samples if row["phase"] == "HOLD"]
        )
        project_lower = float(specs[policy_index]["minimum"])
        project_upper = float(specs[policy_index]["maximum"])
        runtime_tolerance = 2.0e-3
        project_tolerance = 1.0e-6
        slope = float(np.polyfit(commanded, measured, 1)[0])
        per_joint.append(
            {
                "joint": name,
                "policy_index": policy_index,
                "mapping_pass": True,
                "sign_slope": slope,
                "sign_pass": bool(slope > 0.5),
                "readback_max_hold_error_rad": float(np.max(np.abs(hold_error))),
                "readback_pass": bool(np.max(np.abs(hold_error)) <= 0.02),
                "measured_min_rad": float(np.min(measured)),
                "measured_max_rad": float(np.max(measured)),
                "runtime_limit_pass": bool(
                    np.min(measured) >= runtime_lower[local_index] - runtime_tolerance
                    and np.max(measured) <= runtime_upper[local_index] + runtime_tolerance
                ),
                "project_limit_pass": bool(
                    np.min(measured) >= project_lower - project_tolerance
                    and np.max(measured) <= project_upper + project_tolerance
                ),
            }
        )

    replay: dict[str, Any] | None = None
    if args.replay_event_log:
        with np.load(args.replay_event_log.resolve(), allow_pickle=False) as archive:
            trace_names = archive["joint_names"].astype(str).tolist()
            control = np.asarray(archive["control_frame"], dtype=np.int64)
            executed = np.asarray(archive["EXECUTED_COMMAND"], dtype=np.float64)
        if trace_names != names:
            raise RuntimeError("B01 replay joint-name order mismatch")
        first_rows = np.flatnonzero(np.r_[True, np.diff(control) != 0])
        replay_commands = executed[first_rows]
        replay_commands = replay_commands[: min(args.replay_frames, len(replay_commands))]
        write_state(replay_commands[0])
        replay_measured: list[np.ndarray] = []
        replay_velocity: list[np.ndarray] = []
        for command in replay_commands:
            measured, measured_qd = step(command)
            replay_measured.append(measured.copy())
            replay_velocity.append(measured_qd.copy())
        replay_measured_array = np.asarray(replay_measured)
        replay_velocity_array = np.asarray(replay_velocity)
        target_index = names.index("left_hand_middle_1_joint")
        max_frame = int(np.argmax(replay_measured_array[:, target_index]))
        replay = {
            "source_event_log": str(args.replay_event_log.resolve()),
            "source_event_log_sha256": sha256_file(args.replay_event_log.resolve()),
            "frames": int(len(replay_commands)),
            "joint": names[target_index],
            "policy_index": target_index,
            "maximum_measured_rad": float(replay_measured_array[max_frame, target_index]),
            "maximum_measured_frame": max_frame,
            "command_at_maximum_rad": float(replay_commands[max_frame, target_index]),
            "velocity_at_maximum_rad_s": float(replay_velocity_array[max_frame, target_index]),
            "minimum_measured_rad": float(np.min(replay_measured_array[:, target_index])),
            "authoritative_limit_violation_samples": int(
                np.count_nonzero(
                    (replay_measured_array[:, target_index] < specs[target_index]["minimum"] - 1.0e-6)
                    | (replay_measured_array[:, target_index] > specs[target_index]["maximum"] + 1.0e-6)
                )
            ),
            "environment_isolation": isolation_roots,
        }

    status = {
        "mapping_pass_count": 14 if args.replay_only else sum(row["mapping_pass"] for row in per_joint),
        "sign_pass_count": None if args.replay_only else sum(row["sign_pass"] for row in per_joint),
        "readback_pass_count": None if args.replay_only else sum(row["readback_pass"] for row in per_joint),
        "runtime_hard_limit_pass_count": None if args.replay_only else sum(row["runtime_limit_pass"] for row in per_joint),
        "project_hard_limit_pass_count": None if args.replay_only else sum(row["project_limit_pass"] for row in per_joint),
    }
    overall_pass = (
        bool(replay is not None and replay["authoritative_limit_violation_samples"] == 0)
        if args.replay_only
        else all(value == 14 for value in status.values())
    )
    report = {
        "schema_version": "final_eval35_dex3_zero_contact_articulation_audit_v1",
        "status": "PASS" if overall_pass else "FAIL",
        "scope": "same scene/articulation/actuator/timebase as final EVAL35; task object collision-isolated",
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "joint_contract": str(CONTRACT),
        "joint_contract_sha256": sha256_file(CONTRACT),
        "source_scene": str(source_scene),
        "source_scene_sha256": sha256_file(source_scene),
        "physics_dt_s": dt,
        "control_dt_s": dt * substeps,
        "substeps": substeps,
        "dex3_safety_inset_rad": inset,
        "solver_iterations": {
            "source_position": original_solver_position_iterations,
            "source_velocity": original_solver_velocity_iterations,
            "runtime_position": int(
                articulation_prim.GetAttribute(
                    "physxArticulation:solverPositionIterationCount"
                ).Get()
            ),
            "runtime_velocity": int(
                articulation_prim.GetAttribute(
                    "physxArticulation:solverVelocityIterationCount"
                ).Get()
            ),
        },
        "isolation_roots": isolation_roots,
        "disabled_collision_prims": disabled_colliders,
        "disabled_rigid_bodies": disabled_rigid_bodies,
        "mapping": mapping_rows,
        "per_joint": per_joint,
        "summary": status,
        "replay": replay,
        "sweep_rows": sweep_rows,
    }
    output = args.output_dir.resolve()
    atomic_json(output / "DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json", report)
    print(json.dumps({"status": report["status"], **status, "replay": replay}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        print("[Dex3Audit] entering main", flush=True)
        exit_code = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 2
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
