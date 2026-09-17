#!/usr/bin/env python3
"""Task-independent Isaac characterization of G1 Dex3 hard-bound tracking.

No policy, dataset sample, camera, task object, or real-robot interface is used.
Each controlled Dex3 joint is isolated and swept toward/away from both hard
bounds with the exact controller contract used by the Policy-B Isaac runner.
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
from policy_b_isaac_control_contract import (
    CONTROL_FPS,
    PHYSICS_DT,
    CONTROLLER_CONTRACT,
    build_implicit_actuators,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE_STAGE = ROOT / "isaaclab_doll_handoff_scene/generated/doll_handoff_g1_model_preview.usda"
HARD_FREEZE = (
    ROOT
    / "outputs/common_g1_deployment_safety/nearest_bound_joint_position_v1/freeze_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/policy_b_isaac_validation/dex3_controller_characterization"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--repetitions", type=int, default=1)
parser.add_argument("--hold-frames", type=int, default=10)
parser.add_argument("--settle-seconds", type=float, default=0.2)
parser.add_argument(
    "--command-velocities-rad-s",
    type=float,
    nargs="+",
    default=[0.15, 0.5, 1.2],
)
parser.add_argument(
    "--inward-offsets-rad",
    type=float,
    nargs="+",
    default=[0.005, 0.001, 0.0002, 0.00005, 0.0],
)
parser.add_argument("--interior-start-offset-rad", type=float, default=0.2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.repetitions < 1 or args.hold_frames < 4:
    parser.error("repetitions>=1 and hold-frames>=4 are required")
if any(value <= 0 for value in args.command_velocities_rad_s):
    parser.error("all command velocities must be positive")
if any(value < 0 for value in args.inward_offsets_rad):
    parser.error("all inward offsets must be non-negative")
if sorted(args.inward_offsets_rad, reverse=True) != list(args.inward_offsets_rad):
    parser.error("inward offsets must be supplied in progressively decreasing order")
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: 0.0 for key in ("p50", "p90", "p95", "p99", "p99_9")}
    return {
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)),
    }


def ceil_quantum(value: float, quantum: float) -> float:
    return float(math.ceil(value / quantum - 1e-12) * quantum)


def main() -> int:
    import carb
    import omni.usd
    import torch
    from pxr import UsdPhysics
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hard = json.loads(HARD_FREEZE.read_text(encoding="utf-8"))
    if hard.get("status") != "FROZEN_COMMON_DEPLOYMENT_SAFETY_ADAPTER":
        raise RuntimeError("hard-limit projection freeze is not authoritative")
    names = list(hard["joint_names"])
    rows = list(hard["joints"])
    if len(names) != 28 or [row["joint_name"] for row in rows] != names:
        raise RuntimeError("hard-limit freeze joint order is malformed")
    dex_indices = [index for index, row in enumerate(rows) if row["group"] == "dex3"]
    if dex_indices != list(range(14, 28)):
        raise RuntimeError("expected exactly 14 Dex3 policy channels after the 14 arms")
    lower = np.asarray([row["effective_lower_rad"] for row in rows], dtype=np.float64)
    upper = np.asarray([row["effective_upper_rad"] for row in rows], dtype=np.float64)

    expected_substeps = int(round((1.0 / CONTROL_FPS) / PHYSICS_DT))
    if expected_substeps != CONTROLLER_CONTRACT["simulation"]["physics_substeps_per_control_frame"]:
        raise RuntimeError("shared controller contract has inconsistent timebase")

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/hydra/readTransformsFromFabricInRenderDelegate", True)
    if not omni.usd.get_context().open_stage(str(SCENE_STAGE)):
        raise RuntimeError(f"failed to open {SCENE_STAGE}")
    stage = omni.usd.get_context().get_stage()
    disabled_collision_prims = 0
    disabled_rigid_bodies = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/DollHandoffEnvironment"):
            continue
        collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        if collision:
            collision.CreateCollisionEnabledAttr(False)
            disabled_collision_prims += 1
        rigid = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        if rigid:
            rigid.CreateRigidBodyEnabledAttr(False)
            disabled_rigid_bodies += 1

    sim = SimulationContext(
        SimulationCfg(
            dt=PHYSICS_DT,
            device=CONTROLLER_CONTRACT["simulation"]["device"],
            use_fabric=CONTROLLER_CONTRACT["simulation"]["use_fabric"],
        )
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
        raise RuntimeError(f"named Isaac policy mapping failed; missing={missing}")

    target = robot.data.default_joint_pos.torch.clone().to(robot.device, dtype=torch.float32)
    zero = torch.zeros_like(target)
    base = target.clone()
    midpoint = (lower + upper) / 2.0
    base[0, ids] = torch.as_tensor(midpoint, device=robot.device, dtype=torch.float32)

    def update_robot() -> None:
        robot.update(PHYSICS_DT)

    def step_control(command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target[0, ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        for _ in range(expected_substeps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            update_robot()
        measured = robot.data.joint_pos.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        velocity = robot.data.joint_vel.torch[0, ids].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(measured).all() or not np.isfinite(velocity).all():
            raise RuntimeError("non-finite measured state during characterization")
        return measured, velocity

    def reset_at(command: np.ndarray) -> None:
        state = base.clone()
        state[0, ids] = torch.as_tensor(command, device=robot.device, dtype=torch.float32)
        target.copy_(state)
        robot.write_joint_state_to_sim(state, zero)
        sim.forward()
        robot.update(0.0)
        settle_steps = max(1, int(round(args.settle_seconds / PHYSICS_DT)))
        for _ in range(settle_steps):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            update_robot()

    phase_codes = {
        "toward_ramp": 0,
        "hold_after_toward": 1,
        "away_ramp": 2,
        "hold_after_away": 3,
    }
    traces: dict[str, list[Any]] = {
        key: []
        for key in (
            "test_id",
            "repetition",
            "policy_index",
            "bound_code",
            "phase_code",
            "frame_in_test",
            "requested_offset_rad",
            "requested_velocity_rad_s",
            "commanded_rad",
            "measured_rad",
            "measured_velocity_rad_s",
            "command_velocity_rad_s",
            "hard_clearance_rad",
            "hard_penetration_rad",
            "outward_tracking_error_rad",
        )
    }
    test_reports: list[dict[str, Any]] = []
    test_id = 0

    for policy_index in dex_indices:
        joint = names[policy_index]
        for bound_code, bound_name in enumerate(("lower", "upper")):
            sign = 1.0 if bound_name == "lower" else -1.0
            hard_bound = lower[policy_index] if bound_name == "lower" else upper[policy_index]
            interior = hard_bound + sign * args.interior_start_offset_rad
            if not lower[policy_index] < interior < upper[policy_index]:
                raise RuntimeError(f"interior characterization start is invalid for {joint} {bound_name}")
            for speed in args.command_velocities_rad_s:
                print(
                    f"[Dex3Characterization] joint={joint} bound={bound_name} speed={speed:.3f}",
                    flush=True,
                )
                for repetition in range(args.repetitions):
                    command = midpoint.copy()
                    command[policy_index] = interior
                    reset_at(command)
                    previous_command = float(interior)
                    sequences = [
                        ("toward", list(args.inward_offsets_rad)),
                        ("away", list(reversed(args.inward_offsets_rad[:-1]))),
                    ]
                    for direction, requested_offsets in sequences:
                        ramp_phase = f"{direction}_ramp"
                        hold_phase = f"hold_after_{direction}"
                        for requested_offset in requested_offsets:
                            target_value = hard_bound + sign * requested_offset
                            if target_value < lower[policy_index] or target_value > upper[policy_index]:
                                raise RuntimeError(
                                    "requested characterization target is outside hard interval"
                                )
                            start_global = len(traces["test_id"])
                            frame_in_test = 0
                            distance = abs(target_value - previous_command)
                            ramp_frames = max(
                                1, int(math.ceil(distance * CONTROL_FPS / speed))
                            )
                            ramp_values = np.linspace(
                                previous_command, target_value, ramp_frames + 1
                            )[1:]
                            schedule = [
                                (ramp_phase, ramp_values),
                                (hold_phase, np.full(args.hold_frames, target_value)),
                            ]
                            for phase_name, values in schedule:
                                for scalar in values:
                                    scalar32 = float(np.float32(scalar))
                                    command[policy_index] = scalar32
                                    measured, measured_velocity = step_control(command)
                                    command_velocity = (
                                        scalar32 - previous_command
                                    ) * CONTROL_FPS
                                    previous_command = scalar32
                                    if bound_name == "lower":
                                        clearance = measured[policy_index] - hard_bound
                                        penetration = max(
                                            hard_bound - measured[policy_index], 0.0
                                        )
                                        outward_error = max(
                                            scalar32 - measured[policy_index], 0.0
                                        )
                                    else:
                                        clearance = hard_bound - measured[policy_index]
                                        penetration = max(
                                            measured[policy_index] - hard_bound, 0.0
                                        )
                                        outward_error = max(
                                            measured[policy_index] - scalar32, 0.0
                                        )
                                    values_to_add = {
                                        "test_id": test_id,
                                        "repetition": repetition,
                                        "policy_index": policy_index,
                                        "bound_code": bound_code,
                                        "phase_code": phase_codes[phase_name],
                                        "frame_in_test": frame_in_test,
                                        "requested_offset_rad": requested_offset,
                                        "requested_velocity_rad_s": speed,
                                        "commanded_rad": scalar32,
                                        "measured_rad": measured[policy_index],
                                        "measured_velocity_rad_s": measured_velocity[
                                            policy_index
                                        ],
                                        "command_velocity_rad_s": command_velocity,
                                        "hard_clearance_rad": clearance,
                                        "hard_penetration_rad": penetration,
                                        "outward_tracking_error_rad": outward_error,
                                    }
                                    for key, value in values_to_add.items():
                                        traces[key].append(value)
                                    frame_in_test += 1

                            stop_global = len(traces["test_id"])
                            sl = slice(start_global, stop_global)
                            phases = np.asarray(
                                traces["phase_code"][sl], dtype=np.int8
                            )
                            measured_values = np.asarray(
                                traces["measured_rad"][sl], dtype=np.float64
                            )
                            commanded_values = np.asarray(
                                traces["commanded_rad"][sl], dtype=np.float64
                            )
                            penetration_values = np.asarray(
                                traces["hard_penetration_rad"][sl], dtype=np.float64
                            )
                            outward_values = np.asarray(
                                traces["outward_tracking_error_rad"][sl], dtype=np.float64
                            )
                            hold_indices = np.flatnonzero(
                                phases == phase_codes[hold_phase]
                            )
                            steady_indices = hold_indices[len(hold_indices) // 2 :]
                            steady_error = (
                                measured_values[steady_indices]
                                - commanded_values[steady_indices]
                            )
                            positive_penetration = penetration_values[
                                penetration_values > 0.0
                            ]
                            test_reports.append(
                                {
                                    "test_id": test_id,
                                    "joint": joint,
                                    "policy_index": policy_index,
                                    "bound": bound_name,
                                    "approach_direction": direction,
                                    "repetition": repetition,
                                    "requested_offset_rad": requested_offset,
                                    "requested_velocity_rad_s": speed,
                                    "actual_max_command_velocity_rad_s": float(
                                        np.max(
                                            np.abs(
                                                np.asarray(
                                                    traces["command_velocity_rad_s"][sl]
                                                )
                                            )
                                        )
                                    ),
                                    "sample_count": stop_global - start_global,
                                    "commanded_min_rad": float(
                                        np.min(commanded_values)
                                    ),
                                    "commanded_max_rad": float(
                                        np.max(commanded_values)
                                    ),
                                    "measured_min_rad": float(np.min(measured_values)),
                                    "measured_max_rad": float(np.max(measured_values)),
                                    "maximum_hard_bound_penetration_rad": float(
                                        np.max(penetration_values)
                                    ),
                                    "mean_hard_bound_penetration_all_samples_rad": float(
                                        np.mean(penetration_values)
                                    ),
                                    "hard_bound_penetration_percentiles_all_samples_rad": percentiles(
                                        penetration_values
                                    ),
                                    "hard_bound_penetration_positive_sample_count": int(
                                        positive_penetration.size
                                    ),
                                    "hard_bound_penetration_percentiles_positive_samples_rad": percentiles(
                                        positive_penetration
                                    ),
                                    "maximum_outward_tracking_error_rad": float(
                                        np.max(outward_values)
                                    ),
                                    "maximum_ramp_outward_tracking_error_rad": float(
                                        np.max(
                                            outward_values[
                                                phases == phase_codes[ramp_phase]
                                            ]
                                        )
                                    ),
                                    "maximum_hold_outward_tracking_error_rad": float(
                                        np.max(
                                            outward_values[
                                                phases == phase_codes[hold_phase]
                                            ]
                                        )
                                    ),
                                    "steady_state_error_mean_rad": float(
                                        np.mean(steady_error)
                                    ),
                                    "steady_state_error_rmse_rad": float(
                                        np.sqrt(np.mean(np.square(steady_error)))
                                    ),
                                    "steady_state_error_max_abs_rad": float(
                                        np.max(np.abs(steady_error))
                                    ),
                                }
                            )
                            test_id += 1

    arrays = {
        key: np.asarray(
            values,
            dtype=(
                np.int32
                if key in {"test_id", "policy_index", "frame_in_test"}
                else np.int16
                if key == "repetition"
                else np.int8
                if key in {"bound_code", "phase_code"}
                else np.float64
            ),
        )
        for key, values in traces.items()
    }
    atomic_npz(
        output / "raw_tracking_trace.npz",
        **arrays,
        joint_names=np.asarray(names),
        phase_names=np.asarray(
            ["toward_ramp", "hold_after_toward", "away_ramp", "hold_after_away"]
        ),
        bound_names=np.asarray(["lower", "upper"]),
        policy_invoked=np.asarray(False),
        task_objects_physical=np.asarray(False),
        real_robot_commands=np.asarray(False),
    )

    derivation_quantum = 1e-5
    safety_factor = 2.0
    per_joint: list[dict[str, Any]] = []
    for policy_index in dex_indices:
        joint_row: dict[str, Any] = {
            "joint": names[policy_index],
            "policy_index": policy_index,
            "hard_lower_rad": float(lower[policy_index]),
            "hard_upper_rad": float(upper[policy_index]),
        }
        for bound_code, bound_name in enumerate(("lower", "upper")):
            mask = (arrays["policy_index"] == policy_index) & (
                arrays["bound_code"] == bound_code
            )
            penetration = arrays["hard_penetration_rad"][mask]
            positive = penetration[penetration > 0]
            near = mask & (arrays["requested_offset_rad"] <= 0.0050000001)
            toward_or_hold = near & np.isin(
                arrays["phase_code"],
                [phase_codes["toward_ramp"], phase_codes["hold_after_toward"]],
            )
            observed_outward = float(
                np.max(arrays["outward_tracking_error_rad"][toward_or_hold])
            )
            derived_margin = ceil_quantum(
                max(derivation_quantum, safety_factor * observed_outward),
                derivation_quantum,
            )
            exact_bound = mask & (arrays["requested_offset_rad"] == 0.0)
            velocity_dependency = {}
            for speed in args.command_velocities_rad_s:
                speed_mask = mask & np.isclose(arrays["requested_velocity_rad_s"], speed)
                speed_penetration = arrays["hard_penetration_rad"][speed_mask]
                speed_outward = arrays["outward_tracking_error_rad"][speed_mask]
                velocity_dependency[f"{speed:.6f}"] = {
                    "sample_count": int(np.count_nonzero(speed_mask)),
                    "maximum_hard_bound_penetration_rad": float(
                        np.max(speed_penetration)
                    ),
                    "maximum_outward_tracking_error_rad": float(np.max(speed_outward)),
                }
            direction_dependency = {}
            for phase_name in (
                "toward_ramp",
                "hold_after_toward",
                "away_ramp",
                "hold_after_away",
            ):
                phase_mask = mask & (arrays["phase_code"] == phase_codes[phase_name])
                direction_dependency[phase_name] = {
                    "sample_count": int(np.count_nonzero(phase_mask)),
                    "maximum_hard_bound_penetration_rad": float(
                        np.max(arrays["hard_penetration_rad"][phase_mask])
                    ),
                    "maximum_outward_tracking_error_rad": float(
                        np.max(arrays["outward_tracking_error_rad"][phase_mask])
                    ),
                }
            bound_report = {
                "sample_count": int(np.count_nonzero(mask)),
                "penetrating_sample_count": int(positive.size),
                "penetrating_sample_percentage": float(100.0 * positive.size / penetration.size),
                "maximum_hard_bound_penetration_rad": float(np.max(penetration)),
                "mean_hard_bound_penetration_all_samples_rad": float(np.mean(penetration)),
                "mean_hard_bound_penetration_positive_samples_rad": float(np.mean(positive))
                if positive.size
                else 0.0,
                "hard_bound_penetration_percentiles_all_samples_rad": percentiles(penetration),
                "hard_bound_penetration_percentiles_positive_samples_rad": percentiles(positive),
                "maximum_exact_bound_penetration_rad": float(
                    np.max(arrays["hard_penetration_rad"][exact_bound])
                ),
                "maximum_near_bound_toward_or_hold_outward_tracking_error_rad": observed_outward,
                "derived_margin_rad": derived_margin,
                "velocity_dependency": velocity_dependency,
                "approach_phase_dependency": direction_dependency,
            }
            joint_row[bound_name] = bound_report
        joint_row["margin_lower_rad"] = joint_row["lower"]["derived_margin_rad"]
        joint_row["margin_upper_rad"] = joint_row["upper"]["derived_margin_rad"]
        joint_row["lower_safe_rad"] = float(
            np.float32(lower[policy_index] + joint_row["margin_lower_rad"])
        )
        joint_row["upper_safe_rad"] = float(
            np.float32(upper[policy_index] - joint_row["margin_upper_rad"])
        )
        if not joint_row["lower_safe_rad"] < joint_row["upper_safe_rad"]:
            raise RuntimeError(f"derived margin emptied interval for {names[policy_index]}")
        per_joint.append(joint_row)

    report = {
        "schema_version": "g1_dex3_isaac_joint_limit_tracking_characterization_v1",
        "status": "PASS_CHARACTERIZATION_COMPLETE",
        "scope": "task-independent no-object no-policy simulated Dex3 boundary tracking",
        "margin_label": "SIMULATION_CONTROLLER_MARGIN_ONLY",
        "policy_invoked": False,
        "dataset_accessed": False,
        "camera_created_or_moved": False,
        "task_objects": {
            "used": False,
            "collision_prims_disabled": disabled_collision_prims,
            "rigid_bodies_disabled": disabled_rigid_bodies,
        },
        "real_robot_commands": False,
        "real_hardware_calibration_required_before_transmission": True,
        "protocol": {
            "repetitions": args.repetitions,
            "hold_frames": args.hold_frames,
            "settle_seconds": args.settle_seconds,
            "command_velocities_rad_s": args.command_velocities_rad_s,
            "inward_offsets_rad": args.inward_offsets_rad,
            "interior_start_offset_rad": args.interior_start_offset_rad,
            "sweep_structure": (
                "one reset per joint/bound/velocity/repetition; continuous progressive "
                "toward sweep through all offsets followed by an away sweep"
            ),
            "control_fps": CONTROL_FPS,
            "physics_dt_s": PHYSICS_DT,
            "physics_substeps_per_control_frame": expected_substeps,
            "isolated_joint_testing": True,
            "other_dex3_joint_target": "midpoint of each active hard interval",
        },
        "margin_derivation": {
            "rule": (
                "For each joint and bound, take twice the worst measured outward tracking "
                "deviation across all toward/hold samples at commanded offsets <= 0.005 rad, "
                "then round outward to 1e-5 rad with a minimum of 1e-5 rad."
            ),
            "safety_factor": safety_factor,
            "rounding_quantum_rad": derivation_quantum,
            "selected_before_policy_rerun": True,
            "policy_success_or_task_success_used": False,
        },
        "controller_contract": CONTROLLER_CONTRACT,
        "controller_contract_source": {
            "path": str(ROOT / "tools/policy_b_isaac_control_contract.py"),
            "sha256": sha256_file(ROOT / "tools/policy_b_isaac_control_contract.py"),
        },
        "characterization_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "hard_limit_freeze": {
            "path": str(HARD_FREEZE),
            "sha256": sha256_file(HARD_FREEZE),
        },
        "scene_asset": {"path": str(SCENE_STAGE), "sha256": sha256_file(SCENE_STAGE)},
        "joint_names": names,
        "per_joint": per_joint,
        "test_count": len(test_reports),
        "sample_count": int(arrays["test_id"].size),
        "raw_trace": str(output / "raw_tracking_trace.npz"),
        "test_reports": test_reports,
    }
    atomic_json(output / "characterization.json", report)

    md = [
        "# Isaac Dex3 joint-limit tracking characterization",
        "",
        "Status: **PASS_CHARACTERIZATION_COMPLETE**",
        "",
        "Label: **SIMULATION_CONTROLLER_MARGIN_ONLY**",
        "",
        "No Policy B inference, Dataset-B access, camera mutation, task-object physics, or real-robot command path was used.",
        "",
        "| Joint | Lower max penetration | Lower outward error | Lower margin | Upper max penetration | Upper outward error | Upper margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_joint:
        md.append(
            "| {joint} | {lp:.9f} | {lo:.9f} | {lm:.9f} | {up:.9f} | {uo:.9f} | {um:.9f} |".format(
                joint=row["joint"],
                lp=row["lower"]["maximum_hard_bound_penetration_rad"],
                lo=row["lower"]["maximum_near_bound_toward_or_hold_outward_tracking_error_rad"],
                lm=row["margin_lower_rad"],
                up=row["upper"]["maximum_hard_bound_penetration_rad"],
                uo=row["upper"]["maximum_near_bound_toward_or_hold_outward_tracking_error_rad"],
                um=row["margin_upper_rad"],
            )
        )
    md.extend(
        [
            "",
            "Margins are twice the worst observed near-bound toward/hold outward deviation, rounded outward to 1e-5 rad. They are not selected using Policy-B pass/fail behavior.",
            "",
            "A separate real-hardware no-object boundary/tracking calibration is mandatory before any physical Dex3 command transmission.",
        ]
    )
    (output / "characterization.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "test_count": report["test_count"],
                "sample_count": report["sample_count"],
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
