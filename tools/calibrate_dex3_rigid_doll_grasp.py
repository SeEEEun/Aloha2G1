#!/usr/bin/env python3
"""Build and later execute fixed, policy-independent Dex3 rigid-doll tests.

CPU-safe commands (``validate-config``, ``build-primitives``, ``summarize``)
never import Isaac.  The explicit ``run-isaac`` subcommand launches the separate
Isaac runner and is intentionally not invoked by preparation or tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs/doll_handoff_rigid_proxy_v1.json"
DEFAULT_OUTPUT = ROOT / "outputs/paper_metrics/rigid_proxy"
ISAAC_RUNNER = ROOT / "tools/run_dex3_rigid_doll_grasp_isaac.py"
ISAAC_PYTHON = Path("/home/jbnu/miniconda3/envs/isaaclab6/bin/python")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != "doll_handoff_rigid_proxy_v1":
        raise ValueError("unexpected rigid-proxy schema")
    candidates = config["material_candidates"]
    if [row["name"] for row in candidates] != ["LOW", "MEDIUM", "HIGH"] or len(candidates) != 3:
        raise ValueError("the bounded calibration must contain exactly LOW/MEDIUM/HIGH")
    for row in candidates:
        static = float(row["static_friction"])
        dynamic = float(row["dynamic_friction"])
        if not 0.0 <= dynamic <= static:
            raise ValueError(f"invalid friction candidate: {row}")
    scene = read_json(Path(config["source_scene"]["layout"]))
    scene_usd_path = Path(config["source_scene"]["scene_usd"])
    preview_path = Path(config["source_scene"]["g1_preview_usd"])
    scene_usd = scene_usd_path.read_text(encoding="utf-8")
    preview_usd = preview_path.read_text(encoding="utf-8")
    source_scene_files_exist = all(
        Path(config["source_scene"][key]).is_file()
        for key in ("layout", "scene_usd", "g1_preview_usd")
    )
    radius = float(config["object"]["collision_dimensions_m"]["radius"])
    checks = {
        "mass_matches_scene": np.isclose(
            float(config["object"]["mass_kg"]), float(scene["doll"]["mass_kg"]), atol=1e-12
        ),
        "diameter_matches_scene": np.isclose(
            2.0 * radius, float(scene["doll"]["diameter_m"]), atol=1e-12
        ),
        "diameter_and_radius_are_consistent": np.isclose(
            float(config["object"]["collision_dimensions_m"]["diameter"]),
            2.0 * radius,
            atol=1e-12,
        ),
        "sphere_collision": config["object"]["collision_shape"] == "SPHERE",
        "three_material_candidates_only": len(candidates) == 3,
        "no_deformable": not bool(config["object"]["deformable"]),
        "no_magnet": not bool(config["object"]["magnetic"]),
        "no_attachment": not bool(config["object"]["constraint_attachment"])
        and not bool(config["object"]["weld_to_hand"]),
        "no_policy_logic": not bool(config["policy_specific_logic"]),
        "source_scene_files_exist": source_scene_files_exist,
        "usd_doll_is_rigid_body": 'def Xform "Doll" (' in scene_usd
        and 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]' in scene_usd,
        "usd_doll_mass_matches": f'float physics:mass = {float(config["object"]["mass_kg"]):.3f}'
        in scene_usd,
        "usd_body_is_collision_sphere": 'def Sphere "Body" (' in scene_usd
        and 'prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"]' in scene_usd,
        "usd_collision_radius_matches": f'double radius = {radius:g}' in scene_usd,
        "usd_visual_nubs_are_not_collision_shapes": 'def Sphere "Nub1" (\n            prepend apiSchemas = ["MaterialBindingAPI"]'
        in scene_usd
        and 'def Sphere "Nub2" (\n            prepend apiSchemas = ["MaterialBindingAPI"]'
        in scene_usd,
        "g1_preview_references_scene": "prepend references = @doll_handoff_scene.usda@"
        in preview_usd,
        "thirty_hz": np.isclose(float(config["timing"]["control_fps_hz"]), 30.0),
    }
    if not all(checks.values()):
        raise RuntimeError(f"rigid-proxy static validation failed: {checks}")
    return {
        "schema_version": "doll_handoff_rigid_proxy_static_validation_v1",
        "status": "PASS",
        "config": str(path.resolve()),
        "config_sha256": sha256_file(path),
        "checks": checks,
        "freeze_status": config["status"],
        "isaac_executed": False,
        "source_file_sha256": {
            "layout": sha256_file(Path(config["source_scene"]["layout"])),
            "scene_usd": sha256_file(scene_usd_path),
            "g1_preview_usd": sha256_file(preview_path),
        },
    }


def minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(alpha, dtype=np.float64), 0.0, 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _segment(start: np.ndarray, end: np.ndarray, frames: int) -> np.ndarray:
    alpha = minimum_jerk(np.linspace(0.0, 1.0, max(2, int(frames)), dtype=np.float64))
    return start[None, :] + alpha[:, None] * (end - start)[None, :]


def build_primitive(side: str, config_path: Path, output_dir: Path) -> dict[str, Any]:
    if side not in ("left", "right"):
        raise ValueError(side)
    validation = validate_config(config_path)
    config = read_json(config_path)
    # These imports are CPU MuJoCo/SciPy only.  No Isaac, torch, or policy is loaded.
    from tools.doll_handoff_retargeting.common import load_common_config, load_json, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.doll_handoff_retargeting.retarget import SharedTemporalIK
    from tools.evaluation.contracts import authoritative_joint_ranges

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal, nominal_report = g1.derive_task_ready_nominal(common, scene)
    proposed = load_json(ROOT / "configs/doll_handoff_retargeting/proposed_config.template.json")
    primitives = g1.derive_hand_primitives(scene, proposed, nominal)
    solver = SharedTemporalIK(common, g1, nominal, natural_arm_enabled=True)
    fps = float(config["timing"]["control_fps_hz"])
    durations = config["calibration"]["stage_durations_s"]
    stage_order = [
        "OPEN_AROUND_DOLL",
        "CLOSE_GRASP",
        "HOLD_ON_TABLE",
        "LIFT",
        "HOLD_ELEVATED",
        "LOWER",
        "RELEASE",
        "RETRACT",
    ]
    object_center = np.asarray(
        config["calibration"]["object_reset_center_world_xyz_m"], dtype=np.float64
    )
    offsets = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in config["calibration"][
            "active_whole_hand_target_offsets_from_object_center_m"
        ].items()
    }
    g1.assign(nominal, primitives["states"]["left"]["OPEN"], primitives["states"]["right"]["OPEN"])
    static_tool = primitives["wrist_to_grasp_frame"]
    nominal_tool_model = {
        arm: g1.static_tool_pose_state(arm, static_tool[arm])[0].copy()
        for arm in ("left", "right")
    }
    nominal_tool_rotation = {
        arm: g1.static_tool_pose_state(arm, static_tool[arm])[1].copy()
        for arm in ("left", "right")
    }
    active_world = {
        "OPEN_AROUND_DOLL": object_center + offsets["OPEN_AROUND_DOLL"],
        "GRASP": object_center + offsets["GRASP"],
        "LIFT": object_center + offsets["LIFT"],
        "LOWER": object_center + offsets["LOWER"],
        "RETRACT": object_center + offsets["RETRACT"],
    }
    active_model = {key: g1.world_to_model_position(value) for key, value in active_world.items()}
    stage_endpoints = {
        "OPEN_AROUND_DOLL": active_model["OPEN_AROUND_DOLL"],
        "CLOSE_GRASP": active_model["GRASP"],
        "HOLD_ON_TABLE": active_model["GRASP"],
        "LIFT": active_model["LIFT"],
        "HOLD_ELEVATED": active_model["LIFT"],
        "LOWER": active_model["LOWER"],
        "RELEASE": active_model["LOWER"],
        "RETRACT": active_model["RETRACT"],
    }
    position_parts: list[np.ndarray] = []
    hand_parts: list[np.ndarray] = []
    labels: list[str] = []
    # The declared calibration protocol starts with an already-open hand
    # around the object.  Starting from the remote nominal tool position would
    # sweep an open hand through the unconstrained sphere before closure.
    current_position = active_model["OPEN_AROUND_DOLL"]
    current_hand = primitives["states"][side]["OPEN"]
    grasp_hand = primitives["states"][side]["GRASP"]
    for stage in stage_order:
        frames = max(2, int(round(float(durations[stage]) * fps)))
        endpoint = stage_endpoints[stage]
        position = _segment(current_position, endpoint, frames)
        if stage == "CLOSE_GRASP":
            hand = _segment(current_hand, grasp_hand, frames)
        elif stage == "RELEASE":
            hand = _segment(current_hand, primitives["states"][side]["OPEN"], frames)
        else:
            hand = np.repeat(current_hand[None, :], frames, axis=0)
        if position_parts:
            position = position[1:]
            hand = hand[1:]
        position_parts.append(position)
        hand_parts.append(hand)
        labels.extend([stage] * len(position))
        current_position = endpoint
        current_hand = hand[-1]
    active_position = np.concatenate(position_parts)
    active_hand = np.concatenate(hand_parts)
    count = len(active_position)
    target_position = {
        arm: (
            active_position
            if arm == side
            else np.repeat(nominal_tool_model[arm][None, :], count, axis=0)
        )
        for arm in ("left", "right")
    }
    target_rotation = {
        arm: np.repeat(nominal_tool_rotation[arm][None, :, :], count, axis=0)
        for arm in ("left", "right")
    }
    targets = {
        # SharedTemporalIK takes its sequence length from the common wrist key
        # even when the direct static whole-hand objective is selected.
        "left_wrist_position": target_position["left"],
        "right_wrist_position": target_position["right"],
        "left_wrist_rotation": target_rotation["left"],
        "right_wrist_rotation": target_rotation["right"],
        "left_tool_position_model": target_position["left"],
        "right_tool_position_model": target_position["right"],
        "left_tool_rotation_model": target_rotation["left"],
        "right_tool_rotation_model": target_rotation["right"],
        "static_wrist_to_tool": static_tool,
        "direct_static_grasp_frame_position_objective": True,
        "task_orientation_constrained": False,
        "orientation_gauge_weight_multiplier": np.zeros(count),
        "explicit_bimanual_objective": False,
    }
    solution = solver.solve(targets)
    arm_q = np.asarray(solution["q"], dtype=np.float64)
    left_hand = (
        active_hand
        if side == "left"
        else np.repeat(primitives["states"]["left"]["OPEN"][None, :], count, axis=0)
    )
    right_hand = (
        active_hand
        if side == "right"
        else np.repeat(primitives["states"]["right"]["OPEN"][None, :], count, axis=0)
    )
    canonical_names, _ = authoritative_joint_ranges()
    source_names = list(g1.arm_joint_names) + list(g1.hand_joint_names["left"]) + list(g1.hand_joint_names["right"])
    source_q = np.column_stack((arm_q, left_hand, right_hand))
    lookup = {name: index for index, name in enumerate(source_names)}
    if set(canonical_names) != set(source_names):
        raise RuntimeError("calibration named joints differ from canonical 28-D contract")
    full_q = source_q[:, [lookup[name] for name in canonical_names]]
    achieved_static_objective = []
    achieved_physical = []
    clearances = []
    for frame in range(count):
        g1.assign(arm_q[frame], left_hand[frame], right_hand[frame])
        achieved_static_objective.append(
            g1.static_tool_pose_state(side, static_tool[side])[0]
        )
        achieved_physical.append(g1.whole_hand_grasp_pose(side)[:3, 3])
        state = g1.posture_clearance_state(arm_q[frame])
        clearances.append(min(state["TORSO"]["minimum_distance_m"], state["CROSS_ARM"]["minimum_distance_m"]))
    achieved_static_objective = np.asarray(achieved_static_objective)
    achieved_physical = np.asarray(achieved_physical)
    static_objective_error = np.linalg.norm(
        achieved_static_objective - active_position, axis=1
    )
    closed_stages = ("HOLD_ON_TABLE", "LIFT", "HOLD_ELEVATED", "LOWER")
    closed_mask = np.isin(np.asarray(labels), closed_stages)
    physical_closed_error = np.linalg.norm(
        achieved_physical[closed_mask] - active_position[closed_mask], axis=1
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{side}_fixed_grasp_primitive.npz"
    temporary = npz_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            source_episode_id=np.asarray(f"RIGID_DOLL_CALIBRATION_{side.upper()}"),
            fps=np.asarray(fps),
            frame_index=np.arange(count, dtype=np.int64),
            timestamp_s=np.arange(count, dtype=np.float64) / fps,
            stage=np.asarray(labels),
            joint_names=np.asarray(canonical_names),
            commanded_q_rad=full_q.astype(np.float32),
            active_side=np.asarray(side),
            target_whole_hand_position_model_m=active_position.astype(np.float32),
            achieved_whole_hand_position_model_m=achieved_physical.astype(np.float32),
            achieved_static_wrist_attached_objective_position_model_m=achieved_static_objective.astype(
                np.float32
            ),
            target_whole_hand_position_world_m=g1.model_to_world_position(active_position).astype(np.float32),
            active_hand_open_q_rad=primitives["states"][side]["OPEN"].astype(np.float32),
            active_hand_grasp_q_rad=primitives["states"][side]["GRASP"].astype(np.float32),
            policy_independent=np.asarray(True),
            learned_policy_used=np.asarray(False),
            config_sha256=np.asarray(validation["config_sha256"]),
        )
    os.replace(temporary, npz_path)
    static_thresholds = config["calibration"]["static_validation_thresholds"]
    joint_limit_violation_count = int(
        np.count_nonzero(
            (arm_q < g1.arm_limits[:, 0] - 1e-9)
            | (arm_q > g1.arm_limits[:, 1] + 1e-9)
        )
    )
    report = {
        "schema_version": "doll_handoff_fixed_grasp_primitive_v1",
        "status": (
            "READY"
            if max(physical_closed_error)
            <= float(static_thresholds["maximum_whole_hand_target_error_m"])
            and min(clearances)
            >= float(static_thresholds["minimum_generic_torso_or_cross_arm_clearance_m"])
            and joint_limit_violation_count
            <= int(static_thresholds["joint_limit_violation_count"])
            else "STATIC_VALIDATION_FAIL"
        ),
        "side": side,
        "config": str(config_path.resolve()),
        "config_sha256": validation["config_sha256"],
        "trajectory": str(npz_path.resolve()),
        "trajectory_sha256": sha256_file(npz_path),
        "frames": count,
        "duration_s": (count - 1) / fps,
        "stage_order": stage_order,
        "max_tool_position_error_m": float(np.max(physical_closed_error)),
        "mean_tool_position_error_m": float(np.mean(physical_closed_error)),
        "tool_position_error_scope": "actual physical three-pad circumcenter during closed HOLD_ON_TABLE/LIFT/HOLD_ELEVATED/LOWER stages",
        "physical_error_evaluated_stages": list(closed_stages),
        "max_static_wrist_attached_objective_error_m": float(
            np.max(static_objective_error)
        ),
        "mean_static_wrist_attached_objective_error_m": float(
            np.mean(static_objective_error)
        ),
        "minimum_generic_torso_or_cross_arm_clearance_m": float(np.min(clearances)),
        "joint_limit_violation_count": joint_limit_violation_count,
        "static_validation_thresholds": static_thresholds,
        "solver_backend": solution["backend"],
        "common_natural_arm_enabled": solution["natural_arm_enabled"],
        "nominal_posture": nominal_report,
        "policy_independent": True,
        "learned_policy_used": False,
        "isaac_executed": False,
    }
    atomic_json(output_dir / f"{side}_fixed_grasp_primitive.json", report)
    if report["status"] != "READY":
        raise RuntimeError(f"{side} primitive failed static validation: {report}")
    return report


def summarize(results: list[Path], output: Path) -> dict[str, Any]:
    rows = []
    for path in results:
        row = read_json(path)
        rows.append(
            {
                **row,
                "result_path": str(path.resolve()),
                "result_sha256": sha256_file(path),
            }
        )
    keys = {(str(row["side"]), str(row["material_candidate"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate side/material calibration result")
    passes = {
        material: all(
            any(
                row["side"] == side
                and row["material_candidate"] == material
                and bool(row.get(f"{side.upper()}_LIFT_PASS", False))
                for row in rows
            )
            for side in ("left", "right")
        )
        for material in ("LOW", "MEDIUM", "HIGH")
    }
    recommended = next((name for name in ("LOW", "MEDIUM", "HIGH") if passes[name]), None)
    report = {
        "schema_version": "doll_handoff_grasp_calibration_summary_v1",
        "status": "CALIBRATION_COMPLETE" if len(rows) == 6 else "PARTIAL",
        "expected_grid": [[side, material] for material in ("LOW", "MEDIUM", "HIGH") for side in ("left", "right")],
        "results": rows,
        "candidate_passes_both_hands": passes,
        "predeclared_selection_rule": "LOWEST_FRICTION_CANDIDATE_WITH_BILATERAL_ARTIFACT_FREE_LIFT_PASS",
        "recommended_candidate_under_predeclared_rule": recommended,
        "selection_performed": False,
        "policy_results_consulted": False,
    }
    atomic_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-config")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "static_validation.json")
    build = sub.add_parser("build-primitives")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "primitives")
    run = sub.add_parser("run-isaac")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--side", choices=("left", "right"), required=True)
    run.add_argument("--material", choices=("LOW", "MEDIUM", "HIGH"), required=True)
    run.add_argument("--primitive", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--headless", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--results", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "calibration_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "validate-config":
        report = validate_config(args.config)
        atomic_json(args.output, report)
    elif args.command == "build-primitives":
        reports = [build_primitive(side, args.config, args.output_dir) for side in ("left", "right")]
        atomic_json(args.output_dir / "primitive_manifest.json", {"status": "READY", "primitives": reports})
    elif args.command == "summarize":
        summarize(args.results, args.output)
    else:
        primitive = args.primitive or DEFAULT_OUTPUT / "primitives" / f"{args.side}_fixed_grasp_primitive.npz"
        command = [
            str(ISAAC_PYTHON),
            str(ISAAC_RUNNER),
            "--config",
            str(args.config.resolve()),
            "--side",
            args.side,
            "--material",
            args.material,
            "--primitive",
            str(primitive.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
        ]
        if args.headless:
            command.append("--headless")
        # This branch is an explicit future GPU/Isaac operation.  CPU preparation
        # never calls it, and there is no policy/checkpoint argument.
        raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
