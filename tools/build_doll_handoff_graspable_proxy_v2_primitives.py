#!/usr/bin/env python3
"""Build the fixed mirrored arm placement/lift paths for doll proxy v2.

This is a CPU MuJoCo/IK preparation command.  It does not import Isaac or read
any dataset, policy, or checkpoint.  One declared whole-hand center and its
vertical 5 cm offset are solved once and reused for every geometry/profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def minimum_jerk(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, count, dtype=np.float64)
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start[None] + blend[:, None] * (end - start)[None]


def build_side(side: str, config: dict[str, Any], config_path: Path, output: Path) -> dict[str, Any]:
    from tools.doll_handoff_retargeting.common import load_common_config, load_json, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.evaluation.contracts import authoritative_joint_ranges

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal, nominal_report = g1.derive_task_ready_nominal(common, scene)
    proposed = load_json(ROOT / "configs/doll_handoff_retargeting/proposed_config.template.json")
    primitives = g1.derive_hand_primitives(scene, proposed, nominal)
    # Config hand vectors use the authoritative [thumb, middle, index] order;
    # G1Kinematics exposes [thumb, index, middle], so the mapping is named rather
    # than positional.  The P1 wrist-attached frame is then oriented so the
    # thumb-to-opposition closing direction and the index-to-middle spread both
    # lie in the horizontal plane.  This is one mirrored construction derived
    # from digit topology, not a side- or geometry-specific physics fit.
    canonical_names, _ = authoritative_joint_ranges()
    power: dict[str, np.ndarray] = {}
    reference_profile = str(
        config["scripted_arm_placement"].get(
            "arm_placement_reference_hand_profile", "P1"
        )
    )
    for hand_side, canonical_slice in (("left", slice(14, 21)), ("right", slice(21, 28))):
        by_name = dict(
            zip(
                canonical_names[canonical_slice],
                config["hand_states"][hand_side][f"POWER_GRASP_{reference_profile}"],
                strict=True,
            )
        )
        power[hand_side] = np.asarray(
            [by_name[name] for name in g1.hand_joint_names[hand_side]], dtype=np.float64
        )
    g1.assign(nominal, power["left"], power["right"])
    static_tool = {
        arm: np.linalg.inv(g1.wrist_pose(arm)) @ g1.whole_hand_grasp_pose(arm)
        for arm in ("left", "right")
    }
    origin_model = g1.world_to_model_position(np.zeros(3, dtype=np.float64))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3, dtype=np.float64)[axis])
            - g1.model_to_world_position(np.zeros(3, dtype=np.float64))
            for axis in range(3)
        ]
    )
    if not np.allclose(world_from_model.T @ world_from_model, np.eye(3), atol=1.0e-10):
        raise RuntimeError("model/world rotation is not orthonormal")
    desired_rotation_model: dict[str, np.ndarray] = {}
    closing_world: dict[str, np.ndarray] = {}
    explicit_rotations = config["scripted_arm_placement"].get(
        "desired_whole_hand_rotation_world_by_side"
    )
    for arm in ("left", "right"):
        current_world = world_from_model @ g1.whole_hand_grasp_pose(arm)[:3, :3]
        closing = current_world[:, 0].copy()
        closing[2] = 0.0
        closing /= np.linalg.norm(closing)
        spread = current_world[:, 2].copy()
        spread[2] = 0.0
        spread -= closing * float(spread @ closing)
        spread /= np.linalg.norm(spread)
        transverse = np.cross(spread, closing)
        horizontal_world = np.column_stack((closing, transverse, spread))
        if explicit_rotations is not None:
            # Optional task-independent placement contract used by isolated hand
            # mechanics diagnostics.  Matrix columns are the desired whole-hand
            # X/Y/Z axes in world coordinates.  This places an enclosure; it
            # does not solve to object-surface or fingertip targets.
            desired_world = np.asarray(explicit_rotations[arm], dtype=np.float64)
        else:
            angle = np.deg2rad(
                float(
                    config["scripted_arm_placement"]["orientation_closing_axis_roll_deg"][arm]
                )
            )
            closing_axis_rotation = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(angle), -np.sin(angle)],
                    [0.0, np.sin(angle), np.cos(angle)],
                ],
                dtype=np.float64,
            )
            desired_world = horizontal_world @ closing_axis_rotation
        if not np.allclose(desired_world.T @ desired_world, np.eye(3), atol=1.0e-10):
            raise RuntimeError("desired whole-hand rotation is not orthonormal")
        if not np.isclose(np.linalg.det(desired_world), 1.0, atol=1.0e-10):
            raise RuntimeError("invalid desired whole-hand rotation")
        desired_rotation_model[arm] = world_from_model.T @ desired_world
        closing_world[arm] = closing
    nominal_position = {
        arm: g1.static_tool_pose_state(arm, static_tool[arm])[0].copy()
        for arm in ("left", "right")
    }
    nominal_rotation = {
        arm: g1.static_tool_pose_state(arm, static_tool[arm])[1].copy()
        for arm in ("left", "right")
    }
    placement = config["scripted_arm_placement"]
    if "whole_hand_center_world_xyz_m_by_side" in placement:
        start_world = np.asarray(
            placement["whole_hand_center_world_xyz_m_by_side"][side],
            dtype=np.float64,
        )
    else:
        start_world = np.asarray(
            placement["whole_hand_center_world_xyz_m"], dtype=np.float64
        )
    end_world = start_world + np.asarray(placement["lift_offset_world_xyz_m"], dtype=np.float64)
    fps = float(config["timing"]["control_fps_hz"])
    approach_count = max(
        2, int(round(float(config["timing"]["power_close_s"]) * fps))
    )
    lift_count = max(
        2, int(round(float(config["timing"]["lift_transition_s"]) * fps))
    )
    start_model = g1.world_to_model_position(start_world)
    end_model = g1.world_to_model_position(end_world)
    historical_path = Path(config["historical_clear_open_orientation_sources"][side])
    with np.load(historical_path, allow_pickle=False) as historical:
        historical_stages = historical["stage"].astype(str)
        historical_q = np.asarray(
            historical["commanded_q_rad"][
                np.flatnonzero(historical_stages == "HOLD_ON_TABLE")[-1]
            ],
            dtype=np.float64,
        )
    lateral_approach = np.asarray([1.0 if side == "left" else -1.0, 0.0, 0.0])
    pregrasp_world = start_world - float(placement["pregrasp_retreat_m"]) * lateral_approach
    pregrasp_model = g1.world_to_model_position(pregrasp_world)
    approach_position = minimum_jerk(pregrasp_model, start_model, approach_count)
    lift_position = minimum_jerk(start_model, end_model, lift_count)
    active_position = np.concatenate(
        (approach_position, lift_position[1:]),
        axis=0,
    )
    active_rotation = np.repeat(
        desired_rotation_model[side][None], len(active_position), axis=0
    )
    count = len(active_position)
    target_position = {
        arm: active_position
        if arm == side
        else np.repeat(nominal_position[arm][None], count, axis=0)
        for arm in ("left", "right")
    }
    target_rotation = {
        arm: np.repeat(
            nominal_rotation[arm][None], count, axis=0
        )
        for arm in ("left", "right")
    }
    target_rotation[side] = active_rotation
    block = slice(0, 7) if side == "left" else slice(7, 14)
    arm_q = np.repeat(nominal[None], count, axis=0)
    arm_q[0] = historical_q[:14]
    declared_seeds = placement.get("arm_ik_seed_q_rad_by_side", {})
    previous = np.asarray(
        declared_seeds.get(side, historical_q[block]), dtype=np.float64
    ).copy()
    if previous.shape != (7,):
        raise RuntimeError(f"{side} arm IK seed must be a named seven-joint vector")
    lower = g1.arm_limits[block, 0] + 1.0e-7
    upper = g1.arm_limits[block, 1] - 1.0e-7
    solve_rows: list[dict[str, Any]] = []
    position_weight = float(config["scripted_arm_placement"]["scripted_ik_position_weight"])
    orientation_weight = float(
        config["scripted_arm_placement"]["scripted_ik_orientation_weight"]
    )
    # These declared values are residual scales for a deterministic 7-DoF
    # active-arm pose solve, not learned-policy or controller gains.
    position_scale = 100.0 * position_weight / 3.0
    orientation_scale = orientation_weight / 3.0
    continuity_scale = float(
        config["scripted_arm_placement"]["scripted_ik_frame_continuity_weight"]
    )
    for frame in range(count):
        target_p = target_position[side][frame]
        target_r = target_rotation[side][frame]
        fixed = nominal.copy()

        def residual(active_q: np.ndarray) -> np.ndarray:
            full = fixed.copy()
            full[block] = active_q
            g1.assign(full, power["left"], power["right"])
            current_p, current_r, _, _ = g1.static_tool_pose_state(side, static_tool[side])
            orientation_error = Rotation.from_matrix(current_r.T @ target_r).as_rotvec()
            return np.r_[
                position_scale * (current_p - target_p),
                orientation_scale * orientation_error,
                continuity_scale * (active_q - previous),
            ]

        solution = least_squares(
            residual,
            np.clip(previous, lower, upper),
            bounds=(lower, upper),
            max_nfev=300,
            xtol=1.0e-11,
            ftol=1.0e-11,
            gtol=1.0e-11,
        )
        arm_q[frame] = fixed
        arm_q[frame, block] = solution.x
        previous = solution.x.copy()
        g1.assign(arm_q[frame], power["left"], power["right"])
        achieved_p, achieved_r, _, _ = g1.static_tool_pose_state(side, static_tool[side])
        solve_rows.append(
            {
                "success": bool(solution.success),
                "nfev": int(solution.nfev),
                "position_error_m": float(np.linalg.norm(achieved_p - target_p)),
                "orientation_error_rad": float(
                    np.linalg.norm(Rotation.from_matrix(achieved_r.T @ target_r).as_rotvec())
                ),
            }
        )
    left_hand = np.repeat(primitives["states"]["left"]["OPEN"][None], count, axis=0)
    right_hand = np.repeat(primitives["states"]["right"]["OPEN"][None], count, axis=0)
    source_names = list(g1.arm_joint_names) + list(g1.hand_joint_names["left"]) + list(
        g1.hand_joint_names["right"]
    )
    lookup = {name: index for index, name in enumerate(source_names)}
    full = np.column_stack((arm_q, left_hand, right_hand))[:, [lookup[name] for name in canonical_names]]
    achieved = []
    for frame in range(count):
        g1.assign(arm_q[frame], power["left"], power["right"])
        achieved.append(g1.whole_hand_grasp_pose(side)[:3, 3])
    achieved = g1.model_to_world_position(np.asarray(achieved))
    labels = np.asarray(["APPROACH_POWER"] * approach_count + ["LIFT"] * (lift_count - 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=full.astype(np.float32),
            stage=labels,
            joint_names=np.asarray(canonical_names),
            target_whole_hand_position_world_m=g1.model_to_world_position(active_position).astype(np.float32),
            achieved_whole_hand_position_world_m=achieved.astype(np.float32),
            control_fps_hz=np.asarray(fps),
            side=np.asarray(side),
            policy_independent=np.asarray(True),
            config_sha256=np.asarray(sha256_file(config_path)),
            open_arm_q_rad=arm_q[0].astype(np.float32),
            approach_arm_q_rad=arm_q[:approach_count].astype(np.float32),
            lift_arm_q_rad=arm_q[approach_count - 1 :].astype(np.float32),
            historical_clear_open_orientation_source=np.asarray(str(historical_path.resolve())),
            historical_clear_open_orientation_source_sha256=np.asarray(
                sha256_file(historical_path)
            ),
            ik_position_error_m=np.asarray(
                [row["position_error_m"] for row in solve_rows], dtype=np.float32
            ),
            ik_orientation_error_rad=np.asarray(
                [row["orientation_error_rad"] for row in solve_rows], dtype=np.float32
            ),
        )
    os.replace(temporary, output)
    error = np.linalg.norm(achieved - g1.model_to_world_position(active_position), axis=1)
    return {
        "side": side,
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "frames": count,
        "approach_frames": approach_count,
        "lift_frames": lift_count,
        "pregrasp_center_world_m": pregrasp_world.tolist(),
        "declared_start_center_world_m": start_world.tolist(),
        "declared_end_center_world_m": end_world.tolist(),
        "maximum_achieved_position_error_m": float(np.max(error)),
        "maximum_ik_position_error_m": max(row["position_error_m"] for row in solve_rows),
        "maximum_ik_orientation_error_rad": max(
            row["orientation_error_rad"] for row in solve_rows
        ),
        "all_ik_frames_success": all(row["success"] for row in solve_rows),
        "nominal_posture": nominal_report,
        "policy_or_checkpoint_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/doll_handoff_graspable_proxy_v2.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/doll_handoff_graspable_proxy_v2/scripted_primitives",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    schema = config.get("schema_version")
    if schema not in {
        "doll_handoff_graspable_proxy_v2",
        "doll_handoff_measured_proxy_v3",
        "dex3_measured_doll_graspability_sanity_v1",
        "dex3_simple_graspable_doll_proxy_v1",
    }:
        raise RuntimeError("unexpected config")
    revision = (
        "simple_v1"
        if schema == "dex3_simple_graspable_doll_proxy_v1"
        else "sanity_v1"
        if schema == "dex3_measured_doll_graspability_sanity_v1"
        else "v3"
        if schema == "doll_handoff_measured_proxy_v3"
        else "v2"
    )
    reports = [
        build_side(
            side,
            config,
            config_path,
            args.output_dir / f"{side}_{revision}_arm_primitive.npz",
        )
        for side in ("left", "right")
    ]
    manifest = {
        "schema_version": f"doll_handoff_graspable_proxy_{revision}_scripted_arm_manifest",
        "status": "READY",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_specific_adjustment": False,
        "side_specific_adjustment": False,
        "primitives": reports,
        "learned_policy_used": False,
        "real_robot": False,
    }
    atomic_json(args.output_dir / "MANIFEST.json", manifest)
    print((args.output_dir / "MANIFEST.json").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
