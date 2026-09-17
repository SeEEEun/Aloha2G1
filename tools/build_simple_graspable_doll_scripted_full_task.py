#!/usr/bin/env python3
"""Build one policy-free, start-to-finish Doll-Handoff command.

The doll/material contract is read-only.  The command reuses the bilateral P14
single-hand grasp, searches only a small declared set of right-palm acquisition
placements, and emits one 30 Hz trajectory through bin release.
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
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import (  # noqa: E402
    canonical_row,
    hand_model,
    minimum_jerk,
    preliminary_candidate_audit,
    solve_bounded_pose,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import (  # noqa: E402
    AUTHORITATIVE_REFERENCES,
    authoritative_joint_ranges,
)


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task"
    / "p14_bilateral"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG)
    if config["active_calibration_profile"] != "P14":
        raise RuntimeError("bilateral P14 is not active")
    trial_root = (
        ROOT
        / "outputs/dex3_simple_graspable_doll_proxy_v1"
        / "hand_calibration_v2/p14_three_digit_preload/trials"
    )
    trials = {side: read_json(trial_root / side / "trial_result.json") for side in ("left", "right")}
    if any(trials[side]["status"] != "PASS" for side in trials):
        raise RuntimeError("bilateral P14 gate is not PASS")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal_arm, _ = g1.derive_task_ready_nominal(common, scene)
    names, _ = authoritative_joint_ranges()
    left_open = hand_model(g1, names, config, "left", "OPEN")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    left_pre = hand_model(g1, names, config, "left", "PRESHAPE")
    right_pre = hand_model(g1, names, config, "right", "PRESHAPE")
    left_power = hand_model(g1, names, config, "left", "POWER_GRASP_P14")
    right_power = hand_model(g1, names, config, "right", "POWER_GRASP_P14")
    p1 = {side: hand_model(g1, names, config, side, "POWER_GRASP_P1") for side in ("left", "right")}

    primitive_paths = {side: Path(config["source_arm_primitives"][side]) for side in ("left", "right")}
    primitives: dict[str, dict[str, np.ndarray]] = {}
    for side, path in primitive_paths.items():
        with np.load(path, allow_pickle=False) as archive:
            primitives[side] = {key: np.asarray(archive[key]) for key in archive.files}
    reference_arm = {side: np.asarray(primitives[side]["lift_arm_q_rad"][-1], dtype=np.float64) for side in ("left", "right")}
    origin = g1.model_to_world_position(np.zeros(3))
    world_from_model = np.column_stack(
        [g1.model_to_world_position(np.eye(3)[axis]) - origin for axis in range(3)]
    )
    static_tool: dict[str, np.ndarray] = {}
    reference_rotation_world: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        g1.assign(reference_arm[side], p1["left"], p1["right"])
        static_tool[side] = np.linalg.inv(g1.wrist_pose(side)) @ g1.whole_hand_grasp_pose(side)
        g1.assign(reference_arm[side], left_power, right_power)
        _, rotation_model, _, _ = g1.static_tool_pose_state(side, static_tool[side])
        reference_rotation_world[side] = world_from_model @ rotation_model

    # Measured P14 load-relative tool offsets are derived from the untouched
    # bilateral physics logs, not from object or fingertip target tuning.
    object_centers: dict[str, np.ndarray] = {}
    tool_offsets: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        with np.load(trials[side]["event_log"], allow_pickle=False) as archive:
            mask = archive["stage"].astype(str) == "HOLD_ELEVATED"
            object_centers[side] = np.median(archive["object_position_world_m"][mask][-240:], axis=0)
        tool = np.asarray(primitives[side]["target_whole_hand_position_world_m"][-1], dtype=np.float64)
        tool_offsets[side] = tool - object_centers[side]

    fps = float(config["timing"]["control_fps_hz"])
    left_arm_open = np.asarray(primitives["left"]["approach_arm_q_rad"][0], dtype=np.float64)
    left_arm_grasp = np.asarray(primitives["left"]["approach_arm_q_rad"][-1], dtype=np.float64)
    left_lift_path = np.asarray(primitives["left"]["lift_arm_q_rad"], dtype=np.float64)
    left_lift_tool = np.asarray(primitives["left"]["target_whole_hand_position_world_m"][-1], dtype=np.float64)
    handoff_object = np.asarray([0.4175, 0.205, object_centers["left"][2]], dtype=np.float64)
    left_handoff_tool = handoff_object + tool_offsets["left"]
    left_rotation_model = world_from_model.T @ reference_rotation_world["left"]
    right_rotation_model = world_from_model.T @ reference_rotation_world["right"]
    left_transport_count = int(round(2.0 * fps))
    left_transport, left_reports = solve_path(
        g1,
        "left",
        static_tool["left"],
        g1.world_to_model_position(minimum_jerk(left_lift_tool, left_handoff_tool, left_transport_count)),
        np.repeat(left_rotation_model[None], left_transport_count, axis=0),
        left_lift_path[-1],
        left_power,
        right_open,
    )
    left_handoff_arm = left_transport[-1]

    # Exactly twelve palm-only candidates: four long-axis separations and
    # three yaw values.  Doll, material, finger vector, and arm target of the
    # retaining left hand are invariant.
    candidates: list[dict[str, Any]] = []
    for dx in (0.015, 0.025, 0.035, 0.045):
        for yaw in (-35.0, 0.0, 35.0):
            right_world = handoff_object + tool_offsets["right"] + np.asarray([dx, 0.0, 0.005])
            rotation_world = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ reference_rotation_world["right"]
            try:
                arm, pose_report = solve_bounded_pose(
                    g1,
                    "right",
                    static_tool["right"],
                    g1.world_to_model_position(right_world),
                    world_from_model.T @ rotation_world,
                    left_handoff_arm,
                    reference_arm["right"][7:14],
                    left_power,
                    right_power,
                    2026083000 + int(round(dx * 1000)) * 10 + int(yaw + 40),
                )
                audit = preliminary_candidate_audit(
                    g1,
                    arm,
                    left_power,
                    right_power,
                    handoff_object,
                    np.asarray(config["geometry_candidates"][0]["dimensions_m"], dtype=np.float64),
                )
                candidates.append(
                    {
                        "dx_m": dx,
                        "yaw_deg": yaw,
                        "right_tool_world_m": right_world,
                        "right_rotation_world": rotation_world,
                        "arm": arm,
                        "pose_report": pose_report,
                        "audit": audit,
                    }
                )
            except RuntimeError as error:
                candidates.append({"dx_m": dx, "yaw_deg": yaw, "error": str(error)})
    feasible = [
        row
        for row in candidates
        if "audit" in row
        and row["audit"]["right_hand_doll_contact"]
        and row["audit"]["no_left_right_hand_overlap"]
        and row["audit"]["minimum_left_right_clearance_m"] >= 0.0
    ]
    if not feasible:
        serializable = [
            {key: value for key, value in row.items() if key not in {"arm", "right_rotation_world"}}
            for row in candidates
        ]
        atomic_json(output / "handoff_candidate_audit.json", {"status": "FAIL", "candidates": serializable})
        raise RuntimeError("no collision-free right acquisition contact in bounded palm search")
    chosen = max(
        feasible,
        key=lambda row: (
            row["audit"]["right_contact_count_at_1mm_tolerance"],
            row["audit"]["minimum_left_right_clearance_m"],
            -abs(row["dx_m"]),
        ),
    )
    right_acquire_arm = np.asarray(chosen["arm"], dtype=np.float64)

    # Release-order audit showed that opening opposition first avoids the
    # thumb/thumb sweep.  Then retreat the open left palm in Cartesian space,
    # away from the right hand, instead of interpolating through the robot.
    left_retreat_world = left_handoff_tool + np.asarray([-0.10, -0.06, 0.04])
    left_retreat_path, left_retreat_reports = solve_path(
        g1,
        "left",
        static_tool["left"],
        g1.world_to_model_position(minimum_jerk(left_handoff_tool, left_retreat_world, round(1.5 * fps))),
        np.repeat(left_rotation_model[None], round(1.5 * fps), axis=0),
        right_acquire_arm,
        left_open,
        right_power,
    )
    left_parked = left_retreat_path[-1]
    # Once the left hand is open and parked, return the right palm to its
    # validated load-relative P14 geometry before transport.
    right_hold_world = handoff_object + tool_offsets["right"]
    right_hold_arm, right_hold_report = solve_bounded_pose(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(right_hold_world),
        right_rotation_model,
        left_parked,
        right_acquire_arm[7:14],
        left_open,
        right_power,
        2026083091,
    )
    bin_object = np.asarray([0.738212049, 0.099787664, 1.045], dtype=np.float64)
    bin_tool = bin_object + tool_offsets["right"]
    right_transport_count = int(round(2.5 * fps))
    right_transport, right_reports = solve_path(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(minimum_jerk(right_hold_world, bin_tool, right_transport_count)),
        np.repeat(right_rotation_model[None], right_transport_count, axis=0),
        right_hold_arm,
        left_open,
        right_power,
    )

    rows: list[np.ndarray] = []
    stages: list[str] = []

    def append(arm: np.ndarray, left: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left, right))
        stages.append(stage)

    for _ in range(round(0.5 * fps)):
        append(left_arm_open, left_open, right_open, "LEFT_OPEN")
    left_pre_path = minimum_jerk(left_open, left_pre, round(1.5 * fps))
    left_approach = np.asarray(primitives["left"]["approach_arm_q_rad"], dtype=np.float64)
    for arm, hand in zip(left_approach[1:], left_pre_path[1:], strict=True):
        append(arm, hand, right_open, "LEFT_PRESHAPE_APPROACH")
    for hand in minimum_jerk(left_pre, left_power, round(1.5 * fps))[1:]:
        append(left_arm_grasp, hand, right_open, "LEFT_POWER_GRASP")
    for _ in range(round(1.0 * fps)):
        append(left_arm_grasp, left_power, right_open, "GRAVITY_RETENTION")
    for arm in left_lift_path[1:]:
        append(arm, left_power, right_open, "LEFT_LIFT_5CM")
    for _ in range(round(1.0 * fps)):
        append(left_lift_path[-1], left_power, right_open, "HOLD_ELEVATED")
    for arm in left_transport[1:]:
        append(arm, left_power, right_open, "LEFT_TRANSPORT")
    for _ in range(round(0.5 * fps)):
        append(left_handoff_arm, left_power, right_open, "LEFT_HANDOFF_HOLD")
    for arm in minimum_jerk(left_handoff_arm, right_acquire_arm, round(2.0 * fps))[1:]:
        append(arm, left_power, right_open, "RIGHT_APPROACH")
    for hand in minimum_jerk(right_open, right_pre, round(1.0 * fps))[1:]:
        append(right_acquire_arm, left_power, hand, "RIGHT_PRESHAPE")
    for hand in minimum_jerk(right_pre, right_power, round(1.5 * fps))[1:]:
        append(right_acquire_arm, left_power, hand, "RIGHT_ACQUIRE")
    for _ in range(round(0.75 * fps)):
        append(right_acquire_arm, left_power, right_power, "DUAL_CONTACT_HOLD")
    # Collision-free release order established exhaustively over the six
    # thumb-joint permutations: open index+middle, then thumb_1, thumb_0,
    # thumb_2.  This changes no endpoints or grasp target.
    left_release = left_power.copy()
    target = left_release.copy()
    target[3:] = left_open[3:]
    for hand in minimum_jerk(left_release, target, 20)[1:]:
        append(right_acquire_arm, hand, right_power, "LEFT_RELEASE_INDEX_MIDDLE")
    left_release = target
    for joint_index in (1, 0, 2):
        target = left_release.copy()
        target[joint_index] = left_open[joint_index]
        for hand in minimum_jerk(left_release, target, 15)[1:]:
            append(right_acquire_arm, hand, right_power, "LEFT_RELEASE_THUMB")
        left_release = target
    for _ in range(round(0.5 * fps)):
        append(right_acquire_arm, left_open, right_power, "RIGHT_POST_RELEASE_HOLD")
    for arm in left_retreat_path[1:]:
        append(arm, left_open, right_power, "LEFT_RETREAT")
    for arm in minimum_jerk(left_parked, right_hold_arm, round(1.0 * fps))[1:]:
        append(arm, left_open, right_power, "RIGHT_SETTLE_TO_POWER_GRASP")
    for _ in range(round(1.0 * fps)):
        append(right_hold_arm, left_open, right_power, "RIGHT_OWNED_HOLD")
    for arm in right_transport[1:]:
        append(arm, left_open, right_power, "RIGHT_TRANSPORT_TO_BIN")
    for _ in range(round(0.5 * fps)):
        append(right_transport[-1], left_open, right_power, "RIGHT_HOLD_OVER_BIN")
    for hand in minimum_jerk(right_power, right_open, round(1.0 * fps))[1:]:
        append(right_transport[-1], left_open, hand, "RIGHT_RELEASE")
    for _ in range(round(2.0 * fps)):
        append(right_transport[-1], left_open, right_open, "POST_RELEASE")

    commands = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    joint_contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray([float(row["minimum"]) for row in joint_contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in joint_contract["joint_specs"]])
    violations = (commands < lower[None] - 1e-9) | (commands > upper[None] + 1e-9)
    lookup = {name: index for index, name in enumerate(names)}
    arm = commands[:, [lookup[name] for name in g1.arm_joint_names]]
    left_model = commands[:, [lookup[name] for name in g1.hand_joint_names["left"]]]
    right_model = commands[:, [lookup[name] for name in g1.hand_joint_names["right"]]]
    geometry = g1.trajectory_geometry(arm, left_model, right_model, 1e-5)
    collision_counts = {key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()}
    command_path = output / "scripted_full_task_command.npz"
    temporary = command_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=commands.astype(np.float32),
            stage=labels,
            joint_names=np.asarray(names),
            control_fps_hz=np.asarray(fps),
            object_initial_center_world_m=np.asarray([0.3, 0.15, 0.8375], dtype=np.float32),
            handoff_object_center_world_m=handoff_object.astype(np.float32),
            bin_object_target_world_m=bin_object.astype(np.float32),
            policy_independent=np.asarray(True),
            scripted_full_task=np.asarray(True),
        )
    os.replace(temporary, command_path)
    serializable_candidates = [
        {key: value for key, value in row.items() if key not in {"arm", "right_rotation_world"}}
        for row in candidates
    ]
    report = {
        "schema_version": "simple_graspable_doll_scripted_full_task_v1",
        "status": "READY" if not np.any(violations) else "FAIL",
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "bilateral_trials": {side: {"path": str(trial_root / side / "trial_result.json"), "sha256": sha256_file(trial_root / side / "trial_result.json"), "status": trials[side]["status"]} for side in trials},
        "doll_parameters_changed": False,
        "selected_handoff_candidate": {key: value for key, value in chosen.items() if key not in {"arm", "right_rotation_world"}},
        "bounded_candidates": serializable_candidates,
        "right_hold_pose_report": right_hold_report,
        "ik_max_position_error_m": max(row["position_error_m"] for row in left_reports + left_retreat_reports + right_reports),
        "ik_max_orientation_error_rad": max(row["orientation_error_rad"] for row in left_reports + left_retreat_reports + right_reports),
        "joint_limit_violations": int(np.count_nonzero(violations)),
        "robot_collision_frame_counts": collision_counts,
        "robot_collision_pairs": geometry["collision_pairs"],
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(output / "build_report.json", report)
    print((output / "build_report.json").read_text(encoding="utf-8"), end="")
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
