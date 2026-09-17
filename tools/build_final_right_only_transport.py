#!/usr/bin/env python3
"""Build one bounded RIGHT-only grasp/transport physics command.

The scientific doll/material/controller config is read-only.  A candidate may
change only the RIGHT object-relative wrist pose and RIGHT Dex3 command.  The
result starts from a normal scene reset and contains no object pose writes,
state restoration, policy, attachment, or runtime feedback.
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
    rotations_between,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import (  # noqa: E402
    AUTHORITATIVE_REFERENCES,
    authoritative_joint_ranges,
)


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
P14_ROOT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "hand_calibration_v2/p14_three_digit_preload"
)
DEFAULT_OUTPUT = ROOT / "outputs/final_task_completion_v1/01_right_transport_grasp"


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


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def candidate_hand(
    g1: G1Kinematics,
    names: list[str],
    config: dict[str, Any],
    profile: str,
) -> tuple[np.ndarray, dict[str, float]]:
    hand = hand_model(g1, names, config, "right", "POWER_GRASP_P14").copy()
    deltas: dict[str, float] = {}
    if profile == "P14":
        return hand, deltas
    if profile == "OPPOSE":
        deltas = {
            "right_hand_thumb_1_joint": -0.05,
            "right_hand_thumb_2_joint": -0.04,
        }
    elif profile == "CRADLE":
        deltas = {
            "right_hand_thumb_1_joint": -0.04,
            "right_hand_thumb_2_joint": -0.03,
            "right_hand_index_0_joint": 0.08,
            "right_hand_middle_0_joint": 0.08,
        }
    elif profile == "TRIPOD_WRAP":
        # R10 retained strong opposed thumb/index loads but left the middle
        # digit marginal during translation.  This bounded variant preserves
        # that opposition and adds closure only to the under-loaded chain.
        deltas = {
            "right_hand_thumb_1_joint": -0.04,
            "right_hand_thumb_2_joint": -0.03,
            "right_hand_index_0_joint": 0.08,
            "right_hand_middle_0_joint": 0.10,
            "right_hand_middle_1_joint": 0.15,
        }
    else:
        raise ValueError(profile)
    lookup = {name: index for index, name in enumerate(g1.hand_joint_names["right"])}
    for name, delta in deltas.items():
        hand[lookup[name]] += delta
    limits = np.asarray(g1.hand_limits["right"], dtype=np.float64)
    if np.any(hand < limits[:, 0]) or np.any(hand > limits[:, 1]):
        raise RuntimeError(f"candidate hand exceeds named limits: {profile}")
    return hand, deltas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--hand-profile",
        choices=("P14", "OPPOSE", "CRADLE", "TRIPOD_WRAP"),
        default="P14",
    )
    parser.add_argument("--scope", choices=("R5", "R6"), default="R5")
    parser.add_argument(
        "--bin-offset-x-m",
        type=float,
        default=0.0,
        help="Release-only bin-centering correction; doll/physics are unchanged.",
    )
    parser.add_argument(
        "--bin-offset-y-m",
        type=float,
        default=0.0,
        help="Release-only bin-centering correction; doll/physics are unchanged.",
    )
    parser.add_argument(
        "--release-object-z-m",
        type=float,
        help="Optional object-COM descent target used only for R6 release.",
    )
    parser.add_argument(
        "--transport-object-z-m",
        type=float,
        help=(
            "Optional object-COM clearance height for the short and full "
            "horizontal R6 segments.  This changes only the common arm path; "
            "the candidate wrist orientation, Dex3 grasp, doll, and release "
            "target remain unchanged."
        ),
    )
    parser.add_argument(
        "--release-frames",
        type=int,
        default=46,
        help="Minimum-jerk RIGHT open duration including both endpoints.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = (
        args.output.resolve()
        if args.output is not None
        else (DEFAULT_OUTPUT / "candidates" / args.candidate_id).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG)
    if config["active_calibration_profile"] != "P14":
        raise RuntimeError("authoritative bilateral P14 provenance is not active")
    right_result_path = P14_ROOT / "trials/right/trial_result.json"
    right_result = read_json(right_result_path)
    if right_result["status"] != "PASS":
        raise RuntimeError("frozen RIGHT standalone P14 evidence is not PASS")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    names, _ = authoritative_joint_ranges()
    left_open = hand_model(g1, names, config, "left", "OPEN")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    right_preshape = hand_model(g1, names, config, "right", "PRESHAPE")
    right_hand, hand_deltas = candidate_hand(
        g1, names, config, args.hand_profile
    )
    p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
    p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")

    primitive_path = Path(config["source_arm_primitives"]["right"])
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    reference_grasp_arm = np.asarray(
        primitive["approach_arm_q_rad"][-1], dtype=np.float64
    )
    reference_lift_arm = np.asarray(
        primitive["lift_arm_q_rad"][-1], dtype=np.float64
    )
    reference_grasp_world = np.asarray(
        primitive["target_whole_hand_position_world_m"][44], dtype=np.float64
    )
    reference_pregrasp_world = np.asarray(
        primitive["target_whole_hand_position_world_m"][0], dtype=np.float64
    )
    reference_lift_world = np.asarray(
        primitive["target_whole_hand_position_world_m"][-1], dtype=np.float64
    )

    g1.assign(reference_grasp_arm, p1_left, p1_right)
    static_tool = (
        np.linalg.inv(g1.wrist_pose("right"))
        @ g1.whole_hand_grasp_pose("right")
    )
    g1.assign(reference_grasp_arm, left_open, right_hand)
    _, reference_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool
    )
    origin = g1.model_to_world_position(np.zeros(3, dtype=np.float64))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3, dtype=np.float64)[axis]) - origin
            for axis in range(3)
        ]
    )
    reference_rotation_world = world_from_model @ reference_rotation_model
    delta_rotation_world = Rotation.from_euler(
        "xyz",
        [args.roll_deg, args.pitch_deg, args.yaw_deg],
        degrees=True,
    ).as_matrix()
    candidate_rotation_world = delta_rotation_world @ reference_rotation_world
    candidate_rotation_model = world_from_model.T @ candidate_rotation_world

    object_spawn_world = np.asarray(
        [
            *config["object"]["center_world_xy_m_by_side"]["right"],
            float(config["object"]["table_surface_world_z_m"])
            + 0.5 * float(config["object"]["visual_dimensions_m"][2]),
        ],
        dtype=np.float64,
    )
    candidate_grasp_world = object_spawn_world + delta_rotation_world @ (
        reference_grasp_world - object_spawn_world
    )
    candidate_pregrasp_world = candidate_grasp_world + delta_rotation_world @ (
        reference_pregrasp_world - reference_grasp_world
    )

    orientation_count = 31
    orientation_arm, orientation_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(reference_grasp_world, candidate_grasp_world, orientation_count)
        ),
        rotations_between(
            reference_rotation_model, candidate_rotation_model, orientation_count
        ),
        reference_grasp_arm,
        left_open,
        right_hand,
    )
    candidate_grasp_arm = orientation_arm[-1]
    reverse_pregrasp_arm, pregrasp_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(candidate_grasp_world, candidate_pregrasp_world, 31)
        ),
        np.repeat(candidate_rotation_model[None], 31, axis=0),
        candidate_grasp_arm,
        left_open,
        right_open,
    )
    candidate_pregrasp_arm = reverse_pregrasp_arm[-1]
    approach_arm, approach_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(candidate_pregrasp_world, candidate_grasp_world, 45)
        ),
        np.repeat(candidate_rotation_model[None], 45, axis=0),
        candidate_pregrasp_arm,
        left_open,
        right_preshape,
    )
    candidate_grasp_arm = approach_arm[-1]

    initial_lift_world = candidate_grasp_world + np.asarray([0.0, 0.0, 0.063])
    initial_lift_arm, initial_lift_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(candidate_grasp_world, initial_lift_world, 45)
        ),
        np.repeat(candidate_rotation_model[None], 45, axis=0),
        candidate_grasp_arm,
        left_open,
        right_hand,
    )

    with np.load(right_result["event_log"], allow_pickle=False) as archive:
        labels = archive["stage"].astype(str)
        reference_object_high = np.median(
            np.asarray(archive["object_position_world_m"], dtype=np.float64)[
                labels == "HOLD_ELEVATED"
            ][-240:],
            axis=0,
        )
    reference_tool_to_object = reference_lift_world - reference_object_high
    candidate_tool_to_object = delta_rotation_world @ reference_tool_to_object
    predicted_object_after_initial_lift = (
        initial_lift_world - candidate_tool_to_object
    )

    high_object_world = predicted_object_after_initial_lift + np.asarray(
        [0.0, 0.0, 0.100]
    )
    high_tool_world = high_object_world + candidate_tool_to_object
    vertical_arm, vertical_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(initial_lift_world, high_tool_world, 121)
        ),
        np.repeat(candidate_rotation_model[None], 121, axis=0),
        initial_lift_arm[-1],
        left_open,
        right_hand,
    )

    bin_xy = np.asarray(scene["bin"]["center_world_xy_m"], dtype=np.float64)
    bin_xy = bin_xy + np.asarray(
        [args.bin_offset_x_m, args.bin_offset_y_m], dtype=np.float64
    )
    direction_xy = bin_xy - high_object_world[:2]
    direction_xy /= max(np.linalg.norm(direction_xy), 1.0e-12)
    short_object_world = high_object_world.copy()
    short_object_world[:2] += 0.075 * direction_xy
    short_object_world[2] = max(short_object_world[2], 1.012)
    if args.transport_object_z_m is not None:
        short_object_world[2] = float(args.transport_object_z_m)
    short_tool_world = short_object_world + candidate_tool_to_object
    short_arm, short_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(high_tool_world, short_tool_world, 121)
        ),
        np.repeat(candidate_rotation_model[None], 121, axis=0),
        vertical_arm[-1],
        left_open,
        right_hand,
    )

    bin_object_world = np.r_[bin_xy, max(1.035, short_object_world[2])]
    bin_tool_world = bin_object_world + candidate_tool_to_object
    bin_arm = np.asarray([short_arm[-1]], dtype=np.float64)
    bin_reports: list[dict[str, Any]] = []
    release_object_world = bin_object_world.copy()
    release_object_world[2] = (
        float(scene["table"]["surface_height_m"])
        + float(scene["bin"]["outer_dimensions_xyz_m"][2])
        + 0.5 * float(config["geometry_candidates"][0]["dimensions_m"][2])
        + 0.005
    )
    if args.release_object_z_m is not None:
        release_object_world[2] = float(args.release_object_z_m)
    release_tool_world = release_object_world + candidate_tool_to_object
    descent_arm = np.asarray([short_arm[-1]], dtype=np.float64)
    descent_reports: list[dict[str, Any]] = []
    if args.scope == "R6":
        bin_arm, bin_reports = solve_path(
            g1,
            "right",
            static_tool,
            g1.world_to_model_position(
                minimum_jerk(short_tool_world, bin_tool_world, 241)
            ),
            np.repeat(candidate_rotation_model[None], 241, axis=0),
            short_arm[-1],
            left_open,
            right_hand,
        )
        descent_arm, descent_reports = solve_path(
            g1,
            "right",
            static_tool,
            g1.world_to_model_position(
                minimum_jerk(bin_tool_world, release_tool_world, 61)
            ),
            np.repeat(candidate_rotation_model[None], 61, axis=0),
            bin_arm[-1],
            left_open,
            right_hand,
        )

    fps = float(config["timing"]["control_fps_hz"])
    rows: list[np.ndarray] = []
    stages: list[str] = []

    def append(arm: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left_open, right))
        stages.append(stage)

    for _ in range(15):
        append(candidate_pregrasp_arm, right_open, "RIGHT_OPEN")
    preshape_path = minimum_jerk(right_open, right_preshape, 45)
    for arm, hand in zip(approach_arm[1:], preshape_path[1:], strict=True):
        append(arm, hand, "RIGHT_PRESHAPE_APPROACH")
    for hand in minimum_jerk(right_preshape, right_hand, 45)[1:]:
        append(candidate_grasp_arm, hand, "RIGHT_POWER_GRASP")
    for _ in range(30):
        append(candidate_grasp_arm, right_hand, "GRAVITY_RETENTION")
    for arm in initial_lift_arm[1:]:
        append(arm, right_hand, "RIGHT_INITIAL_LIFT")
    for _ in range(30):
        append(initial_lift_arm[-1], right_hand, "HOLD_ELEVATED")
    for arm in vertical_arm[1:]:
        append(arm, right_hand, "RIGHT_VERTICAL_100MM")
    for _ in range(30):
        append(vertical_arm[-1], right_hand, "RIGHT_HIGH_STABILIZATION")
    for arm in short_arm[1:]:
        append(arm, right_hand, "RIGHT_SHORT_HORIZONTAL")
    for _ in range(30):
        append(short_arm[-1], right_hand, "RIGHT_SHORT_STABILIZATION")
    if args.scope == "R6":
        for arm in bin_arm[1:]:
            append(arm, right_hand, "RIGHT_TRANSPORT_TO_BIN")
        for _ in range(30):
            append(bin_arm[-1], right_hand, "RIGHT_HOLD_OVER_BIN")
        for arm in descent_arm[1:]:
            append(arm, right_hand, "RIGHT_CONTROLLED_BIN_DESCENT")
        for _ in range(15):
            append(descent_arm[-1], right_hand, "RIGHT_PRE_RELEASE_STABILIZATION")
        if args.release_frames < 3:
            raise RuntimeError("--release-frames must be at least 3")
        for hand in minimum_jerk(
            right_hand, right_open, args.release_frames
        )[1:]:
            append(descent_arm[-1], hand, "RIGHT_RELEASE")
        for _ in range(120):
            append(descent_arm[-1], right_open, "POST_RELEASE")

    commands = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    joint_contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray(
        [float(row["minimum"]) for row in joint_contract["joint_specs"]]
    )
    upper = np.asarray(
        [float(row["maximum"]) for row in joint_contract["joint_specs"]]
    )
    violations = (commands < lower[None] - 1.0e-9) | (
        commands > upper[None] + 1.0e-9
    )
    name_to_index = {name: index for index, name in enumerate(names)}
    arm_indices = [name_to_index[name] for name in g1.arm_joint_names]
    left_indices = [name_to_index[name] for name in g1.hand_joint_names["left"]]
    right_indices = [name_to_index[name] for name in g1.hand_joint_names["right"]]
    geometry = g1.trajectory_geometry(
        commands[:, arm_indices],
        commands[:, left_indices],
        commands[:, right_indices],
        1.0e-5,
    )
    collision_counts = {
        key: int(np.count_nonzero(value))
        for key, value in geometry["collision_flags"].items()
    }
    reports = (
        orientation_reports
        + pregrasp_reports
        + approach_reports
        + initial_lift_reports
        + vertical_reports
        + short_reports
        + bin_reports
        + descent_reports
    )
    maximum_position_error = max(float(row["position_error_m"]) for row in reports)
    maximum_orientation_error = max(
        float(row["orientation_error_rad"]) for row in reports
    )
    maximum_arm_step = float(
        np.max(np.abs(np.diff(commands[:, arm_indices], axis=0)), initial=0.0)
    )
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and maximum_position_error <= 0.001
        and maximum_orientation_error <= 0.02
        and maximum_arm_step <= 0.03
    )
    command_path = output / f"right_only_{args.scope.lower()}_command.npz"
    save_npz(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": labels,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(fps),
            "policy_independent": np.asarray(True),
            "state_restoration_used": np.asarray(False),
            "object_pose_writes": np.asarray(0),
            "runtime_object_feedback_used": np.asarray(False),
            "candidate_id": np.asarray(args.candidate_id),
            "candidate_right_hand_model_order_7d_rad": right_hand.astype(np.float32),
            "candidate_rotation_delta_rpy_deg": np.asarray(
                [args.roll_deg, args.pitch_deg, args.yaw_deg], dtype=np.float32
            ),
            "predicted_object_after_initial_lift_world_m": predicted_object_after_initial_lift.astype(np.float32),
            "high_object_target_world_m": high_object_world.astype(np.float32),
            "short_object_target_world_m": short_object_world.astype(np.float32),
            "bin_object_target_world_m": bin_object_world.astype(np.float32),
            "release_object_target_world_m": release_object_world.astype(np.float32),
            "transport_object_clearance_z_m": np.asarray(
                short_object_world[2], dtype=np.float32
            ),
        },
    )
    report = {
        "schema_version": "final_right_only_transport_candidate_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "candidate_id": args.candidate_id,
        "scope": args.scope,
        "object_or_material_changed": False,
        "controller_gains_changed": False,
        "state_restoration_used": False,
        "prohibited_mechanisms_used": False,
        "candidate": {
            "rotation_delta_rpy_deg": [
                args.roll_deg,
                args.pitch_deg,
                args.yaw_deg,
            ],
            "hand_profile": args.hand_profile,
            "hand_model_order_7d_rad": right_hand,
            "hand_joint_delta_rad": hand_deltas,
            "bin_offset_xy_m": [args.bin_offset_x_m, args.bin_offset_y_m],
            "release_frames": args.release_frames,
            "grasp_tool_world_m": candidate_grasp_world,
            "pregrasp_tool_world_m": candidate_pregrasp_world,
            "predicted_tool_to_object_world_m": candidate_tool_to_object,
        },
        "waypoints": {
            "predicted_object_after_initial_lift_world_m": predicted_object_after_initial_lift,
            "high_object_world_m": high_object_world,
            "short_object_world_m": short_object_world,
            "bin_object_world_m": bin_object_world,
            "release_object_world_m": release_object_world,
        },
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_ik_position_error_m": maximum_position_error,
            "maximum_ik_orientation_error_rad": maximum_orientation_error,
            "maximum_adjacent_arm_step_rad": maximum_arm_step,
            "maximum_adjacent_arm_step_gate_rad": 0.03,
        },
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "frames": int(len(commands)),
        "duration_s": float(len(commands) / fps),
        "config": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "right_p14_result": str(right_result_path),
        "right_p14_result_sha256": sha256_file(right_result_path),
        "right_p14_event_sha256": sha256_file(Path(right_result["event_log"])),
        "builder": str(Path(__file__).resolve()),
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(output / "offline_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
