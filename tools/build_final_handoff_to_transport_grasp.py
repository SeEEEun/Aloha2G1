#!/usr/bin/env python3
"""Connect the validated LEFT handoff prefix to the selected RIGHT transport grasp.

The command starts at frame zero and reuses the exact successful acquisition
prefix through LEFT index/middle partial relaxation.  While LEFT still supports
the doll, RIGHT follows one minimum-jerk SE(3) path into the already validated
R14 transport grasp.  LEFT release is runtime-gated on sustained RIGHT
three-digit support.  No object state, material, gain, or safety setting is
changed and no state restoration or runtime object feedback is used.
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
SOURCE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task"
    / "p14_bilateral/backward_constructed_handoff/B2_PATH_F40"
    / "right_preload_partial_left_relax_exact_endpoint_v4/full"
    / "scripted_full_task_command.npz"
)
SELECTED = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp"
    / "SELECTED_RIGHT_TRANSPORT_GRASP.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/final_task_completion_v1/02_handoff_to_transport_grasp"
    / "H01_BACKWARD_TO_R14"
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


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate", "full"), default="gate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = read_json(CONFIG)
    selected = read_json(SELECTED)
    if selected.get("status") != "PASS":
        raise RuntimeError("selected RIGHT transport grasp is not PASS")
    if selected["scientific_config_sha256"] != sha256_file(CONFIG):
        raise RuntimeError("scientific doll/controller config changed")
    if sha256_file(SOURCE) != "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab":
        raise RuntimeError("verified continuous handoff prefix changed")

    with np.load(SOURCE, allow_pickle=False) as archive:
        source_commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        source_stages = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
        handoff_object = np.asarray(
            archive["handoff_object_center_world_m"], dtype=np.float64
        )
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or not np.isclose(fps, 30.0):
        raise RuntimeError("authoritative 28D command interface changed")

    # Retain the exact successful prefix through partial LEFT relaxation.
    partial_rows = np.flatnonzero(
        source_stages == "LEFT_INDEX_MIDDLE_PARTIAL_RELAX_AFTER_RIGHT_PRELOAD"
    )
    if len(partial_rows) != 29:
        raise RuntimeError("verified handoff prefix stage changed")
    prefix_end = int(partial_rows[-1] + 1)
    rows = [row.copy() for row in source_commands[:prefix_end]]
    stages = source_stages[:prefix_end].tolist()

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    left_open = hand_model(g1, names, config, "left", "OPEN")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
    p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")
    right_transport = np.asarray(
        selected["right_transport_hold_model_order_7d_rad"], dtype=np.float64
    )

    current_arm = rows[-1][arm_indices].copy()
    partial_left = rows[-1][left_indices].copy()
    acquisition_hand = rows[-1][right_indices].copy()

    primitive_path = Path(config["source_arm_primitives"]["right"])
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    reference_grasp_arm = np.asarray(
        primitive["approach_arm_q_rad"][-1], dtype=np.float64
    )
    reference_lift_world = np.asarray(
        primitive["target_whole_hand_position_world_m"][-1], dtype=np.float64
    )
    g1.assign(reference_grasp_arm, p1_left, p1_right)
    static_tool_right = (
        np.linalg.inv(g1.wrist_pose("right"))
        @ g1.whole_hand_grasp_pose("right")
    )
    g1.assign(reference_grasp_arm, left_open, right_transport)
    _, reference_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool_right
    )
    origin = g1.model_to_world_position(np.zeros(3, dtype=np.float64))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3, dtype=np.float64)[axis])
            - origin
            for axis in range(3)
        ]
    )
    reference_rotation_world = world_from_model @ reference_rotation_model
    rotation_delta = Rotation.from_euler("x", 10.0, degrees=True).as_matrix()
    transport_rotation_world = rotation_delta @ reference_rotation_world
    transport_rotation_model = world_from_model.T @ transport_rotation_world

    p14_result = read_json(
        ROOT
        / "outputs/dex3_simple_graspable_doll_proxy_v1/hand_calibration_v2"
        / "p14_three_digit_preload/trials/right/trial_result.json"
    )
    with np.load(p14_result["event_log"], allow_pickle=False) as archive:
        mask = archive["stage"].astype(str) == "HOLD_ELEVATED"
        p14_object_high = np.median(
            np.asarray(archive["object_position_world_m"], dtype=np.float64)[mask][
                -240:
            ],
            axis=0,
        )
    reference_tool_to_object = reference_lift_world - p14_object_high
    transport_tool_to_object = rotation_delta @ reference_tool_to_object
    transport_tool_world = handoff_object + transport_tool_to_object

    # Establish 35 mm of LEFT radial clearance while the already-validated
    # acquisition posture carries the opposing load.  The first direct path
    # audit showed that keeping the old LEFT thumb at its original patch would
    # overlap the selected +10 degree RIGHT thumb.  This is a staged load
    # transfer, not a new grasp or wrist-pose search.
    static_tools: dict[str, np.ndarray] = {"right": static_tool_right}
    left_primitive_path = Path(config["source_arm_primitives"]["left"])
    with np.load(left_primitive_path, allow_pickle=False) as archive:
        left_primitive = {key: np.asarray(archive[key]) for key in archive.files}
    left_reference_arm = np.asarray(
        left_primitive["approach_arm_q_rad"][-1], dtype=np.float64
    )
    g1.assign(left_reference_arm, p1_left, p1_right)
    static_tools["left"] = (
        np.linalg.inv(g1.wrist_pose("left")) @ g1.whole_hand_grasp_pose("left")
    )
    g1.assign(current_arm, partial_left, acquisition_hand)
    left_start_model, left_rotation_model, _, _ = g1.static_tool_pose_state(
        "left", static_tools["left"]
    )
    left_start_world = g1.model_to_world_position(left_start_model)
    radial = left_start_world - handoff_object
    radial /= max(np.linalg.norm(radial), 1.0e-12)
    left_preclear_world = left_start_world + 0.035 * radial
    preclear_count = 61
    preclear_arm, preclear_reports = solve_path(
        g1,
        "left",
        static_tools["left"],
        g1.world_to_model_position(
            minimum_jerk(left_start_world, left_preclear_world, preclear_count)
        ),
        np.repeat(left_rotation_model[None], preclear_count, axis=0),
        current_arm,
        partial_left,
        acquisition_hand,
    )

    # Follow the known collision-free family to the standalone P14 endpoint
    # first, then apply the selected R14 +10 degree transport presentation as a
    # separate short path.  Splitting these motions avoids the direct-path IK
    # branch jump found by H01.
    g1.assign(preclear_arm[-1], partial_left, acquisition_hand)
    start_model, start_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool_right
    )
    start_world = g1.model_to_world_position(start_model)
    p14_tool_world = handoff_object + reference_tool_to_object
    p14_count = 181
    p14_arm, p14_reports = solve_path(
        g1,
        "right",
        static_tool_right,
        g1.world_to_model_position(
            minimum_jerk(start_world, p14_tool_world, p14_count)
        ),
        np.einsum(
            "ij,tjk->tik",
            world_from_model.T,
            rotations_between(
                world_from_model @ start_rotation_model,
                reference_rotation_world,
                p14_count,
            ),
        ),
        preclear_arm[-1],
        partial_left,
        acquisition_hand,
        clearance_target_m=0.001,
    )
    transition_count = 61
    transition_arm, transition_reports = solve_path(
        g1,
        "right",
        static_tool_right,
        g1.world_to_model_position(
            minimum_jerk(p14_tool_world, transport_tool_world, transition_count)
        ),
        np.einsum(
            "ij,tjk->tik",
            world_from_model.T,
            rotations_between(
                reference_rotation_world,
                transport_rotation_world,
                transition_count,
            ),
        ),
        p14_arm[-1],
        partial_left,
        right_transport,
        clearance_target_m=0.001,
    )

    def append(arm: np.ndarray, left: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left, right))
        stages.append(stage)

    for arm in preclear_arm[1:]:
        append(
            arm,
            partial_left,
            acquisition_hand,
            "LEFT_SUPPORTED_RADIAL_PRECLEARANCE",
        )
    for arm in p14_arm[1:]:
        append(
            arm,
            partial_left,
            acquisition_hand,
            "RIGHT_APPROACH_TRANSPORT_ENDPOINT",
        )
    right_hand_path = minimum_jerk(acquisition_hand, right_transport, transition_count)
    for arm, right in zip(
        transition_arm[1:], right_hand_path[1:], strict=True
    ):
        append(arm, partial_left, right, "RIGHT_CONVERGE_TO_TRANSPORT_GRASP")
    for _ in range(30):
        append(
            transition_arm[-1],
            partial_left,
            right_transport,
            "RIGHT_THREE_DIGIT_VERIFICATION",
        )

    # Complete only the remaining 10 mm of the declared 45 mm LEFT radial exit
    # while opening. RIGHT remains fixed at the transport endpoint.
    left_clear_world = left_start_world + 0.045 * radial
    release_count = 61
    release_arm, release_reports = solve_path(
        g1,
        "left",
        static_tools["left"],
        g1.world_to_model_position(
            minimum_jerk(left_preclear_world, left_clear_world, release_count)
        ),
        np.repeat(left_rotation_model[None], release_count, axis=0),
        transition_arm[-1],
        left_open,
        right_transport,
    )
    left_release_path = minimum_jerk(partial_left, left_open, release_count)
    for arm, left in zip(release_arm[1:], left_release_path[1:], strict=True):
        append(arm, left, right_transport, "LEFT_THUMB_RELEASE")
    for _ in range(30):
        append(
            release_arm[-1],
            left_open,
            right_transport,
            "RIGHT_POST_RELEASE_RETENTION",
        )

    left_retreat_world = left_clear_world + np.asarray([-0.10, -0.06, 0.04])
    retreat_count = 61
    retreat_arm, retreat_reports = solve_path(
        g1,
        "left",
        static_tools["left"],
        g1.world_to_model_position(
            minimum_jerk(left_clear_world, left_retreat_world, retreat_count)
        ),
        np.repeat(left_rotation_model[None], retreat_count, axis=0),
        release_arm[-1],
        left_open,
        right_transport,
    )
    for arm in retreat_arm[1:]:
        append(arm, left_open, right_transport, "LEFT_SAFE_RETREAT")
    for _ in range(30):
        append(
            retreat_arm[-1], left_open, right_transport, "RIGHT_OWNED_HOLD"
        )

    full_reports: list[dict[str, Any]] = []
    if args.mode == "full":
        high_object = handoff_object + np.asarray([0.0, 0.0, 0.100])
        high_tool = high_object + transport_tool_to_object
        vertical_arm, vertical_reports = solve_path(
            g1,
            "right",
            static_tool_right,
            g1.world_to_model_position(
                minimum_jerk(transport_tool_world, high_tool, 121)
            ),
            np.repeat(transport_rotation_model[None], 121, axis=0),
            retreat_arm[-1],
            left_open,
            right_transport,
        )
        for arm in vertical_arm[1:]:
            append(arm, left_open, right_transport, "RIGHT_VERTICAL_CLEARANCE")
        for _ in range(30):
            append(
                vertical_arm[-1],
                left_open,
                right_transport,
                "RIGHT_HIGH_STABILIZATION",
            )

        bin_xy = np.asarray(scene["bin"]["center_world_xy_m"], dtype=np.float64)
        bin_xy += np.asarray([0.015, -0.010])
        direction = bin_xy - high_object[:2]
        direction /= max(np.linalg.norm(direction), 1.0e-12)
        short_object = high_object.copy()
        short_object[:2] += 0.075 * direction
        short_object[2] = max(short_object[2], 1.012)
        short_tool = short_object + transport_tool_to_object
        short_arm, short_reports = solve_path(
            g1,
            "right",
            static_tool_right,
            g1.world_to_model_position(
                minimum_jerk(high_tool, short_tool, 121)
            ),
            np.repeat(transport_rotation_model[None], 121, axis=0),
            vertical_arm[-1],
            left_open,
            right_transport,
        )
        for arm in short_arm[1:]:
            append(arm, left_open, right_transport, "RIGHT_SHORT_HORIZONTAL")
        for _ in range(30):
            append(
                short_arm[-1],
                left_open,
                right_transport,
                "RIGHT_SHORT_STABILIZATION",
            )

        bin_object = np.r_[bin_xy, max(1.035, short_object[2])]
        bin_tool = bin_object + transport_tool_to_object
        bin_arm, bin_reports = solve_path(
            g1,
            "right",
            static_tool_right,
            g1.world_to_model_position(minimum_jerk(short_tool, bin_tool, 241)),
            np.repeat(transport_rotation_model[None], 241, axis=0),
            short_arm[-1],
            left_open,
            right_transport,
        )
        for arm in bin_arm[1:]:
            append(arm, left_open, right_transport, "RIGHT_TRANSPORT_TO_BIN")
        for _ in range(30):
            append(
                bin_arm[-1], left_open, right_transport, "RIGHT_HOLD_OVER_BIN"
            )

        release_object = np.r_[bin_xy, 1.000]
        release_tool = release_object + transport_tool_to_object
        descent_arm, descent_reports = solve_path(
            g1,
            "right",
            static_tool_right,
            g1.world_to_model_position(
                minimum_jerk(bin_tool, release_tool, 61)
            ),
            np.repeat(transport_rotation_model[None], 61, axis=0),
            bin_arm[-1],
            left_open,
            right_transport,
        )
        for arm in descent_arm[1:]:
            append(
                arm, left_open, right_transport, "RIGHT_CONTROLLED_BIN_DESCENT"
            )
        for _ in range(15):
            append(
                descent_arm[-1],
                left_open,
                right_transport,
                "RIGHT_PRE_RELEASE_STABILIZATION",
            )
        for right in minimum_jerk(right_transport, right_open, 16)[1:]:
            append(descent_arm[-1], left_open, right, "RIGHT_RELEASE")
        for _ in range(120):
            append(descent_arm[-1], left_open, right_open, "POST_RELEASE")
        full_reports = (
            vertical_reports
            + short_reports
            + bin_reports
            + descent_reports
        )

    commands = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (commands < lower[None] - 1.0e-9) | (
        commands > upper[None] + 1.0e-9
    )
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
        preclear_reports
        + p14_reports
        + transition_reports
        + release_reports
        + retreat_reports
        + full_reports
    )
    maximum_position_error = max(float(row["position_error_m"]) for row in reports)
    maximum_orientation_error = max(
        float(row["orientation_error_rad"]) for row in reports
    )
    arm_steps = np.abs(np.diff(commands[:, arm_indices], axis=0))
    maximum_arm_step = float(np.max(arm_steps, initial=0.0))
    maximum_new_arm_step = float(
        np.max(arm_steps[prefix_end - 1 :], initial=0.0)
    )
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and maximum_position_error <= 0.001
        and maximum_orientation_error <= 0.02
        # The immutable successful prefix contains one 0.03396 rad LEFT
        # transport step. New construction must meet the unchanged 0.03 rad
        # local path gate; inherited evidence is reported separately.
        and maximum_new_arm_step <= 0.03
    )

    command_path = output / f"handoff_to_transport_{args.mode}_command.npz"
    save_npz(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": labels,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(fps),
            "policy_independent": np.asarray(True),
            "scripted_full_task": np.asarray(args.mode == "full"),
            "runtime_right_three_digit_gate_required": np.asarray(True),
            "right_three_digit_gate_minimum_s": np.asarray(0.5),
            "attachment_used": np.asarray(False),
            "object_follow_used": np.asarray(False),
            "object_pose_writes": np.asarray(0),
            "state_restoration_used": np.asarray(False),
            "selected_right_transport_hand_model_order_7d_rad": right_transport.astype(np.float32),
            "handoff_object_center_world_m": handoff_object.astype(np.float32),
            "transport_tool_target_world_m": transport_tool_world.astype(np.float32),
        },
    )
    report = {
        "schema_version": "final_handoff_to_transport_grasp_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "mode": args.mode,
        "construction": "backward from selected RIGHT transport-capable grasp",
        "source_verified_prefix": str(SOURCE),
        "source_verified_prefix_sha256": sha256_file(SOURCE),
        "selected_right_grasp": str(SELECTED),
        "selected_right_grasp_sha256": sha256_file(SELECTED),
        "selected_right_transport_hand_model_order_7d_rad": right_transport,
        "transport_tool_target_world_m": transport_tool_world,
        "transport_rotation_world": transport_rotation_world,
        "handoff_object_center_world_m": handoff_object,
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_ik_position_error_m": maximum_position_error,
            "maximum_ik_orientation_error_rad": maximum_orientation_error,
            "maximum_adjacent_arm_step_rad": maximum_arm_step,
            "maximum_inherited_prefix_arm_step_rad": float(
                np.max(arm_steps[: prefix_end - 1], initial=0.0)
            ),
            "maximum_new_adjacent_arm_step_rad": maximum_new_arm_step,
            "maximum_new_adjacent_arm_step_gate_rad": 0.03,
        },
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "runtime_release_gate": {
            "stage": "RIGHT_THREE_DIGIT_VERIFICATION",
            "minimum_simultaneous_support_s": 0.5,
            "left_release_stage": "LEFT_THUMB_RELEASE",
        },
        "doll_or_material_changed": False,
        "controller_gains_changed": False,
        "state_restoration_used": False,
        "runtime_object_feedback_used": False,
        "prohibited_mechanisms_used": False,
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(output / f"offline_{args.mode}_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
