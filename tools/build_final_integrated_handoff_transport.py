#!/usr/bin/env python3
"""Integrate the verified handoff/keyframe retreat into the selected R14 grasp.

This is a normal frame-zero command.  It contains no state restoration.  The
first prefix is the exact continuous run that physically established RIGHT-only
ownership for one second.  It is followed by the exact T5 LEFT radial-clearance
segment whose initial command equals that prefix byte-for-byte.  Only after
LEFT is clear does RIGHT follow a bounded path into the independently validated
transport-capable R14 grasp.
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
from scipy.ndimage import gaussian_filter1d
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
HANDOFF_SOURCE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task"
    / "p14_bilateral/backward_constructed_handoff/B2_PATH_F40"
    / "right_preload_partial_left_relax_exact_endpoint_v4/full"
    / "scripted_full_task_command.npz"
)
T5_TAIL = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/success_first_common_execution"
    / "tail_debug/t5_t4_grip_robust_final_transport_v1/tail_command.npz"
)
T5_REPORT = T5_TAIL.parent / "offline_report.json"
SELECTED = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp"
    / "SELECTED_RIGHT_TRANSPORT_GRASP.json"
)
H07_EVENT = (
    ROOT
    / "outputs/final_task_completion_v1/02_handoff_to_transport_grasp"
    / "H07_LEFT_CLEAR_FIXED_BRIDGE_REFERENCE/physics_right_sensor/event_log.npz"
)
R14_EVENT = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
    / "R14_RELEASE_D1_CENTERED_DEEP_FAST/physics_r6/event_log.npz"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/final_task_completion_v1/02_handoff_to_transport_grasp"
    / "H04_INTEGRATED_POSTHANDOFF_TO_R14"
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
    parser.add_argument(
        "--conversion-order",
        choices=("simultaneous", "wrist_then_fingers", "fingers_then_wrist"),
        default="simultaneous",
        help=(
            "Bounded handoff-to-transport-grasp conversion order.  Endpoints, "
            "physics, and the final R14 grasp are identical for every order."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = read_json(CONFIG)
    selected = read_json(SELECTED)
    t5_report = read_json(T5_REPORT)
    expected = {
        HANDOFF_SOURCE: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
        T5_TAIL: "7399ed969e78c8542d0aaf12ed372f4ae47fe949f3941efcf5d1b572df4e73f5",
        CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
        H07_EVENT: "75d40b8c03d7e4de16dae2a1dd35663cb42f10b72cb15bc5af5bdfb8b1175ca9",
        R14_EVENT: "75c4c4d9da333c23f78ebd7c5ee6a3c63e35c786fbe245788c1a92fc80efe02b",
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"verified dependency changed: {path}")
    if selected.get("status") != "PASS" or t5_report.get("status") != "OFFLINE_PASS":
        raise RuntimeError("selected grasp or T5 clearance evidence is not valid")

    with np.load(HANDOFF_SOURCE, allow_pickle=False) as archive:
        handoff_commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        handoff_stages = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
        declared_handoff_object = np.asarray(
            archive["handoff_object_center_world_m"], dtype=np.float64
        )
    with np.load(T5_TAIL, allow_pickle=False) as archive:
        tail_commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        tail_stages = archive["stage"].astype(str)
        tail_names = archive["joint_names"].astype(str).tolist()
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or tail_names != names or not np.isclose(fps, 30.0):
        raise RuntimeError("authoritative 28D interface changed")
    if not np.array_equal(handoff_commands[539], tail_commands[0]):
        raise RuntimeError("verified keyframe command splice is not exact")
    clearance_end = int(np.flatnonzero(tail_stages == "RIGHT_TRANSPORT_GRIP_VERIFICATION")[-1] + 1)
    if clearance_end != 85:
        raise RuntimeError("verified T5 clearance prefix changed")

    # Continuous source through the end of its one-second RIGHT-only hold,
    # followed by T5 frames 1..84 (frame 0 is the identical splice state).
    rows = [row.copy() for row in handoff_commands[:540]]
    stages = handoff_stages[:540].tolist()
    rows.extend(row.copy() for row in tail_commands[1:clearance_end])
    stages.extend(tail_stages[1:clearance_end].tolist())
    inherited_end = len(rows)

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
    current_left = rows[-1][left_indices].copy()
    current_right = rows[-1][right_indices].copy()
    if not np.allclose(current_left, left_open, atol=1.0e-7):
        raise RuntimeError("T5 verified clearance does not leave LEFT open")

    # T5 proves a 45 mm radial exit. Before entering the object-centered R14
    # endpoint, move the already-open LEFT hand farther out of the shared
    # volume while the verified T4 wrap remains fixed. This does not alter the
    # handoff, doll, or RIGHT grasp endpoints.
    left_primitive_path = Path(config["source_arm_primitives"]["left"])
    with np.load(left_primitive_path, allow_pickle=False) as archive:
        left_primitive = {key: np.asarray(archive[key]) for key in archive.files}
    left_reference_arm = np.asarray(
        left_primitive["approach_arm_q_rad"][-1], dtype=np.float64
    )
    g1.assign(left_reference_arm, p1_left, p1_right)
    static_tool_left = (
        np.linalg.inv(g1.wrist_pose("left")) @ g1.whole_hand_grasp_pose("left")
    )
    g1.assign(current_arm, current_left, current_right)
    left_start_model, left_start_rotation_model, _, _ = g1.static_tool_pose_state(
        "left", static_tool_left
    )
    left_start_world = g1.model_to_world_position(left_start_model)
    left_retreat_world = left_start_world + np.asarray([-0.08, -0.06, 0.04])
    left_retreat_arm, left_retreat_reports = solve_path(
        g1,
        "left",
        static_tool_left,
        g1.world_to_model_position(
            minimum_jerk(left_start_world, left_retreat_world, 91)
        ),
        np.repeat(left_start_rotation_model[None], 91, axis=0),
        current_arm,
        left_open,
        current_right,
    )

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
    post_clear_object = np.asarray(
        t5_report["post_clearance_object_reference_world_m"], dtype=np.float64
    )
    # Recover the actual object-to-tool SE(3) of the selected, physically
    # successful R14 grasp. Apply that fixed transform to the stabilized H07
    # handoff object pose. This preserves grasp geometry despite the doll's
    # different post-handoff orientation; it is fixed offline provenance, not
    # runtime pose feedback.
    with np.load(R14_EVENT, allow_pickle=False) as archive:
        r14_mask = archive["stage"].astype(str) == "RIGHT_HIGH_STABILIZATION"
        r14_object_position = np.median(
            np.asarray(archive["object_position_world_m"], dtype=np.float64)[
                r14_mask
            ][-240:],
            axis=0,
        )
        r14_object_rotation = Rotation.from_quat(
            np.asarray(archive["object_quaternion_xyzw"], dtype=np.float64)[
                r14_mask
            ][-240:]
        ).mean().as_matrix()
        r14_command = np.asarray(
            archive["commanded_q_rad"], dtype=np.float64
        )[np.flatnonzero(r14_mask)[-1]]
    g1.assign(
        r14_command[arm_indices],
        r14_command[left_indices],
        r14_command[right_indices],
    )
    selected_tool_model, selected_tool_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool_right
    )
    selected_tool_world = g1.model_to_world_position(selected_tool_model)
    selected_tool_rotation_world = world_from_model @ selected_tool_rotation_model
    selected_object_to_tool_translation = r14_object_rotation.T @ (
        selected_tool_world - r14_object_position
    )
    selected_object_to_tool_rotation = (
        r14_object_rotation.T @ selected_tool_rotation_world
    )
    with np.load(H07_EVENT, allow_pickle=False) as archive:
        bridge_mask = archive["stage"].astype(str) == "RIGHT_P14_BRIDGE_STABILIZATION"
        bridge_object_reference = np.median(
            np.asarray(archive["object_position_world_m"], dtype=np.float64)[
                bridge_mask
            ],
            axis=0,
        )
        bridge_object_rotation = Rotation.from_quat(
            np.asarray(archive["object_quaternion_xyzw"], dtype=np.float64)[
                bridge_mask
            ]
        ).mean().as_matrix()
    p14_tool_world = post_clear_object + reference_tool_to_object
    transport_tool_world = bridge_object_reference + (
        bridge_object_rotation @ selected_object_to_tool_translation
    )
    transport_rotation_world = (
        bridge_object_rotation @ selected_object_to_tool_rotation
    )
    transport_rotation_model = world_from_model.T @ transport_rotation_world
    transport_tool_to_object = transport_tool_world - bridge_object_reference

    g1.assign(left_retreat_arm[-1], current_left, current_right)
    start_model, start_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool_right
    )
    start_world = g1.model_to_world_position(start_model)
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
        left_retreat_arm[-1],
        left_open,
        current_right,
    )
    p14_endpoint_arm = p14_arm[-1].copy()
    p14_endpoint_report = p14_reports[-1]
    # The Cartesian seed path crosses a wrist singularity even though both
    # endpoints are reachable. Use a smooth joint-space bridge between those
    # exact endpoint solutions; this preserves the arm branch continuously.
    p14_arm = minimum_jerk(left_retreat_arm[-1], p14_endpoint_arm, 241)
    r14_count = 121
    r14_arm, r14_reports = solve_path(
        g1,
        "right",
        static_tool_right,
        g1.world_to_model_position(
            minimum_jerk(p14_tool_world, transport_tool_world, r14_count)
        ),
        np.einsum(
            "ij,tjk->tik",
            world_from_model.T,
            rotations_between(
                reference_rotation_world,
                transport_rotation_world,
                r14_count,
            ),
        ),
        p14_arm[-1],
        left_open,
        right_transport,
    )
    r14_endpoint_arm = r14_arm[-1].copy()
    r14_endpoint_report = r14_reports[-1]
    # Retain the collision-aware Cartesian branch, but remove its isolated IK
    # sample switches with a bounded zero-phase joint-space regularization.
    # Direct endpoint interpolation was rejected in H08 because it crossed the
    # torso. This stays locally around the collision-free Cartesian path and
    # restores the exact start/end states after filtering.
    r14_raw_arm = r14_arm.copy()
    r14_arm = gaussian_filter1d(r14_raw_arm, sigma=4.0, axis=0, mode="nearest")
    r14_progress = np.linspace(0.0, 1.0, len(r14_arm))
    r14_blend = np.asarray(
        [10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5 for u in r14_progress]
    )
    r14_arm += (
        (1.0 - r14_blend)[:, None] * (r14_raw_arm[0] - r14_arm[0])[None]
        + r14_blend[:, None] * (r14_raw_arm[-1] - r14_arm[-1])[None]
    )

    def append(arm: np.ndarray, left: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left, right))
        stages.append(stage)

    for arm in left_retreat_arm[1:]:
        append(arm, left_open, current_right, "LEFT_FINAL_SAFE_RETREAT")
    for arm in p14_arm[1:]:
        append(arm, left_open, current_right, "RIGHT_POSTHANDOFF_APPROACH_P14")
    for _ in range(30):
        append(
            p14_arm[-1],
            left_open,
            current_right,
            "RIGHT_P14_BRIDGE_STABILIZATION",
        )
    right_transition = minimum_jerk(current_right, right_transport, 91)
    if args.conversion_order == "simultaneous":
        simultaneous_right = minimum_jerk(current_right, right_transport, r14_count)
        for arm, right in zip(r14_arm[1:], simultaneous_right[1:], strict=True):
            append(arm, left_open, right, "RIGHT_CONVERGE_TO_TRANSPORT_GRASP")
    elif args.conversion_order == "wrist_then_fingers":
        # Preserve the physically validated T4 enclosure while the wrist moves
        # to the selected R14 object-relative endpoint.  Only after a brief
        # stabilization does the hand converge to the exact, immutable R14 7D
        # transport vector.
        for arm in r14_arm[1:]:
            append(arm, left_open, current_right, "RIGHT_R14_WRIST_ALIGNMENT")
        for _ in range(30):
            append(
                r14_arm[-1],
                left_open,
                current_right,
                "RIGHT_R14_WRIST_ALIGNMENT_STABILIZATION",
            )
        for right in right_transition[1:]:
            append(
                r14_arm[-1],
                left_open,
                right,
                "RIGHT_R14_FINGER_ENCLOSURE",
            )
    else:
        # Complementary bounded diagnostic: establish the immutable R14 hand
        # vector at the stable bridge pose before changing wrist SE(3).
        for right in right_transition[1:]:
            append(
                p14_arm[-1],
                left_open,
                right,
                "RIGHT_R14_FINGER_ENCLOSURE",
            )
        for _ in range(30):
            append(
                p14_arm[-1],
                left_open,
                right_transport,
                "RIGHT_R14_FINGER_ENCLOSURE_STABILIZATION",
            )
        for arm in r14_arm[1:]:
            append(arm, left_open, right_transport, "RIGHT_R14_WRIST_ALIGNMENT")
    for _ in range(30):
        append(
            r14_arm[-1],
            left_open,
            right_transport,
            "RIGHT_TRANSPORT_GRASP_VERIFICATION",
        )

    high_object = bridge_object_reference + np.asarray([0.0, 0.0, 0.100])
    high_tool = high_object + transport_tool_to_object
    vertical_arm, vertical_reports = solve_path(
        g1,
        "right",
        static_tool_right,
        g1.world_to_model_position(
            minimum_jerk(transport_tool_world, high_tool, 121)
        ),
        np.repeat(transport_rotation_model[None], 121, axis=0),
        r14_arm[-1],
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

    full_reports: list[dict[str, Any]] = []
    if args.mode == "full":
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
                short_arm[-1], left_open, right_transport, "RIGHT_SHORT_STABILIZATION"
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
            append(bin_arm[-1], left_open, right_transport, "RIGHT_HOLD_OVER_BIN")

        release_object = np.r_[bin_xy, 1.000]
        release_tool = release_object + transport_tool_to_object
        descent_arm, descent_reports = solve_path(
            g1,
            "right",
            static_tool_right,
            g1.world_to_model_position(minimum_jerk(bin_tool, release_tool, 61)),
            np.repeat(transport_rotation_model[None], 61, axis=0),
            bin_arm[-1],
            left_open,
            right_transport,
        )
        for arm in descent_arm[1:]:
            append(arm, left_open, right_transport, "RIGHT_CONTROLLED_BIN_DESCENT")
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
        full_reports = short_reports + bin_reports + descent_reports

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
    # P14 is only a joint-space guide waypoint after the Cartesian solver
    # exposed a singular direct path. The physical contract is the selected
    # R14 endpoint, which is solved essentially exactly, plus all subsequent
    # Cartesian transport endpoints.
    reports = (
        left_retreat_reports
        + [r14_endpoint_report]
        + vertical_reports
        + full_reports
    )
    maximum_position_error = max(float(row["position_error_m"]) for row in reports)
    maximum_orientation_error = max(
        float(row["orientation_error_rad"]) for row in reports
    )
    arm_steps = np.abs(np.diff(commands[:, arm_indices], axis=0))
    maximum_arm_step = float(np.max(arm_steps, initial=0.0))
    maximum_new_arm_step = float(
        np.max(arm_steps[inherited_end - 1 :], initial=0.0)
    )
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and maximum_position_error <= 0.001
        and maximum_orientation_error <= 0.02
        and maximum_new_arm_step <= 0.03
    )

    command_path = output / f"integrated_handoff_transport_{args.mode}_command.npz"
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
            "debug_keyframe_restoration_used": np.asarray(False),
            "selected_right_transport_hand_model_order_7d_rad": right_transport.astype(np.float32),
            "transport_grasp_conversion_order": np.asarray(args.conversion_order),
            "post_clearance_object_reference_world_m": post_clear_object.astype(np.float32),
            "bridge_object_reference_world_m": bridge_object_reference.astype(np.float32),
            "bridge_object_rotation_world": bridge_object_rotation.astype(np.float32),
            "selected_object_to_tool_translation_object_m": selected_object_to_tool_translation.astype(np.float32),
            "selected_object_to_tool_rotation_object": selected_object_to_tool_rotation.astype(np.float32),
        },
    )
    report = {
        "schema_version": "final_integrated_handoff_transport_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "mode": args.mode,
        "transport_grasp_conversion_order": args.conversion_order,
        "construction": [
            "exact continuous frame-zero handoff through 1 s RIGHT-only ownership",
            "exact T5 LEFT radial-clearance command segment, integrated without restoration",
            f"bounded P14-to-selected-R14 endpoint convergence ({args.conversion_order})",
            "selected R14 vertical and transport construction",
        ],
        "handoff_source": str(HANDOFF_SOURCE),
        "handoff_source_sha256": sha256_file(HANDOFF_SOURCE),
        "t5_tail_source": str(T5_TAIL),
        "t5_tail_source_sha256": sha256_file(T5_TAIL),
        "selected_right_grasp": str(SELECTED),
        "selected_right_grasp_sha256": sha256_file(SELECTED),
        "splice_command_exact": True,
        "post_clearance_object_reference_world_m": post_clear_object,
        "bridge_object_reference_world_m": bridge_object_reference,
        "bridge_object_reference_event": str(H07_EVENT),
        "bridge_object_reference_event_sha256": sha256_file(H07_EVENT),
        "bridge_object_rotation_world": bridge_object_rotation,
        "selected_r14_event": str(R14_EVENT),
        "selected_r14_event_sha256": sha256_file(R14_EVENT),
        "selected_object_to_tool_translation_object_m": selected_object_to_tool_translation,
        "selected_object_to_tool_rotation_object": selected_object_to_tool_rotation,
        "transport_tool_target_world_m": transport_tool_world,
        "selected_right_transport_hand_model_order_7d_rad": right_transport,
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_ik_position_error_m": maximum_position_error,
            "maximum_ik_orientation_error_rad": maximum_orientation_error,
            "maximum_adjacent_arm_step_rad": maximum_arm_step,
            "maximum_inherited_arm_step_rad": float(
                np.max(arm_steps[: inherited_end - 1], initial=0.0)
            ),
            "maximum_new_adjacent_arm_step_rad": maximum_new_arm_step,
            "maximum_new_adjacent_arm_step_gate_rad": 0.03,
            "p14_endpoint_ik": p14_endpoint_report,
            "r14_endpoint_ik": r14_endpoint_report,
        },
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "state_restoration_used": False,
        "runtime_object_feedback_used": False,
        "doll_or_material_changed": False,
        "controller_gains_changed": False,
        "prohibited_mechanisms_used": False,
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(output / f"offline_{args.mode}_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
