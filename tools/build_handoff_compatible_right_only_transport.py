#!/usr/bin/env python3
"""Build a RIGHT-only transport test from a proven handoff-owned grasp.

The seed object-to-tool transform and 7D hand target are recovered from the
stable, table-free RIGHT-only retention segment of R26.  A trial starts from a
normal scene reset and physically closes on the object; the seed state is not
restored.  Optional deltas are a small, predeclared local fallback family.
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
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import (  # noqa: E402
    canonical_row,
    hand_model,
    minimum_jerk,
    solve_bounded_pose,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import AUTHORITATIVE_REFERENCES, authoritative_joint_ranges  # noqa: E402


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
R26_COMMAND = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
    / "R26_HANDOFF_R14_INDEX0_P010/continuous_r14_hand_t5_full_command.npz"
)
R26_EVENT = R26_COMMAND.parent / "physics_full/event_log.npz"
R14_EVENT = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
    / "R14_RELEASE_D1_CENTERED_DEEP_FAST/physics_r6/event_log.npz"
)
HANDOFF_SOURCE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task/p14_bilateral"
    / "backward_constructed_handoff/B2_PATH_F40/right_preload_partial_left_relax_exact_endpoint_v4"
    / "full/scripted_full_task_command.npz"
)
EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    R26_COMMAND: "2c525c492baaf88a47270679bc112d35a15f721933f0d62365b54625f0f4805e",
    R26_EVENT: "d96a362c341f6bd551ed607a908ae8986908fe9b55164ec531ff73199577d956",
    R14_EVENT: "75c4c4d9da333c23f78ebd7c5ee6a3c63e35c786fbe245788c1a92fc80efe02b",
    HANDOFF_SOURCE: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
}
DEFAULT_ROOT = (
    ROOT
    / "outputs/final_methodology_preserving_completion"
    / "03_fallback_transport_grasp"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--radial-offset-mm", type=float, default=0.0)
    parser.add_argument("--local-roll-deg", type=float, default=0.0)
    parser.add_argument("--local-pitch-deg", type=float, default=0.0)
    parser.add_argument("--local-yaw-deg", type=float, default=0.0)
    parser.add_argument("--thumb1-delta-rad", type=float, default=0.0)
    parser.add_argument("--middle0-delta-rad", type=float, default=0.0)
    parser.add_argument("--pregrasp-scale", type=float, default=1.0)
    parser.add_argument(
        "--transport-object-z-m",
        type=float,
        help=(
            "Optional common bin-clearance object-COM height. It changes only "
            "the arm path and preserves the candidate grasp, doll, and gains."
        ),
    )
    parser.add_argument(
        "--r14-to-r26-alpha",
        type=float,
        default=1.0,
        help="Structured transform/hand interpolation: 0=verified R14, 1=R26 ownership.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if abs(args.radial_offset_mm) > 5.0:
        raise ValueError("fallback radial offset is bounded to +/-5 mm")
    if max(abs(args.local_roll_deg), abs(args.local_pitch_deg), abs(args.local_yaw_deg)) > 5.0:
        raise ValueError("fallback local orientation deltas are bounded to +/-5 deg")
    if max(abs(args.thumb1_delta_rad), abs(args.middle0_delta_rad)) > 0.05:
        raise ValueError("fallback finger deltas are bounded to +/-0.05 rad")
    if not 0.0 <= args.r14_to_r26_alpha <= 1.0:
        raise ValueError("--r14-to-r26-alpha must be in [0,1]")
    if not 1.0 <= args.pregrasp_scale <= 2.5:
        raise ValueError("--pregrasp-scale must be in [1,2.5]")
    output = (args.output or (DEFAULT_ROOT / "candidates" / args.candidate_id)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable seed changed: {path}: {actual}")

    config = read_json(CONFIG)
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    names, _ = authoritative_joint_ranges()
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    left_open = hand_model(g1, names, config, "left", "OPEN")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    right_preshape = hand_model(g1, names, config, "right", "PRESHAPE")
    p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
    p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")

    primitive_path = Path(config["source_arm_primitives"]["right"])
    with np.load(primitive_path, allow_pickle=False) as archive:
        primitive = {key: np.asarray(archive[key]) for key in archive.files}
    reference_arm = np.asarray(primitive["approach_arm_q_rad"][-1], dtype=np.float64)
    g1.assign(reference_arm, p1_left, p1_right)
    static_tool = np.linalg.inv(g1.wrist_pose("right")) @ g1.whole_hand_grasp_pose("right")

    with np.load(R26_EVENT, allow_pickle=False) as archive:
        stage = archive["stage"].astype(str)
        stable = stage == "RIGHT_POST_RELEASE_RETENTION"
        if np.count_nonzero(stable) != 240:
            raise RuntimeError("R26 stable ownership segment changed")
        stable_rows = np.flatnonzero(stable)[-120:]
        command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        object_position = np.median(
            np.asarray(archive["object_position_world_m"], dtype=np.float64)[stable_rows],
            axis=0,
        )
        object_rotation = Rotation.from_quat(
            np.asarray(archive["object_quaternion_xyzw"], dtype=np.float64)[stable_rows]
        ).mean().as_matrix()
        seed_q = command[stable_rows[-1]]
        seed_forces = {
            digit: float(
                np.mean(np.asarray(archive[f"{digit}_force_n"], dtype=np.float64)[stable_rows])
            )
            for digit in ("thumb", "index", "middle")
        }
        seed_speed = float(
            np.max(
                np.linalg.norm(
                    np.asarray(archive["object_linear_velocity_m_s"], dtype=np.float64)[stable_rows],
                    axis=1,
                )
            )
        )
        seed_table_force = float(
            np.max(np.asarray(archive["table_contact_force_n"], dtype=np.float64)[stable_rows])
        )

    right_name_lookup = {
        name: index for index, name in enumerate(g1.hand_joint_names["right"])
    }
    origin = g1.model_to_world_position(np.zeros(3, dtype=np.float64))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3, dtype=np.float64)[axis]) - origin
            for axis in range(3)
        ]
    )
    with np.load(R14_EVENT, allow_pickle=False) as archive:
        r14_stage = archive["stage"].astype(str)
        r14_command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        r14_q = r14_command[np.flatnonzero(r14_stage == "GRAVITY_RETENTION")[-1]]
        r14_pregrasp_q = r14_command[np.flatnonzero(r14_stage == "RIGHT_OPEN")[-1]]
    with np.load(R26_COMMAND, allow_pickle=False) as archive:
        r26_stage = archive["stage"].astype(str)
        r26_command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        r26_q = r26_command[
            np.flatnonzero(r26_stage == "RIGHT_SUPPORTED_R14_HAND_ENCLOSURE")[-1]
        ]
    with np.load(HANDOFF_SOURCE, allow_pickle=False) as archive:
        handoff_object_center = np.asarray(
            archive["handoff_object_center_world_m"], dtype=np.float64
        )

    def commanded_tool_world(q: np.ndarray) -> np.ndarray:
        g1.assign(q[arm_indices], q[left_indices], q[right_indices])
        position_model, rotation_model, _, _ = g1.static_tool_pose_state(
            "right", static_tool
        )
        value = np.eye(4, dtype=np.float64)
        value[:3, 3] = g1.model_to_world_position(position_model)
        value[:3, :3] = world_from_model @ rotation_model
        return value

    # Use the commanded acquisition geometry.  The settled post-lift offset is
    # retained only as physical support evidence: after a doll seats inside a
    # closed hand it is not a valid OPEN->PRESHAPE table-approach target.
    r14_tool_world = commanded_tool_world(r14_q)
    r14_pregrasp_tool_world = commanded_tool_world(r14_pregrasp_q)
    g1.assign(r14_q[arm_indices], r14_q[left_indices], r14_q[right_indices])
    r14_object_world = np.eye(4, dtype=np.float64)
    r14_object_world[:3, 3] = np.asarray(
        [
            *config["object"]["center_world_xy_m_by_side"]["right"],
            float(config["object"]["table_surface_world_z_m"])
            + 0.5 * float(config["object"]["visual_dimensions_m"][2]),
        ],
        dtype=np.float64,
    )
    r14_object_to_tool = np.linalg.inv(r14_object_world) @ r14_tool_world
    r26_tool_world = commanded_tool_world(r26_q)
    r26_object_world = np.eye(4, dtype=np.float64)
    r26_object_world[:3, 3] = handoff_object_center
    r26_object_to_tool = np.linalg.inv(r26_object_world) @ r26_tool_world
    alpha = float(args.r14_to_r26_alpha)
    blended_object_to_tool = np.eye(4, dtype=np.float64)
    blended_object_to_tool[:3, 3] = (
        (1.0 - alpha) * r14_object_to_tool[:3, 3]
        + alpha * r26_object_to_tool[:3, 3]
    )
    blended_object_to_tool[:3, :3] = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(
            np.stack([r14_object_to_tool[:3, :3], r26_object_to_tool[:3, :3]])
        ),
    )([alpha]).as_matrix()[0]
    object_to_tool = blended_object_to_tool
    right_hand = (1.0 - alpha) * r14_q[right_indices] + alpha * r26_q[right_indices]
    right_hand[right_name_lookup["right_hand_thumb_1_joint"]] += args.thumb1_delta_rad
    right_hand[right_name_lookup["right_hand_middle_0_joint"]] += args.middle0_delta_rad
    limits = np.asarray(g1.hand_limits["right"], dtype=np.float64)
    if np.any(right_hand < limits[:, 0]) or np.any(right_hand > limits[:, 1]):
        raise RuntimeError("local fallback hand target exceeds named limits")

    radial = object_to_tool[:3, 3]
    radial_direction = radial / max(np.linalg.norm(radial), 1.0e-12)
    object_to_tool[:3, 3] += 0.001 * args.radial_offset_mm * radial_direction
    local_delta = Rotation.from_euler(
        "xyz",
        [args.local_roll_deg, args.local_pitch_deg, args.local_yaw_deg],
        degrees=True,
    ).as_matrix()
    object_to_tool[:3, :3] = object_to_tool[:3, :3] @ local_delta

    spawn_object_world = np.eye(4, dtype=np.float64)
    spawn_object_world[:3, 3] = np.asarray(
        [
            *config["object"]["center_world_xy_m_by_side"]["right"],
            float(config["object"]["table_surface_world_z_m"])
            + 0.5 * float(config["object"]["visual_dimensions_m"][2]),
        ],
        dtype=np.float64,
    )
    grasp_tool_world = spawn_object_world @ object_to_tool
    grasp_rotation_model = world_from_model.T @ grasp_tool_world[:3, :3]
    grasp_arm, grasp_report = solve_bounded_pose(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(grasp_tool_world[:3, 3]),
        grasp_rotation_model,
        reference_arm,
        reference_arm[7:],
        left_open,
        right_hand,
        20260831,
    )

    pregrasp_tool_world = grasp_tool_world.copy()
    candidate_from_r14_rotation = (
        grasp_tool_world[:3, :3] @ r14_tool_world[:3, :3].T
    )
    pregrasp_tool_world[:3, 3] += args.pregrasp_scale * candidate_from_r14_rotation @ (
        r14_pregrasp_tool_world[:3, 3] - r14_tool_world[:3, 3]
    )
    reverse_pregrasp, reverse_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(grasp_tool_world[:3, 3], pregrasp_tool_world[:3, 3], 46)
        ),
        np.repeat(grasp_rotation_model[None], 46, axis=0),
        grasp_arm,
        left_open,
        right_open,
    )
    pregrasp_arm = reverse_pregrasp[-1]
    # The reverse path is already a continuous, collision-aware solve from the
    # exact grasp branch to pregrasp.  Reversing those same samples preserves
    # that branch exactly; a second forward solve can independently choose a
    # different redundant-arm branch and create an artificial endpoint jump.
    approach_arm = minimum_jerk(pregrasp_arm, grasp_arm, 46)
    approach_reports = [reverse_reports[-1], grasp_report]

    lift_tool = grasp_tool_world.copy()
    lift_tool[2, 3] += 0.063
    lift_arm, lift_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(grasp_tool_world[:3, 3], lift_tool[:3, 3], 61)
        ),
        np.repeat(grasp_rotation_model[None], 61, axis=0),
        grasp_arm,
        left_open,
        right_hand,
    )
    lift_endpoint_report = lift_reports[-1]
    lift_arm = minimum_jerk(grasp_arm, lift_arm[-1], 61)
    high_tool = lift_tool.copy()
    high_tool[2, 3] += 0.100
    vertical_arm, vertical_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(minimum_jerk(lift_tool[:3, 3], high_tool[:3, 3], 181)),
        np.repeat(grasp_rotation_model[None], 181, axis=0),
        lift_arm[-1],
        left_open,
        right_hand,
    )
    vertical_endpoint_report = vertical_reports[-1]
    vertical_arm = minimum_jerk(lift_arm[-1], vertical_arm[-1], 181)

    tool_to_object_translation = grasp_tool_world[:3, 3] - spawn_object_world[:3, 3]
    high_object = high_tool[:3, 3] - tool_to_object_translation
    bin_xy = np.asarray(scene["bin"]["center_world_xy_m"], dtype=np.float64)
    bin_xy += np.asarray([0.015, -0.010])
    direction = bin_xy - high_object[:2]
    direction /= max(np.linalg.norm(direction), 1.0e-12)
    short_object = high_object.copy()
    short_object[:2] += 0.075 * direction
    short_object[2] = max(short_object[2], 1.012)
    if args.transport_object_z_m is not None:
        short_object[2] = float(args.transport_object_z_m)
    short_tool = short_object + tool_to_object_translation
    short_arm, short_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(minimum_jerk(high_tool[:3, 3], short_tool, 181)),
        np.repeat(grasp_rotation_model[None], 181, axis=0),
        vertical_arm[-1],
        left_open,
        right_hand,
    )
    short_endpoint_report = short_reports[-1]
    short_arm = minimum_jerk(vertical_arm[-1], short_arm[-1], 181)
    bin_object = np.r_[bin_xy, max(1.035, short_object[2])]
    bin_tool = bin_object + tool_to_object_translation
    bin_arm, bin_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(minimum_jerk(short_tool, bin_tool, 301)),
        np.repeat(grasp_rotation_model[None], 301, axis=0),
        short_arm[-1],
        left_open,
        right_hand,
    )
    bin_endpoint_report = bin_reports[-1]
    bin_arm = minimum_jerk(short_arm[-1], bin_arm[-1], 301)
    release_object = bin_object.copy()
    release_object[2] = 1.0
    release_tool = release_object + tool_to_object_translation
    descent_arm, descent_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(minimum_jerk(bin_tool, release_tool, 91)),
        np.repeat(grasp_rotation_model[None], 91, axis=0),
        bin_arm[-1],
        left_open,
        right_hand,
    )
    descent_endpoint_report = descent_reports[-1]
    descent_arm = minimum_jerk(bin_arm[-1], descent_arm[-1], 91)

    rows: list[np.ndarray] = []
    stages: list[str] = []

    def append(arm: np.ndarray, hand: np.ndarray, stage_name: str) -> None:
        rows.append(canonical_row(names, g1, arm, left_open, hand))
        stages.append(stage_name)

    for _ in range(15):
        append(pregrasp_arm, right_open, "RIGHT_OPEN")
    preshape = minimum_jerk(right_open, right_preshape, 46)
    for arm, hand in zip(approach_arm[1:], preshape[1:], strict=True):
        append(arm, hand, "RIGHT_PRESHAPE_APPROACH")
    for hand in minimum_jerk(right_preshape, right_hand, 61)[1:]:
        append(grasp_arm, hand, "RIGHT_POWER_GRASP")
    for _ in range(45):
        append(grasp_arm, right_hand, "GRAVITY_RETENTION")
    for arm in lift_arm[1:]:
        append(arm, right_hand, "RIGHT_INITIAL_LIFT")
    for _ in range(45):
        append(lift_arm[-1], right_hand, "HOLD_ELEVATED")
    for arm in vertical_arm[1:]:
        append(arm, right_hand, "RIGHT_VERTICAL_100MM")
    for _ in range(45):
        append(vertical_arm[-1], right_hand, "RIGHT_HIGH_STABILIZATION")
    for arm in short_arm[1:]:
        append(arm, right_hand, "RIGHT_SHORT_HORIZONTAL")
    for _ in range(45):
        append(short_arm[-1], right_hand, "RIGHT_SHORT_STABILIZATION")
    for arm in bin_arm[1:]:
        append(arm, right_hand, "RIGHT_TRANSPORT_TO_BIN")
    for _ in range(45):
        append(bin_arm[-1], right_hand, "RIGHT_HOLD_OVER_BIN")
    for arm in descent_arm[1:]:
        append(arm, right_hand, "RIGHT_CONTROLLED_BIN_DESCENT")
    for _ in range(30):
        append(descent_arm[-1], right_hand, "RIGHT_PRE_RELEASE_STABILIZATION")
    for hand in minimum_jerk(right_hand, right_open, 16)[1:]:
        append(descent_arm[-1], hand, "RIGHT_RELEASE")
    for _ in range(120):
        append(descent_arm[-1], right_open, "POST_RELEASE")

    commands = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (commands < lower[None] - 1.0e-9) | (commands > upper[None] + 1.0e-9)
    geometry = g1.trajectory_geometry(
        commands[:, arm_indices], commands[:, left_indices], commands[:, right_indices], 1.0e-5
    )
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()
    }
    # The execution paths are continuous joint-space minimum-jerk bridges
    # between collision-audited Cartesian endpoints.  Intermediate redundant
    # IK samples are diagnostics only and are not executed; scoring them would
    # reintroduce the branch-switch artifact this builder explicitly removes.
    reports = [
        grasp_report,
        reverse_reports[-1],
        lift_endpoint_report,
        vertical_endpoint_report,
        short_endpoint_report,
        bin_endpoint_report,
        descent_endpoint_report,
    ]
    max_position_error = max(float(row["position_error_m"]) for row in reports)
    max_orientation_error = max(float(row["orientation_error_rad"]) for row in reports)
    max_arm_step = float(np.max(np.abs(np.diff(commands[:, arm_indices], axis=0)), initial=0.0))
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and max_position_error <= 0.001
        and max_orientation_error <= 0.02
        and max_arm_step <= 0.03
    )
    command_path = output / "right_only_transport_to_bin_command.npz"
    save_npz(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": labels,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(30.0),
            "candidate_id": np.asarray(args.candidate_id),
            "candidate_right_hand_model_order_7d_rad": right_hand.astype(np.float32),
            "transport_object_clearance_z_m": np.asarray(
                short_object[2], dtype=np.float32
            ),
            "seed_object_to_tool_4x4": object_to_tool.astype(np.float64),
            "policy_independent": np.asarray(True),
            "state_restoration_used": np.asarray(False),
            "object_pose_writes": np.asarray(0),
            "runtime_object_feedback_used": np.asarray(False),
            "attachment_used": np.asarray(False),
            "object_follow_used": np.asarray(False),
        },
    )
    report = {
        "schema_version": "handoff_compatible_right_only_transport_candidate_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "candidate_id": args.candidate_id,
        "seed": {
            "source": "R26 stable RIGHT-only post-release retention",
            "command": str(R26_COMMAND),
            "command_sha256": sha256_file(R26_COMMAND),
            "event_log": str(R26_EVENT),
            "event_log_sha256": sha256_file(R26_EVENT),
            "mean_force_n": seed_forces,
            "maximum_speed_m_s": seed_speed,
            "maximum_table_force_n": seed_table_force,
            "object_to_tool_4x4": object_to_tool,
            "right_hand_model_order_7d_rad": right_hand,
        },
        "local_variation": {
            "r14_to_r26_alpha": alpha,
            "pregrasp_scale": args.pregrasp_scale,
            "radial_offset_mm": args.radial_offset_mm,
            "local_rpy_deg": [args.local_roll_deg, args.local_pitch_deg, args.local_yaw_deg],
            "thumb1_delta_rad": args.thumb1_delta_rad,
            "middle0_delta_rad": args.middle0_delta_rad,
            "transport_object_clearance_z_m": short_object[2],
        },
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_ik_position_error_m": max_position_error,
            "maximum_ik_orientation_error_rad": max_orientation_error,
            "maximum_adjacent_arm_step_rad": max_arm_step,
            "endpoint_reports": {
                "grasp": grasp_report,
                "pregrasp": reverse_reports[-1],
                "initial_lift": lift_endpoint_report,
                "vertical": vertical_endpoint_report,
                "short_horizontal": short_endpoint_report,
                "bin": bin_endpoint_report,
                "descent": descent_endpoint_report,
            },
        },
        "frames": int(len(commands)),
        "duration_s": float(len(commands) / 30.0),
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "doll_or_material_changed": False,
        "controller_gains_changed": False,
        "state_restoration_used": False,
        "runtime_object_feedback_used": False,
        "prohibited_mechanisms_used": False,
        "policy_used": False,
        "real_robot_used": False,
    }
    atomic_json(output / "offline_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
