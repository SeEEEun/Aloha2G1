#!/usr/bin/env python3
"""Build a debug-only tail from the verified post-handoff keyframe."""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

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
KEYFRAME = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "success_first_common_execution/post_handoff_keyframe/POST_HANDOFF_KEYFRAME.npz"
)
KEYFRAME_MANIFEST = KEYFRAME.with_name("POST_HANDOFF_KEYFRAME_MANIFEST.json")
V9 = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral/backward_constructed_handoff"
    / "B2_PATH_F40/left_object_radial_clearance45_then_right_p14_v9/full"
)
V9_COMMAND = V9 / "scripted_full_task_command.npz"
V9_EVENT = V9 / "physics_right_sensor/event_log.npz"
EXPECTED_V9_COMMAND_SHA256 = (
    "eccae4213378b6e6a6e406a687f43a4d753b8010a1d7b0395bc7b60f8ad405f2"
)
EXPECTED_V9_EVENT_SHA256 = (
    "ea7bdbac5f6e2c91975dfe0982d181d4f5e38db22f69f7e9eb42d0736e6ec4dc"
)
T5_EVENT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "success_first_common_execution/tail_debug"
    / "t5_t4_grip_robust_final_transport_v1/physics_right_sensor/event_log.npz"
)
EXPECTED_T5_EVENT_SHA256 = (
    "4b1964b164cc6fb6c3d0e76c90764ad28b91eac57aa3af9a4d267325dec04ecc"
)
OUTPUT_ROOT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "success_first_common_execution/tail_debug"
)
TRANSPORT_GRIP_PROFILES: dict[str, dict[str, Any]] = {
    "INDEX08": {
        "candidate": "T2_INDEX08_SLOW_TRANSPORT",
        "directory": "t2_index08_slow_transport",
        "joint_delta_rad": {"right_hand_index_0_joint": 0.08},
    },
    "ALL_SMALL": {
        "candidate": "T3_ALL_SMALL_SLOW_TRANSPORT",
        "directory": "t3_all_small_slow_transport",
        "joint_delta_rad": {
            "right_hand_thumb_1_joint": 0.04,
            "right_hand_index_0_joint": 0.08,
            "right_hand_middle_0_joint": 0.04,
        },
    },
    "ALL_MEDIUM": {
        "candidate": "T4_THREE_DIGIT_WRAP_SLOW_TRANSPORT",
        "directory": "t4_three_digit_wrap_slow_transport",
        "joint_delta_rad": {
            "right_hand_thumb_1_joint": 0.04,
            "right_hand_index_0_joint": 0.18,
            "right_hand_index_1_joint": -0.08,
            "right_hand_middle_0_joint": 0.04,
        },
    },
}


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


def save_command(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport-grip-profile",
        choices=tuple(TRANSPORT_GRIP_PROFILES),
        default="INDEX08",
    )
    parser.add_argument(
        "--transport-path-profile",
        choices=(
            "LEGACY",
            "ROBUST_FINAL_V1",
            "ROBUST_FINAL_V2_QUASISTATIC",
            "ROBUST_FINAL_V3_SUPPORTIVE_CARRY",
            "ROBUST_FINAL_V4_WITHIN_RETENTION_HORIZON",
        ),
        default="LEGACY",
        help=(
            "ROBUST_FINAL_V1 preserves the T4 grasp, slows the 145 mm lift, "
            "uses the smallest bounded orientation deviation that removed the "
            "bin-path IK branch jump, and adds stabilization/descent stages."
        ),
    )
    args = parser.parse_args()
    profile = TRANSPORT_GRIP_PROFILES[args.transport_grip_profile]
    robust_final = args.transport_path_profile in {
        "ROBUST_FINAL_V1",
        "ROBUST_FINAL_V2_QUASISTATIC",
        "ROBUST_FINAL_V3_SUPPORTIVE_CARRY",
        "ROBUST_FINAL_V4_WITHIN_RETENTION_HORIZON",
    }
    quasistatic_horizontal = (
        args.transport_path_profile == "ROBUST_FINAL_V2_QUASISTATIC"
    )
    supportive_carry = (
        args.transport_path_profile == "ROBUST_FINAL_V3_SUPPORTIVE_CARRY"
    )
    within_retention_horizon = (
        args.transport_path_profile
        == "ROBUST_FINAL_V4_WITHIN_RETENTION_HORIZON"
    )
    if robust_final and args.transport_grip_profile != "ALL_MEDIUM":
        raise RuntimeError("ROBUST_FINAL_V1 requires the frozen successful T4 grip")
    output = OUTPUT_ROOT / (
        "t8_t4_grip_within_retention_horizon_v4"
        if within_retention_horizon
        else "t7_t4_grip_supportive_carry_v3"
        if supportive_carry
        else "t6_t4_grip_quasistatic_transport_v2"
        if quasistatic_horizontal
        else "t5_t4_grip_robust_final_transport_v1"
        if robust_final
        else profile["directory"]
    )
    keyframe_manifest = read_json(KEYFRAME_MANIFEST)
    if (
        keyframe_manifest["status"] != "POST_HANDOFF_KEYFRAME_VERIFIED"
        or sha256_file(KEYFRAME) != keyframe_manifest["keyframe_sha256"]
        or sha256_file(V9_COMMAND) != EXPECTED_V9_COMMAND_SHA256
        or sha256_file(V9_EVENT) != EXPECTED_V9_EVENT_SHA256
    ):
        raise RuntimeError("frozen tail-debug dependency changed")
    if supportive_carry and sha256_file(T5_EVENT) != EXPECTED_T5_EVENT_SHA256:
        raise RuntimeError("frozen T5 physical boundary log changed")
    config = read_json(CONFIG)
    names, _ = authoritative_joint_ranges()
    with np.load(KEYFRAME, allow_pickle=False) as archive:
        keyframe = {key: np.asarray(archive[key]) for key in archive.files}
    if keyframe["joint_names"].astype(str).tolist() != names:
        raise RuntimeError("keyframe named order changed")
    with np.load(V9_COMMAND, allow_pickle=False) as archive:
        v9_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        v9_stage = archive["stage"].astype(str)
        if archive["joint_names"].astype(str).tolist() != names:
            raise RuntimeError("v9 command named order changed")
        fps = float(np.asarray(archive["control_fps_hz"]).item())
        bin_object_xy = np.asarray(
            archive["bin_object_target_world_m"], dtype=np.float64
        )[:2]
    with np.load(V9_EVENT, allow_pickle=False) as archive:
        v9_physics_stage = archive["stage"].astype(str)
        mask = v9_physics_stage == "RIGHT_ACQUISITION_HOLD_AFTER_LEFT_CLEARANCE"
        post_clearance_object = np.median(
            np.asarray(archive["object_position_world_m"], dtype=np.float64)[mask],
            axis=0,
        )

    if not np.array_equal(v9_q[510].astype(np.float32), keyframe["commanded_q_rad"]):
        raise RuntimeError("tail prefix does not start at POST_HANDOFF_KEYFRAME")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    left_open = v9_q[510, left_indices]
    right_acquisition = v9_q[510, right_indices]
    right_transport_grip = right_acquisition.copy()
    right_name_to_index = {
        name: index for index, name in enumerate(g1.hand_joint_names["right"])
    }
    for name, delta in profile["joint_delta_rad"].items():
        right_transport_grip[right_name_to_index[name]] += float(delta)
    arm_start = v9_q[583, arm_indices]

    # Use the same frozen whole-hand tool definition as all earlier P14 work.
    p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
    p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")
    primitive_path = Path(config["source_arm_primitives"]["right"])
    with np.load(primitive_path, allow_pickle=False) as archive:
        reference_arm = np.asarray(archive["lift_arm_q_rad"][-1], dtype=np.float64)
    g1.assign(reference_arm, p1_left, p1_right)
    static_tool = np.linalg.inv(g1.wrist_pose("right")) @ g1.whole_hand_grasp_pose(
        "right"
    )
    g1.assign(arm_start, left_open, right_transport_grip)
    tool_position_model, tool_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool
    )
    origin = g1.model_to_world_position(np.zeros(3))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3)[axis]) - origin
            for axis in range(3)
        ]
    )
    tool_position_world = g1.model_to_world_position(tool_position_model)
    tool_rotation_world = world_from_model @ tool_rotation_model

    # The object must remain physically supported; this fixed displacement only
    # declares a robot waypoint and never writes or follows the runtime object.
    bin_transport_object = np.asarray(
        [bin_object_xy[0], bin_object_xy[1], 1.045], dtype=np.float64
    )
    vertical_object = post_clearance_object.copy()
    vertical_object[2] = bin_transport_object[2]
    vertical_tool = tool_position_world + (vertical_object - post_clearance_object)
    bin_tool = vertical_tool + (bin_transport_object - vertical_object)

    vertical_count = 241 if robust_final else 121
    vertical_arm, vertical_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            minimum_jerk(tool_position_world, vertical_tool, vertical_count)
        ),
        np.repeat(
            (world_from_model.T @ tool_rotation_world)[None],
            vertical_count,
            axis=0,
        ),
        arm_start,
        left_open,
        right_transport_grip,
    )
    horizontal_count = (
        46
        if within_retention_horizon
        else 721
        if quasistatic_horizontal
        else 241
        if robust_final
        else 181
    )
    horizontal_rotations = np.repeat(
        (world_from_model.T @ tool_rotation_world)[None],
        horizontal_count,
        axis=0,
    )
    orientation_deviation_fraction = 0.0
    orientation_deviation_rad = 0.0
    support_pitch_rad = 0.0
    support_yaw_fraction = 0.0
    tilt_arm = np.asarray([vertical_arm[-1]], dtype=np.float64)
    tilt_reports: list[dict[str, Any]] = []
    horizontal_positions_world = minimum_jerk(
        vertical_tool, bin_tool, horizontal_count
    )
    fixed_support_object_world = None
    fixed_object_to_tool_offset_world = None
    if robust_final:
        # The prior exact-orientation route developed a late 0.42 rad IK step.
        # A bounded offline audit tested fixed fractions of the orientation
        # difference to the already reachable frozen v9 bin posture.  The
        # smallest tested fraction satisfying max arm step < 0.03 rad was 0.05
        # (1.7064 degrees); the fingers and all physics remain unchanged.
        g1.assign(
            v9_q[790, arm_indices],
            v9_q[790, left_indices],
            v9_q[790, right_indices],
        )
        _, known_bin_rotation_model, _, _ = g1.static_tool_pose_state(
            "right", static_tool
        )
        start_rotation_model = world_from_model.T @ tool_rotation_world
        rotation_vector = Rotation.from_matrix(
            start_rotation_model.T @ known_bin_rotation_model
        ).as_rotvec()
        orientation_deviation_fraction = 0.05
        end_rotation_model = start_rotation_model @ Rotation.from_rotvec(
            orientation_deviation_fraction * rotation_vector
        ).as_matrix()
        orientation_deviation_rad = float(
            np.linalg.norm(orientation_deviation_fraction * rotation_vector)
        )
        horizontal_rotations = rotations_between(
            start_rotation_model,
            end_rotation_model,
            horizontal_count,
        )
        if supportive_carry:
            # T5/T6 proved that time scaling does not prevent escape: the
            # frozen high-pose grasp has a ~2.5 s tangential-retention horizon.
            # Use the executed T5 stabilization COM only to construct a fixed
            # offline pivot.  There is no runtime object measurement/feedback.
            with np.load(T5_EVENT, allow_pickle=False) as archive:
                labels = archive["stage"].astype(str)
                stable_mask = labels == "RIGHT_VERTICAL_STABILIZATION"
                fixed_support_object_world = np.median(
                    np.asarray(
                        archive["object_position_world_m"], dtype=np.float64
                    )[stable_mask],
                    axis=0,
                )
            fixed_object_to_tool_offset_world = (
                fixed_support_object_world - vertical_tool
            )
            support_pitch_rad = float(np.radians(20.0))
            support_pitch_world = Rotation.from_rotvec(
                support_pitch_rad * np.asarray([0.0, 1.0, 0.0])
            ).as_matrix()
            support_rotation_world = support_pitch_world @ tool_rotation_world
            support_rotation_model = world_from_model.T @ support_rotation_world
            support_tool_world = (
                fixed_support_object_world
                - support_pitch_world @ fixed_object_to_tool_offset_world
            )
            tilt_count = 61
            tilt_arm, tilt_reports = solve_path(
                g1,
                "right",
                static_tool,
                g1.world_to_model_position(
                    minimum_jerk(vertical_tool, support_tool_world, tilt_count)
                ),
                rotations_between(
                    start_rotation_model, support_rotation_model, tilt_count
                ),
                vertical_arm[-1],
                left_open,
                right_transport_grip,
            )

            # A four-candidate bounded audit over 25/50/75/100% of the known
            # reachable bin-yaw change found 25% to be the smallest continuous
            # branch (max step < 0.03 rad) when combined with the support pitch.
            known_bin_rotation_world = world_from_model @ known_bin_rotation_model
            known_yaw_vector_world = Rotation.from_matrix(
                known_bin_rotation_world @ tool_rotation_world.T
            ).as_rotvec()
            support_yaw_fraction = 0.25
            support_yaw_world = Rotation.from_rotvec(
                support_yaw_fraction * known_yaw_vector_world
            ).as_matrix()
            end_rotation_world = support_yaw_world @ support_rotation_world
            end_rotation_model = world_from_model.T @ end_rotation_world
            horizontal_rotations = rotations_between(
                support_rotation_model,
                end_rotation_model,
                horizontal_count,
            )
            object_centers = minimum_jerk(
                fixed_support_object_world,
                bin_transport_object,
                horizontal_count,
            )
            horizontal_positions_world = []
            for object_center, rotation_model in zip(
                object_centers, horizontal_rotations, strict=True
            ):
                rotation_world = world_from_model @ rotation_model
                delta_world = rotation_world @ tool_rotation_world.T
                horizontal_positions_world.append(
                    object_center
                    - delta_world @ fixed_object_to_tool_offset_world
                )
            horizontal_positions_world = np.asarray(horizontal_positions_world)
            bin_tool = horizontal_positions_world[-1]
            orientation_deviation_rad = float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        tool_rotation_world.T @ end_rotation_world
                    ).as_rotvec()
                )
            )
    horizontal_arm, horizontal_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(
            horizontal_positions_world
        ),
        horizontal_rotations,
        tilt_arm[-1],
        left_open,
        right_transport_grip,
    )
    descent_arm = np.asarray([horizontal_arm[-1]], dtype=np.float64)
    descent_reports: list[dict[str, Any]] = []
    descent_object = bin_transport_object.copy()
    descent_tool = bin_tool.copy()
    if robust_final:
        table_z = float(scene["table"]["surface_height_m"])
        bin_height = float(scene["bin"]["outer_dimensions_xyz_m"][2])
        collider_height = float(config["geometry_candidates"][0]["dimensions_m"][2])
        release_clearance = 0.005
        descent_object[2] = (
            table_z + bin_height + 0.5 * collider_height + release_clearance
        )
        if supportive_carry:
            end_rotation_world = world_from_model @ horizontal_rotations[-1]
            end_delta_world = end_rotation_world @ tool_rotation_world.T
            descent_tool = (
                descent_object
                - end_delta_world @ fixed_object_to_tool_offset_world
            )
        else:
            descent_tool = bin_tool + (descent_object - bin_transport_object)
        descent_count = 10 if within_retention_horizon else 61
        descent_arm, descent_reports = solve_path(
            g1,
            "right",
            static_tool,
            g1.world_to_model_position(
                minimum_jerk(bin_tool, descent_tool, descent_count)
            ),
            np.repeat(horizontal_rotations[-1][None], descent_count, axis=0),
            horizontal_arm[-1],
            left_open,
            right_transport_grip,
        )

    rows = [v9_q[510].copy()]
    stages = ["POST_HANDOFF_KEYFRAME"]
    for right in minimum_jerk(right_acquisition, right_transport_grip, 11)[1:]:
        rows.append(
            canonical_row(
                names,
                g1,
                v9_q[510, arm_indices],
                left_open,
                right,
            )
        )
        stages.append("RIGHT_TRANSPORT_GRIP_PRELOAD")
    for _ in range(15):
        rows.append(
            canonical_row(
                names,
                g1,
                v9_q[510, arm_indices],
                left_open,
                right_transport_grip,
            )
        )
        stages.append("RIGHT_TRANSPORT_GRIP_HOLD")
    for source_row in v9_q[540:584]:
        rows.append(
            canonical_row(
                names,
                g1,
                source_row[arm_indices],
                left_open,
                right_transport_grip,
            )
        )
        stages.append("LEFT_OBJECT_RADIAL_CLEARANCE")
    for _ in range(15):
        rows.append(
            canonical_row(
                names, g1, arm_start, left_open, right_transport_grip
            )
        )
        stages.append("RIGHT_TRANSPORT_GRIP_VERIFICATION")
    for arm in vertical_arm[1:]:
        rows.append(canonical_row(names, g1, arm, left_open, right_transport_grip))
        stages.append("RIGHT_TRANSPORT_VERTICAL_CLEARANCE")
    if robust_final:
        for _ in range(15):
            rows.append(
                canonical_row(
                    names, g1, vertical_arm[-1], left_open, right_transport_grip
                )
            )
            stages.append("RIGHT_VERTICAL_STABILIZATION")
    if supportive_carry:
        for arm in tilt_arm[1:]:
            rows.append(
                canonical_row(names, g1, arm, left_open, right_transport_grip)
            )
            stages.append("RIGHT_SUPPORTIVE_CARRY_PIVOT")
    for arm in horizontal_arm[1:]:
        rows.append(canonical_row(names, g1, arm, left_open, right_transport_grip))
        stages.append("RIGHT_TRANSPORT_TO_BIN")
    for _ in range(
        3 if within_retention_horizon else 15 if robust_final else 30
    ):
        rows.append(
            canonical_row(
                names, g1, horizontal_arm[-1], left_open, right_transport_grip
            )
        )
        stages.append("RIGHT_HOLD_OVER_BIN")
    if robust_final:
        for arm in descent_arm[1:]:
            rows.append(
                canonical_row(names, g1, arm, left_open, right_transport_grip)
            )
            stages.append("RIGHT_CONTROLLED_BIN_DESCENT")
        for _ in range(3 if within_retention_horizon else 15):
            rows.append(
                canonical_row(
                    names, g1, descent_arm[-1], left_open, right_transport_grip
                )
            )
            stages.append("RIGHT_PRE_RELEASE_STABILIZATION")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    release_count = 16 if within_retention_horizon else 46
    for right in minimum_jerk(
        right_transport_grip, right_open, release_count
    )[1:]:
        rows.append(canonical_row(names, g1, descent_arm[-1], left_open, right))
        stages.append("RIGHT_RELEASE")
    for _ in range(120):
        rows.append(
            canonical_row(names, g1, descent_arm[-1], left_open, right_open)
        )
        stages.append("BIN_SETTLE")

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
        vertical_reports + tilt_reports + horizontal_reports + descent_reports
    )
    max_position_error = max(float(row["position_error_m"]) for row in reports)
    max_orientation_error = max(float(row["orientation_error_rad"]) for row in reports)
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and max_position_error <= 0.001
        and max_orientation_error <= 0.02
    )
    command_path = output / "tail_command.npz"
    save_command(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": labels,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(fps),
            "state_restoration_debug_only": np.asarray(True),
            "final_continuous_validation": np.asarray(False),
            "runtime_right_three_digit_gate_required": np.asarray(False),
            "attachment_used": np.asarray(False),
            "object_follow_used": np.asarray(False),
            "runtime_object_pose_feedback_used": np.asarray(False),
            "post_clearance_object_reference_world_m": post_clearance_object.astype(
                np.float32
            ),
            "bin_transport_object_target_world_m": bin_transport_object.astype(
                np.float32
            ),
            "bin_release_object_target_world_m": descent_object.astype(np.float32),
            "transport_orientation_deviation_rad": np.asarray(
                orientation_deviation_rad
            ),
        },
    )
    report = {
        "schema_version": "success_first_post_handoff_tail_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "candidate": (
            "T8_T4_GRIP_WITHIN_RETENTION_HORIZON_V4"
            if within_retention_horizon
            else "T7_T4_GRIP_SUPPORTIVE_CARRY_V3"
            if supportive_carry
            else "T6_T4_GRIP_QUASISTATIC_TRANSPORT_V2"
            if quasistatic_horizontal
            else "T5_T4_GRIP_ROBUST_FINAL_TRANSPORT_V1"
            if robust_final
            else profile["candidate"]
        ),
        "construction": [
            "exact POST_HANDOFF_KEYFRAME command",
            "exact physically validated v9 LEFT 45 mm radial clearance",
            "bounded temporary RIGHT transport preload",
            (
                "8 s vertical minimum-jerk clearance to the existing 1.045 m bin target"
                if robust_final
                else "4 s vertical minimum-jerk clearance to the existing 1.045 m bin target"
            ),
            (
                "0.5 s terminal stabilization with the unchanged T4 grasp"
                if robust_final
                else "no separate terminal stabilization"
            ),
            (
                "2 s geometry-derived 20 degree object-centered supportive-carry pivot with the T4 finger hold unchanged"
                if supportive_carry
                else "no separate supportive-carry pivot"
            ),
            (
                "1.5 s continuous minimum-jerk bin traverse scheduled inside the repeatable 2.5 s physical retention horizon"
                if within_retention_horizon
                else "8 s object-relative minimum-jerk transport with the smallest bounded reachable 8.532 degree bin-yaw component"
                if supportive_carry
                else "24 s quasistatic minimum-jerk horizontal transport, time-scaled from the T5 physical 0.04 m/s contact-loss boundary, with the same 1.7064 degree orientation path"
                if quasistatic_horizontal
                else "8 s horizontal minimum-jerk transport with the minimum bounded 1.7064 degree orientation deviation"
                if robust_final
                else "6 s horizontal minimum-jerk transport"
            ),
            "0.5 s over-bin stabilization",
            (
                "0.3 s controlled descent to a geometry-derived 5 mm rim clearance"
                if within_retention_horizon
                else "2 s controlled descent to a geometry-derived 5 mm rim clearance"
                if robust_final
                else "no controlled descent"
            ),
            "1.5 s physical hand opening",
            "4 s free bin settle",
        ],
        "post_clearance_object_reference_world_m": post_clearance_object,
        "bin_transport_object_target_world_m": bin_transport_object,
        "bin_release_object_target_world_m": descent_object,
        "tool_waypoints_world_m": {
            "start": tool_position_world,
            "vertical": vertical_tool,
            "bin": bin_tool,
            "release": descent_tool,
        },
        "transport_path_profile": args.transport_path_profile,
        "transport_orientation_deviation_fraction": orientation_deviation_fraction,
        "transport_orientation_deviation_rad": orientation_deviation_rad,
        "transport_orientation_deviation_deg": float(
            np.degrees(orientation_deviation_rad)
        ),
        "supportive_carry_pitch_deg": float(np.degrees(support_pitch_rad)),
        "supportive_carry_bin_yaw_fraction": support_yaw_fraction,
        "supportive_carry_fixed_object_reference_world_m": fixed_support_object_world,
        "supportive_carry_fixed_object_to_tool_offset_world_m": fixed_object_to_tool_offset_world,
        "supportive_carry_runtime_object_feedback": False,
        "orientation_candidate_selection": (
            "smallest bounded tested fraction with max arm step < 0.03 rad"
            if robust_final
            else "exact preservation"
        ),
        "right_acquisition_7d_rad": right_acquisition,
        "right_transport_grip_7d_rad": right_transport_grip,
        "right_transport_grip_joint_delta_rad": profile["joint_delta_rad"],
        "right_transport_grasp_changed": True,
        "doll_changed": False,
        "runtime_object_feedback_used": False,
        "object_pose_writes": 0,
        "prohibited_mechanisms_used": False,
        "joint_limit_violations": int(np.count_nonzero(violations)),
        "robot_collision_frame_counts": collision_counts,
        "robot_collision_pairs": geometry["collision_pairs"],
        "ik_max_position_error_m": max_position_error,
        "ik_max_orientation_error_rad": max_orientation_error,
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "keyframe": str(KEYFRAME),
        "keyframe_sha256": sha256_file(KEYFRAME),
        "v9_command_sha256": sha256_file(V9_COMMAND),
        "v9_event_sha256": sha256_file(V9_EVENT),
        "config_sha256": sha256_file(CONFIG),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(output / "offline_report.json", report)
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
    )
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
