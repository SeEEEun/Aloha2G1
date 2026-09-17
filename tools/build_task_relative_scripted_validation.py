#!/usr/bin/env python3
"""Re-express only the scripted validator in authoritative task-relative frames."""

from __future__ import annotations

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

from tools.build_doll_handoff_proxy_v2_handoff_gate import minimum_jerk, solve_path  # noqa: E402
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402


SOURCE = (
    ROOT
    / "outputs/final_direct_physical_eval35/00_pre_eval35_execution_freeze"
    / "physx_qualification/commands/scripted_full_task_limit_safe_regression.npz"
)
REGISTRATION = ROOT / "configs/contact_eval_common_task_registration_v1.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
AB_MANIFEST = ROOT / "outputs/final_direct_physical_eval35/00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
FROZEN_HANDOFF_SOURCE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task/p14_bilateral"
    / "backward_constructed_handoff/B2_PATH_F40/right_preload_partial_left_relax_exact_endpoint_v4"
    / "full/scripted_full_task_command.npz"
)
OUTPUT_DIR = ROOT / "outputs/final_direct_physical_eval35/00_authoritative_physical_scene/task_relative_scripted_validation"
OUTPUT = OUTPUT_DIR / "corrected_task_relative_scripted_regression.npz"
REPORT = OUTPUT_DIR / "TASK_FRAME_NUMERICAL_PREFLIGHT.json"
TASK_RELATIVE_END = 206
HANDOFF_BRIDGE_START = 207
FROZEN_HANDOFF_TAIL_START = 266
EXPECTED_SOURCE_SHA256 = "770a270e1e8927dda7f1b1b75fe76e1596566b7e182c61e03876f9125d0a5d17"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def wrist_targets(
    g1: G1Kinematics,
    arms: np.ndarray,
    left_hands: np.ndarray,
    right_hands: np.ndarray,
    side: str,
    world_from_model: np.ndarray,
    rotation_delta_world: np.ndarray,
    old_object_position: np.ndarray,
    new_object_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_positions_world: list[np.ndarray] = []
    old_rotations_world: list[np.ndarray] = []
    for arm, left, right in zip(arms, left_hands, right_hands, strict=True):
        g1.assign(arm, left, right)
        wrist_model = np.asarray(g1.wrist_pose(side), dtype=np.float64)
        old_positions_world.append(g1.model_to_world_position(wrist_model[:3, 3]))
        old_rotations_world.append(world_from_model @ wrist_model[:3, :3])
    old_positions = np.asarray(old_positions_world)
    old_rotations = np.asarray(old_rotations_world)
    new_positions = (
        np.einsum("ij,tj->ti", rotation_delta_world, old_positions - old_object_position)
        + new_object_position
    )
    new_rotations_world = np.einsum("ij,tjk->tik", rotation_delta_world, old_rotations)
    new_rotations_model = np.einsum("ij,tjk->tik", world_from_model.T, new_rotations_world)
    return old_positions, old_rotations, new_positions, new_rotations_model


def actual_wrist_world(
    g1: G1Kinematics,
    arm: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    side: str,
    world_from_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g1.assign(arm, left, right)
    wrist = np.asarray(g1.wrist_pose(side), dtype=np.float64)
    return g1.model_to_world_position(wrist[:3, 3]), world_from_model @ wrist[:3, :3]


def transform_error(
    actual_position: np.ndarray,
    actual_rotation: np.ndarray,
    expected_position: np.ndarray,
    expected_rotation: np.ndarray,
) -> tuple[float, float]:
    position_error = float(np.linalg.norm(actual_position - expected_position))
    rotation_error = float(
        np.linalg.norm(Rotation.from_matrix(actual_rotation.T @ expected_rotation).as_rotvec())
    )
    return position_error, rotation_error


def main() -> int:
    if sha256_file(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("scripted validation source command changed")
    registration = read_json(REGISTRATION)
    config = read_json(CONFIG)
    ab_before = sha256_file(AB_MANIFEST)
    ab_manifest = read_json(AB_MANIFEST)
    ab_command_hashes_before = {
        row["physical_command"]: sha256_file(Path(row["physical_command"]))
        for row in ab_manifest["records"]
    }
    ab_declared_hash_mismatches = [
        row["physical_command"]
        for row in ab_manifest["records"]
        if ab_command_hashes_before[row["physical_command"]] != row["physical_command_sha256"]
    ]
    if ab_declared_hash_mismatches:
        raise RuntimeError("an ACT-A/B command already differs from its frozen manifest")
    with np.load(SOURCE, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(FROZEN_HANDOFF_SOURCE, allow_pickle=False) as archive:
        frozen_handoff_object_center = np.asarray(
            archive["handoff_object_center_world_m"], dtype=np.float64
        )
    command = np.asarray(source["commanded_q_rad"], dtype=np.float64)
    stages = source["stage"].astype(str)
    names = source["joint_names"].astype(str).tolist()
    if len(command) != 3309 or stages[FROZEN_HANDOFF_TAIL_START] != "LEFT_HANDOFF_HOLD":
        raise RuntimeError("unexpected scripted command phase layout")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = np.asarray([lookup[name] for name in g1.arm_joint_names])
    left_indices = np.asarray([lookup[name] for name in g1.hand_joint_names["left"]])
    right_indices = np.asarray([lookup[name] for name in g1.hand_joint_names["right"]])
    arms = command[:, arm_indices]
    left_hands = command[:, left_indices]
    right_hands = command[:, right_indices]

    origin = g1.model_to_world_position(np.zeros(3, dtype=np.float64))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3, dtype=np.float64)[axis]) - origin
            for axis in range(3)
        ]
    )
    old_object_position = np.asarray(
        [
            *config["object"]["center_world_xy_m_by_side"]["left"],
            config["object"]["table_surface_world_z_m"]
            + config["object"]["visual_dimensions_m"][2] / 2.0,
        ],
        dtype=np.float64,
    )
    new_object_position = np.asarray(
        [
            *registration["registered_doll_center_world_xy_m"],
            old_object_position[2],
        ],
        dtype=np.float64,
    )
    rotation_delta_world = Rotation.from_quat(
        registration["registered_doll_orientation_quaternion_xyzw"]
    ).as_matrix()
    count = TASK_RELATIVE_END + 1
    left_old_pos, left_old_rot, left_target_pos, left_target_rot_model = wrist_targets(
        g1, arms[:count], left_hands[:count], right_hands[:count], "left",
        world_from_model, rotation_delta_world, old_object_position, new_object_position,
    )
    left_solved, left_reports = solve_path(
        g1,
        "left",
        np.eye(4, dtype=np.float64),
        g1.world_to_model_position(left_target_pos),
        left_target_rot_model,
        arms[0],
        left_hands[0],
        right_hands[0],
        clearance_target_m=0.0,
    )
    corrected = command.copy()
    corrected[:count, arm_indices[:7]] = left_solved[:, :7]

    # The doll is now physically held at the registered pose. Move that held
    # LEFT grasp into the already validated handoff task frame. The bridge ends
    # at the exact old handoff joint state; every command from LEFT_HANDOFF_HOLD
    # onward (including RIGHT acquisition, transport, and bin placement) is
    # preserved exactly.
    bridge_count = FROZEN_HANDOFF_TAIL_START - HANDOFF_BRIDGE_START + 1
    new_bridge_start_position, new_bridge_start_rotation = actual_wrist_world(
        g1,
        corrected[TASK_RELATIVE_END, arm_indices],
        corrected[TASK_RELATIVE_END, left_indices],
        corrected[TASK_RELATIVE_END, right_indices],
        "left",
        world_from_model,
    )
    old_handoff_position, old_handoff_rotation = actual_wrist_world(
        g1,
        arms[FROZEN_HANDOFF_TAIL_START],
        left_hands[FROZEN_HANDOFF_TAIL_START],
        right_hands[FROZEN_HANDOFF_TAIL_START],
        "left",
        world_from_model,
    )
    bridge_positions_world = minimum_jerk(
        new_bridge_start_position, old_handoff_position, bridge_count
    )
    bridge_rotations_world = Rotation.from_matrix(
        np.stack((new_bridge_start_rotation, old_handoff_rotation))
    )
    from scipy.spatial.transform import Slerp

    bridge_rotations_world = Slerp([0.0, 1.0], bridge_rotations_world)(
        np.linspace(0.0, 1.0, bridge_count)
    ).as_matrix()
    bridge_rotations_model = np.einsum(
        "ij,tjk->tik", world_from_model.T, bridge_rotations_world
    )
    bridge_solved, bridge_reports = solve_path(
        g1,
        "left",
        np.eye(4, dtype=np.float64),
        g1.world_to_model_position(bridge_positions_world),
        bridge_rotations_model,
        corrected[TASK_RELATIVE_END, arm_indices],
        left_hands[TASK_RELATIVE_END],
        right_hands[TASK_RELATIVE_END],
        clearance_target_m=0.0,
    )
    corrected[HANDOFF_BRIDGE_START:FROZEN_HANDOFF_TAIL_START + 1, arm_indices[:7]] = bridge_solved[:, :7]
    corrected[HANDOFF_BRIDGE_START:FROZEN_HANDOFF_TAIL_START + 1, arm_indices[7:]] = arms[
        HANDOFF_BRIDGE_START:FROZEN_HANDOFF_TAIL_START + 1, 7:
    ]
    corrected[FROZEN_HANDOFF_TAIL_START:] = command[FROZEN_HANDOFF_TAIL_START:]

    joint_contract = read_json(JOINT_CONTRACT)
    lower = np.asarray([row["minimum"] for row in joint_contract["joint_specs"]], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in joint_contract["joint_specs"]], dtype=np.float64)
    violations = (corrected < lower - 1.0e-9) | (corrected > upper + 1.0e-9)
    if np.any(violations):
        raise RuntimeError("corrected scripted validator violates authoritative joint limits")

    geometry = g1.trajectory_geometry(
        corrected[:, arm_indices], corrected[:, left_indices], corrected[:, right_indices], 1.0e-5
    )
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()
    }
    maximum_arm_step = float(np.max(np.abs(np.diff(corrected[:, arm_indices], axis=0)), initial=0.0))

    key_frames = {
        "left_grasp": 102,
        "left_lift": 176,
    }
    key_audit: dict[str, Any] = {}
    for label, frame in key_frames.items():
        rows: dict[str, Any] = {}
        for side, target_positions, old_positions, old_rotations in (
            ("left", left_target_pos, left_old_pos, left_old_rot),
        ):
            actual_position, actual_rotation = actual_wrist_world(
                g1,
                corrected[frame, arm_indices],
                corrected[frame, left_indices],
                corrected[frame, right_indices],
                side,
                world_from_model,
            )
            expected_rotation = rotation_delta_world @ old_rotations[frame]
            position_error, rotation_error = transform_error(
                actual_position, actual_rotation, target_positions[frame], expected_rotation
            )
            old_relative_position = old_rotations[frame].T @ (old_object_position - old_positions[frame])
            new_relative_position = actual_rotation.T @ (new_object_position - actual_position)
            rows[side] = {
                "actual_wrist_world_position_m": actual_position,
                "target_wrist_world_position_m": target_positions[frame],
                "position_error_m": position_error,
                "orientation_error_deg": float(np.degrees(rotation_error)),
                "old_wrist_frame_object_vector_m": old_relative_position,
                "new_wrist_frame_object_vector_m": new_relative_position,
                "relative_object_vector_error_m": float(np.linalg.norm(new_relative_position - old_relative_position)),
            }
        key_audit[label] = {"frame": frame, "stage": stages[frame], "hands": rows}

    frozen_handoff_tail_exact = bool(
        np.array_equal(
            corrected[FROZEN_HANDOFF_TAIL_START:],
            command[FROZEN_HANDOFF_TAIL_START:],
        )
    )
    bin_frames_exact = bool(np.array_equal(corrected[2140:], command[2140:]))
    dex3_exact = bool(np.array_equal(corrected[:, 14:], command[:, 14:]))
    ab_command_hashes_after = {
        path: sha256_file(Path(path)) for path in ab_command_hashes_before
    }
    ab_commands_unchanged = ab_command_hashes_after == ab_command_hashes_before
    task_position_error = max(row["position_error_m"] for row in left_reports + bridge_reports)
    task_orientation_error = max(row["orientation_error_rad"] for row in left_reports + bridge_reports)
    handoff_pose_audit: dict[str, Any] = {}
    for frame in (280, 1440, 1800):
        per_hand: dict[str, Any] = {}
        for side in ("left", "right"):
            corrected_position, corrected_rotation = actual_wrist_world(
                g1, corrected[frame, arm_indices], corrected[frame, left_indices],
                corrected[frame, right_indices], side, world_from_model
            )
            source_position, source_rotation = actual_wrist_world(
                g1, command[frame, arm_indices], command[frame, left_indices],
                command[frame, right_indices], side, world_from_model
            )
            position_error, rotation_error = transform_error(
                corrected_position, corrected_rotation, source_position, source_rotation
            )
            per_hand[side] = {
                "corrected_wrist_world_position_m": corrected_position,
                "frozen_source_wrist_world_position_m": source_position,
                "position_difference_m": position_error,
                "orientation_difference_deg": float(np.degrees(rotation_error)),
                "wrist_to_frozen_handoff_object_vector_world_m": frozen_handoff_object_center - corrected_position,
            }
        handoff_pose_audit[str(frame)] = {
            "stage": stages[frame], "hands": per_hand
        }
    preflight_pass = bool(
        task_position_error <= 0.001
        and task_orientation_error <= np.deg2rad(0.5)
        and frozen_handoff_tail_exact
        and bin_frames_exact
        and dex3_exact
        and not any(collision_counts.values())
        and maximum_arm_step <= 0.05
        and ab_before == sha256_file(AB_MANIFEST)
        and ab_commands_unchanged
    )

    output_values = dict(source)
    output_values.update(
        commanded_q_rad=corrected.astype(np.float32),
        task_relative_scripted_validation=np.asarray(True),
        authoritative_object_registration=np.asarray(str(REGISTRATION.resolve())),
        authoritative_object_registration_sha256=np.asarray(sha256_file(REGISTRATION)),
        task_relative_end_control_frame=np.asarray(TASK_RELATIVE_END),
        handoff_bridge_start_control_frame=np.asarray(HANDOFF_BRIDGE_START),
        exact_frozen_handoff_tail_start_control_frame=np.asarray(FROZEN_HANDOFF_TAIL_START),
        source_scripted_validation_sha256=np.asarray(EXPECTED_SOURCE_SHA256),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **output_values)
    os.replace(temporary, OUTPUT)
    report = {
        "schema_version": "task_relative_scripted_validation_preflight_v1",
        "status": "PASS" if preflight_pass else "FAIL",
        "source_command": str(SOURCE.resolve()),
        "source_command_sha256": sha256_file(SOURCE),
        "output_command": str(OUTPUT.resolve()),
        "output_command_sha256": sha256_file(OUTPUT),
        "registration": str(REGISTRATION.resolve()),
        "registration_sha256": sha256_file(REGISTRATION),
        "old_object_pose": {
            "position_xyz_m": old_object_position,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "authoritative_object_pose": {
            "position_xyz_m": new_object_position,
            "quaternion_xyzw": registration["registered_doll_orientation_quaternion_xyzw"],
        },
        "phase_contract": {
            "object_and_handoff_relative_frames_inclusive": [0, TASK_RELATIVE_END],
            "held_object_bridge_to_frozen_handoff_frames_inclusive": [HANDOFF_BRIDGE_START, FROZEN_HANDOFF_TAIL_START],
            "exact_frozen_handoff_right_acquisition_transport_and_bin_tail_frames_inclusive": [FROZEN_HANDOFF_TAIL_START, len(command) - 1],
        },
        "key_frame_relative_transform_audit": key_audit,
        "frozen_handoff_object_center_world_m": frozen_handoff_object_center,
        "frozen_handoff_pose_audit": handoff_pose_audit,
        "maximum_task_relative_wrist_position_error_m": task_position_error,
        "maximum_task_relative_wrist_orientation_error_deg": float(np.degrees(task_orientation_error)),
        "maximum_adjacent_arm_joint_step_rad": maximum_arm_step,
        "offline_collision_counts": collision_counts,
        "dex3_commands_exactly_preserved": dex3_exact,
        "frozen_handoff_and_transport_tail_exactly_preserved": frozen_handoff_tail_exact,
        "bin_relative_tail_exactly_preserved": bin_frames_exact,
        "act_ab_command_manifest_sha256_before": ab_before,
        "act_ab_command_manifest_sha256_after": sha256_file(AB_MANIFEST),
        "act_ab_physical_command_count_verified": len(ab_command_hashes_before),
        "act_ab_physical_command_hashes_before": ab_command_hashes_before,
        "act_ab_physical_command_hashes_after": ab_command_hashes_after,
        "act_ab_trajectories_modified": False,
        "doll_physics_modified": False,
        "bin_modified": False,
        "common_execution_layer_modified": False,
    }
    atomic_json(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(OUTPUT),
        "sha256": report["output_command_sha256"],
        "maximum_position_error_m": task_position_error,
        "maximum_orientation_error_deg": report["maximum_task_relative_wrist_orientation_error_deg"],
        "collision_counts": collision_counts,
        "maximum_arm_step_rad": maximum_arm_step,
    }, indent=2))
    return 0 if preflight_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
