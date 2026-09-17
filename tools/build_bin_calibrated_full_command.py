#!/usr/bin/env python3
"""Replace only the obsolete absolute-Z bin descent in the frozen full command.

The command prefix through RIGHT_HOLD_OVER_BIN is byte/numerically preserved.
The replacement descent is derived once from the selected bin bottom/rim and
the unchanged object-speed gate, then solved as a smooth fixed-orientation
RIGHT-wrist path.  No grasp, handoff, transport, physics, gain, or threshold is
changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import (  # noqa: E402
    canonical_row,
    minimum_jerk,
    rotations_between,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import AUTHORITATIVE_REFERENCES  # noqa: E402


SOURCE_COMMAND = (
    ROOT
    / "outputs/final_task_completion_v1/08_r26_retiming/variants/R26_T4_4P0X"
    / "retimed_r26_full_command.npz"
)
SELECTED_HEIGHT_EVENT = (
    ROOT
    / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm"
    / "physics_exact_replay/event_log.npz"
)
OUTPUT = (
    ROOT
    / "outputs/final_bin_calibrated_completion/01_selected_bin/height_105mm"
    / "bin_calibrated_full_command.npz"
)
REPORT = OUTPUT.with_name("BIN_RELATIVE_DROP_PATH.json")
SELECTED_BIN_HEIGHT_M = 0.105
BOTTOM_THICKNESS_M = 0.006
DESCENT_FRAMES = 361
STABILIZATION_FRAMES = 30
RELEASE_FRAMES = 45
SETTLE_FRAMES = 120
MAXIMUM_NEW_ADJACENT_ARM_STEP_RAD = 0.03
EXPECTED_SOURCE_SHA256 = "cb680b12d78cff61c53e6e466d605b3732508a365f571b1ceee62aba54d4f52c"


def solve_minimum_orientation_deviation_path(
    g1: G1Kinematics,
    positions_model: np.ndarray,
    preferred_rotation_model: np.ndarray,
    seed_full_arm: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Prioritize exact wrist position while minimizing required orientation drift."""
    block = slice(7, 14)
    lower = g1.arm_limits[block, 0] + 1e-7
    upper = g1.arm_limits[block, 1] - 1e-7
    full = np.asarray(seed_full_arm, dtype=np.float64).copy()
    previous = full[block].copy()
    rows: list[np.ndarray] = []
    reports: list[dict[str, float]] = []
    for target_position in positions_model:
        fixed = full.copy()

        def residual(active: np.ndarray) -> np.ndarray:
            candidate = fixed.copy()
            candidate[block] = active
            g1.assign(candidate, left_hand, right_hand)
            position, rotation, _, _ = g1.static_tool_pose_state(
                "right", np.eye(4, dtype=np.float64)
            )
            clearance = g1.posture_clearance_state(candidate)
            return np.r_[
                150.0 * (position - target_position),
                0.05
                * Rotation.from_matrix(rotation.T @ preferred_rotation_model).as_rotvec(),
                250.0 * max(0.0, -float(clearance["TORSO"]["minimum_distance_m"])),
                150.0
                * max(0.0, -float(clearance["CROSS_ARM"]["minimum_distance_m"])),
                0.004 * (active - previous),
            ]

        solution = least_squares(
            residual,
            np.clip(previous, lower, upper),
            bounds=(lower, upper),
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        full[block] = solution.x
        previous = solution.x.copy()
        g1.assign(full, left_hand, right_hand)
        position, rotation, _, _ = g1.static_tool_pose_state(
            "right", np.eye(4, dtype=np.float64)
        )
        clearance = g1.posture_clearance_state(full)
        rows.append(full.copy())
        reports.append(
            {
                "position_error_m": float(np.linalg.norm(position - target_position)),
                "orientation_deviation_rad": float(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            rotation.T @ preferred_rotation_model
                        ).as_rotvec()
                    )
                ),
                "torso_clearance_m": float(clearance["TORSO"]["minimum_distance_m"]),
                "cross_arm_clearance_m": float(
                    clearance["CROSS_ARM"]["minimum_distance_m"]
                ),
            }
        )
    return np.asarray(rows), reports


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main() -> int:
    if sha256(SOURCE_COMMAND) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("source full-command hash changed")
    with np.load(SOURCE_COMMAND, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(SELECTED_HEIGHT_EVENT, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    command = np.asarray(source["commanded_q_rad"], dtype=np.float64)
    stage = source["stage"].astype(str)
    names = source["joint_names"].astype(str).tolist()
    hold = np.flatnonzero(stage == "RIGHT_HOLD_OVER_BIN")
    if len(hold) == 0:
        raise RuntimeError("source command lacks RIGHT_HOLD_OVER_BIN")
    prefix_end = int(hold[-1])
    prefix = command[: prefix_end + 1].copy()
    prefix_stage = stage[: prefix_end + 1].copy()

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = np.asarray([lookup[name] for name in g1.arm_joint_names])
    left_indices = np.asarray([lookup[name] for name in g1.hand_joint_names["left"]])
    right_indices = np.asarray([lookup[name] for name in g1.hand_joint_names["right"]])
    start = prefix[-1]
    start_arm = start[arm_indices]
    left_hand = start[left_indices]
    right_hold = start[right_indices]
    g1.assign(start_arm, left_hand, right_hold)
    start_wrist_model = np.asarray(g1.wrist_pose("right"), dtype=np.float64)
    start_wrist_world_position = g1.model_to_world_position(start_wrist_model[:3, 3])
    start_wrist_model_rotation = start_wrist_model[:3, :3].copy()

    event_hold = event["stage"].astype(str) == "RIGHT_HOLD_OVER_BIN"
    if not np.any(event_hold):
        raise RuntimeError("selected-height event lacks RIGHT_HOLD_OVER_BIN")
    held_object_center_world = np.median(
        event["object_position_world_m"][event_hold], axis=0
    )
    held_object_center_z = float(held_object_center_world[2])

    table_z = float(scene["table"]["surface_height_m"])
    collision_height = 0.070
    # Descend to the geometry-defined supported center before release.  This
    # removes free fall without moving the bin bottom or weakening the speed gate.
    free_fall_gap_m = 0.0
    target_object_center_z = (
        table_z + BOTTOM_THICKNESS_M + 0.5 * collision_height + free_fall_gap_m
    )
    descent_distance = held_object_center_z - target_object_center_z
    if descent_distance <= 0.0:
        raise RuntimeError("selected rim-relative descent is not downward")
    held_object_center_model = g1.world_to_model_position(held_object_center_world)
    wrist_to_object = np.eye(4, dtype=np.float64)
    wrist_to_object[:3, 3] = start_wrist_model_rotation.T @ (
        held_object_center_model - start_wrist_model[:3, 3]
    )
    target_object_center_world = np.asarray(
        [
            float(scene["bin"]["center_world_xy_m"][0]),
            float(scene["bin"]["center_world_xy_m"][1]),
            target_object_center_z,
        ],
        dtype=np.float64,
    )
    positions_world = minimum_jerk(
        held_object_center_world, target_object_center_world, DESCENT_FRAMES
    )
    orientation_candidates = []
    feasible_orientation_paths: list[tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]] = []
    selected_orientation_relief_deg: float | None = None
    descent_arm = np.empty((0, 14), dtype=np.float64)
    solve_reports: list[dict[str, Any]] = []
    bin_center_xy = np.asarray(scene["bin"]["center_world_xy_m"], dtype=np.float64)
    opening_half_xy = 0.5 * np.asarray(
        scene["bin"]["opening_dimensions_xy_m"], dtype=np.float64
    )
    opening_min_xy = bin_center_xy - opening_half_xy
    opening_max_xy = bin_center_xy + opening_half_xy
    right_inner_wall_x = float(opening_max_xy[0])
    for relief_deg in (60.0, 70.0, 80.0, 90.0):
        relief_arm = start_arm.copy()
        relief_arm[13] = max(
            float(g1.arm_limits[13, 0] + 1e-7),
            relief_arm[13] - np.deg2rad(relief_deg),
        )
        g1.assign(relief_arm, left_hand, right_hold)
        relief_rotation = np.asarray(g1.wrist_pose("right"), dtype=np.float64)[:3, :3]
        candidate_rotations = rotations_between(
            start_wrist_model_rotation, relief_rotation, DESCENT_FRAMES
        )
        candidate_arm, raw_reports = solve_path(
            g1,
            "right",
            wrist_to_object,
            g1.world_to_model_position(positions_world),
            candidate_rotations,
            start_arm,
            left_hand,
            right_hold,
            clearance_target_m=0.0,
        )
        candidate_steps = np.max(
            np.abs(np.diff(candidate_arm[:, 7:14], axis=0)), axis=1, initial=0.0
        )
        candidate_report = {
            "relief_deg": relief_deg,
            "maximum_adjacent_arm_step_rad": float(np.max(candidate_steps, initial=0.0)),
            "maximum_position_error_m": max(row["position_error_m"] for row in raw_reports),
            "maximum_orientation_error_rad": max(
                row["orientation_error_rad"] for row in raw_reports
            ),
            "minimum_torso_clearance_m": min(row["torso_clearance_m"] for row in raw_reports),
            "minimum_cross_arm_clearance_m": min(
                row["cross_arm_clearance_m"] for row in raw_reports
            ),
        }
        g1.assign(candidate_arm[-1], left_hand, right_hold)
        hand_body_position: dict[str, list[float]] = {}
        for body_name in (
            "right_hand_palm_link",
            "right_hand_thumb_2_link",
            "right_hand_index_1_link",
            "right_hand_middle_1_link",
        ):
            body_id = mujoco.mj_name2id(
                g1.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id >= 0:
                hand_body_position[body_name] = g1.model_to_world_position(
                    g1.data.xpos[body_id]
                ).tolist()
        hand_body_x = {
            name: float(position[0]) for name, position in hand_body_position.items()
        }
        candidate_report["endpoint_hand_body_position_world_m"] = hand_body_position
        candidate_report["maximum_endpoint_hand_body_x_world_m"] = max(
            hand_body_x.values(), default=float("inf")
        )
        candidate_report["predicted_right_wall_clearance_m"] = (
            right_inner_wall_x
            - candidate_report["maximum_endpoint_hand_body_x_world_m"]
        )
        center_edge_clearances = []
        for position in hand_body_position.values():
            xy = np.asarray(position[:2], dtype=np.float64)
            center_edge_clearances.extend((xy - opening_min_xy).tolist())
            center_edge_clearances.extend((opening_max_xy - xy).tolist())
        candidate_report["minimum_hand_body_center_to_opening_edge_m"] = min(
            center_edge_clearances, default=-float("inf")
        )
        candidate_report["pass"] = bool(
            candidate_report["maximum_adjacent_arm_step_rad"]
            <= MAXIMUM_NEW_ADJACENT_ARM_STEP_RAD
            and candidate_report["maximum_position_error_m"] <= 0.001
            and candidate_report["maximum_orientation_error_rad"] <= 0.02
            and candidate_report["minimum_torso_clearance_m"] >= 0.0
            and candidate_report["minimum_cross_arm_clearance_m"] >= 0.0
        )
        orientation_candidates.append(candidate_report)
        print("orientation_relief_candidate", candidate_report)
        if candidate_report["pass"]:
            normalized_reports = [
                {
                    **row,
                    "orientation_deviation_rad": float(
                        np.deg2rad(relief_deg) + row["orientation_error_rad"]
                    ),
                }
                for row in raw_reports
            ]
            feasible_orientation_paths.append(
                (candidate_report, candidate_arm, normalized_reports)
            )
    if not feasible_orientation_paths:
        raise RuntimeError("bounded minimum-orientation descent audit found no continuous path")
    selected_report, descent_arm, solve_reports = max(
        feasible_orientation_paths,
        key=lambda value: value[0]["minimum_hand_body_center_to_opening_edge_m"],
    )
    selected_orientation_relief_deg = float(selected_report["relief_deg"])
    truncated_at_path_index: int | None = None
    print(
        "descent_ik",
        f"max_position_error_m={max(row['position_error_m'] for row in solve_reports):.9f}",
        f"endpoint_position_error_m={solve_reports[-1]['position_error_m']:.9f}",
        f"max_orientation_deviation_rad={max(row['orientation_deviation_rad'] for row in solve_reports):.9f}",
    )
    print("descent_endpoint_right_arm", descent_arm[-1, 7:14].tolist())
    print("descent_right_arm_limits", g1.arm_limits[7:14].tolist())
    g1.assign(descent_arm[-1], left_hand, right_hold)
    endpoint_position_model, _, _, _ = g1.static_tool_pose_state(
        "right", wrist_to_object
    )
    print("descent_target_position_model", g1.world_to_model_position(positions_world[-1]).tolist())
    print("descent_actual_position_model", endpoint_position_model.tolist())
    if max(row["position_error_m"] for row in solve_reports) > 0.005:
        raise RuntimeError("continuous rim-relative descent prefix exceeds 5 mm IK error")
    if min(row["torso_clearance_m"] for row in solve_reports) < 0.0:
        raise RuntimeError("rim-relative descent has torso collision")
    if min(row["cross_arm_clearance_m"] for row in solve_reports) < 0.0:
        raise RuntimeError("rim-relative descent has cross-arm collision")

    right_open = np.asarray(
        json.loads(
            (ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json").read_text()
        )["hand_states"]["right"]["OPEN"],
        dtype=np.float64,
    )
    rows = [row.copy() for row in prefix]
    labels = prefix_stage.tolist()
    for arm in descent_arm[1:]:
        rows.append(canonical_row(names, g1, arm, left_hand, right_hold))
        labels.append("RIGHT_RIM_RELATIVE_CONTROLLED_DESCENT")
    for _ in range(STABILIZATION_FRAMES):
        rows.append(canonical_row(names, g1, descent_arm[-1], left_hand, right_hold))
        labels.append("RIGHT_PRE_RELEASE_STABILIZATION")
    for hand in minimum_jerk(right_hold, right_open, RELEASE_FRAMES)[1:]:
        rows.append(canonical_row(names, g1, descent_arm[-1], left_hand, hand))
        labels.append("RIGHT_RELEASE")
    for arm in descent_arm[-2::-1]:
        rows.append(canonical_row(names, g1, arm, left_hand, right_open))
        labels.append("RIGHT_POST_RELEASE_RETREAT")
    for _ in range(SETTLE_FRAMES):
        rows.append(canonical_row(names, g1, descent_arm[0], left_hand, right_open))
        labels.append("BIN_SETTLE")
    result = np.asarray(rows, dtype=np.float64)
    labels_array = np.asarray(labels)

    joint_contract = json.loads(Path(AUTHORITATIVE_REFERENCES["joint_ranges"]).read_text())
    lower = np.asarray([float(row["minimum"]) for row in joint_contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in joint_contract["joint_specs"]])
    violations = (result < lower[None] - 1e-9) | (result > upper[None] + 1e-9)
    if np.any(violations):
        raise RuntimeError("bin-relative command violates authoritative joint limits")
    new_steps = np.max(
        np.abs(np.diff(result[prefix_end:, arm_indices], axis=0)), axis=1, initial=0.0
    )
    maximum_new_step = float(np.max(new_steps, initial=0.0))
    if maximum_new_step > MAXIMUM_NEW_ADJACENT_ARM_STEP_RAD:
        raise RuntimeError(f"new adjacent arm step exceeds gate: {maximum_new_step}")

    geometry = g1.trajectory_geometry(
        result[:, arm_indices], result[:, left_indices], result[:, right_indices], 1e-5
    )
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()
    }
    if any(collision_counts.values()):
        raise RuntimeError(f"offline robot collision in bin-relative command: {collision_counts}")

    g1.assign(descent_arm[-1], left_hand, right_hold)
    achieved_wrist_model = np.asarray(g1.wrist_pose("right"), dtype=np.float64)
    achieved_wrist_world_position = g1.model_to_world_position(
        achieved_wrist_model[:3, 3]
    )
    achieved_object_model, _, _, _ = g1.static_tool_pose_state(
        "right", wrist_to_object
    )
    achieved_object_world = g1.model_to_world_position(achieved_object_model)
    achieved_descent_distance = float(
        held_object_center_z - achieved_object_world[2]
    )
    predicted_release_object_center_z = float(achieved_object_world[2])

    output_values = dict(source)
    output_values.update(
        commanded_q_rad=result.astype(np.float32),
        stage=labels_array,
        selected_bin_height_m=np.asarray(SELECTED_BIN_HEIGHT_M),
        selected_bin_rim_world_z_m=np.asarray(table_z + SELECTED_BIN_HEIGHT_M),
        rim_relative_drop_path=np.asarray(True),
        target_release_object_center_world_z_m=np.asarray(target_object_center_z),
        predicted_reachable_release_object_center_world_z_m=np.asarray(
            predicted_release_object_center_z
        ),
        theoretical_free_fall_gap_m=np.asarray(free_fall_gap_m),
        theoretical_free_fall_speed_m_s=np.asarray(
            np.sqrt(2.0 * 9.81 * free_fall_gap_m)
        ),
        source_prefix_end_control_frame=np.asarray(prefix_end),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **output_values)
    os.replace(temporary, OUTPUT)
    report = {
        "schema_version": "bin_calibrated_full_command_v1",
        "policy_independent": True,
        "source_command": SOURCE_COMMAND,
        "source_command_sha256": sha256(SOURCE_COMMAND),
        "selected_height_event": SELECTED_HEIGHT_EVENT,
        "selected_height_event_sha256": sha256(SELECTED_HEIGHT_EVENT),
        "output_command": OUTPUT,
        "output_command_sha256": sha256(OUTPUT),
        "prefix_frames_preserved": prefix_end + 1,
        "prefix_numeric_identical": bool(np.array_equal(result[: prefix_end + 1], command[: prefix_end + 1])),
        "selected_bin_height_m": SELECTED_BIN_HEIGHT_M,
        "selected_rim_world_z_m": table_z + SELECTED_BIN_HEIGHT_M,
        "held_object_center_z_m": held_object_center_z,
        "target_release_object_center_z_m": target_object_center_z,
        "descent_distance_m": descent_distance,
        "continuous_branch_truncated_at_path_index": truncated_at_path_index,
        "orientation_relief_candidates": orientation_candidates,
        "selected_minimum_orientation_relief_deg": selected_orientation_relief_deg,
        "achieved_wrist_world_position_m": achieved_wrist_world_position,
        "achieved_object_world_position_m": achieved_object_world,
        "achieved_descent_distance_m": achieved_descent_distance,
        "predicted_reachable_release_object_center_z_m": predicted_release_object_center_z,
        "theoretical_free_fall_gap_m": free_fall_gap_m,
        "theoretical_free_fall_speed_m_s": float(np.sqrt(2.0 * 9.81 * free_fall_gap_m)),
        "start_wrist_world_position_m": start_wrist_world_position,
        "held_object_center_world_m": held_object_center_world,
        "target_object_center_world_m": target_object_center_world,
        "wrist_to_object_transform": wrist_to_object,
        "wrist_orientation_held": False,
        "minimum_jerk": True,
        "descent_frames": DESCENT_FRAMES,
        "stabilization_frames": STABILIZATION_FRAMES,
        "release_frames": RELEASE_FRAMES,
        "settle_frames": SETTLE_FRAMES,
        "maximum_new_adjacent_arm_step_rad": maximum_new_step,
        "maximum_new_adjacent_arm_step_gate_rad": MAXIMUM_NEW_ADJACENT_ARM_STEP_RAD,
        "maximum_ik_position_error_m": max(row["position_error_m"] for row in solve_reports),
        "maximum_required_wrist_orientation_deviation_rad": max(
            row["orientation_deviation_rad"] for row in solve_reports
        ),
        "offline_collision_counts": collision_counts,
        "doll_grasp_handoff_transport_prefix_modified": False,
        "doll_physics_modified": False,
        "gains_modified": False,
        "thresholds_modified": False,
        "act_ab_used": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False, default=default) + "\n")
    print(OUTPUT)
    print(f"sha256={report['output_command_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
