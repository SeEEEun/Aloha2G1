#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

TWIST_PYTHON = "/home/jbnu/miniconda3/envs/twist/bin/python"

try:
    import numpy as np
except ModuleNotFoundError:
    if Path(sys.executable).resolve() != Path(TWIST_PYTHON).resolve():
        os.execv(TWIST_PYTHON, [TWIST_PYTHON, __file__, *sys.argv[1:]])
    raise

import argparse
import math

import transform_aloha_tcp_to_g1_targets as transform_mod
import validate_g1_targets_and_sparse_ik as ik_mod


DEFAULT_INPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/aloha_tcp_trajectory.npz")
DEFAULT_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEFAULT_SEARCH_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_task_ready_anchor_search.npz")
DEFAULT_BEST_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/task_ready_g1_cartesian_targets.npz")
SAMPLE_FRAME_INDICES = np.array([0, 100, 300, 500, 712], dtype=np.int64)
CENTER_X_VALUES = np.array([0.15, 0.20, 0.25, 0.30, 0.35], dtype=np.float64)
CENTER_Z_VALUES = np.array([0.65, 0.70, 0.75, 0.80, 0.85, 0.90], dtype=np.float64)
HAND_SEPARATION_VALUES = np.array([0.20, 0.24, 0.28, 0.32, 0.36, 0.40], dtype=np.float64)
CENTER_SCALE_VALUES = np.array([0.45, 0.50, 0.55, 0.60, 0.65], dtype=np.float64)
RELATIVE_SCALE_VALUES = np.array([0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 1.00], dtype=np.float64)
ALIGN_RPY_VALUES = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, -5.0, 0.0]], dtype=np.float64)
HAND_DISTANCE_MIN_M = 0.12
HAND_DISTANCE_MAX_M = 0.60
POSITION_TOLERANCE_M = 0.01
MAX_ITERATIONS = 300
DAMPING = 0.01
MAX_JOINT_STEP_RAD = 0.04
PREVIOUS_Q_WEIGHT = 0.02
NEUTRAL_Q_WEIGHT = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a task-ready G1 bimanual anchor using sparse position-only IK.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--search-output", type=Path, default=DEFAULT_SEARCH_OUTPUT)
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fail(message: str) -> RuntimeError:
    return RuntimeError(message)


def require_writable_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise fail(f"Output already exists. Use --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def sample_indices(frame_count: int) -> np.ndarray:
    valid = []
    for idx in SAMPLE_FRAME_INDICES.tolist():
        if idx < frame_count:
            valid.append(idx)
    if 0 not in valid:
        valid.insert(0, 0)
    if frame_count - 1 not in valid:
        valid.append(frame_count - 1)
    return np.asarray(sorted(set(valid)), dtype=np.int64)


def make_align_rotation(rpy_deg: np.ndarray) -> np.ndarray:
    return transform_mod.rotation_from_rpy_deg(np.asarray(rpy_deg, dtype=np.float64))


def position_only_state(model_info: dict[str, object], data) -> dict[str, np.ndarray]:
    arm_dof_ids = model_info["arm_dof_ids"]
    left_pos, left_quat, left_jac = ik_mod.compute_palm_pose_and_jacobian(
        model_info["model"], data, model_info["left_body_id"], ik_mod.LEFT_PALM_OFFSET, arm_dof_ids[:7]
    )
    right_pos, right_quat, right_jac = ik_mod.compute_palm_pose_and_jacobian(
        model_info["model"], data, model_info["right_body_id"], ik_mod.RIGHT_PALM_OFFSET, arm_dof_ids[7:]
    )
    return {
        "left_pos": left_pos,
        "left_quat": left_quat,
        "right_pos": right_pos,
        "right_quat": right_quat,
        "left_jac_pos": left_jac[:3],
        "right_jac_pos": right_jac[:3],
    }


def generate_initial_guesses(neutral_arm_q: np.ndarray, target_left_pos: np.ndarray, target_right_pos: np.ndarray) -> list[np.ndarray]:
    center = 0.5 * (target_left_pos + target_right_pos)
    separation = abs(float(target_right_pos[1] - target_left_pos[1]))
    shoulder_pitch = float(np.interp(center[0], [0.15, 0.35], [-0.2, -0.8]))
    elbow = float(np.interp(center[2], [0.65, 0.90], [1.2, 0.35]))
    roll_mag = float(np.interp(separation, [0.20, 0.40], [0.05, 0.18]))

    seeds = [np.array(neutral_arm_q, dtype=np.float64, copy=True)]
    for pitch_delta, elbow_delta in [(0.0, 0.0), (-0.15, -0.2), (0.15, 0.2)]:
        q = np.array(neutral_arm_q, dtype=np.float64, copy=True)
        q[0] = shoulder_pitch + pitch_delta
        q[1] = roll_mag
        q[2] = 0.0
        q[3] = elbow + elbow_delta
        q[4:7] = 0.0
        q[7] = shoulder_pitch + pitch_delta
        q[8] = -roll_mag
        q[9] = 0.0
        q[10] = elbow + elbow_delta
        q[11:14] = 0.0
        seeds.append(q)
    return seeds


def solve_position_only_frame(
    model_info: dict[str, object],
    target_left_pos: np.ndarray,
    target_right_pos: np.ndarray,
    init_arm_q: np.ndarray,
    previous_arm_q: np.ndarray,
) -> dict[str, object]:
    model = model_info["model"]
    stand_qpos = model_info["stand_qpos"]
    arm_qpos_ids = model_info["arm_qpos_ids"]
    joint_limits = model_info["joint_limits"]
    neutral_arm_q = model_info["stand_arm_q"]
    data = ik_mod.mujoco.MjData(model)
    best: dict[str, object] | None = None

    initial_guesses = [np.array(init_arm_q, dtype=np.float64, copy=True)]
    initial_guesses.extend(generate_initial_guesses(neutral_arm_q, target_left_pos, target_right_pos))
    initial_guesses.append(np.array(neutral_arm_q, dtype=np.float64, copy=True))

    deduped_guesses: list[np.ndarray] = []
    for guess in initial_guesses:
        if not any(np.allclose(guess, existing, atol=1e-9) for existing in deduped_guesses):
            deduped_guesses.append(guess)

    for initial_q in deduped_guesses:
        q = np.array(initial_q, dtype=np.float64, copy=True)
        for iteration in range(1, MAX_ITERATIONS + 1):
            ik_mod.assign_arm_qpos(data, stand_qpos, arm_qpos_ids, q)
            ik_mod.mujoco.mj_forward(model, data)
            current = position_only_state(model_info, data)
            left_err_vec = target_left_pos - current["left_pos"]
            right_err_vec = target_right_pos - current["right_pos"]
            left_err = float(np.linalg.norm(left_err_vec))
            right_err = float(np.linalg.norm(right_err_vec))
            limit_margin, near_mask, violation = ik_mod.arm_limit_metrics(q, joint_limits)
            result = {
                "arm_q": q.copy(),
                "left_pos": current["left_pos"].copy(),
                "right_pos": current["right_pos"].copy(),
                "left_quat": current["left_quat"].copy(),
                "right_quat": current["right_quat"].copy(),
                "left_pos_err": left_err,
                "right_pos_err": right_err,
                "iteration_count": iteration,
                "limit_margin": limit_margin,
                "near_mask": near_mask.copy(),
                "violation": violation,
                "converged": left_err <= POSITION_TOLERANCE_M and right_err <= POSITION_TOLERANCE_M and not violation,
            }
            if best is None or (left_err + right_err) < (best["left_pos_err"] + best["right_pos_err"]):
                best = result
            if result["converged"]:
                return result

            zero = np.zeros((3, 7), dtype=np.float64)
            J = np.vstack(
                [
                    np.hstack([current["left_jac_pos"], zero]),
                    np.hstack([zero, current["right_jac_pos"]]),
                ]
            )
            error = np.concatenate([left_err_vec, right_err_vec])
            prev_w = math.sqrt(PREVIOUS_Q_WEIGHT)
            neutral_w = math.sqrt(NEUTRAL_Q_WEIGHT)
            J_aug = np.vstack(
                [
                    J,
                    prev_w * np.eye(14, dtype=np.float64),
                    neutral_w * np.eye(14, dtype=np.float64),
                ]
            )
            error_aug = np.concatenate(
                [
                    error,
                    prev_w * (previous_arm_q - q),
                    neutral_w * (neutral_arm_q - q),
                ]
            )
            system = J_aug @ J_aug.T + (DAMPING**2) * np.eye(J_aug.shape[0], dtype=np.float64)
            dq = J_aug.T @ np.linalg.solve(system, error_aug)
            dq = np.clip(dq, -MAX_JOINT_STEP_RAD, MAX_JOINT_STEP_RAD)
            q = np.clip(q + dq, joint_limits[:, 0], joint_limits[:, 1])

    if best is None:
        raise fail("Position-only IK produced no result")
    return best


def anchor_start_positions(center_xyz: np.ndarray, hand_separation: float) -> tuple[np.ndarray, np.ndarray]:
    d_start = np.array([0.0, -hand_separation, 0.0], dtype=np.float64)
    left = center_xyz - 0.5 * d_start
    right = center_xyz + 0.5 * d_start
    return left, right


def build_task_ready_targets(
    aloha: dict[str, np.ndarray],
    center_xyz: np.ndarray,
    hand_separation: float,
    center_scale: float,
    relative_scale: float,
    align_rpy_deg: np.ndarray,
    left_start_quat: np.ndarray,
    right_start_quat: np.ndarray,
) -> dict[str, np.ndarray]:
    left_aloha = np.asarray(aloha["left_position_xyz_m"], dtype=np.float64)
    right_aloha = np.asarray(aloha["right_position_xyz_m"], dtype=np.float64)
    center_aloha = 0.5 * (left_aloha + right_aloha)
    relative_aloha = right_aloha - left_aloha
    delta_center = center_aloha - center_aloha[0]
    delta_relative = relative_aloha - relative_aloha[0]

    align_rot = make_align_rotation(align_rpy_deg)
    g1_delta_center = center_scale * (delta_center @ align_rot.T)
    g1_delta_relative = relative_scale * (delta_relative @ align_rot.T)
    d_start = np.array([0.0, -hand_separation, 0.0], dtype=np.float64)
    center_g1 = center_xyz[None, :] + g1_delta_center
    relative_g1 = d_start[None, :] + g1_delta_relative
    left_g1 = center_g1 - 0.5 * relative_g1
    right_g1 = center_g1 + 0.5 * relative_g1
    left_quat = np.repeat(left_start_quat[None, :], left_g1.shape[0], axis=0)
    right_quat = np.repeat(right_start_quat[None, :], right_g1.shape[0], axis=0)
    hand_distance = np.linalg.norm(right_g1 - left_g1, axis=1)
    return {
        "timestamp_s": np.asarray(aloha["timestamp_s"], dtype=np.float64),
        "g1_left_target_position_xyz_m": left_g1,
        "g1_right_target_position_xyz_m": right_g1,
        "g1_left_target_quaternion_wxyz": left_quat,
        "g1_right_target_quaternion_wxyz": right_quat,
        "g1_center_start_xyz_m": center_xyz,
        "g1_hand_separation_start_m": np.asarray(hand_separation, dtype=np.float64),
        "center_scale": np.asarray(center_scale, dtype=np.float64),
        "relative_scale": np.asarray(relative_scale, dtype=np.float64),
        "align_rotation_rpy_deg": np.asarray(align_rpy_deg, dtype=np.float64),
        "aloha_delta_center_xyz_m": delta_center,
        "aloha_delta_relative_xyz_m": delta_relative,
        "g1_delta_center_xyz_m": g1_delta_center,
        "g1_delta_relative_xyz_m": g1_delta_relative,
        "g1_hands_distance_m": hand_distance,
    }


def evaluate_representative_frames(
    model_info: dict[str, object],
    targets: dict[str, np.ndarray],
    indices: np.ndarray,
    start_arm_q: np.ndarray,
) -> dict[str, np.ndarray | float | int | bool]:
    left_err = np.zeros(indices.shape[0], dtype=np.float64)
    right_err = np.zeros(indices.shape[0], dtype=np.float64)
    converged = np.zeros(indices.shape[0], dtype=bool)
    min_margin = np.zeros(indices.shape[0], dtype=np.float64)
    near_limit_count = 0
    previous_q = np.asarray(start_arm_q, dtype=np.float64)
    violation = False

    for i, frame_idx in enumerate(indices):
        result = solve_position_only_frame(
            model_info,
            targets["g1_left_target_position_xyz_m"][frame_idx],
            targets["g1_right_target_position_xyz_m"][frame_idx],
            previous_q if i > 0 else np.asarray(start_arm_q, dtype=np.float64),
            previous_q,
        )
        left_err[i] = float(result["left_pos_err"])
        right_err[i] = float(result["right_pos_err"])
        converged[i] = bool(result["converged"])
        min_margin[i] = float(result["limit_margin"])
        near_limit_count += int(np.sum(result["near_mask"]))
        violation = violation or bool(result["violation"])
        previous_q = np.asarray(result["arm_q"], dtype=np.float64)

    left_pos = targets["g1_left_target_position_xyz_m"]
    right_pos = targets["g1_right_target_position_xyz_m"]
    max_step = float(
        max(
            np.max(np.linalg.norm(np.diff(left_pos, axis=0), axis=1)),
            np.max(np.linalg.norm(np.diff(right_pos, axis=0), axis=1)),
        )
    )
    mean_pos = float(np.mean(np.concatenate([left_err, right_err])))
    max_pos = float(np.max(np.concatenate([left_err, right_err])))
    nonconverged = int(np.sum(~converged))
    score = (
        10.0 * mean_pos
        + 5.0 * max_pos
        + 0.5 * nonconverged
        + 0.2 * near_limit_count
        + 0.1 * max_step
    )
    return {
        "left_position_error_m": left_err,
        "right_position_error_m": right_err,
        "converged": converged,
        "converged_count": int(np.sum(converged)),
        "joint_limit_min_margin_rad": min_margin,
        "joint_limit_violation": violation,
        "near_joint_limit_count": near_limit_count,
        "mean_position_error": mean_pos,
        "max_position_error": max_pos,
        "max_frame_to_frame_target_displacement": max_step,
        "score": score,
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    search_output = args.search_output.expanduser().resolve()
    best_output = args.best_output.expanduser().resolve()
    require_writable_output(search_output, args.overwrite)
    require_writable_output(best_output, args.overwrite)

    aloha = transform_mod.load_npz(input_path)
    frame_count = int(aloha["timestamp_s"].shape[0])
    rep_indices = sample_indices(frame_count)
    model_info = ik_mod.validate_model(model_path)

    anchor_centers = []
    anchor_separations = []
    anchor_valid = []
    anchor_frame0_left_err = []
    anchor_frame0_right_err = []
    anchor_joint_margin = []
    anchor_start_arm_q = []
    anchor_left_start_pos = []
    anchor_right_start_pos = []
    anchor_left_start_quat = []
    anchor_right_start_quat = []

    for cx in CENTER_X_VALUES:
        for cz in CENTER_Z_VALUES:
            center_xyz = np.array([cx, 0.0, cz], dtype=np.float64)
            for sep in HAND_SEPARATION_VALUES:
                left_start_pos, right_start_pos = anchor_start_positions(center_xyz, float(sep))
                frame0 = solve_position_only_frame(
                    model_info,
                    left_start_pos,
                    right_start_pos,
                    np.asarray(model_info["stand_arm_q"], dtype=np.float64),
                    np.asarray(model_info["stand_arm_q"], dtype=np.float64),
                )
                valid = (
                    frame0["left_pos_err"] <= POSITION_TOLERANCE_M
                    and frame0["right_pos_err"] <= POSITION_TOLERANCE_M
                    and not frame0["violation"]
                    and HAND_DISTANCE_MIN_M <= sep <= HAND_DISTANCE_MAX_M
                )
                anchor_centers.append(center_xyz.copy())
                anchor_separations.append(float(sep))
                anchor_valid.append(bool(valid))
                anchor_frame0_left_err.append(float(frame0["left_pos_err"]))
                anchor_frame0_right_err.append(float(frame0["right_pos_err"]))
                anchor_joint_margin.append(float(frame0["limit_margin"]))
                anchor_start_arm_q.append(np.asarray(frame0["arm_q"], dtype=np.float64))
                anchor_left_start_pos.append(np.asarray(frame0["left_pos"], dtype=np.float64))
                anchor_right_start_pos.append(np.asarray(frame0["right_pos"], dtype=np.float64))
                anchor_left_start_quat.append(np.asarray(frame0["left_quat"], dtype=np.float64))
                anchor_right_start_quat.append(np.asarray(frame0["right_quat"], dtype=np.float64))

    anchor_centers_arr = np.asarray(anchor_centers, dtype=np.float64)
    anchor_separations_arr = np.asarray(anchor_separations, dtype=np.float64)
    anchor_valid_arr = np.asarray(anchor_valid, dtype=bool)
    anchor_frame0_left_err_arr = np.asarray(anchor_frame0_left_err, dtype=np.float64)
    anchor_frame0_right_err_arr = np.asarray(anchor_frame0_right_err, dtype=np.float64)
    anchor_joint_margin_arr = np.asarray(anchor_joint_margin, dtype=np.float64)
    anchor_start_arm_q_arr = np.asarray(anchor_start_arm_q, dtype=np.float64)
    anchor_left_start_pos_arr = np.asarray(anchor_left_start_pos, dtype=np.float64)
    anchor_right_start_pos_arr = np.asarray(anchor_right_start_pos, dtype=np.float64)
    anchor_left_start_quat_arr = np.asarray(anchor_left_start_quat, dtype=np.float64)
    anchor_right_start_quat_arr = np.asarray(anchor_right_start_quat, dtype=np.float64)

    surviving_anchor_indices = np.flatnonzero(anchor_valid_arr)

    coarse_center_scale_values = np.array([0.45, 0.55, 0.65], dtype=np.float64)
    coarse_relative_scale_values = np.array([0.45, 0.60, 1.00], dtype=np.float64)
    tested = []

    for anchor_idx in surviving_anchor_indices.tolist():
        for center_scale in coarse_center_scale_values:
            for relative_scale in coarse_relative_scale_values:
                for align_rpy_deg in ALIGN_RPY_VALUES:
                    targets = build_task_ready_targets(
                        aloha,
                        anchor_centers_arr[anchor_idx],
                        float(anchor_separations_arr[anchor_idx]),
                        float(center_scale),
                        float(relative_scale),
                        align_rpy_deg,
                        anchor_left_start_quat_arr[anchor_idx],
                        anchor_right_start_quat_arr[anchor_idx],
                    )
                    finite = np.isfinite(targets["g1_left_target_position_xyz_m"]).all() and np.isfinite(targets["g1_right_target_position_xyz_m"]).all()
                    hand_dist = targets["g1_hands_distance_m"]
                    hand_dist_ok = bool(np.all((hand_dist >= HAND_DISTANCE_MIN_M) & (hand_dist <= HAND_DISTANCE_MAX_M)))
                    if finite and hand_dist_ok:
                        metrics = evaluate_representative_frames(model_info, targets, rep_indices, anchor_start_arm_q_arr[anchor_idx])
                    else:
                        metrics = {
                            "left_position_error_m": np.full(rep_indices.shape[0], np.inf, dtype=np.float64),
                            "right_position_error_m": np.full(rep_indices.shape[0], np.inf, dtype=np.float64),
                            "converged": np.zeros(rep_indices.shape[0], dtype=bool),
                            "converged_count": 0,
                            "joint_limit_min_margin_rad": np.full(rep_indices.shape[0], -np.inf, dtype=np.float64),
                            "joint_limit_violation": True,
                            "near_joint_limit_count": 0,
                            "mean_position_error": np.inf,
                            "max_position_error": np.inf,
                            "max_frame_to_frame_target_displacement": np.inf,
                            "score": np.inf,
                        }
                    valid = (
                        finite
                        and hand_dist_ok
                        and bool(metrics["left_position_error_m"][0] <= POSITION_TOLERANCE_M)
                        and bool(metrics["right_position_error_m"][0] <= POSITION_TOLERANCE_M)
                        and not bool(metrics["joint_limit_violation"])
                    )
                    tested.append(
                        {
                            "anchor_idx": anchor_idx,
                            "center_xyz": anchor_centers_arr[anchor_idx].copy(),
                            "hand_separation": float(anchor_separations_arr[anchor_idx]),
                            "center_scale": float(center_scale),
                            "relative_scale": float(relative_scale),
                            "align_rpy_deg": np.asarray(align_rpy_deg, dtype=np.float64).copy(),
                            "valid": valid,
                            "metrics": metrics,
                        }
                    )

    finite_coarse = [item for item in tested if np.isfinite(item["metrics"]["score"])]
    finite_coarse.sort(key=lambda item: item["metrics"]["score"])
    top_seed = finite_coarse[:15]

    fine_keys = {(item["anchor_idx"], item["center_scale"], item["relative_scale"], tuple(item["align_rpy_deg"].tolist())) for item in tested}
    for item in top_seed:
        center_matches = np.where(np.isclose(CENTER_SCALE_VALUES, item["center_scale"], atol=1e-9))[0]
        rel_matches = np.where(np.isclose(RELATIVE_SCALE_VALUES, item["relative_scale"], atol=1e-9))[0]
        if center_matches.size == 0 or rel_matches.size == 0:
            continue
        center_idx = int(center_matches[0])
        rel_idx = int(rel_matches[0])
        center_candidates = CENTER_SCALE_VALUES[max(0, center_idx - 1) : min(len(CENTER_SCALE_VALUES), center_idx + 2)]
        rel_candidates = RELATIVE_SCALE_VALUES[max(0, rel_idx - 1) : min(len(RELATIVE_SCALE_VALUES), rel_idx + 2)]
        for center_scale in center_candidates:
            for relative_scale in rel_candidates:
                key = (item["anchor_idx"], float(center_scale), float(relative_scale), tuple(item["align_rpy_deg"].tolist()))
                if key in fine_keys:
                    continue
                targets = build_task_ready_targets(
                    aloha,
                    anchor_centers_arr[item["anchor_idx"]],
                    float(anchor_separations_arr[item["anchor_idx"]]),
                    float(center_scale),
                    float(relative_scale),
                    item["align_rpy_deg"],
                    anchor_left_start_quat_arr[item["anchor_idx"]],
                    anchor_right_start_quat_arr[item["anchor_idx"]],
                )
                finite = np.isfinite(targets["g1_left_target_position_xyz_m"]).all() and np.isfinite(targets["g1_right_target_position_xyz_m"]).all()
                hand_dist = targets["g1_hands_distance_m"]
                hand_dist_ok = bool(np.all((hand_dist >= HAND_DISTANCE_MIN_M) & (hand_dist <= HAND_DISTANCE_MAX_M)))
                if finite and hand_dist_ok:
                    metrics = evaluate_representative_frames(model_info, targets, rep_indices, anchor_start_arm_q_arr[item["anchor_idx"]])
                else:
                    metrics = {
                        "left_position_error_m": np.full(rep_indices.shape[0], np.inf, dtype=np.float64),
                        "right_position_error_m": np.full(rep_indices.shape[0], np.inf, dtype=np.float64),
                        "converged": np.zeros(rep_indices.shape[0], dtype=bool),
                        "converged_count": 0,
                        "joint_limit_min_margin_rad": np.full(rep_indices.shape[0], -np.inf, dtype=np.float64),
                        "joint_limit_violation": True,
                        "near_joint_limit_count": 0,
                        "mean_position_error": np.inf,
                        "max_position_error": np.inf,
                        "max_frame_to_frame_target_displacement": np.inf,
                        "score": np.inf,
                    }
                valid = (
                    finite
                    and hand_dist_ok
                    and bool(metrics["left_position_error_m"][0] <= POSITION_TOLERANCE_M)
                    and bool(metrics["right_position_error_m"][0] <= POSITION_TOLERANCE_M)
                    and not bool(metrics["joint_limit_violation"])
                )
                tested.append(
                    {
                        "anchor_idx": item["anchor_idx"],
                        "center_xyz": anchor_centers_arr[item["anchor_idx"]].copy(),
                        "hand_separation": float(anchor_separations_arr[item["anchor_idx"]]),
                        "center_scale": float(center_scale),
                        "relative_scale": float(relative_scale),
                        "align_rpy_deg": item["align_rpy_deg"].copy(),
                        "valid": valid,
                        "metrics": metrics,
                    }
                )
                fine_keys.add(key)

    scores = np.array([item["metrics"]["score"] for item in tested], dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    best_idx = int(order[0])
    best = tested[best_idx]
    best_anchor_idx = int(best["anchor_idx"])
    best_targets = build_task_ready_targets(
        aloha,
        best["center_xyz"],
        best["hand_separation"],
        best["center_scale"],
        best["relative_scale"],
        best["align_rpy_deg"],
        anchor_left_start_quat_arr[best_anchor_idx],
        anchor_right_start_quat_arr[best_anchor_idx],
    )

    task_ready_payload = {
        "timestamp_s": best_targets["timestamp_s"],
        "g1_left_target_position_xyz_m": best_targets["g1_left_target_position_xyz_m"],
        "g1_right_target_position_xyz_m": best_targets["g1_right_target_position_xyz_m"],
        "g1_left_target_quaternion_wxyz": best_targets["g1_left_target_quaternion_wxyz"],
        "g1_right_target_quaternion_wxyz": best_targets["g1_right_target_quaternion_wxyz"],
        "g1_task_ready_left_start_position_xyz_m": anchor_left_start_pos_arr[best_anchor_idx],
        "g1_task_ready_right_start_position_xyz_m": anchor_right_start_pos_arr[best_anchor_idx],
        "g1_task_ready_left_start_quaternion_wxyz": anchor_left_start_quat_arr[best_anchor_idx],
        "g1_task_ready_right_start_quaternion_wxyz": anchor_right_start_quat_arr[best_anchor_idx],
        "g1_task_ready_arm_q": anchor_start_arm_q_arr[best_anchor_idx],
        "g1_center_start_xyz_m": best["center_xyz"],
        "g1_hand_separation_start_m": np.asarray(best["hand_separation"], dtype=np.float64),
        "center_scale": np.asarray(best["center_scale"], dtype=np.float64),
        "relative_scale": np.asarray(best["relative_scale"], dtype=np.float64),
        "align_rotation_rpy_deg": best["align_rpy_deg"],
        "aloha_delta_center_xyz_m": best_targets["aloha_delta_center_xyz_m"],
        "aloha_delta_relative_xyz_m": best_targets["aloha_delta_relative_xyz_m"],
        "g1_delta_center_xyz_m": best_targets["g1_delta_center_xyz_m"],
        "g1_delta_relative_xyz_m": best_targets["g1_delta_relative_xyz_m"],
    }

    np.savez(
        search_output,
        anchor_center_xyz_m=anchor_centers_arr,
        anchor_hand_separation_m=anchor_separations_arr,
        anchor_frame0_valid=anchor_valid_arr,
        anchor_frame0_left_position_error_m=anchor_frame0_left_err_arr,
        anchor_frame0_right_position_error_m=anchor_frame0_right_err_arr,
        anchor_frame0_joint_limit_min_margin_rad=anchor_joint_margin_arr,
        tested_center_xyz_m=np.asarray([item["center_xyz"] for item in tested], dtype=np.float64),
        tested_hand_separation_m=np.asarray([item["hand_separation"] for item in tested], dtype=np.float64),
        tested_center_scale=np.asarray([item["center_scale"] for item in tested], dtype=np.float64),
        tested_relative_scale=np.asarray([item["relative_scale"] for item in tested], dtype=np.float64),
        tested_align_rotation_rpy_deg=np.asarray([item["align_rpy_deg"] for item in tested], dtype=np.float64),
        valid_anchor_mask=np.asarray([item["valid"] for item in tested], dtype=bool),
        sample_frame_indices=rep_indices,
        left_position_error_m=np.asarray([item["metrics"]["left_position_error_m"] for item in tested], dtype=np.float64),
        right_position_error_m=np.asarray([item["metrics"]["right_position_error_m"] for item in tested], dtype=np.float64),
        converged_mask=np.asarray([item["metrics"]["converged"] for item in tested], dtype=bool),
        converged_count=np.asarray([item["metrics"]["converged_count"] for item in tested], dtype=np.int64),
        joint_limit_min_margin_rad=np.asarray([item["metrics"]["joint_limit_min_margin_rad"] for item in tested], dtype=np.float64),
        near_joint_limit_count=np.asarray([item["metrics"]["near_joint_limit_count"] for item in tested], dtype=np.int64),
        mean_position_error=np.asarray([item["metrics"]["mean_position_error"] for item in tested], dtype=np.float64),
        max_position_error=np.asarray([item["metrics"]["max_position_error"] for item in tested], dtype=np.float64),
        max_frame_to_frame_target_displacement=np.asarray([item["metrics"]["max_frame_to_frame_target_displacement"] for item in tested], dtype=np.float64),
        score=scores,
        rank=ranks,
        best_index=np.asarray(best_idx, dtype=np.int64),
    )
    np.savez(best_output, **task_ready_payload)

    search_npz = np.load(search_output, allow_pickle=False)
    best_npz = np.load(best_output, allow_pickle=False)

    total_candidates = len(tested)
    frame0_pass_count = int(np.sum(anchor_valid_arr))
    all5_count = int(np.sum(np.asarray([item["metrics"]["converged_count"] == rep_indices.shape[0] for item in tested], dtype=bool)))
    top10 = order[: min(10, total_candidates)]

    print(f"generated python file: {Path(__file__).resolve()}")
    print(f"search npz: {search_output}")
    print(f"best targets npz: {best_output}")
    print(f"evaluated total candidates: {total_candidates}")
    print(f"frame 0 IK passing anchors: {frame0_pass_count}")
    print(f"all 5 representative frames converged candidates: {all5_count}")
    print("top 10 candidates:")
    for rank_idx, idx in enumerate(top10, start=1):
        item = tested[int(idx)]
        print(
            f"  {rank_idx:02d}: score={item['metrics']['score']:.6f} conv={item['metrics']['converged_count']}/5 "
            f"center={item['center_xyz'].tolist()} sep={item['hand_separation']:.2f} "
            f"c_scale={item['center_scale']:.2f} r_scale={item['relative_scale']:.2f} align={item['align_rpy_deg'].tolist()}"
        )
    print(f"best center xyz: {best['center_xyz'].tolist()}")
    print(f"best hand separation: {best['hand_separation']:.2f}")
    print(f"best center_scale: {best['center_scale']:.2f}")
    print(f"best relative_scale: {best['relative_scale']:.2f}")
    print(f"best align rotation: {best['align_rpy_deg'].tolist()}")
    for i, frame_idx in enumerate(rep_indices):
        print(
            f"frame {int(frame_idx)} position error: "
            f"left={best['metrics']['left_position_error_m'][i]:.6f} "
            f"right={best['metrics']['right_position_error_m'][i]:.6f}"
        )
    print(f"joint-limit minimum margin: {float(np.min(best['metrics']['joint_limit_min_margin_rad'])):.6f}")
    print(f"search npz keys: {sorted(search_npz.files)}")
    print(f"best npz keys: {sorted(best_npz.files)}")
    ready_for_full = bool(best["metrics"]["converged_count"] == rep_indices.shape[0] and np.max(best["metrics"]["left_position_error_m"]) <= POSITION_TOLERANCE_M and np.max(best["metrics"]["right_position_error_m"]) <= POSITION_TOLERANCE_M)
    print(f"ready for full 713-frame IK: {ready_for_full}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
