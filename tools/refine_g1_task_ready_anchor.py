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

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import search_g1_task_ready_anchor as anchor_mod
import validate_g1_targets_and_sparse_ik as ik_mod


DEFAULT_INPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/aloha_tcp_trajectory.npz")
DEFAULT_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEFAULT_REFINED_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_task_ready_anchor_refined_search.npz")
DEFAULT_FULL_IK_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_full_trajectory_ik_top3.npz")
DEFAULT_FRAME300_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_frame300_diagnostics.npz")

REPRESENTATIVE_FRAMES = np.array([0, 100, 300, 500, 712], dtype=np.int64)
STAGE1_CENTER = np.array([0.3, 0.0, 0.85], dtype=np.float64)
STAGE1_CENTER_SCALE = 0.45
HAND_SEPARATION_VALUES = np.array([0.34, 0.35, 0.36, 0.37, 0.38], dtype=np.float64)
RELATIVE_SCALE_VALUES = np.array([0.42, 0.435, 0.45, 0.465, 0.48], dtype=np.float64)
ALIGN_PITCH_VALUES_DEG = np.array([-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0], dtype=np.float64)

CENTER_X_VALUES = np.array([0.29, 0.295, 0.30, 0.305, 0.31], dtype=np.float64)
CENTER_Z_VALUES = np.array([0.84, 0.845, 0.85, 0.855, 0.86], dtype=np.float64)
CENTER_SCALE_VALUES = np.array([0.42, 0.435, 0.45, 0.465, 0.48], dtype=np.float64)

POSITION_TOLERANCE_M = 0.01
HAND_DISTANCE_MIN_M = 0.12
HAND_DISTANCE_MAX_M = 0.60
ORIENTATION_WEIGHT_BASE = 0.15
ORIENTATION_WEIGHT_SCALES = np.array([1.0, 0.5, 0.25, 0.1], dtype=np.float64)
FULL_TRAJECTORY_MAX_ITERATIONS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine G1 task-ready anchor search and run top-3 full-trajectory IK.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--refined-output", type=Path, default=DEFAULT_REFINED_OUTPUT)
    parser.add_argument("--full-ik-output", type=Path, default=DEFAULT_FULL_IK_OUTPUT)
    parser.add_argument("--frame300-output", type=Path, default=DEFAULT_FRAME300_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fail(message: str) -> RuntimeError:
    return RuntimeError(message)


def require_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise fail(f"Output already exists. Use --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_aloha(path: Path) -> dict[str, np.ndarray]:
    payload = anchor_mod.transform_mod.load_npz(path)
    required = [
        "timestamp_s",
        "left_position_xyz_m",
        "right_position_xyz_m",
        "left_quaternion_wxyz",
        "right_quaternion_wxyz",
        "hands_distance_m",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise fail(f"Missing required ALOHA FK keys: {missing}")
    frame_count = int(payload["timestamp_s"].shape[0])
    if frame_count <= 712:
        raise fail(f"Expected at least 713 frames, found {frame_count}")
    for key in required:
        value = np.asarray(payload[key])
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise fail(f"{key} contains NaN or inf")
    return payload


def quat_to_rot(quat_wxyz: np.ndarray) -> np.ndarray:
    return ik_mod.quat_to_mat(np.asarray(quat_wxyz, dtype=np.float64))


def orientation_error_rad(current_quat: np.ndarray, target_quat: np.ndarray) -> float:
    current_rot = quat_to_rot(current_quat)
    target_rot = quat_to_rot(target_quat)
    return float(np.linalg.norm(ik_mod.rotation_log_world(current_rot, target_rot)))


def hand_distance_ok(left_pos: np.ndarray, right_pos: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(right_pos - left_pos, axis=1)
    return (dist >= HAND_DISTANCE_MIN_M) & (dist <= HAND_DISTANCE_MAX_M)


def build_candidate_targets(
    aloha: dict[str, np.ndarray],
    center_xyz: np.ndarray,
    hand_separation: float,
    center_scale: float,
    relative_scale: float,
    align_rpy_deg: np.ndarray,
    left_start_quat: np.ndarray,
    right_start_quat: np.ndarray,
) -> dict[str, np.ndarray]:
    targets = anchor_mod.build_task_ready_targets(
        aloha,
        center_xyz,
        hand_separation,
        center_scale,
        relative_scale,
        align_rpy_deg,
        left_start_quat,
        right_start_quat,
    )
    targets["g1_task_ready_left_start_position_xyz_m"] = targets["g1_left_target_position_xyz_m"][0].copy()
    targets["g1_task_ready_right_start_position_xyz_m"] = targets["g1_right_target_position_xyz_m"][0].copy()
    targets["g1_task_ready_left_start_quaternion_wxyz"] = left_start_quat.copy()
    targets["g1_task_ready_right_start_quaternion_wxyz"] = right_start_quat.copy()
    return targets


def frame_success(left_err: float, right_err: float) -> bool:
    return left_err <= POSITION_TOLERANCE_M and right_err <= POSITION_TOLERANCE_M


def max_frame_step(targets: dict[str, np.ndarray]) -> float:
    left_diff = np.linalg.norm(np.diff(targets["g1_left_target_position_xyz_m"], axis=0), axis=1)
    right_diff = np.linalg.norm(np.diff(targets["g1_right_target_position_xyz_m"], axis=0), axis=1)
    return float(max(np.max(left_diff), np.max(right_diff)))


def score_candidate(
    mean_pos: float,
    max_pos: float,
    mean_ori: float,
    failed_frame_count: int,
    solver_failure_count: int,
    joint_violation_count: int,
) -> float:
    penalty = 0.5 * failed_frame_count + 1.0 * solver_failure_count + 5.0 * joint_violation_count
    return float(mean_pos + 2.0 * max_pos + 0.1 * mean_ori + penalty)


def evaluate_representative_candidate(
    model_info: dict[str, object],
    targets: dict[str, np.ndarray],
    indices: np.ndarray,
    start_arm_q: np.ndarray,
) -> dict[str, np.ndarray | float | int | bool]:
    count = indices.shape[0]
    left_pos_err = np.zeros(count, dtype=np.float64)
    right_pos_err = np.zeros(count, dtype=np.float64)
    left_ori_err = np.zeros(count, dtype=np.float64)
    right_ori_err = np.zeros(count, dtype=np.float64)
    solver_success = np.zeros(count, dtype=bool)
    iterations = np.zeros(count, dtype=np.int64)
    margin = np.zeros(count, dtype=np.float64)
    joint_delta_l2 = np.zeros(count, dtype=np.float64)
    joint_delta_maxabs = np.zeros(count, dtype=np.float64)
    both_hands_success = np.zeros(count, dtype=bool)
    hands_within_count = np.zeros(count, dtype=np.int64)
    arm_q = np.zeros((count, 14), dtype=np.float64)
    achieved_left_pos = np.zeros((count, 3), dtype=np.float64)
    achieved_right_pos = np.zeros((count, 3), dtype=np.float64)
    achieved_left_quat = np.zeros((count, 4), dtype=np.float64)
    achieved_right_quat = np.zeros((count, 4), dtype=np.float64)
    near_mask = np.zeros((count, 14), dtype=bool)
    violation_mask = np.zeros(count, dtype=bool)

    previous_q = np.array(start_arm_q, dtype=np.float64, copy=True)
    for out_idx, frame_idx in enumerate(indices):
        init_q = previous_q if out_idx > 0 else np.array(start_arm_q, dtype=np.float64, copy=True)
        result = anchor_mod.solve_position_only_frame(
            model_info,
            targets["g1_left_target_position_xyz_m"][frame_idx],
            targets["g1_right_target_position_xyz_m"][frame_idx],
            init_q,
            previous_q,
        )
        arm_q[out_idx] = np.asarray(result["arm_q"], dtype=np.float64)
        achieved_left_pos[out_idx] = np.asarray(result["left_pos"], dtype=np.float64)
        achieved_right_pos[out_idx] = np.asarray(result["right_pos"], dtype=np.float64)
        achieved_left_quat[out_idx] = np.asarray(result["left_quat"], dtype=np.float64)
        achieved_right_quat[out_idx] = np.asarray(result["right_quat"], dtype=np.float64)
        left_pos_err[out_idx] = float(result["left_pos_err"])
        right_pos_err[out_idx] = float(result["right_pos_err"])
        left_ori_err[out_idx] = orientation_error_rad(result["left_quat"], targets["g1_left_target_quaternion_wxyz"][frame_idx])
        right_ori_err[out_idx] = orientation_error_rad(result["right_quat"], targets["g1_right_target_quaternion_wxyz"][frame_idx])
        iterations[out_idx] = int(result["iteration_count"])
        margin[out_idx] = float(result["limit_margin"])
        near_mask[out_idx] = np.asarray(result["near_mask"], dtype=bool)
        violation_mask[out_idx] = bool(result["violation"])
        solver_success[out_idx] = bool(result["converged"])
        if out_idx == 0:
            delta = arm_q[out_idx] - np.asarray(start_arm_q, dtype=np.float64)
        else:
            delta = arm_q[out_idx] - arm_q[out_idx - 1]
        joint_delta_l2[out_idx] = float(np.linalg.norm(delta))
        joint_delta_maxabs[out_idx] = float(np.max(np.abs(delta)))
        hands_within_count[out_idx] = int((left_pos_err[out_idx] <= POSITION_TOLERANCE_M) + (right_pos_err[out_idx] <= POSITION_TOLERANCE_M))
        both_hands_success[out_idx] = frame_success(left_pos_err[out_idx], right_pos_err[out_idx]) and not violation_mask[out_idx]
        previous_q = arm_q[out_idx]

    concatenated_pos = np.concatenate([left_pos_err, right_pos_err])
    concatenated_ori = np.concatenate([left_ori_err, right_ori_err])
    failed_count = int(np.sum(~both_hands_success))
    solver_failure_count = int(np.sum(~solver_success))
    violation_count = int(np.sum(violation_mask))
    return {
        "left_position_error_m": left_pos_err,
        "right_position_error_m": right_pos_err,
        "left_orientation_error_rad": left_ori_err,
        "right_orientation_error_rad": right_ori_err,
        "solver_success": solver_success,
        "iteration_count": iterations,
        "joint_limit_min_margin_rad": margin,
        "joint_delta_l2_rad": joint_delta_l2,
        "joint_delta_maxabs_rad": joint_delta_maxabs,
        "hands_within_1cm_count": hands_within_count,
        "both_hands_within_1cm": both_hands_success,
        "both_hands_within_1cm_count": int(np.sum(both_hands_success)),
        "mean_position_error_m": float(np.mean(concatenated_pos)),
        "max_position_error_m": float(np.max(concatenated_pos)),
        "p95_position_error_m": float(np.percentile(concatenated_pos, 95.0)),
        "mean_orientation_error_rad": float(np.mean(concatenated_ori)),
        "joint_limit_violation_count": violation_count,
        "solver_failure_count": solver_failure_count,
        "score": score_candidate(
            float(np.mean(concatenated_pos)),
            float(np.max(concatenated_pos)),
            float(np.mean(concatenated_ori)),
            failed_count,
            solver_failure_count,
            violation_count,
        ),
        "min_joint_limit_margin_rad_overall": float(np.min(margin)),
        "arm_q": arm_q,
        "achieved_left_position_xyz_m": achieved_left_pos,
        "achieved_right_position_xyz_m": achieved_right_pos,
        "achieved_left_quaternion_wxyz": achieved_left_quat,
        "achieved_right_quaternion_wxyz": achieved_right_quat,
        "near_joint_limit_mask": near_mask,
    }


def compute_task_ready_start(
    model_info: dict[str, object],
    center_xyz: np.ndarray,
    hand_separation: float,
) -> dict[str, np.ndarray | float | bool]:
    left_start_pos, right_start_pos = anchor_mod.anchor_start_positions(center_xyz, hand_separation)
    frame0 = anchor_mod.solve_position_only_frame(
        model_info,
        left_start_pos,
        right_start_pos,
        np.asarray(model_info["stand_arm_q"], dtype=np.float64),
        np.asarray(model_info["stand_arm_q"], dtype=np.float64),
    )
    valid = (
        float(frame0["left_pos_err"]) <= POSITION_TOLERANCE_M
        and float(frame0["right_pos_err"]) <= POSITION_TOLERANCE_M
        and not bool(frame0["violation"])
        and HAND_DISTANCE_MIN_M <= hand_separation <= HAND_DISTANCE_MAX_M
    )
    return {
        "valid": valid,
        "center_xyz": center_xyz.copy(),
        "hand_separation": float(hand_separation),
        "left_start_position_xyz_m": np.asarray(frame0["left_pos"], dtype=np.float64),
        "right_start_position_xyz_m": np.asarray(frame0["right_pos"], dtype=np.float64),
        "left_start_quaternion_wxyz": np.asarray(frame0["left_quat"], dtype=np.float64),
        "right_start_quaternion_wxyz": np.asarray(frame0["right_quat"], dtype=np.float64),
        "arm_q": np.asarray(frame0["arm_q"], dtype=np.float64),
        "left_pos_err": float(frame0["left_pos_err"]),
        "right_pos_err": float(frame0["right_pos_err"]),
        "joint_limit_min_margin_rad": float(frame0["limit_margin"]),
    }


def candidate_sort_key(candidate: dict[str, object]) -> tuple[float, float, float, float]:
    metrics = candidate["metrics"]
    return (
        -float(metrics["both_hands_within_1cm_count"]),
        float(metrics["max_position_error_m"]),
        float(metrics["score"]),
        -float(metrics["min_joint_limit_margin_rad_overall"]),
    )


def finalize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(records, key=candidate_sort_key)
    for rank, record in enumerate(ordered, start=1):
        record["rank"] = rank
    return ordered


def build_stage_arrays(records: list[dict[str, object]], rep_count: int) -> dict[str, np.ndarray]:
    n = len(records)
    out = {
        "center_xyz_m": np.zeros((n, 3), dtype=np.float64),
        "hand_separation_m": np.zeros(n, dtype=np.float64),
        "center_scale": np.zeros(n, dtype=np.float64),
        "relative_scale": np.zeros(n, dtype=np.float64),
        "align_rotation_rpy_deg": np.zeros((n, 3), dtype=np.float64),
        "valid_anchor_mask": np.zeros(n, dtype=bool),
        "left_position_error_m": np.zeros((n, rep_count), dtype=np.float64),
        "right_position_error_m": np.zeros((n, rep_count), dtype=np.float64),
        "left_orientation_error_rad": np.zeros((n, rep_count), dtype=np.float64),
        "right_orientation_error_rad": np.zeros((n, rep_count), dtype=np.float64),
        "solver_success": np.zeros((n, rep_count), dtype=bool),
        "iteration_count": np.zeros((n, rep_count), dtype=np.int64),
        "joint_limit_min_margin_rad": np.zeros((n, rep_count), dtype=np.float64),
        "joint_delta_l2_rad": np.zeros((n, rep_count), dtype=np.float64),
        "joint_delta_maxabs_rad": np.zeros((n, rep_count), dtype=np.float64),
        "hands_within_1cm_count": np.zeros((n, rep_count), dtype=np.int64),
        "both_hands_within_1cm": np.zeros((n, rep_count), dtype=bool),
        "both_hands_within_1cm_count": np.zeros(n, dtype=np.int64),
        "mean_position_error_m": np.zeros(n, dtype=np.float64),
        "max_position_error_m": np.zeros(n, dtype=np.float64),
        "p95_position_error_m": np.zeros(n, dtype=np.float64),
        "mean_orientation_error_rad": np.zeros(n, dtype=np.float64),
        "min_joint_limit_margin_rad_overall": np.zeros(n, dtype=np.float64),
        "score": np.zeros(n, dtype=np.float64),
        "rank": np.zeros(n, dtype=np.int64),
    }
    finite_large = 1e9
    finite_negative_large = -1e9
    for i, record in enumerate(records):
        metrics = record["metrics"]
        out["center_xyz_m"][i] = record["center_xyz"]
        out["hand_separation_m"][i] = float(record["hand_separation"])
        out["center_scale"][i] = float(record["center_scale"])
        out["relative_scale"][i] = float(record["relative_scale"])
        out["align_rotation_rpy_deg"][i] = record["align_rpy_deg"]
        out["valid_anchor_mask"][i] = bool(record["valid_anchor"])
        for key in [
            "left_position_error_m",
            "right_position_error_m",
            "left_orientation_error_rad",
            "right_orientation_error_rad",
            "solver_success",
            "iteration_count",
            "joint_limit_min_margin_rad",
            "joint_delta_l2_rad",
            "joint_delta_maxabs_rad",
            "hands_within_1cm_count",
            "both_hands_within_1cm",
        ]:
            value = np.asarray(metrics[key])
            if np.issubdtype(value.dtype, np.floating):
                fill = finite_negative_large if "margin" in key else finite_large
                value = np.nan_to_num(value, nan=fill, posinf=fill, neginf=fill)
            out[key][i] = value
        for key in [
            "both_hands_within_1cm_count",
            "mean_position_error_m",
            "max_position_error_m",
            "p95_position_error_m",
            "mean_orientation_error_rad",
            "min_joint_limit_margin_rad_overall",
            "score",
            "rank",
        ]:
            scalar = metrics[key] if key in metrics else record[key]
            if isinstance(scalar, (float, np.floating)):
                fill = finite_negative_large if "margin" in key else finite_large
                scalar = np.nan_to_num(scalar, nan=fill, posinf=fill, neginf=fill)
            out[key][i] = scalar
    return out


def stage1_search(
    aloha: dict[str, np.ndarray],
    model_info: dict[str, object],
    rep_indices: np.ndarray,
) -> tuple[list[dict[str, object]], dict[float, dict[str, object]]]:
    anchor_cache: dict[float, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    for hand_separation in HAND_SEPARATION_VALUES.tolist():
        anchor_state = compute_task_ready_start(model_info, STAGE1_CENTER, float(hand_separation))
        anchor_cache[float(hand_separation)] = anchor_state
        for relative_scale in RELATIVE_SCALE_VALUES.tolist():
            for pitch_deg in ALIGN_PITCH_VALUES_DEG.tolist():
                align_rpy_deg = np.array([0.0, pitch_deg, 0.0], dtype=np.float64)
                if anchor_state["valid"]:
                    targets = build_candidate_targets(
                        aloha,
                        STAGE1_CENTER,
                        float(hand_separation),
                        STAGE1_CENTER_SCALE,
                        float(relative_scale),
                        align_rpy_deg,
                        np.asarray(anchor_state["left_start_quaternion_wxyz"], dtype=np.float64),
                        np.asarray(anchor_state["right_start_quaternion_wxyz"], dtype=np.float64),
                    )
                    finite = np.isfinite(targets["g1_left_target_position_xyz_m"]).all() and np.isfinite(targets["g1_right_target_position_xyz_m"]).all()
                    hand_ok = bool(np.all(hand_distance_ok(targets["g1_left_target_position_xyz_m"], targets["g1_right_target_position_xyz_m"])))
                    if finite and hand_ok:
                        metrics = evaluate_representative_candidate(
                            model_info,
                            targets,
                            rep_indices,
                            np.asarray(anchor_state["arm_q"], dtype=np.float64),
                        )
                    else:
                        metrics = evaluate_representative_candidate(
                            model_info,
                            build_candidate_targets(
                                aloha,
                                STAGE1_CENTER,
                                float(hand_separation),
                                STAGE1_CENTER_SCALE,
                                float(relative_scale),
                                align_rpy_deg,
                                np.asarray(anchor_state["left_start_quaternion_wxyz"], dtype=np.float64),
                                np.asarray(anchor_state["right_start_quaternion_wxyz"], dtype=np.float64),
                            ),
                            rep_indices,
                            np.asarray(anchor_state["arm_q"], dtype=np.float64),
                        )
                        metrics["score"] = float(metrics["score"] + 100.0)
                else:
                    inf_vec = np.full(rep_indices.shape[0], np.inf, dtype=np.float64)
                    metrics = {
                        "left_position_error_m": inf_vec.copy(),
                        "right_position_error_m": inf_vec.copy(),
                        "left_orientation_error_rad": inf_vec.copy(),
                        "right_orientation_error_rad": inf_vec.copy(),
                        "solver_success": np.zeros(rep_indices.shape[0], dtype=bool),
                        "iteration_count": np.zeros(rep_indices.shape[0], dtype=np.int64),
                        "joint_limit_min_margin_rad": np.full(rep_indices.shape[0], -np.inf, dtype=np.float64),
                        "joint_delta_l2_rad": np.zeros(rep_indices.shape[0], dtype=np.float64),
                        "joint_delta_maxabs_rad": np.zeros(rep_indices.shape[0], dtype=np.float64),
                        "hands_within_1cm_count": np.zeros(rep_indices.shape[0], dtype=np.int64),
                        "both_hands_within_1cm": np.zeros(rep_indices.shape[0], dtype=bool),
                        "both_hands_within_1cm_count": 0,
                        "mean_position_error_m": np.inf,
                        "max_position_error_m": np.inf,
                        "p95_position_error_m": np.inf,
                        "mean_orientation_error_rad": np.inf,
                        "joint_limit_violation_count": 1,
                        "solver_failure_count": int(rep_indices.shape[0]),
                        "score": np.inf,
                        "min_joint_limit_margin_rad_overall": -np.inf,
                        "arm_q": np.zeros((rep_indices.shape[0], 14), dtype=np.float64),
                        "achieved_left_position_xyz_m": np.zeros((rep_indices.shape[0], 3), dtype=np.float64),
                        "achieved_right_position_xyz_m": np.zeros((rep_indices.shape[0], 3), dtype=np.float64),
                        "achieved_left_quaternion_wxyz": np.zeros((rep_indices.shape[0], 4), dtype=np.float64),
                        "achieved_right_quaternion_wxyz": np.zeros((rep_indices.shape[0], 4), dtype=np.float64),
                        "near_joint_limit_mask": np.zeros((rep_indices.shape[0], 14), dtype=bool),
                    }
                records.append(
                    {
                        "center_xyz": STAGE1_CENTER.copy(),
                        "hand_separation": float(hand_separation),
                        "center_scale": float(STAGE1_CENTER_SCALE),
                        "relative_scale": float(relative_scale),
                        "align_rpy_deg": align_rpy_deg.copy(),
                        "valid_anchor": bool(anchor_state["valid"]),
                        "metrics": metrics,
                    }
                )
    return finalize_records(records), anchor_cache


def stage2_search(
    aloha: dict[str, np.ndarray],
    model_info: dict[str, object],
    rep_indices: np.ndarray,
    stage1_best: dict[str, object],
) -> tuple[list[dict[str, object]], dict[tuple[float, float], dict[str, object]]]:
    anchor_cache: dict[tuple[float, float], dict[str, object]] = {}
    records: list[dict[str, object]] = []
    hand_separation = float(stage1_best["hand_separation"])
    relative_scale = float(stage1_best["relative_scale"])
    align_rpy_deg = np.asarray(stage1_best["align_rpy_deg"], dtype=np.float64)
    for center_x in CENTER_X_VALUES.tolist():
        for center_z in CENTER_Z_VALUES.tolist():
            center_xyz = np.array([center_x, 0.0, center_z], dtype=np.float64)
            anchor_state = compute_task_ready_start(model_info, center_xyz, hand_separation)
            anchor_cache[(float(center_x), float(center_z))] = anchor_state
            for center_scale in CENTER_SCALE_VALUES.tolist():
                if anchor_state["valid"]:
                    targets = build_candidate_targets(
                        aloha,
                        center_xyz,
                        hand_separation,
                        float(center_scale),
                        relative_scale,
                        align_rpy_deg,
                        np.asarray(anchor_state["left_start_quaternion_wxyz"], dtype=np.float64),
                        np.asarray(anchor_state["right_start_quaternion_wxyz"], dtype=np.float64),
                    )
                    finite = np.isfinite(targets["g1_left_target_position_xyz_m"]).all() and np.isfinite(targets["g1_right_target_position_xyz_m"]).all()
                    hand_ok = bool(np.all(hand_distance_ok(targets["g1_left_target_position_xyz_m"], targets["g1_right_target_position_xyz_m"])))
                    if finite and hand_ok:
                        metrics = evaluate_representative_candidate(
                            model_info,
                            targets,
                            rep_indices,
                            np.asarray(anchor_state["arm_q"], dtype=np.float64),
                        )
                    else:
                        metrics = evaluate_representative_candidate(
                            model_info,
                            targets,
                            rep_indices,
                            np.asarray(anchor_state["arm_q"], dtype=np.float64),
                        )
                        metrics["score"] = float(metrics["score"] + 100.0)
                else:
                    inf_vec = np.full(rep_indices.shape[0], np.inf, dtype=np.float64)
                    metrics = {
                        "left_position_error_m": inf_vec.copy(),
                        "right_position_error_m": inf_vec.copy(),
                        "left_orientation_error_rad": inf_vec.copy(),
                        "right_orientation_error_rad": inf_vec.copy(),
                        "solver_success": np.zeros(rep_indices.shape[0], dtype=bool),
                        "iteration_count": np.zeros(rep_indices.shape[0], dtype=np.int64),
                        "joint_limit_min_margin_rad": np.full(rep_indices.shape[0], -np.inf, dtype=np.float64),
                        "joint_delta_l2_rad": np.zeros(rep_indices.shape[0], dtype=np.float64),
                        "joint_delta_maxabs_rad": np.zeros(rep_indices.shape[0], dtype=np.float64),
                        "hands_within_1cm_count": np.zeros(rep_indices.shape[0], dtype=np.int64),
                        "both_hands_within_1cm": np.zeros(rep_indices.shape[0], dtype=bool),
                        "both_hands_within_1cm_count": 0,
                        "mean_position_error_m": np.inf,
                        "max_position_error_m": np.inf,
                        "p95_position_error_m": np.inf,
                        "mean_orientation_error_rad": np.inf,
                        "joint_limit_violation_count": 1,
                        "solver_failure_count": int(rep_indices.shape[0]),
                        "score": np.inf,
                        "min_joint_limit_margin_rad_overall": -np.inf,
                        "arm_q": np.zeros((rep_indices.shape[0], 14), dtype=np.float64),
                        "achieved_left_position_xyz_m": np.zeros((rep_indices.shape[0], 3), dtype=np.float64),
                        "achieved_right_position_xyz_m": np.zeros((rep_indices.shape[0], 3), dtype=np.float64),
                        "achieved_left_quaternion_wxyz": np.zeros((rep_indices.shape[0], 4), dtype=np.float64),
                        "achieved_right_quaternion_wxyz": np.zeros((rep_indices.shape[0], 4), dtype=np.float64),
                        "near_joint_limit_mask": np.zeros((rep_indices.shape[0], 14), dtype=bool),
                    }
                records.append(
                    {
                        "center_xyz": center_xyz.copy(),
                        "hand_separation": hand_separation,
                        "center_scale": float(center_scale),
                        "relative_scale": relative_scale,
                        "align_rpy_deg": align_rpy_deg.copy(),
                        "valid_anchor": bool(anchor_state["valid"]),
                        "metrics": metrics,
                    }
                )
    return finalize_records(records), anchor_cache


def make_sparse_args(orientation_weight: float) -> argparse.Namespace:
    return argparse.Namespace(
        position_weight=1.0,
        orientation_weight=float(orientation_weight),
        damping=0.01,
        max_iterations=FULL_TRAJECTORY_MAX_ITERATIONS,
        max_joint_step_rad=0.04,
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.25,
    )


def build_best_candidate_targets(
    aloha: dict[str, np.ndarray],
    candidate: dict[str, object],
    model_info: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    anchor_state = compute_task_ready_start(model_info, np.asarray(candidate["center_xyz"], dtype=np.float64), float(candidate["hand_separation"]))
    targets = build_candidate_targets(
        aloha,
        np.asarray(candidate["center_xyz"], dtype=np.float64),
        float(candidate["hand_separation"]),
        float(candidate["center_scale"]),
        float(candidate["relative_scale"]),
        np.asarray(candidate["align_rpy_deg"], dtype=np.float64),
        np.asarray(anchor_state["left_start_quaternion_wxyz"], dtype=np.float64),
        np.asarray(anchor_state["right_start_quaternion_wxyz"], dtype=np.float64),
    )
    targets["g1_task_ready_arm_q"] = np.asarray(anchor_state["arm_q"], dtype=np.float64)
    return targets, np.asarray(anchor_state["arm_q"], dtype=np.float64)


def run_frame300_diagnostics(
    aloha: dict[str, np.ndarray],
    model_info: dict[str, object],
    rep_indices: np.ndarray,
    best_candidate: dict[str, object],
    best_metrics: dict[str, object],
) -> dict[str, np.ndarray]:
    targets, start_arm_q = build_best_candidate_targets(aloha, best_candidate, model_info)
    frame300_idx = int(np.where(rep_indices == 300)[0][0])
    prev_q = best_metrics["arm_q"][frame300_idx - 1]
    frame_idx = 300

    base_result = anchor_mod.solve_position_only_frame(
        model_info,
        targets["g1_left_target_position_xyz_m"][frame_idx],
        targets["g1_right_target_position_xyz_m"][frame_idx],
        np.asarray(prev_q, dtype=np.float64),
        np.asarray(prev_q, dtype=np.float64),
    )
    error_vec = targets["g1_right_target_position_xyz_m"][frame_idx] - np.asarray(base_result["right_pos"], dtype=np.float64)
    base_ori_error = orientation_error_rad(base_result["right_quat"], targets["g1_right_target_quaternion_wxyz"][frame_idx])
    hand_distance = float(np.linalg.norm(np.asarray(base_result["right_pos"]) - np.asarray(base_result["left_pos"])))

    scale_errors = np.zeros(ORIENTATION_WEIGHT_SCALES.shape[0], dtype=np.float64)
    scale_ori_errors = np.zeros(ORIENTATION_WEIGHT_SCALES.shape[0], dtype=np.float64)
    scale_success = np.zeros(ORIENTATION_WEIGHT_SCALES.shape[0], dtype=bool)
    scale_iterations = np.zeros(ORIENTATION_WEIGHT_SCALES.shape[0], dtype=np.int64)
    scale_mode = np.empty(ORIENTATION_WEIGHT_SCALES.shape[0], dtype="<U24")

    for idx, scale in enumerate(ORIENTATION_WEIGHT_SCALES.tolist()):
        args = make_sparse_args(ORIENTATION_WEIGHT_BASE * float(scale))
        result = ik_mod.solve_sparse_frame(
            model_info,
            targets["g1_left_target_position_xyz_m"][frame_idx],
            targets["g1_left_target_quaternion_wxyz"][frame_idx],
            targets["g1_right_target_position_xyz_m"][frame_idx],
            targets["g1_right_target_quaternion_wxyz"][frame_idx],
            np.asarray(prev_q, dtype=np.float64),
            np.asarray(prev_q, dtype=np.float64),
            args,
        )
        scale_errors[idx] = float(result["right_pos_err"])
        scale_ori_errors[idx] = float(result["right_ori_err"])
        scale_success[idx] = bool(result["converged"])
        scale_iterations[idx] = int(result["iteration_count"])
        scale_mode[idx] = ik_mod.MODE_NAMES[int(result["mode_index"])]

    return {
        "frame_index": np.asarray(frame_idx, dtype=np.int64),
        "candidate_center_xyz_m": np.asarray(best_candidate["center_xyz"], dtype=np.float64),
        "candidate_hand_separation_m": np.asarray(best_candidate["hand_separation"], dtype=np.float64),
        "candidate_center_scale": np.asarray(best_candidate["center_scale"], dtype=np.float64),
        "candidate_relative_scale": np.asarray(best_candidate["relative_scale"], dtype=np.float64),
        "candidate_align_rotation_rpy_deg": np.asarray(best_candidate["align_rpy_deg"], dtype=np.float64),
        "target_right_position_xyz_m": targets["g1_right_target_position_xyz_m"][frame_idx].copy(),
        "solved_right_position_xyz_m": np.asarray(base_result["right_pos"], dtype=np.float64),
        "right_xyz_error_vector_m": error_vec,
        "right_position_error_norm_m": np.asarray(np.linalg.norm(error_vec), dtype=np.float64),
        "target_right_quaternion_wxyz": targets["g1_right_target_quaternion_wxyz"][frame_idx].copy(),
        "solved_right_quaternion_wxyz": np.asarray(base_result["right_quat"], dtype=np.float64),
        "right_orientation_error_rad": np.asarray(base_ori_error, dtype=np.float64),
        "right_arm_q": np.asarray(base_result["arm_q"], dtype=np.float64)[7:].copy(),
        "joint_limit_min_margin_rad": np.asarray(base_result["limit_margin"], dtype=np.float64),
        "joint_delta_vs_prev_l2_rad": np.asarray(np.linalg.norm(np.asarray(base_result["arm_q"], dtype=np.float64) - np.asarray(prev_q, dtype=np.float64)), dtype=np.float64),
        "hands_distance_m": np.asarray(hand_distance, dtype=np.float64),
        "orientation_weight_scales": ORIENTATION_WEIGHT_SCALES.copy(),
        "orientation_scaled_right_position_error_m": scale_errors,
        "orientation_scaled_right_orientation_error_rad": scale_ori_errors,
        "orientation_scaled_success": scale_success,
        "orientation_scaled_iteration_count": scale_iterations,
        "orientation_scaled_mode": scale_mode,
    }


def run_full_trajectory_top3(
    aloha: dict[str, np.ndarray],
    model_info: dict[str, object],
    top3: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    frame_count = int(aloha["timestamp_s"].shape[0])
    candidate_count = len(top3)
    joint_trajectory = np.zeros((candidate_count, frame_count, 14), dtype=np.float64)
    solved_left_pos = np.zeros((candidate_count, frame_count, 3), dtype=np.float64)
    solved_right_pos = np.zeros((candidate_count, frame_count, 3), dtype=np.float64)
    solved_left_quat = np.zeros((candidate_count, frame_count, 4), dtype=np.float64)
    solved_right_quat = np.zeros((candidate_count, frame_count, 4), dtype=np.float64)
    target_left_pos = np.zeros((candidate_count, frame_count, 3), dtype=np.float64)
    target_right_pos = np.zeros((candidate_count, frame_count, 3), dtype=np.float64)
    target_left_quat = np.zeros((candidate_count, frame_count, 4), dtype=np.float64)
    target_right_quat = np.zeros((candidate_count, frame_count, 4), dtype=np.float64)
    left_pos_err = np.zeros((candidate_count, frame_count), dtype=np.float64)
    right_pos_err = np.zeros((candidate_count, frame_count), dtype=np.float64)
    left_ori_err = np.zeros((candidate_count, frame_count), dtype=np.float64)
    right_ori_err = np.zeros((candidate_count, frame_count), dtype=np.float64)
    success_flag = np.zeros((candidate_count, frame_count), dtype=bool)
    retry_mode = np.empty((candidate_count, frame_count), dtype="<U16")
    min_margin = np.zeros((candidate_count, frame_count), dtype=np.float64)
    frame_step_l2 = np.zeros((candidate_count, frame_count), dtype=np.float64)
    frame_step_maxabs = np.zeros((candidate_count, frame_count), dtype=np.float64)
    candidate_center_xyz = np.zeros((candidate_count, 3), dtype=np.float64)
    candidate_hand_sep = np.zeros(candidate_count, dtype=np.float64)
    candidate_center_scale = np.zeros(candidate_count, dtype=np.float64)
    candidate_relative_scale = np.zeros(candidate_count, dtype=np.float64)
    candidate_align = np.zeros((candidate_count, 3), dtype=np.float64)
    task_ready_arm_q = np.zeros((candidate_count, 14), dtype=np.float64)

    summary_success_count = np.zeros(candidate_count, dtype=np.int64)
    summary_success_rate = np.zeros(candidate_count, dtype=np.float64)
    summary_left_mean = np.zeros(candidate_count, dtype=np.float64)
    summary_left_max = np.zeros(candidate_count, dtype=np.float64)
    summary_left_p95 = np.zeros(candidate_count, dtype=np.float64)
    summary_right_mean = np.zeros(candidate_count, dtype=np.float64)
    summary_right_max = np.zeros(candidate_count, dtype=np.float64)
    summary_right_p95 = np.zeros(candidate_count, dtype=np.float64)
    summary_both_mean = np.zeros(candidate_count, dtype=np.float64)
    summary_both_max = np.zeros(candidate_count, dtype=np.float64)
    summary_both_p95 = np.zeros(candidate_count, dtype=np.float64)
    summary_ori_mean = np.zeros(candidate_count, dtype=np.float64)
    summary_ori_max = np.zeros(candidate_count, dtype=np.float64)
    summary_within_1cm = np.zeros(candidate_count, dtype=np.float64)
    summary_within_2cm = np.zeros(candidate_count, dtype=np.float64)
    summary_min_margin = np.zeros(candidate_count, dtype=np.float64)
    summary_max_step = np.zeros(candidate_count, dtype=np.float64)
    summary_mean_step = np.zeros(candidate_count, dtype=np.float64)

    failure_mask = np.zeros((candidate_count, frame_count), dtype=bool)
    gt_1cm_mask = np.zeros((candidate_count, frame_count), dtype=bool)
    gt_2cm_mask = np.zeros((candidate_count, frame_count), dtype=bool)

    for cand_idx, candidate in enumerate(top3):
        targets, init_task_ready_q = build_best_candidate_targets(aloha, candidate, model_info)
        candidate_center_xyz[cand_idx] = np.asarray(candidate["center_xyz"], dtype=np.float64)
        candidate_hand_sep[cand_idx] = float(candidate["hand_separation"])
        candidate_center_scale[cand_idx] = float(candidate["center_scale"])
        candidate_relative_scale[cand_idx] = float(candidate["relative_scale"])
        candidate_align[cand_idx] = np.asarray(candidate["align_rpy_deg"], dtype=np.float64)
        task_ready_arm_q[cand_idx] = init_task_ready_q

        target_left_pos[cand_idx] = targets["g1_left_target_position_xyz_m"]
        target_right_pos[cand_idx] = targets["g1_right_target_position_xyz_m"]
        target_left_quat[cand_idx] = targets["g1_left_target_quaternion_wxyz"]
        target_right_quat[cand_idx] = targets["g1_right_target_quaternion_wxyz"]

        prev_q = init_task_ready_q.copy()
        prev_success_q = init_task_ready_q.copy()
        for frame_idx in range(frame_count):
            attempts = [
                ("prev_full", prev_q.copy(), ORIENTATION_WEIGHT_BASE),
                ("task_ready_full", init_task_ready_q.copy(), ORIENTATION_WEIGHT_BASE),
                ("prev_half_ori", prev_q.copy(), 0.5 * ORIENTATION_WEIGHT_BASE),
            ]
            chosen: dict[str, object] | None = None
            chosen_label = "fail_hold"
            best_pos_sum = math.inf
            for label, init_q, ori_weight in attempts:
                result = ik_mod.solve_sparse_frame(
                    model_info,
                    targets["g1_left_target_position_xyz_m"][frame_idx],
                    targets["g1_left_target_quaternion_wxyz"][frame_idx],
                    targets["g1_right_target_position_xyz_m"][frame_idx],
                    targets["g1_right_target_quaternion_wxyz"][frame_idx],
                    init_q,
                    prev_q,
                    make_sparse_args(ori_weight),
                )
                pos_sum = float(result["left_pos_err"] + result["right_pos_err"])
                if pos_sum < best_pos_sum:
                    chosen = result
                    chosen_label = label
                    best_pos_sum = pos_sum
                if frame_success(float(result["left_pos_err"]), float(result["right_pos_err"])) and not bool(result["violation"]):
                    chosen = result
                    chosen_label = label
                    break
            if chosen is None:
                raise fail("Full trajectory IK had no result")

            if not frame_success(float(chosen["left_pos_err"]), float(chosen["right_pos_err"])) or bool(chosen["violation"]):
                failure_mask[cand_idx, frame_idx] = True
                chosen_label = "fail_hold"
                hold_q = prev_success_q.copy()
                data = ik_mod.mujoco.MjData(model_info["model"])
                ik_mod.assign_arm_qpos(data, model_info["stand_qpos"], model_info["arm_qpos_ids"], hold_q)
                ik_mod.mujoco.mj_forward(model_info["model"], data)
                current = ik_mod.current_bimanual_state(model_info, data)
                chosen = {
                    "arm_q": hold_q,
                    "left_pos": current["left_pos"],
                    "right_pos": current["right_pos"],
                    "left_quat": current["left_quat"],
                    "right_quat": current["right_quat"],
                    "left_pos_err": float(np.linalg.norm(targets["g1_left_target_position_xyz_m"][frame_idx] - current["left_pos"])),
                    "right_pos_err": float(np.linalg.norm(targets["g1_right_target_position_xyz_m"][frame_idx] - current["right_pos"])),
                    "left_ori_err": orientation_error_rad(current["left_quat"], targets["g1_left_target_quaternion_wxyz"][frame_idx]),
                    "right_ori_err": orientation_error_rad(current["right_quat"], targets["g1_right_target_quaternion_wxyz"][frame_idx]),
                    "limit_margin": ik_mod.arm_limit_metrics(hold_q, model_info["joint_limits"])[0],
                    "converged": False,
                }
            else:
                prev_success_q = np.asarray(chosen["arm_q"], dtype=np.float64).copy()
                success_flag[cand_idx, frame_idx] = True

            q_now = np.asarray(chosen["arm_q"], dtype=np.float64)
            joint_trajectory[cand_idx, frame_idx] = q_now
            solved_left_pos[cand_idx, frame_idx] = np.asarray(chosen["left_pos"], dtype=np.float64)
            solved_right_pos[cand_idx, frame_idx] = np.asarray(chosen["right_pos"], dtype=np.float64)
            solved_left_quat[cand_idx, frame_idx] = np.asarray(chosen["left_quat"], dtype=np.float64)
            solved_right_quat[cand_idx, frame_idx] = np.asarray(chosen["right_quat"], dtype=np.float64)
            left_pos_err[cand_idx, frame_idx] = float(chosen["left_pos_err"])
            right_pos_err[cand_idx, frame_idx] = float(chosen["right_pos_err"])
            left_ori_err[cand_idx, frame_idx] = float(chosen["left_ori_err"])
            right_ori_err[cand_idx, frame_idx] = float(chosen["right_ori_err"])
            retry_mode[cand_idx, frame_idx] = chosen_label
            min_margin[cand_idx, frame_idx] = float(chosen["limit_margin"])
            if frame_idx == 0:
                delta = q_now - init_task_ready_q
            else:
                delta = q_now - joint_trajectory[cand_idx, frame_idx - 1]
            frame_step_l2[cand_idx, frame_idx] = float(np.linalg.norm(delta))
            frame_step_maxabs[cand_idx, frame_idx] = float(np.max(np.abs(delta)))
            prev_q = q_now.copy()

        hand_max_err = np.maximum(left_pos_err[cand_idx], right_pos_err[cand_idx])
        ori_concat = np.concatenate([left_ori_err[cand_idx], right_ori_err[cand_idx]])
        summary_success_count[cand_idx] = int(np.sum(success_flag[cand_idx]))
        summary_success_rate[cand_idx] = float(np.mean(success_flag[cand_idx]))
        summary_left_mean[cand_idx] = float(np.mean(left_pos_err[cand_idx]))
        summary_left_max[cand_idx] = float(np.max(left_pos_err[cand_idx]))
        summary_left_p95[cand_idx] = float(np.percentile(left_pos_err[cand_idx], 95.0))
        summary_right_mean[cand_idx] = float(np.mean(right_pos_err[cand_idx]))
        summary_right_max[cand_idx] = float(np.max(right_pos_err[cand_idx]))
        summary_right_p95[cand_idx] = float(np.percentile(right_pos_err[cand_idx], 95.0))
        summary_both_mean[cand_idx] = float(np.mean(hand_max_err))
        summary_both_max[cand_idx] = float(np.max(hand_max_err))
        summary_both_p95[cand_idx] = float(np.percentile(hand_max_err, 95.0))
        summary_ori_mean[cand_idx] = float(np.mean(ori_concat))
        summary_ori_max[cand_idx] = float(np.max(ori_concat))
        within_1cm = (left_pos_err[cand_idx] <= 0.01) & (right_pos_err[cand_idx] <= 0.01)
        within_2cm = (left_pos_err[cand_idx] <= 0.02) & (right_pos_err[cand_idx] <= 0.02)
        summary_within_1cm[cand_idx] = float(np.mean(within_1cm))
        summary_within_2cm[cand_idx] = float(np.mean(within_2cm))
        summary_min_margin[cand_idx] = float(np.min(min_margin[cand_idx]))
        summary_max_step[cand_idx] = float(np.max(frame_step_maxabs[cand_idx]))
        summary_mean_step[cand_idx] = float(np.mean(frame_step_l2[cand_idx]))
        gt_1cm_mask[cand_idx] = ~within_1cm
        gt_2cm_mask[cand_idx] = ~within_2cm

    return {
        "frame_count": np.asarray(frame_count, dtype=np.int64),
        "timestamp_s": np.asarray(aloha["timestamp_s"], dtype=np.float64),
        "candidate_center_xyz_m": candidate_center_xyz,
        "candidate_hand_separation_m": candidate_hand_sep,
        "candidate_center_scale": candidate_center_scale,
        "candidate_relative_scale": candidate_relative_scale,
        "candidate_align_rotation_rpy_deg": candidate_align,
        "candidate_task_ready_arm_q": task_ready_arm_q,
        "joint_trajectory_q": joint_trajectory,
        "solved_left_position_xyz_m": solved_left_pos,
        "solved_right_position_xyz_m": solved_right_pos,
        "solved_left_quaternion_wxyz": solved_left_quat,
        "solved_right_quaternion_wxyz": solved_right_quat,
        "target_left_position_xyz_m": target_left_pos,
        "target_right_position_xyz_m": target_right_pos,
        "target_left_quaternion_wxyz": target_left_quat,
        "target_right_quaternion_wxyz": target_right_quat,
        "left_position_error_m": left_pos_err,
        "right_position_error_m": right_pos_err,
        "left_orientation_error_rad": left_ori_err,
        "right_orientation_error_rad": right_ori_err,
        "success_flag": success_flag,
        "retry_mode": retry_mode,
        "joint_limit_min_margin_rad": min_margin,
        "frame_joint_step_l2_rad": frame_step_l2,
        "frame_joint_step_maxabs_rad": frame_step_maxabs,
        "summary_success_count": summary_success_count,
        "summary_success_rate": summary_success_rate,
        "summary_left_position_error_mean_m": summary_left_mean,
        "summary_left_position_error_max_m": summary_left_max,
        "summary_left_position_error_p95_m": summary_left_p95,
        "summary_right_position_error_mean_m": summary_right_mean,
        "summary_right_position_error_max_m": summary_right_max,
        "summary_right_position_error_p95_m": summary_right_p95,
        "summary_both_position_error_mean_m": summary_both_mean,
        "summary_both_position_error_max_m": summary_both_max,
        "summary_both_position_error_p95_m": summary_both_p95,
        "summary_orientation_error_mean_rad": summary_ori_mean,
        "summary_orientation_error_max_rad": summary_ori_max,
        "summary_within_1cm_ratio": summary_within_1cm,
        "summary_within_2cm_ratio": summary_within_2cm,
        "summary_joint_limit_min_margin_rad": summary_min_margin,
        "summary_frame_joint_step_maxabs_rad": summary_max_step,
        "summary_frame_joint_step_mean_l2_rad": summary_mean_step,
        "failure_mask": failure_mask,
        "gt_1cm_mask": gt_1cm_mask,
        "gt_2cm_mask": gt_2cm_mask,
    }


def verify_npz(path: Path, required_keys: list[str]) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    payload = {key: data[key] for key in data.files}
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise fail(f"Missing keys in {path.name}: {missing}")
    for key, value in payload.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise fail(f"{path.name}:{key} contains NaN or inf")
    return payload


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    refined_output = args.refined_output.expanduser().resolve()
    full_ik_output = args.full_ik_output.expanduser().resolve()
    frame300_output = args.frame300_output.expanduser().resolve()

    require_output(refined_output, args.overwrite)
    require_output(full_ik_output, args.overwrite)
    require_output(frame300_output, args.overwrite)

    aloha = load_aloha(input_path)
    model_info = ik_mod.validate_model(model_path)
    rep_indices = REPRESENTATIVE_FRAMES.copy()

    stage1_records, _stage1_anchor_cache = stage1_search(aloha, model_info, rep_indices)
    stage1_best = stage1_records[0]
    stage2_records, _stage2_anchor_cache = stage2_search(aloha, model_info, rep_indices, stage1_best)
    stage2_best = stage2_records[0]

    combined_records = finalize_records(stage1_records + stage2_records)
    top3 = combined_records[:3]

    refined_payload = {
        "representative_frames": rep_indices,
        "stage1_candidate_count": np.asarray(len(stage1_records), dtype=np.int64),
        "stage2_candidate_count": np.asarray(len(stage2_records), dtype=np.int64),
        "stage1_best_center_xyz_m": np.asarray(stage1_best["center_xyz"], dtype=np.float64),
        "stage1_best_hand_separation_m": np.asarray(stage1_best["hand_separation"], dtype=np.float64),
        "stage1_best_center_scale": np.asarray(stage1_best["center_scale"], dtype=np.float64),
        "stage1_best_relative_scale": np.asarray(stage1_best["relative_scale"], dtype=np.float64),
        "stage1_best_align_rotation_rpy_deg": np.asarray(stage1_best["align_rpy_deg"], dtype=np.float64),
        "stage2_best_center_xyz_m": np.asarray(stage2_best["center_xyz"], dtype=np.float64),
        "stage2_best_hand_separation_m": np.asarray(stage2_best["hand_separation"], dtype=np.float64),
        "stage2_best_center_scale": np.asarray(stage2_best["center_scale"], dtype=np.float64),
        "stage2_best_relative_scale": np.asarray(stage2_best["relative_scale"], dtype=np.float64),
        "stage2_best_align_rotation_rpy_deg": np.asarray(stage2_best["align_rpy_deg"], dtype=np.float64),
        "best_center_xyz_m": np.asarray(combined_records[0]["center_xyz"], dtype=np.float64),
        "best_hand_separation_m": np.asarray(combined_records[0]["hand_separation"], dtype=np.float64),
        "best_center_scale": np.asarray(combined_records[0]["center_scale"], dtype=np.float64),
        "best_relative_scale": np.asarray(combined_records[0]["relative_scale"], dtype=np.float64),
        "best_align_rotation_rpy_deg": np.asarray(combined_records[0]["align_rpy_deg"], dtype=np.float64),
    }
    stage1_arrays = build_stage_arrays(stage1_records, rep_indices.shape[0])
    stage2_arrays = build_stage_arrays(stage2_records, rep_indices.shape[0])
    combined_arrays = build_stage_arrays(combined_records, rep_indices.shape[0])
    for prefix, arrays in [("stage1_", stage1_arrays), ("stage2_", stage2_arrays), ("combined_", combined_arrays)]:
        for key, value in arrays.items():
            refined_payload[prefix + key] = value
    np.savez_compressed(refined_output, **refined_payload)

    frame300_payload = run_frame300_diagnostics(aloha, model_info, rep_indices, stage2_best, stage2_best["metrics"])
    np.savez_compressed(frame300_output, **frame300_payload)

    full_payload = run_full_trajectory_top3(aloha, model_info, top3)
    np.savez_compressed(full_ik_output, **full_payload)

    refined_check = verify_npz(
        refined_output,
        [
            "representative_frames",
            "stage1_center_xyz_m",
            "stage2_center_xyz_m",
            "combined_center_xyz_m",
            "best_center_xyz_m",
        ],
    )
    frame300_check = verify_npz(
        frame300_output,
        [
            "frame_index",
            "target_right_position_xyz_m",
            "solved_right_position_xyz_m",
            "orientation_weight_scales",
        ],
    )
    full_check = verify_npz(
        full_ik_output,
        [
            "frame_count",
            "joint_trajectory_q",
            "success_flag",
            "summary_success_rate",
            "candidate_center_xyz_m",
        ],
    )
    if int(full_check["frame_count"]) != 713:
        raise fail(f"Expected 713 frames in full IK output, found {int(full_check['frame_count'])}")
    if full_check["joint_trajectory_q"].shape[1:] != (713, 14):
        raise fail(f"Unexpected joint trajectory shape: {full_check['joint_trajectory_q'].shape}")

    print(f"Stage 1 evaluation count: {len(stage1_records)}")
    print(
        "Stage 1 best:",
        f"center={stage1_best['center_xyz'].tolist()}",
        f"sep={stage1_best['hand_separation']:.3f}",
        f"center_scale={stage1_best['center_scale']:.3f}",
        f"relative_scale={stage1_best['relative_scale']:.3f}",
        f"align={stage1_best['align_rpy_deg'].tolist()}",
        f"both<=1cm={stage1_best['metrics']['both_hands_within_1cm_count']}/5",
        f"max_err={stage1_best['metrics']['max_position_error_m']:.6f}",
    )
    print(f"Stage 2 evaluation count: {len(stage2_records)}")
    print(
        "Stage 2 best:",
        f"center={stage2_best['center_xyz'].tolist()}",
        f"sep={stage2_best['hand_separation']:.3f}",
        f"center_scale={stage2_best['center_scale']:.3f}",
        f"relative_scale={stage2_best['relative_scale']:.3f}",
        f"align={stage2_best['align_rpy_deg'].tolist()}",
        f"both<=1cm={stage2_best['metrics']['both_hands_within_1cm_count']}/5",
        f"max_err={stage2_best['metrics']['max_position_error_m']:.6f}",
    )
    full_success_exists = any(int(item["metrics"]["both_hands_within_1cm_count"]) == rep_indices.shape[0] for item in combined_records)
    print(f"All 5 representative frames within 1 cm exists: {full_success_exists}")
    print("Top 10 refined candidates:")
    for idx, item in enumerate(combined_records[:10], start=1):
        metrics = item["metrics"]
        print(
            f"  {idx}. center={item['center_xyz'].tolist()} sep={item['hand_separation']:.3f} "
            f"center_scale={item['center_scale']:.3f} relative_scale={item['relative_scale']:.3f} "
            f"align={item['align_rpy_deg'].tolist()} both<=1cm={metrics['both_hands_within_1cm_count']}/5 "
            f"max_err={metrics['max_position_error_m']:.6f} p95={metrics['p95_position_error_m']:.6f} "
            f"score={metrics['score']:.6f} min_margin={metrics['min_joint_limit_margin_rad_overall']:.6f}"
        )
    print("Frame 300 orientation-weight diagnostics:")
    for scale, pos_err, ori_err, success, mode in zip(
        frame300_check["orientation_weight_scales"],
        frame300_check["orientation_scaled_right_position_error_m"],
        frame300_check["orientation_scaled_right_orientation_error_rad"],
        frame300_check["orientation_scaled_success"],
        frame300_check["orientation_scaled_mode"],
    ):
        print(
            f"  scale={float(scale):.2f} right_pos_err={float(pos_err):.6f} "
            f"right_ori_err={float(ori_err):.6f} success={bool(success)} mode={str(mode)}"
        )
    print("Top-3 full trajectory IK:")
    for idx in range(full_check["candidate_center_xyz_m"].shape[0]):
        print(
            f"  candidate {idx + 1}: center={full_check['candidate_center_xyz_m'][idx].tolist()} "
            f"sep={float(full_check['candidate_hand_separation_m'][idx]):.3f} "
            f"center_scale={float(full_check['candidate_center_scale'][idx]):.3f} "
            f"relative_scale={float(full_check['candidate_relative_scale'][idx]):.3f} "
            f"align={full_check['candidate_align_rotation_rpy_deg'][idx].tolist()} "
            f"success_rate={float(full_check['summary_success_rate'][idx]):.4f} "
            f"max_err={float(full_check['summary_both_position_error_max_m'][idx]):.6f} "
            f"p95_err={float(full_check['summary_both_position_error_p95_m'][idx]):.6f} "
            f"min_margin={float(full_check['summary_joint_limit_min_margin_rad'][idx]):.6f}"
        )
    print("Generated files:")
    for path in [refined_output, full_ik_output, frame300_output]:
        print(f"  {path}")
    print("NPZ keys and shapes:")
    for name, payload in [("refined", refined_check), ("frame300", frame300_check), ("full_ik", full_check)]:
        print(f"  [{name}]")
        for key in sorted(payload.keys()):
            print(f"    {key}: {payload[key].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
