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
from types import SimpleNamespace

import transform_aloha_tcp_to_g1_targets as transform_mod
import validate_g1_targets_and_sparse_ik as ik_mod


DEFAULT_INPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/aloha_tcp_trajectory.npz")
DEFAULT_MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEFAULT_SEARCH_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/g1_target_transform_search.npz")
DEFAULT_BEST_OUTPUT = Path("/home/jbnu/aloha_g1_dataset/GoPark/derived/best_g1_cartesian_targets.npz")
SCALE_VALUES = np.array([0.60, 0.70, 0.75, 0.80, 0.90, 1.00], dtype=np.float64)
ORIENTATION_WEIGHT_VALUES = np.array([0.00, 0.03, 0.05, 0.10, 0.15], dtype=np.float64)
ALIGN_RPY_VALUES = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
        [0.0, 5.0, 0.0],
        [0.0, -5.0, 0.0],
        [5.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
MIN_REASONABLE_HAND_DISTANCE_M = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search G1 target transform parameters using sparse IK.")
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


def make_transform_args(input_path: Path, model_path: Path, scale: float, orientation_weight: float, align_rpy_deg: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        input=input_path,
        output=Path("unused.npz"),
        g1_model=model_path,
        scale=float(scale),
        orientation_weight=float(orientation_weight),
        align_rotation_rpy_deg=tuple(float(x) for x in align_rpy_deg),
        left_tool_rpy_deg=(0.0, 0.0, 0.0),
        right_tool_rpy_deg=(0.0, 0.0, 0.0),
        overwrite=True,
    )


def make_ik_args() -> SimpleNamespace:
    return SimpleNamespace(
        position_weight=1.0,
        orientation_weight=0.15,
        damping=0.01,
        max_iterations=200,
        max_joint_step_rad=0.04,
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.25,
    )


def run_sparse_ik_quiet(model_info: dict[str, object], payload: dict[str, np.ndarray], sample_indices: np.ndarray) -> dict[str, np.ndarray]:
    sample_count = int(sample_indices.shape[0])
    both_q = np.zeros((sample_count, 14), dtype=np.float64)
    left_q = np.zeros((sample_count, 7), dtype=np.float64)
    right_q = np.zeros((sample_count, 7), dtype=np.float64)
    converged = np.zeros(sample_count, dtype=bool)
    ik_mode = np.empty(sample_count, dtype="<U20")
    iteration_count = np.zeros(sample_count, dtype=np.int64)
    left_pos_err = np.zeros(sample_count, dtype=np.float64)
    right_pos_err = np.zeros(sample_count, dtype=np.float64)
    left_ori_err = np.zeros(sample_count, dtype=np.float64)
    right_ori_err = np.zeros(sample_count, dtype=np.float64)
    min_margin = np.zeros(sample_count, dtype=np.float64)
    near_mask = np.zeros((sample_count, 14), dtype=bool)
    achieved_left_pos = np.zeros((sample_count, 3), dtype=np.float64)
    achieved_right_pos = np.zeros((sample_count, 3), dtype=np.float64)
    achieved_left_quat = np.zeros((sample_count, 4), dtype=np.float64)
    achieved_right_quat = np.zeros((sample_count, 4), dtype=np.float64)

    previous_solution = np.asarray(model_info["stand_arm_q"], dtype=np.float64)
    ik_args = make_ik_args()
    violation_detected = False
    for out_idx, frame_idx in enumerate(sample_indices):
        result = ik_mod.solve_sparse_frame(
            model_info,
            payload["g1_left_target_position_xyz_m"][frame_idx],
            payload["g1_left_target_quaternion_wxyz"][frame_idx],
            payload["g1_right_target_position_xyz_m"][frame_idx],
            payload["g1_right_target_quaternion_wxyz"][frame_idx],
            previous_solution if out_idx > 0 else np.asarray(model_info["stand_arm_q"], dtype=np.float64),
            previous_solution,
            ik_args,
        )
        both_q[out_idx] = result["arm_q"]
        left_q[out_idx] = result["arm_q"][:7]
        right_q[out_idx] = result["arm_q"][7:]
        converged[out_idx] = bool(result["converged"])
        ik_mode[out_idx] = ik_mod.MODE_NAMES[int(result["mode_index"])]
        iteration_count[out_idx] = int(result["iteration_count"])
        left_pos_err[out_idx] = float(result["left_pos_err"])
        right_pos_err[out_idx] = float(result["right_pos_err"])
        left_ori_err[out_idx] = float(result["left_ori_err"])
        right_ori_err[out_idx] = float(result["right_ori_err"])
        min_margin[out_idx] = float(result["limit_margin"])
        near_mask[out_idx] = np.asarray(result["near_mask"], dtype=bool)
        achieved_left_pos[out_idx] = result["left_pos"]
        achieved_right_pos[out_idx] = result["right_pos"]
        achieved_left_quat[out_idx] = result["left_quat"]
        achieved_right_quat[out_idx] = result["right_quat"]
        previous_solution = np.asarray(result["arm_q"], dtype=np.float64)
        violation_detected = violation_detected or bool(result["violation"])

    return {
        "g1_left_arm_q": left_q,
        "g1_right_arm_q": right_q,
        "g1_bimanual_arm_q": both_q,
        "converged": converged,
        "ik_mode": ik_mode,
        "iteration_count": iteration_count,
        "left_position_error_m": left_pos_err,
        "right_position_error_m": right_pos_err,
        "left_orientation_error_rad": left_ori_err,
        "right_orientation_error_rad": right_ori_err,
        "joint_limit_min_margin_rad": min_margin,
        "joint_limit_near_mask": near_mask,
        "achieved_left_position_xyz_m": achieved_left_pos,
        "achieved_right_position_xyz_m": achieved_right_pos,
        "achieved_left_quaternion_wxyz": achieved_left_quat,
        "achieved_right_quaternion_wxyz": achieved_right_quat,
        "joint_limit_violation": np.asarray(violation_detected),
    }


def compute_combo_metrics(payload: dict[str, np.ndarray], sparse: dict[str, np.ndarray], orientation_weight: float, sample_indices: np.ndarray) -> dict[str, object]:
    left_pos = sparse["left_position_error_m"]
    right_pos = sparse["right_position_error_m"]
    left_ori = sparse["left_orientation_error_rad"]
    right_ori = sparse["right_orientation_error_rad"]
    mean_position_error = float(np.mean(np.concatenate([left_pos, right_pos])))
    max_position_error = float(np.max(np.concatenate([left_pos, right_pos])))
    mean_orientation_error = float(np.mean(np.concatenate([left_ori, right_ori])))
    converged_count = int(np.sum(sparse["converged"]))
    fallback_count = int(np.sum(sparse["ik_mode"] != ik_mod.MODE_NAMES[0]))
    nonconverged_count = int(sample_indices.shape[0] - converged_count)
    min_margin = float(np.min(sparse["joint_limit_min_margin_rad"]))
    left_start = np.asarray(payload["g1_left_start_position_xyz_m"], dtype=np.float64)
    right_start = np.asarray(payload["g1_right_start_position_xyz_m"], dtype=np.float64)
    max_displacement = float(
        max(
            np.max(np.linalg.norm(payload["g1_left_target_position_xyz_m"] - left_start[None, :], axis=1)),
            np.max(np.linalg.norm(payload["g1_right_target_position_xyz_m"] - right_start[None, :], axis=1)),
        )
    )
    frame0_ok = bool(left_pos[0] <= 0.01 and right_pos[0] <= 0.01)
    min_hand_distance = float(np.min(payload["g1_hands_distance_m"]))
    valid = (
        frame0_ok
        and not bool(sparse["joint_limit_violation"])
        and np.isfinite(payload["g1_left_target_position_xyz_m"]).all()
        and np.isfinite(payload["g1_right_target_position_xyz_m"]).all()
        and min_hand_distance > MIN_REASONABLE_HAND_DISTANCE_M
    )
    score = 10.0 * mean_position_error + 5.0 * max_position_error + 0.2 * fallback_count + 1.0 * nonconverged_count
    if orientation_weight > 0.0:
        score += 0.5 * mean_orientation_error
    if not valid:
        score += 1000.0
    return {
        "mean_position_error": mean_position_error,
        "left_position_error_mean": float(np.mean(left_pos)),
        "right_position_error_mean": float(np.mean(right_pos)),
        "max_position_error": max_position_error,
        "mean_orientation_error": mean_orientation_error,
        "converged_count": converged_count,
        "fallback_count": fallback_count,
        "nonconverged_count": nonconverged_count,
        "minimum_joint_limit_margin_rad": min_margin,
        "maximum_target_displacement_m": max_displacement,
        "frame0_ok": frame0_ok,
        "min_hand_distance_m": min_hand_distance,
        "valid": valid,
        "score": score,
    }


def build_best_targets_payload(best_payload: dict[str, np.ndarray], best_metrics: dict[str, object], best_sparse: dict[str, np.ndarray], sample_indices: np.ndarray, best_scale: float, best_orientation_weight: float, best_align_rpy_deg: np.ndarray) -> dict[str, np.ndarray]:
    out = dict(best_payload)
    out["search_best_scale"] = np.asarray(best_scale, dtype=np.float64)
    out["search_best_orientation_weight"] = np.asarray(best_orientation_weight, dtype=np.float64)
    out["search_best_align_rotation_rpy_deg"] = np.asarray(best_align_rpy_deg, dtype=np.float64)
    out["search_best_score"] = np.asarray(best_metrics["score"], dtype=np.float64)
    out["search_sample_frame_indices"] = sample_indices
    out["search_sample_converged"] = sparse_bool = np.asarray(best_sparse["converged"], dtype=bool)
    out["search_sample_ik_mode"] = np.asarray(best_sparse["ik_mode"])
    out["search_sample_left_position_error_m"] = np.asarray(best_sparse["left_position_error_m"], dtype=np.float64)
    out["search_sample_right_position_error_m"] = np.asarray(best_sparse["right_position_error_m"], dtype=np.float64)
    out["search_sample_left_orientation_error_rad"] = np.asarray(best_sparse["left_orientation_error_rad"], dtype=np.float64)
    out["search_sample_right_orientation_error_rad"] = np.asarray(best_sparse["right_orientation_error_rad"], dtype=np.float64)
    out["search_sample_joint_limit_min_margin_rad"] = np.asarray(best_sparse["joint_limit_min_margin_rad"], dtype=np.float64)
    out["search_sample_converged_count"] = np.asarray(int(np.sum(sparse_bool)), dtype=np.int64)
    return out


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    search_output = args.search_output.expanduser().resolve()
    best_output = args.best_output.expanduser().resolve()
    require_writable_output(search_output, args.overwrite)
    require_writable_output(best_output, args.overwrite)

    model_info = ik_mod.validate_model(model_path)
    sample_indices = ik_mod.sample_frame_indices(transform_mod.load_npz(input_path)["timestamp_s"].shape[0])
    total_combos = int(len(SCALE_VALUES) * len(ORIENTATION_WEIGHT_VALUES) * len(ALIGN_RPY_VALUES))

    combo_scale = np.zeros(total_combos, dtype=np.float64)
    combo_orientation_weight = np.zeros(total_combos, dtype=np.float64)
    combo_align_rpy_deg = np.zeros((total_combos, 3), dtype=np.float64)
    score_arr = np.zeros(total_combos, dtype=np.float64)
    valid_arr = np.zeros(total_combos, dtype=bool)
    mean_pos_arr = np.zeros(total_combos, dtype=np.float64)
    left_mean_pos_arr = np.zeros(total_combos, dtype=np.float64)
    right_mean_pos_arr = np.zeros(total_combos, dtype=np.float64)
    max_pos_arr = np.zeros(total_combos, dtype=np.float64)
    mean_ori_arr = np.zeros(total_combos, dtype=np.float64)
    converged_count_arr = np.zeros(total_combos, dtype=np.int64)
    fallback_count_arr = np.zeros(total_combos, dtype=np.int64)
    nonconverged_count_arr = np.zeros(total_combos, dtype=np.int64)
    min_margin_arr = np.zeros(total_combos, dtype=np.float64)
    max_disp_arr = np.zeros(total_combos, dtype=np.float64)
    frame0_ok_arr = np.zeros(total_combos, dtype=bool)
    min_hand_distance_arr = np.zeros(total_combos, dtype=np.float64)
    frame_left_pos_err = np.zeros((total_combos, sample_indices.shape[0]), dtype=np.float64)
    frame_right_pos_err = np.zeros((total_combos, sample_indices.shape[0]), dtype=np.float64)
    frame_left_ori_err = np.zeros((total_combos, sample_indices.shape[0]), dtype=np.float64)
    frame_right_ori_err = np.zeros((total_combos, sample_indices.shape[0]), dtype=np.float64)
    frame_converged = np.zeros((total_combos, sample_indices.shape[0]), dtype=bool)
    frame_mode = np.empty((total_combos, sample_indices.shape[0]), dtype="<U20")
    frame_min_margin = np.zeros((total_combos, sample_indices.shape[0]), dtype=np.float64)

    best_index = -1
    best_payload: dict[str, np.ndarray] | None = None
    best_sparse: dict[str, np.ndarray] | None = None
    best_metrics: dict[str, object] | None = None

    combo_index = 0
    for scale in SCALE_VALUES:
        for orientation_weight in ORIENTATION_WEIGHT_VALUES:
            for align_rpy_deg in ALIGN_RPY_VALUES:
                transform_args = make_transform_args(input_path, model_path, float(scale), float(orientation_weight), align_rpy_deg)
                payload, _ = transform_mod.transform(transform_args)
                transform_mod.validate_payload(payload)
                sparse = run_sparse_ik_quiet(model_info, payload, sample_indices)
                metrics = compute_combo_metrics(payload, sparse, float(orientation_weight), sample_indices)

                combo_scale[combo_index] = scale
                combo_orientation_weight[combo_index] = orientation_weight
                combo_align_rpy_deg[combo_index] = align_rpy_deg
                score_arr[combo_index] = float(metrics["score"])
                valid_arr[combo_index] = bool(metrics["valid"])
                mean_pos_arr[combo_index] = float(metrics["mean_position_error"])
                left_mean_pos_arr[combo_index] = float(metrics["left_position_error_mean"])
                right_mean_pos_arr[combo_index] = float(metrics["right_position_error_mean"])
                max_pos_arr[combo_index] = float(metrics["max_position_error"])
                mean_ori_arr[combo_index] = float(metrics["mean_orientation_error"])
                converged_count_arr[combo_index] = int(metrics["converged_count"])
                fallback_count_arr[combo_index] = int(metrics["fallback_count"])
                nonconverged_count_arr[combo_index] = int(metrics["nonconverged_count"])
                min_margin_arr[combo_index] = float(metrics["minimum_joint_limit_margin_rad"])
                max_disp_arr[combo_index] = float(metrics["maximum_target_displacement_m"])
                frame0_ok_arr[combo_index] = bool(metrics["frame0_ok"])
                min_hand_distance_arr[combo_index] = float(metrics["min_hand_distance_m"])
                frame_left_pos_err[combo_index] = sparse["left_position_error_m"]
                frame_right_pos_err[combo_index] = sparse["right_position_error_m"]
                frame_left_ori_err[combo_index] = sparse["left_orientation_error_rad"]
                frame_right_ori_err[combo_index] = sparse["right_orientation_error_rad"]
                frame_converged[combo_index] = sparse["converged"]
                frame_mode[combo_index] = sparse["ik_mode"]
                frame_min_margin[combo_index] = sparse["joint_limit_min_margin_rad"]

                if best_index < 0 or score_arr[combo_index] < score_arr[best_index]:
                    best_index = combo_index
                    best_payload = payload
                    best_sparse = sparse
                    best_metrics = metrics
                combo_index += 1

    if best_index < 0 or best_payload is None or best_sparse is None or best_metrics is None:
        raise fail("Search did not produce a best candidate")

    np.savez(
        search_output,
        sample_frame_indices=sample_indices,
        scale_values=combo_scale,
        orientation_weight_values=combo_orientation_weight,
        align_rotation_rpy_deg_values=combo_align_rpy_deg,
        score=score_arr,
        valid=valid_arr,
        mean_position_error=mean_pos_arr,
        left_position_error_mean=left_mean_pos_arr,
        right_position_error_mean=right_mean_pos_arr,
        max_position_error=max_pos_arr,
        mean_orientation_error=mean_ori_arr,
        converged_frame_count=converged_count_arr,
        fallback_count=fallback_count_arr,
        nonconverged_count=nonconverged_count_arr,
        minimum_joint_limit_margin_rad=min_margin_arr,
        maximum_target_displacement_m=max_disp_arr,
        frame0_ok=frame0_ok_arr,
        min_hand_distance_m=min_hand_distance_arr,
        frame_left_position_error_m=frame_left_pos_err,
        frame_right_position_error_m=frame_right_pos_err,
        frame_left_orientation_error_rad=frame_left_ori_err,
        frame_right_orientation_error_rad=frame_right_ori_err,
        frame_converged=frame_converged,
        frame_ik_mode=frame_mode,
        frame_joint_limit_min_margin_rad=frame_min_margin,
        best_index=np.asarray(best_index, dtype=np.int64),
        model_path=np.asarray(str(model_path)),
        input_path=np.asarray(str(input_path)),
        search_score_definition=np.asarray(
            "10*mean_position_error + 5*max_position_error + 0.5*mean_orientation_error(if ow>0) + 0.2*fallback_count + 1.0*nonconverged_count"
        ),
    )

    best_scale = float(combo_scale[best_index])
    best_orientation_weight = float(combo_orientation_weight[best_index])
    best_align_rpy_deg = combo_align_rpy_deg[best_index].copy()
    best_targets_payload = build_best_targets_payload(
        best_payload,
        best_metrics,
        best_sparse,
        sample_indices,
        best_scale,
        best_orientation_weight,
        best_align_rpy_deg,
    )
    np.savez(best_output, **best_targets_payload)

    order = np.argsort(score_arr)
    top_k = min(10, total_combos)
    print(f"total combinations: {total_combos}")
    print("top 10 combinations:")
    for rank, idx in enumerate(order[:top_k], start=1):
        print(
            f"  {rank:02d}: score={score_arr[idx]:.6f} valid={bool(valid_arr[idx])} "
            f"scale={combo_scale[idx]:.2f} ow={combo_orientation_weight[idx]:.2f} "
            f"align={combo_align_rpy_deg[idx].tolist()} conv={int(converged_count_arr[idx])}/5 "
            f"fallback={int(fallback_count_arr[idx])} max_pos={max_pos_arr[idx]:.6f}"
        )
    print(f"best scale: {best_scale:.2f}")
    print(f"best orientation_weight: {best_orientation_weight:.2f}")
    print(f"best align_rotation_rpy_deg: {best_align_rpy_deg.tolist()}")
    print("best representative frame errors:")
    for i, frame_idx in enumerate(sample_indices):
        print(
            f"  frame {int(frame_idx)}: "
            f"left_pos={best_sparse['left_position_error_m'][i]:.6f} "
            f"right_pos={best_sparse['right_position_error_m'][i]:.6f} "
            f"left_ori={best_sparse['left_orientation_error_rad'][i]:.6f} "
            f"right_ori={best_sparse['right_orientation_error_rad'][i]:.6f} "
            f"converged={bool(best_sparse['converged'][i])} "
            f"mode={best_sparse['ik_mode'][i]}"
        )
    print(f"converged count: {int(best_metrics['converged_count'])}/5")
    print(f"fallback count: {int(best_metrics['fallback_count'])}")
    print(f"joint-limit minimum margin: {float(best_metrics['minimum_joint_limit_margin_rad']):.6f}")
    print(f"best_g1_cartesian_targets.npz: {best_output}")
    full_ik_ready = bool(best_metrics["converged_count"] >= 4 and best_metrics["max_position_error"] <= 0.03 and best_metrics["valid"])
    print(f"ready for full IK: {full_ik_ready}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
