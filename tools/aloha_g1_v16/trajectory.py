"""Semantic-progress contact-carrier trajectories and coupled temporal IK."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from aloha_g1_v15.kinematics import ActiveG1Dex3
from aloha_g1_v15.translator import mapped_relative

from .carrier import (
    _arm_torso_penetration,
    _assign_side,
    build_left_pinch_carrier,
    build_right_hook_carrier,
    inverse_pose,
)


def prohibited_penetration_vector(runtime: ActiveG1Dex3) -> np.ndarray:
    """Fixed-length active-model penetration vector used inside IK."""
    depths = {
        "arm_torso": 0.0,
        "arm_arm": 0.0,
        "hand_torso": 0.0,
        "hand_hand": 0.0,
        "same_hand": 0.0,
    }
    for row in runtime.penetrating_contacts(tolerance=0.0):
        bodies = row["bodies"]
        depth = -float(row["distance_m"])
        left = any(value.startswith("left_") for value in bodies)
        right = any(value.startswith("right_") for value in bodies)
        hand = ["hand" in value for value in bodies]
        arm = [any(token in value for token in ("shoulder", "elbow", "wrist")) for value in bodies]
        torso = [any(token in value for token in ("torso", "waist", "pelvis")) for value in bodies]
        if any(arm) and any(torso):
            depths["arm_torso"] = max(depths["arm_torso"], depth)
        if left and right and any(arm):
            depths["arm_arm"] = max(depths["arm_arm"], depth)
        if any(hand) and any(torso):
            depths["hand_torso"] = max(depths["hand_torso"], depth)
        if left and right and all(hand):
            depths["hand_hand"] = max(depths["hand_hand"], depth)
        if all(hand) and (left ^ right):
            depths["same_hand"] = max(depths["same_hand"], depth)
    return np.asarray(list(depths.values()), dtype=np.float64)


def smoothstep(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def path_progress(values: np.ndarray, start: int, end: int) -> np.ndarray:
    segment = np.asarray(values[start : end + 1], dtype=np.float64)
    if len(segment) <= 1:
        return np.ones(len(segment), dtype=np.float64)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(segment, axis=0), axis=1))]
    if distance[-1] <= np.finfo(np.float64).eps:
        return np.linspace(0.0, 1.0, len(segment))
    return distance / distance[-1]


def slerp_pair(start: np.ndarray, end: np.ndarray, progress: np.ndarray) -> np.ndarray:
    progress = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((start, end))))(progress).as_matrix()


def source_preserving_orientation_segment(
    source_rotation: np.ndarray,
    calibration: np.ndarray,
    start: int,
    end: int,
    start_target: np.ndarray,
    end_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source-relative rotation and add only a smooth endpoint residual."""
    relative = mapped_relative(source_rotation, start, end, calibration)
    base = np.einsum("ij,tjk->tik", start_target, relative)
    endpoint_registration = end_target @ base[-1].T
    source_step = Rotation.from_matrix(
        np.einsum("tji,tjk->tik", source_rotation[start:end], source_rotation[start + 1 : end + 1])
    ).magnitude()
    cumulative = np.r_[0.0, np.cumsum(np.abs(source_step))]
    if cumulative[-1] <= 1e-12:
        progress = np.linspace(0.0, 1.0, end - start + 1)
    else:
        progress = cumulative / cumulative[-1]
    correction = slerp_pair(np.eye(3), endpoint_registration, smoothstep(progress))
    return np.einsum("tij,tjk->tik", correction, base), correction


def semantic_translation_residual(
    base: np.ndarray,
    source_position: np.ndarray,
    anchor_indices: list[int],
    anchor_targets: list[np.ndarray],
    end_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """C1-smooth residual driven by source arc progress between semantic anchors."""
    base = np.asarray(base, dtype=np.float64)
    residual = np.zeros_like(base)
    anchors = list(zip(anchor_indices, [np.asarray(value) for value in anchor_targets]))
    deltas = [target - base[index] for index, target in anchors]
    first_index = anchors[0][0]
    initial_progress = smoothstep(path_progress(source_position, 0, first_index))
    residual[: first_index + 1] = initial_progress[:, None] * deltas[0]
    for pair in range(len(anchors) - 1):
        start = anchors[pair][0]
        end = anchors[pair + 1][0]
        progress = smoothstep(path_progress(source_position, start, end))
        residual[start : end + 1] = (
            (1.0 - progress[:, None]) * deltas[pair]
            + progress[:, None] * deltas[pair + 1]
        )
    last_index = anchors[-1][0]
    residual[last_index : end_index + 1] = deltas[-1]
    if end_index + 1 < len(base):
        count = len(base) - end_index
        decay = smoothstep(np.linspace(0.0, 1.0, count))
        residual[end_index:] = (1.0 - decay[:, None]) * deltas[-1]
    return base + residual, residual


def carrier_rotation_path(
    base_rotation: np.ndarray,
    source_rotation: np.ndarray,
    calibration: np.ndarray,
    semantic_anchors: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    length = len(base_rotation)
    output = np.asarray(base_rotation, dtype=np.float64).copy()
    correction = np.repeat(np.eye(3)[None], length, axis=0)
    first_index, first_target = semantic_anchors[0]
    first_registration = first_target @ output[first_index].T
    initial_progress = smoothstep(np.linspace(0.0, 1.0, first_index + 1))
    initial_correction = slerp_pair(np.eye(3), first_registration, initial_progress)
    output[: first_index + 1] = np.einsum(
        "tij,tjk->tik", initial_correction, output[: first_index + 1]
    )
    correction[: first_index + 1] = initial_correction
    for (start, start_target), (end, end_target) in zip(semantic_anchors[:-1], semantic_anchors[1:]):
        segment, local_correction = source_preserving_orientation_segment(
            source_rotation, calibration, start, end, start_target, end_target
        )
        output[start : end + 1] = segment
        # Registration relative to the original v14-derived carrier reference.
        correction[start : end + 1] = np.einsum(
            "tij,tkj->tik", segment, base_rotation[start : end + 1]
        )
    final_index, final_target = semantic_anchors[-1]
    final_registration = final_target @ base_rotation[final_index].T
    output[final_index:] = np.einsum(
        "ij,tjk->tik", final_registration, base_rotation[final_index:]
    )
    correction[final_index:] = final_registration
    return output, correction


def continuous_hand_profile(
    length: int,
    open_q: np.ndarray,
    contact_q: np.ndarray,
    acquire_progress: np.ndarray,
    release_progress: np.ndarray,
    acquire_start: int,
    acquire_end: int,
    release_start: int,
    release_end: int,
) -> np.ndarray:
    output = np.repeat(np.asarray(open_q, dtype=np.float64)[None], length, axis=0)
    acquire = smoothstep(acquire_progress[acquire_start : acquire_end + 1])
    output[acquire_start : acquire_end + 1] = (
        (1.0 - acquire[:, None]) * open_q + acquire[:, None] * contact_q
    )
    output[acquire_end : release_start + 1] = contact_q
    release = smoothstep(release_progress[release_start : release_end + 1])
    output[release_start : release_end + 1] = (
        (1.0 - release[:, None]) * contact_q + release[:, None] * open_q
    )
    output[release_end:] = open_q
    return output


def _rotation_residual(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def solve_coupled_side_trajectory(
    runtime: ActiveG1Dex3,
    side: str,
    reference_arm_q: np.ndarray,
    hand_reference: np.ndarray,
    carrier_target: np.ndarray,
    carrier_builder: Callable[[ActiveG1Dex3, np.ndarray, int], np.ndarray],
    table_height: float,
    *,
    carrier_position_weight: float,
    carrier_rotation_weight: float,
    reference_weight: float,
    temporal_weight: float,
    maximum_step_rad: float,
    max_nfev: int,
    collision_weight: float = 1200.0,
    maximum_hand_step_rad: float | None = None,
    opposite_hand_reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Continuation solve with arm and all seven hand joints as variables."""
    reference_arm_q = np.asarray(reference_arm_q, dtype=np.float64)
    hand_reference = np.asarray(hand_reference, dtype=np.float64)
    length = len(reference_arm_q)
    arm_output = reference_arm_q.copy()
    hand_output = hand_reference.copy()
    if opposite_hand_reference is None:
        opposite_hand_reference = np.repeat(
            runtime.open_hand_q["right" if side == "left" else "left"][None],
            length,
            axis=0,
        )
    opposite_hand_reference = np.asarray(opposite_hand_reference, dtype=np.float64)
    if opposite_hand_reference.shape != (length, 7):
        raise ValueError("opposite-hand trajectory must have shape [T,7]")
    arm_limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    block = slice(0, 7) if side == "left" else slice(7, 14)
    side_limits = arm_limits[block]
    hand_limits = runtime.hand_limits[side]
    full_lower = np.r_[side_limits[:, 0], hand_limits[:, 0]] + 1e-7
    full_upper = np.r_[side_limits[:, 1], hand_limits[:, 1]] - 1e-7
    trust_step = np.r_[
        np.repeat(maximum_step_rad, 7),
        np.repeat(maximum_step_rad if maximum_hand_step_rad is None else maximum_hand_step_rad, 7),
    ]
    carrier_position_error = np.empty(length)
    carrier_rotation_error = np.empty(length)
    collision_penetration = np.empty(length)
    for index in range(length):
        reference = np.r_[reference_arm_q[index, block], hand_reference[index]]
        if index == 0:
            previous = reference.copy()
            lower, upper = full_lower, full_upper
        else:
            previous = np.r_[arm_output[index - 1, block], hand_output[index - 1]]
            # A strict continuation trust region prevents the reference seed's
            # own branch changes from being copied into the v16 trajectory.
            lower = np.maximum(full_lower, previous - trust_step)
            upper = np.minimum(full_upper, previous + trust_step)
            invalid = lower >= upper
            lower[invalid], upper[invalid] = full_lower[invalid], full_upper[invalid]
        seed = np.clip(previous if index else reference, lower + 1e-9, upper - 1e-9)

        def state(value: np.ndarray):
            hand = value[7:]
            arm = np.asarray(reference_arm_q[index], dtype=np.float64).copy()
            arm[block] = value[:7]
            if side == "left":
                runtime.assign(arm, hand, opposite_hand_reference[index])
            else:
                runtime.assign(arm, opposite_hand_reference[index], hand)
            carrier = carrier_builder(runtime, hand, index)
            palm = runtime.palm_pose(side)
            return arm, hand, carrier, palm

        def residual(value: np.ndarray) -> np.ndarray:
            _, _, carrier, palm = state(value)
            return np.r_[
                carrier_position_weight * (carrier[:3, 3] - carrier_target[index, :3, 3]),
                carrier_rotation_weight * _rotation_residual(
                    carrier[:3, :3], carrier_target[index, :3, :3]
                ),
                reference_weight * (value - reference),
                temporal_weight * (value - previous),
                collision_weight * prohibited_penetration_vector(runtime),
                np.array([collision_weight * max(0.0, table_height + 0.002 - palm[2, 3])]),
            ]

        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            ftol=2e-8,
            xtol=2e-8,
            gtol=2e-8,
            x_scale="jac",
        )
        arm, hand, carrier, _ = state(solution.x)
        arm_output[index] = arm
        hand_output[index] = hand
        carrier_position_error[index] = np.linalg.norm(
            carrier[:3, 3] - carrier_target[index, :3, 3]
        )
        carrier_rotation_error[index] = _rotation_residual(
            carrier[:3, :3], carrier_target[index, :3, :3]
        ).dot(_rotation_residual(carrier[:3, :3], carrier_target[index, :3, :3])) ** 0.5
        collision_penetration[index] = _arm_torso_penetration(runtime)
        if index % 100 == 0:
            print(
                f"[V16_COUPLED_{side.upper()}] {index}/{length - 1} "
                f"carrier_mm={carrier_position_error[index] * 1000.0:.3f}",
                flush=True,
            )
    arm_step = np.max(np.abs(np.diff(arm_output[:, block], axis=0)), axis=1)
    q_step = np.max(np.abs(np.diff(np.c_[arm_output[:, block], hand_output], axis=0)), axis=1)
    branch_threshold = max(0.20, 1.5 * maximum_step_rad)
    branch_frames = (np.flatnonzero(arm_step > branch_threshold) + 1).tolist()
    metrics = {
        "side": side,
        "carrier_position_error_m": carrier_position_error,
        "carrier_rotation_error_rad": carrier_rotation_error,
        "arm_torso_penetration_m": collision_penetration,
        "maximum_joint_step_rad": float(np.max(q_step)),
        "maximum_arm_joint_step_rad": float(np.max(arm_step)),
        "branch_discontinuity_frames": branch_frames,
        "branch_discontinuity_count": len(branch_frames),
        "carrier_position_5mm_rate": float(np.mean(carrier_position_error <= 0.005)),
        "maximum_carrier_position_error_m": float(np.max(carrier_position_error)),
        "maximum_carrier_rotation_error_rad": float(np.max(carrier_rotation_error)),
    }
    return arm_output, hand_output, metrics


def pose_deviation(reference: np.ndarray, corrected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(corrected[:, :3, 3] - reference[:, :3, 3], axis=1)
    rotation = Rotation.from_matrix(
        np.einsum("tij,tkj->tik", corrected[:, :3, :3], reference[:, :3, :3])
    ).magnitude()
    return translation, rotation


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "percentile_95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }
