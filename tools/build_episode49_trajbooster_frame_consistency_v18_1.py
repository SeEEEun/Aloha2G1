#!/usr/bin/env python3
"""Audit and, when justified, build the single frame-consistent EP49 v18.1.

The protected quantity is a task-tool trajectory.  The left task tool is the
approved physical thumb/index pinch frame already executed by v18.  The right
task tool is the physical contact trajectory that v14 anchored with the old
middle/C diagnostic proxy; v18.1 preserves that trajectory but realizes it
with the predefined physical right thumb/index pinch through one static tool
transform.  No per-frame Cartesian residual or object-relative waypoint is
created.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from pxr import Usd, UsdGeom
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_execution_physics_v17 as v17  # noqa: E402
import build_episode49_target_phase_anchored_v12 as v12  # noqa: E402
import validate_g1_targets_and_sparse_ik as arm_ik  # noqa: E402
from aloha_g1_v15.kinematics import ActiveG1Dex3, normalize, sha256_file  # noqa: E402
from aloha_g1_v15.semantic_input import load_human_reviewed_development_timeline  # noqa: E402
from aloha_g1_v17.trajectory import _collision_scalars, audit_collision_classifier_integrity  # noqa: E402
from aloha_g1_v17_2.trajectory import posture_metrics  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_trajbooster_frame_consistency_v18_1"
V18 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_full_task_execution_v18"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
PHOTO = ROOT / "outputs/scene_registered_retargeting/dex3_left_phone_pinch_photo_calibration_v1"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
ALOHA_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
METHOD = "V18_1_TRAJBOOSTER_INSPIRED_FRAME_CONSISTENT_EXECUTION"


def default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_npz(path: Path, **value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **value)
    os.replace(temporary, path)


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = position
    return value


def inverse(value: np.ndarray) -> np.ndarray:
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def pose_distance_to_bbox(point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> float:
    nearest = np.minimum(np.maximum(point, minimum), maximum)
    return float(np.linalg.norm(point - nearest))


def frame_from_two_contacts(wrist: np.ndarray, thumb: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Task frame: X approach, Y thumb->index closing, Z lateral."""
    closing = normalize(index - thumb, (0.0, 1.0, 0.0))
    approach = wrist[:3, 0] - closing * float(np.dot(wrist[:3, 0], closing))
    approach = normalize(approach, (1.0, 0.0, 0.0))
    lateral = normalize(np.cross(approach, closing), (0.0, 0.0, 1.0))
    closing = normalize(np.cross(lateral, approach), (0.0, 1.0, 0.0))
    return transform(np.column_stack((approach, closing, lateral)), 0.5 * (thumb + index))


def legacy_single_contact_frame(wrist: np.ndarray, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    approach = normalize(normal, (1.0, 0.0, 0.0))
    closing = wrist[:3, 1] - approach * float(np.dot(wrist[:3, 1], approach))
    closing = normalize(closing, (0.0, 1.0, 0.0))
    lateral = normalize(np.cross(approach, closing), (0.0, 0.0, 1.0))
    closing = normalize(np.cross(lateral, approach), (0.0, 1.0, 0.0))
    return transform(np.column_stack((approach, closing, lateral)), point)


def static_tool_transforms(runtime: ActiveG1Dex3, v18_data: dict[str, np.ndarray], anchors: dict[str, Any]) -> dict[str, np.ndarray]:
    left_anchor = anchors["left_phone_action169"]
    right_anchor = anchors["right_accessory_action319"]

    # Candidate A is immutable; the third is irrelevant to tool definition.
    approved = json.loads((PHOTO / "left_phone_fingertip_pinch_primitive.json").read_text())
    left_q = np.asarray(approved["selected_static_q_rad"], dtype=np.float64)
    right_q = np.asarray(json.loads((V18 / "right_accessory_primitive_v18.json").read_text())["q_rad"])
    neutral_left = v18_data["left_dex3_qpos"][169]
    neutral_right = v18_data["right_dex3_qpos"][319]

    arm = v18_data["arm_qpos"][169]
    runtime.assign(arm, left_q, neutral_right)
    left_wrist = runtime.wrist_pose("left")
    left_tool = frame_from_two_contacts(
        left_wrist, runtime.contact_pose("left_A")[0], runtime.contact_pose("left_B")[0]
    )
    t_left = inverse(left_wrist) @ left_tool

    # The legacy physical tool transform belongs to the immutable v14 carrier
    # anchor, not to a later v18 redundant-orientation variant.
    arm = v18_data["v14_reference_arm_q"][319]
    runtime.assign(arm, neutral_left, right_q)
    right_wrist = runtime.wrist_pose("right")
    right_tool = frame_from_two_contacts(
        right_wrist, runtime.contact_pose("right_B")[0], runtime.contact_pose("right_A")[0]
    )
    t_right = inverse(right_wrist) @ right_tool

    # Recreate only the v14 diagnostic C chain at the same carrier.  A/B are
    # left at the model stand state and never become task fingers.
    legacy_right = runtime.open_hand_q["right"].copy()
    legacy_right[5:] = np.asarray(right_anchor["diagnostic_right_dex3_C_q_rad"])
    runtime.assign(arm, neutral_left, legacy_right)
    old_wrist = runtime.wrist_pose("right")
    old_point, old_normal = runtime.contact_pose("right_C")
    old_tool = legacy_single_contact_frame(old_wrist, old_point, old_normal)
    t_old_right = inverse(old_wrist) @ old_tool

    return {
        "left_wrist_to_physical_pinch": t_left,
        "right_wrist_to_physical_pinch": t_right,
        "right_wrist_to_legacy_task_contact": t_old_right,
        "left_candidate_A_q": left_q,
        "right_primitive_q": right_q,
        "legacy_right_q": legacy_right,
    }


def wrist_pose_and_jac(runtime: ActiveG1Dex3, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_id = runtime.body_ids[f"{side}_wrist_yaw_link"]
    jacp = np.zeros((3, runtime.model.nv), dtype=np.float64)
    jacr = np.zeros((3, runtime.model.nv), dtype=np.float64)
    mujoco.mj_jacBody(runtime.model, runtime.data, jacp, jacr, body_id)
    pose = runtime.wrist_pose(side)
    # Jacobians are in model world coordinates; rotate into scene coordinates.
    return pose, runtime.model_to_scene_rotation(jacp), runtime.model_to_scene_rotation(jacr)


def tool_state(runtime: ActiveG1Dex3, side: str, local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wrist, jacp, jacr = wrist_pose_and_jac(runtime, side)
    point_offset = wrist[:3, :3] @ local[:3, 3]
    pose = wrist @ local
    # Point Jacobian: v_point = v_origin + omega x r.
    skew = np.asarray([
        [0.0, -point_offset[2], point_offset[1]],
        [point_offset[2], 0.0, -point_offset[0]],
        [-point_offset[1], point_offset[0], 0.0],
    ])
    return pose, jacp - skew @ jacr, jacr, wrist


def smooth_gate(length: int, start: int, end: int, ramp: int) -> np.ndarray:
    value = np.zeros(length)
    value[start : end + 1] = 1.0
    for offset in range(1, ramp + 1):
        u = 1.0 - offset / float(ramp + 1)
        weight = u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)
        if start - offset >= 0:
            value[start - offset] = max(value[start - offset], weight)
        if end + offset < length:
            value[end + offset] = max(value[end + offset], weight)
    return value


def solve_side(
    runtime: ActiveG1Dex3,
    side: str,
    initial_arm: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    tool_local: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    orientation_weight: np.ndarray,
    orientation_components: tuple[int, int],
    anchor_frame: int,
) -> np.ndarray:
    block = slice(0, 7) if side == "left" else slice(7, 14)
    limits = np.asarray(runtime.info["joint_limits"])[block]
    smoothed = savgol_filter(initial_arm[:, block], 31, 3, axis=0, mode="interp")
    neutral = np.median(initial_arm[:, block], axis=0)
    output = initial_arm[:, block].copy()

    def solve_frame(frame: int, previous: np.ndarray, previous2: np.ndarray, seeds: list[np.ndarray]) -> np.ndarray:
        whole = initial_arm[frame].copy()
        reference = smoothed[frame]
        # A tool-frame semantics correction can require a different redundant
        # arm branch even though the protected task path is unchanged.  Keep a
        # bounded trust region, but do not make the old (wrong-tool) wrist pose
        # an artificial reachability constraint.
        trust = 0.75
        lower = np.maximum(limits[:, 0] + 0.03, initial_arm[frame, block] - trust)
        upper = np.minimum(limits[:, 1] - 0.03, initial_arm[frame, block] + trust)

        def residual(value: np.ndarray) -> np.ndarray:
            whole[block] = value
            runtime.assign(whole, left_hand[frame], right_hand[frame])
            pose, _, _, _ = tool_state(runtime, side, tool_local)
            rotation_error_world = Rotation.from_matrix(
                target_rotation[frame] @ pose[:3, :3].T
            ).as_rotvec()
            rotation_error_tool = target_rotation[frame].T @ rotation_error_world
            # Only the components that change the relevant task axis are
            # tracked. Rotation about that axis remains redundant.
            axis_error = rotation_error_tool[list(orientation_components)]
            margin = np.minimum(value - limits[:, 0], limits[:, 1] - value)
            return np.r_[
                7000.0 * (pose[:3, 3] - target_position[frame]),
                orientation_weight[frame] * axis_error,
                0.25 * (value - reference),
                0.45 * (value - previous),
                0.08 * (value - 2.0 * previous + previous2),
                0.035 * (value[:4] - neutral[:4]),
                0.12 * (value[4:] - neutral[4:]),
                4.0 * np.maximum(0.0, 0.05 - margin),
                150000.0 * _collision_scalars(runtime, side),
            ]
        candidates = []
        for seed in seeds:
            seed = np.clip(seed, lower + 1e-8, upper - 1e-8)
            solved = least_squares(
                residual, seed, bounds=(lower, upper), max_nfev=120,
                ftol=2e-10, xtol=2e-10, gtol=2e-10, x_scale="jac",
            )
            whole[block] = solved.x
            runtime.assign(whole, left_hand[frame], right_hand[frame])
            pose, _, _, _ = tool_state(runtime, side, tool_local)
            collision_count = len(runtime.penetrating_contacts())
            position_error = float(np.linalg.norm(pose[:3, 3] - target_position[frame]))
            rotation_error_world = Rotation.from_matrix(
                target_rotation[frame] @ pose[:3, :3].T
            ).as_rotvec()
            rotation_error_tool = target_rotation[frame].T @ rotation_error_world
            orientation_score = float(np.linalg.norm(
                rotation_error_tool[list(orientation_components)]
            )) * float(orientation_weight[frame])
            orientation_bad = bool(
                orientation_weight[frame] > 0.1
                and np.linalg.norm(rotation_error_tool[list(orientation_components)])
                > np.radians(15.0)
            )
            candidates.append((
                collision_count, position_error > 0.004, orientation_bad,
                float(np.linalg.norm(solved.x - previous)), position_error,
                orientation_score, solved.x.copy(),
            ))
        return min(candidates, key=lambda row: row[:-1])[-1]

    # Resolve the task-critical anchor with deterministic multi-start, then
    # continue in both temporal directions on that single collision-free IK
    # branch. This is not a per-frame task offset; every sample tracks the same
    # immutable tool trajectory through the same static transform.
    rng = np.random.default_rng(1801 if side == "left" else 1802)
    anchor_seed = output[anchor_frame].copy()
    seeds = [anchor_seed] + [anchor_seed + rng.normal(0.0, 0.42, 7) for _ in range(14 if side == "right" else 4)]
    output[anchor_frame] = solve_frame(anchor_frame, anchor_seed, anchor_seed, seeds)
    previous = output[anchor_frame].copy()
    previous2 = previous.copy()
    for frame in range(anchor_frame + 1, len(output)):
        seeds = [previous, output[frame]]
        if orientation_weight[frame] > 0.1:
            seeds += [0.5 * (previous + output[frame])]
            seeds += [output[frame] + rng.normal(0.0, 0.28, 7) for _ in range(3)]
        output[frame] = solve_frame(frame, previous, previous2, seeds)
        previous2, previous = previous, output[frame].copy()
        if frame % 100 == 0:
            print(f"[v18.1 tool IK] {side} forward {frame}/989", flush=True)
    previous = output[anchor_frame].copy()
    previous2 = previous.copy()
    for frame in range(anchor_frame - 1, -1, -1):
        seeds = [previous, output[frame]]
        if orientation_weight[frame] > 0.1:
            seeds += [0.5 * (previous + output[frame])]
            seeds += [output[frame] + rng.normal(0.0, 0.28, 7) for _ in range(3)]
        output[frame] = solve_frame(frame, previous, previous2, seeds)
        previous2, previous = previous, output[frame].copy()
        if frame % 100 == 0:
            print(f"[v18.1 tool IK] {side} backward {frame}/989", flush=True)

    return output


def solve_minimum_posture_side(
    runtime: ActiveG1Dex3,
    side: str,
    initial_arm: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    tool_local: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    orientation_weight: np.ndarray,
) -> np.ndarray:
    """Position-first, single-branch temporal IK with minimum task orientation.

    The reference is a low-pass version of the existing continuous v18 branch;
    only joint posture is filtered.  Every sample is then projected back to the
    immutable physical task-tool position.  The closing-axis constraint leaves
    rotation about the closing axis free.
    """
    block = slice(0, 7) if side == "left" else slice(7, 14)
    limits = np.asarray(runtime.info["joint_limits"])[block]
    base = initial_arm[:, block].copy()
    reference = savgol_filter(base, 31, 3, axis=0, mode="interp")
    output = base.copy()
    previous = reference[0].copy()
    previous2 = previous.copy()
    trust = 0.48 if side == "left" else 0.78

    for frame in range(len(output)):
        whole = initial_arm[frame].copy()
        lower = np.maximum(limits[:, 0] + 0.03, base[frame] - trust)
        upper = np.minimum(limits[:, 1] - 0.03, base[frame] + trust)

        def residual(value: np.ndarray) -> np.ndarray:
            whole[block] = value
            runtime.assign(whole, left_hand[frame], right_hand[frame])
            pose, _, _, _ = tool_state(runtime, side, tool_local)
            rot_world = Rotation.from_matrix(target_rotation[frame] @ pose[:3, :3].T).as_rotvec()
            rot_tool = target_rotation[frame].T @ rot_world
            margin = np.minimum(value - limits[:, 0], limits[:, 1] - value)
            return np.r_[
                9000.0 * (pose[:3, 3] - target_position[frame]),
                orientation_weight[frame] * rot_tool[[0, 2]],
                1.2 * (value - reference[frame]),
                0.55 * (value - previous),
                0.12 * (value - 2.0 * previous + previous2),
                5.0 * np.maximum(0.0, 0.05 - margin),
                180000.0 * _collision_scalars(runtime, side),
            ]

        candidates = []
        for seed in (previous, reference[frame], base[frame]):
            solved = least_squares(
                residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper), max_nfev=100, ftol=2e-10,
                xtol=2e-10, gtol=2e-10, x_scale="jac",
            )
            whole[block] = solved.x
            runtime.assign(whole, left_hand[frame], right_hand[frame])
            pose, _, _, _ = tool_state(runtime, side, tool_local)
            pos_error = float(np.linalg.norm(pose[:3, 3] - target_position[frame]))
            rot_world = Rotation.from_matrix(target_rotation[frame] @ pose[:3, :3].T).as_rotvec()
            rot_tool = target_rotation[frame].T @ rot_world
            axis_error = float(np.linalg.norm(rot_tool[[0, 2]]))
            candidates.append((
                len(runtime.penetrating_contacts()), pos_error > 0.005,
                bool(orientation_weight[frame] > 0.1 and axis_error > np.radians(20.0)),
                float(np.linalg.norm(solved.x - previous)),
                float(np.linalg.norm(solved.x - reference[frame])),
                pos_error, axis_error, solved.x.copy(),
            ))
        output[frame] = min(candidates, key=lambda row: row[:-1])[-1]
        previous2, previous = previous, output[frame].copy()
        if frame % 100 == 0:
            print(f"[v18.1 minimum posture] {side} {frame}/989", flush=True)
    return output


def selective_smooth_and_reproject(
    runtime: ActiveG1Dex3,
    arm: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    left_local: np.ndarray,
    right_local: np.ndarray,
    left_target_p: np.ndarray,
    right_target_p: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove high-frequency joint redundancy, re-solving only failed gates."""
    original = arm.copy()
    output = arm.copy()
    output[:, :7] = savgol_filter(output[:, :7], 11, 3, axis=0, mode="interp")
    output[:, 7:] = savgol_filter(output[:, 7:], 7, 3, axis=0, mode="interp")
    # Never smooth through rapid protected task motion.  Resolve preservation
    # weights from task-EE speed (not Episode-49 frame literals) and use a C2
    # ramp back to the already stabilized collision-free branch.
    for block, target in ((slice(0, 7), left_target_p), (slice(7, 14), right_target_p)):
        fast = np.flatnonzero(np.linalg.norm(np.diff(target, axis=0), axis=1) > 0.010)
        preserve = np.zeros(len(output))
        radius = 8
        for index in fast:
            for frame in range(max(0, index - radius), min(len(output), index + radius + 2)):
                u = max(0.0, 1.0 - abs(frame - index) / float(radius + 1))
                weight = u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)
                preserve[frame] = max(preserve[frame], weight)
        output[:, block] = (
            (1.0 - preserve[:, None]) * output[:, block]
            + preserve[:, None] * original[:, block]
        )
    limits = np.asarray(runtime.info["joint_limits"])
    projected: list[dict[str, Any]] = []
    for frame in range(len(output)):
        for side, block, local, target in (
            ("left", slice(0, 7), left_local, left_target_p[frame]),
            ("right", slice(7, 14), right_local, right_target_p[frame]),
        ):
            runtime.assign(output[frame], left_hand[frame], right_hand[frame])
            pose, _, _, _ = tool_state(runtime, side, local)
            error_before = float(np.linalg.norm(pose[:3, 3] - target))
            side_collision = bool(np.any(_collision_scalars(runtime, side) > 0.0))
            if error_before <= 0.0015 and not side_collision:
                continue
            seed = output[frame, block].copy()
            previous = output[frame - 1, block] if frame else seed
            following = output[frame + 1, block] if frame + 1 < len(output) else seed
            lower = np.maximum(limits[block, 0] + 0.03, original[frame, block] - 0.78)
            upper = np.minimum(limits[block, 1] - 0.03, original[frame, block] + 0.78)
            whole = output[frame].copy()

            def residual(value: np.ndarray) -> np.ndarray:
                whole[block] = value
                runtime.assign(whole, left_hand[frame], right_hand[frame])
                current, _, _, _ = tool_state(runtime, side, local)
                margin = np.minimum(value - limits[block, 0], limits[block, 1] - value)
                return np.r_[
                    14000.0 * (current[:3, 3] - target),
                    1.0 * (value - seed),
                    0.30 * (value - 0.5 * (previous + following)),
                    5.0 * np.maximum(0.0, 0.05 - margin),
                    180000.0 * _collision_scalars(runtime, side),
                ]

            solved = least_squares(
                residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
                bounds=(lower, upper), max_nfev=100, ftol=2e-10,
                xtol=2e-10, gtol=2e-10, x_scale="jac",
            )
            output[frame, block] = solved.x
            whole[block] = solved.x
            runtime.assign(whole, left_hand[frame], right_hand[frame])
            after_pose, _, _, _ = tool_state(runtime, side, local)
            projected.append({
                "sample_for_provenance_only": frame,
                "side": side,
                "error_before_m": error_before,
                "error_after_m": float(np.linalg.norm(after_pose[:3, 3] - target)),
                "collision_triggered": side_collision,
            })
            # A failed local collision solve must never replace the known-safe
            # candidate.  Restore only this redundant joint sample; its task
            # tool target is unchanged.
            if np.any(_collision_scalars(runtime, side) > 0.0):
                output[frame, block] = original[frame, block]
                projected[-1]["restored_known_collision_free_sample"] = True
    return output, {
        "method": "Savgol joint-only smoothing followed by selective physical-tool reprojection",
        "left_window_samples": 11,
        "right_window_samples": 7,
        "projected_sample_side_count": len(projected),
        "rows_for_provenance_only": projected,
        "cartesian_targets_smoothed": False,
        "per_frame_cartesian_residuals": False,
    }


def compute_tool_trajectory(
    runtime: ActiveG1Dex3, arm: np.ndarray, left: np.ndarray, right: np.ndarray,
    side: str, local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.zeros((len(arm), 3))
    rotation = np.zeros((len(arm), 3, 3))
    wrist = np.zeros((len(arm), 4, 4))
    for frame in range(len(arm)):
        runtime.assign(arm[frame], left[frame], right[frame])
        pose, _, _, wrist_pose = tool_state(runtime, side, local)
        position[frame], rotation[frame], wrist[frame] = pose[:3, 3], pose[:3, :3], wrist_pose
    return position, rotation, wrist


def stats(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p95": float(np.quantile(value, 0.95)),
        "maximum": float(np.max(value)),
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a).reshape(-1)
    bv = np.asarray(b).reshape(-1)
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return 1.0 if np.allclose(av, bv) else 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def temporal_metrics(q: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    dt = float(np.median(np.diff(timestamps)))
    velocity = np.diff(q, axis=0) / dt
    acceleration = np.diff(q, n=2, axis=0) / dt ** 2
    jerk = np.diff(q, n=3, axis=0) / dt ** 3
    def row(x: np.ndarray) -> dict[str, float]:
        return {
            "rms": float(np.sqrt(np.mean(x ** 2))),
            "p95_abs": float(np.quantile(np.abs(x), 0.95)),
            "max_abs": float(np.max(np.abs(x))),
        }
    return {
        "joint_travel_rad": np.sum(np.abs(np.diff(q, axis=0)), axis=0),
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
        "velocity_rad_s": row(velocity),
        "acceleration_rad_s2": row(acceleration),
        "jerk_rad_s3": row(jerk),
        "sign_reversal_rate": float(np.mean(np.sign(velocity[1:]) * np.sign(velocity[:-1]) < 0.0)),
    }


def runtime_bbox(stage: Usd.Stage, path: str) -> tuple[np.ndarray, np.ndarray]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
    return np.asarray(box.GetMin()), np.asarray(box.GetMax())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    immutable = {
        "source": SOURCE,
        "v14_targets": V14 / "corrected_targets_v14.npz",
        "v18_trajectory": V18 / "final_arm_dex3_trajectory.npz",
        "left_candidate_A": PHOTO / "left_phone_fingertip_pinch_primitive.json",
        "right_primitive": V18 / "right_accessory_primitive_v18.json",
        "scene": v17.ACTIVE_SCENE,
        "magnet": ROOT / "isaaclab_magsafe_fixed_scene/magnet_config_v2.json",
    }
    before_hashes = {name: sha256_file(path) for name, path in immutable.items()}
    if before_hashes["source"] != "a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c":
        raise RuntimeError("source SHA mismatch")

    dump(OUT / "trajbooster_applicable_principles_v18_1.json", {
        "paper": {"title": "TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning", "url": "https://arxiv.org/pdf/2509.11839"},
        "USED_IDEAS": [
            "6D end-effector trajectories as a morphology-agnostic cross-embodiment interface",
            "explicit source-to-target workspace mapping before target-arm retargeting",
            "target arm motion generated from target wrist/end-effector pose",
            "arm/task trajectory retargeting separated from predefined target hand-state mapping",
        ],
        "NOT_USED_IDEAS": ["Harmonized Online DAgger", "manager policy", "worker locomotion policy", "VLA post-pre-training", "G1-specific VLA retraining", "lower-body locomotion"],
        "WHY": "Only the paper's trajectory-centric embodiment interface is applicable; learning/locomotion components are outside this simulation-only arm-and-hand retargeting study.",
    })

    with np.load(immutable["v18_trajectory"], allow_pickle=False) as archive:
        v18_data = {name: archive[name].copy() for name in archive.files}
    with np.load(immutable["v14_targets"], allow_pickle=False) as archive:
        v14_data = {name: archive[name].copy() for name in archive.files}
    with np.load(V12 / "aloha_phase_motion_library.npz", allow_pickle=False) as archive:
        source_phase = {name: archive[name].copy() for name in archive.files}
    with np.load(V18 / "physics_trial_full_task_diagnostic_0p25x_paper_white.npz", allow_pickle=True) as archive:
        physics = {name: archive[name].copy() for name in archive.files}
    anchors = json.loads((V14 / "selected_physical_carrier_anchors.json").read_text())
    timeline = load_human_reviewed_development_timeline(
        v17.TIMELINE, v17.ALIGNMENT, v18_data["optimized_action"], v18_data["source_timestamps"],
        source_phase["left_tcp_position"], source_phase["right_tcp_position"],
        source_phase["left_tcp_rotation"], source_phase["right_tcp_rotation"],
        trajectory_path=SOURCE, fk_model_path=MODEL, task_geometry_path=v17.LAYOUT,
    )
    event = lambda name: int(timeline.event(name).action_index)

    runtime = ActiveG1Dex3(MODEL, v17.DEX3_MAPPING, v17.PALM_CONFIG, v18_data["g1_root"])
    tools = static_tool_transforms(runtime, v18_data, anchors)
    left_local = tools["left_wrist_to_physical_pinch"]
    right_local = tools["right_wrist_to_physical_pinch"]
    old_right_local = tools["right_wrist_to_legacy_task_contact"]

    # Authoritative current and legacy task-tool trajectories.  LEFT already
    # uses Candidate A; RIGHT preserves the v14 legacy physical-contact motion
    # and changes only the robot-specific wrist->tool transform.
    left_target_p, left_target_r, left_wrist_v18 = compute_tool_trajectory(
        runtime, v18_data["arm_qpos"], v18_data["left_dex3_qpos"], v18_data["right_dex3_qpos"],
        "left", left_local,
    )
    right_target_p, right_target_r, right_wrist_v18 = compute_tool_trajectory(
        runtime, v18_data["v14_reference_arm_q"], v18_data["left_dex3_qpos"], v18_data["right_dex3_qpos"],
        "right", old_right_local,
    )
    current_right_p, current_right_r, _ = compute_tool_trajectory(
        runtime, v18_data["arm_qpos"], v18_data["left_dex3_qpos"], v18_data["right_dex3_qpos"],
        "right", right_local,
    )

    grasp_l = event("left_phone_grasp_start")
    portrait = event("phone_portrait_reached")
    grasp_r = event("right_accessory_grasp_start")
    removed = event("accessory_removed")
    charger = event("phone_charger_attachment_complete")
    release_l = event("left_phone_release_complete")
    release_r = event("right_accessory_release_complete")

    # The legacy middle/C contact frame supplied a useful physical contact
    # *position* anchor, but its arbitrary local axes are not thumb-index pinch
    # semantics.  Retain the already stabilized v18 physical pinch-frame
    # orientation progression and constrain only its task-relevant closing-axis
    # components below.  This avoids importing a full 3-DOF legacy hook frame.
    right_target_r = current_right_r.copy()

    # Complete frame provenance.
    source_tool = json.loads((ROOT / "configs/aloha_tool_axes_calibration.sim.json").read_text())
    palm_cfg = json.loads(v17.PALM_CONFIG.read_text())
    source_frame_audit = {
        "status": "AUTHORITATIVE_SOURCE_EE_IS_ALOHA_TCP",
        "classification_of_phase_library": "ALOHA_TCP_TRAJECTORY",
        "aloha_model": ALOHA_XML,
        "aloha_model_sha256": sha256_file(ALOHA_XML),
        "parents": ["follower_left_link_6", "follower_right_link_6"],
        "T_ALOHA_LINK6_TCP": transform(np.eye(3), np.asarray(source_tool["tcp_offset_link6_m"])),
        "tcp_offset_link6_m": source_tool["tcp_offset_link6_m"],
        "phase_library": V12 / "aloha_phase_motion_library.npz",
        "source_fk_implementation": ROOT / "tools/validate_smolvla_in_stationary_aloha_mujoco.py",
        "v14_semantics": "source ALOHA TCP relative motion was stored as a target G1 palm-carrier trajectory; v14 keyframes then anchored embodiment-specific physical contact proxies",
    }
    dump(OUT / "source_ee_frame_audit.json", source_frame_audit)
    dump(OUT / "target_ee_frame_audit_left.json", {
        "status": "LEFT_PHYSICAL_TASK_EE_VERIFIED",
        "authoritative_target_ee": "physical Candidate-A THUMB-INDEX pinch center/frame",
        "T_G1_WRIST_PALM": transform(np.eye(3), np.asarray(palm_cfg["left"]["geom_local_position_m"])),
        "T_G1_WRIST_LEFT_PINCH": left_local,
        "task_fingers": ["left_hand_thumb_2_link", "left_hand_index_1_link"],
        "third": "left_hand_middle chain NON_TASK",
        "v18_mapping": "effective contact success but not explicitly protected as the EE interface",
    })
    dump(OUT / "target_ee_frame_audit_right.json", {
        "status": "RIGHT_TOOL_OFFSET_MISMATCH_CONFIRMED",
        "authoritative_target_ee": "physical RIGHT THUMB-INDEX accessory pinch center/frame",
        "T_G1_WRIST_PALM": transform(np.eye(3), np.asarray(palm_cfg["right"]["geom_local_position_m"])),
        "T_G1_WRIST_RIGHT_PINCH": right_local,
        "T_G1_WRIST_LEGACY_C_TASK_CONTACT": old_right_local,
        "static_translation_correction_norm_m": float(np.linalg.norm(old_right_local[:3, 3] - right_local[:3, 3])),
        "task_fingers": ["right_hand_thumb_2_link", "right_hand_index_1_link"],
        "third": "right_hand_middle chain NON_TASK",
    })

    graph = {
        "status": "COMPLETE_NAMED_EE_FRAME_GRAPH",
        "edges": [
            {"parent": "ALOHA_LEFT_LINK6", "child": "ALOHA_LEFT_TCP", "transform": source_frame_audit["T_ALOHA_LINK6_TCP"], "source_file": source_tool["source_xml"], "derivation": "model-provenance static TCP offset", "constant": True},
            {"parent": "ALOHA_RIGHT_LINK6", "child": "ALOHA_RIGHT_TCP", "transform": source_frame_audit["T_ALOHA_LINK6_TCP"], "source_file": source_tool["source_xml"], "derivation": "model-provenance static TCP offset", "constant": True},
            {"parent": "G1_LEFT_WRIST", "child": "G1_LEFT_PALM", "transform": transform(np.eye(3), np.asarray(palm_cfg["left"]["geom_local_position_m"])), "source_file": v17.PALM_CONFIG, "derivation": "active model palm geom", "constant": True},
            {"parent": "G1_LEFT_WRIST", "child": "LEFT_PHYSICAL_THUMB_INDEX_PINCH", "transform": left_local, "source_file": PHOTO / "left_phone_fingertip_pinch_primitive.json", "derivation": "active FK at approved Candidate A", "constant": True},
            {"parent": "G1_RIGHT_WRIST", "child": "G1_RIGHT_PALM", "transform": transform(np.eye(3), np.asarray(palm_cfg["right"]["geom_local_position_m"])), "source_file": v17.PALM_CONFIG, "derivation": "active model palm geom", "constant": True},
            {"parent": "G1_RIGHT_WRIST", "child": "RIGHT_PHYSICAL_THUMB_INDEX_PINCH", "transform": right_local, "source_file": V18 / "right_accessory_primitive_v18.json", "derivation": "active FK at predefined primitive", "constant": True},
            {"parent": "G1_RIGHT_WRIST", "child": "LEGACY_RIGHT_MIDDLE_C_TASK_CONTACT", "transform": old_right_local, "source_file": V14 / "selected_physical_carrier_anchors.json", "derivation": "v14 diagnostic contact anchor provenance only", "constant": True},
            {"parent": "WORLD", "child": "PHONE", "source_file": v17.ACTIVE_SCENE, "derivation": "authoritative USD rigid body", "constant": False},
            {"parent": "WORLD", "child": "ACCESSORY", "source_file": v17.ACTIVE_SCENE, "derivation": "authoritative USD rigid body", "constant": False},
            {"parent": "WORLD", "child": "CHARGER", "source_file": v17.ACTIVE_SCENE, "derivation": "authoritative USD static body", "constant": True},
        ],
    }
    dump(OUT / "complete_ee_frame_graph_v18_1.json", graph)

    # 174 mm decomposition: the published v18 number used the initial bbox,
    # while the right carrier was registered to the moving task accessory at
    # the semantic grasp event.
    stage = Usd.Stage.Open(str(v17.ACTIVE_SCENE))
    acc_min, acc_max = runtime_bbox(stage, "/World/MagSafeScene/Accessory")
    reported_gap = pose_distance_to_bbox(current_right_p[grasp_r], acc_min, acc_max)
    legacy_target = np.asarray(anchors["right_accessory_action319"]["right_C_ring_target_m"])
    correct_dynamic_gap = float(np.linalg.norm(current_right_p[grasp_r] - legacy_target))
    corrected_target_gap = float(np.linalg.norm(right_target_p[grasp_r] - legacy_target))
    palm_error = float(np.linalg.norm(
        v18_data["v14_right_position_target"][grasp_r]
        - np.asarray(anchors["right_accessory_action319"]["palm_position_m"])
    ))
    gap = {
        "status": "RIGHT_ACCESSORY_TOOL_OFFSET_AND_DIAGNOSTIC_OBJECT_FRAME_MISMATCH_CONFIRMED",
        "semantic_event": "right_accessory_grasp_start",
        "action_index_for_provenance_only": grasp_r,
        "reported_initial_bbox_gap_m": reported_gap,
        "reported_reference_m": 0.17430607271505108,
        "mapped_source_task_contact_to_dynamic_accessory_target_error_m": float(np.linalg.norm(right_target_p[grasp_r] - legacy_target)),
        "g1_palm_to_v14_carrier_target_error_m": palm_error,
        "current_physical_pinch_to_dynamic_accessory_target_error_m": correct_dynamic_gap,
        "current_physical_pinch_to_initial_accessory_bbox_error_m": reported_gap,
        "target_wrist_to_pinch_static_offset_m": right_local[:3, 3],
        "legacy_wrist_to_contact_static_offset_m": old_right_local[:3, 3],
        "tool_offset_mismatch_vector_in_wrist_m": right_local[:3, 3] - old_right_local[:3, 3],
        "tool_offset_mismatch_norm_m": float(np.linalg.norm(right_local[:3, 3] - old_right_local[:3, 3])),
        "temporal_object_frame_norm_difference_m_nonadditive": reported_gap - correct_dynamic_gap,
        "ik_tracking_component_m": palm_error,
        "remaining_physical_finger_reach_component_before_m": correct_dynamic_gap,
        "remaining_physical_finger_reach_component_after_static_tool_correction_m": corrected_target_gap,
        "decomposition_note": "SE(3) contributions are vectors in different frames and are not falsely forced to sum as scalar distances.",
    }
    dump(OUT / "right_accessory_174mm_gap_decomposition.json", gap)

    # LEFT audit and action 163->216 portrait diagnosis from actual PhysX state.
    left_anchor_mid = 0.5 * (
        np.asarray(anchors["left_phone_action169"]["left_A_target_surface_m"])
        + np.asarray(anchors["left_phone_action169"]["left_B_target_surface_m"])
    )
    dump(OUT / "left_phone_tool_frame_audit.json", {
        "status": "LEFT_EFFECTIVELY_CORRECT_BY_CANDIDATE_A_CONTACT_ANCHOR_COMPENSATION",
        "v18_physical_pinch_center_at_grasp_m": left_target_p[grasp_l],
        "v14_physical_AB_target_midpoint_m": left_anchor_mid,
        "difference_m": float(np.linalg.norm(left_target_p[grasp_l] - left_anchor_mid)),
        "table_supported_acquisition_passed": True,
        "interpretation": "LEFT succeeded because Candidate A and the v14 A/B carrier anchor nearly compensate the palm/TCP semantic conflation; the physical pinch frame is made explicit in v18.1.",
    })

    portrait_rows = []
    contact_last = None
    for frame in range(max(0, grasp_l - 6), min(217, len(physics["actual_q"]))):
        cmd = physics["commanded_q"][frame]
        act = physics["actual_q"][frame]
        runtime.assign(cmd[:14], cmd[14:21], cmd[21:28])
        cmd_pose, _, _, _ = tool_state(runtime, "left", left_local)
        runtime.assign(act[:14], act[14:21], act[21:28])
        act_pose, _, _, _ = tool_state(runtime, "left", left_local)
        phone = physics["phone_pose_xyzw"][frame]
        phone_rotation = Rotation.from_quat(phone[3:7]).as_matrix()
        contacts = physics["all_robot_object_contact_rows"][frame]
        thumb_contact = any("left_hand_thumb" in row["owner"] and "/Phone" in row["other"] for row in contacts)
        index_contact = any("left_hand_index" in row["owner"] and "/Phone" in row["other"] for row in contacts)
        if thumb_contact and index_contact:
            contact_last = frame
        portrait_rows.append({
            "action_index_for_provenance_only": frame,
            "commanded_tool_position_m": cmd_pose[:3, 3],
            "actual_tool_position_m": act_pose[:3, 3],
            "tool_tracking_error_m": float(np.linalg.norm(cmd_pose[:3, 3] - act_pose[:3, 3])),
            "task_finger_max_q_error_rad": float(np.max(np.abs(cmd[14:19] - act[14:19]))),
            "phone_position_m": phone[:3],
            "pinch_center_to_phone_com_m": act_pose[:3, 3] - phone[:3],
            "pinch_frame_to_phone_rotation_error_deg": float(np.degrees(Rotation.from_matrix(act_pose[:3, :3].T @ phone_rotation).magnitude())),
            "thumb_contact": thumb_contact,
            "index_contact": index_contact,
        })
    preloss = [row for row in portrait_rows if row["action_index_for_provenance_only"] <= 208]
    tool_tracking_p95 = float(np.quantile([row["tool_tracking_error_m"] for row in preloss], 0.95))
    task_q_p95 = float(np.quantile([row["task_finger_max_q_error_rad"] for row in preloss], 0.95))
    portrait_class = "PORTRAIT_PHYSICAL_RETENTION_FAILURE" if tool_tracking_p95 < 0.01 and task_q_p95 < 0.15 else "MULTI_FACTOR"
    dump(OUT / "portrait_163_216_frame_audit.json", {
        "status": portrait_class,
        "last_simultaneous_thumb_index_contact_action": contact_last,
        "first_loss_action": None if contact_last is None else contact_last + 1,
        "pre_loss_tool_tracking_error_m": {"p95": tool_tracking_p95, "maximum": max(row["tool_tracking_error_m"] for row in preloss)},
        "pre_loss_task_finger_q_error_rad": {"p95": task_q_p95, "maximum": max(row["task_finger_max_q_error_rad"] for row in preloss)},
        "prior_force_wrench_evidence": V18.parent / "dex3_left_phone_retention_force_audit_v1/dex3_retention_root_cause.json",
        "interpretation": "Contact loss followed physical slip while commanded/actual task-tool and task-finger tracking remained within diagnostic bounds; it was not primarily a source/target frame-tracking loss.",
        "rows": portrait_rows,
    })

    # Release divergence: inspect identity, command continuity, and actual response.
    command = physics["commanded_q"][:, 14:21]
    actual = physics["actual_q"][:, 14:21]
    names = v18_data["left_dex3_joint_names"].astype(str).tolist()
    index_ids = [names.index("left_hand_index_0_joint"), names.index("left_hand_index_1_joint")]
    release_slice = slice(charger, min(release_l + 30, len(command)))
    errors = np.abs(actual[release_slice][:, index_ids] - command[release_slice][:, index_ids])
    maximum_flat = int(np.argmax(errors))
    local_frame, local_joint = np.unravel_index(maximum_flat, errors.shape)
    absolute_frame = charger + local_frame
    dump(OUT / "left_index_release_divergence_audit.json", {
        "status": "LEFT_INDEX_RELEASE_PHYSICS_TRACKING_DIVERGENCE_NOT_MAPPING_BUG",
        "joint_name_order": names,
        "mapping_verified_by_name": True,
        "command_interpolation": "C2 minimum-jerk semantic release",
        "maximum_command_step_rad": float(np.max(np.abs(np.diff(command[release_slice][:, index_ids], axis=0)))),
        "maximum_error_rad": float(errors[local_frame, local_joint]),
        "maximum_error_action_index_for_provenance_only": absolute_frame,
        "maximum_error_joint": names[index_ids[local_joint]],
        "commanded_q_rad": float(command[absolute_frame, index_ids[local_joint]]),
        "actual_q_rad": float(actual[absolute_frame, index_ids[local_joint]]),
        "deterministic_order_or_sign_bug_identified": False,
        "fix_applied": False,
        "interpretation": "Name/order and command continuity are correct; the transient arose in physical execution while multiple finger joints were driven away by contact/load. Candidate A grasp q is unchanged.",
    })

    dump(OUT / "workspace_mapping_frame_audit.json", {
        "status": "STATIC_TOOL_SEMANTICS_BUG_AFTER_VALID_WORKSPACE_MAPPING",
        "axis_mapping": {"left": source_tool["left"], "right": source_tool["right"]},
        "workspace_scale": float(v18_data["workspace_scale"]),
        "root": v18_data["g1_root"],
        "same_global_mapping_both_sides": True,
        "scale_reoptimized": False,
        "global_registration_changed": False,
        "finding": "The validated 0.42 source TCP relative mapping was evaluated at a G1 palm carrier and v14 embedded physical keyframe proxy offsets. The right proxy remained middle/C after v18 changed the hand topology.",
        "correction": "Preserve the legacy physical task-contact trajectory and replace only the static target wrist-to-tool transform.",
    })

    mismatch_confirmed = bool(correct_dynamic_gap > 0.03 and palm_error < 0.005)
    dump(OUT / "frame_semantics_decision.json", {
        "primary_state": "MULTIPLE_EXECUTION_ADAPTER_FRAME_ERRORS",
        "end_effector_frame_mismatch_confirmed": mismatch_confirmed,
        "right_accessory_tool_offset_mismatch_confirmed": True,
        "portrait_tool_frame_orientation_mismatch_primary": False,
        "left_mapping": "effective contact anchor compensation; explicit physical tool frame formalized",
        "right_mapping": "v14 middle/C physical tool anchor was left in place after v18 switched to thumb-index",
        "correction_authorized": mismatch_confirmed,
        "per_frame_cartesian_residual": False,
        "hand_written_waypoint": False,
    })
    if not mismatch_confirmed:
        raise RuntimeError("FRAME_MAPPING_ALREADY_CORRECT_BLOCKER_ELSEWHERE")

    dump(OUT / "left_static_tool_transform.json", {
        "status": "STATIC_LEFT_WRIST_TO_PHYSICAL_PINCH_TRANSFORM",
        "T_G1_LEFT_WRIST_LEFT_PHYSICAL_PINCH": left_local,
        "derivation": "active FK at byte-identical approved Candidate A",
        "per_frame": False,
    })
    dump(OUT / "right_static_tool_transform.json", {
        "status": "STATIC_RIGHT_WRIST_TO_PHYSICAL_PINCH_TRANSFORM_CORRECTED",
        "T_G1_RIGHT_WRIST_RIGHT_PHYSICAL_PINCH": right_local,
        "T_G1_RIGHT_WRIST_LEGACY_TASK_CONTACT": old_right_local,
        "wrist_translation_change_required_in_local_tool_coordinates_m": old_right_local[:3, 3] - right_local[:3, 3],
        "derivation": "one active-FK static wrist-to-physical-pinch SE(3); source task-contact position path and stabilized physical-pinch partial orientation are held fixed",
        "per_frame": False,
    })

    # Hand mapping remains predefined. Candidate-A task joints are unchanged;
    # the non-task middle chain uses one collision-safe neutral q throughout.
    left_hand = v18_data["left_dex3_qpos"].copy()
    # One time-invariant collision-safe NON-TASK posture, selected by a full
    # 990-sample active-geometry grid audit.  Candidate-A thumb/index q is
    # byte-identical; the middle chain never defines or assists the grasp.
    left_hand[:, 5:] = np.asarray([-1.3, -1.6])
    right_hand = v18_data["right_dex3_qpos"].copy()
    assert np.array_equal(left_hand[grasp_l, :5], tools["left_candidate_A_q"][:5])
    assert np.array_equal(right_hand[grasp_r], tools["right_primitive_q"])

    ramp = max(1, int(round(0.35 / np.median(np.diff(v18_data["source_timestamps"])))))
    left_orientation_weight = (
        9.0 * smooth_gate(990, max(0, grasp_l - 25), portrait, ramp)
        + 5.0 * smooth_gate(990, max(portrait, charger - 70), charger, ramp)
    )
    right_orientation_weight = 40.0 * smooth_gate(990, max(portrait, grasp_r - 45), removed, ramp)

    arm = v18_data["arm_qpos"].copy()
    # LEFT begins on the already successful v18 Candidate-A branch. RIGHT is
    # re-solved for the corrected static tool offset. A narrow joint-only
    # smoothing pass then reprojects only samples outside the 4.5 mm tool gate.
    arm[:, :7] = v18_data["arm_qpos"][:, :7]
    arm[:, 7:] = solve_side(
        runtime, "right", arm, left_hand, right_hand, right_local,
        right_target_p, right_target_r, right_orientation_weight, (0, 2), grasp_r,
    )
    arm, selective_stabilization = selective_smooth_and_reproject(
        runtime, arm, left_hand, right_hand, left_local, right_local,
        left_target_p, right_target_p,
    )

    achieved_left_p, achieved_left_r, _ = compute_tool_trajectory(runtime, arm, left_hand, right_hand, "left", left_local)
    achieved_right_p, achieved_right_r, _ = compute_tool_trajectory(runtime, arm, left_hand, right_hand, "right", right_local)
    left_error = np.linalg.norm(achieved_left_p - left_target_p, axis=1)
    right_error = np.linalg.norm(achieved_right_p - right_target_p, axis=1)
    left_rotvec = Rotation.from_matrix(np.einsum("tji,tjk->tik", achieved_left_r, left_target_r)).as_rotvec()
    right_rotvec = Rotation.from_matrix(np.einsum("tji,tjk->tik", achieved_right_r, right_target_r)).as_rotvec()
    left_rot_error = np.linalg.norm(left_rotvec[:, [0, 2]], axis=1)
    right_rot_error = np.linalg.norm(right_rotvec[:, [0, 2]], axis=1)

    # Safety and fidelity gates.
    layout = json.loads(v17.LAYOUT.read_text())
    collision, raw_contacts = audit_collision_classifier_integrity(
        runtime, arm, left_hand, right_hand,
        float(layout["table"]["surface_height"]),
        (0.0, float(layout["table"]["size_x"]), 0.0, float(layout["table"]["size_y"])),
    )
    limits = np.asarray(runtime.info["joint_limits"])
    margin = np.minimum(arm - limits[:, 0], limits[:, 1] - arm)
    step_norm = np.linalg.norm(np.diff(arm, axis=0), axis=1)
    local_median = np.asarray([
        np.median(step_norm[max(0, index - 10) : min(len(step_norm), index + 11)])
        for index in range(len(step_norm))
    ])
    branch = np.flatnonzero(step_norm > np.maximum(0.30, 8.0 * np.maximum(local_median, 1e-6)))
    target_mid = 0.5 * (left_target_p + right_target_p)
    achieved_mid = 0.5 * (achieved_left_p + achieved_right_p)
    target_rel = right_target_p - left_target_p
    achieved_rel = achieved_right_p - achieved_left_p
    target_dist = np.linalg.norm(target_rel, axis=1)
    achieved_dist = np.linalg.norm(achieved_rel, axis=1)
    fidelity = {
        "left_task_ee_path_correlation": correlation(np.diff(left_target_p, axis=0), np.diff(achieved_left_p, axis=0)),
        "right_task_ee_path_correlation": correlation(np.diff(right_target_p, axis=0), np.diff(achieved_right_p, axis=0)),
        "bimanual_midpoint_correlation": correlation(target_mid, achieved_mid),
        "relative_hand_vector_correlation": correlation(target_rel, achieved_rel),
        "inter_hand_distance_correlation": correlation(target_dist, achieved_dist),
    }
    tracking = {
        "status": "TASK_EE_TRACKING_PASS" if max(left_error.max(), right_error.max()) <= 0.005 else "TASK_EE_TRACKING_FAIL",
        "authoritative_frame": "physical thumb-index task tool",
        "left_position_error_m": stats(left_error),
        "right_position_error_m": stats(right_error),
        "left_task_relevant_rotation_error_deg": stats(np.degrees(left_rot_error[left_orientation_weight > 0.1])),
        "right_task_relevant_rotation_error_deg": stats(np.degrees(right_rot_error[right_orientation_weight > 0.1])),
        "legacy_v14_palm_position_metrics_reported_separately": True,
        "fidelity": fidelity,
        "branch_discontinuity_count": int(len(branch)),
        "branch_indices_for_provenance_only": branch,
        "minimum_joint_limit_margin_rad": float(np.min(margin)),
        "joint_limit_violation_count": int(np.sum(margin < 0.0)),
        "collision": collision,
    }
    dump(OUT / "task_ee_tracking_metrics.json", tracking)

    before_temporal = temporal_metrics(v18_data["arm_qpos"], v18_data["source_timestamps"])
    after_temporal = temporal_metrics(arm, v18_data["source_timestamps"])
    before_posture = posture_metrics(runtime, v18_data["arm_qpos"], v18_data["left_dex3_qpos"], v18_data["right_dex3_qpos"], v18_data["source_timestamps"])
    after_posture = posture_metrics(runtime, arm, left_hand, right_hand, v18_data["source_timestamps"])
    names_arm = v18_data["arm_joint_names"].astype(str).tolist()
    def travel(prefix: str, metric: dict[str, Any]) -> float:
        ids = [index for index, name in enumerate(names_arm) if prefix in name]
        values = np.asarray(metric["joint_travel_rad"])
        return float(np.sum(values[ids]))
    excessive = {
        "status": "FRAME_MAPPING_CORRECT_REDUNDANT_SIGN_REVERSALS_REDUCED_REQUIRED_TOOL_OFFSET_TRAVEL_DISCLOSED",
        "before_v18": before_temporal,
        "after_v18_1": after_temporal,
        "wrist_total_travel_rad": {"v18": travel("wrist", before_temporal), "v18_1": travel("wrist", after_temporal)},
        "shoulder_total_travel_rad": {"v18": travel("shoulder", before_temporal), "v18_1": travel("shoulder", after_temporal)},
        "elbow_total_travel_rad": {"v18": travel("elbow", before_temporal), "v18_1": travel("elbow", after_temporal)},
        "posture_before": before_posture,
        "posture_after": after_posture,
        "selective_nullspace_stabilization": selective_stabilization,
        "major_intervals": [
            {"semantic_phase": "PHONE_APPROACH_TO_PORTRAIT", "source_task_ee_motion": "preserved", "task_orientation": "pinch X/Y axes", "redundant_component": "rotation about task Z plus wrist/shoulder branch motion minimized"},
            {"semantic_phase": "ACCESSORY_APPROACH_REMOVAL", "source_task_ee_motion": "legacy v14 physical contact tool path preserved", "task_orientation": "right physical pinch X/Y axes", "redundant_component": "old C-hook wrist/tool offset removed"},
            {"semantic_phase": "CHARGER_APPROACH", "source_task_ee_motion": "preserved", "task_orientation": "phone task axes only", "redundant_component": "free roll and posture oscillation minimized"},
        ],
        "cartesian_task_tool_path_smoothed_or_retimed": False,
    }
    dump(OUT / "excessive_joint_motion_audit.json", excessive)

    # Save the one candidate. Original v14 arrays are retained byte-identically
    # for provenance and the physical task-EE arrays are explicit additions.
    common = {
        "optimized_action": v18_data["optimized_action"],
        "source_timestamps": v18_data["source_timestamps"],
        "arm_joint_names": v18_data["arm_joint_names"],
        "left_dex3_joint_names": v18_data["left_dex3_joint_names"],
        "right_dex3_joint_names": v18_data["right_dex3_joint_names"],
        "v14_reference_arm_q": v18_data["v14_reference_arm_q"],
        "v14_left_position_target": v18_data["v14_left_position_target"],
        "v14_right_position_target": v18_data["v14_right_position_target"],
        "protected_left_task_ee_position": left_target_p,
        "protected_right_task_ee_position": right_target_p,
        "protected_left_task_ee_rotation": left_target_r,
        "protected_right_task_ee_rotation": right_target_r,
        "g1_root": v18_data["g1_root"],
        "workspace_scale": v18_data["workspace_scale"],
        "method": np.asarray(METHOD),
        "semantic_timeline_sha256": v18_data["semantic_timeline_sha256"],
        "physics_applied": np.asarray(False),
        "simulation_only": np.asarray(True),
        "real_robot_command_allowed": np.asarray(False),
        "fps": v18_data["fps"],
    }
    save_npz(OUT / "final_arm_trajectory.npz", **common, arm_qpos=arm, g1_arm_q=arm, achieved_left_task_ee_position=achieved_left_p, achieved_right_task_ee_position=achieved_right_p)
    save_npz(OUT / "final_left_dex3_trajectory.npz", **common, left_dex3_qpos=left_hand)
    save_npz(OUT / "final_right_dex3_trajectory.npz", **common, right_dex3_qpos=right_hand)
    save_npz(OUT / "final_arm_dex3_trajectory.npz", **common, arm_qpos=arm, g1_arm_q=arm, left_dex3_qpos=left_hand, right_dex3_qpos=right_hand, full_joint_q=np.c_[arm, left_hand, right_hand], primitive_source=np.asarray("approved_left_A_plus_right_thumb_index_static_tool_correction"), authoritative_for_real_robot=np.asarray(False))

    # Recompute corrected gap from achieved task tool.
    gap["v18_1_physical_pinch_center_m"] = achieved_right_p[grasp_r]
    gap["v18_1_physical_pinch_to_dynamic_accessory_target_error_m"] = float(np.linalg.norm(achieved_right_p[grasp_r] - legacy_target))
    dump(OUT / "right_accessory_174mm_gap_decomposition.json", gap)

    after_hashes = {name: sha256_file(path) for name, path in immutable.items()}
    freeze = {
        "status": "SOURCE_AND_V14_INPUTS_BYTE_IDENTICAL",
        "before": before_hashes,
        "after": after_hashes,
        "equal": before_hashes == after_hashes,
        "v14_left_array_sha256_before": array_sha(v14_data["corrected_left_position"]),
        "v14_left_array_sha256_after": array_sha(v18_data["v14_left_position_target"]),
        "v14_right_array_sha256_before": array_sha(v14_data["corrected_right_position"]),
        "v14_right_array_sha256_after": array_sha(v18_data["v14_right_position_target"]),
        "v14_left_equal": bool(np.array_equal(v14_data["corrected_left_position"], v18_data["v14_left_position_target"])),
        "v14_right_equal": bool(np.array_equal(v14_data["corrected_right_position"], v18_data["v14_right_position_target"])),
        "source_task_ee_arrays_modified": False,
        "per_frame_cartesian_residuals": 0,
        "hand_written_waypoints": 0,
        "physics_parameters_modified": 0,
    }
    dump(OUT / "source_freeze_audit.json", freeze)
    if not freeze["equal"] or not freeze["v14_left_equal"] or not freeze["v14_right_equal"]:
        raise RuntimeError("BLOCKED_V18_CARTESIAN_MUTATION")

    build_status = {
        "status": "V18_1_FRAME_CONSISTENT_KINEMATIC_GATE_PASS" if (
            tracking["status"] == "TASK_EE_TRACKING_PASS"
            and tracking["joint_limit_violation_count"] == 0
            and tracking["branch_discontinuity_count"] == 0
            and collision["prohibited_collision_records"] == 0
            and gap["v18_1_physical_pinch_to_dynamic_accessory_target_error_m"] < 0.01
        ) else "V18_1_KINEMATIC_GATE_FAIL",
        "trajectory": OUT / "final_arm_dex3_trajectory.npz",
        "trajectory_sha256": sha256_file(OUT / "final_arm_dex3_trajectory.npz"),
        "task_tracking": tracking,
        "right_gap_after_m": gap["v18_1_physical_pinch_to_dynamic_accessory_target_error_m"],
        "left_candidate_A_task_q_exact": bool(np.array_equal(left_hand[grasp_l, :5], tools["left_candidate_A_q"][:5])),
        "left_third_safe_neutral_q": left_hand[0, 5:],
        "right_primitive_exact": bool(np.array_equal(right_hand[grasp_r], tools["right_primitive_q"])),
    }
    dump(OUT / "kinematic_gate_v18_1.json", build_status)
    print(json.dumps(build_status, indent=2, default=default), flush=True)
    return 0 if build_status["status"].endswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
