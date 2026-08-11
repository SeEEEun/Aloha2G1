"""Semantic-driven wrist-orientation and continuous Dex3 integration.

All phase boundaries are resolved through :class:`SemanticTimeline`.  This
module intentionally has no episode-specific frame constants or trajectory
length assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from aloha_magsafe_semantics.schema import SemanticTimeline

from .kinematics import ActiveG1Dex3, normalize, ring_material_gap, transform


C_LEFT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
C_RIGHT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def interpolate_rotation(start: np.ndarray, end: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    key = Rotation.from_matrix(np.stack((start, end)))
    return Slerp([0.0, 1.0], key)(np.clip(values, 0.0, 1.0)).as_matrix()


def mapped_relative(source: np.ndarray, start: int, end: int, calibration: np.ndarray) -> np.ndarray:
    segment = np.asarray(source[start : end + 1], dtype=np.float64)
    delta = np.einsum("ji,tjk->tik", segment[0], segment)
    return np.einsum("ij,tjk,kl->til", calibration.T, delta, calibration)


def rotation_arc_progress(source: np.ndarray, start: int, end: int) -> np.ndarray:
    segment = np.asarray(source[start : end + 1], dtype=np.float64)
    if len(segment) == 1:
        return np.ones(1)
    steps = Rotation.from_matrix(np.einsum("tji,tjk->tik", segment[:-1], segment[1:])).magnitude()
    cumulative = np.r_[0.0, np.cumsum(np.abs(steps))]
    if cumulative[-1] <= np.finfo(np.float64).eps:
        return np.linspace(0.0, 1.0, len(segment))
    return cumulative / cumulative[-1]


def source_preserving_segment(
    source: np.ndarray,
    start: int,
    end: int,
    calibration: np.ndarray,
    start_target: np.ndarray,
    end_target: np.ndarray | None,
) -> np.ndarray:
    relative = mapped_relative(source, start, end, calibration)
    base = np.einsum("ij,tjk->tik", start_target, relative)
    if end_target is None:
        return base
    correction = end_target @ base[-1].T
    progress = rotation_arc_progress(source, start, end)
    world_correction = interpolate_rotation(np.eye(3), correction, progress)
    return np.einsum("tij,tjk->tik", world_correction, base)


def semantic_rotation_path(
    source: np.ndarray,
    calibration: np.ndarray,
    initial: np.ndarray,
    segments: list[tuple[int, int, np.ndarray | None]],
    length: int,
) -> np.ndarray:
    output = np.repeat(initial[None], length, axis=0)
    current = initial
    for start, end, endpoint in segments:
        if end < start:
            raise ValueError("semantic orientation segment is reversed")
        values = source_preserving_segment(source, start, end, calibration, current, endpoint)
        output[start : end + 1] = values
        current = values[-1]
    final_end = segments[-1][1] if segments else 0
    output[final_end:] = current
    return output


def compose_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return transform(np.asarray(rotation, dtype=np.float64), np.asarray(translation, dtype=np.float64))


def inverse_pose(value: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -value[:3, :3].T @ value[:3, 3]
    return result


@dataclass
class LeftCandidate:
    assignment: str
    stage: str
    arm_q: np.ndarray
    finger_q: np.ndarray
    wrist_pose: np.ndarray
    palm_position: np.ndarray
    contact_positions: dict[str, np.ndarray]
    target_positions: dict[str, np.ndarray]
    gaps_m: dict[str, float]
    normal_alignment: dict[str, float]
    position_error_m: float
    joint_margin_rad: float
    score: float
    valid: bool

    def record(self) -> dict[str, Any]:
        return {
            "assignment": self.assignment,
            "stage": self.stage,
            "arm_q_rad": self.arm_q,
            "finger_q_rad": self.finger_q,
            "wrist_pose": self.wrist_pose,
            "palm_position_m": self.palm_position,
            "contact_positions_m": self.contact_positions,
            "target_positions_m": self.target_positions,
            "gaps_m": self.gaps_m,
            "normal_alignment": self.normal_alignment,
            "position_error_m": self.position_error_m,
            "joint_margin_rad": self.joint_margin_rad,
            "score": self.score,
            "valid": self.valid,
        }


def _side_arm(base: np.ndarray, side: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(base, dtype=np.float64).copy()
    result[:7 if side == "left" else 0] = result[:7 if side == "left" else 0]
    block = slice(0, 7) if side == "left" else slice(7, 14)
    result[block] = value
    return result


def search_left_opposed_phone_contact(
    runtime: ActiveG1Dex3,
    base_arm_q: np.ndarray,
    palm_target: np.ndarray,
    phone_pose: np.ndarray,
    phone_dimensions: np.ndarray,
    assignment: str,
    stage: str,
    finger_seed: np.ndarray | None = None,
    *,
    random_seed: int = 0,
) -> LeftCandidate:
    """Search full left arm + A/B FK for opposite phone surfaces."""
    if assignment not in ("A_SCREEN_B_BACK", "A_BACK_B_SCREEN"):
        raise ValueError(assignment)
    arm_base = np.asarray(base_arm_q, dtype=np.float64)
    palm_target = np.asarray(palm_target, dtype=np.float64)
    half = 0.5 * np.asarray(phone_dimensions, dtype=np.float64)
    a_sign = -1.0 if assignment == "A_SCREEN_B_BACK" else 1.0
    b_sign = -a_sign
    a_limits = runtime.contacts["left_A"].limits
    b_limits = runtime.contacts["left_B"].limits
    limits = np.vstack((runtime.info["joint_limits"][:7], a_limits, b_limits))
    contact_lower = np.array([-half[0] + 0.001, -half[2] + 0.002] * 2)
    contact_upper = np.array([-half[0] + 0.045, half[2] - 0.002] * 2)
    lower = np.r_[limits[:, 0], contact_lower]
    upper = np.r_[limits[:, 1], contact_upper]
    if finger_seed is None:
        finger_seed = np.r_[runtime.open_hand_q["left"][:5]]
    seeds = [np.r_[
        arm_base[:7], finger_seed,
        -half[0] + 0.012, 0.018,
        -half[0] + 0.012, -0.018,
    ]]
    rng = np.random.default_rng(random_seed)
    for _ in range(7):
        arm = np.clip(
            arm_base[:7] + rng.normal(0.0, 0.16, 7),
            limits[:7, 0] + 1e-6,
            limits[:7, 1] - 1e-6,
        )
        fingers = limits[7:, 0] + rng.random(5) * (limits[7:, 1] - limits[7:, 0])
        seeds.append(np.r_[arm, fingers, rng.uniform(contact_lower, contact_upper)])

    best: LeftCandidate | None = None
    for seed in seeds:
        seed = np.clip(seed, lower + 1e-8, upper - 1e-8)

        def state(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
            arm = _side_arm(arm_base, "left", value[:7])
            hand = runtime.open_hand_q["left"].copy()
            hand[:5] = value[7:12]
            runtime.assign(arm, hand, runtime.open_hand_q["right"])
            palm = runtime.palm_pose("left")
            return arm, hand, {
                "A": runtime.contact_pose("left_A"),
                "B": runtime.contact_pose("left_B"),
                "palm": (palm[:3, 3], palm[:3, 2]),
            }

        def targets(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_a, z_a, x_b, z_b = value[-4:]
            ta = phone_pose[:3, 3] + phone_pose[:3, :3] @ np.array([x_a, a_sign * half[1], z_a])
            tb = phone_pose[:3, 3] + phone_pose[:3, :3] @ np.array([x_b, b_sign * half[1], z_b])
            return ta, tb

        def residual(value: np.ndarray) -> np.ndarray:
            _, _, points = state(value)
            ta, tb = targets(value)
            desired_a_normal = -a_sign * phone_pose[:3, 1]
            desired_b_normal = -b_sign * phone_pose[:3, 1]
            return np.r_[
                1200.0 * (points["palm"][0] - palm_target),
                320.0 * (points["A"][0] - ta),
                320.0 * (points["B"][0] - tb),
                0.8 * (points["A"][1] - desired_a_normal),
                0.8 * (points["B"][1] - desired_b_normal),
                0.025 * (value[:7] - arm_base[:7]),
                0.006 * (value[7:12] - finger_seed),
            ]

        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=350,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        arm, hand, points = state(solution.x)
        ta, tb = targets(solution.x)
        gaps = {"A": float(np.linalg.norm(points["A"][0] - ta)), "B": float(np.linalg.norm(points["B"][0] - tb))}
        alignments = {
            "A": float(np.dot(points["A"][1], -a_sign * phone_pose[:3, 1])),
            "B": float(np.dot(points["B"][1], -b_sign * phone_pose[:3, 1])),
        }
        position_error = float(np.linalg.norm(points["palm"][0] - palm_target))
        margin = runtime.arm_joint_margin(arm)
        score = 1000.0 * max(gaps.values()) + 500.0 * position_error + 0.03 * float(np.linalg.norm(arm[:7] - arm_base[:7]))
        valid = bool(
            max(gaps.values()) <= 0.005
            and position_error <= 0.005
            and min(alignments.values()) >= 0.5
            and margin >= -1e-8
        )
        wrist = runtime.wrist_pose("left")
        candidate = LeftCandidate(
            assignment=assignment,
            stage=stage,
            arm_q=arm,
            finger_q=hand,
            wrist_pose=wrist,
            palm_position=points["palm"][0],
            contact_positions={"A": points["A"][0], "B": points["B"][0]},
            target_positions={"A": ta, "B": tb},
            gaps_m=gaps,
            normal_alignment=alignments,
            position_error_m=position_error,
            joint_margin_rad=margin,
            score=score,
            valid=valid,
        )
        if best is None or candidate.score < best.score:
            best = candidate
    if best is None:
        raise RuntimeError("left candidate search produced no result")
    return best


@dataclass
class RightCandidate:
    family: str
    arm_q: np.ndarray
    hand_q: np.ndarray
    wrist_pose: np.ndarray
    palm_position: np.ndarray
    contact_position: np.ndarray
    ring_target: np.ndarray
    gap_m: float
    insertion_alignment: float
    position_error_m: float
    joint_margin_rad: float
    ring_angle_deg: float
    score: float
    valid: bool

    def record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "arm_q_rad": self.arm_q,
            "hand_q_rad": self.hand_q,
            "wrist_pose": self.wrist_pose,
            "palm_position_m": self.palm_position,
            "C_contact_position_m": self.contact_position,
            "ring_target_m": self.ring_target,
            "C_ring_gap_m": self.gap_m,
            "insertion_alignment": self.insertion_alignment,
            "position_error_m": self.position_error_m,
            "joint_margin_rad": self.joint_margin_rad,
            "ring_angle_deg": self.ring_angle_deg,
            "score": self.score,
            "valid": self.valid,
        }


def search_right_ring_contact(
    runtime: ActiveG1Dex3,
    base_arm_q: np.ndarray,
    palm_target: np.ndarray,
    accessory_pose: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    depth: float,
    family: str,
    source_approach: np.ndarray,
    *,
    random_seed: int = 0,
) -> RightCandidate:
    if family not in ("AXIAL_APERTURE_INSERTION", "RADIAL_GAP_INSERTION"):
        raise ValueError(family)
    arm_base = np.asarray(base_arm_q, dtype=np.float64)
    c_limits = runtime.contacts["right_C"].limits
    limits = np.vstack((runtime.info["joint_limits"][7:], c_limits))
    lower, upper = limits[:, 0], limits[:, 1]
    angles = np.linspace(-math.pi, math.pi, 13)[:-1]
    if family == "RADIAL_GAP_INSERTION":
        # The active asset gap is centered on its local +X direction.
        angles = np.radians(np.linspace(-15.0, 15.0, 5))
    rng = np.random.default_rng(random_seed)
    best: RightCandidate | None = None
    for angle in angles:
        radius = inner_radius + 0.0005
        local_target = np.array([radius * math.cos(angle), -0.5 * depth, radius * math.sin(angle)])
        target = accessory_pose[:3, 3] + accessory_pose[:3, :3] @ local_target
        if family == "AXIAL_APERTURE_INSERTION":
            desired_direction = normalize(accessory_pose[:3, 1])
        else:
            desired_direction = normalize(accessory_pose[:3, :3] @ np.array([-math.cos(angle), 0.0, -math.sin(angle)]))
        seeds = [np.r_[arm_base[7:], runtime.open_hand_q["right"][-2:]]]
        for _ in range(4):
            seeds.append(np.r_[
                np.clip(arm_base[7:] + rng.normal(0.0, 0.15, 7), lower[:7], upper[:7]),
                lower[7:] + rng.random(2) * (upper[7:] - lower[7:]),
            ])
        for seed in seeds:
            seed = np.clip(seed, lower + 1e-8, upper - 1e-8)

            def state(value: np.ndarray):
                arm = _side_arm(arm_base, "right", value[:7])
                hand = runtime.open_hand_q["right"].copy()
                hand[-2:] = value[7:]
                runtime.assign(arm, runtime.open_hand_q["left"], hand)
                palm = runtime.palm_pose("right")[:3, 3]
                contact, normal = runtime.contact_pose("right_C")
                return arm, hand, palm, contact, normal

            def residual(value: np.ndarray) -> np.ndarray:
                _, _, palm, contact, normal = state(value)
                return np.r_[
                    1200.0 * (palm - palm_target),
                    350.0 * (contact - target),
                    0.9 * (normal - desired_direction),
                    0.025 * (value[:7] - arm_base[7:]),
                    0.006 * (value[7:] - runtime.open_hand_q["right"][-2:]),
                ]

            solution = least_squares(
                residual,
                seed,
                bounds=(lower, upper),
                max_nfev=320,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            arm, hand, palm, contact, normal = state(solution.x)
            gap = float(np.linalg.norm(contact - target))
            alignment = float(np.dot(normal, desired_direction))
            approach_alignment = float(np.dot(normalize(runtime.wrist_pose("right")[:3, 0]), normalize(source_approach)))
            position_error = float(np.linalg.norm(palm - palm_target))
            margin = runtime.arm_joint_margin(arm)
            score = 1000.0 * gap + 500.0 * position_error + 0.2 * (1.0 - alignment) + 0.08 * (1.0 - approach_alignment)
            valid = bool(gap <= 0.005 and position_error <= 0.005 and alignment >= 0.75 and margin >= -1e-8)
            candidate = RightCandidate(
                family=family,
                arm_q=arm,
                hand_q=hand,
                wrist_pose=runtime.wrist_pose("right"),
                palm_position=palm,
                contact_position=contact,
                ring_target=target,
                gap_m=gap,
                insertion_alignment=alignment,
                position_error_m=position_error,
                joint_margin_rad=margin,
                ring_angle_deg=float(np.degrees(angle)),
                score=score,
                valid=valid,
            )
            if best is None or candidate.score < best.score:
                best = candidate
    if best is None:
        raise RuntimeError("right candidate search produced no result")
    return best


def solve_arm_orientation_trajectory(
    runtime: ActiveG1Dex3,
    initial_q: np.ndarray,
    position_left: np.ndarray,
    position_right: np.ndarray,
    rotation_left: np.ndarray,
    rotation_right: np.ndarray,
    *,
    orientation_gain: float,
    posture_gain: float,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Task-Jacobian solve with position correction preceding orientation."""
    q = np.asarray(initial_q, dtype=np.float64).copy()
    nominal = np.asarray(initial_q, dtype=np.float64)
    limits = np.asarray(runtime.info["joint_limits"], dtype=np.float64)
    for frame in range(len(q)):
        for side, target_p, target_r, block in (
            ("left", position_left[frame], rotation_left[frame], slice(0, 7)),
            ("right", position_right[frame], rotation_right[frame], slice(7, 14)),
        ):
            initial = q[frame, block].copy()
            previous = q[frame - 1, block].copy() if frame else initial.copy()
            whole = q[frame].copy()
            center = 0.5 * (limits[block, 0] + limits[block, 1])
            orientation_weight = 4.0 + 8.0 * orientation_gain

            def residual(value: np.ndarray) -> np.ndarray:
                whole[block] = value
                runtime.assign(whole, runtime.open_hand_q["left"], runtime.open_hand_q["right"])
                position, rotation, _ = runtime.palm_state(side)
                return np.r_[
                    2500.0 * (position - target_p),
                    orientation_weight * rotation_error(rotation, target_r),
                    0.025 * (value - initial),
                    0.008 * (value - previous),
                    posture_gain * (value - center),
                ]

            solution = least_squares(
                residual,
                np.clip(initial, limits[block, 0] + 1e-8, limits[block, 1] - 1e-8),
                bounds=(limits[block, 0] + 1e-8, limits[block, 1] - 1e-8),
                jac="2-point",
                max_nfev=max(20, iterations * 3),
                ftol=2e-9,
                xtol=2e-9,
                gtol=2e-9,
                x_scale="jac",
            )
            q[frame, block] = solution.x
    achieved_left_p = np.empty_like(position_left)
    achieved_right_p = np.empty_like(position_right)
    achieved_left_r = np.empty_like(rotation_left)
    achieved_right_r = np.empty_like(rotation_right)
    for frame, value in enumerate(q):
        runtime.assign(value, runtime.open_hand_q["left"], runtime.open_hand_q["right"])
        achieved_left_p[frame], achieved_left_r[frame], _ = runtime.palm_state("left")
        achieved_right_p[frame], achieved_right_r[frame], _ = runtime.palm_state("right")
    left_error = np.linalg.norm(achieved_left_p - position_left, axis=1)
    right_error = np.linalg.norm(achieved_right_p - position_right, axis=1)
    left_rotation_error = Rotation.from_matrix(
        np.einsum("tij,tkj->tik", rotation_left, achieved_left_r)
    ).magnitude()
    right_rotation_error = Rotation.from_matrix(
        np.einsum("tij,tkj->tik", rotation_right, achieved_right_r)
    ).magnitude()
    metrics = {
        "left_position_error_m": left_error,
        "right_position_error_m": right_error,
        "left_orientation_error_rad": left_rotation_error,
        "right_orientation_error_rad": right_rotation_error,
        "simultaneous_5mm_rate": float(np.mean((left_error <= 0.005) & (right_error <= 0.005))),
        "minimum_joint_margin_rad": float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q))),
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
        "achieved_left_position": achieved_left_p,
        "achieved_right_position": achieved_right_p,
        "achieved_left_rotation": achieved_left_r,
        "achieved_right_rotation": achieved_right_r,
    }
    return q, metrics


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def continuous_hand_trajectories(
    timeline: SemanticTimeline,
    source_action: np.ndarray,
    runtime: ActiveG1Dex3,
    left_acquire: np.ndarray,
    left_hold: np.ndarray,
    right_preinsert: np.ndarray,
    right_hook: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    length = timeline.trajectory_length
    left = np.repeat(runtime.open_hand_q["left"][None], length, axis=0)
    right = np.repeat(runtime.open_hand_q["right"][None], length, axis=0)
    left_grasp = timeline.event("left_phone_grasp_start").action_index
    portrait = timeline.event("phone_portrait_reached").action_index
    charger = timeline.event("phone_charger_attachment_complete").action_index
    left_release = timeline.event("left_phone_release_complete").action_index
    right_grasp = timeline.event("right_accessory_grasp_start").action_index
    detachment = timeline.event("accessory_detachment_start").action_index
    removed = timeline.event("accessory_removed").action_index
    right_release = timeline.event("right_accessory_release_complete").action_index
    if None in (left_grasp, portrait, charger, left_release, right_grasp, detachment, removed, right_release):
        raise ValueError("task-critical semantic timeline is unresolved")
    left_grasp, portrait, charger, left_release = map(int, (left_grasp, portrait, charger, left_release))
    right_grasp, detachment, removed, right_release = map(int, (right_grasp, detachment, removed, right_release))

    acquisition = smoothstep(timeline.progress("phone_acquisition")[left_grasp : portrait + 1])
    split = np.clip(2.0 * acquisition, 0.0, 1.0)
    second = np.clip(2.0 * acquisition - 1.0, 0.0, 1.0)
    left[left_grasp : portrait + 1] = (
        (1.0 - split[:, None]) * left_acquire
        + split[:, None] * ((1.0 - second[:, None]) * left_acquire + second[:, None] * left_hold)
    )
    left[portrait : charger + 1] = left_hold
    left_release_progress = smoothstep(timeline.progress("left_release")[charger : left_release + 1])
    left[charger : left_release + 1] = (
        (1.0 - left_release_progress[:, None]) * left_hold
        + left_release_progress[:, None] * runtime.open_hand_q["left"]
    )
    left[left_release:] = runtime.open_hand_q["left"]

    right[right_grasp] = right_preinsert
    insertion = smoothstep(timeline.progress("accessory_acquisition")[right_grasp : detachment + 1])
    right[right_grasp : detachment + 1] = (
        (1.0 - insertion[:, None]) * right_preinsert + insertion[:, None] * right_hook
    )
    right[detachment : right_release + 1] = right_hook
    right_release_progress = smoothstep(timeline.progress("right_release")[removed : right_release + 1])
    right[removed : right_release + 1] = (
        (1.0 - right_release_progress[:, None]) * right_hook
        + right_release_progress[:, None] * runtime.open_hand_q["right"]
    )
    right[right_release:] = runtime.open_hand_q["right"]
    phases = {
        "left": timeline.sample_arrays["left_gripper_phase"],
        "right": timeline.sample_arrays["right_gripper_phase"],
    }
    return left, right, phases


def build_phone_object_trajectory(
    timeline: SemanticTimeline,
    wrist_pose_left: np.ndarray,
    phone_initial: np.ndarray,
    phone_on_charger: np.ndarray,
    wrist_from_phone_lock: np.ndarray,
) -> np.ndarray:
    length = timeline.trajectory_length
    phone = np.repeat(phone_initial[None], length, axis=0)
    grasp = int(timeline.event("left_phone_grasp_start").action_index)
    portrait = int(timeline.event("phone_portrait_reached").action_index)
    charger = int(timeline.event("phone_charger_attachment_complete").action_index)
    carrier = np.einsum("tij,jk->tik", wrist_pose_left, wrist_from_phone_lock)
    acquisition = smoothstep(timeline.progress("phone_acquisition")[grasp : portrait + 1])
    rotations = np.empty((len(acquisition), 3, 3))
    for offset, progress in enumerate(acquisition):
        rotations[offset] = interpolate_rotation(
            phone_initial[:3, :3], carrier[grasp + offset, :3, :3], np.array([progress])
        )[0]
    phone[grasp : portrait + 1, :3, :3] = rotations
    phone[grasp : portrait + 1, :3, 3] = (
        (1.0 - acquisition[:, None]) * phone_initial[:3, 3]
        + acquisition[:, None] * carrier[grasp : portrait + 1, :3, 3]
    )
    phone[portrait : charger + 1] = carrier[portrait : charger + 1]
    phone[charger:] = phone_on_charger
    return phone


def build_accessory_object_trajectory(
    timeline: SemanticTimeline,
    phone_pose: np.ndarray,
    phone_from_accessory: np.ndarray,
    wrist_pose_right: np.ndarray,
    wrist_from_accessory_lock: np.ndarray,
) -> np.ndarray:
    attached = np.einsum("tij,jk->tik", phone_pose, phone_from_accessory)
    result = attached.copy()
    detachment = int(timeline.event("accessory_detachment_start").action_index)
    removed = int(timeline.event("accessory_removed").action_index)
    release = int(timeline.event("right_accessory_release_complete").action_index)
    carrier = np.einsum("tij,jk->tik", wrist_pose_right, wrist_from_accessory_lock)
    progress = smoothstep(timeline.progress("accessory_removal")[detachment : removed + 1])
    for offset, value in enumerate(progress):
        frame = detachment + offset
        result[frame, :3, :3] = interpolate_rotation(
            attached[frame, :3, :3], carrier[frame, :3, :3], np.array([value])
        )[0]
        result[frame, :3, 3] = (
            (1.0 - value) * attached[frame, :3, 3] + value * carrier[frame, :3, 3]
        )
    result[removed : release + 1] = carrier[removed : release + 1]
    result[release:] = carrier[release]
    return result


def swept_right_c_audit(
    runtime: ActiveG1Dex3,
    arm_q: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    accessory_pose: np.ndarray,
    timeline: SemanticTimeline,
    inner_radius: float,
    outer_radius: float,
    depth: float,
    *,
    maximum_tip_step_m: float,
    minimum_substeps: int,
) -> dict[str, Any]:
    transitions = (
        ("PREINSERT_TO_INSERT", "right_accessory_grasp_start", "accessory_detachment_start"),
        ("INSERT_TO_HOOK", "accessory_detachment_start", "accessory_removed"),
        ("HOOK_TO_REMOVE", "accessory_removed", "right_accessory_release_complete"),
    )
    records = []
    penetration_count = 0
    maximum_penetration = 0.0
    maximum_step = 0.0
    for label, start_name, end_name in transitions:
        start, end = timeline.interval(start_name, end_name)
        segment_records = []
        for frame in range(start, end):
            runtime.assign(arm_q[frame], left_q[frame], right_q[frame])
            p0, _ = runtime.contact_pose("right_C")
            runtime.assign(arm_q[frame + 1], left_q[frame + 1], right_q[frame + 1])
            p1, _ = runtime.contact_pose("right_C")
            estimated = max(minimum_substeps, int(math.ceil(np.linalg.norm(p1 - p0) / maximum_tip_step_m)))
            local_max = 0.0
            local_hits = 0
            previous = None
            for substep in range(estimated + 1):
                fraction = substep / estimated
                aq = (1.0 - fraction) * arm_q[frame] + fraction * arm_q[frame + 1]
                lq = (1.0 - fraction) * left_q[frame] + fraction * left_q[frame + 1]
                rq = (1.0 - fraction) * right_q[frame] + fraction * right_q[frame + 1]
                pose = (1.0 - fraction) * accessory_pose[frame] + fraction * accessory_pose[frame + 1]
                # Linear rotation entries are projected back to SO(3); object
                # attachment is diagnostic and never enters the arm solve.
                u, _, vt = np.linalg.svd(pose[:3, :3])
                pose[:3, :3] = u @ vt
                runtime.assign(aq, lq, rq)
                point, _ = runtime.contact_pose("right_C")
                signed, _ = ring_material_gap(point, pose, inner_radius, outer_radius, depth)
                if signed < 0.0:
                    local_hits += 1
                    local_max = max(local_max, -signed)
                if previous is not None:
                    maximum_step = max(maximum_step, float(np.linalg.norm(point - previous)))
                previous = point
            penetration_count += local_hits
            maximum_penetration = max(maximum_penetration, local_max)
            segment_records.append({
                "frame_pair": [frame, frame + 1],
                "substeps": estimated,
                "material_penetration_samples": local_hits,
                "maximum_material_penetration_m": local_max,
            })
        records.append({"transition": label, "segments": segment_records})
    return {
        "transitions": records,
        "approximate_C_tip_maximum_substep_m": maximum_step,
        "minimum_substeps_per_frame_transition": minimum_substeps,
        "ring_material_penetration_sample_count": penetration_count,
        "maximum_ring_material_penetration_m": maximum_penetration,
        "pass": penetration_count == 0,
    }
