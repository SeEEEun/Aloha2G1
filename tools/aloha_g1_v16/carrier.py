"""Active-geometry contact-carrier construction and endpoint search.

Unlike the v15 diagnostic, this module does not treat a fingertip collision
pad *center* as the only possible contact point.  A contact point is optimized
inside the calibrated active collision-pad extent.  Arm and finger variables
are solved together and the wrist pose follows from that solution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aloha_g1_v15.kinematics import ActiveG1Dex3, normalize, transform


def inverse_pose(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -value[:3, :3].T @ value[:3, 3]
    return result


def rotation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(Rotation.from_matrix(np.asarray(a) @ np.asarray(b).T).magnitude())


def _contact_patch_state(
    runtime: ActiveG1Dex3,
    label: str,
    tangent_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point, normal and distal-link rotation in the scene frame."""
    spec = runtime.contacts[label]
    body = runtime.body_ids[spec.link]
    body_rotation_model = runtime.data.xmat[body].reshape(3, 3)
    local = spec.local_position + np.array(
        [0.0, float(tangent_offset[0]), float(tangent_offset[1])], dtype=np.float64
    )
    point_model = runtime.data.xpos[body] + body_rotation_model @ local
    point = runtime.model_to_scene_position(point_model)
    body_rotation = runtime.model_to_scene_rotation(body_rotation_model)
    normal = normalize(body_rotation @ spec.local_normal)
    return point, normal, body_rotation


def _proper_basis(primary: np.ndarray, secondary_hint: np.ndarray) -> np.ndarray:
    primary = normalize(primary)
    secondary = np.asarray(secondary_hint, dtype=np.float64)
    secondary = secondary - primary * float(np.dot(primary, secondary))
    if np.linalg.norm(secondary) < 1e-8:
        fallback = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(primary, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        secondary = fallback - primary * float(np.dot(primary, fallback))
    secondary = normalize(secondary)
    third = normalize(np.cross(primary, secondary))
    secondary = normalize(np.cross(third, primary))
    basis = np.column_stack((primary, secondary, third))
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-10):
        raise RuntimeError("contact-carrier basis is not orthonormal")
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1e-10):
        raise RuntimeError("contact-carrier basis is not right handed")
    return basis


def build_left_pinch_carrier(
    runtime: ActiveG1Dex3,
    a_tangent_offset: np.ndarray,
    b_tangent_offset: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build LEFT_PHONE_PINCH_CARRIER from active A/B pad points."""
    a, normal_a, _ = _contact_patch_state(runtime, "left_A", a_tangent_offset)
    b, normal_b, _ = _contact_patch_state(runtime, "left_B", b_tangent_offset)
    origin = 0.5 * (a + b)
    palm = runtime.palm_pose("left")[:3, 3]
    approach = origin - palm
    basis = _proper_basis(b - a, approach)
    return transform(basis, origin), {
        "A": a,
        "B": b,
        "normal_A": normal_a,
        "normal_B": normal_b,
        "palm": palm,
        "approach": normalize(approach),
    }


def build_right_hook_carrier(
    runtime: ActiveG1Dex3,
    c_tangent_offset: np.ndarray,
    ring_center: np.ndarray,
    hook_hint: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build RIGHT_ACCESSORY_HOOK_CARRIER from the active C pad point."""
    c, normal_c, _ = _contact_patch_state(runtime, "right_C", c_tangent_offset)
    radial = np.asarray(ring_center, dtype=np.float64) - c
    hook = radial + np.asarray(hook_hint, dtype=np.float64)
    basis = _proper_basis(normal_c, hook)
    return transform(basis, c), {
        "C": c,
        "normal_C": normal_c,
        "ring_center": np.asarray(ring_center, dtype=np.float64),
        "hook_hint": normalize(hook_hint),
    }


def _arm_torso_penetration(runtime: ActiveG1Dex3) -> float:
    depth = 0.0
    for row in runtime.penetrating_contacts(tolerance=0.0):
        names = "|".join(row["bodies"])
        if any(token in names for token in ("shoulder", "elbow", "wrist")) and any(
            token in names for token in ("torso", "waist", "pelvis")
        ):
            depth = max(depth, -float(row["distance_m"]))
    return depth


def _assign_side(
    runtime: ActiveG1Dex3,
    reference_arm: np.ndarray,
    side: str,
    side_arm: np.ndarray,
    hand: np.ndarray,
) -> np.ndarray:
    arm = np.asarray(reference_arm, dtype=np.float64).copy()
    block = slice(0, 7) if side == "left" else slice(7, 14)
    arm[block] = side_arm
    if side == "left":
        runtime.assign(arm, hand, runtime.open_hand_q["right"])
    else:
        runtime.assign(arm, runtime.open_hand_q["left"], hand)
    return arm


def _phone_face_target(
    phone_pose: np.ndarray,
    phone_half: np.ndarray,
    side_sign: float,
    xz: np.ndarray,
) -> np.ndarray:
    local = np.array([float(xz[0]), side_sign * phone_half[1], float(xz[1])])
    return phone_pose[:3, 3] + phone_pose[:3, :3] @ local


@dataclass
class LeftCarrierCandidate:
    assignment: str
    initial_arm_q: np.ndarray
    charger_arm_q: np.ndarray
    hand_q: np.ndarray
    patch_offsets: dict[str, np.ndarray]
    phone_local_contacts: dict[str, np.ndarray]
    initial_carrier: np.ndarray
    charger_carrier: np.ndarray
    initial_wrist: np.ndarray
    charger_wrist: np.ndarray
    initial_gaps_m: dict[str, float]
    charger_gaps_m: dict[str, float]
    normal_alignment: dict[str, dict[str, float]]
    rigid_translation_error_m: float
    rigid_rotation_error_rad: float
    arm_torso_penetration_m: dict[str, float]
    minimum_joint_margin_rad: float
    score: float
    valid: bool
    optimizer: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return dict(self.__dict__)


def search_left_common_rigid_carrier(
    runtime: ActiveG1Dex3,
    reference_arm_initial: np.ndarray,
    reference_arm_charger: np.ndarray,
    phone_initial: np.ndarray,
    phone_charger: np.ndarray,
    phone_dimensions: np.ndarray,
    assignment: str,
    *,
    hand_seed: np.ndarray | None = None,
    random_seed: int = 0,
    seed_count: int = 18,
    normal_weight: float = 0.22,
    minimum_normal_alignment: float = 0.25,
) -> LeftCarrierCandidate:
    """Jointly solve both endpoints for one physical rigid pinch carrier.

    The finger configuration, pad contact points and phone-local contact points
    are shared by both endpoints.  This is the direct numerical test of the
    one-rigid-carrier hypothesis.
    """
    if assignment not in ("A_SCREEN_B_BACK", "A_BACK_B_SCREEN"):
        raise ValueError(assignment)
    a_sign = -1.0 if assignment == "A_SCREEN_B_BACK" else 1.0
    b_sign = -a_sign
    half = 0.5 * np.asarray(phone_dimensions, dtype=np.float64)
    arm_limits = np.asarray(runtime.info["joint_limits"][:7], dtype=np.float64)
    hand_limits = np.vstack((runtime.contacts["left_A"].limits, runtime.contacts["left_B"].limits))
    a_extent = runtime.contacts["left_A"].half_extent
    b_extent = runtime.contacts["left_B"].half_extent
    # arm initial, arm charger, A/B hand, A uv, B uv, A xz, B xz
    lower = np.r_[
        arm_limits[:, 0], arm_limits[:, 0], hand_limits[:, 0],
        -a_extent[1], -a_extent[2], -b_extent[1], -b_extent[2],
        -half[0] + 0.001, -half[2] + 0.001,
        -half[0] + 0.001, -half[2] + 0.001,
    ]
    upper = np.r_[
        arm_limits[:, 1], arm_limits[:, 1], hand_limits[:, 1],
        a_extent[1], a_extent[2], b_extent[1], b_extent[2],
        -0.004, half[2] - 0.001,
        -0.004, half[2] - 0.001,
    ]
    if hand_seed is None:
        hand_seed = runtime.open_hand_q["left"][:5]
    base = np.r_[
        reference_arm_initial[:7], reference_arm_charger[:7], hand_seed,
        0.0, 0.0, 0.0, 0.0,
        -half[0] + 0.012, 0.018, -half[0] + 0.012, -0.018,
    ]
    rng = np.random.default_rng(random_seed)
    seeds = [np.clip(base, lower + 1e-7, upper - 1e-7)]
    for _ in range(seed_count - 1):
        value = base.copy()
        value[:7] += rng.normal(0.0, 0.28, 7)
        value[7:14] += rng.normal(0.0, 0.28, 7)
        value[14:19] = hand_limits[:, 0] + rng.random(5) * (hand_limits[:, 1] - hand_limits[:, 0])
        value[19:23] = rng.uniform(lower[19:23], upper[19:23])
        value[23:] = rng.uniform(lower[23:], upper[23:])
        seeds.append(np.clip(value, lower + 1e-7, upper - 1e-7))

    def endpoint(value: np.ndarray, which: int):
        side_q = value[:7] if which == 0 else value[7:14]
        reference = reference_arm_initial if which == 0 else reference_arm_charger
        hand = runtime.open_hand_q["left"].copy()
        hand[:5] = value[14:19]
        arm = _assign_side(runtime, reference, "left", side_q, hand)
        carrier, state = build_left_pinch_carrier(runtime, value[19:21], value[21:23])
        return arm, hand, carrier, state, runtime.wrist_pose("left")

    def targets(value: np.ndarray, phone: np.ndarray):
        return (
            _phone_face_target(phone, half, a_sign, value[23:25]),
            _phone_face_target(phone, half, b_sign, value[25:27]),
        )

    def residual(value: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        relative_rotations = []
        for which, phone in enumerate((phone_initial, phone_charger)):
            _, _, carrier, state, _ = endpoint(value, which)
            target_a, target_b = targets(value, phone)
            desired_a = -a_sign * phone[:3, 1]
            desired_b = -b_sign * phone[:3, 1]
            relative_rotations.append(phone[:3, :3].T @ carrier[:3, :3])
            rows.extend((
                900.0 * (state["A"] - target_a),
                900.0 * (state["B"] - target_b),
                normal_weight * (state["normal_A"] - desired_a),
                normal_weight * (state["normal_B"] - desired_b),
                np.array([220.0 * _arm_torso_penetration(runtime)]),
            ))
        rows.extend((
            2.5 * Rotation.from_matrix(relative_rotations[1] @ relative_rotations[0].T).as_rotvec(),
            0.006 * (value[:7] - reference_arm_initial[:7]),
            0.006 * (value[7:14] - reference_arm_charger[:7]),
            0.002 * (value[14:19] - hand_seed),
            0.001 * value[19:23],
        ))
        return np.concatenate(rows)

    best: LeftCarrierCandidate | None = None
    for seed_index, seed in enumerate(seeds):
        solution = least_squares(
            residual,
            seed,
            bounds=(lower + 1e-8, upper - 1e-8),
            max_nfev=650,
            ftol=2e-11,
            xtol=2e-11,
            gtol=2e-11,
            x_scale="jac",
        )
        endpoints = [endpoint(solution.x, value) for value in (0, 1)]
        target_pairs = [targets(solution.x, phone) for phone in (phone_initial, phone_charger)]
        gaps = [
            {
                "A": float(np.linalg.norm(endpoint_row[3]["A"] - target_pair[0])),
                "B": float(np.linalg.norm(endpoint_row[3]["B"] - target_pair[1])),
            }
            for endpoint_row, target_pair in zip(endpoints, target_pairs)
        ]
        normals = []
        for endpoint_row, phone in zip(endpoints, (phone_initial, phone_charger)):
            normals.append({
                "A": float(np.dot(endpoint_row[3]["normal_A"], -a_sign * phone[:3, 1])),
                "B": float(np.dot(endpoint_row[3]["normal_B"], -b_sign * phone[:3, 1])),
            })
        initial_relative = inverse_pose(phone_initial) @ endpoints[0][2]
        charger_relative = inverse_pose(phone_charger) @ endpoints[1][2]
        translation_error = float(np.linalg.norm(initial_relative[:3, 3] - charger_relative[:3, 3]))
        rotation_error = rotation_distance(initial_relative[:3, :3], charger_relative[:3, :3])
        penetrations = {}
        for name, endpoint_row in zip(("initial", "charger"), endpoints):
            endpoint(solution.x, 0 if name == "initial" else 1)
            penetrations[name] = _arm_torso_penetration(runtime)
        margins = [runtime.arm_joint_margin(row[0]) for row in endpoints]
        maximum_gap = max(*(gaps[0].values()), *(gaps[1].values()))
        score = (
            1000.0 * maximum_gap
            + 200.0 * translation_error
            + 1.5 * rotation_error
            + 500.0 * max(penetrations.values())
            + 0.002 * float(np.linalg.norm(solution.x[:7] - reference_arm_initial[:7]))
            + 0.002 * float(np.linalg.norm(solution.x[7:14] - reference_arm_charger[:7]))
        )
        valid = bool(
            maximum_gap <= 0.005
            and translation_error <= 0.005
            and math.degrees(rotation_error) <= 5.0
            and max(penetrations.values()) <= 1e-5
            and min(margins) >= -1e-8
            and min(value for row in normals for value in row.values()) >= minimum_normal_alignment
        )
        candidate = LeftCarrierCandidate(
            assignment=assignment,
            initial_arm_q=endpoints[0][0],
            charger_arm_q=endpoints[1][0],
            hand_q=endpoints[0][1],
            patch_offsets={"A": solution.x[19:21], "B": solution.x[21:23]},
            phone_local_contacts={
                "A": np.array([solution.x[23], a_sign * half[1], solution.x[24]]),
                "B": np.array([solution.x[25], b_sign * half[1], solution.x[26]]),
            },
            initial_carrier=endpoints[0][2],
            charger_carrier=endpoints[1][2],
            initial_wrist=endpoints[0][4],
            charger_wrist=endpoints[1][4],
            initial_gaps_m=gaps[0],
            charger_gaps_m=gaps[1],
            normal_alignment={"initial": normals[0], "charger": normals[1]},
            rigid_translation_error_m=translation_error,
            rigid_rotation_error_rad=rotation_error,
            arm_torso_penetration_m=penetrations,
            minimum_joint_margin_rad=float(min(margins)),
            score=score,
            valid=valid,
            optimizer={
                "seed_index": seed_index,
                "success": bool(solution.success),
                "status": int(solution.status),
                "cost": float(solution.cost),
                "nfev": int(solution.nfev),
                "normal_weight": float(normal_weight),
                "minimum_normal_alignment": float(minimum_normal_alignment),
            },
        )
        if best is None or (not candidate.valid, candidate.score) < (not best.valid, best.score):
            best = candidate
    if best is None:
        raise RuntimeError("left common-carrier search returned no candidate")
    return best


@dataclass
class RightCarrierCandidate:
    family: str
    arm_q: np.ndarray
    hand_q: np.ndarray
    patch_offset: np.ndarray
    carrier: np.ndarray
    wrist: np.ndarray
    contact_position: np.ndarray
    target_position: np.ndarray
    contact_gap_m: float
    insertion_alignment: float
    arm_torso_penetration_m: float
    minimum_joint_margin_rad: float
    ring_angle_rad: float
    score: float
    valid: bool
    optimizer: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return dict(self.__dict__)


def search_right_hook_anchor(
    runtime: ActiveG1Dex3,
    reference_arm: np.ndarray,
    accessory_pose: np.ndarray,
    inner_radius: float,
    depth: float,
    family: str,
    source_approach: np.ndarray,
    *,
    hand_seed: np.ndarray | None = None,
    random_seed: int = 0,
    seed_count: int = 18,
) -> RightCarrierCandidate:
    """Jointly solve right arm, C finger and active C-pad contact point."""
    if family not in ("AXIAL_APERTURE_INSERTION", "RADIAL_GAP_INSERTION"):
        raise ValueError(family)
    limits_arm = np.asarray(runtime.info["joint_limits"][7:], dtype=np.float64)
    limits_c = runtime.contacts["right_C"].limits
    extent = runtime.contacts["right_C"].half_extent
    lower = np.r_[limits_arm[:, 0], limits_c[:, 0], -extent[1], -extent[2], -math.pi]
    upper = np.r_[limits_arm[:, 1], limits_c[:, 1], extent[1], extent[2], math.pi]
    if hand_seed is None:
        hand_seed = runtime.open_hand_q["right"][-2:]
    base = np.r_[reference_arm[7:], hand_seed, 0.0, 0.0, math.pi]
    rng = np.random.default_rng(random_seed)
    seeds = [np.clip(base, lower + 1e-7, upper - 1e-7)]
    for _ in range(seed_count - 1):
        value = base.copy()
        value[:7] += rng.normal(0.0, 0.32, 7)
        value[7:9] = limits_c[:, 0] + rng.random(2) * (limits_c[:, 1] - limits_c[:, 0])
        value[9:11] = rng.uniform(lower[9:11], upper[9:11])
        if family == "RADIAL_GAP_INSERTION":
            value[11] = rng.normal(0.0, math.radians(18.0))
        else:
            value[11] = rng.uniform(-math.pi, math.pi)
        seeds.append(np.clip(value, lower + 1e-7, upper - 1e-7))

    def target(value: np.ndarray):
        angle = float(value[11])
        radial = np.array([math.cos(angle), 0.0, math.sin(angle)])
        local = radial * inner_radius
        if family == "AXIAL_APERTURE_INSERTION":
            local[1] = -0.5 * depth
            desired = normalize(accessory_pose[:3, 1])
            hook_hint = -(accessory_pose[:3, :3] @ radial)
        else:
            local[1] = 0.0
            desired = normalize(-(accessory_pose[:3, :3] @ radial))
            hook_hint = normalize(accessory_pose[:3, 1])
        return accessory_pose[:3, 3] + accessory_pose[:3, :3] @ local, desired, hook_hint

    def state(value: np.ndarray):
        hand = runtime.open_hand_q["right"].copy()
        hand[-2:] = value[7:9]
        arm = _assign_side(runtime, reference_arm, "right", value[:7], hand)
        target_position, desired, hook_hint = target(value)
        carrier, carrier_state = build_right_hook_carrier(
            runtime, value[9:11], accessory_pose[:3, 3], hook_hint
        )
        return arm, hand, carrier, carrier_state, runtime.wrist_pose("right"), target_position, desired

    def residual(value: np.ndarray) -> np.ndarray:
        _, _, _, state_row, _, target_position, desired = state(value)
        return np.r_[
            950.0 * (state_row["C"] - target_position),
            0.35 * (state_row["normal_C"] - desired),
            0.020 * (value[:7] - reference_arm[7:]),
            0.003 * (value[7:9] - hand_seed),
            0.001 * value[9:11],
            np.array([250.0 * _arm_torso_penetration(runtime)]),
        ]

    best: RightCarrierCandidate | None = None
    for seed_index, seed in enumerate(seeds):
        solution = least_squares(
            residual,
            seed,
            bounds=(lower + 1e-8, upper - 1e-8),
            max_nfev=500,
            ftol=2e-11,
            xtol=2e-11,
            gtol=2e-11,
            x_scale="jac",
        )
        arm, hand, carrier, state_row, wrist, target_position, desired = state(solution.x)
        gap = float(np.linalg.norm(state_row["C"] - target_position))
        alignment = float(np.dot(state_row["normal_C"], desired))
        penetration = _arm_torso_penetration(runtime)
        margin = runtime.arm_joint_margin(arm)
        approach_alignment = float(np.dot(carrier[:3, 0], normalize(source_approach)))
        score = (
            1000.0 * gap + 0.25 * (1.0 - alignment)
            + 0.08 * (1.0 - approach_alignment) + 500.0 * penetration
            + 0.004 * float(np.linalg.norm(solution.x[:7] - reference_arm[7:]))
        )
        valid = bool(gap <= 0.005 and penetration <= 1e-5 and margin >= -1e-8)
        candidate = RightCarrierCandidate(
            family=family,
            arm_q=arm,
            hand_q=hand,
            patch_offset=solution.x[9:11],
            carrier=carrier,
            wrist=wrist,
            contact_position=state_row["C"],
            target_position=target_position,
            contact_gap_m=gap,
            insertion_alignment=alignment,
            arm_torso_penetration_m=penetration,
            minimum_joint_margin_rad=margin,
            ring_angle_rad=float(solution.x[11]),
            score=score,
            valid=valid,
            optimizer={
                "seed_index": seed_index,
                "success": bool(solution.success),
                "status": int(solution.status),
                "cost": float(solution.cost),
                "nfev": int(solution.nfev),
            },
        )
        if best is None or (not candidate.valid, candidate.score) < (not best.valid, best.score):
            best = candidate
    if best is None:
        raise RuntimeError("right hook-carrier search returned no candidate")
    return best
