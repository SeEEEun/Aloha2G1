#!/usr/bin/env python3
"""Evidence-driven G1 forward-root sweep for Episode-49 v14.

The sweep changes only the *total* G1 forward offset used to place the active
G1/Dex3 model in the fixed MagSafe scene.  Every contact test uses one arm and
the relevant Dex3 joints in the same FK state.  Dex3 configurations written by
this program are key-frame diagnostics only; no hand trajectory is generated.

This program deliberately does not edit the authoritative scene.  A separate
apply/finalize step is permitted only after this program finds a candidate at
or below 0.30 m that passes physical reach, static clearance, and ALOHA motion
fidelity gates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from pxr import Usd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_physical_contact_anchored_v13 as v13
import build_episode49_target_phase_anchored_v12 as v12
import retarget_episode49_optimized_action_to_g1 as core
import robot_model_preview_common as preview_common
import solve_episode49_target_phase_anchored_v12_ik as ikv12

OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
V13 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_physical_contact_anchored_v13"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
PREVIEW_CONFIG = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
ACTIVE_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
REGISTRATION = ROOT / "configs/magsafe_task_frame_registration.sim.json"
PHASE_LIBRARY = V12 / "aloha_phase_motion_library.npz"
BACKUP = ROOT / "backups/g1_root_forward_015_pre_v14_20260807_110041"

SCALE = 0.42
LAG = 7
FPS = 30.0
THRESHOLD = 0.005
OLD_OFFSET = 0.15
MAX_OFFSET = 0.30
METHOD = "ALOHA_PRIMARY_ROOT_REGISTERED_ARM_RETARGETING"
PHONE_MODE = "B_EDGE_A_FRONT"
R_SCENE_FROM_MODEL = v12.R_SCENE_FROM_MODEL.copy()
MODEL_ROOT = v12.MODEL_ROOT.copy()
PALM_OFFSET = {
    "left": np.array([0.0415, 0.003, 0.0]),
    "right": np.array([0.0415, -0.003, 0.0]),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(value, indent=2, default=default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz(path: Path, **value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **value)
    os.replace(tmp, path)


def normalize(value, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    value = np.asarray(value, float)
    norm = float(np.linalg.norm(value))
    if norm > 1e-12:
        return value / norm
    fallback = np.asarray(fallback, float)
    return fallback / np.linalg.norm(fallback)


def make_transform(rotation=np.eye(3), position=np.zeros(3)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation, float)
    result[:3, 3] = np.asarray(position, float)
    return result


def model_to_scene(position: np.ndarray, root_position: np.ndarray) -> np.ndarray:
    value = np.asarray(position, float)
    return (R_SCENE_FROM_MODEL @ (value - MODEL_ROOT).T).T + root_position


def rotation_model_to_scene(rotation: np.ndarray) -> np.ndarray:
    return R_SCENE_FROM_MODEL @ np.asarray(rotation, float)


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise RuntimeError(f"missing body: {name}")
    return result


def point_obb_penetration(points: np.ndarray, pose: np.ndarray, dimensions: np.ndarray) -> float:
    """Return deepest sampled-point penetration into an oriented box."""
    local = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    half = 0.5 * np.asarray(dimensions)
    inside = np.all(np.abs(local) <= half + 1e-12, axis=1)
    if not np.any(inside):
        return 0.0
    depth = np.min(half - np.abs(local[inside]), axis=1)
    return float(max(0.0, np.max(depth)))


def point_aabb_signed_distance(point: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    point = np.asarray(point)
    outside = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    if np.any(outside > 0.0):
        return float(np.linalg.norm(outside))
    return -float(np.min(np.minimum(point - lower, upper - point)))


@dataclass
class ContactContext:
    info: dict
    runtime: dict
    phone_dimensions: np.ndarray
    phone_initial: np.ndarray
    phone_action319: np.ndarray
    accessory_action319: np.ndarray
    phone_on_pad: np.ndarray
    ring_inner: float
    ring_outer: float
    ring_depth: float
    table_bounds: tuple[float, float, float, float]
    table_z: float
    old_q: np.ndarray
    left_source_approach_169: np.ndarray
    right_source_approach_319: np.ndarray
    left_source_approach_523: np.ndarray


class FullContactSolver:
    def __init__(self, context: ContactContext):
        self.c = context
        self.model = context.info["model"]
        self.data = mujoco.MjData(self.model)
        self.body = {
            label: body_id(self.model, context.runtime[label]["link"])
            for label in ("left_A", "left_B", "right_C")
        }
        self.wrist = {
            side: body_id(self.model, f"{side}_wrist_yaw_link") for side in ("left", "right")
        }
        self.phone_previous: dict[str, np.ndarray] = {}
        self.ring_previous: np.ndarray | None = None
        self.v13_charger = json.loads((V13 / "charger_active_fk_contact_reach_audit.json").read_text())
        self.v13_selected = json.loads((V13 / "selected_physical_carrier_anchors.json").read_text())

    def _assign(self, arm_q: np.ndarray, finger_values: dict[str, np.ndarray]) -> None:
        self.data.qpos[:] = self.c.info["stand_qpos"]
        self.data.qpos[self.c.info["arm_qpos_ids"]] = arm_q
        for label, value in finger_values.items():
            self.data.qpos[self.c.runtime[label]["qpos"]] = value
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _contact_proxy(self, label: str, root_position: np.ndarray):
        spec = self.c.runtime[label]
        link = self.body[label]
        rotation_model = self.data.xmat[link].reshape(3, 3)
        position_model = self.data.xpos[link]
        proxy_model = position_model + rotation_model @ spec["proxy"]
        proxy_scene = model_to_scene(proxy_model, root_position)
        normal_scene = rotation_model_to_scene(rotation_model @ spec["normal_local"])
        hull_model = position_model + spec["hull"] @ rotation_model.T
        hull_scene = model_to_scene(hull_model, root_position)
        return proxy_scene, normalize(normal_scene), hull_scene

    def _wrist_palm(self, side: str, root_position: np.ndarray):
        wrist = self.wrist[side]
        rotation = rotation_model_to_scene(self.data.xmat[wrist].reshape(3, 3))
        position = model_to_scene(self.data.xpos[wrist], root_position)
        palm = position + rotation @ PALM_OFFSET[side]
        # The diagnostic MuJoCo model attaches Dex3 directly to the wrist and
        # has no separately named palm body.  The active USD palm visual was
        # audited in v13; express its measured hull through the verified palm
        # proxy on the wrist instead of inventing a MuJoCo palm link.
        palm_hull_scene = palm + self.c.runtime[f"{side}_palm_visual"] @ rotation.T
        return position, rotation, palm, palm_hull_scene

    def _configuration_collision(self, side: str, root_position: np.ndarray) -> dict:
        torso_pairs = []
        torso_proximity = []
        cross_arm_pairs = []
        for contact in self.data.contact:
            bodies = [
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[geom]),
                ) or "world"
                for geom in (contact.geom1, contact.geom2)
            ]
            side_body = any(
                body.startswith(f"{side}_")
                and any(token in body for token in ("shoulder", "elbow", "wrist", "hand"))
                for body in bodies
            )
            torso = any(any(token in body for token in ("torso", "waist", "pelvis")) for body in bodies)
            if side_body and torso:
                pair = "|".join(sorted(bodies))
                torso_proximity.append({"pair": pair, "signed_distance_m": float(contact.dist)})
                # MuJoCo exposes contacts inside its detection margin.  The
                # physical gate is penetration, not a zero-distance housing
                # touch; retain all near pairs but reject only negative depth.
                if float(contact.dist) < -1e-5:
                    torso_pairs.append(pair)
            left = any(body.startswith("left_") and any(token in body for token in ("shoulder", "elbow", "wrist")) for body in bodies)
            right = any(body.startswith("right_") and any(token in body for token in ("shoulder", "elbow", "wrist")) for body in bodies)
            if left and right:
                if float(contact.dist) < -1e-5:
                    cross_arm_pairs.append("|".join(sorted(bodies)))

        x0, x1, y0, y1 = self.c.table_bounds
        min_table_clearance = math.inf
        table_penetration = 0.0
        for geom_id in range(self.model.ngeom):
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[geom_id]),
            ) or ""
            if not body_name.startswith(f"{side}_"):
                continue
            if not any(token in body_name for token in ("shoulder", "elbow", "wrist", "hand")):
                continue
            vertices = model_to_scene(ikv12.geom_world_vertices(self.model, self.data, geom_id), root_position)
            inside_xy = (
                (vertices[:, 0] >= x0) & (vertices[:, 0] <= x1)
                & (vertices[:, 1] >= y0) & (vertices[:, 1] <= y1)
            )
            if np.any(inside_xy):
                min_table_clearance = min(min_table_clearance, float(np.min(vertices[inside_xy, 2] - self.c.table_z)))
                table_penetration = max(table_penetration, float(max(0.0, self.c.table_z - np.min(vertices[inside_xy, 2]))))
        if not np.isfinite(min_table_clearance):
            min_table_clearance = 1.0
        return {
            "torso_hand_collision": bool(torso_pairs),
            "torso_hand_contact_pairs": sorted(set(torso_pairs)),
            "torso_hand_proximity_pairs": torso_proximity,
            "collision_penetration_tolerance_m": 1e-5,
            "arm_arm_collision": bool(cross_arm_pairs),
            "arm_arm_contact_pairs": sorted(set(cross_arm_pairs)),
            "arm_table_penetration_m": table_penetration,
            "minimum_arm_table_vertical_clearance_m": min_table_clearance,
            "pass": not torso_pairs and not cross_arm_pairs and table_penetration <= 1e-4,
        }

    def _torso_penetration_scalar(self, side: str) -> float:
        """Maximum MuJoCo arm/hand-to-torso penetration in the current state."""
        maximum = 0.0
        for contact in self.data.contact:
            bodies = [
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[geom]),
                ) or "world"
                for geom in (contact.geom1, contact.geom2)
            ]
            side_body = any(
                body.startswith(f"{side}_")
                and any(token in body for token in ("shoulder", "elbow", "wrist", "hand"))
                for body in bodies
            )
            torso = any(any(token in body for token in ("torso", "waist", "pelvis")) for body in bodies)
            if side_body and torso:
                maximum = max(maximum, max(0.0, -float(contact.dist)))
        return maximum

    def _phone_state(self, x: np.ndarray, root_position: np.ndarray):
        arm = self.c.info["stand_arm_q"].copy()
        arm[:7] = x[:7]
        self._assign(arm, {"left_A": x[7:10], "left_B": x[10:12]})
        a, na, ahull = self._contact_proxy("left_A", root_position)
        b, nb, bhull = self._contact_proxy("left_B", root_position)
        wrist, wrist_rotation, palm, palm_hull = self._wrist_palm("left", root_position)
        shoulder = model_to_scene(
            self.data.xpos[body_id(self.model, "left_shoulder_pitch_link")], root_position
        )
        return arm, a, na, ahull, b, nb, bhull, wrist, wrist_rotation, palm, palm_hull, shoulder

    def _phone_targets(self, x: np.ndarray, phone_pose: np.ndarray):
        dimensions = self.c.phone_dimensions
        x_surface, z_surface = x[-2:]
        # V13's selected and user-approved phone contact semantics:
        # thumb A on the front/back face adjacent to the left edge and index B
        # on the physical left long-side edge.  This is contact geometry, not a
        # hand-written arm waypoint.
        a_local = np.array([x_surface, -0.5 * dimensions[1], z_surface])
        b_local = np.array([-0.5 * dimensions[0], 0.0, z_surface])
        a_target = phone_pose[:3, 3] + phone_pose[:3, :3] @ a_local
        b_target = phone_pose[:3, 3] + phone_pose[:3, :3] @ b_local
        desired_a_normal = phone_pose[:3, 1]  # opposite the front face outward normal
        desired_b_normal = phone_pose[:3, 0]  # opposite the left-edge outward normal
        return a_target, b_target, desired_a_normal, desired_b_normal

    def solve_phone(
        self,
        candidate_id: str,
        root_position: np.ndarray,
        phone_pose: np.ndarray,
        action_index: int,
        approach: np.ndarray,
        previous: np.ndarray | None = None,
    ) -> dict:
        arm_limits = self.c.info["joint_limits"][:7]
        finger_limits = np.vstack((self.c.runtime["left_A"]["limits"], self.c.runtime["left_B"]["limits"]))
        dimensions = self.c.phone_dimensions
        # At contact acquisition the phone bottom is flush with the table.
        # Restrict the physical pinch to the upper side region so the distal
        # links do not solve the contact equation by passing through the table.
        z_lower = 0.025 if action_index == 169 else -0.020
        lower = np.r_[
            arm_limits[:, 0], finger_limits[:, 0],
            -0.5 * dimensions[0] + 0.001, z_lower,
        ]
        upper = np.r_[
            arm_limits[:, 1], finger_limits[:, 1],
            -0.5 * dimensions[0] + 0.030, 0.5 * dimensions[2] - 0.002,
        ]
        old_arm = self.c.old_q[action_index, :7]
        old_finger = np.asarray(
            self.v13_selected["left_phone_contact_start"]["left_dex3_AB_diagnostic_q"], float
        )
        if action_index == 523:
            old_finger = np.r_[
                self.v13_charger["left_A"]["diagnostic_finger_q_rad"],
                self.v13_charger["left_B"]["diagnostic_finger_q_rad"],
            ]
        arm_seeds = [old_arm]
        if action_index == 523:
            arm_a = np.asarray(self.v13_charger["left_A"]["diagnostic_arm_q_rad"], float)
            arm_b = np.asarray(self.v13_charger["left_B"]["diagnostic_arm_q_rad"], float)
            arm_seeds.extend((arm_a, arm_b, 0.5 * (arm_a + arm_b)))
        if previous is not None:
            arm_seeds.insert(0, previous[:7])
        seeds = []
        if previous is not None:
            seeds.append(previous)
        for arm_seed in arm_seeds:
            seeds.append(np.r_[arm_seed, old_finger, -0.5 * dimensions[0] + 0.012, 0.0])
        seeds.append(np.r_[old_arm, np.mean(finger_limits, axis=1), -0.5 * dimensions[0] + 0.018, 0.018])
        # Deterministic multistart around the collision-free v12 branch.  The
        # contact equations are redundant, so a single least-squares basin can
        # otherwise find a fingertip-perfect but shoulder/torso-colliding arm.
        rng = np.random.default_rng(169 if action_index == 169 else 523)
        for _ in range(6 if action_index == 169 else 2):
            arm_seed = np.clip(
                old_arm + rng.normal(0.0, 0.28, 7),
                arm_limits[:, 0] + 1e-4, arm_limits[:, 1] - 1e-4,
            )
            finger_seed = finger_limits[:, 0] + rng.random(5) * (
                finger_limits[:, 1] - finger_limits[:, 0]
            )
            seeds.append(np.r_[
                arm_seed, finger_seed,
                -0.5 * dimensions[0] + rng.uniform(0.005, 0.026),
                rng.uniform(max(z_lower, 0.006), 0.030),
            ])

        rows = []
        approach = normalize(approach, [0, 1, 0])
        for seed_id, seed in enumerate(seeds):
            seed = np.clip(seed, lower + 1e-8, upper - 1e-8)

            def residual(x):
                state = self._phone_state(x, root_position)
                _, pa, na, _, pb, nb, _, _, _, palm, _, _ = state
                ta, tb, dna, dnb = self._phone_targets(x, phone_pose)
                carrier_direction = normalize(0.5 * (pa + pb) - palm, approach)
                torso_penetration = self._torso_penetration_scalar("left")
                return np.r_[
                    300.0 * (pa - ta), 300.0 * (pb - tb),
                    0.15 * (na - dna), 0.15 * (nb - dnb),
                    0.04 * (carrier_direction - approach),
                    0.0100 * (x[:7] - old_arm),
                    0.0001 * (x[7:12] - np.mean(finger_limits, axis=1)),
                    [80.0 * torso_penetration],
                ]

            solution = least_squares(
                residual, seed, bounds=(lower, upper), max_nfev=1800,
                ftol=1e-11, xtol=1e-11, gtol=1e-11,
            )
            x = solution.x
            state = self._phone_state(x, root_position)
            (arm, pa, na, ahull, pb, nb, bhull, wrist, wrist_rotation,
             palm, palm_hull, shoulder) = state
            ta, tb, dna, dnb = self._phone_targets(x, phone_pose)
            gap_a = float(np.linalg.norm(pa - ta))
            gap_b = float(np.linalg.norm(pb - tb))
            penetration = max(
                point_obb_penetration(ahull, phone_pose, dimensions),
                point_obb_penetration(bhull, phone_pose, dimensions),
                point_obb_penetration(palm_hull, phone_pose, dimensions),
            )
            collision = self._configuration_collision("left", root_position)
            approach_alignment = float(np.dot(normalize(0.5 * (pa + pb) - palm), approach))
            normal_a = float(np.dot(na, dna))
            normal_b = float(np.dot(nb, dnb))
            margin = THRESHOLD - max(gap_a, gap_b)
            valid = bool(
                max(gap_a, gap_b) <= THRESHOLD
                and penetration <= THRESHOLD
                and collision["pass"]
            )
            row = {
                "candidate_id": f"{candidate_id}_seed_{seed_id}",
                "action_index": action_index,
                "contact_semantics": PHONE_MODE,
                "arm_q_rad": arm[:7],
                "diagnostic_left_dex3_AB_q_rad": x[7:12],
                "surface_parameters_phone_local_m": {"x": x[-2], "z": x[-1]},
                "wrist_position_m": wrist,
                "wrist_rotation": wrist_rotation,
                "palm_position_m": palm,
                "left_A_contact_position_m": pa,
                "left_B_contact_position_m": pb,
                "left_A_target_surface_m": ta,
                "left_B_target_surface_m": tb,
                "left_A_gap_m": gap_a,
                "left_B_gap_m": gap_b,
                "left_A_contact_normal_alignment": normal_a,
                "left_B_contact_normal_alignment": normal_b,
                "mapped_aloha_approach_alignment": approach_alignment,
                "maximum_phone_penetration_m": penetration,
                "shoulder_position_m": shoulder,
                "shoulder_to_contact_midpoint_m": float(np.linalg.norm(0.5 * (pa + pb) - shoulder)),
                "physical_contact_margin_m": margin,
                "collision": collision,
                "optimizer_success": bool(solution.success),
                "optimizer_cost": float(solution.cost),
                "valid": valid,
                "dex3_configuration_diagnostic_only": True,
            }
            row["selection_score"] = (
                1000.0 * max(gap_a, gap_b)
                + 1000.0 * penetration
                + 0.05 * (2.0 - normal_a - normal_b)
                + 0.02 * (1.0 - approach_alignment)
                + (100.0 if not collision["pass"] else 0.0)
            )
            rows.append(row)
            if valid and max(gap_a, gap_b) < 5e-5 and normal_a > 0.4 and normal_b > 0.4:
                break
        valid_rows = [row for row in rows if row["valid"]]
        selected = min(valid_rows or rows, key=lambda row: row["selection_score"])
        selected["all_seed_results"] = [
            {
                "candidate_id": row["candidate_id"],
                "max_gap_mm": 1000 * max(row["left_A_gap_m"], row["left_B_gap_m"]),
                "penetration_mm": 1000 * row["maximum_phone_penetration_m"],
                "valid": row["valid"],
            }
            for row in rows
        ]
        return selected

    def _ring_state(self, x: np.ndarray, root_position: np.ndarray):
        arm = self.c.info["stand_arm_q"].copy()
        arm[7:14] = x[:7]
        self._assign(arm, {"right_C": x[7:9]})
        pc, nc, chull = self._contact_proxy("right_C", root_position)
        wrist, wrist_rotation, palm, palm_hull = self._wrist_palm("right", root_position)
        shoulder = model_to_scene(
            self.data.xpos[body_id(self.model, "right_shoulder_pitch_link")], root_position
        )
        return arm, pc, nc, chull, wrist, wrist_rotation, palm, palm_hull, shoulder

    def solve_ring(self, root_position: np.ndarray, previous: np.ndarray | None = None) -> dict:
        arm_limits = self.c.info["joint_limits"][7:14]
        finger_limits = self.c.runtime["right_C"]["limits"]
        lower = np.r_[arm_limits[:, 0], finger_limits[:, 0], -math.pi]
        upper = np.r_[arm_limits[:, 1], finger_limits[:, 1], math.pi]
        old_arm = self.c.old_q[319, 7:14]
        local_q = np.asarray(
            self.v13_selected["right_accessory"]["carrier_local"]["right_dex3_C_diagnostic_q"], float
        )
        seeds = []
        if previous is not None:
            seeds.append(previous)
        for theta in (0.0, -math.pi / 2, math.pi / 2, math.pi):
            seeds.append(np.r_[old_arm, local_q, theta])
        seeds.append(np.r_[old_arm, np.mean(finger_limits, axis=1), -math.pi / 3])
        accessory = self.c.accessory_action319
        ring_normal = normalize(accessory[:3, 1])
        approach = normalize(self.c.right_source_approach_319, ring_normal)
        rows = []
        for seed_id, seed in enumerate(seeds):
            seed = np.clip(seed, lower + 1e-8, upper - 1e-8)

            def target(x):
                theta = x[-1]
                local = np.array([
                    self.c.ring_inner * math.cos(theta),
                    -0.5 * self.c.ring_depth,
                    self.c.ring_inner * math.sin(theta),
                ])
                return accessory[:3, 3] + accessory[:3, :3] @ local

            def residual(x):
                _, pc, nc, _, _, _, palm, _, _ = self._ring_state(x, root_position)
                carrier_direction = normalize(pc - palm, approach)
                return np.r_[
                    300.0 * (pc - target(x)),
                    0.20 * (nc - ring_normal),
                    0.04 * (carrier_direction - approach),
                    0.0003 * (x[:7] - old_arm),
                ]

            solution = least_squares(
                residual, seed, bounds=(lower, upper), max_nfev=2200,
                ftol=1e-11, xtol=1e-11, gtol=1e-11,
            )
            x = solution.x
            arm, pc, nc, chull, wrist, wrist_rotation, palm, palm_hull, shoulder = self._ring_state(x, root_position)
            ring_target = target(x)
            gap = float(np.linalg.norm(pc - ring_target))
            direction_alignment = float(np.dot(nc, ring_normal))
            phone_penetration = max(
                point_obb_penetration(chull, self.c.phone_action319, self.c.phone_dimensions),
                point_obb_penetration(palm_hull, self.c.phone_action319, self.c.phone_dimensions),
            )
            collision = self._configuration_collision("right", root_position)
            margin = THRESHOLD - gap
            valid = bool(
                gap <= THRESHOLD
                and direction_alignment >= 0.5
                and phone_penetration <= THRESHOLD
                and collision["pass"]
            )
            row = {
                "candidate_id": f"RIGHT_C_INNER_RING_seed_{seed_id}",
                "action_index": 319,
                "arm_q_rad": arm[7:14],
                "diagnostic_right_dex3_C_q_rad": x[7:9],
                "ring_angle_deg": math.degrees(float(x[-1])),
                "wrist_position_m": wrist,
                "wrist_rotation": wrist_rotation,
                "palm_position_m": palm,
                "right_C_contact_position_m": pc,
                "right_C_ring_target_m": ring_target,
                "right_C_ring_gap_m": gap,
                "ring_insertion_direction_alignment": direction_alignment,
                "mapped_aloha_approach_alignment": float(np.dot(normalize(pc - palm), approach)),
                "maximum_wrist_phone_penetration_m": phone_penetration,
                "shoulder_position_m": shoulder,
                "shoulder_to_contact_m": float(np.linalg.norm(pc - shoulder)),
                "global_wrist_to_ring_m": float(np.linalg.norm(wrist - ring_target)),
                "physical_contact_margin_m": margin,
                "collision": collision,
                "optimizer_success": bool(solution.success),
                "optimizer_cost": float(solution.cost),
                "valid": valid,
                "dex3_configuration_diagnostic_only": True,
            }
            row["selection_score"] = (
                1000.0 * gap + 1000.0 * phone_penetration
                + 0.05 * (1.0 - direction_alignment)
                + (100.0 if not collision["pass"] else 0.0)
            )
            rows.append(row)
            if valid and gap < 5e-5 and direction_alignment > 0.9:
                break
        valid_rows = [row for row in rows if row["valid"]]
        selected = min(valid_rows or rows, key=lambda row: row["selection_score"])
        selected["all_seed_results"] = [
            {
                "candidate_id": row["candidate_id"],
                "gap_mm": 1000 * row["right_C_ring_gap_m"],
                "direction_alignment": row["ring_insertion_direction_alignment"],
                "valid": row["valid"],
            }
            for row in rows
        ]
        return selected


def static_clearance(context: ContactContext, root_position: np.ndarray) -> dict:
    model = context.info["model"]
    data = mujoco.MjData(model)
    data.qpos[:] = context.info["stand_qpos"]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    x0, x1, y0, y1 = context.table_bounds
    table_lower = np.array([x0, y0, context.table_z - 0.045])
    table_upper = np.array([x1, y1, context.table_z])
    categories = {
        "torso": ("torso", "waist"),
        "pelvis": ("pelvis", "hip"),
        "leg_foot": ("knee", "ankle", "foot"),
        "shoulder": ("shoulder",),
        "initial_hand": ("wrist", "hand"),
    }
    category_min = {name: math.inf for name in categories}
    penetrations = []
    object_penetrations = []
    for geom_id in range(model.ngeom):
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
        ) or ""
        vertices = model_to_scene(ikv12.geom_world_vertices(model, data, geom_id), root_position)
        signed = min(point_aabb_signed_distance(point, table_lower, table_upper) for point in vertices)
        for category, tokens in categories.items():
            if any(token in body for token in tokens):
                category_min[category] = min(category_min[category], signed)
        if signed < -1e-4:
            penetrations.append({"body": body, "penetration_m": -signed})
        if any(token in body for token in ("torso", "waist", "shoulder", "elbow", "wrist", "hand")):
            phone_pen = point_obb_penetration(vertices, context.phone_initial, context.phone_dimensions)
            if phone_pen > 1e-4:
                object_penetrations.append({"body": body, "object": "phone", "penetration_m": phone_pen})
    for key in category_min:
        if not np.isfinite(category_min[key]):
            category_min[key] = 1.0
    return {
        "category_minimum_table_clearance_m": category_min,
        "minimum_table_clearance_m": min(category_min.values()),
        "table_penetrations": penetrations,
        "initial_robot_fixed_object_penetrations": object_penetrations,
        "static_collision_pass": not penetrations and not object_penetrations,
        "table_aabb_m": {"lower": table_lower, "upper": table_upper},
        "stand_qpos_used": True,
    }


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, float)
    target = np.asarray(target, float)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vh = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def fidelity_for_root(
    row: dict,
    base_model_l: np.ndarray,
    base_model_r: np.ndarray,
    base_model_lr: np.ndarray,
    base_model_rr: np.ndarray,
    source_progress: np.ndarray,
) -> dict:
    left169 = np.asarray(row["action169"]["palm_position_m"], float)
    right319 = np.asarray(row["action319"]["palm_position_m"], float)
    left523 = np.asarray(row["action523"]["palm_position_m"], float)
    source_landmarks = np.vstack((base_model_l[169], base_model_r[319], base_model_l[523]))
    target_landmarks = np.vstack((left169, right319, left523))
    global_r, global_t = kabsch(source_landmarks, target_landmarks)
    base_l = (global_r @ base_model_l.T).T + global_t
    base_r = (global_r @ base_model_r.T).T + global_t
    base_lr = np.einsum("ij,tjk->tik", global_r, base_model_lr)
    base_rr = np.einsum("ij,tjk->tik", global_r, base_model_rr)
    right334 = right319 + global_r @ (base_model_r[334] - base_model_r[319])
    # Keyframe contact poses are physical constraints.  The continuous
    # residual is still solved only at the approved phase-boundary knots.
    residual_l169 = left169 - base_l[169]
    residual_l523 = left523 - base_l[523]
    residual_delta = residual_l523 - residual_l169
    residual_r319 = right319 - base_r[319]
    residual_r334 = right334 - base_r[334]
    profiles = {
        "VERY_STRONG_ALOHA": {"magnitude": 18.0, "velocity": 300.0, "acceleration": 2400.0, "bimanual": 90.0},
        "STRONG_ALOHA": {"magnitude": 8.0, "velocity": 150.0, "acceleration": 1100.0, "bimanual": 45.0},
        "BALANCED": {"magnitude": 3.0, "velocity": 70.0, "acceleration": 420.0, "bimanual": 18.0},
    }
    candidates = {}
    arrays = {}
    acquisition_progress_193 = float(
        (source_progress[193] - source_progress[169])
        / max(source_progress[216] - source_progress[169], 1e-12)
    )
    for profile_name, profile in profiles.items():
        # The free quadratic solver preferred changing residual in the long,
        # low-displacement portrait-hold phase, which reduced L2 path fidelity.
        # Sweep only the approved boundary coefficient that divides the same
        # endpoint correction between the source-rich L1 and L3 phases.  L2
        # remains a constant residual and therefore retains its exact source
        # phase-relative curve.
        for split in np.linspace(0.20, 0.80, 25):
            residual_mid = residual_l169 + float(split) * residual_delta
            anchors_l = {
                0: residual_l169,
                169: residual_l169,
                193: residual_l169 + float(split) * acquisition_progress_193 * residual_delta,
                216: residual_mid,
                373: residual_mid,
                523: residual_l523,
                579: residual_l523,
                639: residual_l523,
                695: residual_l523,
            }
            # The physical right grasp/removal relation has the same mapped
            # phase displacement at 319 and 334.  A constant hand-specific
            # registration is the lowest time-varying-energy solution and does
            # not deform any source right-arm phase.
            anchors_r = {
                0: residual_r319,
                169: residual_r319,
                193: residual_r319,
                216: residual_r319,
                319: residual_r319,
                322: residual_r319,
                334: residual_r334,
                373: residual_r334,
                523: residual_r334,
                579: residual_r334,
                639: residual_r334,
                695: residual_r334,
            }
            knots_l, knots_r = v12.solve_residual_knots(anchors_l, anchors_r, profile)
            residual_l = v12.blend_knot_values(knots_l, source_progress)
            residual_r = v12.blend_knot_values(knots_r, source_progress)
            corrected_l = base_l + residual_l
            corrected_r = base_r + residual_r
            phases, bimanual, minimum = v12.phase_fidelity(
                base_l, base_r, corrected_l, corrected_r,
                base_lr, base_rr, base_lr, base_rr,
            )
            anchor_error = max(
                np.linalg.norm(corrected_l[169] - left169),
                np.linalg.norm(corrected_l[523] - left523),
                np.linalg.norm(corrected_r[319] - right319),
                np.linalg.norm(corrected_r[334] - right334),
            )
            velocity_energy = float(np.sum(np.diff(residual_l, axis=0) ** 2) + np.sum(np.diff(residual_r, axis=0) ** 2))
            acceleration_energy = float(np.sum(np.diff(residual_l, n=2, axis=0) ** 2) + np.sum(np.diff(residual_r, n=2, axis=0) ** 2))
            max_residual = float(max(np.linalg.norm(residual_l, axis=1).max(), np.linalg.norm(residual_r, axis=1).max()))
            hard_pass = bool(
                minimum["path_shape"] >= 0.90
                and minimum["speed"] >= 0.90
                and minimum["rotation_progress"] >= 0.90
                and anchor_error <= 1e-8
            )
            name = f"{profile_name}_SPLIT_{split:.3f}"
            candidates[name] = {
                "weights": profile,
                "left_endpoint_correction_split_fraction_at_action216_373": float(split),
                "source_progress_fraction_at_action193": acquisition_progress_193,
                "anchor_error_m": anchor_error,
                "minimum_major_phase_fidelity": minimum,
                "per_phase": phases,
                "bimanual": bimanual,
                "max_time_varying_residual_translation_m": max_residual,
                "residual_velocity_energy": velocity_energy,
                "residual_acceleration_energy": acceleration_energy,
                "hard_fidelity_gate_pass": hard_pass,
            }
            arrays[name] = {
                "knots_l": knots_l, "knots_r": knots_r,
                "residual_l": residual_l, "residual_r": residual_r,
                "corrected_l": corrected_l, "corrected_r": corrected_r,
            }
    eligible = [name for name, value in candidates.items() if value["hard_fidelity_gate_pass"]]
    ranking_pool = eligible or list(candidates)
    selected_name = max(
        ranking_pool,
        key=lambda name: (
            candidates[name]["minimum_major_phase_fidelity"]["path_shape"],
            candidates[name]["minimum_major_phase_fidelity"]["speed"],
            -candidates[name]["max_time_varying_residual_translation_m"],
        ),
    )
    return {
        "workspace_scale": SCALE,
        "global_rotation": global_r,
        "global_translation_m": global_t,
        "same_global_transform_both_hands_all_samples": True,
        "global_similarity_excluded_from_deformation": True,
        "landmark_fit_residuals_m": target_landmarks - ((global_r @ source_landmarks.T).T + global_t),
        "candidates": candidates,
        "selected_candidate": selected_name,
        "hard_fidelity_gate_pass": bool(candidates[selected_name]["hard_fidelity_gate_pass"]),
        "arrays": arrays[selected_name],
        "base_l": base_l,
        "base_r": base_r,
        "base_lr": base_lr,
        "base_rr": base_rr,
        "right_anchor334": right334,
    }


def sweep_row(offset: float, original_root: np.ndarray, forward: np.ndarray, solver: FullContactSolver,
              context: ContactContext, previous: dict[str, np.ndarray] | None = None) -> dict:
    root_position = original_root + offset * forward
    previous = previous or {}
    action169 = solver.solve_phone(
        "ACTION169_PHONE", root_position, context.phone_initial, 169,
        context.left_source_approach_169, previous.get("action169"),
    )
    action319 = solver.solve_ring(root_position, previous.get("action319"))
    action523 = solver.solve_phone(
        "ACTION523_CHARGER", root_position, context.phone_on_pad, 523,
        context.left_source_approach_523, previous.get("action523"),
    )
    clearance = static_clearance(context, root_position)
    all_physical = bool(
        action169["valid"] and action319["valid"] and action523["valid"]
        and clearance["static_collision_pass"]
    )
    return {
        "total_forward_offset_m": float(offset),
        "root_xyz_m": root_position,
        "action169": action169,
        "action319": action319,
        "action523": action523,
        "phone_center_to_pad_m": 0.0,
        "phone_normal_error_deg": 0.0,
        "static_clearance": clearance,
        "all_physical_gates_pass": all_physical,
        "solver_state": {
            "action169": np.r_[
                action169["arm_q_rad"], action169["diagnostic_left_dex3_AB_q_rad"],
                action169["surface_parameters_phone_local_m"]["x"],
                action169["surface_parameters_phone_local_m"]["z"],
            ],
            "action319": np.r_[
                action319["arm_q_rad"], action319["diagnostic_right_dex3_C_q_rad"],
                math.radians(action319["ring_angle_deg"]),
            ],
            "action523": np.r_[
                action523["arm_q_rad"], action523["diagnostic_left_dex3_AB_q_rad"],
                action523["surface_parameters_phone_local_m"]["x"],
                action523["surface_parameters_phone_local_m"]["z"],
            ],
        },
    }


def flat_row(row: dict) -> dict:
    return {
        "total_forward_offset_m": row["total_forward_offset_m"],
        "root_x": row["root_xyz_m"][0],
        "root_y": row["root_xyz_m"][1],
        "root_z": row["root_xyz_m"][2],
        "action169_A_gap_mm": 1000 * row["action169"]["left_A_gap_m"],
        "action169_B_gap_mm": 1000 * row["action169"]["left_B_gap_m"],
        "action169_pass": row["action169"]["valid"],
        "action319_C_gap_mm": 1000 * row["action319"]["right_C_ring_gap_m"],
        "action319_pass": row["action319"]["valid"],
        "action523_A_gap_mm": 1000 * row["action523"]["left_A_gap_m"],
        "action523_B_gap_mm": 1000 * row["action523"]["left_B_gap_m"],
        "action523_phone_pad_mm": 1000 * row["phone_center_to_pad_m"],
        "action523_phone_normal_deg": row["phone_normal_error_deg"],
        "action523_pass": row["action523"]["valid"],
        "static_collision_pass": row["static_clearance"]["static_collision_pass"],
        "all_physical_gates_pass": row["all_physical_gates_pass"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flat_row(row) for row in rows]
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    os.replace(tmp, path)


def stripped_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "solver_state"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        SOURCE, TIMELINE, ALIGNMENT, LAYOUT, PREVIEW_CONFIG, ACTIVE_SCENE,
        FIXED_SCENE, REGISTRATION, PHASE_LIBRARY,
        V12 / "position_only_exact_arm_trajectory.npz",
        V12 / "target_phone_pose_trajectory.npz",
        V12 / "target_accessory_pose_trajectory.npz",
        V13 / "charger_active_fk_contact_reach_audit.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not BACKUP.is_dir():
        raise RuntimeError(f"required pre-v14 backup is missing: {BACKUP}")

    hashes_before = {str(path.resolve()): sha(path) for path in required}
    backup_hashes = {str(path.relative_to(BACKUP)): sha(path) for path in BACKUP.rglob("*") if path.is_file()}
    dump(OUT / "input_hash_audit.json", {
        "immutable_input_sha256": hashes_before,
        "pre_v14_backup": str(BACKUP.resolve()),
        "backup_file_sha256": backup_hashes,
        "optimized_action_expected_shape": [990, 14],
        "action_to_observation_lag_frames": LAG,
        "workspace_scale": SCALE,
        "no_source_motion_recomputation": True,
    })

    preview = json.loads(PREVIEW_CONFIG.read_text())
    registration = json.loads(REGISTRATION.read_text())
    original_root = np.asarray(preview["g1"]["position_xyz_m"], float)
    stage = Usd.Stage.Open(str(ACTIVE_SCENE))
    if stage is None:
        raise RuntimeError(ACTIVE_SCENE)
    geometry = preview_common.task_geometry(stage, original_root.tolist())
    forward = np.asarray(geometry["table_forward_direction"], float)
    applied_015, _, _ = preview_common.offset_root_pose(
        stage, original_root.tolist(), preview["g1"]["orientation_wxyz"],
        forward_offset_m=OLD_OFFSET,
    )
    applied_015 = np.asarray(applied_015, float)
    active_root = v12.active_transform(stage, "/World/G1")[:3, 3]
    provenance_error = float(np.max(np.abs(applied_015 - active_root)))
    if provenance_error > 1e-6:
        dump(OUT / "forward_direction_audit.json", {
            "status": "BLOCKED_ROOT_DIRECTION_PROVENANCE",
            "original_root": original_root,
            "forward_direction": forward,
            "expected_current_root": applied_015,
            "active_root": active_root,
            "maximum_error_m": provenance_error,
        })
        raise RuntimeError("BLOCKED_ROOT_DIRECTION_PROVENANCE")
    dump(OUT / "forward_direction_audit.json", {
        "status": "ROOT_FORWARD_DIRECTION_PROVENANCE_PASS",
        "source_code": str((ROOT / "isaaclab_magsafe_fixed_scene/robot_model_preview_common.py").resolve()),
        "source_function": "task_geometry + offset_root_pose",
        "original_root_xyz_m": original_root,
        "verified_forward_unit_vector": forward,
        "forward_norm": float(np.linalg.norm(forward)),
        "expected_current_015_root_xyz_m": applied_015,
        "active_composed_usd_root_xyz_m": active_root,
        "reproduction_max_error_m": provenance_error,
        "prompt_approximation_not_used": True,
    })

    layout = json.loads(LAYOUT.read_text())
    phone_initial = v12.active_transform(stage, "/World/MagSafeScene/Phone")
    accessory_initial = v12.active_transform(stage, "/World/MagSafeScene/Accessory")
    charger_root = v12.active_transform(stage, "/World/MagSafeScene/Charger")
    pad = v12.active_transform(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    pad_vertical = normalize(pad[:3, 1])
    pad_normal = normalize(pad[:3, 2])
    desired_phone_rotation = np.column_stack((
        pad_vertical, -pad_normal, np.cross(pad_vertical, -pad_normal),
    ))
    if np.linalg.det(desired_phone_rotation) < 0:
        desired_phone_rotation[:, 2] *= -1
    phone_on_pad = make_transform(desired_phone_rotation, pad[:3, 3])
    with np.load(V12 / "target_phone_pose_trajectory.npz", allow_pickle=False) as values:
        phone_action319 = values["pose"][319].copy()
    with np.load(V12 / "target_accessory_pose_trajectory.npz", allow_pickle=False) as values:
        accessory_action319 = values["pose"][319].copy()

    with np.load(SOURCE, allow_pickle=False) as values:
        action = values["optimized_action"].copy()
        timestamps = values["timestamp"].copy()
        fps = float(values["fps"])
    with np.load(PHASE_LIBRARY, allow_pickle=False) as values:
        if not np.array_equal(values["optimized_action"], action):
            raise RuntimeError("phase library optimized_action hash/content mismatch")
        if not np.array_equal(values["timestamps"], timestamps):
            raise RuntimeError("phase library timestamps mismatch")
        left_source_p = values["left_tcp_position"].copy()
        right_source_p = values["right_tcp_position"].copy()
        left_source_r = values["left_tcp_rotation"].copy()
        right_source_r = values["right_tcp_rotation"].copy()
    if action.shape != (990, 14) or not np.isfinite(action).all() or fps != FPS:
        raise RuntimeError("immutable source action invariant failed")
    with np.load(V12 / "globally_registered_base_targets.npz", allow_pickle=False) as values:
        base_model_l = values["base_aloha_derived_left_position_model"].copy()
        base_model_r = values["base_aloha_derived_right_position_model"].copy()
        old_global_r = values["global_registration_rotation"].copy()
        base_model_lr = np.einsum("ji,tjk->tik", old_global_r, values["globally_registered_left_rotation"])
        base_model_rr = np.einsum("ji,tjk->tik", old_global_r, values["globally_registered_right_rotation"])
    with np.load(V12 / "position_only_exact_arm_trajectory.npz", allow_pickle=False) as values:
        old_q = values["g1_arm_q"].copy()

    info = core.ik.validate_model(core.G1_XML)
    _, runtime = v13.hand_geometry_audit(stage, info)
    left_approach_169 = normalize(left_source_p[169] - left_source_p[157], [0, 1, 0])
    right_approach_319 = normalize(right_source_p[319] - right_source_p[307], [0, 1, 0])
    left_approach_523 = normalize(left_source_p[523] - left_source_p[511], [0, 1, 0])
    # The source directions are mapped by the same verified fixed scene-axis
    # convention as the phase library.  They are a score, never a new path.
    left_approach_169 = normalize(R_SCENE_FROM_MODEL @ left_approach_169)
    right_approach_319 = normalize(R_SCENE_FROM_MODEL @ right_approach_319)
    left_approach_523 = normalize(R_SCENE_FROM_MODEL @ left_approach_523)
    context = ContactContext(
        info=info, runtime=runtime,
        phone_dimensions=np.asarray(layout["phone"]["size_landscape_xyz"], float),
        phone_initial=phone_initial,
        phone_action319=phone_action319,
        accessory_action319=accessory_action319,
        phone_on_pad=phone_on_pad,
        ring_inner=0.5 * float(layout["accessory"]["main_inner_diameter"]),
        ring_outer=0.5 * float(layout["accessory"]["main_outer_diameter"]),
        ring_depth=float(layout["accessory"]["main_depth"]),
        table_bounds=(0.0, float(layout["table"]["size_x"]), 0.0, float(layout["table"]["size_y"])),
        table_z=float(layout["table"]["surface_height"]),
        old_q=old_q,
        left_source_approach_169=left_approach_169,
        right_source_approach_319=right_approach_319,
        left_source_approach_523=left_approach_523,
    )
    dump(OUT / "configs_snapshot_before.json", {
        "registration": registration,
        "preview_config": preview,
        "active_object_frames": {
            "phone": phone_initial,
            "accessory": accessory_initial,
            "charger_root": charger_root,
            "charger_pad": pad,
        },
        "scene_hashes": {
            str(LAYOUT.resolve()): sha(LAYOUT),
            str(FIXED_SCENE.resolve()): sha(FIXED_SCENE),
            str(ACTIVE_SCENE.resolve()): sha(ACTIVE_SCENE),
            str(REGISTRATION.resolve()): sha(REGISTRATION),
        },
        "backup": str(BACKUP.resolve()),
    })

    solver = FullContactSolver(context)
    debug_action169 = os.environ.get("V14_DEBUG_ACTION169_OFFSET")
    if debug_action169:
        debug_offset = float(debug_action169)
        debug_root = original_root + debug_offset * forward
        debug = solver.solve_phone(
            "DEBUG_ACTION169", debug_root, context.phone_initial, 169,
            context.left_source_approach_169,
        )
        print(json.dumps(debug, indent=2, default=default), flush=True)
        return 0
    coarse_offsets = np.round(np.arange(0.15, 0.3001, 0.01), 3)
    coarse_rows = []
    previous = None
    for offset in coarse_offsets:
        print(f"[V14_ROOT_SWEEP] coarse total offset={offset:.3f}", flush=True)
        row = sweep_row(float(offset), original_root, forward, solver, context, previous)
        coarse_rows.append(row)
        previous = row["solver_state"]
        print(json.dumps(flat_row(row), indent=2), flush=True)
    write_csv(OUT / "root_sweep_coarse.csv", coarse_rows)
    dump(OUT / "root_sweep_coarse.json", {
        "definition": "each value is TOTAL forward offset from original root; never cumulative",
        "rows": [stripped_row(row) for row in coarse_rows],
    })

    first_pass_index = next((i for i, row in enumerate(coarse_rows) if row["all_physical_gates_pass"]), None)
    if first_pass_index is None:
        dump(OUT / "root_selection_report.json", {
            "status": "BLOCKED_ROOT_FORWARD_RANGE",
            "maximum_authorized_total_forward_offset_m": MAX_OFFSET,
            "limiting_actions_at_030": {
                "action169": coarse_rows[-1]["action169"]["valid"],
                "action319": coarse_rows[-1]["action319"]["valid"],
                "action523": coarse_rows[-1]["action523"]["valid"],
                "static_clearance": coarse_rows[-1]["static_clearance"]["static_collision_pass"],
            },
            "authoritative_scene_modified": False,
        })
        dump(OUT / "physical_reachability_by_root.json", {
            f"{row['total_forward_offset_m']:.3f}": stripped_row(row) for row in coarse_rows
        })
        dump(OUT / "static_clearance_by_root.json", {
            f"{row['total_forward_offset_m']:.3f}": row["static_clearance"] for row in coarse_rows
        })
        print("BLOCKED_ROOT_FORWARD_RANGE", flush=True)
        return 2

    coarse_pass = coarse_rows[first_pass_index]
    coarse_fail_offset = coarse_offsets[max(0, first_pass_index - 1)]
    coarse_pass_offset = coarse_pass["total_forward_offset_m"]
    fine_offsets = list(np.round(np.arange(coarse_fail_offset, coarse_pass_offset + 0.0001, 0.001), 3))
    practical = round(min(MAX_OFFSET, coarse_pass_offset + 0.005), 3)
    if practical not in fine_offsets:
        fine_offsets.append(practical)
    fine_offsets = sorted(set(float(value) for value in fine_offsets))
    fine_rows = []
    previous = coarse_rows[max(0, first_pass_index - 1)]["solver_state"]
    for offset in fine_offsets:
        print(f"[V14_ROOT_SWEEP] fine total offset={offset:.3f}", flush=True)
        row = sweep_row(offset, original_root, forward, solver, context, previous)
        fine_rows.append(row)
        previous = row["solver_state"]
        print(json.dumps(flat_row(row), indent=2), flush=True)
    write_csv(OUT / "root_sweep_fine.csv", fine_rows)
    dump(OUT / "root_sweep_fine.json", {
        "resolution_m": 0.001,
        "first_coarse_transition": [float(coarse_fail_offset), float(coarse_pass_offset)],
        "rows": [stripped_row(row) for row in fine_rows],
    })

    source_progress = v12.combined_source_progress(
        left_source_p, right_source_p, left_source_r, right_source_r
    )
    fidelity_by_root = {}
    for row in fine_rows:
        if not row["all_physical_gates_pass"]:
            continue
        fidelity = fidelity_for_root(
            row, base_model_l, base_model_r, base_model_lr, base_model_rr, source_progress
        )
        row["fidelity"] = fidelity
        fidelity_by_root[f"{row['total_forward_offset_m']:.3f}"] = {
            key: value for key, value in fidelity.items()
            if key not in ("arrays", "base_l", "base_r", "base_lr", "base_rr")
        }
    dump(OUT / "aloha_fidelity_by_root.json", fidelity_by_root)

    eligible = [
        row for row in fine_rows
        if row["all_physical_gates_pass"]
        and row.get("fidelity", {}).get("hard_fidelity_gate_pass", False)
    ]
    dump(OUT / "physical_reachability_by_root.json", {
        f"{row['total_forward_offset_m']:.3f}": stripped_row(row)
        for row in coarse_rows + fine_rows
    })
    dump(OUT / "static_clearance_by_root.json", {
        f"{row['total_forward_offset_m']:.3f}": row["static_clearance"]
        for row in coarse_rows + fine_rows
    })
    if not eligible:
        dump(OUT / "root_selection_report.json", {
            "status": "BLOCKED_ALOHA_FIDELITY",
            "physically_feasible_offsets_m": [
                row["total_forward_offset_m"] for row in fine_rows if row["all_physical_gates_pass"]
            ],
            "fidelity_by_root": fidelity_by_root,
            "authoritative_scene_modified": False,
        })
        print("BLOCKED_ALOHA_FIDELITY", flush=True)
        return 3

    eligible.sort(key=lambda row: row["total_forward_offset_m"])
    minimum = eligible[0]
    minimum_margin = min(
        minimum["action169"]["physical_contact_margin_m"],
        minimum["action319"]["physical_contact_margin_m"],
        minimum["action523"]["physical_contact_margin_m"],
    )
    selected = minimum
    reason = "smallest total forward offset passing physical, static-clearance, and ALOHA-fidelity gates"
    if minimum_margin < 0.002:
        target_offset = round(minimum["total_forward_offset_m"] + 0.005, 3)
        margin_rows = [row for row in eligible if abs(row["total_forward_offset_m"] - target_offset) < 5e-7]
        if not margin_rows:
            print(f"[V14_ROOT_SWEEP] evaluating explicit practical margin offset={target_offset:.3f}", flush=True)
            margin_row = sweep_row(target_offset, original_root, forward, solver, context, minimum["solver_state"])
            if margin_row["all_physical_gates_pass"]:
                margin_row["fidelity"] = fidelity_for_root(
                    margin_row, base_model_l, base_model_r, base_model_lr, base_model_rr, source_progress
                )
                fine_rows.append(margin_row)
                if margin_row["fidelity"]["hard_fidelity_gate_pass"]:
                    margin_rows = [margin_row]
        if margin_rows:
            candidate = margin_rows[0]
            min_f = minimum["fidelity"]["candidates"][minimum["fidelity"]["selected_candidate"]]["minimum_major_phase_fidelity"]
            can_f = candidate["fidelity"]["candidates"][candidate["fidelity"]["selected_candidate"]]["minimum_major_phase_fidelity"]
            if can_f["path_shape"] + 1e-9 >= min_f["path_shape"]:
                selected = candidate
                reason = (
                    "minimum feasible root had <2 mm contact margin; selected the explicitly "
                    "required +5 mm practical-margin candidate because it remained collision-free "
                    "and did not reduce minimum path fidelity"
                )

    selected_fidelity = selected["fidelity"]
    selected_arrays = selected_fidelity["arrays"]
    selected_offset = float(selected["total_forward_offset_m"])
    selected_root = np.asarray(selected["root_xyz_m"])
    save_npz(
        OUT / "global_task_registration_candidate_v14.npz",
        global_rotation=selected_fidelity["global_rotation"],
        global_translation=selected_fidelity["global_translation_m"],
        base_left_position=selected_fidelity["base_l"],
        base_right_position=selected_fidelity["base_r"],
        base_left_rotation=selected_fidelity["base_lr"],
        base_right_rotation=selected_fidelity["base_rr"],
    )
    save_npz(
        OUT / "phase_residual_candidate_v14.npz",
        residual_knots=v12.KNOTS,
        left_knot_values=selected_arrays["knots_l"],
        right_knot_values=selected_arrays["knots_r"],
        left_translation_residual=selected_arrays["residual_l"],
        right_translation_residual=selected_arrays["residual_r"],
    )
    save_npz(
        OUT / "corrected_targets_candidate_v14.npz",
        optimized_action=action,
        timestamps=timestamps,
        action_indices=np.arange(990),
        base_left_position=selected_fidelity["base_l"],
        base_right_position=selected_fidelity["base_r"],
        corrected_left_position=selected_arrays["corrected_l"],
        corrected_right_position=selected_arrays["corrected_r"],
        corrected_left_rotation=selected_fidelity["base_lr"],
        corrected_right_rotation=selected_fidelity["base_rr"],
        global_registration_rotation=selected_fidelity["global_rotation"],
        global_registration_translation=selected_fidelity["global_translation_m"],
        left_translation_residual=selected_arrays["residual_l"],
        right_translation_residual=selected_arrays["residual_r"],
        selected_root_xyz=selected_root,
        selected_total_forward_offset_m=np.array(selected_offset),
        workspace_scale=np.array(SCALE),
        method=np.array(METHOD),
        diagnostic_only=np.array(True),
        real_robot_command_allowed=np.array(False),
    )
    dump(OUT / "root_selection_report.json", {
        "status": "ROOT_CANDIDATE_SELECTED_PENDING_AUTHORITATIVE_APPLY",
        "minimum_physical_and_fidelity_forward_offset_m": minimum["total_forward_offset_m"],
        "minimum_candidate_contact_margin_mm": 1000 * minimum_margin,
        "selected_total_forward_offset_m": selected_offset,
        "selected_root_xyz_m": selected_root,
        "verified_forward_direction": forward,
        "selection_reason": reason,
        "selected_physical_metrics": stripped_row(selected),
        "selected_fidelity": {
            key: value for key, value in selected_fidelity.items()
            if key not in ("arrays", "base_l", "base_r", "base_lr", "base_rr")
        },
        "authoritative_scene_modified": False,
        "next_step_authorized": True,
    })
    dump(OUT / "selected_physical_carrier_anchors_candidate.json", {
        "selected_total_forward_offset_m": selected_offset,
        "root_xyz_m": selected_root,
        "left_phone_action169": selected["action169"],
        "right_accessory_action319": selected["action319"],
        "left_charger_action523": selected["action523"],
        "phone_on_pad_pose": phone_on_pad,
        "phone_center_to_pad_m": 0.0,
        "phone_normal_error_deg": 0.0,
        "dex3_keyframe_values_diagnostic_only": True,
    })
    print(json.dumps({
        "status": "ROOT_CANDIDATE_SELECTED_PENDING_AUTHORITATIVE_APPLY",
        "minimum_offset_m": minimum["total_forward_offset_m"],
        "selected_offset_m": selected_offset,
        "selected_root_xyz_m": selected_root,
        "reason": reason,
    }, indent=2, default=default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
