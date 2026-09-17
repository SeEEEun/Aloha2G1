"""Clean model-derived ALOHA and G1/Dex3 kinematics for this audit."""
from __future__ import annotations

import contextlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .common import (
    ARM_JOINT_NAMES,
    SIDES,
    apply_pose,
    apply_rotation,
    inverse_transform,
    pose_from_scene_robot,
    require_hash,
    transform,
)


ALOHA_ARM_NAMES = tuple(f"left_joint_{index}" for index in range(6)) + tuple(
    f"right_joint_{index}" for index in range(6)
)
ALOHA_MODEL_ARM_NAMES = tuple(
    f"follower_left_joint_{index}" for index in range(6)
) + tuple(f"follower_right_joint_{index}" for index in range(6))
ALOHA_MODEL_GRIPPERS = {
    "left": (
        "follower_left_right_carriage_joint",
        "follower_left_left_carriage_joint",
    ),
    "right": (
        "follower_right_right_carriage_joint",
        "follower_right_left_carriage_joint",
    ),
}
ALOHA_WRIST_BODIES = {
    "left": "follower_left_link_6",
    "right": "follower_right_link_6",
}
ALOHA_TIP_GEOMS = {
    "left": (
        "follower_left_gripper_right_tip",
        "follower_left_gripper_left_tip",
    ),
    "right": (
        "follower_right_gripper_right_tip",
        "follower_right_gripper_left_tip",
    ),
}


def _name_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise KeyError(f"active model missing {name!r} ({object_type})")
    return int(value)


def _normalize(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length > np.finfo(np.float64).eps:
        return value / length
    fallback_value = np.asarray(fallback, dtype=np.float64)
    return fallback_value / np.linalg.norm(fallback_value)


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(target) @ np.asarray(current).T).as_rotvec()


def _triangle_circumcenter(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the unique circumcenter/radius of three non-collinear 3-D points."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (3, 3):
        raise ValueError(f"three contact points required, got {points.shape}")
    first, second, third = points
    u = second - first
    v = third - first
    normal = np.cross(u, v)
    normal_squared = float(normal @ normal)
    if normal_squared <= 1e-16:
        raise RuntimeError("whole-hand contact centers are collinear")
    center = first + (
        np.cross(normal, u) * float(v @ v)
        + np.cross(v, normal) * float(u @ u)
    ) / (2.0 * normal_squared)
    return center, float(np.linalg.norm(center - first))


class ALOHAKinematics:
    """Stationary ALOHA wrist and physical jaw-center FK from named model geometry."""

    def __init__(self, common: Mapping[str, Any], scene: Mapping[str, Any]):
        path = Path(common["models"]["aloha_xml"])
        require_hash(path, common["models"]["aloha_xml_sha256"], "ALOHA model")
        self.path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.root_pose = pose_from_scene_robot(scene, "aloha")
        self.arm_joint_ids = np.asarray(
            [_name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ALOHA_MODEL_ARM_NAMES],
            dtype=np.int64,
        )
        self.arm_qpos_ids = np.asarray(
            [self.model.jnt_qposadr[value] for value in self.arm_joint_ids], dtype=np.int64
        )
        self.arm_limits = np.asarray(
            [self.model.jnt_range[value] for value in self.arm_joint_ids], dtype=np.float64
        )
        self.gripper_joint_ids = {
            side: np.asarray(
                [_name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
                dtype=np.int64,
            )
            for side, names in ALOHA_MODEL_GRIPPERS.items()
        }
        self.gripper_qpos_ids = {
            side: np.asarray([self.model.jnt_qposadr[value] for value in ids], dtype=np.int64)
            for side, ids in self.gripper_joint_ids.items()
        }
        self.gripper_limits = {
            side: np.asarray(self.model.jnt_range[ids[1]], dtype=np.float64)
            for side, ids in self.gripper_joint_ids.items()
        }
        self.wrist_ids = {
            side: _name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for side, name in ALOHA_WRIST_BODIES.items()
        }
        self.tip_geom_ids = {
            side: tuple(_name_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names)
            for side, names in ALOHA_TIP_GEOMS.items()
        }
        self.link6_to_tcp = self._derive_link6_to_tcp()

    def _derive_link6_to_tcp(self) -> dict[str, np.ndarray]:
        self.data.qpos[:] = 0.0
        for side in SIDES:
            midpoint = float(np.mean(self.gripper_limits[side]))
            self.data.qpos[self.gripper_qpos_ids[side]] = midpoint
        mujoco.mj_forward(self.model, self.data)
        result: dict[str, np.ndarray] = {}
        for side in SIDES:
            body = self.wrist_ids[side]
            rotation = np.asarray(self.data.xmat[body], dtype=np.float64).reshape(3, 3)
            tips = [
                np.asarray(self.data.geom_xpos[geom], dtype=np.float64)
                for geom in self.tip_geom_ids[side]
            ]
            center = 0.5 * (tips[0] + tips[1])
            closing = _normalize(tips[1] - tips[0], (0.0, 1.0, 0.0))
            approach = rotation[:, 0] - closing * float(rotation[:, 0] @ closing)
            approach = _normalize(approach, (1.0, 0.0, 0.0))
            lateral = _normalize(np.cross(approach, closing), (0.0, 0.0, 1.0))
            closing = _normalize(np.cross(lateral, approach), (0.0, 1.0, 0.0))
            local = rotation.T @ (center - np.asarray(self.data.xpos[body], dtype=np.float64))
            local_rotation = rotation.T @ np.column_stack(
                (approach, closing, lateral)
            )
            if not np.isclose(np.linalg.det(local_rotation), 1.0, atol=1e-10):
                raise RuntimeError(f"invalid {side} ALOHA interaction-frame rotation")
            result[side] = transform(local_rotation, local)
        return result

    def channel_report(self) -> dict[str, Any]:
        return {
            "arm_dataset_names": list(ALOHA_ARM_NAMES),
            "arm_model_joint_names": list(ALOHA_MODEL_ARM_NAMES),
            "arm_model_joint_limits_rad": self.arm_limits,
            "left_gripper_model_joints": list(ALOHA_MODEL_GRIPPERS["left"]),
            "right_gripper_model_joints": list(ALOHA_MODEL_GRIPPERS["right"]),
            "left_gripper_range_m": self.gripper_limits["left"],
            "right_gripper_range_m": self.gripper_limits["right"],
            "increasing_gripper_value_is_open": True,
            "link6_to_task_tcp": self.link6_to_tcp,
            "tcp_derivation": (
                "static jaw-tip midpoint with X from link-6 forward projected "
                "orthogonal to the jaw-closing axis, Y along the two named jaw tips, "
                "and Z completing a right-handed physical interaction frame"
            ),
        }

    def shoulder_wrist_reach_geometry(self) -> dict[str, Any]:
        """Derive source arm scale from named ALOHA joint-anchor geometry.

        The six-axis ALOHA chain has intersecting shoulder/wrist axes.  For the
        workspace scale we use the two translating links between joint 1 -> 2
        and joint 2 -> 3.  This is morphology-only and independent of task data.
        """
        self.data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        report: dict[str, Any] = {
            "derivation": (
                "active-model joint-anchor distances from arm joint 1 to 2 and "
                "joint 2 to 3; no scene object, episode, phase, or target success"
            ),
            "sides": {},
        }
        totals: list[float] = []
        for side, block in (("left", slice(0, 6)), ("right", slice(6, 12))):
            ids = self.arm_joint_ids[block]
            shoulder = np.asarray(self.data.xanchor[int(ids[1])], dtype=np.float64)
            elbow = np.asarray(self.data.xanchor[int(ids[2])], dtype=np.float64)
            wrist_base = np.asarray(self.data.xanchor[int(ids[3])], dtype=np.float64)
            upper = float(np.linalg.norm(elbow - shoulder))
            forearm = float(np.linalg.norm(wrist_base - elbow))
            total = upper + forearm
            totals.append(total)
            report["sides"][side] = {
                "joint_names": [
                    ALOHA_MODEL_ARM_NAMES[block.start + index]
                    for index in (1, 2, 3)
                ],
                "shoulder_anchor_local_m": shoulder,
                "elbow_anchor_local_m": elbow,
                "wrist_base_anchor_local_m": wrist_base,
                "upper_effective_length_m": upper,
                "forearm_effective_length_m": forearm,
                "effective_reach_m": total,
            }
        if not np.allclose(totals, totals[0], atol=1e-12, rtol=0.0):
            raise RuntimeError(f"asymmetric ALOHA reach geometry: {totals}")
        report["common_effective_reach_m"] = float(totals[0])
        return report

    def wrist_orientation_capacity(self) -> dict[str, Any]:
        """Report a morphology-only scalar capacity for the last three wrist axes."""
        report: dict[str, Any] = {
            "derivation": "L2 norm of active-model joint-range spans for wrist joints 3:6",
            "sides": {},
        }
        capacities: list[float] = []
        for side, block in (("left", slice(0, 6)), ("right", slice(6, 12))):
            limits = self.arm_limits[block][3:6]
            spans = limits[:, 1] - limits[:, 0]
            capacity = float(np.linalg.norm(spans))
            capacities.append(capacity)
            report["sides"][side] = {
                "joint_names": [
                    ALOHA_MODEL_ARM_NAMES[block.start + index]
                    for index in (3, 4, 5)
                ],
                "joint_range_spans_rad": spans,
                "capacity_l2_rad": capacity,
            }
        report["common_capacity_l2_rad"] = float(min(capacities))
        return report

    def map_action(self, action: np.ndarray) -> tuple[np.ndarray, int]:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 14 or not np.isfinite(action).all():
            raise ValueError(f"ALOHA action must be finite [T,14], got {action.shape}")
        qpos = np.zeros((len(action), self.model.nq), dtype=np.float64)
        qpos[:, self.arm_qpos_ids[:6]] = action[:, :6]
        qpos[:, self.arm_qpos_ids[6:]] = action[:, 7:13]
        clipped = 0
        for side, index in (("left", 6), ("right", 13)):
            lower, upper = self.gripper_limits[side]
            value = np.clip(action[:, index], lower, upper)
            clipped += int(np.count_nonzero(value != action[:, index]))
            qpos[:, self.gripper_qpos_ids[side][0]] = value
            qpos[:, self.gripper_qpos_ids[side][1]] = value
        return qpos, clipped

    def fk(self, action: np.ndarray) -> dict[str, Any]:
        qpos, clipped = self.map_action(action)
        count = len(qpos)
        output: dict[str, Any] = {"qpos": qpos, "gripper_clipped_values": clipped}
        for side in SIDES:
            for name in ("wrist", "tcp"):
                output[f"{side}_{name}_position_local"] = np.empty((count, 3), dtype=np.float64)
                output[f"{side}_{name}_rotation_local"] = np.empty((count, 3, 3), dtype=np.float64)
        for frame, value in enumerate(qpos):
            self.data.qpos[:] = value
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            for side in SIDES:
                body = self.wrist_ids[side]
                wrist = transform(
                    np.asarray(self.data.xmat[body], dtype=np.float64).reshape(3, 3),
                    np.asarray(self.data.xpos[body], dtype=np.float64),
                )
                tcp = wrist @ self.link6_to_tcp[side]
                for name, pose in (("wrist", wrist), ("tcp", tcp)):
                    output[f"{side}_{name}_position_local"][frame] = pose[:3, 3]
                    output[f"{side}_{name}_rotation_local"][frame] = pose[:3, :3]
        for side in SIDES:
            for name in ("wrist", "tcp"):
                output[f"{side}_{name}_position_world"] = apply_pose(
                    self.root_pose, output[f"{side}_{name}_position_local"]
                )
                output[f"{side}_{name}_rotation_world"] = apply_rotation(
                    self.root_pose, output[f"{side}_{name}_rotation_local"]
                )
        return output


@dataclass(frozen=True)
class ContactSpec:
    side: str
    digit: str
    link: str
    joint_names: tuple[str, ...]
    local_position: np.ndarray
    local_normal: np.ndarray
    half_extent: np.ndarray


class G1Kinematics:
    """Named fixed-base G1 arm, Dex3 task fingers, FK, Jacobian and collision audit."""

    def __init__(self, common: Mapping[str, Any], scene: Mapping[str, Any]):
        path = Path(common["models"]["g1_xml"])
        require_hash(path, common["models"]["g1_xml_sha256"], "G1 model")
        self.path = path
        # Model validation is intentionally silent: review logs should contain audit results,
        # not a repeated dump of every model joint.
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.root_pose = pose_from_scene_robot(scene, "g1")
        key = _name_id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.stand_qpos = np.asarray(self.model.key_qpos[key], dtype=np.float64).copy()
        self.model_pelvis = self.stand_qpos[:3].copy()
        self.arm_joint_names = np.asarray(ARM_JOINT_NAMES, dtype="U64")
        self.arm_joint_ids = np.asarray(
            [_name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES],
            dtype=np.int64,
        )
        self.arm_qpos_ids = np.asarray(
            [self.model.jnt_qposadr[value] for value in self.arm_joint_ids], dtype=np.int64
        )
        self.arm_dof_ids = np.asarray(
            [self.model.jnt_dofadr[value] for value in self.arm_joint_ids], dtype=np.int64
        )
        self.arm_limits = np.asarray(
            [self.model.jnt_range[value] for value in self.arm_joint_ids], dtype=np.float64
        )
        self.wrist_ids = {
            side: _name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
            for side in SIDES
        }
        self.shoulder_anchor_ids = {
            side: _name_id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{side}_shoulder_pitch_link",
            )
            for side in SIDES
        }
        mapping_path = Path(common["models"]["dex3_whole_hand_geometry"])
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        require_hash(path, mapping["active_model_sha256"], "Dex3 mapping active model")
        if Path(mapping["active_model"]).resolve() != path.resolve():
            raise RuntimeError("Dex3 task-finger mapping references a different active model")
        if tuple(mapping.get("task_digits", ())) != ("thumb", "index", "middle"):
            raise RuntimeError("Doll-Handoff requires all three Dex3 task digits")
        self.mapping_path = mapping_path
        self.contacts: dict[str, ContactSpec] = {}
        self.contact_role: dict[str, str] = {}
        for side in SIDES:
            for role in ("A", "B", "C"):
                row = mapping[side][role]
                digit = str(row["digit_chain"])
                self.contact_role[f"{side}_{digit}"] = role
                self.contacts[f"{side}_{digit}"] = ContactSpec(
                    side=side,
                    digit=digit,
                    link=str(row["distal_link"]),
                    joint_names=tuple(row["joint_names"]),
                    local_position=np.asarray(row["local_position_xyz_m"], dtype=np.float64),
                    local_normal=np.asarray(row["local_normal"], dtype=np.float64),
                    half_extent=np.asarray(row["pad_half_extent_m"], dtype=np.float64),
                )
        self.body_ids = {
            spec.link: _name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec.link)
            for spec in self.contacts.values()
        }
        self.hand_joint_names = {
            side: tuple(
                name
                for digit in ("thumb", "index", "middle")
                for name in self.contacts[f"{side}_{digit}"].joint_names
            )
            for side in SIDES
        }
        self.hand_joint_ids = {
            side: np.asarray(
                [_name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
                dtype=np.int64,
            )
            for side, names in self.hand_joint_names.items()
        }
        self.hand_qpos_ids = {
            side: np.asarray([self.model.jnt_qposadr[value] for value in ids], dtype=np.int64)
            for side, ids in self.hand_joint_ids.items()
        }
        self.hand_limits = {
            side: np.asarray([self.model.jnt_range[value] for value in ids], dtype=np.float64)
            for side, ids in self.hand_joint_ids.items()
        }
        self.open_hand_q = {
            side: np.clip(
                self.stand_qpos[ids],
                self.hand_limits[side][:, 0] + 1e-8,
                self.hand_limits[side][:, 1] - 1e-8,
            )
            for side, ids in self.hand_qpos_ids.items()
        }
        self.posture_joint_ids = {
            side: {
                name: _name_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_{name}_joint",
                )
                for name in ("shoulder_pitch", "elbow", "wrist_yaw")
            }
            for side in SIDES
        }
        self.clearance_pairs = self._derive_posture_clearance_pairs()
        self.assign(self.stand_qpos[self.arm_qpos_ids])
        self.arm_characteristic_length_m = {}
        for side in SIDES:
            ids = self.posture_joint_ids[side]
            shoulder = np.asarray(self.data.xanchor[ids["shoulder_pitch"]])
            elbow = np.asarray(self.data.xanchor[ids["elbow"]])
            wrist = np.asarray(self.data.xanchor[ids["wrist_yaw"]])
            self.arm_characteristic_length_m[side] = 0.5 * (
                float(np.linalg.norm(elbow - shoulder))
                + float(np.linalg.norm(wrist - elbow))
            )

    def _derive_posture_clearance_pairs(self) -> dict[str, tuple[tuple[int, int], ...]]:
        """Select task-independent robot geometry pairs for smooth posture clearance."""
        by_body: dict[str, list[int]] = {}
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                or f"body_{body}"
            )
            by_body.setdefault(name, []).append(geom)
        torso_geoms = [
            geom
            for body, geoms in by_body.items()
            if any(token in body for token in ("pelvis", "waist", "torso"))
            for geom in geoms
        ]
        arm_bodies = {
            side: [
                body
                for body in by_body
                if body.startswith(f"{side}_")
                and any(
                    token in body
                    for token in (
                        "shoulder_yaw",
                        "shoulder_roll",
                        "elbow",
                        "wrist_roll",
                        "wrist_pitch",
                        "wrist_yaw",
                    )
                )
            ]
            for side in SIDES
        }
        arm_geoms = {
            side: [geom for body in arm_bodies[side] for geom in by_body[body]]
            for side in SIDES
        }
        # Cross-arm posture clearance excludes wrist-yaw/palm/finger geometry so
        # legitimate bimanual hand interaction is not repelled.
        proximal_cross_geoms = {
            side: [
                geom
                for body in arm_bodies[side]
                if any(
                    token in body
                    for token in ("shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
                )
                for geom in by_body[body]
            ]
            for side in SIDES
        }
        candidates = {
            "TORSO": tuple(
                (arm, torso)
                for side in SIDES
                for arm in arm_geoms[side]
                for torso in torso_geoms
            ),
            "CROSS_ARM": tuple(
                (left, right)
                for left in proximal_cross_geoms["left"]
                for right in proximal_cross_geoms["right"]
            ),
        }
        return {
            category: tuple(
                pair for pair in pairs if self._geom_pair_collision_eligible(pair)
            )
            for category, pairs in candidates.items()
        }

    def _geom_pair_collision_eligible(self, pair: tuple[int, int]) -> bool:
        """Match MuJoCo's generic collision masks and adjacent-body exclusion."""
        first, second = map(int, pair)
        mask_enabled = bool(
            (int(self.model.geom_contype[first]) & int(self.model.geom_conaffinity[second]))
            or (
                int(self.model.geom_contype[second])
                & int(self.model.geom_conaffinity[first])
            )
        )
        body_first = int(self.model.geom_bodyid[first])
        body_second = int(self.model.geom_bodyid[second])
        directly_adjacent = bool(
            int(self.model.body_parentid[body_first]) == body_second
            or int(self.model.body_parentid[body_second]) == body_first
        )
        return mask_enabled and body_first != body_second and not directly_adjacent

    def model_to_world_position(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        shifted = value - self.model_pelvis
        return shifted @ self.root_pose[:3, :3].T + self.root_pose[:3, 3]

    def world_to_model_position(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        return (value - self.root_pose[:3, 3]) @ self.root_pose[:3, :3] + self.model_pelvis

    def model_to_world_rotation(self, value: np.ndarray) -> np.ndarray:
        return np.einsum("ij,...jk->...ik", self.root_pose[:3, :3], np.asarray(value))

    def world_to_model_rotation(self, value: np.ndarray) -> np.ndarray:
        return np.einsum("ij,...jk->...ik", self.root_pose[:3, :3].T, np.asarray(value))

    def assign(
        self,
        arm_q: np.ndarray,
        left_hand: np.ndarray | None = None,
        right_hand: np.ndarray | None = None,
    ) -> None:
        self.data.qpos[:] = self.stand_qpos
        self.data.qpos[self.arm_qpos_ids] = np.asarray(arm_q, dtype=np.float64)
        if left_hand is not None:
            self.data.qpos[self.hand_qpos_ids["left"]] = np.asarray(left_hand, dtype=np.float64)
        if right_hand is not None:
            self.data.qpos[self.hand_qpos_ids["right"]] = np.asarray(right_hand, dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def wrist_pose(self, side: str) -> np.ndarray:
        body = self.wrist_ids[side]
        return transform(
            np.asarray(self.data.xmat[body], dtype=np.float64).reshape(3, 3),
            np.asarray(self.data.xpos[body], dtype=np.float64),
        )

    def fixed_shoulder_anchors_model(self) -> dict[str, np.ndarray]:
        """Return the fixed shoulder-chain origins from the active model."""
        self.assign(self.stand_qpos[self.arm_qpos_ids])
        return {
            side: np.asarray(self.data.xpos[body], dtype=np.float64).copy()
            for side, body in self.shoulder_anchor_ids.items()
        }

    def arm_landmarks(self, arm_q: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """Named shoulder/elbow/wrist joint anchors for both seven-DoF arms."""
        self.assign(arm_q)
        return {
            side: {
                name: np.asarray(self.data.xanchor[joint], dtype=np.float64).copy()
                for name, joint in self.posture_joint_ids[side].items()
            }
            for side in SIDES
        }

    def sew_angles(
        self,
        arm_q: np.ndarray,
        nominal_elbow_guides_model: Mapping[str, np.ndarray],
    ) -> dict[str, float]:
        """Signed shoulder-elbow-wrist swivel relative to nominal morphology guides."""
        landmarks = self.arm_landmarks(arm_q)
        values: dict[str, float] = {}
        for side in SIDES:
            shoulder = landmarks[side]["shoulder_pitch"]
            elbow = landmarks[side]["elbow"]
            wrist = landmarks[side]["wrist_yaw"]
            axis = _normalize(wrist - shoulder, (1.0, 0.0, 0.0))
            elbow_radial = elbow - shoulder
            elbow_radial -= axis * float(elbow_radial @ axis)
            elbow_radial = _normalize(
                elbow_radial, (0.0, 1.0 if side == "left" else -1.0, 0.0)
            )
            reference = np.asarray(
                nominal_elbow_guides_model[side], dtype=np.float64
            ).copy()
            reference -= axis * float(reference @ axis)
            reference = _normalize(
                reference, (0.0, 1.0 if side == "left" else -1.0, 0.0)
            )
            values[side] = float(
                math.atan2(
                    float(axis @ np.cross(reference, elbow_radial)),
                    float(reference @ elbow_radial),
                )
            )
        return values

    def manipulability_state(self, arm_q: np.ndarray) -> dict[str, Any]:
        """Characteristic-length-scaled 6-D wrist Jacobian singular values."""
        state = self.wrist_state(arm_q)
        output: dict[str, Any] = {
            "metric": (
                "minimum singular value of [J_position; L_characteristic * "
                "J_rotation], with L derived from active upper/forearm geometry"
            ),
            "characteristic_length_m": self.arm_characteristic_length_m,
        }
        for side in SIDES:
            jacobian = np.asarray(state[f"{side}_jacobian"], dtype=np.float64).copy()
            jacobian[3:] *= float(self.arm_characteristic_length_m[side])
            singular = np.linalg.svd(jacobian, compute_uv=False)
            output[side] = {
                "singular_values": singular,
                "minimum_singular_value": float(np.min(singular)),
                "condition_number": float(
                    np.max(singular) / max(np.min(singular), 1e-12)
                ),
            }
        return output

    def _clearance_pair_distance(self, pair: tuple[int, int]) -> float:
        from_to = np.zeros(6, dtype=np.float64)
        distance = float(
            mujoco.mj_geomDistance(
                self.model,
                self.data,
                int(pair[0]),
                int(pair[1]),
                0.5,
                from_to,
            )
        )
        # MuJoCo mesh-mesh distance can return exactly zero while providing two
        # separated closest points. Preserve negative penetration, otherwise use
        # the actual closest-point separation rather than a false zero margin.
        closest_point_distance = float(np.linalg.norm(from_to[3:] - from_to[:3]))
        if distance == 0.0 and closest_point_distance > 1e-12:
            return closest_point_distance
        return distance

    def posture_clearance_state(
        self,
        arm_q: np.ndarray,
        selected_pairs: Mapping[str, tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        """Minimum actual-model torso and proximal cross-arm clearance."""
        self.assign(arm_q)
        output: dict[str, Any] = {
            "pair_policy": (
                "active collision geometries: shoulder-yaw/elbow/wrist links to "
                "pelvis/waist/torso; proximal left/right arms excluding hands"
            )
        }
        for category, pairs in self.clearance_pairs.items():
            candidates = (
                (selected_pairs[category],)
                if selected_pairs is not None and category in selected_pairs
                else pairs
            )
            values = [(self._clearance_pair_distance(pair), pair) for pair in candidates]
            distance, pair = min(values, key=lambda row: row[0])
            names: list[str] = []
            for geom in pair:
                body = int(self.model.geom_bodyid[int(geom)])
                names.append(
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body
                    )
                    or f"body_{body}"
                )
            output[category] = {
                "minimum_distance_m": float(distance),
                "closest_geom_pair": tuple(map(int, pair)),
                "closest_body_pair": tuple(names),
            }
        return output

    def shoulder_wrist_reach_geometry(self) -> dict[str, Any]:
        """Derive a conservative folded-arm radius from named active-model joints."""
        self.assign(self.stand_qpos[self.arm_qpos_ids])
        report: dict[str, Any] = {
            "derivation": (
                "two-link shoulder-pitch to elbow to wrist-yaw joint-anchor geometry "
                "at the active model's maximum positive elbow flexion"
            ),
            "sides": {},
        }
        radii: list[float] = []
        for side in SIDES:
            joint_ids = {
                name: _name_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_{name}_joint",
                )
                for name in ("shoulder_pitch", "elbow", "wrist_yaw")
            }
            shoulder = np.asarray(
                self.data.xanchor[joint_ids["shoulder_pitch"]], dtype=np.float64
            )
            elbow = np.asarray(
                self.data.xanchor[joint_ids["elbow"]], dtype=np.float64
            )
            wrist = np.asarray(
                self.data.xanchor[joint_ids["wrist_yaw"]], dtype=np.float64
            )
            upper = float(np.linalg.norm(elbow - shoulder))
            forearm = float(np.linalg.norm(wrist - elbow))
            elbow_max = float(self.model.jnt_range[joint_ids["elbow"], 1])
            folded = float(
                math.sqrt(
                    max(
                        0.0,
                        upper**2
                        + forearm**2
                        + 2.0 * upper * forearm * math.cos(elbow_max),
                    )
                )
            )
            radii.append(folded)
            report["sides"][side] = {
                "shoulder_pitch_anchor_model_m": shoulder,
                "elbow_anchor_model_m": elbow,
                "wrist_yaw_anchor_model_m": wrist,
                "upper_effective_length_m": upper,
                "forearm_effective_length_m": forearm,
                "maximum_positive_elbow_flexion_rad": elbow_max,
                "conservative_folded_radius_m": folded,
            }
        report["common_minimum_shoulder_to_wrist_radius_m"] = max(radii)
        return report

    def wrist_orientation_capacity(self) -> dict[str, Any]:
        """Report the same last-three-axis capacity used for source morphology."""
        report: dict[str, Any] = {
            "derivation": "L2 norm of active-model joint-range spans for wrist roll/pitch/yaw",
            "sides": {},
        }
        capacities: list[float] = []
        for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
            limits = self.arm_limits[block][4:7]
            spans = limits[:, 1] - limits[:, 0]
            capacity = float(np.linalg.norm(spans))
            capacities.append(capacity)
            report["sides"][side] = {
                "joint_names": self.arm_joint_names[block][4:7],
                "joint_range_spans_rad": spans,
                "capacity_l2_rad": capacity,
            }
        report["common_capacity_l2_rad"] = float(min(capacities))
        return report

    def wrist_state(self, arm_q: np.ndarray) -> dict[str, np.ndarray]:
        self.assign(arm_q)
        output: dict[str, np.ndarray] = {}
        for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
            body = self.wrist_ids[side]
            pose = self.wrist_pose(side)
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body)
            dofs = self.arm_dof_ids[block]
            output[f"{side}_position"] = pose[:3, 3].copy()
            output[f"{side}_rotation"] = pose[:3, :3].copy()
            output[f"{side}_jacobian"] = np.vstack((jacp[:, dofs], jacr[:, dofs]))
        return output

    def static_tool_point_state(
        self, side: str, wrist_to_tool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Current static wrist-attached tool point and its side-arm Jacobian.

        ``wrist_state`` or ``assign`` must have been called for the desired q first.
        The point is independent of changing finger joints, matching the required
        one-static-transform physical whole-hand grasp-frame representation.
        """
        position, _, jacobian_position, _ = self.static_tool_pose_state(
            side, wrist_to_tool
        )
        return position, jacobian_position

    def static_tool_pose_state(
        self, side: str, wrist_to_tool: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Static wrist-attached tool pose and side-arm position/rotation Jacobians."""
        wrist_to_tool = np.asarray(wrist_to_tool, dtype=np.float64)
        wrist = self.wrist_pose(side)
        pose = wrist @ wrist_to_tool
        position = pose[:3, 3]
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jac(
            self.model,
            self.data,
            jacp,
            jacr,
            position,
            self.wrist_ids[side],
        )
        block = slice(0, 7) if side == "left" else slice(7, 14)
        dofs = self.arm_dof_ids[block]
        return position, pose[:3, :3], jacp[:, dofs], jacr[:, dofs]

    def contact_pose(self, side: str, digit: str) -> tuple[np.ndarray, np.ndarray]:
        spec = self.contacts[f"{side}_{digit}"]
        body = self.body_ids[spec.link]
        rotation = np.asarray(self.data.xmat[body], dtype=np.float64).reshape(3, 3)
        position = np.asarray(self.data.xpos[body], dtype=np.float64) + rotation @ spec.local_position
        normal = _normalize(rotation @ spec.local_normal, (1.0, 0.0, 0.0))
        return position, normal

    def pinch_pose(self, side: str) -> np.ndarray:
        """Legacy thumb-index diagnostic; not the active Doll-Handoff tool frame."""
        wrist = self.wrist_pose(side)
        thumb, _ = self.contact_pose(side, "thumb")
        index, _ = self.contact_pose(side, "index")
        closing = _normalize(index - thumb, (0.0, 1.0, 0.0))
        approach = wrist[:3, 0] - closing * float(np.dot(wrist[:3, 0], closing))
        approach = _normalize(approach, (1.0, 0.0, 0.0))
        lateral = _normalize(np.cross(approach, closing), (0.0, 0.0, 1.0))
        closing = _normalize(np.cross(lateral, approach), (0.0, 1.0, 0.0))
        return transform(np.column_stack((approach, closing, lateral)), 0.5 * (thumb + index))

    def aperture(self, side: str) -> float:
        """Legacy thumb-index aperture diagnostic."""
        thumb, _ = self.contact_pose(side, "thumb")
        index, _ = self.contact_pose(side, "index")
        return float(np.linalg.norm(index - thumb))

    def whole_hand_grasp_pose(self, side: str) -> np.ndarray:
        """Physical three-contact enclosure frame for the active Dex3 geometry.

        The origin is the circumcenter of the thumb/index/middle distal-pad
        centers. X points from the thumb toward the index-middle opposition
        centroid. Z is the middle-to-index spread direction orthogonalized to X,
        and Y = Z x X. This definition is embodiment geometry, not a task-phase
        Cartesian correction.
        """
        points = np.stack(
            [self.contact_pose(side, digit)[0] for digit in ("thumb", "index", "middle")]
        )
        center, _ = _triangle_circumcenter(points)
        closing = _normalize(0.5 * (points[1] + points[2]) - points[0], (1.0, 0.0, 0.0))
        spread = points[1] - points[2]
        spread = _normalize(spread - closing * float(spread @ closing), (0.0, 0.0, 1.0))
        transverse = _normalize(np.cross(spread, closing), (0.0, 1.0, 0.0))
        spread = _normalize(np.cross(closing, transverse), (0.0, 0.0, 1.0))
        rotation = np.column_stack((closing, transverse, spread))
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
            raise RuntimeError(f"invalid {side} whole-hand grasp-frame rotation")
        return transform(rotation, center)

    def whole_hand_enclosure_radius(self, side: str) -> float:
        points = np.stack(
            [self.contact_pose(side, digit)[0] for digit in ("thumb", "index", "middle")]
        )
        _, radius = _triangle_circumcenter(points)
        return radius

    def derive_hand_primitives(
        self,
        scene: Mapping[str, Any],
        proposed: Mapping[str, Any],
        nominal_arm: np.ndarray,
    ) -> dict[str, Any]:
        settings = proposed["hand_primitive_derivation"]
        left_open = self.open_hand_q["left"].copy()
        right_open = self.open_hand_q["right"].copy()
        left_endpoint = left_open.copy()
        lookup = {name: index for index, name in enumerate(self.hand_joint_names["left"])}
        for name, value in settings["closure_endpoint_left_by_name"].items():
            left_endpoint[lookup[name]] = float(value)
        right_endpoint = right_open.copy()
        left_values = dict(zip(self.hand_joint_names["left"], left_endpoint))
        right_lookup = {name: index for index, name in enumerate(self.hand_joint_names["right"])}
        for name in self.hand_joint_names["right"]:
            left_name = name.replace("right_", "left_", 1)
            sign = 1.0 if name == "right_hand_thumb_0_joint" else -1.0
            right_endpoint[right_lookup[name]] = sign * left_values[left_name]
        for side, endpoint in (("left", left_endpoint), ("right", right_endpoint)):
            limits = self.hand_limits[side]
            if np.any(endpoint < limits[:, 0]) or np.any(endpoint > limits[:, 1]):
                raise RuntimeError(f"{side} closure endpoint violates active-model limits")

        doll_diameter = float(scene["doll"]["diameter_m"])
        target_grasp_radius = 0.5 * doll_diameter * float(
            settings["grasp_circumradius_ratio_of_provisional_doll_radius"]
        )
        target_preshape_radius = 0.5 * (
            doll_diameter + float(settings["preshape_diameter_clearance_m"])
        )

        def state_at(alpha: float) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
            left = left_open + alpha * (left_endpoint - left_open)
            right = right_open + alpha * (right_endpoint - right_open)
            self.assign(nominal_arm, left, right)
            return left, right, {
                side: self.whole_hand_enclosure_radius(side) for side in SIDES
            }

        def alpha_for_radius(target: float) -> float:
            low, high = 0.0, 1.0
            _, _, endpoints = state_at(high)
            _, _, openings = state_at(low)
            if not all(endpoints[side] <= target <= openings[side] for side in SIDES):
                raise RuntimeError(
                    f"requested enclosure radius {target} is outside common closure path: "
                    f"open={openings} endpoint={endpoints}"
                )
            for _ in range(64):
                middle = 0.5 * (low + high)
                _, _, values = state_at(middle)
                if float(np.mean(list(values.values()))) > target:
                    low = middle
                else:
                    high = middle
            return 0.5 * (low + high)

        alpha_grasp = alpha_for_radius(target_grasp_radius)
        alpha_preshape = alpha_for_radius(target_preshape_radius)
        left_grasp, right_grasp, grasp_radii = state_at(alpha_grasp)
        left_pre, right_pre, preshape_radii = state_at(alpha_preshape)
        states = {
            "left": {
                "OPEN": left_open,
                "PRESHAPE": left_pre,
                "GRASP": left_grasp,
                "HOLD": left_grasp,
                "RELEASE": left_open,
            },
            "right": {
                "OPEN": right_open,
                "PRESHAPE": right_pre,
                "GRASP": right_grasp,
                "HOLD": right_grasp,
                "RELEASE": right_open,
            },
        }
        self.assign(nominal_arm, left_grasp, right_grasp)
        wrist_to_grasp_frame = {
            side: inverse_transform(self.wrist_pose(side))
            @ self.whole_hand_grasp_pose(side)
            for side in SIDES
        }
        geometry: dict[str, Any] = {}
        for side in SIDES:
            points = np.stack(
                [
                    self.contact_pose(side, digit)[0]
                    for digit in ("thumb", "index", "middle")
                ]
            )
            grasp_pose = self.whole_hand_grasp_pose(side)
            wrist = self.wrist_pose(side)
            geometry[side] = {
                "pad_centers_model_m": {
                    digit: points[index]
                    for index, digit in enumerate(("thumb", "index", "middle"))
                },
                "pad_centers_wrist_m": {
                    digit: wrist[:3, :3].T @ (points[index] - wrist[:3, 3])
                    for index, digit in enumerate(("thumb", "index", "middle"))
                },
                "pairwise_pad_center_distances_m": {
                    "thumb_index": float(np.linalg.norm(points[0] - points[1])),
                    "thumb_middle": float(np.linalg.norm(points[0] - points[2])),
                    "index_middle": float(np.linalg.norm(points[1] - points[2])),
                },
                "enclosure_radius_m": self.whole_hand_enclosure_radius(side),
                "wrist_to_grasp_frame": wrist_to_grasp_frame[side],
                "grasp_frame_rotation_model": grasp_pose[:3, :3],
            }
        return {
            "status": "SIM_ONLY_NOT_REAL_DEX3_CALIBRATED",
            "states": states,
            "left_endpoint": left_endpoint,
            "right_endpoint": right_endpoint,
            "alpha_preshape": alpha_preshape,
            "alpha_grasp": alpha_grasp,
            "target_preshape_enclosure_radius_m": target_preshape_radius,
            "target_grasp_enclosure_radius_m": target_grasp_radius,
            "achieved_preshape_enclosure_radius_m": preshape_radii,
            "achieved_grasp_enclosure_radius_m": grasp_radii,
            "wrist_to_grasp_frame": wrist_to_grasp_frame,
            "joint_names": self.hand_joint_names,
            "task_digits": ("thumb", "index", "middle"),
            "grasp_frame_definition": (
                "circumcenter of the three named distal pad centers; X thumb-to-"
                "index/middle opposition, Z middle-to-index spread, Y=Z cross X"
            ),
            "geometry_at_grasp": geometry,
        }

    def derive_task_ready_nominal(
        self, common: Mapping[str, Any], scene: Mapping[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        row = common["canonical_task_ready_posture"]
        width, depth = map(float, scene["table"]["size_xy_m"])
        surface = float(scene["table"]["surface_height_m"])
        targets_world = {}
        for side in SIDES:
            fx, fy = map(float, row[f"{side}_wrist_world_xy_fraction_of_table"])
            targets_world[side] = np.asarray(
                [fx * width, fy * depth, surface + float(row["wrist_height_above_table_m"])],
                dtype=np.float64,
            )
        targets = {side: self.world_to_model_position(value) for side, value in targets_world.items()}
        desired_rotation = np.asarray(row["wrist_rotation_in_g1_model"], dtype=np.float64)
        stand_arm = self.stand_qpos[self.arm_qpos_ids].copy()

        def residual(q: np.ndarray) -> np.ndarray:
            state = self.wrist_state(q)
            values: list[np.ndarray] = []
            for side in SIDES:
                values.append(5.0 * (targets[side] - state[f"{side}_position"]))
                values.append(0.15 * _rotation_error(state[f"{side}_rotation"], desired_rotation))
            values.append(0.005 * (q - stand_arm))
            return np.concatenate(values)

        solution = least_squares(
            residual,
            stand_arm,
            bounds=(self.arm_limits[:, 0] + 1e-7, self.arm_limits[:, 1] - 1e-7),
            max_nfev=1000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        q = solution.x
        state = self.wrist_state(q)
        report = {
            "method": row["derivation"],
            "target_world": targets_world,
            "target_model": targets,
            "desired_wrist_rotation_model": desired_rotation,
            "q": q,
            "joint_names": self.arm_joint_names,
            "solver_success": bool(solution.success),
            "solver_cost": float(solution.cost),
            "function_evaluations": int(solution.nfev),
            "position_error_m": {
                side: float(np.linalg.norm(state[f"{side}_position"] - targets[side]))
                for side in SIDES
            },
            "orientation_error_rad": {
                side: float(
                    np.linalg.norm(_rotation_error(state[f"{side}_rotation"], desired_rotation))
                )
                for side in SIDES
            },
        }
        if max(report["position_error_m"].values()) > 1e-3:
            raise RuntimeError(f"task-ready nominal derivation failed: {report}")
        return q, report

    def trajectory_geometry(
        self,
        arm: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        penetration_tolerance: float,
        static_wrist_to_tool: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        count = len(arm)
        output: dict[str, Any] = {}
        for side in SIDES:
            for frame in ("wrist", "grasp"):
                output[f"{side}_{frame}_position_model"] = np.empty((count, 3))
                output[f"{side}_{frame}_rotation_model"] = np.empty((count, 3, 3))
                output[f"{side}_{frame}_position_world"] = np.empty((count, 3))
                output[f"{side}_{frame}_rotation_world"] = np.empty((count, 3, 3))
            if static_wrist_to_tool is not None:
                output[f"{side}_static_tool_position_model"] = np.empty((count, 3))
                output[f"{side}_static_tool_rotation_model"] = np.empty((count, 3, 3))
                output[f"{side}_static_tool_position_world"] = np.empty((count, 3))
                output[f"{side}_static_tool_rotation_world"] = np.empty((count, 3, 3))
        category_flags = {
            key: np.zeros(count, dtype=bool)
            for key in (
                "ARM_TORSO",
                "CROSS_ARM",
                "DISTAL_HAND_HAND",
                "WRIST_OR_PALM_TORSO",
                "OTHER",
            )
        }
        pairs: dict[str, set[str]] = {key: set() for key in category_flags}
        records: list[dict[str, Any]] = []
        for frame in range(count):
            self.assign(arm[frame], left_hand[frame], right_hand[frame])
            for side in SIDES:
                for name, pose in (
                    ("wrist", self.wrist_pose(side)),
                    ("grasp", self.whole_hand_grasp_pose(side)),
                ):
                    output[f"{side}_{name}_position_model"][frame] = pose[:3, 3]
                    output[f"{side}_{name}_rotation_model"][frame] = pose[:3, :3]
                    output[f"{side}_{name}_position_world"][frame] = self.model_to_world_position(
                        pose[:3, 3]
                    )
                    output[f"{side}_{name}_rotation_world"][frame] = self.model_to_world_rotation(
                        pose[:3, :3]
                    )
                if static_wrist_to_tool is not None:
                    pose = self.wrist_pose(side) @ np.asarray(
                        static_wrist_to_tool[side], dtype=np.float64
                    )
                    output[f"{side}_static_tool_position_model"][frame] = pose[:3, 3]
                    output[f"{side}_static_tool_rotation_model"][frame] = pose[:3, :3]
                    output[f"{side}_static_tool_position_world"][frame] = (
                        self.model_to_world_position(pose[:3, 3])
                    )
                    output[f"{side}_static_tool_rotation_world"][frame] = (
                        self.model_to_world_rotation(pose[:3, :3])
                    )
            for contact in self.data.contact:
                if float(contact.dist) >= -abs(penetration_tolerance):
                    continue
                bodies = []
                geoms = []
                for geom in (int(contact.geom1), int(contact.geom2)):
                    geoms.append(
                        mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom
                        )
                        or f"geom_{geom}"
                    )
                    body_id = int(self.model.geom_bodyid[geom])
                    bodies.append(
                        mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                        or f"body_{body_id}"
                    )
                if bodies[0] == bodies[1]:
                    continue
                joined = "|".join(sorted(bodies))
                sides = {
                    side
                    for side in SIDES
                    if any(name.startswith(f"{side}_") for name in bodies)
                }
                # Adjacent or internal same-side finger contacts are not arm/body failures.
                if len(sides) == 1 and all("hand_" in name or "wrist" in name for name in bodies):
                    continue
                torso = any(
                    any(token in name for token in ("torso", "pelvis", "waist"))
                    for name in bodies
                )
                wrist_or_palm = any(
                    any(token in name for token in ("wrist", "hand_palm")) for name in bodies
                )
                arm_involved = any(
                    any(token in name for token in ("shoulder", "upper_arm", "elbow", "forearm", "wrist"))
                    for name in bodies
                )
                distal_fingers = all(
                    "hand_" in name
                    and any(token in name for token in ("thumb", "index", "middle"))
                    for name in bodies
                )
                if len(sides) == 2 and distal_fingers:
                    category = "DISTAL_HAND_HAND"
                elif len(sides) == 2:
                    category = "CROSS_ARM"
                elif torso and wrist_or_palm:
                    category = "WRIST_OR_PALM_TORSO"
                elif torso and arm_involved:
                    category = "ARM_TORSO"
                else:
                    category = "OTHER"
                category_flags[category][frame] = True
                pairs[category].add(joined)
                records.append(
                    {
                        "frame": frame,
                        "category": category,
                        "pair": joined,
                        "body_1": bodies[0],
                        "body_2": bodies[1],
                        "geom_1": geoms[0],
                        "geom_2": geoms[1],
                        "contact_position_model_m": np.asarray(
                            contact.pos, dtype=np.float64
                        ).copy(),
                        "contact_position_world_m": self.model_to_world_position(
                            np.asarray(contact.pos, dtype=np.float64)
                        ),
                        "contact_normal_model": np.asarray(
                            contact.frame[:3], dtype=np.float64
                        ).copy(),
                        "contact_normal_world": self.root_pose[:3, :3]
                        @ np.asarray(contact.frame[:3], dtype=np.float64),
                        "penetration_m": -float(contact.dist),
                    }
                )
        output["collision_flags"] = category_flags
        output["collision_pairs"] = {key: sorted(value) for key, value in pairs.items()}
        output["collision_records"] = records
        return output


__all__ = ["ALOHAKinematics", "G1Kinematics", "ALOHA_ARM_NAMES"]
