"""Active-model G1 arm and Dex3 kinematics used by the v15 translator."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import validate_g1_targets_and_sparse_ik as arm_ik


R_SCENE_FROM_MODEL = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
MODEL_ROOT = np.array([0.0, 0.0, 0.7922728583], dtype=np.float64)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: np.ndarray, fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > np.finfo(np.float64).eps:
        return value / norm
    fallback_array = np.asarray(fallback, dtype=np.float64)
    return fallback_array / np.linalg.norm(fallback_array)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


@dataclass(frozen=True)
class ContactSpec:
    label: str
    side: str
    role: str
    link: str
    joint_names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    limits: np.ndarray
    local_position: np.ndarray
    local_normal: np.ndarray
    half_extent: np.ndarray


class ActiveG1Dex3:
    """Name-mapped zero-dynamics FK for the active G1/Dex3 model."""

    def __init__(
        self,
        model_path: str | Path,
        mapping_path: str | Path,
        palm_path: str | Path,
        root_position: np.ndarray,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.mapping_path = Path(mapping_path).resolve()
        self.palm_path = Path(palm_path).resolve()
        self.info = arm_ik.validate_model(self.model_path)
        self.model: mujoco.MjModel = self.info["model"]
        self.data = mujoco.MjData(self.model)
        self.root_position = np.asarray(root_position, dtype=np.float64)
        mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        palm = json.loads(self.palm_path.read_text(encoding="utf-8"))
        if Path(mapping["active_model"]).resolve() != self.model_path:
            raise ValueError("Dex3 mapping does not reference the active model")
        if mapping["active_model_sha256"] != sha256_file(self.model_path):
            raise ValueError("active G1/Dex3 model hash mismatch")
        self.palm_offset = {
            side: np.asarray(palm[side]["geom_local_position_m"], dtype=np.float64)
            for side in ("left", "right")
        }
        self.contacts: dict[str, ContactSpec] = {}
        for side in ("left", "right"):
            for role in ("A", "B", "C"):
                row = mapping[side][role]
                joint_names = tuple(row["joint_names"])
                joint_ids = np.asarray([self.joint_id(name) for name in joint_names], dtype=np.int64)
                qpos_ids = np.asarray([self.model.jnt_qposadr[value] for value in joint_ids], dtype=np.int64)
                limits = np.asarray([self.model.jnt_range[value] for value in joint_ids], dtype=np.float64)
                self.contacts[f"{side}_{role}"] = ContactSpec(
                    label=f"{side}_{role}",
                    side=side,
                    role=role,
                    link=row["distal_link"],
                    joint_names=joint_names,
                    joint_ids=joint_ids,
                    qpos_ids=qpos_ids,
                    limits=limits,
                    local_position=np.asarray(row["local_position_xyz_m"], dtype=np.float64),
                    local_normal=np.asarray(row["local_normal"], dtype=np.float64),
                    half_extent=np.asarray(row["pad_half_extent_m"], dtype=np.float64),
                )
        self.hand_joint_names = {
            "left": tuple(
                mapping["left"][role]["joint_names"][index]
                for role in ("A", "B", "C")
                for index in range(len(mapping["left"][role]["joint_names"]))
            ),
            "right": tuple(
                mapping["right"][role]["joint_names"][index]
                for role in ("A", "B", "C")
                for index in range(len(mapping["right"][role]["joint_names"]))
            ),
        }
        self.hand_qpos_ids = {
            side: np.asarray(
                [self.model.jnt_qposadr[self.joint_id(name)] for name in names], dtype=np.int64
            )
            for side, names in self.hand_joint_names.items()
        }
        self.hand_limits = {
            side: np.asarray(
                [self.model.jnt_range[self.joint_id(name)] for name in names], dtype=np.float64
            )
            for side, names in self.hand_joint_names.items()
        }
        stand = np.asarray(self.info["stand_qpos"], dtype=np.float64)
        self.open_hand_q = {
            side: np.clip(
                stand[ids].copy(),
                self.hand_limits[side][:, 0] + 1e-8,
                self.hand_limits[side][:, 1] - 1e-8,
            )
            for side, ids in self.hand_qpos_ids.items()
        }
        self.body_ids = {
            name: self.body_id(name)
            for name in {
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                *(spec.link for spec in self.contacts.values()),
            }
        }

    def joint_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if value < 0:
            raise KeyError(f"active model missing joint {name}")
        return int(value)

    def body_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if value < 0:
            raise KeyError(f"active model missing body {name}")
        return int(value)

    def model_to_scene_position(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        return (R_SCENE_FROM_MODEL @ (value - MODEL_ROOT).T).T + self.root_position

    def scene_to_model_position(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        return (R_SCENE_FROM_MODEL.T @ (value - self.root_position).T).T + MODEL_ROOT

    @staticmethod
    def model_to_scene_rotation(value: np.ndarray) -> np.ndarray:
        return R_SCENE_FROM_MODEL @ np.asarray(value, dtype=np.float64)

    def assign(self, arm_q: np.ndarray, left_q: np.ndarray | None = None, right_q: np.ndarray | None = None) -> None:
        self.data.qpos[:] = self.info["stand_qpos"]
        self.data.qpos[self.info["arm_qpos_ids"]] = np.asarray(arm_q, dtype=np.float64)
        if left_q is not None:
            self.data.qpos[self.hand_qpos_ids["left"]] = np.asarray(left_q, dtype=np.float64)
        if right_q is not None:
            self.data.qpos[self.hand_qpos_ids["right"]] = np.asarray(right_q, dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def wrist_pose(self, side: str) -> np.ndarray:
        body = self.body_ids[f"{side}_wrist_yaw_link"]
        rotation = self.model_to_scene_rotation(self.data.xmat[body].reshape(3, 3))
        position = self.model_to_scene_position(self.data.xpos[body])
        return transform(rotation, position)

    def palm_pose(self, side: str) -> np.ndarray:
        wrist = self.wrist_pose(side)
        return transform(wrist[:3, :3], wrist[:3, 3] + wrist[:3, :3] @ self.palm_offset[side])

    def contact_pose(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        spec = self.contacts[label]
        body = self.body_ids[spec.link]
        rotation_model = self.data.xmat[body].reshape(3, 3)
        position_model = self.data.xpos[body] + rotation_model @ spec.local_position
        position = self.model_to_scene_position(position_model)
        normal = normalize(self.model_to_scene_rotation(rotation_model) @ spec.local_normal)
        return position, normal

    def palm_state(self, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = arm_ik.current_bimanual_state(self.info, self.data)
        position_model = state[f"{side}_pos"]
        rotation_model = Rotation.from_quat(state[f"{side}_quat"][[1, 2, 3, 0]]).as_matrix()
        jacobian = state[f"{side}_jac"].copy()
        jacobian[:3] = R_SCENE_FROM_MODEL @ jacobian[:3]
        jacobian[3:] = R_SCENE_FROM_MODEL @ jacobian[3:]
        return (
            self.model_to_scene_position(position_model),
            self.model_to_scene_rotation(rotation_model),
            jacobian,
        )

    def arm_joint_margin(self, arm_q: np.ndarray) -> float:
        limits = np.asarray(self.info["joint_limits"], dtype=np.float64)
        return float(np.min(np.minimum(arm_q - limits[:, 0], limits[:, 1] - arm_q)))

    def finger_joint_margin(self, side: str, q: np.ndarray) -> float:
        limits = self.hand_limits[side]
        return float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))

    def penetrating_contacts(self, tolerance: float = 1e-5) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for contact in self.data.contact:
            if float(contact.dist) >= -abs(tolerance):
                continue
            geom_names = []
            body_names = []
            for geom in (contact.geom1, contact.geom2):
                geom_names.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
                    or f"geom_{int(geom)}"
                )
                body = int(self.model.geom_bodyid[int(geom)])
                body_names.append(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
                    or f"body_{body}"
                )
            records.append({
                "distance_m": float(contact.dist),
                "geoms": geom_names,
                "bodies": body_names,
            })
        return records


def nearest_box_surface(point: np.ndarray, pose: np.ndarray, dimensions: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float64)
    local = pose[:3, :3].T @ (point - pose[:3, 3])
    half = 0.5 * np.asarray(dimensions, dtype=np.float64)
    outside = np.maximum(np.abs(local) - half, 0.0)
    if np.any(outside > 0.0):
        nearest_local = np.clip(local, -half, half)
        signed = float(np.linalg.norm(local - nearest_local))
    else:
        margins = half - np.abs(local)
        axis = int(np.argmin(margins))
        nearest_local = local.copy()
        nearest_local[axis] = np.copysign(half[axis], local[axis] if local[axis] else 1.0)
        signed = -float(margins[axis])
    nearest = pose[:3, 3] + pose[:3, :3] @ nearest_local
    return signed, nearest, local


def ring_material_gap(
    point: np.ndarray,
    pose: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    depth: float,
) -> tuple[float, np.ndarray]:
    local = pose[:3, :3].T @ (np.asarray(point, dtype=np.float64) - pose[:3, 3])
    radius = float(np.linalg.norm(local[[0, 2]]))
    axial = abs(float(local[1]))
    radial_gap = max(inner_radius - radius, radius - outer_radius, 0.0)
    axial_gap = max(axial - 0.5 * depth, 0.0)
    if radial_gap > 0.0 or axial_gap > 0.0:
        return float(np.hypot(radial_gap, axial_gap)), local
    penetration = min(radius - inner_radius, outer_radius - radius, 0.5 * depth - axial)
    return -float(max(0.0, penetration)), local
