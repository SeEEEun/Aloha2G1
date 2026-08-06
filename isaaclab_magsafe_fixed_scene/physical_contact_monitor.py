"""Pair-resolved PhysX contact measurements for robot MagSafe replays.

This module only authors contact-report APIs into the replay composition layer.
It never edits a referenced robot or scene asset.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


CONTACT_FIELDS = [
    "sim_time",
    "source_frame",
    "task_phase",
    "robot",
    "contact_pair",
    "sensor_prim",
    "other_prim",
    "contact_x",
    "contact_y",
    "contact_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "normal_force",
    "tangential_force",
    "friction_force_x",
    "friction_force_y",
    "friction_force_z",
    "contact_duration",
    "relative_velocity",
    "separation",
    "penetration",
    "gripper_aperture",
    "grasp_state",
    "accessory_attached",
    "phone_charger_state",
    "phone_position",
    "phone_orientation",
]


def enable_contact_reporting(stage, root_paths: Iterable[str], threshold: float = 0.0) -> list[str]:
    """Apply PhysxContactReportAPI to rigid bodies below replay-layer roots."""
    from pxr import PhysxSchema, UsdPhysics

    enabled: list[str] = []
    seen: set[str] = set()
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root:
            continue
        pending = [root]
        while pending:
            prim = pending.pop(0)
            pending.extend(prim.GetChildren())
            path = str(prim.GetPath())
            if path in seen or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            seen.add(path)
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(float(threshold))
            enabled.append(path)
    stage.GetRootLayer().Save()
    return enabled


def select_rigid_bodies(stage, root_paths: Iterable[str], predicate: Callable[[str], bool]) -> list[str]:
    """Return rigid-body paths below roots which satisfy a path predicate."""
    from pxr import UsdPhysics

    result: list[str] = []
    for root_path in root_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root:
            continue
        pending = [root]
        while pending:
            prim = pending.pop(0)
            pending.extend(prim.GetChildren())
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and predicate(path):
                result.append(path)
    return sorted(set(result))


@dataclass
class ContactSample:
    sim_time: float
    source_frame: int
    task_phase: str
    sensor_prim: str
    other_prim: str
    point: np.ndarray
    normal: np.ndarray
    normal_force: float
    tangential_force: float
    friction_force: np.ndarray
    separation: float
    penetration: float
    duration: float
    relative_velocity: float
    aperture: float

    @property
    def pair(self) -> str:
        return f"{self.sensor_prim} <-> {self.other_prim}"


class PhysicalContactMonitor:
    """Read raw PhysX contacts, preserving signed separation per contact point."""

    def __init__(
        self,
        sensor,
        filter_paths: list[str],
        output_csv: str | Path,
        *,
        robot: str,
        physics_dt: float,
        relevant_pair: Callable[[str, str], bool],
        relative_velocity: Callable[[str, str], float] | None = None,
    ):
        self.sensor = sensor
        self.filter_paths = list(filter_paths)
        self.sensor_paths: list[str] = []
        self.robot = robot
        self.dt = float(physics_dt)
        self.relevant_pair = relevant_pair
        self.relative_velocity_fn = relative_velocity
        self.output_path = Path(output_csv)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CONTACT_FIELDS)
        self._writer.writeheader()
        self._active_duration: dict[tuple[str, str], float] = {}
        self.latest: list[ContactSample] = []
        self.max_penetration = 0.0
        self.max_normal_force = 0.0
        self.contact_count = 0
        self.api_error: str | None = None
        self.tangential_force_method = "PhysX_RigidContactView.get_friction_data"

    @staticmethod
    def _numpy(value) -> np.ndarray:
        if hasattr(value, "numpy"):
            return np.asarray(value.numpy())
        if hasattr(value, "torch"):
            return value.torch.detach().cpu().numpy()
        return np.asarray(value)

    def sample(
        self,
        *,
        sim_time: float,
        source_frame: int,
        task_phase: str,
        apertures: dict[str, float],
        grasp_states: dict[str, str],
        accessory_attached: bool,
        phone_charger_state: str,
        phone_position: np.ndarray,
        phone_orientation: np.ndarray,
    ) -> list[ContactSample]:
        try:
            view = self.sensor.contact_view
            self.sensor_paths = list(self.sensor.body_physx_view.prim_paths[: self.sensor.num_sensors])
            forces, points, normals, separations, counts, starts = view.get_contact_data(self.dt)
            forces = self._numpy(forces).reshape(-1)
            points = self._numpy(points).reshape(-1, 3)
            normals = self._numpy(normals).reshape(-1, 3)
            separations = self._numpy(separations).reshape(-1)
            counts = self._numpy(counts).reshape(len(self.sensor_paths), -1).astype(np.int64)
            starts = self._numpy(starts).reshape(len(self.sensor_paths), -1).astype(np.int64)
            friction, _, friction_counts, friction_starts = view.get_friction_data(self.dt)
            friction = self._numpy(friction).reshape(-1, 3)
            friction_counts = self._numpy(friction_counts).reshape(len(self.sensor_paths), -1).astype(np.int64)
            friction_starts = self._numpy(friction_starts).reshape(len(self.sensor_paths), -1).astype(np.int64)
        except Exception as exc:
            self.api_error = f"{type(exc).__name__}: {exc}"
            raise

        current_keys: set[tuple[str, str]] = set()
        samples: list[ContactSample] = []
        phone_position_text = " ".join(f"{v:.9g}" for v in phone_position)
        phone_orientation_text = " ".join(f"{v:.9g}" for v in phone_orientation)
        for sensor_index, owner in enumerate(self.sensor_paths):
            for filter_index in range(counts.shape[1]):
                other = (
                    self.filter_paths[filter_index]
                    if filter_index < len(self.filter_paths)
                    else f"FILTER_{filter_index}"
                )
                friction_start = int(friction_starts[sensor_index, filter_index])
                friction_count = int(friction_counts[sensor_index, filter_index])
                pair_tangent = float(
                    np.linalg.norm(
                        np.sum(friction[friction_start:friction_start + friction_count], axis=0)
                    )
                )
                pair_friction = np.sum(
                    friction[friction_start:friction_start + friction_count], axis=0
                ).astype(float)
                start = int(starts[sensor_index, filter_index])
                count = int(counts[sensor_index, filter_index])
                for contact_index in range(start, start + count):
                    if not self.relevant_pair(owner, other):
                        continue
                    key = tuple(sorted((owner, other)))
                    current_keys.add(key)
                    duration = self._active_duration.get(key, 0.0) + self.dt
                    self._active_duration[key] = duration
                    separation = float(separations[contact_index])
                    penetration = max(0.0, -separation)
                    normal_force = abs(float(forces[contact_index]))
                    rel_vel = (
                        float(self.relative_velocity_fn(owner, other))
                        if self.relative_velocity_fn is not None
                        else math.nan
                    )
                    aperture = next((v for k, v in apertures.items() if k in owner), math.nan)
                    grasp_state = next((v for k, v in grasp_states.items() if k in owner), "NONE")
                    sample = ContactSample(
                        sim_time,
                        int(source_frame),
                        task_phase,
                        owner,
                        other,
                        points[contact_index].astype(float),
                        normals[contact_index].astype(float),
                        normal_force,
                        pair_tangent,
                        pair_friction,
                        separation,
                        penetration,
                        duration,
                        rel_vel,
                        aperture,
                    )
                    samples.append(sample)
                    self._writer.writerow(
                {
                    "sim_time": f"{sim_time:.9f}",
                    "source_frame": source_frame,
                    "task_phase": task_phase,
                    "robot": self.robot,
                    "contact_pair": sample.pair,
                    "sensor_prim": owner,
                    "other_prim": other,
                    "contact_x": f"{sample.point[0]:.9g}",
                    "contact_y": f"{sample.point[1]:.9g}",
                    "contact_z": f"{sample.point[2]:.9g}",
                    "normal_x": f"{sample.normal[0]:.9g}",
                    "normal_y": f"{sample.normal[1]:.9g}",
                    "normal_z": f"{sample.normal[2]:.9g}",
                    "normal_force": f"{normal_force:.9g}",
                    "tangential_force": f"{pair_tangent:.9g}",
                    "friction_force_x": f"{pair_friction[0]:.9g}",
                    "friction_force_y": f"{pair_friction[1]:.9g}",
                    "friction_force_z": f"{pair_friction[2]:.9g}",
                    "contact_duration": f"{duration:.9g}",
                    "relative_velocity": f"{rel_vel:.9g}",
                    "separation": f"{separation:.9g}",
                    "penetration": f"{penetration:.9g}",
                    "gripper_aperture": f"{aperture:.9g}",
                    "grasp_state": grasp_state,
                    "accessory_attached": int(accessory_attached),
                    "phone_charger_state": phone_charger_state,
                    "phone_position": phone_position_text,
                    "phone_orientation": phone_orientation_text,
                }
            )
                    self.max_penetration = max(self.max_penetration, penetration)
                    self.max_normal_force = max(self.max_normal_force, normal_force)
                    self.contact_count += 1
        for key in list(self._active_duration):
            if key not in current_keys:
                del self._active_duration[key]
        self.latest = samples
        return samples

    def close(self) -> None:
        self._file.flush()
        self._file.close()
