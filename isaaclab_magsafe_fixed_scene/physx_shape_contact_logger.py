"""Exact collider/material contact logging from PhysX contact-report callbacks."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


FIELDS = [
    "sim_time", "source_frame", "event_type", "actor0", "actor1",
    "collider0", "collider1", "material0", "material1",
    "contact_x", "contact_y", "contact_z", "normal_x", "normal_y", "normal_z",
    "impulse_x", "impulse_y", "impulse_z", "normal_force", "tangential_force",
    "friction_force_x", "friction_force_y", "friction_force_z",
    "phone_local_contact_x", "phone_local_contact_y", "phone_local_contact_z",
    "phone_local_normal_x", "phone_local_normal_y", "phone_local_normal_z",
    "pad_velocity_x", "pad_velocity_y", "pad_velocity_z",
    "phone_surface_velocity_x", "phone_surface_velocity_y", "phone_surface_velocity_z",
    "relative_tangent_velocity_x", "relative_tangent_velocity_y", "relative_tangent_velocity_z",
    "separation", "penetration",
]


class PhysxShapeContactLogger:
    """Capture exact shape and material paths unavailable from tensor contact views."""

    def __init__(self, path: str | Path, *, physics_dt: float, start_frame: int, end_frame: int):
        import omni.physx

        self.dt = float(physics_dt)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.sim_time = 0.0
        self.source_frame = 0
        self.file = Path(path).open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.file.flush()
        self.latest: list[dict] = []
        self.body_states: dict[str, dict[str, np.ndarray]] = {}
        self.api_error: str | None = None
        self.subscription = (
            omni.physx.get_physx_simulation_interface()
            .subscribe_contact_report_events(self._callback)
        )

    @staticmethod
    def _path(encoded) -> str:
        from pxr import PhysicsSchemaTools

        return str(PhysicsSchemaTools.intToSdfPath(encoded))

    def set_context(
        self,
        sim_time: float,
        source_frame: int,
        body_states: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.sim_time = float(sim_time)
        self.source_frame = int(source_frame)
        self.body_states = body_states or {}
        self.latest = []

    @staticmethod
    def _rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = quaternion / np.linalg.norm(quaternion)
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    def _point_velocity(self, actor: str, point: np.ndarray) -> np.ndarray:
        state = self.body_states.get(actor)
        if state is None:
            return np.full(3, np.nan)
        return state["linear"] + np.cross(state["angular"], point - state["position"])

    def _callback(self, headers, data) -> None:
        try:
            self._callback_impl(headers, data)
        except Exception as exc:
            self.api_error = f"{type(exc).__name__}: {exc}"

    def _callback_impl(self, headers, data) -> None:
        if not self.start_frame <= self.source_frame <= self.end_frame:
            return
        for header in headers:
            actor0, actor1 = self._path(header.actor0), self._path(header.actor1)
            collider0, collider1 = self._path(header.collider0), self._path(header.collider1)
            combined = (collider0 + " " + collider1).lower()
            if "phone" not in combined:
                continue
            if not any(token in combined for token in ("stationaryaloha", "table", "accessory")):
                continue
            for index in range(
                header.contact_data_offset,
                header.contact_data_offset + header.num_contact_data,
            ):
                item = data[index]
                point = np.asarray(item.position, dtype=float)
                normal = np.asarray(item.normal, dtype=float)
                impulse = np.asarray(item.impulse, dtype=float)
                normal_impulse = abs(float(np.dot(impulse, normal)))
                tangent_vector = impulse - np.dot(impulse, normal) * normal
                tangent_impulse = float(np.linalg.norm(tangent_vector))
                friction_force = tangent_vector / self.dt
                phone_is_0 = "/Phone" in actor0
                phone_actor = actor0 if phone_is_0 else actor1
                pad_actor = actor1 if phone_is_0 else actor0
                phone_state = self.body_states.get(phone_actor)
                pad_velocity = self._point_velocity(pad_actor, point)
                phone_velocity = self._point_velocity(phone_actor, point)
                relative_velocity = pad_velocity - phone_velocity
                relative_tangent = relative_velocity - np.dot(relative_velocity, normal) * normal
                if phone_state is not None:
                    phone_rotation = self._rotation_wxyz(phone_state["quaternion"])
                    phone_local_contact = phone_rotation.T @ (point - phone_state["position"])
                    phone_local_normal = phone_rotation.T @ normal
                else:
                    phone_local_contact = np.full(3, np.nan)
                    phone_local_normal = np.full(3, np.nan)
                separation = float(item.separation)
                row = {
                    "sim_time": f"{self.sim_time:.9f}",
                    "source_frame": self.source_frame,
                    "event_type": str(header.type),
                    "actor0": actor0, "actor1": actor1,
                    "collider0": collider0, "collider1": collider1,
                    "material0": self._path(item.material0),
                    "material1": self._path(item.material1),
                    "contact_x": f"{point[0]:.9g}", "contact_y": f"{point[1]:.9g}",
                    "contact_z": f"{point[2]:.9g}",
                    "normal_x": f"{normal[0]:.9g}", "normal_y": f"{normal[1]:.9g}",
                    "normal_z": f"{normal[2]:.9g}",
                    "impulse_x": f"{impulse[0]:.9g}", "impulse_y": f"{impulse[1]:.9g}",
                    "impulse_z": f"{impulse[2]:.9g}",
                    "normal_force": f"{normal_impulse / self.dt:.9g}",
                    "tangential_force": f"{tangent_impulse / self.dt:.9g}",
                    "friction_force_x": f"{friction_force[0]:.9g}",
                    "friction_force_y": f"{friction_force[1]:.9g}",
                    "friction_force_z": f"{friction_force[2]:.9g}",
                    "phone_local_contact_x": f"{phone_local_contact[0]:.9g}",
                    "phone_local_contact_y": f"{phone_local_contact[1]:.9g}",
                    "phone_local_contact_z": f"{phone_local_contact[2]:.9g}",
                    "phone_local_normal_x": f"{phone_local_normal[0]:.9g}",
                    "phone_local_normal_y": f"{phone_local_normal[1]:.9g}",
                    "phone_local_normal_z": f"{phone_local_normal[2]:.9g}",
                    "pad_velocity_x": f"{pad_velocity[0]:.9g}",
                    "pad_velocity_y": f"{pad_velocity[1]:.9g}",
                    "pad_velocity_z": f"{pad_velocity[2]:.9g}",
                    "phone_surface_velocity_x": f"{phone_velocity[0]:.9g}",
                    "phone_surface_velocity_y": f"{phone_velocity[1]:.9g}",
                    "phone_surface_velocity_z": f"{phone_velocity[2]:.9g}",
                    "relative_tangent_velocity_x": f"{relative_tangent[0]:.9g}",
                    "relative_tangent_velocity_y": f"{relative_tangent[1]:.9g}",
                    "relative_tangent_velocity_z": f"{relative_tangent[2]:.9g}",
                    "separation": f"{separation:.9g}",
                    "penetration": f"{max(0.0, -separation):.9g}",
                }
                self.writer.writerow(row)
                self.latest.append(row)

    def close(self) -> None:
        self.subscription = None
        self.file.flush()
        self.file.close()
