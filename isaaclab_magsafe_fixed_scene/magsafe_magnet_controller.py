"""MagSafe state machines and bounded spring-damper wrench controller."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np


class MagnetState(str, Enum):
    DETACHED = "DETACHED"
    ATTRACTING = "ATTRACTING"
    ATTACHED = "ATTACHED"
    COOLDOWN = "COOLDOWN"


@dataclass
class BodyState:
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


@dataclass
class PairResult:
    state: MagnetState
    force: np.ndarray
    torque: np.ndarray
    distance: float
    angle_deg: float
    linear_speed: float
    angular_speed: float
    attach_event: bool = False
    detach_event: bool = False
    penetration: float = 0.0
    invalid: bool = False
    pad_normal_angle_deg: float = 0.0


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v + 2.0 * (w * np.cross(qv, v) + np.cross(qv, np.cross(qv, v)))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def clamp_norm(v: np.ndarray, maximum: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= maximum or n < 1e-12 else v * (maximum / n)


class MagneticPair:
    def __init__(
        self,
        name: str,
        cfg: dict,
        source_frame_local: np.ndarray,
        source_normal_local: np.ndarray,
        target_frame_local: np.ndarray,
        target_normal_local: np.ndarray,
        source_mass: float,
        max_acceleration: float,
        initial_state: MagnetState | None = None,
        target_orientation_world: np.ndarray | None = None,
    ):
        self.name = name
        self.cfg = cfg
        self.source_frame_local = source_frame_local
        self.source_normal_local = source_normal_local
        self.target_frame_local = target_frame_local
        self.target_normal_local = target_normal_local
        self.source_mass = source_mass
        self.max_acceleration = max_acceleration
        self.state = initial_state or MagnetState(cfg["initial_state"])
        self.cooldown_until = 0.0
        self.target_orientation_world = target_orientation_world

    def _frame(self, body: BodyState, local_p: np.ndarray, local_n: np.ndarray):
        return body.position + quat_rotate(body.quaternion_wxyz, local_p), quat_rotate(body.quaternion_wxyz, local_n)

    def update(
        self,
        now: float,
        source: BodyState,
        target: BodyState,
        *,
        external_force_norm: float = 0.0,
        external_torque_norm: float = 0.0,
        penetration: float = 0.0,
        correct_face: bool = True,
    ) -> PairResult:
        sp, sn = self._frame(source, self.source_frame_local, self.source_normal_local)
        tp, tn = self._frame(target, self.target_frame_local, self.target_normal_local)
        ep = tp - sp
        distance = float(np.linalg.norm(ep))
        vrel = target.linear_velocity - source.linear_velocity
        wrel = target.angular_velocity - source.angular_velocity
        # Facing normals must oppose one another at a mating interface.
        desired = -tn
        normal_angle = math.degrees(math.acos(float(np.clip(np.dot(sn, desired), -1.0, 1.0))))
        if self.target_orientation_world is not None:
            desired_q = self.target_orientation_world / np.linalg.norm(self.target_orientation_world)
            source_q = source.quaternion_wxyz / np.linalg.norm(source.quaternion_wxyz)
            qerr = quat_multiply(desired_q, np.array([source_q[0], -source_q[1], -source_q[2], -source_q[3]]))
            if qerr[0] < 0.0:
                qerr = -qerr
            angle = 2.0 * math.acos(float(np.clip(qerr[0], -1.0, 1.0)))
            s = float(np.linalg.norm(qerr[1:]))
            e_r = qerr[1:] / s * angle if s > 1e-9 else np.zeros(3)
        else:
            cross = np.cross(sn, desired)
            dot = float(np.clip(np.dot(sn, desired), -1.0, 1.0))
            angle = math.acos(dot)
            axis_norm = float(np.linalg.norm(cross))
            e_r = cross / axis_norm * angle if axis_norm > 1e-9 else np.zeros(3)
        attach_event = detach_event = False

        if self.state == MagnetState.COOLDOWN and now >= self.cooldown_until:
            self.state = MagnetState.DETACHED
        if self.state == MagnetState.DETACHED and distance < self.cfg["capture_radius_m"] and correct_face:
            self.state = MagnetState.ATTRACTING
        elif self.state == MagnetState.ATTRACTING and (distance >= self.cfg["capture_radius_m"] or not correct_face):
            self.state = MagnetState.DETACHED
        if self.state == MagnetState.ATTRACTING:
            ready = (
                distance < self.cfg["attach_distance_m"]
                and math.degrees(angle) < self.cfg["attach_angle_deg"]
                and np.linalg.norm(vrel) < self.cfg["attach_linear_speed_mps"]
                and np.linalg.norm(wrel) < self.cfg["attach_angular_speed_rps"]
                and penetration <= 0.002
                and correct_face
            )
            if ready:
                self.state = MagnetState.ATTACHED
                attach_event = True
        if self.state == MagnetState.ATTACHED and (
            external_force_norm > self.cfg["break_force_n"] or external_torque_norm > self.cfg["break_torque_nm"]
        ):
            self.state = MagnetState.COOLDOWN
            self.cooldown_until = now + self.cfg["cooldown_s"]
            detach_event = True

        active = self.state in (MagnetState.ATTRACTING, MagnetState.ATTACHED)
        force = np.zeros(3)
        torque = np.zeros(3)
        if active:
            x = np.clip(1.0 - distance / self.cfg["capture_radius_m"], 0.0, 1.0)
            falloff = x * x * (3.0 - 2.0 * x)
            if self.state == MagnetState.ATTACHED:
                falloff = 1.0
            # vrel/omega_rel are target minus source. Therefore a stable
            # damping term has a plus sign with this convention (equivalent to
            # ``-kd * (source_velocity - target_velocity)``).
            force = falloff * (self.cfg["kp_position"] * ep + self.cfg["kd_position"] * vrel)
            torque = falloff * (self.cfg["kp_orientation"] * e_r + self.cfg["kd_orientation"] * wrel)
            force = clamp_norm(force, min(self.cfg["max_force_n"], self.source_mass * self.max_acceleration))
            torque = clamp_norm(torque, self.cfg["max_torque_nm"])
        invalid = not np.all(np.isfinite(np.r_[force, torque, distance, angle]))
        if invalid:
            force[:] = torque[:] = 0.0
        return PairResult(
            self.state, force, torque, distance, math.degrees(angle),
            float(np.linalg.norm(vrel)), float(np.linalg.norm(wrel)),
            attach_event, detach_event, penetration, invalid, normal_angle,
        )


class MagnetCsvLogger:
    FIELDS = [
        "simulation_time", "pair_name", "state", "center_distance", "normal_angular_error_deg",
        "relative_linear_speed", "relative_angular_speed", "force_norm", "torque_norm",
        "attach_event", "detach_event", "joint_state", "penetration", "nan_inf",
        "pad_normal_error_deg", "phone_long_axis_x", "phone_long_axis_y", "phone_long_axis_z",
        "phone_distance_to_charger_base",
    ]

    def __init__(self, path: str | Path):
        self.file = Path(path).open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    def write(self, t: float, name: str, r: PairResult, joint_state: str) -> None:
        self.writer.writerow({
            "simulation_time": f"{t:.9f}", "pair_name": name, "state": r.state.value,
            "center_distance": f"{r.distance:.9g}", "normal_angular_error_deg": f"{r.angle_deg:.7g}",
            "relative_linear_speed": f"{r.linear_speed:.7g}", "relative_angular_speed": f"{r.angular_speed:.7g}",
            "force_norm": f"{np.linalg.norm(r.force):.7g}", "torque_norm": f"{np.linalg.norm(r.torque):.7g}",
            "attach_event": int(r.attach_event), "detach_event": int(r.detach_event),
            "joint_state": joint_state, "penetration": f"{r.penetration:.7g}", "nan_inf": int(r.invalid),
            "pad_normal_error_deg": f"{r.pad_normal_angle_deg:.7g}",
            "phone_long_axis_x": f"{getattr(r, 'long_axis', [float('nan')]*3)[0]:.7g}",
            "phone_long_axis_y": f"{getattr(r, 'long_axis', [float('nan')]*3)[1]:.7g}",
            "phone_long_axis_z": f"{getattr(r, 'long_axis', [float('nan')]*3)[2]:.7g}",
            "phone_distance_to_charger_base": f"{getattr(r, 'base_distance', float('nan')):.7g}",
        })
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
