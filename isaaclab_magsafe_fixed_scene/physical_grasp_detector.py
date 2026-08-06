"""Conservative contact-based grasp and physical-task state detectors."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class GraspState(str, Enum):
    NONE = "NONE"
    CONTACT_CANDIDATE = "CONTACT_CANDIDATE"
    BILATERAL_CONTACT = "BILATERAL_CONTACT"
    STABLE_GRASP = "STABLE_GRASP"
    SLIPPING = "SLIPPING"
    RELEASED = "RELEASED"


class TaskPhase(str, Enum):
    INITIAL_STABLE = "INITIAL_STABLE"
    ACCESSORY_APPROACH = "ACCESSORY_APPROACH"
    ACCESSORY_CONTACT = "ACCESSORY_CONTACT"
    ACCESSORY_GRASP = "ACCESSORY_GRASP"
    ACCESSORY_PULL = "ACCESSORY_PULL"
    ACCESSORY_DETACH = "ACCESSORY_DETACH"
    PHONE_APPROACH = "PHONE_APPROACH"
    PHONE_CONTACT = "PHONE_CONTACT"
    PHONE_GRASP = "PHONE_GRASP"
    PHONE_MOVE = "PHONE_MOVE"
    CHARGER_APPROACH = "CHARGER_APPROACH"
    PHONE_RELEASE = "PHONE_RELEASE"
    MAGNETIC_CAPTURE = "MAGNETIC_CAPTURE"
    FINAL_ATTACHED = "FINAL_ATTACHED"
    FAILED = "FAILED"


@dataclass
class GraspObservation:
    frame: int
    state: GraspState
    total_force: float
    aperture: float
    relative_speed: float
    slip: float
    opposing_dot: float
    pad_groups: tuple[str, ...]


class GraspDetector:
    """Require two pad groups, opposing normals, force, dwell and low slip."""

    FIELDS = [
        "sim_time", "source_frame", "robot", "gripper", "object", "state",
        "contact_count", "pad_groups", "normal_force", "aperture",
        "relative_velocity", "accumulated_slip", "opposing_normal_dot",
    ]

    def __init__(
        self,
        output_csv: str | Path,
        *,
        robot: str,
        physics_dt: float,
        force_threshold: float = 1.0,
        stable_time: float = 0.20,
        max_relative_speed: float = 0.05,
        max_slip: float = 0.010,
    ):
        self.robot = robot
        self.dt = physics_dt
        self.force_threshold = force_threshold
        self.stable_steps = max(1, round(stable_time / physics_dt))
        self.max_relative_speed = max_relative_speed
        self.max_slip = max_slip
        self.states: dict[tuple[str, str], GraspState] = {}
        self.dwell: dict[tuple[str, str], int] = {}
        self.slip: dict[tuple[str, str], float] = {}
        self.events: dict[str, int] = {}
        self.path = Path(output_csv)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()

    @staticmethod
    def _pad_group(path: str) -> str:
        lower = path.lower()
        if "carriage_left" in lower or "thumb" in lower:
            return "opposing_a"
        if "carriage_right" in lower or "index" in lower or "middle" in lower:
            return "opposing_b"
        return "other"

    def update(
        self,
        *,
        sim_time: float,
        frame: int,
        gripper: str,
        object_name: str,
        contacts,
        aperture: float,
        relative_speed: float,
    ) -> GraspObservation:
        key = (gripper, object_name)
        relevant = [
            c for c in contacts
            if gripper in c.sensor_prim and object_name.lower() in c.other_prim.lower()
        ]
        groups = tuple(sorted({self._pad_group(c.sensor_prim) for c in relevant if self._pad_group(c.sensor_prim) != "other"}))
        total_force = sum(c.normal_force for c in relevant)
        opposing_dot = 1.0
        for i, a in enumerate(relevant):
            for b in relevant[i + 1:]:
                if self._pad_group(a.sensor_prim) != self._pad_group(b.sensor_prim):
                    opposing_dot = min(opposing_dot, float(np.dot(a.normal, b.normal)))
        bilateral = {"opposing_a", "opposing_b"}.issubset(groups)
        force_ok = total_force >= self.force_threshold
        speed_ok = relative_speed <= self.max_relative_speed
        opposing = opposing_dot <= -0.3
        previous = self.states.get(key, GraspState.NONE)
        slip = self.slip.get(key, 0.0)
        if relevant:
            slip += max(0.0, relative_speed) * self.dt
        if not relevant:
            state = GraspState.RELEASED if previous not in (GraspState.NONE, GraspState.RELEASED) else GraspState.NONE
            self.dwell[key] = 0
            slip = 0.0
        elif not bilateral:
            state = GraspState.CONTACT_CANDIDATE
            self.dwell[key] = 0
        elif not opposing or not force_ok:
            state = GraspState.BILATERAL_CONTACT
            self.dwell[key] = 0
        else:
            self.dwell[key] = self.dwell.get(key, 0) + int(speed_ok)
            if slip > self.max_slip:
                state = GraspState.SLIPPING
            elif self.dwell[key] >= self.stable_steps:
                state = GraspState.STABLE_GRASP
            else:
                state = GraspState.BILATERAL_CONTACT
        self.states[key] = state
        self.slip[key] = slip
        event_key = f"{gripper}:{object_name}:{state.value}"
        self.events.setdefault(event_key, frame)
        self.writer.writerow(
            {
                "sim_time": f"{sim_time:.9f}", "source_frame": frame, "robot": self.robot,
                "gripper": gripper, "object": object_name, "state": state.value,
                "contact_count": len(relevant), "pad_groups": "|".join(groups),
                "normal_force": f"{total_force:.9g}", "aperture": f"{aperture:.9g}",
                "relative_velocity": f"{relative_speed:.9g}", "accumulated_slip": f"{slip:.9g}",
                "opposing_normal_dot": f"{opposing_dot:.9g}",
            }
        )
        return GraspObservation(frame, state, total_force, aperture, relative_speed, slip, opposing_dot, groups)

    def close(self) -> None:
        self.file.flush()
        self.file.close()


class TaskPhaseDetector:
    """Monotonic task phase detector driven by measured physical events."""

    ORDER = list(TaskPhase)[:-1]

    def __init__(self):
        self.phase = TaskPhase.INITIAL_STABLE
        self.transitions: list[tuple[int, str]] = [(0, self.phase.value)]

    def advance(self, frame: int, phase: TaskPhase) -> None:
        if phase == TaskPhase.FAILED:
            self.phase = phase
        elif self.ORDER.index(phase) > self.ORDER.index(self.phase):
            self.phase = phase
        else:
            return
        self.transitions.append((frame, self.phase.value))

