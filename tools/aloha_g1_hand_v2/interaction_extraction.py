"""Authoritative source-gripper event and contact reconstruction.

No object pose is inferred here.  With the current source dataset, contact
geometry is reconstructed only in the ALOHA TCP frame from the two named tip
collision boxes in the authoritative stationary ALOHA model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from aloha_g1_dataset_v1 import core as v1


REPRESENTATION_FALLBACK = "SOURCE_TCP_RELATIVE_CONTACT_GEOMETRY_FALLBACK"
REPRESENTATION_OBJECT = "OBJECT_RELATIVE_CONTACT_GEOMETRY"


def _runs(labels: np.ndarray) -> list[dict[str, int | str]]:
    values = np.asarray(labels).astype(str)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("semantic phase labels must be a non-empty vector")
    starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    return [
        {"phase": str(values[start]), "start_frame": int(start), "end_frame": int(end)}
        for start, end in zip(starts, ends)
    ]


def detect_first_complete_grasp(labels: np.ndarray) -> dict[str, int | None | str]:
    """Select the first signal-derived OPEN→PREGRASP→GRASP→HOLD cycle.

    RELEASE is the first release run following that HOLD and may be absent if
    the demonstration ends while still holding.  No frame or episode ID is an
    input to this state machine.
    """
    runs = _runs(labels)
    for index in range(2, len(runs) - 1):
        phases = [str(row["phase"]) for row in runs[index - 2 : index + 2]]
        if phases == ["OPEN", "PREGRASP", "GRASP", "HOLD"]:
            release = next(
                (
                    int(row["start_frame"])
                    for row in runs[index + 2 :]
                    if str(row["phase"]) == "RELEASE"
                ),
                None,
            )
            return {
                "selection_rule": "first_complete_OPEN_PREGRASP_GRASP_HOLD_cycle",
                "pregrasp_start": int(runs[index - 1]["start_frame"]),
                "grasp_onset": int(runs[index]["start_frame"]),
                "hold_start": int(runs[index + 1]["start_frame"]),
                "release": release,
            }
    raise ValueError("no complete OPEN→PREGRASP→GRASP→HOLD semantic cycle")


@dataclass(frozen=True)
class AlohaPadSpec:
    label: str
    geom_name: str
    geom_id: int


class AlohaContactExtractor:
    """Extract opposing jaw-tip surface points from authoritative model FK."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve()
        self.model, _ = v1.aloha_fk.load_validated_model(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.tcp_offset = np.asarray(v1.aloha_fk.TCP_OFFSET_LOCAL, dtype=np.float64)
        self._terminal = {
            side: self._body_id(f"follower_{side}_link_6") for side in ("left", "right")
        }
        self._pads = {
            side: (
                AlohaPadSpec(
                    "contact_A_tcp",
                    f"follower_{side}_gripper_right_tip",
                    self._geom_id(f"follower_{side}_gripper_right_tip"),
                ),
                AlohaPadSpec(
                    "contact_B_tcp",
                    f"follower_{side}_gripper_left_tip",
                    self._geom_id(f"follower_{side}_gripper_left_tip"),
                ),
            )
            for side in ("left", "right")
        }

    def _body_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if value < 0:
            raise KeyError(name)
        return int(value)

    def _geom_id(self, name: str) -> int:
        value = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if value < 0:
            raise KeyError(name)
        return int(value)

    def _assign(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        row = np.asarray(action, dtype=np.float64)
        if row.shape != (14,) or not np.isfinite(row).all():
            raise ValueError("source action must be finite shape [14]")
        mapped, clipped_count = v1.aloha_fk.mapped_qpos(row[None, :])
        self.data.qpos[:] = 0.0
        self.data.qpos[:16] = mapped[0]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return mapped[0], bool(clipped_count)

    def _inner_surface_point_world(self, geom_id: int, tcp_world: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64).copy()
        rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy()
        delta_local = rotation.T @ (tcp_world - center)
        # The named tips are boxes.  Select the face whose normal points toward
        # the TCP center; this derives both face axis and sign from model data.
        if int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise RuntimeError("ALOHA named gripper tip must be a box geom")
        axis = int(np.argmax(np.abs(delta_local)))
        sign = 1.0 if delta_local[axis] >= 0.0 else -1.0
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = sign * float(self.model.geom_size[geom_id, axis])
        point = center + rotation @ offset
        return point, {
            "box_face_axis": axis,
            "box_face_sign": sign,
            "box_half_extent_m": float(self.model.geom_size[geom_id, axis]),
            "geom_center_world_m": center,
            "surface_offset_geom_m": offset,
        }

    def extract(
        self,
        action: np.ndarray,
        side: str,
        *,
        authoritative_object_pose: np.ndarray | None = None,
        object_pose_provenance: str | None = None,
    ) -> dict[str, Any]:
        if side not in self._pads:
            raise ValueError(side)
        if authoritative_object_pose is not None and not object_pose_provenance:
            raise ValueError("object-frame contact extraction requires explicit authoritative provenance")
        if authoritative_object_pose is not None:
            raise NotImplementedError(
                "OBJECT_RELATIVE_CONTACT_GEOMETRY is reserved until integrated authoritative metadata exists"
            )
        mapped, clipped = self._assign(action)
        terminal_id = self._terminal[side]
        terminal_position = np.asarray(self.data.xpos[terminal_id], dtype=np.float64).copy()
        terminal_rotation = np.asarray(self.data.xmat[terminal_id], dtype=np.float64).reshape(3, 3).copy()
        tcp_position = terminal_position + terminal_rotation @ self.tcp_offset
        tcp_rotation = terminal_rotation
        contacts_tcp: dict[str, np.ndarray] = {}
        provenance: dict[str, Any] = {}
        for pad in self._pads[side]:
            world, details = self._inner_surface_point_world(pad.geom_id, tcp_position)
            contacts_tcp[pad.label] = tcp_rotation.T @ (world - tcp_position)
            provenance[pad.label] = {
                "geom_name": pad.geom_name,
                "geom_id": pad.geom_id,
                **details,
            }
        point_a = contacts_tcp["contact_A_tcp"]
        point_b = contacts_tcp["contact_B_tcp"]
        center = 0.5 * (point_a + point_b)
        closing = point_b - point_a
        width = float(np.linalg.norm(closing))
        if width <= np.finfo(np.float64).eps:
            raise RuntimeError("degenerate ALOHA contact width")
        closing /= width
        approach = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        approach -= closing * float(np.dot(approach, closing))
        approach /= np.linalg.norm(approach)
        lateral = np.cross(approach, closing)
        lateral /= np.linalg.norm(lateral)
        result = {
            "schema_version": "aloha_source_interaction_v2",
            "representation_mode": REPRESENTATION_FALLBACK,
            "object_relative_metadata_status": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
            "object_pose_used": False,
            "simulation_scene_used_as_source_annotation": False,
            "side": side,
            "source_model": str(self.model_path),
            "source_action": np.asarray(action, dtype=np.float64),
            "mapped_qpos": mapped,
            "source_gripper_command_m": float(action[6 if side == "left" else 13]),
            "source_gripper_command_was_model_clipped": clipped,
            "tcp_offset_terminal_m": self.tcp_offset,
            "contact_A_tcp_m": point_a,
            "contact_B_tcp_m": point_b,
            "grasp_center_tcp_m": center,
            "closing_axis_tcp": closing,
            "approach_axis_tcp": approach,
            "lateral_axis_tcp": lateral,
            "gripper_width_m": width,
            "contact_assignment": {
                "contact_A_tcp": "Dex3 thumb",
                "contact_B_tcp": "Dex3 index",
            },
            "geometry_approximation": {
                "name": "NAMED_ALOHA_TIP_BOX_INNER_FACE_CENTER",
                "definition": "surface center of each named tip box face pointing toward the TCP center",
                "hidden_offset": False,
                "jaw_pad_provenance": provenance,
            },
        }
        if not all(
            np.isfinite(np.asarray(result[key])).all()
            for key in (
                "contact_A_tcp_m",
                "contact_B_tcp_m",
                "closing_axis_tcp",
                "approach_axis_tcp",
                "gripper_width_m",
            )
        ):
            raise RuntimeError("non-finite source interaction geometry")
        return result
