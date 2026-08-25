"""Fail-closed read-only parent-pose contract for a movable helmet mount."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from deployment_camera_config import DeploymentCamera


def _matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise RuntimeError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise RuntimeError(f"{label} has an invalid homogeneous final row")
    return result


class ParentLinkPoseAdapter:
    """Validate a locked pose or consume timestamp-aligned FK from a JSON bridge.

    The JSON bridge is intentionally read-only and may be atomically refreshed
    by a separate robot-state/FK process.  Its schema is:
    ``parent_link``, ``host_monotonic_timestamp_ns``, and
    ``task_from_parent_matrix``.  This adapter has no DDS publisher or command
    dependency.
    """

    def __init__(
        self,
        camera: DeploymentCamera,
        *,
        locked_parent_pose_id: str | None,
        fk_json: Path | None,
        maximum_fk_time_offset_ms: float = 100.0,
    ):
        self.camera = camera
        self.mount = camera.mount
        self.maximum_offset_ns = int(maximum_fk_time_offset_ms * 1e6)
        movable = self.mount.get("parent_link_can_move")
        if movable is not True:
            self.mode = "STATIC_PARENT"
            self.locked_parent_pose_id = None
            self.fk_json = None
            return
        configured_locked = self.mount.get("locked_parent_pose_id")
        configured_fk = self.mount.get("parent_link_fk_provider")
        unresolved = {None, "", "CAMERA_NOT_MOUNTED", "EXTRINSIC_NOT_CALIBRATED"}
        if configured_locked not in unresolved:
            if locked_parent_pose_id != configured_locked:
                raise RuntimeError(
                    f"camera requires locked parent pose {configured_locked!r}; "
                    f"runtime reported {locked_parent_pose_id!r}"
                )
            self.mode = "LOCKED_PARENT_POSE"
            self.locked_parent_pose_id = configured_locked
            self.fk_json = None
        elif configured_fk not in unresolved:
            if fk_json is None:
                raise RuntimeError(
                    "movable camera parent requires --parent-fk-json for the configured FK provider"
                )
            _matrix(self.mount.get("parent_from_camera_matrix"), "mount.parent_from_camera_matrix")
            self.mode = "TIMESTAMP_ALIGNED_PARENT_FK"
            self.locked_parent_pose_id = None
            self.fk_json = fk_json.resolve()
        else:
            raise RuntimeError("movable camera parent has neither locked pose nor FK contract")

    def read(self, camera_host_monotonic_ns: int) -> dict[str, Any]:
        if self.mode in {"STATIC_PARENT", "LOCKED_PARENT_POSE"}:
            return {
                "mode": self.mode,
                "parent_link": self.mount.get("parent_link"),
                "locked_parent_pose_id": self.locked_parent_pose_id,
                "task_from_camera_matrix": self.camera.task_from_camera.tolist(),
                "timestamp_aligned_fk_used": False,
            }
        assert self.fk_json is not None
        value = json.loads(self.fk_json.read_text(encoding="utf-8"))
        if value.get("parent_link") != self.mount.get("parent_link"):
            raise RuntimeError("FK bridge parent_link differs from calibrated camera parent")
        stamp = int(value["host_monotonic_timestamp_ns"])
        offset = abs(int(camera_host_monotonic_ns) - stamp)
        if offset > self.maximum_offset_ns:
            raise RuntimeError(
                f"parent FK is not timestamp-aligned: {offset / 1e6:.3f} ms exceeds "
                f"{self.maximum_offset_ns / 1e6:.3f} ms"
            )
        task_from_parent = _matrix(value["task_from_parent_matrix"], "task_from_parent_matrix")
        parent_from_camera = _matrix(
            self.mount["parent_from_camera_matrix"], "mount.parent_from_camera_matrix"
        )
        return {
            "mode": self.mode,
            "parent_link": self.mount["parent_link"],
            "fk_provider": self.mount["parent_link_fk_provider"],
            "fk_json": str(self.fk_json),
            "camera_host_monotonic_timestamp_ns": int(camera_host_monotonic_ns),
            "fk_host_monotonic_timestamp_ns": stamp,
            "absolute_time_offset_ms": offset / 1e6,
            "task_from_camera_matrix": (task_from_parent @ parent_from_camera).tolist(),
            "timestamp_aligned_fk_used": True,
        }


__all__ = ["ParentLinkPoseAdapter"]
