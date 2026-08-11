"""Regression guard against stale robot renders with moving joint trajectories."""
from __future__ import annotations

from typing import Mapping, Any


class StaticRobotRenderError(AssertionError):
    """Raised when moving q is paired with a static or empty robot mask."""


def assert_moving_q_has_moving_robot_masks(
    q_motion_peak_to_peak_rad: float,
    cameras: Mapping[str, Mapping[str, Any]],
    *,
    minimum_q_motion_rad: float = 1.0e-3,
    minimum_mask_xor_pixels: int = 100,
) -> None:
    """Require nonempty, changing robot masks whenever the input q moves."""
    if q_motion_peak_to_peak_rad <= minimum_q_motion_rad:
        raise StaticRobotRenderError("input q trajectory is static")
    if not cameras:
        raise StaticRobotRenderError("no rendered camera audits were supplied")
    failures = []
    for name, row in cameras.items():
        moving = (
            not bool(row.get("robot_masks_identical_at_all_keyframes", True))
            and int(row.get("maximum_keyframe_mask_xor_pixels", 0))
            > minimum_mask_xor_pixels
            and bool(row.get("robot_mask_nonempty_all_frames", False))
        )
        if not moving:
            failures.append(name)
    if failures:
        raise StaticRobotRenderError(
            "BLOCKED_RENDERED_ROBOT_STATIC_DESPITE_MOVING_Q: " + ",".join(failures)
        )

