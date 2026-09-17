"""G1 training-schema v1 adapters and LeRobot packaging utilities.

This package is deliberately isolated from the retargeting implementation.  It
only consumes completed retargeted trajectories and never changes their values.
"""

from .constants import CANONICAL_JOINT_NAMES, FPS, SCHEMA_VERSION

__all__ = ["CANONICAL_JOINT_NAMES", "FPS", "SCHEMA_VERSION"]
