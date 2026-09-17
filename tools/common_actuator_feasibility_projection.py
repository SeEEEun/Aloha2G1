#!/usr/bin/env python3
"""Generic named-joint actuator-feasibility projection for absolute positions.

The layer is deliberately policy-, episode-, phase-, and task-independent.  It
preserves every non-projectable scalar and every already-valid projectable
scalar exactly.  Invalid projectable scalars are replaced only by the nearest
closed-interval bound, while a complete correction record is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_STATUS = "FROZEN_COMMON_DEPLOYMENT_SAFETY_ADAPTER"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProjectionResult:
    policy_raw_action: np.ndarray
    hardware_feasible_action: np.ndarray
    correction_magnitude: np.ndarray
    records: list[dict[str, Any]]
    summary: dict[str, Any]


class NamedJointPositionProjector:
    """Componentwise Euclidean projection onto configured named intervals."""

    def __init__(self, config: dict[str, Any]):
        if config.get("status") != EXPECTED_STATUS:
            raise RuntimeError("common actuator-feasibility config is not frozen")
        if config.get("algorithm") != "componentwise_nearest_closed_interval_bound":
            raise RuntimeError("unsupported actuator-feasibility algorithm")
        if config.get("command_semantics") != "absolute_joint_position_rad":
            raise RuntimeError("projector requires absolute joint-position commands")
        rows = list(config["joints"])
        self.config = config
        self.names = [row["joint_name"] for row in rows]
        self.lower = np.asarray([row["effective_lower_rad"] for row in rows], dtype=np.float64)
        self.upper = np.asarray([row["effective_upper_rad"] for row in rows], dtype=np.float64)
        self.projectable = np.asarray([row["projectable"] for row in rows], dtype=bool)
        self.groups = [row["group"] for row in rows]
        if len(rows) == 0 or len(set(self.names)) != len(rows):
            raise RuntimeError("projection config has missing or duplicate named joints")
        if np.any(self.lower >= self.upper):
            raise RuntimeError("projection config contains an empty joint interval")
        if any(flag != (group == "dex3") for flag, group in zip(self.projectable, self.groups)):
            raise RuntimeError("only Dex3 joints may be projectable in this frozen adapter")

    @classmethod
    def from_path(cls, path: Path, *, verify_implementation: bool = True) -> "NamedJointPositionProjector":
        path = path.resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if verify_implementation:
            expected = config["implementation"]["sha256"]
            actual = sha256_file(Path(config["implementation"]["path"]))
            if actual != expected:
                raise RuntimeError(
                    f"common projection implementation hash changed: {actual} != {expected}"
                )
        return cls(config)

    def project(
        self,
        policy_raw_action: np.ndarray,
        *,
        inference_index: int | None = None,
        global_row_offset: int = 0,
    ) -> ProjectionResult:
        raw = np.asarray(policy_raw_action)
        if raw.ndim != 2 or raw.shape[1] != len(self.names):
            raise ValueError(
                f"action must have shape [rows,{len(self.names)}], received {raw.shape}"
            )
        if not np.issubdtype(raw.dtype, np.floating):
            raise TypeError(f"action dtype must be floating, received {raw.dtype}")
        if not np.isfinite(raw).all():
            raise ValueError("policy_raw_action contains NaN/Inf")

        feasible = raw.copy()
        low_mask = (raw < self.lower[None, :]) & self.projectable[None, :]
        high_mask = (raw > self.upper[None, :]) & self.projectable[None, :]
        feasible[low_mask] = np.broadcast_to(self.lower, raw.shape)[low_mask]
        feasible[high_mask] = np.broadcast_to(self.upper, raw.shape)[high_mask]
        correction = np.abs(feasible.astype(np.float64) - raw.astype(np.float64))
        changed = low_mask | high_mask

        # These are strict invariants, not approximate checks.
        if not np.array_equal(feasible[:, ~self.projectable], raw[:, ~self.projectable]):
            raise RuntimeError("projection changed a non-projectable (arm) scalar")
        if not np.array_equal(feasible[~changed], raw[~changed]):
            raise RuntimeError("projection changed an already-valid scalar")
        if np.any(feasible[:, self.projectable] < self.lower[self.projectable][None, :]) or np.any(
            feasible[:, self.projectable] > self.upper[self.projectable][None, :]
        ):
            raise RuntimeError("projected action is still outside a configured interval")

        records: list[dict[str, Any]] = []
        for row, column in np.argwhere(changed):
            bound = "lower" if low_mask[row, column] else "upper"
            records.append(
                {
                    "inference_index": inference_index,
                    "action_row_in_chunk": int(row),
                    "global_action_row": int(global_row_offset + row),
                    "policy_index": int(column),
                    "joint": self.names[column],
                    "raw_value_rad": float(raw[row, column]),
                    "projected_value_rad": float(feasible[row, column]),
                    "correction_magnitude_rad": float(correction[row, column]),
                    "active_bound": bound,
                    "effective_lower_rad": float(self.lower[column]),
                    "effective_upper_rad": float(self.upper[column]),
                }
            )
        affected = correction[changed]
        summary = {
            "input_shape": list(raw.shape),
            "projected_scalar_count": int(np.count_nonzero(changed)),
            "affected_frame_count": int(np.count_nonzero(np.any(changed, axis=1))),
            "affected_joint_count": int(np.count_nonzero(np.any(changed, axis=0))),
            "affected_joint_names": [
                self.names[index] for index in np.flatnonzero(np.any(changed, axis=0))
            ],
            "lower_projection_count": int(np.count_nonzero(low_mask)),
            "upper_projection_count": int(np.count_nonzero(high_mask)),
            "mean_projection_magnitude_over_projected_scalars_rad": float(np.mean(affected))
            if affected.size
            else 0.0,
            "maximum_projection_magnitude_rad": float(np.max(affected)) if affected.size else 0.0,
            "mean_projection_magnitude_over_all_scalars_rad": float(np.mean(correction)),
            "rms_projection_magnitude_over_all_scalars_rad": float(
                np.sqrt(np.mean(np.square(correction)))
            ),
            "arm_outputs_bitwise_preserved": bool(
                np.array_equal(feasible[:, ~self.projectable], raw[:, ~self.projectable])
            ),
            "valid_dex3_outputs_bitwise_preserved": bool(
                np.array_equal(feasible[:, self.projectable][~changed[:, self.projectable]],
                               raw[:, self.projectable][~changed[:, self.projectable]])
            ),
            "projected_dex3_limit_violation_count": 0,
            "comparison_tolerance_rad": 0.0,
        }
        return ProjectionResult(raw.copy(), feasible, correction, records, summary)


__all__ = ["NamedJointPositionProjector", "ProjectionResult", "sha256_file"]
