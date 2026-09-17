#!/usr/bin/env python3
"""Policy-independent two-stage G1/Dex3 absolute-position safety projection.

Stage 1 projects invalid Dex3 values to authoritative hard bounds.  Stage 2
projects values outside empirically derived simulation-controller-safe bounds
to the nearest safe value.  Arms and already-safe Dex3 values are preserved
bitwise.  Raw, hard-projected, and deployment-safe arrays remain distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_STATUS = "FROZEN_COMMON_DEPLOYMENT_SAFETY_ADAPTER"
EXPECTED_LABEL = "SIMULATION_CONTROLLER_MARGIN_ONLY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeploymentProjectionResult:
    policy_raw_action: np.ndarray
    hard_limit_projected_action: np.ndarray
    deployment_safe_action: np.ndarray
    hard_limit_correction_magnitude: np.ndarray
    deployment_margin_correction_magnitude: np.ndarray
    total_correction_magnitude: np.ndarray
    hard_limit_records: list[dict[str, Any]]
    deployment_margin_records: list[dict[str, Any]]
    records: list[dict[str, Any]]
    summary: dict[str, Any]

    @property
    def hardware_feasible_action(self) -> np.ndarray:
        """Compatibility alias; the authoritative v2 name is deployment_safe_action."""

        return self.deployment_safe_action


class NamedJointDeploymentSafetyProjector:
    """Two-stage componentwise projection for the frozen named interface."""

    def __init__(self, config: dict[str, Any]):
        if config.get("status") != EXPECTED_STATUS:
            raise RuntimeError("common deployment-safety config is not frozen")
        if config.get("margin_label") != EXPECTED_LABEL:
            raise RuntimeError("deployment margin is not labeled simulation-only")
        if config.get("algorithm") != "hard_then_simulation_safe_nearest_closed_interval":
            raise RuntimeError("unsupported deployment-safety algorithm")
        if config.get("command_semantics") != "absolute_joint_position_rad":
            raise RuntimeError("projector requires absolute joint-position commands")
        rows = list(config["joints"])
        self.config = config
        self.names = [row["joint_name"] for row in rows]
        self.hard_lower = np.asarray([row["hard_lower_rad"] for row in rows], dtype=np.float64)
        self.hard_upper = np.asarray([row["hard_upper_rad"] for row in rows], dtype=np.float64)
        self.safe_lower = np.asarray([row["safe_lower_rad"] for row in rows], dtype=np.float64)
        self.safe_upper = np.asarray([row["safe_upper_rad"] for row in rows], dtype=np.float64)
        self.margin_lower = self.safe_lower - self.hard_lower
        self.margin_upper = self.hard_upper - self.safe_upper
        self.projectable = np.asarray([row["projectable"] for row in rows], dtype=bool)
        self.groups = [row["group"] for row in rows]
        if len(rows) == 0 or len(set(self.names)) != len(rows):
            raise RuntimeError("deployment config has missing or duplicate named joints")
        if any(flag != (group == "dex3") for flag, group in zip(self.projectable, self.groups)):
            raise RuntimeError("only Dex3 joints may be projectable")
        if np.any(self.hard_lower >= self.hard_upper):
            raise RuntimeError("empty hard interval")
        if np.any(self.safe_lower >= self.safe_upper):
            raise RuntimeError("empty deployment-safe interval")
        if np.any(self.safe_lower < self.hard_lower) or np.any(self.safe_upper > self.hard_upper):
            raise RuntimeError("deployment-safe interval is not contained by hard interval")
        if np.any(self.safe_lower[~self.projectable] != self.hard_lower[~self.projectable]) or np.any(
            self.safe_upper[~self.projectable] != self.hard_upper[~self.projectable]
        ):
            raise RuntimeError("arm safe intervals must exactly equal arm hard intervals")

    @classmethod
    def from_path(
        cls, path: Path, *, verify_implementation: bool = True
    ) -> "NamedJointDeploymentSafetyProjector":
        path = path.resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if verify_implementation:
            expected = config["implementation"]["sha256"]
            actual = sha256_file(Path(config["implementation"]["path"]))
            if actual != expected:
                raise RuntimeError(
                    f"common deployment implementation hash changed: {actual} != {expected}"
                )
        return cls(config)

    def project(
        self,
        policy_raw_action: np.ndarray,
        *,
        inference_index: int | None = None,
        global_row_offset: int = 0,
    ) -> DeploymentProjectionResult:
        raw = np.asarray(policy_raw_action)
        if raw.ndim != 2 or raw.shape[1] != len(self.names):
            raise ValueError(
                f"action must have shape [rows,{len(self.names)}], received {raw.shape}"
            )
        if not np.issubdtype(raw.dtype, np.floating):
            raise TypeError(f"action dtype must be floating, received {raw.dtype}")
        if not np.isfinite(raw).all():
            raise ValueError("policy_raw_action contains NaN/Inf")

        hard = raw.copy()
        hard_low_mask = (raw < self.hard_lower[None, :]) & self.projectable[None, :]
        hard_high_mask = (raw > self.hard_upper[None, :]) & self.projectable[None, :]
        hard[hard_low_mask] = np.broadcast_to(self.hard_lower, raw.shape)[hard_low_mask]
        hard[hard_high_mask] = np.broadcast_to(self.hard_upper, raw.shape)[hard_high_mask]
        hard_changed = hard_low_mask | hard_high_mask

        safe = hard.copy()
        safe_low_mask = (hard < self.safe_lower[None, :]) & self.projectable[None, :]
        safe_high_mask = (hard > self.safe_upper[None, :]) & self.projectable[None, :]
        safe[safe_low_mask] = np.broadcast_to(self.safe_lower, raw.shape)[safe_low_mask]
        safe[safe_high_mask] = np.broadcast_to(self.safe_upper, raw.shape)[safe_high_mask]
        safe_changed = safe_low_mask | safe_high_mask

        hard_correction = np.abs(hard.astype(np.float64) - raw.astype(np.float64))
        margin_correction = np.abs(safe.astype(np.float64) - hard.astype(np.float64))
        total_correction = np.abs(safe.astype(np.float64) - raw.astype(np.float64))

        if not np.array_equal(hard[:, ~self.projectable], raw[:, ~self.projectable]):
            raise RuntimeError("hard projection changed an arm scalar")
        if not np.array_equal(safe[:, ~self.projectable], raw[:, ~self.projectable]):
            raise RuntimeError("deployment projection changed an arm scalar")
        if not np.array_equal(hard[~hard_changed], raw[~hard_changed]):
            raise RuntimeError("hard projection changed an already-hard-valid scalar")
        if not np.array_equal(safe[~safe_changed], hard[~safe_changed]):
            raise RuntimeError("deployment projection changed an already-safe scalar")
        if np.any(hard[:, self.projectable] < self.hard_lower[self.projectable][None, :]) or np.any(
            hard[:, self.projectable] > self.hard_upper[self.projectable][None, :]
        ):
            raise RuntimeError("hard-projected action remains outside a hard interval")
        if np.any(safe[:, self.projectable] < self.safe_lower[self.projectable][None, :]) or np.any(
            safe[:, self.projectable] > self.safe_upper[self.projectable][None, :]
        ):
            raise RuntimeError("deployment-safe action remains outside a safe interval")

        def base_record(row: int, column: int) -> dict[str, Any]:
            return {
                "inference_index": inference_index,
                "action_row_in_chunk": int(row),
                "global_action_row": int(global_row_offset + row),
                "policy_index": int(column),
                "joint": self.names[column],
            }

        hard_records: list[dict[str, Any]] = []
        for row, column in np.argwhere(hard_changed):
            active = "lower" if hard_low_mask[row, column] else "upper"
            hard_records.append(
                {
                    **base_record(row, column),
                    "projection_stage": "hard_limit",
                    "active_bound": active,
                    "policy_raw_value_rad": float(raw[row, column]),
                    "hard_limit_projected_value_rad": float(hard[row, column]),
                    "deployment_safe_value_rad": float(safe[row, column]),
                    "incremental_correction_magnitude_rad": float(
                        hard_correction[row, column]
                    ),
                    "total_raw_to_deployment_safe_correction_rad": float(
                        total_correction[row, column]
                    ),
                    "hard_lower_rad": float(self.hard_lower[column]),
                    "hard_upper_rad": float(self.hard_upper[column]),
                }
            )

        margin_records: list[dict[str, Any]] = []
        for row, column in np.argwhere(safe_changed):
            active = "lower" if safe_low_mask[row, column] else "upper"
            margin_records.append(
                {
                    **base_record(row, column),
                    "projection_stage": "simulation_controller_margin",
                    "margin_label": EXPECTED_LABEL,
                    "active_bound": active,
                    "policy_raw_value_rad": float(raw[row, column]),
                    "hard_limit_projected_value_rad": float(hard[row, column]),
                    "deployment_safe_value_rad": float(safe[row, column]),
                    "incremental_correction_magnitude_rad": float(
                        margin_correction[row, column]
                    ),
                    "total_raw_to_deployment_safe_correction_rad": float(
                        total_correction[row, column]
                    ),
                    "safe_lower_rad": float(self.safe_lower[column]),
                    "safe_upper_rad": float(self.safe_upper[column]),
                    "margin_lower_rad": float(self.margin_lower[column]),
                    "margin_upper_rad": float(self.margin_upper[column]),
                }
            )

        hard_affected = hard_correction[hard_changed]
        margin_affected = margin_correction[safe_changed]
        total_changed = raw != safe
        total_affected = total_correction[total_changed]
        summary = {
            "input_shape": list(raw.shape),
            "hard_limit_projected_scalar_count": int(np.count_nonzero(hard_changed)),
            "deployment_margin_projected_scalar_count": int(np.count_nonzero(safe_changed)),
            "total_modified_scalar_count": int(np.count_nonzero(total_changed)),
            "affected_frame_count": int(np.count_nonzero(np.any(total_changed, axis=1))),
            "affected_joint_count": int(np.count_nonzero(np.any(total_changed, axis=0))),
            "affected_joint_names": [
                self.names[index] for index in np.flatnonzero(np.any(total_changed, axis=0))
            ],
            "hard_lower_projection_count": int(np.count_nonzero(hard_low_mask)),
            "hard_upper_projection_count": int(np.count_nonzero(hard_high_mask)),
            "safe_lower_projection_count": int(np.count_nonzero(safe_low_mask)),
            "safe_upper_projection_count": int(np.count_nonzero(safe_high_mask)),
            "mean_hard_projection_magnitude_over_modified_scalars_rad": float(
                np.mean(hard_affected)
            )
            if hard_affected.size
            else 0.0,
            "maximum_hard_projection_magnitude_rad": float(np.max(hard_affected))
            if hard_affected.size
            else 0.0,
            "mean_deployment_margin_correction_over_modified_scalars_rad": float(
                np.mean(margin_affected)
            )
            if margin_affected.size
            else 0.0,
            "maximum_deployment_margin_correction_rad": float(np.max(margin_affected))
            if margin_affected.size
            else 0.0,
            "maximum_total_raw_to_deployment_safe_correction_rad": float(
                np.max(total_affected)
            )
            if total_affected.size
            else 0.0,
            "arm_outputs_bitwise_preserved": bool(
                np.array_equal(safe[:, ~self.projectable], raw[:, ~self.projectable])
            ),
            "hard_valid_values_preserved_by_hard_stage": bool(
                np.array_equal(hard[~hard_changed], raw[~hard_changed])
            ),
            "safe_values_preserved_by_margin_stage": bool(
                np.array_equal(safe[~safe_changed], hard[~safe_changed])
            ),
            "hard_limit_violation_count_after_hard_projection": 0,
            "hard_limit_violation_count_after_deployment_projection": 0,
            "safe_interval_violation_count_after_deployment_projection": 0,
            "comparison_tolerance_rad": 0.0,
            "margin_label": EXPECTED_LABEL,
        }
        records = sorted(
            hard_records + margin_records,
            key=lambda item: (
                -1 if item["inference_index"] is None else item["inference_index"],
                item["action_row_in_chunk"],
                item["policy_index"],
                item["projection_stage"],
            ),
        )
        return DeploymentProjectionResult(
            policy_raw_action=raw.copy(),
            hard_limit_projected_action=hard,
            deployment_safe_action=safe,
            hard_limit_correction_magnitude=hard_correction,
            deployment_margin_correction_magnitude=margin_correction,
            total_correction_magnitude=total_correction,
            hard_limit_records=hard_records,
            deployment_margin_records=margin_records,
            records=records,
            summary=summary,
        )


__all__ = [
    "NamedJointDeploymentSafetyProjector",
    "DeploymentProjectionResult",
    "sha256_file",
]
