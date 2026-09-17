#!/usr/bin/env python3
"""Policy-independent jerk-limited realization of a selected 28D position target."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OTGResult:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    target_position: np.ndarray
    audit: dict[str, Any]


class CommonJerkLimitedOTG:
    """Generate a fail-closed Ruckig reference from measured state to one target."""

    def __init__(self, config: dict[str, Any]):
        if config.get("schema_version") != "common_g1_28d_ruckig_otg_v1":
            raise RuntimeError("unsupported common OTG config")
        if config.get("policy_independent") is not True:
            raise RuntimeError("OTG config is not policy independent")
        rows = sorted(config["joints"], key=lambda row: row["joint_index"])
        if len(rows) != 28 or [row["joint_index"] for row in rows] != list(range(28)):
            raise RuntimeError("OTG config does not define canonical 28D order")
        self.config = config
        self.names = [row["joint_name"] for row in rows]
        self.safe_lower = np.asarray([row["target_safe_lower_rad"] for row in rows], dtype=np.float64)
        self.safe_upper = np.asarray([row["target_safe_upper_rad"] for row in rows], dtype=np.float64)
        self.max_velocity = np.asarray([row["max_velocity_rad_s"] for row in rows], dtype=np.float64)
        self.authoritative_velocity_ceiling = np.asarray(
            [row["authoritative_velocity_ceiling_rad_s"] for row in rows],
            dtype=np.float64,
        )
        self.max_acceleration = np.asarray(
            [row["max_acceleration_rad_s2"] for row in rows], dtype=np.float64
        )
        self.max_jerk = np.asarray([row["max_jerk_rad_s3"] for row in rows], dtype=np.float64)
        self.control_period = float(config["control_period_s"])
        if np.any(self.safe_lower >= self.safe_upper) or min(
            self.max_velocity.min(), self.max_acceleration.min(), self.max_jerk.min()
        ) <= 0.0:
            raise RuntimeError("OTG config has an empty interval or non-positive constraint")

    @classmethod
    def from_path(cls, path: Path) -> "CommonJerkLimitedOTG":
        path = path.resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        implementation = config["implementation"]
        actual = sha256_file(Path(implementation["path"]))
        if actual != implementation["sha256"]:
            raise RuntimeError(f"common OTG implementation hash mismatch: {actual}")
        return cls(config)

    def generate(
        self,
        *,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_acceleration: np.ndarray,
        target_position: np.ndarray,
        steps: int = 50,
    ) -> OTGResult:
        from ruckig import InputParameter, OutputParameter, Result, Ruckig

        q = np.asarray(current_position, dtype=np.float64)
        dq = np.asarray(current_velocity, dtype=np.float64)
        ddq = np.asarray(current_acceleration, dtype=np.float64)
        target = np.asarray(target_position, dtype=np.float64)
        if any(value.shape != (28,) for value in (q, dq, ddq, target)):
            raise ValueError("OTG requires current q/dq/ddq and target q as finite (28,) arrays")
        if not all(np.isfinite(value).all() for value in (q, dq, ddq, target)):
            raise ValueError("OTG input contains NaN/Inf")
        if steps < 1:
            raise ValueError("OTG steps must be positive")
        low = np.flatnonzero(target < self.safe_lower)
        high = np.flatnonzero(target > self.safe_upper)
        if len(low) or len(high):
            raise ValueError(
                f"OTG target is outside the deployment-safe interval: low={low.tolist()} high={high.tolist()}"
            )

        otg = Ruckig(28, self.control_period)
        inp = InputParameter(28)
        out = OutputParameter(28)
        inp.current_position = q.tolist()
        inp.current_velocity = dq.tolist()
        inp.current_acceleration = ddq.tolist()
        inp.target_position = target.tolist()
        inp.target_velocity = np.zeros(28, dtype=np.float64).tolist()
        inp.target_acceleration = np.zeros(28, dtype=np.float64).tolist()
        if np.any(np.abs(dq) > self.authoritative_velocity_ceiling + 1e-9):
            violating = np.flatnonzero(
                np.abs(dq) > self.authoritative_velocity_ceiling + 1e-9
            )
            raise RuntimeError(
                "measured velocity exceeds the authoritative actuator ceiling: "
                f"joints={violating.tolist()}"
            )
        effective_max_velocity = np.minimum(
            np.maximum(self.max_velocity, np.abs(dq)),
            self.authoritative_velocity_ceiling,
        )
        inp.max_velocity = effective_max_velocity.tolist()
        inp.min_velocity = (-effective_max_velocity).tolist()
        inp.max_acceleration = self.max_acceleration.tolist()
        inp.min_acceleration = (-self.max_acceleration).tolist()
        inp.max_jerk = self.max_jerk.tolist()
        inp.min_position = self.safe_lower.tolist()
        inp.max_position = self.safe_upper.tolist()

        # Current feedback is intentionally not projected or hidden. Ruckig's documented
        # braking behavior accepts a current state outside dynamic limits while requiring
        # the target and all generated commands to be valid.
        try:
            valid = bool(otg.validate_input(inp, False, True))
        except Exception as error:
            raise RuntimeError(f"Ruckig input validation failed closed: {error}") from error
        if not valid:
            raise RuntimeError("Ruckig input validation returned false; no command generated")

        positions = []
        velocities = []
        accelerations = []
        results = []
        calculation_us = []
        for _ in range(steps):
            result = otg.update(inp, out)
            if result not in (Result.Working, Result.Finished):
                raise RuntimeError(f"Ruckig update failed closed with result {result}")
            position = np.asarray(out.new_position, dtype=np.float64)
            velocity = np.asarray(out.new_velocity, dtype=np.float64)
            acceleration = np.asarray(out.new_acceleration, dtype=np.float64)
            if not all(np.isfinite(value).all() for value in (position, velocity, acceleration)):
                raise RuntimeError("Ruckig generated NaN/Inf; no command may be used")
            positions.append(position)
            velocities.append(velocity)
            accelerations.append(acceleration)
            results.append(str(result))
            calculation_us.append(float(out.calculation_duration))
            out.pass_to_input(inp)

        position_array = np.stack(positions)
        velocity_array = np.stack(velocities)
        acceleration_array = np.stack(accelerations)
        velocity_excess = np.maximum(
            np.abs(velocity_array) - effective_max_velocity[None, :], 0.0
        )
        acceleration_excess = np.maximum(
            np.abs(acceleration_array) - self.max_acceleration[None, :], 0.0
        )
        generated_jerk = np.diff(
            np.vstack((ddq[None, :], acceleration_array)), axis=0
        ) / self.control_period
        jerk_excess = np.maximum(np.abs(generated_jerk) - self.max_jerk[None, :], 0.0)
        maximum_excesses = {
            "velocity": float(velocity_excess.max()),
            "acceleration": float(acceleration_excess.max()),
            "jerk": float(jerk_excess.max()),
        }
        if max(maximum_excesses.values()) > 1e-8:
            worst_quantity = max(maximum_excesses, key=maximum_excesses.get)
            excess_array = {
                "velocity": velocity_excess,
                "acceleration": acceleration_excess,
                "jerk": jerk_excess,
            }[worst_quantity]
            worst_frame, worst_joint = np.unravel_index(
                int(np.argmax(excess_array)), excess_array.shape
            )
            raise RuntimeError(
                "Ruckig output exceeded a configured dynamic constraint: "
                f"quantity={worst_quantity} excess={maximum_excesses[worst_quantity]:.12g} "
                f"frame={worst_frame} joint={self.names[worst_joint]}"
            )
        audit = {
            "implementation": "Ruckig community Python online position OTG",
            "ruckig_version": "0.19.4",
            "steps": int(steps),
            "current_position_rad": q,
            "current_velocity_rad_s": dq,
            "current_acceleration_rad_s2": ddq,
            "target_position_rad": target,
            "target_velocity_rad_s": np.zeros(28),
            "target_acceleration_rad_s2": np.zeros(28),
            "nominal_max_velocity_rad_s": self.max_velocity,
            "effective_max_velocity_rad_s": effective_max_velocity,
            "measured_velocity_above_nominal_joint_count": int(
                np.count_nonzero(np.abs(dq) > self.max_velocity)
            ),
            "maximum_generated_velocity_rad_s": float(np.max(np.abs(velocity_array))),
            "maximum_generated_acceleration_rad_s2": float(np.max(np.abs(acceleration_array))),
            "maximum_generated_jerk_rad_s3": float(np.max(np.abs(generated_jerk))),
            "maximum_velocity_constraint_excess": float(velocity_excess.max()),
            "maximum_acceleration_constraint_excess": float(acceleration_excess.max()),
            "maximum_jerk_constraint_excess": float(jerk_excess.max()),
            "calculation_duration_us_mean": float(np.mean(calculation_us)),
            "calculation_duration_us_max": float(np.max(calculation_us)),
            "final_result": results[-1],
            "failed_closed": False,
            "raw_policy_modified": False,
            "future_policy_calls_used": False,
            "task_phase_episode_object_specific_logic": False,
        }
        return OTGResult(position_array, velocity_array, acceleration_array, target, audit)


__all__ = ["CommonJerkLimitedOTG", "OTGResult", "sha256_file"]
