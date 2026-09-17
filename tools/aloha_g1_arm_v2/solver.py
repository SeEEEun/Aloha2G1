"""Acceptance-aware shared temporal IK used identically by Dataset A and B."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from aloha_g1_dataset_v1.core import SharedTemporalIK


def acceptance_key(
    position_error_m: float,
    orientation_error_rad: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    nominal_distance: float,
) -> tuple[float, float, float, float, float]:
    """Rank gate-valid 6D states before sub-tolerance position refinements."""
    accepted = (
        position_error_m <= position_tolerance_m
        and orientation_error_rad <= orientation_tolerance_rad
    )
    normalized = max(
        position_error_m / position_tolerance_m,
        orientation_error_rad / orientation_tolerance_rad,
    )
    return (
        0.0 if accepted else 1.0,
        normalized,
        position_error_m,
        orientation_error_rad,
        nominal_distance,
    )


class AcceptanceAwareTemporalIK(SharedTemporalIK):
    """v1 DLS budgets/weights with corrected best-effort state retention.

    v1 ranked any position-valid state by position error before considering the
    orientation acceptance gate.  This class makes joint position+orientation
    acceptance the first key while preserving every numerical budget, weight,
    limit, update bound, temporal term, and smoothing parameter.
    """

    def _solve_frame_v2(
        self,
        targets: dict[str, np.ndarray],
        index: int,
        initial: np.ndarray,
        previous: np.ndarray,
        previous2: np.ndarray,
        iterations: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        q = np.asarray(initial, dtype=np.float64).copy()
        limits = self.g1.limits
        damping = float(self.config["damping"])
        max_update = float(self.config["max_joint_update_rad"])
        max_frame_step = float(self.config["max_frame_joint_step_rad"])
        frame_lower = np.maximum(limits[:, 0], previous - max_frame_step)
        frame_upper = np.minimum(limits[:, 1], previous + max_frame_step)
        q = np.clip(q, frame_lower, frame_upper)
        acceptance_position = float(self.config["position_tolerance_m"])
        stopping_position = min(acceptance_position, 5e-4)
        acceptance_orientation = float(self.config["orientation_tolerance_rad"])
        best = q.copy()
        best_key = (math.inf,) * 5
        best_residuals = (math.inf,) * 4
        accepted_seen = False
        used = 0
        numerical_fallback = False
        for used in range(1, iterations + 1):
            jacobian, error, residuals = self._system(
                q, targets, index, previous, previous2
            )
            position_max = max(residuals[:2])
            orientation_max = max(residuals[2:])
            key = acceptance_key(
                position_max,
                orientation_max,
                acceptance_position,
                acceptance_orientation,
                float(np.linalg.norm(q - self.g1.nominal_q)),
            )
            if key < best_key:
                best_key = key
                best = q.copy()
                best_residuals = residuals
            accepted_seen = accepted_seen or key[0] == 0.0
            if position_max <= stopping_position and orientation_max <= acceptance_orientation:
                break
            normal = jacobian.T @ jacobian + (damping * damping) * np.eye(14)
            right = jacobian.T @ error
            try:
                delta = np.linalg.solve(normal, right)
            except np.linalg.LinAlgError:
                numerical_fallback = True
                delta = np.linalg.lstsq(normal, right, rcond=None)[0]
            q = np.clip(
                q + np.clip(delta, -max_update, max_update), frame_lower, frame_upper
            )
        _, _, residuals = self._system(q, targets, index, previous, previous2)
        position_max = max(residuals[:2])
        orientation_max = max(residuals[2:])
        key = acceptance_key(
            position_max,
            orientation_max,
            acceptance_position,
            acceptance_orientation,
            float(np.linalg.norm(q - self.g1.nominal_q)),
        )
        if key < best_key:
            best_key = key
            best = q.copy()
            best_residuals = residuals
        accepted_seen = accepted_seen or key[0] == 0.0
        return best, {
            "iterations": int(used),
            "budget": int(iterations),
            "budget_exhausted": bool(used >= iterations),
            "accepted_state_seen": bool(accepted_seen),
            "returned_accepted": bool(best_key[0] == 0.0),
            "position_error_max_m": float(max(best_residuals[:2])),
            "orientation_error_max_rad": float(max(best_residuals[2:])),
            "numerical_lstsq_fallback": bool(numerical_fallback),
        }

    def solve(self, targets: dict[str, np.ndarray]) -> dict[str, Any]:
        count = len(targets["left_wrist_position"])
        raw = np.empty((count, 14), dtype=np.float64)
        initial_meta: list[dict[str, Any]] = []
        previous = self.g1.nominal_q.copy()
        previous2 = previous.copy()
        for index in range(count):
            maximum = int(
                self.config["max_iterations_initial_frame"]
                if index == 0
                else self.config["max_iterations_per_frame"]
            )
            raw[index], meta = self._solve_frame_v2(
                targets, index, previous, previous, previous2, maximum
            )
            initial_meta.append(meta)
            previous2, previous = previous, raw[index].copy()

        window = min(
            int(self.config["temporal_smoothing_window"]),
            count if count % 2 else count - 1,
        )
        polyorder = int(self.config["temporal_smoothing_polyorder"])
        if window >= max(polyorder + 2, 3):
            smoothed = savgol_filter(raw, window, polyorder, axis=0, mode="interp")
        else:
            smoothed = raw.copy()
        smoothed = np.clip(smoothed, self.g1.limits[:, 0], self.g1.limits[:, 1])

        final = np.empty_like(raw)
        reprojection_meta: list[dict[str, Any]] = []
        previous = smoothed[0].copy()
        previous2 = previous.copy()
        for index in range(count):
            final[index], meta = self._solve_frame_v2(
                targets,
                index,
                smoothed[index],
                previous,
                previous2,
                int(self.config["max_iterations_reprojection"]),
            )
            reprojection_meta.append(meta)
            previous2, previous = previous, final[index].copy()
        return {
            "q": final,
            "raw_q": raw,
            "smoothed_q": smoothed,
            "initial_iterations": np.asarray(
                [row["iterations"] for row in initial_meta], dtype=np.int64
            ),
            "reprojection_iterations": np.asarray(
                [row["iterations"] for row in reprojection_meta], dtype=np.int64
            ),
            "initial_metadata": initial_meta,
            "reprojection_metadata": reprojection_meta,
            "solver_bug_fix": "acceptance-aware best-state retention",
        }


def static_pose_solve(
    g1: Any,
    target_positions: dict[str, np.ndarray],
    target_rotations: dict[str, np.ndarray],
    seeds: list[np.ndarray],
    *,
    iterations: int,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    orientation_weight: float = 0.005,
) -> dict[str, Any]:
    """Deterministic unconstrained-per-frame 6D feasibility query."""
    best: dict[str, Any] | None = None
    identity = np.eye(14, dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        q = np.clip(np.asarray(seed, dtype=np.float64), g1.limits[:, 0], g1.limits[:, 1])
        nominal = q.copy()
        used = 0
        fallback = False
        for used in range(1, iterations + 1):
            state = g1.wrist_state(q)
            rows: list[np.ndarray] = []
            errors: list[np.ndarray] = []
            position_errors: list[float] = []
            orientation_errors: list[float] = []
            for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
                jacobian = np.zeros((6, 14), dtype=np.float64)
                jacobian[:, block] = state[f"{side}_jacobian"]
                position_error = (
                    np.asarray(target_positions[side]) - state[f"{side}_position"]
                )
                from aloha_g1_dataset_v1.core import rotation_error_vector

                orientation_error = rotation_error_vector(
                    state[f"{side}_rotation"], np.asarray(target_rotations[side])
                )
                rows.extend((3.0 * jacobian[:3], orientation_weight * jacobian[3:]))
                errors.extend((3.0 * position_error, orientation_weight * orientation_error))
                position_errors.append(float(np.linalg.norm(position_error)))
                orientation_errors.append(float(np.linalg.norm(orientation_error)))
            rows.append(0.001 * identity)
            errors.append(0.001 * (nominal - q))
            position_max = max(position_errors)
            orientation_max = max(orientation_errors)
            key = acceptance_key(
                position_max,
                orientation_max,
                position_tolerance_m,
                orientation_tolerance_rad,
                float(np.linalg.norm(q - nominal)),
            )
            record = {
                "q": q.copy(),
                "key": key,
                "position_errors_m": position_errors,
                "orientation_errors_rad": orientation_errors,
                "iterations": int(used),
                "seed_index": int(seed_index),
                "numerical_lstsq_fallback": bool(fallback),
            }
            if best is None or key < best["key"]:
                best = record
            if key[0] == 0.0 and position_max <= min(position_tolerance_m, 5e-4):
                break
            jacobian = np.vstack(rows)
            error = np.concatenate(errors)
            normal = jacobian.T @ jacobian + 4e-6 * identity
            try:
                delta = np.linalg.solve(normal, jacobian.T @ error)
            except np.linalg.LinAlgError:
                fallback = True
                delta = np.linalg.lstsq(normal, jacobian.T @ error, rcond=None)[0]
            q = np.clip(q + np.clip(delta, -0.035, 0.035), g1.limits[:, 0], g1.limits[:, 1])
    if best is None:
        raise RuntimeError("static pose solve produced no candidate")
    best["accepted"] = best["key"][0] == 0.0
    best["minimum_joint_limit_margin_rad"] = float(
        np.min(np.minimum(best["q"] - g1.limits[:, 0], g1.limits[:, 1] - best["q"]))
    )
    return best


__all__ = ["AcceptanceAwareTemporalIK", "acceptance_key", "static_pose_solve"]
