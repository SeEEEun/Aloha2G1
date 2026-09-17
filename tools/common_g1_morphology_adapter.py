"""Method-blind G1 morphology feasibility projection for A/B target SE(3).

The adapter sees only incoming bilateral wrist SE(3), a previous common IK
realization, common hand state, and the G1 model.  It has no representation,
object, event, policy, scorer, or physical-outcome input.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from tools.doll_handoff_feasibility.solver import GenericG1FeasibilityResolver
from tools.doll_handoff_retargeting.common import SIDES
from tools.doll_handoff_retargeting.models import G1Kinematics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/common_g1_morphology_adapter_v1.json"
GENERIC_CONFIG = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"


@dataclass
class MorphologyAdapterResult:
    raw_targets: dict[str, np.ndarray]
    feasible_targets: dict[str, np.ndarray]
    projected_q: np.ndarray
    position_projection_m: np.ndarray
    orientation_projection_rad: np.ndarray
    achieved_positions_model: dict[str, np.ndarray]
    achieved_rotations_model: dict[str, np.ndarray]
    metadata: dict[str, Any]


class CommonG1MorphologyAdapter(GenericG1FeasibilityResolver):
    """One common constrained target-to-G1 projection with no method branch."""

    def __init__(
        self,
        common: Mapping[str, Any],
        g1: G1Kinematics,
        nominal_q: np.ndarray,
        config_path: str | Path = DEFAULT_CONFIG,
    ):
        # Do not call the historical resolver constructor: its old frozen input
        # contract is unrelated to this TRAIN-only single-variable reset.  The
        # mature task-independent tracking/collision algorithms are reused with
        # the current shared model/config supplied explicitly here.
        self.adapter_config_path = Path(config_path).resolve()
        self.adapter_config = json.loads(self.adapter_config_path.read_text(encoding="utf-8"))
        if self.adapter_config.get("method_blind") is not True:
            raise RuntimeError("common morphology adapter must be method-blind")
        forbidden = (
            "representation_mode_input_allowed",
            "episode_specific_parameters_allowed",
            "phase_specific_parameters_allowed",
            "object_pose_objective_allowed",
            "task_success_input_allowed",
            "physical_outcome_input_allowed",
        )
        if any(bool(self.adapter_config.get(key)) for key in forbidden):
            raise RuntimeError("common morphology adapter enables a forbidden input")
        self.config = json.loads(GENERIC_CONFIG.read_text(encoding="utf-8"))
        collision_projection = self.adapter_config["collision_temporal_projection"]
        self.config["collision_repair"]["maximum_event_iterations"] = int(
            collision_projection["maximum_event_iterations"]
        )
        self.config["collision_repair"]["window_padding_candidates_frames"] = list(
            collision_projection["window_padding_candidates_frames"]
        )
        self.config["collision_repair"]["correction_scale_schedule"] = list(
            collision_projection["correction_scale_schedule"]
        )
        self.common = copy.deepcopy(dict(common))
        self.scene: dict[str, Any] = {}
        self.g1 = g1
        self.nominal = np.asarray(nominal_q, dtype=np.float64)
        self.acceptance = self.config["unchanged_acceptance"]
        self.strict_tolerance = float(self.acceptance["strict_position_tolerance_m"])
        self.physical_tolerance = float(self.acceptance["physical_position_tolerance_m"])
        self.collision_tolerance = float(self.acceptance["collision_penetration_tolerance_m"])
        self.maximum_position_projection = float(
            self.adapter_config["projection_bounds"]["maximum_position_projection_m"]
        )
        # Identity makes the inherited generic resolver operate on wrist SE(3)
        # rather than any representation-specific tool or grasp frame.
        self.transforms = {side: np.eye(4, dtype=np.float64) for side in SIDES}
        self.shoulders = self.g1.fixed_shoulder_anchors_model()
        reach = self.g1.shoulder_wrist_reach_geometry()
        self.outer_tool_radius = {
            side: float(reach["sides"][side]["upper_effective_length_m"])
            + float(reach["sides"][side]["forearm_effective_length_m"])
            for side in SIDES
        }

    def _anchor_repair(
        self,
        frame: int,
        q: np.ndarray,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Clear one collision under the declared morphology projection bound."""
        settings = self.config["collision_repair"]
        current = q[frame].copy()
        position_bound = np.full(
            2,
            self.maximum_position_projection
            + float(self.adapter_config["position_tolerance_interior_m"]),
            dtype=np.float64,
        )
        tracked: list[tuple[int, int]] = []
        reports: list[dict[str, Any]] = []
        for discovery in range(int(settings["maximum_pair_discovery_passes"])):
            records = self._records(current, left_hand[frame], right_hand[frame])
            for record in records:
                pair = tuple(record["geom_pair"])
                if pair not in tracked:
                    tracked.append(pair)
            if not records:
                break

            def position_error(value: np.ndarray) -> np.ndarray:
                position, _ = self._pose(value)
                return np.asarray(
                    [
                        np.linalg.norm(position[side] - source_model[side][frame])
                        for side in SIDES
                    ],
                    dtype=np.float64,
                )

            def distances(value: np.ndarray) -> np.ndarray:
                return np.asarray(
                    [
                        self._distance(value, left_hand[frame], right_hand[frame], pair)
                        for pair in tracked
                    ],
                    dtype=np.float64,
                )

            def objective(value: np.ndarray) -> float:
                errors = position_error(value)
                return (
                    float(settings["source_position_objective_scale_per_m"]) ** 2
                    * float(errors @ errors)
                    + float(settings["frozen_q_deviation_weight"])
                    * float(np.dot(value - q_before[frame], value - q_before[frame]))
                    + float(settings["nominal_q_deviation_weight"])
                    * float(np.dot(value - self.nominal, value - self.nominal))
                )

            def constraint(value: np.ndarray) -> np.ndarray:
                return np.concatenate(
                    (
                        position_bound - position_error(value),
                        distances(value) - float(settings["clearance_m"]),
                    )
                )

            result = minimize(
                objective,
                current,
                method="SLSQP",
                bounds=list(
                    zip(
                        self.g1.arm_limits[:, 0] + 1e-8,
                        self.g1.arm_limits[:, 1] - 1e-8,
                    )
                ),
                constraints={"type": "ineq", "fun": constraint},
                options={
                    "maxiter": int(settings["maximum_anchor_iterations"]),
                    "ftol": 1e-10,
                    "disp": False,
                },
            )
            candidate = np.asarray(result.x, dtype=np.float64)
            old_count = len(records)
            new_records = self._records(candidate, left_hand[frame], right_hand[frame])
            margin = float(np.min(constraint(candidate)))
            accepted = bool(margin >= -1e-6 and len(new_records) < old_count)
            reports.append(
                {
                    "discovery_pass": discovery,
                    "optimizer_success": bool(result.success),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "minimum_constraint_margin": margin,
                    "old_contact_count": old_count,
                    "new_contact_count": len(new_records),
                    "accepted": accepted,
                }
            )
            if not accepted:
                break
            current = candidate
        return current, {
            "frame": frame,
            "tracked_geom_pairs": [list(pair) for pair in tracked],
            "position_bound_m": position_bound,
            "bound_derivation": self.adapter_config["projection_bounds"]["position_derivation"],
            "delta_norm_rad": float(np.linalg.norm(current - q[frame])),
            "reports": reports,
            "final_contact_count": len(self._records(current, left_hand[frame], right_hand[frame])),
        }

    def _collision_candidate_safe(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> bool:
        """Accept only strict collision progress inside common morphology bounds."""
        before_collision = before["collision"]
        after_collision = after["collision"]
        collision_reduced = (
            int(after_collision["hard_collision_frame_count"])
            < int(before_collision["hard_collision_frame_count"])
            or (
                int(after_collision["hard_collision_frame_count"])
                == int(before_collision["hard_collision_frame_count"])
                and int(after_collision["contact_frame_count"])
                < int(before_collision["contact_frame_count"])
            )
        )
        maximum_source_error = self.maximum_position_projection + float(
            self.adapter_config["position_tolerance_interior_m"]
        )
        return bool(
            collision_reduced
            and float(after["maximum_source_residual_m"]) <= maximum_source_error + 1e-6
            and self._temporal_passes(after["temporal"])
            and int(after["temporal"]["branch_discontinuity_count"])
            <= int(before["temporal"]["branch_discontinuity_count"])
        )

    def _minimum_nominal_alpha(
        self,
        q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> float:
        """Smallest common-nominal interpolation with no forbidden contact."""
        if not self._records(q, left_hand, right_hand):
            return 0.0
        settings = self.adapter_config["nominal_clearance_projection"]
        step = float(settings["alpha_grid_step"])
        high = None
        low = 0.0
        alpha = step
        while alpha <= 1.0 + 1e-12:
            candidate = (1.0 - min(alpha, 1.0)) * q + min(alpha, 1.0) * self.nominal
            if not self._records(candidate, left_hand, right_hand):
                high = min(alpha, 1.0)
                low = max(0.0, high - step)
                break
            alpha += step
        if high is None:
            # The common nominal itself is expected to be collision-free for all
            # hand states.  Fail closed if the active model contradicts that.
            return float("inf")
        for _ in range(int(settings["bisection_iterations"])):
            middle = 0.5 * (low + high)
            candidate = (1.0 - middle) * q + middle * self.nominal
            if self._records(candidate, left_hand, right_hand):
                low = middle
            else:
                high = middle
        return min(1.0, high + float(settings["alpha_safety_margin"]))

    def _nominal_clearance_projection(
        self,
        q: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Clear residual collision using one smooth common nominal envelope."""
        settings = self.adapter_config["nominal_clearance_projection"]
        if not bool(settings["enabled"]):
            return q, {"enabled": False}
        required = np.asarray(
            [
                self._minimum_nominal_alpha(q[frame], left_hand[frame], right_hand[frame])
                for frame in range(len(q))
            ],
            dtype=np.float64,
        )
        if not np.isfinite(required).all():
            return q, {
                "enabled": True,
                "status": "COMMON_NOMINAL_NOT_COLLISION_FREE",
                "required_alpha": required,
                "selected": None,
            }
        candidates: list[dict[str, Any]] = []
        selected_q: np.ndarray | None = None
        for sigma in map(float, settings["temporal_sigma_candidates_frames"]):
            smooth = gaussian_filter1d(required, sigma=sigma, mode="nearest")
            positive = smooth > 1e-12
            scale = float(np.max(required[positive] / smooth[positive])) if np.any(positive) else 1.0
            alpha = np.clip(smooth * max(1.0, scale), 0.0, 1.0)
            alpha = np.maximum(alpha, required)
            candidate = (1.0 - alpha[:, None]) * q + alpha[:, None] * self.nominal
            candidate = candidate.astype(np.float32).astype(np.float64)
            state = self._candidate_state(candidate, source_model, left_hand, right_hand, fps)
            maximum_error = float(state["maximum_source_residual_m"])
            projection_max = max(0.0, maximum_error - float(self.adapter_config["position_tolerance_interior_m"]))
            passes = bool(
                int(state["collision"]["hard_collision_frame_count"]) == 0
                and int(state["collision"]["contact_frame_count"]) == 0
                and self._temporal_passes(state["temporal"])
                and int(state["temporal"]["branch_discontinuity_count"]) == 0
                and projection_max <= self.maximum_position_projection + 1e-9
            )
            row = {
                "sigma_frames": sigma,
                "alpha_mean": float(np.mean(alpha)),
                "alpha_max": float(np.max(alpha)),
                "maximum_source_residual_m": maximum_error,
                "implied_maximum_projection_m": projection_max,
                "contact_frame_count": int(state["collision"]["contact_frame_count"]),
                "hard_collision_frame_count": int(state["collision"]["hard_collision_frame_count"]),
                "temporal": state["temporal"],
                "passes": passes,
            }
            candidates.append(row)
            if passes:
                selected_q = candidate
                break
        return (
            selected_q if selected_q is not None else q,
            {
                "enabled": True,
                "status": "PASS" if selected_q is not None else "NO_CANDIDATE_PASSED",
                "required_alpha": required,
                "required_alpha_nonzero_frames": int(np.count_nonzero(required)),
                "required_alpha_mean": float(np.mean(required)),
                "required_alpha_max": float(np.max(required)),
                "candidates": candidates,
                "selected": next((row for row in candidates if row["passes"]), None),
            },
        )

    @staticmethod
    def _nearest_orientation_inside_tolerance(
        achieved: np.ndarray,
        raw: np.ndarray,
        tolerance: float,
    ) -> tuple[np.ndarray, float]:
        relative = np.asarray(raw, dtype=np.float64) @ np.asarray(achieved, dtype=np.float64).T
        vector = Rotation.from_matrix(relative).as_rotvec()
        angle = float(np.linalg.norm(vector))
        if angle <= tolerance:
            return np.asarray(raw, dtype=np.float64).copy(), 0.0
        feasible = Rotation.from_rotvec(vector * (tolerance / angle)).as_matrix() @ achieved
        return feasible, angle - tolerance

    def adapt(
        self,
        incoming_targets: Mapping[str, np.ndarray],
        initial_q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
        include_orientation: bool,
    ) -> MorphologyAdapterResult:
        required = {
            f"{side}_wrist_position" for side in SIDES
        } | {f"{side}_wrist_rotation" for side in SIDES}
        missing = required - set(incoming_targets)
        if missing:
            raise KeyError(f"missing method-blind target arrays: {sorted(missing)}")
        raw_position_model = {
            side: np.asarray(incoming_targets[f"{side}_wrist_position"], dtype=np.float64).copy()
            for side in SIDES
        }
        raw_rotation_model = {
            side: np.asarray(incoming_targets[f"{side}_wrist_rotation"], dtype=np.float64).copy()
            for side in SIDES
        }
        count = len(raw_position_model["left"])
        if any(len(value) != count for value in (*raw_position_model.values(), *raw_rotation_model.values())):
            raise ValueError("inconsistent target lengths")
        q_input = np.asarray(initial_q, dtype=np.float64)
        if q_input.shape != (count, 14):
            raise ValueError(f"initial common IK q has shape {q_input.shape}, expected {(count, 14)}")
        source_world = {
            side: self.g1.model_to_world_position(raw_position_model[side]) for side in SIDES
        }
        q_tracked, tracking_report = self._track(
            q_input,
            raw_position_model,
            raw_rotation_model,
            np.asarray(left_hand, dtype=np.float64),
            np.asarray(right_hand, dtype=np.float64),
            float(fps),
        )
        q_projected, collision_report = self._collision_repair(
            q_tracked,
            q_input,
            raw_position_model,
            np.asarray(left_hand, dtype=np.float64),
            np.asarray(right_hand, dtype=np.float64),
            float(fps),
        )
        nominal_clearance_report: dict[str, Any] = {"enabled": False}
        if int(collision_report["final"]["collision"]["hard_collision_frame_count"]) > 0:
            q_projected, nominal_clearance_report = self._nominal_clearance_projection(
                q_projected,
                raw_position_model,
                np.asarray(left_hand, dtype=np.float64),
                np.asarray(right_hand, dtype=np.float64),
                float(fps),
            )
        q_projected = np.asarray(q_projected, dtype=np.float32).astype(np.float64)
        achieved_position_model, achieved_rotation_model = self._pose_arrays(q_projected)
        achieved_world = {
            side: self.g1.model_to_world_position(achieved_position_model[side]) for side in SIDES
        }
        realized_world, position_projection, projection_reason = self._nearest_realized_targets(
            source_world,
            achieved_world,
            q_projected,
            np.linalg.norm(q_tracked - q_input, axis=1) > 1e-8,
            np.linalg.norm(q_projected - q_tracked, axis=1) > 1e-8,
        )
        feasible_position_model = {
            side: self.g1.world_to_model_position(realized_world[side]) for side in SIDES
        }
        orientation_projection = np.zeros((count, 2), dtype=np.float64)
        feasible_rotation_model = {
            side: raw_rotation_model[side].copy() for side in SIDES
        }
        if include_orientation:
            tolerance = float(self.adapter_config["orientation_tolerance_interior_rad"])
            for side_index, side in enumerate(SIDES):
                for frame in range(count):
                    feasible, magnitude = self._nearest_orientation_inside_tolerance(
                        achieved_rotation_model[side][frame],
                        raw_rotation_model[side][frame],
                        tolerance,
                    )
                    feasible_rotation_model[side][frame] = feasible
                    orientation_projection[frame, side_index] = magnitude
        feasible_targets = {
            **{f"{side}_wrist_position": feasible_position_model[side] for side in SIDES},
            **{f"{side}_wrist_rotation": feasible_rotation_model[side] for side in SIDES},
        }
        raw_targets = {
            **{f"{side}_wrist_position": raw_position_model[side] for side in SIDES},
            **{f"{side}_wrist_rotation": raw_rotation_model[side] for side in SIDES},
        }
        bounds = self.adapter_config["projection_bounds"]
        final_state = self._candidate_state(
            q_projected,
            raw_position_model,
            np.asarray(left_hand, dtype=np.float64),
            np.asarray(right_hand, dtype=np.float64),
            float(fps),
        )
        final_collision = final_state["collision"]
        temporal = self._temporal_metrics(q_projected, left_hand, right_hand, float(fps))
        metadata = {
            "schema_version": "common_g1_morphology_adapter_result_v1",
            "method_blind": True,
            "representation_mode_consumed": False,
            "episode_index_consumed": False,
            "object_pose_consumed": False,
            "task_or_physical_outcome_consumed": False,
            "include_orientation": bool(include_orientation),
            "tracking": tracking_report,
            "collision_repair": collision_report,
            "nominal_clearance_projection": nominal_clearance_report,
            "projection_reason": projection_reason,
            "position_projection_m": {
                "mean": float(np.mean(position_projection)),
                "p95": float(np.quantile(position_projection, 0.95)),
                "max": float(np.max(position_projection)),
                "morphology_bound": float(bounds["maximum_position_projection_m"]),
                "within_bound": bool(np.max(position_projection) <= float(bounds["maximum_position_projection_m"]) + 1e-12),
            },
            "orientation_projection_rad": {
                "mean": float(np.mean(orientation_projection)),
                "p95": float(np.quantile(orientation_projection, 0.95)),
                "max": float(np.max(orientation_projection)),
                "morphology_bound": float(bounds["maximum_orientation_projection_rad"]),
                "within_bound": bool(np.max(orientation_projection) <= float(bounds["maximum_orientation_projection_rad"]) + 1e-12),
            },
            "temporal": temporal,
            "hard_self_collision_frames": int(final_collision["hard_collision_frame_count"]),
            "self_collision_contact_frames": int(final_collision["contact_frame_count"]),
            "finite": bool(
                np.isfinite(q_projected).all()
                and all(np.isfinite(value).all() for value in feasible_targets.values())
            ),
        }
        return MorphologyAdapterResult(
            raw_targets=raw_targets,
            feasible_targets=feasible_targets,
            projected_q=q_projected,
            position_projection_m=position_projection,
            orientation_projection_rad=orientation_projection,
            achieved_positions_model=achieved_position_model,
            achieved_rotations_model=achieved_rotation_model,
            metadata=metadata,
        )
