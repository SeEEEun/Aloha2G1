"""Frozen-wrist left Dex3 thumb/index interaction IK."""
from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares

from aloha_g1_v15.kinematics import ActiveG1Dex3

from .collision_eval import (
    CollisionClassifier,
    body_digit,
    body_side,
    is_torso,
    is_wrist_or_palm,
)


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


class Dex3InteractionIK:
    """Optimize only left thumb/index joints; arm, wrist, and third stay fixed."""

    def __init__(
        self,
        runtime: ActiveG1Dex3,
        v1_config: Mapping[str, Any],
        v2_config: Mapping[str, Any],
    ):
        self.runtime = runtime
        self.v1_config = v1_config
        self.config = v2_config["dex3_ik"]
        if v2_config["method"] != "proposed" or v2_config["side"] != "left":
            raise ValueError("hand v2 currently supports Proposed left-hand diagnostics only")
        self.side = "left"
        self.classifier = CollisionClassifier(runtime, dict(v1_config))
        self.names = tuple(runtime.hand_joint_names[self.side])
        self.limits = np.asarray(runtime.hand_limits[self.side], dtype=np.float64)

        task_names = tuple(runtime.contacts["left_A"].joint_names) + tuple(
            runtime.contacts["left_B"].joint_names
        )
        third_names = tuple(runtime.contacts["left_C"].joint_names)
        self.task_indices = np.asarray([self.names.index(name) for name in task_names], dtype=np.int64)
        self.third_indices = np.asarray([self.names.index(name) for name in third_names], dtype=np.int64)
        if len(self.task_indices) != int(self.config["optimized_dof"]):
            raise RuntimeError("unexpected task-finger DoF")

        primitive_path = v1_config["hand_mapping"]["primitive_source"]
        primitive_data = json.loads(open(primitive_path, encoding="utf-8").read())
        row = primitive_data["primitives"][self.config["reference_primitive"]]
        lookup = dict(zip(row["joint_names"], row["qpos"]))
        self.reference_q = np.asarray([lookup[name] for name in self.names], dtype=np.float64)
        self.third_neutral_q = runtime.open_hand_q[self.side][self.third_indices].copy()
        self.template_q = self.reference_q.copy()
        self.template_q[self.third_indices] = self.third_neutral_q
        epsilon = float(self.config["bound_epsilon_rad"])
        self.lower = self.limits[self.task_indices, 0] + epsilon
        self.upper = self.limits[self.task_indices, 1] - epsilon
        if np.any(self.lower >= self.upper):
            raise RuntimeError("invalid Dex3 task-finger limits")

    def full_q(self, task_q: np.ndarray) -> np.ndarray:
        value = self.template_q.copy()
        value[self.task_indices] = np.asarray(task_q, dtype=np.float64)
        value[self.third_indices] = self.third_neutral_q
        return value

    def _contact_pose(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        spec = self.runtime.contacts[label]
        body = self.runtime.body_ids[spec.link]
        rotation = np.asarray(self.runtime.data.xmat[body], dtype=np.float64).reshape(3, 3)
        position = np.asarray(self.runtime.data.xpos[body], dtype=np.float64) + rotation @ spec.local_position
        normal = rotation @ spec.local_normal
        normal /= np.linalg.norm(normal)
        return position, normal

    def _left_collision_depths(self) -> np.ndarray:
        """Fixed-length penetration residual ordered by safety priority."""
        depths = np.zeros(6, dtype=np.float64)
        for record in self.classifier.records():
            if not any(body_side(name) == "left" and body_digit(name) for name in record.bodies):
                continue
            depth = max(0.0, -float(record.distance_m) - self.classifier.tolerance)
            digits = {body_digit(name) for name in record.bodies} - {None}
            sides = {body_side(name) for name in record.bodies} - {None}
            if any(is_wrist_or_palm(name) for name in record.bodies):
                depths[0] = max(depths[0], depth)
            elif any(is_torso(name) for name in record.bodies):
                depths[1] = max(depths[1], depth)
            elif digits == {"THUMB", "INDEX"} and len(sides) == 1:
                depths[2] = max(depths[2], depth)
            elif len(sides) == 2:
                depths[3] = max(depths[3], depth)
            elif "THIRD" in digits:
                depths[4] = max(depths[4], depth)
            else:
                depths[5] = max(depths[5], depth)
        return depths

    def _residual(
        self,
        task_q: np.ndarray,
        arm_q: np.ndarray,
        right_q: np.ndarray,
        target: Mapping[str, Any],
        previous: np.ndarray,
        previous2: np.ndarray,
        temporal: bool,
    ) -> np.ndarray:
        q = self.full_q(task_q)
        self.runtime.assign(arm_q, q, right_q)
        thumb, thumb_normal = self._contact_pose("left_A")
        index, index_normal = self._contact_pose("left_B")
        position_scale = float(self.config["contact_position_scale_m"])
        residuals: list[np.ndarray] = [
            np.sqrt(float(self.config["contact_weight"]))
            * (thumb - np.asarray(target["thumb_target_m"]))
            / position_scale,
            np.sqrt(float(self.config["contact_weight"]))
            * (index - np.asarray(target["index_target_m"]))
            / position_scale,
            np.sqrt(float(self.config["orientation_weight"]))
            * np.cross(thumb_normal, np.asarray(target["thumb_target_normal"])),
            np.sqrt(float(self.config["orientation_weight"]))
            * np.cross(index_normal, np.asarray(target["index_target_normal"])),
            np.sqrt(float(self.config["reference_regularization_weight"]))
            * (task_q - self.reference_q[self.task_indices])
            / float(self.config["regularization_scale_rad"]),
        ]
        margin = float(self.config["joint_limit_margin_rad"])
        lower_gap = task_q - self.lower
        upper_gap = self.upper - task_q
        residuals.extend(
            (
                np.sqrt(float(self.config["joint_limit_margin_weight"]))
                * np.maximum(0.0, margin - lower_gap)
                / margin,
                np.sqrt(float(self.config["joint_limit_margin_weight"]))
                * np.maximum(0.0, margin - upper_gap)
                / margin,
                np.sqrt(float(self.config["collision_weight"]))
                * self._left_collision_depths()
                / float(self.config["collision_penetration_scale_m"]),
            )
        )
        if temporal:
            residuals.extend(
                (
                    np.sqrt(float(self.config["temporal_velocity_weight"]))
                    * (task_q - previous[self.task_indices]),
                    np.sqrt(float(self.config["temporal_acceleration_weight"]))
                    * (task_q - 2.0 * previous[self.task_indices] + previous2[self.task_indices]),
                )
            )
        return np.concatenate(residuals)

    def evaluate(
        self,
        q: np.ndarray,
        arm_q: np.ndarray,
        right_q: np.ndarray,
        target: Mapping[str, Any],
    ) -> dict[str, Any]:
        q = np.asarray(q, dtype=np.float64)
        self.runtime.assign(arm_q, q, right_q)
        thumb, thumb_normal = self._contact_pose("left_A")
        index, index_normal = self._contact_pose("left_B")
        third, _ = self._contact_pose("left_C")
        thumb_error = float(np.linalg.norm(thumb - np.asarray(target["thumb_target_m"])))
        index_error = float(np.linalg.norm(index - np.asarray(target["index_target_m"])))
        target_center = 0.5 * (
            np.asarray(target["thumb_target_m"]) + np.asarray(target["index_target_m"])
        )
        pinch_center = 0.5 * (thumb + index)
        records = self.classifier.records()
        left_records = [
            row
            for row in records
            if any(body_side(name) == "left" and body_digit(name) for name in row.bodies)
        ]
        wrist_records = [
            row for row in left_records if any(is_wrist_or_palm(name) for name in row.bodies)
        ]
        torso_records = [row for row in left_records if any(is_torso(name) for name in row.bodies)]
        self_records = [row for row in left_records if row.enhanced_same_hand]
        limits_ok = bool(
            q.shape == (7,)
            and np.isfinite(q).all()
            and np.all(q >= self.limits[:, 0] - 1e-9)
            and np.all(q <= self.limits[:, 1] + 1e-9)
        )
        return {
            "dex3_q": q,
            "dex3_q_shape": list(q.shape),
            "finite": bool(np.isfinite(q).all()),
            "joint_limits_ok": limits_ok,
            "joint_limit_violation_count": int(
                np.count_nonzero((q < self.limits[:, 0] - 1e-9) | (q > self.limits[:, 1] + 1e-9))
            ),
            "thumb_position_m": thumb,
            "index_position_m": index,
            "third_position_m": third,
            "thumb_target_error_m": thumb_error,
            "index_target_error_m": index_error,
            "mean_task_finger_error_m": 0.5 * (thumb_error + index_error),
            "max_task_finger_error_m": max(thumb_error, index_error),
            "physical_pinch_center_error_m": float(np.linalg.norm(pinch_center - target_center)),
            "thumb_orientation_error_rad": _angle(
                thumb_normal, np.asarray(target["thumb_target_normal"])
            ),
            "index_orientation_error_rad": _angle(
                index_normal, np.asarray(target["index_target_normal"])
            ),
            "actual_thumb_index_center_separation_m": float(np.linalg.norm(index - thumb)),
            "distance_from_phone_pinch_initialization_rad": float(np.linalg.norm(q - self.reference_q)),
            "third_finger_policy": "ACTIVE_MODEL_OPEN_NEUTRAL_FIXED",
            "third_finger_q": q[self.third_indices],
            "third_finger_neutral_exact": bool(np.array_equal(q[self.third_indices], self.third_neutral_q)),
            "finger_self_collision_count": len(self_records),
            "finger_wrist_collision_count": len(wrist_records),
            "finger_palm_collision_count": 0,
            "finger_palm_collision_note": "active model palm collision geoms belong to wrist_yaw_link and are counted as wrist",
            "finger_torso_collision_count": len(torso_records),
            "third_finger_task_object_contact": "NOT_EVALUATED_SOURCE_OBJECT_POSE_UNAVAILABLE",
            "total_collision_pairs": len(records),
            "hand_related_collision_pairs": sum(
                any(body_digit(name) is not None for name in row.bodies) for row in records
            ),
            "left_hand_related_collision_pairs": len(left_records),
            "v1_gate_collision_pairs": sum(row.v1_gate_relevant for row in records),
            "collision_records": [
                {
                    "pair": row.pair,
                    "distance_m": row.distance_m,
                    "category": row.category,
                    "cause_group": row.cause_group,
                    "v1_gate_relevant": row.v1_gate_relevant,
                    "enhanced_same_hand": row.enhanced_same_hand,
                }
                for row in records
            ],
        }

    def solve(
        self,
        arm_q: np.ndarray,
        right_q: np.ndarray,
        target: Mapping[str, Any],
        *,
        initial_q: np.ndarray | None = None,
        previous_q: np.ndarray | None = None,
        previous2_q: np.ndarray | None = None,
        temporal: bool = False,
    ) -> dict[str, Any]:
        initial = self.reference_q if initial_q is None else np.asarray(initial_q, dtype=np.float64)
        previous = self.reference_q if previous_q is None else np.asarray(previous_q, dtype=np.float64)
        previous2 = previous if previous2_q is None else np.asarray(previous2_q, dtype=np.float64)
        x0 = np.clip(initial[self.task_indices], self.lower, self.upper)
        objective_initial = float(
            np.linalg.norm(
                self._residual(x0, arm_q, right_q, target, previous, previous2, temporal)
            )
        )
        solution = least_squares(
            lambda value: self._residual(
                value, arm_q, right_q, target, previous, previous2, temporal
            ),
            x0,
            bounds=(self.lower, self.upper),
            max_nfev=int(self.config["max_function_evaluations"]),
            xtol=float(self.config["xtol"]),
            ftol=float(self.config["ftol"]),
            gtol=float(self.config["gtol"]),
        )
        q = self.full_q(solution.x)
        metrics = self.evaluate(q, arm_q, right_q, target)
        objective_final = float(
            np.linalg.norm(
                self._residual(solution.x, arm_q, right_q, target, previous, previous2, temporal)
            )
        )
        mean_error = float(metrics["mean_task_finger_error_m"])
        at_bound = bool(
            np.any(np.isclose(solution.x, self.lower, atol=5e-5))
            or np.any(np.isclose(solution.x, self.upper, atol=5e-5))
        )
        feasible = bool(mean_error <= float(self.config["contact_feasible_mean_error_m"]))
        if feasible:
            blocker = None
        elif at_bound:
            blocker = "DEX3_FINGER_WORKSPACE_LIMIT"
        else:
            blocker = "FROZEN_WRIST_INCOMPATIBLE"
        return {
            "q": q,
            "metrics": metrics,
            "solver": {
                "success": bool(solution.success),
                "status": int(solution.status),
                "message": str(solution.message),
                "function_evaluations": int(solution.nfev),
                "cost": float(solution.cost),
                "optimality": float(solution.optimality),
                "objective_norm_initial": objective_initial,
                "objective_norm_final": objective_final,
                "contact_feasible": feasible,
                "active_joint_bound": at_bound,
                "blocker_classification": blocker,
                "temporal_terms_enabled": temporal,
            },
        }
