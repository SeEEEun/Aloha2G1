"""Active-model construction of feasible semantic thumb/index primitives."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from aloha_g1_dataset_v1.core import (
    inverse_transform,
    physical_pinch_frame,
    raw_wrist_pose,
)
from aloha_g1_hand_v2.collision_eval import (
    CollisionClassifier,
    body_digit,
    body_side,
    is_wrist_or_palm,
)


PHASES = ("OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE")


def angle_rad(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


class SemanticPrimitiveBuilder:
    """Optimize five task-finger DoF; never alter arm or third-finger DoF."""

    def __init__(
        self,
        runtime: Any,
        v1_config: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.runtime = runtime
        self.v1 = v1_config
        self.config = config
        self.optimizer = config["primitive_optimizer"]
        self.classifier = CollisionClassifier(runtime, dict(v1_config))
        self.nominal_arm = np.asarray(v1_config["nominal_g1_arm_q"], dtype=np.float64)
        self.labels = {
            side: tuple(v1_config["target_frames"][f"{side}_physical_pinch_contacts"])
            for side in ("left", "right")
        }
        self.old_tools = {
            side: np.asarray(
                v1_config["target_frames"][f"{side}_wrist_to_physical_pinch"],
                dtype=np.float64,
            )
            for side in ("left", "right")
        }
        self.task_indices: dict[str, np.ndarray] = {}
        self.third_indices: dict[str, np.ndarray] = {}
        self.tip_geoms: dict[str, tuple[int, int]] = {}
        for side in ("left", "right"):
            names = tuple(runtime.hand_joint_names[side])
            third_names = tuple(runtime.contacts[f"{side}_C"].joint_names)
            third = np.asarray([names.index(name) for name in third_names], dtype=np.int64)
            task = np.asarray([index for index in range(7) if index not in set(third)], dtype=np.int64)
            if len(task) != 5 or len(third) != 2:
                raise RuntimeError(f"unexpected {side} Dex3 topology")
            self.task_indices[side] = task
            self.third_indices[side] = third
            role_by_digit = {
                body_digit(runtime.contacts[f"{side}_{role}"].link): role
                for role in ("A", "B")
            }
            self.tip_geoms[side] = (
                self._collision_geom(runtime.contacts[f"{side}_{role_by_digit['THUMB']}"].link),
                self._collision_geom(runtime.contacts[f"{side}_{role_by_digit['INDEX']}"].link),
            )
        self.seeds = self._load_seeds(Path(self.optimizer["left_seed_source"]))

    def _collision_geom(self, body_name: str) -> int:
        body = self.runtime.body_ids[body_name]
        candidates = [
            geom
            for geom in range(self.runtime.model.ngeom)
            if int(self.runtime.model.geom_bodyid[geom]) == body
            and (
                int(self.runtime.model.geom_contype[geom]) != 0
                or int(self.runtime.model.geom_conaffinity[geom]) != 0
            )
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one active collision geom on {body_name}, got {candidates}")
        return int(candidates[0])

    def _load_seeds(self, source: Path) -> dict[str, np.ndarray]:
        artifact = json.loads(source.read_text(encoding="utf-8"))
        row = artifact["seed"]
        lookup = dict(zip(row["joint_order"], row["joint_values"]))
        left_names = tuple(self.runtime.hand_joint_names["left"])
        left = self.runtime.open_hand_q["left"].copy()
        for index in self.task_indices["left"]:
            left[index] = float(lookup[left_names[index]])

        right_names = tuple(self.runtime.hand_joint_names["right"])
        right = self.runtime.open_hand_q["right"].copy()
        for index in self.task_indices["right"]:
            name = right_names[index]
            left_name = name.replace("right_", "left_", 1)
            sign = 1.0 if "thumb_0_joint" in name else -1.0
            right[index] = sign * float(lookup[left_name])
        return {"left": left, "right": right}

    def _assign_side(self, side: str, q: np.ndarray) -> None:
        left = q if side == "left" else self.runtime.open_hand_q["left"]
        right = q if side == "right" else self.runtime.open_hand_q["right"]
        self.runtime.assign(self.nominal_arm, left, right)

    def surface_aperture(self, side: str, q: np.ndarray) -> tuple[float, np.ndarray]:
        self._assign_side(side, q)
        from_to = np.zeros(6, dtype=np.float64)
        distance = mujoco.mj_geomDistance(
            self.runtime.model,
            self.runtime.data,
            self.tip_geoms[side][0],
            self.tip_geoms[side][1],
            0.3,
            from_to,
        )
        return float(distance), from_to

    def wrist_to_pinch(self, side: str, q: np.ndarray) -> np.ndarray:
        self._assign_side(side, q)
        wrist = raw_wrist_pose(self.runtime, side)
        pinch = physical_pinch_frame(self.runtime, side, self.labels[side])
        return inverse_transform(wrist) @ pinch

    def _internal_collision_depths(self, side: str) -> np.ndarray:
        """Fixed task-finger overlap/wrist/third residual in safety priority order."""
        result = np.zeros(4, dtype=np.float64)
        for record in self.classifier.records():
            sides = {body_side(name) for name in record.bodies} - {None}
            if sides != {side}:
                continue
            digits = {body_digit(name) for name in record.bodies} - {None}
            if not digits.intersection({"THUMB", "INDEX"}):
                continue
            depth = max(0.0, -float(record.distance_m) - self.classifier.tolerance)
            if digits == {"THUMB", "INDEX"}:
                result[0] = max(result[0], depth)
            elif "THUMB" in digits and any(is_wrist_or_palm(name) for name in record.bodies):
                result[1] = max(result[1], depth)
            elif "INDEX" in digits and any(is_wrist_or_palm(name) for name in record.bodies):
                result[2] = max(result[2], depth)
            elif "THIRD" in digits:
                result[3] = max(result[3], depth)
        return result

    def evaluate(self, side: str, q: np.ndarray) -> dict[str, Any]:
        q = np.asarray(q, dtype=np.float64)
        aperture, closest = self.surface_aperture(side, q)
        frame = self.wrist_to_pinch(side, q)
        self._assign_side(side, q)
        first, first_normal = self.runtime.contact_pose(self.labels[side][0])
        second, second_normal = self.runtime.contact_pose(self.labels[side][1])
        old = self.old_tools[side]
        rotation_delta = Rotation.from_matrix(old[:3, :3].T @ frame[:3, :3]).magnitude()
        task_limits = self.runtime.hand_limits[side]
        internal = self._internal_collision_depths(side)
        target = float(
            self.config["object_class_geometry"][side]["target_surface_aperture_m"]
        )
        index_role = next(
            role
            for role in ("A", "B")
            if body_digit(self.runtime.contacts[f"{side}_{role}"].link) == "INDEX"
        )
        pad_allowance = float(
            np.sum(
                [
                    self.runtime.contacts[f"{side}_{role}"].half_extent[0]
                    for role in ("A", "B")
                ]
            )
        )
        return {
            "q": q,
            "finite": bool(np.isfinite(q).all()),
            "joint_limits_ok": bool(
                q.shape == (7,)
                and np.all(q >= task_limits[:, 0] - 1e-9)
                and np.all(q <= task_limits[:, 1] + 1e-9)
            ),
            "minimum_joint_margin_rad": float(
                np.min(np.minimum(q - task_limits[:, 0], task_limits[:, 1] - q))
            ),
            "surface_aperture_m": aperture,
            "target_object_class_aperture_m": target,
            "surface_aperture_error_m": aperture - target,
            "surface_aperture_allowance_m": pad_allowance,
            "object_class_opening_compatible": bool(
                aperture >= target - self.classifier.tolerance
                and aperture - target <= pad_allowance
            ),
            "closest_surface_points_m": closest.reshape(2, 3),
            "thumb_index_contact_center_separation_m": float(np.linalg.norm(second - first)),
            "pad_normal_opposition_angle_deg": float(
                np.degrees(angle_rad(first_normal, -second_normal))
            ),
            "pad_normal_mutual_angle_deg": float(
                np.degrees(angle_rad(first_normal, second_normal))
            ),
            "wrist_to_pinch": frame,
            "old_wrist_to_pinch": old,
            "translation_delta_m": float(np.linalg.norm(frame[:3, 3] - old[:3, 3])),
            "pinch_center_delta_m": float(np.linalg.norm(frame[:3, 3] - old[:3, 3])),
            "rotation_delta_rad": float(rotation_delta),
            "closing_axis_delta_rad": angle_rad(frame[:3, 1], old[:3, 1]),
            "same_hand_internal_collision_depths_m": internal,
            "same_hand_internal_collision": bool(np.any(internal > 0.0)),
            "thumb_index_pathological_overlap": bool(aperture < -self.classifier.tolerance),
            "index_pad_largest_half_extent_m": float(
                np.max(self.runtime.contacts[f"{side}_{index_role}"].half_extent)
            ),
        }

    def _optimize_grasp(self, side: str) -> dict[str, Any]:
        task = self.task_indices[side]
        third = self.third_indices[side]
        seed = self.seeds[side].copy()
        seed[third] = self.runtime.open_hand_q[side][third]
        limits = self.runtime.hand_limits[side][task]
        epsilon = float(self.optimizer["joint_bound_epsilon_rad"])
        lower = limits[:, 0] + epsilon
        upper = limits[:, 1] - epsilon
        target_aperture = float(
            self.config["object_class_geometry"][side]["target_surface_aperture_m"]
        )
        old = self.old_tools[side]

        def full_q(value: np.ndarray) -> np.ndarray:
            q = seed.copy()
            q[task] = value
            q[third] = self.runtime.open_hand_q[side][third]
            return q

        def residual(value: np.ndarray) -> np.ndarray:
            q = full_q(value)
            aperture, _ = self.surface_aperture(side, q)
            frame = self.wrist_to_pinch(side, q)
            self._assign_side(side, q)
            _, first_normal = self.runtime.contact_pose(self.labels[side][0])
            _, second_normal = self.runtime.contact_pose(self.labels[side][1])
            rotation = Rotation.from_matrix(old[:3, :3].T @ frame[:3, :3]).as_rotvec()
            tool_weight = np.sqrt(float(self.optimizer["tool_compatibility_weight"]))
            return np.concatenate(
                (
                    np.asarray(
                        [
                            (aperture - target_aperture)
                            / float(self.optimizer["surface_aperture_scale_m"]),
                            (float(np.dot(first_normal, second_normal)) + 1.0)
                            / float(self.optimizer["pad_normal_opposition_scale"]),
                        ]
                    ),
                    tool_weight
                    * (frame[:3, 3] - old[:3, 3])
                    / float(self.optimizer["tool_translation_scale_m"]),
                    tool_weight
                    * rotation
                    / float(self.optimizer["tool_rotation_scale_rad"]),
                    np.sqrt(float(self.optimizer["seed_regularization_weight"]))
                    * (value - seed[task]),
                    np.sqrt(float(self.optimizer["internal_collision_weight"]))
                    * self._internal_collision_depths(side)
                    / float(self.optimizer["internal_collision_scale_m"]),
                )
            )

        initial = np.clip(seed[task], lower, upper)
        solution = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=int(self.optimizer["max_function_evaluations"]),
            xtol=float(self.optimizer["xtol"]),
            ftol=float(self.optimizer["ftol"]),
            gtol=float(self.optimizer["gtol"]),
        )
        q = full_q(solution.x)
        return {
            "q": q,
            "metrics": self.evaluate(side, q),
            "solver": {
                "success": bool(solution.success),
                "status": int(solution.status),
                "message": str(solution.message),
                "function_evaluations": int(solution.nfev),
                "cost": float(solution.cost),
                "optimality": float(solution.optimality),
                "deterministic_seed_q": seed,
                "optimized_indices": task,
                "third_indices_not_optimized": third,
            },
        }

    def _pregrasp(self, side: str, open_q: np.ndarray, grasp_q: np.ndarray) -> dict[str, Any]:
        samples = int(self.optimizer["pregrasp_interpolation_samples"])
        grasp_metrics = self.evaluate(side, grasp_q)
        clearance = 2.0 * float(grasp_metrics["index_pad_largest_half_extent_m"])
        target = float(grasp_metrics["surface_aperture_m"]) + clearance
        candidates: list[tuple[tuple[float, float], float, np.ndarray, dict[str, Any]]] = []
        for alpha in np.linspace(0.0, 1.0, samples):
            q = (1.0 - alpha) * open_q + alpha * grasp_q
            metrics = self.evaluate(side, q)
            unsafe = float(
                metrics["same_hand_internal_collision"]
                or metrics["thumb_index_pathological_overlap"]
            )
            score = (unsafe, abs(float(metrics["surface_aperture_m"]) - target))
            candidates.append((score, float(alpha), q, metrics))
        selected = min(candidates, key=lambda row: (row[0], row[1]))
        return {
            "q": selected[2],
            "metrics": selected[3],
            "interpolation_alpha": selected[1],
            "target_surface_aperture_m": target,
            "clearance_m": clearance,
            "selection_rule": self.optimizer["pregrasp_clearance_rule"],
            "candidate_count": samples,
        }

    def build(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "schema_version": "active_model_semantic_primitives_v2_1",
            "exact_contact_targets_used": False,
            "arm_optimized": False,
            "third_finger_optimized_here": False,
            "sides": {},
        }
        for side in ("left", "right"):
            grasp = self._optimize_grasp(side)
            open_q = self.runtime.open_hand_q[side].copy()
            pregrasp = self._pregrasp(side, open_q, grasp["q"])
            states = {
                "OPEN": open_q,
                "PREGRASP": pregrasp["q"],
                "GRASP": grasp["q"],
                "HOLD": grasp["q"].copy(),
                "RELEASE": open_q.copy(),
            }
            output["sides"][side] = {
                "role": self.config["roles"][side],
                "joint_names": list(self.runtime.hand_joint_names[side]),
                "task_indices": self.task_indices[side],
                "third_indices": self.third_indices[side],
                "seed_q": self.seeds[side],
                "states": states,
                "grasp_optimization": grasp,
                "pregrasp_construction": pregrasp,
                "state_metrics_before_third_search": {
                    phase: self.evaluate(side, q) for phase, q in states.items()
                },
            }
        return output


def tool_compatibility(
    primitives: Mapping[str, Any], v1_config: Mapping[str, Any]
) -> dict[str, Any]:
    position_threshold = float(v1_config["ik"]["position_tolerance_m"])
    orientation_threshold = float(v1_config["ik"]["orientation_tolerance_rad"])
    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        metrics = primitives["sides"][side]["grasp_optimization"]["metrics"]
        translation = float(metrics["translation_delta_m"])
        rotation = float(metrics["rotation_delta_rad"])
        compatible = bool(
            translation <= position_threshold and rotation <= orientation_threshold
        )
        sides[side] = {
            "old_wrist_to_pinch": metrics["old_wrist_to_pinch"],
            "candidate_wrist_to_pinch": metrics["wrist_to_pinch"],
            "translation_delta_m": translation,
            "translation_delta_mm": 1000.0 * translation,
            "rotation_delta_rad": rotation,
            "rotation_delta_deg": float(np.degrees(rotation)),
            "pinch_center_delta_m": float(metrics["pinch_center_delta_m"]),
            "pinch_center_delta_mm": 1000.0 * float(metrics["pinch_center_delta_m"]),
            "closing_axis_delta_rad": float(metrics["closing_axis_delta_rad"]),
            "closing_axis_delta_deg": float(np.degrees(metrics["closing_axis_delta_rad"])),
            "translation_threshold_m": position_threshold,
            "rotation_threshold_rad": orientation_threshold,
            "compatible_with_v1_static_tool_transform": compatible,
        }
    rerun = not all(
        row["compatible_with_v1_static_tool_transform"] for row in sides.values()
    )
    return {
        "schema_version": "proposed_hand_v2_1_tool_transform_compatibility",
        "threshold_provenance": {
            "translation": "v1 shared IK position_tolerance_m",
            "rotation": "v1 shared IK orientation_tolerance_rad",
        },
        "sides": sides,
        "tool_transform_changed_requires_arm_rerun": rerun,
        "classification": (
            "TOOL_TRANSFORM_CHANGED_REQUIRES_ARM_RERUN"
            if rerun
            else "V1_TOOL_TRANSFORM_COMPATIBLE"
        ),
    }


__all__ = ["PHASES", "SemanticPrimitiveBuilder", "angle_rad", "tool_compatibility"]
