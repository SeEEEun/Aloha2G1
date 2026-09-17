"""Task-independent constrained tracking and self-collision feasibility projection."""
from __future__ import annotations

import collections
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize

from tools.doll_handoff_retargeting.common import branch_flags
from tools.doll_handoff_retargeting.models import G1Kinematics

from .common import (
    FROZEN_ROOT,
    OUTPUT_ROOT,
    REPOSITORY,
    SIDES,
    contiguous_segments,
    load_json,
    load_trajectory,
    rotation_error_rad,
    sha256_file,
    stable_episode_id,
    stable_json_sha256,
    verify_frozen_contract,
    write_json,
)


DEFAULT_CONFIG = REPOSITORY / "configs/doll_handoff_g1_feasibility_resolver.json"


@dataclass
class EpisodeResult:
    episode: int
    q_before: np.ndarray
    q_after: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    source_position_world: dict[str, np.ndarray]
    source_orientation_model: dict[str, np.ndarray]
    realized_position_world: dict[str, np.ndarray]
    realized_orientation_model: dict[str, np.ndarray]
    achieved_position_world: dict[str, np.ndarray]
    achieved_orientation_model: dict[str, np.ndarray]
    projection_translation_m: np.ndarray
    projection_orientation_rad: np.ndarray
    projection_reason: np.ndarray
    metadata: dict[str, Any]


class GenericG1FeasibilityResolver:
    """One common target-G1 backend with no method, phase, or episode branch.

    Frozen common-natural-arm q is the seed and nominal redundancy realization.
    This layer only repairs infeasible target tracking and robot self-penetration.
    Source targets are retained verbatim and a separate nearest feasible target is
    emitted whenever the constrained q cannot satisfy the strict source target.
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        output_root: str | Path = OUTPUT_ROOT,
    ):
        self.freeze = verify_frozen_contract()
        self.config_path = Path(config_path).resolve()
        self.config = load_json(self.config_path)
        self.output_root = Path(output_root).resolve()
        if self.output_root == FROZEN_ROOT.resolve():
            raise RuntimeError("feasibility output may not overwrite frozen motion")
        if self.config.get("episode_specific_parameters_allowed") is not False:
            raise RuntimeError("episode-specific feasibility parameters are forbidden")
        if self.config.get("source_target_mutation_allowed") is not False:
            raise RuntimeError("source target mutation must remain forbidden")
        frozen = self.config["frozen_input"]
        for key in (
            "implementation_sha256",
            "trajectory_file_set_sha256",
            "cartesian_target_array_set_sha256",
        ):
            if str(frozen[key]) != str(self.freeze[key]):
                raise RuntimeError(f"resolver/frozen hash mismatch for {key}")
        common_path = FROZEN_ROOT / "frozen_approval/config/common_config.json"
        scene_path = FROZEN_ROOT / "frozen_approval/scene/scene_layout.json"
        tool_path = FROZEN_ROOT / "frozen_approval/config/tool_frame_report.json"
        natural_path = (
            FROZEN_ROOT / "frozen_approval/config/common_natural_arm_solver.json"
        )
        if sha256_file(natural_path) != frozen["common_natural_arm_solver_sha256"]:
            raise RuntimeError("frozen common natural-arm solver hash mismatch")
        self.common = load_json(common_path)
        self.scene = load_json(scene_path)
        self.tool = load_json(tool_path)
        self.g1 = G1Kinematics(self.common, self.scene)
        self.transforms = {
            side: np.asarray(
                self.tool["g1"][f"{side}_wrist_to_grasp_frame"],
                dtype=np.float64,
            )
            for side in SIDES
        }
        self.nominal = np.asarray(
            self.common["resolved"]["natural_arm_redundancy"]["nominal_q"],
            dtype=np.float64,
        )
        self.acceptance = self.config["unchanged_acceptance"]
        self.strict_tolerance = float(
            self.acceptance["strict_position_tolerance_m"]
        )
        self.physical_tolerance = float(
            self.acceptance["physical_position_tolerance_m"]
        )
        self.collision_tolerance = float(
            self.acceptance["collision_penetration_tolerance_m"]
        )
        self.shoulders = self.g1.fixed_shoulder_anchors_model()
        reach = self.g1.shoulder_wrist_reach_geometry()
        self.outer_tool_radius = {
            side: float(reach["sides"][side]["upper_effective_length_m"])
            + float(reach["sides"][side]["forearm_effective_length_m"])
            + float(np.linalg.norm(self.transforms[side][:3, 3]))
            for side in SIDES
        }

    def _source_targets(
        self, values: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        positions = {
            side: np.asarray(
                values[f"target_{side}_interaction_frame_position_world"],
                dtype=np.float64,
            ).copy()
            for side in SIDES
        }
        rotations = {
            side: np.einsum(
                "tij,jk->tik",
                np.asarray(
                    values[f"target_{side}_wrist_rotation_model"],
                    dtype=np.float64,
                ),
                self.transforms[side][:3, :3],
            )
            for side in SIDES
        }
        return positions, rotations

    def _source_model(
        self, source_world: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {
            side: self.g1.world_to_model_position(source_world[side])
            for side in SIDES
        }

    def _pose(
        self, q: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.g1.wrist_state(q)
        positions: dict[str, np.ndarray] = {}
        rotations: dict[str, np.ndarray] = {}
        for side in SIDES:
            position, rotation, _, _ = self.g1.static_tool_pose_state(
                side, self.transforms[side]
            )
            positions[side] = np.asarray(position, dtype=np.float64).copy()
            rotations[side] = np.asarray(rotation, dtype=np.float64).copy()
        return positions, rotations

    def _pose_arrays(
        self, q: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        positions = {
            side: np.empty((len(q), 3), dtype=np.float64) for side in SIDES
        }
        rotations = {
            side: np.empty((len(q), 3, 3), dtype=np.float64) for side in SIDES
        }
        for frame, value in enumerate(q):
            current_position, current_rotation = self._pose(value)
            for side in SIDES:
                positions[side][frame] = current_position[side]
                rotations[side][frame] = current_rotation[side]
        return positions, rotations

    def _position_errors(
        self, q: np.ndarray, source_model: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        positions, _ = self._pose_arrays(q)
        by_side = {
            side: np.linalg.norm(positions[side] - source_model[side], axis=1)
            for side in SIDES
        }
        return np.maximum(by_side["left"], by_side["right"]), by_side

    def _temporal_metrics(
        self,
        q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> dict[str, float | int]:
        full = np.column_stack((q, left_hand, right_hand))
        step = np.abs(np.diff(full, axis=0))
        acceleration = np.abs(np.diff(full, n=2, axis=0)) * fps**2
        flags = branch_flags(
            q,
            float(self.acceptance["branch_absolute_step_norm_rad"]),
            float(self.acceptance["branch_local_multiplier"]),
        )
        violations = (q < self.g1.arm_limits[:, 0] - 1e-9) | (
            q > self.g1.arm_limits[:, 1] + 1e-9
        )
        maximum_step = float(np.max(step, initial=0.0))
        return {
            "maximum_joint_step_rad": maximum_step,
            "maximum_velocity_rad_s": maximum_step * fps,
            "maximum_acceleration_rad_s2": float(
                np.max(acceleration, initial=0.0)
            ),
            "branch_discontinuity_count": int(np.count_nonzero(flags)),
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
        }

    def _temporal_passes(self, metrics: Mapping[str, float | int]) -> bool:
        return bool(
            int(metrics["joint_limit_violation_count"]) == 0
            and float(metrics["maximum_joint_step_rad"])
            <= float(self.acceptance["maximum_joint_step_rad"]) + 1e-7
            and float(metrics["maximum_velocity_rad_s"])
            <= float(self.acceptance["maximum_velocity_rad_s"]) + 1e-7
            and float(metrics["maximum_acceleration_rad_s2"])
            <= float(self.acceptance["maximum_acceleration_rad_s2"]) + 1e-5
        )

    def _tracking_bounds(
        self, frame: int, q: np.ndarray, fps: float
    ) -> tuple[np.ndarray, np.ndarray]:
        settings = self.config["constrained_tracking"]
        lower = self.g1.arm_limits[:, 0].copy() + 1e-8
        upper = self.g1.arm_limits[:, 1].copy() - 1e-8
        step = float(self.acceptance["maximum_joint_step_rad"]) - float(
            settings["joint_step_numerical_interior_rad"]
        )
        second = (
            float(self.acceptance["maximum_acceleration_rad_s2"])
            - float(settings["acceleration_numerical_interior_rad_s2"])
        ) / fps**2
        count = len(q)
        if frame > 0:
            lower = np.maximum(lower, q[frame - 1] - step)
            upper = np.minimum(upper, q[frame - 1] + step)
        if frame + 1 < count:
            lower = np.maximum(lower, q[frame + 1] - step)
            upper = np.minimum(upper, q[frame + 1] + step)
        if frame > 1:
            lower = np.maximum(lower, 2.0 * q[frame - 1] - q[frame - 2] - second)
            upper = np.minimum(upper, 2.0 * q[frame - 1] - q[frame - 2] + second)
        if frame > 0 and frame + 1 < count:
            lower = np.maximum(
                lower, 0.5 * (q[frame + 1] + q[frame - 1] - second)
            )
            upper = np.minimum(
                upper, 0.5 * (q[frame + 1] + q[frame - 1] + second)
            )
        if frame + 2 < count:
            lower = np.maximum(lower, 2.0 * q[frame + 1] - q[frame + 2] - second)
            upper = np.minimum(upper, 2.0 * q[frame + 1] - q[frame + 2] + second)
        invalid = lower >= upper
        if np.any(invalid):
            center = np.clip(
                q[frame, invalid],
                self.g1.arm_limits[invalid, 0] + 2e-8,
                self.g1.arm_limits[invalid, 1] - 2e-8,
            )
            lower[invalid] = center - 1e-9
            upper[invalid] = center + 1e-9
        return lower, upper

    def _tracking_active_mask(
        self, error: np.ndarray
    ) -> np.ndarray:
        settings = self.config["constrained_tracking"]
        active = np.zeros(len(error), dtype=bool)
        padding = int(settings["activation_padding_frames"])
        for frame in np.flatnonzero(
            error > float(settings["activation_position_error_m"])
        ):
            active[max(0, frame - padding) : min(len(error), frame + padding + 1)] = True
        return active

    @staticmethod
    def _cap_branch_step(
        current: np.ndarray,
        candidate: np.ndarray,
        neighbors: Iterable[np.ndarray],
        radius: float,
    ) -> np.ndarray:
        """Keep a frame update inside the branch-continuous neighbor balls.

        The current point is already part of a branch-continuous trajectory.
        Restricting the optimizer result along the current-to-candidate segment
        is therefore deterministic and does not select an elbow branch.
        """
        direction = np.asarray(candidate, dtype=np.float64) - np.asarray(
            current, dtype=np.float64
        )
        alpha = 1.0
        for neighbor in neighbors:
            neighbor = np.asarray(neighbor, dtype=np.float64)
            if np.linalg.norm(current + alpha * direction - neighbor) <= radius:
                continue
            low = 0.0
            high = alpha
            for _ in range(48):
                middle = 0.5 * (low + high)
                if np.linalg.norm(current + middle * direction - neighbor) <= radius:
                    low = middle
                else:
                    high = middle
            alpha = low
        return np.asarray(current, dtype=np.float64) + alpha * direction

    def _track(
        self,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        source_rotation: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.config["constrained_tracking"]
        q = np.asarray(q_before, dtype=np.float64).copy()
        initial_error, _ = self._position_errors(q, source_model)
        active = self._tracking_active_mask(initial_error)
        before_temporal = self._temporal_metrics(q, left_hand, right_hand, fps)
        pass_reports: list[dict[str, Any]] = []
        for pass_index in range(int(settings["bidirectional_passes"])):
            pass_before = q.copy()
            order = np.flatnonzero(active)
            if pass_index % 2:
                order = order[::-1]
            accepted = 0
            attempted = 0
            for raw_frame in order:
                frame = int(raw_frame)
                lower, upper = self._tracking_bounds(frame, q, fps)
                x0 = np.minimum(np.maximum(q[frame], lower), upper)
                midpoint = 0.5 * (
                    (q[frame - 1] if frame > 0 else q[frame])
                    + (q[frame + 1] if frame + 1 < len(q) else q[frame])
                )
                old_position, _ = self._pose(q[frame])
                old_error = max(
                    float(
                        np.linalg.norm(
                            old_position[side] - source_model[side][frame]
                        )
                    )
                    for side in SIDES
                )

                def residual(value: np.ndarray) -> np.ndarray:
                    position, rotation = self._pose(value)
                    values: list[np.ndarray] = [
                        float(settings["position_residual_scale_per_m"])
                        * np.concatenate(
                            [
                                position[side] - source_model[side][frame]
                                for side in SIDES
                            ]
                        )
                    ]
                    # The source orientation remains immutable and non-gating.
                    # A tiny bearing-axis term merely breaks redundant numerical
                    # ties; it is not a relaxed acceptance threshold.
                    values.append(
                        0.002
                        * np.concatenate(
                            [
                                np.cross(
                                    rotation[side][:, 0],
                                    source_rotation[side][frame][:, 0],
                                )
                                for side in SIDES
                            ]
                        )
                    )
                    values.extend(
                        (
                            float(settings["frozen_q_residual_scale"])
                            * (value - q_before[frame]),
                            float(settings["neighbor_q_residual_scale"])
                            * (value - midpoint),
                            float(settings["nominal_q_residual_scale"])
                            * (value - self.nominal),
                        )
                    )
                    return np.concatenate(values)

                attempted += 1
                try:
                    solution = least_squares(
                        residual,
                        x0,
                        bounds=(lower, upper),
                        max_nfev=int(settings["maximum_function_evaluations"]),
                        ftol=1e-9,
                        xtol=1e-9,
                        gtol=1e-9,
                    )
                except ValueError:
                    continue
                neighbors = []
                if frame > 0:
                    neighbors.append(q[frame - 1])
                if frame + 1 < len(q):
                    neighbors.append(q[frame + 1])
                candidate = self._cap_branch_step(
                    q[frame],
                    solution.x,
                    neighbors,
                    float(self.acceptance["branch_absolute_step_norm_rad"])
                    - float(settings["branch_step_numerical_interior_rad"]),
                )
                new_position, _ = self._pose(candidate)
                new_error = max(
                    float(
                        np.linalg.norm(
                            new_position[side] - source_model[side][frame]
                        )
                    )
                    for side in SIDES
                )
                if new_error <= old_error - float(settings["minimum_improvement_m"]):
                    q[frame] = candidate
                    accepted += 1
            serialized = q.astype(np.float32).astype(np.float64)
            temporal = self._temporal_metrics(
                serialized, left_hand, right_hand, fps
            )
            if (
                not self._temporal_passes(temporal)
                or int(temporal["branch_discontinuity_count"])
                > int(before_temporal["branch_discontinuity_count"])
            ):
                q = pass_before
                retained = False
            else:
                q = serialized
                retained = True
            current_error, _ = self._position_errors(q, source_model)
            pass_reports.append(
                {
                    "pass_index": pass_index,
                    "direction": "forward" if pass_index % 2 == 0 else "backward",
                    "attempted_frames": attempted,
                    "accepted_frame_updates": accepted,
                    "pass_retained": retained,
                    "physical_source_residual_frames": int(
                        np.count_nonzero(current_error > self.physical_tolerance)
                    ),
                    "maximum_source_residual_m": float(np.max(current_error)),
                    "mean_source_residual_m": float(np.mean(current_error)),
                    "temporal": temporal,
                }
            )
        final_error, _ = self._position_errors(q, source_model)
        return q, {
            "algorithm": settings["algorithm"],
            "active_frame_count": int(np.count_nonzero(active)),
            "initial_strict_source_residual_frames": int(
                np.count_nonzero(initial_error > self.strict_tolerance)
            ),
            "initial_physical_source_residual_frames": int(
                np.count_nonzero(initial_error > self.physical_tolerance)
            ),
            "initial_maximum_source_residual_m": float(np.max(initial_error)),
            "final_strict_source_residual_frames": int(
                np.count_nonzero(final_error > self.strict_tolerance)
            ),
            "final_physical_source_residual_frames": int(
                np.count_nonzero(final_error > self.physical_tolerance)
            ),
            "final_maximum_source_residual_m": float(np.max(final_error)),
            "final_mean_source_residual_m": float(np.mean(final_error)),
            "before_temporal": before_temporal,
            "passes": pass_reports,
        }

    def _body_name(self, geom: int) -> str:
        body = int(self.g1.model.geom_bodyid[int(geom)])
        return (
            mujoco.mj_id2name(
                self.g1.model, mujoco.mjtObj.mjOBJ_BODY, body
            )
            or f"body_{body}"
        )

    @staticmethod
    def _is_torso(body: str) -> bool:
        return any(token in body for token in ("torso", "pelvis", "waist"))

    @staticmethod
    def _is_arm(body: str) -> bool:
        return any(
            token in body
            for token in ("shoulder", "upper_arm", "elbow", "forearm", "wrist")
        )

    @staticmethod
    def _is_finger(body: str) -> bool:
        return "hand_" in body and any(
            token in body for token in ("thumb", "index", "middle")
        )

    @staticmethod
    def _is_palm_or_hand(body: str) -> bool:
        return "hand_palm" in body or "wrist" in body or "hand_" in body

    def _contact_class(self, first: str, second: str) -> str | None:
        bodies = (first, second)
        sides = {
            side
            for side in SIDES
            if any(body.startswith(f"{side}_") for body in bodies)
        }
        if any(self._is_torso(body) for body in bodies) and any(
            self._is_arm(body) for body in bodies
        ):
            return "ARM_TORSO_INVALID"
        if len(sides) == 2 and all(self._is_finger(body) for body in bodies):
            return "DISTAL_HAND_HAND_CONTACT"
        if len(sides) == 2 and any(self._is_palm_or_hand(body) for body in bodies):
            if any("hand_palm" in body or "wrist" in body for body in bodies):
                return "PALM_HAND_INVALID"
            return "CROSS_ARM_INVALID"
        if len(sides) == 2:
            return "CROSS_ARM_INVALID"
        return None

    def _records(
        self,
        q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> list[dict[str, Any]]:
        self.g1.assign(q, left_hand, right_hand)
        records: list[dict[str, Any]] = []
        for contact in self.g1.data.contact:
            distance = float(contact.dist)
            if distance >= -self.collision_tolerance:
                continue
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            bodies = (self._body_name(pair[0]), self._body_name(pair[1]))
            classification = self._contact_class(*bodies)
            if classification is None:
                continue
            records.append(
                {
                    "classification": classification,
                    "geom_pair": pair,
                    "body_pair": tuple(sorted(bodies)),
                    "distance_m": distance,
                    "penetration_depth_m": -distance,
                }
            )
        return records

    @staticmethod
    def _segment_is_hard(
        classification: str, depth_m: float, duration_s: float
    ) -> bool:
        if classification == "DISTAL_HAND_HAND_CONTACT":
            return depth_m >= 0.015 and duration_s >= 0.25
        if classification == "ARM_TORSO_INVALID":
            return depth_m >= 0.015 or (
                depth_m >= 0.005 and duration_s >= 1.0
            )
        return depth_m >= 0.005 or duration_s >= 0.25

    def _collision_metrics(
        self,
        q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> dict[str, Any]:
        by_frame: dict[int, list[dict[str, Any]]] = {}
        by_class: dict[str, list[int]] = collections.defaultdict(list)
        maximum_depth = 0.0
        for frame in range(len(q)):
            records = self._records(q[frame], left_hand[frame], right_hand[frame])
            if records:
                by_frame[frame] = records
            for classification in {row["classification"] for row in records}:
                by_class[classification].append(frame)
            maximum_depth = max(
                maximum_depth,
                max(
                    (float(row["penetration_depth_m"]) for row in records),
                    default=0.0,
                ),
            )
        segments: list[dict[str, Any]] = []
        hard_frames: set[int] = set()
        hard_classes: collections.Counter[str] = collections.Counter()
        for classification, frames in by_class.items():
            for start, end in contiguous_segments(frames):
                records = [
                    row
                    for frame in range(start, end + 1)
                    for row in by_frame.get(frame, [])
                    if row["classification"] == classification
                ]
                depth = max(
                    float(row["penetration_depth_m"]) for row in records
                )
                duration = float((end - start + 1) / fps)
                hard = self._segment_is_hard(classification, depth, duration)
                if hard:
                    hard_frames.update(range(start, end + 1))
                    hard_classes[classification] += end - start + 1
                segments.append(
                    {
                        "classification": classification,
                        "start_frame": start,
                        "end_frame": end,
                        "frame_count": end - start + 1,
                        "duration_s": duration,
                        "maximum_penetration_depth_m": depth,
                        "hard": hard,
                    }
                )
        return {
            "contact_frame_count": len(by_frame),
            "hard_collision_frame_count": len(hard_frames),
            "maximum_penetration_depth_m": maximum_depth,
            "hard_class_frame_counts": dict(hard_classes),
            "segments": segments,
            "by_frame": by_frame,
            "hard_frames": sorted(hard_frames),
        }

    def _distance(
        self,
        q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        pair: tuple[int, int],
    ) -> float:
        self.g1.assign(q, left_hand, right_hand)
        return float(
            mujoco.mj_geomDistance(
                self.g1.model,
                self.g1.data,
                int(pair[0]),
                int(pair[1]),
                0.10,
                None,
            )
        )

    def _anchor_repair(
        self,
        frame: int,
        q: np.ndarray,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.config["collision_repair"]
        current = q[frame].copy()
        initial_position, _ = self._pose(current)
        initial_error = np.asarray(
            [
                np.linalg.norm(
                    initial_position[side] - source_model[side][frame]
                )
                for side in SIDES
            ],
            dtype=np.float64,
        )
        position_bound = np.maximum(
            self.strict_tolerance, initial_error + 1e-6
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
                        np.linalg.norm(
                            position[side] - source_model[side][frame]
                        )
                        for side in SIDES
                    ],
                    dtype=np.float64,
                )

            def distances(value: np.ndarray) -> np.ndarray:
                return np.asarray(
                    [
                        self._distance(
                            value,
                            left_hand[frame],
                            right_hand[frame],
                            pair,
                        )
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
            new_records = self._records(
                candidate, left_hand[frame], right_hand[frame]
            )
            margin = float(np.min(constraint(candidate)))
            accepted = bool(
                margin >= -1e-6 and len(new_records) < old_count
            )
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
            "initial_position_bound_m": position_bound,
            "delta_norm_rad": float(np.linalg.norm(current - q[frame])),
            "reports": reports,
            "final_contact_count": len(
                self._records(current, left_hand[frame], right_hand[frame])
            ),
        }

    @staticmethod
    def _smooth_delta(
        length: int,
        anchor_index: int,
        anchor_value: np.ndarray,
        settings: Mapping[str, Any],
    ) -> np.ndarray:
        identity = np.eye(length, dtype=np.float64)
        first = np.diff(identity, axis=0)
        second = np.diff(identity, n=2, axis=0)
        matrix = float(settings["window_deviation_weight"]) * identity
        matrix += float(settings["window_velocity_weight"]) * (first.T @ first)
        matrix += float(settings["window_acceleration_weight"]) * (
            second.T @ second
        )
        right = np.zeros((length, len(anchor_value)), dtype=np.float64)
        matrix[anchor_index, anchor_index] += float(settings["anchor_weight"])
        right[anchor_index] += float(settings["anchor_weight"]) * anchor_value
        boundary = float(settings["boundary_zero_weight"])
        for index in sorted(
            {0, min(1, length - 1), max(0, length - 2), length - 1}
        ):
            matrix[index, index] += boundary
        try:
            return np.linalg.solve(matrix, right)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(matrix, right, rcond=None)[0]

    def _candidate_state(
        self,
        q: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> dict[str, Any]:
        error, _ = self._position_errors(q, source_model)
        collision = self._collision_metrics(q, left_hand, right_hand, fps)
        temporal = self._temporal_metrics(q, left_hand, right_hand, fps)
        return {
            "strict_source_residual_frames": int(
                np.count_nonzero(error > self.strict_tolerance)
            ),
            "physical_source_residual_frames": int(
                np.count_nonzero(error > self.physical_tolerance)
            ),
            "maximum_source_residual_m": float(np.max(error)),
            "mean_source_residual_m": float(np.mean(error)),
            "collision": collision,
            "temporal": temporal,
        }

    def _collision_candidate_safe(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> bool:
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
        return bool(
            collision_reduced
            and int(after["physical_source_residual_frames"])
            <= int(before["physical_source_residual_frames"])
            and float(after["maximum_source_residual_m"])
            <= max(
                float(before["maximum_source_residual_m"]),
                self.physical_tolerance,
            )
            + 1e-6
            and self._temporal_passes(after["temporal"])
            and int(after["temporal"]["branch_discontinuity_count"])
            <= int(before["temporal"]["branch_discontinuity_count"])
        )

    def _collision_repair(
        self,
        q_input: np.ndarray,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.config["collision_repair"]
        q = np.asarray(q_input, dtype=np.float64).copy()
        initial = self._candidate_state(
            q, source_model, left_hand, right_hand, fps
        )
        initial_segments = [
            row for row in initial["collision"]["segments"] if row["hard"]
        ]
        budget = int(settings["maximum_event_iterations"]) * max(
            1, len(initial_segments)
        )
        iterations: list[dict[str, Any]] = []
        for iteration in range(budget):
            before = self._candidate_state(
                q, source_model, left_hand, right_hand, fps
            )
            hard_frames = list(before["collision"]["hard_frames"])
            if not hard_frames:
                break
            anchor_frame = max(
                hard_frames,
                key=lambda frame: max(
                    (
                        float(row["penetration_depth_m"])
                        for row in before["collision"]["by_frame"].get(frame, [])
                    ),
                    default=0.0,
                ),
            )
            repaired, anchor_report = self._anchor_repair(
                anchor_frame,
                q,
                q_before,
                source_model,
                left_hand,
                right_hand,
            )
            delta_value = repaired - q[anchor_frame]
            attempts: list[dict[str, Any]] = []
            candidates: list[tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]] = []
            if float(np.linalg.norm(delta_value)) <= 1e-12:
                iterations.append(
                    {
                        "iteration": iteration,
                        "anchor_frame": anchor_frame,
                        "accepted": False,
                        "reason": "NO_FEASIBLE_ANCHOR_CORRECTION",
                        "anchor_report": anchor_report,
                    }
                )
                break
            for padding in settings["window_padding_candidates_frames"]:
                start = max(0, anchor_frame - int(padding))
                end = min(len(q) - 1, anchor_frame + int(padding))
                delta = self._smooth_delta(
                    end - start + 1,
                    anchor_frame - start,
                    delta_value,
                    settings,
                )
                for scale in settings["correction_scale_schedule"]:
                    candidate = q.copy()
                    candidate[start : end + 1] = np.clip(
                        candidate[start : end + 1] + float(scale) * delta,
                        self.g1.arm_limits[:, 0] + 1e-8,
                        self.g1.arm_limits[:, 1] - 1e-8,
                    )
                    candidate = candidate.astype(np.float32).astype(np.float64)
                    after = self._candidate_state(
                        candidate, source_model, left_hand, right_hand, fps
                    )
                    safe = self._collision_candidate_safe(before, after)
                    attempt = {
                        "padding_frames": int(padding),
                        "scale": float(scale),
                        "window_start": start,
                        "window_end": end,
                        "accepted_by_hard_gates": safe,
                        "after": self._compact_candidate_state(after),
                    }
                    attempts.append(attempt)
                    if safe:
                        score = (
                            int(after["collision"]["hard_collision_frame_count"]),
                            int(after["collision"]["contact_frame_count"]),
                            float(after["collision"]["maximum_penetration_depth_m"]),
                            int(after["physical_source_residual_frames"]),
                            float(after["mean_source_residual_m"]),
                            float(np.mean(np.linalg.norm(candidate - q_before, axis=1))),
                            int(padding),
                            -float(scale),
                        )
                        candidates.append((score, candidate, attempt))
            if not candidates:
                iterations.append(
                    {
                        "iteration": iteration,
                        "anchor_frame": anchor_frame,
                        "accepted": False,
                        "reason": "NO_WINDOW_CANDIDATE_PASSED_UNCHANGED_HARD_GATES",
                        "before": self._compact_candidate_state(before),
                        "anchor_report": anchor_report,
                        "attempts": attempts,
                    }
                )
                break
            candidates.sort(key=lambda row: row[0])
            _, q, selected = candidates[0]
            iterations.append(
                {
                    "iteration": iteration,
                    "anchor_frame": anchor_frame,
                    "accepted": True,
                    "before": self._compact_candidate_state(before),
                    "anchor_report": anchor_report,
                    "selected": selected,
                    "attempts": attempts,
                }
            )
        final = self._candidate_state(q, source_model, left_hand, right_hand, fps)
        return q, {
            "algorithm": settings["algorithm"],
            "initial": self._compact_candidate_state(initial),
            "final": self._compact_candidate_state(final),
            "initial_hard_segments": initial_segments,
            "iterations": iterations,
        }

    @staticmethod
    def _compact_candidate_state(state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "strict_source_residual_frames": int(
                state["strict_source_residual_frames"]
            ),
            "physical_source_residual_frames": int(
                state["physical_source_residual_frames"]
            ),
            "maximum_source_residual_m": float(state["maximum_source_residual_m"]),
            "mean_source_residual_m": float(state["mean_source_residual_m"]),
            "collision": {
                key: state["collision"][key]
                for key in (
                    "contact_frame_count",
                    "hard_collision_frame_count",
                    "maximum_penetration_depth_m",
                    "hard_class_frame_counts",
                    "segments",
                )
            },
            "temporal": dict(state["temporal"]),
        }

    def _nearest_realized_targets(
        self,
        source_world: Mapping[str, np.ndarray],
        achieved_world: Mapping[str, np.ndarray],
        q: np.ndarray,
        tracking_changed: np.ndarray,
        collision_changed: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        settings = self.config["nearest_target_projection"]
        tolerance = float(settings["strict_tolerance_interior_m"])
        count = len(q)
        realized = {
            side: np.asarray(source_world[side], dtype=np.float64).copy()
            for side in SIDES
        }
        magnitude = np.zeros((count, 2), dtype=np.float64)
        reasons = np.full((count, 2), "NONE", dtype="U48")
        source_model = self._source_model(source_world)
        for side_index, side in enumerate(SIDES):
            delta = achieved_world[side] - source_world[side]
            distance = np.linalg.norm(delta, axis=1)
            active = distance > tolerance
            direction = np.zeros_like(delta)
            direction[active] = delta[active] / distance[active, None]
            magnitude[active, side_index] = distance[active] - tolerance
            realized[side][active] = (
                source_world[side][active]
                + magnitude[active, side_index, None] * direction[active]
            )
            for frame in np.flatnonzero(active):
                reach = float(
                    np.linalg.norm(source_model[side][frame] - self.shoulders[side])
                )
                previous = max(0, int(frame) - 1)
                target_step = float(
                    np.linalg.norm(
                        source_world[side][frame] - source_world[side][previous]
                    )
                )
                if collision_changed[frame]:
                    reason = "COLLISION_CLEARANCE_ACTIVE"
                elif reach >= self.outer_tool_radius[side] - self.physical_tolerance:
                    reason = "WORKSPACE_BOUNDARY_ACTIVE"
                elif target_step > self.strict_tolerance:
                    reason = "TEMPORAL_LIMIT_ACTIVE"
                elif tracking_changed[frame]:
                    reason = "CONSTRAINED_TRACKING_ACTIVE"
                else:
                    reason = "NEAREST_FEASIBLE_IK_POINT"
                reasons[frame, side_index] = reason
        return realized, magnitude, reasons

    def solve_episode(self, episode: int, export: bool = True) -> EpisodeResult:
        values = load_trajectory(episode)
        q_before = values["g1_arm_qpos"].astype(np.float64)
        left_hand = values["left_dex3_qpos"].astype(np.float64)
        right_hand = values["right_dex3_qpos"].astype(np.float64)
        source_world, source_rotation = self._source_targets(values)
        source_model = self._source_model(source_world)
        timestamp = values["timestamp"].astype(np.float64)
        fps = float(1.0 / np.median(np.diff(timestamp))) if len(timestamp) > 1 else 30.0
        q_tracked, tracking_report = self._track(
            q_before,
            source_model,
            source_rotation,
            left_hand,
            right_hand,
            fps,
        )
        q_after, collision_report = self._collision_repair(
            q_tracked,
            q_before,
            source_model,
            left_hand,
            right_hand,
            fps,
        )
        q_after = q_after.astype(np.float32).astype(np.float64)
        geometry = self.g1.trajectory_geometry(
            q_after,
            left_hand,
            right_hand,
            self.collision_tolerance,
            self.transforms,
        )
        achieved_world = {
            side: geometry[f"{side}_static_tool_position_world"]
            for side in SIDES
        }
        achieved_rotation = {
            side: geometry[f"{side}_static_tool_rotation_model"]
            for side in SIDES
        }
        tracking_changed = np.linalg.norm(q_tracked - q_before, axis=1) > 1e-8
        collision_changed = np.linalg.norm(q_after - q_tracked, axis=1) > 1e-8
        realized_world, projection_magnitude, projection_reason = (
            self._nearest_realized_targets(
                source_world,
                achieved_world,
                q_after,
                tracking_changed,
                collision_changed,
            )
        )
        projection_orientation = np.zeros((len(q_after), 2), dtype=np.float64)
        final_source_error = np.maximum(
            *[
                np.linalg.norm(achieved_world[side] - source_world[side], axis=1)
                for side in SIDES
            ]
        )
        final_realized_error = np.maximum(
            *[
                np.linalg.norm(achieved_world[side] - realized_world[side], axis=1)
                for side in SIDES
            ]
        )
        orientation_error = {
            side: np.asarray(
                [
                    rotation_error_rad(
                        achieved_rotation[side][frame],
                        source_rotation[side][frame],
                    )
                    for frame in range(len(q_after))
                ],
                dtype=np.float64,
            )
            for side in SIDES
        }
        metadata = {
            "schema_version": "doll_handoff_generic_g1_feasibility_episode_v1",
            "episode_index": int(episode),
            "stable_episode_id": stable_episode_id(episode),
            "solver_class": type(self).__name__,
            "task_independent": True,
            "episode_specific_parameters": 0,
            "phase_specific_cartesian_parameters": 0,
            "source_targets_modified": False,
            "source_orientation_modified": False,
            "ownership_timing_modified": False,
            "hands_modified": False,
            "frozen_common_natural_arm_q_used_as_seed": True,
            "tracking": tracking_report,
            "collision_repair": collision_report,
            "changed_arm_frame_count": int(
                np.count_nonzero(np.linalg.norm(q_after - q_before, axis=1) > 1e-8)
            ),
            "projection_active_frame_count": int(
                np.count_nonzero(np.any(projection_magnitude > 0.0, axis=1))
            ),
            "projection_translation_m": {
                "mean": float(np.mean(projection_magnitude)),
                "median": float(np.median(projection_magnitude)),
                "max": float(np.max(projection_magnitude)),
            },
            "projection_orientation_rad": {
                "mean": 0.0,
                "max": 0.0,
            },
            "source_position_residual_m": {
                "mean": float(np.mean(final_source_error)),
                "median": float(np.median(final_source_error)),
                "max": float(np.max(final_source_error)),
            },
            "realized_position_residual_m": {
                "mean": float(np.mean(final_realized_error)),
                "median": float(np.median(final_realized_error)),
                "max": float(np.max(final_realized_error)),
            },
            "source_orientation_residual_rad": {
                side: {
                    "mean": float(np.mean(orientation_error[side])),
                    "max": float(np.max(orientation_error[side])),
                }
                for side in SIDES
            },
            "final_temporal": self._temporal_metrics(
                q_after, left_hand, right_hand, fps
            ),
            "config_sha256": sha256_file(self.config_path),
        }
        result = EpisodeResult(
            episode=int(episode),
            q_before=q_before,
            q_after=q_after,
            left_hand=left_hand,
            right_hand=right_hand,
            source_position_world=source_world,
            source_orientation_model=source_rotation,
            realized_position_world=realized_world,
            realized_orientation_model={
                side: source_rotation[side].copy() for side in SIDES
            },
            achieved_position_world=achieved_world,
            achieved_orientation_model=achieved_rotation,
            projection_translation_m=projection_magnitude,
            projection_orientation_rad=projection_orientation,
            projection_reason=projection_reason,
            metadata=metadata,
        )
        if export:
            self.export_episode(result, values, geometry)
        return result

    def load_exported_episode(self, episode: int) -> EpisodeResult:
        """Load a config-matching exported result for deterministic resumption."""
        frozen = load_trajectory(episode)
        trajectory_path = (
            self.output_root
            / "after/trajectories"
            / f"{stable_episode_id(episode)}.npz"
        )
        metric_path = (
            self.output_root
            / "after/metrics"
            / f"{stable_episode_id(episode)}.solver.json"
        )
        if not trajectory_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(trajectory_path)
        with np.load(trajectory_path, allow_pickle=False) as payload:
            values = {name: np.asarray(payload[name]) for name in payload.files}
        exported_config = str(np.asarray(values["feasibility_config_sha256"]).item())
        if exported_config != sha256_file(self.config_path):
            raise RuntimeError(
                f"cached ep{episode:03d} uses a different resolver config"
            )
        immutable_keys = (
            "target_left_wrist_position_model",
            "target_right_wrist_position_model",
            "target_left_wrist_rotation_model",
            "target_right_wrist_rotation_model",
            "target_left_interaction_frame_position_world",
            "target_right_interaction_frame_position_world",
            "ownership_state",
            "left_hand_phase",
            "right_hand_phase",
            "event_names",
            "event_frames",
        )
        if not all(
            np.array_equal(np.asarray(frozen[key]), np.asarray(values[key]))
            for key in immutable_keys
        ):
            raise RuntimeError(f"cached ep{episode:03d} changed an immutable array")
        q_after = values["g1_arm_qpos"].astype(np.float64)
        left_hand = frozen["left_dex3_qpos"].astype(np.float64)
        right_hand = frozen["right_dex3_qpos"].astype(np.float64)
        source_world, source_rotation = self._source_targets(frozen)
        geometry = self.g1.trajectory_geometry(
            q_after,
            left_hand,
            right_hand,
            self.collision_tolerance,
            self.transforms,
        )
        return EpisodeResult(
            episode=int(episode),
            q_before=frozen["g1_arm_qpos"].astype(np.float64),
            q_after=q_after,
            left_hand=left_hand,
            right_hand=right_hand,
            source_position_world=source_world,
            source_orientation_model=source_rotation,
            realized_position_world={
                side: values[
                    f"realized_feasible_{side}_interaction_frame_position_world"
                ].astype(np.float64)
                for side in SIDES
            },
            realized_orientation_model={
                side: values[
                    f"realized_feasible_{side}_interaction_frame_orientation_model"
                ].astype(np.float64)
                for side in SIDES
            },
            achieved_position_world={
                side: geometry[f"{side}_static_tool_position_world"]
                for side in SIDES
            },
            achieved_orientation_model={
                side: geometry[f"{side}_static_tool_rotation_model"]
                for side in SIDES
            },
            projection_translation_m=values[
                "feasibility_projection_translation_m"
            ].astype(np.float64),
            projection_orientation_rad=values[
                "feasibility_projection_orientation_rad"
            ].astype(np.float64),
            projection_reason=values["feasibility_projection_reason"].astype(str),
            metadata=load_json(metric_path),
        )

    def export_episode(
        self,
        result: EpisodeResult,
        frozen_values: Mapping[str, np.ndarray] | None = None,
        geometry: Mapping[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        if frozen_values is None:
            frozen_values = load_trajectory(result.episode)
        values = {name: np.asarray(value).copy() for name, value in frozen_values.items()}
        if geometry is None:
            geometry = self.g1.trajectory_geometry(
                result.q_after,
                result.left_hand,
                result.right_hand,
                self.collision_tolerance,
                self.transforms,
            )
        values["g1_arm_qpos"] = result.q_after.astype(np.float32)
        replay_names = values["replay_joint_names"].astype(str).tolist()
        replay = values["replay_named_joint_qpos"].astype(np.float32)
        for index, name in enumerate(self.g1.arm_joint_names.astype(str)):
            replay[:, replay_names.index(name)] = result.q_after[:, index].astype(
                np.float32
            )
        values["replay_named_joint_qpos"] = replay
        for side in SIDES:
            values[f"achieved_{side}_wrist_position_world"] = geometry[
                f"{side}_wrist_position_world"
            ].astype(np.float32)
            values[f"achieved_{side}_physical_grasp_frame_position_world"] = (
                geometry[f"{side}_grasp_position_world"].astype(np.float32)
            )
            task_origin = values["task_frame_origin_world_xyz_m"].astype(np.float64)
            values[f"achieved_{side}_physical_grasp_frame_position_task"] = (
                geometry[f"{side}_grasp_position_world"] - task_origin
            ).astype(np.float32)
            values[
                f"source_{side}_interaction_frame_position_world"
            ] = result.source_position_world[side].astype(np.float32)
            values[
                f"source_{side}_interaction_frame_orientation_model"
            ] = result.source_orientation_model[side].astype(np.float32)
            values[
                f"realized_feasible_{side}_interaction_frame_position_world"
            ] = result.realized_position_world[side].astype(np.float32)
            values[
                f"realized_feasible_{side}_interaction_frame_orientation_model"
            ] = result.realized_orientation_model[side].astype(np.float32)
            values[
                f"achieved_{side}_static_whole_hand_position_world"
            ] = result.achieved_position_world[side].astype(np.float32)
            values[
                f"achieved_{side}_static_whole_hand_orientation_model"
            ] = result.achieved_orientation_model[side].astype(np.float32)
        values["feasibility_projection_translation_m"] = (
            result.projection_translation_m.astype(np.float32)
        )
        values["feasibility_projection_orientation_rad"] = (
            result.projection_orientation_rad.astype(np.float32)
        )
        values["feasibility_projection_reason"] = result.projection_reason
        realized_error = np.maximum(
            *[
                np.linalg.norm(
                    result.achieved_position_world[side]
                    - result.realized_position_world[side],
                    axis=1,
                )
                for side in SIDES
            ]
        )
        values["ik_success_per_frame"] = realized_error <= self.strict_tolerance
        values["feasibility_resolver"] = np.asarray(type(self).__name__)
        values["feasibility_config_sha256"] = np.asarray(
            sha256_file(self.config_path)
        )
        values["source_targets_modified"] = np.asarray(False)
        values["ownership_timing_modified"] = np.asarray(False)
        trajectory_directory = self.output_root / "after/trajectories"
        metric_directory = self.output_root / "after/metrics"
        trajectory_directory.mkdir(parents=True, exist_ok=True)
        metric_directory.mkdir(parents=True, exist_ok=True)
        trajectory_path = trajectory_directory / f"{stable_episode_id(result.episode)}.npz"
        temporary = trajectory_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        temporary.replace(trajectory_path)
        metric_path = metric_directory / f"{stable_episode_id(result.episode)}.solver.json"
        write_json(metric_path, result.metadata)
        return trajectory_path, metric_path


def resolver_implementation_hash(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    paths = [
        Path(config_path).resolve(),
        Path(__file__).resolve(),
        Path(__file__).with_name("common.py").resolve(),
        Path(__file__).with_name("diagnostics.py").resolve(),
        Path(__file__).with_name("evaluate.py").resolve(),
        Path(__file__).with_name("render_review.py").resolve(),
        Path(__file__).with_name("finalize.py").resolve(),
        REPOSITORY / "tools/run_doll_handoff_g1_feasibility.py",
    ]
    files = {
        str(path.relative_to(REPOSITORY)): sha256_file(path) for path in paths
    }
    return {
        "implementation_sha256": stable_json_sha256(files),
        "files": files,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "EpisodeResult",
    "GenericG1FeasibilityResolver",
    "resolver_implementation_hash",
]
