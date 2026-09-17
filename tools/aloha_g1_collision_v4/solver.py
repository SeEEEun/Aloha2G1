"""Shared signed-distance local-window collision repair for Dataset A and B."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from aloha_g1_dataset_v1.core import G1Kinematics, branch_flags
from aloha_g1_feasibility_v3.solver import (
    FeasibilityResult,
    FrozenEpisode,
    SharedConstrainedFeasibilitySolver,
)
from aloha_g1_hand_v2.collision_eval import CollisionClassifier

from .audit import v4_category
from .common import METHOD_TO_DATASET, SIDES, V3_ROOT


PRIORITY = {
    "ARM_TORSO": 1,
    "CROSS_ARM": 2,
    "HAND_OPPOSITE_ARM": 3,
    "WRIST_OPPOSITE_HAND": 3,
    "HAND_HAND": 4,
    "THIRD_OPPOSITE_HAND": 5,
    "OTHER_PROHIBITED": 6,
}


@dataclass
class CollisionRepairResult:
    episode: FrozenEpisode
    q: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    orientation_slack: np.ndarray
    orientation_slack_requested: np.ndarray
    changed_mask: np.ndarray
    metadata: dict[str, Any]


def load_v3_episode(method: str, episode_id: int) -> FrozenEpisode:
    dataset_name = METHOD_TO_DATASET[method]
    directory = V3_ROOT / dataset_name / f"episode_{episode_id:06d}"
    with np.load(directory / "g1_arm_action.npz", allow_pickle=False) as payload:
        q = payload["action"].astype(np.float64)
        targets: dict[str, np.ndarray] = {}
        for side in SIDES:
            targets[f"{side}_wrist_position"] = payload[
                f"target_{side}_wrist_position"
            ].astype(np.float64)
            targets[f"{side}_wrist_rotation"] = payload[
                f"target_{side}_wrist_rotation"
            ].astype(np.float64)
            targets[f"{side}_tool_position"] = payload[
                f"target_{side}_task_tool_position"
            ].astype(np.float64)
        representation = str(payload["representation"])
        timestamps = payload["timestamps"].astype(np.float64)
        fps = float(payload["fps"])
    with np.load(directory / "g1_hand_action.npz", allow_pickle=False) as payload:
        left = payload["left_action"].astype(np.float64)
        right = payload["right_action"].astype(np.float64)
        left_phase = payload["left_phase"].astype(str)
        right_phase = payload["right_phase"].astype(str)
    return FrozenEpisode(
        method=method,
        dataset_name=dataset_name,
        episode_id=episode_id,
        q_v2=q,
        targets=targets,
        left_hand_v2=left,
        right_hand_v2=right,
        left_phase=left_phase,
        right_phase=right_phase,
        timestamps=timestamps,
        fps=fps,
        representation=representation,
    )


def _collision_groups(
    frames: list[int], padding: int, merge_gap: int, frame_count: int
) -> list[dict[str, Any]]:
    if not frames:
        return []
    # Merging padded overlaps prevents two independently generated corrections
    # from being added in the same local trajectory segment.
    threshold = max(int(merge_gap), 2 * int(padding))
    groups: list[list[int]] = [[int(frames[0])]]
    for frame in frames[1:]:
        if int(frame) - groups[-1][-1] <= threshold:
            groups[-1].append(int(frame))
        else:
            groups.append([int(frame)])
    return [
        {
            "core_frames": values,
            "window_start": max(0, values[0] - int(padding)),
            "window_end": min(frame_count - 1, values[-1] + int(padding)),
        }
        for values in groups
    ]


def _smooth_delta(
    length: int,
    anchor_indices: list[int],
    anchor_values: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Quadratic minimum-velocity/acceleration fit with fixed zero boundaries."""
    identity = np.eye(length, dtype=np.float64)
    first = np.diff(identity, axis=0)
    second = np.diff(identity, n=2, axis=0)
    matrix = float(config["window_deviation_weight"]) * identity
    matrix += float(config["window_velocity_weight"]) * (first.T @ first)
    matrix += float(config["window_acceleration_weight"]) * (second.T @ second)
    right = np.zeros((length, 14), dtype=np.float64)
    anchor_weight = float(config["core_anchor_weight"])
    for index, value in zip(anchor_indices, anchor_values):
        matrix[index, index] += anchor_weight
        right[index] += anchor_weight * value
    boundary_weight = float(config["boundary_zero_weight"])
    boundary_indices = sorted({0, min(1, length - 1), max(0, length - 2), length - 1})
    for index in boundary_indices:
        matrix[index, index] += boundary_weight
    try:
        return np.linalg.solve(matrix, right)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, right, rcond=None)[0]


class SharedCollisionWindowSolver:
    """Method-independent repair using v3 constraints and fixed method targets."""

    def __init__(
        self,
        runtime_config: Mapping[str, Any],
        frozen_v3: Mapping[str, Any],
        candidate: Mapping[str, Any],
        g1: G1Kinematics,
        collision_runtime: Any,
        classifier: CollisionClassifier,
    ):
        self.runtime_config = runtime_config
        self.v3 = frozen_v3
        self.config = copy.deepcopy(dict(candidate))
        solver_config = copy.deepcopy(dict(frozen_v3["solver_parameters"]))
        solver_config.update(
            {
                "collision_clearance_m": float(candidate["d_safe_m"]),
                "collision_penalty": float(candidate["collision_barrier_strength"]),
                "v2_deviation_weight": float(candidate["deviation_from_v3_weight"]),
                "nominal_posture_weight": float(candidate["nominal_posture_weight"]),
                "previous_q_weight": float(candidate["local_previous_q_weight"]),
                "next_q_weight": float(candidate["local_next_q_weight"]),
                "acceleration_reference_weight": float(
                    candidate["local_acceleration_weight"]
                ),
                "max_stage1_iterations": int(candidate["max_frame_iterations"]),
                "max_stage2_iterations": int(candidate["max_frame_iterations"]),
                "max_collision_outer_passes": int(candidate["max_pair_discovery_passes"]),
            }
        )
        self.solver_config = solver_config
        self.g1 = g1
        self.runtime = collision_runtime
        self.classifier = classifier
        self.helper = SharedConstrainedFeasibilitySolver(
            runtime_config,
            solver_config,
            frozen_v3["hand_temporal_projection"],
            g1,
            collision_runtime,
            classifier,
        )
        self.limits = np.asarray(g1.limits, dtype=np.float64)
        validation = runtime_config["validation"]
        self.max_step = float(validation["maximum_joint_step_rad"])
        self.max_acceleration = float(validation["maximum_acceleration_rad_s2"])
        self.branch_absolute = float(validation["branch_absolute_step_norm_rad"])
        self.branch_multiplier = float(validation["branch_local_multiplier"])

    def _records(self, q: np.ndarray, left: np.ndarray, right: np.ndarray) -> list[Any]:
        return self.helper._records(q, left, right)

    def _targets(self, episode: FrozenEpisode, frame: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        return (
            {
                side: episode.targets[f"{side}_wrist_position"][frame]
                for side in SIDES
            },
            {
                side: episode.targets[f"{side}_wrist_rotation"][frame]
                for side in SIDES
            },
        )

    def _task_feasible(self, episode: FrozenEpisode, frame: int, q: np.ndarray) -> bool:
        position, rotation = self._targets(episode, frame)
        return self.helper._task_feasible(q, position, rotation)

    def _distance(self, q: np.ndarray, left: np.ndarray, right: np.ndarray, pair: tuple[int, int]) -> float:
        self.runtime.assign(q, left, right)
        return float(
            mujoco.mj_geomDistance(
                self.runtime.model,
                self.runtime.data,
                int(pair[0]),
                int(pair[1]),
                0.10,
                None,
            )
        )

    def _ordered_pairs(self, records: list[Any], existing: list[tuple[int, int]]) -> list[tuple[int, int]]:
        values: dict[tuple[int, int], int] = {pair: 99 for pair in existing}
        for record in records:
            pair = tuple(int(value) for value in record.geom_ids)
            values[pair] = min(values.get(pair, 99), PRIORITY[v4_category(record.bodies)])
        return [pair for pair, _ in sorted(values.items(), key=lambda row: (row[1], row[0]))]

    def _strict_frame(
        self,
        episode: FrozenEpisode,
        frame: int,
        q: np.ndarray,
        pairs: list[tuple[int, int]],
        previous: np.ndarray | None,
        previous2: np.ndarray | None,
        following: np.ndarray | None,
        following2: np.ndarray | None,
    ) -> bool:
        if not self._task_feasible(episode, frame, q):
            return False
        interior = float(self.solver_config["hard_joint_interior_rad"])
        if np.any(q < self.limits[:, 0] + interior - 1e-9) or np.any(
            q > self.limits[:, 1] - interior + 1e-9
        ):
            return False
        clearance = float(self.config["d_safe_m"])
        left = episode.left_hand_v2[frame]
        right = episode.right_hand_v2[frame]
        if any(self._distance(q, left, right, pair) < clearance - 1e-7 for pair in pairs):
            return False
        if self._records(q, left, right):
            return False
        step_interior = float(self.solver_config["temporal_step_constraint_interior_rad"])
        acceleration_interior = float(
            self.solver_config["temporal_acceleration_constraint_interior_rad_s2"]
        ) / episode.fps**2
        step_limit = self.max_step - step_interior
        acceleration_limit = self.max_acceleration / episode.fps**2 - acceleration_interior
        if previous is not None and np.max(np.abs(q - previous)) > step_limit + 1e-7:
            return False
        if following is not None and np.max(np.abs(following - q)) > step_limit + 1e-7:
            return False
        if previous is not None and previous2 is not None and np.max(
            np.abs(q - 2.0 * previous + previous2)
        ) > acceleration_limit + 1e-7:
            return False
        if previous is not None and following is not None and np.max(
            np.abs(following - 2.0 * q + previous)
        ) > acceleration_limit + 1e-7:
            return False
        if following is not None and following2 is not None and np.max(
            np.abs(following2 - 2.0 * following + q)
        ) > acceleration_limit + 1e-7:
            return False
        return True

    def _repair_frame(
        self,
        episode: FrozenEpisode,
        frame: int,
        initial: np.ndarray,
        q_reference: np.ndarray,
        pairs: list[tuple[int, int]],
        previous: np.ndarray | None = None,
        previous2: np.ndarray | None = None,
        following: np.ndarray | None = None,
        following2: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        position, rotation = self._targets(episode, frame)
        current = np.asarray(initial, dtype=np.float64).copy()
        tracked = list(pairs)
        attempts: list[dict[str, Any]] = []
        for _ in range(int(self.config["max_pair_discovery_passes"])):
            records = self._records(
                current,
                episode.left_hand_v2[frame],
                episode.right_hand_v2[frame],
            )
            tracked = self._ordered_pairs(records, tracked)
            if self._strict_frame(
                episode,
                frame,
                current,
                tracked,
                previous,
                previous2,
                following,
                following2,
            ):
                break
            candidate, optimizer = self.helper._project_frame(
                current,
                q_reference,
                position,
                rotation,
                episode.left_hand_v2[frame],
                episode.right_hand_v2[frame],
                tuple(tracked),
                previous,
                previous2,
                following,
                following2,
                int(self.config["max_frame_iterations"]),
            )
            old_records = self._records(
                current,
                episode.left_hand_v2[frame],
                episode.right_hand_v2[frame],
            )
            new_records = self._records(
                candidate,
                episode.left_hand_v2[frame],
                episode.right_hand_v2[frame],
            )
            strict = self._strict_frame(
                episode,
                frame,
                candidate,
                tracked,
                previous,
                previous2,
                following,
                following2,
            )
            attempts.append(
                {
                    **optimizer,
                    "old_collision_pair_count": len(old_records),
                    "new_collision_pair_count": len(new_records),
                    "strict_frame_pass": strict,
                }
            )
            has_temporal_context = any(
                value is not None
                for value in (previous, previous2, following, following2)
            )
            # A collision reduction is not a valid local-window repair if it
            # violates an inherited temporal hard constraint.  Anchor solves
            # have no neighbours yet; refinement solves must pass every hard
            # frame constraint before entering the trajectory.
            if strict or (
                not has_temporal_context
                and len(new_records) < len(old_records)
            ):
                current = candidate
            else:
                break
        return current, {
            "tracked_pairs": [list(pair) for pair in tracked],
            "attempts": attempts,
            "final_collision_pair_count": len(
                self._records(
                    current,
                    episode.left_hand_v2[frame],
                    episode.right_hand_v2[frame],
                )
            ),
        }

    def _trajectory_diagnostics(self, episode: FrozenEpisode, q: np.ndarray) -> dict[str, Any]:
        task_fail = 0
        collision_frames = 0
        penetration: list[float] = []
        for frame in range(len(q)):
            task_fail += int(not self._task_feasible(episode, frame, q[frame]))
            records = self._records(
                q[frame], episode.left_hand_v2[frame], episode.right_hand_v2[frame]
            )
            collision_frames += int(bool(records))
            penetration.extend(max(0.0, -float(row.distance_m)) for row in records)
        full = np.column_stack((q, episode.left_hand_v2, episode.right_hand_v2))
        step = np.abs(np.diff(full, axis=0))
        acceleration = np.abs(np.diff(full, n=2, axis=0)) * episode.fps**2
        branches = branch_flags(q, self.branch_absolute, self.branch_multiplier)
        violations = (q < self.limits[:, 0] - 1e-9) | (
            q > self.limits[:, 1] + 1e-9
        )
        return {
            "task_failed_frames": task_fail,
            "collision_frames": collision_frames,
            "maximum_penetration_depth_m": max(penetration, default=0.0),
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "maximum_step_rad": float(np.max(step, initial=0.0)),
            "maximum_velocity_rad_s": float(np.max(step, initial=0.0) * episode.fps),
            "maximum_acceleration_rad_s2": float(np.max(acceleration, initial=0.0)),
            "branch_discontinuity_count": int(np.count_nonzero(branches)),
        }

    def _event_is_safe(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> bool:
        return bool(
            after["collision_frames"] < before["collision_frames"]
            and after["task_failed_frames"] <= before["task_failed_frames"]
            and after["joint_limit_violation_count"] == 0
            and after["maximum_step_rad"] <= self.max_step + 1e-9
            and after["maximum_velocity_rad_s"]
            <= float(self.runtime_config["validation"]["maximum_velocity_rad_s"])
            + 1e-9
            and after["maximum_acceleration_rad_s2"]
            <= self.max_acceleration + 1e-9
            and after["branch_discontinuity_count"]
            <= before["branch_discontinuity_count"]
        )

    def _penetration_at(self, episode: FrozenEpisode, q: np.ndarray, frame: int) -> float:
        return max(
            (
                max(0.0, -float(row.distance_m))
                for row in self._records(
                    q[frame],
                    episode.left_hand_v2[frame],
                    episode.right_hand_v2[frame],
                )
            ),
            default=0.0,
        )

    def _anchor_candidates(
        self,
        episode: FrozenEpisode,
        q: np.ndarray,
        frames: list[int],
    ) -> list[int]:
        """Select anchors by one deterministic, episode-agnostic rule."""
        if not frames:
            return []
        ranked = sorted(
            frames,
            key=lambda frame: (
                -self._penetration_at(episode, q, frame),
                frame,
            ),
        )
        ordered = ranked + [frames[0], frames[-1]]
        output: list[int] = []
        for frame in ordered:
            if frame not in output:
                output.append(frame)
            if len(output) >= int(self.config["anchor_candidates_per_iteration"]):
                break
        return output

    def _keyframe_solution(
        self,
        episode: FrozenEpisode,
        q: np.ndarray,
        frame: int,
    ) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, Any]]:
        records = self._records(
            q[frame], episode.left_hand_v2[frame], episode.right_hand_v2[frame]
        )
        pairs = self._ordered_pairs(records, [])
        repaired, anchor_report = self._repair_frame(
            episode,
            frame,
            q[frame],
            episode.q_v2[frame],
            pairs,
        )
        return repaired, pairs, anchor_report

    def _keyframe_window_candidate(
        self,
        episode: FrozenEpisode,
        q: np.ndarray,
        frame: int,
        repaired: np.ndarray,
        pairs: list[tuple[int, int]],
        anchor_report: Mapping[str, Any],
        scale: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        padding = int(self.config["window_padding_frames"])
        start = max(0, frame - padding)
        end = min(len(q) - 1, frame + padding)
        delta = _smooth_delta(
            end - start + 1,
            [frame - start],
            np.asarray([repaired - q[frame]], dtype=np.float64),
            self.config,
        )
        candidate = q.copy()
        interior = float(self.solver_config["hard_joint_interior_rad"])
        candidate[start : end + 1] = np.clip(
            q[start : end + 1] + float(scale) * delta,
            self.limits[:, 0] + interior,
            self.limits[:, 1] - interior,
        )
        # The exported representation is float32; candidate selection uses
        # the exact serialized values rather than optimistic float64 values.
        candidate = candidate.astype(np.float32).astype(np.float64)
        return candidate, {
            "anchor_frame": frame,
            "window_start": start,
            "window_end": end,
            "scale": float(scale),
            "anchor_delta_norm_rad": float(np.linalg.norm(repaired - q[frame])),
            "tracked_pairs": [list(pair) for pair in pairs],
            "anchor_solver": anchor_report,
        }

    def solve(self, episode: FrozenEpisode) -> CollisionRepairResult:
        q_v3 = episode.q_v2.astype(np.float64)
        q = q_v3.copy()
        initial_frames = [
            frame
            for frame in range(len(q))
            if self._records(
                q[frame], episode.left_hand_v2[frame], episode.right_hand_v2[frame]
            )
        ]
        padding = int(self.config["window_padding_frames"])
        groups = _collision_groups(
            initial_frames,
            padding,
            int(self.config["event_merge_gap_frames"]),
            len(q),
        )
        reports: list[dict[str, Any]] = []
        for group_index, group in enumerate(groups):
            start = int(group["window_start"])
            end = int(group["window_end"])
            initial_core = [int(value) for value in group["core_frames"]]
            before_group = self._trajectory_diagnostics(episode, q)
            iteration_reports: list[dict[str, Any]] = []
            accepted_any = False
            for iteration in range(int(self.config["max_event_iterations"])):
                current_frames = [
                    frame
                    for frame in range(start, end + 1)
                    if self._records(
                        q[frame],
                        episode.left_hand_v2[frame],
                        episode.right_hand_v2[frame],
                    )
                ]
                if not current_frames:
                    break
                before = self._trajectory_diagnostics(episode, q)
                attempts: list[dict[str, Any]] = []
                feasible_attempts: list[tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]] = []
                for frame in self._anchor_candidates(episode, q, current_frames):
                    repaired, pairs, anchor_report = self._keyframe_solution(
                        episode, q, frame
                    )
                    for scale in self.config["correction_scale_schedule"]:
                        candidate_q, attempt = self._keyframe_window_candidate(
                            episode,
                            q,
                            frame,
                            repaired,
                            pairs,
                            anchor_report,
                            float(scale),
                        )
                        after = self._trajectory_diagnostics(episode, candidate_q)
                        safe = self._event_is_safe(before, after)
                        attempt.update({"after": after, "accepted_by_hard_gates": safe})
                        attempts.append(attempt)
                        if safe:
                            score = (
                                after["collision_frames"],
                                after["maximum_penetration_depth_m"],
                                after["task_failed_frames"],
                                float(np.mean(np.linalg.norm(candidate_q - q_v3, axis=1))),
                                frame,
                                -float(scale),
                            )
                            feasible_attempts.append((score, candidate_q, attempt))
                if not feasible_attempts:
                    iteration_reports.append(
                        {
                            "iteration": iteration,
                            "before": before,
                            "attempts": attempts,
                            "accepted": False,
                        }
                    )
                    break
                feasible_attempts.sort(key=lambda row: row[0])
                _, q, selected_attempt = feasible_attempts[0]
                accepted_any = True
                iteration_reports.append(
                    {
                        "iteration": iteration,
                        "before": before,
                        "attempts": attempts,
                        "accepted": True,
                        "selected": selected_attempt,
                    }
                )
            after_group = self._trajectory_diagnostics(episode, q)
            reports.append(
                {
                    "group_index": group_index,
                    **group,
                    "initial_core_frames": initial_core,
                    "before": before_group,
                    "after": after_group,
                    "accepted": accepted_any,
                    "classification": (
                        "REPAIRED_WITHIN_TASK_EQUIVALENCE"
                        if accepted_any
                        else "COLLISION_UNAVOIDABLE_WITHIN_TASK_EQUIVALENCE"
                    ),
                    "iteration_reports": iteration_reports,
                }
            )

        q = q.astype(np.float32).astype(np.float64)
        orientation_error = np.empty((len(q), 2), dtype=np.float64)
        for frame in range(len(q)):
            position, rotation = self._targets(episode, frame)
            _, orientation_error[frame] = self.helper._task_values(q[frame], position, rotation)
        tolerance = float(self.solver_config["orientation_tolerance_rad"])
        requested = np.maximum(0.0, orientation_error - tolerance)
        used = np.minimum(
            requested, float(self.solver_config["orientation_slack_bound_rad"])
        )
        changed_mask = np.linalg.norm(q - q_v3, axis=1) > 1e-8
        return CollisionRepairResult(
            episode=episode,
            q=q,
            left_hand=episode.left_hand_v2.copy(),
            right_hand=episode.right_hand_v2.copy(),
            orientation_slack=used,
            orientation_slack_requested=requested,
            changed_mask=changed_mask,
            metadata={
                "solver_class": type(self).__name__,
                "collision_config": self.config,
                "inherited_feasibility_v3_config": self.solver_config,
                "initial_collision_frames": len(initial_frames),
                "final_diagnostics": self._trajectory_diagnostics(episode, q),
                "event_reports": reports,
                "collision_priority": PRIORITY,
                "finite_difference_gradient": (
                    "SciPy SLSQP deterministic finite differences over MuJoCo "
                    "mj_geomDistance hard constraints"
                ),
                "method_specific_collision_logic": False,
            },
        )

    def as_v3_result(self, result: CollisionRepairResult) -> FeasibilityResult:
        return FeasibilityResult(
            episode=result.episode,
            q=result.q,
            left_hand=result.left_hand,
            right_hand=result.right_hand,
            orientation_slack=result.orientation_slack,
            orientation_slack_requested=result.orientation_slack_requested,
            metadata={
                "solver_class": type(self).__name__,
                "solver_parameters": self.solver_config,
                "hand_temporal_projection": {
                    "algorithm": "frozen_feasibility_v3_hand_action",
                    "conditionally_applied": False,
                    "trigger_is_identical_for_a_b": True,
                    "endpoint_state_definitions_changed": False,
                    "semantic_labels_changed": False,
                    "v3_hand_action_byte_identical": True,
                },
                "repair_records": result.metadata["event_reports"],
            },
        )


__all__ = [
    "PRIORITY",
    "CollisionRepairResult",
    "load_v3_episode",
    "SharedCollisionWindowSolver",
]
