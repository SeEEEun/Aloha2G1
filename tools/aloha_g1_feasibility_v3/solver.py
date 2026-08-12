"""One deterministic constrained feasibility solver shared by Dataset A and B."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.optimize import minimize

from aloha_g1_dataset_v1.core import G1Kinematics, rotation_error_vector
from aloha_g1_hand_v2.collision_eval import CollisionClassifier

from .common import METHOD_TO_DATASET, V2_ARM_ROOT, V2_INTEGRATED_ROOT


SIDES = ("left", "right")


@dataclass
class FrozenEpisode:
    method: str
    dataset_name: str
    episode_id: int
    q_v2: np.ndarray
    targets: dict[str, np.ndarray]
    left_hand_v2: np.ndarray
    right_hand_v2: np.ndarray
    left_phase: np.ndarray
    right_phase: np.ndarray
    timestamps: np.ndarray
    fps: float
    representation: str


@dataclass
class FeasibilityResult:
    episode: FrozenEpisode
    q: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    orientation_slack: np.ndarray
    orientation_slack_requested: np.ndarray
    metadata: dict[str, Any]


def load_frozen_episode(method: str, episode_id: int) -> FrozenEpisode:
    dataset_name = METHOD_TO_DATASET[method]
    with np.load(
        V2_ARM_ROOT
        / method
        / f"episode_{episode_id:06d}"
        / "g1_arm_action.npz",
        allow_pickle=False,
    ) as payload:
        q = payload["action"].astype(np.float64)
        targets = {
            f"{side}_wrist_position": payload[
                f"target_{side}_wrist_position"
            ].astype(np.float64)
            for side in SIDES
        }
        targets.update(
            {
                f"{side}_wrist_rotation": payload[
                    f"target_{side}_wrist_rotation"
                ].astype(np.float64)
                for side in SIDES
            }
        )
        targets.update(
            {
                f"{side}_tool_position": payload[
                    f"target_{side}_task_tool_position"
                ].astype(np.float64)
                for side in SIDES
            }
        )
        representation = str(payload["representation"])
        timestamps = payload["timestamps"].astype(np.float64)
        fps = float(payload["fps"])
    with np.load(
        V2_INTEGRATED_ROOT
        / dataset_name
        / f"episode_{episode_id:06d}"
        / "g1_hand_action.npz",
        allow_pickle=False,
    ) as payload:
        left = payload["left_action"].astype(np.float64)
        right = payload["right_action"].astype(np.float64)
        left_phase = payload["left_phase"].astype(str)
        right_phase = payload["right_phase"].astype(str)
    if not (
        len(q)
        == len(left)
        == len(right)
        == len(left_phase)
        == len(right_phase)
        == len(timestamps)
    ):
        raise RuntimeError(f"episode {episode_id}: frozen v2 length mismatch")
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


def _temporal_extrema(value: np.ndarray, fps: float) -> dict[str, float]:
    step = np.abs(np.diff(value, axis=0))
    acceleration = np.abs(np.diff(value, n=2, axis=0)) * fps**2
    maximum_step = float(np.max(step, initial=0.0))
    return {
        "maximum_step_rad": maximum_step,
        "maximum_velocity_rad_s": maximum_step * fps,
        "maximum_acceleration_rad_s2": float(
            np.max(acceleration, initial=0.0)
        ),
    }


def _quintic_impulse(frame_count: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, int(frame_count) + 1)
    step = 10.0 * edges**3 - 15.0 * edges**4 + 6.0 * edges**5
    impulse = np.maximum(np.diff(step), 0.0)
    impulse /= np.sum(impulse)
    return impulse


def _causal_fir(value: np.ndarray, frame_count: int) -> np.ndarray:
    impulse = _quintic_impulse(frame_count)
    padded = np.vstack(
        (np.repeat(value[:1], frame_count - 1, axis=0), value)
    )
    output = np.empty_like(value)
    offset = frame_count - 1
    for joint in range(value.shape[1]):
        filtered = np.convolve(padded[:, joint], impulse, mode="full")
        output[:, joint] = filtered[offset : offset + len(value)]
    return output


def project_hand_temporally(
    left: np.ndarray,
    right: np.ndarray,
    fps: float,
    validation: Mapping[str, Any],
    hand_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    combined = np.column_stack((left, right))
    before = _temporal_extrema(combined, fps)
    passes = bool(
        before["maximum_step_rad"]
        <= float(validation["maximum_joint_step_rad"])
        and before["maximum_velocity_rad_s"]
        <= float(validation["maximum_velocity_rad_s"])
        and before["maximum_acceleration_rad_s2"]
        <= float(validation["maximum_acceleration_rad_s2"])
    )
    if passes:
        projected = combined.copy()
    else:
        projected = _causal_fir(
            combined, int(hand_config["transition_frames"])
        )
    after = _temporal_extrema(projected, fps)
    return (
        projected[:, :7],
        projected[:, 7:],
        {
            "algorithm": hand_config["algorithm"],
            "conditionally_applied": not passes,
            "trigger_is_identical_for_a_b": True,
            "endpoint_state_definitions_changed": False,
            "semantic_labels_changed": False,
            "before": before,
            "after": after,
        },
    )


class SharedConstrainedFeasibilitySolver:
    """Two-stage hard-box task projection followed by collision repair.

    The class has no representation branch. Dataset A and B supply different
    frozen targets and fixed hand states to this identical optimization path.
    """

    def __init__(
        self,
        runtime_config: Mapping[str, Any],
        solver_config: Mapping[str, Any],
        hand_temporal_config: Mapping[str, Any],
        g1: G1Kinematics,
        collision_runtime: Any,
        classifier: CollisionClassifier,
    ):
        self.runtime_config = runtime_config
        self.config = dict(solver_config)
        self.hand_temporal_config = dict(hand_temporal_config)
        self.g1 = g1
        self.runtime = collision_runtime
        self.classifier = classifier
        self.limits = np.asarray(g1.limits, dtype=np.float64)
        self.position_tolerance = float(self.config["position_tolerance_m"])
        self.orientation_tolerance = float(
            self.config["orientation_tolerance_rad"]
        )
        self.slack_bound = float(self.config["orientation_slack_bound_rad"])
        validation = runtime_config["validation"]
        self.maximum_step = float(validation["maximum_joint_step_rad"]) - float(
            self.config["temporal_step_constraint_interior_rad"]
        )
        fps = float(runtime_config["source_dataset"]["fps"])
        self.maximum_second_difference = (
            float(validation["maximum_acceleration_rad_s2"])
            - float(
                self.config[
                    "temporal_acceleration_constraint_interior_rad_s2"
                ]
            )
        ) / fps**2

    def _task_values(
        self,
        q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        state = self.g1.wrist_state(q)
        position = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(target_position[side])
                    - state[f"{side}_position"]
                )
                for side in SIDES
            ],
            dtype=np.float64,
        )
        orientation = np.asarray(
            [
                np.linalg.norm(
                    rotation_error_vector(
                        state[f"{side}_rotation"],
                        np.asarray(target_rotation[side]),
                    )
                )
                for side in SIDES
            ],
            dtype=np.float64,
        )
        return position, orientation

    def _records(
        self, q: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> list[Any]:
        self.runtime.assign(q, left, right)
        return [
            record
            for record in self.classifier.records()
            if record.v1_gate_relevant
        ]

    def _distances(
        self,
        q: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        geom_pairs: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        self.runtime.assign(q, left, right)
        return np.asarray(
            [
                mujoco.mj_geomDistance(
                    self.runtime.model,
                    self.runtime.data,
                    pair[0],
                    pair[1],
                    0.10,
                    None,
                )
                for pair in geom_pairs
            ],
            dtype=np.float64,
        )

    def _hard_constraint_values(
        self,
        q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        left: np.ndarray,
        right: np.ndarray,
        geom_pairs: tuple[tuple[int, int], ...],
        previous: np.ndarray | None,
        previous2: np.ndarray | None,
        following: np.ndarray | None,
        following2: np.ndarray | None,
    ) -> np.ndarray:
        position, orientation = self._task_values(
            q, target_position, target_rotation
        )
        values: list[np.ndarray] = [
            self.position_tolerance - position,
            self.orientation_tolerance + self.slack_bound - orientation,
        ]
        if geom_pairs:
            values.append(
                self._distances(q, left, right, geom_pairs)
                - float(self.config["collision_clearance_m"])
            )
        if previous is not None:
            values.append(self.maximum_step - np.abs(q - previous))
        if following is not None:
            values.append(self.maximum_step - np.abs(following - q))
        if previous is not None and previous2 is not None:
            values.append(
                self.maximum_second_difference
                - np.abs(q - 2.0 * previous + previous2)
            )
        if previous is not None and following is not None:
            values.append(
                self.maximum_second_difference
                - np.abs(following - 2.0 * q + previous)
            )
        if following is not None and following2 is not None:
            values.append(
                self.maximum_second_difference
                - np.abs(following2 - 2.0 * following + q)
            )
        return np.concatenate(values)

    def _objective(
        self,
        q: np.ndarray,
        q_v2: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        left: np.ndarray,
        right: np.ndarray,
        geom_pairs: tuple[tuple[int, int], ...],
        previous: np.ndarray | None,
        previous2: np.ndarray | None,
        following: np.ndarray | None,
    ) -> float:
        position, orientation = self._task_values(
            q, target_position, target_rotation
        )
        slack = np.maximum(0.0, orientation - self.orientation_tolerance)
        value = float(self.config["task_position_weight"]) * float(
            np.dot(position, position)
        )
        value += float(self.config["task_orientation_weight"]) * float(
            np.dot(orientation, orientation)
        )
        value += float(self.config["orientation_slack_penalty"]) * float(
            np.dot(slack, slack)
        )
        value += float(self.config["v2_deviation_weight"]) * float(
            np.dot(q - q_v2, q - q_v2)
        )
        value += float(self.config["nominal_posture_weight"]) * float(
            np.dot(q - self.g1.nominal_q, q - self.g1.nominal_q)
        )
        if previous is not None:
            value += float(self.config["previous_q_weight"]) * float(
                np.dot(q - previous, q - previous)
            )
        if following is not None:
            value += float(self.config["next_q_weight"]) * float(
                np.dot(q - following, q - following)
            )
        if previous is not None and previous2 is not None:
            acceleration = q - 2.0 * previous + previous2
            value += float(
                self.config["acceleration_reference_weight"]
            ) * float(np.dot(acceleration, acceleration))
        margin = np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q)
        margin_hinge = np.maximum(
            0.0, float(self.config["joint_soft_margin_rad"]) - margin
        )
        value += float(self.config["joint_margin_weight"]) * float(
            np.dot(margin_hinge, margin_hinge)
        )
        if geom_pairs:
            distances = self._distances(q, left, right, geom_pairs)
            barrier = np.maximum(
                0.0,
                float(self.config["collision_clearance_m"]) - distances,
            )
            value += float(self.config["collision_penalty"]) * float(
                np.dot(barrier, barrier)
            )
        return value

    def _project_frame(
        self,
        initial: np.ndarray,
        q_v2: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        left: np.ndarray,
        right: np.ndarray,
        geom_pairs: tuple[tuple[int, int], ...],
        previous: np.ndarray | None,
        previous2: np.ndarray | None,
        following: np.ndarray | None,
        following2: np.ndarray | None,
        maximum_iterations: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        interior = float(self.config["hard_joint_interior_rad"])
        lower = self.limits[:, 0] + interior
        upper = self.limits[:, 1] - interior
        initial = np.minimum(np.maximum(np.asarray(initial), lower), upper)
        result = minimize(
            lambda value: self._objective(
                value,
                q_v2,
                target_position,
                target_rotation,
                left,
                right,
                geom_pairs,
                previous,
                previous2,
                following,
            ),
            initial,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints={
                "type": "ineq",
                "fun": lambda value: self._hard_constraint_values(
                    value,
                    target_position,
                    target_rotation,
                    left,
                    right,
                    geom_pairs,
                    previous,
                    previous2,
                    following,
                    following2,
                ),
            },
            options={
                "maxiter": int(maximum_iterations),
                "ftol": float(self.config["slsqp_ftol"]),
                "disp": False,
            },
        )
        q = np.asarray(result.x, dtype=np.float64)
        constraints = self._hard_constraint_values(
            q,
            target_position,
            target_rotation,
            left,
            right,
            geom_pairs,
            previous,
            previous2,
            following,
            following2,
        )
        return q, {
            "optimizer_success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "minimum_constraint_margin": float(np.min(constraints)),
            "constraints_satisfied": bool(np.min(constraints) >= -1e-7),
            "tracked_collision_geom_pairs": len(geom_pairs),
        }

    def _task_feasible(
        self,
        q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
    ) -> bool:
        position, orientation = self._task_values(
            q, target_position, target_rotation
        )
        return bool(
            np.all(position <= self.position_tolerance)
            and np.all(
                orientation
                <= self.orientation_tolerance + self.slack_bound
            )
        )

    def solve(self, episode: FrozenEpisode) -> FeasibilityResult:
        validation = self.runtime_config["validation"]
        left, right, hand_meta = project_hand_temporally(
            episode.left_hand_v2,
            episode.right_hand_v2,
            episode.fps,
            validation,
            self.hand_temporal_config,
        )
        q = episode.q_v2.copy()
        frame_metadata: list[dict[str, Any]] = [
            {
                "changed": False,
                "stage1_calls": 0,
                "stage2_calls": 0,
                "stage1_success": None,
                "stage2_success": None,
            }
            for _ in range(len(q))
        ]
        initial_collision = np.zeros(len(q), dtype=bool)
        initial_task_feasible = np.zeros(len(q), dtype=bool)
        for frame in range(len(q)):
            target_position = {
                side: episode.targets[f"{side}_wrist_position"][frame]
                for side in SIDES
            }
            target_rotation = {
                side: episode.targets[f"{side}_wrist_rotation"][frame]
                for side in SIDES
            }
            initial_task_feasible[frame] = self._task_feasible(
                q[frame], target_position, target_rotation
            )
            initial_collision[frame] = bool(
                self._records(q[frame], left[frame], right[frame])
            )

        for trajectory_pass in range(
            int(self.config["max_trajectory_repair_passes"])
        ):
            changed_this_pass = 0
            for frame in range(len(q)):
                target_position = {
                    side: episode.targets[f"{side}_wrist_position"][frame]
                    for side in SIDES
                }
                target_rotation = {
                    side: episode.targets[f"{side}_wrist_rotation"][frame]
                    for side in SIDES
                }
                previous = q[frame - 1] if frame > 0 else None
                previous2 = q[frame - 2] if frame > 1 else previous
                following = q[frame + 1] if frame + 1 < len(q) else None
                following2 = q[frame + 2] if frame + 2 < len(q) else following
                current = q[frame].copy()
                task_ok = self._task_feasible(
                    current, target_position, target_rotation
                )
                if not task_ok:
                    candidate, meta = self._project_frame(
                        current,
                        episode.q_v2[frame],
                        target_position,
                        target_rotation,
                        left[frame],
                        right[frame],
                        (),
                        previous,
                        previous2,
                        following,
                        following2,
                        int(self.config["max_stage1_iterations"]),
                    )
                    old_position, old_orientation = self._task_values(
                        current, target_position, target_rotation
                    )
                    new_position, new_orientation = self._task_values(
                        candidate, target_position, target_rotation
                    )
                    old_score = (
                        int(np.any(old_position > self.position_tolerance)),
                        int(
                            np.any(
                                old_orientation
                                > self.orientation_tolerance + self.slack_bound
                            )
                        ),
                        float(np.max(old_position)),
                        float(np.max(old_orientation)),
                    )
                    new_score = (
                        int(np.any(new_position > self.position_tolerance)),
                        int(
                            np.any(
                                new_orientation
                                > self.orientation_tolerance + self.slack_bound
                            )
                        ),
                        float(np.max(new_position)),
                        float(np.max(new_orientation)),
                    )
                    frame_metadata[frame]["stage1_calls"] += 1
                    frame_metadata[frame]["stage1_success"] = bool(
                        new_score[:2] == (0, 0)
                    )
                    frame_metadata[frame]["stage1_optimizer"] = meta
                    if meta["constraints_satisfied"] and new_score < old_score:
                        current = candidate

                tracked: list[tuple[int, int]] = []
                stage2_meta: list[dict[str, Any]] = []
                for _ in range(int(self.config["max_collision_outer_passes"])):
                    records = self._records(
                        current, left[frame], right[frame]
                    )
                    for record in records:
                        pair = tuple(int(value) for value in record.geom_ids)
                        if pair not in tracked:
                            tracked.append(pair)
                    if not records:
                        break
                    candidate, meta = self._project_frame(
                        current,
                        episode.q_v2[frame],
                        target_position,
                        target_rotation,
                        left[frame],
                        right[frame],
                        tuple(tracked),
                        previous,
                        previous2,
                        following,
                        following2,
                        int(self.config["max_stage2_iterations"]),
                    )
                    stage2_meta.append(meta)
                    old_records = len(records)
                    new_records = len(
                        self._records(candidate, left[frame], right[frame])
                    )
                    old_task = self._task_feasible(
                        current, target_position, target_rotation
                    )
                    new_task = self._task_feasible(
                        candidate, target_position, target_rotation
                    )
                    if meta["constraints_satisfied"] and (
                        int(not new_task), new_records
                    ) < (
                        int(not old_task),
                        old_records,
                    ):
                        current = candidate
                    elif (
                        meta["constraints_satisfied"]
                        and new_task
                        and new_records == 0
                    ):
                        current = candidate
                        break
                    else:
                        break
                if stage2_meta:
                    frame_metadata[frame]["stage2_calls"] += len(stage2_meta)
                    frame_metadata[frame]["stage2_success"] = not bool(
                        self._records(current, left[frame], right[frame])
                    )
                    frame_metadata[frame]["stage2_optimizer"] = stage2_meta
                if not np.array_equal(current, q[frame]):
                    q[frame] = current
                    frame_metadata[frame]["changed"] = True
                    frame_metadata[frame]["last_changed_pass"] = trajectory_pass
                    changed_this_pass += 1
            if changed_this_pass == 0:
                break

        # Metrics and exported labels are evaluated in their actual float32 form.
        q = q.astype(np.float32).astype(np.float64)
        left = left.astype(np.float32).astype(np.float64)
        right = right.astype(np.float32).astype(np.float64)
        orientation_error = np.empty((len(q), 2), dtype=np.float64)
        final_task_feasible = np.zeros(len(q), dtype=bool)
        final_collision = np.zeros(len(q), dtype=bool)
        for frame in range(len(q)):
            target_position = {
                side: episode.targets[f"{side}_wrist_position"][frame]
                for side in SIDES
            }
            target_rotation = {
                side: episode.targets[f"{side}_wrist_rotation"][frame]
                for side in SIDES
            }
            _, orientation_error[frame] = self._task_values(
                q[frame], target_position, target_rotation
            )
            final_task_feasible[frame] = self._task_feasible(
                q[frame], target_position, target_rotation
            )
            final_collision[frame] = bool(
                self._records(q[frame], left[frame], right[frame])
            )
        slack_requested = np.maximum(
            0.0, orientation_error - self.orientation_tolerance
        )
        slack = np.minimum(slack_requested, self.slack_bound)
        return FeasibilityResult(
            episode=episode,
            q=q,
            left_hand=left,
            right_hand=right,
            orientation_slack=slack,
            orientation_slack_requested=slack_requested,
            metadata={
                "solver_class": type(self).__name__,
                "solver_parameters": self.config,
                "hand_temporal_projection": hand_meta,
                "initial_task_feasible_frames": int(
                    np.count_nonzero(initial_task_feasible)
                ),
                "final_task_feasible_frames": int(
                    np.count_nonzero(final_task_feasible)
                ),
                "initial_prohibited_collision_frames": int(
                    np.count_nonzero(initial_collision)
                ),
                "final_prohibited_collision_frames": int(
                    np.count_nonzero(final_collision)
                ),
                "changed_arm_frames": int(
                    np.count_nonzero(
                        np.linalg.norm(q - episode.q_v2, axis=1) > 1e-8
                    )
                ),
                "orientation_slack_bound_exhausted_frames": int(
                    np.count_nonzero(
                        np.any(
                            slack_requested > self.slack_bound + 1e-12,
                            axis=1,
                        )
                    )
                ),
                "temporal_constraint_interior": {
                    "step_rad": float(
                        self.config["temporal_step_constraint_interior_rad"]
                    ),
                    "acceleration_rad_s2": float(
                        self.config[
                            "temporal_acceleration_constraint_interior_rad_s2"
                        ]
                    ),
                    "acceptance_limits_changed": False,
                },
                "stage1_call_count": int(
                    sum(row["stage1_calls"] for row in frame_metadata)
                ),
                "stage2_call_count": int(
                    sum(row["stage2_calls"] for row in frame_metadata)
                ),
                "repair_records": [
                    {"frame": frame, **row}
                    for frame, row in enumerate(frame_metadata)
                    if row["stage1_calls"] or row["stage2_calls"]
                ],
            },
        )
