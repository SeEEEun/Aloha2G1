"""Global full-pose conditioning fix for the common feasibility resolver.

The frozen resolver's optimizer hard-coded B's non-gating spherical-object
orientation policy.  That is not representation-neutral: applying it to A can
create wrist-orientation failures in previously valid frames.  This subclass
changes no configured number.  It derives the rotational residual scale from
the already-frozen common natural-arm position/orientation weights and applies
one identical full-pose acceptance ordering to every frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from tools.doll_handoff_feasibility.common import rotation_error_rad

from .common import OUTPUT_ROOT, SIDES, load_a_trajectory, sha256_file
from .wrist_resolver import RepresentationNeutralWristResolver


class FullPoseCommonWristResolver(RepresentationNeutralWristResolver):
    """Frozen common resolver plus a method-independent 6-D acceptance fix."""

    def __init__(self, *args: Any, output_root: str | Path = OUTPUT_ROOT, **kwargs: Any):
        super().__init__(*args, output_root=output_root, **kwargs)
        self.adapter_path = Path(__file__).resolve()
        self.adapter_sha256 = sha256_file(self.adapter_path)
        shared = self.common["shared_temporal_ik"]
        self.orientation_tolerance = float(shared["orientation_tolerance_rad"])
        self.orientation_residual_scale = float(
            self.config["constrained_tracking"]["position_residual_scale_per_m"]
        ) * float(shared["orientation_weight"]) / float(shared["position_weight"])
        self._active_source_rotation: Mapping[str, np.ndarray] | None = None

    def _cache_paths(self, episode: int) -> tuple[Path, Path]:
        stem = self._stable_id(episode)
        return (
            self.output_root / "after_full_pose/trajectories" / f"{stem}.npz",
            self.output_root / "after_full_pose/metrics" / f"{stem}.solver.json",
        )

    @staticmethod
    def _stable_id(episode: int) -> str:
        from .common import stable_episode_id

        return stable_episode_id(episode)

    @staticmethod
    def _rotation_vector(actual: np.ndarray, desired: np.ndarray) -> np.ndarray:
        relative = np.asarray(actual, dtype=np.float64).T @ np.asarray(
            desired, dtype=np.float64
        )
        return Rotation.from_matrix(relative).as_rotvec()

    def _frame_pose_values(
        self,
        q: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        source_rotation: Mapping[str, np.ndarray],
        frame: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self._pose(q)
        return (
            np.asarray(
                [
                    np.linalg.norm(position[side] - source_model[side][frame])
                    for side in SIDES
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    rotation_error_rad(
                        rotation[side], source_rotation[side][frame]
                    )
                    for side in SIDES
                ],
                dtype=np.float64,
            ),
        )

    def _frame_pose_key(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> tuple[float, ...]:
        normalized = np.concatenate(
            (
                position / self.strict_tolerance,
                orientation / self.orientation_tolerance,
            )
        )
        return (
            float(np.count_nonzero(normalized > 1.0)),
            float(np.max(normalized)),
            float(np.sum(np.maximum(normalized - 1.0, 0.0))),
            float(np.max(position)),
            float(np.max(orientation)),
            float(np.sum(normalized)),
        )

    def _track(
        self,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        source_rotation: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Frozen bidirectional tracker with a representation-neutral 6-D gate."""
        settings = self.config["constrained_tracking"]
        q = np.asarray(q_before, dtype=np.float64).copy()
        initial_position_error, _ = self._position_errors(q, source_model)
        initial_orientation_error = np.empty(len(q), dtype=np.float64)
        for frame, value in enumerate(q):
            _, orientation = self._frame_pose_values(
                value, source_model, source_rotation, frame
            )
            initial_orientation_error[frame] = float(np.max(orientation))
        normalized_error = np.maximum(
            initial_position_error / self.strict_tolerance,
            initial_orientation_error / self.orientation_tolerance,
        )
        active = self._tracking_active_mask(
            normalized_error * self.strict_tolerance
        )
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
                old_position, old_orientation = self._frame_pose_values(
                    q[frame], source_model, source_rotation, frame
                )
                old_key = self._frame_pose_key(old_position, old_orientation)

                def residual(value: np.ndarray) -> np.ndarray:
                    position, rotation = self._pose(value)
                    values = [
                        float(settings["position_residual_scale_per_m"])
                        * np.concatenate(
                            [
                                position[side] - source_model[side][frame]
                                for side in SIDES
                            ]
                        ),
                        self.orientation_residual_scale
                        * np.concatenate(
                            [
                                self._rotation_vector(
                                    rotation[side], source_rotation[side][frame]
                                )
                                for side in SIDES
                            ]
                        ),
                        float(settings["frozen_q_residual_scale"])
                        * (value - q_before[frame]),
                        float(settings["neighbor_q_residual_scale"])
                        * (value - midpoint),
                        float(settings["nominal_q_residual_scale"])
                        * (value - self.nominal),
                    ]
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
                new_position, new_orientation = self._frame_pose_values(
                    candidate, source_model, source_rotation, frame
                )
                new_key = self._frame_pose_key(new_position, new_orientation)
                if new_key < old_key:
                    q[frame] = candidate
                    accepted += 1
            serialized = q.astype(np.float32).astype(np.float64)
            temporal = self._temporal_metrics(serialized, left_hand, right_hand, fps)
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
            current_orientation = np.empty(len(q), dtype=np.float64)
            for frame, value in enumerate(q):
                _, orientation = self._frame_pose_values(
                    value, source_model, source_rotation, frame
                )
                current_orientation[frame] = float(np.max(orientation))
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
                    "orientation_hard_frames": int(
                        np.count_nonzero(
                            current_orientation > self.orientation_tolerance
                        )
                    ),
                    "maximum_source_residual_m": float(np.max(current_error)),
                    "maximum_orientation_residual_rad": float(
                        np.max(current_orientation)
                    ),
                    "mean_source_residual_m": float(np.mean(current_error)),
                    "temporal": temporal,
                }
            )
        q, orientation_constraint_report = self._orientation_constraint_repair(
            q,
            q_before,
            source_model,
            source_rotation,
            left_hand,
            right_hand,
            fps,
        )
        q, orientation_window_report = self._orientation_window_repair(
            q,
            q_before,
            source_model,
            source_rotation,
            left_hand,
            right_hand,
            fps,
        )
        final_error, _ = self._position_errors(q, source_model)
        final_orientation = np.empty(len(q), dtype=np.float64)
        for frame, value in enumerate(q):
            _, orientation = self._frame_pose_values(
                value, source_model, source_rotation, frame
            )
            final_orientation[frame] = float(np.max(orientation))
        return q, {
            "algorithm": (
                settings["algorithm"]
                + "_WITH_GLOBAL_FULL_POSE_ACCEPTANCE_CONDITIONING"
            ),
            "common_bug_fix": (
                "replace B-hard-coded non-gating orientation tie with the "
                "representation target's declared full-pose constraint"
            ),
            "configured_parameters_changed": False,
            "orientation_residual_scale_derivation": (
                "constrained_tracking.position_residual_scale_per_m * "
                "shared_temporal_ik.orientation_weight / "
                "shared_temporal_ik.position_weight"
            ),
            "orientation_residual_scale": self.orientation_residual_scale,
            "active_frame_count": int(np.count_nonzero(active)),
            "initial_strict_source_residual_frames": int(
                np.count_nonzero(initial_position_error > self.strict_tolerance)
            ),
            "initial_physical_source_residual_frames": int(
                np.count_nonzero(initial_position_error > self.physical_tolerance)
            ),
            "initial_orientation_hard_frames": int(
                np.count_nonzero(
                    initial_orientation_error > self.orientation_tolerance
                )
            ),
            "initial_maximum_source_residual_m": float(
                np.max(initial_position_error)
            ),
            "initial_maximum_orientation_residual_rad": float(
                np.max(initial_orientation_error)
            ),
            "final_strict_source_residual_frames": int(
                np.count_nonzero(final_error > self.strict_tolerance)
            ),
            "final_physical_source_residual_frames": int(
                np.count_nonzero(final_error > self.physical_tolerance)
            ),
            "final_orientation_hard_frames": int(
                np.count_nonzero(final_orientation > self.orientation_tolerance)
            ),
            "final_maximum_source_residual_m": float(np.max(final_error)),
            "final_maximum_orientation_residual_rad": float(
                np.max(final_orientation)
            ),
            "final_mean_source_residual_m": float(np.mean(final_error)),
            "before_temporal": before_temporal,
            "passes": pass_reports,
            "orientation_constraint_repair": orientation_constraint_report,
            "orientation_window_repair": orientation_window_report,
        }

    def _orientation_constraint_repair(
        self,
        q_input: np.ndarray,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        source_rotation: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Minimize source translation subject to immutable SO(3) feasibility."""
        settings = self.config["constrained_tracking"]
        q = np.asarray(q_input, dtype=np.float64).copy()
        reports: list[dict[str, Any]] = []
        for pass_index in range(int(settings["bidirectional_passes"])):
            pass_before = q.copy()
            hard_frames = []
            for frame, value in enumerate(q):
                _, orientation = self._frame_pose_values(
                    value, source_model, source_rotation, frame
                )
                if np.any(orientation > self.orientation_tolerance):
                    hard_frames.append(frame)
            order = np.asarray(hard_frames, dtype=np.int64)
            if pass_index % 2:
                order = order[::-1]
            accepted = 0
            optimizer_successes = 0
            for raw_frame in order:
                frame = int(raw_frame)
                lower, upper = self._tracking_bounds(frame, q, fps)
                x0 = np.minimum(np.maximum(q[frame], lower), upper)
                midpoint = 0.5 * (
                    (q[frame - 1] if frame > 0 else q[frame])
                    + (q[frame + 1] if frame + 1 < len(q) else q[frame])
                )
                old_position, old_orientation = self._frame_pose_values(
                    q[frame], source_model, source_rotation, frame
                )

                def objective(value: np.ndarray) -> float:
                    position, _ = self._frame_pose_values(
                        value, source_model, source_rotation, frame
                    )
                    return (
                        float(settings["position_residual_scale_per_m"]) ** 2
                        * float(position @ position)
                        + float(settings["frozen_q_residual_scale"]) ** 2
                        * float(
                            np.dot(
                                value - q_before[frame],
                                value - q_before[frame],
                            )
                        )
                        + float(settings["neighbor_q_residual_scale"]) ** 2
                        * float(np.dot(value - midpoint, value - midpoint))
                        + float(settings["nominal_q_residual_scale"]) ** 2
                        * float(np.dot(value - self.nominal, value - self.nominal))
                    )

                def constraint(value: np.ndarray) -> np.ndarray:
                    _, orientation = self._frame_pose_values(
                        value, source_model, source_rotation, frame
                    )
                    return self.orientation_tolerance - orientation

                result = minimize(
                    objective,
                    x0,
                    method="SLSQP",
                    bounds=list(zip(lower, upper)),
                    constraints={"type": "ineq", "fun": constraint},
                    options={
                        "maxiter": int(
                            self.config["collision_repair"][
                                "maximum_anchor_iterations"
                            ]
                        ),
                        "ftol": 1e-10,
                        "disp": False,
                    },
                )
                optimizer_successes += int(bool(result.success))
                neighbors = []
                if frame > 0:
                    neighbors.append(q[frame - 1])
                if frame + 1 < len(q):
                    neighbors.append(q[frame + 1])
                candidate = self._cap_branch_step(
                    q[frame],
                    np.asarray(result.x, dtype=np.float64),
                    neighbors,
                    float(self.acceptance["branch_absolute_step_norm_rad"])
                    - float(settings["branch_step_numerical_interior_rad"]),
                )
                new_position, new_orientation = self._frame_pose_values(
                    candidate, source_model, source_rotation, frame
                )
                old_orientation_key = (
                    int(np.count_nonzero(old_orientation > self.orientation_tolerance)),
                    float(np.max(old_orientation)),
                    float(np.sum(old_orientation)),
                    float(np.max(old_position)),
                )
                new_orientation_key = (
                    int(np.count_nonzero(new_orientation > self.orientation_tolerance)),
                    float(np.max(new_orientation)),
                    float(np.sum(new_orientation)),
                    float(np.max(new_position)),
                )
                if (
                    float(np.min(constraint(candidate))) >= -1e-6
                    and new_orientation_key < old_orientation_key
                ):
                    q[frame] = candidate
                    accepted += 1
            serialized = q.astype(np.float32).astype(np.float64)
            temporal = self._temporal_metrics(serialized, left_hand, right_hand, fps)
            if (
                not self._temporal_passes(temporal)
                or int(temporal["branch_discontinuity_count"])
                > int(
                    self._temporal_metrics(
                        pass_before, left_hand, right_hand, fps
                    )["branch_discontinuity_count"]
                )
            ):
                q = pass_before
                retained = False
            else:
                q = serialized
                retained = True
            remaining = 0
            maximum = 0.0
            for frame, value in enumerate(q):
                _, orientation = self._frame_pose_values(
                    value, source_model, source_rotation, frame
                )
                remaining += int(np.any(orientation > self.orientation_tolerance))
                maximum = max(maximum, float(np.max(orientation)))
            reports.append(
                {
                    "pass_index": pass_index,
                    "direction": "forward" if pass_index % 2 == 0 else "backward",
                    "attempted_frames": len(order),
                    "optimizer_successes": optimizer_successes,
                    "accepted_frame_updates": accepted,
                    "pass_retained": retained,
                    "remaining_orientation_hard_frames": remaining,
                    "maximum_orientation_residual_rad": maximum,
                    "temporal": temporal,
                }
            )
        return q, {
            "algorithm": (
                "global_bidirectional_box_constrained_nearest_translation_"
                "with_immutable_so3_gate"
            ),
            "task_independent": True,
            "configured_parameters_changed": False,
            "orientation_projection": "NONE",
            "passes": reports,
        }

    def _orientation_arrays(
        self,
        q: np.ndarray,
        source_rotation: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        _, rotations = self._pose_arrays(q)
        by_side = np.asarray(
            [
                [
                    rotation_error_rad(
                        rotations[side][frame], source_rotation[side][frame]
                    )
                    for side in SIDES
                ]
                for frame in range(len(q))
            ],
            dtype=np.float64,
        )
        return np.max(by_side, axis=1), by_side

    def _orientation_anchor(
        self,
        frame: int,
        q: np.ndarray,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        source_rotation: Mapping[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.config["constrained_tracking"]
        lower = self.g1.arm_limits[:, 0] + 1e-8
        upper = self.g1.arm_limits[:, 1] - 1e-8
        midpoint = 0.5 * (
            (q[frame - 1] if frame > 0 else q[frame])
            + (q[frame + 1] if frame + 1 < len(q) else q[frame])
        )

        def orientation_vector(value: np.ndarray) -> np.ndarray:
            _, rotation = self._pose(value)
            return np.concatenate(
                [
                    self._rotation_vector(
                        rotation[side], source_rotation[side][frame]
                    )
                    for side in SIDES
                ]
            )

        seed_result = least_squares(
            orientation_vector,
            np.minimum(np.maximum(q[frame], lower), upper),
            bounds=(lower, upper),
            max_nfev=int(settings["maximum_function_evaluations"]),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )

        def objective(value: np.ndarray) -> float:
            position, _ = self._frame_pose_values(
                value, source_model, source_rotation, frame
            )
            return (
                float(settings["position_residual_scale_per_m"]) ** 2
                * float(position @ position)
                + float(settings["frozen_q_residual_scale"]) ** 2
                * float(
                    np.dot(value - q_before[frame], value - q_before[frame])
                )
                + float(settings["neighbor_q_residual_scale"]) ** 2
                * float(np.dot(value - midpoint, value - midpoint))
                + float(settings["nominal_q_residual_scale"]) ** 2
                * float(np.dot(value - self.nominal, value - self.nominal))
            )

        def constraint(value: np.ndarray) -> np.ndarray:
            _, orientation = self._frame_pose_values(
                value, source_model, source_rotation, frame
            )
            return self.orientation_tolerance - orientation

        result = minimize(
            objective,
            np.asarray(seed_result.x, dtype=np.float64),
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints={"type": "ineq", "fun": constraint},
            options={
                "maxiter": int(
                    self.config["collision_repair"]["maximum_anchor_iterations"]
                ),
                "ftol": 1e-10,
                "disp": False,
            },
        )
        candidate = np.asarray(result.x, dtype=np.float64)
        margin = float(np.min(constraint(candidate)))
        return candidate, {
            "frame": frame,
            "orientation_seed_success": bool(seed_result.success),
            "orientation_seed_function_evaluations": int(seed_result.nfev),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "optimizer_iterations": int(result.nit),
            "minimum_orientation_constraint_margin_rad": margin,
            "delta_norm_rad": float(np.linalg.norm(candidate - q[frame])),
        }

    def _orientation_window_repair(
        self,
        q_input: np.ndarray,
        q_before: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        source_rotation: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Use the frozen minimum-acceleration windows for an SO(3) conflict."""
        settings = self.config["collision_repair"]
        q = np.asarray(q_input, dtype=np.float64).copy()
        initial_error, _ = self._orientation_arrays(q, source_rotation)
        initial_collision = self._collision_metrics(
            q, left_hand, right_hand, fps
        )
        iterations: list[dict[str, Any]] = []
        for iteration in range(int(settings["maximum_event_iterations"])):
            before_error, _ = self._orientation_arrays(q, source_rotation)
            hard_frames = np.flatnonzero(
                before_error > self.orientation_tolerance
            )
            if not len(hard_frames):
                break
            anchor_frame = int(
                hard_frames[np.argmax(before_error[hard_frames])]
            )
            repaired, anchor_report = self._orientation_anchor(
                anchor_frame,
                q,
                q_before,
                source_model,
                source_rotation,
            )
            delta_value = repaired - q[anchor_frame]
            candidates: list[
                tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]
            ] = []
            attempts: list[dict[str, Any]] = []
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
                    after_error, _ = self._orientation_arrays(
                        candidate, source_rotation
                    )
                    temporal = self._temporal_metrics(
                        candidate, left_hand, right_hand, fps
                    )
                    hard_before = int(len(hard_frames))
                    hard_after = int(
                        np.count_nonzero(
                            after_error > self.orientation_tolerance
                        )
                    )
                    safe = bool(
                        hard_after < hard_before
                        and self._temporal_passes(temporal)
                        and int(temporal["branch_discontinuity_count"]) == 0
                    )
                    collision = None
                    if safe:
                        collision = self._collision_metrics(
                            candidate, left_hand, right_hand, fps
                        )
                        safe = bool(
                            int(collision["hard_collision_frame_count"])
                            <= int(
                                initial_collision["hard_collision_frame_count"]
                            )
                        )
                    attempt = {
                        "padding_frames": int(padding),
                        "scale": float(scale),
                        "window_start": start,
                        "window_end": end,
                        "hard_orientation_frames_before": hard_before,
                        "hard_orientation_frames_after": hard_after,
                        "maximum_orientation_residual_rad": float(
                            np.max(after_error)
                        ),
                        "accepted_by_common_hard_gates": safe,
                        "temporal": temporal,
                    }
                    if collision is not None:
                        attempt["hard_collision_frame_count"] = int(
                            collision["hard_collision_frame_count"]
                        )
                    attempts.append(attempt)
                    if safe:
                        source_error, _ = self._position_errors(
                            candidate, source_model
                        )
                        score = (
                            hard_after,
                            float(np.max(after_error)),
                            float(np.mean(source_error)),
                            float(
                                np.mean(
                                    np.linalg.norm(
                                        candidate - q_before, axis=1
                                    )
                                )
                            ),
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
                        "reason": (
                            "NO_MINIMUM_ACCELERATION_WINDOW_PASSED_COMMON_GATES"
                        ),
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
                    "anchor_report": anchor_report,
                    "selected": selected,
                    "attempts": attempts,
                }
            )
        final_error, _ = self._orientation_arrays(q, source_rotation)
        return q, {
            "algorithm": (
                "signed_orientation_anchor_projection_with_frozen_"
                "minimum_acceleration_windows"
            ),
            "task_independent": True,
            "configured_parameters_changed": False,
            "orientation_projection": "NONE",
            "initial_hard_orientation_frame_count": int(
                np.count_nonzero(initial_error > self.orientation_tolerance)
            ),
            "final_hard_orientation_frame_count": int(
                np.count_nonzero(final_error > self.orientation_tolerance)
            ),
            "initial_maximum_orientation_residual_rad": float(
                np.max(initial_error)
            ),
            "final_maximum_orientation_residual_rad": float(
                np.max(final_error)
            ),
            "iterations": iterations,
        }

    def _candidate_state(
        self,
        q: np.ndarray,
        source_model: Mapping[str, np.ndarray],
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
    ) -> dict[str, Any]:
        state = super()._candidate_state(
            q, source_model, left_hand, right_hand, fps
        )
        if self._active_source_rotation is not None:
            _, rotations = self._pose_arrays(q)
            errors = np.asarray(
                [
                    rotation_error_rad(
                        rotations[side][frame],
                        self._active_source_rotation[side][frame],
                    )
                    for side in SIDES
                    for frame in range(len(q))
                ],
                dtype=np.float64,
            ).reshape(2, len(q)).T
            per_frame = np.max(errors, axis=1)
            state["orientation_hard_frame_count"] = int(
                np.count_nonzero(per_frame > self.orientation_tolerance)
            )
            state["maximum_orientation_residual_rad"] = float(
                np.max(per_frame)
            )
            state["mean_orientation_residual_rad"] = float(np.mean(errors))
        return state

    def _collision_candidate_safe(
        self, before: Mapping[str, Any], after: Mapping[str, Any]
    ) -> bool:
        orientation_safe = True
        if "orientation_hard_frame_count" in before:
            orientation_safe = bool(
                int(after["orientation_hard_frame_count"])
                <= int(before["orientation_hard_frame_count"])
                and float(after["maximum_orientation_residual_rad"])
                <= max(
                    float(before["maximum_orientation_residual_rad"]),
                    self.orientation_tolerance,
                )
                + 1e-6
            )
        return bool(
            orientation_safe and super()._collision_candidate_safe(before, after)
        )

    @staticmethod
    def _compact_candidate_state(state: Mapping[str, Any]) -> dict[str, Any]:
        compact = RepresentationNeutralWristResolver._compact_candidate_state(state)
        for key in (
            "orientation_hard_frame_count",
            "maximum_orientation_residual_rad",
            "mean_orientation_residual_rad",
        ):
            if key in state:
                compact[key] = state[key]
        return compact

    def solve_episode(self, episode: int, export: bool = True):
        values = load_a_trajectory(episode)
        _, rotations = self._source_targets(values)
        self._active_source_rotation = rotations
        try:
            return super().solve_episode(episode, export=export)
        finally:
            self._active_source_rotation = None

    def export_episode(self, result, frozen_values=None, geometry=None):
        result.metadata.update(
            {
                "common_solver_bug_fix": (
                    "GLOBAL_FULL_POSE_ACCEPTANCE_CONDITIONING"
                ),
                "common_solver_bug_fix_applies_to_a_and_b": True,
                "common_solver_bug_fix_parameters_changed": False,
                "orientation_projection_added": False,
                "orientation_residual_scale": self.orientation_residual_scale,
                "orientation_residual_scale_source": (
                    "frozen shared natural-arm IK position/orientation weights"
                ),
            }
        )
        return super().export_episode(result, frozen_values, geometry)


__all__ = ["FullPoseCommonWristResolver"]
