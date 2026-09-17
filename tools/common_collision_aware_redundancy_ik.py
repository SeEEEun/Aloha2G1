"""Common method-blind collision-aware redundant IK for the G1 arms.

The solver realizes an already-selected bilateral wrist target.  It consumes no
method, episode, object, event, scorer, or outcome input.  Collision clearance
is obtained by selecting shoulder/elbow/forearm redundancy, never by editing the
incoming target arrays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from tools.common_g1_morphology_adapter import CommonG1MorphologyAdapter
from tools.doll_handoff_retargeting.common import SIDES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/common_collision_aware_redundancy_ik_v1.json"


@dataclass
class CollisionAwareIKResult:
    q: np.ndarray
    metadata: dict[str, Any]


class CommonCollisionAwareRedundancyIK:
    """Realize fixed targets using one deterministic collision-aware branch."""

    def __init__(
        self,
        adapter: CommonG1MorphologyAdapter,
        config_path: str | Path = DEFAULT_CONFIG,
    ):
        self.adapter = adapter
        self.g1 = adapter.g1
        self.nominal = np.asarray(adapter.nominal, dtype=np.float64)
        self.stand = self.g1.stand_qpos[self.g1.arm_qpos_ids].astype(np.float64)
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        forbidden = (
            "representation_mode_input_allowed",
            "episode_index_input_allowed",
            "object_pose_input_allowed",
            "task_success_input_allowed",
            "physical_outcome_input_allowed",
            "incoming_target_mutation_allowed",
        )
        if self.config.get("method_blind") is not True:
            raise RuntimeError("collision-aware IK must be method-blind")
        if any(bool(self.config.get(key)) for key in forbidden):
            raise RuntimeError("collision-aware IK enables a forbidden input")
        self.position_tolerance = float(self.config["position_tolerance_interior_m"])
        self.orientation_tolerance = float(
            self.config["orientation_tolerance_interior_rad"]
        )
        self.clearance = float(self.config["collision_clearance_m"])
        self.lower = self.g1.arm_limits[:, 0].astype(np.float64) + 1e-7
        self.upper = self.g1.arm_limits[:, 1].astype(np.float64) - 1e-7

    @staticmethod
    def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return Rotation.from_matrix(
            np.asarray(target, dtype=np.float64)
            @ np.asarray(current, dtype=np.float64).T
        ).as_rotvec()

    def _errors(
        self,
        q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        include_orientation: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self.adapter._pose(np.asarray(q, dtype=np.float64))
        position_error = np.asarray(
            [
                np.linalg.norm(position[side] - target_position[side])
                for side in SIDES
            ],
            dtype=np.float64,
        )
        orientation_error = np.asarray(
            [
                np.linalg.norm(
                    self._rotation_error(rotation[side], target_rotation[side])
                )
                if include_orientation
                else 0.0
                for side in SIDES
            ],
            dtype=np.float64,
        )
        return position_error, orientation_error

    def _task_residual(
        self,
        q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        include_orientation: bool,
        reference_terms: Sequence[tuple[float, np.ndarray]],
        settings: Mapping[str, Any],
    ) -> np.ndarray:
        position, rotation = self.adapter._pose(q)
        values = [
            float(settings["position_scale_per_m"])
            * np.concatenate(
                [position[side] - target_position[side] for side in SIDES]
            )
        ]
        if include_orientation:
            values.append(
                float(settings["orientation_scale_per_rad"])
                * np.concatenate(
                    [
                        self._rotation_error(rotation[side], target_rotation[side])
                        for side in SIDES
                    ]
                )
            )
        values.extend(
            np.sqrt(max(weight, 0.0)) * (q - reference)
            for weight, reference in reference_terms
            if weight > 0.0
        )
        return np.concatenate(values)

    def _tracked_collision_pairs(
        self,
        q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        tracked: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        records = self.adapter._records(q, left_hand, right_hand)
        for record in records:
            pair = tuple(map(int, record["geom_pair"]))
            if pair not in tracked:
                tracked.append(pair)
        return records

    def _constraint_values(
        self,
        q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        include_orientation: bool,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        tracked: Sequence[tuple[int, int]],
        step_reference: np.ndarray | None,
        step_norm_limit: float | None,
    ) -> np.ndarray:
        position_error, orientation_error = self._errors(
            q, target_position, target_rotation, include_orientation
        )
        values: list[float] = list(self.position_tolerance - position_error)
        if include_orientation:
            values.extend(self.orientation_tolerance - orientation_error)
        values.extend(
            self.adapter._distance(q, left_hand, right_hand, pair) - self.clearance
            for pair in tracked
        )
        if step_reference is not None and step_norm_limit is not None:
            values.append(step_norm_limit - float(np.linalg.norm(q - step_reference)))
        return np.asarray(values, dtype=np.float64)

    def _clear_contacts(
        self,
        start: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        include_orientation: bool,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        input_q: np.ndarray,
        previous: np.ndarray | None,
        predicted: np.ndarray | None,
        lower: np.ndarray,
        upper: np.ndarray,
        settings: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        current = np.clip(np.asarray(start, dtype=np.float64), lower, upper)
        tracked: list[tuple[int, int]] = []
        passes: list[dict[str, Any]] = []
        best = current.copy()
        best_key: tuple[Any, ...] | None = None
        for discovery in range(int(settings["maximum_pair_discovery_passes"])):
            records = self._tracked_collision_pairs(
                current, left_hand, right_hand, tracked
            )
            position_error, orientation_error = self._errors(
                current, target_position, target_rotation, include_orientation
            )
            step_limit = (
                float(settings["maximum_step_norm_rad"])
                if previous is not None and "maximum_step_norm_rad" in settings
                else None
            )
            step_valid = bool(
                previous is None
                or step_limit is None
                or np.linalg.norm(current - previous) <= step_limit + 1e-9
            )
            key = (
                0 if float(np.max(position_error)) <= self.position_tolerance else 1,
                0
                if (not include_orientation or float(np.max(orientation_error)) <= self.orientation_tolerance)
                else 1,
                0 if step_valid else 1,
                len(records),
                max(
                    (float(record["penetration_depth_m"]) for record in records),
                    default=0.0,
                ),
                float(np.max(position_error)),
                float(np.max(orientation_error)),
                float(np.linalg.norm(current - (previous if previous is not None else input_q))),
                float(np.linalg.norm(current - self.nominal)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = current.copy()
            if not records and key[0] == 0 and key[1] == 0 and step_valid:
                break

            reference_terms: list[tuple[float, np.ndarray]] = [
                (float(settings["input_q_weight"]), input_q),
                (float(settings["natural_q_weight"]), self.nominal),
            ]
            if previous is not None:
                reference_terms.append((float(settings["previous_q_weight"]), previous))
            if predicted is not None and "predicted_q_weight" in settings:
                reference_terms.append((float(settings["predicted_q_weight"]), predicted))

            def objective(value: np.ndarray) -> float:
                residual = self._task_residual(
                    value,
                    target_position,
                    target_rotation,
                    include_orientation,
                    reference_terms,
                    settings,
                )
                return float(residual @ residual)

            def constraint(value: np.ndarray) -> np.ndarray:
                return self._constraint_values(
                    value,
                    target_position,
                    target_rotation,
                    include_orientation,
                    left_hand,
                    right_hand,
                    tracked,
                    previous,
                    step_limit,
                )

            result = minimize(
                objective,
                current,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints={"type": "ineq", "fun": constraint},
                options={
                    "maxiter": int(settings["maximum_collision_iterations"]),
                    "ftol": 1e-11,
                    "disp": False,
                },
            )
            candidate = np.clip(np.asarray(result.x, dtype=np.float64), lower, upper)
            margin = float(np.min(constraint(candidate)))
            passes.append(
                {
                    "discovery_pass": discovery,
                    "optimizer_success": bool(result.success),
                    "optimizer_message": str(result.message),
                    "optimizer_iterations": int(result.nit),
                    "tracked_pair_count": len(tracked),
                    "constraint_margin": margin,
                    "contact_count_before": len(records),
                    "contact_count_after": len(
                        self.adapter._records(candidate, left_hand, right_hand)
                    ),
                }
            )
            candidate_position, candidate_orientation = self._errors(
                candidate, target_position, target_rotation, include_orientation
            )
            candidate_records = self.adapter._records(candidate, left_hand, right_hand)
            candidate_key = (
                0 if float(np.max(candidate_position)) <= self.position_tolerance else 1,
                0
                if (not include_orientation or float(np.max(candidate_orientation)) <= self.orientation_tolerance)
                else 1,
                0 if margin >= -1e-6 else 1,
                len(candidate_records),
                max(
                    (
                        float(record["penetration_depth_m"])
                        for record in candidate_records
                    ),
                    default=0.0,
                ),
                float(np.max(candidate_position)),
                float(np.max(candidate_orientation)),
                float(np.linalg.norm(candidate - (previous if previous is not None else input_q))),
                float(np.linalg.norm(candidate - self.nominal)),
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best = candidate.copy()
            current = candidate
            if margin < -1e-6 and not result.success:
                break
        final_records = self.adapter._records(best, left_hand, right_hand)
        position_error, orientation_error = self._errors(
            best, target_position, target_rotation, include_orientation
        )
        return best, {
            "tracked_geom_pairs": [list(pair) for pair in tracked],
            "passes": passes,
            "contact_count": len(final_records),
            "maximum_penetration_m": max(
                (float(record["penetration_depth_m"]) for record in final_records),
                default=0.0,
            ),
            "position_error_max_m": float(np.max(position_error)),
            "orientation_error_max_rad": float(np.max(orientation_error)),
            "collision_free": not final_records,
        }

    def _arm_candidates(
        self,
        side: str,
        input_q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        include_orientation: bool,
        settings: Mapping[str, Any],
    ) -> list[np.ndarray]:
        block = slice(0, 7) if side == "left" else slice(7, 14)
        lower = self.lower[block]
        upper = self.upper[block]
        seeds = [input_q[block], self.nominal[block], self.stand[block]]
        halton = qmc.Halton(d=7, scramble=False).random(
            int(settings["halton_seed_count_per_arm"])
        )
        seeds.extend(lower + row * (upper - lower) for row in halton)
        output: list[tuple[tuple[float, ...], np.ndarray]] = []
        for seed in seeds:
            def residual(value: np.ndarray) -> np.ndarray:
                q = input_q.copy()
                q[block] = value
                position, rotation = self.adapter._pose(q)
                values = [
                    float(settings["position_scale_per_m"])
                    * (position[side] - target_position[side]),
                ]
                if include_orientation:
                    values.append(
                        float(settings["orientation_scale_per_rad"])
                        * self._rotation_error(rotation[side], target_rotation[side])
                    )
                values.append(0.001 * (value - self.nominal[block]))
                return np.concatenate(values)

            solution = least_squares(
                residual,
                np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=int(settings["maximum_position_iterations"]),
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            q = input_q.copy()
            q[block] = solution.x
            position_error, orientation_error = self._errors(
                q, target_position, target_rotation, include_orientation
            )
            side_index = 0 if side == "left" else 1
            if position_error[side_index] > self.position_tolerance:
                continue
            if include_orientation and orientation_error[side_index] > self.orientation_tolerance:
                continue
            if any(np.linalg.norm(solution.x - value) < 1e-5 for _, value in output):
                continue
            output.append(
                (
                    (
                        float(position_error[side_index]),
                        float(orientation_error[side_index]),
                        float(np.linalg.norm(solution.x - input_q[block])),
                        float(np.linalg.norm(solution.x - self.nominal[block])),
                    ),
                    solution.x.copy(),
                )
            )
        output.sort(key=lambda item: item[0])
        return [value for _, value in output[: int(settings["per_arm_candidate_keep"])]]

    def _anchor_solution(
        self,
        input_q: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        include_orientation: bool,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.config["anchor_search"]
        by_side = {
            side: self._arm_candidates(
                side,
                input_q,
                target_position,
                target_rotation,
                include_orientation,
                settings,
            )
            for side in SIDES
        }
        if any(not by_side[side] for side in SIDES):
            return input_q.copy(), {
                "status": "NO_CARTESIAN_ARM_CANDIDATES",
                "candidate_counts": {side: len(by_side[side]) for side in SIDES},
            }
        combinations: list[tuple[tuple[Any, ...], np.ndarray]] = []
        for left in by_side["left"]:
            for right in by_side["right"]:
                q = np.concatenate((left, right))
                records = self.adapter._records(q, left_hand, right_hand)
                position_error, orientation_error = self._errors(
                    q, target_position, target_rotation, include_orientation
                )
                combinations.append(
                    (
                        (
                            0 if not records else 1,
                            len(records),
                            max(
                                (
                                    float(record["penetration_depth_m"])
                                    for record in records
                                ),
                                default=0.0,
                            ),
                            float(np.max(position_error)),
                            float(np.max(orientation_error)),
                            float(np.linalg.norm(q - input_q)),
                            float(np.linalg.norm(q - self.nominal)),
                        ),
                        q,
                    )
                )
        combinations.sort(key=lambda item: item[0])
        attempts: list[dict[str, Any]] = []
        best_key: tuple[Any, ...] | None = None
        best = input_q.copy()
        for rank, (_, seed) in enumerate(
            combinations[: int(settings["bilateral_candidate_keep"])]
        ):
            candidate, report = self._clear_contacts(
                seed,
                target_position,
                target_rotation,
                include_orientation,
                left_hand,
                right_hand,
                input_q,
                None,
                None,
                self.lower,
                self.upper,
                settings,
            )
            position_error, orientation_error = self._errors(
                candidate, target_position, target_rotation, include_orientation
            )
            records = self.adapter._records(candidate, left_hand, right_hand)
            key = (
                0 if float(np.max(position_error)) <= self.position_tolerance else 1,
                0
                if (not include_orientation or float(np.max(orientation_error)) <= self.orientation_tolerance)
                else 1,
                0 if not records else 1,
                len(records),
                float(np.max(position_error)),
                float(np.max(orientation_error)),
                float(np.linalg.norm(candidate - input_q)),
                float(np.linalg.norm(candidate - self.nominal)),
            )
            attempts.append({"rank": rank, "key": list(key), "report": report})
            if best_key is None or key < best_key:
                best_key = key
                best = candidate.copy()
            if key[0] == 0 and key[1] == 0 and key[2] == 0:
                break
        return best, {
            "status": "PASS"
            if best_key is not None and best_key[0] == 0 and best_key[1] == 0 and best_key[2] == 0
            else "NO_COLLISION_FREE_ANCHOR",
            "candidate_counts": {side: len(by_side[side]) for side in SIDES},
            "combination_count": len(combinations),
            "attempts": attempts,
        }

    def _frame_solution(
        self,
        input_q: np.ndarray,
        previous: np.ndarray,
        previous2: np.ndarray,
        target_position: Mapping[str, np.ndarray],
        target_rotation: Mapping[str, np.ndarray],
        include_orientation: bool,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self.config["temporal_realization"]
        predicted = np.clip(2.0 * previous - previous2, self.lower, self.upper)
        step = float(settings["maximum_joint_step_rad"])
        acceleration = float(settings["maximum_joint_acceleration_step_rad"])
        lower = np.maximum(self.lower, previous - step)
        upper = np.minimum(self.upper, previous + step)
        if bool(
            settings.get(
                "acceleration_box_used_during_bidirectional_branch_propagation",
                False,
            )
        ):
            lower = np.maximum(lower, predicted - acceleration)
            upper = np.minimum(upper, predicted + acceleration)
        invalid = lower >= upper
        if np.any(invalid):
            lower[invalid] = previous[invalid] - 1e-9
            upper[invalid] = previous[invalid] + 1e-9
        seeds = (previous, predicted)
        candidates: list[tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]] = []
        for seed_index, seed in enumerate(seeds):
            reference_terms = (
                (float(settings["previous_q_weight"]), previous),
                (float(settings["predicted_q_weight"]), predicted),
                (float(settings["input_q_weight"]), input_q),
                (float(settings["natural_q_weight"]), self.nominal),
            )
            solution = least_squares(
                lambda value: self._task_residual(
                    value,
                    target_position,
                    target_rotation,
                    include_orientation,
                    reference_terms,
                    settings,
                ),
                np.clip(seed, lower, upper),
                bounds=(lower, upper),
                max_nfev=int(settings["maximum_position_iterations_per_frame"]),
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            candidate, collision_report = self._clear_contacts(
                solution.x,
                target_position,
                target_rotation,
                include_orientation,
                left_hand,
                right_hand,
                input_q,
                previous,
                predicted,
                lower,
                upper,
                settings,
            )
            position_error, orientation_error = self._errors(
                candidate, target_position, target_rotation, include_orientation
            )
            records = self.adapter._records(candidate, left_hand, right_hand)
            step_norm = float(np.linalg.norm(candidate - previous))
            key = (
                0 if float(np.max(position_error)) <= self.position_tolerance else 1,
                0
                if (not include_orientation or float(np.max(orientation_error)) <= self.orientation_tolerance)
                else 1,
                0 if not records else 1,
                len(records),
                float(np.max(position_error)),
                float(np.max(orientation_error)),
                step_norm,
                float(np.linalg.norm(candidate - input_q)),
                float(np.linalg.norm(candidate - self.nominal)),
            )
            candidates.append(
                (
                    key,
                    candidate,
                    {
                        "seed_index": seed_index,
                        "step_norm_rad": step_norm,
                        "collision": collision_report,
                    },
                )
            )
        candidates.sort(key=lambda item: item[0])
        selected_key, selected_q, selected_report = candidates[0]
        selected_records = self.adapter._records(
            selected_q, left_hand, right_hand
        )
        selected_position, selected_orientation = self._errors(
            selected_q,
            target_position,
            target_rotation,
            include_orientation,
        )
        selected_step = float(np.linalg.norm(selected_q - previous))
        requires_global_seed = bool(
            selected_records
            or float(np.max(selected_position)) > self.position_tolerance
            or (
                include_orientation
                and float(np.max(selected_orientation)) > self.orientation_tolerance
            )
            or selected_step > float(settings["maximum_step_norm_rad"]) + 1e-9
        )
        if requires_global_seed and bool(
            settings.get("unbounded_local_continuation_enabled", False)
        ):
            # First continue the already selected collision-free branch without
            # an artificial one-frame trust region.  This is a local redundant
            # IK continuation, not a new target or a method-specific rescue.
            fallback, continuation_report = self._clear_contacts(
                previous,
                target_position,
                target_rotation,
                include_orientation,
                left_hand,
                right_hand,
                previous,
                None,
                None,
                self.lower,
                self.upper,
                self.config["anchor_search"],
            )
            continuation_position, continuation_orientation = self._errors(
                fallback,
                target_position,
                target_rotation,
                include_orientation,
            )
            continuation_records = self.adapter._records(
                fallback, left_hand, right_hand
            )
            continuation_key = (
                0
                if float(np.max(continuation_position)) <= self.position_tolerance
                else 1,
                0
                if (
                    not include_orientation
                    or float(np.max(continuation_orientation))
                    <= self.orientation_tolerance
                )
                else 1,
                0 if not continuation_records else 1,
                len(continuation_records),
                float(np.max(continuation_position)),
                float(np.max(continuation_orientation)),
                float(np.linalg.norm(fallback - previous)),
                float(np.linalg.norm(fallback - input_q)),
                float(np.linalg.norm(fallback - self.nominal)),
            )
            fallback_report: dict[str, Any] = {
                "local_continuation": continuation_report
            }
            if (
                continuation_key[0]
                or continuation_key[1]
                or continuation_key[2]
            ) and bool(settings.get("deterministic_global_fallback_enabled", False)):
                fallback, global_report = self._anchor_solution(
                    previous,
                    target_position,
                    target_rotation,
                    include_orientation,
                    left_hand,
                    right_hand,
                )
                fallback_report["deterministic_global_search"] = global_report
            fallback_position, fallback_orientation = self._errors(
                fallback,
                target_position,
                target_rotation,
                include_orientation,
            )
            fallback_records = self.adapter._records(
                fallback, left_hand, right_hand
            )
            fallback_step = float(np.linalg.norm(fallback - previous))
            fallback_key = (
                0 if float(np.max(fallback_position)) <= self.position_tolerance else 1,
                0
                if (
                    not include_orientation
                    or float(np.max(fallback_orientation)) <= self.orientation_tolerance
                )
                else 1,
                0 if not fallback_records else 1,
                len(fallback_records),
                float(np.max(fallback_position)),
                float(np.max(fallback_orientation)),
                fallback_step,
                float(np.linalg.norm(fallback - input_q)),
                float(np.linalg.norm(fallback - self.nominal)),
            )
            if fallback_key < selected_key:
                return fallback, {
                    "seed_index": "deterministic_global_redundancy_fallback",
                    "step_norm_rad": fallback_step,
                    "collision": {
                        "contact_count": len(fallback_records),
                        "collision_free": not fallback_records,
                    },
                    "anchor_fallback": fallback_report,
                }
        return selected_q, selected_report

    def solve(
        self,
        incoming_targets: Mapping[str, np.ndarray],
        input_q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        fps: float,
        include_orientation: bool,
    ) -> CollisionAwareIKResult:
        """Return a temporally propagated collision-free realization.

        Incoming target arrays are copied for calculation and never changed.
        """
        target_position = {
            side: np.asarray(
                incoming_targets[f"{side}_wrist_position"], dtype=np.float64
            ).copy()
            for side in SIDES
        }
        target_rotation = {
            side: np.asarray(
                incoming_targets[f"{side}_wrist_rotation"], dtype=np.float64
            ).copy()
            for side in SIDES
        }
        q_input = np.asarray(input_q, dtype=np.float64)
        left_hand = np.asarray(left_hand, dtype=np.float64)
        right_hand = np.asarray(right_hand, dtype=np.float64)
        count = len(q_input)
        if q_input.shape != (count, 14):
            raise ValueError("common IK input must be T x 14")
        before = self.adapter._collision_metrics(q_input, left_hand, right_hand, fps)
        hard_frames = list(map(int, before["hard_frames"]))
        if not hard_frames:
            return CollisionAwareIKResult(
                q=q_input.copy(),
                metadata={
                    "schema_version": "common_collision_aware_redundancy_ik_result_v1",
                    "method_blind": True,
                    "incoming_target_modified": False,
                    "include_orientation": bool(include_orientation),
                    "anchor_frame": None,
                    "anchor": {"status": "NOT_REQUIRED"},
                    "before_hard_collision_frames": 0,
                    "after_hard_collision_frames": 0,
                    "before_contact_frames": int(before["contact_frame_count"]),
                    "after_contact_frames": int(before["contact_frame_count"]),
                    "frame_reports": [],
                },
            )
        def frame_targets(frame: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            return (
                {side: target_position[side][frame] for side in SIDES},
                {side: target_rotation[side][frame] for side in SIDES},
            )
        anchor = max(
            hard_frames,
            key=lambda frame: max(
                (
                    float(record["penetration_depth_m"])
                    for record in before["by_frame"].get(frame, [])
                ),
                default=0.0,
            ),
        )
        anchor_position, anchor_rotation = frame_targets(anchor)
        anchor_q, anchor_report = self._anchor_solution(
            q_input[anchor],
            anchor_position,
            anchor_rotation,
            include_orientation,
            left_hand[anchor],
            right_hand[anchor],
        )
        q = np.empty_like(q_input)
        q[anchor] = anchor_q
        event_reports: list[dict[str, Any]] = []
        for direction in (-1, 1):
            previous = anchor_q.copy()
            previous2 = previous.copy()
            frames = (
                range(anchor - 1, -1, -1)
                if direction < 0
                else range(anchor + 1, count)
            )
            for frame in frames:
                current_position, current_rotation = frame_targets(frame)
                value, report = self._frame_solution(
                    q_input[frame],
                    previous,
                    previous2,
                    current_position,
                    current_rotation,
                    include_orientation,
                    left_hand[frame],
                    right_hand[frame],
                )
                q[frame] = value
                if report["collision"]["contact_count"] or frame in hard_frames:
                    event_reports.append(
                        {"frame": int(frame), "direction": direction, **report}
                    )
                previous2, previous = previous, value.copy()
        q = q.astype(np.float32).astype(np.float64)
        after = self.adapter._collision_metrics(q, left_hand, right_hand, fps)
        positions, rotations = self.adapter._pose_arrays(q)
        position_error = np.maximum(
            *[
                np.linalg.norm(positions[side] - target_position[side], axis=1)
                for side in SIDES
            ]
        )
        orientation_error = np.maximum(
            *[
                np.asarray(
                    [
                        np.linalg.norm(
                            self._rotation_error(
                                rotations[side][frame], target_rotation[side][frame]
                            )
                        )
                        for frame in range(count)
                    ],
                    dtype=np.float64,
                )
                for side in SIDES
            ]
        ) if include_orientation else np.zeros(count, dtype=np.float64)
        return CollisionAwareIKResult(
            q=q,
            metadata={
                "schema_version": "common_collision_aware_redundancy_ik_result_v1",
                "method_blind": True,
                "representation_mode_consumed": False,
                "episode_index_consumed": False,
                "object_pose_consumed": False,
                "task_or_physical_outcome_consumed": False,
                "incoming_target_modified": False,
                "include_orientation": bool(include_orientation),
                "anchor_frame": int(anchor) if anchor is not None else None,
                "anchor": anchor_report,
                "before_hard_collision_frames": int(before["hard_collision_frame_count"]),
                "after_hard_collision_frames": int(after["hard_collision_frame_count"]),
                "before_contact_frames": int(before["contact_frame_count"]),
                "after_contact_frames": int(after["contact_frame_count"]),
                "position_residual_m": {
                    "mean": float(np.mean(position_error)),
                    "p95": float(np.quantile(position_error, 0.95)),
                    "max": float(np.max(position_error)),
                },
                "orientation_residual_rad": {
                    "mean": float(np.mean(orientation_error)),
                    "p95": float(np.quantile(orientation_error, 0.95)),
                    "max": float(np.max(orientation_error)),
                },
                "event_reports": event_reports,
            },
        )
