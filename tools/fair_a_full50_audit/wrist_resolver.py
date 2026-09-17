"""Representation-neutral wrist adapter for the frozen common B resolver.

Only the realization frame is adapted.  Every numerical routine, parameter,
joint limit, collision body, temporal gate, signed-distance repair, and nearest
target projection is inherited from GenericG1FeasibilityResolver unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import tools.doll_handoff_feasibility.solver as frozen_solver_module
from tools.doll_handoff_feasibility.common import (
    rotation_error_rad,
    sha256_file,
    write_json,
)
from tools.doll_handoff_feasibility.solver import (
    DEFAULT_CONFIG,
    EpisodeResult,
    GenericG1FeasibilityResolver,
)

from .common import (
    OUTPUT_ROOT,
    SIDES,
    a_trajectory_path,
    array_sha256,
    atomic_npz,
    load_a_trajectory,
    load_json,
    sha256_file as local_sha256_file,
    stable_episode_id,
)


IMMUTABLE_SOURCE_KEYS = (
    "target_left_wrist_position_model",
    "target_right_wrist_position_model",
    "target_left_wrist_rotation_model",
    "target_right_wrist_rotation_model",
    "target_left_interaction_frame_position_world",
    "target_right_interaction_frame_position_world",
    "target_left_interaction_frame_position_task",
    "target_right_interaction_frame_position_task",
    "ownership_state",
    "left_hand_phase",
    "right_hand_phase",
    "event_names",
    "event_frames",
)


def bind_fair_a_inputs() -> None:
    """Bind the frozen solver's input hooks to Fair A in this audit process."""
    frozen_solver_module.load_trajectory = load_a_trajectory
    frozen_solver_module.stable_episode_id = stable_episode_id


class RepresentationNeutralWristResolver(GenericG1FeasibilityResolver):
    """The frozen generic realization algorithm operating at a wrist frame."""

    realization_frame = "G1_WRIST_ORIGIN"

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        output_root: str | Path = OUTPUT_ROOT,
    ):
        bind_fair_a_inputs()
        super().__init__(config_path=config_path, output_root=output_root)
        identity = np.eye(4, dtype=np.float64)
        self.transforms = {side: identity.copy() for side in SIDES}
        reach = self.g1.shoulder_wrist_reach_geometry()
        self.outer_tool_radius = {
            side: float(reach["sides"][side]["upper_effective_length_m"])
            + float(reach["sides"][side]["forearm_effective_length_m"])
            for side in SIDES
        }
        self.adapter_path = Path(__file__).resolve()
        self.adapter_sha256 = local_sha256_file(self.adapter_path)

    def _source_targets(
        self, values: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        positions = {
            side: self.g1.model_to_world_position(
                np.asarray(
                    values[f"target_{side}_wrist_position_model"],
                    dtype=np.float64,
                )
            )
            for side in SIDES
        }
        rotations = {
            side: np.asarray(
                values[f"target_{side}_wrist_rotation_model"],
                dtype=np.float64,
            ).copy()
            for side in SIDES
        }
        return positions, rotations

    def _pose(
        self, q: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        state = self.g1.wrist_state(q)
        return (
            {
                side: np.asarray(
                    state[f"{side}_position"], dtype=np.float64
                ).copy()
                for side in SIDES
            },
            {
                side: np.asarray(
                    state[f"{side}_rotation"], dtype=np.float64
                ).copy()
                for side in SIDES
            },
        )

    def _cache_paths(self, episode: int) -> tuple[Path, Path]:
        stem = stable_episode_id(episode)
        return (
            self.output_root / "after/trajectories" / f"{stem}.npz",
            self.output_root / "after/metrics" / f"{stem}.solver.json",
        )

    def load_exported_episode(self, episode: int) -> EpisodeResult:
        frozen = load_a_trajectory(episode)
        trajectory_path, metric_path = self._cache_paths(episode)
        if not trajectory_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(trajectory_path)
        with np.load(trajectory_path, allow_pickle=False) as payload:
            values = {name: np.asarray(payload[name]) for name in payload.files}
        if str(np.asarray(values["feasibility_config_sha256"]).item()) != sha256_file(
            self.config_path
        ):
            raise RuntimeError(f"cached ep{episode:03d} resolver config mismatch")
        if str(np.asarray(values["realization_frame_adapter_sha256"]).item()) != (
            self.adapter_sha256
        ):
            raise RuntimeError(f"cached ep{episode:03d} adapter mismatch")
        if str(np.asarray(values["fair_a_input_trajectory_sha256"]).item()) != (
            local_sha256_file(a_trajectory_path(episode))
        ):
            raise RuntimeError(f"cached ep{episode:03d} input trajectory mismatch")
        checks = {
            key: np.array_equal(np.asarray(frozen[key]), np.asarray(values[key]))
            for key in IMMUTABLE_SOURCE_KEYS
        }
        if not all(checks.values()):
            raise RuntimeError(f"cached ep{episode:03d} mutated source arrays")
        q_after = values["g1_arm_qpos"].astype(np.float64)
        left = frozen["left_dex3_qpos"].astype(np.float64)
        right = frozen["right_dex3_qpos"].astype(np.float64)
        source_world, source_rotation = self._source_targets(frozen)
        geometry = self.g1.trajectory_geometry(
            q_after, left, right, self.collision_tolerance, self.transforms
        )
        return EpisodeResult(
            episode=int(episode),
            q_before=frozen["g1_arm_qpos"].astype(np.float64),
            q_after=q_after,
            left_hand=left,
            right_hand=right,
            source_position_world=source_world,
            source_orientation_model=source_rotation,
            realized_position_world={
                side: values[
                    f"realized_feasible_{side}_realization_frame_position_world"
                ].astype(np.float64)
                for side in SIDES
            },
            realized_orientation_model={
                side: values[
                    f"realized_feasible_{side}_realization_frame_orientation_model"
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
            frozen_values = load_a_trajectory(result.episode)
        values = {
            name: np.asarray(value).copy() for name, value in frozen_values.items()
        }
        if geometry is None:
            geometry = self.g1.trajectory_geometry(
                result.q_after,
                result.left_hand,
                result.right_hand,
                self.collision_tolerance,
                self.transforms,
            )
        for key in IMMUTABLE_SOURCE_KEYS:
            if not np.array_equal(values[key], np.asarray(frozen_values[key])):
                raise RuntimeError(f"immutable source array changed before export: {key}")

        values["g1_arm_qpos"] = result.q_after.astype(np.float32)
        replay_names = values["replay_joint_names"].astype(str).tolist()
        replay = values["replay_named_joint_qpos"].astype(np.float32).copy()
        for index, name in enumerate(self.g1.arm_joint_names.astype(str)):
            replay[:, replay_names.index(name)] = result.q_after[:, index].astype(
                np.float32
            )
        values["replay_named_joint_qpos"] = replay
        task_origin = values["task_frame_origin_world_xyz_m"].astype(np.float64)
        orientation_error = np.empty((len(result.q_after), 2), dtype=np.float64)
        realized_position_error = np.empty_like(orientation_error)
        orientation_tolerance = float(
            self.common["shared_temporal_ik"]["orientation_tolerance_rad"]
        )
        for side_index, side in enumerate(SIDES):
            values[f"achieved_{side}_wrist_position_world"] = geometry[
                f"{side}_wrist_position_world"
            ].astype(np.float32)
            values[f"achieved_{side}_physical_grasp_frame_position_world"] = geometry[
                f"{side}_grasp_position_world"
            ].astype(np.float32)
            values[f"achieved_{side}_physical_grasp_frame_position_task"] = (
                geometry[f"{side}_grasp_position_world"] - task_origin
            ).astype(np.float32)
            values[f"source_{side}_realization_frame_position_world"] = (
                result.source_position_world[side].astype(np.float32)
            )
            values[f"source_{side}_realization_frame_orientation_model"] = (
                result.source_orientation_model[side].astype(np.float32)
            )
            values[f"realized_feasible_{side}_realization_frame_position_world"] = (
                result.realized_position_world[side].astype(np.float32)
            )
            values[f"realized_feasible_{side}_realization_frame_orientation_model"] = (
                result.realized_orientation_model[side].astype(np.float32)
            )
            values[f"achieved_{side}_realization_frame_position_world"] = (
                result.achieved_position_world[side].astype(np.float32)
            )
            values[f"achieved_{side}_realization_frame_orientation_model"] = (
                result.achieved_orientation_model[side].astype(np.float32)
            )
            realized_position_error[:, side_index] = np.linalg.norm(
                result.achieved_position_world[side]
                - result.realized_position_world[side],
                axis=1,
            )
            orientation_error[:, side_index] = [
                rotation_error_rad(
                    result.achieved_orientation_model[side][frame],
                    result.realized_orientation_model[side][frame],
                )
                for frame in range(len(result.q_after))
            ]
        values["inter_grasp_frame_distance_m"] = np.linalg.norm(
            geometry["right_grasp_position_world"]
            - geometry["left_grasp_position_world"],
            axis=1,
        ).astype(np.float32)
        values["feasibility_projection_translation_m"] = (
            result.projection_translation_m.astype(np.float32)
        )
        values["feasibility_projection_orientation_rad"] = (
            result.projection_orientation_rad.astype(np.float32)
        )
        values["feasibility_projection_reason"] = result.projection_reason
        values["ik_success_per_frame"] = np.all(
            (realized_position_error <= self.strict_tolerance)
            & (orientation_error <= orientation_tolerance),
            axis=1,
        )
        values["feasibility_resolver"] = np.asarray(
            "FROZEN_GENERIC_G1_RESOLVER_WITH_REPRESENTATION_FRAME_ADAPTER"
        )
        values["feasibility_config_sha256"] = np.asarray(
            sha256_file(self.config_path)
        )
        values["realization_frame"] = np.asarray(self.realization_frame)
        values["realization_frame_adapter_sha256"] = np.asarray(
            self.adapter_sha256
        )
        values["fair_a_input_trajectory_sha256"] = np.asarray(
            local_sha256_file(a_trajectory_path(result.episode))
        )
        values["source_targets_modified"] = np.asarray(False)
        values["source_targets_modified_semantically"] = np.asarray(False)
        values["ownership_timing_modified"] = np.asarray(False)
        values["interaction_frame_used_as_solver_target"] = np.asarray(False)
        values["episode_specific_correction"] = np.asarray(False)
        values["phase_specific_correction"] = np.asarray(False)

        result.metadata.update(
            {
                "schema_version": "fair_a_common_feasibility_episode_audit_v1",
                "stable_episode_id": stable_episode_id(result.episode),
                "realization_frame": self.realization_frame,
                "realization_frame_adapter": str(self.adapter_path),
                "realization_frame_adapter_sha256": self.adapter_sha256,
                "frozen_generic_solver_class": (
                    "GenericG1FeasibilityResolver"
                ),
                "frozen_generic_solver_parameters_changed": False,
                "source_targets_modified_semantically": False,
                "interaction_frame_used_as_solver_target": False,
                "ownership_used_by_solver": False,
                "bimanual_semantic_target_used_by_solver": False,
                "episode_specific_correction": False,
                "phase_specific_correction": False,
                "input_trajectory": str(a_trajectory_path(result.episode)),
                "input_trajectory_sha256": local_sha256_file(
                    a_trajectory_path(result.episode)
                ),
                "immutable_source_array_sha256": {
                    key: array_sha256(np.asarray(frozen_values[key]))
                    for key in IMMUTABLE_SOURCE_KEYS
                },
                "realized_full_6d_success_rate": float(
                    np.mean(values["ik_success_per_frame"])
                ),
                "realized_orientation_residual_rad": {
                    "mean": float(np.mean(orientation_error)),
                    "p95": float(np.percentile(orientation_error, 95)),
                    "max": float(np.max(orientation_error)),
                    "hard_frame_count": int(
                        np.count_nonzero(
                            np.any(
                                orientation_error > orientation_tolerance,
                                axis=1,
                            )
                        )
                    ),
                },
            }
        )
        trajectory_path, metric_path = self._cache_paths(result.episode)
        atomic_npz(trajectory_path, values)
        write_json(metric_path, result.metadata)
        return trajectory_path, metric_path


__all__ = [
    "IMMUTABLE_SOURCE_KEYS",
    "RepresentationNeutralWristResolver",
    "bind_fair_a_inputs",
]
