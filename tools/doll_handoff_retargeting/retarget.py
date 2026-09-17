"""A/B representations, common hand adapter, shared temporal IK, and validation."""
from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .common import SIDES, branch_flags, rotation_errors, scalar_stats
from .events import EpisodeEvents
from .models import G1Kinematics
from .source import SourceEpisode


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(target) @ np.asarray(current).T).as_rotvec()


def _mean_rotation(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot average an empty rotation list")
    return Rotation.from_matrix(np.stack(values)).mean().as_matrix()


def derive_orientation_alignment(
    fk_by_episode: Mapping[int, Mapping[str, Any]],
    g1: G1Kinematics,
    primitives: Mapping[str, Any],
) -> dict[str, Any]:
    """One pooled static source-axis alignment per side and representation."""
    desired_wrist = np.eye(3, dtype=np.float64)
    result: dict[str, Any] = {
        "derivation": (
            "pooled mean of all 50 source first-frame model orientations aligned to "
            "one canonical G1 task-ready wrist orientation; no episode/frame/phase offsets"
        ),
        "canonical_g1_wrist_rotation_model": desired_wrist,
        "sides": {},
    }
    for side in SIDES:
        source_wrist_model_rotations = [
            g1.world_to_model_rotation(np.asarray(fk[f"{side}_wrist_rotation_world"])[0])
            for _, fk in sorted(fk_by_episode.items())
        ]
        source_interaction_model_rotations = [
            g1.world_to_model_rotation(np.asarray(fk[f"{side}_tcp_rotation_world"])[0])
            for _, fk in sorted(fk_by_episode.items())
        ]
        source_wrist_reference = _mean_rotation(source_wrist_model_rotations)
        source_interaction_reference = _mean_rotation(
            source_interaction_model_rotations
        )
        tool = np.asarray(
            primitives["wrist_to_grasp_frame"][side], dtype=np.float64
        )
        baseline_alignment = source_wrist_reference.T @ desired_wrist
        proposed_alignment = source_interaction_reference.T @ (
            desired_wrist @ tool[:3, :3]
        )
        result["sides"][side] = {
            "source_wrist_first_frame_pooled_mean_rotation_in_g1_model": (
                source_wrist_reference
            ),
            "source_interaction_first_frame_pooled_mean_rotation_in_g1_model": (
                source_interaction_reference
            ),
            "source_wrist_to_g1_wrist_axis_alignment": baseline_alignment,
            "source_interaction_to_g1_grasp_axis_alignment": proposed_alignment,
            "wrist_to_grasp_frame": tool,
            "a0_frame_convention_bug_repaired": (
                "A0 pooled the source TCP frame and applied it to wrist rotations; "
                "fair A pools and applies the source wrist frame consistently"
            ),
            "static": True,
            "episode_specific": False,
            "frame_specific": False,
            "phase_specific": False,
        }
    return result


def derive_baseline_workspace_mapping(
    common: Mapping[str, Any],
    scene: Mapping[str, Any],
    fk_by_episode: Mapping[int, Mapping[str, Any]],
    aloha_reach: Mapping[str, Any],
    aloha_orientation_capacity: Mapping[str, Any],
    g1: G1Kinematics,
    nominal_q: np.ndarray,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one morphology-only wrist workspace mapping for fair Baseline A.

    All episodes contribute only to a pooled neutral source wrist anchor.  Scale
    comes from robot morphology: bilateral neutral wrist separation on task X,
    and the active-model arm-chain reach ratio on task Y/Z.  A common radial
    outer-reach projection prevents singular overextension.  No object location,
    gripper event, handoff phase, or target-task success is consulted.
    """
    registration = common["task_registration"]
    task_origin = np.asarray(scene["task_frame"]["origin_world_xyz_m"], dtype=np.float64)
    scale = float(registration["uniform_metric_scale"])
    translation = np.asarray(
        registration["global_translation_correction_world_xyz_m"], dtype=np.float64
    )
    quaternion = np.asarray(
        registration["global_rotation_correction_world_wxyz"], dtype=np.float64
    )
    registration_rotation = Rotation.from_quat(
        quaternion[[1, 2, 3, 0]]
    ).as_matrix()

    def register(values: np.ndarray) -> np.ndarray:
        relative = scale * (np.asarray(values, dtype=np.float64) - task_origin)
        return relative @ registration_rotation.T + task_origin + translation

    first = {
        side: np.stack(
            [
                register(np.asarray(fk[f"{side}_wrist_position_world"])[0])
                for _, fk in sorted(fk_by_episode.items())
            ]
        )
        for side in SIDES
    }
    source_side_anchor = {side: np.mean(value, axis=0) for side, value in first.items()}
    source_anchor = 0.5 * (
        source_side_anchor["left"] + source_side_anchor["right"]
    )
    source_separation = source_side_anchor["right"] - source_side_anchor["left"]

    nominal = g1.wrist_state(np.asarray(nominal_q, dtype=np.float64))
    target_side_anchor = {
        side: g1.model_to_world_position(nominal[f"{side}_position"])
        for side in SIDES
    }
    target_anchor = 0.5 * (
        target_side_anchor["left"] + target_side_anchor["right"]
    )
    target_separation = target_side_anchor["right"] - target_side_anchor["left"]
    if abs(float(source_separation[0])) <= 1e-9:
        raise RuntimeError("pooled source wrist separation has no task-X component")
    lateral_scale = abs(float(target_separation[0] / source_separation[0]))

    source_reach = float(aloha_reach["common_effective_reach_m"])
    g1_reach_report = g1.shoulder_wrist_reach_geometry()
    g1_reaches = [
        float(value["upper_effective_length_m"])
        + float(value["forearm_effective_length_m"])
        for value in g1_reach_report["sides"].values()
    ]
    g1_reach = min(g1_reaches)
    reach_scale = g1_reach / source_reach
    axis_scale = np.asarray(
        [lateral_scale, reach_scale, reach_scale], dtype=np.float64
    )
    outer_fraction = float(
        baseline["workspace_mapping_policy"]["outer_reach_fraction"]
    )
    if not 0.0 < outer_fraction <= 1.0:
        raise ValueError("baseline outer_reach_fraction must be in (0,1]")
    outer_reach = outer_fraction * g1_reach
    source_orientation_capacity = float(
        aloha_orientation_capacity["common_capacity_l2_rad"]
    )
    target_orientation_capacity_report = g1.wrist_orientation_capacity()
    target_orientation_capacity = float(
        target_orientation_capacity_report["common_capacity_l2_rad"]
    )
    orientation_span_ratios = {
        side: np.asarray(
            target_orientation_capacity_report["sides"][side][
                "joint_range_spans_rad"
            ],
            dtype=np.float64,
        )
        / np.asarray(
            aloha_orientation_capacity["sides"][side][
                "joint_range_spans_rad"
            ],
            dtype=np.float64,
        )
        for side in SIDES
    }
    orientation_deviation_scale = min(
        1.0,
        min(float(np.min(value)) for value in orientation_span_ratios.values()),
    )

    pooled = {
        side: np.concatenate(
            [
                register(np.asarray(fk[f"{side}_wrist_position_world"]))
                for _, fk in sorted(fk_by_episode.items())
            ],
            axis=0,
        )
        for side in SIDES
    }
    shoulders = g1.fixed_shoulder_anchors_model()

    def map_positions(side: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mapped_world = target_anchor + (
            np.asarray(values, dtype=np.float64) - source_anchor
        ) * axis_scale
        mapped_model = g1.world_to_model_position(mapped_world)
        delta = mapped_model - shoulders[side]
        radii = np.linalg.norm(delta, axis=1)
        clipped = radii > outer_reach
        if np.any(clipped):
            mapped_model[clipped] = shoulders[side] + (
                delta[clipped]
                * (outer_reach / radii[clipped])[:, None]
            )
        return mapped_model, clipped

    naive_radii: dict[str, np.ndarray] = {}
    mapped_radii: dict[str, np.ndarray] = {}
    clip_counts: dict[str, int] = {}
    for side in SIDES:
        naive_model = g1.world_to_model_position(pooled[side])
        naive_radii[side] = np.linalg.norm(
            naive_model - shoulders[side], axis=1
        )
        mapped_model, clipped = map_positions(side, pooled[side])
        mapped_radii[side] = np.linalg.norm(
            mapped_model - shoulders[side], axis=1
        )
        clip_counts[side] = int(np.count_nonzero(clipped))

    def stats(values: np.ndarray) -> dict[str, Any]:
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }

    return {
        "schema_version": "fair_baseline_global_workspace_mapping_v1",
        "method": (
            "one task-axis affine mapping followed by one common shoulder-radial "
            "outer-reach projection"
        ),
        "derivation": (
            "pooled all-50 first-frame ALOHA wrist midpoint -> active-model G1 "
            "task-ready wrist midpoint; task-X scale from pooled bilateral neutral "
            "wrist separation; task-Y/Z scale from active-model arm-chain reach ratio"
        ),
        "source_episode_count": len(fk_by_episode),
        "source_anchor_world_m": source_anchor,
        "source_side_anchor_world_m": source_side_anchor,
        "target_anchor_world_m": target_anchor,
        "target_side_anchor_world_m": target_side_anchor,
        "source_neutral_separation_world_m": source_separation,
        "target_neutral_separation_world_m": target_separation,
        "task_axis_scale_xyz": axis_scale,
        "source_effective_reach_m": source_reach,
        "target_effective_reach_m": g1_reach,
        "outer_reach_fraction": outer_fraction,
        "outer_reach_limit_m": outer_reach,
        "source_wrist_orientation_capacity_l2_rad": source_orientation_capacity,
        "target_wrist_orientation_capacity_l2_rad": target_orientation_capacity,
        "wrist_joint_range_span_ratios": orientation_span_ratios,
        "l2_orientation_capacity_ratio_diagnostic": min(
            1.0, target_orientation_capacity / source_orientation_capacity
        ),
        "orientation_deviation_scale": orientation_deviation_scale,
        "orientation_scale_derivation": (
            "minimum active-model G1/ALOHA ratio across the last-three wrist-joint "
            "range spans; applied as one isotropic scalar to pooled-neutral-relative "
            "SO(3) rotation vectors so rotation direction is preserved"
        ),
        "fixed_g1_shoulder_anchors_model_m": shoulders,
        "naive_a0_shoulder_radius_m": {
            side: stats(value) for side, value in naive_radii.items()
        },
        "mapped_shoulder_radius_m": {
            side: stats(value) for side, value in mapped_radii.items()
        },
        "global_clip_counts": clip_counts,
        "global_clip_fractions": {
            side: clip_counts[side] / len(pooled[side]) for side in SIDES
        },
        "orientation_mapping": (
            "one pooled static source-wrist to G1-wrist axis alignment per side; "
            "the A0 TCP/wrist frame mismatch is removed"
        ),
        "task_frame_registration_changed": False,
        "uniform_metric_task_scale": scale,
        "uses_scene_object_coordinates": False,
        "uses_events_or_ownership": False,
        "uses_episode_specific_parameters": False,
        "uses_phase_specific_parameters": False,
        "uses_interaction_or_grasp_frame": False,
    }


class RepresentationBuilder:
    """Form fair A/B wrist targets under one metric task-frame registration."""

    def __init__(
        self,
        common: Mapping[str, Any],
        scene: Mapping[str, Any],
        g1: G1Kinematics,
        alignment: Mapping[str, Any],
        proposed: Mapping[str, Any],
        baseline_workspace_mapping: Mapping[str, Any],
    ):
        self.common = common
        self.scene = scene
        self.g1 = g1
        self.alignment = alignment
        self.proposed = proposed
        self.baseline_workspace_mapping = baseline_workspace_mapping
        registration = common["task_registration"]
        self.scale = float(registration["uniform_metric_scale"])
        self.task_origin = np.asarray(
            scene["task_frame"]["origin_world_xyz_m"], dtype=np.float64
        )
        self.translation = np.asarray(
            registration["global_translation_correction_world_xyz_m"],
            dtype=np.float64,
        )
        quat = np.asarray(
            registration["global_rotation_correction_world_wxyz"], dtype=np.float64
        )
        self.rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        if self.scale != 1.0:
            raise RuntimeError("this audit's explicit model/unit check requires metric scale 1.0")

        self.shoulder_anchors = self.g1.fixed_shoulder_anchors_model()
        self.palm_twist_rad = {
            side: math.radians(
                float(
                    proposed["orientation_semantics"]["static_palm_twist_deg"][side]
                )
            )
            for side in SIDES
        }

    def _registered_position(
        self,
        world: np.ndarray,
        registration_entry: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        world = np.asarray(world, dtype=np.float64)
        if registration_entry is not None:
            target_from_source = np.asarray(
                registration_entry.get(
                    "pre_workspace_source_to_target_transform_matrix",
                    registration_entry["source_to_target_transform_matrix"],
                ),
                dtype=np.float64,
            )
            if target_from_source.shape != (4, 4):
                raise ValueError("episode source-to-target transform must be 4x4")
            return world @ target_from_source[:3, :3].T + target_from_source[:3, 3]
        relative = self.scale * (world - self.task_origin)
        return relative @ self.rotation.T + self.task_origin + self.translation

    def _registered_rotation(
        self,
        world: np.ndarray,
        registration_entry: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        if registration_entry is not None:
            target_from_source = np.asarray(
                registration_entry.get(
                    "pre_workspace_source_to_target_transform_matrix",
                    registration_entry["source_to_target_transform_matrix"],
                ),
                dtype=np.float64,
            )
            return np.einsum(
                "ij,tjk->tik",
                target_from_source[:3, :3],
                np.asarray(world, dtype=np.float64),
            )
        return np.einsum("ij,tjk->tik", self.rotation, np.asarray(world, dtype=np.float64))

    def _baseline_wrist_position(
        self, side: str, registered_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mapping = self.baseline_workspace_mapping
        source_anchor = np.asarray(mapping["source_anchor_world_m"], dtype=np.float64)
        target_anchor = np.asarray(mapping["target_anchor_world_m"], dtype=np.float64)
        axis_scale = np.asarray(mapping["task_axis_scale_xyz"], dtype=np.float64)
        mapped_world = target_anchor + (
            np.asarray(registered_world, dtype=np.float64) - source_anchor
        ) * axis_scale
        mapped_model = self.g1.world_to_model_position(mapped_world)
        shoulder = np.asarray(
            mapping["fixed_g1_shoulder_anchors_model_m"][side], dtype=np.float64
        )
        delta = mapped_model - shoulder
        radius = np.linalg.norm(delta, axis=1)
        limit = float(mapping["outer_reach_limit_m"])
        clipped = radius > limit
        if np.any(clipped):
            mapped_model[clipped] = shoulder + (
                delta[clipped] * (limit / radius[clipped])[:, None]
            )
        return mapped_model, clipped

    def _embodiment_wrist_orientation_gauge(
        self, side: str, grasp_position_model: np.ndarray
    ) -> np.ndarray:
        """Resolve spherical-object orientation from fixed G1 morphology only."""
        position = np.asarray(grasp_position_model, dtype=np.float64)
        approach = position - self.shoulder_anchors[side][None, :]
        lengths = np.linalg.norm(approach, axis=1, keepdims=True)
        if np.any(lengths <= 1e-9):
            raise RuntimeError("grasp target coincides with a G1 shoulder anchor")
        approach = approach / lengths
        model_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        up = model_up[None, :] - approach * (approach @ model_up)[:, None]
        up_lengths = np.linalg.norm(up, axis=1, keepdims=True)
        if np.any(up_lengths <= 1e-9):
            raise RuntimeError("shoulder-to-grasp ray is parallel to model up")
        up = up / up_lengths
        lateral = np.cross(up, approach)
        lateral = lateral / np.linalg.norm(lateral, axis=1, keepdims=True)
        up = np.cross(approach, lateral)
        base = np.stack((approach, lateral, up), axis=2)
        twist = Rotation.from_rotvec(
            np.asarray([self.palm_twist_rad[side], 0.0, 0.0], dtype=np.float64)
        ).as_matrix()
        return np.einsum("tij,jk->tik", base, twist)

    def proposed_semantics_report(self) -> dict[str, Any]:
        return {
            "method_name": self.proposed["method_name"],
            "orientation_policy": self.proposed["orientation_semantics"]["policy"],
            "orientation_weight_by_ownership": self.proposed[
                "orientation_semantics"
            ]["weight_multiplier_relative_to_common_by_ownership"],
            "static_palm_twist_deg": {
                side: math.degrees(self.palm_twist_rad[side]) for side in SIDES
            },
            "fixed_shoulder_anchors_model_m": self.shoulder_anchors,
            "spherical_object_orientation_gauge": True,
            "handoff_cartesian_offset_m": 0.0,
            "removed_handoff_only_offset_m": 0.0495531958180063,
            "removed_offset_reason": (
                "twice a contact-pad tangential half-extent is collision clearance, "
                "not a rigid wrist-to-pinch transform"
            ),
            "handoff_scope": (
                "source-derived ownership semantics change objective weights and "
                "whole-hand synergy states only"
            ),
            "episode_specific_parameters": False,
            "frame_specific_offsets": False,
            "scene_object_waypoints": False,
        }

    def build(
        self,
        representation_mode: str,
        fk: Mapping[str, Any],
        event: EpisodeEvents | None = None,
        registration_entry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if representation_mode not in {"WRIST", "INTERACTION"}:
            raise ValueError(representation_mode)
        if representation_mode == "INTERACTION" and event is None:
            raise ValueError("INTERACTION requires the common source-derived event audit")
        target: dict[str, Any] = {
            "representation_mode": representation_mode,
            "metric_scale": self.scale,
            "episode_registration_bound": registration_entry is not None,
        }
        if registration_entry is not None:
            target["episode_registration_entry_sha256"] = str(
                registration_entry["entry_sha256"]
            )
            target["registered_object_pose"] = dict(
                registration_entry["target_object_pose"]
            )
            target["registered_bin_pose"] = dict(
                registration_entry["target_bin_pose"]
            )
            target["registered_table_task_origin_xyz_m"] = np.asarray(
                registration_entry["target_table_task_origin_xyz_m"],
                dtype=np.float64,
            )
            target["registered_task_rotation_world"] = np.asarray(
                registration_entry["source_to_target_transform_matrix"],
                dtype=np.float64,
            )[:3, :3]

        tool_model: dict[str, np.ndarray] = {}
        tool_rotation_model: dict[str, np.ndarray] = {}
        for side in SIDES:
            source_world = self._registered_position(
                np.asarray(fk[f"{side}_tcp_position_world"]),
                registration_entry,
            )
            world_rotation = self._registered_rotation(
                np.asarray(fk[f"{side}_tcp_rotation_world"]),
                registration_entry,
            )
            tool_model[side] = self.g1.world_to_model_position(source_world)
            source_rotation_model = self.g1.world_to_model_rotation(world_rotation)
            proposed_alignment = np.asarray(
                self.alignment["sides"][side][
                    "source_interaction_to_g1_grasp_axis_alignment"
                ]
            )
            tool_rotation_model[side] = np.einsum(
                "tij,jk->tik", source_rotation_model, proposed_alignment
            )
            target[f"source_{side}_tool_position_world"] = source_world
            target[f"source_{side}_tool_position_model"] = tool_model[side].copy()

        midpoint = 0.5 * (tool_model["left"] + tool_model["right"])
        source_relative = tool_model["right"] - tool_model["left"]
        relative = source_relative.copy()
        constrained_frame_count = 0

        for side in SIDES:
            world = self.g1.model_to_world_position(tool_model[side])
            target[f"{side}_tool_position_world"] = world
            task_origin = np.asarray(
                target.get("registered_table_task_origin_xyz_m", self.task_origin),
                dtype=np.float64,
            )
            task_rotation = np.asarray(
                target.get("registered_task_rotation_world", np.eye(3)),
                dtype=np.float64,
            )
            target[f"{side}_tool_position_task"] = (
                world - task_origin
            ) @ task_rotation
            target[f"{side}_tool_position_model"] = tool_model[side]
            target[f"{side}_tool_rotation_model"] = tool_rotation_model[side]

        target["target_tool_midpoint_model"] = midpoint
        target["target_tool_relative_model"] = relative
        target["source_tool_relative_model"] = source_relative
        target["target_inter_hand_distance_m"] = np.linalg.norm(relative, axis=1)
        # Everything below the representation block consumes the same pair of
        # wrist SE(3) targets.  These fields are deliberately common: the IK
        # implementation must not branch on representation identity or accept
        # method-specific residuals/weights.
        target["static_wrist_to_tool"] = {
            side: np.asarray(
                self.alignment["sides"][side]["wrist_to_grasp_frame"],
                dtype=np.float64,
            )
            for side in SIDES
        }

        if representation_mode == "WRIST":
            for side in SIDES:
                # A is a direct registered TCP/wrist trajectory baseline.  The
                # source TCP is the only point on the ALOHA tool whose task
                # meaning survives a change of hand morphology.  Convert that
                # pose to the G1 wrist origin with one fixed target-tool
                # transform.  This is coordinate compatibility, not a grasp
                # correction: it never reads an object pose, ownership state,
                # policy output, or physical outcome.
                local = np.asarray(
                    self.alignment["sides"][side]["wrist_to_grasp_frame"],
                    dtype=np.float64,
                )
                wrist_rotation = np.einsum(
                    "tij,jk->tik", tool_rotation_model[side], local[:3, :3].T
                )
                wrist_position = tool_model[side] - np.einsum(
                    "tij,j->ti", wrist_rotation, local[:3, 3]
                )
                target[f"{side}_wrist_position"] = wrist_position
                target[f"{side}_wrist_rotation"] = wrist_rotation
            target["representation"] = "trajectory-centric independent wrist-level 6D"
            target["baseline_fixed_tool_compatibility_transform"] = {
                side: np.asarray(
                    self.alignment["sides"][side]["wrist_to_grasp_frame"]
                ).copy()
                for side in SIDES
            }
            target["baseline_source_pose"] = "REGISTERED_ALOHA_TASK_TCP_SE3"
            target["baseline_object_pose_used"] = False
            target["baseline_ownership_state_used"] = False
            target["baseline_global_reach_clip_counts"] = {
                side: 0 for side in SIDES
            }
            target["representation_specific_solver_residuals"] = 0
        else:
            assert event is not None
            for side in SIDES:
                local = np.asarray(
                    self.alignment["sides"][side]["wrist_to_grasp_frame"]
                )
                wrist_rotation = self._embodiment_wrist_orientation_gauge(
                    side, tool_model[side]
                )
                wrist_position = tool_model[side] - np.einsum(
                    "tij,j->ti", wrist_rotation, local[:3, 3]
                )
                target[f"{side}_wrist_position"] = wrist_position
                target[f"{side}_wrist_rotation"] = wrist_rotation
                target[f"{side}_tool_rotation_model"] = np.einsum(
                    "tij,jk->tik", wrist_rotation, local[:3, :3]
                )
            target["representation"] = (
                "interaction-centric source grasp frame to active-model-derived "
                "three-finger whole-hand grasp frame"
            )
            # The bilateral interaction relation is encoded in the two wrist
            # pose targets themselves.  It is not an extra method-specific IK
            # residual, weight schedule, or solver mode.
            coordination = self.proposed["bimanual_interaction"]
            target["representation_specific_solver_residuals"] = 0
            target["ownership_state"] = event.ownership_labels.copy()
            target["handoff_constrained_frame_count"] = 0
            target["handoff_cartesian_offset_m"] = 0.0
            target["bimanual_relative_vector_policy"] = coordination[
                "relative_vector_policy"
            ]
            target["bimanual_absolute_spacing_policy"] = coordination[
                "absolute_spacing_policy"
            ]

        # Workspace placement is deliberately downstream of the sole A/B
        # representation switch.  Applying one rigid transform here moves the
        # completed target frames without re-solving B's interaction geometry
        # against a new shoulder bearing (which would not be a rigid task-frame
        # registration).  A and B therefore receive byte-identical R,t and
        # retain their pre-workspace wrist/tool relations exactly.
        workspace = (
            np.asarray(
                registration_entry["common_workspace_transform_matrix"],
                dtype=np.float64,
            )
            if registration_entry is not None
            and "common_workspace_transform_matrix" in registration_entry
            else None
        )
        if workspace is not None:
            if workspace.shape != (4, 4):
                raise ValueError("common workspace transform must be 4x4")
            workspace_rotation = workspace[:3, :3]
            workspace_translation = workspace[:3, 3]

            def move_model_positions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                base_world = self.g1.model_to_world_position(values)
                moved_world = base_world @ workspace_rotation.T + workspace_translation
                return self.g1.world_to_model_position(moved_world), moved_world

            def move_model_rotations(values: np.ndarray) -> np.ndarray:
                base_world = self.g1.model_to_world_rotation(values)
                moved_world = np.einsum(
                    "ij,tjk->tik", workspace_rotation, base_world
                )
                return self.g1.world_to_model_rotation(moved_world)

            task_origin = np.asarray(
                target["registered_table_task_origin_xyz_m"], dtype=np.float64
            )
            task_rotation = np.asarray(
                target["registered_task_rotation_world"], dtype=np.float64
            )
            for side in SIDES:
                moved_tool_model, moved_tool_world = move_model_positions(
                    np.asarray(target[f"{side}_tool_position_model"])
                )
                moved_tool_rotation = move_model_rotations(
                    np.asarray(target[f"{side}_tool_rotation_model"])
                )
                moved_wrist_model, _ = move_model_positions(
                    np.asarray(target[f"{side}_wrist_position"])
                )
                moved_wrist_rotation = move_model_rotations(
                    np.asarray(target[f"{side}_wrist_rotation"])
                )
                tool_model[side] = moved_tool_model
                tool_rotation_model[side] = moved_tool_rotation
                target[f"source_{side}_tool_position_world"] = moved_tool_world
                target[f"source_{side}_tool_position_model"] = moved_tool_model.copy()
                target[f"{side}_tool_position_world"] = moved_tool_world
                target[f"{side}_tool_position_task"] = (
                    moved_tool_world - task_origin
                ) @ task_rotation
                target[f"{side}_tool_position_model"] = moved_tool_model
                target[f"{side}_tool_rotation_model"] = moved_tool_rotation
                target[f"{side}_wrist_position"] = moved_wrist_model
                target[f"{side}_wrist_rotation"] = moved_wrist_rotation

            midpoint = 0.5 * (tool_model["left"] + tool_model["right"])
            source_relative = tool_model["right"] - tool_model["left"]
            relative = source_relative.copy()
            target["target_tool_midpoint_model"] = midpoint
            target["target_tool_relative_model"] = relative
            target["source_tool_relative_model"] = source_relative
            target["target_inter_hand_distance_m"] = np.linalg.norm(relative, axis=1)
            target["common_workspace_transform_matrix"] = workspace.copy()
            target["common_workspace_transform_applied_post_representation"] = True

        target["explicit_grasp_frame_objective"] = False
        target["direct_static_grasp_frame_position_objective"] = False
        target["task_orientation_constrained"] = True
        target["orientation_gauge_weight_multiplier"] = 1.0
        target["explicit_bimanual_objective"] = False
        target["explicit_bimanual_target_constraint"] = False
        target["bimanual_coordination_weights"] = {
            "midpoint": 0.0,
            "relative_motion": 0.0,
        }

        target["bimanual_reconstruction_max_error_m"] = float(
            max(
                np.max(
                    np.abs(
                        0.5 * (tool_model["left"] + tool_model["right"])
                        - midpoint
                    )
                ),
                np.max(
                    np.abs(
                        (tool_model["right"] - tool_model["left"]) - relative
                    )
                ),
            )
        )
        fingerprint = hashlib.sha256()
        for name in (
            "left_wrist_position",
            "right_wrist_position",
            "left_wrist_rotation",
            "right_wrist_rotation",
            "left_tool_position_model",
            "right_tool_position_model",
            "left_tool_rotation_model",
            "right_tool_rotation_model",
            "target_tool_midpoint_model",
            "target_tool_relative_model",
        ):
            value = np.ascontiguousarray(np.asarray(target[name], dtype=np.float64))
            fingerprint.update(name.encode("utf-8"))
            fingerprint.update(b"\0")
            fingerprint.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            fingerprint.update(value.tobytes())
        target["cartesian_target_sha256"] = fingerprint.hexdigest()
        target["finite"] = bool(
            all(
                np.isfinite(value).all()
                for value in target.values()
                if isinstance(value, np.ndarray)
                and np.issubdtype(value.dtype, np.number)
            )
        )
        return target


class HandMapper:
    """One source-timeline-to-Dex3 mapping shared by every representation."""

    def __init__(
        self,
        common: Mapping[str, Any],
        g1: G1Kinematics,
        primitives: Mapping[str, Any],
    ):
        self.common = common
        self.g1 = g1
        self.primitives = primitives
        maximum_delta = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(primitives["states"][side]["GRASP"])
                        - np.asarray(primitives["states"][side]["OPEN"])
                    )
                )
            )
            for side in SIDES
        )
        validation = common["validation"]
        fps = float(common["source_channels"]["expected_fps"])
        by_step = math.ceil(1.5 * maximum_delta / float(validation["maximum_joint_step_rad"]))
        by_velocity = math.ceil(
            1.5 * maximum_delta * fps / float(validation["maximum_velocity_rad_s"])
        )
        by_acceleration = math.ceil(
            math.sqrt(
                6.0
                * maximum_delta
                * fps**2
                / float(validation["maximum_acceleration_rad_s2"])
            )
        )
        self.transition_frames = max(2, by_step, by_velocity, by_acceleration)

    @staticmethod
    def _smooth_commands(
        labels: np.ndarray,
        states: Mapping[str, np.ndarray],
        transition_frames: int,
    ) -> np.ndarray:
        labels = np.asarray(labels).astype(str)
        width = len(next(iter(states.values())))
        output = np.empty((len(labels), width), dtype=np.float64)
        active = str(labels[0])
        current = np.asarray(states[active], dtype=np.float64).copy()
        start = current.copy()
        age = transition_frames
        output[0] = current
        for frame in range(1, len(labels)):
            label = str(labels[frame])
            if label != active:
                active = label
                start = current.copy()
                age = 0
            age += 1
            u = min(1.0, age / max(transition_frames, 1))
            blend = u * u * (3.0 - 2.0 * u)
            current = start + blend * (np.asarray(states[active]) - start)
            output[frame] = current
        return output

    def map_common(self, event: EpisodeEvents) -> dict[str, Any]:
        """Map the one common source timeline to identical Dex3 commands.

        Spatial representation is deliberately absent from this interface.  A
        and B must receive byte-identical finger supervision for a matched
        source episode.
        """
        labels: dict[str, np.ndarray] = {}
        commands: dict[str, np.ndarray] = {}
        for side in SIDES:
            states = self.primitives["states"][side]
            labels[side] = event.semantic_labels[side].copy()
            mapping = {name: np.asarray(value) for name, value in states.items()}
            commands[side] = self._smooth_commands(
                labels[side], mapping, self.transition_frames
            )
        violations = 0
        for side in SIDES:
            limits = self.g1.hand_limits[side]
            violations += int(
                np.count_nonzero(
                    (commands[side] < limits[:, 0] - 1e-9)
                    | (commands[side] > limits[:, 1] + 1e-9)
                )
            )
        return {
            "left": commands["left"],
            "right": commands["right"],
            "left_phase": labels["left"],
            "right_phase": labels["right"],
            "transition_frames": self.transition_frames,
            "joint_limit_violation_count": violations,
            "calibration_status": self.primitives["status"],
        }

    def map(self, method: str, event: EpisodeEvents) -> dict[str, Any]:
        """Compatibility entry point; ``method`` cannot affect hand commands."""
        if method not in {"baseline", "interaction_frame", "interaction_bimanual", "proposed"}:
            raise ValueError(method)
        return self.map_common(event)


class SharedTemporalIK:
    """Common temporal DLS with task-priority natural-arm null-space posture."""

    def __init__(
        self,
        common: Mapping[str, Any],
        g1: G1Kinematics,
        nominal_q: np.ndarray,
        natural_arm_enabled: bool | None = None,
    ):
        self.config = common["shared_temporal_ik"]
        self.natural_config = common["natural_arm_redundancy"]
        self.g1 = g1
        self.nominal = np.asarray(nominal_q, dtype=np.float64)
        self.stand = g1.stand_qpos[g1.arm_qpos_ids].copy()
        self.natural_arm_enabled = (
            bool(self.natural_config["enabled"])
            if natural_arm_enabled is None
            else bool(natural_arm_enabled)
        )
        nominal_landmarks = self.g1.arm_landmarks(self.nominal)
        self.nominal_elbow_guides = {
            side: nominal_landmarks[side]["elbow"]
            - nominal_landmarks[side]["shoulder_pitch"]
            for side in SIDES
        }
        self.nominal_manipulability = {
            side: float(
                self.g1.manipulability_state(self.nominal)[side][
                    "minimum_singular_value"
                ]
            )
            for side in SIDES
        }
        nominal_clearance = self.g1.posture_clearance_state(self.nominal)
        self.clearance_targets = {
            "TORSO": float(nominal_clearance["TORSO"]["minimum_distance_m"])
            * float(
                self.natural_config[
                    "torso_clearance_reference_fraction_of_nominal"
                ]
            ),
            "CROSS_ARM": float(
                nominal_clearance["CROSS_ARM"]["minimum_distance_m"]
            )
            * float(
                self.natural_config[
                    "cross_arm_clearance_reference_fraction_of_nominal"
                ]
            ),
        }
        self.natural_reference_report = {
            "enabled": self.natural_arm_enabled,
            "scope": self.natural_config["scope"],
            "formulation": self.natural_config["formulation"],
            "nominal_q": self.nominal,
            "nominal_elbow_guides_model": self.nominal_elbow_guides,
            "preferred_sew_angle_rad": {side: 0.0 for side in SIDES},
            "nominal_manipulability_min_singular_value": self.nominal_manipulability,
            "manipulability_metric": self.g1.manipulability_state(self.nominal)[
                "metric"
            ],
            "nominal_clearance": nominal_clearance,
            "clearance_targets_m": self.clearance_targets,
            "episode_specific_parameters": False,
            "phase_specific_parameters": False,
            "task_specific_elbow_target": False,
        }

    @staticmethod
    def _wrapped_angle(value: float) -> float:
        return float(math.atan2(math.sin(value), math.cos(value)))

    @staticmethod
    def _normalized_direction(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 1e-12 else np.zeros_like(value)

    def _natural_features(
        self,
        q: np.ndarray,
        selected_pairs: Mapping[str, tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        sew = self.g1.sew_angles(q, self.nominal_elbow_guides)
        manipulability = self.g1.manipulability_state(q)
        clearance = self.g1.posture_clearance_state(q, selected_pairs)
        return {
            "sew": sew,
            "manipulability": {
                side: float(manipulability[side]["minimum_singular_value"])
                for side in SIDES
            },
            "clearance": {
                category: float(clearance[category]["minimum_distance_m"])
                for category in ("TORSO", "CROSS_ARM")
            },
            "clearance_pairs": {
                category: tuple(clearance[category]["closest_geom_pair"])
                for category in ("TORSO", "CROSS_ARM")
            },
        }

    def _natural_direction(
        self, q: np.ndarray, previous: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """One global robot-level posture direction; no task or phase inputs."""
        q = np.asarray(q, dtype=np.float64)
        previous_sew = self.g1.sew_angles(previous, self.nominal_elbow_guides)
        base = self._natural_features(q)
        selected_pairs = base["clearance_pairs"]
        eps = float(self.natural_config["finite_difference_step_rad"])
        sew_jacobian = {side: np.zeros(14, dtype=np.float64) for side in SIDES}
        manip_gradient = {side: np.zeros(14, dtype=np.float64) for side in SIDES}
        clearance_gradient = {
            category: np.zeros(14, dtype=np.float64)
            for category in ("TORSO", "CROSS_ARM")
        }
        for joint in range(14):
            plus = q.copy()
            minus = q.copy()
            plus[joint] = min(plus[joint] + eps, self.g1.arm_limits[joint, 1])
            minus[joint] = max(minus[joint] - eps, self.g1.arm_limits[joint, 0])
            width = float(plus[joint] - minus[joint])
            if width <= 1e-12:
                continue
            high = self._natural_features(plus, selected_pairs)
            low = self._natural_features(minus, selected_pairs)
            for side in SIDES:
                sew_jacobian[side][joint] = self._wrapped_angle(
                    high["sew"][side] - low["sew"][side]
                ) / width
                manip_gradient[side][joint] = (
                    high["manipulability"][side]
                    - low["manipulability"][side]
                ) / width
            for category in clearance_gradient:
                clearance_gradient[category][joint] = (
                    high["clearance"][category] - low["clearance"][category]
                ) / width

        sew_preference = np.zeros(14, dtype=np.float64)
        sew_continuity = np.zeros(14, dtype=np.float64)
        for side in SIDES:
            sew_preference -= self._wrapped_angle(base["sew"][side]) * sew_jacobian[side]
            delta = self._wrapped_angle(base["sew"][side] - previous_sew[side])
            sew_continuity -= delta * sew_jacobian[side]

        span = np.maximum(self.g1.arm_limits[:, 1] - self.g1.arm_limits[:, 0], 1e-6)
        half_span = 0.5 * span
        center = 0.5 * (self.g1.arm_limits[:, 0] + self.g1.arm_limits[:, 1])
        nominal = (self.nominal - q) / span
        joint_centering = (center - q) / np.square(half_span)

        manipulability = np.zeros(14, dtype=np.float64)
        for side in SIDES:
            threshold = float(
                self.natural_config[
                    "manipulability_reference_fraction_of_nominal"
                ]
            ) * self.nominal_manipulability[side]
            deficit = max(0.0, threshold - base["manipulability"][side])
            if deficit > 0.0:
                manipulability += (
                    deficit / max(threshold * threshold, 1e-12)
                ) * manip_gradient[side]

        clearance = np.zeros(14, dtype=np.float64)
        clearance_deficits: dict[str, float] = {}
        for category, threshold in self.clearance_targets.items():
            deficit = max(0.0, threshold - base["clearance"][category])
            clearance_deficits[category] = deficit
            if deficit > 0.0:
                clearance += (
                    deficit / max(threshold * threshold, 1e-12)
                ) * clearance_gradient[category]

        components = {
            "sew_preference": sew_preference,
            "sew_continuity": sew_continuity,
            "nominal_posture": nominal,
            "joint_centering": joint_centering,
            "manipulability": manipulability,
            "collision_clearance": clearance,
        }
        weights = self.natural_config["component_step_weights"]
        direction = sum(
            float(weights[name]) * self._normalized_direction(value)
            for name, value in components.items()
        )
        audit = {
            "features": base,
            "previous_sew": previous_sew,
            "clearance_deficits_m": clearance_deficits,
            "component_direction_norms": {
                name: float(np.linalg.norm(value)) for name, value in components.items()
            },
            "combined_direction_norm": float(np.linalg.norm(direction)),
        }
        return direction, audit

    def _system(
        self,
        q: np.ndarray,
        targets: Mapping[str, np.ndarray],
        frame: int,
        previous: np.ndarray,
        previous2: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[float, float, float, float],
        np.ndarray,
    ]:
        state = self.g1.wrist_state(q)
        wp = float(self.config["position_weight"])
        wo = float(self.config["orientation_weight"])
        rows: list[np.ndarray] = []
        primary_rows: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        residuals: list[float] = []
        direct_tool = bool(
            targets.get("direct_static_grasp_frame_position_objective", False)
        )
        orientation_constrained = bool(
            targets.get("task_orientation_constrained", True)
        )
        orientation_weight_value = targets.get(
            "orientation_gauge_weight_multiplier",
            1.0 if orientation_constrained else 0.0,
        )
        orientation_gauge_multiplier = float(
            np.asarray(orientation_weight_value)[frame]
            if np.asarray(orientation_weight_value).ndim
            else orientation_weight_value
        )
        for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
            jacobian = np.zeros((6, 14), dtype=np.float64)
            if direct_tool:
                current_position, current_rotation, jacp, jacr = (
                    self.g1.static_tool_pose_state(
                        side, np.asarray(targets["static_wrist_to_tool"][side])
                    )
                )
                jacobian[:3, block] = jacp
                jacobian[3:, block] = jacr
                position = (
                    np.asarray(targets[f"{side}_tool_position_model"])[frame]
                    - current_position
                )
                orientation = _rotation_error(
                    current_rotation,
                    np.asarray(targets[f"{side}_tool_rotation_model"])[frame],
                )
            else:
                jacobian[:, block] = state[f"{side}_jacobian"]
                position = (
                    np.asarray(targets[f"{side}_wrist_position"])[frame]
                    - state[f"{side}_position"]
                )
                orientation = _rotation_error(
                    state[f"{side}_rotation"],
                    np.asarray(targets[f"{side}_wrist_rotation"])[frame],
                )
            rows.append(wp * jacobian[:3])
            primary_rows.append(wp * jacobian[:3])
            errors.append(wp * position)
            if orientation_gauge_multiplier > 0.0:
                rows.append(wo * orientation_gauge_multiplier * jacobian[3:])
                primary_rows.append(
                    wo * orientation_gauge_multiplier * jacobian[3:]
                )
                errors.append(wo * orientation_gauge_multiplier * orientation)
            residuals.extend(
                (
                    float(np.linalg.norm(position)),
                    float(np.linalg.norm(orientation))
                    if orientation_constrained
                    else 0.0,
                )
            )
        if bool(targets.get("explicit_bimanual_objective", False)):
            point_positions: dict[str, np.ndarray] = {}
            point_jacobians: dict[str, np.ndarray] = {}
            for side, block in (("left", slice(0, 7)), ("right", slice(7, 14))):
                position, local_jacobian = self.g1.static_tool_point_state(
                    side, np.asarray(targets["static_wrist_to_tool"][side])
                )
                jacobian = np.zeros((3, 14), dtype=np.float64)
                jacobian[:, block] = local_jacobian
                point_positions[side] = position
                point_jacobians[side] = jacobian
            current_midpoint = 0.5 * (
                point_positions["left"] + point_positions["right"]
            )
            current_relative = point_positions["right"] - point_positions["left"]
            midpoint_jacobian = 0.5 * (
                point_jacobians["left"] + point_jacobians["right"]
            )
            relative_jacobian = point_jacobians["right"] - point_jacobians["left"]
            weights = targets["bimanual_coordination_weights"]
            midpoint_weight = wp * float(np.asarray(weights["midpoint"])[frame])
            relative_weight = wp * float(
                np.asarray(weights["relative_motion"])[frame]
            )
            if midpoint_weight > 0.0:
                rows.append(midpoint_weight * midpoint_jacobian)
                primary_rows.append(midpoint_weight * midpoint_jacobian)
                errors.append(
                    midpoint_weight
                    * (
                        np.asarray(targets["target_tool_midpoint_model"])[frame]
                        - current_midpoint
                    )
                )
            if relative_weight > 0.0:
                rows.append(relative_weight * relative_jacobian)
                primary_rows.append(relative_weight * relative_jacobian)
                errors.append(
                    relative_weight
                    * (
                        np.asarray(targets["target_tool_relative_model"])[frame]
                        - current_relative
                    )
                )
        identity = np.eye(14, dtype=np.float64)
        wv = float(self.config["velocity_regularization_weight"])
        wa = float(self.config["acceleration_regularization_weight"])
        wn = float(self.config["nominal_regularization_weight"])
        rows.extend((wv * identity, wa * identity, wn * identity))
        errors.extend(
            (
                wv * (previous - q),
                wa * (2.0 * previous - previous2 - q),
                wn * (self.nominal - q),
            )
        )
        return (
            np.vstack(rows),
            np.concatenate(errors),
            tuple(residuals),  # type: ignore[arg-type]
            np.vstack(primary_rows),
        )

    def _key(self, residuals: tuple[float, float, float, float], q: np.ndarray) -> tuple[float, ...]:
        pos = max(residuals[0], residuals[2])
        ori = max(residuals[1], residuals[3])
        ptol = float(self.config["position_tolerance_m"])
        otol = float(self.config["orientation_tolerance_rad"])
        accepted = pos <= ptol and ori <= otol
        clearance_violation = 0.0
        if accepted and self.natural_arm_enabled:
            clearance = self.g1.posture_clearance_state(q)
            clearance_violation = float(
                any(
                    float(clearance[category]["minimum_distance_m"]) < 0.0
                    for category in ("TORSO", "CROSS_ARM")
                )
            )
        return (
            0.0 if accepted else 1.0,
            clearance_violation,
            max(pos / ptol, ori / otol),
            pos,
            ori,
            float(np.linalg.norm(q - self.nominal)),
        )

    def _project_frame_trust_region(
        self,
        candidate: np.ndarray,
        previous: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        frame_step: float | None,
    ) -> np.ndarray:
        """Apply the common natural-arm temporal trust region.

        This constraint acts only on the redundant joint realization.  It has no
        episode, task phase, semantic state, or Cartesian target input.
        """
        value = np.clip(np.asarray(candidate, dtype=np.float64), lower, upper)
        if not self.natural_arm_enabled or frame_step is None:
            return value
        limit = float(self.natural_config["maximum_frame_step_norm_rad"])
        delta = value - np.asarray(previous, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > limit:
            value = np.asarray(previous, dtype=np.float64) + delta * (limit / norm)
        return np.clip(value, lower, upper)

    def _solve_seed(
        self,
        targets: Mapping[str, np.ndarray],
        frame: int,
        seed: np.ndarray,
        previous: np.ndarray,
        previous2: np.ndarray,
        iterations: int,
        frame_step: float | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        lower = self.g1.arm_limits[:, 0].copy()
        upper = self.g1.arm_limits[:, 1].copy()
        if frame_step is not None:
            lower = np.maximum(lower, previous - frame_step)
            upper = np.minimum(upper, previous + frame_step)
        q = self._project_frame_trust_region(
            np.asarray(seed, dtype=np.float64), previous, lower, upper, frame_step
        )
        natural_direction = np.zeros(14, dtype=np.float64)
        natural_audit: dict[str, Any] = {"enabled": False}
        if self.natural_arm_enabled:
            natural_direction, natural_audit = self._natural_direction(q, previous)
            natural_audit["enabled"] = True
        best = q.copy()
        best_key = (math.inf,) * 6
        best_residuals = (math.inf,) * 4
        damping = float(self.config["damping"])
        maximum_update = float(self.config["max_joint_update_rad"])
        fallback = False
        maximum_nullspace_update_norm = 0.0
        used = 0
        for used in range(1, iterations + 1):
            jacobian, error, residuals, primary_jacobian = self._system(
                q, targets, frame, previous, previous2
            )
            key = self._key(residuals, q)
            if key < best_key:
                best_key, best, best_residuals = key, q.copy(), residuals
            if key[0] == 0.0 and key[3] <= min(
                0.001, float(self.config["position_tolerance_m"])
            ):
                break
            normal = jacobian.T @ jacobian + damping**2 * np.eye(14)
            right = jacobian.T @ error
            try:
                update = np.linalg.solve(normal, right)
            except np.linalg.LinAlgError:
                fallback = True
                update = np.linalg.lstsq(normal, right, rcond=None)[0]
            if (
                self.natural_arm_enabled
                and used == 1
                and np.any(natural_direction)
            ):
                null_damping = float(self.natural_config["nullspace_damping"])
                task_gram = (
                    primary_jacobian @ primary_jacobian.T
                    + null_damping**2
                    * np.eye(primary_jacobian.shape[0], dtype=np.float64)
                )
                try:
                    primary_pinv = primary_jacobian.T @ np.linalg.solve(
                        task_gram,
                        np.eye(task_gram.shape[0], dtype=np.float64),
                    )
                except np.linalg.LinAlgError:
                    primary_pinv = np.linalg.pinv(primary_jacobian)
                projector = np.eye(14, dtype=np.float64) - primary_pinv @ primary_jacobian
                null_update = projector @ natural_direction
                limit = float(
                    self.natural_config["maximum_nullspace_joint_update_rad"]
                )
                null_update = np.clip(null_update, -limit, limit)
                base_candidate = self._project_frame_trust_region(
                    q + np.clip(update, -maximum_update, maximum_update),
                    previous,
                    lower,
                    upper,
                    frame_step,
                )
                base_clearance = self.g1.posture_clearance_state(base_candidate)
                accepted_scale = 1.0
                while accepted_scale >= 1.0 / 64.0:
                    candidate = self._project_frame_trust_region(
                        q
                        + np.clip(
                            update + accepted_scale * null_update,
                            -maximum_update,
                            maximum_update,
                        ),
                        previous,
                        lower,
                        upper,
                        frame_step,
                    )
                    candidate_clearance = self.g1.posture_clearance_state(candidate)
                    safe = True
                    for category, threshold in self.clearance_targets.items():
                        base_distance = float(
                            base_clearance[category]["minimum_distance_m"]
                        )
                        candidate_distance = float(
                            candidate_clearance[category]["minimum_distance_m"]
                        )
                        required = (
                            threshold
                            if base_distance >= threshold
                            else base_distance - 1e-6
                        )
                        safe &= candidate_distance >= required
                    if safe:
                        break
                    accepted_scale *= 0.5
                if accepted_scale < 1.0 / 64.0:
                    accepted_scale = 0.0
                null_update *= accepted_scale
                natural_audit["clearance_line_search_scale"] = accepted_scale
                maximum_nullspace_update_norm = max(
                    maximum_nullspace_update_norm,
                    float(np.linalg.norm(null_update)),
                )
                update += null_update
            q = self._project_frame_trust_region(
                q + np.clip(update, -maximum_update, maximum_update),
                previous,
                lower,
                upper,
                frame_step,
            )
        _, _, residuals, _ = self._system(q, targets, frame, previous, previous2)
        key = self._key(residuals, q)
        if key < best_key:
            best_key, best, best_residuals = key, q.copy(), residuals
        return best, {
            "iterations": used,
            "accepted": bool(best_key[0] == 0.0),
            "position_error_max_m": float(max(best_residuals[0], best_residuals[2])),
            "orientation_error_max_rad": float(max(best_residuals[1], best_residuals[3])),
            "numerical_lstsq_fallback": fallback,
            "natural_arm": natural_audit,
            "maximum_nullspace_update_norm_rad": maximum_nullspace_update_norm,
        }

    def solve(self, targets: Mapping[str, np.ndarray]) -> dict[str, Any]:
        start_time = time.monotonic()
        count = len(np.asarray(targets["left_wrist_position"]))
        raw = np.empty((count, 14), dtype=np.float64)
        initial_meta: list[dict[str, Any]] = []
        previous = self.nominal.copy()
        previous2 = previous.copy()
        for frame in range(count):
            iterations = int(
                self.config["max_iterations_initial_frame"]
                if frame == 0
                else self.config["max_iterations_per_frame"]
            )
            frame_step = None if frame == 0 else float(self.config["max_frame_joint_step_rad"])
            seeds = (self.nominal, self.stand) if frame == 0 else (previous,)
            candidates = [
                self._solve_seed(
                    targets,
                    frame,
                    seed,
                    previous,
                    previous2,
                    iterations,
                    frame_step,
                )
                for seed in seeds
            ]
            selected = min(
                candidates,
                key=lambda item: (
                    0 if item[1]["accepted"] else 1,
                    item[1]["position_error_max_m"]
                    / float(self.config["position_tolerance_m"])
                    + item[1]["orientation_error_max_rad"]
                    / float(self.config["orientation_tolerance_rad"]),
                ),
            )
            raw[frame] = selected[0]
            initial_meta.append(selected[1])
            previous2, previous = previous, raw[frame].copy()

        window = min(
            int(self.config["temporal_smoothing_window"]),
            count if count % 2 else count - 1,
        )
        polyorder = int(self.config["temporal_smoothing_polyorder"])
        smoothed = (
            savgol_filter(raw, window, polyorder, axis=0, mode="interp")
            if window >= max(polyorder + 2, 3)
            else raw.copy()
        )
        smoothed = np.clip(
            smoothed, self.g1.arm_limits[:, 0], self.g1.arm_limits[:, 1]
        )
        final = np.empty_like(raw)
        reprojection_meta: list[dict[str, Any]] = []
        previous = smoothed[0].copy()
        previous2 = previous.copy()
        for frame in range(count):
            value, meta = self._solve_seed(
                targets,
                frame,
                smoothed[frame],
                previous,
                previous2,
                int(self.config["max_iterations_reprojection"]),
                None
                if frame == 0
                else float(self.config["reprojection_max_frame_joint_step_rad"]),
            )
            final[frame] = value
            reprojection_meta.append(meta)
            previous2, previous = previous, value.copy()
        return {
            "q": final,
            "raw_q": raw,
            "smoothed_q": smoothed,
            "initial_meta": initial_meta,
            "reprojection_meta": reprojection_meta,
            "elapsed_sec": float(time.monotonic() - start_time),
            "backend": self.config["backend"],
            "natural_arm_enabled": self.natural_arm_enabled,
            "natural_arm_reference": self.natural_reference_report,
        }


@dataclass
class ConversionResult:
    method: str
    episode: SourceEpisode
    fk: dict[str, Any]
    events: EpisodeEvents
    targets: dict[str, Any]
    hands: dict[str, Any]
    solver: dict[str, Any]
    geometry: dict[str, Any]
    metrics: dict[str, Any]
    validation: dict[str, Any]


def validate_result(
    common: Mapping[str, Any],
    scene: Mapping[str, Any],
    g1: G1Kinematics,
    method: str,
    episode: SourceEpisode,
    fk: Mapping[str, Any],
    events: EpisodeEvents,
    targets: Mapping[str, Any],
    hands: Mapping[str, Any],
    solver: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm = np.asarray(solver["q"])
    count = len(arm)
    fps = episode.fps
    ik_cfg = common["shared_temporal_ik"]
    validation_cfg = common["validation"]
    wrist_position_error: dict[str, np.ndarray] = {}
    wrist_orientation_error: dict[str, np.ndarray] = {}
    grasp_position_error: dict[str, np.ndarray] = {}
    grasp_orientation_error: dict[str, np.ndarray] = {}
    static_tool_position_error: dict[str, np.ndarray] = {}
    static_tool_orientation_error: dict[str, np.ndarray] = {}
    direct_grasp = bool(
        targets.get("direct_static_grasp_frame_position_objective", False)
    )
    orientation_constrained = bool(
        targets.get("task_orientation_constrained", True)
    )
    for side in SIDES:
        wrist_position_error[side] = np.linalg.norm(
            np.asarray(geometry[f"{side}_wrist_position_model"])
            - np.asarray(targets[f"{side}_wrist_position"]),
            axis=1,
        )
        wrist_orientation_error[side] = rotation_errors(
            np.asarray(geometry[f"{side}_wrist_rotation_model"]),
            np.asarray(targets[f"{side}_wrist_rotation"]),
        )
        grasp_position_error[side] = np.linalg.norm(
            np.asarray(geometry[f"{side}_grasp_position_model"])
            - np.asarray(targets[f"{side}_tool_position_model"]),
            axis=1,
        )
        grasp_orientation_error[side] = rotation_errors(
            np.asarray(geometry[f"{side}_grasp_rotation_model"]),
            np.asarray(targets[f"{side}_tool_rotation_model"]),
        )
        if direct_grasp:
            static_tool_position_error[side] = np.linalg.norm(
                np.asarray(geometry[f"{side}_static_tool_position_model"])
                - np.asarray(targets[f"{side}_tool_position_model"]),
                axis=1,
            )
            static_tool_orientation_error[side] = rotation_errors(
                np.asarray(geometry[f"{side}_static_tool_rotation_model"]),
                np.asarray(targets[f"{side}_tool_rotation_model"]),
            )
        else:
            static_tool_position_error[side] = grasp_position_error[side]
            static_tool_orientation_error[side] = grasp_orientation_error[side]
    ik_success = np.ones(count, dtype=bool)
    ik_task_position_error = (
        static_tool_position_error if direct_grasp else wrist_position_error
    )
    ik_task_orientation_error = (
        static_tool_orientation_error if direct_grasp else wrist_orientation_error
    )
    for side in SIDES:
        ik_success &= ik_task_position_error[side] <= float(
            ik_cfg["position_tolerance_m"]
        )
        if orientation_constrained:
            ik_success &= ik_task_orientation_error[side] <= float(
                ik_cfg["orientation_tolerance_rad"]
            )
    physically_usable_ik = np.ones(count, dtype=bool)
    for side in SIDES:
        physically_usable_ik &= ik_task_position_error[side] <= float(
            validation_cfg["physically_usable_position_tolerance_m"]
        )
        if orientation_constrained:
            physically_usable_ik &= ik_task_orientation_error[side] <= float(
                validation_cfg["physically_usable_orientation_tolerance_rad"]
            )
    actual_left = np.asarray(
        geometry[
            "left_static_tool_position_model"
            if direct_grasp
            else "left_grasp_position_model"
        ]
    )
    actual_right = np.asarray(
        geometry[
            "right_static_tool_position_model"
            if direct_grasp
            else "right_grasp_position_model"
        ]
    )
    actual_midpoint = 0.5 * (actual_left + actual_right)
    actual_relative = actual_right - actual_left
    target_midpoint = np.asarray(targets["target_tool_midpoint_model"])
    target_relative = np.asarray(targets["target_tool_relative_model"])
    midpoint_error = np.linalg.norm(actual_midpoint - target_midpoint, axis=1)
    relative_error = np.linalg.norm(actual_relative - target_relative, axis=1)
    relative_motion_error = np.linalg.norm(
        (actual_relative - actual_relative[0])
        - (target_relative - target_relative[0]),
        axis=1,
    )
    inter_grasp = np.linalg.norm(actual_relative, axis=1)
    target_inter = np.linalg.norm(target_relative, axis=1)
    relation_error = np.abs(
        (inter_grasp - inter_grasp[0]) - (target_inter - target_inter[0])
    )

    arm_limit_mask = (arm < g1.arm_limits[:, 0] - 1e-9) | (
        arm > g1.arm_limits[:, 1] + 1e-9
    )
    finite = bool(
        all(
            np.isfinite(value).all()
            for value in (
                episode.action,
                episode.state,
                arm,
                hands["left"],
                hands["right"],
                actual_left,
                actual_right,
            )
        )
    )
    nan_inf_count = int(
        sum(
            np.count_nonzero(~np.isfinite(value))
            for value in (arm, hands["left"], hands["right"], actual_left, actual_right)
        )
    )
    step = np.abs(np.diff(arm, axis=0))
    velocity = step * fps
    acceleration = np.abs(np.diff(arm, n=2, axis=0)) * fps**2
    jerk = np.abs(np.diff(arm, n=3, axis=0)) * fps**3
    branches = branch_flags(
        arm,
        float(validation_cfg["branch_absolute_step_norm_rad"]),
        float(validation_cfg["branch_local_multiplier"]),
    )
    collision_flags = geometry["collision_flags"]
    invalid_collision = (
        collision_flags["ARM_TORSO"]
        | collision_flags["CROSS_ARM"]
        | collision_flags["WRIST_OR_PALM_TORSO"]
        | collision_flags["OTHER"]
    )
    distal_hand_contact = collision_flags["DISTAL_HAND_HAND"]

    sew_values = {side: np.empty(count, dtype=np.float64) for side in SIDES}
    manipulability_values = {
        side: np.empty(count, dtype=np.float64) for side in SIDES
    }
    torso_clearance = np.empty(count, dtype=np.float64)
    cross_arm_clearance = np.empty(count, dtype=np.float64)
    nominal_deviation = np.linalg.norm(
        arm
        - np.asarray(
            solver["natural_arm_reference"]["nominal_q"], dtype=np.float64
        ),
        axis=1,
    )
    nominal_guides = solver["natural_arm_reference"][
        "nominal_elbow_guides_model"
    ]
    for frame, q_value in enumerate(arm):
        sew = g1.sew_angles(q_value, nominal_guides)
        manip = g1.manipulability_state(q_value)
        clearance = g1.posture_clearance_state(q_value)
        for side in SIDES:
            sew_values[side][frame] = sew[side]
            manipulability_values[side][frame] = manip[side][
                "minimum_singular_value"
            ]
        torso_clearance[frame] = clearance["TORSO"]["minimum_distance_m"]
        cross_arm_clearance[frame] = clearance["CROSS_ARM"]["minimum_distance_m"]
    sew_unwrapped = {side: np.unwrap(values) for side, values in sew_values.items()}
    sew_step = {
        side: np.abs(np.diff(values)) for side, values in sew_unwrapped.items()
    }

    if targets.get("episode_registration_bound"):
        doll_center = np.asarray(
            targets["registered_object_pose"]["position_xyz_m"],
            dtype=np.float64,
        )
    else:
        doll_center = np.asarray(
            [
                *scene["doll"]["center_world_xy_m"],
                float(scene["table"]["surface_height_m"])
                + 0.5 * float(scene["doll"]["diameter_m"]),
            ],
            dtype=np.float64,
        )
    registered_bin = targets.get("registered_bin_pose")
    if registered_bin is not None:
        bin_position = np.asarray(
            registered_bin["position_xyz_m"], dtype=np.float64
        )
        bin_rotation = Rotation.from_quat(
            np.asarray(registered_bin["quaternion_xyzw"], dtype=np.float64)
        ).as_matrix()
        bin_center_xy = bin_position[:2]
    else:
        bin_position = np.asarray(
            [
                *scene["bin"]["center_world_xy_m"],
                float(scene["table"]["surface_height_m"]),
            ],
            dtype=np.float64,
        )
        bin_rotation = np.eye(3, dtype=np.float64)
        bin_center_xy = bin_position[:2]
    opening = np.asarray(scene["bin"]["opening_dimensions_xy_m"], dtype=np.float64)
    rim_z = float(bin_position[2]) + float(scene["bin"]["outer_dimensions_xyz_m"][2])
    left_grasp = events.frames["LEFT_GRASP"]
    right_grasp = events.frames["RIGHT_GRASP"]
    left_release = events.frames["LEFT_RELEASE"]
    final_release = events.frames["RIGHT_FINAL_RELEASE"]
    source_left_grasp_distance = (
        float(
            np.linalg.norm(
                np.asarray(fk["left_tcp_position_world"])[int(left_grasp)] - doll_center
            )
        )
        if left_grasp is not None
        else None
    )
    source_left_grasp_position = (
        np.asarray(fk["left_tcp_position_world"])[int(left_grasp)]
        if left_grasp is not None
        else np.full(3, np.nan)
    )
    source_release = (
        np.asarray(fk["right_tcp_position_world"])[int(final_release)]
        if final_release is not None
        else np.full(3, np.nan)
    )
    target_left_grasp_distance = (
        float(
            np.linalg.norm(
                np.asarray(geometry["left_grasp_position_world"])[int(left_grasp)]
                - doll_center
            )
        )
        if left_grasp is not None
        else None
    )
    target_left_grasp_position = (
        np.asarray(geometry["left_grasp_position_world"])[int(left_grasp)]
        if left_grasp is not None
        else np.full(3, np.nan)
    )
    release_position = (
        np.asarray(geometry["right_grasp_position_world"])[int(final_release)]
        if final_release is not None
        else np.full(3, np.nan)
    )
    release_bin_local = bin_rotation.T @ (release_position - bin_position)
    release_inside = bool(
        final_release is not None
        and np.all(np.abs(release_bin_local[:2]) <= 0.5 * opening)
    )
    source_release_inside = bool(
        final_release is not None
        and np.all(np.abs(source_release[:2] - bin_center_xy) <= 0.5 * opening)
    )
    handoff_start, handoff_end = events.handoff_window
    handoff_slice = slice(handoff_start, handoff_end + 1)
    handoff_frame = handoff_start + int(
        np.argmin(events.inter_hand_distance_m[handoff_slice])
    )

    semantic_valid = bool(
        right_grasp is not None
        and left_release is not None
        and int(right_grasp) < int(left_release)
    )
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in collision_flags.items()
    }
    def phase_sequence(labels: np.ndarray) -> list[str]:
        labels = np.asarray(labels).astype(str)
        if not len(labels):
            return []
        return labels[np.r_[True, labels[1:] != labels[:-1]]].tolist()

    metrics = {
        "schema_version": "doll_handoff_retargeting_metrics_v1",
        "method": method,
        "episode_index": episode.record.episode_index,
        "stable_episode_id": episode.record.stable_episode_id,
        "source_name": episode.record.source_name,
        "frame_count": count,
        "fps": fps,
        "duration_sec": episode.record.duration_sec,
        "conversion_attempted": True,
        "cartesian_target_sha256": targets["cartesian_target_sha256"],
        "finite": finite,
        "nan_inf_count": nan_inf_count,
        "joint_limit_violation_count": int(np.count_nonzero(arm_limit_mask))
        + int(hands["joint_limit_violation_count"]),
        "arm_joint_limit_violation_count": int(np.count_nonzero(arm_limit_mask)),
        "hand_joint_limit_violation_count": int(hands["joint_limit_violation_count"]),
        "ik_success_rate": float(np.mean(ik_success)),
        "ik_failed_frame_count": int(np.count_nonzero(~ik_success)),
        "physically_usable_ik_success_rate": float(
            np.mean(physically_usable_ik)
        ),
        "physically_unusable_ik_frame_count": int(
            np.count_nonzero(~physically_usable_ik)
        ),
        "physically_usable_ik_thresholds": {
            "position_m": float(
                validation_cfg["physically_usable_position_tolerance_m"]
            ),
            "orientation_rad": float(
                validation_cfg["physically_usable_orientation_tolerance_rad"]
            ),
            "policy": validation_cfg["physically_usable_tolerance_policy"],
        },
        "mean_ik_task_error_m": float(
            np.mean(
                np.column_stack(
                    (
                        ik_task_position_error["left"],
                        ik_task_position_error["right"],
                    )
                )
            )
        ),
        "max_ik_task_error_m": float(
            np.max(
                np.column_stack(
                    (
                        ik_task_position_error["left"],
                        ik_task_position_error["right"],
                    )
                )
            )
        ),
        "maximum_joint_step_rad": float(np.max(step, initial=0.0)),
        "maximum_joint_velocity_rad_s": float(np.max(velocity, initial=0.0)),
        "maximum_joint_acceleration_rad_s2": float(np.max(acceleration, initial=0.0)),
        "maximum_joint_jerk_rad_s3": float(np.max(jerk, initial=0.0)),
        "joint_step_rad": scalar_stats(step),
        "joint_velocity_rad_s": scalar_stats(velocity),
        "joint_acceleration_rad_s2": scalar_stats(acceleration),
        "joint_jerk_rad_s3": scalar_stats(jerk),
        "branch_discontinuity_count": int(np.count_nonzero(branches)),
        "collisions": {
            "frame_counts": collision_counts,
            "pairs": geometry["collision_pairs"],
            "invalid_self_body_collision_frames": int(np.count_nonzero(invalid_collision)),
            "distal_hand_contact_review_frames": int(
                np.count_nonzero(distal_hand_contact)
            ),
            "all_self_penetration_frames": int(
                np.count_nonzero(invalid_collision | distal_hand_contact)
            ),
            "records": geometry["collision_records"],
            "distal_hand_contact_is_kinematic_failure": False,
            "intended_task_object_contact_separate": True,
        },
        "task_space": {
            "left_wrist_target_error_m": scalar_stats(wrist_position_error["left"]),
            "right_wrist_target_error_m": scalar_stats(wrist_position_error["right"]),
            "left_wrist_orientation_error_rad": scalar_stats(
                wrist_orientation_error["left"]
            ),
            "right_wrist_orientation_error_rad": scalar_stats(
                wrist_orientation_error["right"]
            ),
            "left_physical_grasp_frame_target_error_m": scalar_stats(
                grasp_position_error["left"]
            ),
            "right_physical_grasp_frame_target_error_m": scalar_stats(
                grasp_position_error["right"]
            ),
            "left_physical_grasp_frame_orientation_error_rad": scalar_stats(
                grasp_orientation_error["left"]
            ),
            "right_physical_grasp_frame_orientation_error_rad": scalar_stats(
                grasp_orientation_error["right"]
            ),
            "left_static_canonical_grasp_frame_target_error_m": scalar_stats(
                static_tool_position_error["left"]
            ),
            "right_static_canonical_grasp_frame_target_error_m": scalar_stats(
                static_tool_position_error["right"]
            ),
            "left_static_canonical_grasp_frame_orientation_error_rad": scalar_stats(
                static_tool_orientation_error["left"]
            ),
            "right_static_canonical_grasp_frame_orientation_error_rad": scalar_stats(
                static_tool_orientation_error["right"]
            ),
            "physical_grasp_frame_objective_explicit": bool(
                targets["explicit_grasp_frame_objective"]
            ),
            "direct_static_canonical_grasp_frame_position_objective": direct_grasp,
            "task_orientation_constrained": orientation_constrained,
            "orientation_weight_multiplier": scalar_stats(
                np.atleast_1d(targets.get("orientation_gauge_weight_multiplier", 0.0))
            ),
        },
        "bimanual": {
            "midpoint_error_m": scalar_stats(midpoint_error),
            "relative_vector_error_m": scalar_stats(relative_error),
            "relative_vector_motion_error_m": scalar_stats(relative_motion_error),
            "inter_hand_relation_error_m": scalar_stats(relation_error),
            "inter_grasp_frame_distance_m": scalar_stats(inter_grasp),
            "target_inter_hand_distance_m": scalar_stats(target_inter),
            "objective_explicit": bool(
                targets.get("explicit_bimanual_objective", False)
                or targets.get("explicit_bimanual_target_constraint", False)
            ),
            "solver_residual_objective": bool(
                targets.get("explicit_bimanual_objective", False)
            ),
            "target_representation_constraint": bool(
                targets.get("explicit_bimanual_target_constraint", False)
            ),
            "absolute_spacing_clamp": targets.get(
                "bimanual_absolute_spacing_policy"
            ),
            "handoff_cartesian_offset_m": targets.get(
                "handoff_cartesian_offset_m", 0.0
            ),
            "handoff_constrained_frame_count": targets.get(
                "handoff_constrained_frame_count", 0
            ),
            "relative_vector_policy": targets.get(
                "bimanual_relative_vector_policy",
                "change-from-frame-zero relation; realized G1 initial spacing retained",
            ),
        },
        "semantics": {
            "source_semantic_valid": events.source_semantic_valid,
            "source_anomalies": events.anomalies,
            "right_grasp_before_left_release": semantic_valid,
            "dual_hold_frames": (
                int(left_release) - int(right_grasp)
                if semantic_valid
                else 0
            ),
            "dual_hold_sec": (
                (int(left_release) - int(right_grasp)) / fps if semantic_valid else 0.0
            ),
            "left_phase_sequence": phase_sequence(hands["left_phase"]),
            "right_phase_sequence": phase_sequence(hands["right_phase"]),
            "ownership_sequence": phase_sequence(events.ownership_labels),
            "ownership_claim_scope": (
                "SOURCE_DERIVED_INTERACTION_SEMANTICS_NOT_MEASURED_OBJECT_CONTACT"
            ),
            "ownership_transition_validity": semantic_valid,
            "semantic_phase_validity": semantic_valid,
        },
        "scene_diagnostics": {
            "diagnostic_only_not_used_to_alter_trajectory": True,
            "doll_center_world_m": doll_center,
            "bin_opening_center_world_m": [*bin_center_xy, rim_z],
            "source_left_task_frame_to_doll_at_left_grasp_m": source_left_grasp_distance,
            "source_left_task_frame_at_left_grasp_world_m": source_left_grasp_position,
            "source_left_grasp_minus_modeled_doll_world_m": source_left_grasp_position
            - doll_center,
            "target_left_grasp_frame_to_doll_at_left_grasp_m": target_left_grasp_distance,
            "target_left_grasp_frame_at_left_grasp_world_m": target_left_grasp_position,
            "target_left_grasp_minus_modeled_doll_world_m": target_left_grasp_position
            - doll_center,
            "handoff_reference_frame": handoff_frame,
            "left_grasp_frame_at_handoff_world_m": np.asarray(
                geometry["left_grasp_position_world"]
            )[handoff_frame],
            "right_grasp_frame_at_handoff_world_m": np.asarray(
                geometry["right_grasp_position_world"]
            )[handoff_frame],
            "inter_grasp_frame_distance_at_handoff_m": float(
                inter_grasp[handoff_frame]
            ),
            "right_final_release_grasp_frame_world_m": release_position,
            "right_final_release_xy_inside_bin_opening": release_inside,
            "right_final_release_horizontal_distance_to_opening_center_m": float(
                np.linalg.norm(release_position[:2] - bin_center_xy)
            )
            if final_release is not None
            else None,
            "right_final_release_height_relative_to_bin_opening_m": float(
                release_position[2] - rim_z
            )
            if final_release is not None
            else None,
            "source_right_final_release_world_m": source_release,
            "source_right_final_release_xy_inside_bin_opening": source_release_inside,
            "source_right_final_release_horizontal_distance_to_opening_center_m": float(
                np.linalg.norm(source_release[:2] - bin_center_xy)
            )
            if final_release is not None
            else None,
            "source_right_final_release_height_relative_to_bin_opening_m": float(
                source_release[2] - rim_z
            )
            if final_release is not None
            else None,
        },
        "solver": {
            "backend": solver["backend"],
            "elapsed_sec": solver["elapsed_sec"],
            "same_backend_for_a_b": True,
            "natural_arm_enabled": bool(solver["natural_arm_enabled"]),
            "initial_iteration_mean": float(
                np.mean([row["iterations"] for row in solver["initial_meta"]])
            ),
            "reprojection_iteration_mean": float(
                np.mean([row["iterations"] for row in solver["reprojection_meta"]])
            ),
        },
        "natural_arm": {
            "enabled": bool(solver["natural_arm_enabled"]),
            "common_a_b_layer": True,
            "primary_cartesian_targets_changed": False,
            "elbow_sew_representation": (
                "signed elbow radial angle about shoulder-to-wrist axis relative "
                "to the task-ready nominal morphology guide"
            ),
            "left_sew_angle_rad": scalar_stats(sew_unwrapped["left"]),
            "right_sew_angle_rad": scalar_stats(sew_unwrapped["right"]),
            "left_sew_range_rad": float(np.ptp(sew_unwrapped["left"])),
            "right_sew_range_rad": float(np.ptp(sew_unwrapped["right"])),
            "maximum_frame_to_frame_sew_change_rad": float(
                max(
                    np.max(sew_step["left"], initial=0.0),
                    np.max(sew_step["right"], initial=0.0),
                )
            ),
            "nominal_posture_deviation_l2_rad": scalar_stats(nominal_deviation),
            "left_manipulability_min_singular_value": scalar_stats(
                manipulability_values["left"]
            ),
            "right_manipulability_min_singular_value": scalar_stats(
                manipulability_values["right"]
            ),
            "manipulability_metric": solver["natural_arm_reference"][
                "manipulability_metric"
            ],
            "torso_clearance_m": scalar_stats(torso_clearance),
            "cross_arm_proximal_clearance_m": scalar_stats(cross_arm_clearance),
            "clearance_pair_policy": (
                "actual active-model shoulder-yaw/elbow/wrist collision geometry "
                "against torso and proximal opposite arm; interacting hands excluded"
            ),
            "maximum_nullspace_update_norm_rad": float(
                max(
                    [
                        row.get("maximum_nullspace_update_norm_rad", 0.0)
                        for row in solver["initial_meta"]
                        + solver["reprojection_meta"]
                    ],
                    default=0.0,
                )
            ),
        },
        "ik_success_per_frame": ik_success,
    }
    checks = {
        "finite": finite and nan_inf_count == 0,
        "frame_mapping": bool(targets["finite"])
        and float(targets["bimanual_reconstruction_max_error_m"]) <= 1e-10,
        "joint_limits": metrics["joint_limit_violation_count"] == 0,
        "ik": metrics["ik_success_rate"] >= float(ik_cfg["required_success_rate"]),
        "temporal": bool(
            metrics["maximum_joint_step_rad"]
            <= float(validation_cfg["maximum_joint_step_rad"])
            and metrics["maximum_joint_velocity_rad_s"]
            <= float(validation_cfg["maximum_velocity_rad_s"])
            and metrics["maximum_joint_acceleration_rad_s2"]
            <= float(validation_cfg["maximum_acceleration_rad_s2"])
            and metrics["branch_discontinuity_count"] == 0
        ),
        "collision": metrics["collisions"]["invalid_self_body_collision_frames"]
        <= int(validation_cfg["prohibited_collision_frames_allowed"]),
        "source_semantics": events.source_semantic_valid,
    }
    order = (
        ("FAIL_DATA", "finite"),
        ("FAIL_FRAME_MAPPING", "frame_mapping"),
        ("FAIL_JOINT_LIMIT", "joint_limits"),
        ("FAIL_IK", "ik"),
        ("FAIL_TEMPORAL", "temporal"),
        ("FAIL_COLLISION", "collision"),
        ("WARN_SOURCE_SEMANTICS", "source_semantics"),
    )
    status = "PASS"
    first_failure = None
    for candidate, check in order:
        if not checks[check]:
            status = candidate
            first_failure = check
            break
    metrics["status"] = status
    metrics["kinematic_pass"] = status == "PASS"
    validation = {
        "status": status,
        "pass": status == "PASS",
        "checks": checks,
        "first_failure_gate": first_failure,
        "thresholds": {
            "position_tolerance_m": ik_cfg["position_tolerance_m"],
            "orientation_tolerance_rad": ik_cfg["orientation_tolerance_rad"],
            "task_orientation_constrained": orientation_constrained,
            "orientation_weight_multiplier": scalar_stats(
                np.atleast_1d(
                    targets.get("orientation_gauge_weight_multiplier", 0.0)
                )
            ),
            "required_ik_success_rate": ik_cfg["required_success_rate"],
            "maximum_joint_step_rad": validation_cfg["maximum_joint_step_rad"],
            "maximum_velocity_rad_s": validation_cfg["maximum_velocity_rad_s"],
            "maximum_acceleration_rad_s2": validation_cfg[
                "maximum_acceleration_rad_s2"
            ],
        },
        "trajectory_changed_from_scene_diagnostics": False,
        "physics_success_tuning": "NOT_PERFORMED",
        "dataset_packaging": "NOT_PERFORMED",
        "policy_training": "NOT_PERFORMED",
        "real_robot_command": "NOT_PERFORMED",
    }
    return metrics, validation


__all__ = [
    "ConversionResult",
    "HandMapper",
    "RepresentationBuilder",
    "SharedTemporalIK",
    "derive_orientation_alignment",
    "validate_result",
]
