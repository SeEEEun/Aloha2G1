"""Single common entry architecture for both doll-handoff retargeting methods."""
from __future__ import annotations

import copy
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .common import (
    BASELINE_TEMPLATE,
    COMPARISON,
    CONFIG_OUTPUT,
    COMMON_TEMPLATE,
    METHODS,
    SUPPORTED_METHODS,
    OUTPUT,
    PROPOSED_TEMPLATE,
    SIDES,
    atomic_csv,
    atomic_json,
    atomic_npz,
    implementation_fingerprint,
    load_common_config,
    load_json,
    load_scene,
    scalar_stats,
    sha256_file,
)
from .events import EventAuditor
from .models import ALOHAKinematics, G1Kinematics
from .retarget import (
    ConversionResult,
    HandMapper,
    RepresentationBuilder,
    SharedTemporalIK,
    derive_baseline_workspace_mapping,
    derive_orientation_alignment,
    validate_result,
)
from .source import SourceRepository


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DollHandoffPipeline:
    """One source/FK/IK/validation path with representation-only A/B hooks."""

    def __init__(
        self,
        common_path: str | Path = COMMON_TEMPLATE,
        output_root: str | Path = OUTPUT,
        natural_arm_enabled: bool | None = None,
        registration_path: str | Path | None = None,
    ):
        self.output = Path(output_root).resolve()
        self.config_output = self.output / "config"
        self.common_path = Path(common_path).resolve()
        self.common = load_common_config(self.common_path)
        self.scene = load_scene(self.common)
        self.registration_path = (
            Path(registration_path).resolve() if registration_path is not None else None
        )
        self.registration_manifest = (
            load_json(self.registration_path) if self.registration_path is not None else None
        )
        if self.registration_manifest is not None:
            if self.registration_manifest.get("status") != "PASS":
                raise RuntimeError("episode registration manifest did not pass")
            if not bool(self.registration_manifest.get("common_for_A_B")):
                raise RuntimeError("episode registration is not common for A/B")
            self.registration_by_source = {
                str(row["source_name"]): row
                for row in self.registration_manifest["entries"]
            }
            for source, row in self.registration_by_source.items():
                unhashed = copy.deepcopy(row)
                expected = str(unhashed.pop("entry_sha256"))
                actual = _stable_digest(unhashed)
                if actual != expected:
                    raise RuntimeError(
                        f"episode registration entry hash mismatch: {source}"
                    )
        else:
            self.registration_by_source = {}
        self.aloha = ALOHAKinematics(self.common, self.scene)
        self.g1 = G1Kinematics(self.common, self.scene)
        self.sources = SourceRepository(
            self.common, self.aloha, self.output / "source_audit"
        )
        self.source_manifest = self.sources.audit()
        self.event_auditor = EventAuditor(
            self.common, self.scene, self.sources, self.output / "event_audit"
        )
        self.event_summary = self.event_auditor.run()
        self.nominal_q, self.nominal_report = self.g1.derive_task_ready_nominal(
            self.common, self.scene
        )
        self.proposed_template = load_json(PROPOSED_TEMPLATE)
        self.baseline_template = load_json(BASELINE_TEMPLATE)
        self.primitives = self.g1.derive_hand_primitives(
            self.scene, self.proposed_template, self.nominal_q
        )
        self.alignment = derive_orientation_alignment(
            self.event_auditor.fk_cache, self.g1, self.primitives
        )
        self.baseline_workspace_mapping = derive_baseline_workspace_mapping(
            self.common,
            self.scene,
            self.event_auditor.fk_cache,
            self.aloha.shoulder_wrist_reach_geometry(),
            self.aloha.wrist_orientation_capacity(),
            self.g1,
            self.nominal_q,
            self.baseline_template,
        )
        self.representation = RepresentationBuilder(
            self.common,
            self.scene,
            self.g1,
            self.alignment,
            self.proposed_template,
            self.baseline_workspace_mapping,
        )
        self.hand_mapper = HandMapper(self.common, self.g1, self.primitives)
        self.solver = SharedTemporalIK(
            self.common,
            self.g1,
            self.nominal_q,
            natural_arm_enabled=natural_arm_enabled,
        )
        self.implementation_sha256, self.implementation_files = implementation_fingerprint()
        self.resolved_configs = self._resolved_configs()
        self._write_configuration_artifacts()
        self._write_static_audits()

    def _resolved_configs(self) -> dict[str, dict[str, Any]]:
        common = copy.deepcopy(self.common)
        common["resolved"] = {
            "source_audit": {
                "enumerated_count": self.source_manifest["enumerated_count"],
                "valid_count": self.source_manifest["valid_count"],
                "invalid_count": self.source_manifest["invalid_count"],
                "manifest_path": str(self.output / "source_audit/source_manifest.json"),
                "manifest_sha256": sha256_file(
                    self.output / "source_audit/source_manifest.json"
                ),
            },
            "event_detector": self.event_auditor.detector_config,
            "canonical_g1_arm_joint_names": self.g1.arm_joint_names,
            "canonical_g1_nominal_q": self.nominal_q,
            "canonical_task_ready_report": self.nominal_report,
            "g1_model_pelvis_xyz_m": self.g1.model_pelvis,
            "aloha_link6_to_task_tcp": self.aloha.link6_to_tcp,
            "shared_hand_transition_frames": self.hand_mapper.transition_frames,
            "natural_arm_redundancy": self.solver.natural_reference_report,
            "implementation_sha256": self.implementation_sha256,
            "episode_registration": (
                {
                    "manifest": str(self.registration_path),
                    "manifest_sha256": sha256_file(self.registration_path),
                    "content_sha256": self.registration_manifest.get("content_sha256"),
                    "semantics": self.registration_manifest.get(
                        "registration_semantics"
                    ),
                    "common_for_A_B": True,
                }
                if self.registration_manifest is not None
                else None
            ),
        }
        common["runtime_fingerprint"] = _stable_digest(common["resolved"])

        baseline = copy.deepcopy(self.baseline_template)
        baseline["resolved"] = {
            "orientation_alignment": {
                side: self.alignment["sides"][side][
                    "source_interaction_to_g1_grasp_axis_alignment"
                ]
                for side in SIDES
            },
            "common_hand_states": self.primitives["states"],
            "common_hand_transition_frames": self.hand_mapper.transition_frames,
            "joint_names": self.primitives["joint_names"],
            "fixed_target_tool_to_g1_wrist": self.primitives[
                "wrist_to_grasp_frame"
            ],
            "legacy_global_workspace_mapping_active": False,
            "common_runtime_fingerprint": common["runtime_fingerprint"],
        }
        baseline["runtime_fingerprint"] = _stable_digest(baseline["resolved"])

        proposed = copy.deepcopy(self.proposed_template)
        proposed["resolved"] = {
            "interaction_and_ownership_semantics": (
                self.representation.proposed_semantics_report()
            ),
            "states": self.primitives["states"],
            "joint_names": self.primitives["joint_names"],
            "wrist_to_grasp_frame": self.primitives["wrist_to_grasp_frame"],
            "target_grasp_enclosure_radius_m": self.primitives[
                "target_grasp_enclosure_radius_m"
            ],
            "achieved_grasp_enclosure_radius_m": self.primitives[
                "achieved_grasp_enclosure_radius_m"
            ],
            "target_preshape_enclosure_radius_m": self.primitives[
                "target_preshape_enclosure_radius_m"
            ],
            "achieved_preshape_enclosure_radius_m": self.primitives[
                "achieved_preshape_enclosure_radius_m"
            ],
            "geometry_at_grasp": self.primitives["geometry_at_grasp"],
            "common_hand_transition_frames": self.hand_mapper.transition_frames,
            "common_runtime_fingerprint": common["runtime_fingerprint"],
        }
        proposed["runtime_fingerprint"] = _stable_digest(proposed["resolved"])
        return {"common": common, "baseline": baseline, "proposed": proposed}

    def _write_configuration_artifacts(self) -> None:
        self.config_output.mkdir(parents=True, exist_ok=True)
        names = {
            "common": self.config_output / "common_config.json",
            "baseline": self.config_output / "baseline_config.json",
            "proposed": self.config_output / "proposed_config.json",
        }
        frozen_manifest = self.config_output / "freeze_manifest.json"
        frozen = frozen_manifest.is_file() and load_json(frozen_manifest).get("status") == "FROZEN"
        for key, path in names.items():
            candidate = copy.deepcopy(self.resolved_configs[key])
            candidate["status"] = (
                "IMMUTABLE_AFTER_SMOKE_TEST" if frozen else "SMOKE_TEST_CANDIDATE"
            )
            if frozen and path.is_file():
                existing = load_json(path)
                if existing.get("runtime_fingerprint") != candidate.get("runtime_fingerprint"):
                    raise RuntimeError(
                        f"frozen {key} runtime differs from current implementation/config"
                    )
                continue
            atomic_json(path, candidate)
        self.config_paths = names

    def _write_static_audits(self) -> None:
        task_frame = {
            "schema_version": "doll_handoff_task_frame_v1",
            "scene_config": self.common["scene_config"],
            "scene_config_sha256": self.common["scene_config_sha256"],
            "task_origin_world_xyz_m": self.scene["task_frame"][
                "origin_world_xyz_m"
            ],
            "task_axes": {
                "x": self.scene["task_frame"]["x_axis"],
                "y": self.scene["task_frame"]["y_axis"],
                "z": self.scene["task_frame"]["z_axis"],
            },
            "task_rotation_world": np.eye(3),
            "aloha_root": self.scene["aloha"],
            "g1_root": self.scene["g1"],
            "g1_model_pelvis_xyz_m": self.g1.model_pelvis,
            "uniform_metric_scale": self.common["task_registration"][
                "uniform_metric_scale"
            ],
            "global_translation_correction_world_xyz_m": self.common[
                "task_registration"
            ]["global_translation_correction_world_xyz_m"],
            "global_rotation_correction_world_wxyz": self.common[
                "task_registration"
            ]["global_rotation_correction_world_wxyz"],
            "registration_derivation": self.common["task_registration"].get(
                "derivation"
            ),
            "episode_specific_registration": self.registration_manifest is not None,
            "episode_registration_manifest": (
                str(self.registration_path)
                if self.registration_path is not None
                else None
            ),
        }
        atomic_json(self.config_output / "task_frame_report.json", task_frame)
        tool_report = {
            "schema_version": "doll_handoff_interaction_frame_report_v2",
            "calibration_status": "SIM_ONLY_NOT_REAL_DEX3_CALIBRATED",
            "aloha": {
                "frame": "static physical jaw-closing-region interaction frame",
                "link6_to_task_tcp": self.aloha.link6_to_tcp,
                "provenance": self.aloha.channel_report()["tcp_derivation"],
            },
            "g1": {
                "task_fingers": ["thumb", "index", "middle"],
                "whole_hand_geometry": str(self.g1.mapping_path),
                "joint_names": self.primitives["joint_names"],
                "canonical_states": self.primitives["states"],
                "grasp_frame_definition": self.primitives[
                    "grasp_frame_definition"
                ],
                "left_wrist_to_grasp_frame": self.primitives[
                    "wrist_to_grasp_frame"
                ]["left"],
                "right_wrist_to_grasp_frame": self.primitives[
                    "wrist_to_grasp_frame"
                ]["right"],
                "target_grasp_enclosure_radius_m": self.primitives[
                    "target_grasp_enclosure_radius_m"
                ],
                "achieved_grasp_enclosure_radius_m": self.primitives[
                    "achieved_grasp_enclosure_radius_m"
                ],
                "geometry_at_grasp": self.primitives["geometry_at_grasp"],
                "transform_scope": "one static transform per side for all episodes and frames",
            },
            "source_to_target_axis_alignment": self.alignment,
            "proposed_task_ee_and_bimanual_semantics": (
                self.representation.proposed_semantics_report()
            ),
        }
        atomic_json(self.config_output / "tool_frame_report.json", tool_report)
        unit_audit = {
            "source_model_length_unit": "meter",
            "target_model_length_unit": "meter",
            "scene_length_unit": "meter",
            "source_task_motion_scale": 1.0,
            "target_task_motion_scale": 1.0,
            "shared_global_scale": 1.0,
            "non_unit_scale_justified": False,
            "baseline_representation_workspace_normalization": "DISABLED",
            "model_workspace_observation": (
                "Both representations use the same 1:1 metric task registration. "
                "WRIST transfers the registered source task-TCP pose through one "
                "fixed target-tool-to-G1-wrist transform; INTERACTION supplies the "
                "registered whole-hand interaction target. No pooled workspace "
                "normalization or representation-specific reach clipping is active."
            ),
        }
        atomic_json(self.config_output / "model_unit_audit.json", unit_audit)
        atomic_json(
            self.config_output / "baseline_workspace_mapping_report.json",
            self.baseline_workspace_mapping,
        )
        mapping = self.baseline_workspace_mapping
        mapping_markdown = "\n".join(
            [
                "# Legacy Baseline-A workspace mapping diagnostic",
                "",
                "Status: **INACTIVE AFTER SINGLE-VARIABLE RESET**",
                "",
                "This historical pooled mapping is computed only to preserve an "
                "audit trail. It is not read by the WRIST target-generation branch.",
                "",
                f"- Method: {mapping['method']}",
                f"- Task-axis scale XYZ: `{mapping['task_axis_scale_xyz']}`",
                f"- Source anchor world: `{mapping['source_anchor_world_m']}`",
                f"- Target anchor world: `{mapping['target_anchor_world_m']}`",
                f"- Source reach: `{mapping['source_effective_reach_m']:.9f} m`",
                f"- G1 reach: `{mapping['target_effective_reach_m']:.9f} m`",
                f"- Common outer reach limit: `{mapping['outer_reach_limit_m']:.9f} m`",
                f"- Global SO(3) deviation scale: `{mapping['orientation_deviation_scale']:.9f}`",
                f"- Global clip counts: `{mapping['global_clip_counts']}`",
                "- Episode-specific parameters: **NO**",
                "- Phase-specific parameters: **NO**",
                "- Interaction/grasp-frame information: **NO**",
                "- Common metric task-frame registration changed: **NO**",
                "",
                "## Historical A0 diagnosis",
                "",
                "A0 directly copied metric source wrist origins. Its pooled "
                "shoulder-radius extrema exceeded the active G1 arm-chain reach, "
                "and it computed alignment from the source TCP while applying it "
                "to source wrist rotations. The single-variable reset instead uses "
                "the registered task TCP and one fixed tool-compatibility transform.",
                "",
                f"- A0 left shoulder radius: `{mapping['naive_a0_shoulder_radius_m']['left']}`",
                f"- A0 right shoulder radius: `{mapping['naive_a0_shoulder_radius_m']['right']}`",
                f"- Fair-A left shoulder radius: `{mapping['mapped_shoulder_radius_m']['left']}`",
                f"- Fair-A right shoulder radius: `{mapping['mapped_shoulder_radius_m']['right']}`",
                "",
            ]
        )
        (self.config_output / "baseline_workspace_mapping_report.md").write_text(
            mapping_markdown, encoding="utf-8"
        )
        fairness = {
            "schema_version": "doll_handoff_ab_fairness_v1",
            "pass": True,
            "shared": [
                "50 sorted source recordings",
                "timestamps and 30 Hz sample rate",
                "ALOHA source FK model and static jaw frame",
                "G1 model and fixed root",
                "scene task frame and metric 1:1 registration",
                "controlled arm joints and joint limits",
                "task-ready nominal posture",
                "temporal IK backend, budgets, tolerances and weights",
                "velocity and acceleration regularization",
                "collision checker",
                "numerical gates",
                "source-derived semantic event timeline",
                "Dex3 phase labels, commands, limits, and transition smoothing",
            ],
            "only_differences": {
                "baseline": "WRIST spatial target generation",
                "proposed": "INTERACTION spatial target generation",
            },
            "single_scientific_switch": {
                "name": "representation_mode",
                "baseline_value": "WRIST",
                "proposed_value": "INTERACTION",
            },
            "common_dex3_mapping": True,
            "shared_solver_config": self.common["shared_temporal_ik"],
            "method_specific_solver_parameters": 0,
            "episode_specific_logic": 0,
            "frame_specific_logic": 0,
            "object_waypoints": 0,
        }
        atomic_json(self.config_output / "fairness_report.json", fairness)
        reuse = """# Doll-Handoff code reuse audit

| component | source file | reused / modified / replaced | reason |
|---|---|---|---|
| Stationary ALOHA model | `/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml` | reused | Validated six-joint arm and prismatic gripper geometry is task-independent. |
| ALOHA loader/FK | `tools/doll_handoff_retargeting/models.py` | replaced | Resolves named joints and derives the physical jaw midpoint directly; no old episode or task configuration is imported. |
| G1 + Dex3 model | `/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml` | reused | Active target kinematics and limits are task-independent. |
| Dex3 whole-hand geometry | `configs/doll_handoff_retargeting/dex3_whole_hand.sim.json` | replaced | Thumb, index, and middle pad geometry is resolved from the active model for one predefined whole-hand synergy. |
| Source event detector | `tools/doll_handoff_retargeting/events.py` | replaced | Pooled doll-handoff transition structure, smoothing, hysteresis, and debounce are derived from all 50 new sources. |
| A/B representations | `tools/doll_handoff_retargeting/retarget.py` | replaced | Baseline wrist and proposed interaction-frame representations share metric task registration without object waypoints. |
| Temporal IK | `tools/doll_handoff_retargeting/retarget.py::SharedTemporalIK` | modified | Acceptance-aware first-frame solve, causal temporal regularization, smoothing, and reprojection are identical for A and B. |
| Collision audit | `tools/doll_handoff_retargeting/models.py::trajectory_geometry` | replaced | Named arm/body categories distinguish invalid self/body penetration from task-object diagnostics. |
| Scene geometry | `isaaclab_doll_handoff_scene/scene_layout.json` | reused | Approved table, black frame, doll, bin, roots, task frame, and cameras remain immutable. |
| Review renderer | `tools/render_doll_handoff_comparisons.py` | replaced | Reads only the approved scene config and new trajectories; source and A/B panels are synchronized. |

No prior task geometry, phase labels, episode anchors, or object-placement constants are active in this pipeline.
"""
        (self.output / "code_reuse_audit.md").write_text(reuse, encoding="utf-8")

    def convert(self, method: str, episode_index: int) -> ConversionResult:
        if method not in SUPPORTED_METHODS:
            raise ValueError(method)
        representation_mode = (
            "WRIST" if method == "baseline" else "INTERACTION"
        )
        episode = self.sources.episode(episode_index)
        registration_entry = None
        if self.registration_manifest is not None:
            registration_entry = self.registration_by_source.get(
                episode.record.source_name
            )
            if registration_entry is None:
                raise RuntimeError(
                    f"missing common episode registration: {episode.record.source_name}"
                )
        fk = self.event_auditor.fk_cache.get(episode_index) or self.aloha.fk(
            episode.state
        )
        events = self.event_auditor.events[episode_index]
        targets = self.representation.build(
            representation_mode,
            fk,
            events,
            registration_entry=registration_entry,
        )
        # The representation switch is spatial only.  Dex3 supervision is a
        # common target-embodiment realization of the one source timeline.
        hands = self.hand_mapper.map_common(events)
        solver = self.solver.solve(targets)
        geometry = self.g1.trajectory_geometry(
            solver["q"],
            hands["left"],
            hands["right"],
            float(self.common["validation"]["collision_penetration_tolerance_m"]),
            targets.get("static_wrist_to_tool"),
        )
        metrics, validation = validate_result(
            self.common,
            self.scene,
            self.g1,
            method,
            episode,
            fk,
            events,
            targets,
            hands,
            solver,
            geometry,
        )
        return ConversionResult(
            method=method,
            episode=episode,
            fk=fk,
            events=events,
            targets=targets,
            hands=hands,
            solver=solver,
            geometry=geometry,
            metrics=metrics,
            validation=validation,
        )

    def _paths(self, method: str, episode_index: int) -> dict[str, Path]:
        stable = self.sources.records[episode_index].stable_episode_id
        return {
            "trajectory": self.output / method / "trajectories" / f"{stable}.npz",
            "metrics": self.output / method / "metrics" / f"{stable}.json",
            "validation": self.output
            / method
            / "metrics"
            / f"{stable}.validation.json",
            "manifest": self.output / method / "metrics" / f"{stable}.manifest.json",
        }

    def export(self, result: ConversionResult) -> dict[str, Path]:
        episode = result.episode
        paths = self._paths(result.method, episode.record.episode_index)
        arm = np.asarray(result.solver["q"], dtype=np.float32)
        left = np.asarray(result.hands["left"], dtype=np.float32)
        right = np.asarray(result.hands["right"], dtype=np.float32)
        names = np.concatenate(
            (
                self.g1.arm_joint_names,
                np.asarray(self.g1.hand_joint_names["left"], dtype="U64"),
                np.asarray(self.g1.hand_joint_names["right"], dtype="U64"),
            )
        )
        replay = np.column_stack((arm, left, right)).astype(np.float32)
        config_key = (
            result.method if result.method in self.config_paths else "proposed"
        )
        atomic_npz(
            paths["trajectory"],
            timestamp=episode.timestamps.astype(np.float64),
            source_frame_index=episode.frame_index.astype(np.int64),
            g1_arm_joint_names=self.g1.arm_joint_names,
            g1_arm_qpos=arm,
            left_dex3_joint_names=np.asarray(
                self.g1.hand_joint_names["left"], dtype="U64"
            ),
            left_dex3_qpos=left,
            right_dex3_joint_names=np.asarray(
                self.g1.hand_joint_names["right"], dtype="U64"
            ),
            right_dex3_qpos=right,
            left_hand_phase=np.asarray(result.hands["left_phase"], dtype="U12"),
            right_hand_phase=np.asarray(result.hands["right_phase"], dtype="U12"),
            ownership_state=np.asarray(result.events.ownership_labels, dtype="U20"),
            method=np.asarray(result.method),
            representation_mode=np.asarray(
                result.targets["representation_mode"]
            ),
            episode_registration_bound=np.asarray(
                result.targets["episode_registration_bound"]
            ),
            episode_registration_entry_sha256=np.asarray(
                result.targets.get("episode_registration_entry_sha256", "")
            ),
            registered_object_position_world=np.asarray(
                result.targets.get("registered_object_pose", {}).get(
                    "position_xyz_m", [np.nan, np.nan, np.nan]
                ),
                dtype=np.float64,
            ),
            registered_object_quaternion_xyzw=np.asarray(
                result.targets.get("registered_object_pose", {}).get(
                    "quaternion_xyzw", [np.nan, np.nan, np.nan, np.nan]
                ),
                dtype=np.float64,
            ),
            registered_bin_position_world=np.asarray(
                result.targets.get("registered_bin_pose", {}).get(
                    "position_xyz_m", [np.nan, np.nan, np.nan]
                ),
                dtype=np.float64,
            ),
            registered_bin_quaternion_xyzw=np.asarray(
                result.targets.get("registered_bin_pose", {}).get(
                    "quaternion_xyzw", [np.nan, np.nan, np.nan, np.nan]
                ),
                dtype=np.float64,
            ),
            registered_table_task_origin_world=np.asarray(
                result.targets.get(
                    "registered_table_task_origin_xyz_m",
                    self.scene["task_frame"]["origin_world_xyz_m"],
                ),
                dtype=np.float64,
            ),
            source_episode_id=np.asarray(episode.record.stable_episode_id),
            source_directory_name=np.asarray(episode.record.source_name),
            source_motion_key=np.asarray(self.common["source_channels"]["motion_key"]),
            source_command_key=np.asarray(self.common["source_channels"]["command_key"]),
            replay_joint_names=names,
            replay_named_joint_qpos=replay,
            full_model_qpos_generated=np.asarray(False),
            replay_artifact_not_future_policy_schema=np.asarray(True),
            target_left_wrist_position_model=np.asarray(
                result.targets["left_wrist_position"], dtype=np.float32
            ),
            target_right_wrist_position_model=np.asarray(
                result.targets["right_wrist_position"], dtype=np.float32
            ),
            target_left_wrist_rotation_model=np.asarray(
                result.targets["left_wrist_rotation"], dtype=np.float32
            ),
            target_right_wrist_rotation_model=np.asarray(
                result.targets["right_wrist_rotation"], dtype=np.float32
            ),
            target_left_interaction_frame_position_world=np.asarray(
                result.targets["left_tool_position_world"], dtype=np.float32
            ),
            target_right_interaction_frame_position_world=np.asarray(
                result.targets["right_tool_position_world"], dtype=np.float32
            ),
            target_left_interaction_frame_position_task=np.asarray(
                result.targets["left_tool_position_task"], dtype=np.float32
            ),
            target_right_interaction_frame_position_task=np.asarray(
                result.targets["right_tool_position_task"], dtype=np.float32
            ),
            achieved_left_wrist_position_world=np.asarray(
                result.geometry["left_wrist_position_world"], dtype=np.float32
            ),
            achieved_right_wrist_position_world=np.asarray(
                result.geometry["right_wrist_position_world"], dtype=np.float32
            ),
            achieved_left_physical_grasp_frame_position_world=np.asarray(
                result.geometry["left_grasp_position_world"], dtype=np.float32
            ),
            achieved_right_physical_grasp_frame_position_world=np.asarray(
                result.geometry["right_grasp_position_world"], dtype=np.float32
            ),
            achieved_left_physical_grasp_frame_position_task=np.asarray(
                result.geometry["left_grasp_position_world"]
                - np.asarray(self.scene["task_frame"]["origin_world_xyz_m"]),
                dtype=np.float32,
            ),
            achieved_right_physical_grasp_frame_position_task=np.asarray(
                result.geometry["right_grasp_position_world"]
                - np.asarray(self.scene["task_frame"]["origin_world_xyz_m"]),
                dtype=np.float32,
            ),
            task_frame_origin_world_xyz_m=np.asarray(
                self.scene["task_frame"]["origin_world_xyz_m"], dtype=np.float64
            ),
            uniform_metric_scale=np.asarray(1.0, dtype=np.float64),
            inter_grasp_frame_distance_m=np.linalg.norm(
                np.asarray(result.geometry["right_grasp_position_world"])
                - np.asarray(result.geometry["left_grasp_position_world"]),
                axis=1,
            ).astype(np.float32),
            ik_success_per_frame=np.asarray(
                result.metrics["ik_success_per_frame"], dtype=bool
            ),
            event_names=np.asarray(list(result.events.frames), dtype="U32"),
            event_frames=np.asarray(
                [
                    -1 if value is None else int(value)
                    for value in result.events.frames.values()
                ],
                dtype=np.int64,
            ),
            config_sha256=np.asarray(sha256_file(self.config_paths[config_key])),
            common_config_sha256=np.asarray(sha256_file(self.config_paths["common"])),
            implementation_sha256=np.asarray(self.implementation_sha256),
            cartesian_target_sha256=np.asarray(
                result.targets["cartesian_target_sha256"]
            ),
            sim_hand_calibration=np.asarray("SIM_ONLY_NOT_REAL_DEX3_CALIBRATED"),
            policy_schema_declared=np.asarray(False),
            real_robot_command_allowed=np.asarray(False),
        )
        metrics = dict(result.metrics)
        metrics.pop("ik_success_per_frame", None)
        atomic_json(paths["metrics"], metrics)
        atomic_json(paths["validation"], result.validation)
        manifest = {
            "schema_version": "doll_handoff_retargeted_episode_v1",
            "method": result.method,
            "representation_mode": result.targets["representation_mode"],
            "episode_registration_bound": result.targets[
                "episode_registration_bound"
            ],
            "episode_registration_entry_sha256": result.targets.get(
                "episode_registration_entry_sha256"
            ),
            "episode_index": episode.record.episode_index,
            "stable_episode_id": episode.record.stable_episode_id,
            "source_name": episode.record.source_name,
            "status": result.validation["status"],
            "conversion_attempted": True,
            "trajectory": str(paths["trajectory"]),
            "trajectory_sha256": sha256_file(paths["trajectory"]),
            "metrics": str(paths["metrics"]),
            "metrics_sha256": sha256_file(paths["metrics"]),
            "validation": str(paths["validation"]),
            "config": str(self.config_paths[config_key]),
            "config_sha256": sha256_file(self.config_paths[config_key]),
            "common_config": str(self.config_paths["common"]),
            "common_config_sha256": sha256_file(self.config_paths["common"]),
            "implementation_sha256": self.implementation_sha256,
            "cartesian_target_sha256": result.targets[
                "cartesian_target_sha256"
            ],
            "scene_config_sha256": self.common["scene_config_sha256"],
            "dataset_packaging": False,
            "policy_training": False,
            "physics_tuning": False,
            "real_robot_commands": False,
        }
        atomic_json(paths["manifest"], manifest)
        return paths

    def export_failure(
        self, method: str, episode_index: int, error: BaseException
    ) -> dict[str, Path]:
        paths = self._paths(method, episode_index)
        record = self.sources.records[episode_index]
        atomic_npz(
            paths["trajectory"],
            timestamp=np.asarray([], dtype=np.float64),
            g1_arm_joint_names=self.g1.arm_joint_names,
            g1_arm_qpos=np.empty((0, 14), dtype=np.float32),
            left_dex3_qpos=np.empty((0, 7), dtype=np.float32),
            right_dex3_qpos=np.empty((0, 7), dtype=np.float32),
            method=np.asarray(method),
            source_episode_id=np.asarray(record.stable_episode_id),
            conversion_attempted=np.asarray(True),
            policy_schema_declared=np.asarray(False),
        )
        reason = f"{type(error).__name__}: {error}"
        metrics = {
            "method": method,
            "episode_index": episode_index,
            "stable_episode_id": record.stable_episode_id,
            "source_name": record.source_name,
            "status": "FAIL_EXCEPTION",
            "kinematic_pass": False,
            "conversion_attempted": True,
            "frame_count": 0,
            "ik_success_rate": 0.0,
            "failure_reason": reason,
        }
        validation = {
            "status": "FAIL_EXCEPTION",
            "pass": False,
            "checks": {},
            "first_failure_gate": "exception",
            "reason": reason,
            "dataset_packaging": "NOT_PERFORMED",
            "policy_training": "NOT_PERFORMED",
        }
        atomic_json(paths["metrics"], metrics)
        atomic_json(paths["validation"], validation)
        atomic_json(
            paths["manifest"],
            {
                "method": method,
                "episode_index": episode_index,
                "stable_episode_id": record.stable_episode_id,
                "source_name": record.source_name,
                "status": "FAIL_EXCEPTION",
                "conversion_attempted": True,
                "reason": reason,
                "trajectory": str(paths["trajectory"]),
                "dataset_packaging": False,
                "policy_training": False,
            },
        )
        return paths

    def run(self, method: str, episode_indices: Iterable[int]) -> dict[str, Any]:
        indices = list(map(int, episode_indices))
        counts: dict[str, int] = {}
        for offset, episode_index in enumerate(indices, 1):
            try:
                result = self.convert(method, episode_index)
                paths = self.export(result)
                status = result.validation["status"]
                counts[status] = counts.get(status, 0) + 1
                print(
                    f"[{offset:02d}/{len(indices):02d}] method={method} "
                    f"episode={episode_index:03d} source={result.episode.record.source_name} "
                    f"status={status} ik={result.metrics['ik_success_rate']:.4f} "
                    f"output={paths['trajectory']}",
                    flush=True,
                )
            except Exception as error:
                self.export_failure(method, episode_index, error)
                counts["FAIL_EXCEPTION"] = counts.get("FAIL_EXCEPTION", 0) + 1
                print(
                    f"[{offset:02d}/{len(indices):02d}] method={method} "
                    f"episode={episode_index:03d} status=FAIL_EXCEPTION "
                    f"reason={type(error).__name__}: {error}",
                    flush=True,
                )
                traceback.print_exc()
        summary = self.write_method_summary(method, require_all=len(indices) == 50)
        self.write_smoke_review_candidate()
        return {"counts": counts, "summary": summary}

    def write_method_summary(self, method: str, require_all: bool) -> dict[str, Any]:
        # Select only canonical per-episode metrics.  The same directory also
        # contains manifests, validation records, and this aggregate summary;
        # broad ``*.json`` matching would accidentally ingest those on a rerun.
        metric_paths = [
            self._paths(method, episode_index)["metrics"]
            for episode_index in range(len(self.sources.records))
            if self._paths(method, episode_index)["metrics"].is_file()
        ]
        rows = [load_json(path) for path in metric_paths]
        if require_all and len(rows) != 50:
            raise RuntimeError(f"{method}: expected 50 metric files, found {len(rows)}")
        summary_rows: list[dict[str, Any]] = []
        for value in rows:
            task = value.get("task_space", {})
            bimanual = value.get("bimanual", {})
            collisions = value.get("collisions", {})
            scene_diag = value.get("scene_diagnostics", {})
            summary_rows.append(
                {
                    "episode_index": value["episode_index"],
                    "stable_episode_id": value["stable_episode_id"],
                    "source_name": value["source_name"],
                    "status": value["status"],
                    "kinematic_pass": value.get("kinematic_pass", False),
                    "ik_success_rate": value.get("ik_success_rate", 0.0),
                    "mean_wrist_error_m": 0.5
                    * (
                        task.get("left_wrist_target_error_m", {}).get("mean", 0.0)
                        + task.get("right_wrist_target_error_m", {}).get("mean", 0.0)
                    ),
                    "mean_physical_grasp_frame_error_m": 0.5
                    * (
                        task.get("left_physical_grasp_frame_target_error_m", {}).get("mean", 0.0)
                        + task.get("right_physical_grasp_frame_target_error_m", {}).get("mean", 0.0)
                    ),
                    "mean_bimanual_relation_error_m": bimanual.get(
                        "inter_hand_relation_error_m", {}
                    ).get("mean", 0.0),
                    "invalid_collision_frames": collisions.get(
                        "invalid_self_body_collision_frames", 0
                    ),
                    "release_inside_bin": scene_diag.get(
                        "right_final_release_xy_inside_bin_opening", False
                    ),
                    "maximum_joint_velocity_rad_s": value.get(
                        "maximum_joint_velocity_rad_s", 0.0
                    ),
                    "maximum_joint_acceleration_rad_s2": value.get(
                        "maximum_joint_acceleration_rad_s2", 0.0
                    ),
                }
            )
        if summary_rows:
            atomic_csv(
                self.output / method / "metrics" / "episode_summary.csv", summary_rows
            )
        numeric = [
            "ik_success_rate",
            "mean_wrist_error_m",
            "mean_physical_grasp_frame_error_m",
            "mean_bimanual_relation_error_m",
            "invalid_collision_frames",
            "maximum_joint_velocity_rad_s",
            "maximum_joint_acceleration_rad_s2",
        ]
        aggregate = {
            key: scalar_stats(np.asarray([float(row[key]) for row in summary_rows]))
            for key in numeric
        }
        summary = {
            "method": method,
            "metric_file_count": len(rows),
            "converted_count": sum(bool(row.get("conversion_attempted")) for row in rows),
            "kinematic_pass_count": sum(bool(row.get("kinematic_pass")) for row in rows),
            "kinematic_fail_count": sum(not bool(row.get("kinematic_pass")) for row in rows),
            "status_counts": {
                status: sum(row.get("status") == status for row in rows)
                for status in sorted({row.get("status") for row in rows})
            },
            "release_inside_bin_count": sum(
                bool(row["release_inside_bin"]) for row in summary_rows
            ),
            "aggregate": aggregate,
            "failed_episode_indices": [
                int(row["episode_index"])
                for row in rows
                if not bool(row.get("kinematic_pass"))
            ],
            "dataset_packaging": "NOT_STARTED_BY_DESIGN",
            "policy_training": "NOT_STARTED_BY_DESIGN",
        }
        atomic_json(self.output / method / "metrics" / "aggregate_summary.json", summary)
        return summary

    def write_smoke_review_candidate(self) -> bool:
        smoke = list(map(int, self.common["render"]["smoke_episode_indices"]))
        complete = all(
            self._paths(method, episode)["metrics"].is_file()
            and self._paths(method, episode)["manifest"].is_file()
            and load_json(self._paths(method, episode)["manifest"]).get(
                "implementation_sha256"
            )
            == self.implementation_sha256
            for method in METHODS
            for episode in smoke
        )
        if not complete:
            return False
        smoke_rows = {
            method: {
                str(episode): load_json(self._paths(method, episode)["validation"])[
                    "status"
                ]
                for episode in smoke
            }
            for method in METHODS
        }
        # Configuration files are content-addressed by every trajectory.  Never
        # mutate them after export merely to record smoke outcomes; doing so
        # would leave every archive pointing at a stale hash.  Mutable run
        # status belongs only in the separate candidate manifest below.
        manifest = {
            "status": "AWAITING_HUMAN_SMOKE_REVIEW_NOT_FROZEN",
            "smoke_episode_indices": smoke,
            "smoke_execution": smoke_rows,
            "smoke_execution_integrity_pass": True,
            "config_sha256": {
                key: sha256_file(path) for key, path in self.config_paths.items()
            },
            "full_batch_allowed": False,
            "note": (
                "Smoke artifacts exist, but configuration freeze and all-50 execution "
                "require explicit human visual approval."
            ),
        }
        atomic_json(self.config_output / "smoke_review_candidate_manifest.json", manifest)
        print(
            "[CONFIG] smoke review candidate written; NOT frozen "
            f"common={manifest['config_sha256']['common']} "
            f"baseline={manifest['config_sha256']['baseline']} "
            f"proposed={manifest['config_sha256']['proposed']}",
            flush=True,
        )
        return True


__all__ = ["DollHandoffPipeline"]
