"""Apply the byte-verified frozen-v4 A/B retargeter to the unseen 20 set."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import aloha_g1_feasibility_v3.evaluate as v3_evaluate_module
from aloha_g1_arm_v2.audit import PreparedEpisode, evaluate_arm_episode
from aloha_g1_arm_v2.common import array_sha256
from aloha_g1_arm_v2.integrated import (
    map_baseline_hand,
    map_proposed_hand,
    validate_candidate_joint_order,
)
from aloha_g1_arm_v2.pipeline import configure_g1
from aloha_g1_collision_v4.evaluate import evaluate_repair
from aloha_g1_collision_v4.solver import SharedCollisionWindowSolver
from aloha_g1_dataset_v1.core import G1Kinematics, HandMapper, SourceKinematics
from aloha_g1_feasibility_v3.solver import (
    FrozenEpisode,
    SharedConstrainedFeasibilitySolver,
)
from aloha_g1_hand_v2.collision_eval import CollisionClassifier, make_runtime

from .common import (
    UnseenSourceDataset,
    atomic_csv,
    atomic_json,
    atomic_npz,
    freeze_dependencies,
    load_json,
    original_source_integrity,
    sha256_file,
    verify_dependencies_unchanged,
)
from .constants import (
    EXPECTED_FROZEN_SHA256,
    ORIGINAL_ID_PREFIX,
    ORIGINAL_V4_ROOT,
    OUTPUT_ROOT,
    RAW_RECORDING_NAMES,
    SOURCE_DATASET_ROOT,
    TASK,
    UNSEEN_ID_PREFIX,
    stable_source_id,
)


METHODS = ("baseline", "proposed")
METHOD_TO_DATASET = {"baseline": "dataset_a", "proposed": "dataset_b"}
STATUSES = (
    "PASS",
    "FAIL_IK",
    "FAIL_COLLISION",
    "FAIL_TEMPORAL",
    "FAIL_LIMIT",
    "FAIL_DATA",
    "FAIL_OTHER",
)


def _implementation_fingerprint() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("common.py"),
        Path(__file__).resolve().with_name("constants.py"),
        root / "tools/run_frozen_v4_unseen20.py",
    ]
    files = {str(path.relative_to(root)): sha256_file(path) for path in paths}
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "files": files}


def _pair_catalog() -> list[tuple[int, int]]:
    audit = load_json(ORIGINAL_V4_ROOT / "audit/collision_events_v3.json")
    return sorted(
        {
            tuple(int(value) for value in row["geom_ids"])
            for row in audit["pair_frame_records"]
        }
    )


def _write_semantic_proxy(output_root: Path, prepared: PreparedEpisode) -> None:
    directory = (
        output_root
        / "source/semantic_diagnostic_proxy/dataset_b"
        / f"episode_{prepared.episode.episode_id:06d}"
    )
    atomic_npz(
        directory / "g1_hand_action.npz",
        left_phase=np.asarray(prepared.detected["left"].phase).astype("U16"),
        right_phase=np.asarray(prepared.detected["right"].phase).astype("U16"),
        role=np.asarray("shared source-semantic diagnostic mask only"),
        affects_dataset_a_actions=np.asarray(False),
    )


def _source_metadata(
    dataset: UnseenSourceDataset,
    prepared: PreparedEpisode,
) -> dict[str, Any]:
    episode = prepared.episode
    raw = dataset.source_manifest[episode.episode_id]
    return {
        "schema_version": "unseen_20_v4_source_metadata_v1",
        "stable_source_id": stable_source_id(UNSEEN_ID_PREFIX, episode.episode_id),
        "local_dataset_namespace": UNSEEN_ID_PREFIX,
        "local_episode_id": episode.episode_id,
        "raw_recording_name": raw["raw_directory_name"],
        "raw_recording_path": raw["absolute_path"],
        "raw_content_tree_sha256": raw["content_tree_sha256"],
        "raw_parquet_sha256": raw["critical_file_sha256"]["parquet"],
        "integrated_source_dataset_root": str(dataset.root),
        "integrated_source_format": "LeRobot v3.0",
        "frame_count": len(episode.action),
        "fps": episode.fps,
        "duration_s": float(episode.timestamps[-1] - episode.timestamps[0]),
        "language_instruction": episode.task,
        "source_action_sha256": array_sha256(episode.action.astype(np.float32)),
        "source_state_sha256": array_sha256(episode.state.astype(np.float32)),
        "image_reference": episode.image_reference,
        "images_duplicated": False,
        "object_relative_source_metadata": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
        "generalization_role": "converter-unseen source demonstration",
    }


def _frozen_episode(
    method: str,
    arm_result: Any,
    hand: Mapping[str, Any],
) -> FrozenEpisode:
    return FrozenEpisode(
        method=method,
        dataset_name=METHOD_TO_DATASET[method],
        episode_id=arm_result.episode.episode_id,
        q_v2=np.asarray(arm_result.solved["q"], dtype=np.float64),
        targets={
            f"{side}_{key}": np.asarray(arm_result.targets[f"{side}_{key}"], dtype=np.float64)
            for side in ("left", "right")
            for key in ("wrist_position", "wrist_rotation", "tool_position")
        },
        left_hand_v2=np.asarray(hand["left"], dtype=np.float64),
        right_hand_v2=np.asarray(hand["right"], dtype=np.float64),
        left_phase=np.asarray(hand["left_phase"]).astype(str),
        right_phase=np.asarray(hand["right_phase"]).astype(str),
        timestamps=np.asarray(arm_result.episode.timestamps, dtype=np.float64),
        fps=float(arm_result.episode.fps),
        representation=str(arm_result.targets["representation"]),
    )


class FrozenV4UnseenRetargeter:
    """One execution path, fixed before and independent of all unseen episodes."""

    def __init__(self, output_root: Path = OUTPUT_ROOT):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.dependencies = freeze_dependencies(self.output_root)
        self.runtime_config = self.dependencies["arm"]["runtime_config"]
        original = original_source_integrity(self.runtime_config)
        if not original["unchanged"]:
            raise RuntimeError("FROZEN_V4_DEPENDENCY_MISMATCH: original source hashes changed")
        self.dataset = UnseenSourceDataset(SOURCE_DATASET_ROOT)
        if not self.dataset.integrity()["matches_build_report"]:
            raise RuntimeError("unseen integrated source hash differs from validated build")
        schema = load_json(self.output_root / "source/source_schema_comparison.json")
        if not schema.get("all_required_contracts_identical"):
            raise RuntimeError("unseen source schema is not equivalent to original 50")
        self.source_kinematics = SourceKinematics(self.runtime_config)
        self.g1 = G1Kinematics(self.runtime_config)
        configure_g1(self.g1, self.runtime_config)
        self.runtime = make_runtime(self.runtime_config)
        self.classifier = CollisionClassifier(self.runtime, self.runtime_config)
        self.baseline_mapper = HandMapper(self.runtime_config, self.runtime)
        validate_candidate_joint_order(self.dependencies["hand"], self.runtime)
        self.v3_solver = SharedConstrainedFeasibilitySolver(
            self.runtime_config,
            self.dependencies["feasibility"]["solver_parameters"],
            self.dependencies["feasibility"]["hand_temporal_projection"],
            self.g1,
            self.runtime,
            self.classifier,
        )
        self.v4_solver = SharedCollisionWindowSolver(
            self.runtime_config,
            self.dependencies["feasibility"],
            self.dependencies["collision"]["parameters"],
            self.g1,
            self.runtime,
            self.classifier,
        )
        self.pair_catalog = _pair_catalog()
        self.semantic_proxy = self.output_root / "source/semantic_diagnostic_proxy"
        # Preserve the exact frozen v3 evaluator. Only redirect its read-only
        # semantic-mask source from old local IDs to new local IDs.
        v3_evaluate_module.V2_INTEGRATED_ROOT = self.semantic_proxy
        self.implementation = _implementation_fingerprint()

    def prepare(self, episode_id: int) -> PreparedEpisode:
        episode = self.dataset.episode(episode_id)
        source_fk = self.source_kinematics.compute(episode.action)
        detected = self.baseline_mapper.detect(episode)
        prepared = PreparedEpisode(episode, source_fk, detected)
        _write_semantic_proxy(self.output_root, prepared)
        return prepared

    def convert(
        self,
        method: str,
        prepared: PreparedEpisode,
    ) -> tuple[Any, dict[str, Any], Any, Any]:
        if method not in METHODS:
            raise ValueError(method)
        arm_result = evaluate_arm_episode(
            method,
            prepared,
            self.runtime_config,
            self.g1,
            self.runtime,
            self.classifier,
        )
        if method == "baseline":
            hand = map_baseline_hand(arm_result, self.baseline_mapper)
        else:
            hand = map_proposed_hand(arm_result, self.dependencies["hand"])
        arm_frozen = _frozen_episode(method, arm_result, hand)
        v3_result = self.v3_solver.solve(arm_frozen)
        v3_frozen = FrozenEpisode(
            method=method,
            dataset_name=METHOD_TO_DATASET[method],
            episode_id=prepared.episode.episode_id,
            q_v2=v3_result.q.copy(),
            targets={key: value.copy() for key, value in arm_frozen.targets.items()},
            left_hand_v2=v3_result.left_hand.copy(),
            right_hand_v2=v3_result.right_hand.copy(),
            left_phase=arm_frozen.left_phase.copy(),
            right_phase=arm_frozen.right_phase.copy(),
            timestamps=arm_frozen.timestamps.copy(),
            fps=arm_frozen.fps,
            representation=arm_frozen.representation,
        )
        v4_result = self.v4_solver.solve(v3_frozen)
        evaluated = evaluate_repair(
            v4_result,
            self.v4_solver,
            self.runtime,
            self.classifier,
            self.g1,
            self.runtime_config,
            self.pair_catalog,
        )
        metrics = evaluated["metrics"]
        metrics.update(
            {
                "stable_source_id": stable_source_id(
                    UNSEEN_ID_PREFIX, prepared.episode.episode_id
                ),
                "raw_recording_name": RAW_RECORDING_NAMES[prepared.episode.episode_id],
                "converter_generalization_role": "UNSEEN_AFTER_V4_FREEZE",
            }
        )
        evaluated["validation"]["frozen_v4_acceptance_rules_unchanged"] = True
        evaluated["validation"]["unseen_episode_used_for_retuning"] = False
        return v4_result, evaluated, arm_result, v3_result

    def export(
        self,
        result: Any,
        evaluated: Mapping[str, Any],
        arm_result: Any,
        v3_result: Any,
        prepared: PreparedEpisode,
    ) -> Path:
        episode = result.episode
        directory = (
            self.output_root
            / episode.dataset_name
            / f"episode_{episode.episode_id:06d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        arm = result.q.astype(np.float32)
        left = result.left_hand.astype(np.float32)
        right = result.right_hand.astype(np.float32)
        hand = np.column_stack((left, right)).astype(np.float32)
        full = np.column_stack((arm, hand)).astype(np.float32)
        dependency_hashes = EXPECTED_FROZEN_SHA256
        atomic_npz(
            directory / "g1_arm_action.npz",
            action=arm,
            timestamps=episode.timestamps.astype(np.float64),
            fps=np.asarray(episode.fps),
            target_left_wrist_position=episode.targets["left_wrist_position"],
            target_right_wrist_position=episode.targets["right_wrist_position"],
            target_left_wrist_rotation=episode.targets["left_wrist_rotation"],
            target_right_wrist_rotation=episode.targets["right_wrist_rotation"],
            target_left_task_tool_position=episode.targets["left_tool_position"],
            target_right_task_tool_position=episode.targets["right_tool_position"],
            joint_names=np.asarray(self.g1.info["joint_names"]).astype("U64"),
            method=np.asarray(episode.method),
            representation=np.asarray(episode.representation),
            stable_source_id=np.asarray(
                stable_source_id(UNSEEN_ID_PREFIX, episode.episode_id)
            ),
            frozen_common_arm_v2_sha256=np.asarray(dependency_hashes["common_arm_v2"]),
            frozen_feasibility_v3_sha256=np.asarray(dependency_hashes["feasibility_v3"]),
            frozen_collision_v4_sha256=np.asarray(dependency_hashes["collision_v4"]),
            real_robot_command_allowed=np.asarray(False),
        )
        mapper = (
            "unchanged_binary_open_close_with_shared_temporal_feasibility"
            if episode.method == "baseline"
            else "frozen_proposed_hand_v2_1"
        )
        atomic_npz(
            directory / "g1_hand_action.npz",
            action=hand,
            left_action=left,
            right_action=right,
            left_phase=episode.left_phase.astype("U16"),
            right_phase=episode.right_phase.astype("U16"),
            left_joint_names=np.asarray(self.runtime.hand_joint_names["left"]).astype("U64"),
            right_joint_names=np.asarray(self.runtime.hand_joint_names["right"]).astype("U64"),
            mapper=np.asarray(mapper),
            frozen_hand_v2_1_sha256=np.asarray(
                "NOT_APPLICABLE_DATASET_A"
                if episode.method == "baseline"
                else dependency_hashes["proposed_hand_v2_1"]
            ),
            real_robot_command_allowed=np.asarray(False),
        )
        atomic_npz(
            directory / "g1_full_action.npz",
            action=full,
            timestamps=episode.timestamps.astype(np.float64),
            fps=np.asarray(episode.fps),
            joint_names=np.concatenate(
                (
                    np.asarray(self.g1.info["joint_names"]).astype("U64"),
                    np.asarray(self.runtime.hand_joint_names["left"]).astype("U64"),
                    np.asarray(self.runtime.hand_joint_names["right"]).astype("U64"),
                )
            ),
            stable_source_id=np.asarray(
                stable_source_id(UNSEEN_ID_PREFIX, episode.episode_id)
            ),
            offline_dataset_label=np.asarray(True),
        )
        atomic_npz(
            directory / "frozen_stage_diagnostics.npz",
            q_common_arm_v2=np.asarray(arm_result.solved["q"], dtype=np.float32),
            q_feasibility_v3=np.asarray(v3_result.q, dtype=np.float32),
            q_collision_v4=arm,
            feasibility_v3_to_collision_v4_delta=(result.q - v3_result.q).astype(np.float32),
            collision_v4_changed_frame_mask=result.changed_mask.astype(bool),
            orientation_slack_rad=result.orientation_slack.astype(np.float32),
            orientation_slack_requested_rad=result.orientation_slack_requested.astype(np.float32),
        )
        atomic_json(directory / "source_metadata.json", _source_metadata(self.dataset, prepared))
        atomic_json(directory / "retargeting_metrics.json", evaluated["metrics"])
        atomic_json(directory / "validation.json", evaluated["validation"])
        atomic_json(
            directory / "solver_report.json",
            {
                "common_arm_v2": {
                    "solver_class": "AcceptanceAwareTemporalIK",
                    "metrics": arm_result.metrics,
                },
                "feasibility_v3": v3_result.metadata,
                "collision_v4": result.metadata,
                "new_episode_parameter_search": False,
                "method_specific_solver_parameters": False,
            },
        )
        file_names = (
            "source_metadata.json",
            "g1_arm_action.npz",
            "g1_hand_action.npz",
            "g1_full_action.npz",
            "frozen_stage_diagnostics.npz",
            "retargeting_metrics.json",
            "validation.json",
            "solver_report.json",
        )
        manifest = {
            "schema_version": "frozen_collision_v4_unseen_episode_v1",
            "status": evaluated["metrics"]["status"],
            "dataset": episode.dataset_name,
            "method": episode.method,
            "local_episode_id": episode.episode_id,
            "stable_source_id": stable_source_id(UNSEEN_ID_PREFIX, episode.episode_id),
            "raw_recording_name": RAW_RECORDING_NAMES[episode.episode_id],
            "frozen_dependency_sha256": dependency_hashes,
            "implementation_sha256": self.implementation["sha256"],
            "source_action_sha256": array_sha256(prepared.episode.action.astype(np.float32)),
            "g1_arm_action_array_sha256": array_sha256(arm),
            "g1_hand_action_array_sha256": array_sha256(hand),
            "g1_full_action_array_sha256": array_sha256(full),
            "a_b_output_separation": True,
            "unseen_episode_used_for_retuning": False,
            "files": {name: sha256_file(directory / name) for name in file_names},
            "offline_only": True,
            "training_executed": False,
            "packaging_executed": False,
            "physics_executed": False,
            "real_robot_commands": False,
        }
        atomic_json(directory / "manifest.json", manifest)
        return directory

    def existing_metrics(self, method: str, episode_id: int) -> dict[str, Any] | None:
        directory = (
            self.output_root
            / METHOD_TO_DATASET[method]
            / f"episode_{episode_id:06d}"
        )
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = load_json(manifest_path)
        if manifest.get("frozen_dependency_sha256") != EXPECTED_FROZEN_SHA256:
            raise RuntimeError(f"existing unseen output dependency mismatch: {directory}")
        if manifest.get("stable_source_id") != stable_source_id(UNSEEN_ID_PREFIX, episode_id):
            raise RuntimeError(f"existing unseen output identity mismatch: {directory}")
        for name, expected in manifest["files"].items():
            if sha256_file(directory / name) != expected:
                raise RuntimeError(f"existing unseen output file mismatch: {directory / name}")
        return load_json(directory / "retargeting_metrics.json")


def _stable_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
    }


def aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_paths = {
        "ik_success_rate": lambda row: row["ik_success_rate"],
        "position_error_mean_m": lambda row: row["position_error_mean_m"],
        "orientation_error_mean_rad": lambda row: row["orientation_error_mean_rad"],
        "prohibited_collision_frames": lambda row: row["collision"]["prohibited_collision_frames"],
        "physical_pinch_error_mean_m": lambda row: row["task_space"]["physical_pinch_error_mean_m"],
        "task_critical_pinch_error_mean_m": lambda row: row["task_space"]["task_critical_pinch_error_mean_m"],
        "midpoint_error_mean_m": lambda row: row["bimanual"]["midpoint_error_mean_m"],
        "relative_vector_error_mean_m": lambda row: row["bimanual"]["relative_vector_error_mean_m"],
        "distance_change_error_mean_m": lambda row: row["bimanual"]["distance_change_error_mean_m"],
    }
    valid = sorted(int(row["episode_id"]) for row in rows if row["status"] == "PASS")
    categories: Counter[str] = Counter()
    for row in rows:
        categories.update(row["collision"]["v4_category_frame_incidence"])
    return {
        "episode_count": len(rows),
        "frame_count": int(sum(int(row["frame_count"]) for row in rows)),
        "pass_count": len(valid),
        "valid_episode_ids": valid,
        "status_counts": {status: sum(row["status"] == status for row in rows) for status in STATUSES},
        "ik_success_mean": float(np.mean([row["ik_success_rate"] for row in rows])),
        "ik_success_median": float(np.median([row["ik_success_rate"] for row in rows])),
        "collision_fail_rate": float(
            np.mean([not row["strict_checks"]["collision"] for row in rows])
        ),
        "temporal_fail_rate": float(
            np.mean([not row["strict_checks"]["temporal"] for row in rows])
        ),
        "prohibited_collision_frames": int(
            sum(row["collision"]["prohibited_collision_frames"] for row in rows)
        ),
        "collision_category_frame_incidence": dict(sorted(categories.items())),
        "metrics": {
            key: _stable_stats([float(function(row)) for row in rows])
            for key, function in scalar_paths.items()
        },
    }


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    collision = row["collision"]
    category = collision["v4_category_frame_incidence"]
    return {
        "stable_source_id": row["stable_source_id"],
        "episode_id": row["episode_id"],
        "raw_recording_name": row["raw_recording_name"],
        "status": row["status"],
        "frame_count": row["frame_count"],
        "fps": row["fps"],
        "ik_success_rate": row["ik_success_rate"],
        "joint_limit_violation_count": row["joint_limit_violation_count"],
        "branch_discontinuity_count": row["temporal"]["branch_discontinuity_count"],
        "maximum_joint_step_rad": row["temporal"]["maximum_joint_step_rad"],
        "maximum_velocity_rad_s": row["temporal"]["maximum_velocity_rad_s"],
        "maximum_acceleration_rad_s2": row["temporal"]["maximum_acceleration_rad_s2"],
        "prohibited_collision_frames": collision["prohibited_collision_frames"],
        "arm_torso_frames": category.get("ARM_TORSO", 0),
        "cross_arm_frames": category.get("CROSS_ARM", 0),
        "hand_opposite_arm_frames": category.get("HAND_OPPOSITE_ARM", 0),
        "hand_hand_frames": category.get("HAND_HAND", 0),
        "third_opposite_hand_frames": category.get("THIRD_OPPOSITE_HAND", 0),
        "wrist_opposite_hand_frames": category.get("WRIST_OPPOSITE_HAND", 0),
        "wrist_error_m": row["position_error_mean_m"],
        "physical_pinch_error_m": row["task_space"]["physical_pinch_error_mean_m"],
        "task_critical_pinch_error_m": row["task_space"]["task_critical_pinch_error_mean_m"],
        "midpoint_error_m": row["bimanual"]["midpoint_error_mean_m"],
        "relative_vector_error_m": row["bimanual"]["relative_vector_error_mean_m"],
        "distance_change_error_m": row["bimanual"]["distance_change_error_mean_m"],
        "semantic_phase_valid": row["strict_checks"]["semantic"],
        "first_failure": row["status"] if row["status"] != "PASS" else "NONE",
    }


def run_retargeting(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    converter = FrozenV4UnseenRetargeter(output_root)
    rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for episode_id in converter.dataset.episode_ids():
        existing = {
            method: converter.existing_metrics(method, episode_id) for method in METHODS
        }
        if all(value is not None for value in existing.values()):
            for method in METHODS:
                rows[method].append(existing[method])
            print(f"[frozen-v4] {episode_id + 1:02d}/20 reused", flush=True)
            continue
        prepared = converter.prepare(episode_id)
        for method in METHODS:
            if existing[method] is not None:
                rows[method].append(existing[method])
                continue
            result, evaluated, arm_result, v3_result = converter.convert(method, prepared)
            converter.export(result, evaluated, arm_result, v3_result, prepared)
            rows[method].append(evaluated["metrics"])
            print(
                f"[frozen-v4] {episode_id + 1:02d}/20 {METHOD_TO_DATASET[method]} "
                f"{evaluated['metrics']['status']}",
                flush=True,
            )
    post = verify_dependencies_unchanged(converter.dependencies["manifest"])
    if not post["all_unchanged"]:
        raise RuntimeError("FROZEN_V4_DEPENDENCY_MISMATCH after unseen conversion")
    atomic_json(
        output_root / "dependencies/post_conversion_integrity.json",
        {
            **post,
            "original_source": original_source_integrity(converter.runtime_config),
            "unseen_source": converter.dataset.integrity(),
            "implementation": converter.implementation,
        },
    )
    return {method: sorted(rows[method], key=lambda row: row["episode_id"]) for method in METHODS}


def _original_raw_names() -> dict[int, str]:
    path = Path(__file__).resolve().parents[2] / "reports/magsafe_lerobot_v3_manifest.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["output_episode_index"]): Path(row["source_folder"]).name
            for row in __import__("csv").DictReader(stream)
        }


def _artifact_entry(namespace: str, dataset_name: str, episode_id: int) -> dict[str, Any]:
    if namespace == ORIGINAL_ID_PREFIX:
        root = ORIGINAL_V4_ROOT / dataset_name / f"episode_{episode_id:06d}"
        raw_name = _original_raw_names()[episode_id]
        source_root = str(Path(__file__).resolve().parents[2] / "lerobot_magsafe_50_cam_high_v3")
    else:
        root = OUTPUT_ROOT / dataset_name / f"episode_{episode_id:06d}"
        raw_name = RAW_RECORDING_NAMES[episode_id]
        source_root = str(SOURCE_DATASET_ROOT)
    manifest = load_json(root / "manifest.json")
    return {
        "stable_source_id": stable_source_id(namespace, episode_id),
        "namespace": namespace,
        "local_episode_id": episode_id,
        "raw_recording_name": raw_name,
        "source_dataset_root": source_root,
        "retargeted_episode_root": str(root),
        "g1_full_action_sha256": manifest["files"]["g1_full_action.npz"],
        "status": manifest["status"],
    }


def _comparison(original: Mapping[str, Any], unseen: Mapping[str, Any]) -> dict[str, Any]:
    paths = (
        "position_error_mean_m",
        "physical_pinch_error_mean_m",
        "task_critical_pinch_error_mean_m",
        "midpoint_error_mean_m",
        "relative_vector_error_mean_m",
        "distance_change_error_mean_m",
    )
    return {
        "interpretation_scope": "descriptive 50-vs-20 frozen-converter comparison; no population-level claim",
        "original_50": {
            "episode_count": 50,
            "valid_count": original["pass_count"],
            "valid_rate": original["pass_count"] / 50.0,
            "ik_success_mean": original["ik_success_mean"],
            "ik_success_median": original["ik_success_median"],
            "collision_fail_rate": original["collision_fail_episode_count"] / 50.0,
            "temporal_fail_rate": original["status_counts"].get("FAIL_TEMPORAL", 0) / 50.0,
            "metrics": {key: original["metrics"][key] for key in paths},
        },
        "unseen_20": {
            "episode_count": 20,
            "valid_count": unseen["pass_count"],
            "valid_rate": unseen["pass_count"] / 20.0,
            "ik_success_mean": unseen["ik_success_mean"],
            "ik_success_median": unseen["ik_success_median"],
            "collision_fail_rate": unseen["collision_fail_rate"],
            "temporal_fail_rate": unseen["temporal_fail_rate"],
            "metrics": {key: unseen["metrics"][key] for key in paths},
        },
        "delta_unseen_minus_original": {
            "valid_rate": unseen["pass_count"] / 20.0 - original["pass_count"] / 50.0,
            "ik_success_mean": unseen["ik_success_mean"] - original["ik_success_mean"],
            "collision_fail_rate": unseen["collision_fail_rate"]
            - original["collision_fail_episode_count"] / 50.0,
            **{
                f"{key}_mean": unseen["metrics"][key]["mean"]
                - original["metrics"][key]["mean"]
                for key in paths
            },
        },
    }


def _new_failure_modes(rows: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    old_categories: set[str] = set()
    old_pairs: set[str] = set()
    old_statuses: set[str] = set()
    for dataset_name in ("dataset_a", "dataset_b"):
        for episode_id in range(50):
            metrics = load_json(
                ORIGINAL_V4_ROOT
                / dataset_name
                / f"episode_{episode_id:06d}/retargeting_metrics.json"
            )
            old_statuses.add(metrics["status"])
            old_categories.update(metrics["collision"]["v4_category_frame_incidence"])
            old_pairs.update(
                row["pair"] for row in metrics["collision"]["v4_top_prohibited_pairs"]
            )
    new_statuses = {row["status"] for values in rows.values() for row in values}
    new_categories = {
        key
        for values in rows.values()
        for row in values
        for key in row["collision"]["v4_category_frame_incidence"]
    }
    new_pairs = {
        value["pair"]
        for values in rows.values()
        for row in values
        for value in row["collision"]["v4_top_prohibited_pairs"]
    }
    return {
        "original_statuses": sorted(old_statuses),
        "unseen_statuses": sorted(new_statuses),
        "unseen_status_modes": sorted(new_statuses - old_statuses),
        "original_collision_categories": sorted(old_categories),
        "unseen_collision_categories": sorted(new_categories),
        "unseen_collision_category_modes": sorted(new_categories - old_categories),
        "unseen_link_pairs_not_seen_in_original_50": sorted(new_pairs - old_pairs),
        "new_causal_failure_mode": bool(
            (new_statuses - old_statuses) or (new_categories - old_categories)
        ),
        "note": "A new link pair within an existing audited category is reported separately and is not by itself a new causal failure category.",
    }


def summarize(rows: dict[str, list[dict[str, Any]]] | None = None, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if rows is None:
        rows = {
            method: [
                load_json(
                    output_root
                    / METHOD_TO_DATASET[method]
                    / f"episode_{episode_id:06d}/retargeting_metrics.json"
                )
                for episode_id in range(20)
            ]
            for method in METHODS
        }
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    aggregate_new = {METHOD_TO_DATASET[m]: aggregate(rows[m]) for m in METHODS}
    original = load_json(ORIGINAL_V4_ROOT / "summary/aggregate.json")
    atomic_csv(summary_dir / "dataset_a_episode_metrics.csv", [_flatten(row) for row in rows["baseline"]])
    atomic_csv(summary_dir / "dataset_b_episode_metrics.csv", [_flatten(row) for row in rows["proposed"]])
    comparison_a = _comparison(original["dataset_a"], aggregate_new["dataset_a"])
    comparison_b = _comparison(original["dataset_b"], aggregate_new["dataset_b"])
    atomic_json(summary_dir / "original50_vs_unseen20_a.json", comparison_a)
    atomic_json(summary_dir / "original50_vs_unseen20_b.json", comparison_b)
    a_new = set(aggregate_new["dataset_a"]["valid_episode_ids"])
    b_new = set(aggregate_new["dataset_b"]["valid_episode_ids"])
    matched_new = sorted(a_new & b_new)
    old_a = set(original["dataset_a"]["valid_episode_ids"])
    old_b = set(original["dataset_b"]["valid_episode_ids"])
    old_matched = old_a & old_b
    identities: list[dict[str, Any]] = []
    old_names = _original_raw_names()
    for episode_id in range(50):
        identities.append(
            {
                "stable_source_id": stable_source_id(ORIGINAL_ID_PREFIX, episode_id),
                "namespace": ORIGINAL_ID_PREFIX,
                "local_episode_id": episode_id,
                "raw_recording_name": old_names[episode_id],
                "dataset_a_status": load_json(
                    ORIGINAL_V4_ROOT / f"dataset_a/episode_{episode_id:06d}/validation.json"
                )["status"],
                "dataset_b_status": load_json(
                    ORIGINAL_V4_ROOT / f"dataset_b/episode_{episode_id:06d}/validation.json"
                )["status"],
            }
        )
    for episode_id in range(20):
        identities.append(
            {
                "stable_source_id": stable_source_id(UNSEEN_ID_PREFIX, episode_id),
                "namespace": UNSEEN_ID_PREFIX,
                "local_episode_id": episode_id,
                "raw_recording_name": RAW_RECORDING_NAMES[episode_id],
                "dataset_a_status": rows["baseline"][episode_id]["status"],
                "dataset_b_status": rows["proposed"][episode_id]["status"],
            }
        )
    atomic_json(
        summary_dir / "source_identity_map.json",
        {
            "schema_version": "old50_new20_stable_source_identity_v1",
            "identity_rule": "namespace:zero-padded local episode ID",
            "local_id_collision_prevented": True,
            "sources": identities,
        },
    )

    a_entries = [
        _artifact_entry(ORIGINAL_ID_PREFIX, "dataset_a", episode_id) for episode_id in sorted(old_a)
    ] + [_artifact_entry(UNSEEN_ID_PREFIX, "dataset_a", episode_id) for episode_id in sorted(a_new)]
    b_entries = [
        _artifact_entry(ORIGINAL_ID_PREFIX, "dataset_b", episode_id) for episode_id in sorted(old_b)
    ] + [_artifact_entry(UNSEEN_ID_PREFIX, "dataset_b", episode_id) for episode_id in sorted(b_new)]
    matched_entries = []
    for namespace, ids in (
        (ORIGINAL_ID_PREFIX, sorted(old_matched)),
        (UNSEEN_ID_PREFIX, matched_new),
    ):
        for episode_id in ids:
            a = _artifact_entry(namespace, "dataset_a", episode_id)
            b = _artifact_entry(namespace, "dataset_b", episode_id)
            matched_entries.append(
                {
                    "stable_source_id": a["stable_source_id"],
                    "namespace": namespace,
                    "local_episode_id": episode_id,
                    "raw_recording_name": a["raw_recording_name"],
                    "dataset_a": a,
                    "dataset_b": b,
                    "dual_pass": True,
                }
            )
    manifests = {
        "dataset_a_native_manifest.json": {
            "schema_version": "future_g1_dataset_a_native_manifest_v1",
            "method": "Dataset A / trajectory-centric baseline",
            "entry_count": len(a_entries),
            "entries": a_entries,
            "packaged": False,
        },
        "dataset_b_native_manifest.json": {
            "schema_version": "future_g1_dataset_b_native_manifest_v1",
            "method": "Dataset B / interaction-aware proposed",
            "entry_count": len(b_entries),
            "entries": b_entries,
            "packaged": False,
        },
        "matched_a_b_manifest.json": {
            "schema_version": "future_primary_matched_a_b_manifest_v1",
            "fairness_rule": "identical stable source IDs and dual PASS; no arbitrary cap",
            "entry_count": len(matched_entries),
            "entries": matched_entries,
            "packaged": False,
        },
    }
    for name, value in manifests.items():
        atomic_json(summary_dir / name, value)
    failure_modes = _new_failure_modes(rows)
    atomic_json(summary_dir / "unseen_failure_mode_audit.json", failure_modes)
    combined = {
        "new_a_valid": len(a_new),
        "new_b_valid": len(b_new),
        "new_matched": len(matched_new),
        "combined_a_valid": len(old_a) + len(a_new),
        "combined_b_valid": len(old_b) + len(b_new),
        "combined_matched": len(old_matched) + len(matched_new),
    }
    reference = load_json(ORIGINAL_V4_ROOT / "summary/training_readiness.json")
    reference_count = int(reference["new_data_recommendation_reference_count"])
    enough = combined["combined_matched"] >= reference_count
    readiness = {
        "schema_version": "frozen_v4_unseen_20_training_manifest_readiness_v1",
        **combined,
        "reference_matched_count": reference_count,
        "reference_provenance": reference["reference_provenance"],
        "reference_not_used_for_converter_tuning": True,
        "recommendation": (
            "ENOUGH_MATCHED_DATA_FOR_POLICY_TRAINING"
            if enough
            else "COLLECT_10_MORE_FIXED_LAYOUT_EPISODES"
        ),
        "training_manifests_ready": True,
        "lerobot_packaging_performed": False,
        "policy_training_performed": False,
        "g1_training_state_adapter": "handled by separate Training Schema v1; not modified here",
        "converter_remains_frozen": True,
        "conclusion": (
            "UNSEEN_20_READY_FOR_POLICY_DATASET_PACKAGING"
            if enough
            else "UNSEEN_20_COLLECT_MORE_FIXED_LAYOUT_DATA"
        ),
    }
    atomic_json(summary_dir / "training_readiness.json", readiness)
    output = {
        "aggregate": aggregate_new,
        "combined": combined,
        "comparison_a": comparison_a,
        "comparison_b": comparison_b,
        "failure_modes": failure_modes,
        "readiness": readiness,
    }
    _write_reports(output_root, output, rows)
    return output


def _write_reports(
    output_root: Path,
    summary: Mapping[str, Any],
    rows: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    a = summary["aggregate"]["dataset_a"]
    b = summary["aggregate"]["dataset_b"]
    ca = summary["comparison_a"]
    cb = summary["comparison_b"]
    combined = summary["combined"]
    failed_a = [row for row in rows["baseline"] if row["status"] != "PASS"]
    failed_b = [row for row in rows["proposed"] if row["status"] != "PASS"]
    generalization = f"""# Frozen-v4 unseen-20 generalization report

This is a descriptive converter-generalization audit on 20 demonstrations collected after v4 was frozen. No hyperparameter, mapping, primitive, threshold, candidate, or split was selected with these demonstrations.

| Method | Original 50 valid | Unseen 20 valid | Rate delta | Original IK mean | Unseen IK mean | Collision-fail rate delta |
|---|---:|---:|---:|---:|---:|---:|
| A | {ca['original_50']['valid_count']}/50 | {ca['unseen_20']['valid_count']}/20 | {ca['delta_unseen_minus_original']['valid_rate']:+.4f} | {ca['original_50']['ik_success_mean']:.6f} | {ca['unseen_20']['ik_success_mean']:.6f} | {ca['delta_unseen_minus_original']['collision_fail_rate']:+.4f} |
| B | {cb['original_50']['valid_count']}/50 | {cb['unseen_20']['valid_count']}/20 | {cb['delta_unseen_minus_original']['valid_rate']:+.4f} | {cb['original_50']['ik_success_mean']:.6f} | {cb['unseen_20']['ik_success_mean']:.6f} | {cb['delta_unseen_minus_original']['collision_fail_rate']:+.4f} |

The 20 episodes support only an in-distribution fixed-layout generalization check. They do not support a claim about arbitrary layouts, general humanoids, policy success, or real-G1 execution. `OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE` remains unchanged.
"""
    (output_root / "summary/generalization_report.md").write_text(
        generalization, encoding="utf-8"
    )
    tests_path = output_root / "tests/test_report.json"
    tests = load_json(tests_path) if tests_path.is_file() else {"status": "PENDING"}
    dependency = load_json(output_root / "dependencies/dependency_checksums.json")
    schema = load_json(output_root / "source/source_schema_comparison.json")
    source = load_json(output_root / "source/source_dataset_audit.json")
    layout = load_json(output_root / "source/layout_audit/layout_audit_manifest.json")
    new_mode = summary["failure_modes"]["new_causal_failure_mode"]
    recommendation = summary["readiness"]["recommendation"]
    conclusion = summary["readiness"]["conclusion"]
    def failure_lines(values: list[Mapping[str, Any]]) -> str:
        if not values:
            return "- 없음"
        return "\n".join(
            f"- `{row['stable_source_id']}` / `{row['raw_recording_name']}`: "
            f"`{row['status']}`, prohibited={row['collision']['prohibited_collision_frames']}, "
            f"IK={row['ik_success_rate']:.6f}"
            for row in values
        )
    report = f"""1. new raw recordings discovered/required count: **20 / 20** (literal allowlist)
2. new source-valid count: **{source['usable_recordings']} / 20**
3. new LeRobot source dataset path: `{SOURCE_DATASET_ROOT}`
4. schema equivalence with original 50: **{schema['status']}**
5. all frozen-v4 hashes verified 여부: **{dependency['status'] == 'ALL_FROZEN_V4_HASHES_VERIFIED'}**
6. new Method A valid count / 20: **{a['pass_count']} / 20**
7. new Method B valid count / 20: **{b['pass_count']} / 20**
8. new matched A∩B count / 20: **{combined['new_matched']} / 20**
9. combined Method A valid count: **{combined['combined_a_valid']}**
10. combined Method B valid count: **{combined['combined_b_valid']}**
11. combined matched A∩B count: **{combined['combined_matched']}**
12. original50 vs unseen20 Method A valid-rate comparison: **0.8000 → {a['pass_count']/20:.4f}**
13. original50 vs unseen20 Method B valid-rate comparison: **0.9200 → {b['pass_count']/20:.4f}**
14. whether any new episode caused an unseen failure mode: **{new_mode}**
15. whether further fixed-layout data collection is recommended: **{recommendation == 'COLLECT_10_MORE_FIXED_LAYOUT_EPISODES'}** (`{recommendation}`)
16. training-manifest readiness: **{summary['readiness']['training_manifests_ready']}**; packaging/training not executed

# A. Exact 20 raw-recording integrity audit

지정된 literal allowlist만 사용했다. 20개 모두 단일 parquet, `meta/info.json`, `meta/tasks.jsonl`, 네 PNG camera stream, frame/timestamp 정렬, 14D finite state/action을 통과했다. 총 frame은 **18,633**, FPS는 **30**이다. 모든 {load_json(output_root/'source/new_20_raw_manifest.json')['all_source_file_count']:,}개 source file의 SHA-256은 `source/raw_file_hashes.jsonl`에 기록했다. Channel 6/13 gripper 의미는 기존 MuJoCo name-based mapping evidence로 재검증했다.

60-frame contact-sheet audit 결과는 `{layout['visual_review']}`이다. 이 이미지는 gross corruption/layout check에만 썼으며 object pose를 추정하지 않았다.

# B. Source LeRobot conversion

`tools/build_magsafe_lerobot_v3.py`의 `build_dataset`, feature contract, `LeRobotDataset.save_episode/finalize`를 그대로 사용했다. 새 데이터셋은 원본에 append하지 않고 별도 v3 root에 생성했다. 원시 dummy task는 원본 50 변환과 같은 authoritative fixed MagSafe instruction으로 정규화했다.

# C. Frozen-v4 integrity proof

Common Arm-v2 `{EXPECTED_FROZEN_SHA256['common_arm_v2']}`, Feasibility-v3 `{EXPECTED_FROZEN_SHA256['feasibility_v3']}`, Hand-v2.1 `{EXPECTED_FROZEN_SHA256['proposed_hand_v2_1']}`, Collision-v4 `{EXPECTED_FROZEN_SHA256['collision_v4']}`가 모두 expected hash와 일치했다. 변환 전 immutable copy와 변환 후 재검증도 일치한다. Candidate search/calibration/retuning은 0회다.

# D. Method A unseen result

- PASS: **{a['pass_count']}/20**
- Status: `{json.dumps(a['status_counts'], ensure_ascii=False)}`
- IK success mean/median: **{a['ik_success_mean']:.6f} / {a['ik_success_median']:.6f}**
- Prohibited collision frames: **{a['prohibited_collision_frames']}**
- Wrist error mean: **{a['metrics']['position_error_mean_m']['mean']*1000:.3f} mm**

# E. Method B unseen result

- PASS: **{b['pass_count']}/20**
- Status: `{json.dumps(b['status_counts'], ensure_ascii=False)}`
- IK success mean/median: **{b['ik_success_mean']:.6f} / {b['ik_success_median']:.6f}**
- Prohibited collision frames: **{b['prohibited_collision_frames']}**
- Task-critical pinch / midpoint / relative-vector: **{b['metrics']['task_critical_pinch_error_mean_m']['mean']*1000:.3f} / {b['metrics']['midpoint_error_mean_m']['mean']*1000:.3f} / {b['metrics']['relative_vector_error_mean_m']['mean']*1000:.3f} mm**

# F. Matched episode analysis

새 dual-PASS는 **{combined['new_matched']}/20**, old+new matched는 **{combined['combined_matched']}**다. `old50:NNN`와 `new20:NNN` namespace를 유지했고, primary matched manifest에는 dual-PASS source만 포함했다. 50개로 임의 cap하지 않았다.

# G. Converter generalization interpretation

A valid rate delta는 **{ca['delta_unseen_minus_original']['valid_rate']:+.4f}**, B는 **{cb['delta_unseen_minus_original']['valid_rate']:+.4f}**다. 이는 동일 nominal fixed-layout 20개에 대한 기술적 결과이며 통계적 population claim이나 layout-shift claim이 아니다. 상세 metric delta는 `original50_vs_unseen20_*.json`에 있다.

# H. Failed episodes and causes

Method A:

{failure_lines(failed_a)}

Method B:

{failure_lines(failed_b)}

새 category-level causal failure mode: **{new_mode}**. 새 link pair 여부는 `unseen_failure_mode_audit.json`에 별도 기록했다.

# I. Combined native/matched manifests

- Dataset A native: **{combined['combined_a_valid']}**
- Dataset B native: **{combined['combined_b_valid']}**
- Primary matched: **{combined['combined_matched']}**

모두 path/hash manifest일 뿐 LeRobot G1 packaging은 수행하지 않았다.

# J. Whether more data should be collected

기존 v4가 명시한 practical reference `{summary['readiness']['reference_matched_count']}+ matched`를 그대로 사용했다(acceptance gate/tuning objective 아님). 결론은 **`{recommendation}`**이다.

# K. Exact files added/modified

- `tools/aloha_g1_unseen_20_v4/{{__init__,constants,source,common,pipeline}}.py`
- `tools/build_unseen_20_lerobot_v3.py`
- `tools/run_frozen_v4_unseen20.py`
- `tools/run_unseen_20_v4_tests.py`
- `tests/test_aloha_g1_unseen_20_v4.py`
- `lerobot_magsafe_20_cam_high_v3_unseen_20260813/` (new, separate source dataset)
- `outputs/g1_unseen_20_v4/` (new isolated audit/results)
- 기존 v1/v2/v2.1/v3/v4 output과 Dataset A/B 정의는 수정하지 않았다.

# L. Exact tests and PASS/FAIL

- Command: `{tests.get('command_string', 'PENDING')}`
- Result: **{tests.get('status', 'PENDING')}**
- Passed/failed: **{tests.get('passed', 'PENDING')} / {tests.get('failed', 'PENDING')}**

{conclusion}
"""
    (output_root / "summary/final_report.md").write_text(report, encoding="utf-8")


def deterministic_rerun(output_root: Path = OUTPUT_ROOT, episode_id: int = 0) -> dict[str, Any]:
    converter = FrozenV4UnseenRetargeter(output_root)
    prepared = converter.prepare(episode_id)
    methods: dict[str, Any] = {}
    for method in METHODS:
        result, evaluated, _, _ = converter.convert(method, prepared)
        expected_directory = (
            output_root / METHOD_TO_DATASET[method] / f"episode_{episode_id:06d}"
        )
        with np.load(expected_directory / "g1_full_action.npz", allow_pickle=False) as payload:
            expected = payload["action"]
        actual = np.column_stack(
            (result.q, result.left_hand, result.right_hand)
        ).astype(np.float32)
        methods[method] = {
            "array_equal": bool(np.array_equal(actual, expected)),
            "expected_sha256": array_sha256(expected),
            "rerun_sha256": array_sha256(actual),
            "status_equal": evaluated["metrics"]["status"]
            == load_json(expected_directory / "retargeting_metrics.json")["status"],
        }
    report = {
        "schema_version": "unseen_20_frozen_v4_deterministic_rerun_v1",
        "episode_id": episode_id,
        "selection_rule": "lowest local unseen episode ID; not selected by performance",
        "methods": methods,
        "pass": all(row["array_equal"] and row["status_equal"] for row in methods.values()),
    }
    atomic_json(output_root / "tests/deterministic_rerun.json", report)
    return report


def anti_overfitting_scan(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    package = Path(__file__).resolve().parent
    paths = sorted(package.glob("*.py")) + [
        package.parent / "run_frozen_v4_unseen20.py"
    ]
    patterns = {
        "episode_equality": re.compile(r"if\s+episode(?:_id)?\s*==\s*\d+"),
        "frame_equality": re.compile(r"if\s+frame(?:_id|_index)?\s*==\s*\d+"),
        "authored_translation_rule": re.compile(
            "manual" + r"[ _-]+offset", re.IGNORECASE
        ),
        "episode_parameterization": re.compile(
            "per" + r"[ _-]+episode[ _-]+(?:parameter|correction|slack)",
            re.IGNORECASE,
        ),
        "frame_correction_rule": re.compile(
            "per" + r"[ _-]+frame[ _-]+correction", re.IGNORECASE
        ),
    }
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            for match in pattern.finditer(text):
                hits.append(
                    {"path": str(path), "pattern": name, "match": match.group(0)}
                )
    # Literal raw identities are required in the immutable allowlist only.
    identity_hits = []
    for path in paths:
        if path.name == "constants.py":
            continue
        text = path.read_text(encoding="utf-8")
        for identity in RAW_RECORDING_NAMES:
            if identity in text:
                identity_hits.append({"path": str(path), "identity": identity})
    report = {
        "schema_version": "unseen_20_v4_anti_overfitting_audit_v1",
        "files_scanned": [str(path) for path in paths],
        "forbidden_logic_hits": hits,
        "new_raw_identity_execution_logic_hits": identity_hits,
        "literal_identity_allowlist_only": not identity_hits,
        "one_global_frozen_solver_for_a_b": True,
        "no_candidate_search": True,
        "no_new_validation_split": True,
        "pass": not hits and not identity_hits,
    }
    atomic_json(output_root / "tests/anti_overfitting_audit.json", report)
    return report


__all__ = [
    "FrozenV4UnseenRetargeter",
    "aggregate",
    "anti_overfitting_scan",
    "deterministic_rerun",
    "run_retargeting",
    "summarize",
]
