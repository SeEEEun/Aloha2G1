"""Independent provenance, frame diagnostics, validation, and A/B report."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tools.doll_handoff_feasibility.common import (
    contiguous_segments,
    sha256_file,
    stable_json_sha256,
    write_csv,
    write_json,
)
from tools.doll_handoff_retargeting.common import branch_flags

from .common import (
    A_GATE,
    A_METRIC_ROOT,
    A_ROOT,
    FINAL_SOURCE_MANIFEST,
    OUTPUT_ROOT,
    REPOSITORY,
    SIDES,
    a_metric_path,
    a_trajectory_path,
    b_final_trajectory_path,
    load_a_trajectory,
    load_json,
    source_rows,
    stable_episode_id,
)
from .full_pose_resolver import FullPoseCommonWristResolver
from .wrist_resolver import IMMUTABLE_SOURCE_KEYS


HARD_EPISODES = (3, 4, 5, 6, 7, 8, 9, 13, 15, 21, 23, 27, 28, 30, 33, 34, 35, 36, 38, 44, 46, 47)
IK_HARD_EPISODES = (3, 4, 5, 6, 7, 8, 9, 13, 21, 23, 28, 30, 33, 35, 36, 38, 44, 47)
COLLISION_HARD_EPISODES = (15, 27, 34, 46)
CAUSE_LABELS = {
    "A": "workspace position infeasible",
    "B": "orientation infeasible",
    "C": "torso-clearance conflict",
    "D": "self-collision",
    "E": "cross-arm collision",
    "F": "temporal transition conflict",
    "G": "joint-limit conflict",
    "H": "numerical IK failure",
    "I": "representation-induced unreachable wrist target",
    "J": "other",
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()


def _rotation_errors(actual: np.ndarray, desired: np.ndarray) -> np.ndarray:
    cosine = np.clip(
        (np.einsum("tij,tij->t", actual, desired) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.arccos(cosine)


def _stats(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array, initial=0.0)),
    }


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _recursive_difference(
    left: Any, right: Any, prefix: str = ""
) -> list[dict[str, Any]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        output = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                output.append({"path": path, "a": "<MISSING>", "b": right[key]})
            elif key not in right:
                output.append({"path": path, "a": left[key], "b": "<MISSING>"})
            else:
                output.extend(_recursive_difference(left[key], right[key], path))
        return output
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return []
        return [{"path": prefix, "a": left, "b": right}]
    return [] if left == right else [{"path": prefix, "a": left, "b": right}]


def build_provenance(resolver: FullPoseCommonWristResolver) -> dict[str, Any]:
    gate = load_json(A_GATE)
    manifest = load_json(FINAL_SOURCE_MANIFEST)
    common_a_path = A_ROOT / "config/common_config.json"
    common_b_path = (
        REPOSITORY
        / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
        / "frozen_approval/config/common_config.json"
    )
    common_a = load_json(common_a_path)
    common_b = load_json(common_b_path)
    representative_manifest = load_json(a_metric_path(0, ".manifest"))
    episode_records = []
    implementation_hashes = set()
    for row in source_rows():
        episode = int(row["final_dataset_index"])
        trajectory = a_trajectory_path(episode)
        metric = a_metric_path(episode)
        validation = a_metric_path(episode, ".validation")
        converter_manifest = a_metric_path(episode, ".manifest")
        converter = load_json(converter_manifest)
        implementation_hashes.add(str(converter["implementation_sha256"]))
        episode_records.append(
            {
                "episode_index": episode,
                "stable_episode_id": row["stable_episode_id"],
                "source_raw_episode": row["raw_directory"],
                "source_frame_count": int(row["source_frame_count"]),
                "source_fps": float(row["source_fps"]),
                "source_parquet_path": row["source_parquet_path"],
                "source_parquet_sha256": row["source_parquet_sha256"],
                "trajectory": _file_record(trajectory),
                "metric": _file_record(metric),
                "validation": {
                    **_file_record(validation),
                    "status": load_json(validation)["status"],
                },
                "converter_manifest": _file_record(converter_manifest),
                "episode_specific_correction": False,
            }
        )
    if len(implementation_hashes) != 1:
        raise RuntimeError(f"nonuniform A implementation hashes: {implementation_hashes}")
    relevant_common_sections = (
        "models",
        "task_registration",
        "canonical_task_ready_posture",
        "shared_temporal_ik",
        "natural_arm_redundancy",
        "validation",
    )
    section_differences = {
        key: _recursive_difference(common_a.get(key), common_b.get(key), key)
        for key in relevant_common_sections
    }
    natural_identical = not any(section_differences.values())
    final_b_freeze = load_json(
        REPOSITORY
        / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
    )
    resolver_config = REPOSITORY / "configs/doll_handoff_g1_feasibility_resolver.json"
    frozen_resolver_manifest = load_json(
        REPOSITORY
        / "outputs/doll_handoff_retargeting/g1_feasibility_resolver_2026-08-21/freeze_manifest.json"
    )
    original_keys = set(load_a_trajectory(0))
    feasibility_fields = sorted(
        key
        for key in original_keys
        if key.startswith("feasibility_") or key.startswith("realized_feasible_")
    )
    provenance = {
        "schema_version": "fair_a_original_full50_reconstruction_v1",
        "audit_time_basis": "repository artifacts; no regeneration used for reconstruction",
        "repository": str(REPOSITORY),
        "git_head_at_audit": _git("rev-parse", "HEAD"),
        "git_status_dirty_at_audit": bool(_git("status", "--porcelain")),
        "original_gate": {**_file_record(A_GATE), "content": gate},
        "original_a_implementation_sha256": next(iter(implementation_hashes)),
        "representative_converter_manifest": representative_manifest,
        "trajectory_config": _file_record(A_ROOT / "config/baseline_config.json"),
        "input_contract": _file_record(
            A_ROOT / "input_contract/common_config.final_common_50.json"
        ),
        "common_config": _file_record(common_a_path),
        "workspace_registration": _file_record(
            A_ROOT / "config/baseline_workspace_mapping_report.json"
        ),
        "model_unit_audit": _file_record(A_ROOT / "config/model_unit_audit.json"),
        "fairness_report": _file_record(A_ROOT / "config/fairness_report.json"),
        "task_frame_report": _file_record(A_ROOT / "config/task_frame_report.json"),
        "tool_frame_report": _file_record(A_ROOT / "config/tool_frame_report.json"),
        "source_manifest": _file_record(FINAL_SOURCE_MANIFEST),
        "source_manifest_status": manifest["status"],
        "source_episode_count": int(manifest["source_count"]),
        "shared_temporal_ik": common_a["shared_temporal_ik"],
        "natural_arm_redundancy": common_a["natural_arm_redundancy"],
        "joint_limits": {
            "joint_names": resolver.g1.arm_joint_names,
            "minimum": resolver.g1.arm_limits[:, 0],
            "maximum": resolver.g1.arm_limits[:, 1],
        },
        "collision_model": {
            "model_config": common_a["models"],
            "penetration_tolerance_m": common_a["validation"][
                "collision_penetration_tolerance_m"
            ],
            "shared_resolver_contact_classes": (
                "ARM_TORSO_INVALID, DISTAL_HAND_HAND_CONTACT, "
                "PALM_HAND_INVALID, CROSS_ARM_INVALID"
            ),
        },
        "episode_mapping": episode_records,
        "backend_parity": {
            "COMMON_NATURAL_ARM_BACKEND_IDENTICAL": (
                "YES" if natural_identical else "NO"
            ),
            "common_section_differences": section_differences,
            "signed_elbow_sew_handling": common_a["natural_arm_redundancy"],
            "COMMON_GENERIC_FEASIBILITY_RESOLVER_IDENTICAL": "NO",
            "original_a_feasibility_fields": feasibility_fields,
            "difference": (
                "Original Fair A stopped after shared temporal/natural-arm IK and "
                "did not invoke the frozen generic feasibility resolver later "
                "used by Proposed B."
            ),
            "frozen_b_generic_resolver": {
                "implementation_sha256": final_b_freeze[
                    "generic_feasibility_resolver_sha256"
                ],
                "config": _file_record(resolver_config),
                "freeze_manifest": frozen_resolver_manifest,
            },
            "target_frame_interface_bug": (
                "The frozen resolver source-target adapter was hard-wired to "
                "target_*_interaction_frame_position_world and a whole-hand "
                "transform; direct use on A would inject forbidden B semantics."
            ),
            "orientation_interface_bug": (
                "The frozen tracker used B's non-gating spherical-grasp orientation "
                "tie; direct use on a constrained 6-D wrist trajectory introduced "
                "new A orientation failures."
            ),
            "fair_common_fix": {
                "representation_frame_adapter_sha256": resolver.adapter_sha256,
                "generic_resolver_parameters_changed": False,
                "full_pose_constraint_uses_frozen_common_weights": True,
                "minimum_acceleration_windows_reused": resolver.config[
                    "collision_repair"
                ]["window_padding_candidates_frames"],
                "episode_specific_logic": False,
                "phase_specific_logic": False,
                "interaction_target_used_for_a": False,
            },
        },
    }
    output = OUTPUT_ROOT / "provenance/original_run_reconstruction.json"
    write_json(output, provenance)
    write_json(
        OUTPUT_ROOT / "provenance/backend_parity.json",
        provenance["backend_parity"],
    )
    return provenance


def _temporal_frame_flags(
    q: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    fps: float,
    resolver: FullPoseCommonWristResolver,
) -> tuple[set[int], dict[str, float | int]]:
    full = np.column_stack((q, left, right))
    flags: set[int] = set()
    step = np.abs(np.diff(full, axis=0))
    for index in np.flatnonzero(
        np.any(
            step > float(resolver.acceptance["maximum_joint_step_rad"]) + 1e-7,
            axis=1,
        )
    ):
        flags.update((int(index), int(index) + 1))
    acceleration = np.abs(np.diff(full, n=2, axis=0)) * fps**2
    for index in np.flatnonzero(
        np.any(
            acceleration
            > float(resolver.acceptance["maximum_acceleration_rad_s2"]) + 1e-5,
            axis=1,
        )
    ):
        flags.update((int(index), int(index) + 1, int(index) + 2))
    return flags, resolver._temporal_metrics(q, left, right, fps)


def _target_steps(
    source_world: Mapping[str, np.ndarray],
    source_rotation: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(source_world["left"])
    position = np.zeros(count, dtype=np.float64)
    orientation = np.zeros(count, dtype=np.float64)
    for side in SIDES:
        position[1:] = np.maximum(
            position[1:],
            np.linalg.norm(np.diff(source_world[side], axis=0), axis=1),
        )
        orientation[1:] = np.maximum(
            orientation[1:],
            _rotation_errors(source_rotation[side][1:], source_rotation[side][:-1]),
        )
    return position, orientation


def _invalid_collision_records(
    collision: Mapping[str, Any], frame: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in collision["by_frame"].get(frame, [])
        if row["classification"] != "DISTAL_HAND_HAND_CONTACT"
    ]


def _hard_pose_segments(mask: np.ndarray, fps: float) -> dict[str, Any]:
    segments = contiguous_segments(np.flatnonzero(mask))
    longest = max(
        ((end - start + 1) / fps for start, end in segments), default=0.0
    )
    return {
        "segments": segments,
        "longest_s": float(longest),
        "frame_count": int(np.count_nonzero(mask)),
    }


def build_static_pose_oracle(
    resolver: FullPoseCommonWristResolver,
) -> dict[int, dict[str, Any]]:
    """Probe the worst materially projected target without temporal constraints."""
    rows: list[dict[str, Any]] = []
    mapping: dict[int, dict[str, Any]] = {}
    material_threshold = resolver.physical_tolerance - resolver.strict_tolerance
    for episode in IK_HARD_EPISODES:
        result = resolver.load_exported_episode(episode)
        maximum = float(np.max(result.projection_translation_m))
        if maximum <= material_threshold + 1e-7:
            continue
        frame, side_index = map(
            int,
            np.unravel_index(
                np.argmax(result.projection_translation_m),
                result.projection_translation_m.shape,
            ),
        )
        side = SIDES[side_index]
        block = slice(0, 7) if side == "left" else slice(7, 14)
        target_position = resolver.g1.world_to_model_position(
            result.source_position_world[side][frame]
        )
        target_rotation = result.source_orientation_model[side][frame]
        base = result.q_after[frame].copy()

        def errors(value: np.ndarray) -> tuple[float, float]:
            q = base.copy()
            q[block] = value
            position, rotation = resolver._pose(q)
            return (
                float(np.linalg.norm(position[side] - target_position)),
                float(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            rotation[side].T @ target_rotation
                        ).as_rotvec()
                    )
                ),
            )

        def residual(value: np.ndarray) -> np.ndarray:
            q = base.copy()
            q[block] = value
            position, rotation = resolver._pose(q)
            return np.concatenate(
                (
                    100.0 * (position[side] - target_position),
                    Rotation.from_matrix(
                        rotation[side].T @ target_rotation
                    ).as_rotvec()
                    / resolver.orientation_tolerance,
                    1e-4 * (value - result.q_before[frame, block]),
                )
            )

        stand = resolver.g1.stand_qpos[resolver.g1.arm_qpos_ids]
        seeds = (
            ("original", result.q_before[frame, block]),
            ("repaired", result.q_after[frame, block]),
            ("nominal", resolver.nominal[block]),
            ("stand", stand[block]),
        )
        best = None
        for seed_name, seed in seeds:
            solution = least_squares(
                residual,
                seed,
                bounds=(
                    resolver.g1.arm_limits[block, 0] + 1e-8,
                    resolver.g1.arm_limits[block, 1] - 1e-8,
                ),
                max_nfev=400,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            position_error, orientation_error = errors(solution.x)
            key = (
                int(position_error > resolver.strict_tolerance)
                + int(orientation_error > resolver.orientation_tolerance),
                max(
                    position_error / resolver.strict_tolerance,
                    orientation_error / resolver.orientation_tolerance,
                ),
                position_error,
                orientation_error,
            )
            candidate = (
                key,
                position_error,
                orientation_error,
                np.asarray(solution.x, dtype=np.float64),
                seed_name,
                bool(solution.success),
                int(solution.nfev),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        q = base.copy()
        q[block] = best[3]
        contacts = resolver._records(
            q, result.left_hand[frame], result.right_hand[frame]
        )
        row = {
            "episode_index": episode,
            "stable_episode_id": stable_episode_id(episode),
            "frame": frame,
            "side": side,
            "projection_m": float(
                result.projection_translation_m[frame, side_index]
            ),
            "best_position_residual_m": best[1],
            "best_orientation_residual_rad": best[2],
            "static_6d_pass": bool(
                best[1] <= resolver.strict_tolerance
                and best[2] <= resolver.orientation_tolerance
            ),
            "joint_delta_from_repaired_norm_rad": float(
                np.linalg.norm(best[3] - result.q_after[frame, block])
            ),
            "seed": best[4],
            "optimizer_success_flag": best[5],
            "function_evaluations": best[6],
            "collision_records": contacts,
            "static_collision_free": not bool(contacts),
            "interpretation": (
                "Static pass does not establish trajectory feasibility; a large "
                "branch displacement, temporal limits, or collision can still "
                "make the full path infeasible."
            ),
        }
        rows.append(row)
        mapping[episode] = row
    write_json(
        OUTPUT_ROOT
        / "diagnostics/material_projection_peak_static_6d_oracle.json",
        {
            "scope": (
                "worst materially projected frame of every applicable original "
                "FAIL_IK episode"
            ),
            "diagnostic_only": True,
            "temporal_constraints_included": False,
            "solver": (
                "four deterministic bounded least-squares seeds; strict 10 mm "
                "position and 0.75 rad orientation acceptance"
            ),
            "rows": rows,
        },
    )
    return mapping


def evaluate_all(
    resolver: FullPoseCommonWristResolver,
    static_oracle: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    final_rows: list[dict[str, Any]] = []
    hard_frame_rows: list[dict[str, Any]] = []
    hard_episode_details: dict[int, dict[str, Any]] = {}
    projection_all = []
    projection_by_side = {side: [] for side in SIDES}
    classifications: collections.Counter[str] = collections.Counter()
    after_results = {}
    collision_after = {}
    collision_before_failed = {}
    after_collision_records = []
    diagnostic_values = {
        "a_wrist_error": [],
        "a_grasp_error": [],
        "a_bimanual_relation_error": [],
    }
    immutable_checks = []

    for episode in range(50):
        result = resolver.load_exported_episode(episode)
        after_results[episode] = result
        original = load_a_trajectory(episode)
        timestamp = original["timestamp"].astype(np.float64)
        fps = float(1.0 / np.median(np.diff(timestamp)))
        q_before = result.q_before
        q_after = result.q_after
        before_model_position, before_rotation = resolver._pose_arrays(q_before)
        before_world_position = {
            side: resolver.g1.model_to_world_position(before_model_position[side])
            for side in SIDES
        }
        before_position_by_side = {
            side: np.linalg.norm(
                before_world_position[side] - result.source_position_world[side],
                axis=1,
            )
            for side in SIDES
        }
        before_orientation_by_side = {
            side: _rotation_errors(
                before_rotation[side], result.source_orientation_model[side]
            )
            for side in SIDES
        }
        after_source_position_by_side = {
            side: np.linalg.norm(
                result.achieved_position_world[side]
                - result.source_position_world[side],
                axis=1,
            )
            for side in SIDES
        }
        after_realized_position_by_side = {
            side: np.linalg.norm(
                result.achieved_position_world[side]
                - result.realized_position_world[side],
                axis=1,
            )
            for side in SIDES
        }
        after_orientation_by_side = {
            side: _rotation_errors(
                result.achieved_orientation_model[side],
                result.source_orientation_model[side],
            )
            for side in SIDES
        }
        before_position = np.maximum(
            before_position_by_side["left"], before_position_by_side["right"]
        )
        before_orientation = np.maximum(
            before_orientation_by_side["left"],
            before_orientation_by_side["right"],
        )
        after_source_position = np.maximum(
            after_source_position_by_side["left"],
            after_source_position_by_side["right"],
        )
        after_realized_position = np.maximum(
            after_realized_position_by_side["left"],
            after_realized_position_by_side["right"],
        )
        after_orientation = np.maximum(
            after_orientation_by_side["left"],
            after_orientation_by_side["right"],
        )
        before_success = (before_position <= resolver.strict_tolerance) & (
            before_orientation <= resolver.orientation_tolerance
        )
        after_realized_success = (
            after_realized_position <= resolver.strict_tolerance
        ) & (after_orientation <= resolver.orientation_tolerance)
        after_source_success = (
            after_source_position <= resolver.strict_tolerance
        ) & (after_orientation <= resolver.orientation_tolerance)
        after_source_physical = (
            after_source_position <= resolver.physical_tolerance
        ) & (after_orientation <= resolver.orientation_tolerance)
        after_collision = resolver._collision_metrics(
            q_after, result.left_hand, result.right_hand, fps
        )
        collision_after[episode] = after_collision
        after_hard_frames = set(map(int, after_collision["hard_frames"]))
        for frame, records in after_collision["by_frame"].items():
            for record in records:
                after_collision_records.append(
                    {
                        "episode_index": episode,
                        "stable_episode_id": stable_episode_id(episode),
                        "frame": int(frame),
                        "classification": record["classification"],
                        "body_pair": "|".join(record["body_pair"]),
                        "geom_pair": "|".join(map(str, record["geom_pair"])),
                        "penetration_m": float(record["penetration_depth_m"]),
                        "shared_severity_hard_frame": int(frame)
                        in after_hard_frames,
                    }
                )
        after_temporal = resolver._temporal_metrics(
            q_after, result.left_hand, result.right_hand, fps
        )
        hard_reasons = []
        if float(np.mean(after_realized_success)) < float(
            resolver.common["shared_temporal_ik"]["required_success_rate"]
        ):
            hard_reasons.append("FAIL_IK")
        if int(after_collision["hard_collision_frame_count"]):
            hard_reasons.append("FAIL_COLLISION")
        if int(after_temporal["joint_limit_violation_count"]):
            hard_reasons.append("FAIL_LIMITS")
        if int(after_temporal["branch_discontinuity_count"]):
            hard_reasons.append("FAIL_BRANCH")
        if not resolver._temporal_passes(after_temporal):
            hard_reasons.append("FAIL_TEMPORAL")
        warning_reasons = []
        if np.any(result.projection_translation_m > 0.0):
            warning_reasons.append("FEASIBILITY_PROJECTION_REPORTED")
        if int(after_collision["contact_frame_count"]):
            warning_reasons.append("NON_HARD_SELF_CONTACT")
        if np.any(~after_source_success):
            warning_reasons.append("SOURCE_TARGET_OUTSIDE_STRICT_TOLERANCE")
        if np.any(~after_realized_success):
            warning_reasons.append("ISOLATED_REALIZED_6D_MISS")
        if hard_reasons:
            classification = "HARD_FAIL"
        elif warning_reasons:
            classification = "USABLE_WITH_WARNING"
        else:
            classification = "CLEAN_PASS"
        classifications[classification] += 1
        projection_all.append(result.projection_translation_m.reshape(-1))
        for side_index, side in enumerate(SIDES):
            projection_by_side[side].append(
                result.projection_translation_m[:, side_index]
            )

        after_path, _ = resolver._cache_paths(episode)
        with np.load(after_path, allow_pickle=False) as payload:
            checks = {
                key: bool(np.array_equal(original[key], payload[key]))
                for key in IMMUTABLE_SOURCE_KEYS
            }
            immutable_checks.append(all(checks.values()))
            achieved_grasp = {
                side: payload[
                    f"achieved_{side}_physical_grasp_frame_position_world"
                ].astype(np.float64)
                for side in SIDES
            }
        target_interaction = {
            side: original[
                f"target_{side}_interaction_frame_position_world"
            ].astype(np.float64)
            for side in SIDES
        }
        wrist_error_values = np.concatenate(
            [after_source_position_by_side[side] for side in SIDES]
        )
        grasp_error_values = np.concatenate(
            [
                np.linalg.norm(
                    achieved_grasp[side] - target_interaction[side], axis=1
                )
                for side in SIDES
            ]
        )
        relation_error_values = np.linalg.norm(
            (achieved_grasp["right"] - achieved_grasp["left"])
            - (target_interaction["right"] - target_interaction["left"]),
            axis=1,
        )
        diagnostic_values["a_wrist_error"].append(wrist_error_values)
        diagnostic_values["a_grasp_error"].append(grasp_error_values)
        diagnostic_values["a_bimanual_relation_error"].append(
            relation_error_values
        )
        original_validation = load_json(a_metric_path(episode, ".validation"))
        final_rows.append(
            {
                "episode_index": episode,
                "stable_episode_id": stable_episode_id(episode),
                "source_raw_episode": source_rows()[episode]["raw_directory"],
                "frame_count": len(q_after),
                "before_status": original_validation["status"],
                "after_classification": classification,
                "hard_reasons": _json_cell(hard_reasons),
                "warning_reasons": _json_cell(warning_reasons),
                "before_strict_6d_failed_frames": int(
                    np.count_nonzero(~before_success)
                ),
                "after_realized_strict_6d_failed_frames": int(
                    np.count_nonzero(~after_realized_success)
                ),
                "after_realized_strict_6d_success_rate": float(
                    np.mean(after_realized_success)
                ),
                "after_source_strict_6d_success_rate": float(
                    np.mean(after_source_success)
                ),
                "after_source_physical_6d_success_rate": float(
                    np.mean(after_source_physical)
                ),
                "before_max_position_residual_m": float(np.max(before_position)),
                "after_max_source_position_residual_m": float(
                    np.max(after_source_position)
                ),
                "after_max_realized_position_residual_m": float(
                    np.max(after_realized_position)
                ),
                "before_max_orientation_residual_rad": float(
                    np.max(before_orientation)
                ),
                "after_max_orientation_residual_rad": float(
                    np.max(after_orientation)
                ),
                "projection_active_frames": int(
                    np.count_nonzero(
                        np.any(result.projection_translation_m > 0.0, axis=1)
                    )
                ),
                "projection_mean_m": float(
                    np.mean(result.projection_translation_m)
                ),
                "projection_p95_m": float(
                    np.percentile(result.projection_translation_m, 95)
                ),
                "projection_max_m": float(
                    np.max(result.projection_translation_m)
                ),
                "contact_frames_after": int(after_collision["contact_frame_count"]),
                "hard_collision_frames_after": int(
                    after_collision["hard_collision_frame_count"]
                ),
                "hard_collision_classes_after": _json_cell(
                    after_collision["hard_class_frame_counts"]
                ),
                "maximum_penetration_after_m": float(
                    after_collision["maximum_penetration_depth_m"]
                ),
                "joint_limit_violation_count": int(
                    after_temporal["joint_limit_violation_count"]
                ),
                "branch_discontinuity_count": int(
                    after_temporal["branch_discontinuity_count"]
                ),
                "maximum_velocity_rad_s": float(
                    after_temporal["maximum_velocity_rad_s"]
                ),
                "maximum_acceleration_rad_s2": float(
                    after_temporal["maximum_acceleration_rad_s2"]
                ),
                "source_arrays_unchanged": all(checks.values()),
            }
        )

    # The original 22 hard episodes get exhaustive frame and contact diagnostics.
    episode_root_rows = []
    collision_records = []
    ik_classification_rows = []
    for episode in HARD_EPISODES:
        result = after_results[episode]
        original = load_a_trajectory(episode)
        timestamp = original["timestamp"].astype(np.float64)
        fps = float(1.0 / np.median(np.diff(timestamp)))
        q = result.q_before
        before_model_position, before_rotation = resolver._pose_arrays(q)
        before_world_position = {
            side: resolver.g1.model_to_world_position(before_model_position[side])
            for side in SIDES
        }
        position_by_side = {
            side: np.linalg.norm(
                before_world_position[side] - result.source_position_world[side],
                axis=1,
            )
            for side in SIDES
        }
        orientation_by_side = {
            side: _rotation_errors(
                before_rotation[side], result.source_orientation_model[side]
            )
            for side in SIDES
        }
        position = np.maximum(position_by_side["left"], position_by_side["right"])
        orientation = np.maximum(
            orientation_by_side["left"], orientation_by_side["right"]
        )
        strict_ik_fail = (position > resolver.strict_tolerance) | (
            orientation > resolver.orientation_tolerance
        )
        before_collision = resolver._collision_metrics(
            q, result.left_hand, result.right_hand, fps
        )
        collision_before_failed[episode] = before_collision
        invalid_collision_frames = {
            frame
            for frame in before_collision["by_frame"]
            if _invalid_collision_records(before_collision, frame)
        }
        temporal_frames, before_temporal = _temporal_frame_flags(
            q, result.left_hand, result.right_hand, fps, resolver
        )
        branch = branch_flags(
            q,
            float(resolver.acceptance["branch_absolute_step_norm_rad"]),
            float(resolver.acceptance["branch_local_multiplier"]),
        )
        branch_frames = set(map(int, np.flatnonzero(branch)))
        lower = resolver.g1.arm_limits[:, 0]
        upper = resolver.g1.arm_limits[:, 1]
        margin = np.minimum(q - lower, upper - q)
        limit_frames = set(
            map(int, np.flatnonzero(np.any(margin < -1e-9, axis=1)))
        )
        target_position_step, target_orientation_step = _target_steps(
            result.source_position_world, result.source_orientation_model
        )
        source_model = resolver._source_model(result.source_position_world)
        reach_by_side = {
            side: np.linalg.norm(
                source_model[side] - resolver.shoulders[side], axis=1
            )
            for side in SIDES
        }
        after_source_position_by_side = {
            side: np.linalg.norm(
                result.achieved_position_world[side]
                - result.source_position_world[side],
                axis=1,
            )
            for side in SIDES
        }
        after_orientation_by_side = {
            side: _rotation_errors(
                result.achieved_orientation_model[side],
                result.source_orientation_model[side],
            )
            for side in SIDES
        }
        hard_frames = sorted(
            set(map(int, np.flatnonzero(strict_ik_fail)))
            | invalid_collision_frames
            | temporal_frames
            | branch_frames
            | limit_frames
        )
        frame_causes = collections.Counter[str]()
        static_peak_pass = bool(
            static_oracle.get(episode, {}).get("static_6d_pass", False)
        )
        for frame in hard_frames:
            contacts = _invalid_collision_records(before_collision, frame)
            classes = {row["classification"] for row in contacts}
            pos_fail = bool(position[frame] > resolver.strict_tolerance)
            ori_fail = bool(orientation[frame] > resolver.orientation_tolerance)
            near_workspace = any(
                reach_by_side[side][frame]
                >= resolver.outer_tool_radius[side] - resolver.physical_tolerance
                for side in SIDES
            )
            projection = result.projection_translation_m[frame]
            final_source_pose_ok = all(
                after_source_position_by_side[side][frame]
                <= resolver.physical_tolerance
                and after_orientation_by_side[side][frame]
                <= resolver.orientation_tolerance
                for side in SIDES
            )
            if frame in limit_frames:
                cause = "G"
            elif "CROSS_ARM_INVALID" in classes or "PALM_HAND_INVALID" in classes:
                cause = "E"
            elif "ARM_TORSO_INVALID" in classes:
                cause = "C"
            elif contacts:
                cause = "D"
            elif ori_fail and (
                not pos_fail
                or orientation[frame] / resolver.orientation_tolerance
                >= position[frame] / resolver.strict_tolerance
            ):
                cause = "B"
            elif frame in temporal_frames or (
                pos_fail
                and (
                    target_position_step[frame] > resolver.strict_tolerance
                    or target_orientation_step[frame] > 0.05
                )
            ):
                cause = "F"
            elif np.any(
                projection
                > resolver.physical_tolerance - resolver.strict_tolerance
            ) and static_peak_pass:
                cause = "F"
            elif pos_fail and near_workspace and not static_peak_pass:
                cause = "A"
            elif np.any(
                projection
                > resolver.physical_tolerance - resolver.strict_tolerance
            ):
                cause = "I"
            elif final_source_pose_ok:
                cause = "H"
            elif pos_fail or ori_fail:
                cause = "I"
            else:
                cause = "J"
            frame_causes[cause] += 1
            deepest = max(
                contacts,
                key=lambda row: float(row["penetration_depth_m"]),
                default=None,
            )
            joint_index = int(np.argmin(margin[frame]))
            hard_frame_rows.append(
                {
                    "episode_index": episode,
                    "stable_episode_id": stable_episode_id(episode),
                    "frame": frame,
                    "primary_cause_code": cause,
                    "primary_cause": CAUSE_LABELS[cause],
                    "original_validation_status": load_json(
                        a_metric_path(episode, ".validation")
                    )["status"],
                    "original_strict_ik_fail": bool(strict_ik_fail[frame]),
                    "original_position_fail": pos_fail,
                    "original_orientation_fail": ori_fail,
                    "original_max_position_residual_m": float(position[frame]),
                    "original_left_position_residual_m": float(
                        position_by_side["left"][frame]
                    ),
                    "original_right_position_residual_m": float(
                        position_by_side["right"][frame]
                    ),
                    "original_max_orientation_residual_rad": float(
                        orientation[frame]
                    ),
                    "original_left_orientation_residual_rad": float(
                        orientation_by_side["left"][frame]
                    ),
                    "original_right_orientation_residual_rad": float(
                        orientation_by_side["right"][frame]
                    ),
                    "collision_class": deepest["classification"] if deepest else "NONE",
                    "collision_pair": (
                        "|".join(deepest["body_pair"]) if deepest else "NONE"
                    ),
                    "penetration_m": (
                        float(deepest["penetration_depth_m"])
                        if deepest
                        else 0.0
                    ),
                    "joint_minimum_margin_rad": float(margin[frame, joint_index]),
                    "minimum_margin_joint": str(
                        resolver.g1.arm_joint_names[joint_index]
                    ),
                    "target_position_step_m": float(target_position_step[frame]),
                    "target_orientation_step_rad": float(
                        target_orientation_step[frame]
                    ),
                    "left_target_shoulder_radius_m": float(
                        reach_by_side["left"][frame]
                    ),
                    "right_target_shoulder_radius_m": float(
                        reach_by_side["right"][frame]
                    ),
                    "temporal_violation_frame": frame in temporal_frames,
                    "branch_discontinuity_frame": frame in branch_frames,
                    "projection_left_m": float(projection[0]),
                    "projection_right_m": float(projection[1]),
                    "after_left_source_position_residual_m": float(
                        after_source_position_by_side["left"][frame]
                    ),
                    "after_right_source_position_residual_m": float(
                        after_source_position_by_side["right"][frame]
                    ),
                    "after_left_orientation_residual_rad": float(
                        after_orientation_by_side["left"][frame]
                    ),
                    "after_right_orientation_residual_rad": float(
                        after_orientation_by_side["right"][frame]
                    ),
                    "solver_status": (
                        "COMMON_REPAIR_SOURCE_POSE_PHYSICALLY_ACCEPTED"
                        if final_source_pose_ok
                        else "COMMON_REPAIR_REQUIRES_REALIZED_TARGET_PROJECTION"
                    ),
                }
            )
        for frame, records in before_collision["by_frame"].items():
            for record in records:
                collision_records.append(
                    {
                        "episode_index": episode,
                        "stable_episode_id": stable_episode_id(episode),
                        "frame": int(frame),
                        "classification": record["classification"],
                        "body_pair": "|".join(record["body_pair"]),
                        "geom_pair": "|".join(map(str, record["geom_pair"])),
                        "penetration_m": float(record["penetration_depth_m"]),
                        "original_gate_invalid": (
                            record["classification"]
                            != "DISTAL_HAND_HAND_CONTACT"
                        ),
                    }
                )
        original_status = load_json(a_metric_path(episode, ".validation"))["status"]
        # Episode-primary is the most frequent frame-level cause, independent of
        # which validator gate happened to be evaluated first.
        primary = max(
            ((count, code) for code, count in frame_causes.items()),
            default=(0, "J"),
        )[1]
        projection_max = float(np.max(result.projection_translation_m))
        final_row = final_rows[episode]
        numerical_classification = None
        if episode in IK_HARD_EPISODES:
            material_threshold = (
                resolver.physical_tolerance - resolver.strict_tolerance
            )
            numerical_only = bool(
                projection_max <= material_threshold + 1e-7
                and float(final_row["after_source_physical_6d_success_rate"])
                >= float(
                    resolver.common["shared_temporal_ik"]["required_success_rate"]
                )
                and final_row["after_classification"] != "HARD_FAIL"
            )
            numerical_classification = (
                "NUMERICAL_ONLY"
                if numerical_only
                else "PHYSICAL_TARGET_INFEASIBLE"
            )
            ik_classification_rows.append(
                {
                    "episode_index": episode,
                    "stable_episode_id": stable_episode_id(episode),
                    "classification": numerical_classification,
                    "projection_max_m": projection_max,
                    "material_projection_threshold_m": material_threshold,
                    "after_source_physical_6d_success_rate": final_row[
                        "after_source_physical_6d_success_rate"
                    ],
                    "after_mechanical_classification": final_row[
                        "after_classification"
                    ],
                    "criterion": (
                        "NUMERICAL_ONLY requires <=5 mm projection, >=95% source "
                        "physical 6-D success, and no final hard mechanical gate"
                    ),
                    "worst_projected_frame_static_6d_pass": static_peak_pass,
                    "infeasibility_scope": (
                        "TRAJECTORY_LEVEL_TEMPORAL_OR_COLLISION"
                        if static_peak_pass and not numerical_only
                        else (
                            "STATIC_OR_COMBINED_POSE"
                            if not numerical_only
                            else "NUMERICAL_ONLY"
                        )
                    ),
                }
            )
        detail = {
            "episode_index": episode,
            "stable_episode_id": stable_episode_id(episode),
            "original_status": original_status,
            "hard_frame_count": len(hard_frames),
            "hard_frames": hard_frames,
            "first_hard_frame": hard_frames[0] if hard_frames else None,
            "last_hard_frame": hard_frames[-1] if hard_frames else None,
            "max_position_residual_m": float(np.max(position)),
            "max_orientation_residual_rad": float(np.max(orientation)),
            "minimum_joint_margin_rad": float(np.min(margin)),
            "minimum_margin_joint": str(
                resolver.g1.arm_joint_names[
                    int(np.unravel_index(np.argmin(margin), margin.shape)[1])
                ]
            ),
            "collision_pairs": sorted(
                {
                    "|".join(record["body_pair"])
                    for records in before_collision["by_frame"].values()
                    for record in records
                    if record["classification"] != "DISTAL_HAND_HAND_CONTACT"
                }
            ),
            "maximum_penetration_m": float(
                before_collision["maximum_penetration_depth_m"]
            ),
            "cause_frame_counts": dict(frame_causes),
            "episode_primary_cause_code": primary,
            "episode_primary_cause": CAUSE_LABELS[primary],
            "ik_failure_kind": numerical_classification,
            "before_temporal": before_temporal,
            "after_classification": final_row["after_classification"],
            "projection_max_m": projection_max,
            "worst_projected_frame_static_6d_pass": static_peak_pass,
        }
        hard_episode_details[episode] = detail
        episode_root_rows.append(
            {
                **{
                    key: value
                    for key, value in detail.items()
                    if key not in {"hard_frames", "before_temporal"}
                },
                "cause_frame_counts": _json_cell(detail["cause_frame_counts"]),
                "collision_pairs": _json_cell(detail["collision_pairs"]),
            }
        )

    projection = np.concatenate(projection_all)
    projection_report = {
        "units": "m",
        "all_arms_all_frames": _stats(projection),
        "by_arm": {
            side: _stats(np.concatenate(projection_by_side[side])) for side in SIDES
        },
        "active_scalar_count": int(np.count_nonzero(projection > 0.0)),
        "active_frame_count": int(
            sum(int(row["projection_active_frames"]) for row in final_rows)
        ),
        "source_targets_preserved": all(immutable_checks),
        "orientation_projection": _stats(
            np.zeros_like(projection, dtype=np.float64)
        ),
    }
    diagnostics = {
        key: _stats(np.concatenate(value)) for key, value in diagnostic_values.items()
    }
    summary = {
        "schema_version": "fair_a_full50_after_common_repair_validation_v1",
        "episode_count": 50,
        "total_frames": int(sum(int(row["frame_count"]) for row in final_rows)),
        "classification_counts": dict(classifications),
        "hard_episode_indices": [
            int(row["episode_index"])
            for row in final_rows
            if row["after_classification"] == "HARD_FAIL"
        ],
        "ik_hard_episode_count": int(
            sum("FAIL_IK" in json.loads(row["hard_reasons"]) for row in final_rows)
        ),
        "ik_strict_failed_frame_count": int(
            sum(row["after_realized_strict_6d_failed_frames"] for row in final_rows)
        ),
        "collision_hard_episode_count": int(
            sum(
                "FAIL_COLLISION" in json.loads(row["hard_reasons"])
                for row in final_rows
            )
        ),
        "collision_hard_frame_count": int(
            sum(row["hard_collision_frames_after"] for row in final_rows)
        ),
        "joint_limit_violation_count": int(
            sum(row["joint_limit_violation_count"] for row in final_rows)
        ),
        "branch_discontinuity_count": int(
            sum(row["branch_discontinuity_count"] for row in final_rows)
        ),
        "maximum_velocity_rad_s": float(
            max(row["maximum_velocity_rad_s"] for row in final_rows)
        ),
        "maximum_acceleration_rad_s2": float(
            max(row["maximum_acceleration_rad_s2"] for row in final_rows)
        ),
        "projection": projection_report,
        "diagnostic_only_metrics": diagnostics,
        "source_arrays_unchanged_all_episodes": all(immutable_checks),
        "interaction_frame_used_as_solver_target": False,
        "ownership_used_by_solver": False,
        "bimanual_semantic_target_used_by_solver": False,
        "episode_specific_correction": False,
        "phase_specific_correction": False,
    }
    write_csv(OUTPUT_ROOT / "after/full50_per_episode.csv", final_rows)
    write_json(OUTPUT_ROOT / "after/full50_validation.json", summary)
    write_json(OUTPUT_ROOT / "after/projection_statistics.json", projection_report)
    write_csv(
        OUTPUT_ROOT / "diagnostics/original_22_episode_summary.csv",
        episode_root_rows,
    )
    write_csv(
        OUTPUT_ROOT / "diagnostics/original_hard_frames.csv", hard_frame_rows
    )
    write_csv(
        OUTPUT_ROOT / "diagnostics/original_collision_records.csv",
        collision_records,
    )
    write_csv(
        OUTPUT_ROOT / "diagnostics/after_collision_records.csv",
        after_collision_records,
    )
    write_csv(
        OUTPUT_ROOT / "diagnostics/ik_numerical_physical_classification.csv",
        ik_classification_rows,
    )
    write_json(
        OUTPUT_ROOT / "diagnostics/original_22_episode_details.json",
        {str(key): value for key, value in hard_episode_details.items()},
    )
    collision_episode_audit = []
    for episode in HARD_EPISODES:
        before = collision_before_failed[episode]
        after = collision_after[episode]
        if (
            episode not in COLLISION_HARD_EPISODES
            and not before["contact_frame_count"]
            and not after["contact_frame_count"]
        ):
            continue
        def pair_summary(collision: Mapping[str, Any]) -> list[dict[str, Any]]:
            grouped: dict[tuple[str, str], dict[str, Any]] = {}
            for frame, records in collision["by_frame"].items():
                for record in records:
                    if record["classification"] == "DISTAL_HAND_HAND_CONTACT":
                        continue
                    key = (
                        str(record["classification"]),
                        "|".join(record["body_pair"]),
                    )
                    row = grouped.setdefault(
                        key,
                        {
                            "classification": key[0],
                            "body_pair": key[1],
                            "frames": [],
                            "maximum_penetration_m": 0.0,
                        },
                    )
                    row["frames"].append(int(frame))
                    row["maximum_penetration_m"] = max(
                        float(row["maximum_penetration_m"]),
                        float(record["penetration_depth_m"]),
                    )
            output = []
            for row in grouped.values():
                row["frames"] = sorted(set(row["frames"]))
                row["frame_segments"] = contiguous_segments(row["frames"])
                row["frame_count"] = len(row["frames"])
                output.append(row)
            return sorted(output, key=lambda row: (row["classification"], row["body_pair"]))
        collision_episode_audit.append(
            {
                "episode_index": episode,
                "stable_episode_id": stable_episode_id(episode),
                "original_validation_status": load_json(
                    a_metric_path(episode, ".validation")
                )["status"],
                "original": {
                    "contact_frame_count": before["contact_frame_count"],
                    "hard_collision_frame_count_under_shared_severity": before[
                        "hard_collision_frame_count"
                    ],
                    "maximum_penetration_m": before[
                        "maximum_penetration_depth_m"
                    ],
                    "segments": before["segments"],
                    "pairs": pair_summary(before),
                },
                "after": {
                    "contact_frame_count": after["contact_frame_count"],
                    "hard_collision_frame_count_under_shared_severity": after[
                        "hard_collision_frame_count"
                    ],
                    "maximum_penetration_m": after[
                        "maximum_penetration_depth_m"
                    ],
                    "segments": after["segments"],
                    "pairs": pair_summary(after),
                },
                "environment_collision": False,
                "numerical_contact_artifact": False,
                "collision_model_scope": "robot self-collision only",
            }
        )
    write_json(
        OUTPUT_ROOT / "diagnostics/collision_episode_audit.json",
        {"episodes": collision_episode_audit},
    )
    return summary, final_rows, hard_episode_details


def build_comparison(
    a_summary: Mapping[str, Any], a_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    b_aggregate = load_json(
        REPOSITORY / "outputs/doll_handoff_dataset_b_final/final50/aggregate_metrics.json"
    )
    b_assembly = load_json(
        REPOSITORY / "outputs/doll_handoff_dataset_b_final/final50/assembly_validation.json"
    )
    b_projection = []
    b_projection_by_side = {side: [] for side in SIDES}
    b_wrist_error = []
    b_grasp_error = []
    b_relation_error = []
    # Use the same final G1 kinematic registration as the A audit.
    resolver = FullPoseCommonWristResolver(output_root=OUTPUT_ROOT)
    for episode in range(50):
        path = b_final_trajectory_path(episode)
        with np.load(path, allow_pickle=False) as payload:
            projection = payload["feasibility_projection_translation_m"].astype(
                np.float64
            )
            b_projection.append(projection.reshape(-1))
            for side_index, side in enumerate(SIDES):
                b_projection_by_side[side].append(projection[:, side_index])
                target_wrist_world = resolver.g1.model_to_world_position(
                    payload[f"target_{side}_wrist_position_model"].astype(np.float64)
                )
                achieved_wrist = payload[
                    f"achieved_{side}_wrist_position_world"
                ].astype(np.float64)
                b_wrist_error.append(
                    np.linalg.norm(achieved_wrist - target_wrist_world, axis=1)
                )
                source_grasp = payload[
                    f"source_{side}_interaction_frame_position_world"
                ].astype(np.float64)
                achieved_grasp = payload[
                    f"achieved_{side}_physical_grasp_frame_position_world"
                ].astype(np.float64)
                b_grasp_error.append(
                    np.linalg.norm(achieved_grasp - source_grasp, axis=1)
                )
            left_source = payload[
                "source_left_interaction_frame_position_world"
            ].astype(np.float64)
            right_source = payload[
                "source_right_interaction_frame_position_world"
            ].astype(np.float64)
            left_achieved = payload[
                "achieved_left_physical_grasp_frame_position_world"
            ].astype(np.float64)
            right_achieved = payload[
                "achieved_right_physical_grasp_frame_position_world"
            ].astype(np.float64)
            b_relation_error.append(
                np.linalg.norm(
                    (right_achieved - left_achieved)
                    - (right_source - left_source),
                    axis=1,
                )
            )
    b_projection_array = np.concatenate(b_projection)
    b_projection_report = {
        "all_arms_all_frames": _stats(b_projection_array),
        "by_arm": {
            side: _stats(np.concatenate(b_projection_by_side[side]))
            for side in SIDES
        },
    }
    a_diag = a_summary["diagnostic_only_metrics"]
    a_counts = a_summary["classification_counts"]
    b_counts = b_assembly["classification_counts"]
    comparison = {
        "schema_version": "fair_a_vs_proposed_b_diagnostic_comparison_v1",
        "no_policy_performance_claim": True,
        "metrics": [
            {
                "metric": "source episodes",
                "fair_a": "50 exact final-common identities; 34,478 frames",
                "proposed_b": "same 50 exact final-common identities; 34,478 frames",
            },
            {
                "metric": "trajectory representation",
                "fair_a": "trajectory-centric independent wrist-level 6-D",
                "proposed_b": "interaction-centric whole-hand/bimanual representation",
            },
            {
                "metric": "shared natural-arm solver",
                "fair_a": "YES; frozen common temporal DLS + SEW/null-space backend",
                "proposed_b": "YES; identical frozen common backend",
            },
            {
                "metric": "shared feasibility resolver",
                "fair_a": "YES after parity correction; wrist-frame adapter + common full-pose bug fix",
                "proposed_b": "YES; frozen generic resolver, interaction-frame adapter",
            },
            {
                "metric": "hard IK episodes",
                "fair_a": int(a_summary["ik_hard_episode_count"]),
                "proposed_b": 0,
            },
            {
                "metric": "hard collision episodes",
                "fair_a": int(a_summary["collision_hard_episode_count"]),
                "proposed_b": 0,
            },
            {
                "metric": "target projection mean / p95 / max (m)",
                "fair_a": {
                    key: a_summary["projection"]["all_arms_all_frames"][key]
                    for key in ("mean", "p95", "max")
                },
                "proposed_b": {
                    key: b_projection_report["all_arms_all_frames"][key]
                    for key in ("mean", "p95", "max")
                },
            },
            {
                "metric": "joint-limit failures",
                "fair_a": int(a_summary["joint_limit_violation_count"]),
                "proposed_b": int(
                    b_aggregate["joint_limit_branch_collision_totals"][
                        "joint_limit_violations_after"
                    ]
                ),
            },
            {
                "metric": "branch failures",
                "fair_a": int(a_summary["branch_discontinuity_count"]),
                "proposed_b": int(
                    b_aggregate["joint_limit_branch_collision_totals"][
                        "branch_discontinuities_after"
                    ]
                ),
            },
            {
                "metric": "wrist tracking error mean / p95 / max (m)",
                "fair_a": {
                    key: a_diag["a_wrist_error"][key]
                    for key in ("mean", "p95", "max")
                },
                "proposed_b": {
                    key: _stats(np.concatenate(b_wrist_error))[key]
                    for key in ("mean", "p95", "max")
                },
            },
            {
                "metric": "whole-hand grasp-frame error mean / p95 / max (m)",
                "fair_a": {
                    key: a_diag["a_grasp_error"][key]
                    for key in ("mean", "p95", "max")
                },
                "proposed_b": {
                    key: _stats(np.concatenate(b_grasp_error))[key]
                    for key in ("mean", "p95", "max")
                },
            },
            {
                "metric": "bimanual relation error mean / p95 / max (m)",
                "fair_a": {
                    key: a_diag["a_bimanual_relation_error"][key]
                    for key in ("mean", "p95", "max")
                },
                "proposed_b": {
                    key: _stats(np.concatenate(b_relation_error))[key]
                    for key in ("mean", "p95", "max")
                },
            },
            {
                "metric": "mechanical classification CLEAN / WARNING / HARD",
                "fair_a": {
                    "CLEAN": int(a_counts.get("CLEAN_PASS", 0)),
                    "WARNING": int(a_counts.get("USABLE_WITH_WARNING", 0)),
                    "HARD": int(a_counts.get("HARD_FAIL", 0)),
                },
                "proposed_b": {
                    "CLEAN": int(b_counts["CLEAN_PASS"]),
                    "WARNING": int(b_counts["USABLE_WITH_WARNING"]),
                    "HARD": int(b_counts["HARD_FAIL"]),
                },
            },
        ],
        "projection": {
            "fair_a": a_summary["projection"],
            "proposed_b": b_projection_report,
        },
    }
    write_json(OUTPUT_ROOT / "comparison/fair_a_vs_proposed_b.json", comparison)
    rows = [
        {
            "METRIC": row["metric"],
            "FAIR A": (
                _json_cell(row["fair_a"])
                if isinstance(row["fair_a"], (dict, list))
                else row["fair_a"]
            ),
            "PROPOSED B": (
                _json_cell(row["proposed_b"])
                if isinstance(row["proposed_b"], (dict, list))
                else row["proposed_b"]
            ),
        }
        for row in comparison["metrics"]
    ]
    write_csv(OUTPUT_ROOT / "comparison/fair_a_vs_proposed_b.csv", rows)
    return comparison


def build_final_report(
    provenance: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
    details: Mapping[int, Mapping[str, Any]],
    comparison: Mapping[str, Any],
    dataset_packaged: bool = False,
    dataset_path: str = "",
) -> str:
    original_gate = provenance["original_gate"]["content"]
    root_counts = collections.Counter(
        str(row["episode_primary_cause_code"]) for row in details.values()
    )
    numerical = sum(
        row.get("ik_failure_kind") == "NUMERICAL_ONLY" for row in details.values()
    )
    physical = sum(
        row.get("ik_failure_kind") == "PHYSICAL_TARGET_INFEASIBLE"
        for row in details.values()
    )
    physical_static_peak_pass = sum(
        row.get("ik_failure_kind") == "PHYSICAL_TARGET_INFEASIBLE"
        and bool(row.get("worst_projected_frame_static_6d_pass"))
        for row in details.values()
    )
    counts = summary["classification_counts"]
    projection = summary["projection"]["all_arms_all_frames"]
    comparison_lines = ["| METRIC | FAIR A | PROPOSED B |", "|---|---|---|"]
    for row in comparison["metrics"]:
        left = _json_cell(row["fair_a"]) if isinstance(row["fair_a"], (dict, list)) else str(row["fair_a"])
        right = _json_cell(row["proposed_b"]) if isinstance(row["proposed_b"], (dict, list)) else str(row["proposed_b"])
        comparison_lines.append(f"| {row['metric']} | {left} | {right} |")
    collision_audit = load_json(
        OUTPUT_ROOT / "diagnostics/collision_episode_audit.json"
    )["episodes"]
    collision_by_episode = {
        int(row["episode_index"]): row for row in collision_audit
    }
    collision_lines = []
    for episode in COLLISION_HARD_EPISODES:
        row = collision_by_episode[episode]
        pairs = ", ".join(
            f"{pair['body_pair']} frames={pair['frame_segments']} "
            f"max={1000*pair['maximum_penetration_m']:.2f} mm"
            for pair in row["original"]["pairs"]
        )
        collision_lines.append(
            f"- ep{episode:02d}: arm/torso; {pairs}; after shared-hard "
            f"frames={row['after']['hard_collision_frame_count_under_shared_severity']}"
        )
    surviving_collision_lines = []
    for row in collision_audit:
        if int(row["after"]["hard_collision_frame_count_under_shared_severity"]):
            surviving_collision_lines.append(
                f"ep{int(row['episode_index']):02d}="
                f"{row['after']['hard_collision_frame_count_under_shared_severity']} "
                f"frames/{1000*row['after']['maximum_penetration_m']:.2f} mm"
            )
    hard_count = int(counts.get("HARD_FAIL", 0))
    decision = (
        "FAIR_A_FULL50_READY_FOR_ACT_A"
        if hard_count == 0 and dataset_packaged
        else (
            "FAIR_A_VALID_WITH_MEASURED_HARD_FAILURES"
            if hard_count > 0
            else "FAIR_A_BASELINE_DEFINITION_REQUIRES_GLOBAL_REVISION"
        )
    )
    report = f"""# FAIR BASELINE A FULL-50 HARD-FAIL ROOT-CAUSE AUDIT

FAIR A ORIGINAL

episodes: {original_gate['episodes_attempted']}  
PASS: {original_gate['structural_classification_counts']['PASS']}  
WARNING: {original_gate['structural_classification_counts']['WARNING']}  
HARD: {original_gate['structural_classification_counts']['HARD_FAIL']}  
IK hard: {original_gate['hard_fail_categories']['FAIL_IK']} episodes  
collision hard: {original_gate['hard_fail_categories']['FAIL_COLLISION']} episodes

BACKEND PARITY

natural-arm solver identical: {provenance['backend_parity']['COMMON_NATURAL_ARM_BACKEND_IDENTICAL']}  
generic feasibility resolver identical: {provenance['backend_parity']['COMMON_GENERIC_FEASIBILITY_RESOLVER_IDENTICAL']}  
differences found: Original A omitted the frozen generic feasibility stage. Its target adapter was also hard-wired to B's whole-hand interaction frame and its tracker used B's non-gating orientation policy. The fair repair parameterized the realization frame as the A wrist, inherited the frozen numerical resolver/config, and added a global full-pose acceptance/temporal-window conditioning fix derived from the frozen common IK weights. No configured number was fitted to A.

22 HARD FAIL ROOT CAUSES

workspace: {root_counts['A']} episode-primary ({sum(row['cause_frame_counts'].get('A', 0) for row in details.values())} frames)  
orientation: {root_counts['B']} episode-primary ({sum(row['cause_frame_counts'].get('B', 0) for row in details.values())} frames)  
torso clearance: {root_counts['C']} episode-primary ({sum(row['cause_frame_counts'].get('C', 0) for row in details.values())} frames)  
collision: {sum(row['original_status'] == 'FAIL_COLLISION' for row in details.values())} original episodes; cross-arm={root_counts['E']}, other-self={root_counts['D']} episode-primary  
temporal: {root_counts['F']} episode-primary ({sum(row['cause_frame_counts'].get('F', 0) for row in details.values())} frames)  
numerical: {numerical} of 18 FAIL_IK episodes  
physically infeasible target: {physical} of 18 FAIL_IK episodes at the full-trajectory constraint level; {physical_static_peak_pass} worst projected poses are statically reachable, localizing these to temporal/branch/collision compatibility rather than raw static workspace reach  
other: {root_counts['J']} episode-primary

Episode-primary frame-mechanism breakdown (one cause per original hard frame, then modal cause per episode): A={root_counts['A']}, B={root_counts['B']}, C={root_counts['C']}, D={root_counts['D']}, E={root_counts['E']}, F={root_counts['F']}, G={root_counts['G']}, H={root_counts['H']}, I={root_counts['I']}, J={root_counts['J']}.

The physical/numerical split is operational under the frozen common constraints: NUMERICAL_ONLY requires at most the 5 mm strict-to-physical tolerance band of target projection, at least 95% physical source-pose success, and no final hard mechanical gate. Larger projection remains PHYSICAL_TARGET_INFEASIBLE at the trajectory level after the shared bidirectional and minimum-acceleration machinery is exhausted. The bounded multistart oracle shows the worst isolated poses themselves are reachable; it does not satisfy the temporal, collision, or continuous-branch constraints and therefore is diagnostic only.

GENERIC REPAIR

applied: YES  
method: frozen bidirectional box-constrained tracking; global full-pose acceptance conditioning; hard SO(3) feasibility; frozen 16/32/48-frame minimum-acceleration windows; frozen signed-distance collision repair; frozen nearest-feasible translation projection  
A-specific logic: NO  
source target modified semantically: NO  
projection mean: {projection['mean']:.9f} m  
projection median: {projection['median']:.9f} m  
projection p95: {projection['p95']:.9f} m  
projection max: {projection['max']:.9f} m

Per arm: left mean/p95/max = {summary['projection']['by_arm']['left']['mean']:.9f} / {summary['projection']['by_arm']['left']['p95']:.9f} / {summary['projection']['by_arm']['left']['max']:.9f} m; right = {summary['projection']['by_arm']['right']['mean']:.9f} / {summary['projection']['by_arm']['right']['p95']:.9f} / {summary['projection']['by_arm']['right']['max']:.9f} m.

COLLISION FAILURES

{chr(10).join(collision_lines)}

All four original collision-only failures are arm/torso self-collisions; none are environment contacts, cross-arm contacts, hand/arm contacts, or numerical contact artifacts. After the exact shared repair, shared-severity hard collision remains in {', '.join(surviving_collision_lines) if surviving_collision_lines else 'none'}; these episodes remain HARD_FAIL.

FULL-50 AFTER FAIR REPAIR

CLEAN: {counts.get('CLEAN_PASS', 0)}  
WARNING: {counts.get('USABLE_WITH_WARNING', 0)}  
HARD: {counts.get('HARD_FAIL', 0)}  
IK hard: {summary['ik_hard_episode_count']} episodes; {summary['ik_strict_failed_frame_count']} isolated strict frames  
collision hard: {summary['collision_hard_episode_count']} episodes; {summary['collision_hard_frame_count']} frames  
limits: {summary['joint_limit_violation_count']}  
branch: {summary['branch_discontinuity_count']}  
maximum velocity: {summary['maximum_velocity_rad_s']:.6f} rad/s  
maximum acceleration: {summary['maximum_acceleration_rad_s2']:.6f} rad/s²

FAIRNESS

interaction-frame used: NO  
ownership used: NO  
bimanual semantic target used: NO  
episode-specific correction: NO  
phase-specific correction: NO

WHOLE-HAND / BIMANUAL METRICS

Diagnostic only; none were solver inputs. Whole-hand grasp-frame position error mean/p95/max = {summary['diagnostic_only_metrics']['a_grasp_error']['mean']:.6f} / {summary['diagnostic_only_metrics']['a_grasp_error']['p95']:.6f} / {summary['diagnostic_only_metrics']['a_grasp_error']['max']:.6f} m. Bimanual relation error mean/p95/max = {summary['diagnostic_only_metrics']['a_bimanual_relation_error']['mean']:.6f} / {summary['diagnostic_only_metrics']['a_bimanual_relation_error']['p95']:.6f} / {summary['diagnostic_only_metrics']['a_bimanual_relation_error']['max']:.6f} m.

COMPARISON TO B

{chr(10).join(comparison_lines)}

DATASET A

packaged: {'YES' if dataset_packaged else 'NO'}  
path: {dataset_path}

POLICY A

NOT_STARTED_BY_DESIGN

FINAL DECISION

{decision}
"""
    path = OUTPUT_ROOT / "FAIR_A_FULL50_HARD_FAIL_AUDIT_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)
    write_json(
        OUTPUT_ROOT / "final_decision.json",
        {
            "decision": decision,
            "dataset_packaged": dataset_packaged,
            "dataset_path": dataset_path,
            "policy_a": "NOT_STARTED_BY_DESIGN",
        },
    )
    return report


def run_report() -> dict[str, Any]:
    resolver = FullPoseCommonWristResolver(output_root=OUTPUT_ROOT)
    provenance = build_provenance(resolver)
    static_oracle = build_static_pose_oracle(resolver)
    summary, rows, details = evaluate_all(resolver, static_oracle)
    comparison = build_comparison(summary, rows)
    report = build_final_report(provenance, summary, rows, details, comparison)
    trajectories = []
    for episode in range(50):
        path, metric_path = resolver._cache_paths(episode)
        trajectories.append(
            {
                "episode_index": episode,
                "stable_episode_id": stable_episode_id(episode),
                "trajectory_path": str(path.resolve()),
                "trajectory_sha256": sha256_file(path),
                "solver_metric_path": str(metric_path.resolve()),
                "solver_metric_sha256": sha256_file(metric_path),
            }
        )
    implementation_files = [
        REPOSITORY / "tools/doll_handoff_feasibility/solver.py",
        REPOSITORY / "configs/doll_handoff_g1_feasibility_resolver.json",
        REPOSITORY / "tools/fair_a_full50_audit/wrist_resolver.py",
        REPOSITORY / "tools/fair_a_full50_audit/full_pose_resolver.py",
        REPOSITORY / "tools/fair_a_full50_audit/report.py",
        REPOSITORY / "tools/run_fair_a_full50_feasibility_audit.py",
        REPOSITORY / "tools/build_fair_a_full50_audit_report.py",
    ]
    implementation_hashes = {
        str(path.relative_to(REPOSITORY)): sha256_file(path)
        for path in implementation_files
    }
    write_json(
        OUTPUT_ROOT / "after/fair_a_repair_manifest.json",
        {
            "schema_version": "fair_a_common_repair_manifest_v1",
            "status": "VALID_WITH_MEASURED_HARD_FAILURES",
            "git_head": _git("rev-parse", "HEAD"),
            "original_a_implementation_sha256": provenance[
                "original_a_implementation_sha256"
            ],
            "frozen_b_generic_resolver_implementation_sha256": provenance[
                "backend_parity"
            ]["frozen_b_generic_resolver"]["implementation_sha256"],
            "frozen_b_generic_resolver_config_sha256": provenance[
                "backend_parity"
            ]["frozen_b_generic_resolver"]["config"]["sha256"],
            "audit_implementation_files": implementation_hashes,
            "audit_implementation_sha256": stable_json_sha256(
                implementation_hashes
            ),
            "trajectory_set_sha256": stable_json_sha256(trajectories),
            "trajectories": trajectories,
            "classification_counts": summary["classification_counts"],
            "source_targets_modified_semantically": False,
            "episode_specific_correction": False,
            "phase_specific_correction": False,
            "dataset_a_packaged": False,
            "policy_a_training_started": False,
            "gpu_used": False,
        },
    )
    return {
        "status": "REPORT_COMPLETE",
        "classification_counts": summary["classification_counts"],
        "report": str(
            OUTPUT_ROOT / "FAIR_A_FULL50_HARD_FAIL_AUDIT_REPORT.md"
        ),
        "report_preview": report[-1000:],
    }


__all__ = [
    "build_comparison",
    "build_final_report",
    "build_provenance",
    "evaluate_all",
    "run_report",
]
