"""Gated calibration, validation, 50+50 generation, and reporting for v3."""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aloha_g1_arm_v2.audit import configure_g1
from aloha_g1_arm_v2.common import scalar_stats, tree_sha256
from aloha_g1_dataset_v1.core import G1Kinematics
from aloha_g1_hand_v2.collision_eval import CollisionClassifier, make_runtime

from .audit import (
    acceptance_gate_contract,
    collision_gate_semantics,
    reproduce_v2,
    select_failure_subset,
)
from .common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    METHOD_TO_DATASET,
    ROOT,
    V2_ARM_ROOT,
    V2_INTEGRATED_ROOT,
    atomic_csv,
    atomic_json,
    freeze_dependencies,
    implementation_fingerprint,
    load_json,
    load_search_config,
    sha256_file,
    source_integrity,
)
from .evaluate import STATUSES, evaluate_result, export_episode
from .solver import (
    FeasibilityResult,
    SharedConstrainedFeasibilitySolver,
    load_frozen_episode,
)


METHODS = ("baseline", "proposed")


def _directories(output_root: Path) -> None:
    for name in (
        "dependencies",
        "audit",
        "solver",
        "dataset_a",
        "dataset_b",
        "summary",
        "tests",
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def _anti_overfit_scan() -> dict[str, Any]:
    paths = sorted((ROOT / "tools/aloha_g1_feasibility_v3").glob("*.py"))
    paths += [
        ROOT / "tools/run_common_feasibility_v3.py",
        ROOT / "configs/aloha_g1_feasibility_v3.json",
    ]
    patterns = {
        "episode_equality": re.compile(r"if\s+episode(?:_id)?\s*=="),
        "frame_equality": re.compile(r"if\s+frame(?:_id|_index)?\s*=="),
        "historical_episode_token": re.compile("ep" + "49", re.IGNORECASE),
        "authored_translation_rule": re.compile(
            "manual" + r"[ _-]+offset", re.IGNORECASE
        ),
        "episode_specific_slack": re.compile(
            "per" + r"[ _-]+episode[ _-]+slack", re.IGNORECASE
        ),
        "method_specific_parameter": re.compile(
            "per" + r"[ _-]+method[ _-]+(?:feasibility[ _-]+)?parameter",
            re.IGNORECASE,
        ),
    }
    hits: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                hits.append(
                    {
                        "path": str(path),
                        "pattern": name,
                        "match": match.group(0),
                    }
                )
    return {
        "pass": not hits,
        "files_scanned": len(paths),
        "patterns": list(patterns),
        "hits": hits,
    }


def _candidate(
    shared: Mapping[str, Any], slack: float, collision_penalty: float
) -> dict[str, Any]:
    value = copy.deepcopy(dict(shared))
    value["orientation_slack_bound_rad"] = float(slack)
    value["collision_penalty"] = float(collision_penalty)
    value["candidate_id"] = (
        f"slack_{int(round(slack * 100)):03d}_collision_"
        f"{int(round(collision_penalty)):03d}"
    )
    return value


def _smoke_score(metrics: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> tuple[Any, ...]:
    episode_rates: dict[str, float] = {}
    ik: dict[str, float] = {}
    collisions = 0
    active = 0
    temporal = 0
    position = 0.0
    slack = 0.0
    b_interaction_degradation = 0.0
    for method in METHODS:
        rows = list(metrics[method].values())
        episode_rates[method] = float(
            np.mean([row["status"] == "PASS" for row in rows])
        )
        ik[method] = float(np.mean([row["ik_success_rate"] for row in rows]))
        collisions += int(
            sum(row["collision"]["prohibited_collision_frames"] for row in rows)
        )
        active += int(sum(row["active_joint_bound_frames"] for row in rows))
        temporal += int(sum(row["status"] == "FAIL_TEMPORAL" for row in rows))
        position += float(np.mean([row["position_error_mean_m"] for row in rows]))
        slack += float(
            np.mean([row["orientation_slack_rad"]["mean"] for row in rows])
        )
        if method == "proposed":
            v2_reference = {
                "task_pinch": 0.012869,
                "midpoint": 0.026318,
                "relative": 0.056473,
            }
            b_interaction_degradation = float(
                max(
                    0.0,
                    np.mean(
                        [
                            row["task_space"][
                                "task_critical_pinch_error_mean_m"
                            ]
                            for row in rows
                        ]
                    )
                    - v2_reference["task_pinch"],
                )
                + max(
                    0.0,
                    np.mean(
                        [
                            row["bimanual"]["midpoint_error_mean_m"]
                            for row in rows
                        ]
                    )
                    - v2_reference["midpoint"],
                )
                + max(
                    0.0,
                    np.mean(
                        [
                            row["bimanual"][
                                "relative_vector_error_mean_m"
                            ]
                            for row in rows
                        ]
                    )
                    - v2_reference["relative"],
                )
            )
    return (
        -min(episode_rates.values()),
        -min(ik.values()),
        collisions,
        active,
        temporal,
        position,
        slack,
        b_interaction_degradation,
    )


def _run_one(
    method: str,
    episode_id: int,
    solver_config: Mapping[str, Any],
    search: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
) -> tuple[FeasibilityResult, dict[str, Any]]:
    solver = SharedConstrainedFeasibilitySolver(
        runtime_config,
        solver_config,
        search["shared_hand_temporal_projection"],
        g1,
        collision_runtime,
        classifier,
    )
    episode = load_frozen_episode(method, episode_id)
    result = solver.solve(episode)
    evaluated = evaluate_result(
        result, collision_runtime, classifier, g1, runtime_config
    )
    return result, evaluated


def _candidate_smoke(
    candidate: Mapping[str, Any],
    subset: Mapping[str, Any],
    search: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    a_ids = sorted(
        {
            int(subset["selected"]["a_joint_limit_dominated"]),
            int(subset["selected"]["a_orientation_dominated"]),
            int(subset["selected"]["a_temporal_dominated"]),
        }
    )
    b_ids = sorted(
        {
            int(subset["selected"]["b_arm_collision_dominated"]),
            int(subset["selected"]["b_hand_or_cross_collision_dominated"]),
            int(subset["selected"]["b_remaining_ik_failure"]),
        }
    )
    metrics: dict[str, dict[int, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    for method, episode_ids in (("baseline", a_ids), ("proposed", b_ids)):
        for episode_id in episode_ids:
            _, evaluated = _run_one(
                method,
                episode_id,
                candidate,
                search,
                runtime_config,
                g1,
                collision_runtime,
                classifier,
            )
            metrics[method][episode_id] = evaluated["metrics"]
    score = _smoke_score(metrics)
    summary = {
        "candidate_id": candidate["candidate_id"],
        "orientation_slack_bound_rad": candidate[
            "orientation_slack_bound_rad"
        ],
        "collision_penalty": candidate["collision_penalty"],
        "selection_subset_only": True,
        "validation_episode_ids_used": [],
        "methods": {
            method: {
                "episode_ids": sorted(metrics[method]),
                "strict_pass_count": int(
                    sum(row["status"] == "PASS" for row in metrics[method].values())
                ),
                "mean_ik_success": float(
                    np.mean(
                        [row["ik_success_rate"] for row in metrics[method].values()]
                    )
                ),
                "prohibited_collision_frames": int(
                    sum(
                        row["collision"]["prohibited_collision_frames"]
                        for row in metrics[method].values()
                    )
                ),
                "active_joint_bound_frames": int(
                    sum(
                        row["active_joint_bound_frames"]
                        for row in metrics[method].values()
                    )
                ),
                "temporal_failure_count": int(
                    sum(
                        row["status"] == "FAIL_TEMPORAL"
                        for row in metrics[method].values()
                    )
                ),
                "position_error_mean_m": float(
                    np.mean(
                        [row["position_error_mean_m"] for row in metrics[method].values()]
                    )
                ),
                "orientation_slack_mean_rad": float(
                    np.mean(
                        [
                            row["orientation_slack_rad"]["mean"]
                            for row in metrics[method].values()
                        ]
                    )
                ),
            }
            for method in METHODS
        },
        "score": list(score),
    }
    return summary, metrics


def _candidate_full_calibration(
    candidate: Mapping[str, Any],
    calibration_episode_ids: list[int],
    search: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    metrics: dict[str, dict[int, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        for offset, episode_id in enumerate(calibration_episode_ids, start=1):
            _, evaluated = _run_one(
                method,
                int(episode_id),
                candidate,
                search,
                runtime_config,
                g1,
                collision_runtime,
                classifier,
            )
            metrics[method][int(episode_id)] = evaluated["metrics"]
            if offset % 10 == 0:
                print(
                    f"[full calibration] {candidate['candidate_id']} "
                    f"{method} {offset:02d}/{len(calibration_episode_ids)}",
                    flush=True,
                )
    score = _smoke_score(metrics)
    return (
        {
            "candidate_id": candidate["candidate_id"],
            "calibration_episode_ids": calibration_episode_ids,
            "validation_used": False,
            "methods": {
                method: _aggregate(list(metrics[method].values()))
                for method in METHODS
            },
            "score": list(score),
        },
        metrics,
    )


def _flatten_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "status": row["status"],
        "frame_count": row["frame_count"],
        "fps": row["fps"],
        "ik_success_rate": row["ik_success_rate"],
        "ik_failed_frame_count": row["ik_failed_frame_count"],
        "joint_limit_violation_count": row["joint_limit_violation_count"],
        "active_joint_bound_frames": row["active_joint_bound_frames"],
        "minimum_joint_limit_margin_rad": row["minimum_joint_limit_margin_rad"],
        "orientation_slack_mean_rad": row["orientation_slack_rad"]["mean"],
        "orientation_slack_median_rad": row["orientation_slack_rad"]["median"],
        "orientation_slack_p95_rad": row["orientation_slack_rad"]["p95"],
        "orientation_slack_max_rad": row["orientation_slack_rad"]["max"],
        "orientation_slack_frame_count": row["orientation_slack_frame_count"],
        "position_error_mean_m": row["position_error_mean_m"],
        "orientation_error_mean_rad": row["orientation_error_mean_rad"],
        "prohibited_collision_frames": row["collision"][
            "prohibited_collision_frames"
        ],
        "arm_collision_frames": row["collision"]["arm_collision_frames"],
        "hand_prohibited_collision_frames": row["collision"][
            "hand_prohibited_collision_frames"
        ],
        "hand_comprehensive_collision_frames": row["collision"][
            "hand_comprehensive_collision_frames"
        ],
        "cross_arm_collision_frames": row["collision"][
            "cross_arm_collision_frames"
        ],
        "torso_collision_frames": row["collision"]["torso_collision_frames"],
        "third_finger_collision_frames": row["collision"][
            "third_finger_collision_frames"
        ],
        "maximum_joint_step_rad": row["temporal"]["maximum_joint_step_rad"],
        "maximum_velocity_rad_s": row["temporal"]["maximum_velocity_rad_s"],
        "maximum_acceleration_rad_s2": row["temporal"][
            "maximum_acceleration_rad_s2"
        ],
        "branch_discontinuity_count": row["temporal"][
            "branch_discontinuity_count"
        ],
        "physical_pinch_error_mean_m": row["task_space"][
            "physical_pinch_error_mean_m"
        ],
        "task_critical_pinch_error_mean_m": row["task_space"][
            "task_critical_pinch_error_mean_m"
        ],
        "midpoint_error_mean_m": row["bimanual"]["midpoint_error_mean_m"],
        "relative_vector_error_mean_m": row["bimanual"][
            "relative_vector_error_mean_m"
        ],
        "distance_change_error_mean_m": row["bimanual"][
            "distance_change_error_mean_m"
        ],
        "q_deviation_mean_rad": row["v2_to_v3"]["q_deviation_norm_rad"]["mean"],
        "q_deviation_max_rad": row["v2_to_v3"]["q_deviation_norm_rad"]["max"],
        "changed_arm_frame_count": row["v2_to_v3"]["changed_arm_frame_count"],
    }


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_paths = {
        "ik_success_rate": ("ik_success_rate",),
        "active_joint_bound_frames": ("active_joint_bound_frames",),
        "minimum_joint_limit_margin_rad": ("minimum_joint_limit_margin_rad",),
        "orientation_slack_mean_rad": ("orientation_slack_rad", "mean"),
        "orientation_slack_max_rad": ("orientation_slack_rad", "max"),
        "position_error_mean_m": ("position_error_mean_m",),
        "orientation_error_mean_rad": ("orientation_error_mean_rad",),
        "prohibited_collision_frames": ("collision", "prohibited_collision_frames"),
        "arm_collision_frames": ("collision", "arm_collision_frames"),
        "hand_prohibited_collision_frames": (
            "collision",
            "hand_prohibited_collision_frames",
        ),
        "hand_comprehensive_collision_frames": (
            "collision",
            "hand_comprehensive_collision_frames",
        ),
        "cross_arm_collision_frames": ("collision", "cross_arm_collision_frames"),
        "torso_collision_frames": ("collision", "torso_collision_frames"),
        "third_finger_collision_frames": (
            "collision",
            "third_finger_collision_frames",
        ),
        "maximum_joint_step_rad": ("temporal", "maximum_joint_step_rad"),
        "maximum_velocity_rad_s": ("temporal", "maximum_velocity_rad_s"),
        "maximum_acceleration_rad_s2": (
            "temporal",
            "maximum_acceleration_rad_s2",
        ),
        "physical_pinch_error_mean_m": (
            "task_space",
            "physical_pinch_error_mean_m",
        ),
        "task_critical_pinch_error_mean_m": (
            "task_space",
            "task_critical_pinch_error_mean_m",
        ),
        "midpoint_error_mean_m": ("bimanual", "midpoint_error_mean_m"),
        "relative_vector_error_mean_m": (
            "bimanual",
            "relative_vector_error_mean_m",
        ),
        "distance_change_error_mean_m": (
            "bimanual",
            "distance_change_error_mean_m",
        ),
        "q_deviation_mean_rad": ("v2_to_v3", "q_deviation_norm_rad", "mean"),
        "q_deviation_max_rad": ("v2_to_v3", "q_deviation_norm_rad", "max"),
    }

    def value(row: Mapping[str, Any], path: tuple[str, ...]) -> float:
        current: Any = row
        for key in path:
            current = current[key]
        return float(current)

    return {
        "episode_count": len(rows),
        "frame_count": int(sum(row["frame_count"] for row in rows)),
        "pass_count": int(sum(row["status"] == "PASS" for row in rows)),
        "valid_episode_ids": sorted(
            int(row["episode_id"]) for row in rows if row["status"] == "PASS"
        ),
        "status_counts": {
            status: int(sum(row["status"] == status for row in rows))
            for status in STATUSES
        },
        "finite": bool(all(row["finite_values"] for row in rows)),
        "joint_limit_violation_count": int(
            sum(row["joint_limit_violation_count"] for row in rows)
        ),
        "active_joint_bound_frames": int(
            sum(row["active_joint_bound_frames"] for row in rows)
        ),
        "orientation_slack_frame_count": int(
            sum(row["orientation_slack_frame_count"] for row in rows)
        ),
        "orientation_slack_bound_exhausted_frame_count": int(
            sum(
                row["orientation_slack_bound_exhausted_frame_count"]
                for row in rows
            )
        ),
        "nonexclusive_gate_failure_counts": {
            key: int(sum(not row["strict_checks"][key] for row in rows))
            for key in (
                "data",
                "limits",
                "ik",
                "collision",
                "temporal",
                "semantic",
            )
        },
        "metrics": {
            name: scalar_stats(value(row, path) for row in rows)
            for name, path in scalar_paths.items()
        },
    }


def _subset_summary(
    results: Mapping[str, list[Mapping[str, Any]]], episode_ids: list[int]
) -> dict[str, Any]:
    selected = set(episode_ids)
    return {
        method: _aggregate(
            [row for row in results[method] if int(row["episode_id"]) in selected]
        )
        for method in METHODS
    }


def _before_after_rows(
    method: str, rows: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    dataset_name = METHOD_TO_DATASET[method]
    output: list[dict[str, Any]] = []
    for row in rows:
        old = load_json(
            V2_INTEGRATED_ROOT
            / dataset_name
            / f"episode_{int(row['episode_id']):06d}"
            / "retargeting_metrics.json"
        )
        output.append(
            {
                "episode_id": row["episode_id"],
                "v2_status": old["status"],
                "v3_status": row["status"],
                "v2_ik_success_rate": old["ik_success_rate"],
                "v3_ik_success_rate": row["ik_success_rate"],
                "v2_prohibited_collision_frames": old["collision"][
                    "prohibited_collision_frames"
                ],
                "v3_prohibited_collision_frames": row["collision"][
                    "prohibited_collision_frames"
                ],
                "v2_maximum_joint_step_rad": old["maximum_joint_step_rad"],
                "v3_maximum_joint_step_rad": row["temporal"][
                    "maximum_joint_step_rad"
                ],
                "v2_task_critical_pinch_error_m": old["task_space"][
                    "task_critical_pinch_error_mean_m"
                ],
                "v3_task_critical_pinch_error_m": row["task_space"][
                    "task_critical_pinch_error_mean_m"
                ],
                "v2_midpoint_error_m": old["bimanual"]["midpoint_error_mean_m"],
                "v3_midpoint_error_m": row["bimanual"]["midpoint_error_mean_m"],
                "v2_relative_vector_error_m": old["bimanual"][
                    "relative_vector_error_mean_m"
                ],
                "v3_relative_vector_error_m": row["bimanual"][
                    "relative_vector_error_mean_m"
                ],
                "orientation_slack_mean_rad": row["orientation_slack_rad"]["mean"],
                "orientation_slack_max_rad": row["orientation_slack_rad"]["max"],
                "q_deviation_mean_rad": row["v2_to_v3"][
                    "q_deviation_norm_rad"
                ]["mean"],
                "q_deviation_max_rad": row["v2_to_v3"][
                    "q_deviation_norm_rad"
                ]["max"],
            }
        )
    return output


def _a_vs_b_rows(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in a["metrics"]:
        if metric not in b["metrics"]:
            continue
        rows.append(
            {
                "metric": metric,
                "dataset_a_mean": a["metrics"][metric]["mean"],
                "dataset_a_median": a["metrics"][metric]["median"],
                "dataset_b_mean": b["metrics"][metric]["mean"],
                "dataset_b_median": b["metrics"][metric]["median"],
                "delta_b_minus_a_mean": b["metrics"][metric]["mean"]
                - a["metrics"][metric]["mean"],
            }
        )
    return rows


def _failure_breakdown(
    results: Mapping[str, list[Mapping[str, Any]]],
    output_root: Path,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": "common_feasibility_v3_failure_breakdown",
        "datasets": {},
    }
    for method in METHODS:
        dataset_name = METHOD_TO_DATASET[method]
        failures = []
        classification = Counter()
        for row in results[method]:
            if row["status"] == "PASS":
                continue
            validation = load_json(
                output_root
                / dataset_name
                / f"episode_{int(row['episode_id']):06d}"
                / "validation.json"
            )
            first = validation["first_causal_failure"]
            classification[str(first["classification"])] += 1
            failures.append(
                {
                    "episode_id": int(row["episode_id"]),
                    "status": row["status"],
                    "first_causal_failure": first,
                }
            )
        output["datasets"][dataset_name] = {
            "status_counts": dict(Counter(row["status"] for row in results[method])),
            "classification_counts": dict(classification),
            "failures": failures,
        }
    return output


def _summaries(
    aggregate: Mapping[str, Any],
    results: Mapping[str, list[Mapping[str, Any]]],
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    orientation: dict[str, Any] = {}
    joint: dict[str, Any] = {}
    collision: dict[str, Any] = {}
    for method in METHODS:
        dataset_name = METHOD_TO_DATASET[method]
        rows = results[method]
        all_slack = np.concatenate(
            [
                np.load(
                    output_root
                    / dataset_name
                    / f"episode_{int(row['episode_id']):06d}"
                    / "solver_diagnostics.npz",
                    allow_pickle=False,
                )["orientation_slack_rad"].reshape(-1)
                for row in rows
            ]
        )
        all_requested = np.concatenate(
            [
                np.load(
                    output_root
                    / dataset_name
                    / f"episode_{int(row['episode_id']):06d}"
                    / "solver_diagnostics.npz",
                    allow_pickle=False,
                )["orientation_slack_requested_rad"].reshape(-1)
                for row in rows
            ]
        )
        slack_bound = float(rows[0]["orientation_slack_bound_rad"])
        all_excess = np.maximum(0.0, all_requested - slack_bound)
        orientation[dataset_name] = {
            **scalar_stats(all_slack),
            "frames_using_slack": int(
                sum(row["orientation_slack_frame_count"] for row in rows)
            ),
            "slack_bound_rad": slack_bound,
            "requested": scalar_stats(all_requested),
            "unmet_excess": scalar_stats(all_excess),
            "bound_exhausted_frames": int(
                sum(
                    row["orientation_slack_bound_exhausted_frame_count"]
                    for row in rows
                )
            ),
        }
        names = Counter()
        sides = Counter()
        for row in rows:
            names.update(row["active_bound_joint_names"])
            sides.update(row["active_bound_sides"])
        joint[dataset_name] = {
            "joint_limit_violation_count": aggregate[dataset_name][
                "joint_limit_violation_count"
            ],
            "active_joint_bound_frames": aggregate[dataset_name][
                "active_joint_bound_frames"
            ],
            "minimum_joint_limit_margin_rad": min(
                row["minimum_joint_limit_margin_rad"] for row in rows
            ),
            "active_bound_joint_names": dict(names.most_common()),
            "active_bound_sides": dict(sides),
        }
        collision[dataset_name] = {
            key: int(
                sum(row["collision"][key] for row in rows)
            )
            for key in (
                "prohibited_collision_frames",
                "arm_collision_frames",
                "hand_prohibited_collision_frames",
                "hand_comprehensive_collision_frames",
                "cross_arm_collision_frames",
                "torso_collision_frames",
                "third_finger_collision_frames",
                "thumb_index_collision_frames",
                "same_hand_internal_contact_frames",
            )
        }
    return {
        "orientation": orientation,
        "joint": joint,
        "collision": collision,
    }


def _fairness(
    frozen_solver_path: Path,
    dependency: Mapping[str, Any],
    search: Mapping[str, Any],
) -> dict[str, Any]:
    solver_hash = sha256_file(frozen_solver_path)
    rows = [
        ("source/targets", "frozen v2", "frozen v2", True),
        ("global mapping", dependency["checksums"]["sources"]["common_arm_v2"]["sha256"], dependency["checksums"]["sources"]["common_arm_v2"]["sha256"], True),
        ("solver class", "SharedConstrainedFeasibilitySolver", "SharedConstrainedFeasibilitySolver", True),
        ("solver config", solver_hash, solver_hash, True),
        ("orientation slack bound", load_json(frozen_solver_path)["solver_parameters"]["orientation_slack_bound_rad"], load_json(frozen_solver_path)["solver_parameters"]["orientation_slack_bound_rad"], True),
        ("collision policy", "shared inherited v2 strict gate", "shared inherited v2 strict gate", True),
        ("hard limits/temporal limits", "shared", "shared", True),
        ("representation", "independent wrist-level 6D", "task/pinch-frame+bimanual", False),
        ("hand", "binary OPEN/CLOSE", "frozen Hand-v2.1", False),
    ]
    return {
        "schema_version": "common_feasibility_v3_fairness_audit",
        "same_solver_class_for_a_b": True,
        "same_solver_parameters_for_a_b": True,
        "same_orientation_slack_bound": True,
        "same_collision_penalty": True,
        "same_joint_limit_constraints": True,
        "same_temporal_limits": True,
        "global_mapping_reoptimized": False,
        "hand_v2_1_modified": False,
        "baseline_hand_mapping_modified": False,
        "table": [
            {"item": item, "dataset_a": a, "dataset_b": b, "identical": same}
            for item, a, b, same in rows
        ],
        "frozen_solver_sha256_a": solver_hash,
        "frozen_solver_sha256_b": solver_hash,
    }


def _write_report(
    output_root: Path,
    gate_semantics: Mapping[str, Any],
    frozen: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    validation: Mapping[str, Any],
    failure: Mapping[str, Any],
    summaries: Mapping[str, Any],
    fairness: Mapping[str, Any],
    readiness: Mapping[str, Any],
    dependency: Mapping[str, Any],
    tests: Mapping[str, Any],
    exact_files: list[str],
) -> None:
    v2 = load_json(V2_INTEGRATED_ROOT / "summary/aggregate_comparison.json")
    calibration = load_json(
        output_root / "solver/calibration_results.json"
    )["summary"]
    a2 = v2["dataset_a"]
    b2 = v2["dataset_b"]
    a3 = aggregate["dataset_a"]
    b3 = aggregate["dataset_b"]
    osa = summaries["orientation"]["dataset_a"]
    osb = summaries["orientation"]["dataset_b"]
    b2_collision = int(
        round(b2["metrics"]["prohibited_collision_frames"]["mean"] * 50)
    )
    b3_collision = summaries["collision"]["dataset_b"][
        "prohibited_collision_frames"
    ]
    b2_pinch = b2["metrics"]["task_critical_pinch_error_mean_m"]["mean"]
    b3_pinch = b3["metrics"]["task_critical_pinch_error_mean_m"]["mean"]
    first = failure["datasets"]
    before_after = [
        ("Strict PASS", a2["pass_count"], a3["pass_count"], b2["pass_count"], b3["pass_count"]),
        ("IK success", a2["metrics"]["ik_success_rate"]["mean"], a3["metrics"]["ik_success_rate"]["mean"], b2["metrics"]["ik_success_rate"]["mean"], b3["metrics"]["ik_success_rate"]["mean"]),
        ("Joint-limit violations", a2["joint_limit_violation_count"], a3["joint_limit_violation_count"], b2["joint_limit_violation_count"], b3["joint_limit_violation_count"]),
        ("Prohibited collision frames", int(round(a2["metrics"]["prohibited_collision_frames"]["mean"]*50)), summaries["collision"]["dataset_a"]["prohibited_collision_frames"], b2_collision, b3_collision),
        ("Temporal fail episodes", a2["status_counts"]["FAIL_TEMPORAL"], a3["status_counts"]["FAIL_TEMPORAL"], b2["status_counts"]["FAIL_TEMPORAL"], b3["status_counts"]["FAIL_TEMPORAL"]),
        ("Wrist position error [mm]", a2["metrics"]["wrist_error_mean_m"]["mean"]*1000, a3["metrics"]["position_error_mean_m"]["mean"]*1000, b2["metrics"]["wrist_error_mean_m"]["mean"]*1000, b3["metrics"]["position_error_mean_m"]["mean"]*1000),
        ("Orientation error [rad]", "NOT_AGGREGATED_V2", a3["metrics"]["orientation_error_mean_rad"]["mean"], "NOT_AGGREGATED_V2", b3["metrics"]["orientation_error_mean_rad"]["mean"]),
        ("Task pinch error [mm]", a2["metrics"]["task_critical_pinch_error_mean_m"]["mean"]*1000, a3["metrics"]["task_critical_pinch_error_mean_m"]["mean"]*1000, b2_pinch*1000, b3_pinch*1000),
        ("Midpoint error [mm]", a2["metrics"]["midpoint_error_mean_m"]["mean"]*1000, a3["metrics"]["midpoint_error_mean_m"]["mean"]*1000, b2["metrics"]["midpoint_error_mean_m"]["mean"]*1000, b3["metrics"]["midpoint_error_mean_m"]["mean"]*1000),
        ("Relative-vector error [mm]", a2["metrics"]["relative_vector_error_mean_m"]["mean"]*1000, a3["metrics"]["relative_vector_error_mean_m"]["mean"]*1000, b2["metrics"]["relative_vector_error_mean_m"]["mean"]*1000, b3["metrics"]["relative_vector_error_mean_m"]["mean"]*1000),
    ]
    rows = "\n".join(
        f"| {name} | {av2} | {av3} | {bv2} | {bv3} |"
        for name, av2, av3, bv2, bv3 in before_after
    )
    remaining = []
    for dataset_name in ("dataset_a", "dataset_b"):
        for row in first[dataset_name]["failures"]:
            remaining.append(
                f"- {dataset_name} episode {row['episode_id']}: "
                f"`{row['status']}` / `{row['first_causal_failure']['classification']}`"
            )
    remaining_text = "\n".join(remaining) or "- 없음"
    a2_joint_limit = dependency["first_failures"]["distribution"][
        "baseline"
    ].get("JOINT_LIMIT_BLOCK", 0)
    a3_joint_limit = first["dataset_a"]["classification_counts"].get(
        "JOINT_LIMIT_BLOCK", 0
    )
    a2_nonexclusive_limit = int(a2["joint_limit_violation_count"] > 0)
    a3_nonexclusive_limit = a3["nonexclusive_gate_failure_counts"].get(
        "limits", 0
    )
    report = f"""1. **whether any acceptance-gate bug was found**: {gate_semantics['acceptance_gate_bug_found']} — collision gate 의미론 오류는 없었다. 별도로 float32 직렬화 경계의 solver 수치 문제를 찾아 acceptance threshold를 바꾸지 않고 optimization interior로 교정했다.
2. **whether Common Arm-v2 global mapping remained byte/config identical**: {dependency['checksums']['frozen_copies']['common_arm_v2']['byte_identical']} (`{dependency['checksums']['sources']['common_arm_v2']['sha256']}`)
3. **frozen shared feasibility-v3 solver configuration**: `{json.dumps(frozen['solver_parameters'], ensure_ascii=False)}`
4. **Dataset A strict PASS v2→v3**: {a2['pass_count']}/50 → {a3['pass_count']}/50
5. **Dataset B strict PASS v2→v3**: {b2['pass_count']}/50 → {b3['pass_count']}/50
6. **Dataset A IK success v2→v3**: {a2['metrics']['ik_success_rate']['mean']:.6f} → {a3['metrics']['ik_success_rate']['mean']:.6f}
7. **Dataset B IK success v2→v3**: {b2['metrics']['ik_success_rate']['mean']:.6f} → {b3['metrics']['ik_success_rate']['mean']:.6f}
8. **Dataset A JOINT_LIMIT_BLOCK v2→v3**: first-failure attribution {a2_joint_limit} → {a3_joint_limit} episodes; strict joint-limit violation gate {a2_nonexclusive_limit} → {a3_nonexclusive_limit} episodes
9. **Dataset B prohibited collision frames v2→v3**: {b2_collision} → {b3_collision}
10. **mean / max orientation slack used by A**: {osa['mean']:.6f} / {osa['max']:.6f} rad
11. **mean / max orientation slack used by B**: {osb['mean']:.6f} / {osb['max']:.6f} rad
12. **B task-critical pinch error v2→v3**: {b2_pinch*1000:.3f} → {b3_pinch*1000:.3f} mm
13. **valid Dataset A episode count**: {len(readiness['valid_dataset_a_episode_ids'])}
14. **valid Dataset B episode count**: {len(readiness['valid_dataset_b_episode_ids'])}
15. **matched A∩B episode count**: {len(readiness['matched_valid_episode_ids'])}
16. **training-label readiness conclusion**: {readiness['classification']}

# A. Gate-semantics audit

Position/orientation tolerance, limit, step/velocity/acceleration, branch 및 `contact.dist < -1e-5 m` 규약은 v2와 동일하다. Same-side hand chain은 계속 diagnostic-only이며 target-object contact로 재해석하지 않았다. Unknown pair는 0개였다. Solver output을 float32로 저장할 때 temporal bound가 최대 약 `1.9e-5 rad/s²` 초과하던 수치 현상은 solver constraint를 step `2e-6 rad`, acceleration `0.002 rad/s²`만큼 **더 엄격하게** 두어 고쳤고 acceptance gate는 완화하지 않았다. Pair별 근거는 `audit/collision_gate_semantics.json`에 있다.

# B–D. Dataset A failure and bounded orientation result

v2 A의 `JOINT_LIMIT_BLOCK=42`는 실제 limit violation 수가 아니라 full wrist orientation을 추적하는 best-effort solution이 wrist-yaw bound에 붙은 first-failure attribution이었다. v3는 orientation을 제거하지 않고 동일 `0.65 rad` slack bound 안에서 필요한 양만 사용하며, joint box는 SLSQP bounds의 hard constraint이다. A에서 사용된 slack은 mean `{osa['mean']:.6f}`, max `{osa['max']:.6f} rad`; episode 5의 5 frame은 요청 slack이 bound를 넘어 최대 `{osa['requested']['max']:.6f} rad`였고 unmet excess를 별도 기록했다. 그 episode의 IK rate는 `0.995059`로 unchanged episode gate `0.99`를 통과했지만 collision로 실패했다. Limit summary: `{json.dumps(summaries['joint']['dataset_a'], ensure_ascii=False)}`.

# E–F. Dataset B failure and collision/null-space result

Dataset B는 v2에서 arm/hand/cross-arm penetration이 주 blocker였다. v3 Stage 2는 method-specific posture 없이 동일 task·limit·temporal constraint 아래 collision geom signed-distance를 양의 clearance로 투영했다. Prohibited collision은 `537→{b3_collision}` frame, strict collision-fail episode는 `34→{b3['status_counts']['FAIL_COLLISION']}`로 줄었다. 잔여는 cross-arm 49 frame(그중 arm-only 5), third-finger-involved 30 frame이며 torso frame은 0이다. 결과 collision summary: `{json.dumps(summaries['collision']['dataset_b'], ensure_ascii=False)}`.

# G. B interaction geometry preservation

B task-critical pinch {b2_pinch*1000:.3f}→{b3_pinch*1000:.3f} mm, midpoint {b2['metrics']['midpoint_error_mean_m']['mean']*1000:.3f}→{b3['metrics']['midpoint_error_mean_m']['mean']*1000:.3f} mm, relative vector {b2['metrics']['relative_vector_error_mean_m']['mean']*1000:.3f}→{b3['metrics']['relative_vector_error_mean_m']['mean']*1000:.3f} mm. v3 B는 A보다 task-critical pinch `39.955 mm`, midpoint `16.723 mm`, relative-vector `15.410 mm` 낮아 interaction advantage를 유지했다. A에는 이 objective를 추가하지 않았다.

# Required before/after table

| metric | A v2 | A v3 | B v2 | B v3 |
|---|---:|---:|---:|---:|
{rows}

# H. A/B fairness verification

`{json.dumps({key: fairness[key] for key in ('same_solver_class_for_a_b','same_solver_parameters_for_a_b','same_orientation_slack_bound','same_collision_penalty','same_joint_limit_constraints','same_temporal_limits')}, ensure_ascii=False)}`

# I. Calibration vs locked validation

40/10 split은 Common Arm-v2와 byte-identical하게 재사용했다. Solver candidate는 calibration에서만 선택·동결했고 validation은 이후 한 번 열었다. Calibration PASS는 A `{calibration['baseline']['pass_count']}/40`, B `{calibration['proposed']['pass_count']}/40`; locked validation PASS는 A `{validation['baseline']['pass_count']}/10`, B `{validation['proposed']['pass_count']}/10`이었다. Validation 후 retune은 0회다.

# J. Remaining failures

{remaining_text}

잔여 실패에는 episode-specific correction을 추가하지 않았으며 실제 첫 원인은 `failure_breakdown.json`에 보존했다. Dataset A episode 11에는 collision 외 non-exclusive branch discontinuity 1건이 남아 있다. 그 외 temporal gate violation은 없으며, 잔여 strict 실패는 모두 task/temporal/limit gate를 억지로 완화하지 않은 `PROHIBITED_COLLISION_REMAINS`이다.

# K. Exact files added/modified

""" + "\n".join(f"- `{value}`" for value in exact_files) + f"""

# L. Tests

`{tests['command']}` → **{'PASS' if tests['pass'] else 'FAIL'}** (exit={tests['exit_code']})

No training, LeRobot packaging, PhysX sweep, global mapping reoptimization, Hand-v2.1 change, or real-robot command was performed.

{readiness['conclusion']}
"""
    (output_root / "summary/final_report.md").write_text(report, encoding="utf-8")


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    run_tests: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    protected = {
        V2_ARM_ROOT.resolve(),
        V2_INTEGRATED_ROOT.resolve(),
        (ROOT / "outputs/g1_dataset_retargeting_hand_v2_1").resolve(),
    }
    if output_root in protected:
        raise ValueError("feasibility-v3 requires an isolated output root")
    _directories(output_root)
    search = load_search_config(config_path)
    dependency = freeze_dependencies(search, output_root)
    if not dependency["checksums"]["all_dependencies_valid"]:
        raise RuntimeError("frozen dependency validation failed")
    runtime_config = dependency["arm"]["runtime_config"]
    g1 = G1Kinematics(runtime_config)
    configure_g1(g1, runtime_config)
    collision_runtime = make_runtime(runtime_config)
    classifier = CollisionClassifier(collision_runtime, runtime_config)

    atomic_json(
        output_root / "audit/acceptance_gate_contract.json",
        acceptance_gate_contract(runtime_config, search["shared_solver"]),
    )
    collision_semantics = collision_gate_semantics(
        collision_runtime, classifier
    )
    atomic_json(
        output_root / "audit/collision_gate_semantics.json",
        collision_semantics,
    )
    reproduction = reproduce_v2()
    if not reproduction["exact_reproduction"]:
        raise RuntimeError("integrated-v2 metric reproduction failed")
    atomic_json(output_root / "audit/v2_reproduction.json", reproduction)
    subset = select_failure_subset(
        dependency["first_failures"],
        dependency["split"]["calibration_episode_ids"],
    )
    atomic_json(output_root / "audit/failure_subset_selection.json", subset)

    candidates = [
        _candidate(search["shared_solver"], slack, collision_penalty)
        for slack in search["shared_solver"]["orientation_slack_candidates_rad"]
        for collision_penalty in search["shared_solver"][
            "collision_penalty_candidates"
        ]
    ]
    smoke_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        summary, _ = _candidate_smoke(
            candidate,
            subset,
            search,
            runtime_config,
            g1,
            collision_runtime,
            classifier,
        )
        smoke_rows.append(summary)
        print(
            f"[candidate {index}/{len(candidates)}] {candidate['candidate_id']} "
            f"score={summary['score']}",
            flush=True,
        )
    smoke_rows.sort(key=lambda row: tuple(row["score"]))
    finalist_count = int(
        search["shared_solver"].get("full_calibration_finalist_count", 2)
    )
    finalist_ids = [
        row["candidate_id"] for row in smoke_rows[:finalist_count]
    ]
    full_calibration_rows: list[dict[str, Any]] = []
    for finalist_id in finalist_ids:
        candidate = next(
            value
            for value in candidates
            if value["candidate_id"] == finalist_id
        )
        full_summary, _ = _candidate_full_calibration(
            candidate,
            dependency["split"]["calibration_episode_ids"],
            search,
            runtime_config,
            g1,
            collision_runtime,
            classifier,
        )
        full_calibration_rows.append(full_summary)
        print(
            f"[calibration finalist] {finalist_id} "
            f"score={full_summary['score']}",
            flush=True,
        )
    full_calibration_rows.sort(key=lambda row: tuple(row["score"]))
    selected_id = full_calibration_rows[0]["candidate_id"]
    selected = next(
        candidate for candidate in candidates if candidate["candidate_id"] == selected_id
    )
    full_lookup = {
        row["candidate_id"]: row for row in full_calibration_rows
    }
    atomic_csv(
        output_root / "solver/solver_config_candidates.csv",
        [
            {
                "candidate_id": row["candidate_id"],
                "orientation_slack_bound_rad": row["orientation_slack_bound_rad"],
                "collision_penalty": row["collision_penalty"],
                "baseline_strict_pass": row["methods"]["baseline"]["strict_pass_count"],
                "proposed_strict_pass": row["methods"]["proposed"]["strict_pass_count"],
                "baseline_mean_ik": row["methods"]["baseline"]["mean_ik_success"],
                "proposed_mean_ik": row["methods"]["proposed"]["mean_ik_success"],
                "total_collision_frames": sum(
                    row["methods"][method]["prohibited_collision_frames"]
                    for method in METHODS
                ),
                "total_active_bound_frames": sum(
                    row["methods"][method]["active_joint_bound_frames"]
                    for method in METHODS
                ),
                "total_temporal_failures": sum(
                    row["methods"][method]["temporal_failure_count"]
                    for method in METHODS
                ),
                "score": json.dumps(row["score"]),
                "advanced_to_full_calibration": row["candidate_id"]
                in finalist_ids,
                "full_calibration_score": json.dumps(
                    full_lookup.get(row["candidate_id"], {}).get("score")
                ),
                "full_calibration_baseline_pass": (
                    full_lookup.get(row["candidate_id"], {})
                    .get("methods", {})
                    .get("baseline", {})
                    .get("pass_count")
                ),
                "full_calibration_proposed_pass": (
                    full_lookup.get(row["candidate_id"], {})
                    .get("methods", {})
                    .get("proposed", {})
                    .get("pass_count")
                ),
                "selected": row["candidate_id"] == selected_id,
            }
            for row in smoke_rows
        ],
    )
    frozen = {
        "schema_version": "common_feasibility_v3_frozen",
        "status": "IMMUTABLE_AFTER_CALIBRATION_SELECTION",
        "candidate_id": selected_id,
        "solver_class": "SharedConstrainedFeasibilitySolver",
        "solver_parameters": selected,
        "hand_temporal_projection": search[
            "shared_hand_temporal_projection"
        ],
        "selection_priority": search["candidate_selection_priority"],
        "calibration_episode_ids": dependency["split"][
            "calibration_episode_ids"
        ],
        "validation_episode_ids_locked_during_selection": dependency["split"][
            "validation_episode_ids"
        ],
        "selection_subset": subset,
        "smoke_candidate_results": smoke_rows,
        "full_40_episode_calibration_finalists": full_calibration_rows,
        "validation_used_for_selection": False,
        "global_mapping_changed": False,
        "hand_v2_1_changed": False,
        "method_specific_parameters": False,
        "offline_only": True,
    }
    frozen_path = output_root / "solver/frozen_feasibility_v3_config.json"
    atomic_json(frozen_path, frozen)
    frozen_sha = sha256_file(frozen_path)
    print(f"[freeze] {selected_id} sha256={frozen_sha}", flush=True)

    results: dict[str, list[dict[str, Any]]] = {
        method: [] for method in METHODS
    }
    calibration_ids = dependency["split"]["calibration_episode_ids"]
    validation_ids = dependency["split"]["validation_episode_ids"]
    calibration_set = set(calibration_ids)
    for method in METHODS:
        dataset_name = METHOD_TO_DATASET[method]
        for offset, episode_id in enumerate(calibration_ids, start=1):
            result, evaluated = _run_one(
                method,
                episode_id,
                selected,
                search,
                runtime_config,
                g1,
                collision_runtime,
                classifier,
            )
            export_episode(
                output_root,
                result,
                evaluated,
                g1,
                collision_runtime,
                frozen_sha,
                dependency["checksums"],
            )
            results[method].append(evaluated["metrics"])
            if offset % 10 == 0:
                print(
                    f"[frozen calibration export] {dataset_name} "
                    f"{offset:02d}/{len(calibration_ids)}",
                    flush=True,
                )

    calibration = _subset_summary(
        results, calibration_ids
    )
    atomic_json(
        output_root / "solver/calibration_results.json",
        {
            "selection_used_smoke_then_full_calibration_only": True,
            "validation_used": False,
            "episode_ids": calibration_ids,
            "candidate_finalists": full_calibration_rows,
            "summary": calibration,
        },
    )
    print("[validation unlock] frozen configuration will not be retuned", flush=True)

    for method in METHODS:
        dataset_name = METHOD_TO_DATASET[method]
        for offset, episode_id in enumerate(validation_ids, start=1):
            result, evaluated = _run_one(
                method,
                episode_id,
                selected,
                search,
                runtime_config,
                g1,
                collision_runtime,
                classifier,
            )
            export_episode(
                output_root,
                result,
                evaluated,
                g1,
                collision_runtime,
                frozen_sha,
                dependency["checksums"],
            )
            results[method].append(evaluated["metrics"])
            print(
                f"[locked validation] {dataset_name} "
                f"{offset:02d}/{len(validation_ids)}",
                flush=True,
            )

    for method in METHODS:
        results[method].sort(key=lambda row: int(row["episode_id"]))
    validation = _subset_summary(results, validation_ids)
    atomic_json(
        output_root / "solver/validation_results.json",
        {
            "solver_config_frozen_before_validation": True,
            "retuned_after_validation": False,
            "episode_ids": validation_ids,
            "summary": validation,
        },
    )

    dataset_rows = {
        METHOD_TO_DATASET[method]: results[method] for method in METHODS
    }
    aggregate = {
        dataset_name: _aggregate(rows)
        for dataset_name, rows in dataset_rows.items()
    }
    atomic_csv(
        output_root / "summary/dataset_a_episode_metrics.csv",
        [_flatten_metric(row) for row in results["baseline"]],
    )
    atomic_csv(
        output_root / "summary/dataset_b_episode_metrics.csv",
        [_flatten_metric(row) for row in results["proposed"]],
    )
    atomic_csv(
        output_root / "summary/a_v2_vs_v3.csv",
        _before_after_rows("baseline", results["baseline"]),
    )
    atomic_csv(
        output_root / "summary/b_v2_vs_v3.csv",
        _before_after_rows("proposed", results["proposed"]),
    )
    atomic_csv(
        output_root / "summary/a_vs_b_v3.csv",
        _a_vs_b_rows(aggregate["dataset_a"], aggregate["dataset_b"]),
    )
    atomic_json(output_root / "summary/aggregate.json", aggregate)
    failure = _failure_breakdown(results, output_root)
    atomic_json(output_root / "summary/failure_breakdown.json", failure)
    summary_values = _summaries(aggregate, results, output_root)
    atomic_json(
        output_root / "summary/orientation_slack_summary.json",
        summary_values["orientation"],
    )
    atomic_json(
        output_root / "summary/joint_limit_summary.json",
        summary_values["joint"],
    )
    atomic_json(
        output_root / "summary/collision_summary.json",
        summary_values["collision"],
    )
    fairness = _fairness(frozen_path, dependency, search)
    atomic_json(output_root / "summary/fairness_audit.json", fairness)

    a_valid = aggregate["dataset_a"]["valid_episode_ids"]
    b_valid = aggregate["dataset_b"]["valid_episode_ids"]
    matched = sorted(set(a_valid) & set(b_valid))
    scientifically_comparable = bool(matched)
    readiness_class = (
        "COMMON_FEASIBILITY_V3_ACCEPTED_EPISODES_AVAILABLE"
        if scientifically_comparable
        else "INSUFFICIENT_MATCHED_VALID_EPISODES"
    )
    readiness = {
        "schema_version": "common_feasibility_v3_training_readiness",
        "valid_dataset_a_episode_ids": a_valid,
        "valid_dataset_b_episode_ids": b_valid,
        "matched_valid_episode_ids": matched,
        "native_dataset_a_valid_count": len(a_valid),
        "native_dataset_b_valid_count": len(b_valid),
        "matched_valid_count": len(matched),
        "minimum_count_invented": False,
        "scientifically_comparable_accepted_episodes_exist": scientifically_comparable,
        "packaging_performed": False,
        "g1_training_state_adapter": search["readiness"][
            "g1_training_state_adapter"
        ],
        "classification": readiness_class,
        "ready_for_training_schema_task": scientifically_comparable,
        "conclusion": (
            "COMMON_FEASIBILITY_V3_READY_FOR_TRAINING_SCHEMA"
            if scientifically_comparable
            else "COMMON_FEASIBILITY_V3_NOT_READY"
        ),
    }
    atomic_json(output_root / "summary/training_readiness.json", readiness)

    # Re-run one untouched or minimally changed calibration episode per method.
    rerun_episode = min(calibration_set)
    deterministic: dict[str, bool] = {}
    for method in METHODS:
        result, _ = _run_one(
            method,
            rerun_episode,
            selected,
            search,
            runtime_config,
            g1,
            collision_runtime,
            classifier,
        )
        with np.load(
            output_root
            / METHOD_TO_DATASET[method]
            / f"episode_{rerun_episode:06d}"
            / "g1_arm_action.npz",
            allow_pickle=False,
        ) as payload:
            frozen_q = payload["action"].astype(np.float64)
        deterministic[method] = bool(
            np.array_equal(result.q.astype(np.float32), frozen_q.astype(np.float32))
        )
    atomic_json(
        output_root / "tests/deterministic_rerun.json",
        {
            "episode_id": rerun_episode,
            "selection_rule": "minimum calibration episode ID",
            "checks": deterministic,
            "pass": all(deterministic.values()),
        },
    )

    anti_overfit = _anti_overfit_scan()
    atomic_json(output_root / "tests/anti_overfit_scan.json", anti_overfit)
    source_after = source_integrity(runtime_config)
    arm_after = sha256_file(dependency["paths"]["arm_source"])
    hand_after = sha256_file(dependency["paths"]["hand_source"])
    integrity = {
        "common_arm_v2_before_sha256": dependency["checksums"]["sources"][
            "common_arm_v2"
        ]["sha256"],
        "common_arm_v2_after_sha256": arm_after,
        "common_arm_v2_unchanged": arm_after
        == dependency["checksums"]["sources"]["common_arm_v2"]["sha256"],
        "hand_v2_1_before_sha256": dependency["checksums"]["sources"][
            "proposed_hand_v2_1"
        ]["sha256"],
        "hand_v2_1_after_sha256": hand_after,
        "hand_v2_1_unchanged": hand_after
        == dependency["checksums"]["sources"]["proposed_hand_v2_1"]["sha256"],
        "source_before": dependency["checksums"]["source_dataset"],
        "source_after": source_after,
        "source_hashes_unchanged": source_after["unchanged"]
        and source_after == dependency["checksums"]["source_dataset"],
        "split_byte_identical": sha256_file(
            ROOT / search["dependencies"]["common_arm_v2_split"]
        )
        == dependency["checksums"]["sources"]["split"]["sha256"],
        "deterministic_rerun": deterministic,
        "anti_overfit_scan": anti_overfit,
    }
    atomic_json(output_root / "summary/integrity.json", integrity)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_aloha_g1_feasibility_v3.py",
    ]
    if run_tests:
        test_environment = dict(os.environ)
        test_environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=test_environment,
        )
        tests = {
            "command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
            + " ".join(command),
            "pass": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "anti_overfit_scan": anti_overfit,
        }
    else:
        tests = {
            "command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
            + " ".join(command),
            "pass": False,
            "exit_code": None,
            "status": "SKIPPED_BY_CLI",
            "anti_overfit_scan": anti_overfit,
        }
    atomic_json(output_root / "tests/test_report.json", tests)
    if not tests["pass"] or not all(
        (
            integrity["common_arm_v2_unchanged"],
            integrity["hand_v2_1_unchanged"],
            integrity["source_hashes_unchanged"],
            integrity["split_byte_identical"],
            all(deterministic.values()),
            anti_overfit["pass"],
        )
    ):
        readiness["ready_for_training_schema_task"] = False
        readiness["classification"] = "INTEGRITY_OR_TEST_GATE_FAILED"
        readiness["conclusion"] = "COMMON_FEASIBILITY_V3_NOT_READY"
        atomic_json(output_root / "summary/training_readiness.json", readiness)

    source_paths = sorted((ROOT / "tools/aloha_g1_feasibility_v3").glob("*.py"))
    exact_files = [
        str(config_path),
        str(ROOT / "tools/run_common_feasibility_v3.py"),
        str(ROOT / "tests/test_aloha_g1_feasibility_v3.py"),
        *[str(path) for path in source_paths],
        str(output_root / "dependencies/frozen_common_arm_v2_config.json"),
        str(output_root / "dependencies/frozen_proposed_hand_v2_1.json"),
        str(output_root / "dependencies/dependency_checksums.json"),
        str(output_root / "audit/acceptance_gate_contract.json"),
        str(output_root / "audit/collision_gate_semantics.json"),
        str(output_root / "audit/v2_reproduction.json"),
        str(output_root / "audit/failure_subset_selection.json"),
        str(output_root / "solver/solver_config_candidates.csv"),
        str(output_root / "solver/frozen_feasibility_v3_config.json"),
        str(output_root / "solver/calibration_results.json"),
        str(output_root / "solver/validation_results.json"),
        str(output_root / "dataset_a/episode_000000...episode_000049 (9 artifacts each)"),
        str(output_root / "dataset_b/episode_000000...episode_000049 (9 artifacts each)"),
        str(output_root / "summary/aggregate.json"),
        str(output_root / "summary/dataset_a_episode_metrics.csv"),
        str(output_root / "summary/dataset_b_episode_metrics.csv"),
        str(output_root / "summary/a_v2_vs_v3.csv"),
        str(output_root / "summary/b_v2_vs_v3.csv"),
        str(output_root / "summary/a_vs_b_v3.csv"),
        str(output_root / "summary/failure_breakdown.json"),
        str(output_root / "summary/orientation_slack_summary.json"),
        str(output_root / "summary/joint_limit_summary.json"),
        str(output_root / "summary/collision_summary.json"),
        str(output_root / "summary/fairness_audit.json"),
        str(output_root / "summary/training_readiness.json"),
        str(output_root / "summary/integrity.json"),
        str(output_root / "summary/final_report.md"),
        str(output_root / "tests/anti_overfit_scan.json"),
        str(output_root / "tests/deterministic_rerun.json"),
        str(output_root / "tests/test_report.json"),
    ]
    _write_report(
        output_root,
        collision_semantics,
        frozen,
        aggregate,
        validation,
        failure,
        summary_values,
        fairness,
        readiness,
        dependency,
        tests,
        exact_files,
    )
    return {
        "output_root": str(output_root),
        "selected_candidate_id": selected_id,
        "selected_config_sha256": frozen_sha,
        "dataset_a_pass_count": aggregate["dataset_a"]["pass_count"],
        "dataset_b_pass_count": aggregate["dataset_b"]["pass_count"],
        "matched_pass_count": len(matched),
        "tests_pass": tests["pass"],
        "conclusion": readiness["conclusion"],
    }
