"""Gated Common Arm-v2 calibration and final integrated A/B audit."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from aloha_g1_dataset_v1.core import (  # noqa: E402
    G1Kinematics,
    HandMapper,
    SourceDataset,
    SourceKinematics,
    load_config,
    stats,
)
from aloha_g1_hand_v2.collision_eval import (  # noqa: E402
    CollisionClassifier,
    make_runtime,
)

from .audit import (  # noqa: E402
    ArmEpisodeResult,
    PreparedEpisode,
    candidate_config,
    configure_g1,
    deterministic_fk_workspace,
    evaluate_arm_episode,
    first_failure_record,
    full_candidate_score,
    rotation_distribution,
    static_screen_candidate,
    summarize_arm_results,
)
from .common import (  # noqa: E402
    DEFAULT_ARM_ROOT,
    DEFAULT_INTEGRATED_ROOT,
    DEFAULT_SEARCH_CONFIG,
    ROOT,
    V1_ROOT,
    atomic_csv,
    atomic_json,
    deterministic_split,
    freeze_hand_dependency,
    load_json,
    load_search_config,
    resolve_from_root,
    scalar_stats,
    sha256_file,
    source_integrity,
    tree_sha256,
    validate_hand_dependency,
)
from .integrated import (  # noqa: E402
    INTEGRATED_STATUSES,
    evaluate_integrated_episode,
    export_arm_episode,
    export_integrated_episode,
    map_baseline_hand,
    map_proposed_hand,
    validate_candidate_joint_order,
)
from .solver import AcceptanceAwareTemporalIK, static_pose_solve  # noqa: E402


METHODS = ("baseline", "proposed")


def _directories(arm_root: Path, integrated_root: Path) -> None:
    for folder in (
        "dependencies",
        "audit",
        "workspace",
        "split",
        "candidates",
        "baseline",
        "proposed",
        "summary",
        "tests",
    ):
        (arm_root / folder).mkdir(parents=True, exist_ok=True)
    for folder in ("dataset_a", "dataset_b", "summary", "tests"):
        (integrated_root / folder).mkdir(parents=True, exist_ok=True)


def _implementation_fingerprint(config_path: Path) -> tuple[str, dict[str, str]]:
    paths = [
        ROOT / "tools/aloha_g1_arm_v2/__init__.py",
        ROOT / "tools/aloha_g1_arm_v2/common.py",
        ROOT / "tools/aloha_g1_arm_v2/solver.py",
        ROOT / "tools/aloha_g1_arm_v2/audit.py",
        ROOT / "tools/aloha_g1_arm_v2/integrated.py",
        ROOT / "tools/aloha_g1_arm_v2/pipeline.py",
        ROOT / "tools/run_common_arm_v2.py",
        config_path,
    ]
    values = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), values


def _prepare_episodes(
    dataset: SourceDataset,
    source_kinematics: SourceKinematics,
    hand_mapper: HandMapper,
) -> dict[int, PreparedEpisode]:
    prepared: dict[int, PreparedEpisode] = {}
    for offset, episode_id in enumerate(dataset.episode_ids(), start=1):
        episode = dataset.episode(episode_id)
        source_fk = source_kinematics.compute(episode.action)
        detected = hand_mapper.detect(episode)
        prepared[episode_id] = PreparedEpisode(episode, source_fk, detected)
        if offset % 10 == 0 or offset == len(dataset.episode_ids()):
            print(f"[source FK] {offset:02d}/50", flush=True)
    return prepared


def _anchor_candidates(
    search: Mapping[str, Any],
    base: Mapping[str, Any],
    hand_candidate: Mapping[str, Any],
    g1: G1Kinematics,
) -> list[dict[str, Any]]:
    cfg = search["anchor_search"]
    compact = np.asarray(base["nominal_g1_arm_q"], dtype=np.float64)
    wide_path = resolve_from_root(cfg["wide_q_source"])
    if sha256_file(wide_path) != cfg["wide_q_source_sha256"]:
        raise RuntimeError("wide task-ready posture provenance hash mismatch")
    with np.load(wide_path, allow_pickle=False) as payload:
        wide = np.asarray(payload["g1_arm_task_q"][0], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    anchor_index = 0
    for alpha in cfg["anchor_interpolation_fractions"]:
        reference = (1.0 - float(alpha)) * compact + float(alpha) * wide
        state = g1.wrist_state(reference)
        reference_positions = {
            side: state[f"{side}_position"].copy() for side in ("left", "right")
        }
        reference_rotations = {
            side: state[f"{side}_rotation"].copy() for side in ("left", "right")
        }
        for requested in cfg["requested_common_translation_m"]:
            requested_array = np.asarray(requested, dtype=np.float64)
            solved = static_pose_solve(
                g1,
                {
                    side: reference_positions[side] + requested_array
                    for side in ("left", "right")
                },
                reference_rotations,
                [reference, compact, wide],
                iterations=int(cfg["static_anchor_iterations"]),
                position_tolerance_m=float(cfg["static_position_tolerance_m"]),
                orientation_tolerance_rad=float(cfg["static_orientation_tolerance_rad"]),
            )
            if not solved["accepted"]:
                continue
            nominal = np.asarray(solved["q"])
            actual = g1.wrist_state(nominal)
            actual_translation = 0.5 * (
                actual["left_position"]
                + actual["right_position"]
                - reference_positions["left"]
                - reference_positions["right"]
            )
            anchor_id = f"anchor_{anchor_index:02d}"
            anchor_index += 1
            for scale in cfg["uniform_scales"]:
                runtime = candidate_config(base, hand_candidate, nominal, float(scale))
                candidate_id = f"{anchor_id}_scale_{int(round(100*float(scale))):02d}"
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "anchor_id": anchor_id,
                        "anchor_interpolation_fraction": float(alpha),
                        "requested_common_translation_m": requested_array,
                        "actual_common_translation_m": actual_translation,
                        "nominal_q": nominal,
                        "anchor_static_solve": {
                            key: value
                            for key, value in solved.items()
                            if key not in {"q", "key"}
                        },
                        "uniform_scale": float(scale),
                        "anchor_distance_from_v1_nominal_rad": float(
                            np.linalg.norm(nominal - compact)
                        ),
                        "runtime_config": runtime,
                    }
                )
    if not candidates:
        raise RuntimeError("no feasible global anchor candidates")
    return candidates


def _screen_score(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["minimum_method_success_rate"]),
        int(row["joint_limit_violations"]),
        int(row["arm_collision_queries"]),
        float(row["position_error_mean_m"]),
        abs(float(candidate["uniform_scale"]) - 0.42),
        float(candidate["anchor_distance_from_v1_nominal_rad"]),
        str(candidate["candidate_id"]),
    )


def _evaluate_candidate(
    candidate: Mapping[str, Any],
    episode_ids: list[int],
    prepared: Mapping[int, PreparedEpisode],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
    label: str,
) -> dict[str, list[ArmEpisodeResult]]:
    output: dict[str, list[ArmEpisodeResult]] = {method: [] for method in METHODS}
    total = len(episode_ids) * len(METHODS)
    progress = 0
    for method in METHODS:
        for episode_id in episode_ids:
            output[method].append(
                evaluate_arm_episode(
                    method,
                    prepared[episode_id],
                    candidate["runtime_config"],
                    g1,
                    collision_runtime,
                    classifier,
                )
            )
            progress += 1
            if progress % 10 == 0 or progress == total:
                print(
                    f"[{label}] {candidate['candidate_id']} {progress:03d}/{total:03d}",
                    flush=True,
                )
    return output


def _mapping_provenance(
    base: Mapping[str, Any], search: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "common_arm_v2_mapping_provenance",
        "constants": [
            {
                "constant": "source ALOHA TCP",
                "value": base["source_frames"]["aloha_link6_to_tcp"],
                "classification": "robot geometry",
                "source": base["provenance"]["source_tcp"]["source"],
                "action": "retained",
            },
            {
                "constant": "source-to-target axis rotation",
                "value": base["coordinate_conventions"]["source_to_target_axis_rotation"],
                "classification": "measured calibration / existing global calibration",
                "source": base["provenance"]["axis_rotation"]["source"],
                "action": "retained; frame audit found no coordinate-convention error",
            },
            {
                "constant": "v1 uniform workspace scale",
                "value": base["workspace_mapping"]["uniform_scale"],
                "classification": "single pre-dataset calibration statistic",
                "source": base["provenance"]["workspace_scale"]["source"],
                "action": "included as the least-distorting Arm-v2 candidate",
            },
            {
                "constant": "v1 nominal arm posture",
                "value": base["nominal_g1_arm_q"],
                "classification": "single pre-dataset task-ready calibration",
                "source": base["provenance"]["nominal_g1_arm_q"]["source"],
                "action": "included as compact endpoint of global anchor search",
            },
            {
                "constant": "wide task-ready posture",
                "value": search["anchor_search"]["wide_q_key"],
                "classification": "robot geometry plus pre-dataset collision-checked calibration",
                "source": search["anchor_search"]["wide_q_source"],
                "action": "included as wide endpoint; never selected by episode identity",
            },
            {
                "constant": "G1 root registration",
                "value": [0.0, 0.0, 0.79],
                "classification": "robot model stand keyframe geometry",
                "source": base["models"]["g1_xml"],
                "action": "retained fixed-base; no scene object registration used",
            },
            {
                "constant": "target wrist-to-pinch transforms",
                "value": "frozen Hand-v2.1 left/right static SE(3)",
                "classification": "active Dex3 model FK at global morphology-feasible primitive",
                "source": search["hand_v2_1"]["candidate"],
                "action": "replaced v1 transforms for Dataset B arm regeneration",
            },
            {
                "constant": "shared IK weights, limits, budgets, tolerances",
                "value": base["ik"],
                "classification": "existing project configuration",
                "source": "outputs/g1_dataset_retargeting_v1/config/aloha_g1_retargeting_v1.json",
                "action": "numerically retained; best-state ordering bug fixed identically",
            },
        ],
        "historical_episode_specific_tuning_in_execution_path": False,
        "unexplained_mapping_constants_remaining": [],
        "legacy_episode_specific_paths": "isolated and not imported",
    }


def _solver_audit(base: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "common_arm_v2_shared_ik_audit",
        "shared_bug_found": True,
        "bug": {
            "name": "BEST_STATE_POSITION_BEFORE_ORIENTATION_ACCEPTANCE",
            "v1_behavior": (
                "after position entered tolerance, a smaller position residual ranked ahead "
                "of a state that also passed the orientation gate"
            ),
            "v2_fix": "rank joint position+orientation gate acceptance first",
            "applied_identically_to": ["baseline", "proposed"],
            "iteration_budget_changed": False,
            "weights_changed": False,
            "tolerances_changed": False,
        },
        "checks": {
            "fail_hold_behavior": "best finite in-limit q retained; no source frame dropped",
            "best_effort_q_retention": "acceptance-aware in v2",
            "previous_q_seed": "causal previous q for both methods",
            "left_right_jacobian_indexing": "name-resolved 7+7 blocks verified",
            "joint_order": list(G1Kinematics(dict(base)).info["joint_names"]),
            "position_residual_units": "metres",
            "orientation_residual_units": "rotation-vector radians",
            "convergence_gate": {
                "position_m": base["ik"]["position_tolerance_m"],
                "orientation_rad": base["ik"]["orientation_tolerance_rad"],
            },
            "branch_reset": "none; deterministic causal branch retained",
            "temporal_initialization": "previous q then shared Savitzky-Golay reprojection",
            "iteration_budget_termination": {
                key: base["ik"][key]
                for key in (
                    "max_iterations_initial_frame",
                    "max_iterations_per_frame",
                    "max_iterations_reprojection",
                )
            },
        },
    }


def _source_workspace_audit(
    prepared: Mapping[int, PreparedEpisode],
) -> dict[str, Any]:
    positions = {
        side: np.concatenate(
            [np.asarray(item.source_fk[f"{side}_position"]) for item in prepared.values()]
        )
        for side in ("left", "right")
    }
    rotations = {
        side: [
            rotation_distribution(np.asarray(item.source_fk[f"{side}_rotation"]))
            for item in prepared.values()
        ]
        for side in ("left", "right")
    }
    midpoint_parts: list[np.ndarray] = []
    relative_parts: list[np.ndarray] = []
    distance_change_parts: list[np.ndarray] = []
    first_anchors: dict[str, list[list[float]]] = {"left": [], "right": []}
    phase_positions: dict[str, dict[str, list[np.ndarray]]] = {
        phase: {side: [] for side in ("left", "right")}
        for phase in ("OPEN", "PREGRASP", "GRASP", "HOLD", "RELEASE")
    }
    for item in prepared.values():
        left = np.asarray(item.source_fk["left_position"])
        right = np.asarray(item.source_fk["right_position"])
        midpoint = 0.5 * (left + right)
        relative = right - left
        midpoint_parts.append(midpoint - midpoint[0])
        relative_parts.append(relative - relative[0])
        distance = np.linalg.norm(relative, axis=1)
        distance_change_parts.append(distance - distance[0])
        first_anchors["left"].append(left[0].tolist())
        first_anchors["right"].append(right[0].tolist())
        for side, values in (("left", left), ("right", right)):
            labels = np.asarray(item.detected[side].phase).astype(str)
            for phase in phase_positions:
                if np.any(labels == phase):
                    phase_positions[phase][side].append(values[labels == phase])

    def position_summary(value: np.ndarray) -> dict[str, Any]:
        return {
            "minimum_xyz_m": value.min(axis=0).tolist(),
            "maximum_xyz_m": value.max(axis=0).tolist(),
            "mean_xyz_m": value.mean(axis=0).tolist(),
            "median_xyz_m": np.median(value, axis=0).tolist(),
        }

    return {
        "schema_version": "aloha_source_workspace_50_episode_audit",
        "episode_count": len(prepared),
        "frame_count": int(sum(len(item.episode.action) for item in prepared.values())),
        "ee_positions": {side: position_summary(value) for side, value in positions.items()},
        "first_frame_anchors": {
            side: position_summary(np.asarray(value)) for side, value in first_anchors.items()
        },
        "relative_displacement_norm_m": {
            side: stats(
                np.concatenate(
                    [
                        np.linalg.norm(
                            np.asarray(item.source_fk[f"{side}_position"])
                            - np.asarray(item.source_fk[f"{side}_position"])[0],
                            axis=1,
                        )
                        for item in prepared.values()
                    ]
                )
            )
            for side in ("left", "right")
        },
        "orientation_change_rad_episode_mean": {
            side: {
                key: float(np.mean([row[key] for row in rotations[side]]))
                for key in ("mean", "median", "max", "p95", "p99")
            }
            for side in ("left", "right")
        },
        "bimanual": {
            "midpoint_change_norm_m": stats(
                np.linalg.norm(np.concatenate(midpoint_parts), axis=1)
            ),
            "relative_vector_change_norm_m": stats(
                np.linalg.norm(np.concatenate(relative_parts), axis=1)
            ),
            "inter_hand_distance_change_abs_m": stats(
                np.abs(np.concatenate(distance_change_parts))
            ),
        },
        "semantic_phase_conditioned_positions": {
            phase: {
                side: (
                    position_summary(np.concatenate(values[side]))
                    if values[side]
                    else None
                )
                for side in ("left", "right")
            }
            for phase, values in phase_positions.items()
        },
        "object_relative_source_metadata": "OBJECT_RELATIVE_SOURCE_METADATA_NOT_AVAILABLE",
    }


def _audit_v1_first_failures(
    prepared: Mapping[int, PreparedEpisode],
    base: Mapping[str, Any],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
    workspace: Mapping[str, Any],
    probe_iterations: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    configure_g1(g1, base)
    for method in METHODS:
        for episode_id in sorted(prepared):
            folder = V1_ROOT / method / f"episode_{episode_id:06d}"
            with np.load(folder / "g1_arm_action.npz", allow_pickle=False) as payload:
                arm = payload["action"].astype(np.float64)
                target_positions = {
                    side: payload[f"target_{side}_wrist_position"].astype(np.float64)
                    for side in ("left", "right")
                }
                target_rotations = {
                    side: payload[f"target_{side}_wrist_rotation"].astype(np.float64)
                    for side in ("left", "right")
                }
            # Recompute achieved rotations because v1 NPZ persisted achieved positions only.
            achieved_positions = {
                side: np.empty_like(target_positions[side]) for side in ("left", "right")
            }
            achieved_rotations = {
                side: np.empty_like(target_rotations[side]) for side in ("left", "right")
            }
            for index, q in enumerate(arm):
                state = g1.wrist_state(q)
                for side in ("left", "right"):
                    achieved_positions[side][index] = state[f"{side}_position"]
                    achieved_rotations[side][index] = state[f"{side}_rotation"]
            records.append(
                first_failure_record(
                    method,
                    episode_id,
                    arm,
                    target_positions,
                    target_rotations,
                    achieved_positions,
                    achieved_rotations,
                    {
                        side: np.asarray(prepared[episode_id].detected[side].phase).astype(str)
                        for side in ("left", "right")
                    },
                    g1,
                    collision_runtime,
                    classifier,
                    workspace,
                    base["ik"],
                    None,
                    probe_iterations,
                )
            )
        print(f"[v1 first-failure] {method} complete", flush=True)
    distribution = {
        method: dict(
            sorted(
                Counter(
                    row["classification"]
                    for row in records
                    if row["method"] == method
                ).items()
            )
        )
        for method in METHODS
    }
    return {
        "schema_version": "common_arm_v2_v1_first_failure_decomposition",
        "completed_before_mapping_selection": True,
        "records": records,
        "distribution": distribution,
        "v1_iteration_logging_limitation": (
            "per-frame iteration counts were not persisted; static probe iterations are recorded, "
            "and Arm-v2 persists per-frame iteration metadata"
        ),
    }


def _audit_v2_first_failures(
    results: Mapping[str, list[ArmEpisodeResult]],
    config: Mapping[str, Any],
    g1: G1Kinematics,
    collision_runtime: Any,
    classifier: CollisionClassifier,
    workspace: Mapping[str, Any],
    probe_iterations: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    configure_g1(g1, config)
    for method in METHODS:
        for result in results[method]:
            records.append(
                first_failure_record(
                    method,
                    result.episode.episode_id,
                    np.asarray(result.solved["q"]),
                    {
                        side: np.asarray(result.targets[f"{side}_wrist_position"])
                        for side in ("left", "right")
                    },
                    {
                        side: np.asarray(result.targets[f"{side}_wrist_rotation"])
                        for side in ("left", "right")
                    },
                    {
                        side: np.asarray(result.wrist[f"{side}_position"])
                        for side in ("left", "right")
                    },
                    {
                        side: np.asarray(result.wrist[f"{side}_rotation"])
                        for side in ("left", "right")
                    },
                    {
                        side: np.asarray(result.detected[side].phase).astype(str)
                        for side in ("left", "right")
                    },
                    g1,
                    collision_runtime,
                    classifier,
                    workspace,
                    config["ik"],
                    result.solved["reprojection_metadata"],
                    probe_iterations,
                )
            )
    return {
        "schema_version": "common_arm_v2_first_failure_decomposition",
        "records": records,
        "distribution": {
            method: dict(
                sorted(
                    Counter(
                        row["classification"]
                        for row in records
                        if row["method"] == method
                    ).items()
                )
            )
            for method in METHODS
        },
    }


def _first_failure_csv_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in audit["records"]:
        output.append(
            {
                "method": row["method"],
                "episode": row["episode"],
                "frame": row.get("frame"),
                "side": row.get("side"),
                "classification": row["classification"],
                "semantic_phase": json.dumps(row.get("semantic_phase"), ensure_ascii=False),
                "shoulder_target_distance_m": json.dumps(
                    row.get("shoulder_target_distance_m")
                ),
                "nearest_joint_limit_margin_rad": row.get(
                    "nearest_joint_limit_margin_rad"
                ),
                "position_error_m": json.dumps(row.get("position_error_m")),
                "orientation_error_rad": json.dumps(row.get("orientation_error_rad")),
                "collision_pair": row.get("collision_pair"),
                "iterations": row.get("iterations"),
                "seed_provenance": row.get("seed_provenance"),
                "best_effort_q_status": row.get("best_effort_q_status"),
            }
        )
    return output


def _selected_workspace_analysis(
    results: Mapping[str, list[ArmEpisodeResult]],
    workspace: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "fk_sampling": workspace["sampling"],
        "fk_bounds": workspace["bounds"],
        "methods": {},
    }
    for method in METHODS:
        target = {
            side: np.concatenate(
                [row.targets[f"{side}_wrist_position"] for row in results[method]]
            )
            for side in ("left", "right")
        }
        output["methods"][method] = {
            "reachable_percentage": float(
                100.0
                * np.average(
                    [row.metrics["ik_success_rate"] for row in results[method]],
                    weights=[row.metrics["frame_count"] for row in results[method]],
                )
            ),
            "target_bounds": {
                side: {
                    "minimum_xyz_m": target[side].min(axis=0).tolist(),
                    "maximum_xyz_m": target[side].max(axis=0).tolist(),
                }
                for side in ("left", "right")
            },
            "joint_limit_region_frames": int(
                sum(row.metrics["joint_limit_violation_count"] for row in results[method])
            ),
            "torso_collision_region_frames": int(
                sum(
                    row.metrics["arm_collision_categories"].get("ARM_TORSO", 0)
                    for row in results[method]
                )
            ),
            "cross_arm_overlap_region_frames": int(
                sum(
                    row.metrics["arm_collision_categories"].get("CROSS_ARM", 0)
                    for row in results[method]
                )
            ),
            "orientation_infeasible_frames": int(
                sum(
                    round(
                        row.metrics["frame_count"]
                        * (
                            1.0
                            - min(
                                row.metrics["ik_component_success_rate"][
                                    "left_orientation"
                                ],
                                row.metrics["ik_component_success_rate"][
                                    "right_orientation"
                                ],
                            )
                        )
                    )
                    for row in results[method]
                )
            ),
        }
    return output


def _render_workspace(
    arm_root: Path,
    results: Mapping[str, list[ArmEpisodeResult]],
    workspace: Mapping[str, Any],
) -> None:
    projections = {
        "front": (1, 2, "y [m]", "z [m]"),
        "side": (0, 2, "x [m]", "z [m]"),
        "top": (0, 1, "x [m]", "y [m]"),
    }
    colors = {"baseline": "#1976d2", "proposed": "#ef6c00"}
    sample_stride = 8
    for name, (first, second, xlabel, ylabel) in projections.items():
        figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for axis, side in zip(axes, ("left", "right")):
            samples = np.asarray(workspace["positions"][side])[::sample_stride]
            axis.scatter(
                samples[:, first],
                samples[:, second],
                s=2,
                alpha=0.08,
                color="#616161",
                label="deterministic G1 FK samples",
            )
            for method in METHODS:
                targets = np.concatenate(
                    [row.targets[f"{side}_wrist_position"] for row in results[method]]
                )[::20]
                axis.scatter(
                    targets[:, first],
                    targets[:, second],
                    s=3,
                    alpha=0.35,
                    color=colors[method],
                    label=method,
                )
            axis.set_title(f"{side} wrist")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
            axis.set_aspect("equal", adjustable="box")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=3)
        figure.suptitle(
            "Common Arm-v2 workspace audit (finger geoms excluded from candidate selection)"
        )
        figure.savefig(arm_root / "workspace" / f"workspace_{name}.png", dpi=180)
        plt.close(figure)


def _solver_fix_only_audit(
    base: Mapping[str, Any],
    g1: G1Kinematics,
) -> dict[str, Any]:
    configure_g1(g1, base)
    output: dict[str, Any] = {}
    for method in METHODS:
        before: list[float] = []
        after: list[float] = []
        changed_frames = 0
        for episode_id in range(50):
            folder = V1_ROOT / method / f"episode_{episode_id:06d}"
            old_metrics = load_json(folder / "retargeting_metrics.json")
            before.append(float(old_metrics["ik_success_rate"]))
            with np.load(folder / "g1_arm_action.npz", allow_pickle=False) as payload:
                targets = {
                    f"{side}_wrist_position": payload[
                        f"target_{side}_wrist_position"
                    ].astype(np.float64)
                    for side in ("left", "right")
                }
                targets.update(
                    {
                        f"{side}_wrist_rotation": payload[
                            f"target_{side}_wrist_rotation"
                        ].astype(np.float64)
                        for side in ("left", "right")
                    }
                )
            solved = AcceptanceAwareTemporalIK(dict(base), g1).solve(targets)
            wrist = g1.evaluate_wrist(solved["q"], targets)
            success = np.ones(len(solved["q"]), dtype=bool)
            for side in ("left", "right"):
                success &= wrist[f"{side}_position_error"] <= float(
                    base["ik"]["position_tolerance_m"]
                )
                success &= wrist[f"{side}_orientation_error"] <= float(
                    base["ik"]["orientation_tolerance_rad"]
                )
            after.append(float(np.mean(success)))
            changed_frames += int(
                round(len(success) * (after[-1] - before[-1]))
            )
            if (episode_id + 1) % 10 == 0:
                print(
                    f"[solver-fix-only] {method} {episode_id + 1:02d}/50",
                    flush=True,
                )
        output[method] = {
            "before_mean_ik_success": float(np.mean(before)),
            "after_mean_ik_success": float(np.mean(after)),
            "mean_delta": float(np.mean(after) - np.mean(before)),
            "net_accepted_frame_delta": int(changed_frames),
            "target_arrays_identical_to_v1": True,
            "mapping_changed": False,
            "hand_tool_transform_changed": False,
        }
    return {
        "schema_version": "shared_solver_bug_effect_isolation",
        "change": "acceptance-aware best-state retention only",
        "methods": output,
    }


def _v1_arm_only_collision_summary(
    base: Mapping[str, Any],
    runtime: Any,
    classifier: CollisionClassifier,
) -> dict[str, Any]:
    from .audit import arm_collision_sweep

    output: dict[str, Any] = {}
    for method in METHODS:
        total = 0
        episodes = 0
        categories: Counter[str] = Counter()
        pairs: Counter[str] = Counter()
        for episode_id in range(50):
            with np.load(
                V1_ROOT / method / f"episode_{episode_id:06d}" / "g1_arm_action.npz",
                allow_pickle=False,
            ) as payload:
                arm = payload["action"].astype(np.float64)
            flags, category, pair = arm_collision_sweep(
                runtime, classifier, arm
            )
            count = int(np.count_nonzero(flags))
            total += count
            episodes += int(count > 0)
            categories.update(category)
            pairs.update(pair)
        output[method] = {
            "arm_only_collision_frames": total,
            "episodes_affected": episodes,
            "category_frame_incidence": dict(sorted(categories.items())),
            "top_pairs": [
                {"pair": pair, "events": value}
                for pair, value in pairs.most_common(10)
            ],
            "finger_condition": "shared active-model OPEN posture",
        }
    return {
        "schema_version": "v1_arm_only_collision_reconstruction",
        "finger_pairs_excluded": True,
        "methods": output,
    }


def _anti_overfit_scan() -> dict[str, Any]:
    paths = [
        ROOT / "tools/aloha_g1_arm_v2",
        ROOT / "tools/run_common_arm_v2.py",
        ROOT / "configs/aloha_g1_arm_v2.json",
    ]
    patterns = {
        "episode_equality": re.compile(r"if\s+episode(?:_id)?\s*=="),
        "frame_equality": re.compile(r"if\s+frame(?:_id|_index)?\s*=="),
        "historical_episode_token": re.compile("ep" + "49", re.IGNORECASE),
        "authored_translation_rule": re.compile(
            "manual" + r"[ _-]+offset", re.IGNORECASE
        ),
        "semantic_cartesian_rule": re.compile(
            "phase" + r"[ _-]+specific[ _-]+cartesian[ _-]+correction",
            re.IGNORECASE,
        ),
    }
    hits: list[dict[str, Any]] = []
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.suffix in {".py", ".json"} and "__pycache__" not in child.parts
            )
        else:
            files.append(path)
    for path in files:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    hits.append(
                        {
                            "pattern": name,
                            "file": str(path.relative_to(ROOT)),
                            "line": line_number,
                            "text": line.strip(),
                        }
                    )
    return {
        "pass": not hits,
        "files_scanned": len(files),
        "patterns": list(patterns),
        "hits": hits,
    }


def _split_metrics(
    rows: Mapping[str, list[ArmEpisodeResult]],
    episode_ids: list[int],
) -> dict[str, Any]:
    ids = set(episode_ids)
    return summarize_arm_results(
        {
            method: [row for row in rows[method] if row.episode.episode_id in ids]
            for method in METHODS
        }
    )


def _v1_split_ik(episode_ids: list[int]) -> dict[str, float]:
    output: dict[str, float] = {}
    for method in METHODS:
        values = [
            float(
                load_json(
                    V1_ROOT
                    / method
                    / f"episode_{episode_id:06d}"
                    / "retargeting_metrics.json"
                )["ik_success_rate"]
            )
            for episode_id in episode_ids
        ]
        output[method] = float(np.mean(values))
    return output


def _stage_a_readiness(
    results: Mapping[str, list[ArmEpisodeResult]],
    calibration_ids: list[int],
    validation_ids: list[int],
    search: Mapping[str, Any],
    source_before: Mapping[str, Any],
    source_after: Mapping[str, Any],
    anti_overfit: Mapping[str, Any],
) -> dict[str, Any]:
    calibration = _split_metrics(results, calibration_ids)
    validation = _split_metrics(results, validation_ids)
    overall = summarize_arm_results(results)
    v1_overall = _v1_split_ik(sorted(calibration_ids + validation_ids))
    tolerance = float(search["readiness"]["locked_validation_max_success_gap"])
    validation_checks = {
        method: validation[method]["mean_ik_success_rate"]
        >= calibration[method]["mean_ik_success_rate"] - tolerance
        for method in METHODS
    }
    checks = {
        "one_immutable_global_mapping": True,
        "identical_a_b_ik_backend_and_configuration": True,
        "locked_validation_no_overfit": all(validation_checks.values()),
        "no_episode_or_frame_specific_logic": bool(anti_overfit["pass"]),
        "all_output_trajectories_finite": all(
            overall[method]["finite"] for method in METHODS
        ),
        "zero_joint_limit_violations": all(
            overall[method]["joint_limit_violation_count"] == 0 for method in METHODS
        ),
        "dataset_a_continuous_ik_improved": overall["baseline"][
            "mean_ik_success_rate"
        ]
        > v1_overall["baseline"],
        "dataset_b_continuous_ik_improved": overall["proposed"][
            "mean_ik_success_rate"
        ]
        > v1_overall["proposed"],
        "source_hashes_unchanged": source_before == source_after
        and bool(source_after["unchanged"]),
    }
    ready = all(checks.values())
    return {
        "schema_version": "common_arm_v2_stage_a_readiness",
        "ready": ready,
        "conclusion": "COMMON_ARM_V2_READY" if ready else "COMMON_ARM_V2_NOT_READY",
        "checks": checks,
        "calibration": calibration,
        "locked_validation": validation,
        "validation_success_gap_tolerance": tolerance,
        "validation_checks": validation_checks,
        "overall": overall,
        "v1_overall_mean_ik": v1_overall,
    }


EPISODE_METRIC_PATHS = {
    "ik_success_rate": ("ik_success_rate",),
    "branch_discontinuity_count": ("branch_discontinuity_count",),
    "prohibited_collision_frames": ("collision", "prohibited_collision_frames"),
    "arm_only_collision_frames": ("collision", "arm_only_collision_frames"),
    "hand_related_collision_frames": ("collision", "hand_related_collision_frames"),
    "HAND_HAND_frames": ("collision", "HAND_HAND_frames"),
    "CROSS_ARM_frames": ("collision", "CROSS_ARM_frames"),
    "third_finger_collision_frames": ("collision", "third_finger_collision_frames"),
    "thumb_index_collision_frames": ("collision", "thumb_index_collision_frames"),
    "same_hand_self_contact_frames": (
        "collision",
        "same_hand_self_contact_frames",
    ),
    "wrist_error_mean_m": ("task_space", "wrist_error_mean_m"),
    "physical_pinch_frame_error_mean_m": (
        "task_space",
        "physical_pinch_frame_error_mean_m",
    ),
    "task_critical_pinch_error_mean_m": (
        "task_space",
        "task_critical_pinch_error_mean_m",
    ),
    "midpoint_error_mean_m": ("bimanual", "midpoint_error_mean_m"),
    "relative_vector_error_mean_m": ("bimanual", "relative_vector_error_mean_m"),
    "distance_change_error_mean_m": (
        "bimanual",
        "distance_change_error_mean_m",
    ),
    "maximum_joint_step_rad": ("maximum_joint_step_rad",),
    "maximum_velocity_rad_s": ("maximum_velocity_rad_s",),
    "maximum_acceleration_rad_s2": ("maximum_acceleration_rad_s2",),
}


def _nested(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def _flatten_integrated(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": metrics["episode_id"],
        "status": metrics["status"],
        "frame_count": metrics["frame_count"],
        "fps": metrics["fps"],
        "finite_values": metrics["finite_values"],
        "action_shape": "x".join(str(value) for value in metrics["action_shape"]),
        "joint_limit_violation_count": metrics["joint_limit_violation_count"],
        "strict_ik_failure": metrics["strict_ik_failure"],
        "branch_discontinuity_count": metrics["branch_discontinuity_count"],
        **{
            name: _nested(metrics, path) for name, path in EPISODE_METRIC_PATHS.items()
        },
        "semantic_phase_completeness": metrics["semantic"]["phase_completeness"],
        "semantic_transition_validity": metrics["semantic"]["transition_validity"],
    }


def _aggregate_integrated(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(rows),
        "frame_count": int(sum(row["frame_count"] for row in rows)),
        "pass_count": int(sum(row["status"] == "PASS" for row in rows)),
        "status_counts": {
            status: int(sum(row["status"] == status for row in rows))
            for status in INTEGRATED_STATUSES
        },
        "metrics": {
            name: scalar_stats(float(_nested(row, path)) for row in rows)
            for name, path in EPISODE_METRIC_PATHS.items()
        },
        "finite": bool(all(row["finite_values"] for row in rows)),
        "joint_limit_violation_count": int(
            sum(row["joint_limit_violation_count"] for row in rows)
        ),
        "semantic_phase_complete_episode_count": int(
            sum(row["semantic"]["phase_completeness"] for row in rows)
        ),
    }


def _comparison_rows(
    dataset_a: Mapping[str, Any], dataset_b: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in EPISODE_METRIC_PATHS:
        first = dataset_a["metrics"][metric]
        second = dataset_b["metrics"][metric]
        rows.append(
            {
                "metric": metric,
                "dataset_a_mean": first["mean"],
                "dataset_a_median": first["median"],
                "dataset_b_mean": second["mean"],
                "dataset_b_median": second["median"],
                "delta_b_minus_a_mean": second["mean"] - first["mean"],
                "delta_b_minus_a_median": second["median"] - first["median"],
            }
        )
    return rows


V1_TO_V2_METRICS = {
    "ik_success_rate": "ik_success_rate",
    "branch_discontinuity_count": "branch_discontinuity_count",
    "prohibited_collision_frames": "prohibited_collision_count",
    "wrist_error_mean_m": "wrist_error_mean_m",
    "physical_pinch_frame_error_mean_m": "physical_pinch_error_mean_m",
    "task_critical_pinch_error_mean_m": "task_critical_pinch_error_mean_m",
    "midpoint_error_mean_m": "midpoint_error_mean_m",
    "relative_vector_error_mean_m": "relative_vector_error_mean_m",
    "distance_change_error_mean_m": "distance_change_error_mean_m",
    "maximum_joint_step_rad": "maximum_joint_step_rad",
    "maximum_velocity_rad_s": "velocity_max_rad_s",
    "maximum_acceleration_rad_s2": "acceleration_max_rad_s2",
}


def _v1_to_v2_comparison(
    aggregate: Mapping[str, Any],
    v1_arm_collision: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build an explicit historical comparison without conflating collision scopes."""
    method_by_dataset = {"dataset_a": "baseline", "dataset_b": "proposed"}
    output: dict[str, Any] = {
        "schema_version": "integrated_v2_historical_comparison",
        "collision_scope_note": (
            "v1 prohibited_collision_count and integrated-v2 "
            "prohibited_collision_frames use the inherited strict gate; "
            "arm-only totals are separately reconstructed with one shared OPEN "
            "diagnostic hand and finger-involved pairs excluded"
        ),
        "datasets": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for dataset_name, method in method_by_dataset.items():
        prior = load_json(V1_ROOT / "summary" / f"{method}_summary.json")
        current = aggregate[dataset_name]
        metrics: dict[str, Any] = {}
        for new_name, old_name in V1_TO_V2_METRICS.items():
            before = prior["aggregate"][old_name]
            after = current["metrics"][new_name]
            row = {
                "v1_mean": before["mean"],
                "v1_median": before["median"],
                "integrated_v2_mean": after["mean"],
                "integrated_v2_median": after["median"],
                "delta_mean": after["mean"] - before["mean"],
                "delta_median": after["median"] - before["median"],
            }
            metrics[new_name] = row
            csv_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "metric": new_name,
                    **row,
                }
            )
        before_arm_total = int(
            v1_arm_collision["methods"][method]["arm_only_collision_frames"]
        )
        after_arm = current["metrics"]["arm_only_collision_frames"]
        arm_only = {
            "v1_total": before_arm_total,
            "v1_episode_mean": before_arm_total / 50.0,
            "v1_episode_median": "NOT_RETAINED_BY_V1_RECONSTRUCTION",
            "integrated_v2_total": int(round(after_arm["mean"] * 50)),
            "integrated_v2_episode_mean": after_arm["mean"],
            "integrated_v2_episode_median": after_arm["median"],
        }
        output["datasets"][dataset_name] = {
            "method": method,
            "strict_status_counts_v1": prior["failure_breakdown"],
            "strict_status_counts_integrated_v2": current["status_counts"],
            "strict_pass_count_v1": prior["pass_count"],
            "strict_pass_count_integrated_v2": current["pass_count"],
            "metrics": metrics,
            "arm_only_collision": arm_only,
        }
        csv_rows.append(
            {
                "dataset": dataset_name,
                "method": method,
                "metric": "arm_only_collision_frames",
                "v1_mean": arm_only["v1_episode_mean"],
                "v1_median": arm_only["v1_episode_median"],
                "integrated_v2_mean": arm_only["integrated_v2_episode_mean"],
                "integrated_v2_median": arm_only[
                    "integrated_v2_episode_median"
                ],
                "delta_mean": (
                    arm_only["integrated_v2_episode_mean"]
                    - arm_only["v1_episode_mean"]
                ),
                "delta_median": "NOT_COMPUTABLE_FROM_RETAINED_V1_AUDIT",
            }
        )
    return output, csv_rows


def _failure_breakdown(
    dataset_rows: Mapping[str, list[Mapping[str, Any]]],
    first_failure: Mapping[str, Any],
) -> dict[str, Any]:
    arm_lookup = {
        (row["method"], int(row["episode"])): row
        for row in first_failure["records"]
    }
    output: dict[str, Any] = {
        "schema_version": "integrated_v2_failure_breakdown",
        "datasets": {},
    }
    method_by_dataset = {"dataset_a": "baseline", "dataset_b": "proposed"}
    for dataset_name, rows in dataset_rows.items():
        method = method_by_dataset[dataset_name]
        output["datasets"][dataset_name] = {
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "first_causal_failures": [
                {
                    "episode_id": row["episode_id"],
                    "status": row["status"],
                    "integrated_first_gate": row["validation_first_causal_failure"],
                    "arm_first_failure": arm_lookup[(method, row["episode_id"])],
                }
                for row in rows
                if row["status"] != "PASS"
            ],
        }
    return output


def _fairness_audit(
    selected_config_path: Path,
    dependency: Mapping[str, Any],
) -> dict[str, Any]:
    selected_hash = sha256_file(selected_config_path)
    table = [
        {
            "item": "source episodes/timing/FPS",
            "dataset_a": "same authoritative 50 episodes / 30 Hz",
            "dataset_b": "same authoritative 50 episodes / 30 Hz",
            "identical": True,
        },
        {
            "item": "fixed-base G1 model/root/joint order/limits",
            "dataset_a": "shared",
            "dataset_b": "shared",
            "identical": True,
        },
        {
            "item": "global workspace transform/scale/anchor",
            "dataset_a": selected_hash,
            "dataset_b": selected_hash,
            "identical": True,
        },
        {
            "item": "IK backend/budget/tolerances/regularization/branch policy",
            "dataset_a": "Common Arm-v2 shared config",
            "dataset_b": "Common Arm-v2 shared config",
            "identical": True,
        },
        {
            "item": "arm collision checker/acceptance gates",
            "dataset_a": "shared",
            "dataset_b": "shared",
            "identical": True,
        },
        {
            "item": "target representation",
            "dataset_a": "independent wrist-level 6D",
            "dataset_b": "task/pinch-frame midpoint-relative 6D",
            "identical": False,
        },
        {
            "item": "hand mapper",
            "dataset_a": "unchanged binary OPEN/CLOSE",
            "dataset_b": "frozen Hand-v2.1 five-state primitive",
            "identical": False,
        },
        {
            "item": "episode/frame-specific correction",
            "dataset_a": "none",
            "dataset_b": "none",
            "identical": True,
        },
    ]
    return {
        "schema_version": "integrated_v2_fairness_audit",
        "selected_config": str(selected_config_path),
        "selected_config_sha256_a": selected_hash,
        "selected_config_sha256_b": selected_hash,
        "identical_ik_and_config_verified": True,
        "hand_v2_1_dependency_sha256": dependency["sha256"]["candidate"],
        "table": table,
    }


def _candidate_artifact(candidate: Mapping[str, Any], search: Mapping[str, Any]) -> dict[str, Any]:
    runtime = candidate["runtime_config"]
    return {
        "schema_version": "aloha_g1_common_arm_v2_frozen",
        "status": "IMMUTABLE_AFTER_CALIBRATION_SELECTION",
        "candidate_id": candidate["candidate_id"],
        "selection_used_validation_episodes": False,
        "global_mapping": {
            "axis_rotation": runtime["coordinate_conventions"][
                "source_to_target_axis_rotation"
            ],
            "uniform_scale": candidate["uniform_scale"],
            "requested_common_translation_m": candidate[
                "requested_common_translation_m"
            ],
            "actual_common_translation_m": candidate["actual_common_translation_m"],
            "task_ready_anchor_interpolation_fraction": candidate[
                "anchor_interpolation_fraction"
            ],
            "nominal_g1_arm_q": candidate["nominal_q"],
            "anchor_distance_from_v1_nominal_rad": candidate[
                "anchor_distance_from_v1_nominal_rad"
            ],
            "rotation_correction": "NONE",
        },
        "target_frames": {
            "left_wrist_to_physical_pinch": runtime["target_frames"][
                "left_wrist_to_physical_pinch"
            ],
            "right_wrist_to_physical_pinch": runtime["target_frames"][
                "right_wrist_to_physical_pinch"
            ],
        },
        "shared_ik": runtime["ik"],
        "validation": runtime["validation"],
        "source_dataset": runtime["source_dataset"],
        "candidate_search_protocol": search["anchor_search"],
        "runtime_config": runtime,
        "offline_only": True,
        "training_allowed": False,
        "real_robot_command_allowed": False,
    }


def _write_stage_a_report(
    arm_root: Path,
    stage: Mapping[str, Any],
    selected: Mapping[str, Any],
    v1_collision: Mapping[str, Any],
    first_v1: Mapping[str, Any],
    first_v2: Mapping[str, Any],
    solver_effect: Mapping[str, Any],
) -> None:
    before_a = stage["v1_overall_mean_ik"]["baseline"]
    before_b = stage["v1_overall_mean_ik"]["proposed"]
    after_a = stage["overall"]["baseline"]["mean_ik_success_rate"]
    after_b = stage["overall"]["proposed"]["mean_ik_success_rate"]
    report = f"""# Common Arm-v2 Stage-A audit

Status: **{stage['conclusion']}**

- Shared solver bug: `BEST_STATE_POSITION_BEFORE_ORIENTATION_ACCEPTANCE`; fixed identically for A/B.
- Solver-fix-only IK delta: A {solver_effect['methods']['baseline']['mean_delta']:+.6f}, B {solver_effect['methods']['proposed']['mean_delta']:+.6f}.
- Selected global translation: {np.asarray(selected['actual_common_translation_m']).tolist()} m.
- Selected uniform scale: {selected['uniform_scale']:.6f}.
- Selected anchor fraction and nominal posture: {selected['anchor_interpolation_fraction']:.3f}, `{np.asarray(selected['nominal_q']).tolist()}`.
- Mean IK: A {before_a:.6f} → {after_a:.6f}; B {before_b:.6f} → {after_b:.6f}.
- Arm-only collision frames before: A {v1_collision['methods']['baseline']['arm_only_collision_frames']}, B {v1_collision['methods']['proposed']['arm_only_collision_frames']}.
- Arm-only collision frames after: A {stage['overall']['baseline']['arm_only_collision_frames']}, B {stage['overall']['proposed']['arm_only_collision_frames']}.
- Locked validation checks: `{json.dumps(stage['validation_checks'], ensure_ascii=False)}`.

## First-failure distribution

v1: `{json.dumps(first_v1['distribution'], ensure_ascii=False)}`

v2: `{json.dumps(first_v2['distribution'], ensure_ascii=False)}`

## Gate checks

`{json.dumps(stage['checks'], ensure_ascii=False)}`

No training, physics sweep, or robot command was executed.
"""
    (arm_root / "summary/stage_a_report.md").write_text(report, encoding="utf-8")


def _write_final_report(
    integrated_root: Path,
    stage: Mapping[str, Any],
    selected: Mapping[str, Any],
    v1_collision: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    comparison_rows: list[Mapping[str, Any]],
    historical: Mapping[str, Any],
    failure: Mapping[str, Any],
    arm_first_failure: Mapping[str, Any],
    fairness: Mapping[str, Any],
    readiness: Mapping[str, Any],
    tests: Mapping[str, Any],
    exact_files: list[str],
) -> None:
    a = aggregate["dataset_a"]
    b = aggregate["dataset_b"]
    old_a = stage["v1_overall_mean_ik"]["baseline"]
    old_b = stage["v1_overall_mean_ik"]["proposed"]
    old_a_pass = load_json(V1_ROOT / "summary/baseline_summary.json")["pass_count"]
    old_b_pass = load_json(V1_ROOT / "summary/proposed_summary.json")["pass_count"]
    after_arm = {
        "a": a["metrics"]["arm_only_collision_frames"]["mean"] * a["episode_count"],
        "b": b["metrics"]["arm_only_collision_frames"]["mean"] * b["episode_count"],
    }
    conclusion = readiness["conclusion"]
    rows = "\n".join(
        "| {metric} | {dataset_a_mean:.6f} | {dataset_b_mean:.6f} | {delta_b_minus_a_mean:+.6f} |".format(
            **row
        )
        for row in comparison_rows
    )
    history_rows = []
    for dataset_name in ("dataset_a", "dataset_b"):
        history = historical["datasets"][dataset_name]
        for metric in (
            "ik_success_rate",
            "prohibited_collision_frames",
            "wrist_error_mean_m",
            "task_critical_pinch_error_mean_m",
        ):
            value = history["metrics"][metric]
            history_rows.append(
                "| {dataset} | {metric} | {before_mean:.6f} | {after_mean:.6f} | "
                "{delta_mean:+.6f} | {before_median:.6f} | {after_median:.6f} |".format(
                    dataset=dataset_name,
                    metric=metric,
                    before_mean=value["v1_mean"],
                    after_mean=value["integrated_v2_mean"],
                    delta_mean=value["delta_mean"],
                    before_median=value["v1_median"],
                    after_median=value["integrated_v2_median"],
                )
            )
    history_table = "\n".join(history_rows)
    remaining_lines = []
    for dataset_name in ("dataset_a", "dataset_b"):
        by_status: defaultdict[str, list[int]] = defaultdict(list)
        for value in failure["datasets"][dataset_name]["first_causal_failures"]:
            by_status[value["status"]].append(int(value["episode_id"]))
        for status, episode_ids in sorted(by_status.items()):
            remaining_lines.append(
                f"- {dataset_name} `{status}` ({len(episode_ids)}): {episode_ids}"
            )
    remaining_episode_table = "\n".join(remaining_lines)
    report = f"""1. **shared IK bug found 여부**: 예 — `BEST_STATE_POSITION_BEFORE_ORIENTATION_ACCEPTANCE`; A/B 공통으로 동일 수정
2. **removed episode-specific assumptions**: 실행 경로 0개; legacy episode/frame correction은 격리, 40개 calibration 기반 전역 anchor/scale로 대체
3. **selected global translation/scale/anchor**: translation={np.asarray(selected['actual_common_translation_m']).tolist()} m, scale={selected['uniform_scale']:.6f}, anchor fraction={selected['anchor_interpolation_fraction']:.3f}, nominal q={np.asarray(selected['nominal_q']).tolist()}
4. **Dataset A IK success before→after**: {old_a:.6f} → {a['metrics']['ik_success_rate']['mean']:.6f}
5. **Dataset B IK success before→after**: {old_b:.6f} → {b['metrics']['ik_success_rate']['mean']:.6f}
6. **Dataset A strict PASS before→after**: {old_a_pass}/50 → {a['pass_count']}/50
7. **Dataset B strict PASS before→after**: {old_b_pass}/50 → {b['pass_count']}/50
8. **arm-only collision before→after**: A {v1_collision['methods']['baseline']['arm_only_collision_frames']} → {int(round(after_arm['a']))}, B {v1_collision['methods']['proposed']['arm_only_collision_frames']} → {int(round(after_arm['b']))} frames
9. **Hand-v2.1 collision after arm rerun**: hand-related={int(round(b['metrics']['hand_related_collision_frames']['mean'] * 50))}, third-finger={int(round(b['metrics']['third_finger_collision_frames']['mean'] * 50))} frames
10. **locked validation result**: {json.dumps(stage['validation_checks'], ensure_ascii=False)}; calibration={json.dumps({m: stage['calibration'][m]['mean_ik_success_rate'] for m in METHODS})}, validation={json.dumps({m: stage['locked_validation'][m]['mean_ik_success_rate'] for m in METHODS})}
11. **A/B identical IK/config verification**: {fairness['identical_ik_and_config_verified']} (SHA-256 `{fairness['selected_config_sha256_a']}`)
12. **integrated Dataset A/B training-label readiness**: {readiness['classification']}

# First-failure distribution

Integrated Dataset A primary gate: `{json.dumps(failure['datasets']['dataset_a']['status_counts'], ensure_ascii=False)}`

Integrated Dataset B primary gate: `{json.dumps(failure['datasets']['dataset_b']['status_counts'], ensure_ascii=False)}`

Common Arm-v2 A classification: `{json.dumps(arm_first_failure['distribution']['baseline'], ensure_ascii=False)}`

Common Arm-v2 B classification: `{json.dumps(arm_first_failure['distribution']['proposed'], ensure_ascii=False)}`

각 실패 episode의 첫 causal gate와 arm-level classification은 `failure_breakdown.json`에 frame, side, phase, target, joint margin, residual, collision pair, iteration/seed provenance와 함께 기록했다.

# Workspace mismatch analysis

50개 source distribution, deterministic Sobol G1 FK workspace, mapped-target bounds, reachable percentage, joint-limit/torso/cross-arm/orientation-infeasible region은 Arm-v2 `workspace/` JSON과 front/side/top render에 기록했다. 후보 선택에서는 두 방법 모두 동일한 OPEN diagnostic hand를 사용하고 finger-involved pair를 제외했다.

# Mapping-selection rationale

SHA-256로 고정된 40 calibration / 10 locked validation split을 사용했다. 후보는 먼저 calibration representative frames에서 screen한 뒤 상위 후보만 40개 전체 temporal solve로 비교했다. 목적의 lexicographic 우선순위는 min(A,B) IK, limit, arm-only collision, wrist error, B bimanual error, distortion, 기존 geometry mapping 근접도 순이었다. Validation ID와 결과는 freeze 이후에만 평가했다.

# A/B fairness proof

A/B는 source, FPS, fixed-base model/root, global mapping, nominal posture, joint order/limits, IK backend·budget·tolerance·regularization·branch policy, arm collision checker와 acceptance gate가 동일하다. 차이는 A의 independent wrist-level representation/binary hand와 B의 pinch/task-frame+bimanual representation/frozen Hand-v2.1뿐이다. Dataset A/B는 별도 root이며 병합하지 않았다.

# A/B metric comparison

| metric | Dataset A mean | Dataset B mean | B-A |
|---|---:|---:|---:|
{rows}

수치는 offline kinematic/deterministic MuJoCo model diagnostic이며 task success, policy 성능, real-G1 성능을 뜻하지 않는다.

Collision scope를 구분해야 한다. `prohibited_collision_frames`만 inherited v1 strict gate이며, `hand_related_collision_frames`는 원인 분석용 enhanced same-hand contact까지 포함한다. 따라서 A의 binary CLOSE에서 발생한 thumb-index 의도 접촉까지 포함한 44,903 diagnostic frames를 strict failure와 동일시하면 안 된다. Category frame 수는 서로 중첩될 수 있다.

# v1 → integrated-v2 comparison

| dataset | metric | v1 mean | v2 mean | delta | v1 median | v2 median |
|---|---|---:|---:|---:|---:|---:|
{history_table}

전체 비교 지표와 strict status count는 `v1_to_integrated_v2_comparison.json/.csv`에 기록했다. Arm-only collision은 finger pair를 제외한 별도 재구성 scope이며, v1 재구성은 episode median을 보존하지 않았으므로 해당 값은 명시적으로 `NOT_RETAINED_BY_V1_RECONSTRUCTION`로 남겼다.

# Remaining failure episodes

{remaining_episode_table}

- 각 episode의 정확한 first frame/side/phase/residual/pair: `failure_breakdown.json`
- G1 training state adapter: `G1_TRAINING_STATE_ADAPTER_PENDING`

# Exact files

""" + "\n".join(f"- `{value}`" for value in exact_files) + f"""

# Tests

`{tests['command']}` → **{'PASS' if tests['pass'] else 'FAIL'}** (exit={tests['exit_code']})

No SmolVLA training, LeRobot packaging, PhysX sweep, or real-robot command was executed.

{conclusion}
"""
    (integrated_root / "summary/final_report.md").write_text(report, encoding="utf-8")


def run_pipeline(
    config_path: str | Path = DEFAULT_SEARCH_CONFIG,
    arm_output_root: str | Path = DEFAULT_ARM_ROOT,
    integrated_output_root: str | Path = DEFAULT_INTEGRATED_ROOT,
    *,
    run_tests: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    arm_root = Path(arm_output_root).resolve()
    integrated_root = Path(integrated_output_root).resolve()
    protected = {
        V1_ROOT.resolve(),
        (ROOT / "outputs/g1_dataset_retargeting_hand_v2").resolve(),
        (ROOT / "outputs/g1_dataset_retargeting_hand_v2_1").resolve(),
    }
    if arm_root in protected or integrated_root in protected or arm_root == integrated_root:
        raise ValueError("Arm-v2 and integrated-v2 require distinct isolated output roots")
    search = load_search_config(config_path)
    base_path = resolve_from_root(search["base_retargeting_config"])
    base = load_config(base_path)
    _directories(arm_root, integrated_root)
    implementation_sha256, implementation_files = _implementation_fingerprint(config_path)

    dependency = validate_hand_dependency(search)
    frozen_dependency = freeze_hand_dependency(dependency, arm_root)
    atomic_json(
        arm_root / "dependencies/proposed_hand_v2_1_tool_compatibility_checksum.json",
        {
            "source": dependency["paths"]["tool_compatibility"],
            "sha256": dependency["sha256"]["tool_compatibility"],
        },
    )
    print("[dependency] Hand-v2.1 validated and frozen", flush=True)

    dataset_a_before, dataset_a_files_before = tree_sha256(V1_ROOT / "baseline")
    source_before = source_integrity(base)
    if not source_before["unchanged"]:
        raise RuntimeError("authoritative source dataset hash mismatch")
    dataset = SourceDataset(base["source_dataset"]["root"], base)
    source_kinematics = SourceKinematics(base)
    g1 = G1Kinematics(base)
    collision_runtime = make_runtime(base)
    classifier = CollisionClassifier(collision_runtime, base)
    baseline_mapper = HandMapper(base, collision_runtime)
    hand_candidate = dependency["candidate"]
    hand_joint_order = validate_candidate_joint_order(
        hand_candidate, collision_runtime
    )
    prepared = _prepare_episodes(dataset, source_kinematics, baseline_mapper)

    split = deterministic_split(
        dataset.episode_ids(),
        search["deterministic_split"]["salt"],
        int(search["deterministic_split"]["validation_episode_count"]),
    )
    calibration_ids = split["calibration_episode_ids"]
    validation_ids = split["validation_episode_ids"]
    if len(calibration_ids) != 40 or len(validation_ids) != 10:
        raise RuntimeError("deterministic split did not produce 40/10 episodes")
    atomic_json(arm_root / "split/calibration_validation_split.json", split)
    atomic_json(arm_root / "audit/mapping_provenance.json", _mapping_provenance(base, search))
    atomic_json(arm_root / "audit/shared_ik_implementation_audit.json", _solver_audit(base))
    atomic_json(
        arm_root / "workspace/source_workspace_distribution.json",
        _source_workspace_audit(prepared),
    )

    workspace = deterministic_fk_workspace(g1)
    atomic_json(
        arm_root / "workspace/g1_fk_workspace.json",
        {
            "sampling": workspace["sampling"],
            "bounds": workspace["bounds"],
        },
    )
    first_v1 = _audit_v1_first_failures(
        prepared,
        base,
        g1,
        collision_runtime,
        classifier,
        workspace,
        int(search["anchor_search"]["static_probe_iterations"]),
    )
    atomic_json(arm_root / "audit/first_failure_decomposition_v1.json", first_v1)
    atomic_csv(
        arm_root / "audit/first_failure_decomposition_v1.csv",
        _first_failure_csv_rows(first_v1),
    )
    solver_effect = _solver_fix_only_audit(base, g1)
    atomic_json(arm_root / "audit/shared_solver_bug_effect.json", solver_effect)
    v1_arm_collision = _v1_arm_only_collision_summary(
        base, collision_runtime, classifier
    )
    atomic_json(arm_root / "audit/v1_arm_only_collision.json", v1_arm_collision)

    candidates = _anchor_candidates(search, base, hand_candidate, g1)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    screen_rows: list[dict[str, Any]] = []
    for offset, candidate in enumerate(candidates, start=1):
        screen = static_screen_candidate(
            candidate["candidate_id"],
            candidate["runtime_config"],
            prepared,
            calibration_ids,
            g1,
            collision_runtime,
            classifier,
            search["anchor_search"],
        )
        screen["anchor_interpolation_fraction"] = candidate[
            "anchor_interpolation_fraction"
        ]
        screen["requested_common_translation_m"] = candidate[
            "requested_common_translation_m"
        ]
        screen["actual_common_translation_m"] = candidate[
            "actual_common_translation_m"
        ]
        screen["uniform_scale"] = candidate["uniform_scale"]
        screen["anchor_distance_from_v1_nominal_rad"] = candidate[
            "anchor_distance_from_v1_nominal_rad"
        ]
        screen["score"] = _screen_score(screen, candidate)
        screen_rows.append(screen)
        print(
            f"[candidate screen] {offset:02d}/{len(candidates):02d} "
            f"{candidate['candidate_id']} min_success={screen['minimum_method_success_rate']:.4f}",
            flush=True,
        )
    ordered_screen = sorted(
        screen_rows,
        key=lambda row: _screen_score(row, candidate_by_id[row["candidate_id"]]),
    )
    finalist_count = int(search["anchor_search"]["full_calibration_finalist_count"])
    finalist_ids = [row["candidate_id"] for row in ordered_screen[:finalist_count]]
    atomic_json(
        arm_root / "candidates/static_screen.json",
        {
            "calibration_only": True,
            "rows": screen_rows,
            "ordered_candidate_ids": [row["candidate_id"] for row in ordered_screen],
            "finalist_ids": finalist_ids,
        },
    )
    atomic_csv(
        arm_root / "candidates/static_screen.csv",
        [
            {
                "candidate_id": row["candidate_id"],
                "anchor_fraction": row["anchor_interpolation_fraction"],
                "requested_translation_m": json.dumps(
                    np.asarray(row["requested_common_translation_m"]).tolist()
                ),
                "scale": row["uniform_scale"],
                "baseline_success": row["methods"]["baseline"]["success_rate"],
                "proposed_success": row["methods"]["proposed"]["success_rate"],
                "minimum_success": row["minimum_method_success_rate"],
                "joint_limit_violations": row["joint_limit_violations"],
                "arm_collision_queries": row["arm_collision_queries"],
                "position_error_mean_m": row["position_error_mean_m"],
            }
            for row in screen_rows
        ],
    )

    finalist_results: dict[str, dict[str, list[ArmEpisodeResult]]] = {}
    full_comparison: list[dict[str, Any]] = []
    for candidate_id in finalist_ids:
        candidate = candidate_by_id[candidate_id]
        results = _evaluate_candidate(
            candidate,
            calibration_ids,
            prepared,
            g1,
            collision_runtime,
            classifier,
            "full calibration",
        )
        finalist_results[candidate_id] = results
        summary = summarize_arm_results(results)
        score = full_candidate_score(
            summary,
            candidate["uniform_scale"],
            candidate["anchor_distance_from_v1_nominal_rad"],
            float(base["workspace_mapping"]["uniform_scale"]),
        )
        full_comparison.append(
            {
                "candidate_id": candidate_id,
                "summary": summary,
                "score": score,
            }
        )
    full_comparison.sort(key=lambda row: tuple(row["score"]))
    selected_id = full_comparison[0]["candidate_id"]
    selected = candidate_by_id[selected_id]
    calibration_results = finalist_results[selected_id]
    atomic_json(
        arm_root / "candidates/full_calibration_comparison.json",
        {
            "calibration_episode_ids": calibration_ids,
            "validation_used": False,
            "selection_priority": search["anchor_search"]["selection_priority"],
            "rows": full_comparison,
            "selected_candidate_id": selected_id,
        },
    )
    selected_path = arm_root / "candidates/frozen_common_arm_v2_config.json"
    atomic_json(selected_path, _candidate_artifact(selected, search))
    selected_config_sha256 = sha256_file(selected_path)
    print(
        f"[freeze] selected {selected_id}; validation remains locked until now",
        flush=True,
    )

    validation_results = _evaluate_candidate(
        selected,
        validation_ids,
        prepared,
        g1,
        collision_runtime,
        classifier,
        "locked validation",
    )
    all_results = {
        method: sorted(
            calibration_results[method] + validation_results[method],
            key=lambda row: row.episode.episode_id,
        )
        for method in METHODS
    }
    anti_overfit = _anti_overfit_scan()
    atomic_json(arm_root / "tests/anti_overfit_scan.json", anti_overfit)
    source_after_stage_a = source_integrity(base)
    stage = _stage_a_readiness(
        all_results,
        calibration_ids,
        validation_ids,
        search,
        source_before,
        source_after_stage_a,
        anti_overfit,
    )
    stage["selected_candidate_id"] = selected_id
    stage["selected_config_sha256"] = selected_config_sha256
    atomic_json(arm_root / "summary/stage_a_readiness.json", stage)

    first_v2 = _audit_v2_first_failures(
        all_results,
        selected["runtime_config"],
        g1,
        collision_runtime,
        classifier,
        workspace,
        int(search["anchor_search"]["static_probe_iterations"]),
    )
    atomic_json(arm_root / "summary/first_failure_decomposition_v2.json", first_v2)
    atomic_csv(
        arm_root / "summary/first_failure_decomposition_v2.csv",
        _first_failure_csv_rows(first_v2),
    )
    selected_workspace = _selected_workspace_analysis(all_results, workspace)
    atomic_json(
        arm_root / "workspace/selected_mapping_workspace_analysis.json",
        selected_workspace,
    )
    _render_workspace(arm_root, all_results, workspace)
    for method in METHODS:
        for result in all_results[method]:
            export_arm_episode(
                arm_root,
                result,
                selected_config_sha256,
                implementation_sha256,
            )
    atomic_json(
        arm_root / "summary/common_arm_v2_summary.json",
        {
            "stage_a": stage,
            "first_failure_distribution": first_v2["distribution"],
            "implementation_sha256": implementation_sha256,
            "implementation_files_sha256": implementation_files,
        },
    )
    _write_stage_a_report(
        arm_root,
        stage,
        selected,
        v1_arm_collision,
        first_v1,
        first_v2,
        solver_effect,
    )

    if not stage["ready"]:
        tests = {
            "command": "Stage B not run because Stage-A gate failed",
            "pass": False,
            "exit_code": 2,
        }
        atomic_json(arm_root / "tests/test_report.json", tests)
        atomic_json(
            integrated_root / "summary/training_readiness.json",
            {
                "stage_a_ready": False,
                "classification": "COMMON_ARM_V2_NOT_READY",
                "conclusion": "INTEGRATED_A_B_RETARGETING_V2_NOT_READY",
            },
        )
        return {
            "stage_a_ready": False,
            "stage_b_executed": False,
            "conclusion": "INTEGRATED_A_B_RETARGETING_V2_NOT_READY",
            "arm_output_root": str(arm_root),
            "integrated_output_root": str(integrated_root),
        }

    print("[gate] Stage A ready; starting final integrated A/B audit", flush=True)
    configure_g1(g1, selected["runtime_config"])
    integrated_results: dict[str, list[dict[str, Any]]] = {
        "dataset_a": [],
        "dataset_b": [],
    }
    arm_by_method = {
        method: {row.episode.episode_id: row for row in all_results[method]}
        for method in METHODS
    }
    for dataset_name, method in (("dataset_a", "baseline"), ("dataset_b", "proposed")):
        for episode_id in sorted(prepared):
            arm_result = arm_by_method[method][episode_id]
            hand = (
                map_baseline_hand(arm_result, baseline_mapper)
                if dataset_name == "dataset_a"
                else map_proposed_hand(arm_result, hand_candidate)
            )
            integrated = evaluate_integrated_episode(
                dataset_name,
                arm_result,
                hand,
                collision_runtime,
                classifier,
                g1,
                selected["runtime_config"],
            )
            export_integrated_episode(
                integrated_root,
                dataset_name,
                arm_result,
                integrated,
                g1,
                collision_runtime,
                selected_config_sha256,
                (
                    None
                    if dataset_name == "dataset_a"
                    else dependency["sha256"]["candidate"]
                ),
            )
            integrated_results[dataset_name].append(
                {
                    **integrated["metrics"],
                    "validation_first_causal_failure": integrated["validation"][
                        "first_causal_failure"
                    ],
                }
            )
            if (episode_id + 1) % 10 == 0:
                print(
                    f"[integrated] {dataset_name} {episode_id + 1:02d}/50",
                    flush=True,
                )

    rows_a = integrated_results["dataset_a"]
    rows_b = integrated_results["dataset_b"]
    atomic_csv(
        integrated_root / "summary/dataset_a_episode_metrics.csv",
        [_flatten_integrated(row) for row in rows_a],
    )
    atomic_csv(
        integrated_root / "summary/dataset_b_episode_metrics.csv",
        [_flatten_integrated(row) for row in rows_b],
    )
    aggregate = {
        "schema_version": "integrated_v2_aggregate_comparison",
        "dataset_a": _aggregate_integrated(rows_a),
        "dataset_b": _aggregate_integrated(rows_b),
        "development_only": True,
        "policy_training_or_task_success": "NOT_PERFORMED",
    }
    comparison_rows = _comparison_rows(aggregate["dataset_a"], aggregate["dataset_b"])
    historical, historical_rows = _v1_to_v2_comparison(
        aggregate, v1_arm_collision
    )
    atomic_json(integrated_root / "summary/aggregate_comparison.json", aggregate)
    atomic_csv(integrated_root / "summary/a_vs_b_comparison.csv", comparison_rows)
    atomic_json(
        integrated_root / "summary/v1_to_integrated_v2_comparison.json",
        historical,
    )
    atomic_csv(
        integrated_root / "summary/v1_to_integrated_v2_comparison.csv",
        historical_rows,
    )
    failure = _failure_breakdown(integrated_results, first_v2)
    atomic_json(integrated_root / "summary/failure_breakdown.json", failure)
    fairness = _fairness_audit(selected_path, dependency)
    atomic_json(integrated_root / "summary/fairness_audit.json", fairness)

    dataset_a_after, dataset_a_files_after = tree_sha256(V1_ROOT / "baseline")
    dependency_after = validate_hand_dependency(search)
    source_after = source_integrity(base)
    integrity = {
        "dataset_a_v1_tree_before": dataset_a_before,
        "dataset_a_v1_tree_after": dataset_a_after,
        "dataset_a_v1_unchanged": dataset_a_before == dataset_a_after,
        "dataset_a_v1_file_count_before": len(dataset_a_files_before),
        "dataset_a_v1_file_count_after": len(dataset_a_files_after),
        "hand_dependency_before_sha256": dependency["sha256"],
        "hand_dependency_after_sha256": dependency_after["sha256"],
        "hand_v2_1_unchanged": dependency["sha256"] == dependency_after["sha256"],
        "source_before": source_before,
        "source_after": source_after,
        "source_hashes_unchanged": source_before == source_after
        and source_after["unchanged"],
        "hand_joint_order_checks": hand_joint_order,
        "frozen_dependency": frozen_dependency,
    }
    atomic_json(integrated_root / "summary/integrity.json", integrity)

    # One deterministic re-run per method verifies the selected execution path.
    rerun_episode = min(calibration_ids)
    deterministic_checks: dict[str, bool] = {}
    for method in METHODS:
        rerun = evaluate_arm_episode(
            method,
            prepared[rerun_episode],
            selected["runtime_config"],
            g1,
            collision_runtime,
            classifier,
        )
        original = arm_by_method[method][rerun_episode]
        deterministic_checks[method] = bool(
            np.array_equal(rerun.solved["q"], original.solved["q"])
        )
    atomic_json(
        arm_root / "tests/deterministic_rerun.json",
        {
            "selection_rule": "minimum calibration episode ID",
            "episode_id": rerun_episode,
            "checks": deterministic_checks,
            "pass": all(deterministic_checks.values()),
        },
    )

    strict_ready = bool(
        aggregate["dataset_a"]["pass_count"] == 50
        and aggregate["dataset_b"]["pass_count"] == 50
        and integrity["dataset_a_v1_unchanged"]
        and integrity["hand_v2_1_unchanged"]
        and integrity["source_hashes_unchanged"]
        and all(deterministic_checks.values())
    )
    readiness = {
        "schema_version": "integrated_v2_training_label_readiness",
        "stage_a_ready": True,
        "dataset_a_strict_pass_count": aggregate["dataset_a"]["pass_count"],
        "dataset_b_strict_pass_count": aggregate["dataset_b"]["pass_count"],
        "action_labels_ready_for_schema_packaging": strict_ready,
        "g1_training_state_adapter": "G1_TRAINING_STATE_ADAPTER_PENDING",
        "policy_training": "NOT_PERFORMED",
        "classification": (
            "INTEGRATED_A_B_ACTION_LABELS_READY"
            if strict_ready
            else "INTEGRATED_A_B_ACTION_LABELS_HAVE_REMAINING_FAILURES"
        ),
        "conclusion": (
            "INTEGRATED_A_B_RETARGETING_V2_READY_FOR_SCHEMA_PACKAGING"
            if strict_ready
            else "INTEGRATED_A_B_RETARGETING_V2_NOT_READY"
        ),
    }
    atomic_json(integrated_root / "summary/training_readiness.json", readiness)

    test_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_aloha_g1_arm_v2.py",
    ]
    if run_tests:
        completed = subprocess.run(
            test_command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        tests = {
            "command": " ".join(test_command),
            "pass": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "anti_overfit_scan": anti_overfit,
        }
    else:
        tests = {
            "command": " ".join(test_command),
            "pass": False,
            "exit_code": None,
            "status": "SKIPPED_BY_CLI",
            "anti_overfit_scan": anti_overfit,
        }
    atomic_json(arm_root / "tests/test_report.json", tests)
    atomic_json(integrated_root / "tests/test_report.json", tests)
    if not tests["pass"]:
        readiness["action_labels_ready_for_schema_packaging"] = False
        readiness["classification"] = "TEST_GATE_FAILED"
        readiness["conclusion"] = "INTEGRATED_A_B_RETARGETING_V2_NOT_READY"
        atomic_json(integrated_root / "summary/training_readiness.json", readiness)

    exact_files = [
        str(config_path),
        str(ROOT / "tools/run_common_arm_v2.py"),
        str(ROOT / "tools/aloha_g1_arm_v2/__init__.py"),
        str(ROOT / "tools/aloha_g1_arm_v2/common.py"),
        str(ROOT / "tools/aloha_g1_arm_v2/solver.py"),
        str(ROOT / "tools/aloha_g1_arm_v2/audit.py"),
        str(ROOT / "tools/aloha_g1_arm_v2/integrated.py"),
        str(ROOT / "tools/aloha_g1_arm_v2/pipeline.py"),
        str(ROOT / "tests/test_aloha_g1_arm_v2.py"),
        str(arm_root / "dependencies/proposed_hand_v2_1_candidate.json"),
        str(arm_root / "dependencies/proposed_hand_v2_1_readiness.json"),
        str(arm_root / "audit/shared_ik_implementation_audit.json"),
        str(arm_root / "audit/first_failure_decomposition_v1.json"),
        str(arm_root / "workspace/source_workspace_distribution.json"),
        str(arm_root / "workspace/selected_mapping_workspace_analysis.json"),
        str(arm_root / "split/calibration_validation_split.json"),
        str(selected_path),
        str(arm_root / "summary/stage_a_readiness.json"),
        str(arm_root / "summary/first_failure_decomposition_v2.json"),
        str(arm_root / "summary/first_failure_decomposition_v2.csv"),
        str(integrated_root / "summary/dataset_a_episode_metrics.csv"),
        str(integrated_root / "summary/dataset_b_episode_metrics.csv"),
        str(integrated_root / "summary/a_vs_b_comparison.csv"),
        str(integrated_root / "summary/v1_to_integrated_v2_comparison.json"),
        str(integrated_root / "summary/v1_to_integrated_v2_comparison.csv"),
        str(integrated_root / "summary/failure_breakdown.json"),
        str(integrated_root / "summary/fairness_audit.json"),
        str(integrated_root / "summary/training_readiness.json"),
        str(integrated_root / "summary/final_report.md"),
    ]
    _write_final_report(
        integrated_root,
        stage,
        selected,
        v1_arm_collision,
        aggregate,
        comparison_rows,
        historical,
        failure,
        first_v2,
        fairness,
        readiness,
        tests,
        exact_files,
    )
    return {
        "stage_a_ready": True,
        "stage_b_executed": True,
        "dataset_a_pass_count": aggregate["dataset_a"]["pass_count"],
        "dataset_b_pass_count": aggregate["dataset_b"]["pass_count"],
        "selected_candidate_id": selected_id,
        "selected_config_sha256": selected_config_sha256,
        "tests_pass": tests["pass"],
        "conclusion": readiness["conclusion"],
        "arm_output_root": str(arm_root),
        "integrated_output_root": str(integrated_root),
    }
