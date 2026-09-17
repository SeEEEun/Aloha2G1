#!/usr/bin/env python3
"""Finalize the method-consistent Experiment-3 safety result.

This evaluator deliberately scores aborted trajectories only as diagnostic
executed prefixes.  Full-rollout metrics remain unavailable unless every A/B
rollout completes.  It never writes any frozen publication table.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.evaluate_paper_core_source_rollouts as source_evaluator
from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import G1Kinematics
from tools.evaluation.contracts import CANONICAL_PHASES, SEMANTIC_SUCCESS
from tools.evaluation.metrics import (
    evaluate_task_sequence,
    handoff_ordering_metrics,
    path_efficiency_metrics,
    smoothness_metrics,
)
from tools.evaluation.semantic_sequence import semantic_phase_events
from tools.paper_core_source_rollout_common import (
    ROOT,
    atomic_json,
    frozen_interfaces,
    read_json,
    sha256_file,
)


EXPERIMENT_NAME = "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT"
OUTPUT = ROOT / "outputs/paper_core_ab/method_consistent_source_video_rollout"
ROLLOUT_ROOT = OUTPUT / "rollouts"
MANIFEST = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
INITIALIZATION_CONTRACT = OUTPUT / "METHOD_CONSISTENT_INITIALIZATION_CONTRACT.json"
INITIALIZATION_CONTRACT_SHA256 = (
    "88e5d8e0b1234c0d58edd49be99906762e22db8bd9db91d63df9e8b202f205e8"
)
EVALUATION_CONTRACT = OUTPUT / "METHOD_CONSISTENT_EVALUATION_CONTRACT.json"
SEMANTIC_CONTRACT = ROOT / "configs/paper_semantic_task_sequence_v1.json"
TABLE_ROOT = ROOT / "outputs/paper_core_ab/tables"
PRESERVED_COMMON = OUTPUT / "preserved_common_state_diagnostic/files"
METHOD_DATASETS = {
    "a": ROOT / "datasets/doll_handoff_fair_a_heldout8",
    "b": ROOT / "datasets/doll_handoff_proposed_b_heldout8",
}
FPS = 30.0


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def table_inventory() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for table in ("table1_full50_retargeting", "table2_heldout_act_prediction", "table3_source_conditioned_rollout"):
        result[table] = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(TABLE_ROOT.glob(f"{table}.*"))
        ]
    return result


def event_map(reference: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {
        str(name): int(frame)
        for name, frame in zip(
            reference["event_names"].astype(str),
            reference["event_frames"].astype(np.int64),
            strict=True,
        )
    }


def derivatives(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qdot = np.diff(q, axis=0) * FPS
    qddot = np.diff(qdot, axis=0) * FPS
    jerk = np.diff(qddot, axis=0) * FPS
    return qdot, qddot, jerk


def rms(values: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(value))))


def distribution(values: list[float], unit: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("distribution requires finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
        "count": int(len(array)),
        "unit": unit,
    }


def evaluate_prefix(
    method: str,
    output_episode: int,
    entry: dict[str, Any],
    method_actions: np.ndarray,
    offset: int,
    g1: G1Kinematics,
    names: list[str],
    lower: np.ndarray,
    upper: np.ndarray,
    thresholds: Mapping[str, Any],
    semantic_config: Mapping[str, Any],
    initialization_contract: Mapping[str, Any],
) -> dict[str, Any]:
    # Reuse the already-frozen geometry definitions from the first Experiment-3
    # evaluator, but redirect it exclusively to the new output namespace.
    source_evaluator.ROLLOUT_ROOT = ROLLOUT_ROOT
    geometry_row = source_evaluator.evaluate_one(
        method,
        output_episode,
        entry,
        method_actions,
        offset,
        g1,
        names,
        lower,
        upper,
        thresholds,
    )
    path = Path(geometry_row["path"])
    report = read_json(path / "rollout_report.json")
    initial = read_json(path / "initial_condition.json")
    initial_verification = read_json(path / "initial_state_contract_verification.json")
    frozen_method = initialization_contract["entries"][output_episode]["methods"][method]

    if report.get("experiment_name") != EXPERIMENT_NAME:
        raise RuntimeError(f"wrong Experiment-3 namespace: {path}")
    if report["initialization"]["mode"] != "method_consistent_v1":
        raise RuntimeError(f"wrong initialization mode: {path}")
    if report["initialization"]["contract_sha256"] != INITIALIZATION_CONTRACT_SHA256:
        raise RuntimeError(f"initialization contract mismatch: {path}")
    if initial["q_float32_sha256"] != frozen_method["initial_q_float32_sha256"]:
        raise RuntimeError(f"initial q hash mismatch: {path}")
    if initial["initial_state_projection_applied"] or not initial_verification["exact_frozen_state0_unmodified"]:
        raise RuntimeError(f"initial state was modified: {path}")
    if report["official_act_execution"]["coefficient"] != 0.01:
        raise RuntimeError(f"ACT-E1 coefficient changed: {path}")
    if report["official_act_execution"]["control_fps"] != FPS:
        raise RuntimeError(f"control rate changed: {path}")

    requested = int(report["requested_frames"])
    executed = int(report["executed_frames"])
    complete = report["status"] == "PASS" and executed == requested
    if executed < 4:
        raise RuntimeError(f"executed prefix too short for diagnostic metrics: {path}")
    if not report["frame0_eligible"]:
        raise RuntimeError(f"unexpected ineligible episode was executed: {path}")

    with np.load(geometry_row["trajectory_evaluation_arrays"], allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    q = arrays["measured_q"].astype(np.float64)
    full_target = method_actions[offset : offset + requested].astype(np.float64)
    if q.shape != (executed, 28) or full_target.shape != (requested, 28):
        raise RuntimeError(f"trajectory dimensions changed: {path}")

    with np.load(entry["b_trajectory_path"], allow_pickle=False) as archive:
        source_reference = {key: np.asarray(archive[key]) for key in archive.files}
    events = event_map(source_reference)
    semantics = semantic_phase_events(
        q,
        full_target,
        event_frames=events,
        semantic_arrays={
            "left_hand_phase": source_reference["left_hand_phase"],
            "right_hand_phase": source_reference["right_hand_phase"],
            "ownership_state": source_reference["ownership_state"],
        },
        config=semantic_config,
    )
    task_sequence = evaluate_task_sequence(
        semantics["canonical_phase_events_frame"],
        success_kind=SEMANTIC_SUCCESS,
        authoritative=True,
    )
    ordering = geometry_row["handoff_ordering"]
    ordering_score = handoff_ordering_metrics(
        ordering["right_acquire_frame"],
        ordering["left_release_frame"],
        fps=FPS,
        authoritative=True,
    )
    reference_hand = {
        "left": arrays["source_left_interaction_world"],
        "right": arrays["source_right_interaction_world"],
    }
    candidate_hand = {
        "left": arrays["predicted_left_whole_hand_world"],
        "right": arrays["predicted_right_whole_hand_world"],
    }
    path_metrics = path_efficiency_metrics(
        reference_hand,
        candidate_hand,
        success=int(task_sequence["task_sequence_success"]),
        success_kind=SEMANTIC_SUCCESS,
    )
    smoothness = smoothness_metrics(q, fps=FPS)
    failed_checks = list(report["safety_abort"]["failed_checks"])
    diagnostic = {
        **geometry_row,
        "frame0_eligible": bool(report["frame0_eligible"]),
        "full_rollout_complete": bool(complete),
        "evaluation_scope": "ABORTED_EXECUTED_PREFIX_DIAGNOSTIC" if not complete else "FULL_ROLLOUT",
        "initialization_verification": {
            "exact_frozen_state0_unmodified": True,
            "frozen_initial_q_float32_sha256": frozen_method["initial_q_float32_sha256"],
            "immediate_reset_max_abs_rad": initial_verification["immediate_reset_max_abs_from_nominal_rad"],
            "settled_max_abs_rad": initial_verification["settled_max_abs_from_nominal_rad"],
        },
        "semantic_phase_detection": semantics,
        "semantic_task_sequence": task_sequence,
        "handoff_ordering_scored": ordering_score,
        "path_efficiency": path_metrics,
        "measured_smoothness": smoothness,
        "safety_failure_checks": failed_checks,
    }
    atomic_json(path / "method_consistent_prefix_evaluation.json", diagnostic)
    return diagnostic


def aggregate_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["full_rollout_complete"]]
    prefix_arrays = []
    wrist = []
    whole_hand = []
    bimanual = []
    qdot = []
    qddot = []
    jerk = []
    rpl = []
    swpe = []
    ordering_ready = []
    failure_checks: dict[str, int] = {}
    for row in rows:
        with np.load(row["trajectory_evaluation_arrays"], allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        prefix_arrays.append(arrays)
        wrist.extend((arrays["left_wrist_error_m"], arrays["right_wrist_error_m"]))
        whole_hand.extend((arrays["left_whole_hand_error_m"], arrays["right_whole_hand_error_m"]))
        bimanual.append(arrays["bimanual_relation_error_m"])
        q = arrays["measured_q"].astype(np.float64)
        velocity, acceleration, joint_jerk = derivatives(q)
        qdot.append(velocity)
        qddot.append(acceleration)
        jerk.append(joint_jerk)
        rpl.append(float(row["path_efficiency"]["RPL"]))
        swpe.append(float(row["path_efficiency"]["SWPE"]))
        if row["handoff_ordering_scored"].get("status") == "READY":
            ordering_ready.append(row["handoff_ordering_scored"])
        for failed in row["safety_failure_checks"]:
            failure_checks[failed] = failure_checks.get(failed, 0) + 1

    phase_scores = [float(row["semantic_task_sequence"]["phase_completion_score"]) for row in rows]
    phase_counts = [int(row["semantic_task_sequence"]["completed_ordered_phases"]) for row in rows]
    semantic_successes = [int(row["semantic_task_sequence"]["task_sequence_success"]) for row in rows]
    hard_collision_episode = [
        int(row["collision"]["hard_collision_frame_incidence"] > 0) for row in rows
    ]
    prefix_metrics = {
        "scope": "diagnostic executed prefixes only; not Table-3 full-rollout results",
        "executed_frames": int(sum(row["executed_frames"] for row in rows)),
        "duration_seconds": float(sum(row["duration_seconds"] for row in rows)),
        "source_wrist_error_mm": source_evaluator.stats(np.concatenate(wrist), 1000.0),
        "whole_hand_interaction_error_mm": source_evaluator.stats(
            np.concatenate(whole_hand), 1000.0
        ),
        "bimanual_relation_error_mm": source_evaluator.stats(
            np.concatenate(bimanual), 1000.0
        ),
        "joint_smoothness": {
            "qdot_rms_rad_s": rms(np.concatenate(qdot, axis=0)),
            "qddot_rms_rad_s2": rms(np.concatenate(qddot, axis=0)),
            "joint_jerk_rms_rad_s3": rms(np.concatenate(jerk, axis=0)),
            "derivatives_cross_episode_boundaries": False,
        },
        "relative_path_length": distribution(rpl, "ratio"),
        "SWPE": distribution(swpe, "ratio"),
        "semantic_task_sequence_success": {
            "successful": int(sum(semantic_successes)),
            "total": len(rows),
            "success_kind": SEMANTIC_SUCCESS,
            "physical_success_claimed": False,
        },
        "phase_completion": {
            "completed_ordered_phases": int(sum(phase_counts)),
            "possible_ordered_phases": len(rows) * len(CANONICAL_PHASES),
            "mean_score": float(np.mean(phase_scores)),
            "per_episode_scores": phase_scores,
        },
        "handoff_ordering_accuracy": {
            "successful": int(sum(item["score"] for item in ordering_ready)),
            "derivable": len(ordering_ready),
            "total_rollouts": len(rows),
            "status": "READY" if ordering_ready else "NA",
            "reason": None if ordering_ready else "no aborted prefix contains both detected handoff events",
        },
    }
    safety = {
        "safety_abort_episodes": int(sum(row["status"] == "SAFETY_ABORT" for row in rows)),
        "failure_check_episode_counts": dict(sorted(failure_checks.items())),
        "hard_self_collision_failure_episodes": int(sum(hard_collision_episode)),
        "table_contact_failure_episodes": int(
            sum("table_contact" in row["safety_failure_checks"] for row in rows)
        ),
        "command_hard_limit_failure_episodes": int(
            sum("command_hard_limits" in row["safety_failure_checks"] for row in rows)
        ),
        "branch_failure_episodes": int(
            sum("branch" in row["safety_failure_checks"] for row in rows)
        ),
        "adjacent_step_failure_episodes": int(
            sum("adjacent_step" in row["safety_failure_checks"] for row in rows)
        ),
        "velocity_failure_episodes": int(
            sum("velocity" in row["safety_failure_checks"] for row in rows)
        ),
        "measured_prefix_joint_limit_violation_frames_diagnostic": int(
            sum(row["joint_limits"]["violation_frames"] for row in rows)
        ),
        "measured_prefix_branch_discontinuities_diagnostic": int(
            sum(row["branch_discontinuity_count"] for row in rows)
        ),
    }
    return {
        "frame0_eligible_episodes": int(sum(row["frame0_eligible"] for row in rows)),
        "total_episodes": len(rows),
        "completed_rollouts": len(completed),
        "full_rollout_metrics": {
            "status": "NA" if not completed else "READY",
            "reason": "zero complete rollouts; publication metrics are not computed from aborted prefixes"
            if not completed
            else None,
        },
        "diagnostic_prefix_metrics": prefix_metrics,
        "safety": safety,
        "per_episode": rows,
    }


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT",
        "",
        "Status: **NOT_EXECUTABLE**",
        "",
        "The method-consistent frozen state[0] initialization resolved the frame-0 blocker for all 16 method/episode runs. Every eligible rollout subsequently reached an unchanged safety abort, so no full-rollout metric is publication-eligible and Table 3 remains frozen.",
        "",
        "| Metric | ACT-A40 | ACT-B40 |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("frame0_eligible_episodes", "Frame-0 eligible"),
        ("completed_rollouts", "Completed full rollouts"),
    ):
        lines.append(f"| {label} | {result['methods']['a'][key]}/8 | {result['methods']['b'][key]}/8 |")
    for method in ("a", "b"):
        value = result["methods"][method]
        prefix = value["diagnostic_prefix_metrics"]
        safety = value["safety"]
        lines.extend(
            [
                "",
                f"## ACT-{method.upper()}40 diagnostic executed prefixes",
                "",
                f"- Safety aborts: {safety['safety_abort_episodes']}/8",
                f"- Semantic task-sequence success: {prefix['semantic_task_sequence_success']['successful']}/8 (non-physical)",
                f"- Phase completion: {prefix['phase_completion']['completed_ordered_phases']}/{prefix['phase_completion']['possible_ordered_phases']} (mean score {prefix['phase_completion']['mean_score']:.6f})",
                f"- Wrist error: {prefix['source_wrist_error_mm']['mean']:.3f} mm mean / {prefix['source_wrist_error_mm']['p95']:.3f} mm p95",
                f"- Whole-hand interaction error: {prefix['whole_hand_interaction_error_mm']['mean']:.3f} mm mean / {prefix['whole_hand_interaction_error_mm']['p95']:.3f} mm p95",
                f"- Bimanual relation error: {prefix['bimanual_relation_error_mm']['mean']:.3f} mm mean / {prefix['bimanual_relation_error_mm']['p95']:.3f} mm p95",
                f"- Joint jerk RMS: {prefix['joint_smoothness']['joint_jerk_rms_rad_s3']:.3f} rad/s^3",
                f"- Relative path length: {prefix['relative_path_length']['mean']:.6f} mean",
                f"- SWPE: {prefix['SWPE']['mean']:.6f} mean ({SEMANTIC_SUCCESS})",
                f"- Handoff ordering: {prefix['handoff_ordering_accuracy']['successful']}/{prefix['handoff_ordering_accuracy']['derivable']} derivable prefixes",
                f"- Hard self-collision failures: {safety['hard_self_collision_failure_episodes']}/8",
                f"- Table-contact failures: {safety['table_contact_failure_episodes']}/8",
                f"- Command hard-limit failures: {safety['command_hard_limit_failure_episodes']}/8",
                f"- Branch failures: {safety['branch_failure_episodes']}/8",
            ]
        )
    lines.extend(
        [
            "",
            "All geometric, smoothness, phase, path, and SWPE numbers above are explicitly aborted-prefix diagnostics. They are not complete-rollout results and were not written to Table 3.",
            "",
            "No onboard-G1 visual autonomy, physical manipulation success, real-world task success, or sim-to-real result is claimed.",
            "",
            "EXPERIMENT3_NOT_EXECUTABLE",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    table_before = table_inventory()
    evaluation_contract = read_json(EVALUATION_CONTRACT)
    initialization_contract = read_json(INITIALIZATION_CONTRACT)
    semantic_config = read_json(SEMANTIC_CONTRACT)
    batch = read_json(OUTPUT / "batch_result.json")
    if sha256_file(INITIALIZATION_CONTRACT) != INITIALIZATION_CONTRACT_SHA256:
        raise RuntimeError("method-consistent initialization contract changed")
    if evaluation_contract["status"] != "FROZEN_BEFORE_METHOD_CONSISTENT_INFERENCE":
        raise RuntimeError("evaluation contract was not frozen before inference")
    if semantic_config["status"] != "FROZEN_BEFORE_SOURCE_CONDITIONED_ACT_RESULTS":
        raise RuntimeError("semantic evaluator contract changed")
    if batch["run_count"] != 16 or batch["experiment_name"] != EXPERIMENT_NAME:
        raise RuntimeError("the full 16-run batch is absent")
    for table_group in ("table1", "table2"):
        expected = {
            Path(item["path"]).name: item["sha256"]
            for item in evaluation_contract["frozen_inputs"][table_group]
        }
        actual = {
            Path(item["path"]).name: item["sha256"]
            for item in table_before[
                "table1_full50_retargeting" if table_group == "table1" else "table2_heldout_act_prediction"
            ]
        }
        if actual != expected:
            raise RuntimeError(f"frozen {table_group} changed")
    preserved_required = [
        PRESERVED_COMMON / "common_state_experiment3_result.json",
        PRESERVED_COMMON / "paper_core_blocked_report.json",
        PRESERVED_COMMON / "PAPER_CORE_BLOCKED_REPORT.md",
    ]
    if not all(path.is_file() for path in preserved_required):
        raise RuntimeError("previous common-state blocked diagnostic was not preserved")

    manifest = read_json(MANIFEST)
    entries = manifest["entries"]
    offsets = source_evaluator.episode_offsets(entries)
    actions = {
        method: source_evaluator.parquet_actions(path)
        for method, path in METHOD_DATASETS.items()
    }
    names, lower, upper, _ = frozen_interfaces()
    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    thresholds = read_json(
        ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
    )["unchanged_acceptance"]
    rows = {
        method: [
            evaluate_prefix(
                method,
                output_episode,
                entry,
                actions[method],
                offsets[output_episode],
                g1,
                names,
                lower,
                upper,
                thresholds,
                semantic_config,
                initialization_contract,
            )
            for output_episode, entry in enumerate(entries)
        ]
        for method in ("a", "b")
    }

    source_identity = []
    for output_episode in range(8):
        row_a = rows["a"][output_episode]
        row_b = rows["b"][output_episode]
        compared = min(len(row_a["source_frame_hashes"]), len(row_b["source_frame_hashes"]))
        exact = bool(
            row_a["source_video_sha256"] == row_b["source_video_sha256"]
            and np.array_equal(
                np.asarray(row_a["source_frame_hashes"][:compared]),
                np.asarray(row_b["source_frame_hashes"][:compared]),
            )
        )
        if not exact:
            raise RuntimeError(f"A/B source identity differs at heldout episode {output_episode}")
        source_identity.append(
            {
                "heldout_output_episode": output_episode,
                "source_final_episode": int(entries[output_episode]["final_dataset_index"]),
                "source_video_sha256": row_a["source_video_sha256"],
                "exact_decoded_prefix_identity": True,
                "compared_prefix_frames": compared,
            }
        )
    methods = {method: aggregate_method(value) for method, value in rows.items()}
    all_complete = all(methods[method]["completed_rollouts"] == 8 for method in ("a", "b"))
    if all_complete:
        raise RuntimeError("all rollouts completed; this non-executable finalizer must not update Table 3")

    table_after = table_inventory()
    if table_after != table_before:
        raise RuntimeError("a frozen publication table changed during evaluation")
    result = {
        "schema_version": "paper_core_method_consistent_source_video_rollout_result_v1",
        "status": "NOT_EXECUTABLE",
        "terminal_status": "EXPERIMENT3_NOT_EXECUTABLE",
        "experiment_name": EXPERIMENT_NAME,
        "frame0_initialization_result": "PASS_ALL_16",
        "full_rollout_result": "ZERO_OF_16_COMPLETE_DUE_UNCHANGED_POST_FRAME0_SAFETY_ABORTS",
        "evaluation_contract": str(EVALUATION_CONTRACT),
        "evaluation_contract_sha256": sha256_file(EVALUATION_CONTRACT),
        "initialization_contract": str(INITIALIZATION_CONTRACT),
        "initialization_contract_sha256": sha256_file(INITIALIZATION_CONTRACT),
        "batch_result": str(OUTPUT / "batch_result.json"),
        "batch_result_sha256": sha256_file(OUTPUT / "batch_result.json"),
        "source_input_identity": source_identity,
        "methods": methods,
        "publication_tables": {
            "table1_unchanged": True,
            "table2_unchanged": True,
            "table3_updated": False,
            "reason": "no full rollout completed; frozen predeclared Table-3 update gate not met",
            "inventory_before_and_after": table_after,
        },
        "previous_common_state_diagnostic": {
            "preserved": True,
            "files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in preserved_required
            ],
        },
        "claims": {
            "source_video_conditioned_simulation_only": True,
            "g1_onboard_rgb_used": False,
            "onboard_visual_autonomy": False,
            "physical_manipulation_success": False,
            "real_world_task_success": False,
            "sim_to_real": False,
            "real_hardware": False,
        },
        "stopped_without_retry_or_threshold_change": True,
    }
    result_path = OUTPUT / "experiment3_method_consistent_result.json"
    atomic_json(result_path, result)
    report_path = OUTPUT / "METHOD_CONSISTENT_SOURCE_VIDEO_ROLLOUT_REPORT.md"
    atomic_text(report_path, render_report(result))
    checksums = {
        "experiment_result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "report": {"path": str(report_path), "sha256": sha256_file(report_path)},
    }
    atomic_json(OUTPUT / "method_consistent_result_checksums.json", checksums)
    print(json.dumps(result, indent=2, default=source_evaluator.json_default))


if __name__ == "__main__":
    main()
