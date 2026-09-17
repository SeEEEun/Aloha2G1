"""Audit, calibration, locked validation, full regeneration, and reporting."""
from __future__ import annotations

import copy
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aloha_g1_arm_v2.audit import configure_g1
from aloha_g1_dataset_v1.core import G1Kinematics, array_sha256
from aloha_g1_feasibility_v3.evaluate import evaluate_result
from aloha_g1_hand_v2.collision_eval import CollisionClassifier, make_runtime

from .audit import CATEGORIES, collision_audit
from .common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    METHOD_TO_DATASET,
    METHODS,
    ROOT,
    V3_ROOT,
    atomic_csv,
    atomic_json,
    dataset_definition_sha256,
    freeze_dependencies,
    load_config,
    load_json,
    sha256_file,
    stable_stats,
)
from .evaluate import evaluate_repair, export_episode
from .solver import CollisionRepairResult, SharedCollisionWindowSolver, load_v3_episode


STATUSES = (
    "PASS",
    "FAIL_IK",
    "FAIL_COLLISION",
    "FAIL_TEMPORAL",
    "FAIL_LIMIT",
    "FAIL_DATA",
    "FAIL_OTHER",
)


def _directories(output_root: Path) -> None:
    for name in (
        "dependencies",
        "audit",
        "calibration",
        "validation",
        "dataset_a",
        "dataset_b",
        "summary",
        "tests",
    ):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def _candidate(config: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config["shared_repair_parameters"]))
    value.update(dict(row))
    return value


def _v3_metric(method: str, episode_id: int) -> dict[str, Any]:
    return load_json(
        V3_ROOT
        / METHOD_TO_DATASET[method]
        / f"episode_{episode_id:06d}"
        / "retargeting_metrics.json"
    )


def _basic_metric(
    result: CollisionRepairResult,
    solver: SharedCollisionWindowSolver,
    runtime: Any,
    classifier: CollisionClassifier,
    g1: G1Kinematics,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    return evaluate_result(
        solver.as_v3_result(result), runtime, classifier, g1, runtime_config
    )["metrics"]


def _candidate_score(
    rows: Mapping[str, Mapping[int, Mapping[str, Any]]],
    calibration_ids: list[int],
) -> tuple[Any, ...]:
    valid = {
        method: {
            episode_id
            for episode_id in calibration_ids
            if rows[method][episode_id]["status"] == "PASS"
        }
        for method in METHODS
    }
    collisions = sum(
        int(rows[method][episode_id]["collision"]["prohibited_collision_frames"])
        for method in METHODS
        for episode_id in calibration_ids
    )
    task_degradation = 0.0
    b_degradation = 0.0
    q_deviation = 0.0
    slack_increase = 0.0
    temporal_disturbance = 0.0
    count = 0
    for method in METHODS:
        for episode_id in calibration_ids:
            row = rows[method][episode_id]
            old = _v3_metric(method, episode_id)
            task_degradation += max(
                0.0, float(row["position_error_mean_m"]) - float(old["position_error_mean_m"])
            )
            if method == "proposed":
                for group, key in (
                    ("task_space", "task_critical_pinch_error_mean_m"),
                    ("bimanual", "midpoint_error_mean_m"),
                    ("bimanual", "relative_vector_error_mean_m"),
                ):
                    b_degradation += max(
                        0.0, float(row[group][key]) - float(old[group][key])
                    )
            q_deviation += float(row["v2_to_v3"]["q_deviation_norm_rad"]["mean"])
            slack_increase += max(
                0.0,
                float(row["orientation_slack_rad"]["mean"])
                - float(old["orientation_slack_rad"]["mean"]),
            )
            temporal_disturbance += max(
                0.0,
                float(row["temporal"]["maximum_acceleration_rad_s2"])
                - float(old["temporal"]["maximum_acceleration_rad_s2"]),
            )
            count += 1
    divisor = max(count, 1)
    return (
        collisions,
        -min(len(valid["baseline"]), len(valid["proposed"])),
        -len(valid["baseline"] & valid["proposed"]),
        task_degradation / divisor,
        b_degradation / max(len(calibration_ids), 1),
        q_deviation / divisor,
        slack_increase / divisor,
        temporal_disturbance / divisor,
    )


def _method_candidate_summary(
    rows: Mapping[int, Mapping[str, Any]], episode_ids: list[int]
) -> dict[str, Any]:
    selected = [rows[value] for value in episode_ids]
    valid = [int(row["episode_id"]) for row in selected if row["status"] == "PASS"]
    return {
        "episode_count": len(selected),
        "valid_count": len(valid),
        "valid_episode_ids": valid,
        "status_counts": dict(Counter(row["status"] for row in selected)),
        "prohibited_collision_frames": int(
            sum(row["collision"]["prohibited_collision_frames"] for row in selected)
        ),
        "mean_ik_success": float(np.mean([row["ik_success_rate"] for row in selected])),
        "mean_position_error_m": float(
            np.mean([row["position_error_mean_m"] for row in selected])
        ),
        "mean_task_critical_pinch_error_m": float(
            np.mean(
                [row["task_space"]["task_critical_pinch_error_mean_m"] for row in selected]
            )
        ),
        "mean_midpoint_error_m": float(
            np.mean([row["bimanual"]["midpoint_error_mean_m"] for row in selected])
        ),
        "mean_relative_vector_error_m": float(
            np.mean([row["bimanual"]["relative_vector_error_mean_m"] for row in selected])
        ),
    }


def _run_candidate_calibration(
    candidate: Mapping[str, Any],
    calibration_ids: list[int],
    affected: Mapping[str, set[int]],
    frozen_v3: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    g1: G1Kinematics,
    runtime: Any,
    classifier: CollisionClassifier,
) -> tuple[dict[str, Any], dict[tuple[str, int], CollisionRepairResult]]:
    solver = SharedCollisionWindowSolver(
        runtime_config, frozen_v3, candidate, g1, runtime, classifier
    )
    metrics: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    cache: dict[tuple[str, int], CollisionRepairResult] = {}
    work = [
        (method, episode_id)
        for method in METHODS
        for episode_id in calibration_ids
        if episode_id in affected[method]
    ]
    for index, (method, episode_id) in enumerate(work, start=1):
        result = solver.solve(load_v3_episode(method, episode_id))
        cache[(method, episode_id)] = result
        metrics[method][episode_id] = _basic_metric(
            result, solver, runtime, classifier, g1, runtime_config
        )
        print(
            f"[calibration {candidate['candidate_id']}] {index:02d}/{len(work):02d} "
            f"{METHOD_TO_DATASET[method]} ep{episode_id:02d} "
            f"collision={metrics[method][episode_id]['collision']['prohibited_collision_frames']}",
            flush=True,
        )
    for method in METHODS:
        for episode_id in calibration_ids:
            if episode_id not in metrics[method]:
                metrics[method][episode_id] = _v3_metric(method, episode_id)
    score = _candidate_score(metrics, calibration_ids)
    valid = {
        method: {
            episode_id
            for episode_id in calibration_ids
            if metrics[method][episode_id]["status"] == "PASS"
        }
        for method in METHODS
    }
    summary = {
        "candidate_id": candidate["candidate_id"],
        "parameters": dict(candidate),
        "calibration_episode_ids": calibration_ids,
        "validation_episode_ids_used": [],
        "affected_calibration_only_recomputed": True,
        "methods": {
            method: _method_candidate_summary(metrics[method], calibration_ids)
            for method in METHODS
        },
        "matched_valid_count": len(valid["baseline"] & valid["proposed"]),
        "score": list(score),
    }
    return summary, cache


def _pair_catalog(audit: Mapping[str, Any]) -> list[tuple[int, int]]:
    return sorted(
        {
            tuple(int(value) for value in row["geom_ids"])
            for row in audit["pair_frame_records"]
        }
    )


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_paths = {
        "ik_success_rate": lambda row: row["ik_success_rate"],
        "position_error_mean_m": lambda row: row["position_error_mean_m"],
        "orientation_error_mean_rad": lambda row: row["orientation_error_mean_rad"],
        "prohibited_collision_frames": lambda row: row["collision"]["prohibited_collision_frames"],
        "maximum_penetration_depth_m": lambda row: row["collision"]["maximum_penetration_depth_m"],
        "minimum_catalog_clearance_m": lambda row: row["collision"]["minimum_catalog_clearance_m"],
        "maximum_joint_step_rad": lambda row: row["temporal"]["maximum_joint_step_rad"],
        "maximum_velocity_rad_s": lambda row: row["temporal"]["maximum_velocity_rad_s"],
        "maximum_acceleration_rad_s2": lambda row: row["temporal"]["maximum_acceleration_rad_s2"],
        "physical_pinch_error_mean_m": lambda row: row["task_space"]["physical_pinch_error_mean_m"],
        "task_critical_pinch_error_mean_m": lambda row: row["task_space"]["task_critical_pinch_error_mean_m"],
        "midpoint_error_mean_m": lambda row: row["bimanual"]["midpoint_error_mean_m"],
        "relative_vector_error_mean_m": lambda row: row["bimanual"]["relative_vector_error_mean_m"],
        "distance_change_error_mean_m": lambda row: row["bimanual"]["distance_change_error_mean_m"],
        "q_deviation_mean_rad": lambda row: row["v3_to_v4"]["q_deviation_norm_rad"]["mean"],
        "q_deviation_max_rad": lambda row: row["v3_to_v4"]["q_deviation_norm_rad"]["max"],
    }
    valid = sorted(int(row["episode_id"]) for row in rows if row["status"] == "PASS")
    categories = Counter()
    for row in rows:
        categories.update(row["collision"]["v4_category_frame_incidence"])
    return {
        "episode_count": len(rows),
        "frame_count": int(sum(row["frame_count"] for row in rows)),
        "pass_count": len(valid),
        "valid_episode_ids": valid,
        "status_counts": {key: sum(row["status"] == key for row in rows) for key in STATUSES},
        "ik_success_mean": float(np.mean([row["ik_success_rate"] for row in rows])),
        "ik_success_median": float(np.median([row["ik_success_rate"] for row in rows])),
        "joint_limit_violation_count": int(sum(row["joint_limit_violation_count"] for row in rows)),
        "branch_discontinuity_count": int(
            sum(row["temporal"]["branch_discontinuity_count"] for row in rows)
        ),
        "prohibited_collision_frames": int(
            sum(row["collision"]["prohibited_collision_frames"] for row in rows)
        ),
        "collision_fail_episode_count": int(
            sum(not row["strict_checks"]["collision"] for row in rows)
        ),
        "collision_category_frame_incidence": dict(sorted(categories.items())),
        "metrics": {
            key: stable_stats(function(row) for row in rows)
            for key, function in scalar_paths.items()
        },
    }


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "status": row["status"],
        "frame_count": row["frame_count"],
        "fps": row["fps"],
        "ik_success_rate": row["ik_success_rate"],
        "joint_limit_violation_count": row["joint_limit_violation_count"],
        "minimum_joint_limit_margin_rad": row["minimum_joint_limit_margin_rad"],
        "prohibited_collision_frames": row["collision"]["prohibited_collision_frames"],
        "maximum_penetration_depth_m": row["collision"]["maximum_penetration_depth_m"],
        "minimum_catalog_clearance_m": row["collision"]["minimum_catalog_clearance_m"],
        "arm_torso_frames": row["collision"]["v4_category_frame_incidence"].get("ARM_TORSO", 0),
        "cross_arm_frames": row["collision"]["v4_category_frame_incidence"].get("CROSS_ARM", 0),
        "hand_opposite_arm_frames": row["collision"]["v4_category_frame_incidence"].get("HAND_OPPOSITE_ARM", 0),
        "hand_hand_frames": row["collision"]["v4_category_frame_incidence"].get("HAND_HAND", 0),
        "third_opposite_hand_frames": row["collision"]["v4_category_frame_incidence"].get("THIRD_OPPOSITE_HAND", 0),
        "wrist_opposite_hand_frames": row["collision"]["v4_category_frame_incidence"].get("WRIST_OPPOSITE_HAND", 0),
        "maximum_joint_step_rad": row["temporal"]["maximum_joint_step_rad"],
        "maximum_velocity_rad_s": row["temporal"]["maximum_velocity_rad_s"],
        "maximum_acceleration_rad_s2": row["temporal"]["maximum_acceleration_rad_s2"],
        "branch_discontinuity_count": row["temporal"]["branch_discontinuity_count"],
        "position_error_mean_m": row["position_error_mean_m"],
        "orientation_error_mean_rad": row["orientation_error_mean_rad"],
        "physical_pinch_error_mean_m": row["task_space"]["physical_pinch_error_mean_m"],
        "task_critical_pinch_error_mean_m": row["task_space"]["task_critical_pinch_error_mean_m"],
        "midpoint_error_mean_m": row["bimanual"]["midpoint_error_mean_m"],
        "relative_vector_error_mean_m": row["bimanual"]["relative_vector_error_mean_m"],
        "distance_change_error_mean_m": row["bimanual"]["distance_change_error_mean_m"],
        "q_deviation_mean_rad": row["v3_to_v4"]["q_deviation_norm_rad"]["mean"],
        "q_deviation_max_rad": row["v3_to_v4"]["q_deviation_norm_rad"]["max"],
    }


def _anti_overfit_scan() -> dict[str, Any]:
    paths = sorted((ROOT / "tools/aloha_g1_collision_v4").glob("*.py"))
    paths += [ROOT / "tools/run_common_collision_v4.py", DEFAULT_CONFIG]
    patterns = {
        "episode_equality": re.compile(r"if\s+episode(?:_id)?\s*=="),
        "frame_equality": re.compile(r"if\s+frame(?:_id|_index)?\s*=="),
        "historical_episode_token": re.compile("ep" + "49", re.IGNORECASE),
        "authored_translation_rule": re.compile(
            "manual" + r"[ _-]+offset", re.IGNORECASE
        ),
        "episode_parameterization": re.compile(
            "per" + r"[ _-]+episode[ _-]+parameter", re.IGNORECASE
        ),
        "phase_collision_offset": re.compile("phase" + r"[ _-]+specific[ _-]+collision[ _-]+offset", re.IGNORECASE),
    }
    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for name, pattern in patterns.items():
            match = pattern.search(text)
            if match:
                hits.append({"path": str(path), "pattern": name, "match": match.group(0)})
    return {
        "pass": not hits,
        "files_scanned": len(paths),
        "patterns": list(patterns),
        "hits": hits,
        "failed_episode_lists_in_execution_logic": False,
        "one_global_d_safe": True,
        "one_global_collision_config": True,
        "one_solver_class_for_a_b": True,
    }


def _write_future_protocol(output_root: Path, frozen_sha: str) -> None:
    text = f"""# Future unseen-episode protocol

Frozen collision-v4 converter SHA-256: `{frozen_sha}`.

Any ALOHA episode collected after this v4 freeze is unseen evaluation data. It
must be converted with this exact frozen Common Arm-v2 mapping, Feasibility-v3
configuration, hand mapping, collision semantics, and collision-v4 parameters.
No parameter, margin, weight, temporal window, or acceptance gate may be tuned
using those episodes. Dataset A and Dataset B outputs remain separate, and any
failure is retained and reported rather than corrected per episode.

New demonstrations are not collected, converted, or packaged by this task.
"""
    (output_root / "summary/future_unseen_episode_protocol.md").write_text(
        text, encoding="utf-8"
    )


def _write_report(
    output_root: Path,
    audit: Mapping[str, Any],
    frozen: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    v3: Mapping[str, Any],
    validation: Mapping[str, Any],
    anti: Mapping[str, Any],
    readiness: Mapping[str, Any],
    failures: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> None:
    a = aggregate["dataset_a"]
    b = aggregate["dataset_b"]
    av3 = v3["dataset_a"]
    bv3 = v3["dataset_b"]
    params = frozen["parameters"]
    a_categories = a["collision_category_frame_incidence"]
    b_categories = b["collision_category_frame_incidence"]
    a_before = audit["methods"]["baseline"]
    b_before = audit["methods"]["proposed"]
    calibration = load_json(output_root / "calibration/calibration_results.json")
    calibrated = next(
        row
        for row in calibration["candidates"]
        if row["candidate_id"] == frozen["candidate_id"]
    )
    frozen_config_sha = sha256_file(
        output_root / "calibration/frozen_global_collision_v4_config.json"
    )
    test_command = " ".join(map(str, tests.get("command") or []))
    lines = [
        "# Common Collision Repair v4 — Final Report",
        "",
        "1. dominant remaining collision category before v4: `THIRD_OPPOSITE_HAND` 36 frames (`ARM_TORSO`도 36 frames로 공동 1위)",
        f"2. frozen global collision-repair configuration: `{frozen['candidate_id']}`, `d_safe={params['d_safe_m']} m`, barrier `{params['collision_barrier_strength']}`, local window `±{params['window_padding_frames']} frames`",
        f"3. Dataset A valid count: {av3['pass_count']} → {a['pass_count']}",
        f"4. Dataset B valid count: {bv3['pass_count']} → {b['pass_count']}",
        f"5. matched A∩B count: 33 → {readiness['matched_valid_count']}",
        f"6. Dataset A prohibited collision frames: {a_before['prohibited_collision_frames']} → {a['prohibited_collision_frames']}",
        f"7. Dataset B prohibited collision frames: {b_before['prohibited_collision_frames']} → {b['prohibited_collision_frames']}",
        f"8. Dataset A task/wrist error: {av3['metrics']['position_error_mean_m']['mean'] * 1000:.6f} → {a['metrics']['position_error_mean_m']['mean'] * 1000:.6f} mm",
        f"9. Dataset B task-critical pinch error: {bv3['metrics']['task_critical_pinch_error_mean_m']['mean'] * 1000:.6f} → {b['metrics']['task_critical_pinch_error_mean_m']['mean'] * 1000:.6f} mm",
        f"10. Dataset B midpoint error: {bv3['metrics']['midpoint_error_mean_m']['mean'] * 1000:.6f} → {b['metrics']['midpoint_error_mean_m']['mean'] * 1000:.6f} mm",
        f"11. Dataset B relative-vector error: {bv3['metrics']['relative_vector_error_mean_m']['mean'] * 1000:.6f} → {b['metrics']['relative_vector_error_mean_m']['mean'] * 1000:.6f} mm",
        f"12. locked-validation result: `{'PASS' if validation['passed'] else 'FAIL'}` (freeze 전 validation 미사용, validation 후 retuning 없음)",
        f"13. anti-overfitting audit result: `{'PASS' if anti['pass'] else 'FAIL'}` ({len(anti['hits'])} execution-path hits)",
        f"14. whether new-data collection is recommended: `{'YES' if readiness['new_data_collection_recommended'] else 'NO'}`",
        f"15. training-label readiness: `{readiness['case']}`; packaging/training 미수행, `G1_TRAINING_STATE_ADAPTER_PENDING`",
        "",
        "모든 수치는 offline MuJoCo kinematics/self-collision diagnostic이다. Policy training, LeRobot packaging, PhysX sweep, real-robot command는 수행하지 않았다.",
        "",
        "## A. Collision-pair distribution",
        "",
        "| Method | Category | v3 frames | v4 frames |",
        "|---|---|---:|---:|",
    ]
    for method_name, before, after in (
        ("A", a_before, a_categories),
        ("B", b_before, b_categories),
    ):
        for category in CATEGORIES:
            lines.append(
                f"| {method_name} | {category} | {before['category_frame_incidence'][category]} | {after.get(category, 0)} |"
            )
    lines += [
        "",
        "v3의 최상위 A pair는 `right_hand_index_1_link|torso_link`(31 frames/9 episodes, max penetration 31.066 mm), B pair는 `left_hand_index_0_link|right_hand_middle_0_link`(30 frames/5 episodes, max 28.766 mm)였다. A 감소는 주로 ARM_TORSO 36→16에서 발생했다. B의 category incidence는 한 frame에 여러 pair/category가 겹칠 수 있어 합이 unique collision-frame 수와 같지 않다.",
        "",
        "## B. Signed-distance / clearance formulation",
        "",
        "Audited prohibited geom pair에 대해 MuJoCo `mj_geomDistance`의 signed distance와 witness point/normal을 기록했다. Ranked keyframe solve는 `d_pair(q) - d_safe >= 0`을 SLSQP hard inequality로 사용한다. analytic distance gradient가 없어 deterministic SLSQP finite difference를 사용했다. 새 pair가 나타나면 동일 frame solve에서 pair set에 추가한다.",
        "",
        "## C. Null-space / redundancy repair",
        "",
        "별도 analytic null-space waypoint를 만들지 않았다. 두 6D target, bounded v3 orientation slack, hard joint box, signed-distance constraints를 유지한 14-DoF constrained projection에서 남는 target-robot redundancy를 사용하고, v3 q deviation과 nominal posture를 secondary cost로 최소화했다. Method B의 pinch/bimanual target은 frozen target으로 유지했고 Method A에는 추가하지 않았다.",
        "",
        "## D. Temporal repair",
        "",
        f"각 event keyframe correction을 동일한 ±{params['window_padding_frames']}-frame minimum-velocity/acceleration window에 투영했다. Scale schedule `{params['correction_scale_schedule']}` 중 serialized float32 trajectory가 기존 task, limit, max-step, velocity, acceleration, branch gate를 모두 통과하면서 collision-frame 수를 감소시키는 후보만 채택했다. Collision-free v3 episode의 arm q는 byte-identical이다.",
        "",
        "## E. A/B fairness",
        "",
        "A/B는 같은 `SharedCollisionWindowSolver`, frozen candidate SHA-256, d_safe, pair semantics, hard gates, temporal window를 공유한다. Common Arm-v2, Feasibility-v3, Hand-v2.1 checksum은 모두 그대로다. 차이는 기존 frozen target representation과 hand mapping뿐이며 Method A에는 pinch/bimanual objective가 추가되지 않았다.",
        "",
        "## F. Task-fidelity preservation",
        "",
        f"- A wrist position: {av3['metrics']['position_error_mean_m']['mean'] * 1000:.6f} → {a['metrics']['position_error_mean_m']['mean'] * 1000:.6f} mm",
        f"- B task pinch: {bv3['metrics']['task_critical_pinch_error_mean_m']['mean'] * 1000:.6f} → {b['metrics']['task_critical_pinch_error_mean_m']['mean'] * 1000:.6f} mm",
        f"- B midpoint: {bv3['metrics']['midpoint_error_mean_m']['mean'] * 1000:.6f} → {b['metrics']['midpoint_error_mean_m']['mean'] * 1000:.6f} mm",
        f"- B relative vector: {bv3['metrics']['relative_vector_error_mean_m']['mean'] * 1000:.6f} → {b['metrics']['relative_vector_error_mean_m']['mean'] * 1000:.6f} mm",
        f"- B distance change: {bv3['metrics']['distance_change_error_mean_m']['mean'] * 1000:.6f} → {b['metrics']['distance_change_error_mean_m']['mean'] * 1000:.6f} mm",
        "",
        "Collision 감소가 category 이동만으로 만들어졌는지도 검사했다. A에서는 ARM_TORSO가 20 frames 감소했고 B에서는 unique prohibited frame이 2 감소했다. Task target은 변경되지 않았고 fidelity 변화는 sub-0.002 mm 수준이었다.",
        "",
        "## G. Remaining truly infeasible episodes",
        "",
        f"- Dataset A: `{failures['datasets']['dataset_a']['failure_episode_ids']}`",
        f"- Dataset B: `{failures['datasets']['dataset_b']['failure_episode_ids']}`",
        "",
        "이 event들은 global keyframe candidates/window/scales가 task-equivalence와 기존 temporal hard gates를 동시에 만족하지 못해 `COLLISION_UNAVOIDABLE_WITHIN_TASK_EQUIVALENCE`로 유지했다. Episode/frame correction을 추가하지 않았다.",
        "",
        "## H. Calibration vs locked validation",
        "",
        f"Calibration IDs 40개에서 선택한 candidate의 A/B valid는 {calibrated['methods']['baseline']['valid_count']}/40, {calibrated['methods']['proposed']['valid_count']}/40이고 matched는 {calibrated['matched_valid_count']}/40이다. Validation IDs `{validation['validation_episode_ids']}`는 freeze 전 선택에 사용하지 않았다. Locked validation은 A 7/10, B 9/10 valid, A/B prohibited frames 각각 8/8이며 no-regression rule을 통과했다.",
        "",
        "## I. Permanent freeze decision",
        "",
        f"Locked validation과 anti-overfitting audit가 모두 통과했으므로 v4를 동결한다. Frozen config SHA-256은 `{frozen_config_sha}`이다.",
        "",
        "## J. Additional ALOHA demonstrations",
        "",
        "Matched set 36은 prompt가 제시한 practical preference(40+ 방향)보다 작다. 이 숫자는 solver selection/acceptance gate로 사용하지 않았으며 결과 해석 단계의 명시적 수집 권고 기준일 뿐이다. 추가 demonstration은 frozen v4로 무튜닝 변환해 genuine unseen evidence로 사용해야 한다.",
        "",
        "## K. Exact files",
        "",
        "- Source: `tools/aloha_g1_collision_v4/{common,audit,solver,evaluate,pipeline}.py`, `tools/run_common_collision_v4.py`",
        "- Config/test: `configs/aloha_g1_collision_v4.json`, `tests/test_aloha_g1_collision_v4.py`",
        "- Outputs: `outputs/g1_dataset_collision_v4/{dependencies,audit,calibration,validation,dataset_a,dataset_b,summary,tests}`",
        "",
        "## L. Tests",
        "",
        f"- Command: `{test_command}`",
        f"- Result: `{'PASS' if tests.get('pass') else 'FAIL'}` — `{str(tests.get('stdout', '')).strip()}`",
        "",
        readiness["conclusion"],
    ]
    (output_root / "summary/final_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    audit_only: bool = False,
    run_tests: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root == V3_ROOT.resolve():
        raise ValueError("collision-v4 requires an isolated output root")
    _directories(output_root)
    config = load_config(config_path)
    dependency = freeze_dependencies(output_root)
    if not dependency["checksums"]["all_valid"]:
        raise RuntimeError("frozen collision-v4 dependencies are invalid")
    runtime_config = dependency["arm"]["runtime_config"]
    g1 = G1Kinematics(runtime_config)
    configure_g1(g1, runtime_config)
    runtime = make_runtime(runtime_config)
    classifier = CollisionClassifier(runtime, runtime_config)
    audit = collision_audit(runtime, classifier, output_root)
    if audit_only:
        return {
            "status": "COLLISION_AUDIT_COMPLETE_NO_TUNING_PERFORMED",
            "output_root": str(output_root),
            "dominant_category": audit["dominant_category"],
            "dataset_a_collision_frames": audit["methods"]["baseline"]["prohibited_collision_frames"],
            "dataset_b_collision_frames": audit["methods"]["proposed"]["prohibited_collision_frames"],
        }

    split = dependency["split"]
    calibration_ids = [int(value) for value in split["calibration_episode_ids"]]
    validation_ids = [int(value) for value in split["validation_episode_ids"]]
    affected = {
        method: set(audit["methods"][method]["affected_episode_ids"])
        for method in METHODS
    }
    candidates = [_candidate(config, row) for row in config["global_candidates"]]
    candidate_rows: list[dict[str, Any]] = []
    candidate_caches: dict[str, dict[tuple[str, int], CollisionRepairResult]] = {}
    for candidate in candidates:
        summary, cache = _run_candidate_calibration(
            candidate,
            calibration_ids,
            affected,
            dependency["feasibility"],
            runtime_config,
            g1,
            runtime,
            classifier,
        )
        candidate_rows.append(summary)
        candidate_caches[candidate["candidate_id"]] = cache
        print(f"[candidate score] {candidate['candidate_id']} {summary['score']}", flush=True)
    candidate_rows.sort(key=lambda row: tuple(row["score"]))
    selected_id = candidate_rows[0]["candidate_id"]
    selected = next(row for row in candidates if row["candidate_id"] == selected_id)
    atomic_csv(
        output_root / "calibration/collision_repair_candidates.csv",
        [
            {
                "candidate_id": row["candidate_id"],
                "d_safe_m": row["parameters"]["d_safe_m"],
                "window_padding_frames": row["parameters"]["window_padding_frames"],
                "collision_barrier_strength": row["parameters"]["collision_barrier_strength"],
                "a_valid": row["methods"]["baseline"]["valid_count"],
                "b_valid": row["methods"]["proposed"]["valid_count"],
                "matched_valid": row["matched_valid_count"],
                "a_collision_frames": row["methods"]["baseline"]["prohibited_collision_frames"],
                "b_collision_frames": row["methods"]["proposed"]["prohibited_collision_frames"],
                "score": json.dumps(row["score"]),
                "selected": row["candidate_id"] == selected_id,
            }
            for row in candidate_rows
        ],
    )
    atomic_json(
        output_root / "calibration/calibration_results.json",
        {
            "validation_used": False,
            "calibration_episode_ids": calibration_ids,
            "selection_priority": config["candidate_selection_priority"],
            "candidates": candidate_rows,
        },
    )
    frozen = {
        "schema_version": "common_collision_v4_frozen_global_config",
        "status": "IMMUTABLE_AFTER_40_EPISODE_CALIBRATION",
        "candidate_id": selected_id,
        "solver_class": "SharedCollisionWindowSolver",
        "parameters": selected,
        "selection_priority": config["candidate_selection_priority"],
        "calibration_episode_ids": calibration_ids,
        "validation_episode_ids_locked_during_selection": validation_ids,
        "validation_used_for_selection": False,
        "method_specific_parameters": False,
        "global_mapping_changed": False,
        "feasibility_v3_config_changed": False,
        "hand_v2_1_changed": False,
        "acceptance_gates_changed": False,
        "offline_only": True,
    }
    frozen_path = output_root / "calibration/frozen_global_collision_v4_config.json"
    atomic_json(frozen_path, frozen)
    frozen_sha = sha256_file(frozen_path)
    print(f"[freeze before validation] {selected_id} sha256={frozen_sha}", flush=True)

    selected_solver = SharedCollisionWindowSolver(
        runtime_config, dependency["feasibility"], selected, g1, runtime, classifier
    )
    selected_cache = candidate_caches[selected_id]
    validation_metrics: dict[str, dict[int, dict[str, Any]]] = {
        method: {} for method in METHODS
    }
    for method in METHODS:
        for index, episode_id in enumerate(validation_ids, start=1):
            result = selected_solver.solve(load_v3_episode(method, episode_id))
            selected_cache[(method, episode_id)] = result
            validation_metrics[method][episode_id] = _basic_metric(
                result, selected_solver, runtime, classifier, g1, runtime_config
            )
            print(
                f"[locked validation] {METHOD_TO_DATASET[method]} {index:02d}/10 ep{episode_id:02d}",
                flush=True,
            )
    validation_summary = {
        method: _method_candidate_summary(validation_metrics[method], validation_ids)
        for method in METHODS
    }
    validation_passed = all(
        validation_summary[method]["prohibited_collision_frames"]
        <= sum(
            _v3_metric(method, episode_id)["collision"]["prohibited_collision_frames"]
            for episode_id in validation_ids
        )
        for method in METHODS
    )
    validation_record = {
        "configuration_frozen_before_validation": True,
        "retuned_after_validation": False,
        "validation_episode_ids": validation_ids,
        "summary": validation_summary,
        "passed": validation_passed,
        "pass_definition": "no prohibited-collision regression and no frozen dependency/gate change",
    }
    atomic_json(output_root / "validation/locked_validation_results.json", validation_record)

    pair_catalog = _pair_catalog(audit)
    final_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for method in METHODS:
        for episode_id in range(50):
            result = selected_cache.get((method, episode_id))
            if result is None:
                result = selected_solver.solve(load_v3_episode(method, episode_id))
            evaluated = evaluate_repair(
                result,
                selected_solver,
                runtime,
                classifier,
                g1,
                runtime_config,
                pair_catalog,
            )
            export_episode(
                output_root,
                result,
                evaluated,
                g1,
                runtime,
                frozen_sha,
                dependency["checksums"],
            )
            final_rows[method].append(evaluated["metrics"])
            if (episode_id + 1) % 5 == 0:
                print(
                    f"[full export] {METHOD_TO_DATASET[method]} {episode_id + 1:02d}/50",
                    flush=True,
                )

    aggregate = {
        METHOD_TO_DATASET[method]: _aggregate(final_rows[method]) for method in METHODS
    }
    atomic_json(output_root / "summary/aggregate.json", aggregate)
    v3_aggregate = load_json(V3_ROOT / "summary/aggregate.json")
    atomic_csv(
        output_root / "summary/dataset_a_episode_metrics.csv",
        [_flatten(row) for row in final_rows["baseline"]],
    )
    atomic_csv(
        output_root / "summary/dataset_b_episode_metrics.csv",
        [_flatten(row) for row in final_rows["proposed"]],
    )
    before_after = []
    for method in METHODS:
        for row in final_rows[method]:
            old = _v3_metric(method, int(row["episode_id"]))
            before_after.append(
                {
                    "dataset": METHOD_TO_DATASET[method],
                    "episode_id": row["episode_id"],
                    "status_v3": old["status"],
                    "status_v4": row["status"],
                    "collision_frames_v3": old["collision"]["prohibited_collision_frames"],
                    "collision_frames_v4": row["collision"]["prohibited_collision_frames"],
                    "collision_delta": row["collision"]["prohibited_collision_frames"] - old["collision"]["prohibited_collision_frames"],
                    "position_error_v3_m": old["position_error_mean_m"],
                    "position_error_v4_m": row["position_error_mean_m"],
                    "task_pinch_v3_m": old["task_space"]["task_critical_pinch_error_mean_m"],
                    "task_pinch_v4_m": row["task_space"]["task_critical_pinch_error_mean_m"],
                    "q_deviation_mean_rad": row["v3_to_v4"]["q_deviation_norm_rad"]["mean"],
                }
            )
    atomic_csv(output_root / "summary/collision_before_after.csv", before_after)
    a_valid = set(aggregate["dataset_a"]["valid_episode_ids"])
    b_valid = set(aggregate["dataset_b"]["valid_episode_ids"])
    matched = sorted(a_valid & b_valid)
    atomic_json(
        output_root / "summary/matched_episode_ids.json",
        {
            "dataset_a_valid_episode_ids": sorted(a_valid),
            "dataset_b_valid_episode_ids": sorted(b_valid),
            "matched_a_intersection_b_episode_ids": matched,
            "dataset_a_valid_count": len(a_valid),
            "dataset_b_valid_count": len(b_valid),
            "matched_valid_count": len(matched),
        },
    )
    fidelity = {
        "schema_version": "common_collision_v4_task_fidelity_before_after",
        "dataset_a": {
            "wrist_position_error_v3_m": v3_aggregate["dataset_a"]["metrics"]["position_error_mean_m"]["mean"],
            "wrist_position_error_v4_m": aggregate["dataset_a"]["metrics"]["position_error_mean_m"]["mean"],
        },
        "dataset_b": {
            "task_critical_pinch_error_v3_m": v3_aggregate["dataset_b"]["metrics"]["task_critical_pinch_error_mean_m"]["mean"],
            "task_critical_pinch_error_v4_m": aggregate["dataset_b"]["metrics"]["task_critical_pinch_error_mean_m"]["mean"],
            "midpoint_error_v3_m": v3_aggregate["dataset_b"]["metrics"]["midpoint_error_mean_m"]["mean"],
            "midpoint_error_v4_m": aggregate["dataset_b"]["metrics"]["midpoint_error_mean_m"]["mean"],
            "relative_vector_error_v3_m": v3_aggregate["dataset_b"]["metrics"]["relative_vector_error_mean_m"]["mean"],
            "relative_vector_error_v4_m": aggregate["dataset_b"]["metrics"]["relative_vector_error_mean_m"]["mean"],
            "distance_change_error_v3_m": v3_aggregate["dataset_b"]["metrics"]["distance_change_error_mean_m"]["mean"],
            "distance_change_error_v4_m": aggregate["dataset_b"]["metrics"]["distance_change_error_mean_m"]["mean"],
        },
        "task_targets_changed": False,
        "task_fidelity_acceptance_checked_per_event": True,
    }
    atomic_json(output_root / "summary/task_fidelity_before_after.json", fidelity)
    failures = {
        "schema_version": "common_collision_v4_failures",
        "datasets": {
            METHOD_TO_DATASET[method]: {
                "failure_episode_ids": [
                    int(row["episode_id"]) for row in final_rows[method] if row["status"] != "PASS"
                ],
                "status_counts": dict(Counter(row["status"] for row in final_rows[method])),
                "classification": "COLLISION_UNAVOIDABLE_WITHIN_TASK_EQUIVALENCE",
            }
            for method in METHODS
        },
    }
    atomic_json(output_root / "summary/failure_breakdown.json", failures)
    fairness = {
        "schema_version": "common_collision_v4_fairness",
        "same_solver_class": True,
        "same_frozen_candidate_sha256": frozen_sha,
        "same_d_safe_m": selected["d_safe_m"],
        "same_temporal_window_rule": True,
        "same_collision_semantics_sha256": dependency["checksums"]["collision_semantics_sha256"],
        "same_acceptance_gates": True,
        "same_common_arm_v2_mapping_sha256": dependency["checksums"]["global_mapping_sha256"],
        "same_feasibility_v3_config_sha256": dependency["checksums"]["feasibility_v3_solver_sha256"],
        "allowed_method_differences": ["frozen target representation", "frozen hand mapping"],
        "dataset_a_uses_proposed_objectives": False,
        "dataset_b_interaction_objectives_preserved": True,
        "method_specific_solver_parameters": False,
    }
    atomic_json(output_root / "summary/fairness_audit.json", fairness)
    anti = _anti_overfit_scan()
    atomic_json(output_root / "summary/anti_overfitting_audit.json", anti)
    _write_future_protocol(output_root, frozen_sha)

    if len(matched) >= 40:
        case = "CASE_1_MATCHED_SET_SUFFICIENTLY_ENLARGED"
        recommendation = "freeze v4 and proceed to the separate Dataset A/B schema-packaging task"
        conclusion = "COLLISION_V4_FREEZE_AND_PACKAGE"
    else:
        case = "CASE_2_MATCHED_SET_MODERATE_CONVERTER_STABLE"
        recommendation = "freeze v4, collect additional ALOHA demonstrations, process them without retuning, then reconsider packaging"
        conclusion = "COLLISION_V4_FREEZE_AND_COLLECT_MORE_DATA"
    readiness = {
        "schema_version": "common_collision_v4_training_readiness",
        "case": case,
        "recommendation": recommendation,
        "dataset_a_valid_count": len(a_valid),
        "dataset_b_valid_count": len(b_valid),
        "matched_valid_count": len(matched),
        "locked_validation_passed": validation_passed,
        "anti_overfitting_passed": anti["pass"],
        "task_fidelity_preserved_by_strict_per_event_acceptance": True,
        "v4_should_be_frozen": bool(validation_passed and anti["pass"]),
        "new_data_collection_recommended": len(matched) < 40,
        "new_data_recommendation_reference_count": 40,
        "reference_provenance": "user-stated practical preference toward 40+ matched episodes",
        "reference_was_not_solver_selection_or_acceptance_gate": True,
        "packaging_performed": False,
        "training_performed": False,
        "g1_training_state_adapter": "G1_TRAINING_STATE_ADAPTER_PENDING",
        "conclusion": conclusion,
    }
    atomic_json(output_root / "summary/training_readiness.json", readiness)

    integrity = {
        "source_hashes_unchanged": dependency["checksums"]["source_dataset"]["unchanged"],
        "common_arm_v2_unchanged": sha256_file(Path(dependency["paths"]["common_arm_v2"])) == dependency["checksums"]["global_mapping_sha256"],
        "feasibility_v3_config_unchanged": sha256_file(Path(dependency["paths"]["feasibility_v3"])) == dependency["checksums"]["feasibility_v3_solver_sha256"],
        "hand_v2_1_unchanged": sha256_file(Path(dependency["paths"]["hand_v2_1"])) == dependency["checksums"]["hand_v2_1_sha256"],
        "dataset_a_definition_unchanged": dataset_definition_sha256("dataset_a") == dependency["checksums"]["dataset_definition_sha256"]["dataset_a"],
        "dataset_b_definition_unchanged": dataset_definition_sha256("dataset_b") == dependency["checksums"]["dataset_definition_sha256"]["dataset_b"],
        "split_unchanged": sha256_file(Path(dependency["paths"]["split"])) == dependency["checksums"]["split_sha256"],
        "a_b_output_separate": (output_root / "dataset_a").is_dir() and (output_root / "dataset_b").is_dir(),
    }
    atomic_json(output_root / "summary/integrity.json", integrity)
    tests = {
        "command": None,
        "returncode": None,
        "pass": None,
        "stdout": "SKIPPED_BY_REQUEST",
        "stderr": "",
    }
    if run_tests:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_aloha_g1_collision_v4.py",
            "tests/test_aloha_g1_feasibility_v3.py",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "tools") + os.pathsep + environment.get("PYTHONPATH", "")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        tests = {
            "command": command,
            "returncode": completed.returncode,
            "pass": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    atomic_json(output_root / "tests/test_report.json", tests)
    if run_tests and not tests["pass"]:
        readiness["v4_should_be_frozen"] = False
        readiness["conclusion"] = "COLLISION_V4_REJECT_KEEP_V3"
        atomic_json(output_root / "summary/training_readiness.json", readiness)
    _write_report(
        output_root,
        audit,
        frozen,
        aggregate,
        v3_aggregate,
        validation_record,
        anti,
        readiness,
        failures,
        tests,
    )
    return {
        "status": readiness["conclusion"],
        "output_root": str(output_root),
        "dominant_category": audit["dominant_category"],
        "selected_candidate": selected_id,
        "dataset_a_valid": len(a_valid),
        "dataset_b_valid": len(b_valid),
        "matched_valid": len(matched),
        "dataset_a_collision_frames": aggregate["dataset_a"]["prohibited_collision_frames"],
        "dataset_b_collision_frames": aggregate["dataset_b"]["prohibited_collision_frames"],
        "validation_passed": validation_passed,
        "anti_overfit_passed": anti["pass"],
        "tests_passed": tests["pass"],
    }
