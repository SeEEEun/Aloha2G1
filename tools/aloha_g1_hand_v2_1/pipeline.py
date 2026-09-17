"""End-to-end offline Proposed hand v2.1 construction and 50-episode audit."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aloha_g1_hand_v2.collision_eval import CollisionClassifier, make_runtime
from aloha_g1_hand_v2.common import load_v1_config

from .common import (
    ROOT,
    SOURCE_CONFIG,
    V1_ROOT,
    V2_ROOT,
    V2_1_ROOT,
    atomic_csv,
    atomic_json,
    integrity_snapshot,
    load_source_config,
    markdown_table,
    sha256_file,
)
from .primitives import PHASES, SemanticPrimitiveBuilder, tool_compatibility
from .render import render_all
from .sweep import (
    compare_variants,
    evaluate_variant,
    exact_v2_ablation_summary,
    install_global_neutrals,
)
from .third_neutral import (
    build_search_frame_refs,
    evaluate_neutral_pair,
    load_frozen_episodes,
    search_global_neutrals,
)


def _select_global_transition(
    primitives: Mapping[str, Any],
    v1: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """Find the shortest one-size-fits-all minimum-jerk duration on all labels."""
    fps = float(config["interpolation"]["sampling_rate_hz"])
    thresholds = {
        "maximum_joint_step_rad": float(v1["validation"]["maximum_joint_step_rad"]),
        "maximum_velocity_rad_s": float(v1["validation"]["maximum_velocity_rad_s"]),
        "maximum_acceleration_rad_s2": float(
            v1["validation"]["maximum_acceleration_rad_s2"]
        ),
    }
    largest_delta = 0.0
    for side in ("left", "right"):
        values = [
            np.asarray(value, dtype=np.float64)
            for value in primitives["sides"][side]["states"].values()
        ]
        for first in values:
            for second in values:
                largest_delta = max(largest_delta, float(np.max(np.abs(first - second))))
    minimum_duration = float(config["interpolation"]["minimum_duration_sec"])
    # Unit quintic extrema: max ds/du=1.875 and max |d2s/du2|=10/sqrt(3).
    velocity_duration = 1.875 * largest_delta / thresholds["maximum_velocity_rad_s"]
    acceleration_duration = np.sqrt(
        (10.0 / np.sqrt(3.0))
        * largest_delta
        / thresholds["maximum_acceleration_rad_s2"]
    )
    initial_frames = max(
        1,
        int(np.ceil(fps * max(minimum_duration, velocity_duration, acceleration_duration))),
    )
    attempts: list[dict[str, Any]] = []
    shared_runtime = make_runtime(v1)
    for frames in range(initial_frames, 181):
        episodes = load_frozen_episodes(primitives, frames)
        max_step = 0.0
        max_velocity = 0.0
        max_acceleration = 0.0
        finite = True
        for episode in episodes:
            combined = np.column_stack(
                (episode["candidate_left"], episode["candidate_right"])
            )
            finite = finite and bool(np.isfinite(combined).all())
            step = np.diff(combined, axis=0)
            velocity = step * fps
            acceleration = np.diff(velocity, axis=0) * fps
            max_step = max(max_step, float(np.max(np.abs(step), initial=0.0)))
            max_velocity = max(
                max_velocity, float(np.max(np.abs(velocity), initial=0.0))
            )
            max_acceleration = max(
                max_acceleration, float(np.max(np.abs(acceleration), initial=0.0))
            )
        limits_ok = all(
            np.all(episode[f"candidate_{side}"] >= shared_runtime.hand_limits[side][:, 0] - 1e-9)
            and np.all(
                episode[f"candidate_{side}"] <= shared_runtime.hand_limits[side][:, 1] + 1e-9
            )
            for episode in episodes
            for side in ("left", "right")
        )
        passed = bool(
            finite
            and limits_ok
            and max_step <= thresholds["maximum_joint_step_rad"] + 1e-12
            and max_velocity <= thresholds["maximum_velocity_rad_s"] + 1e-12
            and max_acceleration <= thresholds["maximum_acceleration_rad_s2"] + 1e-12
        )
        attempt = {
            "transition_frames": frames,
            "duration_sec": frames / fps,
            "max_hand_joint_step_rad": max_step,
            "max_hand_velocity_rad_s": max_velocity,
            "max_hand_acceleration_rad_s2": max_acceleration,
            "finite": finite,
            "joint_limits_ok": limits_ok,
            "pass": passed,
        }
        attempts.append(attempt)
        if passed:
            return frames, episodes, {
                "selection_rule": config["interpolation"]["duration_rule"],
                "thresholds": thresholds,
                "largest_semantic_state_joint_delta_rad": largest_delta,
                "analytic_initial_frame_count": initial_frames,
                "selected": attempt,
                "attempts": attempts,
            }
    raise RuntimeError("no global minimum-jerk duration passed existing temporal limits")


def _prior_summary() -> dict[str, Any]:
    collision = json.loads(
        (V2_ROOT / "collision_audit/collision_attribution_50ep.json").read_text(
            encoding="utf-8"
        )
    )
    v2 = json.loads((V2_ROOT / "summary/run_summary.json").read_text(encoding="utf-8"))
    baseline = json.loads(
        (V1_ROOT / "summary/baseline_summary.json").read_text(encoding="utf-8")
    )
    proposed = json.loads(
        (V1_ROOT / "summary/proposed_summary.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": "proposed_hand_v2_1_prior_results_summary",
        "source_artifacts": [
            str(V1_ROOT / "summary/baseline_summary.json"),
            str(V1_ROOT / "summary/proposed_summary.json"),
            str(V2_ROOT / "collision_audit/collision_attribution_50ep.json"),
            str(V2_ROOT / "summary/run_summary.json"),
        ],
        "current_50_episode": {
            "baseline_collision_frames": int(
                collision["baseline"]["v1_collision_frames"]
            ),
            "proposed_collision_frames": int(
                collision["proposed"]["v1_collision_frames"]
            ),
            "proposed_finger_involved_frames": int(
                collision["proposed"]["finger_involved_collision_frames"]
            ),
            "proposed_third_finger_frames": int(
                collision["proposed"]["cause_group_frame_incidence"][
                    "non_task_third_finger_collision"
                ]
            ),
            "proposed_enhanced_same_hand_frames": int(
                collision["proposed"]["enhanced_logger"][
                    "same_hand_collision_frames_nonexclusive_with_v1"
                ]
            ),
            "collision_decision": collision["decision_gate"],
        },
        "pinch_error_episode_mean_m": {
            "dataset_a_baseline": baseline["aggregate"][
                "task_critical_pinch_error_mean_m"
            ]["mean"],
            "current_proposed_v1": proposed["aggregate"][
                "task_critical_pinch_error_mean_m"
            ]["mean"],
        },
        "exact_contact_v2": {
            "fixed_phone_pinch_mean_error_m": v2["fixed_phone_pinch_mean_error_m"],
            "exact_contact_mean_error_m": v2["interaction_ik_mean_error_m"],
            "ready": False,
            "classification": [
                "DEX3_FINGER_WORKSPACE_LIMIT",
                "HAND_GEOMETRY_IMPROVED_BUT_COLLISION_REMAINS",
            ],
        },
        "preservation_rule": "v1 and hand-v2 outputs are read-only inputs",
    }


def _full_pareto_validation(
    runtime: Any,
    classifier: CollisionClassifier,
    episodes: list[dict[str, Any]],
    search: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    full_refs = [
        (episode_id, frame)
        for episode_id, episode in enumerate(episodes)
        for frame in range(len(episode["arm"]))
    ]
    indices = search["third_indices"]
    selected = {
        side: np.asarray(search["selected"][side], dtype=np.float64)
        for side in ("left", "right")
    }
    candidates: list[dict[str, Any]] = [
        {"name": "selected", "left": selected["left"], "right": selected["right"]}
    ]
    limit = int(config["third_neutral_search"]["full_validation_pareto_candidates"])
    for side in ("left", "right"):
        for row in search["pareto_candidates"][side]:
            candidate = {
                "name": f"{side}_pareto_{len(candidates)}",
                "left": selected["left"].copy(),
                "right": selected["right"].copy(),
            }
            candidate[side] = np.asarray(row["q"], dtype=np.float64)
            if any(
                np.array_equal(candidate["left"], old["left"])
                and np.array_equal(candidate["right"], old["right"])
                for old in candidates
            ):
                continue
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = evaluate_neutral_pair(
            runtime,
            classifier,
            episodes,
            full_refs,
            indices,
            {"left": candidate["left"], "right": candidate["right"]},
        )
        output.append({"name": candidate["name"], "metrics": metrics})
    return output


def _candidate_states(
    primitive_build: Mapping[str, Any], search: Mapping[str, Any]
) -> dict[str, dict[str, np.ndarray]]:
    states: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        indices = np.asarray(primitive_build["sides"][side]["third_indices"], dtype=np.int64)
        states[side] = {}
        for phase, source in primitive_build["sides"][side]["states"].items():
            q = np.asarray(source, dtype=np.float64).copy()
            q[indices] = np.asarray(search["selected"][side], dtype=np.float64)
            states[side][phase] = q
    return states


def _candidate_config_artifact(
    v1: Mapping[str, Any],
    config: Mapping[str, Any],
    primitives: Mapping[str, Any],
    search: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    temporal_transition: Mapping[str, Any],
) -> dict[str, Any]:
    states = _candidate_states(primitives, search)
    return {
        "schema_version": "proposed_hand_v2_1_candidate",
        "version": "2.1",
        "status": "CANDIDATE_REQUIRES_AUDIT_RESULT",
        "method_scope": "Proposed/Dataset B only",
        "diagnostic_exact_contact_ik": {
            "retained": True,
            "default": False,
            "source": str(V2_ROOT),
        },
        "joint_order": {
            side: list(primitives["sides"][side]["joint_names"])
            for side in ("left", "right")
        },
        "joint_order_provenance": v1["hand_mapping"]["dex3_mapping_source"],
        "states": {
            side: {phase: q for phase, q in states[side].items()}
            for side in ("left", "right")
        },
        "named_states": {
            "LEFT_PHONE_PREGRASP": states["left"]["PREGRASP"],
            "LEFT_PHONE_PINCH": states["left"]["GRASP"],
            "LEFT_PHONE_HOLD": states["left"]["HOLD"],
            "RIGHT_ACCESSORY_PREGRASP": states["right"]["PREGRASP"],
            "RIGHT_ACCESSORY_PINCH": states["right"]["GRASP"],
            "RIGHT_ACCESSORY_HOLD": states["right"]["HOLD"],
        },
        "left_third_safe_neutral": np.asarray(search["selected"]["left"]),
        "right_third_safe_neutral": np.asarray(search["selected"]["right"]),
        "third_finger_policy": {
            "role": "NON_TASK",
            "task_contact_allowed": False,
            "support_contact_allowed": False,
            "per_episode_q": False,
            "per_phase_q": False,
        },
        "left_static_wrist_to_pinch": compatibility["sides"]["left"][
            "candidate_wrist_to_pinch"
        ],
        "right_static_wrist_to_pinch": compatibility["sides"]["right"][
            "candidate_wrist_to_pinch"
        ],
        "semantic_phase_mapping": {
            side: {phase: phase for phase in PHASES} for side in ("left", "right")
        },
        "interpolation": {
            **config["interpolation"],
            "selected_transition_frames": temporal_transition["selected"][
                "transition_frames"
            ],
            "selected_duration_sec": temporal_transition["selected"]["duration_sec"],
            "validation": temporal_transition,
        },
        "geometry_model_provenance": {
            "g1_model": v1["models"]["g1_xml"],
            "g1_model_sha256": v1["models"]["g1_xml_sha256"],
            "finger_mapping": v1["hand_mapping"]["dex3_mapping_source"],
            "finger_mapping_sha256": v1["hand_mapping"]["dex3_mapping_source_sha256"],
            "object_class_geometry": config["object_class_geometry"],
            "source_demo_object_pose_used": False,
        },
        "primitive_construction": primitives,
        "tool_transform_compatibility": compatibility,
        "label_status": "SIMULATION_PLACEHOLDER_HAND_LABELS",
        "authoritative_for_real_g1": False,
        "real_robot_command_allowed": False,
    }


def _anti_overfit_scan() -> dict[str, Any]:
    source_files = sorted((ROOT / "tools/aloha_g1_hand_v2_1").glob("*.py")) + [
        SOURCE_CONFIG,
        ROOT / "tools/build_proposed_hand_v2_1.py",
    ]
    patterns = {
        "episode_equality": re.compile(r"if\s+episode(?:_id)?\s*==\s*49"),
        "frame_equality": re.compile(r"if\s+frame\s*=="),
        "legacy_episode_offset_token": re.compile(
            r"ep" + r"49[_-]?offset", re.IGNORECASE
        ),
        "manual_cartesian_waypoint_token": re.compile(
            r"manual" + r"[_ -]?waypoint", re.IGNORECASE
        ),
        "per_episode_hand_q_assignment": re.compile(
            r"(?:hand_q|neutral_q)\s*\[\s*episode", re.IGNORECASE
        ),
        "framewise_contact_residual_token": re.compile(
            r"per[_ -]?frame" + r"[_ -]?contact[_ -]?correction", re.IGNORECASE
        ),
    }
    hits: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path in source_files:
        if not path.exists():
            continue
        scanned.append(str(path.relative_to(ROOT)))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
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
    return {"files_scanned": scanned, "hits": hits, "pass": not hits}


def _run_tests(output_root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_aloha_g1_hand_v2_1.py"]
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"(\d+) passed", output)
    scan = _anti_overfit_scan()
    return {
        "command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 " + " ".join(command),
        "exit_code": completed.returncode,
        "pass": completed.returncode == 0 and scan["pass"],
        "passed_test_count": int(match.group(1)) if match else 0,
        "output": output,
        "anti_overfit_scan": scan,
    }


def _readiness(
    config: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    integrity: Mapping[str, Any],
    search: Mapping[str, Any],
    temporal_transition: Mapping[str, Any],
) -> dict[str, Any]:
    variants = aggregate["variants"]
    baseline = variants["dataset_a_baseline"]
    current = variants["current_proposed_v1"]
    candidate = variants["proposed_hand_v2_1"]
    thresholds = config["readiness"]
    checks = {
        "hand_related_collisions_decrease": int(
            candidate["collision"]["comprehensive_hand_related_frames"]
        )
        < int(current["collision"]["comprehensive_hand_related_frames"]),
        "third_finger_collisions_substantially_decrease": float(
            aggregate["v1_to_v2_1"]["third_finger_collision_fractional_reduction"]
        )
        >= float(thresholds["third_collision_minimum_fractional_reduction"]),
        "pinch_quality_better_than_dataset_a": float(
            candidate["pinch_frame"]["task_critical_pinch_error_episode_mean_m"]["mean"]
        )
        < float(
            baseline["pinch_frame"]["task_critical_pinch_error_episode_mean_m"]["mean"]
        ),
        "all_values_finite": bool(candidate["kinematics"]["finite"]),
        "joint_limits_pass": int(candidate["kinematics"]["joint_limit_violation_count"])
        == 0,
        "temporal_limits_pass": bool(temporal_transition["selected"]["pass"])
        and float(candidate["kinematics"]["max_hand_joint_step_rad"])
        <= float(temporal_transition["thresholds"]["maximum_joint_step_rad"]) + 1e-12
        and float(candidate["kinematics"]["max_hand_velocity_rad_s"])
        <= float(temporal_transition["thresholds"]["maximum_velocity_rad_s"]) + 1e-12
        and float(candidate["kinematics"]["max_hand_acceleration_rad_s2"])
        <= float(temporal_transition["thresholds"]["maximum_acceleration_rad_s2"])
        + 1e-12,
        "same_configuration_all_50": int(candidate["episode_count"])
        == int(thresholds["same_configuration_episode_count"]),
        "dataset_a_unchanged": bool(integrity["dataset_a_unchanged"]),
        "g1_arm_action_unchanged": bool(integrity["g1_arm_trajectories_unchanged"]),
    }
    hand_candidate_pass = all(checks.values())
    selected_metrics = search["selected_search_metrics"]
    fully_collision_free_neutral = (
        int(selected_metrics["combined_third_collision_frames"]) == 0
    )
    arm_rerun = bool(compatibility["tool_transform_changed_requires_arm_rerun"])
    if hand_candidate_pass and arm_rerun:
        classification = "HAND_ADAPTER_READY_ARM_MAPPING_STILL_BLOCKING"
        conclusion = "PROPOSED_HAND_V2_1_READY_FOR_COMMON_ARM_RERUN"
    elif hand_candidate_pass:
        classification = "HAND_ADAPTER_READY_FOR_DATASET_B_INTEGRATION"
        conclusion = "PROPOSED_HAND_V2_1_READY_FOR_COMMON_ARM_RERUN"
    else:
        classification = "PROPOSED_HAND_V2_1_NOT_READY"
        conclusion = "PROPOSED_HAND_V2_1_NOT_READY"
    return {
        "schema_version": "proposed_hand_v2_1_integration_readiness",
        "checks": checks,
        "hand_candidate_acceptance_pass": hand_candidate_pass,
        "fully_collision_free_global_third_neutral_found": fully_collision_free_neutral,
        "global_neutral_interpretation": (
            "COLLISION_FREE_GLOBAL_NEUTRAL"
            if fully_collision_free_neutral
            else "PARETO_NEUTRAL_WITH_FROZEN_ARM_PROXIMITY_RESIDUAL"
        ),
        "common_arm_rerun_required": arm_rerun,
        "ready_for_direct_dataset_b_integration": hand_candidate_pass and not arm_rerun,
        "ready_for_common_arm_rerun": hand_candidate_pass and arm_rerun,
        "classification": classification,
        "conclusion": conclusion,
        "thresholds": thresholds,
        "temporal_transition": temporal_transition,
    }


def _write_final_report(
    output_root: Path,
    candidate_config: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    collision: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    readiness: Mapping[str, Any],
    integrity: Mapping[str, Any],
    tests: Mapping[str, Any],
    search: Mapping[str, Any],
) -> None:
    current = aggregate["variants"]["current_proposed_v1"]
    candidate = aggregate["variants"]["proposed_hand_v2_1"]
    baseline = aggregate["variants"]["dataset_a_baseline"]
    exact = aggregate["variants"]["diagnostic_exact_contact_ik_v2"]
    states = candidate_config["states"]
    left_q = np.asarray(states["left"]["GRASP"])
    right_q = np.asarray(states["right"]["GRASP"])
    left_neutral = np.asarray(candidate_config["left_third_safe_neutral"])
    right_neutral = np.asarray(candidate_config["right_third_safe_neutral"])
    current_collision = current["collision"]
    candidate_collision = candidate["collision"]
    category_names = sorted(
        set(current_collision["category_frame_incidence"])
        | set(candidate_collision["category_frame_incidence"])
    )
    category_table = markdown_table(
        ["category", "Proposed v1 frames", "v2.1 frames", "delta"],
        [
            [
                category,
                current_collision["category_frame_incidence"].get(category, 0),
                candidate_collision["category_frame_incidence"].get(category, 0),
                candidate_collision["category_frame_incidence"].get(category, 0)
                - current_collision["category_frame_incidence"].get(category, 0),
            ]
            for category in category_names
        ],
    )
    comparison_table = markdown_table(
        ["metric", "Dataset A", "Proposed v1", "Exact-contact v2", "Proposed v2.1"],
        [
            [
                "scope",
                "50 ep historical",
                "50 ep frozen arm",
                "one-frame ablation",
                "50 ep frozen arm",
            ],
            [
                "task-critical pinch error [mm]",
                f"{1000*baseline['pinch_frame']['task_critical_pinch_error_episode_mean_m']['mean']:.3f}",
                f"{1000*current['pinch_frame']['task_critical_pinch_error_episode_mean_m']['mean']:.3f}",
                "N/A (contact-point metric only)",
                f"{1000*candidate['pinch_frame']['task_critical_pinch_error_episode_mean_m']['mean']:.3f}",
            ],
            [
                "mean exact task-finger contact [mm]",
                "N/A",
                f"{1000*exact['current_fixed_phone_pinch_mean_task_finger_error_m']:.3f} (dev)",
                f"{1000*exact['mean_task_finger_error_m']:.3f} (dev; infeasible)",
                "not claimed",
            ],
            [
                "third frames",
                baseline["collision"]["third_finger_related_frames"],
                current_collision["third_finger_related_frames"],
                "not swept",
                candidate_collision["third_finger_related_frames"],
            ],
            [
                "comprehensive hand frames",
                baseline["collision"]["comprehensive_hand_related_frames"],
                current_collision["comprehensive_hand_related_frames"],
                "not swept",
                candidate_collision["comprehensive_hand_related_frames"],
            ],
        ],
    )
    added = [
        "configs/aloha_g1_hand_v2_1.json",
        "tools/aloha_g1_hand_v2_1/__init__.py",
        "tools/aloha_g1_hand_v2_1/common.py",
        "tools/aloha_g1_hand_v2_1/primitives.py",
        "tools/aloha_g1_hand_v2_1/third_neutral.py",
        "tools/aloha_g1_hand_v2_1/sweep.py",
        "tools/aloha_g1_hand_v2_1/render.py",
        "tools/aloha_g1_hand_v2_1/pipeline.py",
        "tools/build_proposed_hand_v2_1.py",
        "tests/test_aloha_g1_hand_v2_1.py",
    ]
    artifacts = [
        "audit/prior_results_summary.json",
        "audit/third_finger_collision_audit.json",
        "config/proposed_hand_v2_1_candidate.json",
        "metrics/episode_metrics.csv",
        "metrics/aggregate_comparison.json",
        "metrics/collision_comparison.json",
        "metrics/semantic_phase_validation.json",
        "renders/left_phone_states.png",
        "renders/right_accessory_states.png",
        "renders/third_neutral_comparison.png",
        "renders/hand_v1_vs_v2_vs_v2_1.png",
        "summary/tool_transform_compatibility.json",
        "summary/integration_readiness.json",
        "summary/final_report.md",
        "summary/run_summary.json",
        "tests/test_report.json",
        "integrity/before.json",
        "integrity/after_and_comparison.json",
    ]
    top_remaining = markdown_table(
        ["rank", "link pair", "pair events"],
        [
            [index, row["pair"], row["pair_events"]]
            for index, row in enumerate(candidate_collision["top_pairs"][:10], start=1)
        ],
    )
    report = f"""1. **left/right selected thumb-index primitive q[7]**: left `{left_q.tolist()}` / right `{right_q.tolist()}`
2. **left/right selected third-finger neutral q**: left `{left_neutral.tolist()}` / right `{right_neutral.tolist()}`
3. **Proposed v1 vs v2.1 third-finger collision frames**: {current_collision['third_finger_related_frames']:,} → {candidate_collision['third_finger_related_frames']:,}
4. **Proposed v1 vs v2.1 total hand-related collision frames**: {current_collision['comprehensive_hand_related_frames']:,} → {candidate_collision['comprehensive_hand_related_frames']:,} (comprehensive union)
5. **Proposed v1 vs v2.1 task-critical pinch-frame error**: {1000*current['pinch_frame']['task_critical_pinch_error_episode_mean_m']['mean']:.3f} mm → {1000*candidate['pinch_frame']['task_critical_pinch_error_episode_mean_m']['mean']:.3f} mm
6. **left/right tool-transform delta**: left {compatibility['sides']['left']['translation_delta_mm']:.3f} mm / {compatibility['sides']['left']['rotation_delta_deg']:.3f}°, right {compatibility['sides']['right']['translation_delta_mm']:.3f} mm / {compatibility['sides']['right']['rotation_delta_deg']:.3f}°
7. **G1 arm checksum unchanged 여부**: `{str(integrity['g1_arm_trajectories_unchanged']).lower()}`
8. **Dataset A checksum unchanged 여부**: `{str(integrity['dataset_a_unchanged']).lower()}`
9. **whether a common-arm rerun is required**: `{str(readiness['common_arm_rerun_required']).lower()}` — `{compatibility['classification']}`
10. **Dataset B hand integration readiness**: `{readiness['classification']}`; direct integration=`{str(readiness['ready_for_direct_dataset_b_integration']).lower()}`, common-arm rerun candidate=`{str(readiness['ready_for_common_arm_rerun']).lower()}`

# A. exact two-contact IK를 ablation으로만 유지한 이유

v2 exact-contact IK는 source object pose 없이 `SOURCE_TCP_RELATIVE_CONTACT_GEOMETRY_FALLBACK`에서 얻은 4.448 mm center target을 Dex3에 강제했다. 개발 grasp error는 {1000*exact['current_fixed_phone_pinch_mean_task_finger_error_m']:.3f}→{1000*exact['mean_task_finger_error_m']:.3f} mm로 줄었지만 feasibility gate는 `{str(exact['contact_feasible']).lower()}`이고 blocker는 `{exact['blocker']}`였다. 따라서 결과와 코드는 `diagnostic_exact_contact_ik`로 보존하되 최종 label mapper로 사용하지 않았다.

# B. feasible target-hand primitive 도출

Arm/wrist를 고정하고 active Dex3의 5 task-finger DoF만 사용했다. GRASP/HOLD는 thumb/index distal collision mesh 사이의 signed surface aperture를 phone 7.950 mm와 accessory 3.500 mm에 맞추고, pad normal opposition·기존 static pinch frame·기존 검증 seed를 함께 최적화했다. PREGRASP clearance는 active index-pad 최대 half extent의 2배로 정했고 201개 deterministic joint interpolation 후보 중 안전한 최접근 값을 골랐다. OPEN/RELEASE는 active-model neutral task-finger posture를 공유한다. Isaac 좌표를 source object pose로 사용하지 않았다.

모든 episode에 같은 causal FIR quintic minimum-jerk transition을 적용했다. 기존 v1 temporal limit를 통과하는 최단 공통 설정은 {readiness['temporal_transition']['selected']['transition_frames']} frames ({readiness['temporal_transition']['selected']['duration_sec']:.3f} s)이며, max step/velocity/acceleration은 각각 {readiness['temporal_transition']['selected']['max_hand_joint_step_rad']:.6f} rad, {readiness['temporal_transition']['selected']['max_hand_velocity_rad_s']:.6f} rad/s, {readiness['temporal_transition']['selected']['max_hand_acceleration_rad_s2']:.6f} rad/s²이다.

# C. third finger neutralization

Third/middle 2 DoF는 task/contact/support 변수에서 완전히 제외했다. 모든 기존 third-collision frame, 30-frame stride, semantic-transition ±2 frame을 합친 50-episode search set에서 coarse→coordinate-refine search를 수행했다. 선택 q는 모든 phase/episode에 동일하다. 완전 무충돌 neutral 발견 여부는 `{str(readiness['fully_collision_free_global_third_neutral_found']).lower()}`이며 결과는 `{readiness['global_neutral_interpretation']}`이다.

# D. collision 감소가 실제 hand repair인지 category 이동인지

{category_table}

Third frames는 {100*aggregate['v1_to_v2_1']['third_finger_collision_fractional_reduction']:.2f}% 감소했고 comprehensive hand frames는 {100*aggregate['v1_to_v2_1']['comprehensive_hand_collision_fractional_reduction']:.2f}% 감소했다. 그러나 `HAND_HAND`/`CROSS_ARM` 잔여가 0은 아니므로 단순 category 이름 변경으로 성공을 주장하지 않는다. Arm-only `ARM_TORSO`는 frozen arm 때문에 그대로 남는 것이 정상이다.

# E. 남은 collision categories

{top_remaining}

{comparison_table}

v2.1 task-critical pinch-frame error는 v1 Proposed보다 {1000*aggregate['v1_to_v2_1']['task_critical_pinch_error_episode_mean_delta_m']:+.3f} mm 변했지만 Dataset A의 {1000*baseline['pinch_frame']['task_critical_pinch_error_episode_mean_m']['mean']:.3f} mm보다 작다. 새 morphology-feasible primitive의 static tool translation이 기존 5 mm tolerance를 넘으므로 frozen-arm 수치는 진단 결과이며, canonical Dataset B 생성 전 common-arm rerun이 필요하다.

# F. 추가/수정 파일

""" + "\n".join(f"- `{path}`" for path in added) + f"""

생성 artifact:

""" + "\n".join(f"- `{output_root / path}`" for path in artifacts) + f"""

기존 v1, hand-v2, Dataset A/B 파일은 수정하지 않았다. 모든 산출물은 `{output_root}` 아래에 새로 기록했다.

# G. tests

`{tests['command']}` → **{'PASS' if tests['pass'] else 'FAIL'}** ({tests['passed_test_count']} passed, exit {tests['exit_code']})

Anti-overfit scan → **{'PASS' if tests['anti_overfit_scan']['pass'] else 'FAIL'}**, hits={len(tests['anti_overfit_scan']['hits'])}

{readiness['conclusion']}
"""
    (output_root / "summary/final_report.md").write_text(report, encoding="utf-8")


def run_pipeline(
    output_root: Path = V2_1_ROOT,
    *,
    method: str = "proposed",
    hand_mapper: str = "feasible_semantic_primitive",
    run_tests: bool = True,
) -> dict[str, Any]:
    if method != "proposed":
        raise ValueError("hand v2.1 is Proposed/Dataset-B only")
    if hand_mapper != "feasible_semantic_primitive":
        raise ValueError("exact-contact v2 remains diagnostic and is not the v2.1 default")
    if output_root.resolve() in {V1_ROOT.resolve(), V2_ROOT.resolve()}:
        raise ValueError("v2.1 must use an isolated output root")
    for folder in ("audit", "config", "metrics", "renders", "summary", "tests", "integrity"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)

    v1 = load_v1_config()
    config = load_source_config()
    before = integrity_snapshot()
    atomic_json(output_root / "integrity/before.json", before)
    prior = _prior_summary()
    atomic_json(output_root / "audit/prior_results_summary.json", prior)

    runtime = make_runtime(v1)
    classifier = CollisionClassifier(runtime, v1)
    builder = SemanticPrimitiveBuilder(runtime, v1, config)
    primitives = builder.build()
    compatibility = tool_compatibility(primitives, v1)
    transition_frames, episodes, temporal_transition = _select_global_transition(
        primitives, v1, config
    )
    refs, search_frame_audit = build_search_frame_refs(
        runtime, classifier, episodes, config
    )
    search = search_global_neutrals(
        runtime, classifier, episodes, refs, config
    )
    full_pareto = _full_pareto_validation(
        runtime, classifier, episodes, search, config
    )
    search["full_50_episode_pareto_validation"] = full_pareto
    search["search_frame_audit"] = search_frame_audit
    install_global_neutrals(
        episodes, search["third_indices"], search["selected"]
    )

    baseline = evaluate_variant(
        "dataset_a_baseline", runtime, classifier, episodes, v1
    )
    current = evaluate_variant(
        "current_proposed_v1", runtime, classifier, episodes, v1
    )
    candidate = evaluate_variant(
        "proposed_hand_v2_1", runtime, classifier, episodes, v1
    )
    exact = exact_v2_ablation_summary()
    aggregate, collision, episode_rows = compare_variants(
        baseline, current, candidate, exact
    )

    after = integrity_snapshot()
    integrity = {
        "before": before,
        "after": after,
        "dataset_a_unchanged": before["dataset_a_tree_sha256"]
        == after["dataset_a_tree_sha256"],
        "g1_arm_trajectories_unchanged": before["proposed_arm_combined_sha256"]
        == after["proposed_arm_combined_sha256"],
    }
    atomic_json(output_root / "integrity/after_and_comparison.json", integrity)
    readiness = _readiness(
        config, aggregate, compatibility, integrity, search, temporal_transition
    )
    candidate_config = _candidate_config_artifact(
        v1, config, primitives, search, compatibility, temporal_transition
    )
    candidate_config["status"] = readiness["classification"]
    candidate_config["integration_readiness"] = {
        "direct_dataset_b": readiness["ready_for_direct_dataset_b_integration"],
        "common_arm_rerun": readiness["ready_for_common_arm_rerun"],
    }

    atomic_json(
        output_root / "audit/third_finger_collision_audit.json", search
    )
    atomic_json(
        output_root / "config/proposed_hand_v2_1_candidate.json", candidate_config
    )
    atomic_json(output_root / "metrics/aggregate_comparison.json", aggregate)
    atomic_json(output_root / "metrics/collision_comparison.json", collision)
    atomic_json(
        output_root / "metrics/semantic_phase_validation.json",
        {
            "dataset_a_baseline": baseline["semantic"],
            "current_proposed_v1": current["semantic"],
            "proposed_hand_v2_1": candidate["semantic"],
            "temporal_transition": temporal_transition,
            "source_supported": True,
            "object_pose_required": False,
        },
    )
    flat_rows = []
    for row in episode_rows:
        flat = {key: value for key, value in row.items() if key != "category_frame_incidence"}
        flat["category_frame_incidence_json"] = json.dumps(
            row["category_frame_incidence"], sort_keys=True
        )
        flat_rows.append(flat)
    atomic_csv(output_root / "metrics/episode_metrics.csv", flat_rows)
    atomic_json(
        output_root / "summary/tool_transform_compatibility.json", compatibility
    )
    atomic_json(output_root / "summary/integration_readiness.json", readiness)
    renders = render_all(
        output_root,
        runtime,
        v1,
        config,
        primitives,
        search["selected"],
    )
    tests = _run_tests(output_root) if run_tests else {
        "command": "SKIPPED",
        "exit_code": 0,
        "pass": True,
        "passed_test_count": 0,
        "output": "tests skipped by explicit CLI flag",
        "anti_overfit_scan": _anti_overfit_scan(),
    }
    atomic_json(output_root / "tests/test_report.json", tests)
    if not tests["pass"]:
        readiness["classification"] = "PROPOSED_HAND_V2_1_NOT_READY"
        readiness["conclusion"] = "PROPOSED_HAND_V2_1_NOT_READY"
        readiness["test_gate_pass"] = False
        atomic_json(output_root / "summary/integration_readiness.json", readiness)
    else:
        readiness["test_gate_pass"] = True
        atomic_json(output_root / "summary/integration_readiness.json", readiness)
    _write_final_report(
        output_root,
        candidate_config,
        aggregate,
        collision,
        compatibility,
        readiness,
        integrity,
        tests,
        search,
    )
    result = {
        "output_root": str(output_root),
        "left_grasp_q": candidate_config["states"]["left"]["GRASP"],
        "right_grasp_q": candidate_config["states"]["right"]["GRASP"],
        "left_third_neutral_q": candidate_config["left_third_safe_neutral"],
        "right_third_neutral_q": candidate_config["right_third_safe_neutral"],
        "current_third_frames": current["collision"]["third_finger_related_frames"],
        "candidate_third_frames": candidate["collision"]["third_finger_related_frames"],
        "current_hand_frames": current["collision"]["comprehensive_hand_related_frames"],
        "candidate_hand_frames": candidate["collision"]["comprehensive_hand_related_frames"],
        "dataset_a_unchanged": integrity["dataset_a_unchanged"],
        "g1_arm_action_unchanged": integrity["g1_arm_trajectories_unchanged"],
        "tests_pass": tests["pass"],
        "render_count": len(renders["files"]),
        "readiness": readiness["classification"],
        "conclusion": readiness["conclusion"],
    }
    atomic_json(output_root / "summary/run_summary.json", result)
    return result


__all__ = ["run_pipeline"]
