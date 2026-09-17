#!/usr/bin/env python3
"""Freeze a PAPER_CORE_BLOCKED result after the paired frame-0 safety audit.

This is deliberately separate from ``finalize_paper_core_ab.py``.  The latter
remains the strict READY gate.  This program records a terminal, evidence-backed
blocked outcome without relaxing rollout safety, changing a policy, or claiming
metrics for trajectories that were never executed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
ROLLOUT = PAPER / "source_conditioned_rollout"
FRAME0_SOURCE_EPISODE = 2
FIRST_STEP_LIMIT_RAD = 0.15


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def table_value(rows: list[dict[str, Any]], metric: str, method: str) -> str:
    matches = [row for row in rows if row["metric"] == metric]
    require(len(matches) == 1, f"table metric is not unique: {metric}")
    return str(matches[0][method])


def load_frame0(method: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    path = ROLLOUT / method / f"heldout_00_source_{FRAME0_SOURCE_EPISODE:02d}"
    report = read_json(path / "rollout_report.json")
    with np.load(path / "rollout_arrays.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return report, arrays, path


def main() -> None:
    table1 = read_json(PAPER / "tables/table1_full50_retargeting.json")
    table2 = read_json(PAPER / "tables/table2_heldout_act_prediction.json")["rows"]
    common = read_json(PAPER / "common48_manifest.json")
    train = read_json(PAPER / "train40_manifest.json")
    heldout = read_json(PAPER / "heldout8_manifest.json")
    training = read_json(PAPER / "act_a_b_training_audit.json")
    experiment2 = read_json(PAPER / "offline_heldout8/experiment2_result.json")
    experiment3_path = ROLLOUT / "experiment3_result.json"
    experiment3 = read_json(experiment3_path)
    ablation = read_json(PAPER / "ablation_status.json")
    initial_contract_path = PAPER / "COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json"
    initial_contract = read_json(initial_contract_path)
    initial_audit = read_json(PAPER / "common_initial_state_isaac_audit.json")

    require(table1["status"] == "PASS", "Experiment 1 changed from PASS")
    require(common["status"] == "PASS" and common["episode_count"] == 48, "COMMON48 invalid")
    require(common["exact_same_final_dataset_indices"], "A/B COMMON48 identities differ")
    excluded = [int(row["final_dataset_index"]) for row in common["excluded_a_hard_episodes"]]
    require(excluded == [35, 46], f"frozen Fair-A hard identities changed: {excluded}")
    require(train["episode_count"] == 40 and heldout["episode_count"] == 8, "split changed")
    require(heldout["split_contract"]["split_seed"] == 20260826, "split seed changed")
    require(heldout["all_heldout_complete"], "held-out source phases incomplete")
    require(training["status"] == "PASS", "paired ACT training is not PASS")
    require(experiment2["status"] == "PASS", "Experiment 2 is not PASS")
    require(experiment3["status"] == "INCOMPLETE_OR_SAFETY_ABORT", "unexpected Experiment 3 state")
    require(initial_contract["status"] == "COMMON_G1_POLICY_INITIAL_STATE_AB_V1", "initial contract changed")
    require(initial_audit["status"] == "PASS", "common Isaac reset audit is not PASS")

    reports: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    paths: dict[str, Path] = {}
    for method in ("a", "b"):
        reports[method], arrays[method], paths[method] = load_frame0(method)
        report = reports[method]
        audit = report["first_raw_chunk_audit"]
        require(report["status"] == "SAFETY_ABORT", f"ACT-{method.upper()} frame0 did not abort")
        require(report["executed_frames"] == 0, "a command was unexpectedly executed")
        require(report["safety_abort"]["frame"] == 0, "abort was not at frame 0")
        require(report["safety_abort"]["when"] == "before_command", "abort happened after command")
        require(audit["checks"]["finite"], "raw chunk is non-finite")
        require(audit["checks"]["command_hard_limits"], "raw chunk violates hard limits")
        require(audit["checks"]["adjacent_step"], "raw chunk internal step gate failed")
        require(audit["checks"]["velocity"], "raw chunk internal velocity gate failed")
        require(audit["checks"]["acceleration"], "raw chunk internal acceleration gate failed")
        require(audit["checks"]["branch"], "raw chunk branch gate failed")
        require(audit["checks"]["hard_self_collision"], "raw chunk collision gate failed")
        require(not audit["checks"]["first_command_delta"], "expected first-command blocker absent")
        require(arrays[method]["commanded_action"].shape == (0, 28), "command array is not empty")
        require(arrays[method]["raw_act_query_chunk"].shape == (1, 50, 28), "raw chunk shape changed")
        require(
            np.array_equal(
                arrays[method]["raw_act_e1_action"],
                arrays[method]["deployment_projected_action"],
            ),
            "common projection altered the frame-0 ensembled action",
        )

    require(np.array_equal(arrays["a"]["joint_names"], arrays["b"]["joint_names"]), "joint order differs")
    require(np.array_equal(arrays["a"]["source_rgb_sha256"], arrays["b"]["source_rgb_sha256"]), "source RGB differs")
    require(np.array_equal(arrays["a"]["source_frame_index"], arrays["b"]["source_frame_index"]), "source clock differs")
    require(np.array_equal(arrays["a"]["measured_state"], arrays["b"]["measured_state"]), "measured reset state differs")

    names = arrays["a"]["joint_names"].astype(str)
    first = {method: arrays[method]["raw_act_e1_action"][0].astype(np.float64) for method in ("a", "b")}
    pair_delta = np.abs(first["a"] - first["b"])
    no_common = np.flatnonzero(pair_delta > 2.0 * FIRST_STEP_LIMIT_RAD + 1e-12)
    require(len(no_common) > 0, "the two observed first-action safe intervals unexpectedly overlap")
    interval_proof = [
        {
            "joint_index": int(index),
            "joint_name": str(names[index]),
            "act_a_first_action_rad": float(first["a"][index]),
            "act_b_first_action_rad": float(first["b"][index]),
            "absolute_separation_rad": float(pair_delta[index]),
            "maximum_separation_for_common_0p15_rad_reset": 2.0 * FIRST_STEP_LIMIT_RAD,
            "common_safe_interval_exists": False,
        }
        for index in no_common
    ]

    frame0 = {}
    for method in ("a", "b"):
        q = arrays[method]["measured_state"][0].astype(np.float64)
        delta = first[method] - q
        top = np.argsort(np.abs(delta))[::-1][:8]
        audit = reports[method]["first_raw_chunk_audit"]
        frame0[method] = {
            "status": reports[method]["status"],
            "executed_frames": 0,
            "raw_query_shape": list(arrays[method]["raw_act_query_chunk"].shape[1:]),
            "raw_query_sha256": read_json(paths[method] / "executed_prefix_safety_records.json")[0]["inference"]["raw_chunk_sha256"],
            "first_command_delta_max_abs_rad": audit["first_command_delta_max_abs_rad"],
            "first_command_delta_arm_l2_rad": audit["first_command_delta_arm_l2_rad_diagnostic_only"],
            "internal_chunk_max_adjacent_step_rad": audit["maximum_adjacent_step_rad"],
            "internal_chunk_max_velocity_rad_s": audit["maximum_velocity_rad_s"],
            "internal_chunk_max_acceleration_rad_s2": audit["maximum_acceleration_rad_s2"],
            "hard_limit_violations": audit["hard_limit_violation_count"],
            "hard_collision_incidence": audit["collision"]["invalid_hard_self_collision_incidence"],
            "common_projection_records_over_full_first_query_chunk": reports[method]["common_deployment_projection"]["projection_record_count"],
            "frame0_ensembled_action_modified_scalars_by_common_projection": int(
                np.count_nonzero(
                    arrays[method]["raw_act_e1_action"][0]
                    != arrays[method]["deployment_projected_action"][0]
                )
            ),
            "failed_checks": reports[method]["safety_abort"]["failed_checks"],
            "largest_current_to_first_action_deltas": [
                {
                    "joint_index": int(index),
                    "joint_name": str(names[index]),
                    "measured_reset_rad": float(q[index]),
                    "first_action_rad": float(first[method][index]),
                    "signed_delta_rad": float(delta[index]),
                }
                for index in top
            ],
            "report": artifact(paths[method] / "rollout_report.json"),
            "arrays": artifact(paths[method] / "rollout_arrays.npz"),
        }

    selected = {
        method: experiment2["methods"][method]["selected_checkpoint_evaluation"]
        for method in ("a", "b")
    }
    training_summary = {
        method: {
            "status": training["methods"][method]["status"],
            "steps": training["methods"][method]["training_steps"],
            "final_logged_loss": training["methods"][method]["last_logged_health"]["loss"],
            "selected_checkpoint": selected[method]["checkpoint"],
            "selected_step": selected[method]["checkpoint_step"],
            "selected_checkpoint_model_sha256": selected[method]["model_sha256"],
            "phase_score": (
                f"{selected[method]['phase_score']['successful']}/"
                f"{selected[method]['phase_score']['total']}"
            ),
        }
        for method in ("a", "b")
    }

    visual_review = {
        "schema_version": "paper_core_human_visual_review_v1",
        "status": "NOT_RUN_SAFETY_ABORT",
        "reason": "Both selected policies stopped at frame 0 before any command; no full rollout video exists to grade.",
        "a_visible_mechanical_smoothness": "NOT_EVALUABLE",
        "b_visible_mechanical_smoothness": "NOT_EVALUABLE",
        "raw_chunk_review_assets": str(PAPER / "offline_heldout8/raw_chunk_visual_review"),
        "claim": "No human-visible full-rollout smoothness claim is made.",
    }
    atomic_json(ROLLOUT / "human_visual_review.json", visual_review)

    result = {
        "schema_version": "paper_core_ab_acceptance_v1",
        "status": "BLOCKED",
        "terminal_marker": "PAPER_CORE_BLOCKED",
        "blocked_at": "EXPERIMENT_3_FRAME_0_PRE_COMMAND_SAFETY_GATE",
        "experiment1": {
            "status": "PASS",
            "table": artifact(PAPER / "tables/table1_full50_retargeting.json"),
            "observation": table1["observation"],
        },
        "common48": {"verified": True, "excluded_a_hard_episodes": excluded},
        "split": {
            "train": train["episode_count"],
            "heldout": heldout["episode_count"],
            "seed": heldout["split_contract"]["split_seed"],
            "heldout_final_dataset_indices": heldout["split_contract"]["heldout_final_dataset_indices"],
        },
        "training": training_summary,
        "experiment2": {
            "status": "PASS",
            "table": artifact(PAPER / "tables/table2_heldout_act_prediction.json"),
            "a_action_rmse_rad": float(table_value(table2, "full valid chunk RMSE", "ACT-A40")),
            "b_action_rmse_rad": float(table_value(table2, "full valid chunk RMSE", "ACT-B40")),
            "a_predicted_wrist_mean_mm": float(table_value(table2, "predicted source wrist error mean", "ACT-A40")),
            "b_predicted_wrist_mean_mm": float(table_value(table2, "predicted source wrist error mean", "ACT-B40")),
            "a_predicted_whole_hand_mean_mm": float(table_value(table2, "predicted whole-hand error mean", "ACT-A40")),
            "b_predicted_whole_hand_mean_mm": float(table_value(table2, "predicted whole-hand error mean", "ACT-B40")),
            "a_predicted_bimanual_mean_mm": float(table_value(table2, "predicted bimanual relation error mean", "ACT-A40")),
            "b_predicted_bimanual_mean_mm": float(table_value(table2, "predicted bimanual relation error mean", "ACT-B40")),
            "phase_score_a": training_summary["a"]["phase_score"],
            "phase_score_b": training_summary["b"]["phase_score"],
        },
        "experiment3": {
            "status": "BLOCKED",
            "planned_episodes": 8,
            "a_completed": 0,
            "b_completed": 0,
            "frame0": frame0,
            "identical_frame0_source_rgb": True,
            "identical_frame0_measured_state": True,
            "common_initial_state": artifact(initial_contract_path),
            "common_initial_state_isaac_audit": artifact(PAPER / "common_initial_state_isaac_audit.json"),
            "first_step_gate_rad": FIRST_STEP_LIMIT_RAD,
            "no_single_reset_can_connect_to_both_observed_first_actions_within_gate": True,
            "no_common_reset_joint_count": int(len(no_common)),
            "no_common_reset_interval_proof": interval_proof,
            "root_cause": (
                "Under the frozen common-state contract, the method-specific absolute-action policies emit "
                "incompatible observed frame-0 hand configurations. Their observed first actions are separated "
                "by more than twice the unchanged first-step gate on five joints. A different common reset "
                "would itself change the state-conditioned predictions and therefore requires a new predeclared, "
                "policy-independent initialization study; choosing one after these results would be performance-informed."
            ),
            "not_caused_by": [
                "NaN_or_Inf",
                "joint_order_mismatch",
                "normalization_or_denormalization_failure",
                "hard_joint_limit_violation",
                "common_projection",
                "raw_chunk_internal_step_velocity_or_acceleration",
                "collision_or_branch_discontinuity",
            ],
            "safety_response": "STOPPED_WITH_ZERO_COMMANDS; no gate relaxation, smoothing, or policy-specific pose was introduced.",
            "proof_scope": (
                "The interval proof applies to the two preserved predictions observed under the frozen common "
                "state. It does not claim that all possible state-conditioned common-reset inputs would fail."
            ),
            "table": artifact(PAPER / "tables/table3_source_conditioned_rollout.json"),
            "evaluation": artifact(experiment3_path),
            "human_visual_review": visual_review,
        },
        "ablation": ablation["status"],
        "real_g1": "OPTIONAL / NOT_REQUIRED_FOR_PAPER_CORE / NOT_RUN",
        "paper_core": "BLOCKED",
        "claim_control": {
            "experiments_1_and_2_support_offline_persistence_of_the_retargeting_difference": True,
            "experiment_3_full_rollout_claim_available": False,
            "autonomous_real_g1_claim": False,
            "physical_manipulation_success_claim": False,
            "sim_to_real_claim": False,
        },
        "transient_isaac_startup_crash_attempt_preserved": artifact(
            ROLLOUT / "startup_crash_attempt1/b_heldout_00.log"
        ),
    }
    atomic_json(PAPER / "paper_core_acceptance.json", result)
    atomic_json(PAPER / "paper_core_blocked_report.json", result)

    lines = [
        "# Paper-Core A/B Acceptance",
        "",
        "Status: **BLOCKED**",
        "",
        "Experiments 1 and 2 are complete and remain PASS. Experiment 3 stopped at its frame-0 pre-command safety gate for both policies; zero commands were executed.",
        "",
        f"The unchanged first-step gate is {FIRST_STEP_LIMIT_RAD:.2f} rad. Under the frozen common state, five joints have ACT-A/ACT-B observed first-action separation above {2 * FIRST_STEP_LIMIT_RAD:.2f} rad; the maximum is {float(np.max(pair_delta)):.6f} rad. No reset value can connect to both of these preserved predictions within the gate. Because ACT is state-conditioned, testing another common reset requires a new predeclared initialization study and fresh inference; it is not inferred from this interval proof.",
        "",
        "No safety threshold, checkpoint, dataset, retargeting result, normalization, common adapter, or policy weight was changed.",
        "",
        "PAPER_CORE_BLOCKED",
        "",
    ]
    text = "\n".join(lines)
    atomic_text(PAPER / "PAPER_CORE_ACCEPTANCE.md", text)
    atomic_text(PAPER / "PAPER_CORE_BLOCKED_REPORT.md", text)
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal_marker": result["terminal_marker"],
                "blocked_at": result["blocked_at"],
                "a_first_command_delta_max_abs_rad": frame0["a"]["first_command_delta_max_abs_rad"],
                "b_first_command_delta_max_abs_rad": frame0["b"]["first_command_delta_max_abs_rad"],
                "no_common_reset_joint_count": int(len(no_common)),
                "maximum_a_b_first_action_separation_rad": float(np.max(pair_delta)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
