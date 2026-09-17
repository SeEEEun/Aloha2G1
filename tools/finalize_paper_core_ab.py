#!/usr/bin/env python3
"""Validate and freeze the minimum-complete paper-core A/B result bundle.

This script is intentionally a terminal gate, not an evaluator.  It consumes
the three already-produced experiment results and refuses READY unless the
paired training, HELDOUT8 prediction, all 16 source-conditioned Isaac runs,
source-input identity, representative assets, and human motion review pass.
It never reads from or communicates with real G1 hardware.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"


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
    require(path.is_file(), f"missing required artifact: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def lookup_table(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    matches = [row for row in rows if row["metric"] == metric]
    require(len(matches) == 1, f"table metric is not unique: {metric}")
    return matches[0]


def main() -> None:
    table1 = read_json(PAPER / "tables/table1_full50_retargeting.json")
    common = read_json(PAPER / "common48_manifest.json")
    train = read_json(PAPER / "train40_manifest.json")
    heldout = read_json(PAPER / "heldout8_manifest.json")
    packaging = read_json(PAPER / "dataset_packaging_audit.json")
    initial_contract_path = PAPER / "COMMON_G1_POLICY_INITIAL_STATE_AB_V1.json"
    initial_contract = read_json(initial_contract_path)
    initial_audit = read_json(PAPER / "common_initial_state_isaac_audit.json")
    training = read_json(PAPER / "act_a_b_training_audit.json")
    experiment2 = read_json(PAPER / "offline_heldout8/experiment2_result.json")
    raw_visual = read_json(
        PAPER
        / "offline_heldout8/raw_chunk_visual_review/raw_chunk_visual_review_assets.json"
    )
    experiment3 = read_json(
        PAPER / "source_conditioned_rollout/experiment3_result.json"
    )
    visual = read_json(
        PAPER / "source_conditioned_rollout/human_visual_review.json"
    )
    ablation = read_json(PAPER / "ablation_status.json")

    require(table1["status"] == "PASS", "Experiment 1 is not PASS")
    require(common["status"] == "PASS", "COMMON48 is not PASS")
    require(common["episode_count"] == 48, "COMMON48 episode count changed")
    require(common["a_episode_count"] == common["b_episode_count"] == 48, "A/B COMMON48 differs")
    require(common["exact_same_final_dataset_indices"], "A/B COMMON48 identities differ")
    excluded = [int(row["final_dataset_index"]) for row in common["excluded_a_hard_episodes"]]
    require(excluded == [35, 46], f"unexpected frozen A-hard identities: {excluded}")
    require(train["status"] == heldout["status"] == "PASS", "split manifest is not PASS")
    require(train["episode_count"] == 40 and heldout["episode_count"] == 8, "split counts changed")
    split = heldout["split_contract"]
    require(split["split_seed"] == 20260826, "split seed changed")
    require(split["a_b_manifests_identical"], "A/B manifests are not identical")
    require(split["selected_once_before_training"], "split was not frozen before training")
    require(heldout["all_heldout_complete"], "held-out source phases are incomplete")
    require(packaging["status"] == "PASS", "dataset packaging audit is not PASS")
    require(
        initial_contract["status"] == "COMMON_G1_POLICY_INITIAL_STATE_AB_V1",
        "paper-specific common initial-state contract is invalid",
    )
    require(
        initial_contract["frozen_before_paper_model_prediction"]
        and not initial_contract["selection_used_policy_predictions_or_performance"],
        "paper initial pose was not selected independently of policy results",
    )
    require(
        initial_contract["symmetry_before_projection"]["equal_full_l2_to_1e_12"],
        "paper initial pose is not symmetric between A40/B40 medians",
    )
    require(initial_audit["status"] == "PASS", "common pose failed Isaac dry reset")
    require(
        initial_audit["contract_sha256"] == sha256_file(initial_contract_path),
        "Isaac dry reset used a different initial-state contract",
    )

    require(training["status"] == "PASS", "paired training audit is not PASS")
    for method in ("a", "b"):
        row = training["methods"][method]
        require(row["status"] == "PASS", f"ACT-{method.upper()}40 training failed")
        require(row["training_steps"] == 100_000, "paired training budget changed")
        require(row["all_logged_losses_finite"], "non-finite training loss")
        require(row["all_logged_gradients_finite"], "non-finite training gradient")
        require(row["final_checkpoint_tensor_audit"]["finite"], "non-finite checkpoint")
    require(
        training["methods"]["a"]["training_steps"]
        == training["methods"]["b"]["training_steps"],
        "A/B training budgets differ",
    )

    require(experiment2["status"] == "PASS", "Experiment 2 is not PASS")
    require(experiment2["input_audit"]["probe_count"] == 72, "Experiment 2 probe count changed")
    require(
        experiment2["input_audit"]["decoded_rgb_tensor_exact_equal_count"] == 72,
        "Experiment 2 A/B RGB inputs differ",
    )
    for method in ("a", "b"):
        selected = experiment2["methods"][method]["selected_checkpoint_evaluation"]
        require(selected["prediction_finite"], "non-finite held-out prediction")
        require(selected["prediction_shape"] == [72, 50, 28], "held-out output shape changed")
        require(selected["phase_score"]["total"] == 64, "phase-score denominator changed")
    require(raw_visual["status"] == "PASS", "raw ACT chunk review assets are incomplete")
    require(
        raw_visual["raw_prediction"]
        and not raw_visual["temporal_ensemble"]
        and not raw_visual["smoothing"]
        and not raw_visual["physics"],
        "raw ACT chunk review is not an unmodified policy-output visualization",
    )
    for phase, row in raw_visual["records"].items():
        path = Path(row["video"])
        require(
            path.is_file() and sha256_file(path) == row["video_sha256"],
            f"raw ACT review video changed: {phase}",
        )
    raw_strip = Path(raw_visual["temporal_strips"]["path"])
    require(
        raw_strip.is_file()
        and sha256_file(raw_strip) == raw_visual["temporal_strips"]["sha256"],
        "raw ACT temporal-strip asset changed",
    )

    require(experiment3["status"] == "PASS", "Experiment 3 is not PASS")
    require(
        experiment3["experiment_name"]
        == "SOURCE-VIDEO-CONDITIONED G1 POLICY ROLLOUT",
        "Experiment 3 name/semantics changed",
    )
    require(experiment3["paired_complete_output_episodes"] == list(range(8)), "not all paired rollouts completed")
    require(
        all(
            row["exact_source_video_and_decoded_frame_identity"]
            for row in experiment3["source_input_identity"]
        ),
        "source ALOHA inputs were not exactly identical for A/B",
    )
    for method in ("a", "b"):
        row = experiment3["methods"][method]
        require(row["completed"] == row["total"] == 8, "rollout completion count changed")
        require(row["primary_paired_eligible"] == 8, "paired eligibility incomplete")
    require(experiment3["representative_assets"]["status"] == "PASS", "Figure 4 assets incomplete")
    require(not experiment3["claims"]["real_robot"], "paper result must remain simulation-only")
    require(not experiment3["claims"]["physical_manipulation_success"], "physical success claim is forbidden")
    require(visual["status"] == "PASS", "human motion review is not PASS")
    require(visual["a_visible_mechanical_smoothness"] == "PASS", "ACT-A visible motion failed")
    require(visual["b_visible_mechanical_smoothness"] == "PASS", "ACT-B visible motion failed")
    require(visual["a_raw_chunk_smoothness"] == "PASS", "raw ACT-A chunks failed visual review")
    require(visual["b_raw_chunk_smoothness"] == "PASS", "raw ACT-B chunks failed visual review")
    require(
        ablation["decision"] in {"RUN", "SKIPPED"}
        and not ablation["primary_experiments_blocked"],
        "optional ablation blocks paper core",
    )

    table2 = experiment2["table2"]
    table3 = experiment3["table3"]
    offline = {
        "a_action_rmse_rad": float(
            lookup_table(table2, "full valid chunk RMSE")["ACT-A40"]
        ),
        "b_action_rmse_rad": float(
            lookup_table(table2, "full valid chunk RMSE")["ACT-B40"]
        ),
        "a_predicted_wrist_mean_mm": float(
            lookup_table(table2, "predicted source wrist error mean")["ACT-A40"]
        ),
        "b_predicted_wrist_mean_mm": float(
            lookup_table(table2, "predicted source wrist error mean")["ACT-B40"]
        ),
        "a_predicted_whole_hand_mean_mm": float(
            lookup_table(table2, "predicted whole-hand error mean")["ACT-A40"]
        ),
        "b_predicted_whole_hand_mean_mm": float(
            lookup_table(table2, "predicted whole-hand error mean")["ACT-B40"]
        ),
        "a_predicted_bimanual_mean_mm": float(
            lookup_table(table2, "predicted bimanual relation error mean")["ACT-A40"]
        ),
        "b_predicted_bimanual_mean_mm": float(
            lookup_table(table2, "predicted bimanual relation error mean")["ACT-B40"]
        ),
        "phase_score_a": lookup_table(table2, "phase-behavior score")["ACT-A40"],
        "phase_score_b": lookup_table(table2, "phase-behavior score")["ACT-B40"],
    }
    rollout = {
        "a_wrist_mean_mm": float(lookup_table(table3, "source wrist error mean")["ACT-A40"]),
        "b_wrist_mean_mm": float(lookup_table(table3, "source wrist error mean")["ACT-B40"]),
        "a_whole_hand_mean_mm": float(lookup_table(table3, "whole-hand error mean")["ACT-A40"]),
        "b_whole_hand_mean_mm": float(lookup_table(table3, "whole-hand error mean")["ACT-B40"]),
        "a_bimanual_mean_mm": float(lookup_table(table3, "bimanual relation error mean")["ACT-A40"]),
        "b_bimanual_mean_mm": float(lookup_table(table3, "bimanual relation error mean")["ACT-B40"]),
        "handoff_ordering_a": lookup_table(table3, "right-acquire-before-left-release")["ACT-A40"],
        "handoff_ordering_b": lookup_table(table3, "right-acquire-before-left-release")["ACT-B40"],
        "collision_frames_a": int(lookup_table(table3, "hard collision frame incidence")["ACT-A40"]),
        "collision_frames_b": int(lookup_table(table3, "hard collision frame incidence")["ACT-B40"]),
    }

    required_assets = [
        PAPER / "tables/table1_full50_retargeting.csv",
        PAPER / "tables/table1_full50_retargeting.md",
        PAPER / "tables/table2_heldout_act_prediction.csv",
        PAPER / "tables/table2_heldout_act_prediction.md",
        PAPER / "tables/table3_source_conditioned_rollout.csv",
        PAPER / "tables/table3_source_conditioned_rollout.md",
        PAPER / "figures/figure1_pipeline.svg",
        PAPER / "figures/figure2_source_vs_fair_a_vs_proposed_b.mp4",
        PAPER / "figures/figure3_interaction_metrics.png",
        PAPER / "figures/figure3_interaction_metrics.pdf",
        PAPER / "figures/figure4_source_vs_act_a_vs_act_b_overview.mp4",
        PAPER / "figures/figure4_source_vs_act_a_vs_act_b_three_quarter.mp4",
    ]
    asset_manifest = [artifact(path) for path in required_assets]
    result = {
        "schema_version": "paper_core_ab_acceptance_v1",
        "status": "READY",
        "terminal_marker": "PAPER_CORE_A_B_RESULTS_READY",
        "experiment1": {
            "status": "PASS",
            "full50_a": "27 CLEAN / 21 WARNING / 2 HARD",
            "full50_b": "13 CLEAN / 37 WARNING / 0 HARD",
            "observation": table1["observation"],
        },
        "common48": {"verified": True, "excluded_a_hard_episodes": excluded},
        "common_initial_state": {
            "contract": artifact(initial_contract_path),
            "isaac_dry_reset": initial_audit,
            "symmetric_train40_midpoint": True,
        },
        "split": {
            "train": train["episode_count"],
            "heldout": heldout["episode_count"],
            "seed": split["split_seed"],
            "heldout_final_dataset_indices": split["heldout_final_dataset_indices"],
        },
        "training": training,
        "experiment2": offline,
        "raw_chunk_visual_review": raw_visual,
        "experiment3": rollout,
        "visual_review": visual,
        "ablation": ablation["decision"],
        "real_g1": "OPTIONAL / NOT_REQUIRED_FOR_PAPER_CORE / NOT_RUN",
        "claim_scope": experiment3["claims"],
        "assets": asset_manifest,
    }
    atomic_json(PAPER / "paper_core_acceptance.json", result)
    lines = [
        "# Paper-Core A/B Acceptance",
        "",
        "Status: **READY**",
        "",
        f"COMMON48 excludes frozen Fair-A hard episodes: {excluded}.",
        f"Split: TRAIN40 / HELDOUT8, seed {split['split_seed']}.",
        f"Held-out phase score: ACT-A {offline['phase_score_a']}; ACT-B {offline['phase_score_b']}.",
        f"Full rollout completion: ACT-A {experiment3['methods']['a']['completed']}/8; ACT-B {experiment3['methods']['b']['completed']}/8.",
        "Real G1: not run and not required for this claim.",
        "",
        "PAPER_CORE_A_B_RESULTS_READY",
        "",
    ]
    atomic_text(PAPER / "PAPER_CORE_ACCEPTANCE.md", "\n".join(lines))
    print(json.dumps({
        "status": result["status"],
        "terminal_marker": result["terminal_marker"],
        "experiment2": offline,
        "experiment3": rollout,
        "asset_count": len(asset_manifest),
    }, indent=2))


if __name__ == "__main__":
    main()
