#!/usr/bin/env python3
"""Finish Experiment 2 named-G1 FK in the Isaac/SciPy environment.

ACT inference and checkpoint selection are produced by
``evaluate_paper_core_act_ab.py --inference-only`` in the LeRobot environment.
This second stage is deterministic CPU FK; it never loads or executes a policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.evaluate_paper_core_act_ab import (
        EVALUATION_CONTRACT,
        METHODS,
        OUTPUT,
        PROBE_LABELS,
        SELECTION_CONTRACT,
        TABLES,
        atomic_csv,
        atomic_json,
        hand_configuration_and_ordering,
        json_default,
        load_json,
        markdown_table,
        parquet_arrays,
        predicted_geometry,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    from evaluate_paper_core_act_ab import (
        EVALUATION_CONTRACT,
        METHODS,
        OUTPUT,
        PROBE_LABELS,
        SELECTION_CONTRACT,
        TABLES,
        atomic_csv,
        atomic_json,
        hand_configuration_and_ordering,
        json_default,
        load_json,
        markdown_table,
        parquet_arrays,
        predicted_geometry,
        sha256_file,
    )


ROOT = Path("/home/jbnu/aloha_g1_dataset")
HELDOUT_MANIFEST = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
INFERENCE_SELECTION = OUTPUT / "experiment2_inference_selection.json"


def build_geometry_probes() -> list[dict[str, Any]]:
    manifest = load_json(HELDOUT_MANIFEST)
    arrays = {method: parquet_arrays(row["dataset"]) for method, row in METHODS.items()}
    offsets = []
    running = 0
    for entry in manifest["entries"]:
        offsets.append(running)
        running += int(entry["frames"])
    probes = []
    for output_episode, entry in enumerate(manifest["entries"]):
        final_episode = int(entry["final_dataset_index"])
        length = int(entry["frames"])
        audit = manifest["complete_source_phase_audit"][str(final_episode)]
        with np.load(entry["a_trajectory_path"], allow_pickle=False) as archive_a, np.load(
            entry["b_trajectory_path"], allow_pickle=False
        ) as archive_b:
            source_wrist = {
                side: archive_a[f"target_{side}_wrist_position_model"].astype(np.float64)
                for side in ("left", "right")
            }
            source_interaction = {}
            for side in ("left", "right"):
                a_value = archive_a[f"target_{side}_interaction_frame_position_world"].astype(
                    np.float64
                )
                b_value = archive_b[f"source_{side}_interaction_frame_position_world"].astype(
                    np.float64
                )
                if not np.array_equal(a_value, b_value):
                    raise RuntimeError(
                        f"source interaction target mismatch ep{final_episode} {side}"
                    )
                source_interaction[side] = b_value
            event_map = {
                str(name): int(frame)
                for name, frame in zip(
                    archive_b["event_names"], archive_b["event_frames"], strict=True
                )
            }
            left_phase = archive_b["left_hand_phase"].astype(str)
            post_release_open = np.flatnonzero(
                (np.arange(length) > event_map["LEFT_RELEASE"]) & (left_phase == "OPEN")
            )
            if not len(post_release_open):
                raise RuntimeError(f"no post-release left-open frame ep{final_episode}")
            prototypes = {
                method: {
                    "right_open": arrays[method]["action"][offsets[output_episode] + event_map["RIGHT_CLOSE_ONSET"] - 1, 21:28].copy(),
                    "right_hold": arrays[method]["action"][offsets[output_episode] + event_map["RIGHT_STABLE_HOLD"], 21:28].copy(),
                    "left_hold": arrays[method]["action"][offsets[output_episode] + event_map["LEFT_STABLE_HOLD"], 14:21].copy(),
                    "left_open": arrays[method]["action"][offsets[output_episode] + int(post_release_open[0]), 14:21].copy(),
                }
                for method in METHODS
            }
            for phase in PROBE_LABELS:
                frame = int(audit["nine_probe_frames"][phase])
                valid = min(50, length - frame)
                probes.append(
                    {
                        "output_episode": output_episode,
                        "final_episode": final_episode,
                        "stable_episode_id": entry["stable_episode_id"],
                        "frame": frame,
                        "phase": phase,
                        "valid": valid,
                        "target": {
                            method: arrays[method]["action"][
                                offsets[output_episode] + frame : offsets[output_episode] + frame + valid
                            ].copy()
                            for method in METHODS
                        },
                        "source_wrist_model": {
                            side: source_wrist[side][frame : frame + valid].copy()
                            for side in ("left", "right")
                        },
                        "source_interaction_world": {
                            side: source_interaction[side][frame : frame + valid].copy()
                            for side in ("left", "right")
                        },
                        "prototypes": prototypes,
                    }
                )
    if len(probes) != 72:
        raise RuntimeError(f"geometry probe count changed: {len(probes)}")
    return probes


def main() -> None:
    inference = load_json(INFERENCE_SELECTION)
    if inference.get("status") != "PASS":
        raise RuntimeError("Experiment-2 inference selection is not PASS")
    probes = build_geometry_probes()
    joint_names = inference["input_audit"]["feature_names"]
    all_results = inference["methods"]
    for method in METHODS:
        archive_path = Path(all_results[method]["selected_prediction_archive"])
        with np.load(archive_path, allow_pickle=False) as archive:
            prediction = archive["prediction"].astype(np.float32)
            if prediction.shape != (72, 50, 28):
                raise RuntimeError(f"selected prediction shape changed: {archive_path}")
            if not np.array_equal(
                archive["final_episode"],
                np.asarray([row["final_episode"] for row in probes], dtype=np.int64),
            ):
                raise RuntimeError(f"prediction/probe identity mismatch: {archive_path}")
        geometry = predicted_geometry(method, prediction, probes, joint_names)
        hands = hand_configuration_and_ordering(method, prediction, probes)
        all_results[method]["common_source_geometry"] = geometry
        all_results[method]["hand_configuration_and_ordering"] = hands
        atomic_json(OUTPUT / method / "selected_checkpoint_result.json", all_results[method])

    table_rows = []
    for label, accessor, units in [
        ("first-action RMSE", ("action_accuracy", "first_action_rmse_rad"), "rad"),
        ("first-4 RMSE", ("action_accuracy", "first_4_frame_rmse_rad"), "rad"),
        ("full valid chunk RMSE", ("action_accuracy", "full_valid_chunk_rmse_rad"), "rad"),
        ("arm RMSE", ("action_accuracy", "arm_full_valid_chunk_rmse_rad"), "rad"),
        ("Dex3 RMSE", ("action_accuracy", "dex3_full_valid_chunk_rmse_rad"), "rad"),
    ]:
        table_rows.append(
            {
                "metric": label,
                "ACT-A40": f"{all_results['a']['selected_checkpoint_evaluation'][accessor[0]][accessor[1]]:.6f}",
                "ACT-B40": f"{all_results['b']['selected_checkpoint_evaluation'][accessor[0]][accessor[1]]:.6f}",
                "unit": units,
            }
        )
    for label, key in [
        ("predicted source wrist error mean", "source_wrist_trajectory_fidelity_error_mm"),
        ("predicted source wrist error p95", "source_wrist_trajectory_fidelity_error_mm"),
        ("predicted whole-hand error mean", "whole_hand_interaction_frame_error_mm"),
        ("predicted whole-hand error p95", "whole_hand_interaction_frame_error_mm"),
        ("predicted bimanual relation error mean", "bimanual_relation_error_mm"),
        ("predicted bimanual relation error p95", "bimanual_relation_error_mm"),
    ]:
        statistic = "p95" if label.endswith("p95") else "mean"
        table_rows.append(
            {
                "metric": label,
                "ACT-A40": f"{all_results['a']['common_source_geometry'][key][statistic]:.3f}",
                "ACT-B40": f"{all_results['b']['common_source_geometry'][key][statistic]:.3f}",
                "unit": "mm",
            }
        )
    for label, key, statistic, unit in [
        (
            "raw arm direction reversals mean",
            "arm_direction_reversals_per_s_mean",
            "mean_across_chunks",
            "reversals/s/joint",
        ),
        (
            "raw Dex3 direction reversals mean",
            "dex3_direction_reversals_per_s_mean",
            "mean_across_chunks",
            "reversals/s/joint",
        ),
        (
            "raw all-joint direction reversals mean",
            "direction_reversals_per_s_mean",
            "mean_across_chunks",
            "reversals/s/joint",
        ),
        (
            "raw maximum adjacent step",
            "max_adjacent_step_rad",
            "maximum_across_chunks",
            "rad",
        ),
        ("raw qdot RMS", "qdot_rms_rad_s", "mean_across_chunks", "rad/s"),
        ("raw qddot RMS", "qddot_rms_rad_s2", "mean_across_chunks", "rad/s^2"),
        ("raw jerk RMS", "jerk_rms_rad_s3", "mean_across_chunks", "rad/s^3"),
        (
            "raw low-motion detrended peak-to-peak",
            "low_motion_detrended_peak_to_peak_rad",
            "mean_across_chunks",
            "rad",
        ),
    ]:
        table_rows.append(
            {
                "metric": label,
                "ACT-A40": f"{all_results['a']['selected_checkpoint_evaluation']['raw_chunk_smoothness'][key][statistic]:.6f}",
                "ACT-B40": f"{all_results['b']['selected_checkpoint_evaluation']['raw_chunk_smoothness'][key][statistic]:.6f}",
                "unit": unit,
            }
        )
    table_rows.extend(
        [
            {
                "metric": "phase-behavior score",
                "ACT-A40": f"{all_results['a']['selected_checkpoint_evaluation']['phase_score']['successful']}/64",
                "ACT-B40": f"{all_results['b']['selected_checkpoint_evaluation']['phase_score']['successful']}/64",
                "unit": "probes",
            },
            {
                "metric": "right-acquire-before-left-release",
                "ACT-A40": f"{all_results['a']['hand_configuration_and_ordering']['handoff_ordering']['successful_episodes']}/8",
                "ACT-B40": f"{all_results['b']['hand_configuration_and_ordering']['handoff_ordering']['successful_episodes']}/8",
                "unit": "episodes",
            },
        ]
    )
    atomic_csv(TABLES / "table2_heldout_act_prediction.csv", table_rows)
    (TABLES / "table2_heldout_act_prediction.md").write_text(
        markdown_table(table_rows), encoding="utf-8"
    )
    result = {
        "schema_version": "paper_core_ab_heldout_act_prediction_v1",
        "status": "PASS",
        "evaluation_contract": str(EVALUATION_CONTRACT),
        "evaluation_contract_sha256": sha256_file(EVALUATION_CONTRACT),
        "checkpoint_selection_contract": str(SELECTION_CONTRACT),
        "checkpoint_selection_contract_sha256": sha256_file(SELECTION_CONTRACT),
        "input_audit": inference["input_audit"],
        "methods": all_results,
        "table2": table_rows,
        "environment_split": {
            "act_inference": "lerobot-smolvla environment",
            "named_g1_fk": "isaaclab6 SciPy/MuJoCo environment",
            "policy_inference_repeated_in_fk_stage": False,
        },
        "claims": {
            "offline_heldout_only": True,
            "physical_manipulation_success_claimed": False,
            "real_robot_claimed": False,
        },
    }
    atomic_json(OUTPUT / "experiment2_result.json", result)
    atomic_json(TABLES / "table2_heldout_act_prediction.json", {"rows": table_rows})
    print(
        json.dumps(
            {
                method: {
                    "selected_step": all_results[method]["checkpoint_selection"]["selected_step"],
                    "phase_score": all_results[method]["selected_checkpoint_evaluation"]["phase_score"],
                    "action_accuracy": all_results[method]["selected_checkpoint_evaluation"]["action_accuracy"],
                    "geometry": all_results[method]["common_source_geometry"],
                    "handoff_ordering": all_results[method]["hand_configuration_and_ordering"]["handoff_ordering"],
                }
                for method in METHODS
            },
            indent=2,
            default=json_default,
        )
    )


if __name__ == "__main__":
    main()
