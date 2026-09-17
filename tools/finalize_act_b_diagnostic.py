#!/usr/bin/env python3
"""Assemble the evidence-backed final ACT-B diagnostic report and table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/policy_b_act"
IMPLEMENTATION = OUTPUT / "audit/installed_act_implementation.json"
COMPATIBILITY = OUTPUT / "audit/dataset_compatibility.json"
IDENTITY = OUTPUT / "audit/dataset_identity_after_training.json"
HEALTH = OUTPUT / "training_health.json"
SELECTION = OUTPUT / "selected_checkpoint.json"
REPEATABILITY = OUTPUT / "raw_diagnostics/fixed_input_repeatability.json"
RAW = OUTPUT / "raw_diagnostics/raw_single_chunk_smoothness.json"
SMOL_STOCHASTICITY = (
    ROOT / "outputs/policy_execution_stability_review/fixed_input_stochasticity/fixed_observation_stochasticity.json"
)
SMOL_PHASE = ROOT / "outputs/policy_b_offline_phase_probe/phase_learning_decision.json"
TEMPORAL = OUTPUT / "temporal_ensemble/official_e0_vs_e1_report.json"
COMPARISON_OUTPUT = OUTPUT / "comparison"
PHASE_ORDER = (
    ("initial_left_approach", "initial"),
    ("immediately_before_left_grasp", "before grasp"),
    ("left_grasp_left_owned", "left grasp"),
    ("left_transport", "left transport"),
    ("handoff_approach", "handoff approach"),
    ("dual_contact_transfer", "dual contact"),
    ("right_owned", "right owned"),
    ("right_transport", "right transport"),
    ("release", "release"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-json", type=Path, required=True)
    parser.add_argument("--isaac-report", type=Path, default=None)
    return parser.parse_args()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def aggregate_value(raw: dict[str, Any], source: str, metric: str, summary: str = "mean_across_chunks") -> float:
    return float(raw["aggregates"][source][metric][summary])


def fmt(value: float) -> str:
    return f"{value:.6g}"


def main() -> None:
    args = parse_args()
    required = (IMPLEMENTATION, COMPATIBILITY, IDENTITY, HEALTH, SELECTION, REPEATABILITY, RAW, SMOL_STOCHASTICITY, SMOL_PHASE)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"final ACT-B evidence missing: {missing}")
    decision = read(args.decision_json.resolve())
    raw_class = decision["ACT_RAW_CHUNK_SMOOTHNESS"]
    root_cause = decision["ROOT_CAUSE"]
    terminal = decision["TERMINAL_STATUS"]
    valid_roots = {
        "PASS": "SMOLVLA_POLICY_CLASS_DOMINANT_BLOCKER",
        "PARTIAL": "MIXED",
        "FAIL": "DATASET_TEMPORAL_SUPERVISION_BLOCKER",
    }
    if raw_class not in valid_roots or root_cause != valid_roots[raw_class]:
        raise RuntimeError("human decision root cause is inconsistent with raw smoothness classification")
    if terminal not in {
        "ACT_B_READY_FOR_A_B_RESEARCH",
        "ACT_B_DIAGNOSTIC_ONLY",
        "POLICY_CLASS_CHANGE_DID_NOT_SOLVE_NONSMOOTHNESS",
    }:
        raise RuntimeError("invalid terminal status")

    implementation = read(IMPLEMENTATION)
    compatibility = read(COMPATIBILITY)
    identity = read(IDENTITY)
    health = read(HEALTH)
    selection = read(SELECTION)
    repeatability = read(REPEATABILITY)
    raw = read(RAW)
    smol_stochasticity = read(SMOL_STOCHASTICITY)
    smol_phase = read(SMOL_PHASE)
    selected_step = int(selection["selected_checkpoint_step"])
    phase_eval = read(
        OUTPUT / "checkpoint_evaluation" / f"checkpoint_{selected_step:06d}_evaluation.json"
    )
    temporal = read(TEMPORAL) if TEMPORAL.is_file() else None
    isaac = read(args.isaac_report.resolve()) if args.isaac_report is not None else None

    if identity["ACTION_ARRAYS_UNCHANGED"] != "YES" or identity["STATE_ARRAYS_UNCHANGED"] != "YES":
        raise RuntimeError("immutable Dataset-B final gate failed")
    if repeatability["ACT_FIXED_INPUT_STOCHASTICITY"] != "NEGLIGIBLE":
        raise RuntimeError("ACT fixed-input stochasticity gate failed")

    with np.load(
        ROOT / "outputs/policy_execution_stability_review/fixed_input_stochasticity/fixed_observation_inference_samples.npz",
        allow_pickle=False,
    ) as archive:
        smol_repeats = archive["raw_policy_chunks"].astype(np.float64)
    smol_first_std = np.std(smol_repeats[:, :, 0, :], axis=1)
    smol_full_std = np.std(smol_repeats, axis=1)
    act_first_std = max(
        float(row["first_action_standard_deviation"]["maximum_joint_rad"])
        for row in repeatability["conditions"].values()
    )
    act_full_std = max(
        float(row["full_chunk_standard_deviation"]["maximum_offset_joint_rad"])
        for row in repeatability["conditions"].values()
    )

    metrics = (
        ("raw arm direction reversals (/s, mean)", "arm_direction_reversals_per_s_mean"),
        ("raw Dex3 direction reversals (/s, mean)", "dex3_direction_reversals_per_s_mean"),
        ("raw max joint step (rad, mean chunk max)", "adjacent_step_max_abs_rad"),
        ("raw qdot RMS (rad/s)", "qdot_rms_rad_s"),
        ("raw qdot max (rad/s, mean chunk max)", "qdot_max_abs_rad_s"),
        ("raw qddot RMS (rad/s^2)", "qddot_rms_rad_s2"),
        ("raw qddot max (rad/s^2, mean chunk max)", "qddot_max_abs_rad_s2"),
        ("raw jerk RMS (rad/s^3)", "jerk_rms_rad_s3"),
        ("raw jerk max (rad/s^3, mean chunk max)", "jerk_max_abs_rad_s3"),
        ("raw 4-10 Hz energy (rad^2)", "mean_4_10_hz_energy_rad2"),
        ("raw target-stationary p-p max (rad, mean)", "target_stationary_detrended_peak_to_peak_max_rad"),
    )
    table_rows = [
        {
            "Metric": "fixed-input first-action max std (rad)",
            "Dataset target": "0",
            "SmolVLA": fmt(float(np.max(smol_first_std))),
            "ACT": fmt(act_first_std),
        },
        {
            "Metric": "fixed-input full-chunk max std (rad)",
            "Dataset target": "0",
            "SmolVLA": fmt(float(np.max(smol_full_std))),
            "ACT": fmt(act_full_std),
        },
    ]
    for label, key in metrics:
        table_rows.append(
            {
                "Metric": label,
                "Dataset target": fmt(aggregate_value(raw, "dataset_target", key)),
                "SmolVLA": fmt(aggregate_value(raw, "smolvla_raw_fixed_noise", key)),
                "ACT": fmt(aggregate_value(raw, "act_raw", key)),
            }
        )
    table_rows.extend(
        [
            {
                "Metric": "phase score",
                "Dataset target": "54/54 by definition",
                "SmolVLA": "54/54",
                "ACT": f"{phase_eval['phase_score']['successful_qualitative_probes']}/54",
            },
            {
                "Metric": "visible raw smoothness",
                "Dataset target": decision["DATASET_TARGET_VISIBLE_SMOOTHNESS"],
                "SmolVLA": decision["SMOLVLA_VISIBLE_SMOOTHNESS"],
                "ACT": decision["ACT_VISIBLE_SMOOTHNESS"],
            },
            {
                "Metric": "Isaac continuous rollout",
                "Dataset target": "N/A",
                "SmolVLA": "COMPLETE_WITH_VISIBLE_JITTER (frozen prior evidence)",
                "ACT": isaac["status"] if isaac is not None else "NOT_RUN_RAW_GATE_NOT_PASSED",
            },
        ]
    )
    COMPARISON_OUTPUT.mkdir(parents=True, exist_ok=True)
    with (COMPARISON_OUTPUT / "dataset_target_vs_smolvla_vs_act.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    phase_lines = []
    for phase, final_label in PHASE_ORDER:
        row = phase_eval["phases"][phase]
        phase_lines.append(
            f"{final_label}: {row['successful_qualitative_probes']}/{row['sample_count']} successful; "
            f"first={row['first_action_rmse_rad']:.6f}, first4={row['first_4_frame_rmse_rad']:.6f}, "
            f"full={row['full_available_chunk_rmse_rad']:.6f}, arms={row['arm_full_available_chunk_rmse_rad']:.6f}, "
            f"Dex3={row['dex3_full_available_chunk_rmse_rad']:.6f} rad"
        )
    repeat = repeatability["conditions"]
    agg_target = raw["aggregates"]["dataset_target"]
    agg_smol = raw["aggregates"]["smolvla_raw_fixed_noise"]
    agg_act = raw["aggregates"]["act_raw"]
    behavior = (
        decision["ISAAC_BEHAVIOR_LIKE"]
        if isaac is not None
        else {key: "NOT_RUN" for key in (
            "LEFT_APPROACH_LIKE", "LEFT_GRASP_MOTION_LIKE", "LEFT_TRANSPORT_LIKE",
            "RIGHT_HANDOFF_APPROACH_LIKE", "BIMANUAL_HANDOFF_POSTURE_LIKE", "LEFT_RELEASE_LIKE",
            "RIGHT_TRANSPORT_TO_BIN_LIKE", "RIGHT_RELEASE_LIKE"
        )}
    )
    isaac_run = "YES" if isaac is not None else "NO"
    isaac_duration = f"{isaac['duration_seconds_at_30_hz']:.3f} s" if isaac else "N/A"
    isaac_jitter = decision["ISAAC_VISIBLE_JITTER"] if isaac else "N/A"
    isaac_abort = json.dumps(isaac["safety_abort"]) if isaac and isaac["safety_abort"] else "NO"
    next_step = (
        "Train Dataset-A -> ACT-A and use ACT for the primary A/B experiment."
        if raw_class == "PASS"
        else "Stop policy replacement and audit Dataset-B temporal targets."
        if raw_class == "FAIL"
        else "Keep ACT diagnostic-only while isolating the residual instability before the A/B experiment."
    )
    report = f"""ACT IMPLEMENTATION

LeRobot commit: {implementation['lerobot_commit']}
ACT class: {implementation['model_class']} / {implementation['configuration_class']}
state dim: 28
action dim: 28
chunk size: 50
temporal ensemble support: YES; official ACTTemporalEnsembler, reference k=0.01, requires n_action_steps=1


ACT TRAINING

status: {health['status']}
steps: {health['configured_steps']}
final loss: {health['final_logged_metrics']['loss']}
selected checkpoint: {health['selected_checkpoint']}
checkpoint SHA256: {health['selected_model_sha256']}


ACT PHASE PROBE

{chr(10).join(phase_lines)}

phase score: {phase_eval['phase_score']['successful_qualitative_probes']}/54 ({phase_eval['phase_score']['phases_with_all_6_successful']}/9 phases all-six)


FIXED INPUT REPEATABILITY

initial std: first max={repeat['initial']['first_action_standard_deviation']['maximum_joint_rad']:.3e} rad; full max={repeat['initial']['full_chunk_standard_deviation']['maximum_offset_joint_rad']:.3e} rad
approach std: first max={repeat['left_approach']['first_action_standard_deviation']['maximum_joint_rad']:.3e} rad; full max={repeat['left_approach']['full_chunk_standard_deviation']['maximum_offset_joint_rad']:.3e} rad
plateau std: first max={repeat['doll_region_plateau']['first_action_standard_deviation']['maximum_joint_rad']:.3e} rad; full max={repeat['doll_region_plateau']['full_chunk_standard_deviation']['maximum_offset_joint_rad']:.3e} rad

ACT_FIXED_INPUT_STOCHASTICITY: {repeatability['ACT_FIXED_INPUT_STOCHASTICITY']}


RAW SINGLE-CHUNK SMOOTHNESS

Dataset target: {decision['DATASET_TARGET_VISIBLE_SMOOTHNESS']}
SmolVLA: {decision['SMOLVLA_VISIBLE_SMOOTHNESS']}
ACT: {decision['ACT_VISIBLE_SMOOTHNESS']}

direction reversals: Dataset arms={agg_target['arm_direction_reversals_per_s_mean']['mean_across_chunks']:.3f}/s Dex3={agg_target['dex3_direction_reversals_per_s_mean']['mean_across_chunks']:.3f}/s; SmolVLA arms={agg_smol['arm_direction_reversals_per_s_mean']['mean_across_chunks']:.3f}/s Dex3={agg_smol['dex3_direction_reversals_per_s_mean']['mean_across_chunks']:.3f}/s; ACT arms={agg_act['arm_direction_reversals_per_s_mean']['mean_across_chunks']:.3f}/s Dex3={agg_act['dex3_direction_reversals_per_s_mean']['mean_across_chunks']:.3f}/s
max step: Dataset={agg_target['adjacent_step_max_abs_rad']['mean_across_chunks']:.6f}; SmolVLA={agg_smol['adjacent_step_max_abs_rad']['mean_across_chunks']:.6f}; ACT={agg_act['adjacent_step_max_abs_rad']['mean_across_chunks']:.6f} rad
qdot: Dataset RMS/max={agg_target['qdot_rms_rad_s']['mean_across_chunks']:.4f}/{agg_target['qdot_max_abs_rad_s']['maximum_across_chunks']:.4f}; SmolVLA={agg_smol['qdot_rms_rad_s']['mean_across_chunks']:.4f}/{agg_smol['qdot_max_abs_rad_s']['maximum_across_chunks']:.4f}; ACT={agg_act['qdot_rms_rad_s']['mean_across_chunks']:.4f}/{agg_act['qdot_max_abs_rad_s']['maximum_across_chunks']:.4f} rad/s
qddot: Dataset RMS/max={agg_target['qddot_rms_rad_s2']['mean_across_chunks']:.3f}/{agg_target['qddot_max_abs_rad_s2']['maximum_across_chunks']:.3f}; SmolVLA={agg_smol['qddot_rms_rad_s2']['mean_across_chunks']:.3f}/{agg_smol['qddot_max_abs_rad_s2']['maximum_across_chunks']:.3f}; ACT={agg_act['qddot_rms_rad_s2']['mean_across_chunks']:.3f}/{agg_act['qddot_max_abs_rad_s2']['maximum_across_chunks']:.3f} rad/s^2
jerk: Dataset RMS/max={agg_target['jerk_rms_rad_s3']['mean_across_chunks']:.2f}/{agg_target['jerk_max_abs_rad_s3']['maximum_across_chunks']:.2f}; SmolVLA={agg_smol['jerk_rms_rad_s3']['mean_across_chunks']:.2f}/{agg_smol['jerk_max_abs_rad_s3']['maximum_across_chunks']:.2f}; ACT={agg_act['jerk_rms_rad_s3']['mean_across_chunks']:.2f}/{agg_act['jerk_max_abs_rad_s3']['maximum_across_chunks']:.2f} rad/s^3
peak-to-peak: Dataset={agg_target['target_stationary_detrended_peak_to_peak_max_rad']['mean_across_chunks']:.6f}; SmolVLA={agg_smol['target_stationary_detrended_peak_to_peak_max_rad']['mean_across_chunks']:.6f}; ACT={agg_act['target_stationary_detrended_peak_to_peak_max_rad']['mean_across_chunks']:.6f} rad (detrended target-stationary joints, mean chunk max)

ACT_RAW_CHUNK_SMOOTHNESS:
{raw_class}


ISAAC ACT DIAGNOSTIC

run: {isaac_run}
duration: {isaac_duration}
visible jitter: {isaac_jitter}
safety abort: {isaac_abort}

LEFT_APPROACH_LIKE: {behavior['LEFT_APPROACH_LIKE']}
LEFT_GRASP_MOTION_LIKE: {behavior['LEFT_GRASP_MOTION_LIKE']}
LEFT_TRANSPORT_LIKE: {behavior['LEFT_TRANSPORT_LIKE']}
RIGHT_HANDOFF_APPROACH_LIKE: {behavior['RIGHT_HANDOFF_APPROACH_LIKE']}
BIMANUAL_HANDOFF_POSTURE_LIKE: {behavior['BIMANUAL_HANDOFF_POSTURE_LIKE']}
LEFT_RELEASE_LIKE: {behavior['LEFT_RELEASE_LIKE']}
RIGHT_TRANSPORT_TO_BIN_LIKE: {behavior['RIGHT_TRANSPORT_TO_BIN_LIKE']}
RIGHT_RELEASE_LIKE: {behavior['RIGHT_RELEASE_LIKE']}


ROOT CAUSE:

{root_cause}


REAL G1

NOT_STARTED_BY_DESIGN


NEXT STEP

{next_step}

{terminal}
"""
    atomic_text(OUTPUT / "FINAL_ACT_B_DIAGNOSTIC_REPORT.md", report)
    payload = {
        "schema_version": "act_b_final_diagnostic_v1",
        "status": "COMPLETE",
        "decision": decision,
        "implementation": implementation,
        "training_health": health,
        "selected_checkpoint": selection,
        "phase_evaluation": phase_eval,
        "repeatability": repeatability,
        "raw_smoothness": raw,
        "temporal_ensemble": temporal,
        "isaac": isaac,
        "comparison_table": table_rows,
        "immutable_dataset": identity,
        "final_report": str(OUTPUT / "FINAL_ACT_B_DIAGNOSTIC_REPORT.md"),
        "real_g1": "NOT_STARTED_BY_DESIGN",
        "terminal_status": terminal,
    }
    atomic_json(OUTPUT / "FINAL_ACT_B_DIAGNOSTIC_MANIFEST.json", payload)
    print(report)


if __name__ == "__main__":
    main()
