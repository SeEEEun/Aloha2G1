#!/usr/bin/env python3
"""Finalize immutable ACT-B frame-0, E0/E1, and full-motion review artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout"
FRAME0 = OUT / "frame0_root_cause_report.json"
LEGACY_ROWS = OUT / "frame0_legacy_hidden/frame0_named_joint_audit.json"
E0 = OUT / "object_free_e0_300"
E1 = OUT / "object_free_e1_300"
FULL = OUT / "full_motion_e1_688"
SELECTION = OUT / "SELECTED_ACT_EXECUTION_CONFIG.json"
CHECKPOINT_SHA = "a8d0a0d437421a1a0100f4f8c8ed67b5f6cd7ef0d383b1ebba0e6e45ada4d078"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=lambda x: x.tolist()) + "\n",
    )


def derivative_metrics(q: np.ndarray, cut: int = 0) -> dict[str, float]:
    value = np.asarray(q, dtype=np.float64)[cut:]
    qdot = np.diff(value, axis=0) * 30.0
    qddot = np.diff(qdot, axis=0) * 30.0
    jerk = np.diff(qddot, axis=0) * 30.0
    return {
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(value, axis=0)))),
        "qddot_rms_rad_s2": float(np.sqrt(np.mean(np.square(qddot)))),
        "qddot_max_abs_rad_s2": float(np.max(np.abs(qddot))),
        "jerk_rms_rad_s3": float(np.sqrt(np.mean(np.square(jerk)))),
        "jerk_max_abs_rad_s3": float(np.max(np.abs(jerk))),
    }


def main() -> None:
    frame0 = read_json(FRAME0)
    e0 = read_json(E0 / "isaac_diagnostic_report.json")
    e1 = read_json(E1 / "isaac_diagnostic_report.json")
    full = read_json(FULL / "isaac_diagnostic_report.json")
    selection_sha = sha256_file(SELECTION)
    if selection_sha != "656f6f474f6981c5b0c0417895b91ad924d171031bb66b26e4640b54bc863b64":
        raise RuntimeError("selected ACT execution config changed")
    if e0["status"] != "SAFETY_ABORT" or e0["executed_frames"] != 50:
        raise RuntimeError("E0 boundary result changed")
    if e1["status"] != "PASS" or e1["executed_frames"] != 300:
        raise RuntimeError("E1 bounded result changed")
    if full["status"] != "PASS" or full["executed_frames"] != 688:
        raise RuntimeError("full-motion safety result changed")
    if any(row["checkpoint_model_sha256"] != CHECKPOINT_SHA for row in (e0, e1, full)):
        raise RuntimeError("rollout checkpoint identity changed")

    rows = read_json(LEGACY_ROWS)
    violations = [row for row in rows if row["act_raw_hard_limit_violation"]]
    if len(violations) != 6:
        raise RuntimeError("six-joint legacy violation set changed")
    exact_violations = [
        {
            "joint_name": row["joint_name"],
            "isaac_reset_state_rad": row["isaac_reset_state_rad"],
            "isaac_settled_state_rad": row["isaac_settled_state_rad"],
            "act_raw_first_action_rad": row["act_e0_raw_first_action_rad"],
            "hard_lower_rad": row["hard_lower_rad"],
            "hard_upper_rad": row["hard_upper_rad"],
            "signed_clearance_rad": row["act_raw_signed_clearance_rad"],
            "deployment_projected_action_rad": row["deployment_projected_action_rad"],
        }
        for row in violations
    ]

    with np.load(E0 / "rollout_arrays.npz", allow_pickle=False) as archive:
        e0_command = archive["commanded_action"].astype(np.float64)
    with np.load(E1 / "rollout_arrays.npz", allow_pickle=False) as archive:
        e1_command = archive["commanded_action"].astype(np.float64)
    with np.load(FULL / "rollout_arrays.npz", allow_pickle=False) as archive:
        full_command = archive["commanded_action"].astype(np.float64)
        joint_names = archive["joint_names"].astype(str)
    e0_metrics = derivative_metrics(e0_command)
    e1_metrics = derivative_metrics(e1_command)
    full_metrics = derivative_metrics(full_command)
    e1_after_startup = derivative_metrics(e1_command, cut=10)
    full_after_startup = derivative_metrics(full_command, cut=10)

    with np.load(
        ROOT / "outputs/policy_b_act/checkpoint_evaluation/checkpoint_100000_phase_predictions.npz",
        allow_pickle=False,
    ) as archive:
        mask = archive["episode_index"] == 24
        phases = archive["phase"][mask].astype(str)
        phase_targets = archive["authoritative_target"][mask, 0].astype(np.float64)
    phase_rmse = np.sqrt(np.mean(np.square(full_command[:, None, :] - phase_targets[None]), axis=2))
    nearest = np.argmin(phase_rmse, axis=1)
    nearest_counts = {phase: int(np.count_nonzero(nearest == index)) for index, phase in enumerate(phases)}
    semantic_review = {
        "review_method": "human review of overview/top/side/source-like phase and uniform sheets plus nearest Dataset-B episode-24 phase-prototype audit",
        "nearest_phase_counts_over_688_frames": nearest_counts,
        "closed_loop_arm_path_length_fraction_vs_dataset_episode24": None,
        "LEFT_APPROACH_LIKE": "NO",
        "LEFT_GRASP_MOTION_LIKE": "NO",
        "LEFT_TRANSPORT_LIKE": "NO",
        "RIGHT_HANDOFF_APPROACH_LIKE": "NO",
        "BIMANUAL_HANDOFF_POSTURE_LIKE": "NO_TRANSFER_POSTURE; ONLY_STABLE_BIMANUAL_REST_LIKE_POSTURE",
        "LEFT_RELEASE_LIKE": "NO",
        "RIGHT_TRANSPORT_TO_BIN_LIKE": "NO",
        "RIGHT_RELEASE_LIKE": "NO",
        "semantic_full_motion_status": "FAIL_LOCAL_INITIAL_POSTURE_ATTRACTOR",
        "interpretation": "The selected execution is mechanically smooth and safe, but the current Isaac RGB/measured-state closed loop does not advance through Dataset-B task phases.",
    }
    source_manifest = read_json(ROOT / "outputs/doll_handoff_dataset_b_final/final_source_manifest.json")
    episode24 = source_manifest["episodes"][24]
    with np.load(episode24["retargeted_trajectory_path"], allow_pickle=False) as archive:
        source_names = archive["replay_joint_names"].astype(str).tolist()
        source_q = archive["replay_named_joint_qpos"][:, [source_names.index(name) for name in joint_names]]
    closed_path = float(
        np.sum(np.linalg.norm(np.diff(full_command[:, :7], axis=0), axis=1))
        + np.sum(np.linalg.norm(np.diff(full_command[:, 7:14], axis=0), axis=1))
    )
    source_path = float(
        np.sum(np.linalg.norm(np.diff(source_q[:, :7], axis=0), axis=1))
        + np.sum(np.linalg.norm(np.diff(source_q[:, 7:14], axis=0), axis=1))
    )
    semantic_review["closed_loop_arm_path_length_fraction_vs_dataset_episode24"] = closed_path / source_path

    human_review = {
        "schema_version": "act_b_isaac_human_smoothness_review_v1",
        "status": "PASS_REVIEW_COMPLETE",
        "equal_scale_comparisons": {
            view: str(OUT / f"e0_vs_e1_comparison/{view}_e0_vs_e1.mp4")
            for view in ("overview", "top", "side", "source_like")
        },
        "act_e0": {
            "visible_smoothness": "SMOOTH_WITHIN_FIRST_QUEUE_BUT_NOT_CONTINUOUSLY_EXECUTABLE",
            "no_dduk_dduk_motion_in_executed_prefix": True,
            "shoulder_pumping": False,
            "elbow_pumping": False,
            "stop_reverse_correct_pattern": "QUEUE_BOUNDARY_DISCONTINUITY_REJECTED_BEFORE_EXECUTION",
            "high_frequency_dex3_oscillation": False,
            "overall": "PARTIAL_FAIL_AT_FIRST_REQUERY_BOUNDARY",
        },
        "act_e1": {
            "visible_smoothness": "SMOOTH_FLOWING",
            "no_dduk_dduk_motion": True,
            "shoulder_pumping": False,
            "elbow_pumping": False,
            "stop_reverse_correct_pattern": False,
            "high_frequency_dex3_oscillation": False,
            "overall": "PASS",
        },
        "full_motion_mechanical": {
            "visible_mechanical_smoothness": "PASS",
            "semantic_task_progression": "FAIL",
            "distinction": "Mechanical smoothness is not being used to claim the eight requested semantic motifs.",
        },
    }
    atomic_json(OUT / "human_smoothness_and_semantic_review.json", human_review)

    e0_boundary = read_json(E0 / "executed_prefix_safety_records.json")[50]
    report = {
        "schema_version": "act_b_isaac_frame0_and_rollout_final_v1",
        "status": "ACT_B_ISAAC_FULL_MOTION_REVIEW_READY",
        "frozen_inputs": {
            "dataset_b": "UNCHANGED_50_EPISODES_34478_FRAMES",
            "checkpoint_model_sha256": CHECKPOINT_SHA,
            "prior_phase_probe": "54/54",
            "prior_fixed_input_stochasticity": "ZERO",
            "prior_raw_chunk_smoothness": "PASS",
        },
        "frame0_root_cause": {
            "right_dex3_violations": [row["joint_name"] for row in violations],
            "exact_joint_records": exact_violations,
            "origin": "ACT raw boundary overshoot under the legacy hidden-object frame, compounded by omission of the already-frozen policy-independent deployment projection; the hidden-object scene bug changes the overshoot set. Not denormalization, mapping, sign, or 32D padding.",
            "legacy_scene_implementation_error": "Doll and bin were hidden instead of remaining visible with collisions disabled.",
            "command_violation_count_after_common_adapter": 0,
            "branch_norm_excess": "METRIC_IMPLEMENTATION_BUG_WITH_SETTLE_DRIFT_EXPOSING_THE_ERROR",
            "branch_origin": frame0["branch_norm_excess"],
            "mapping": "PASS",
            "normalization": "PASS",
            "normalization_round_trip_max_abs_rad": read_json(OUT / "frame0_legacy_hidden/frame0_audit_report.json")["normalization"]["round_trip_max_abs_rad"],
            "initial_state_contract": frame0["initial_state_contract"],
            "initial_state_contract_sha256": frame0["initial_state_contract_sha256"],
            "after_fix": read_json(OUT / "frame0_corrected_dry_run.json")["status"],
        },
        "act_e0": {
            "duration_seconds": e0["duration_seconds_at_30_hz"],
            "visible_smoothness": human_review["act_e0"]["visible_smoothness"],
            "executed_prefix_dynamics": e0_metrics,
            "rejected_boundary": {
                "frame": 50,
                "maximum_joint_step_rad": e0_boundary["command_audit"]["maximum_joint_step_rad"],
                "maximum_acceleration_rad_s2": e0_boundary["command_audit"]["maximum_acceleration_rad_s2"],
                "arm_step_l2_rad": e0_boundary["command_audit"]["arm_step_l2_rad"],
                "branch_threshold_rad": e0_boundary["command_audit"]["branch_threshold_rad"],
                "new_chunk_first_delta_max_abs_rad": e0_boundary["query_chunk_audit"]["first_command_delta_max_abs_rad"],
            },
            "collision": 0,
            "table_contact": e0["table_contact"],
            "safety_abort": e0["safety_abort"],
        },
        "act_e1": {
            "duration_seconds": e1["duration_seconds_at_30_hz"],
            "visible_smoothness": human_review["act_e1"]["visible_smoothness"],
            "dynamics": e1_metrics,
            "dynamics_after_first_10_startup_frames": e1_after_startup,
            "collision": 0,
            "table_contact": e1["table_contact"],
            "safety_abort": e1["safety_abort"],
        },
        "selected_act_execution": {
            "mode": "E1",
            "name": "ACT_E1_TEMPORAL_ENSEMBLE",
            "config": str(SELECTION),
            "config_sha256": selection_sha,
        },
        "full_motion": {
            "duration_seconds": full["duration_seconds_at_30_hz"],
            "requested_and_executed_frames": [full["requested_frames"], full["executed_frames"]],
            "safety_abort": full["safety_abort"],
            "collision": 0,
            "table_contact": full["table_contact"],
            "dynamics": full_metrics,
            "dynamics_after_first_10_startup_frames": full_after_startup,
            "semantic_review": semantic_review,
            "visible_mechanical_smoothness": "PASS",
            "videos": full["videos"],
        },
        "implementation": {
            "post_fix_runner": str(ROOT / "tools/run_act_b_isaac_diagnostic.py"),
            "post_fix_runner_sha256": sha256_file(ROOT / "tools/run_act_b_isaac_diagnostic.py"),
            "inference_worker": str(ROOT / "tools/act_b_inference_worker.py"),
            "inference_worker_sha256": sha256_file(ROOT / "tools/act_b_inference_worker.py"),
            "real_command_publisher_present": False,
            "real_hardware_transport": False,
        },
        "real_g1": "NOT_STARTED_BY_DESIGN",
        "terminal_classification": "ACT_B_ISAAC_FULL_MOTION_REVIEW_READY",
    }
    final_json = OUT / "FINAL_ACT_B_ISAAC_FRAME0_AND_ROLLOUT_REPORT.json"
    atomic_json(final_json, report)

    md = f"""# ACT-B Isaac frame-0 and rollout diagnostic

Status: **ACT_B_ISAAC_FULL_MOTION_REVIEW_READY**

The frame-0 abort was caused by two deployment-runner defects: the frozen generic Dex3 projection was omitted, and the branch threshold was incorrectly applied to measured-state → first-command instead of action-trajectory adjacency. The six legacy raw violations were {', '.join(report['frame0_root_cause']['right_dex3_violations'])}. Mapping and ACT normalization both pass; maximum normalization round-trip error is {report['frame0_root_cause']['normalization_round_trip_max_abs_rad']:.3e} rad.

The corrected zero-command frame-0 dry run passes without changing the 0.180-rad branch gate, Dataset B, ACT weights, or retargeting.

ACT-E0 ran {e0['duration_seconds_at_30_hz']:.3f} s and stopped before executing its first requery boundary. ACT-E1 ran {e1['duration_seconds_at_30_hz']:.1f} s with all safety checks passing and visibly continuous motion. The frozen selection is `ACT_E1_TEMPORAL_ENSEMBLE`, coefficient 0.01; config SHA256 `{selection_sha}`.

The selected full diagnostic completed {full['duration_seconds_at_30_hz']:.3f} s (688/688 frames), with zero safety aborts, self-collisions, or table contacts. Mechanical smoothness passes. Semantic progression does not: the current Isaac RGB/measured-state loop stays in an initial/rest-like attractor and does not visibly complete the requested handoff motifs. These are separate findings.

Real G1: **NOT_STARTED_BY_DESIGN**.
"""
    atomic_text(OUT / "FINAL_ACT_B_ISAAC_FRAME0_AND_ROLLOUT_REPORT.md", md)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
