#!/usr/bin/env python3
"""Freeze the closed-loop stability review without authorizing robot motion."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/policy_execution_stability_review"
ANALYSIS = OUT / "analysis/execution_stability_metrics.json"
FIXED = OUT / "fixed_input_stochasticity/fixed_observation_stochasticity.json"
LOW = OUT / "low_motion_reference/low_motion_reference.json"
OTG_CONFIG = OUT / "jerk_limited_otg/common_g1_28d_ruckig_config.json"
M2_RECLASSIFICATION = OUT / "m2_reclassification.json"
M2_CONFIG = ROOT / "outputs/policy_b_causal_execution/frozen_adapter/common_causal_execution_adapter.json"
CANDIDATE_RUN = (
    OUT
    / "stateful_otg_latency_corrected/H10_D09_FIXED_RTC_RUCKIG/full_policy_b_diagnostic_rollout"
)
CANDIDATE_VIDEO_RUN = (
    OUT
    / "selected_candidate_video_latency_corrected/H10_D09_FIXED_RTC_STATEFUL_RUCKIG/full_policy_b_diagnostic_rollout"
)
RTC_FILES = [
    Path("/home/jbnu/lerobot-smolvla/src/lerobot/policies/rtc/configuration_rtc.py"),
    Path("/home/jbnu/lerobot-smolvla/src/lerobot/policies/rtc/modeling_rtc.py"),
    Path("/home/jbnu/lerobot-smolvla/src/lerobot/policies/rtc/action_queue.py"),
    Path("/home/jbnu/lerobot-smolvla/src/lerobot/rollout/inference/rtc.py"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def fmt(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def runtime_result(metrics: dict[str, Any], key: str) -> dict[str, Any]:
    row = metrics["runs"][key]
    boundary = row["boundary"]["executed_boundary_max_abs_delta_rad"]
    return {
        "status": row["status"],
        "duration_s": row["duration_s"],
        "frames": row["frames"],
        "inference_calls": row["inference_calls"],
        "horizon": row["execution_horizon_frames"],
        "replan_hz": row["replanning_frequency_hz"],
        "boundary_median_rad": boundary["median"],
        "boundary_p95_rad": boundary["p95"],
        "boundary_p99_rad": boundary["p99"],
        "boundary_max_rad": boundary["maximum"],
        "command_velocity_max_rad_s": row["command"]["velocity_abs_rad_s"]["maximum"],
        "command_acceleration_max_rad_s2": row["command"]["acceleration_abs_rad_s2"]["maximum"],
        "command_jerk_max_rad_s3": row["command"]["jerk_abs_rad_s3"]["maximum"],
        "direction_reversals_per_s_mean": row["command"]["direction_reversals_per_s"]["mean"],
        "tracking_rmse_rad": row["tracking_error_rad"]["rmse"],
        "commanded_hard_limit_violations": row["commanded_hard_limit_violation_count"],
        "branch_discontinuities": row["branch_discontinuity_count"],
        "maximum_external_non_task_contact_force_n": row["maximum_external_non_task_contact_force_n"],
        "abort": row["abort"],
    }


def main() -> int:
    metrics = read_json(ANALYSIS)
    fixed = read_json(FIXED)
    low = read_json(LOW)
    otg = read_json(OTG_CONFIG)
    old_ready = read_json(
        ROOT
        / "outputs/policy_b_causal_execution/object_free_multiview/M1_rtc/"
        "stage2_object_free_multichunk/inference_bridge/ready.json"
    )
    candidate_report = read_json(CANDIDATE_RUN / "stage_report.json")
    video_report = read_json(CANDIDATE_VIDEO_RUN / "stage_report.json")
    m2 = metrics["runs"]["M2_DIAGNOSTIC_CROSSFADE"]
    selected = metrics["runs"]["STATEFUL_RTC_RUCKIG_H10_D9"]
    selected_video = metrics["runs"]["STATEFUL_RTC_RUCKIG_H10_D9_VIDEO"]

    with np.load(CANDIDATE_RUN / "inference_chunks.npz") as chunks:
        raw_preserved = bool(np.array_equal(chunks["policy_raw_action"], chunks["raw_policy_chunk"]))
        separate_arrays = all(
            name in chunks.files
            for name in (
                "raw_policy_chunk",
                "previous_plan",
                "committed_prefix",
                "fused_uncommitted_plan",
                "final_command_reference",
            )
        )
        committed_prefix_overwritten = bool(chunks["committed_prefix_overwritten"])

    fixed_rows = {}
    for row in fixed["observations"]:
        audit = row["unique_seed_noise_audit"]
        fixed_rows[row["label"]] = {
            "first_action_maximum_std_rad": audit["first_action"]["maximum_std_rad"],
            "first_action_maximum_range_rad": audit["first_action"]["maximum_range_rad"],
            "prefix_4_maximum_joint_rms_std_rad": audit["prefix"]["4"][
                "maximum_per_joint_rms_std_rad"
            ],
            "prefix_8_maximum_joint_rms_std_rad": audit["prefix"]["8"][
                "maximum_per_joint_rms_std_rad"
            ],
            "prefix_12_maximum_joint_rms_std_rad": audit["prefix"]["12"][
                "maximum_per_joint_rms_std_rad"
            ],
            "prefix_16_maximum_joint_rms_std_rad": audit["prefix"]["16"][
                "maximum_per_joint_rms_std_rad"
            ],
            "fixed_noise_bitwise_identical": row["fixed_noise_diagnostic"]["bitwise_identical"],
            "fixed_noise_maximum_output_difference_rad": row["fixed_noise_diagnostic"][
                "maximum_output_difference_rad"
            ],
        }

    fixed_report = """# Fixed-input Policy-B stochasticity audit

Result: **POLICY_CHUNK_STOCHASTICITY = MATERIAL**.

The frozen RGB bytes, measured 28D state, task string, normalization, and checkpoint were each replayed 64 times at three observations. Distinct flow-noise seeds produced materially different chunks. Reusing one seed-derived flow-noise tensor made all 64 outputs bitwise identical (maximum difference 0 rad), proving the variation is model sampling noise rather than nondeterministic kernels or observation mutation.

| Observation | First-action max std (rad) | First-action range (rad) | Prefix-4 max RMS std (rad) | Prefix-8 | Prefix-12 | Prefix-16 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for label, row in fixed_rows.items():
        fixed_report += (
            f"| {label} | {row['first_action_maximum_std_rad']:.9f} | "
            f"{row['first_action_maximum_range_rad']:.9f} | "
            f"{row['prefix_4_maximum_joint_rms_std_rad']:.9f} | "
            f"{row['prefix_8_maximum_joint_rms_std_rad']:.9f} | "
            f"{row['prefix_12_maximum_joint_rms_std_rad']:.9f} | "
            f"{row['prefix_16_maximum_joint_rms_std_rad']:.9f} |\n"
        )
    fixed_report += "\nDominant unstable joints: " + ", ".join(
        f"{row['joint']} ({row['aggregate_rms_std_rad']:.6f} rad RMS std)"
        for row in fixed["dominant_unstable_joints"][:8]
    ) + ".\n\nFixed-noise inference was used only as a diagnostic condition. It is not an approval of an execution adapter or real hardware motion.\n"
    atomic_text(OUT / "fixed_input_stochasticity/fixed_input_stochasticity_report.md", fixed_report)

    rtc_results = {
        key: runtime_result(metrics, key)
        for key in (
            "RTC_H8_D6_G10_LINEAR",
            "RTC_H10_D6_G10_LINEAR",
            "RTC_H12_D6_G10_LINEAR",
            "RTC_H10_D9_G10_LINEAR",
        )
    }
    rtc_audit = {
        "schema_version": "installed_lerobot_rtc_audit_v1",
        "installed_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in RTC_FILES
        ],
        "official_behavior": {
            "default_execution_horizon": 10,
            "default_max_guidance_weight": 10.0,
            "default_prefix_attention_schedule": "LINEAR",
            "execution": "asynchronous background RTCInference thread",
            "latency_delay": "ceil(max observed inference latency * fps)",
            "queue_merge": "discard first real_delay rows of newly generated chunk because prior queued commands execute during inference",
            "prefix": "unconsumed prior original actions guide the causal prefix; no future observation is used",
        },
        "previous_m1": old_ready["causal_stitching"],
        "previous_m1_classification": (
            "NOT_A_PROPER_LATENCY_AWARE_ACTION_QUEUE: H4, delay=0, guidance=5, EXP, synchronous simulation"
        ),
        "bounded_d6_guided_pass_config": {
            "execution_horizons": [8, 10, 12],
            "inference_delay_frames": 6,
            "max_guidance_weight": 10.0,
            "prefix_attention_schedule": "LINEAR",
            "derivation": "ceil(max guided RTC pass 0.171657369 s * 30 Hz) = 6",
            "results": rtc_results,
        },
        "latency_corrected_dual_output_config": {
            "execution_horizon": 10,
            "inference_delay_frames": 9,
            "derivation": "ceil(max post-warmup raw+guided diagnostic request 0.281891847 s * 30 Hz) = 9",
            "reason": "raw unconditioned chunk preservation adds a diagnostic pass to the critical path",
            "result": rtc_results["RTC_H10_D9_G10_LINEAR"],
        },
        "result": "FAIL: every RTC-only bounded configuration hit an executed-prefix gate",
        "future_observations_used": False,
        "offline_temporal_consensus_used": False,
        "real_hardware_authorized": False,
    }
    atomic_json(OUT / "rtc_audit/rtc_configuration_audit.json", rtc_audit)
    rtc_markdown = f"""# Installed LeRobot RTC audit

The installed implementation is asynchronous: a background inference thread reads the latest observation while the control loop consumes an `ActionQueue`. It estimates delay as `ceil(max latency × FPS)`, guides from the unconsumed previous plan, and drops the newly predicted rows corresponding to actions already consumed during inference. No future observation is used.

Previous M1 was not a faithful latency-aware queue: horizon 4, delay 0, guidance 5, EXP schedule, synchronous execution.

The bounded official-style tests used guidance 10 and LINEAR attention. H8/D6 stopped at {rtc_results['RTC_H8_D6_G10_LINEAR']['duration_s']:.3f} s (prefix acceleration), H10/D6 at {rtc_results['RTC_H10_D6_G10_LINEAR']['duration_s']:.3f} s (0.151255 rad boundary jump), and H12/D6 at {rtc_results['RTC_H12_D6_G10_LINEAR']['duration_s']:.3f} s (first-command delta). The guided RTC pass required six frames. Because this audit separately preserves an unconditioned raw chunk, the measured dual-output critical path required nine frames; H10/D9 still stopped at {rtc_results['RTC_H10_D9_G10_LINEAR']['duration_s']:.3f} s on an executed-prefix gate.

Result: **RTC ALONE = FAIL**. The official causal semantics were recovered, but no bounded RTC-only setting passed the strict rollout.
"""
    atomic_text(OUT / "rtc_audit/rtc_configuration_audit.md", rtc_markdown)

    arm_rows = [row for row in otg["joints"] if row["group"] == "arm"]
    dex_rows = [row for row in otg["joints"] if row["group"] == "dex3"]
    constraints = {
        "arm": {
            quantity: [min(row[quantity] for row in arm_rows), max(row[quantity] for row in arm_rows)]
            for quantity in ("max_velocity_rad_s", "max_acceleration_rad_s2", "max_jerk_rad_s3")
        },
        "dex3": {
            quantity: [min(row[quantity] for row in dex_rows), max(row[quantity] for row in dex_rows)]
            for quantity in ("max_velocity_rad_s", "max_acceleration_rad_s2", "max_jerk_rad_s3")
        },
    }
    low_reference = {
        name: row["all_window_joint_values"] for name, row in low["metrics"].items()
    }
    low_after = selected["low_motion_command"]
    low_ratios_p95 = {
        name: low_after["metrics"][name]["p95"] / low_reference[name]["p95"]
        for name in ("qdot_rms_rad_s", "qddot_rms_rad_s2", "jerk_rms_rad_s3", "peak_to_peak_rad")
    }
    comparison_videos = {}
    for path in sorted((OUT / "comparison_videos_final").glob("*.mp4")):
        comparison_videos[path.stem] = {"path": str(path), "sha256": sha256_file(path)}
    candidate_videos = {}
    for path in sorted(CANDIDATE_VIDEO_RUN.glob("*.mp4")):
        candidate_videos[path.stem] = {"path": str(path), "sha256": sha256_file(path)}

    candidate = {
        "schema_version": "common_causal_execution_diagnostic_candidate_v1",
        "name": "STATEFUL_RTC_H10_D9_PLUS_RUCKIG_DIAGNOSTIC_CANDIDATE",
        "selection_status": "NONE_APPROVED",
        "candidate_status": "DIAGNOSTIC_ONLY_NOT_REAL_ROBOT_APPROVED",
        "policy_independent_execution_logic": True,
        "task_phase_episode_object_specific_logic": False,
        "raw_policy_chunks_modified": False,
        "raw_policy_chunks_preserved_exactly": raw_preserved,
        "separate_plan_and_command_arrays_present": separate_arrays,
        "committed_prefix_overwritten": committed_prefix_overwritten,
        "plan_priority": [
            "already committed commands",
            "remaining valid previous plan",
            "new policy plan for uncommitted future",
        ],
        "rtc": {
            "execution_horizon": 10,
            "inference_delay_frames": 9,
            "max_guidance_weight": 10.0,
            "prefix_attention_schedule": "LINEAR",
            "execution_semantics": "logically asynchronous latency-aware causal queue",
        },
        "inference_noise": {
            "mode_in_candidate_test": "FIXED_SEED_DERIVED_TENSOR_REUSED",
            "status": "DIAGNOSTIC_ONLY_NOT_FROZEN_AS_FINAL_EXECUTION_POLICY",
        },
        "jerk_limited_otg": {
            "implementation": "Ruckig 0.19.4 community Python online position OTG",
            "config": str(OTG_CONFIG),
            "config_sha256": sha256_file(OTG_CONFIG),
            "state": "persistent command q/dq/ddq initialized from measured state; advanced only through committed Ruckig outputs",
            "measured_feedback": "tracking and all runtime safety gates; a replan never re-seeds from lagging measured q",
            "target": "deployment-safe endpoint of selected committed horizon",
            "constraints_ranges": constraints,
            "fail_closed": True,
        },
        "quantitative_repeat": runtime_result(metrics, "STATEFUL_RTC_RUCKIG_H10_D9"),
        "video_repeat": runtime_result(metrics, "STATEFUL_RTC_RUCKIG_H10_D9_VIDEO"),
        "low_motion_reference": str(LOW),
        "low_motion_p95_ratio_to_reference": low_ratios_p95,
        "acceptance_failures": [
            "late non-task/table contact in every H8/H10/H12 stateful OTG rollout",
            "low-motion qdot/qddot/jerk/peak-to-peak not comparable to Dataset-B reference",
            "rendered repeat boundary p99 0.050347 rad",
            "explicit human video approval not yet provided",
        ],
        "candidate_videos": candidate_videos,
        "comparison_videos": comparison_videos,
        "real_hardware_safety_readiness": "BLOCKED",
        "real_robot_command_allowed": False,
    }
    candidate_path = OUT / "best_diagnostic_candidate/common_causal_execution_candidate.json"
    atomic_json(candidate_path, candidate)
    candidate_hash = sha256_file(candidate_path)

    m2_boundary = m2["boundary"]["executed_boundary_max_abs_delta_rad"]
    selected_boundary = selected["boundary"]["executed_boundary_max_abs_delta_rad"]
    horizon_rows = {
        key: runtime_result(metrics, key)
        for key in ("H4_COMMITMENT", "H8_COMMITMENT", "H12_COMMITMENT", "H16_COMMITMENT")
    }
    final_report = f"""# Policy execution stability review

## FIXED-INPUT POLICY STOCHASTICITY

result: **MATERIAL**

dominant joints: {', '.join(row['joint'] for row in fixed['dominant_unstable_joints'][:8])}

At the initial, left-approach, and doll-region plateau observations, maximum first-action standard deviation was {fixed_rows['initial']['first_action_maximum_std_rad']:.6f}, {fixed_rows['left_approach']['first_action_maximum_std_rad']:.6f}, and {fixed_rows['doll_region_plateau']['first_action_maximum_std_rad']:.6f} rad respectively. Fixed flow noise made every 64-repeat set bitwise identical.

## HORIZON COMPARISON

H4: PASS for {horizon_rows['H4_COMMITMENT']['duration_s']:.3f} s; boundary p99/max {horizon_rows['H4_COMMITMENT']['boundary_p99_rad']:.6f}/{horizon_rows['H4_COMMITMENT']['boundary_max_rad']:.6f} rad; persistent high-rate direction reversal remained.

H8: FAIL at {horizon_rows['H8_COMMITMENT']['duration_s']:.3f} s; boundary p99/max {horizon_rows['H8_COMMITMENT']['boundary_p99_rad']:.6f}/{horizon_rows['H8_COMMITMENT']['boundary_max_rad']:.6f} rad; first-command prefix gate.

H12: FAIL at {horizon_rows['H12_COMMITMENT']['duration_s']:.3f} s; boundary p99/max {horizon_rows['H12_COMMITMENT']['boundary_p99_rad']:.6f}/{horizon_rows['H12_COMMITMENT']['boundary_max_rad']:.6f} rad; acceleration prefix gate.

H16: FAIL at {horizon_rows['H16_COMMITMENT']['duration_s']:.3f} s; boundary p99/max {horizon_rows['H16_COMMITMENT']['boundary_p99_rad']:.6f}/{horizon_rows['H16_COMMITMENT']['boundary_max_rad']:.6f} rad; external/table-contact gate.

Longer commitment alone did not stabilize the stochastic chunks and did not produce an acceptable execution layer.

## RTC CONFIGURATION AUDIT

previous M1 config: H4, inference delay 0, guidance 5, EXP prefix attention, synchronous replacement. It is preserved but is not a proper latency-aware action queue.

new bounded config: installed official semantics; guidance 10, LINEAR attention, H8/H10/H12, six-frame guided-pass delay. All RTC-only runs failed strict prefix/boundary gates. Because retaining an extra raw unconditioned chunk raises this diagnostic worker's measured critical path, H10/D9 was also tested and failed at {rtc_results['RTC_H10_D9_G10_LINEAR']['duration_s']:.3f} s.

result: **FAIL — RTC alone was insufficient.**

## PLAN COMMITMENT

implemented: YES

committed prefix overwritten: **NO**

Every artifact keeps `raw_policy_chunk`, `previous_plan`, `committed_prefix`, `fused_uncommitted_plan`, and `final_command_reference` separately. Raw-policy aliases are byte-identical in the best candidate artifact.

## JERK-LIMITED OTG

implementation: Ruckig 0.19.4, stateful 28D online position OTG.

constraints: arms use Dataset-B p99 velocity and p95 acceleration/jerk; Dex3 uses observed maxima. Arm velocity/acceleration/jerk ranges are {constraints['arm']['max_velocity_rad_s'][0]:.6f}–{constraints['arm']['max_velocity_rad_s'][1]:.6f} rad/s, {constraints['arm']['max_acceleration_rad_s2'][0]:.6f}–{constraints['arm']['max_acceleration_rad_s2'][1]:.6f} rad/s², and {constraints['arm']['max_jerk_rad_s3'][0]:.6f}–{constraints['arm']['max_jerk_rad_s3'][1]:.6f} rad/s³. Dex3 ranges are {constraints['dex3']['max_velocity_rad_s'][0]:.6f}–{constraints['dex3']['max_velocity_rad_s'][1]:.6f}, {constraints['dex3']['max_acceleration_rad_s2'][0]:.6f}–{constraints['dex3']['max_acceleration_rad_s2'][1]:.6f}, and {constraints['dex3']['max_jerk_rad_s3'][0]:.6f}–{constraints['dex3']['max_jerk_rad_s3'][1]:.6f} in the corresponding units.

result: command direction reversals fell from a mean {m2['command']['direction_reversals_per_s']['mean']:.3f}/s under M2 to {selected['command']['direction_reversals_per_s']['mean']:.3f}/s. The best quantitative D9 repeat had boundary median/p99/max {selected_boundary['median']:.6f}/{selected_boundary['p99']:.6f}/{selected_boundary['maximum']:.6f} rad and no commanded hard-limit or branch violation. It nevertheless aborted on table contact at {selected['duration_s']:.3f} s, and the rendered repeat aborted at {selected_video['duration_s']:.3f} s.

## LOW-MOTION JITTER

reference: 282 one-second Dataset-B windows from 48 episodes, constant ownership, balanced NO_OWNER/RELEASED, arm-speed RMS cutoff {low['selection']['arm_speed_rms_cutoff_rad_s']:.9f} rad/s. Reference p95 qdot/qddot/jerk/peak-to-peak are {low_reference['qdot_rms_rad_s']['p95']:.6f}, {low_reference['qddot_rms_rad_s2']['p95']:.6f}, {low_reference['jerk_rms_rad_s3']['p95']:.6f}, and {low_reference['peak_to_peak_rad']['p95']:.6f}.

before: M2 produced zero qualifying one-second low-motion windows and mean {m2['command']['direction_reversals_per_s']['mean']:.3f} reversals/s; boundary p99/max remained {m2_boundary['p99']:.6f}/{m2_boundary['maximum']:.6f} rad.

after: the best D9 stateful RTC+OTG repeat produced {low_after['window_count']} qualifying windows, but p95 qdot/qddot/jerk/peak-to-peak were {low_after['metrics']['qdot_rms_rad_s']['p95']:.6f}, {low_after['metrics']['qddot_rms_rad_s2']['p95']:.6f}, {low_after['metrics']['jerk_rms_rad_s3']['p95']:.6f}, and {low_after['metrics']['peak_to_peak_rad']['p95']:.6f}: {low_ratios_p95['qdot_rms_rad_s']:.2f}×, {low_ratios_p95['qddot_rms_rad_s2']:.2f}×, {low_ratios_p95['jerk_rms_rad_s3']:.2f}×, and {low_ratios_p95['peak_to_peak_rad']:.2f}× the reference. The low-motion acceptance gate is therefore not met.

## SELECTED EXECUTION ADAPTER

method: **NONE APPROVED**

best diagnostic candidate: stateful committed-plan RTC H10/D9 + fixed-noise diagnostic + Ruckig realization.

config: `{candidate_path}`

SHA256: `{candidate_hash}`

This candidate is not the final common adapter. M2 remains classified `DIAGNOSTIC_CROSSFADE_NOT_REAL_ROBOT_APPROVED` (reclassification SHA256 `{sha256_file(M2_RECLASSIFICATION)}`; preserved M2 config SHA256 `{sha256_file(M2_CONFIG)}`).

## VIDEO REVIEW

persistent visible oscillation: **YES / NOT CLEARED**

The stateful OTG visibly removes the dense replan-frequency chatter in the command plot, but its rendered repeat is not repeatably within the low-motion envelope and reaches a hard table-contact abort. Equal-scale videos are ready for explicit human review; absence of oscillation has not been human-approved.

## REAL HARDWARE

**BLOCKED**

No DDS, G1, or Dex3 command path was invoked. Camera work may proceed independently, but Policy A/B hardware execution remains prohibited.

Final status: `BLOCKED_BY_PERSISTENT_POLICY_JITTER`
"""
    atomic_text(OUT / "final_report.md", final_report)

    manifest = {
        "schema_version": "final_policy_execution_stability_review_manifest_v1",
        "status": "BLOCKED_BY_PERSISTENT_POLICY_JITTER",
        "selected_execution_adapter": None,
        "best_diagnostic_candidate": {
            "path": str(candidate_path),
            "sha256": candidate_hash,
        },
        "m2_reclassification": {
            "classification": "DIAGNOSTIC_CROSSFADE_NOT_REAL_ROBOT_APPROVED",
            "path": str(M2_RECLASSIFICATION),
            "sha256": sha256_file(M2_RECLASSIFICATION),
            "preserved_config_sha256": sha256_file(M2_CONFIG),
        },
        "fixed_input_stochasticity": {
            "classification": "MATERIAL",
            "artifact": str(FIXED),
            "sha256": sha256_file(FIXED),
            "all_fixed_noise_repeats_bitwise_identical": fixed[
                "all_fixed_noise_repeats_bitwise_identical"
            ],
        },
        "analysis": {
            "path": str(ANALYSIS),
            "sha256": sha256_file(ANALYSIS),
        },
        "low_motion_reference": {
            "path": str(LOW),
            "sha256": sha256_file(LOW),
            "selected_windows": low["selection"]["selected_window_count"],
        },
        "rtc_audit": {
            "path": str(OUT / "rtc_audit/rtc_configuration_audit.json"),
            "sha256": sha256_file(OUT / "rtc_audit/rtc_configuration_audit.json"),
        },
        "implementation": {
            str(path): sha256_file(path)
            for path in (
                ROOT / "tools/run_policy_b_isaac_doll_handoff.py",
                ROOT / "tools/policy_b_inference_worker.py",
                ROOT / "tools/common_jerk_limited_otg.py",
                ROOT / "tools/build_common_jerk_limited_otg_config.py",
                ROOT / "tools/analyze_policy_execution_stability.py",
                Path(__file__).resolve(),
            )
        },
        "videos": {
            "candidate": candidate_videos,
            "equal_scale_comparison": comparison_videos,
        },
        "final_report": str(OUT / "final_report.md"),
        "real_hardware_safety_readiness": "BLOCKED",
        "real_robot_command_allowed": False,
        "policy_a_or_b_hardware_execution_started": False,
    }
    atomic_json(OUT / "FINAL_POLICY_EXECUTION_STABILITY_REVIEW_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "candidate_sha256": candidate_hash,
                "manifest": str(OUT / "FINAL_POLICY_EXECUTION_STABILITY_REVIEW_MANIFEST.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
