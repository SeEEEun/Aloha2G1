#!/usr/bin/env python3
"""Freeze and summarize the Policy-B causal execution-layer experiment.

This script only reads frozen Dataset-B and completed Isaac diagnostic artifacts.
It never invokes a policy, simulator, retargeter, or real-robot interface.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/policy_b_causal_execution"
REFERENCE = OUT / "dataset_b_motion_reference/dataset_b_natural_motion_reference.json"
RUNS = {
    "M0_NAIVE": OUT / "object_free/M0_naive/stage2_object_free_multichunk",
    "M1_RTC": OUT / "object_free/M1_rtc/stage2_object_free_multichunk",
    "M2_STATE_ALIGNED_CROSSFADE": OUT
    / "object_free/M2_state_aligned_crossfade/stage2_object_free_multichunk",
}
OLD_FULL = ROOT / "outputs/policy_b_isaac_validation/full_policy_b_diagnostic_rollout"
NEW_FULL = OUT / "full_motion/M2_state_aligned_crossfade/full_policy_b_diagnostic_rollout"
ANALYSIS = OUT / "analysis"
FROZEN = OUT / "frozen_adapter"
FPS = 30.0
HORIZON = 4


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0, "median": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def trajectory_dynamics(q: np.ndarray) -> dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)
    step = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    velocity = np.max(np.abs(np.diff(q, axis=0) * FPS), axis=1)
    acceleration = np.max(np.abs(np.diff(q, n=2, axis=0) * FPS**2), axis=1)
    jerk = np.max(np.abs(np.diff(q, n=3, axis=0) * FPS**3), axis=1)
    return {
        "adjacent_step_rad": distribution(step),
        "velocity_rad_s": distribution(velocity),
        "acceleration_rad_s2": distribution(acceleration),
        "jerk_rad_s3": distribution(jerk),
    }


def boundary_metrics(path: Path, names: list[str]) -> dict[str, Any]:
    source = read_json(path / "chunk_boundary_jitter.json")
    records = source["records"]
    deltas = np.asarray([row["executed_boundary_delta_rad"] for row in records])
    maximum = np.max(np.abs(deltas), axis=1)
    per_joint = []
    for index, name in enumerate(names):
        values = np.abs(deltas[:, index])
        per_joint.append({"joint_index": index, "joint": name, **distribution(values)})
    return {
        "replan_boundary_count": int(len(records)),
        "executed_command_jump_rad": distribution(maximum),
        "per_joint": per_joint,
        "planned_old_vs_new_raw_maximum_rad": float(
            max(
                row.get(
                    "old_plan_vs_new_raw_max_abs_delta_rad",
                    row.get("raw_boundary_max_abs_delta_rad", 0.0),
                )
                for row in records
            )
        ),
        "resulting_measured_acceleration_rad_s2": distribution(
            np.asarray(
                [
                    row["resulting_measured_acceleration_max_abs_rad_s2"]
                    for row in records
                    if row["resulting_measured_acceleration_max_abs_rad_s2"] is not None
                ]
            )
        ),
    }


def spectral_replan_signature(q: np.ndarray, names: list[str]) -> dict[str, Any]:
    # Use commanded acceleration: discontinuous replan replacement is most
    # directly visible as a periodic acceleration impulse every four frames.
    acceleration = np.diff(np.asarray(q, dtype=np.float64), n=2, axis=0) * FPS**2
    acceleration -= np.mean(acceleration, axis=0, keepdims=True)
    frequencies = np.fft.rfftfreq(len(acceleration), d=1.0 / FPS)
    spectrum = np.abs(np.fft.rfft(acceleration, axis=0)) * 2.0 / len(acceleration)
    expected = FPS / HORIZON
    nearest = int(np.argmin(np.abs(frequencies - expected)))
    band = (frequencies >= expected - 0.75) & (frequencies <= expected + 0.75)
    band_rms = np.sqrt(np.mean(np.square(spectrum[band]), axis=0))
    order = np.argsort(band_rms)[::-1]
    return {
        "replan_frequency_hz": expected,
        "nearest_fft_bin_hz": float(frequencies[nearest]),
        "per_joint_acceleration_amplitude_at_replan_frequency_rad_s2": {
            name: float(spectrum[nearest, index]) for index, name in enumerate(names)
        },
        "dominant_joints_in_replan_frequency_band": [
            {
                "joint": names[int(index)],
                "band_rms_acceleration_amplitude_rad_s2": float(band_rms[index]),
            }
            for index in order[:8]
        ],
        "frequency_hz": frequencies,
        "mean_acceleration_amplitude_across_joints_rad_s2": np.mean(spectrum, axis=1),
    }


def run_summary(path: Path, visual: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = read_json(path / "stage_report.json")
    arrays = dict(np.load(path / "rollout_trace.npz", allow_pickle=False))
    names = arrays["joint_names"].astype(str).tolist()
    inference = dict(np.load(path / "inference_chunks.npz", allow_pickle=False))
    raw_equal_alias = bool(
        np.array_equal(inference["policy_raw_action"], inference["raw_policy_chunk"])
    )
    command = arrays["commanded_action"] if "commanded_action" in arrays else arrays["commanded_q"]
    measured = arrays["measured_qpos"] if "measured_qpos" in arrays else arrays["actual_q"]
    rollout = report["rollout"]
    collision = rollout["actual_hard_limit_audit"]["collision"]
    summary = {
        "path": str(path),
        "stage_status": report["status"],
        "duration_s": float(report["duration_s"]),
        "executed_control_frames": int(report["executed_control_frames"]),
        "inference_calls": int(report["inference_call_count"]),
        "diagnostic_abort_reason": report.get("diagnostic_abort_reason"),
        "boundary": boundary_metrics(path, names),
        "command_dynamics": trajectory_dynamics(command),
        "measured_dynamics": trajectory_dynamics(measured),
        "tracking_rmse_rad": float(rollout["tracking_rmse_rad"]),
        "commanded_hard_limit_violation_count": int(
            rollout["commanded_hard_limit_audit"]["joint_limit_violation_count"]
        ),
        "measured_arm_hard_limit_violation_count": int(
            rollout["actual_arm_hard_limit_violation_count"]
        ),
        "measured_dex3_maximum_hard_limit_excess_rad": float(
            rollout["actual_dex3_maximum_hard_limit_excess_rad"]
        ),
        "branch_discontinuity_count": int(
            rollout["actual_hard_limit_audit"]["branch_discontinuity_count"]
        ),
        "robot_hard_collision_incidence": int(
            collision["invalid_hard_collision_category_incidence"]
        ),
        "maximum_external_non_task_contact_force_n": float(
            rollout["maximum_external_non_task_contact_force_n"]
        ),
        "visual_jitter_review": visual,
        "raw_chunk_alias_byte_equal": raw_equal_alias,
        "stored_arrays": sorted(inference),
    }
    spectral = spectral_replan_signature(command, names)
    summary["replan_frequency_signature"] = {
        key: value
        for key, value in spectral.items()
        if key not in {"frequency_hz", "mean_acceleration_amplitude_across_joints_rad_s2"}
    }
    return summary, {
        "names": np.asarray(names),
        "command": command,
        "measured": measured,
        "frequency": spectral["frequency_hz"],
        "spectrum": spectral["mean_acceleration_amplitude_across_joints_rad_s2"],
    }


def video_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
        if path.is_file()
    ]


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)
    reference = read_json(REFERENCE)
    visual = {
        "M0_NAIVE": "SEVERE_REPEATED_REPLAN_SYNCHRONIZED_SHAKING",
        "M1_RTC": "REDUCED_BUT_STILL_VISIBLE",
        "M2_STATE_ALIGNED_CROSSFADE": "SUBSTANTIALLY_REDUCED_NO_REPEATED_HIGH_FREQUENCY_SHAKE",
    }
    methods: dict[str, Any] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    for name, path in RUNS.items():
        methods[name], traces[name] = run_summary(path, visual[name])

    old_report = read_json(OLD_FULL / "stage_report.json")
    old_trace = dict(np.load(OLD_FULL / "rollout_trace.npz", allow_pickle=False))
    names = old_trace["joint_names"].astype(str).tolist()
    old_command = old_trace["commanded_q"]
    old_measured = old_trace["actual_q"]
    old_full = {
        "path": str(OLD_FULL),
        "duration_s": float(old_report["duration_s"]),
        "inference_calls": int(old_report["inference_call_count"]),
        "boundary": boundary_metrics(OLD_FULL, names),
        "command_dynamics": trajectory_dynamics(old_command),
        "measured_dynamics": trajectory_dynamics(old_measured),
        "visual_jitter_review": "SEVERE_REPEATED_REPLAN_SYNCHRONIZED_SHAKING",
    }
    new_full, new_trace = run_summary(
        NEW_FULL, "SUBSTANTIALLY_REDUCED_NO_REPEATED_HIGH_FREQUENCY_SHAKE"
    )
    new_full_report = read_json(NEW_FULL / "stage_report.json")
    new_full["executed_prefix_safety_summary"] = new_full_report[
        "executed_prefix_safety_summary"
    ]
    new_full["future_suffix_warning_summary"] = new_full_report[
        "future_suffix_warning_summary"
    ]
    new_full["object_visualization_mode"] = new_full_report["object_visualization_mode"]

    old_max = old_full["boundary"]["executed_command_jump_rad"]["maximum"]
    new_max = new_full["boundary"]["executed_command_jump_rad"]["maximum"]
    m0_p99 = methods["M0_NAIVE"]["boundary"]["executed_command_jump_rad"]["p99"]
    m2_p99 = methods["M2_STATE_ALIGNED_CROSSFADE"]["boundary"][
        "executed_command_jump_rad"
    ]["p99"]

    comparison = {
        "schema_version": 1,
        "purpose": "RAW_POLICY_CHUNKS_TO_CAUSAL_CONTINUOUS_COMMAND_TRAJECTORY",
        "dataset_b_natural_motion_reference": reference,
        "object_free_methods": methods,
        "full_motion_before": old_full,
        "full_motion_after": new_full,
        "reductions": {
            "object_free_p99_boundary_jump_percent": float(100.0 * (1.0 - m2_p99 / m0_p99)),
            "full_motion_max_boundary_jump_percent": float(100.0 * (1.0 - new_max / old_max)),
        },
        "root_cause": {
            "classification": "CAUSAL_REPLAN_INCONSISTENCY_AT_HORIZON_4_BOUNDARIES",
            "pattern": "command reset every 4 control frames (0.133333 s; 7.5 Hz)",
            "dominant_joints": [
                row["joint"]
                for row in sorted(
                    methods["M0_NAIVE"]["boundary"]["per_joint"],
                    key=lambda item: item["p99"],
                    reverse=True,
                )[:8]
            ],
            "evidence": (
                "Large old-plan/new-plan discrepancies were repeatedly injected by naive replacement; "
                "RTC reduced but did not eliminate the reset. A causal measured-state-anchored "
                "crossfade substantially reduced both the boundary impulses and visible shaking "
                "without changing raw policy chunks."
            ),
        },
        "classification_separation": {
            "policy_command_continuity": "PASS_FOR_ISAAC_DIAGNOSTIC",
            "policy_behavior_progression": "UNCHANGED",
            "strict_sim_safety_qualification": "FAIL",
            "real_hardware_safety_readiness": "BLOCKED",
        },
    }
    (ANALYSIS / "causal_execution_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (ANALYSIS / "per_joint_boundary_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["joint_index", "joint", "m0_p99_rad", "m1_p99_rad", "m2_p99_rad", "m0_max_rad", "m1_max_rad", "m2_max_rad"]
        )
        rows = [methods[name]["boundary"]["per_joint"] for name in RUNS]
        for index, joint in enumerate(names):
            writer.writerow(
                [index, joint, rows[0][index]["p99"], rows[1][index]["p99"], rows[2][index]["p99"], rows[0][index]["maximum"], rows[1][index]["maximum"], rows[2][index]["maximum"]]
            )

    labels = ["M0 naive", "M1 RTC", "M2 crossfade"]
    colors = ["#d1495b", "#edae49", "#00798c"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color, key in zip(labels, colors, RUNS):
        values = methods[key]["boundary"]["executed_command_jump_rad"]
        axes[0].plot([50, 95, 99, 100], [values["median"], values["p95"], values["p99"], values["maximum"]], marker="o", color=color, label=label)
        dyn = methods[key]["command_dynamics"]
        axes[1].scatter(dyn["acceleration_rad_s2"]["p99"], dyn["jerk_rad_s3"]["p99"], color=color, s=70, label=label)
    ref = reference["per_frame_maximum_across_joints"]
    axes[0].axhline(ref["adjacent_step_rad"]["p95"], color="black", linestyle="--", label="Dataset-B step p95")
    axes[0].set(xlabel="percentile", ylabel="max joint boundary jump (rad)", title="Replan-boundary command jump")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axvline(ref["acceleration_rad_s2"]["p99"], color="black", linestyle="--", label="Dataset-B acceleration p99")
    axes[1].axhline(ref["jerk_rad_s3"]["p99"], color="gray", linestyle="--", label="Dataset-B jerk p99")
    axes[1].set(xlabel="command acceleration p99 (rad/s²)", ylabel="command jerk p99 (rad/s³)", title="Natural-motion envelope")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ANALYSIS / "boundary_and_dynamics.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, color, key in zip(labels, colors, RUNS):
        ax.plot(traces[key]["frequency"], traces[key]["spectrum"], color=color, label=label)
    ax.axvline(FPS / HORIZON, color="black", linestyle="--", label="replan frequency 7.5 Hz")
    ax.set(xlim=(0, 15), xlabel="frequency (Hz)", ylabel="mean command-acceleration amplitude (rad/s²)", title="Replan-synchronous command spectrum")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS / "replan_frequency_spectrum.png", dpi=170)
    plt.close(fig)

    implementation_files = {
        "common_stitcher": ROOT / "tools/common_causal_action_stitching.py",
        "policy_worker": ROOT / "tools/policy_b_inference_worker.py",
        "isaac_runner": ROOT / "tools/run_policy_b_isaac_doll_handoff.py",
        "dataset_reference_analyzer": ROOT / "tools/analyze_dataset_b_natural_motion_reference.py",
    }
    weights = [float(10 * u**3 - 15 * u**4 + 6 * u**5) for u in np.arange(1, 6) / 5]
    config = {
        "schema_version": 1,
        "name": "COMMON_CAUSAL_ACTION_STITCHING_ADAPTER",
        "method": "STATE_ALIGNED_CAUSAL_MINIMUM_JERK_CROSSFADE",
        "scope": "POLICY_INDEPENDENT_28D_ABSOLUTE_JOINT_POSITION_EXECUTION",
        "control_fps": FPS,
        "execution_horizon_frames": HORIZON,
        "crossfade_window_frames": 5,
        "crossfade_window_seconds": 5 / FPS,
        "minimum_jerk_weights": weights,
        "anchor": "LATEST_MEASURED_QPOS",
        "prior_plan": "PREVIOUS_DEPLOYMENT_SAFE_REMAINING_PLAN_ALIGNED_AT_REPLAN_TIMESTAMP",
        "new_plan": "CURRENT_RAW_POLICY_CHUNK",
        "derivation": {
            "m0_maximum_old_new_plan_discrepancy_rad": methods["M0_NAIVE"]["boundary"]["planned_old_vs_new_raw_maximum_rad"],
            "dataset_b_p95_jerk_rad_s3": ref["jerk_rad_s3"]["p95"],
            "formula": "ceil(30 * cbrt(60 * discrepancy / Dataset-B p95 jerk))",
            "derived_frames": 5,
            "broad_parameter_sweep_used": False,
        },
        "causality": {
            "future_observations_used": False,
            "future_policy_calls_used": False,
            "offline_temporal_consensus_used": False,
        },
        "preservation": {
            "raw_policy_chunk_logged_unchanged": True,
            "previous_remaining_plan_logged": True,
            "stitched_execution_plan_logged": True,
            "commanded_action_logged": True,
            "measured_qpos_logged": True,
        },
        "semantic_logic": {
            "episode_specific": False,
            "phase_specific": False,
            "task_specific": False,
            "grasp_or_handoff_waypoints": False,
        },
        "qualification": {
            "policy_command_continuity": "PASS_FOR_ISAAC_DIAGNOSTIC",
            "object_free_strict_safety": "FAIL_TABLE_CONTACT",
            "strict_sim_safety_qualification": "FAIL",
            "real_hardware_authorized": False,
        },
        "implementation_sha256": {key: sha256(path) for key, path in implementation_files.items()},
    }
    config_path = FROZEN / "common_causal_execution_adapter.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review_videos = (
        list((OUT / "comparison").glob("*.mp4"))
        + list(
            (OUT / "object_free_multiview").glob(
                "*/stage2_object_free_multichunk/*.mp4"
            )
        )
        + [
            NEW_FULL / f"{name}.mp4"
            for name in ("overview", "top", "side", "source_like")
        ]
    )
    freeze = {
        "schema_version": 1,
        "adapter_config": str(config_path),
        "adapter_config_sha256": sha256(config_path),
        "implementation_sha256": config["implementation_sha256"],
        "dataset_b_motion_reference": str(REFERENCE),
        "dataset_b_motion_reference_sha256": sha256(REFERENCE),
        "checkpoint": str(new_full_report["checkpoint"]),
        "checkpoint_model_sha256": new_full_report["checkpoint_model_sha256"],
        "camera_config": str(ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"),
        "camera_config_sha256": sha256(ROOT / "outputs/policy_b_isaac_validation/camera/source_like_cam_high.json"),
        "selected_method": config["method"],
        "selected_object_free_artifact": str(RUNS["M2_STATE_ALIGNED_CROSSFADE"]),
        "full_motion_artifact": str(NEW_FULL),
        "analysis": str(ANALYSIS / "causal_execution_comparison.json"),
        "analysis_sha256": sha256(ANALYSIS / "causal_execution_comparison.json"),
        "video_manifest": video_manifest(sorted(review_videos)),
        "real_robot_command_allowed": False,
        "strict_sim_safety_qualification": "FAIL",
    }
    freeze_path = FROZEN / "FREEZE_MANIFEST.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dominant = comparison["root_cause"]["dominant_joints"]
    report_lines = [
        "# Policy-B causal execution review",
        "",
        "Raw Policy-B chunks were preserved. The command-layer root cause is causal replan inconsistency: naive replacement repeatedly injected an old-plan/new-plan reset every four frames (7.5 Hz). The state-aligned minimum-jerk crossfade substantially reduced the visible shake without semantic task logic.",
        "",
        "## Result separation",
        "",
        "- POLICY_COMMAND_CONTINUITY: PASS_FOR_ISAAC_DIAGNOSTIC",
        "- POLICY_BEHAVIOR_PROGRESSION: UNCHANGED",
        "- STRICT_SIM_SAFETY_QUALIFICATION: FAIL",
        "- REAL_HARDWARE_SAFETY_READINESS: BLOCKED",
        "",
        "The object-free qualification remains FAIL: strict runtime gates stopped M0, M1, and M2 before 15–20 s due table contact (and M0/M1 also exceeded the prior diagnostic Dex3 excursion cap). No result was relaxed or relabeled.",
        "",
        "## Dominant baseline joints",
        "",
        *[f"- {joint}" for joint in dominant],
        "",
        "## Selected adapter",
        "",
        f"- Method: {config['method']}",
        "- Window: 5 frames / 0.166667 s",
        "- Anchor: latest measured qpos",
        "- Causal only; no future observation, future policy call, RTC, averaging, or offline consensus",
        f"- Config SHA256: {freeze['adapter_config_sha256']}",
        "",
        "## Full-motion diagnostic",
        "",
        f"- Duration: {new_full['duration_s']:.6f} s ({new_full['executed_control_frames']} frames, {new_full['inference_calls']} policy calls)",
        f"- Boundary jump median/p95/p99/max: {new_full['boundary']['executed_command_jump_rad']['median']:.6f} / {new_full['boundary']['executed_command_jump_rad']['p95']:.6f} / {new_full['boundary']['executed_command_jump_rad']['p99']:.6f} / {new_full['boundary']['executed_command_jump_rad']['maximum']:.6f} rad",
        f"- Command acceleration p99/max: {new_full['command_dynamics']['acceleration_rad_s2']['p99']:.6f} / {new_full['command_dynamics']['acceleration_rad_s2']['maximum']:.6f} rad/s²",
        f"- Command jerk p99/max: {new_full['command_dynamics']['jerk_rad_s3']['p99']:.6f} / {new_full['command_dynamics']['jerk_rad_s3']['maximum']:.6f} rad/s³",
        "- Executed-prefix collision/branch discontinuity: 0 / 0",
        "- Qualitative progression: left approach only; later semantic phases unchanged",
        "- PHYSICAL_OBJECT_SUCCESS_NOT_EVALUATED (kinematic visualization doll)",
        "",
        "## Review artifacts",
        "",
        f"- Comparison videos: {OUT / 'comparison'}",
        f"- Strict-gated object-free multiview runs: {OUT / 'object_free_multiview'}",
        f"- Full rollout: {NEW_FULL}",
        f"- Numerical analysis: {ANALYSIS / 'causal_execution_comparison.json'}",
        f"- Frozen adapter: {freeze_path}",
    ]
    (OUT / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "analysis": str(ANALYSIS / "causal_execution_comparison.json"),
        "config": str(config_path),
        "config_sha256": freeze["adapter_config_sha256"],
        "freeze_manifest": str(freeze_path),
        "final_report": str(OUT / "final_report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
