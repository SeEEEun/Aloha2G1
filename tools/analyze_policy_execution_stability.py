#!/usr/bin/env python3
"""Compare causal execution diagnostics against frozen Dataset-B motion references."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/policy_execution_stability_review/analysis"
LOW_REFERENCE = (
    ROOT / "outputs/policy_execution_stability_review/low_motion_reference/low_motion_reference.json"
)
RUNS = {
    "M2_DIAGNOSTIC_CROSSFADE": ROOT
    / "outputs/policy_b_causal_execution/full_motion/M2_state_aligned_crossfade/full_policy_b_diagnostic_rollout",
    "H4_COMMITMENT": ROOT
    / "outputs/policy_execution_stability_review/horizon_comparison/H04/full_policy_b_diagnostic_rollout",
    "H8_COMMITMENT": ROOT
    / "outputs/policy_execution_stability_review/horizon_comparison/H08/full_policy_b_diagnostic_rollout",
    "H12_COMMITMENT": ROOT
    / "outputs/policy_execution_stability_review/horizon_comparison/H12/full_policy_b_diagnostic_rollout",
    "H16_COMMITMENT": ROOT
    / "outputs/policy_execution_stability_review/horizon_comparison/H16/full_policy_b_diagnostic_rollout",
    "RTC_H8_D6_G10_LINEAR": ROOT
    / "outputs/policy_execution_stability_review/rtc_bounded/H08_D06_G10_LINEAR/full_policy_b_diagnostic_rollout",
    "RTC_H10_D6_G10_LINEAR": ROOT
    / "outputs/policy_execution_stability_review/rtc_bounded/H10_D06_G10_LINEAR/full_policy_b_diagnostic_rollout",
    "RTC_H12_D6_G10_LINEAR": ROOT
    / "outputs/policy_execution_stability_review/rtc_bounded/H12_D06_G10_LINEAR/full_policy_b_diagnostic_rollout",
    "RTC_H10_D9_G10_LINEAR": ROOT
    / "outputs/policy_execution_stability_review/rtc_latency_corrected/H10_D09_G10_LINEAR/full_policy_b_diagnostic_rollout",
    "STATEFUL_RTC_RUCKIG_H8": ROOT
    / "outputs/policy_execution_stability_review/stateful_otg_bounded/H08_D06_FIXED_RTC_RUCKIG/full_policy_b_diagnostic_rollout",
    "STATEFUL_RTC_RUCKIG_H10": ROOT
    / "outputs/policy_execution_stability_review/stateful_otg_validation_v2/H10_D06_FIXED_RTC_RUCKIG/full_policy_b_diagnostic_rollout",
    "STATEFUL_RTC_RUCKIG_H10_VIDEO": ROOT
    / "outputs/policy_execution_stability_review/selected_candidate_video/H10_D06_FIXED_RTC_STATEFUL_RUCKIG/full_policy_b_diagnostic_rollout",
    "STATEFUL_RTC_RUCKIG_H10_D9": ROOT
    / "outputs/policy_execution_stability_review/stateful_otg_latency_corrected/H10_D09_FIXED_RTC_RUCKIG/full_policy_b_diagnostic_rollout",
    "STATEFUL_RTC_RUCKIG_H10_D9_VIDEO": ROOT
    / "outputs/policy_execution_stability_review/selected_candidate_video_latency_corrected/H10_D09_FIXED_RTC_STATEFUL_RUCKIG/full_policy_b_diagnostic_rollout",
    "STATEFUL_RTC_RUCKIG_H12": ROOT
    / "outputs/policy_execution_stability_review/stateful_otg_bounded/H12_D06_FIXED_RTC_RUCKIG/full_policy_b_diagnostic_rollout",
}
FPS = 30.0
ARM_COUNT = 14


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    value = np.asarray(values, dtype=np.float64).ravel()
    value = value[np.isfinite(value)]
    if not len(value):
        return {key: None for key in ("count", "mean", "median", "p95", "p99", "maximum")}
    return {
        "count": int(len(value)),
        "mean": float(np.mean(value)),
        "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95)),
        "p99": float(np.percentile(value, 99)),
        "maximum": float(np.max(value)),
    }


def energy_ratio(position: np.ndarray) -> np.ndarray:
    samples = np.asarray(position, dtype=np.float64)
    time = np.arange(len(samples), dtype=np.float64)
    design = np.column_stack((time, np.ones_like(time)))
    trend = design @ np.linalg.lstsq(design, samples, rcond=None)[0]
    centered = samples - trend
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / FPS)
    denominator = np.sum(power[frequencies > 0.0], axis=0)
    numerator = np.sum(power[(frequencies >= 4.0) & (frequencies <= 10.0)], axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-20)


def reversal_rate(velocity: np.ndarray, deadband: float, duration: float) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    result = np.zeros(velocity.shape[1], dtype=np.float64)
    for joint in range(velocity.shape[1]):
        sign = np.sign(velocity[:, joint])
        sign[np.abs(velocity[:, joint]) <= deadband] = 0.0
        active = sign[sign != 0.0]
        result[joint] = np.count_nonzero(active[1:] != active[:-1]) / max(duration, 1.0 / FPS)
    return result


def trajectory_metrics(q: np.ndarray, deadband: float) -> dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)
    qdot = np.diff(q, axis=0) * FPS
    qddot = np.diff(qdot, axis=0) * FPS
    jerk = np.diff(qddot, axis=0) * FPS
    duration = max((len(q) - 1) / FPS, 1.0 / FPS)
    reversals = reversal_rate(qdot, deadband, duration)
    return {
        "joint_step_abs_rad": distribution(np.abs(np.diff(q, axis=0))),
        "velocity_abs_rad_s": distribution(np.abs(qdot)),
        "acceleration_abs_rad_s2": distribution(np.abs(qddot)),
        "jerk_abs_rad_s3": distribution(np.abs(jerk)),
        "maximum_velocity_across_joints_rad_s": distribution(np.max(np.abs(qdot), axis=1)),
        "maximum_acceleration_across_joints_rad_s2": distribution(np.max(np.abs(qddot), axis=1)),
        "maximum_jerk_across_joints_rad_s3": distribution(np.max(np.abs(jerk), axis=1)),
        "direction_reversals_per_s": distribution(reversals),
        "direction_reversals_per_s_per_joint": reversals.tolist(),
    }


def low_motion_metrics(q: np.ndarray, deadband: float, cutoff: float) -> dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)[:, :ARM_COUNT]
    rows: dict[str, list[float]] = {
        "qdot_rms_rad_s": [],
        "qddot_rms_rad_s2": [],
        "jerk_rms_rad_s3": [],
        "peak_to_peak_rad": [],
        "direction_reversals_per_s": [],
        "energy_ratio_4_10hz": [],
    }
    selected = []
    for start in range(0, max(len(q) - 30 + 1, 0), 5):
        window = q[start : start + 30]
        qdot = np.diff(window, axis=0) * FPS
        arm_speed_rms = float(np.sqrt(np.mean(np.square(qdot))))
        if arm_speed_rms > cutoff:
            continue
        qddot = np.diff(qdot, axis=0) * FPS
        jerk = np.diff(qddot, axis=0) * FPS
        rows["qdot_rms_rad_s"].extend(np.sqrt(np.mean(np.square(qdot), axis=0)))
        rows["qddot_rms_rad_s2"].extend(np.sqrt(np.mean(np.square(qddot), axis=0)))
        rows["jerk_rms_rad_s3"].extend(np.sqrt(np.mean(np.square(jerk), axis=0)))
        rows["peak_to_peak_rad"].extend(np.ptp(window, axis=0))
        rows["direction_reversals_per_s"].extend(reversal_rate(qdot, deadband, 29.0 / FPS))
        rows["energy_ratio_4_10hz"].extend(energy_ratio(window))
        selected.append({"start_frame": start, "arm_speed_rms_rad_s": arm_speed_rms})
    return {
        "definition": "one-second/30-frame arm windows, stride 5, arm qdot RMS at or below frozen Dataset-B cutoff",
        "window_count": len(selected),
        "selected_windows": selected,
        "metrics": {name: distribution(np.asarray(values)) for name, values in rows.items()},
    }


def boundary_metrics(directory: Path, names: list[str]) -> dict[str, Any]:
    report = json.loads((directory / "chunk_boundary_jitter.json").read_text(encoding="utf-8"))
    records = report.get("records", [])
    executed = []
    raw_disagreement = []
    per_joint = []
    for row in records:
        if row.get("executed_boundary_delta_rad") is not None:
            value = np.abs(np.asarray(row["executed_boundary_delta_rad"], dtype=np.float64))
            executed.append(float(value.max()))
            per_joint.append(value)
        if row.get("old_plan_vs_new_raw_delta_rad") is not None:
            raw_disagreement.append(float(np.max(np.abs(row["old_plan_vs_new_raw_delta_rad"]))))
    if per_joint:
        joint_mean = np.mean(np.stack(per_joint), axis=0)
        dominant = np.argsort(joint_mean)[::-1][:8]
        dominant_rows = [
            {"joint": names[index], "mean_abs_boundary_delta_rad": float(joint_mean[index])}
            for index in dominant
        ]
    else:
        dominant_rows = []
    return {
        "count": len(executed),
        "executed_boundary_max_abs_delta_rad": distribution(np.asarray(executed)),
        "raw_old_plan_new_plan_max_abs_disagreement_rad": distribution(np.asarray(raw_disagreement)),
        "dominant_command_boundary_joints": dominant_rows,
        "reported_maximum_resulting_measured_acceleration_rad_s2": report.get(
            "maximum_resulting_measured_acceleration_rad_s2"
        ),
    }


def abort_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    reason = report.get("diagnostic_abort_reason")
    if reason is None:
        return None
    return {
        "reason": reason.get("reason"),
        "frame": reason.get("frame"),
        "failed_runtime_checks": [key for key, value in reason.get("checks", {}).items() if not value],
        "external_non_task_contact_force_n": reason.get("external_non_task_contact_force_n"),
    }


def analyze_run(directory: Path, reference: dict[str, Any]) -> dict[str, Any]:
    report = json.loads((directory / "stage_report.json").read_text(encoding="utf-8"))
    with np.load(directory / "rollout_trace.npz") as trace:
        command = trace["commanded_q"].astype(np.float64)
        measured = trace["actual_q"].astype(np.float64)
        names = trace["joint_names"].tolist()
    deadband = float(reference["selection"]["direction_reversal_deadband_rad_s"])
    cutoff = float(reference["selection"]["arm_speed_rms_cutoff_rad_s"])
    tracking = command - measured
    return {
        "directory": str(directory),
        "status": report["status"],
        "duration_s": float(report["duration_s"]),
        "frames": int(report["executed_control_frames"]),
        "inference_calls": int(report["inference_call_count"]),
        "execution_horizon_frames": int(report["execution_horizon_frames"]),
        "replanning_frequency_hz": FPS / float(report["execution_horizon_frames"]),
        "abort": abort_summary(report),
        "boundary": boundary_metrics(directory, names),
        "command": trajectory_metrics(command, deadband),
        "measured": trajectory_metrics(measured, deadband),
        "tracking_error_rad": {
            "rmse": float(np.sqrt(np.mean(np.square(tracking)))),
            "maximum_abs": float(np.max(np.abs(tracking))),
        },
        "low_motion_command": low_motion_metrics(command, deadband, cutoff),
        "low_motion_measured": low_motion_metrics(measured, deadband, cutoff),
        "commanded_hard_limit_violation_count": int(
            report["rollout"]["commanded_hard_limit_audit"]["joint_limit_violation_count"]
        ),
        "branch_discontinuity_count": int(
            report["rollout"]["commanded_hard_limit_audit"]["branch_discontinuity_count"]
        ),
        "maximum_external_non_task_contact_force_n": float(
            report["rollout"]["maximum_external_non_task_contact_force_n"]
        ),
    }


def main() -> int:
    reference = json.loads(LOW_REFERENCE.read_text(encoding="utf-8"))
    results = {name: analyze_run(path, reference) for name, path in RUNS.items()}
    payload = {
        "schema_version": "policy_execution_stability_analysis_v1",
        "fps": FPS,
        "low_motion_reference": {
            "path": str(LOW_REFERENCE),
            "selection": reference["selection"],
            "metrics": {
                name: values["all_window_joint_values"]
                for name, values in reference["metrics"].items()
            },
        },
        "runs": results,
        "interpretation_guard": (
            "A scalar improvement does not authorize hardware. Human equal-scale video review "
            "and every real-robot gate remain mandatory."
        ),
    }
    atomic_json(OUT / "execution_stability_metrics.json", payload)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "execution_stability_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run", "status", "duration_s", "horizon", "replan_hz", "boundary_p95_rad",
                "boundary_p99_rad", "boundary_max_rad", "command_velocity_max", "command_acceleration_max",
                "command_jerk_max", "tracking_rmse_rad", "low_motion_windows", "low_qdot_rms_p99",
                "low_qddot_rms_p99", "low_jerk_rms_p99", "low_reversal_p99", "low_energy_4_10hz_p99",
                "low_peak_to_peak_p99", "hard_limit_violations", "branch_discontinuities", "max_external_force_n",
            ]
        )
        for name, row in results.items():
            boundary = row["boundary"]["executed_boundary_max_abs_delta_rad"]
            low = row["low_motion_command"]
            writer.writerow(
                [
                    name, row["status"], row["duration_s"], row["execution_horizon_frames"], row["replanning_frequency_hz"],
                    boundary["p95"], boundary["p99"], boundary["maximum"],
                    row["command"]["velocity_abs_rad_s"]["maximum"],
                    row["command"]["acceleration_abs_rad_s2"]["maximum"],
                    row["command"]["jerk_abs_rad_s3"]["maximum"], row["tracking_error_rad"]["rmse"],
                    low["window_count"], low["metrics"]["qdot_rms_rad_s"]["p99"],
                    low["metrics"]["qddot_rms_rad_s2"]["p99"], low["metrics"]["jerk_rms_rad_s3"]["p99"],
                    low["metrics"]["direction_reversals_per_s"]["p99"],
                    low["metrics"]["energy_ratio_4_10hz"]["p99"], low["metrics"]["peak_to_peak_rad"]["p99"],
                    row["commanded_hard_limit_violation_count"], row["branch_discontinuity_count"],
                    row["maximum_external_non_task_contact_force_n"],
                ]
            )

    labels = ["M2", "H4", "RTC-H8", "Stateful RTC+OTG H10/D9"]
    keys = [
        "M2_DIAGNOSTIC_CROSSFADE",
        "H4_COMMITMENT",
        "RTC_H8_D6_G10_LINEAR",
        "STATEFUL_RTC_RUCKIG_H10_D9",
    ]
    metrics = [
        ("boundary p99 (rad)", lambda row: row["boundary"]["executed_boundary_max_abs_delta_rad"]["p99"]),
        ("low-motion qdot RMS p99 (rad/s)", lambda row: row["low_motion_command"]["metrics"]["qdot_rms_rad_s"]["p99"]),
        ("low-motion 4–10 Hz energy p99", lambda row: row["low_motion_command"]["metrics"]["energy_ratio_4_10hz"]["p99"]),
        ("low-motion peak-to-peak p99 (rad)", lambda row: row["low_motion_command"]["metrics"]["peak_to_peak_rad"]["p99"]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, (title, getter) in zip(axes.ravel(), metrics, strict=True):
        values = [getter(results[key]) for key in keys]
        values = [np.nan if value is None else value for value in values]
        axis.bar(labels, values, color=["#777777", "#d95f02", "#7570b3", "#1b9e77"])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT / "execution_stability_comparison.png", dpi=160)
    plt.close(figure)

    comparison_joints = [0, 2, 3, 5, 7, 10]
    comparison_names = [
        "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_pitch_joint", "right_shoulder_pitch_joint", "right_elbow_joint",
    ]
    traces = {}
    for label, key in [
        ("M2 crossfade", "M2_DIAGNOSTIC_CROSSFADE"),
        ("Stateful RTC H10/D9 + Ruckig", "STATEFUL_RTC_RUCKIG_H10_D9"),
    ]:
        with np.load(Path(results[key]["directory"]) / "rollout_trace.npz") as trace:
            traces[label] = trace["commanded_q"].astype(np.float64)
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for axis, joint, joint_name in zip(axes.ravel(), comparison_joints, comparison_names, strict=True):
        for label, values in traces.items():
            limit = min(len(values), int(9.0 * FPS))
            axis.plot(np.arange(limit) / FPS, values[:limit, joint], label=label, linewidth=1.2)
        axis.set_title(joint_name)
        axis.set_ylabel("command q (rad)")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(OUT / "dominant_arm_command_comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"output": str(OUT), "run_count": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
