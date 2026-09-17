#!/usr/bin/env python3
"""Audit fixed-noise Policy-B chunks before any queue or PAINT experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/policy_execution_stability_official_async_paint"
RUN_ROOT = OUTPUT / "single_raw_chunk"
LOW_REFERENCE = (
    ROOT
    / "outputs/policy_execution_stability_review/low_motion_reference/low_motion_reference.json"
)
GLOBAL_REFERENCE = (
    ROOT
    / "outputs/policy_b_causal_execution/dataset_b_motion_reference/dataset_b_natural_motion_reference.json"
)
LABELS = ("initial", "left_approach", "doll_region_plateau")
FPS = 30.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def reversal_rates(velocity: np.ndarray, deadband: float) -> np.ndarray:
    rates = []
    duration = velocity.shape[0] / FPS
    for index in range(velocity.shape[1]):
        values = velocity[:, index]
        signs = np.where(values > deadband, 1, np.where(values < -deadband, -1, 0))
        nonzero = signs[signs != 0]
        count = int(np.count_nonzero(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0
        rates.append(count / duration)
    return np.asarray(rates, dtype=np.float64)


def minimum_motion_window(q: np.ndarray, width: int = 30) -> tuple[int, np.ndarray, float]:
    candidates = []
    for start in range(0, len(q) - width + 1):
        window = q[start : start + width]
        velocity = np.diff(window[:, :14], axis=0) * FPS
        speed = float(np.sqrt(np.mean(np.square(velocity))))
        candidates.append((speed, start, window))
    speed, start, window = min(candidates, key=lambda row: row[0])
    return start, window, speed


def main() -> None:
    low_reference = json.loads(LOW_REFERENCE.read_text(encoding="utf-8"))
    global_reference = json.loads(GLOBAL_REFERENCE.read_text(encoding="utf-8"))
    deadband = float(low_reference["selection"]["direction_reversal_deadband_rad_s"])
    cutoff = float(low_reference["selection"]["arm_speed_rms_cutoff_rad_s"])
    records: list[dict[str, Any]] = []
    chunks = []
    common_names: list[str] | None = None
    noise_hashes = set()

    for label in LABELS:
        stage = RUN_ROOT / label / "stage1_object_free_single"
        report_path = stage / "stage_report.json"
        inference_path = stage / "inference_chunks.npz"
        rollout_path = stage / "rollout_trace.npz"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        with np.load(inference_path, allow_pickle=False) as archive:
            raw = archive["policy_raw_action"][0].astype(np.float64)
            names = archive["joint_names"].astype(str).tolist()
            if not np.array_equal(archive["policy_raw_action"], archive["raw_policy_chunk"]):
                raise RuntimeError(f"raw Policy-B chunk alias differs for {label}")
        with np.load(rollout_path, allow_pickle=False) as archive:
            commanded = archive["commanded_action"].astype(np.float64)
            measured = archive["measured_qpos"].astype(np.float64)
        if raw.shape != (50, 28) or commanded.shape != (50, 28) or measured.shape != (50, 28):
            raise RuntimeError(f"unexpected single-chunk shape for {label}")
        if common_names is None:
            common_names = names
        elif names != common_names:
            raise RuntimeError("single-chunk joint order differs across observations")
        noise = report["policy_worker"]["causal_stitching"]["episode_persistent_flow_noise"]
        noise_path = Path(noise["path"])
        noise_array = np.load(noise_path, allow_pickle=False)
        if hashlib.sha256(noise_array.tobytes()).hexdigest() != noise["sha256"]:
            raise RuntimeError(f"flow-noise tensor hash mismatch for {label}")
        noise_hashes.add(noise["sha256"])

        step = np.diff(raw, axis=0)
        velocity = step * FPS
        acceleration = np.diff(velocity, axis=0) * FPS
        jerk = np.diff(acceleration, axis=0) * FPS
        reversals = reversal_rates(velocity, deadband)
        window_start, window, minimum_arm_speed = minimum_motion_window(raw)
        window_velocity = np.diff(window, axis=0) * FPS
        window_acceleration = np.diff(window_velocity, axis=0) * FPS
        window_jerk = np.diff(window_acceleration, axis=0) * FPS
        window_reversals = reversal_rates(window_velocity, deadband)
        record = {
            "label": label,
            "checkpoint": report["checkpoint"],
            "checkpoint_model_sha256": report["checkpoint_model_sha256"],
            "fixed_observation": report["fixed_observation_single_chunk_diagnostic"],
            "flow_noise": noise,
            "raw_chunk_sha256": hashlib.sha256(raw.astype(np.float32).tobytes()).hexdigest(),
            "raw_chunk_preserved": True,
            "inference_calls": int(report["inference_call_count"]),
            "executed_frames": int(report["executed_control_frames"]),
            "replanning": False,
            "raw_internal": {
                "adjacent_step_abs_rad": distribution(np.abs(step)),
                "velocity_abs_rad_s": distribution(np.abs(velocity)),
                "acceleration_abs_rad_s2": distribution(np.abs(acceleration)),
                "jerk_abs_rad_s3": distribution(np.abs(jerk)),
                "direction_reversals_per_s": {
                    "all_joints": distribution(reversals),
                    "arms": distribution(reversals[:14]),
                    "dex3": distribution(reversals[14:]),
                    "per_joint": {
                        name: float(reversals[index]) for index, name in enumerate(names)
                    },
                },
                "peak_to_peak_rad": {
                    "all_joints": distribution(np.ptp(raw, axis=0)),
                    "maximum": float(np.max(np.ptp(raw, axis=0))),
                },
            },
            "minimum_motion_one_second_window": {
                "start_row": int(window_start),
                "end_row_inclusive": int(window_start + len(window) - 1),
                "arm_qdot_rms_rad_s": minimum_arm_speed,
                "qualifies_for_dataset_b_low_motion_cutoff": minimum_arm_speed <= cutoff,
                "qdot_rms_rad_s": float(np.sqrt(np.mean(np.square(window_velocity)))),
                "qddot_rms_rad_s2": float(np.sqrt(np.mean(np.square(window_acceleration)))),
                "jerk_rms_rad_s3": float(np.sqrt(np.mean(np.square(window_jerk)))),
                "peak_to_peak_rad": float(np.max(np.ptp(window, axis=0))),
                "direction_reversals_per_s_mean": float(np.mean(window_reversals)),
            },
            "isaac_replay": {
                "stage_status": report["status"],
                "tracking_rmse_rad": float(report["rollout"]["tracking_rmse_rad"]),
                "maximum_measured_velocity_rad_s": float(
                    report["rollout"]["actual_trajectory"]["maximum_velocity_rad_s"]
                ),
                "maximum_measured_acceleration_rad_s2": float(
                    report["rollout"]["actual_trajectory"]["maximum_acceleration_rad_s2"]
                ),
                "maximum_external_non_task_contact_force_n": float(
                    report["rollout"]["maximum_external_non_task_contact_force_n"]
                ),
                "failed_checks": [key for key, value in report["checks"].items() if not value],
                "strict_failed_checks": [
                    key for key, value in report["rollout"]["checks"].items() if not value
                ],
            },
            "artifacts": {
                "report": str(report_path),
                "inference_chunks": str(inference_path),
                "rollout_trace": str(rollout_path),
                "overview_video": str(stage / "overview.mp4"),
                "top_video": str(stage / "top.mp4"),
                "side_video": str(stage / "side.mp4"),
                "source_like_video": str(stage / "policy.mp4"),
            },
        }
        records.append(record)
        chunks.append(raw)

    assert common_names is not None
    chunks_array = np.stack(chunks)
    dominant_scores = np.mean(
        [
            np.asarray(
                list(row["raw_internal"]["direction_reversals_per_s"]["per_joint"].values()),
                dtype=np.float64,
            )
            for row in records
        ],
        axis=0,
    )
    dominant_order = np.argsort(dominant_scores)[::-1]

    # The stop decision is representation-level, not a task-success-tuned scalar
    # gate: all three no-replan chunks contain alternating arm corrections at
    # near-control-rate frequency, and the plateau replay also reaches the strict
    # table-contact gate. Queue semantics cannot remove oscillation already encoded
    # inside the committed raw chunk without changing its representation.
    conclusion = {
        "single_raw_chunk_smooth": False,
        "classification": "BLOCKED_BY_POLICY_LEVEL_NONSMOOTHNESS",
        "reason": (
            "All three fixed-observation/fixed-noise chunks contain dense internal "
            "direction reversals without any replan boundary; the no-replan plateau "
            "replay additionally reached the strict table-contact gate."
        ),
        "stop_before_official_async_queue": True,
        "stop_before_paint": True,
        "official_async_s1_run": False,
        "official_async_s2_run": False,
        "paint_compatibility_audit_run": False,
        "paint_pilot_run": False,
        "next_method_class": (
            "policy-side training-time prefix/action conditioning, final helmet-D455 "
            "XR adaptation, or a smooth action representation"
        ),
    }
    aggregate = {
        "schema_version": "policy_b_fixed_noise_single_raw_chunk_smoothness_v1",
        "checkpoint": records[0]["checkpoint"],
        "checkpoint_model_sha256": records[0]["checkpoint_model_sha256"],
        "control_fps": FPS,
        "observations": list(LABELS),
        "episode_persistent_flow_noise_identical_across_replays": len(noise_hashes) == 1,
        "episode_persistent_flow_noise_sha256": next(iter(noise_hashes)),
        "joint_names": common_names,
        "dominant_internal_reversal_joints": [
            {
                "joint": common_names[int(index)],
                "mean_direction_reversals_per_s": float(dominant_scores[index]),
            }
            for index in dominant_order[:10]
        ],
        "dataset_b_references": {
            "low_motion": str(LOW_REFERENCE),
            "global_motion": str(GLOBAL_REFERENCE),
            "low_motion_arm_speed_cutoff_rad_s": cutoff,
            "low_motion_direction_reversals_per_s_p95": low_reference["metrics"][
                "direction_reversals_per_s"
            ]["all_window_joint_values"]["p95"],
            "low_motion_direction_reversals_per_s_p99": low_reference["metrics"][
                "direction_reversals_per_s"
            ]["all_window_joint_values"]["p99"],
            "global_per_frame_maximum_across_joints": global_reference[
                "per_frame_maximum_across_joints"
            ],
        },
        "records": records,
        "conclusion": conclusion,
        "real_hardware_safety_readiness": "BLOCKED",
        "real_robot_command_allowed": False,
    }

    analysis_dir = RUN_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(50) / FPS
    figure, axes = plt.subplots(len(LABELS), 2, figsize=(15, 10), sharex=True)
    left_indices = [0, 1, 2, 3, 5, 6]
    right_indices = [7, 8, 9, 10, 12, 13]
    for row_index, (label, raw) in enumerate(zip(LABELS, chunks_array)):
        for index in left_indices:
            axes[row_index, 0].plot(time_axis, raw[:, index], label=common_names[index], linewidth=1)
        for index in right_indices:
            axes[row_index, 1].plot(time_axis, raw[:, index], label=common_names[index], linewidth=1)
        axes[row_index, 0].set_ylabel(f"{label}\nq (rad)")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].set_title("Left arm — one fixed-noise raw chunk, no replanning")
    axes[0, 1].set_title("Right arm — one fixed-noise raw chunk, no replanning")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(analysis_dir / "single_raw_chunk_arm_positions.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(len(LABELS), 2, figsize=(15, 10), sharex=True)
    for row_index, (label, raw) in enumerate(zip(LABELS, chunks_array)):
        velocity = np.diff(raw, axis=0) * FPS
        velocity_time = np.arange(len(velocity)) / FPS
        for index in left_indices:
            axes[row_index, 0].plot(velocity_time, velocity[:, index], linewidth=0.9)
        for index in right_indices:
            axes[row_index, 1].plot(velocity_time, velocity[:, index], linewidth=0.9)
        axes[row_index, 0].set_ylabel(f"{label}\nqdot (rad/s)")
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].set_title("Left-arm raw-chunk velocity")
    axes[0, 1].set_title("Right-arm raw-chunk velocity")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(analysis_dir / "single_raw_chunk_arm_velocity.png", dpi=180)
    plt.close(figure)

    aggregate["artifacts"] = {
        "arm_position_plot": str(analysis_dir / "single_raw_chunk_arm_positions.png"),
        "arm_velocity_plot": str(analysis_dir / "single_raw_chunk_arm_velocity.png"),
        "normal_speed_overview_video": str(
            RUN_ROOT / "comparison/overview_three_fixed_noise_single_chunks.mp4"
        ),
        "slow_motion_overview_video": str(
            RUN_ROOT / "comparison/overview_three_fixed_noise_single_chunks_slow4x.mp4"
        ),
    }
    atomic_json(analysis_dir / "single_raw_chunk_smoothness.json", aggregate)

    lines = [
        "# Fixed-noise single raw-chunk smoothness audit",
        "",
        "`SINGLE_RAW_CHUNK_SMOOTH = NO`",
        "",
        "All three chunks were inferred from byte-verified frozen RGB, exact frozen float32 state, the frozen task, and one identical stored flow-noise tensor. Each chunk was replayed for all 50 rows without replanning.",
        "",
        "| Observation | Arm reversals/s mean | Arm reversals/s max | Max step (rad) | Max velocity (rad/s) | Max acceleration (rad/s²) | Max jerk (rad/s³) | Table contact (N) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        internal = row["raw_internal"]
        replay = row["isaac_replay"]
        lines.append(
            f"| {row['label']} | {internal['direction_reversals_per_s']['arms']['mean']:.3f} | "
            f"{internal['direction_reversals_per_s']['arms']['maximum']:.3f} | "
            f"{internal['adjacent_step_abs_rad']['maximum']:.6f} | "
            f"{internal['velocity_abs_rad_s']['maximum']:.6f} | "
            f"{internal['acceleration_abs_rad_s2']['maximum']:.6f} | "
            f"{internal['jerk_abs_rad_s3']['maximum']:.6f} | "
            f"{replay['maximum_external_non_task_contact_force_n']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The stop/correction/reversal structure exists inside a single committed raw chunk; it is therefore not created solely by queue replacement. Per the bounded investigation stop rule, S1, S2, and PAINT were not run.",
            "",
            "Real hardware remains blocked.",
        ]
    )
    (analysis_dir / "single_raw_chunk_smoothness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
