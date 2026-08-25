#!/usr/bin/env python3
"""Offline raw Dex3 contact-channel statistics, plots, deltas and data-quality audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summary(value: np.ndarray) -> dict[str, Any]:
    return {
        "mean": np.mean(value, axis=0).tolist(),
        "std": np.std(value, axis=0).tolist(),
        "median": np.median(value, axis=0).tolist(),
        "mad": np.median(np.abs(value - np.median(value, axis=0)), axis=0).tolist(),
        "minimum": np.min(value, axis=0).tolist(),
        "maximum": np.max(value, axis=0).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-seconds", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.recording, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    report: dict[str, Any] = {
        "schema_version": "dex3_contact_readonly_analysis_v1",
        "status": "PASS_OFFLINE_RAW_CHANNEL_ANALYSIS",
        "recording": str(args.recording.resolve()),
        "force_units": "NOT_ASSIGNED",
        "contact_threshold": "NOT_ASSIGNED",
        "per_hand": {},
    }
    plot_payload = {}
    for side in ("left", "right"):
        pressure = np.asarray(data[f"{side}_press_raw"], dtype=np.float64)
        timestamps = np.asarray(data[f"{side}_host_monotonic_timestamp_ns"], dtype=np.int64)
        time_s = (timestamps - timestamps[0]) / 1e9
        baseline_count = max(1, int(np.searchsorted(time_s, args.baseline_seconds, side="right")))
        baseline = pressure[:baseline_count]
        center = np.median(baseline, axis=0)
        delta = pressure - center
        gaps = np.diff(time_s)
        lost = np.asarray(data[f"{side}_press_lost_raw"], dtype=np.uint32)
        flattened = pressure.reshape(len(pressure), -1)
        zero_step_fraction = np.mean(np.diff(flattened, axis=0) == 0, axis=0) if len(pressure) > 1 else np.ones(108)
        plateau_candidates = np.flatnonzero(zero_step_fraction > 0.995).astype(int).tolist()
        report["per_hand"][side] = {
            "samples": len(pressure),
            "raw_shape": list(pressure.shape),
            "baseline_samples": baseline_count,
            "baseline_raw_statistics": summary(baseline),
            "full_recording_raw_statistics": summary(pressure),
            "contact_delta_from_baseline_statistics": summary(delta),
            "maximum_absolute_contact_delta_raw": float(np.max(np.abs(delta))),
            "maximum_interarrival_gap_s": float(gaps.max(initial=0.0)),
            "press_lost_nonzero_count": int(np.count_nonzero(lost)),
            "press_lost_maximum_raw": int(lost.max(initial=0)),
            "constant_plateau_candidate_flat_indices": plateau_candidates,
            "saturation_interpretation": "candidate plateaus only; device saturation limits are not assumed",
        }
        plot_payload[side] = (time_s, pressure, delta)

    left_center = np.median(plot_payload["left"][1], axis=0)
    right_center = np.median(plot_payload["right"][1], axis=0)
    report["left_right_comparison"] = {
        "median_raw_difference_left_minus_right": (left_center - right_center).tolist(),
        "maximum_absolute_median_raw_difference": float(np.max(np.abs(left_center - right_center))),
        "sensor_index_correspondence_assumed": False,
    }
    atomic_json(args.output_dir / "analysis.json", report)
    annotation = {
        "schema_version": "dex3_manual_sensor_index_annotation_v1",
        "status": "NOT_ANNOTATED",
        "force_units": "NOT_ASSIGNED",
        "instructions": "Manually touch one physical region at a time and map only observed raw sensor indices. Do not infer force or copy thresholds.",
        "left": {str(index): None for index in range(9)},
        "right": {str(index): None for index in range(9)},
    }
    atomic_json(args.output_dir / "manual_sensor_index_annotation.template.json", annotation)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for side, (time_s, pressure, delta) in plot_payload.items():
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for sensor in range(9):
            axes[0].plot(time_s, pressure[:, sensor].mean(axis=1), label=f"sensor {sensor}")
            axes[1].plot(time_s, delta[:, sensor].mean(axis=1), label=f"sensor {sensor}")
        axes[0].set_ylabel("raw mean of 12 values")
        axes[1].set_ylabel("raw delta from baseline")
        axes[1].set_xlabel("host monotonic time (s)")
        axes[0].legend(ncol=3, fontsize=8)
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        figure.suptitle(f"Dex3 {side}: raw pressure records (no force units)")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{side}_raw_channels_and_delta.png", dpi=160)
        plt.close(figure)
    print(json.dumps({"status": report["status"], "output": str(args.output_dir.resolve()), "force_units": "NOT_ASSIGNED"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
