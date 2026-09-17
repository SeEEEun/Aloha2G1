#!/usr/bin/env python3
"""Build a low-motion jitter reference from frozen Dataset-B action labels.

All windows stay within one episode. Selection is kinematic and independent of
policy, camera, task success, or deployment rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FPS = 30.0
WINDOW = 30
STRIDE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def reversal_rate(velocity: np.ndarray, deadband: float) -> np.ndarray:
    signs = np.where(velocity > deadband, 1, np.where(velocity < -deadband, -1, 0))
    rates = np.zeros(velocity.shape[1], dtype=np.float64)
    duration = max((len(velocity) - 1) / FPS, 1.0 / FPS)
    for joint in range(velocity.shape[1]):
        active = signs[:, joint]
        active = active[active != 0]
        changes = np.count_nonzero(active[1:] != active[:-1]) if len(active) > 1 else 0
        rates[joint] = changes / duration
    return rates


def energy_ratio_4_10(q: np.ndarray) -> np.ndarray:
    time = np.arange(len(q), dtype=np.float64)
    detrended = np.empty_like(q, dtype=np.float64)
    for joint in range(q.shape[1]):
        coefficients = np.polyfit(time, q[:, joint], deg=1)
        detrended[:, joint] = q[:, joint] - np.polyval(coefficients, time)
    frequencies = np.fft.rfftfreq(len(q), d=1.0 / FPS)
    power = np.square(np.abs(np.fft.rfft(detrended, axis=0)))
    positive = frequencies > 0
    band = (frequencies >= 4.0) & (frequencies <= 10.0)
    denominator = np.sum(power[positive], axis=0)
    numerator = np.sum(power[band], axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-20)


def metrics(q: np.ndarray, deadband: float) -> dict[str, np.ndarray]:
    velocity = np.diff(q, axis=0) * FPS
    acceleration = np.diff(q, n=2, axis=0) * FPS**2
    jerk = np.diff(q, n=3, axis=0) * FPS**3
    return {
        "qdot_rms_rad_s": np.sqrt(np.mean(np.square(velocity), axis=0)),
        "qddot_rms_rad_s2": np.sqrt(np.mean(np.square(acceleration), axis=0)),
        "jerk_rms_rad_s3": np.sqrt(np.mean(np.square(jerk), axis=0)),
        "peak_to_peak_rad": np.ptp(q, axis=0),
        "direction_reversals_per_s": reversal_rate(velocity, deadband),
        "energy_ratio_4_10hz": energy_ratio_4_10(q),
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = []
    all_joint_names = None
    for episode in manifest["episodes"]:
        path = Path(episode["retargeted_trajectory_path"])
        if sha256(path) != episode["retargeted_trajectory_sha256"]:
            raise RuntimeError(f"trajectory hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            q = archive["replay_named_joint_qpos"].astype(np.float64)
            names = archive["replay_joint_names"].astype(str).tolist()
            ownership = archive["ownership_state"].astype(str)
            left_phase = archive["left_hand_phase"].astype(str)
            right_phase = archive["right_hand_phase"].astype(str)
        if all_joint_names is None:
            all_joint_names = names
        elif names != all_joint_names:
            raise RuntimeError("joint ordering mismatch")
        velocity = np.diff(q[:, :14], axis=0) * FPS
        for start in range(0, len(q) - WINDOW + 1, STRIDE):
            ownership_window = ownership[start : start + WINDOW]
            if len(set(ownership_window)) != 1:
                continue
            speed = np.max(np.abs(velocity[start : start + WINDOW - 1]), axis=1)
            candidates.append(
                {
                    "episode": int(episode["final_dataset_index"]),
                    "start": start,
                    "ownership": ownership_window[0],
                    "left_phase": Counter(left_phase[start : start + WINDOW]).most_common(1)[0][0],
                    "right_phase": Counter(right_phase[start : start + WINDOW]).most_common(1)[0][0],
                    "arm_speed_rms_rad_s": float(np.sqrt(np.mean(np.square(speed)))),
                    "arm_speed_max_rad_s": float(np.max(speed)),
                    "q": q[start : start + WINDOW, :14],
                }
            )

    no_owner_speeds = np.asarray(
        [row["arm_speed_rms_rad_s"] for row in candidates if row["ownership"] == "NO_OWNER"]
    )
    if not len(no_owner_speeds):
        raise RuntimeError("no NO_OWNER reference candidates")
    # The lower decile of one-second pre-acquisition/approach windows is the
    # empirical steady-motion regime. This threshold is computed before and
    # independently of any policy rollout.
    cutoff = float(np.percentile(no_owner_speeds, 10))
    eligible = [row for row in candidates if row["arm_speed_rms_rad_s"] <= cutoff]
    # The raw eligible set contains a much longer post-release tail. Balance
    # the two naturally occurring steady regimes so idle RELEASED frames do not
    # numerically swamp the pre-acquisition/approach reference.
    by_ownership = {
        ownership: [row for row in eligible if row["ownership"] == ownership]
        for ownership in ("NO_OWNER", "RELEASED")
    }
    balanced_count = min(len(rows) for rows in by_ownership.values())
    selected = []
    for ownership, rows in by_ownership.items():
        indices = np.linspace(0, len(rows) - 1, balanced_count).round().astype(int)
        selected.extend(rows[int(index)] for index in indices)
    deadband = 0.05 * cutoff
    if len(selected) < 30:
        raise RuntimeError("insufficient low-motion Dataset-B windows")
    metric_rows = [metrics(row["q"], deadband) for row in selected]
    metric_names = list(metric_rows[0])
    stacked = {name: np.stack([row[name] for row in metric_rows]) for name in metric_names}
    assert all_joint_names is not None
    arm_names = all_joint_names[:14]
    report_metrics = {}
    for metric_name, values in stacked.items():
        report_metrics[metric_name] = {
            "all_window_joint_values": distribution(values),
            "per_window_rms_across_joints": distribution(
                np.sqrt(np.mean(np.square(values), axis=1))
            ),
            "per_window_maximum_joint": distribution(np.max(values, axis=1)),
            "per_joint": {
                joint: distribution(values[:, index])
                for index, joint in enumerate(arm_names)
            },
        }

    phase_counts = Counter(row["ownership"] for row in selected)
    episodes = sorted({row["episode"] for row in selected})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "low_motion_windows.npz").open("wb") as stream:
        np.savez_compressed(
            stream,
            q=np.stack([row["q"] for row in selected]).astype(np.float32),
            episode=np.asarray([row["episode"] for row in selected], dtype=np.int64),
            start=np.asarray([row["start"] for row in selected], dtype=np.int64),
            ownership=np.asarray([row["ownership"] for row in selected]),
            joint_names=np.asarray(arm_names),
            speed_cutoff_rad_s=np.asarray(cutoff),
            reversal_deadband_rad_s=np.asarray(deadband),
            **stacked,
        )
    report = {
        "schema_version": 1,
        "name": "DATASET_B_LOW_MOTION_REFERENCE",
        "dataset_manifest": str(args.manifest.resolve()),
        "dataset_manifest_sha256": sha256(args.manifest),
        "episodes_total": len(manifest["episodes"]),
        "fps": FPS,
        "window_frames": WINDOW,
        "window_seconds": WINDOW / FPS,
        "stride_frames": STRIDE,
        "episode_boundaries_crossed": False,
        "selection": {
            "candidate_definition": "one-second windows with constant ownership annotation",
            "cutoff_derivation": "10th percentile of arm-speed RMS among constant NO_OWNER approach/pre-acquisition windows",
            "arm_speed_rms_cutoff_rad_s": cutoff,
            "rollout_results_used_to_choose_cutoff": False,
            "direction_reversal_deadband_rad_s": deadband,
            "eligible_window_count_before_ownership_balance": len(eligible),
            "ownership_balance_rule": "equal deterministic coverage of NO_OWNER and RELEASED steady regimes",
            "selected_window_count": len(selected),
            "selected_episode_count": len(episodes),
            "ownership_distribution": dict(sorted(phase_counts.items())),
        },
        "joint_names": arm_names,
        "metrics": report_metrics,
        "windows": str(output / "low_motion_windows.npz"),
        "windows_sha256": sha256(output / "low_motion_windows.npz"),
    }
    write_json(output / "low_motion_reference.json", report)

    labels = [
        "qdot RMS\n(rad/s)",
        "qddot RMS\n(rad/s²)",
        "jerk RMS\n(rad/s³)",
        "peak-to-peak\n(rad)",
        "reversals/s",
        "4–10 Hz\nenergy ratio",
    ]
    values = [
        stacked["qdot_rms_rad_s"],
        stacked["qddot_rms_rad_s2"],
        stacked["jerk_rms_rad_s3"],
        stacked["peak_to_peak_rad"],
        stacked["direction_reversals_per_s"],
        stacked["energy_ratio_4_10hz"],
    ]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, label, matrix in zip(axes.flat, labels, values):
        axis.boxplot([matrix[:, index] for index in range(14)], showfliers=False)
        axis.set_title(label)
        axis.set_xticks(range(1, 15))
        axis.set_xticklabels(range(14), fontsize=7)
        axis.grid(True, alpha=0.25)
    figure.suptitle("Dataset-B low-motion reference by arm-joint index")
    figure.tight_layout()
    figure.savefig(output / "low_motion_reference.png", dpi=170)
    plt.close(figure)
    print(json.dumps({
        "report": str(output / "low_motion_reference.json"),
        "selected_windows": len(selected),
        "selected_episodes": len(episodes),
        "cutoff_rad_s": cutoff,
        "ownership_distribution": dict(phase_counts),
    }, indent=2))


if __name__ == "__main__":
    main()
