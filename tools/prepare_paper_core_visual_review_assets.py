#!/usr/bin/env python3
"""Create temporal strips and motion traces for the paired rollout review.

The generated assets support, but do not replace, a human smoothness judgment.
They are read-only derivatives of the representative source/A/B videos and
measured Isaac trajectories selected before policy-result inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
PAPER = ROOT / "outputs/paper_core_ab"
RESULT = PAPER / "source_conditioned_rollout/experiment3_result.json"
HELDOUT = PAPER / "heldout8_manifest.json"
FIGURES = PAPER / "figures"
FPS = 30.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_video_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, value = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode comparison frame {index}")
    return value


def temporal_strip(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    assets = result["representative_assets"]
    if assets["status"] != "PASS":
        raise RuntimeError("representative paired rollout is not complete")
    final_episode = int(assets["source_final_episode"])
    phases = manifest["complete_source_phase_audit"][str(final_episode)][
        "nine_probe_frames"
    ]
    selected = [
        ("left grasp", int(phases["left_grasp_owned"])),
        ("left transport", int(phases["left_transport"])),
        ("handoff", int(phases["dual_contact_transfer"])),
        ("right owned", int(phases["right_owned"])),
        ("release", int(phases["release"])),
    ]
    video = Path(assets["videos"]["overview"]["path"])
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {video}")
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    rows = []
    offsets = (-6, -3, 0, 3, 6)
    try:
        for phase, center in selected:
            cells = []
            for offset in offsets:
                index = min(max(center + offset, 0), count - 1)
                cell = cv2.resize(
                    read_video_frame(capture, index),
                    (960, 240),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.rectangle(cell, (0, 208), (960, 240), (0, 0, 0), -1)
                cv2.putText(
                    cell,
                    f"{phase} | t={index / FPS:.2f}s | offset {offset:+d} frames",
                    (8, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cells.append(cell)
            rows.append(np.hstack(cells))
    finally:
        capture.release()
    image = np.vstack(rows)
    output = FIGURES / "figure4_human_review_temporal_strips.png"
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"could not write {output}")
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "source_comparison_video": str(video),
        "source_comparison_video_sha256": sha256_file(video),
        "phases": {phase: frame for phase, frame in selected},
        "offsets_frames": list(offsets),
    }


def motion_trace(result: dict[str, Any]) -> dict[str, Any]:
    representative = result["representative_assets"]
    output_episode = int(representative["heldout_output_episode"])
    trajectories: dict[str, np.ndarray] = {}
    paths = {}
    for method in ("a", "b"):
        row = next(
            item
            for item in result["all_rollouts"][method]
            if int(item["output_episode"]) == output_episode
        )
        path = Path(row["rollout_arrays"])
        with np.load(path, allow_pickle=False) as archive:
            trajectories[method] = archive["measured_state"][1:].astype(np.float64)
        paths[method] = {"path": str(path), "sha256": sha256_file(path)}
    if trajectories["a"].shape != trajectories["b"].shape:
        raise RuntimeError("representative A/B trajectory lengths differ")

    trace: dict[str, dict[str, np.ndarray]] = {}
    for method, q in trajectories.items():
        step = np.diff(q, axis=0)
        qdot = step * FPS
        qddot = np.diff(qdot, axis=0) * FPS
        jerk = np.diff(qddot, axis=0) * FPS
        trace[method] = {
            "arm_step_l2": np.linalg.norm(step[:, :14], axis=1),
            "dex3_step_max": np.max(np.abs(step[:, 14:]), axis=1),
            "arm_qddot_rms": np.sqrt(np.mean(qddot[:, :14] ** 2, axis=1)),
            "dex3_qddot_rms": np.sqrt(np.mean(qddot[:, 14:] ** 2, axis=1)),
            "arm_jerk_rms": np.sqrt(np.mean(jerk[:, :14] ** 2, axis=1)),
            "dex3_jerk_rms": np.sqrt(np.mean(jerk[:, 14:] ** 2, axis=1)),
        }
    specifications = [
        ("arm_step_l2", "Arm adjacent-step L2 (rad)"),
        ("dex3_step_max", "Dex3 max adjacent step (rad)"),
        ("arm_qddot_rms", "Arm framewise qddot RMS (rad/s²)"),
        ("dex3_qddot_rms", "Dex3 framewise qddot RMS (rad/s²)"),
        ("arm_jerk_rms", "Arm framewise jerk RMS (rad/s³)"),
        ("dex3_jerk_rms", "Dex3 framewise jerk RMS (rad/s³)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"a": "#2563eb", "b": "#16a34a"}
    labels = {"a": "ACT-A40", "b": "ACT-B40"}
    summary = {}
    for axis, (key, label) in zip(axes.flat, specifications, strict=True):
        summary[key] = {}
        for method in ("a", "b"):
            value = trace[method][key]
            time = np.arange(len(value)) / FPS
            axis.plot(time, value, color=colors[method], linewidth=0.8, alpha=0.82, label=labels[method])
            summary[key][method] = {
                "rms": float(np.sqrt(np.mean(value**2))),
                "p95": float(np.percentile(value, 95.0)),
                "max": float(np.max(value)),
            }
        axis.set_title(label)
        axis.set_xlabel("source time (s)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Representative Paired ACT-E1 Isaac Motion Continuity", fontsize=15)
    png = FIGURES / "figure4_human_review_motion_continuity.png"
    pdf = FIGURES / "figure4_human_review_motion_continuity.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "png": {"path": str(png), "sha256": sha256_file(png)},
        "pdf": {"path": str(pdf), "sha256": sha256_file(pdf)},
        "representative_rollout_arrays": paths,
        "summary": summary,
    }


def main() -> None:
    result = read_json(RESULT)
    manifest = read_json(HELDOUT)
    if result["status"] != "PASS":
        raise RuntimeError("Experiment 3 must be PASS before human-review assets")
    records = {
        "schema_version": "paper_core_human_visual_review_assets_v1",
        "status": "PASS",
        "representative_selection_rule": manifest[
            "representative_episode_rule_frozen_before_policy_results"
        ],
        "representative_source_final_episode": int(manifest["representative_episode"]),
        "temporal_strips": temporal_strip(result, manifest),
        "motion_continuity": motion_trace(result),
        "human_judgment_pending": True,
    }
    atomic_json(
        PAPER / "source_conditioned_rollout/human_visual_review_assets.json",
        records,
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
