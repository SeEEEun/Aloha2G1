#!/usr/bin/env python3
"""Prepare paper Figures 1--3 from frozen pre-policy evidence.

Figure 4 is generated only after paired source-conditioned rollouts by
``evaluate_paper_core_source_rollouts.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from tools.doll_handoff_retargeting.common import load_common_config, load_scene
    from tools.doll_handoff_retargeting.models import G1Kinematics
    from tools.doll_handoff_retargeting.render import ReviewRenderer
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    from doll_handoff_retargeting.common import load_common_config, load_scene
    from doll_handoff_retargeting.models import G1Kinematics
    from doll_handoff_retargeting.render import ReviewRenderer


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUTPUT = ROOT / "outputs/paper_core_ab/figures"
MANIFEST = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
TABLE1 = ROOT / "outputs/paper_core_ab/tables/table1_full50_retargeting.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def figure1() -> dict[str, Any]:
    path = OUTPUT / "figure1_pipeline.svg"
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="650" viewBox="0 0 1500 650">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#334155"/></marker>
  <style>
    .title { font: 700 28px sans-serif; fill: #0f172a; }
    .head { font: 700 19px sans-serif; fill: #0f172a; }
    .body { font: 15px sans-serif; fill: #334155; }
    .small { font: 13px sans-serif; fill: #475569; }
    .box { stroke-width: 2; rx: 16; ry: 16; }
    .arrow { stroke: #334155; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }
  </style>
</defs>
<rect width="1500" height="650" fill="#f8fafc"/>
<text x="750" y="48" text-anchor="middle" class="title">Interaction-Centric Retargeting to Downstream G1 Policy Evaluation</text>

<rect x="45" y="235" width="245" height="145" class="box" fill="#e2e8f0" stroke="#64748b"/>
<text x="168" y="275" text-anchor="middle" class="head">50 ALOHA demonstrations</text>
<text x="168" y="307" text-anchor="middle" class="body">cam_high RGB + source motion</text>
<text x="168" y="334" text-anchor="middle" class="small">same recordings, timestamps, phases</text>
<text x="168" y="359" text-anchor="middle" class="small">34,478 frames at 30 Hz</text>

<rect x="360" y="105" width="300" height="155" class="box" fill="#dbeafe" stroke="#2563eb"/>
<text x="510" y="143" text-anchor="middle" class="head">Fair A retargeting</text>
<text x="510" y="176" text-anchor="middle" class="body">trajectory-centric</text>
<text x="510" y="201" text-anchor="middle" class="body">6D wrist-level transfer</text>
<text x="510" y="230" text-anchor="middle" class="small">Experiment 1: all 50 episodes</text>

<rect x="360" y="390" width="300" height="155" class="box" fill="#dcfce7" stroke="#16a34a"/>
<text x="510" y="428" text-anchor="middle" class="head">Proposed B retargeting</text>
<text x="510" y="461" text-anchor="middle" class="body">interaction-centric</text>
<text x="510" y="486" text-anchor="middle" class="body">whole-hand + bimanual ownership</text>
<text x="510" y="515" text-anchor="middle" class="small">Experiment 1: all 50 episodes</text>

<rect x="740" y="105" width="250" height="155" class="box" fill="#eff6ff" stroke="#2563eb"/>
<text x="865" y="143" text-anchor="middle" class="head">A40 supervision</text>
<text x="865" y="177" text-anchor="middle" class="body">COMMON48 → TRAIN40</text>
<text x="865" y="204" text-anchor="middle" class="small">original ALOHA RGB</text>
<text x="865" y="229" text-anchor="middle" class="small">Fair-A 28D state/action</text>

<rect x="740" y="390" width="250" height="155" class="box" fill="#f0fdf4" stroke="#16a34a"/>
<text x="865" y="428" text-anchor="middle" class="head">B40 supervision</text>
<text x="865" y="462" text-anchor="middle" class="body">same TRAIN40 episodes</text>
<text x="865" y="489" text-anchor="middle" class="small">same ALOHA RGB</text>
<text x="865" y="514" text-anchor="middle" class="small">Proposed-B 28D state/action</text>

<rect x="1060" y="105" width="190" height="155" class="box" fill="#bfdbfe" stroke="#1d4ed8"/>
<text x="1155" y="151" text-anchor="middle" class="head">ACT-A40</text>
<text x="1155" y="184" text-anchor="middle" class="body">chunk = 50</text>
<text x="1155" y="211" text-anchor="middle" class="small">identical ACT config</text>
<text x="1155" y="234" text-anchor="middle" class="small">100k-step budget</text>

<rect x="1060" y="390" width="190" height="155" class="box" fill="#bbf7d0" stroke="#15803d"/>
<text x="1155" y="436" text-anchor="middle" class="head">ACT-B40</text>
<text x="1155" y="469" text-anchor="middle" class="body">chunk = 50</text>
<text x="1155" y="496" text-anchor="middle" class="small">identical ACT config</text>
<text x="1155" y="519" text-anchor="middle" class="small">100k-step budget</text>

<rect x="1310" y="205" width="160" height="245" class="box" fill="#fef3c7" stroke="#d97706"/>
<text x="1390" y="245" text-anchor="middle" class="head">HELDOUT8</text>
<text x="1390" y="278" text-anchor="middle" class="body">Experiment 2</text>
<text x="1390" y="302" text-anchor="middle" class="small">action + FK metrics</text>
<line x1="1335" y1="326" x2="1445" y2="326" stroke="#d97706"/>
<text x="1390" y="356" text-anchor="middle" class="body">Experiment 3</text>
<text x="1390" y="380" text-anchor="middle" class="small">same source video</text>
<text x="1390" y="404" text-anchor="middle" class="small">ACT-E1 Isaac motion</text>
<text x="1390" y="430" text-anchor="middle" class="small">no physical success claim</text>

<path d="M290 285 C325 285 325 180 360 180" class="arrow"/>
<path d="M290 330 C325 330 325 465 360 465" class="arrow"/>
<path d="M660 182 L740 182" class="arrow"/>
<path d="M660 467 L740 467" class="arrow"/>
<path d="M990 182 L1060 182" class="arrow"/>
<path d="M990 467 L1060 467" class="arrow"/>
<path d="M1250 182 C1280 182 1280 275 1310 275" class="arrow"/>
<path d="M1250 467 C1280 467 1280 380 1310 380" class="arrow"/>

<text x="750" y="620" text-anchor="middle" class="small">Only retargeted state/action supervision differs between paired policy training runs.</text>
</svg>
"""
    atomic_text(path, svg)
    return {"path": str(path), "sha256": sha256_file(path)}


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def source_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode source frame {frame}")
    return image


def label_source(image: np.ndarray, source_id: str, frame: int, timestamp: float) -> np.ndarray:
    value = image.copy()
    cv2.rectangle(value, (0, 0), (640, 106), (14, 14, 18), -1)
    cv2.putText(value, "SOURCE ALOHA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(value, f"{source_id} | frame {frame:04d} | t={timestamp:6.2f}s", (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(value, "visual conditioning/provenance; no G1 onboard image", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 255), 1, cv2.LINE_AA)
    return value


def figure2() -> dict[str, Any]:
    manifest = load_json(MANIFEST)
    final_episode = int(manifest["representative_episode"])
    entry = next(
        row for row in manifest["entries"] if int(row["final_dataset_index"]) == final_episode
    )
    a = load_trajectory(Path(entry["a_trajectory_path"]))
    b = load_trajectory(Path(entry["b_trajectory_path"]))
    source_path = Path(entry["source_rgb_identity"]["canonical_video_path"])
    if sha256_file(source_path) != entry["source_rgb_identity"]["canonical_video_sha256"]:
        raise RuntimeError("representative source video changed")
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    renderer = ReviewRenderer(common, scene, g1)
    event_map = {
        str(name): int(frame)
        for name, frame in zip(b["event_names"], b["event_frames"], strict=True)
    }
    events = {"frames": event_map}
    metrics = {"baseline": {"status": "PASS"}, "proposed": {"status": "PASS"}}
    source = cv2.VideoCapture(str(source_path))
    if not source.isOpened():
        raise RuntimeError(f"could not open source video: {source_path}")
    video_path = OUTPUT / "figure2_source_vs_fair_a_vs_proposed_b.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), renderer.fps, (1920, 480)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    frames = list(range(0, int(entry["frames"]), renderer.stride))
    if frames[-1] != int(entry["frames"]) - 1:
        frames.append(int(entry["frames"]) - 1)
    try:
        for frame in frames:
            timestamp = frame / 30.0
            source_panel = label_source(
                source_frame(source, frame), entry["original_source_recording_id"], frame, timestamp
            )
            panels = [
                renderer.robot_panel(
                    "baseline", a, metrics["baseline"], events, entry["stable_episode_id"], frame, timestamp, "overview"
                ),
                renderer.robot_panel(
                    "proposed", b, metrics["proposed"], events, entry["stable_episode_id"], frame, timestamp, "overview"
                ),
            ]
            writer.write(np.hstack((source_panel, *panels)))
    finally:
        writer.release()
        source.release()
        renderer.close()

    phase_frames = manifest["complete_source_phase_audit"][str(final_episode)][
        "nine_probe_frames"
    ]
    selected = {
        "left_grasp": int(phase_frames["left_grasp_owned"]),
        "left_transport": int(phase_frames["left_transport"]),
        "handoff": int(phase_frames["dual_contact_transfer"]),
        "right_owned": int(phase_frames["right_owned"]),
        "release": int(phase_frames["release"]),
    }
    # Recreate a renderer because the full-video renderer has been closed.
    renderer = ReviewRenderer(common, scene, g1)
    source = cv2.VideoCapture(str(source_path))
    stills = {}
    try:
        for phase, frame in selected.items():
            timestamp = frame / 30.0
            image = np.hstack(
                (
                    label_source(
                        source_frame(source, frame), entry["original_source_recording_id"], frame, timestamp
                    ),
                    renderer.robot_panel(
                        "baseline", a, metrics["baseline"], events, entry["stable_episode_id"], frame, timestamp, "overview"
                    ),
                    renderer.robot_panel(
                        "proposed", b, metrics["proposed"], events, entry["stable_episode_id"], frame, timestamp, "overview"
                    ),
                )
            )
            path = OUTPUT / f"figure2_{phase}_source_vs_fair_a_vs_proposed_b.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"could not write {path}")
            stills[phase] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "frame": frame,
                "source_timestamp_seconds": timestamp,
            }
    finally:
        source.release()
        renderer.close()
    return {
        "representative_source_final_episode": final_episode,
        "representative_rule": manifest[
            "representative_episode_rule_frozen_before_policy_results"
        ],
        "video": {"path": str(video_path), "sha256": sha256_file(video_path)},
        "stills": stills,
        "kinematic_visualization": True,
        "physical_contact_claim": False,
    }


def figure3() -> dict[str, Any]:
    table = load_json(TABLE1)
    rows = table["rows"] if "rows" in table else table
    lookup = {row["metric"]: row for row in rows}
    specifications = [
        ("Wrist trajectory", "Wrist error mean / p95 / max"),
        ("Whole-hand frame", "Whole-hand error mean / p95 / max"),
        ("Bimanual relation", "Bimanual relation error mean / p95 / max"),
    ]
    values = {"Fair A": [], "Proposed B": []}
    p95 = {"Fair A": [], "Proposed B": []}
    data_rows = []
    for label, metric in specifications:
        row = lookup[metric]
        for method_label, key in (("Fair A", "fair_a"), ("Proposed B", "proposed_b")):
            value = row[key]
            if isinstance(value, str):
                parts = [float(part.strip()) for part in value.split("/")]
                mean_value, p95_value = parts[:2]
            else:
                mean_value = float(value["mean"])
                p95_value = float(value["p95"])
            values[method_label].append(mean_value)
            p95[method_label].append(p95_value)
            data_rows.append(
                {
                    "metric": label,
                    "method": method_label,
                    "mean_mm": mean_value,
                    "p95_mm": p95_value,
                }
            )
    csv_path = OUTPUT / "figure3_interaction_metrics.csv"
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)
    os.replace(temporary, csv_path)

    x = np.arange(len(specifications))
    width = 0.34
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    colors = {"Fair A": "#3b82f6", "Proposed B": "#22c55e"}
    for offset, method in ((-width / 2, "Fair A"), (width / 2, "Proposed B")):
        means = np.asarray(values[method])
        upper = np.asarray(p95[method]) - means
        axis.bar(
            x + offset,
            means,
            width,
            yerr=np.vstack((np.zeros_like(upper), upper)),
            capsize=4,
            color=colors[method],
            label=method,
            alpha=0.9,
        )
    axis.set_xticks(x, [row[0] for row in specifications])
    axis.set_ylabel("Position error (mm; bar = mean, whisker = p95)")
    axis.set_title("Full-50 Retargeting: Source-Fidelity Tradeoff")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    png = OUTPUT / "figure3_interaction_metrics.png"
    pdf = OUTPUT / "figure3_interaction_metrics.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return {
        "data": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "png": {"path": str(png), "sha256": sha256_file(png)},
        "pdf": {"path": str(pdf), "sha256": sha256_file(pdf)},
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = {
        "figure1": figure1(),
        "figure2": figure2(),
        "figure3": figure3(),
        "figure4": "GENERATED_AFTER_EXPERIMENT_3",
        "caption_drafts": {
            "figure1": "Pipeline for the paired retargeting and policy experiment. Both branches share source demonstrations, episode split, policy architecture, and optimization; only retargeted state/action supervision differs.",
            "figure2": "Kinematic visualization of the same source frames under Fair A and Proposed B. The doll is visual context; the panels do not assert physical contact or grasp success.",
            "figure3": "Full-50 source-fidelity metrics. Fair A emphasizes source wrist trajectory, whereas Proposed B emphasizes whole-hand and bimanual interaction geometry.",
            "figure4": "The same held-out source ALOHA video conditions ACT-A40 and ACT-B40 Isaac trajectories under identical ACT-E1 execution. The figure evaluates generated motion, not physical manipulation success.",
        },
    }
    atomic_json(OUTPUT / "paper_figure_asset_manifest_pre_experiment3.json", assets)
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
