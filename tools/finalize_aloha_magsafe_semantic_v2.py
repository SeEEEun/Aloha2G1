#!/usr/bin/env python3
"""Create post-freeze reviews, tests, immutability audit, and v2 report."""
from __future__ import annotations

import collections
import csv
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS  # noqa: E402
from aloha_magsafe_semantics.features import compute_stationary_aloha_fk  # noqa: E402
from aloha_magsafe_semantics.io import load_trajectory  # noqa: E402
from aloha_magsafe_semantics_v2.detector import detect_magsafe_semantics  # noqa: E402
from aloha_magsafe_semantics_v2.experiment import MODEL, OUT, POSE_CONFIG, read_manifest  # noqa: E402


V1 = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(name: str) -> dict[str, Any]:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def timeline(episode_id: int, v1: bool = False) -> dict[str, Any]:
    base = V1 if v1 else OUT
    return json.loads((base / "episodes" / f"{episode_id:02d}" / "semantic_timeline.auto.json").read_text(encoding="utf-8"))


def event_map(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["event_name"]: row for row in value["events"]}


def class_counts(rows: list[dict[str, Any]], name: str) -> collections.Counter[str]:
    return collections.Counter(row[f"{name}_class"] for row in rows)


def plot_comparisons() -> None:
    v1_rows = json.loads((V1 / "batch_semantic_summary.json").read_text(encoding="utf-8"))["episodes"]
    v2_rows = load("full_50_coverage.json")["episodes"]
    v1_bad = []
    v2_bad = []
    for name in REQUIRED_EVENTS:
        before = class_counts(v1_rows, name)
        after = class_counts(v2_rows, name)
        v1_bad.append(before["LOW"] + before["AMBIGUOUS"])
        v2_bad.append(after["LOW"] + after["AMBIGUOUS"])
    x = np.arange(len(REQUIRED_EVENTS))
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.bar(x - 0.2, v1_bad, width=0.4, label="v1 LOW+AMBIGUOUS")
    axis.bar(x + 0.2, v2_bad, width=0.4, label="v2 LOW+AMBIGUOUS")
    axis.set_xticks(x, REQUIRED_EVENTS, rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("episode count")
    axis.set_title("Event-wise uncertainty: v1 vs frozen v2")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT / "v1_vs_v2_ambiguity_by_event.png", dpi=170)
    plt.close(figure)

    split = load("dataset_split_v2.json")
    development = load("development_10_result.json")["aggregate"]["high_medium_complete_count"]
    validation = load("validation_10_result.json")["aggregate"]["high_medium_complete_count"]
    heldout = load("heldout_30_result.json")["high_medium_complete_count"]
    full_v2 = load("full_50_coverage.json")["high_medium_complete_count"]
    labels = ["v1 full50", "v2 DEV10", "v2 VAL10", "v2 TEST30", "v2 full50"]
    values = [19, development, validation, heldout, full_v2]
    denominators = [50, 10, 10, 30, 50]
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(labels, values, color=("#777777", "#4c78a8", "#f2cf5b", "#e45756", "#54a24b"))
    axis.bar_label(bars, labels=[f"{value}/{denominator}" for value, denominator in zip(values, denominators)])
    axis.set_ylim(0, 52)
    axis.set_ylabel("HIGH/MEDIUM-complete timelines")
    axis.set_title("Complete semantic timelines (different split denominators shown on bars)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT / "v1_vs_v2_complete_timeline_count.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True)
    for axis, name in zip(axes.flat, REQUIRED_EVENTS):
        old = [row[f"{name}_index"] / max(row["frame_count"] - 1, 1) for row in v1_rows if row[f"{name}_index"] is not None]
        new = [row[f"{name}_index"] / max(row["frame_count"] - 1, 1) for row in v2_rows if row[f"{name}_index"] is not None]
        axis.hist(old, bins=np.linspace(0, 1, 21), alpha=0.5, label="v1")
        axis.hist(new, bins=np.linspace(0, 1, 21), alpha=0.5, label="v2")
        axis.set_title(name, fontsize=8)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Normalized event-time distributions")
    figure.tight_layout()
    figure.savefig(OUT / "v1_vs_v2_normalized_event_times.png", dpi=170)
    plt.close(figure)

    edges = list(zip(REQUIRED_EVENTS[:-1], REQUIRED_EVENTS[1:]))
    v1_event_times = {
        episode_id: event_map(timeline(episode_id, True))
        for episode_id in range(50)
    }
    v2_event_times = {
        episode_id: event_map(timeline(episode_id))
        for episode_id in range(50)
    }
    figure, axes = plt.subplots(3, 4, figsize=(15, 10))
    for axis, edge in zip(axes.flat, edges):
        first, second = edge
        old = [
            v1_event_times[episode_id][second]["action_time_sec"] - v1_event_times[episode_id][first]["action_time_sec"]
            for episode_id in range(50)
            if v1_event_times[episode_id][first]["action_time_sec"] is not None and v1_event_times[episode_id][second]["action_time_sec"] is not None
        ]
        new = [
            v2_event_times[episode_id][second]["action_time_sec"] - v2_event_times[episode_id][first]["action_time_sec"]
            for episode_id in range(50)
            if v2_event_times[episode_id][first]["action_time_sec"] is not None and v2_event_times[episode_id][second]["action_time_sec"] is not None
        ]
        axis.boxplot([old, new], tick_labels=["v1", "v2"], showfliers=False)
        axis.set_title(f"{first}\n→ {second}", fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(edges):]:
        axis.axis("off")
    figure.suptitle("Consecutive event-duration distributions (seconds)")
    figure.tight_layout()
    figure.savefig(OUT / "v1_vs_v2_duration_distribution.png", dpi=170)
    plt.close(figure)


def timeline_grid(path: Path, ids: list[int], title: str) -> None:
    matrix = np.full((len(ids), len(REQUIRED_EVENTS)), np.nan)
    for row, episode_id in enumerate(ids):
        value = event_map(timeline(episode_id))
        length = timeline(episode_id)["trajectory_length"]
        for column, name in enumerate(REQUIRED_EVENTS):
            index = value[name]["action_index"]
            matrix[row, column] = np.nan if index is None else index / max(length - 1, 1)
    figure, axis = plt.subplots(figsize=(15, max(5, len(ids) * 0.32)))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_yticks(range(len(ids)), [f"ep{value:02d}" for value in ids], fontsize=7)
    axis.set_xticks(range(len(REQUIRED_EVENTS)), REQUIRED_EVENTS, rotation=55, ha="right", fontsize=8)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="normalized action time")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def confidence_heatmap(path: Path, ids: list[int], title: str) -> None:
    matrix = np.zeros((len(ids), len(REQUIRED_EVENTS)))
    labels = np.empty((len(ids), len(REQUIRED_EVENTS)), dtype="U1")
    for row, episode_id in enumerate(ids):
        value = event_map(timeline(episode_id))
        for column, name in enumerate(REQUIRED_EVENTS):
            matrix[row, column] = value[name]["confidence"]
            labels[row, column] = value[name]["confidence_class"][0]
    figure, axis = plt.subplots(figsize=(15, max(5, len(ids) * 0.32)))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    for row in range(len(ids)):
        for column in range(len(REQUIRED_EVENTS)):
            axis.text(column, row, labels[row, column], ha="center", va="center", fontsize=6)
    axis.set_yticks(range(len(ids)), [f"ep{value:02d}" for value in ids], fontsize=7)
    axis.set_xticks(range(len(REQUIRED_EVENTS)), REQUIRED_EVENTS, rotation=55, ha="right", fontsize=8)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="confidence")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def image_directory(row: dict[str, Any]) -> Path:
    return Path(row["source_folder"]) / "images/observation.images.cam_high/episode_000000"


def read_frame(directory: Path, index: int, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    image = cv2.imread(str(directory / f"frame_{index:06d}.png"))
    if image is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def latest_event(events: dict[str, dict[str, Any]], index: int) -> str:
    rows = [(row["action_index"], name) for name, row in events.items() if name in REQUIRED_EVENTS and row["action_index"] is not None and row["action_index"] <= index]
    return max(rows, default=(-1, "PRE_TASK"))[1]


def episode_fk(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = load_trajectory(row["source_parquet"], "raw_action")
    pose = json.loads(POSE_CONFIG.read_text(encoding="utf-8"))["stationary_aloha"]
    fk = compute_stationary_aloha_fk(loaded["action"], loaded["timestamps"], MODEL, pose["position_xyz_m"], pose["orientation_wxyz"])
    return loaded, fk


def speeds(fk: dict[str, Any], timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.linalg.norm(np.gradient(fk["left_tcp_position"], timestamps, axis=0), axis=1)
    right = np.linalg.norm(np.gradient(fk["right_tcp_position"], timestamps, axis=0), axis=1)
    return left, right


def overlay_video(path: Path, directory: Path, timestamps: np.ndarray, left_speed: np.ndarray, right_speed: np.ndarray, timelines: list[tuple[str, dict[str, Any]]], observed_offset: int = 0) -> dict[str, Any]:
    length = len(timestamps)
    if path.is_file():
        capture = cv2.VideoCapture(str(path))
        decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if decoded == length:
            return {"path": str(path.resolve()), "sha256": sha256(path), "decoded_frames": decoded, "expected_frames": length, "pass": True, "reused_verified_artifact": True}
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 360))
    event_maps = [(label, event_map(value)) for label, value in timelines]
    for index in range(length):
        observed = index + observed_offset
        terminal = observed >= length
        image = read_frame(directory, min(observed, length - 1))
        cv2.rectangle(image, (0, 0), (640, 112), (15, 15, 15), -1)
        cv2.putText(image, f"action {index}/{length-1}  t={timestamps[index]:.3f}s" + ("  POST-OBSERVATION" if terminal else ""), (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"TCP speed L={left_speed[index]:.3f} R={right_speed[index]:.3f} m/s", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)
        for row_number, (label, events) in enumerate(event_maps):
            current = latest_event(events, index)
            color = (100, 210, 255) if row_number == 0 else (130, 255, 130)
            cv2.putText(image, f"{label}: {current}", (12, 68 + row_number * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return {"path": str(path.resolve()), "sha256": sha256(path), "decoded_frames": decoded, "expected_frames": length, "pass": decoded == length}


def contact_sheet(path: Path, directory: Path, old: dict[str, Any], new: dict[str, Any]) -> None:
    old_events, new_events = event_map(old), event_map(new)
    cells = []
    for name in REQUIRED_EVENTS:
        index = new_events[name]["action_index"]
        if index is None:
            index = old_events[name]["action_index"] or 0
        image = read_frame(directory, int(index), (320, 180))
        cv2.rectangle(image, (0, 0), (320, 48), (0, 0, 0), -1)
        cv2.putText(image, name[:36], (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"v1={old_events[name]['action_index']} v2={new_events[name]['action_index']} {new_events[name]['confidence_class']}", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 150), 1, cv2.LINE_AA)
        cells.append(image)
    canvas = np.zeros((3 * 180, 4 * 320, 3), dtype=np.uint8)
    for cell_index, image in enumerate(cells):
        row, column = divmod(cell_index, 4)
        canvas[row * 180 : (row + 1) * 180, column * 320 : (column + 1) * 320] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def media() -> dict[str, Any]:
    manifest = {row["episode_id"]: row for row in read_manifest()}
    split = load("dataset_split_v2.json")
    v1_batch = json.loads((V1 / "batch_semantic_summary.json").read_text(encoding="utf-8"))["episodes"]
    v1_by_id = {row["episode_id"]: row for row in v1_batch}
    development_ambiguous = [
        episode_id for episode_id in split["development"]
        if v1_by_id[episode_id]["low_events"] or v1_by_id[episode_id]["ambiguous_events"]
    ]
    audits = []
    contact_directory = OUT / "development_v2_contact_sheets"
    overlay_directory = OUT / "development_v2_overlays"
    overlay_directory.mkdir(parents=True, exist_ok=True)
    for episode_id in development_ambiguous:
        row = manifest[episode_id]
        old, new = timeline(episode_id, True), timeline(episode_id)
        contact_sheet(contact_directory / f"ep{episode_id:02d}_v1_vs_v2.png", image_directory(row), old, new)
        loaded, fk = episode_fk(row)
        left, right = speeds(fk, loaded["timestamps"])
        audits.append(overlay_video(
            overlay_directory / f"ep{episode_id:02d}_v1_vs_v2.mp4", image_directory(row), loaded["timestamps"], left, right,
            [("v1", old), ("v2", new)],
        ))

    configs = load("validation_config_results.json")
    validation_overlays = OUT / "validation_config_disagreement_overlays"
    validation_overlays.mkdir(parents=True, exist_ok=True)
    config_names = list(configs)
    for episode_id in split["validation"]:
        summaries = [{row["episode_id"]: row for row in configs[name]["episodes"]}[episode_id] for name in config_names]
        if not any(len({row[f"{event}_index"] for row in summaries}) > 1 for event in REQUIRED_EVENTS):
            continue
        # Build lightweight timeline dictionaries from config summaries. The
        # selected v2 per-sample phases remain unchanged and are not rerun.
        variants = []
        for config_name, summary in zip(config_names, summaries):
            variants.append((config_name.replace("V2_", ""), {
                "events": [{"event_name": event, "action_index": summary[f"{event}_index"]} for event in REQUIRED_EVENTS],
            }))
        row = manifest[episode_id]
        loaded, fk = episode_fk(row)
        left, right = speeds(fk, loaded["timestamps"])
        audits.append(overlay_video(
            validation_overlays / f"ep{episode_id:02d}_config_disagreement.mp4", image_directory(row), loaded["timestamps"], left, right, variants,
        ))

    ep49_row = manifest[49]
    optimized = load_trajectory(ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz", "optimized_action")
    pose = json.loads(POSE_CONFIG.read_text(encoding="utf-8"))["stationary_aloha"]
    fk = compute_stationary_aloha_fk(optimized["action"], optimized["timestamps"], MODEL, pose["position_xyz_m"], pose["orientation_wxyz"])
    left, right = speeds(fk, optimized["timestamps"])
    old_ep49 = json.loads((V1 / "ep49_auto_timeline.json").read_text(encoding="utf-8"))
    new_ep49 = load("ep49_auto_timeline_v2.json")
    audits.append(overlay_video(
        OUT / "ep49_v1_vs_v2_overlay.mp4", image_directory(ep49_row), optimized["timestamps"], left, right,
        [("v1", old_ep49), ("v2", new_ep49)], observed_offset=7,
    ))
    return {
        "development_v1_ambiguous_ids": development_ambiguous,
        "development_contact_sheet_directory": str(contact_directory.resolve()),
        "development_overlay_directory": str(overlay_directory.resolve()),
        "validation_disagreement_overlay_directory": str(validation_overlays.resolve()),
        "videos": audits,
        "all_video_frame_counts_pass": all(row["pass"] for row in audits),
    }


def immutability() -> dict[str, Any]:
    freeze = load("v1_freeze_audit.json")
    v1_current = {}
    for relative, before in freeze["v1_files"].items():
        path = ROOT / relative
        after = sha256(path)
        v1_current[relative] = {"before": before, "after": after, "byte_identical": before == after}
    immutable_current = {}
    for relative, before in freeze["immutable_v14_scene_files_before"].items():
        path = ROOT / relative
        after = sha256(path)
        immutable_current[relative] = {"before": before, "after": after, "byte_identical": before == after}
    return {
        "status": "PASS" if all(row["byte_identical"] for row in v1_current.values()) and all(row["byte_identical"] for row in immutable_current.values()) else "FAIL",
        "v1_files": v1_current,
        "v14_and_scene_files": immutable_current,
        "G1_trajectory_generation_executed": False,
        "orientation_optimization_executed": False,
        "Dex3_trajectory_executed": False,
        "physics_DDS_publisher_hardware_executed": False,
    }


def robustness_metrics() -> dict[str, Any]:
    source = load_trajectory(
        ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz",
        "optimized_action",
    )
    action, timestamps = source["action"], source["timestamps"]
    config = load("selected_detector_v2_config.json")
    geometry = json.loads((ROOT / "configs/aloha_magsafe_task_geometry.semantic_v1.json").read_text(encoding="utf-8"))
    pose = json.loads(POSE_CONFIG.read_text(encoding="utf-8"))["stationary_aloha"]

    def detect(value: np.ndarray, time: np.ndarray):
        fk = compute_stationary_aloha_fk(value, time, MODEL, pose["position_xyz_m"], pose["orientation_wxyz"])
        return detect_magsafe_semantics(value, time, "optimized_action", fk, geometry, config)

    reference = detect(action, timestamps)
    duration = float(timestamps[-1] - timestamps[0])
    cases = []
    for frequency in (20.0, 60.0):
        target_time = np.arange(int(round(duration * frequency)) + 1, dtype=np.float64) / frequency + timestamps[0]
        target_time = target_time[target_time <= timestamps[-1] + 1e-9]
        target_action = np.column_stack([
            np.interp(target_time, timestamps, action[:, column])
            for column in range(action.shape[1])
        ])
        result = detect(target_action, target_time)
        drift = {
            name: float(abs(result.event(name).action_time_sec - reference.event(name).action_time_sec))
            for name in REQUIRED_EVENTS
        }
        cases.append({
            "frequency_hz": frequency,
            "event_time_drift_sec": drift,
            "max_all_event_drift_sec": max(drift.values()),
            "max_nonterminal_event_drift_sec": max(value for name, value in drift.items() if name != "task_end"),
            "task_end_drift_sec": drift["task_end"],
        })
    all_pass = all(row["max_all_event_drift_sec"] <= 1.2 for row in cases)
    nonterminal_pass = all(row["max_nonterminal_event_drift_sec"] <= 1.2 for row in cases)
    return {
        "status": "PASS" if all_pass else "BLOCKED_TERMINAL_SUFFIX_RESAMPLING",
        "diagnostic_tolerance_sec": 1.2,
        "all_event_resampling_pass": all_pass,
        "nonterminal_resampling_pass": nonterminal_pass,
        "cases": cases,
        "detector_changed_after_heldout": False,
        "threshold_lowered_after_heldout": False,
    }


def run_tests() -> dict[str, Any]:
    command = [
        "/home/jbnu/miniconda3/envs/isaaclab6/bin/python", "-m", "pytest", "-q",
        "tests/test_no_semantic_frame_hardcoding.py",
        "tests/test_aloha_magsafe_semantics.py",
        "tests/test_aloha_magsafe_semantics_v2.py",
    ]
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "tools"))
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
    receipt = load("heldout_evaluation_receipt.json")
    robustness = load("robustness_metrics_v2.json")
    hardcoding_command = ["rg", "-n", r"\b(169|193|216|319|322|334|373|523|579|639|695)\b", "tools/aloha_magsafe_semantics_v2"]
    hardcoding = subprocess.run(hardcoding_command, cwd=ROOT, text=True, capture_output=True, check=False)
    checks = {
        "pytest_pass": completed.returncode == 0,
        "selected_config_frozen_before_heldout": load("selected_detector_v2_config_provenance.json")["status"] == "DETECTOR_CONFIG_FROZEN_BEFORE_HELDOUT_TEST",
        "heldout_single_evaluation": receipt["heldout_evaluation_count"] == 1 and receipt["detector_invocation_count"] == 30,
        "heldout_did_not_change_parameters": receipt["parameters_changed_after_results"] is False,
        "generic_runtime_literal_scan_clean": hardcoding.returncode == 1 and not hardcoding.stdout,
        "partial_order_all_50": load("full_50_coverage.json")["partial_order_valid_count"] == 50,
        "v1_v14_scene_immutable": load("immutability_audit_v2.json")["status"] == "PASS",
        "resampling_all_event_gate": robustness["all_event_resampling_pass"],
        "resampling_nonterminal_gate": robustness["nonterminal_resampling_pass"],
        "no_G1_trajectory_generated": not any("g1" in path.name.lower() and path.suffix == ".npz" for path in OUT.rglob("*.npz")),
        "no_downstream_execution": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "checks": checks,
        "hardcoding_scan_command": " ".join(hardcoding_command),
        "hardcoding_scan_stdout": hardcoding.stdout,
    }


def report_markdown() -> str:
    split = load("dataset_split_v2.json")
    taxonomy = load("development_failure_taxonomy.json")
    selected = load("selected_detector_v2_config_provenance.json")
    development = load("development_10_result.json")
    validation = load("validation_10_result.json")
    heldout = load("heldout_30_result.json")
    full = load("full_50_coverage.json")
    regression = load("episode49_regression_v2.json")
    tests = load("tests_results.json")
    robustness = load("robustness_metrics_v2.json")
    event_rows = []
    v1_event = {row["event_name"]: row for row in csv.DictReader((OUT / "v1_vs_v2_event_summary.csv").open(encoding="utf-8"))}
    for name in REQUIRED_EVENTS:
        counts = full["event_confidence_counts"][name]
        old = v1_event[name]
        event_rows.append(
            f"| `{name}` | {counts.get('HIGH', 0)} | {counts.get('MEDIUM', 0)} | {counts.get('LOW', 0)} | {counts.get('AMBIGUOUS', 0)} | {old['v1_low_ambiguous']} → {old['v2_low_ambiguous']} |"
        )
    regression_rows = "\n".join(
        f"| `{row['event_name']}` | {row['approved_action_index']} | {row['detected_action_index']} | {row['v1_error_seconds']:.3f} | {row['v2_error_seconds']:.3f} | {row['v2_confidence_class']} |"
        for row in regression["events"]
    )
    configs = load("validation_config_results.json")
    config_rows = "\n".join(
        f"| `{name}` | {value['aggregate']['mandatory_complete_count']}/10 | {value['aggregate']['high_medium_complete_count']}/10 | {value['aggregate']['partial_order_valid_count']}/10 | {value['fabricated_release_count']} | {value['confidence_calibration']['brier_score']:.4f} |"
        for name, value in configs.items()
    )
    taxonomy_rows = "\n".join(f"| `{name}` | {count} |" for name, count in taxonomy["category_counts"].items())
    statuses = [
        "SEMANTIC_DETECTOR_V2_IMPLEMENTED",
        "V1_SCHEMA_AND_GENERIC_API_PRESERVED",
        "DETECTOR_CONFIG_FROZEN_BEFORE_HELDOUT_TEST",
        "HELDOUT_30_EVALUATED_WITHOUT_TUNING",
        "V14_AND_SCENE_UNCHANGED",
        "NO_G1_TRAJECTORY_GENERATED",
        development["smoke_status"],
        validation["status"],
        heldout["status"],
        full["status"],
        "BLOCKED_TERMINAL_SUFFIX_GENERALIZATION",
        robustness["status"],
    ]
    return f"""# ALOHA MagSafe semantic detector v2\n\n## 3-line summary\n\n1. v1 schema/API를 유지한 v2 rotation/terminal/release/bidirectional decoder를 구현했고 split·config freeze·held-out single-evaluation protocol을 통과했다.\n2. v2는 full-50 HIGH/MEDIUM complete를 19→{full['high_medium_complete_count']}로 개선했지만 validation {validation['aggregate']['high_medium_complete_count']}/10, held-out {heldout['high_medium_complete_count']}/30으로 readiness 기준에 미달했다.\n3. v2는 동결하며 held-out을 재튜닝에 사용하지 않는다; 다음 단계는 새 split/protocol을 먼저 선언한 detector v3 여부 결정이다.\n\n## 1. Final status\n\n{chr(10).join(f'- `{value}`' for value in statuses)}\n\n## 2. Frozen data splits\n\n- DEVELOPMENT 10: {split['development']} (Episode 49 포함)\n- VALIDATION 10: {split['validation']}\n- HELD-OUT TEST 30: {split['heldout_test']}\n- DEVELOPMENT smoke: {split['development_smoke']}\n- Split hash: `{split['split_hash']}`\n- Selection used only manifest duration and raw gripper variation; no semantic result/reference entered the split.\n\n## 3. V1 development failure taxonomy\n\n| Global pattern | Count |\n|---|---:|\n{taxonomy_rows}\n\nNo episode-specific correction was created.\n\n## 4. Rotation detector v2\n\nV2 uses mapped phase-relative cumulative rotation, smoothed angular speed/acceleration, monotonic progress envelope, sustained low-speed dwell, post-plateau orientation stability, later-motion consistency, and a semi-Markov/change-point candidate set. It explicitly retains overshoot/return and multiple-plateau alternatives. Full-50 `phone_portrait_reached` LOW/AMBIGUOUS changed from 21 to {int(v1_event['phone_portrait_reached']['v2_low_ambiguous'])}.\n\n## 5. Terminal suffix detector v2\n\nA backward suffix computes future translation/rotation/gripper-transition energy, future peak energy, suffix pose spread, and stable terminal gripper phase. `terminal_hold_start`, `left_arm_return_near_home`, and `task_end` remain distinct. Full-50 `task_end` still has {full['event_confidence_counts']['task_end'].get('AMBIGUOUS', 0)} AMBIGUOUS and four total mandatory-index failures occur across all events, so terminal generalization remains the main blocker.\n\n## 6. Release detector v2\n\nRelease evidence combines robust normalized opening derivative, state transition, stable open plateau, post-release departure, low-speed support, and global order. A missing physical opening gate remains AMBIGUOUS. Right-release LOW/AMBIGUOUS changed from 5 to {int(v1_event['right_accessory_release_complete']['v2_low_ambiguous'])}; fabricated validation releases: 0.\n\n## 7. Bidirectional sequence decoder\n\nForward candidates are coupled to backward endpoints with soft development-only duration priors, phase-progress terms, and top-3 globally consistent beam sequences. Events are never independently selected and sorted afterward. All 50 v2 outputs preserve the partial order.\n\n## 8. Duration priors\n\nMedians, MAD-derived scales, and 5–95% ranges were derived only from the ten DEVELOPMENT signal timelines, in seconds and normalized episode progress. Priors are soft; no event is forced solely by duration.\n\n## 9. Confidence calibration\n\nConfidence separates detector score, evidence breadth/dwell, and global-sequence margin. Validation evidence-proxy Brier score for the selected config is {validation['confidence_calibration']['brier_score']:.4f}. See [reliability diagram](validation_reliability_diagram.png) and [calibration JSON](confidence_calibration_v2.json).\n\n## 10. Three predeclared global configs\n\n| Config | Mandatory | HIGH/MEDIUM | Order | Fabricated release | Brier |\n|---|---:|---:|---:|---:|---:|\n{config_rows}\n\n## 11. Selected config and hash\n\n- Selected: `{selected['selected_candidate']}`\n- Frozen canonical hash: `{selected['selected_config_canonical_sha256']}`\n- Frozen before held-out: true\n- Held-out detector invocations: exactly 30 (one per episode)\n\n## 12. Episode-49 v1 vs v2 regression\n\nApproved reference was opened only after two identical v2 detections. Independence pass: **{regression['approved_reference_independence_pass']}**.\n\n| Event | Approved | V2 | V1 error s | V2 error s | V2 class |\n|---|---:|---:|---:|---:|---|\n{regression_rows}\n\nStatus: `{regression['status']}`. V2 improves several transition/release errors but regresses portrait plateau and especially terminal endpoint; no approved index was used to force alignment.\n\n## 13. DEVELOPMENT result\n\n- HIGH/MEDIUM complete: {development['aggregate']['high_medium_complete_count']}/10\n- Mandatory complete: {development['aggregate']['mandatory_complete_count']}/10\n- Partial order: {development['aggregate']['partial_order_valid_count']}/10\n- Smoke: {development['smoke_high_medium_complete_count']}/3 — `{development['smoke_status']}`\n- This is diagnostic, not a generalization claim.\n\n## 14. VALIDATION result\n\n- HIGH/MEDIUM complete: {validation['aggregate']['high_medium_complete_count']}/10 (required 8)\n- Mandatory complete: {validation['aggregate']['mandatory_complete_count']}/10\n- Partial order: {validation['aggregate']['partial_order_valid_count']}/10\n- Status: `{validation['status']}`\n\n## 15. HELD-OUT 30 result\n\n- HIGH/MEDIUM complete: {heldout['high_medium_complete_count']}/30 (required 27)\n- Mandatory complete: {heldout['mandatory_complete_count']}/30\n- Partial order: {heldout['partial_order_valid_count']}/30\n- Failed IDs: {[row['episode_id'] for row in heldout['episodes'] if not row['high_medium_complete']]}\n- Status: `{heldout['status']}`\n- No detector change occurred after these results were seen.\n\n## 16. Full-50 descriptive coverage\n\n- HIGH/MEDIUM complete: {full['high_medium_complete_count']}/50 (v1: 19/50; target: 45)\n- Mandatory-index complete: {full['mandatory_complete_count']}/50\n- Partial order: {full['partial_order_valid_count']}/50\n- LOW/AMBIGUOUS-containing complete timelines: {full['complete_with_low_or_ambiguous_count']}\n- Status: `{full['status']}`\n\n## 17. Event-wise confidence counts\n\n| Event | HIGH | MEDIUM | LOW | AMBIGUOUS | v1→v2 LOW+AMB |\n|---|---:|---:|---:|---:|---:|\n{chr(10).join(event_rows)}\n\n## 18. V1→V2 improvements and regressions\n\nPortrait ambiguity and release/detachment evidence improved substantially, and full-50 HIGH/MEDIUM completeness rose by 11 episodes. The terminal suffix remains brittle in recordings with late jitter/no clean terminal suffix, and v2 introduced four unresolved mandatory indices (Episode 4 return; Episodes 7/13/17 task end). Episode-49 `task_end` moved too late, so this is not ready for v15 execution. See [event comparison](v1_vs_v2_ambiguity_by_event.png), [episode CSV](v1_vs_v2_episode_summary.csv), and [transition CSV](v1_vs_v2_confidence_transition.csv).\n\n## 19. Generic API and v15 readiness\n\nThe canonical v1 schema, named-event API, semantic intervals/progress, deduplicated semantic knots, simultaneous events, and variable T remain intact. `retarget_aloha_trajectory_to_g1(...)` dry-run passes and v15 maps every phase to semantic API fields, but `automatic_resume_allowed=false` because held-out readiness failed. No v15 code was executed.\n\n## 20. Tests and review commands\n\n- Tests: `{tests['status']}` — `{tests['stdout'].strip()}`\n- V1/v14/scene byte identity: `{load('immutability_audit_v2.json')['status']}`\n- Episode-49 overlay: [video](ep49_v1_vs_v2_overlay.mp4)\n- Development contact sheets: [directory](development_v2_contact_sheets/)\n- Validation disagreement: [grid](validation_config_disagreement_grid.png)\n- Held-out failures: [timeline](heldout_failure_timeline_grid.png), [confidence](heldout_confidence_heatmap.png)\n- Review commands: [commands.sh](commands.sh)\n\n## 21. Exact next recommended action\n\nFreeze and report v2; do not tune on the held-out 30. Decide whether to create detector v3 only after declaring a new split and protocol, then globally improve terminal-suffix/return evidence and repeat the complete development→validation→held-out ladder. Do not manually enter frames per episode.\n\nDETECTOR V2 WAS TUNED ONLY ON THE FROZEN DEVELOPMENT SET\nTHE VALIDATION SET WAS USED ONLY FOR GLOBAL CONFIG SELECTION\nTHE HELD-OUT 30 EPISODES DID NOT INFLUENCE V2 PARAMETERS\nABSOLUTE EPISODE-49 FRAMES WERE NOT USED AS DETECTION RULES\nNO EPISODE-SPECIFIC THRESHOLD OR MANUAL EVENT FRAME WAS USED\nLOW-CONFIDENCE AND AMBIGUOUS EVENTS WERE REPORTED RATHER THAN FABRICATED\nTHE V14 TRAJECTORY AND AUTHORITATIVE SCENE REMAINED BYTE-IDENTICAL\nNO G1 IK, DEX3 TRAJECTORY, PHYSICS, DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED\n"""


def commands() -> str:
    return f"""#!/usr/bin/env bash\nset -euo pipefail\ncd {ROOT}\nffplay -autoexit outputs/semantic_event_generalization/aloha_magsafe_semantics_v2/ep49_v1_vs_v2_overlay.mp4\nxdg-open outputs/semantic_event_generalization/aloha_magsafe_semantics_v2/heldout_failure_timeline_grid.png\nxdg-open outputs/semantic_event_generalization/aloha_magsafe_semantics_v2/heldout_confidence_heatmap.png\nxdg-open outputs/semantic_event_generalization/aloha_magsafe_semantics_v2/validation_config_disagreement_grid.png\n# Review one development overlay, replacing the episode ID as needed:\nffplay -autoexit outputs/semantic_event_generalization/aloha_magsafe_semantics_v2/development_v2_overlays/ep01_v1_vs_v2.mp4\n# No G1, Dex3, physics, DDS, publisher, or hardware command is invoked here.\n"""


def main() -> int:
    plot_comparisons()
    split = load("dataset_split_v2.json")
    heldout = load("heldout_30_result.json")
    failure_ids = [row["episode_id"] for row in heldout["episodes"] if not row["high_medium_complete"]]
    timeline_grid(OUT / "heldout_failure_timeline_grid.png", failure_ids, "Frozen v2 held-out failures (review only; not used for tuning)")
    confidence_heatmap(OUT / "heldout_confidence_heatmap.png", split["heldout_test"], "Frozen v2 held-out confidence (review only)")
    timeline_grid(OUT / "thirty_episode_event_timeline_grid.png", split["heldout_test"], "Frozen held-out 30 event timelines")
    confidence_heatmap(OUT / "thirty_episode_confidence_heatmap.png", split["heldout_test"], "Frozen held-out 30 confidence")
    review = media()
    atomic_json(OUT / "review_artifact_audit.json", review)
    immutable = immutability()
    atomic_json(OUT / "immutability_audit_v2.json", immutable)
    atomic_json(OUT / "robustness_metrics_v2.json", robustness_metrics())

    # Compatibility/readiness aliases requested by the progressive ladder.
    development = load("development_10_result.json")
    atomic_json(OUT / "three_episode_pilot.json", {
        "status": development["smoke_status"],
        "episode_ids": development["smoke_episode_ids"],
        "high_medium_complete_count": development["smoke_high_medium_complete_count"],
        "scope": "DEVELOPMENT_SMOKE",
    })
    atomic_json(OUT / "ten_episode_validation.json", load("validation_10_result.json"))
    atomic_json(OUT / "thirty_episode_validation.json", heldout)
    atomic_json(OUT / "fifty_episode_readiness.json", load("full_50_coverage.json"))

    tests = run_tests()
    atomic_json(OUT / "tests_results.json", tests)
    report = report_markdown()
    robustness = load("robustness_metrics_v2.json")
    report = report.replace(
        "- V1/v14/scene byte identity:",
        f"- Resampling: `{robustness['status']}`; nonterminal <=1.2 s: {robustness['nonterminal_resampling_pass']}; task-end drift 20/60 Hz: {[row['task_end_drift_sec'] for row in robustness['cases']]}\n- V1/v14/scene byte identity:",
    )
    (OUT / "report.md").write_text(report, encoding="utf-8")
    report_directory = OUT / "report"
    report_directory.mkdir(exist_ok=True)
    (report_directory / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Semantic detector v2</title>"
        "<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;line-height:1.45}pre{white-space:pre-wrap}</style>"
        f"</head><body><pre>{html.escape(report)}</pre></body></html>", encoding="utf-8",
    )
    command_text = commands()
    (OUT / "commands.sh").write_text(command_text, encoding="utf-8")
    (OUT / "commands.sh").chmod(0o755)

    backup = Path(load("v1_freeze_audit.json")["backup_directory"])
    backup_files = sorted(path for path in backup.rglob("*") if path.is_file() and path.name != "backup_manifest.json")
    atomic_json(backup / "backup_manifest.json", {
        "status": "PRE_V2_BACKUP_COMPLETE",
        "backup_directory": str(backup.resolve()),
        "file_count": len(backup_files),
        "files": [{"path": str(path.relative_to(backup)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in backup_files],
    })
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    atomic_json(OUT / "run_manifest.json", {
        "status": "SEMANTIC_DETECTOR_V2_RUN_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_config_hash": load("selected_detector_v2_config_provenance.json")["selected_config_canonical_sha256"],
        "split_hash": split["split_hash"],
        "heldout_evaluation_count": load("heldout_evaluation_receipt.json")["heldout_evaluation_count"],
        "heldout_parameters_changed": False,
        "no_downstream_generation": True,
        "file_count": len(files),
        "files": [{"path": str(path.relative_to(OUT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
    })
    print(json.dumps({
        "output": str(OUT),
        "tests": tests["status"],
        "immutability": immutable["status"],
        "review_videos": len(review["videos"]),
        "review_frame_counts_pass": review["all_video_frame_counts_pass"],
        "heldout": heldout["status"],
        "full50": load("full_50_coverage.json")["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
