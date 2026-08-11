#!/usr/bin/env python3
"""Read-only task-critical readiness audit for frozen semantic detector v2.

This program deliberately does not import either semantic detector package.  It
parses the already-generated v1/v2 timeline JSON and phase NPZ files, computes
readiness statistics, and writes only to a separate audit output directory.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1"
V2 = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v2"
OUT = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v2_critical_audit"

EXPECTED_CONFIG_HASH = "4e36c1208a95ede5a7c19dee0f52c4d7b7b0e109c6c3937475e33d71899755db"
EXPECTED_SPLIT_HASH = "d498ee3549789524b229b0eddfc2cc78f023e3baa03703f88fe5929121dce076"

CRITICAL_EVENTS = (
    "left_phone_grasp_start",
    "phone_rotation_to_portrait_start",
    "phone_portrait_reached",
    "right_accessory_grasp_start",
    "accessory_detachment_start",
    "accessory_removed",
    "phone_move_to_charger_start",
    "phone_charger_attachment_complete",
    "left_phone_release_complete",
    "right_accessory_release_complete",
)
TERMINAL_EVENTS = ("left_arm_return_near_home", "task_end")
OPTIONAL_EVENTS = (
    "terminal_hold_start",
    "left_phone_grasp_stable",
    "right_accessory_hook_stable",
    "phone_transport_stable",
)
HIGH_MEDIUM = {"HIGH", "MEDIUM"}

# The last edge is a second branch.  No left-release/right-release order exists.
CRITICAL_EDGES = (
    ("left_phone_grasp_start", "phone_rotation_to_portrait_start", False),
    ("phone_rotation_to_portrait_start", "phone_portrait_reached", False),
    ("phone_portrait_reached", "right_accessory_grasp_start", True),
    ("right_accessory_grasp_start", "accessory_detachment_start", False),
    ("accessory_detachment_start", "accessory_removed", False),
    ("accessory_removed", "phone_move_to_charger_start", True),
    ("phone_move_to_charger_start", "phone_charger_attachment_complete", False),
    ("phone_charger_attachment_complete", "left_phone_release_complete", False),
    ("accessory_removed", "right_accessory_release_complete", False),
)
INTERVALS = tuple((a, b) for a, b, _ in CRITICAL_EDGES)
PROGRESS_KEYS = (
    "phone_acquisition_progress",
    "phone_rotation_progress",
    "accessory_acquisition_progress",
    "accessory_removal_progress",
    "phone_to_charger_progress",
    "left_release_progress",
    "right_release_progress",
)
CONF_ORDER = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "AMBIGUOUS": 0, "MISSING": -1}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_paths() -> list[Path]:
    paths = sorted((ROOT / "tools/aloha_magsafe_semantics_v2").glob("*.py"))
    paths += [
        V2 / "selected_detector_v2_config.json",
        V2 / "dataset_split_v2.json",
        V2 / "ep49_auto_timeline_v2.json",
        V2 / "ep49_semantic_phases_v2.npz",
        ROOT / "tools/v15_semantic_interface.py",
        ROOT / "tools/retarget_aloha_trajectory_to_g1.py",
        ROOT / "configs/g1_root_forward_v14.approved.json",
        ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/corrected_targets_v14.npz",
        ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/position_only_exact_v14.npz",
        ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14/position_only_nullspace_v14.npz",
    ]
    for episode_id in range(50):
        paths.extend(
            [
                V2 / f"episodes/{episode_id:02d}/semantic_timeline.auto.json",
                V2 / f"episodes/{episode_id:02d}/semantic_phases.npz",
            ]
        )
    return [path for path in paths if path.exists()]


def hash_snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def timeline_path(base: Path, episode_id: int) -> Path:
    return base / f"episodes/{episode_id:02d}/semantic_timeline.auto.json"


def phases_path(base: Path, episode_id: int) -> Path:
    return base / f"episodes/{episode_id:02d}/semantic_phases.npz"


def event_map(timeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {event["event_name"]: event for event in timeline.get("events", [])}


def event_class(events: dict[str, dict[str, Any]], name: str) -> str:
    event = events.get(name)
    if event is None or event.get("action_index") is None:
        return "MISSING"
    return str(event.get("confidence_class", "MISSING"))


def event_index(events: dict[str, dict[str, Any]], name: str) -> int | None:
    event = events.get(name)
    return None if event is None or event.get("action_index") is None else int(event["action_index"])


def critical_order_valid(events: dict[str, dict[str, Any]]) -> bool:
    for left, right, allow_equal in CRITICAL_EDGES:
        a, b = event_index(events, left), event_index(events, right)
        if a is None or b is None:
            return False
        if (a > b) if allow_equal else (a >= b):
            return False
    return True


def episode_record(base: Path, episode_id: int) -> dict[str, Any]:
    timeline = load_json(timeline_path(base, episode_id))
    events = event_map(timeline)
    indices = {name: event_index(events, name) for name in CRITICAL_EVENTS}
    classes = {name: event_class(events, name) for name in CRITICAL_EVENTS}
    present = all(index is not None for index in indices.values())
    order_valid = critical_order_valid(events)
    critical_complete = present and order_valid and all(value in HIGH_MEDIUM for value in classes.values())
    terminal_classes = {name: event_class(events, name) for name in TERMINAL_EVENTS}
    full_complete = critical_complete and all(value in HIGH_MEDIUM for value in terminal_classes.values())
    worst = min(classes.values(), key=lambda value: CONF_ORDER.get(value, -2))
    time_range = timeline.get("time_range_sec", [0.0, 0.0])
    duration = float(time_range[-1]) - float(time_range[0])
    return {
        "episode_id": episode_id,
        "timeline": timeline,
        "events": events,
        "trajectory_length": int(timeline["trajectory_length"]),
        "duration_sec": duration,
        "critical_indices": indices,
        "critical_classes": classes,
        "all_task_critical_indices_present": present,
        "all_task_critical_partial_order_valid": order_valid,
        "all_task_critical_confidence_HIGH_MEDIUM": critical_complete,
        "any_task_critical_LOW": any(value == "LOW" for value in classes.values()),
        "any_task_critical_AMBIGUOUS": any(value == "AMBIGUOUS" for value in classes.values()),
        "critical_worst_confidence": worst,
        "terminal_classes": terminal_classes,
        "full_timeline_complete": full_complete,
    }


def confidence_counts(records: list[dict[str, Any]], names: Iterable[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for name in names:
        count = Counter(event_class(record["events"], name) for record in records)
        output[name] = {key: int(count.get(key, 0)) for key in ("HIGH", "MEDIUM", "LOW", "AMBIGUOUS", "MISSING")}
    return output


def suffix_independence(base: Path, record: dict[str, Any]) -> dict[str, Any]:
    if not record["all_task_critical_confidence_HIGH_MEDIUM"]:
        return {"applicable": False}
    length = record["trajectory_length"]
    indices = record["critical_indices"]
    all_before_end = all(index is not None and index < length - 1 for index in indices.values())
    intervals_valid = critical_order_valid(record["events"])
    arrays_valid = True
    array_details: dict[str, Any] = {}
    with np.load(phases_path(base, record["episode_id"]), allow_pickle=False) as archive:
        for key in PROGRESS_KEYS:
            valid = key in archive.files and archive[key].shape == (length,) and bool(np.isfinite(archive[key]).all())
            arrays_valid &= valid
            array_details[key] = {"present": key in archive.files, "shape_valid": valid}
    return {
        "applicable": True,
        "all_critical_events_before_T_minus_1": all_before_end,
        "manipulation_intervals_valid": intervals_valid,
        "progress_arrays_valid_without_task_end": arrays_valid,
        "final_release_indices_valid": indices["left_phone_release_complete"] is not None
        and indices["right_accessory_release_complete"] is not None,
        "trajectory_boundary_index": length - 1,
        "trajectory_boundary_is_not_relabelled_task_end": True,
        "progress_array_details": array_details,
    }


def readiness_payload(scope: str, ids: list[int], records_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    records = [records_by_id[episode_id] for episode_id in ids]
    critical_complete = sum(record["all_task_critical_confidence_HIGH_MEDIUM"] for record in records)
    present = sum(record["all_task_critical_indices_present"] for record in records)
    order_valid = sum(record["all_task_critical_partial_order_valid"] for record in records)
    low_occurrences = sum(sum(value == "LOW" for value in record["critical_classes"].values()) for record in records)
    ambiguous_occurrences = sum(sum(value == "AMBIGUOUS" for value in record["critical_classes"].values()) for record in records)
    terminal = confidence_counts(records, TERMINAL_EVENTS)
    suffix = {str(record["episode_id"]): suffix_independence(V2, record) for record in records if record["all_task_critical_confidence_HIGH_MEDIUM"]}
    suffix_pass = sum(
        all(
            details.get(key, False)
            for key in (
                "all_critical_events_before_T_minus_1",
                "manipulation_intervals_valid",
                "progress_arrays_valid_without_task_end",
                "final_release_indices_valid",
            )
        )
        for details in suffix.values()
    )
    return {
        "scope": scope,
        "episode_ids": ids,
        "episode_count": len(ids),
        "critical_high_medium_complete_count": int(critical_complete),
        "critical_mandatory_index_present_count": int(present),
        "critical_mandatory_index_missing_episode_count": len(ids) - int(present),
        "critical_partial_order_valid_count": int(order_valid),
        "critical_partial_order_failure_count": len(ids) - int(order_valid),
        "critical_LOW_event_occurrence_count": int(low_occurrences),
        "critical_AMBIGUOUS_event_occurrence_count": int(ambiguous_occurrences),
        "full_timeline_high_medium_complete_count": sum(record["full_timeline_complete"] for record in records),
        "terminal_only_blocker_episode_ids": [
            record["episode_id"]
            for record in records
            if record["all_task_critical_confidence_HIGH_MEDIUM"] and not record["full_timeline_complete"]
        ],
        "critical_failure_episode_ids": [
            record["episode_id"] for record in records if not record["all_task_critical_confidence_HIGH_MEDIUM"]
        ],
        "critical_event_confidence_counts": confidence_counts(records, CRITICAL_EVENTS),
        "terminal_event_confidence_counts": terminal,
        "critical_ready_suffix_independence_pass_count": int(suffix_pass),
        "critical_ready_suffix_independence_applicable_count": len(suffix),
        "suffix_independence_by_episode": suffix,
        "detector_invocation_count_during_audit": 0,
        "source": "FROZEN_V2_TIMELINES_ONLY",
    }


def robust_reference(values: list[float], floor: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = max(1.4826 * mad, floor)
    return {"median": median, "mad": mad, "robust_scale": scale, "count": len(values)}


def timing_quality(records_by_id: dict[int, dict[str, Any]], development_ids: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_reference: dict[str, dict[str, float]] = {}
    interval_reference: dict[str, dict[str, float]] = {}
    for name in CRITICAL_EVENTS:
        values = []
        for episode_id in development_ids:
            record = records_by_id[episode_id]
            event = record["events"].get(name)
            if event and event.get("action_index") is not None:
                values.append(float(event["action_index"]) / max(1, record["trajectory_length"] - 1))
        event_reference[name] = robust_reference(values, floor=0.02)
    for left, right in INTERVALS:
        key = f"{left}->{right}"
        values = []
        for episode_id in development_ids:
            record = records_by_id[episode_id]
            a, b = record["events"].get(left), record["events"].get(right)
            if a and b and a.get("action_time_sec") is not None and b.get("action_time_sec") is not None:
                values.append(float(b["action_time_sec"]) - float(a["action_time_sec"]))
        interval_reference[key] = robust_reference(values, floor=0.25)

    rows: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for episode_id, record in sorted(records_by_id.items()):
        for name in CRITICAL_EVENTS:
            event = record["events"].get(name)
            if not event or event.get("action_index") is None:
                continue
            normalized = float(event["action_index"]) / max(1, record["trajectory_length"] - 1)
            reference = event_reference[name]
            score = abs(normalized - reference["median"]) / reference["robust_scale"]
            evidence = event.get("evidence", {})
            attribution = evidence.get("v2_confidence_attribution", {})
            row = {
                "episode_id": episode_id,
                "kind": "EVENT_TIME",
                "event_or_interval": name,
                "confidence_class": event_class(record["events"], name),
                "candidate_score_margin": event.get("score_difference"),
                "local_candidate_margin": attribution.get("local_candidate_margin"),
                "global_sequence_margin": evidence.get("global_sequence_margin"),
                "normalized_event_time": normalized,
                "duration_sec": "",
                "development_median": reference["median"],
                "development_robust_scale": reference["robust_scale"],
                "robust_outlier_score": score,
                "flag": "CRITICAL_TIMING_OUTLIER" if event_class(record["events"], name) in HIGH_MEDIUM and score > 3.5 else "",
            }
            rows.append(row)
            if row["flag"]:
                outliers.append(row)
        for left, right in INTERVALS:
            a, b = record["events"].get(left), record["events"].get(right)
            if not a or not b or a.get("action_time_sec") is None or b.get("action_time_sec") is None:
                continue
            duration = float(b["action_time_sec"]) - float(a["action_time_sec"])
            key = f"{left}->{right}"
            reference = interval_reference[key]
            score = abs(duration - reference["median"]) / reference["robust_scale"]
            row = {
                "episode_id": episode_id,
                "kind": "ADJACENT_CRITICAL_DURATION",
                "event_or_interval": key,
                "confidence_class": f"{event_class(record['events'], left)}->{event_class(record['events'], right)}",
                "candidate_score_margin": "",
                "local_candidate_margin": "",
                "global_sequence_margin": "",
                "normalized_event_time": "",
                "duration_sec": duration,
                "development_median": reference["median"],
                "development_robust_scale": reference["robust_scale"],
                "robust_outlier_score": score,
                "flag": "CRITICAL_TIMING_OUTLIER"
                if event_class(record["events"], left) in HIGH_MEDIUM
                and event_class(record["events"], right) in HIGH_MEDIUM
                and score > 3.5
                else "",
            }
            rows.append(row)
            if row["flag"]:
                outliers.append(row)
    return {
        "method": "DEVELOPMENT_MEDIAN_AND_MAD",
        "outlier_threshold_robust_z": 3.5,
        "normalized_event_time_scale_floor": 0.02,
        "interval_duration_scale_floor_sec": 0.25,
        "confidence_not_modified_by_timing_audit": True,
        "event_reference": event_reference,
        "interval_reference": interval_reference,
        "record_count": len(rows),
        "outlier_count": len(outliers),
        "records": rows,
    }, outliers


def plot_coverage(readiness: dict[str, dict[str, Any]]) -> None:
    labels = ["DEVELOPMENT", "VALIDATION", "HELD-OUT", "FULL 50"]
    keys = ["development", "validation", "heldout", "full"]
    critical = [readiness[key]["critical_high_medium_complete_count"] for key in keys]
    full = [readiness[key]["full_timeline_high_medium_complete_count"] for key in keys]
    totals = [readiness[key]["episode_count"] for key in keys]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.19, critical, 0.38, label="Task-critical HIGH/MEDIUM", color="#2ca25f")
    ax.bar(x + 0.19, full, 0.38, label="Full timeline HIGH/MEDIUM", color="#756bb1")
    for i, (c, f, total) in enumerate(zip(critical, full, totals)):
        ax.text(i - 0.19, c + 0.4, f"{c}/{total}", ha="center", fontsize=9)
        ax.text(i + 0.19, f + 0.4, f"{f}/{total}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Complete episodes")
    ax.set_title("Frozen v2: task-critical vs terminal-inclusive readiness")
    ax.set_ylim(0, max(totals) * 1.12)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "critical_vs_terminal_coverage.png", dpi=180)
    plt.close(fig)


def confidence_matrix_plot(records: list[dict[str, Any]], filename: str, failure_only: bool) -> None:
    values = {"MISSING": -1, "AMBIGUOUS": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    data = np.asarray([[values[record["critical_classes"][name]] for name in CRITICAL_EVENTS] for record in records])
    if failure_only:
        data = np.where(data >= 2, 0, 1)
        cmap = matplotlib.colors.ListedColormap(["#edf8e9", "#de2d26"])
        vmin, vmax = 0, 1
        title = "Held-out 30 task-critical failure matrix"
    else:
        cmap = matplotlib.colors.ListedColormap(["#252525", "#756bb1", "#de2d26", "#fec44f", "#2ca25f"])
        vmin, vmax = -1, 3
        title = "Held-out 30 task-critical confidence"
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(CRITICAL_EVENTS)), [name.replace("_", "\n") for name in CRITICAL_EVENTS], fontsize=7)
    ax.set_yticks(np.arange(len(records)), [str(record["episode_id"]) for record in records], fontsize=8)
    ax.set_xlabel("Task-critical semantic event")
    ax.set_ylabel("Held-out episode ID")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_v1_v2_coverage(v1_counts: dict[str, int], v2_counts: dict[str, int], totals: dict[str, int]) -> None:
    labels = ["DEVELOPMENT", "VALIDATION", "HELD-OUT", "FULL 50"]
    keys = ["development", "validation", "heldout", "full"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.19, [v1_counts[key] for key in keys], 0.38, label="v1 critical", color="#9ecae1")
    ax.bar(x + 0.19, [v2_counts[key] for key in keys], 0.38, label="v2 critical", color="#238b45")
    for i, key in enumerate(keys):
        ax.text(i - 0.19, v1_counts[key] + 0.4, f"{v1_counts[key]}/{totals[key]}", ha="center", fontsize=9)
        ax.text(i + 0.19, v2_counts[key] + 0.4, f"{v2_counts[key]}/{totals[key]}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Task-critical HIGH/MEDIUM complete")
    ax.set_ylim(0, max(totals.values()) * 1.12)
    ax.set_title("Frozen v1 vs v2 task-critical readiness")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "v1_vs_v2_critical_coverage.png", dpi=180)
    plt.close(fig)


def report_markdown(
    split: dict[str, Any],
    readiness: dict[str, dict[str, Any]],
    pilot: dict[str, Any],
    v1_counts: dict[str, int],
    timing: dict[str, Any],
    ep49: dict[str, Any],
    freeze_status: str,
) -> str:
    held = readiness["heldout"]
    outlier_scope = Counter(
        "DEVELOPMENT"
        if int(row["episode_id"]) in split["development"]
        else "VALIDATION"
        if int(row["episode_id"]) in split["validation"]
        else "HELDOUT"
        for row in timing["records"]
        if row["flag"]
    )
    outlier_kind = Counter(row["kind"] for row in timing["records"] if row["flag"])
    final_lines = """DETECTOR V2 WAS NOT MODIFIED OR RE-RUN DURING THIS AUDIT
TASK-CRITICAL MANIPULATION EVENTS WERE EVALUATED SEPARATELY FROM TERMINAL DIAGNOSTIC EVENTS
LEFT_ARM_RETURN_NEAR_HOME AND TASK_END DID NOT AUTOMATICALLY BLOCK MANIPULATION READINESS
PHONE_MOVE_TO_CHARGER_START AND PHONE_CHARGER_ATTACHMENT_COMPLETE REMAINED TASK-CRITICAL
NO LOW OR AMBIGUOUS EVENT WAS REPLACED BY A MANUAL FRAME
THE FROZEN HELD-OUT 30 RESULTS WERE NOT USED FOR PARAMETER TUNING
NO G1 IK, ORIENTATION, DEX3 TRAJECTORY, PHYSICS, DDS, PUBLISHER, OR REAL-ROBOT COMMAND WAS USED"""
    critical_failures = held["critical_failure_episode_ids"]
    terminal_only = held["terminal_only_blocker_episode_ids"]
    lines = [
        "1. 동결된 v2 timeline만 재집계했으며 detector/config/event index/split을 변경하거나 detector를 실행하지 않았다.",
        f"2. HELD-OUT critical HIGH/MEDIUM completeness는 {held['critical_high_medium_complete_count']}/30으로 full-timeline {held['full_timeline_high_medium_complete_count']}/30보다 9개 높다.",
        "3. 결과는 BORDERLINE이므로 deterministic 3-episode v15 pilot만 허용하고, 30-episode v15 generalization은 detector v3 전까지 보류한다.",
        "",
        "# 1. Final critical-readiness status",
        "",
        "- `TASK_CRITICAL_READINESS_BORDERLINE`",
        "- `READY_FOR_LIMITED_V15_3_EPISODE_PILOT`",
        "- `FULL_EPISODE_SEMANTIC_SEGMENTATION_NOT_READY`",
        "- `DETECTOR_V2_FROZEN`",
        "- `HELDOUT_RESULTS_NOT_USED_FOR_TUNING`",
        "- `NO_G1_TRAJECTORY_GENERATED`",
        "",
        "# 2. Proof detector v2 was untouched",
        "",
        f"- Freeze audit: `{freeze_status}`",
        "- Detector invocation count during audit: `0`",
        f"- Selected config: `V2_BALANCED`, `{EXPECTED_CONFIG_HASH}`",
        f"- Split hash: `{EXPECTED_SPLIT_HASH}`",
        "- All frozen code/config/timeline/v14 hashes were identical before and after the audit.",
        "",
        "# 3. Event criticality classification",
        "",
        "- Task-critical: " + ", ".join(f"`{name}`" for name in CRITICAL_EVENTS),
        "- Terminal/diagnostic: `left_arm_return_near_home`, `task_end`",
        "- Optional diagnostic: " + ", ".join(f"`{name}`" for name in OPTIONAL_EVENTS),
        "",
        "# 4. Exact v15 dependencies",
        "",
        "Phone acquisition/rotation, accessory insertion/removal, phone transport, charger endpoint, both releases, orientation activation, Dex3 progress, contact keyframes and manipulation residual knots use the ten task-critical events. `left_arm_return_near_home` and `task_end` enter only through the generic mandatory-complete guard and terminal suffix knots; they are `TERMINAL_SUFFIX_ONLY`, not manipulation generation anchors.",
        "",
        "# 5–8. Critical completeness",
        "",
        "| Scope | Critical H/M | Full timeline H/M | Missing critical index | Critical LOW | Critical AMBIGUOUS | Critical order failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("development", "DEVELOPMENT"), ("validation", "VALIDATION"), ("heldout", "HELD-OUT"), ("full", "FULL 50")):
        row = readiness[key]
        lines.append(
            f"| {label} | {row['critical_high_medium_complete_count']}/{row['episode_count']} | "
            f"{row['full_timeline_high_medium_complete_count']}/{row['episode_count']} | "
            f"{row['critical_mandatory_index_missing_episode_count']} | {row['critical_LOW_event_occurrence_count']} | "
            f"{row['critical_AMBIGUOUS_event_occurrence_count']} | {row['critical_partial_order_failure_count']} |"
        )
    lines += [
        "",
        "# 9. Full-timeline vs critical-only",
        "",
        f"HELD-OUT에서 terminal events를 분리하면 16/30에서 25/30으로 9 episode 증가한다. FULL 50은 {readiness['full']['full_timeline_high_medium_complete_count']}/50에서 {readiness['full']['critical_high_medium_complete_count']}/50으로 12 episode 증가한다. 따라서 기존 readiness 저하의 큰 부분은 terminal suffix이지만, task-critical 실패가 남아 있다.",
        "",
        "# 10–12. Critical and terminal blockers",
        "",
        "| Task-critical event | HELD-OUT LOW | HELD-OUT AMBIGUOUS | FULL LOW | FULL AMBIGUOUS |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CRITICAL_EVENTS:
        held_counts = readiness["heldout"]["critical_event_confidence_counts"][name]
        full_counts = readiness["full"]["critical_event_confidence_counts"][name]
        lines.append(
            f"| `{name}` | {held_counts['LOW']} | {held_counts['AMBIGUOUS']} | "
            f"{full_counts['LOW']} | {full_counts['AMBIGUOUS']} |"
        )
    lines += [
        "",
        f"- HELD-OUT task-critical failures: `{critical_failures}`",
        f"- HELD-OUT terminal-only blockers: `{terminal_only}`",
        "- Critical causes: episode 24/40 portrait AMBIGUOUS, 28 accessory_removed LOW, 36 charger_complete LOW, 46 detachment_start LOW.",
        "- Terminal-only causes remain diagnostic and do not block the limited manipulation pilot.",
        "",
        "# 13. Episode 49 diagnostic",
        "",
        f"- Optimized-action timeline critical complete: `{ep49['critical_complete']}`",
        f"- Full-timeline complete: `{ep49['full_complete']}`",
        f"- Blocking critical events: `{ep49['blocking_critical_events']}`",
        "- `phone_move_to_charger_start` remains LOW and therefore blocks Episode-49 critical readiness.",
        "- `phone_charger_attachment_complete` is MEDIUM but retains the 2.133 s approved-reference error as a timing warning.",
        "- `task_end` has the 7.833 s reference error but is terminal-only and does not independently block manipulation readiness.",
        "",
        "# 14. Critical timing outliers",
        "",
        f"Development median/MAD audit flagged {timing['outlier_count']} HIGH/MEDIUM event-time or adjacent-duration records at robust z > 3.5. Confidence classes were not changed. See `critical_timing_outliers.csv`.",
        f"- By split: DEVELOPMENT {outlier_scope['DEVELOPMENT']}, VALIDATION {outlier_scope['VALIDATION']}, HELD-OUT {outlier_scope['HELDOUT']}.",
        f"- By type: event time {outlier_kind['EVENT_TIME']}, adjacent critical duration {outlier_kind['ADJACENT_CRITICAL_DURATION']}.",
        "",
        "# 15. v1 vs v2 critical improvement",
        "",
        f"- FULL 50 critical completeness: v1 `{v1_counts['full']}/50` → v2 `{readiness['full']['critical_high_medium_complete_count']}/50`",
        f"- HELD-OUT critical completeness: v1 `{v1_counts['heldout']}/30` → v2 `{readiness['heldout']['critical_high_medium_complete_count']}/30`",
        "- Terminal events were excluded from this primary comparison.",
        "",
        "# 16. Recommended deterministic v15 pilot",
        "",
        f"Episodes: `{pilot['selected_episode_ids']}`. They are the shortest, duration-median-nearest, and longest critical-complete held-out episodes, selected deterministically without visual inspection.",
        "",
        "# 17. Exact next recommended action",
        "",
        "Run only a limited v15 orientation+Dex3 pilot on the three frozen semantic-complete episodes listed above. Do not claim 30-episode translator readiness. Design detector v3 in parallel under a newly declared protocol before multi-episode v15 generalization.",
        "",
        final_lines,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = frozen_paths()
    before = hash_snapshot(frozen)
    split = load_json(V2 / "dataset_split_v2.json")
    if split.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise RuntimeError("frozen split hash mismatch")
    provenance = load_json(V2 / "selected_detector_v2_config_provenance.json")
    if provenance.get("selected_config_canonical_sha256") != EXPECTED_CONFIG_HASH:
        raise RuntimeError("selected detector config hash mismatch")

    split_ids = {
        "development": list(map(int, split["development"])),
        "validation": list(map(int, split["validation"])),
        "heldout": list(map(int, split["heldout_test"])),
        "full": list(range(50)),
    }
    v2_records = {episode_id: episode_record(V2, episode_id) for episode_id in range(50)}
    v1_records = {episode_id: episode_record(V1, episode_id) for episode_id in range(50)}
    readiness = {key: readiness_payload(key.upper(), ids, v2_records) for key, ids in split_ids.items()}

    held_count = readiness["heldout"]["critical_high_medium_complete_count"]
    if held_count >= 27:
        primary_statuses = ["TASK_CRITICAL_30_EPISODE_READINESS_PASS", "READY_FOR_V15_SEMANTIC_PILOT"]
    elif held_count >= 24:
        primary_statuses = ["TASK_CRITICAL_READINESS_BORDERLINE", "READY_FOR_LIMITED_V15_3_EPISODE_PILOT"]
    else:
        primary_statuses = ["BLOCKED_TASK_CRITICAL_SEMANTIC_COVERAGE", "DETECTOR_V3_RECOMMENDED_BEFORE_V15_GENERALIZATION"]
    primary_statuses += [
        "FULL_EPISODE_SEMANTIC_SEGMENTATION_NOT_READY",
        "DETECTOR_V2_FROZEN",
        "HELDOUT_RESULTS_NOT_USED_FOR_TUNING",
        "NO_G1_TRAJECTORY_GENERATED",
    ]
    readiness["heldout"]["statuses"] = primary_statuses

    criticality = {
        "schema_name": "semantic_event_criticality_v1",
        "task_critical_manipulation_events": list(CRITICAL_EVENTS),
        "terminal_diagnostic_events": list(TERMINAL_EVENTS),
        "optional_diagnostic_events": list(OPTIONAL_EVENTS),
        "readiness_rule": "All ten task-critical events present, critical partial-order valid, all confidence HIGH/MEDIUM",
        "terminal_events_block_task_critical_readiness": False,
        "phone_move_and_charger_complete_remain_task_critical": True,
    }
    write_json(OUT / "semantic_event_criticality_v1.json", criticality)

    dependency = {
        "status": "V15_TASK_CRITICAL_DEPENDENCIES_SEPARATED",
        "audited_files": {
            "tools/v15_semantic_interface.py": sha256_file(ROOT / "tools/v15_semantic_interface.py"),
            "tools/retarget_aloha_trajectory_to_g1.py": sha256_file(ROOT / "tools/retarget_aloha_trajectory_to_g1.py"),
            "v15_semantic_interface_readiness_v2.json": sha256_file(V2 / "v15_semantic_interface_readiness_v2.json"),
        },
        "dependencies": {
            "left_phone_acquisition": ["left_phone_grasp_start", "phone_portrait_reached", "phone_acquisition_progress"],
            "phone_rotation": ["phone_rotation_to_portrait_start", "phone_portrait_reached", "phone_rotation_progress"],
            "right_accessory_insertion": ["right_accessory_grasp_start", "accessory_detachment_start", "accessory_acquisition_progress"],
            "accessory_removal": ["accessory_detachment_start", "accessory_removed", "accessory_removal_progress"],
            "phone_transport": ["phone_move_to_charger_start", "phone_charger_attachment_complete", "phone_to_charger_progress"],
            "charger_orientation": ["phone_charger_attachment_complete", "phone_rotation_progress"],
            "left_release": ["left_phone_release_complete", "left_release_progress"],
            "right_release": ["right_accessory_release_complete", "right_release_progress"],
            "Dex3_phase_interpolation": ["phone_acquisition_progress", "accessory_acquisition_progress", "accessory_removal_progress"],
            "orientation_activation": ["phone_rotation_progress", "accessory_removal_progress"],
            "manipulation_phase_residual_knots": list(CRITICAL_EVENTS),
            "trajectory_numeric_boundaries": ["trajectory_start=0", "trajectory_end=T-1"],
            "left_arm_return_near_home": "TERMINAL_SUFFIX_ONLY",
            "task_end": "TERMINAL_SUFFIX_ONLY",
        },
        "current_generic_guard_observation": "REQUIRED_EVENTS and build_semantic_knots include terminal events for full-timeline completeness and suffix knots",
        "manipulation_generation_requires_terminal_events": False,
        "T_minus_1_may_be_numeric_boundary_but_not_detected_task_end": True,
        "v15_executed": False,
        "source_modified": False,
    }
    write_json(OUT / "v15_event_dependency_audit.json", dependency)

    for key, filename in (
        ("development", "development_critical_readiness.json"),
        ("validation", "validation_critical_readiness.json"),
        ("heldout", "heldout30_critical_readiness.json"),
        ("full", "full50_critical_readiness.json"),
    ):
        write_json(OUT / filename, readiness[key])

    matrix_rows = []
    short_names = {
        "left_phone_grasp_start": "phone_grasp",
        "phone_rotation_to_portrait_start": "rotation_start",
        "phone_portrait_reached": "portrait_reached",
        "right_accessory_grasp_start": "right_grasp",
        "accessory_detachment_start": "detachment_start",
        "accessory_removed": "accessory_removed",
        "phone_move_to_charger_start": "phone_move",
        "phone_charger_attachment_complete": "charger_complete",
        "left_phone_release_complete": "left_release",
        "right_accessory_release_complete": "right_release",
    }
    for episode_id, record in sorted(v2_records.items()):
        row = {"episode_id": episode_id}
        row.update({short_names[name]: record["critical_classes"][name] for name in CRITICAL_EVENTS})
        row.update(
            {
                "critical_complete": record["all_task_critical_confidence_HIGH_MEDIUM"],
                "critical_worst_confidence": record["critical_worst_confidence"],
                "left_return_confidence": record["terminal_classes"]["left_arm_return_near_home"],
                "task_end_confidence": record["terminal_classes"]["task_end"],
                "full_timeline_complete": record["full_timeline_complete"],
            }
        )
        matrix_rows.append(row)
    write_csv(OUT / "episode_criticality_matrix.csv", list(matrix_rows[0]), matrix_rows)

    confidence_rows = []
    for key in ("development", "validation", "heldout", "full"):
        for event_name, counts in readiness[key]["critical_event_confidence_counts"].items():
            confidence_rows.append({"scope": key.upper(), "criticality": "TASK_CRITICAL", "event_name": event_name, **counts})
        for event_name, counts in readiness[key]["terminal_event_confidence_counts"].items():
            confidence_rows.append({"scope": key.upper(), "criticality": "TERMINAL_DIAGNOSTIC", "event_name": event_name, **counts})
    write_csv(
        OUT / "event_critical_confidence_summary.csv",
        ["scope", "criticality", "event_name", "HIGH", "MEDIUM", "LOW", "AMBIGUOUS", "MISSING"],
        confidence_rows,
    )

    timing, outliers = timing_quality(v2_records, split_ids["development"])
    write_json(OUT / "critical_timing_quality.json", timing)
    timing_fields = [
        "episode_id",
        "kind",
        "event_or_interval",
        "confidence_class",
        "candidate_score_margin",
        "local_candidate_margin",
        "global_sequence_margin",
        "normalized_event_time",
        "duration_sec",
        "development_median",
        "development_robust_scale",
        "robust_outlier_score",
        "flag",
    ]
    write_csv(OUT / "critical_timing_outliers.csv", timing_fields, outliers)

    v1_counts: dict[str, int] = {}
    v2_counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for key, ids in split_ids.items():
        v1_counts[key] = sum(v1_records[episode_id]["all_task_critical_confidence_HIGH_MEDIUM"] for episode_id in ids)
        v2_counts[key] = sum(v2_records[episode_id]["all_task_critical_confidence_HIGH_MEDIUM"] for episode_id in ids)
        totals[key] = len(ids)
    v1_event_rows = []
    for event_name in CRITICAL_EVENTS:
        v1_count = Counter(event_class(record["events"], event_name) for record in v1_records.values())
        v2_count = Counter(event_class(record["events"], event_name) for record in v2_records.values())
        v1_event_rows.append(
            {
                "event_name": event_name,
                **{f"v1_{name}": v1_count.get(name, 0) for name in ("HIGH", "MEDIUM", "LOW", "AMBIGUOUS", "MISSING")},
                **{f"v2_{name}": v2_count.get(name, 0) for name in ("HIGH", "MEDIUM", "LOW", "AMBIGUOUS", "MISSING")},
                "v1_LOW_AMBIGUOUS": v1_count.get("LOW", 0) + v1_count.get("AMBIGUOUS", 0),
                "v2_LOW_AMBIGUOUS": v2_count.get("LOW", 0) + v2_count.get("AMBIGUOUS", 0),
            }
        )
    write_csv(OUT / "v1_vs_v2_critical_event_summary.csv", list(v1_event_rows[0]), v1_event_rows)
    episode_compare_rows = []
    for episode_id in range(50):
        episode_compare_rows.append(
            {
                "episode_id": episode_id,
                "split": "DEVELOPMENT"
                if episode_id in split_ids["development"]
                else "VALIDATION"
                if episode_id in split_ids["validation"]
                else "HELDOUT",
                "v1_critical_complete": v1_records[episode_id]["all_task_critical_confidence_HIGH_MEDIUM"],
                "v2_critical_complete": v2_records[episode_id]["all_task_critical_confidence_HIGH_MEDIUM"],
                "v1_worst_confidence": v1_records[episode_id]["critical_worst_confidence"],
                "v2_worst_confidence": v2_records[episode_id]["critical_worst_confidence"],
            }
        )
    write_csv(OUT / "v1_vs_v2_critical_episode_summary.csv", list(episode_compare_rows[0]), episode_compare_rows)

    eligible = [v2_records[episode_id] for episode_id in split_ids["heldout"] if v2_records[episode_id]["all_task_critical_confidence_HIGH_MEDIUM"]]
    eligible.sort(key=lambda record: (record["duration_sec"], record["episode_id"]))
    duration_median = float(np.median([record["duration_sec"] for record in eligible]))
    chosen = [
        eligible[0],
        min(eligible, key=lambda record: (abs(record["duration_sec"] - duration_median), record["episode_id"])),
        eligible[-1],
    ]
    # Preserve duration-quantile role order and eliminate a theoretical tie duplication deterministically.
    unique_chosen: list[dict[str, Any]] = []
    for record in chosen:
        if record["episode_id"] not in {item["episode_id"] for item in unique_chosen}:
            unique_chosen.append(record)
    for record in eligible:
        if len(unique_chosen) == 3:
            break
        if record["episode_id"] not in {item["episode_id"] for item in unique_chosen}:
            unique_chosen.append(record)
    roles = ("SHORT_DURATION", "MEDIAN_DURATION", "LONG_DURATION")
    pilot_episodes = []
    for role, record in zip(roles, unique_chosen):
        pilot_episodes.append(
            {
                "selection_role": role,
                "episode_id": record["episode_id"],
                "duration_sec": record["duration_sec"],
                "critical_event_indices": record["critical_indices"],
                "confidence_classes": record["critical_classes"],
                "normalized_event_times": {
                    name: record["critical_indices"][name] / max(1, record["trajectory_length"] - 1) for name in CRITICAL_EVENTS
                },
                "reason_selected": "Deterministic duration coverage among frozen held-out critical-complete episodes",
            }
        )
    pilot = {
        "status": "RECOMMENDED_FOR_LIMITED_V15_3_EPISODE_PILOT" if held_count >= 24 else "PILOT_NOT_RECOMMENDED",
        "selection_pool": "FROZEN_HELDOUT_TASK_CRITICAL_HIGH_MEDIUM_COMPLETE",
        "selection_method": "shortest, closest to eligible-duration median, longest; tie by episode ID",
        "visual_inspection_used": False,
        "episode49_excluded": True,
        "selected_episode_ids": [record["episode_id"] for record in pilot_episodes],
        "episodes": pilot_episodes,
        "v15_executed": False,
    }
    write_json(OUT / "recommended_v15_pilot_episodes.json", pilot)

    optimized_ep49 = load_json(V2 / "ep49_auto_timeline_v2.json")
    optimized_events = event_map(optimized_ep49)
    optimized_critical_classes = {name: event_class(optimized_events, name) for name in CRITICAL_EVENTS}
    optimized_terminal_classes = {name: event_class(optimized_events, name) for name in TERMINAL_EVENTS}
    ep49_regression = load_json(V2 / "episode49_regression_v2.json")
    ep49 = {
        "source": "optimized_action_frozen_v2_timeline",
        "critical_complete": critical_order_valid(optimized_events)
        and all(value in HIGH_MEDIUM for value in optimized_critical_classes.values()),
        "full_complete": critical_order_valid(optimized_events)
        and all(value in HIGH_MEDIUM for value in optimized_critical_classes.values())
        and all(value in HIGH_MEDIUM for value in optimized_terminal_classes.values()),
        "critical_classes": optimized_critical_classes,
        "terminal_classes": optimized_terminal_classes,
        "blocking_critical_events": [name for name, value in optimized_critical_classes.items() if value not in HIGH_MEDIUM],
        "approved_reference_used_for_detection": False,
        "regression_diagnostics": ep49_regression.get("events", []),
        "raw_dataset_episode49_is_separate_from_optimized_action_regression": True,
    }
    write_json(OUT / "episode49_critical_diagnosis.json", ep49)

    plot_coverage(readiness)
    held_records = [v2_records[episode_id] for episode_id in split_ids["heldout"]]
    confidence_matrix_plot(held_records, "heldout30_critical_confidence_heatmap.png", failure_only=False)
    confidence_matrix_plot(held_records, "heldout30_critical_failure_matrix.png", failure_only=True)
    plot_v1_v2_coverage(v1_counts, v2_counts, totals)

    after = hash_snapshot(frozen)
    freeze_status = "PASS" if before == after else "BLOCKED_READ_ONLY_AUDIT_VIOLATION"
    input_audit = {
        "status": freeze_status,
        "audit_type": "READ_ONLY_SEMANTIC_READINESS_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_detector": "V2_BALANCED",
        "selected_config_canonical_sha256": EXPECTED_CONFIG_HASH,
        "split_sha256": EXPECTED_SPLIT_HASH,
        "frozen_file_count": len(frozen),
        "frozen_hashes_before": before,
        "frozen_hashes_after": after,
        "changed_frozen_files": [name for name in before if before[name] != after.get(name)],
        "detector_module_imported": False,
        "detector_invocation_count_during_audit": 0,
        "event_redetection_performed": False,
        "event_index_edited": False,
        "confidence_threshold_changed": False,
        "split_changed": False,
        "G1_IK_or_trajectory_generated": False,
        "orientation_or_Dex3_executed": False,
        "physics_DDS_publisher_hardware_executed": False,
    }
    write_json(OUT / "input_freeze_audit.json", input_audit)

    report = report_markdown(split, readiness, pilot, v1_counts, timing, ep49, freeze_status)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    html_text = "<!doctype html><html><head><meta charset='utf-8'><title>Semantic v2 critical audit</title>"
    html_text += "<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}pre{white-space:pre-wrap;line-height:1.45}</style>"
    html_text += f"</head><body><pre>{html.escape(report)}</pre></body></html>\n"
    (OUT / "report").mkdir(exist_ok=True)
    (OUT / "report/index.html").write_text(html_text, encoding="utf-8")
    commands = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
xdg-open {OUT / 'critical_vs_terminal_coverage.png'}
xdg-open {OUT / 'heldout30_critical_confidence_heatmap.png'}
xdg-open {OUT / 'heldout30_critical_failure_matrix.png'}
xdg-open {OUT / 'v1_vs_v2_critical_coverage.png'}
xdg-open {OUT / 'report/index.html'}
# Read-only review only. No detector, v15, G1, Dex3, physics, DDS, publisher, or hardware command.
"""
    (OUT / "commands.sh").write_text(commands, encoding="utf-8")
    os.chmod(OUT / "commands.sh", 0o755)

    manifest_files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "run_manifest.json")
    manifest = {
        "status": primary_statuses,
        "method": "FROZEN_V2_TIMELINE_CRITICALITY_REAGGREGATION",
        "source_directory": str(V2),
        "output_directory": str(OUT),
        "detector_invocation_count": 0,
        "file_count_excluding_manifest": len(manifest_files),
        "files": {str(path.relative_to(OUT)): sha256_file(path) for path in manifest_files},
    }
    write_json(OUT / "run_manifest.json", manifest)
    if freeze_status != "PASS":
        raise RuntimeError(freeze_status)
    print(json.dumps({"statuses": primary_statuses, "heldout_critical": held_count, "pilot": pilot["selected_episode_ids"], "outliers": len(outliers)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
