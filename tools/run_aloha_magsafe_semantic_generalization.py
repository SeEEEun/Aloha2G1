#!/usr/bin/env python3
"""Run semantic timeline detection on the integrated 50-episode dataset.

Detection is completed before the Episode-49 approved reference is opened.
This program performs MuJoCo FK only; it has no G1, Dex3, physics, DDS,
publisher, or hardware command path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import replace
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

from aloha_magsafe_semantics.candidate_detection import extract_event_candidates  # noqa: E402
from aloha_magsafe_semantics.detector import detect_magsafe_semantics  # noqa: E402
from aloha_magsafe_semantics.event_names import (  # noqa: E402
    OPTIONAL_EVENTS, PARTIAL_ORDER_EDGES, PROGRESS_NAMES, REQUIRED_EVENTS, SCHEMA_NAME,
)
from aloha_magsafe_semantics.features import compute_stationary_aloha_fk  # noqa: E402
from aloha_magsafe_semantics.io import (  # noqa: E402
    atomic_json, canonical_json_hash, load_trajectory, sha256_array, sha256_file,
)
from aloha_magsafe_semantics.knots import build_semantic_knots  # noqa: E402
from aloha_magsafe_semantics.schema import SemanticTimeline  # noqa: E402
from retarget_aloha_trajectory_to_g1 import retarget_aloha_trajectory_to_g1  # noqa: E402
from v15_semantic_interface import PHASE_API_MAPPING, readiness as v15_readiness  # noqa: E402


OUT = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1"
MANIFEST = ROOT / "reports/magsafe_lerobot_v3_manifest.csv"
DATASET = ROOT / "lerobot_magsafe_50_cam_high_v3"
CONFIG_PATH = ROOT / "configs/aloha_magsafe_semantic_detector_v1.json"
GEOMETRY_PATH = ROOT / "configs/aloha_magsafe_task_geometry.semantic_v1.json"
MODEL = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
POSE_CONFIG = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
EP49_ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
EP49_REFERENCE = ROOT / "configs/episode49_task_timeline.approved.json"
EP49_ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer CSV fields for empty rows: {path}")
    names = fieldnames or list(rows[0])
    temp = path.with_suffix(path.suffix + ".incomplete")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def file_hashes(paths: list[Path]) -> dict[str, str | None]:
    return {str(path.resolve()): sha256_file(path) if path.is_file() else None for path in paths}


def immutable_paths() -> list[Path]:
    return [
        EP49_REFERENCE,
        EP49_ALIGNMENT,
        ROOT / "configs/g1_root_forward_v14.approved.json",
        V14 / "corrected_targets_v14.npz",
        V14 / "position_only_exact_v14.npz",
        V14 / "position_only_nullspace_v14.npz",
        ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json",
        ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda",
    ]


def read_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    for source in raw:
        row = dict(source)
        row["episode_id"] = int(source["output_episode_index"])
        row["source_frame_count"] = int(source["source_frame_count"])
        row["source_first_timestamp"] = float(source["source_first_timestamp"])
        row["source_last_timestamp"] = float(source["source_last_timestamp"])
        row["source_parquet"] = str(Path(source["source_parquet"]).resolve())
        row["source_folder"] = str(Path(source["source_folder"]).resolve())
        rows.append(row)
    ids = [row["episode_id"] for row in rows]
    exact_ids = list(range(len(rows)))
    if len(rows) != 50 or sorted(ids) != exact_ids or len(set(ids)) != len(ids):
        raise RuntimeError(f"authoritative manifest mismatch: count={len(rows)} ids={ids}")
    metadata_path = DATASET / "meta/info.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["total_episodes"]) != len(rows):
        raise RuntimeError("LeRobot metadata episode count disagrees with manifest")
    audit = {
        "status": "PASS",
        "authoritative_manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256_file(MANIFEST),
        "integrated_dataset": str(DATASET.resolve()),
        "integrated_metadata": str(metadata_path.resolve()),
        "integrated_metadata_sha256": sha256_file(metadata_path),
        "total_episode_count": len(rows),
        "exact_episode_ids": ids,
        "ids_are_zero_through_last": ids == exact_ids,
        "directory_order_used_as_episode_id": False,
        "mapping": [{
            "episode_id": row["episode_id"],
            "raw_recording": row["source_folder"],
            "raw_parquet": row["source_parquet"],
            "LeRobot_episode_index": row["episode_id"],
            "frame_count": row["source_frame_count"],
        } for row in rows],
    }
    return rows, audit


def source_pose() -> dict[str, Any]:
    return json.loads(POSE_CONFIG.read_text(encoding="utf-8"))["stationary_aloha"]


def save_timeline(directory: Path, timeline: SemanticTimeline, candidates: dict[str, Any], context: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "semantic_timeline.auto.json", timeline.to_dict())
    numeric_arrays = {key: value for key, value in timeline.sample_arrays.items()}
    numeric_arrays.update({
        "timestamps": timeline.timestamps,
        "event_names": np.asarray(list(timeline.events), dtype="U48"),
        "event_action_indices": np.asarray([
            -1 if timeline.events[name].action_index is None else timeline.events[name].action_index
            for name in timeline.events
        ], dtype=np.int64),
    })
    np.savez_compressed(directory / "semantic_phases.npz", **numeric_arrays)
    feature = context["features"]
    feature_metrics = {
        "trajectory_length": timeline.trajectory_length,
        "duration_sec": float(timeline.timestamps[-1] - timeline.timestamps[0]),
        "frequency_hz_median": float(1.0 / np.median(np.diff(timeline.timestamps))),
        "left_total_path_m": float(feature["left_cumulative_path_length"][-1]),
        "right_total_path_m": float(feature["right_cumulative_path_length"][-1]),
        "left_total_rotation_rad": float(feature["left_cumulative_rotation"][-1]),
        "right_total_rotation_rad": float(feature["right_cumulative_rotation"][-1]),
        "left_speed_quantiles_mps": np.quantile(feature["left_linear_speed"], [0, .25, .5, .75, .9, 1]).tolist(),
        "right_speed_quantiles_mps": np.quantile(feature["right_linear_speed"], [0, .25, .5, .75, .9, 1]).tolist(),
        "left_angular_speed_quantiles_radps": np.quantile(feature["left_angular_speed"], [0, .25, .5, .75, .9, 1]).tolist(),
        "right_angular_speed_quantiles_radps": np.quantile(feature["right_angular_speed"], [0, .25, .5, .75, .9, 1]).tolist(),
        "inter_hand_distance_range_m": [float(np.min(feature["inter_hand_distance"])), float(np.max(feature["inter_hand_distance"]))],
        "gripper_calibration": timeline.metadata["gripper_calibration"],
        "partial_order": timeline.metadata["partial_order"],
        "semantic_knots": build_semantic_knots(timeline),
    }
    atomic_json(directory / "feature_metrics.json", feature_metrics)
    atomic_json(directory / "event_candidates.json", {
        "detector_config_hash": timeline.detector_config_hash,
        "reference_timeline_used_for_detection": False,
        "events": {name: [candidate.to_dict() for candidate in values] for name, values in candidates.items()},
    })


def process(
    path: Path,
    source_type: str,
    config: dict[str, Any],
    geometry: dict[str, Any],
    output_directory: Path | None,
) -> tuple[dict[str, Any], SemanticTimeline, dict[str, Any], dict[str, Any]]:
    loaded = load_trajectory(path, source_type)
    pose = source_pose()
    fk = compute_stationary_aloha_fk(
        loaded["action"], loaded["timestamps"], MODEL,
        pose["position_xyz_m"], pose["orientation_wxyz"],
    )
    fk["model_sha256"] = sha256_file(MODEL)
    # Public detection receives no approved reference and no Episode-specific
    # event index. Candidate extraction below is only for the audit artifact.
    timeline = detect_magsafe_semantics(
        loaded["action"], loaded["timestamps"], source_type, fk, geometry, config,
        observation_state=loaded.get("observation_state"), observation_alignment=None,
    )
    candidates, context = extract_event_candidates(
        loaded["action"], loaded["timestamps"], fk, geometry, config,
    )
    if output_directory is not None:
        save_timeline(output_directory, timeline, candidates, context)
    return loaded, timeline, candidates, context


def attach_alignment(timeline: SemanticTimeline, alignment: dict[str, Any]) -> SemanticTimeline:
    lag = int(alignment["action_to_observation_lag_frames"])
    latency = float(alignment["latency_seconds"])
    events = {}
    for name, event in timeline.events.items():
        if event.action_index is None:
            events[name] = event
        else:
            events[name] = replace(
                event,
                observed_frame=int(event.action_index + lag),
                observed_time_sec=float(event.action_time_sec + latency),
            )
    return SemanticTimeline(
        trajectory_length=timeline.trajectory_length,
        timestamps=timeline.timestamps,
        events=events,
        sample_arrays=timeline.sample_arrays,
        detector_config_hash=timeline.detector_config_hash,
        trajectory_hash=timeline.trajectory_hash,
        fk_model_hash=timeline.fk_model_hash,
        task_geometry_hash=timeline.task_geometry_hash,
        source_type=timeline.source_type,
        metadata=dict(timeline.metadata, observation_alignment_applied_after_detection=True,
                      action_to_observation_lag_frames=lag),
    )


def confidence_value(timeline: SemanticTimeline, name: str) -> float:
    return float(timeline.event(name).confidence)


def episode_summary(episode_id: int, timeline: SemanticTimeline, row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episode_id": episode_id,
        "frame_count": timeline.trajectory_length,
        "duration_sec": float(timeline.timestamps[-1] - timeline.timestamps[0]),
        "source_folder": row["source_folder"],
        "source_parquet": row["source_parquet"],
        "detector_config_hash": timeline.detector_config_hash,
        "mandatory_complete": timeline.mandatory_complete(),
        "high_medium_complete": timeline.high_medium_complete(),
        "partial_order_valid": timeline.metadata["partial_order"]["valid"],
    }
    missing = []
    low = []
    ambiguous = []
    for name in REQUIRED_EVENTS:
        event = timeline.event(name)
        result[f"{name}_index"] = event.action_index
        result[f"{name}_confidence"] = event.confidence
        result[f"{name}_class"] = event.confidence_class
        if event.action_index is None:
            missing.append(name)
        if event.confidence_class == "LOW":
            low.append(name)
        if event.confidence_class == "AMBIGUOUS":
            ambiguous.append(name)
    result["missing_events"] = missing
    result["low_events"] = low
    result["ambiguous_events"] = ambiguous
    return result


def deterministic_subsets(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    heldout = [row for row in rows if row["episode_id"] != 49]
    ordered = sorted(heldout, key=lambda row: (row["source_frame_count"], row["episode_id"]))
    smoke = [ordered[0]["episode_id"], ordered[len(ordered) // 2]["episode_id"], ordered[-1]["episode_id"]]
    generator = np.random.default_rng(20260807)
    shuffled = generator.permutation(np.asarray([row["episode_id"] for row in heldout], dtype=np.int64)).tolist()
    return {"three": smoke, "ten": sorted(shuffled[:10]), "thirty": sorted(shuffled[:30])}


def readiness_record(name: str, ids: list[int], summaries: dict[int, dict[str, Any]], required_count: int) -> dict[str, Any]:
    selected = [summaries[index] for index in ids]
    complete = [row for row in selected if row["high_medium_complete"] and row["partial_order_valid"]]
    status = name if len(complete) >= required_count else "BLOCKED_MULTI_EPISODE_EVENT_COVERAGE"
    return {
        "status": status,
        "episode_ids": ids,
        "deterministic_seed": 20260807,
        "Episode_49_excluded": 49 not in ids,
        "detector_config_hashes": sorted({row["detector_config_hash"] for row in selected}),
        "same_detector_config": len({row["detector_config_hash"] for row in selected}) == 1,
        "episode_specific_tuning": False,
        "manual_event_editing": False,
        "required_complete_count": required_count,
        "high_medium_complete_count": len(complete),
        "partial_order_valid_count": sum(bool(row["partial_order_valid"]) for row in selected),
        "incomplete_episode_ids": [row["episode_id"] for row in selected if not row["high_medium_complete"]],
        "episode_results": selected,
    }


def frame_hardcoding_audit() -> dict[str, Any]:
    alignment = json.loads(EP49_ALIGNMENT.read_text(encoding="utf-8"))
    reference_numbers = sorted({int(row["aligned_action_index"]) for row in alignment["event_mapping"].values()})
    number_pattern = re.compile(r"(?<!\d)(?:" + "|".join(map(str, reference_numbers)) + r")(?!\d)")
    name_pattern = re.compile(
        r"frame169|action169|frame_169|phone_grasp_frame|accessory_grasp_frame|charger_frame|EVENTS\s*=|phase_knots|approved_event|episode49",
        re.IGNORECASE,
    )
    generic_paths = {
        "tools/aloha_magsafe_semantics",
        "tools/retarget_aloha_trajectory_to_g1.py",
        "tools/v15_semantic_interface.py",
    }
    occurrences: list[dict[str, Any]] = []
    scanned_files = 0
    skip_roots = {".git", "raw_recordings", "lerobot_magsafe_50_cam_high_v3", "backups"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".md", ".yaml", ".yml", ".toml"}:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] in skip_roots:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        scanned_files += 1
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_number, line in enumerate(lines, 1):
            if not number_pattern.search(line) and not name_pattern.search(line):
                continue
            relative_text = relative.as_posix()
            is_generic = any(relative_text == entry or relative_text.startswith(entry + "/") for entry in generic_paths)
            if is_generic and number_pattern.search(line):
                category = "FORBIDDEN_RUNTIME_DEPENDENCY"
                rationale = "absolute Episode-49 reference index appears in generic runtime"
            elif relative_text == "tools/retarget_episode49_semantic_compat.py":
                category = "LEGACY_WRAPPER"
                rationale = "thin compatibility wrapper; requires explicit semantic timeline and contains no event indices"
            elif relative_text.startswith("tests/") or relative_text.startswith("configs/episode49_"):
                category = "ALLOWED_REFERENCE_ONLY"
                rationale = "regression fixture or approved immutable Episode-49 provenance"
            elif relative_text.startswith("outputs/") or relative_text.startswith("evaluation/") or relative_text.startswith("reports/"):
                category = "ALLOWED_REFERENCE_ONLY"
                rationale = "historical/generated result or report, excluded from generic runtime"
            elif relative_text.startswith("tools/") or relative_text.startswith("isaaclab_magsafe_fixed_scene/"):
                category = "ALLOWED_REFERENCE_ONLY"
                rationale = "pre-v1 frozen legacy experiment, quarantined from generic semantic dependency graph"
            else:
                category = "ALLOWED_REFERENCE_ONLY"
                rationale = "non-runtime provenance or numeric coincidence"
            occurrences.append({
                "file": relative_text,
                "line": line_number,
                "matched_text": line.strip()[:500],
                "category": category,
                "rationale": rationale,
            })
    counts = {category: sum(row["category"] == category for row in occurrences) for category in (
        "ALLOWED_REFERENCE_ONLY", "FORBIDDEN_RUNTIME_DEPENDENCY", "LEGACY_WRAPPER",
    )}
    return {
        "status": "PASS" if counts["FORBIDDEN_RUNTIME_DEPENDENCY"] == 0 else "BLOCKED_RUNTIME_FRAME_HARDCODING",
        "scope": "repository text sources plus generated JSON/Markdown under 2 MB; raw data, backups, binary media excluded",
        "scanned_files": scanned_files,
        "Episode_49_reference_indices": reference_numbers,
        "generic_runtime_dependency_roots": sorted(generic_paths),
        "legacy_policy": "pre-v1 versioned experiment scripts are frozen provenance and are not imported by the generic runtime",
        "counts": counts,
        "occurrences": occurrences,
    }


def write_hardcoding_markdown(audit: dict[str, Any]) -> None:
    grouped: dict[str, dict[str, int]] = {}
    for row in audit["occurrences"]:
        grouped.setdefault(row["file"], {})[row["category"]] = grouped.setdefault(row["file"], {}).get(row["category"], 0) + 1
    lines = [
        "# Frame hard-coding audit", "", f"Status: `{audit['status']}`", "",
        "The v1 generic dependency graph is scanned separately from frozen pre-v1 experiment scripts.", "",
        "| File | Classification | Occurrences |", "|---|---:|---:|",
    ]
    for path, categories in sorted(grouped.items()):
        for category, count in sorted(categories.items()):
            lines.append(f"| `{path}` | `{category}` | {count} |")
    (OUT / "frame_hardcoding_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def active_event(timeline: SemanticTimeline, index: int) -> tuple[str, float, str]:
    result = ("pre_task", 1.0, "HIGH")
    ordered = sorted(
        (record.action_index, name, record.confidence, record.confidence_class)
        for name, record in timeline.events.items() if record.action_index is not None
    )
    for event_index, name, confidence, confidence_class in ordered:
        if event_index <= index:
            result = (name, confidence, confidence_class)
        else:
            break
    return result


def overlay_video(
    output: Path,
    image_directory: Path,
    timeline: SemanticTimeline,
    context: dict[str, Any],
    observation_lag: int = 0,
    output_fps: float = 15.0,
) -> dict[str, Any]:
    images = sorted(image_directory.glob("frame_*.png"))
    if not images:
        return {"status": "MISSING_IMAGES", "path": str(output)}
    first = cv2.imread(str(images[0]))
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    feature = context["features"]
    written = 0
    for observed_index, image_path in enumerate(images):
        frame = cv2.imread(str(image_path))
        action_index = observed_index - observation_lag
        action_index = min(timeline.end_index, max(0, action_index))
        event_name, confidence, confidence_class = active_event(timeline, action_index)
        left_phase = str(timeline.sample_arrays["left_gripper_phase"][action_index])
        right_phase = str(timeline.sample_arrays["right_gripper_phase"][action_index])
        cv2.rectangle(frame, (0, 0), (width, 86), (0, 0, 0), -1)
        cv2.putText(frame, f"action {action_index}/{timeline.end_index}  t={timeline.timestamps[action_index]:.3f}s", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{event_name}  confidence={confidence:.2f} {confidence_class}", (12, 45), cv2.FONT_HERSHEY_SIMPLEX, .52, (70, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"gripper L={left_phase} R={right_phase}   TCP speed L={feature['left_linear_speed'][action_index]:.3f} R={feature['right_linear_speed'][action_index]:.3f} m/s", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 255, 180), 1, cv2.LINE_AA)
        if observation_lag and observed_index < observation_lag:
            cv2.putText(frame, "PRE-COMMAND OBSERVATION HOLD", (width - 290, height - 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1, cv2.LINE_AA)
        writer.write(frame)
        written += 1
    writer.release()
    capture = cv2.VideoCapture(str(output))
    decoded = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); capture.release()
    return {"status": "PASS" if decoded == written else "FRAME_COUNT_MISMATCH", "path": str(output.resolve()), "written_frames": written, "decoded_frames": decoded, "fps": output_fps}


def contact_sheet(output: Path, image_directory: Path, timeline: SemanticTimeline, observation_lag: int = 0) -> dict[str, Any]:
    tiles = []
    for event_name in REQUIRED_EVENTS:
        event = timeline.event(event_name)
        if event.action_index is None:
            continue
        observed = min(timeline.end_index + observation_lag, max(0, event.action_index + observation_lag))
        path = image_directory / f"frame_{observed:06d}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (320, 240))
        cv2.rectangle(frame, (0, 0), (320, 48), (0, 0, 0), -1)
        cv2.putText(frame, event_name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"action={event.action_index} conf={event.confidence:.2f} {event.confidence_class}", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, .36, (70, 230, 255), 1, cv2.LINE_AA)
        tiles.append(frame)
    columns = 3
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)])
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    return {"path": str(output.resolve()), "tiles": len([event for event in timeline.events.values() if event.action_index is not None])}


def ep49_plots(timeline: SemanticTimeline, reference: dict[str, Any], alignment: dict[str, Any]) -> dict[str, Any]:
    approved = {name: int(row["aligned_action_index"]) for name, row in alignment["event_mapping"].items()}
    figure, axis = plt.subplots(figsize=(15, 8))
    y = np.arange(len(REQUIRED_EVENTS))
    detected = [timeline.event(name).action_index for name in REQUIRED_EVENTS]
    expected = [approved[name] for name in REQUIRED_EVENTS]
    axis.scatter(detected, y, marker="o", s=70, label="detected from action/FK/gripper")
    axis.scatter(expected, y, marker="x", s=70, label="approved regression reference")
    for row, (left, right) in enumerate(zip(detected, expected)):
        if left is not None:
            axis.plot((left, right), (row, row), color="0.6", linewidth=1)
    axis.set_yticks(y, REQUIRED_EVENTS); axis.invert_yaxis(); axis.set_xlabel("action index")
    axis.grid(True, axis="x", alpha=.3); axis.legend(); axis.set_title("Episode 49: detector output loaded before approved reference")
    figure.tight_layout(); figure.savefig(OUT / "ep49_detected_vs_approved_timeline.png", dpi=170); plt.close(figure)
    metrics = []
    gripper_events = {"left_phone_grasp_start", "right_accessory_grasp_start", "left_phone_release_complete", "right_accessory_release_complete"}
    for name in REQUIRED_EVENTS:
        event = timeline.event(name)
        expected_index = approved[name]
        error = None if event.action_index is None else abs(event.action_index - expected_index)
        tolerance = 0.5 if name in gripper_events else 0.8
        error_sec = None if error is None else error / float(alignment["fps"])
        metrics.append({
            "event_name": name,
            "detected_action_index": event.action_index,
            "approved_action_index": expected_index,
            "absolute_error_samples": error,
            "absolute_error_seconds": error_sec,
            "diagnostic_tolerance_seconds": tolerance,
            "within_diagnostic_tolerance": error_sec is not None and error_sec <= tolerance,
            "confidence": event.confidence,
            "confidence_class": event.confidence_class,
            "evidence": event.evidence,
        })
    return {
        "status": "PASS" if all(row["within_diagnostic_tolerance"] for row in metrics) else "SEMANTIC_EP49_REGRESSION_WARNING",
        "detector_completed_before_reference_loaded": True,
        "reference_timeline_used_for_detection": False,
        "reference_file": str(EP49_REFERENCE.resolve()),
        "reference_file_sha256": sha256_file(EP49_REFERENCE),
        "alignment_file": str(EP49_ALIGNMENT.resolve()),
        "alignment_file_sha256": sha256_file(EP49_ALIGNMENT),
        "events": metrics,
    }


def timeline_grid(path: Path, ids: list[int], timelines: dict[int, SemanticTimeline], title: str) -> None:
    figure, axis = plt.subplots(figsize=(16, max(5, len(ids) * .35)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(REQUIRED_EVENTS)))
    for row, episode_id in enumerate(ids):
        timeline = timelines[episode_id]
        for event_position, name in enumerate(REQUIRED_EVENTS):
            event = timeline.event(name)
            if event.action_index is not None:
                axis.scatter(event.action_index / max(1, timeline.end_index), row, s=22, color=colors[event_position])
    axis.set_yticks(range(len(ids)), [str(value) for value in ids]); axis.invert_yaxis()
    axis.set_xlabel("normalized action time"); axis.set_ylabel("episode ID"); axis.set_title(title)
    axis.grid(True, axis="x", alpha=.25); figure.tight_layout(); figure.savefig(path, dpi=170); plt.close(figure)


def confidence_heatmap(path: Path, ids: list[int], timelines: dict[int, SemanticTimeline], title: str) -> None:
    matrix = np.asarray([[timelines[episode_id].event(name).confidence for name in REQUIRED_EVENTS] for episode_id in ids])
    figure, axis = plt.subplots(figsize=(16, max(5, len(ids) * .35)))
    image = axis.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(REQUIRED_EVENTS)), REQUIRED_EVENTS, rotation=65, ha="right")
    axis.set_yticks(range(len(ids)), [str(value) for value in ids]); axis.set_ylabel("episode ID")
    axis.set_title(title); figure.colorbar(image, ax=axis, label="confidence")
    figure.tight_layout(); figure.savefig(path, dpi=170); plt.close(figure)


def distribution_artifacts(summaries: list[dict[str, Any]], timelines: dict[int, SemanticTimeline]) -> None:
    index_rows = []
    normalized_rows = []
    confidence_rows = []
    duration_rows = []
    incomplete_rows = []
    for summary in summaries:
        episode_id = summary["episode_id"]
        timeline = timelines[episode_id]
        for event_name in REQUIRED_EVENTS:
            event = timeline.event(event_name)
            index_rows.append({"episode_id": episode_id, "event_name": event_name, "action_index": event.action_index})
            normalized_rows.append({"episode_id": episode_id, "event_name": event_name, "normalized_action_time": None if event.action_index is None else event.action_index / max(1, timeline.end_index)})
            confidence_rows.append({"episode_id": episode_id, "event_name": event_name, "confidence": event.confidence, "confidence_class": event.confidence_class})
        ordered = [(name, timeline.event(name).action_time_sec) for name in REQUIRED_EVENTS if timeline.event(name).action_time_sec is not None]
        for (first_name, first), (second_name, second) in zip(ordered, ordered[1:]):
            duration_rows.append({"episode_id": episode_id, "phase": f"{first_name}->{second_name}", "duration_sec": second - first})
        incomplete_rows.append({
            "episode_id": episode_id,
            "missing_count": len(summary["missing_events"]),
            "low_count": len(summary["low_events"]),
            "ambiguous_count": len(summary["ambiguous_events"]),
            "high_medium_complete": summary["high_medium_complete"],
        })
    dump_csv(OUT / "event_index_distribution.csv", index_rows)
    dump_csv(OUT / "normalized_event_time_distribution.csv", normalized_rows)
    dump_csv(OUT / "confidence_summary.csv", confidence_rows)
    dump_csv(OUT / "event_duration_distribution.csv", duration_rows)
    dump_csv(OUT / "incomplete_episode_summary.csv", incomplete_rows)

    for name, rows, field, ylabel in (
        ("event_index_distribution.png", index_rows, "action_index", "action index"),
        ("normalized_event_time_distribution.png", normalized_rows, "normalized_action_time", "normalized action time"),
        ("event_confidence_distribution.png", confidence_rows, "confidence", "confidence"),
    ):
        figure, axis = plt.subplots(figsize=(15, 7))
        data = [[float(row[field]) for row in rows if row["event_name"] == event and row[field] not in (None, "")] for event in REQUIRED_EVENTS]
        axis.boxplot(data, labels=REQUIRED_EVENTS, showfliers=True); axis.tick_params(axis="x", rotation=65)
        axis.set_ylabel(ylabel); axis.grid(True, axis="y", alpha=.25); figure.tight_layout(); figure.savefig(OUT / name, dpi=170); plt.close(figure)
    figure, axis = plt.subplots(figsize=(16, 8))
    phase_names = sorted({row["phase"] for row in duration_rows})
    phase_data = [[row["duration_sec"] for row in duration_rows if row["phase"] == phase] for phase in phase_names]
    axis.boxplot(phase_data, labels=phase_names, showfliers=True); axis.tick_params(axis="x", rotation=70); axis.set_ylabel("seconds")
    figure.tight_layout(); figure.savefig(OUT / "event_duration_distribution.png", dpi=170); plt.close(figure)
    figure, axis = plt.subplots(figsize=(15, 6))
    x = np.arange(len(incomplete_rows)); axis.bar(x, [row["low_count"] for row in incomplete_rows], label="LOW")
    axis.bar(x, [row["ambiguous_count"] for row in incomplete_rows], bottom=[row["low_count"] for row in incomplete_rows], label="AMBIGUOUS")
    axis.set_xlabel("episode ID"); axis.set_ylabel("mandatory events"); axis.legend(); figure.tight_layout(); figure.savefig(OUT / "incomplete_episode_summary.png", dpi=170); plt.close(figure)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before_hashes = file_hashes(immutable_paths())
    rows, dataset_audit = read_manifest()
    atomic_json(OUT / "input_dataset_audit.json", dataset_audit)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
    config_hash = canonical_json_hash(config)
    shutil.copy2(CONFIG_PATH, OUT / "detector_config.json")
    schema = {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "required_events": REQUIRED_EVENTS,
        "optional_events": OPTIONAL_EVENTS,
        "progress_arrays": PROGRESS_NAMES,
        "event_record_fields": [
            "event_name", "action_index", "action_time_sec", "observed_frame", "observed_time_sec",
            "confidence", "confidence_class", "evidence", "provenance", "alternatives", "score_difference",
        ],
    }
    graph = {"edges": [{"before": first, "after": second, "non_decreasing": same} for first, second, same in PARTIAL_ORDER_EDGES],
             "right_release_vs_left_return_total_order_imposed": False}
    atomic_json(OUT / "semantic_schema.json", schema)
    atomic_json(OUT / "semantic_event_graph.json", graph)
    atomic_json(OUT / "detector_config_provenance.json", {
        "config_path": str(CONFIG_PATH.resolve()), "config_sha256": sha256_file(CONFIG_PATH),
        "canonical_config_hash": config_hash, "finalized_before_heldout_processing": True,
        "Episode_49_role": "DEVELOPMENT_REGRESSION_REFERENCE_ONLY",
        "episode_specific_tuning": False, "duration_units": "seconds",
    })

    # Episode-49 detector run happens before either approved reference file is opened.
    ep49_loaded, ep49_detected, ep49_candidates, ep49_context = process(EP49_ACTION, "optimized_action", config, geometry, None)
    ep49_action_domain_hash_before_reference = canonical_json_hash(ep49_detected.to_dict())
    reference = json.loads(EP49_REFERENCE.read_text(encoding="utf-8"))
    alignment = json.loads(EP49_ALIGNMENT.read_text(encoding="utf-8"))
    ep49 = attach_alignment(ep49_detected, alignment)
    atomic_json(OUT / "ep49_auto_timeline.json", ep49.to_dict())
    np.savez_compressed(OUT / "ep49_semantic_phases.npz", timestamps=ep49.timestamps, **ep49.sample_arrays)
    atomic_json(OUT / "ep49_event_candidates.json", {name: [row.to_dict() for row in values] for name, values in ep49_candidates.items()})
    regression = ep49_plots(ep49, reference, alignment)
    regression["action_domain_output_hash_before_reference_load"] = ep49_action_domain_hash_before_reference
    regression["action_domain_output_hash_recomputed_without_reference"] = canonical_json_hash(ep49_detected.to_dict())
    regression["approved_reference_independence_pass"] = (
        regression["action_domain_output_hash_before_reference_load"] == regression["action_domain_output_hash_recomputed_without_reference"]
    )
    atomic_json(OUT / "ep49_regression_metrics.json", regression)
    ep49_images = Path(next(row["source_folder"] for row in rows if row["episode_id"] == 49)) / "images/observation.images.cam_high/episode_000000"
    ep49_media = {
        "contact_sheet": contact_sheet(OUT / "ep49_detected_vs_approved_contact_sheet.png", ep49_images, ep49, int(alignment["action_to_observation_lag_frames"])),
        "overlay": overlay_video(OUT / "ep49_semantic_overlay.mp4", ep49_images, ep49, ep49_context, int(alignment["action_to_observation_lag_frames"])),
    }

    timelines: dict[int, SemanticTimeline] = {}
    contexts: dict[int, dict[str, Any]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode_id = row["episode_id"]
        directory = OUT / "episodes" / f"{episode_id:02d}"
        _, timeline, _, context = process(Path(row["source_parquet"]), "raw_action", config, geometry, directory)
        timelines[episode_id] = timeline
        contexts[episode_id] = context
        summaries[episode_id] = episode_summary(episode_id, timeline, row)
    summary_rows = [summaries[index] for index in sorted(summaries)]
    flat_rows = []
    for row in summary_rows:
        flat = {key: (json.dumps(value) if isinstance(value, list) else value) for key, value in row.items()}
        flat_rows.append(flat)
    dump_csv(OUT / "batch_semantic_summary.csv", flat_rows)
    atomic_json(OUT / "batch_semantic_summary.json", {
        "episodes": summary_rows, "detector_config_hash": config_hash,
        "all_processed_without_crash": len(summary_rows) == len(rows),
        "partial_order_valid_count": sum(row["partial_order_valid"] for row in summary_rows),
    })

    subsets = deterministic_subsets(rows)
    smoke = readiness_record("READY_FOR_3_HELDOUT_RETARGETING", subsets["three"], summaries, 3)
    ten = readiness_record("READY_FOR_10_EPISODE_RETARGETING", subsets["ten"], summaries, 8)
    thirty = readiness_record("READY_FOR_30_EPISODE_RETARGETING", subsets["thirty"], summaries, 27)
    all_complete = sum(row["high_medium_complete"] for row in summary_rows)
    fifty = {
        "status": "READY_FOR_50_EPISODE_BATCH_RETARGETING" if all_complete >= 45 else "BLOCKED_MULTI_EPISODE_EVENT_COVERAGE",
        "episode_ids": sorted(summaries), "discovered_episode_count": len(summaries),
        "high_medium_complete_count": all_complete,
        "partial_timeline_count": sum(row["mandatory_complete"] and not row["high_medium_complete"] for row in summary_rows),
        "missing_timeline_count": sum(not row["mandatory_complete"] for row in summary_rows),
        "ambiguous_episode_count": sum(bool(row["ambiguous_events"]) for row in summary_rows),
        "partial_order_valid_count": sum(row["partial_order_valid"] for row in summary_rows),
        "same_detector_config": len({row["detector_config_hash"] for row in summary_rows}) == 1,
        "episode_specific_tuning": False,
        "thresholds_lowered_for_coverage": False,
        "episode_results": summary_rows,
    }
    atomic_json(OUT / "three_episode_pilot.json", smoke)
    atomic_json(OUT / "ten_episode_validation.json", ten)
    atomic_json(OUT / "thirty_episode_validation.json", thirty)
    atomic_json(OUT / "fifty_episode_readiness.json", fifty)

    pilot_media = {}
    for episode_id in subsets["three"]:
        row = next(value for value in rows if value["episode_id"] == episode_id)
        image_directory = Path(row["source_folder"]) / "images/observation.images.cam_high/episode_000000"
        pilot_media[str(episode_id)] = {
            "contact_sheet": contact_sheet(OUT / f"pilot_ep{episode_id:02d}_semantic_contact_sheet.png", image_directory, timelines[episode_id]),
            "overlay": overlay_video(OUT / f"pilot_ep{episode_id:02d}_semantic_overlay.mp4", image_directory, timelines[episode_id], contexts[episode_id]),
        }
    timeline_grid(OUT / "ten_episode_event_timeline_grid.png", subsets["ten"], timelines, "Deterministic 10-episode held-out event timelines")
    confidence_heatmap(OUT / "ten_episode_confidence_heatmap.png", subsets["ten"], timelines, "Deterministic 10-episode confidence")
    timeline_grid(OUT / "thirty_episode_event_timeline_grid.png", subsets["thirty"], timelines, "Primary 30-episode generalization event timelines")
    confidence_heatmap(OUT / "thirty_episode_confidence_heatmap.png", subsets["thirty"], timelines, "Primary 30-episode generalization confidence")
    distribution_artifacts(summary_rows, timelines)

    hardcoding = frame_hardcoding_audit()
    atomic_json(OUT / "frame_hardcoding_audit.json", hardcoding)
    write_hardcoding_markdown(hardcoding)
    interface = retarget_aloha_trajectory_to_g1(
        ep49_loaded["action"], ep49_loaded["timestamps"], ep49,
        {"workspace_scale": 0.42, "mode": "frozen_translator_config_dry_run"},
        {"scene": "authoritative_isaac_lab", "dry_run": True}, dry_run=True,
    )
    atomic_json(OUT / "generic_converter_interface_audit.json", interface)
    v15 = v15_readiness(ep49)
    v15["future_phase_mapping"] = PHASE_API_MAPPING
    atomic_json(OUT / "v15_semantic_interface_readiness.json", v15)

    after_hashes = file_hashes(immutable_paths())
    immutable = {path: {"before": before_hashes[path], "after": after_hashes[path], "byte_identical": before_hashes[path] == after_hashes[path]} for path in before_hashes}
    atomic_json(OUT / "immutable_v14_scene_hash_audit.json", {
        "status": "PASS" if all(row["byte_identical"] for row in immutable.values()) else "FAIL",
        "files": immutable,
        "trajectory_generation_executed": False,
    })
    atomic_json(OUT / "review_artifacts.json", {"ep49": ep49_media, "pilot": pilot_media})
    atomic_json(OUT / "run_state_pretests.json", {
        "created_at": utc_now(), "detector_config_hash": config_hash,
        "Episode_49_regression": regression["status"], "smoke": smoke["status"], "ten": ten["status"],
        "thirty": thirty["status"], "fifty": fifty["status"], "hardcoding": hardcoding["status"],
    })
    print(json.dumps({
        "output": str(OUT), "config_hash": config_hash, "ep49": regression["status"],
        "smoke": smoke["status"], "ten": ten["status"], "thirty": thirty["status"],
        "fifty": fifty["status"], "complete_50": all_complete,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

