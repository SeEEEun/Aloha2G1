#!/usr/bin/env python3
"""Materialize one source-derived task-intent timeline shared by A and B."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_direct_physical_eval35/00_preparation"
HELDOUT = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
EVAL10 = ROOT / "outputs/final_contact_constrained_eval/04_eval10_preparation/EVAL10_RETARGETING_MANIFEST.json"
NEW25 = OUT / "NEW_UNSEEN_25_FROZEN_AB_CONVERSION.json"
SOURCE_EVENT_DETECTOR = (
    ROOT / "outputs/dataset_a_final50_retargeting/event_audit/detector_config.json"
)
NEW25_SOURCE_EVENT_DETECTOR = (
    OUT / "runtime_frozen_fair_a/event_audit/detector_config.json"
)
NPZ = OUT / "COMMON_SOURCE_TASK_INTENT_EVAL35.npz"
MANIFEST = OUT / "COMMON_SOURCE_TASK_INTENT_EVAL35.json"
INTENT_EVENTS = (
    "LEFT_CLOSE_ONSET",
    "LEFT_STABLE_HOLD",
    "RIGHT_CLOSE_ONSET",
    "LEFT_RELEASE",
    "RIGHT_FINAL_RELEASE",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def event_dict(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["event_names"].astype(str)
        frames = archive["event_frames"].astype(np.int64)
    return {str(name): int(frame) for name, frame in zip(names, frames, strict=True)}


def make_timeline(frames: int, events: dict[str, int]) -> np.ndarray:
    if any(name not in events or events[name] < 0 for name in INTENT_EVENTS):
        raise RuntimeError(f"incomplete source task events: {events}")
    close, hold, handoff, right_hold, release = (
        events[name] for name in INTENT_EVENTS
    )
    if not 0 <= close <= hold <= handoff <= right_hold <= release < frames:
        raise RuntimeError(f"invalid source intent ordering: {events}, frames={frames}")
    values = np.full(frames, "OPEN_INTENT", dtype="U24")
    values[close:hold] = "LEFT_CLOSE_INTENT"
    values[hold:handoff] = "LEFT_HOLD_INTENT"
    values[handoff:right_hold] = "HANDOFF_INTENT"
    values[right_hold:release] = "RIGHT_HOLD_INTENT"
    values[release:] = "FINAL_RELEASE_INTENT"
    return values


def main() -> int:
    if MANIFEST.is_file() and NPZ.is_file():
        cached = read_json(MANIFEST)
        if cached.get("status") != "FROZEN_BEFORE_ROLLOUT" or sha256_file(NPZ) != cached["artifact_sha256"]:
            raise RuntimeError("existing common intent artifact is invalid")
        print(json.dumps({"status": cached["status"], "cache_hit": str(MANIFEST)}, indent=2))
        return 0
    heldout = read_json(HELDOUT)
    eval10 = read_json(EVAL10)
    new25 = read_json(NEW25)
    if sha256_file(SOURCE_EVENT_DETECTOR) != sha256_file(NEW25_SOURCE_EVENT_DETECTOR):
        raise RuntimeError("authoritative source-only event detector drift")
    if len(heldout["entries"]) != 8 or len(eval10["new_unseen_2"]) != 2 or len(new25["records"]) != 25:
        raise RuntimeError("EVAL35 source partition count mismatch")
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for index, row in enumerate(heldout["entries"]):
        a_path = Path(row["a_trajectory_path"])
        b_path = Path(row["b_trajectory_path"])
        a_events = {
            name: event_dict(a_path)[name] for name in INTENT_EVENTS
        }
        b_events = {
            name: event_dict(b_path)[name] for name in INTENT_EVENTS
        }
        event = a_events
        frames = int(row["frames"])
        stable = row["stable_episode_id"]
        provenance = "PREDEFINED_HELDOUT"
        sources = [str(a_path), str(b_path)]
        source_hashes = [sha256_file(a_path), sha256_file(b_path)]
        timeline = make_timeline(frames, event)
        arrays[f"eval_{index:02d}"] = timeline
        records.append({
            "eval_index": index, "stable_episode_id": stable,
            "provenance": provenance, "frames": frames, "event_frames": event,
            "common_source_detector": str(SOURCE_EVENT_DETECTOR.resolve()),
            "alternate_converter_event_difference": {
                name: [a_events[name], b_events[name]]
                for name in INTENT_EVENTS if a_events[name] != b_events[name]
            },
            "source_artifacts": sources,
            "source_artifact_sha256": source_hashes,
            "timeline_sha256": sha256_array(timeline),
        })
    for offset, row in enumerate(eval10["new_unseen_2"], start=8):
        a_events = {
            name: int(row["a_initial"]["event_frames"][name])
            for name in INTENT_EVENTS
        }
        b_events = {
            name: int(row["b_initial"]["event_frames"][name])
            for name in INTENT_EVENTS
        }
        timeline = make_timeline(int(row["frames"]), a_events)
        arrays[f"eval_{offset:02d}"] = timeline
        records.append({
            "eval_index": offset, "stable_episode_id": row["stable_episode_id"],
            "provenance": "POST_TRAINING_UNSEEN", "frames": int(row["frames"]),
            "event_frames": a_events,
            "common_source_detector": str(SOURCE_EVENT_DETECTOR.resolve()),
            "alternate_converter_event_difference": {
                name: [a_events[name], b_events[name]]
                for name in INTENT_EVENTS if a_events[name] != b_events[name]
            },
            "source_artifacts": [row["a_initial"]["initial_trajectory"], row["b_initial"]["initial_trajectory"]],
            "source_artifact_sha256": [row["a_initial"]["initial_trajectory_sha256"], row["b_initial"]["initial_trajectory_sha256"]],
            "timeline_sha256": sha256_array(timeline),
        })
    for offset, row in enumerate(new25["records"], start=10):
        a_events = {
            name: int(row["a_initial"]["event_frames"][name])
            for name in INTENT_EVENTS
        }
        b_events = {
            name: int(row["b_initial"]["event_frames"][name])
            for name in INTENT_EVENTS
        }
        timeline = make_timeline(int(row["frames"]), a_events)
        arrays[f"eval_{offset:02d}"] = timeline
        records.append({
            "eval_index": offset, "stable_episode_id": row["stable_episode_id"],
            "provenance": "NEW_UNSEEN_25", "frames": int(row["frames"]),
            "event_frames": a_events,
            "common_source_detector": str(SOURCE_EVENT_DETECTOR.resolve()),
            "alternate_converter_event_difference": {
                name: [a_events[name], b_events[name]]
                for name in INTENT_EVENTS if a_events[name] != b_events[name]
            },
            "source_artifacts": [row["a_initial"]["initial_trajectory"], row["b_initial"]["initial_trajectory"]],
            "source_artifact_sha256": [row["a_initial"]["initial_trajectory_sha256"], row["b_initial"]["initial_trajectory_sha256"]],
            "timeline_sha256": sha256_array(timeline),
        })
    if len(records) != 35 or len({row["stable_episode_id"] for row in records}) != 35:
        raise RuntimeError("common intent is not exact unique EVAL35")
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = NPZ.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, NPZ)
    value = {
        "schema_version": "eval35_common_source_task_intent_v1",
        "status": "FROZEN_BEFORE_ROLLOUT", "evaluation_set": "EVAL35",
        "source": "authoritative ALOHA gripper transitions shared identically by A and B",
        "source_event_detector": str(SOURCE_EVENT_DETECTOR.resolve()),
        "source_event_detector_sha256": sha256_file(SOURCE_EVENT_DETECTOR),
        "event_detector_operates_before_retargeting": True,
        "method_specific_phase_labels_used": False,
        "manual_or_episode_specific_timing_used": False,
        "intent_states": [
            "OPEN_INTENT", "LEFT_CLOSE_INTENT", "LEFT_HOLD_INTENT",
            "HANDOFF_INTENT", "RIGHT_HOLD_INTENT", "FINAL_RELEASE_INTENT",
        ],
        "artifact": str(NPZ.resolve()), "artifact_sha256": sha256_file(NPZ),
        "records": records,
    }
    MANIFEST.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "episodes": 35, "artifact": str(NPZ)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
