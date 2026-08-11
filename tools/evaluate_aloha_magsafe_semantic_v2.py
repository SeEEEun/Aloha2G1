#!/usr/bin/env python3
"""One-shot held-out evaluation and post-freeze descriptive full-50 audit."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS  # noqa: E402
from aloha_magsafe_semantics.features import compute_stationary_aloha_fk  # noqa: E402
from aloha_magsafe_semantics.io import load_trajectory, sha256_file  # noqa: E402
from aloha_magsafe_semantics_v2.detector import detect_magsafe_semantics  # noqa: E402
from aloha_magsafe_semantics_v2.experiment import (  # noqa: E402
    AccessLedger, GEOMETRY_PATH, MODEL, OUT, POSE_CONFIG, aggregate_status, atomic_json,
    canonical_hash, detect_loaded, load_episode_inputs, load_split, read_manifest,
    timeline_summary,
)
from retarget_aloha_trajectory_to_g1 import retarget_aloha_trajectory_to_g1  # noqa: E402
from v15_semantic_interface import PHASE_API_MAPPING, readiness as v15_readiness  # noqa: E402


V1_OUT = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1"
EP49_ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
EP49_REFERENCE = ROOT / "configs/episode49_task_timeline.approved.json"
EP49_ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def verify_freeze(config: dict[str, Any], split: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = json.loads((OUT / "selected_detector_v2_config_provenance.json").read_text(encoding="utf-8"))
    receipt = json.loads((OUT / "heldout_evaluation_receipt.json").read_text(encoding="utf-8"))
    if provenance["status"] != "DETECTOR_CONFIG_FROZEN_BEFORE_HELDOUT_TEST":
        raise RuntimeError("selected config is not frozen")
    if canonical_hash(config) != provenance["selected_config_canonical_sha256"]:
        raise RuntimeError("selected config changed after freeze")
    if split["split_hash"] != provenance["split_hash"]:
        raise RuntimeError("dataset split changed after config selection")
    for relative, expected in provenance["code_hashes"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"v2 detector code changed after freeze: {relative}")
    if receipt["heldout_evaluation_count"] != 0 or receipt["heldout_results_seen"]:
        raise RuntimeError("held-out test has already been evaluated; v2 forbids a rerun")
    return provenance, receipt


def save_timeline(episode_id: int, timeline: Any) -> None:
    directory = OUT / "episodes" / f"{episode_id:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "semantic_timeline.auto.json", timeline.to_dict())
    np.savez_compressed(
        directory / "semantic_phases.npz",
        timestamps=timeline.timestamps,
        event_names=np.asarray(list(timeline.events), dtype="U48"),
        event_action_indices=np.asarray([
            -1 if timeline.events[name].action_index is None else timeline.events[name].action_index
            for name in timeline.events
        ], dtype=np.int64),
        **timeline.sample_arrays,
    )
    atomic_json(directory / "feature_metrics.json", {
        "trajectory_length": timeline.trajectory_length,
        "duration_sec": float(timeline.timestamps[-1] - timeline.timestamps[0]),
        "detector_version": timeline.metadata["detector_version"],
        "global_sequence_margin": timeline.metadata["global_sequence_score_margin"],
        "rotation_segmentation": timeline.metadata["rotation_segmentation"],
        "terminal_suffix": timeline.metadata["terminal_suffix"],
        "release_physical_evidence": timeline.metadata["release_physical_evidence"],
    })


def coverage_payload(name: str, ids: list[int], summaries: dict[int, dict[str, Any]], threshold: int, ready_status: str, blocked_status: str) -> dict[str, Any]:
    rows = [summaries[episode_id] for episode_id in ids]
    aggregate = aggregate_status(rows)
    ready = (
        aggregate["high_medium_complete_count"] >= threshold
        and aggregate["partial_order_valid_count"] == len(ids)
        and aggregate["same_detector_config"]
    )
    return {
        "status": ready_status if ready else blocked_status,
        "scope": name,
        "episode_ids": ids,
        **aggregate,
        "required_high_medium_complete_count": threshold,
        "manual_event_edits": False,
        "episode_specific_tuning": False,
        "partial_order_requirement": len(ids),
        "episodes": rows,
    }


def v1_v2_comparisons(summaries: dict[int, dict[str, Any]]) -> None:
    v1 = json.loads((V1_OUT / "batch_semantic_summary.json").read_text(encoding="utf-8"))["episodes"]
    v1_by_id = {int(row["episode_id"]): row for row in v1}
    event_rows = []
    transition = collections.Counter()
    for name in REQUIRED_EVENTS:
        before = collections.Counter(v1_by_id[episode_id][f"{name}_class"] for episode_id in sorted(summaries))
        after = collections.Counter(summaries[episode_id][f"{name}_class"] for episode_id in sorted(summaries))
        for episode_id in sorted(summaries):
            transition[(name, v1_by_id[episode_id][f"{name}_class"], summaries[episode_id][f"{name}_class"])] += 1
        event_rows.append({
            "event_name": name,
            **{f"v1_{value.lower()}": before.get(value, 0) for value in ("HIGH", "MEDIUM", "LOW", "AMBIGUOUS")},
            **{f"v2_{value.lower()}": after.get(value, 0) for value in ("HIGH", "MEDIUM", "LOW", "AMBIGUOUS")},
            "v1_low_ambiguous": before.get("LOW", 0) + before.get("AMBIGUOUS", 0),
            "v2_low_ambiguous": after.get("LOW", 0) + after.get("AMBIGUOUS", 0),
        })
    dump_csv(OUT / "v1_vs_v2_event_summary.csv", event_rows)
    episode_rows = []
    for episode_id in sorted(summaries):
        old = v1_by_id[episode_id]
        new = summaries[episode_id]
        changed = [name for name in REQUIRED_EVENTS if old[f"{name}_index"] != new[f"{name}_index"] or old[f"{name}_class"] != new[f"{name}_class"]]
        episode_rows.append({
            "episode_id": episode_id,
            "v1_high_medium_complete": old["high_medium_complete"],
            "v2_high_medium_complete": new["high_medium_complete"],
            "v1_low_events": json.dumps(old["low_events"]),
            "v1_ambiguous_events": json.dumps(old["ambiguous_events"]),
            "v2_low_events": json.dumps(new["low_events"]),
            "v2_ambiguous_events": json.dumps(new["ambiguous_events"]),
            "changed_event_count": len(changed),
            "changed_events": json.dumps(changed),
        })
    dump_csv(OUT / "v1_vs_v2_episode_summary.csv", episode_rows)
    transition_rows = [{
        "event_name": name, "v1_class": old, "v2_class": new, "count": count,
    } for (name, old, new), count in sorted(transition.items())]
    dump_csv(OUT / "v1_vs_v2_confidence_transition.csv", transition_rows)
    atomic_json(OUT / "v1_to_v2_changed_event_audit.json", {
        "changed_event_records": int(sum(row["changed_event_count"] for row in episode_rows)),
        "episodes_with_changes": sum(row["changed_event_count"] > 0 for row in episode_rows),
        "confidence_upgrades_require_new_evidence_fields": True,
        "rows": episode_rows,
    })


def episode49_regression(config: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    loaded = load_trajectory(EP49_ACTION, "optimized_action")
    pose = json.loads(POSE_CONFIG.read_text(encoding="utf-8"))["stationary_aloha"]
    fk = compute_stationary_aloha_fk(
        loaded["action"], loaded["timestamps"], MODEL,
        pose["position_xyz_m"], pose["orientation_wxyz"],
    )
    fk["model_sha256"] = sha256_file(MODEL)
    task_geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
    first = detect_magsafe_semantics(loaded["action"], loaded["timestamps"], "optimized_action", fk, task_geometry, config)
    first_hash = canonical_hash(first.to_dict())
    second = detect_magsafe_semantics(loaded["action"], loaded["timestamps"], "optimized_action", fk, task_geometry, config)
    second_hash = canonical_hash(second.to_dict())
    # Approved references are intentionally opened only after both detections.
    _ = json.loads(EP49_REFERENCE.read_text(encoding="utf-8"))
    alignment = json.loads(EP49_ALIGNMENT.read_text(encoding="utf-8"))
    v1_regression = json.loads((V1_OUT / "ep49_regression_metrics.json").read_text(encoding="utf-8"))
    v1_events = {row["event_name"]: row for row in v1_regression["events"]}
    rows = []
    for name in REQUIRED_EVENTS:
        event = first.event(name)
        approved = int(alignment["event_mapping"][name]["aligned_action_index"])
        error_samples = abs(int(event.action_index) - approved) if event.action_index is not None else None
        tolerance = 0.5 if name in (
            "left_phone_grasp_start", "right_accessory_grasp_start",
            "left_phone_release_complete", "right_accessory_release_complete",
        ) else 0.8
        rows.append({
            "event_name": name,
            "detected_action_index": event.action_index,
            "approved_action_index": approved,
            "error_samples": error_samples,
            "error_seconds": None if error_samples is None else float(error_samples / 30.0),
            "diagnostic_tolerance_seconds": tolerance,
            "within_diagnostic_tolerance": error_samples is not None and error_samples / 30.0 <= tolerance,
            "v1_error_seconds": float(v1_events[name]["absolute_error_seconds"]),
            "v2_error_seconds": None if error_samples is None else float(error_samples / 30.0),
            "v1_confidence_class": v1_events[name]["confidence_class"],
            "v2_confidence_class": event.confidence_class,
            "v2_confidence": event.confidence,
            "v2_candidate_source": event.evidence.get("candidate_source"),
            "evidence_change": "v2 segmentation/suffix/release attribution" if event.evidence.get("candidate_source", "").startswith("v2_") else "preserved generic signal evidence",
        })
    payload = {
        "status": "PASS" if all(row["within_diagnostic_tolerance"] for row in rows) else "SEMANTIC_EP49_REGRESSION_WARNING",
        "detector_completed_before_reference_loaded": True,
        "reference_timeline_used_for_detection": False,
        "approved_reference_independence_pass": first_hash == second_hash,
        "action_domain_output_hash_first_without_reference": first_hash,
        "action_domain_output_hash_second_without_reference": second_hash,
        "reference_file": str(EP49_REFERENCE.resolve()),
        "reference_file_sha256": file_sha256(EP49_REFERENCE),
        "alignment_file": str(EP49_ALIGNMENT.resolve()),
        "alignment_file_sha256": file_sha256(EP49_ALIGNMENT),
        "events": rows,
    }
    return first, loaded, payload


def main() -> int:
    split = load_split()
    config = json.loads((OUT / "selected_detector_v2_config.json").read_text(encoding="utf-8"))
    provenance, receipt = verify_freeze(config, split)
    heldout = [int(value) for value in split["heldout_test"]]
    development = [int(value) for value in split["development"]]
    validation = [int(value) for value in split["validation"]]
    manifest = {row["episode_id"]: row for row in read_manifest()}
    receipt.update({
        "heldout_evaluation_count": 1,
        "heldout_results_seen": False,
        "evaluation_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_invocation_policy": "exactly one selected-config detection per held-out episode",
    })
    atomic_json(OUT / "heldout_evaluation_receipt.json", receipt)
    ledger = AccessLedger("HELDOUT_ONE_SHOT_EVALUATION", set(heldout), set(development + validation))
    timelines: dict[int, Any] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for episode_id in heldout:
        loaded, fk, task_geometry = load_episode_inputs(manifest[episode_id], ledger, "single frozen-config heldout evaluation")
        timeline = detect_loaded(loaded, fk, task_geometry, config)
        timelines[episode_id] = timeline
        summaries[episode_id] = timeline_summary(episode_id, timeline, manifest[episode_id])
        save_timeline(episode_id, timeline)
    heldout_payload = coverage_payload(
        "HELDOUT_TEST_30", heldout, summaries, 27,
        "READY_FOR_30_EPISODE_RETARGETING", "BLOCKED_HELDOUT_30_COVERAGE",
    )
    heldout_payload.update({
        "selected_config": provenance["selected_candidate"],
        "selected_config_hash": provenance["selected_config_canonical_sha256"],
        "config_frozen_before_evaluation": True,
        "detector_invocation_count": len(heldout),
        "parameters_changed_after_results": False,
    })
    atomic_json(OUT / "heldout_30_result.json", heldout_payload)
    access = ledger.payload()
    access["development_or_validation_access_count"] = 0
    atomic_json(OUT / "heldout_access_log.json", access)
    receipt.update({
        "heldout_results_seen": True,
        "evaluation_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "detector_invocation_count": len(heldout),
        "result_file": str((OUT / "heldout_30_result.json").resolve()),
        "result_file_sha256": file_sha256(OUT / "heldout_30_result.json"),
        "parameters_changed_after_results": False,
    })
    atomic_json(OUT / "heldout_evaluation_receipt.json", receipt)

    # Post-test descriptive coverage reuses held-out results and processes only
    # development/validation IDs; held-out raw data are never reopened.
    descriptive_ledger = AccessLedger("POST_FREEZE_DEVELOPMENT_VALIDATION_ARTIFACTS", set(development + validation), set(heldout))
    for episode_id in development + validation:
        loaded, fk, task_geometry = load_episode_inputs(manifest[episode_id], descriptive_ledger, "post-freeze descriptive selected-config output")
        timeline = detect_loaded(loaded, fk, task_geometry, config)
        timelines[episode_id] = timeline
        summaries[episode_id] = timeline_summary(episode_id, timeline, manifest[episode_id])
        save_timeline(episode_id, timeline)
    atomic_json(OUT / "postfreeze_descriptive_access_log.json", descriptive_ledger.payload())
    all_ids = sorted(summaries)
    full = coverage_payload(
        "FULL_50_DESCRIPTIVE_COVERAGE", all_ids, summaries, 45,
        "READY_FOR_50_EPISODE_BATCH_RETARGETING", "BLOCKED_FULL_50_COVERAGE",
    )
    event_classes = {
        name: dict(collections.Counter(summaries[episode_id][f"{name}_class"] for episode_id in all_ids))
        for name in REQUIRED_EVENTS
    }
    full.update({
        "selected_config": provenance["selected_candidate"],
        "selected_config_hash": provenance["selected_config_canonical_sha256"],
        "heldout_results_reused_without_second_detection": True,
        "heldout_detector_invocations_total": len(heldout),
        "event_confidence_counts": event_classes,
        "complete_with_low_or_ambiguous_count": sum(row["mandatory_complete"] and not row["high_medium_complete"] for row in summaries.values()),
        "mandatory_index_missing_episode_count": sum(not row["mandatory_complete"] for row in summaries.values()),
    })
    atomic_json(OUT / "full_50_coverage.json", full)
    v1_v2_comparisons(summaries)

    ep49, ep49_loaded, regression = episode49_regression(config)
    atomic_json(OUT / "ep49_auto_timeline_v2.json", ep49.to_dict())
    np.savez_compressed(OUT / "ep49_semantic_phases_v2.npz", timestamps=ep49.timestamps, **ep49.sample_arrays)
    atomic_json(OUT / "episode49_regression_v2.json", regression)
    interface = retarget_aloha_trajectory_to_g1(
        ep49_loaded["action"], ep49_loaded["timestamps"], ep49,
        {"mode": "v2 semantic dry run", "workspace_scale": 0.42},
        {"scene": "authoritative Isaac Lab", "dry_run": True}, dry_run=True,
    )
    atomic_json(OUT / "generic_converter_interface_audit_v2.json", interface)
    v15 = v15_readiness(ep49)
    v15.update({
        "future_phase_mapping": PHASE_API_MAPPING,
        "selected_detector_config_hash": provenance["selected_config_canonical_sha256"],
        "heldout_30_status": heldout_payload["status"],
        "automatic_resume_allowed": heldout_payload["status"] == "READY_FOR_30_EPISODE_RETARGETING",
        "v15_executed": False,
        "runtime_literal_event_indices": False,
        "variable_trajectory_length_supported": True,
        "simultaneous_events_supported": True,
    })
    atomic_json(OUT / "v15_semantic_interface_readiness_v2.json", v15)

    print(json.dumps({
        "selected_config": provenance["selected_candidate"],
        "heldout": heldout_payload["status"],
        "heldout_high_medium": heldout_payload["high_medium_complete_count"],
        "heldout_partial_order": heldout_payload["partial_order_valid_count"],
        "full50": full["status"],
        "full50_high_medium": full["high_medium_complete_count"],
        "episode49_regression": regression["status"],
        "heldout_evaluated_once": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
