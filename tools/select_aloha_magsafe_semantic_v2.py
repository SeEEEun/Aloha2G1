#!/usr/bin/env python3
"""Validation-only selection and immutable freeze of one global v2 config."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS  # noqa: E402
from aloha_magsafe_semantics_v2.experiment import (  # noqa: E402
    AccessLedger, OUT, aggregate_status, atomic_json, canonical_hash, detect_loaded,
    load_episode_inputs, load_split, read_manifest, timeline_summary,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_rows(timeline: Any, config_name: str, episode_id: int) -> list[dict[str, Any]]:
    rows = []
    for name in REQUIRED_EVENTS:
        event = timeline.event(name)
        attribution = event.evidence.get("v2_confidence_attribution", {})
        evidence_supported = bool(
            attribution.get("physical_evidence_gate", 0.0) >= 1.0
            and attribution.get("evidence_breadth", 0.0) >= 0.55
            and attribution.get("dwell_stability", 0.0) >= 0.35
        )
        rows.append({
            "config": config_name,
            "episode_id": episode_id,
            "event_name": name,
            "confidence": float(event.confidence),
            "confidence_class": event.confidence_class,
            "evidence_supported_proxy": evidence_supported,
        })
    return rows


def calibration_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidence = np.asarray([row["confidence"] for row in rows], dtype=np.float64)
    target = np.asarray([row["evidence_supported_proxy"] for row in rows], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, 11)
    reliability = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lower) & ((confidence < upper) | ((upper == 1.0) & (confidence <= upper)))
        reliability.append({
            "lower": float(lower), "upper": float(upper), "count": int(np.sum(mask)),
            "mean_confidence": float(np.mean(confidence[mask])) if np.any(mask) else None,
            "evidence_supported_fraction": float(np.mean(target[mask])) if np.any(mask) else None,
        })
    return {
        "proxy_definition": "physical evidence gate + evidence breadth >=0.55 + dwell/stability >=0.35",
        "brier_score": float(np.mean((confidence - target) ** 2)),
        "reliability_bins": reliability,
    }


def reliability_plot(path: Path, metrics_by_config: dict[str, dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    axis.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
    for name, metrics in metrics_by_config.items():
        rows = [row for row in metrics["reliability_bins"] if row["count"]]
        axis.plot(
            [row["mean_confidence"] for row in rows],
            [row["evidence_supported_fraction"] for row in rows],
            marker="o", label=f"{name} (Brier {metrics['brier_score']:.3f})",
        )
    axis.set(xlabel="Predicted confidence", ylabel="Evidence-supported fraction", xlim=(0, 1), ylim=(0, 1), title="Validation reliability (evidence-support proxy)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def disagreement_plot(path: Path, validation: list[int], timelines: dict[str, dict[int, Any]]) -> None:
    names = list(timelines)
    matrix = np.zeros((len(validation), len(REQUIRED_EVENTS)), dtype=np.float64)
    for row, episode_id in enumerate(validation):
        for column, event in enumerate(REQUIRED_EVENTS):
            indices = [timelines[name][episode_id].event(event).action_index for name in names]
            times = [timelines[name][episode_id].event(event).action_time_sec for name in names]
            matrix[row, column] = 0.0 if len(set(indices)) == 1 else max(times) - min(times)
    figure, axis = plt.subplots(figsize=(15, 5.5))
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_yticks(range(len(validation)), [f"ep{value:02d}" for value in validation])
    axis.set_xticks(range(len(REQUIRED_EVENTS)), REQUIRED_EVENTS, rotation=55, ha="right", fontsize=8)
    axis.set_title("Validation disagreement among three predeclared global configs (seconds)")
    figure.colorbar(image, ax=axis, label="max event-time disagreement (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    split = load_split()
    development = set(int(value) for value in split["development"])
    validation = [int(value) for value in split["validation"]]
    heldout = set(int(value) for value in split["heldout_test"])
    development_access = json.loads((OUT / "development_access_log.json").read_text(encoding="utf-8"))
    if development_access["prohibited_access_count"] != 0 or not development_access["heldout_access_prohibition_pass"]:
        raise RuntimeError("development access boundary failed")
    configs_payload = json.loads((OUT / "detector_v2_config_candidates.json").read_text(encoding="utf-8"))
    if configs_payload["status"] != "PREDECLARED_BEFORE_VALIDATION" or len(configs_payload["configs"]) > 3:
        raise RuntimeError("global configs were not predeclared correctly")
    configs = configs_payload["configs"]
    ledger = AccessLedger("VALIDATION_CONFIG_SELECTION", set(validation), development | heldout)
    manifest = {row["episode_id"]: row for row in read_manifest()}
    inputs = {
        episode_id: load_episode_inputs(manifest[episode_id], ledger, "validation global config selection")
        for episode_id in validation
    }
    results: dict[str, Any] = {}
    timelines: dict[str, dict[int, Any]] = {}
    calibration: dict[str, dict[str, Any]] = {}
    for config_name, config in configs.items():
        summaries = []
        config_timelines = {}
        rows = []
        fabricated_release_count = 0
        for episode_id in validation:
            loaded, fk, task_geometry = inputs[episode_id]
            timeline = detect_loaded(loaded, fk, task_geometry, config)
            config_timelines[episode_id] = timeline
            summaries.append(timeline_summary(episode_id, timeline, manifest[episode_id]))
            rows.extend(calibration_rows(timeline, config_name, episode_id))
            release = timeline.event("right_accessory_release_complete")
            attribution = release.evidence.get("v2_confidence_attribution", {})
            if attribution.get("physical_evidence_gate", 0.0) < 1.0 and release.confidence_class in ("HIGH", "MEDIUM"):
                fabricated_release_count += 1
        metrics = calibration_metrics(rows)
        aggregate = aggregate_status(summaries)
        results[config_name] = {
            "config_hash_predeclared": config["config_hash"],
            "scope": "VALIDATION_ONLY",
            "aggregate": aggregate,
            "fabricated_release_count": fabricated_release_count,
            "confidence_calibration": metrics,
            "episodes": summaries,
        }
        timelines[config_name] = config_timelines
        calibration[config_name] = metrics

    ranking = sorted(results, key=lambda name: (
        -results[name]["aggregate"]["partial_order_valid_count"],
        -results[name]["aggregate"]["mandatory_complete_count"],
        -results[name]["aggregate"]["high_medium_complete_count"],
        results[name]["fabricated_release_count"],
        results[name]["confidence_calibration"]["brier_score"],
        name,
    ))
    selected_name = ranking[0]
    selected = configs[selected_name]
    selected_hash = canonical_hash(selected)
    atomic_json(OUT / "selected_detector_v2_config.json", selected)
    code_files = sorted((ROOT / "tools/aloha_magsafe_semantics_v2").glob("*.py"))
    provenance = {
        "status": "DETECTOR_CONFIG_FROZEN_BEFORE_HELDOUT_TEST",
        "selected_candidate": selected_name,
        "selected_config_canonical_sha256": selected_hash,
        "selected_config_file_sha256": None,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_hash": split["split_hash"],
        "selection_priority": [
            "partial_order_valid", "mandatory_event_index_complete", "HIGH_MEDIUM_complete_count",
            "no_fabricated_release_increase", "confidence_calibration_quality", "Episode49_secondary_only",
        ],
        "ranking": ranking,
        "validation_results": {name: {
            "aggregate": value["aggregate"],
            "fabricated_release_count": value["fabricated_release_count"],
            "brier_score": value["confidence_calibration"]["brier_score"],
        } for name, value in results.items()},
        "code_hashes": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_files},
        "development_only_parameter_derivation": True,
        "validation_used_only_for_config_selection": True,
        "heldout_episode_access_count_before_freeze": 0,
        "approved_episode49_reference_used_for_selection": False,
        "episode_specific_tuning": False,
    }
    provenance["selected_config_file_sha256"] = file_sha256(OUT / "selected_detector_v2_config.json")
    atomic_json(OUT / "selected_detector_v2_config_provenance.json", provenance)
    atomic_json(OUT / "validation_config_results.json", results)
    atomic_json(OUT / "validation_10_result.json", {
        "status": "READY_FOR_10_EPISODE_RETARGETING" if results[selected_name]["aggregate"]["high_medium_complete_count"] >= 8 else "BLOCKED_VALIDATION_COVERAGE",
        "selected_config": selected_name,
        "selected_config_hash": selected_hash,
        "episode_ids": validation,
        **results[selected_name],
        "no_episode_specific_tuning": True,
    })
    development_results = json.loads((OUT / "development_config_results.json").read_text(encoding="utf-8"))[selected_name]
    smoke = set(int(value) for value in split["development_smoke"])
    smoke_rows = [row for row in development_results["episodes"] if row["episode_id"] in smoke]
    atomic_json(OUT / "development_10_result.json", {
        "scope": "DEVELOPMENT_DIAGNOSTIC_NOT_GENERALIZATION_CLAIM",
        "selected_config": selected_name,
        "selected_config_hash": selected_hash,
        **development_results,
        "smoke_episode_ids": sorted(smoke),
        "smoke_high_medium_complete_count": sum(row["high_medium_complete"] for row in smoke_rows),
        "smoke_status": "READY_FOR_3_HELDOUT_RETARGETING" if all(row["high_medium_complete"] for row in smoke_rows) else "BLOCKED_DEVELOPMENT_SMOKE",
    })
    calibration_payload = json.loads((OUT / "confidence_calibration_v2.json").read_text(encoding="utf-8"))
    calibration_payload.update({
        "validation_scope": "VALIDATION_ONLY",
        "validation_metrics_by_config": calibration,
        "selected_config": selected_name,
        "heldout_used_for_calibration": False,
    })
    atomic_json(OUT / "confidence_calibration_v2.json", calibration_payload)
    reliability_plot(OUT / "validation_reliability_diagram.png", calibration)
    disagreement_plot(OUT / "validation_config_disagreement_grid.png", validation, timelines)
    access = ledger.payload()
    access["heldout_access_prohibition_pass"] = not any(value in heldout for value in access["accessed_ids"])
    access["development_access_prohibition_pass"] = not any(value in development for value in access["accessed_ids"])
    atomic_json(OUT / "validation_access_log.json", access)
    atomic_json(OUT / "heldout_evaluation_receipt.json", {
        "selected_config_hash": selected_hash,
        "config_frozen": True,
        "heldout_evaluation_count": 0,
        "heldout_results_seen": False,
    })
    print(json.dumps({
        "selected": selected_name,
        "selected_config_hash": selected_hash,
        "ranking": ranking,
        "validation": {name: value["aggregate"] for name, value in results.items()},
        "heldout_access_count": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
