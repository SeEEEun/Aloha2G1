#!/usr/bin/env python3
"""Development-only diagnosis and predeclaration of detector-v2 configs."""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from aloha_magsafe_semantics.event_names import REQUIRED_EVENTS  # noqa: E402
from aloha_magsafe_semantics_v2.experiment import (  # noqa: E402
    AccessLedger, OUT, aggregate_status, atomic_json, canonical_hash, detect_loaded,
    load_episode_inputs, load_split, read_manifest, timeline_summary,
)


V1_OUT = ROOT / "outputs/semantic_event_generalization/aloha_magsafe_semantics_v1"
V1_CONFIG = ROOT / "configs/aloha_magsafe_semantic_detector_v1.json"
BASE_CONFIG = ROOT / "configs/aloha_magsafe_semantic_detector_v2.base.json"

CORE_DURATION_EDGES = (
    ("left_phone_grasp_start", "phone_rotation_to_portrait_start"),
    ("phone_rotation_to_portrait_start", "phone_portrait_reached"),
    ("phone_portrait_reached", "right_accessory_grasp_start"),
    ("right_accessory_grasp_start", "accessory_detachment_start"),
    ("accessory_detachment_start", "accessory_removed"),
    ("accessory_removed", "phone_move_to_charger_start"),
    ("phone_move_to_charger_start", "phone_charger_attachment_complete"),
    ("phone_charger_attachment_complete", "left_phone_release_complete"),
    ("left_phone_release_complete", "left_arm_return_near_home"),
    ("accessory_removed", "right_accessory_release_complete"),
)


def load_v1_timeline(episode_id: int) -> dict[str, Any]:
    return json.loads((V1_OUT / "episodes" / f"{episode_id:02d}" / "semantic_timeline.auto.json").read_text(encoding="utf-8"))


def event_map(timeline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["event_name"]: row for row in timeline["events"]}


def taxonomy_for_event(episode_id: int, name: str, event: dict[str, Any], timeline_length: int, split: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    alternatives = event.get("alternatives", [])
    selected_score = max(event.get("evidence", {}).get("detector_score_components", {}).values(), default=event.get("confidence", 0.0))
    alternative_scores = [float(row.get("score", 0.0)) for row in alternatives]
    close_alternatives = sum(abs(selected_score - score) <= 0.08 for score in alternative_scores)
    margin = event.get("score_difference")
    detail = {
        "confidence_class": event["confidence_class"],
        "confidence": event["confidence"],
        "score_difference": margin,
        "close_alternative_count": close_alternatives,
        "action_index": event["action_index"],
    }
    if name == "phone_portrait_reached":
        return ("MULTIPLE_ROTATION_PLATEAUS" if close_alternatives else "WEAK_ROTATION_PLATEAU"), detail
    if name == "phone_rotation_to_portrait_start":
        return "ROTATION_OVERSHOOT_AND_RETURN", detail
    if name == "task_end":
        if event["action_index"] is not None and event["action_index"] < 0.85 * timeline_length:
            return "TERMINAL_SUFFIX_STARTED_TOO_EARLY", detail
        if close_alternatives:
            return "MULTIPLE_LOW_SPEED_DWELLS", detail
        return "LATE_MEANINGFUL_MOTION_AFTER_CANDIDATE_END", detail
    if name in ("left_phone_release_complete", "right_accessory_release_complete"):
        feature = next(row for row in split["selection_features_by_episode"] if row["episode_id"] == episode_id)
        side = "left" if name.startswith("left_") else "right"
        detail["gripper_range"] = feature[f"{side}_gripper_range"]
        ranges = [row[f"{side}_gripper_range"] for row in split["selection_features_by_episode"] if row["episode_id"] in split["development"]]
        if detail["gripper_range"] <= float(np.quantile(ranges, 0.15)):
            return "RELEASE_SIGNAL_ABSENT", detail
        return "RELEASE_SIGNAL_WEAK", detail
    if name in ("right_accessory_grasp_start", "accessory_detachment_start", "accessory_removed"):
        components = event.get("evidence", {}).get("detector_score_components", {})
        gripper = max((value for key, value in components.items() if "gripper" in key), default=0.0)
        motion = max((value for key, value in components.items() if any(token in key for token in ("motion", "speed", "displacement"))), default=0.0)
        detail.update({"gripper_evidence": gripper, "motion_evidence": motion})
        return ("MOTION_AND_GRIPPER_EVIDENCE_DISAGREE" if abs(gripper - motion) > 0.45 else "CONFIDENCE_MARGIN_TOO_SMALL"), detail
    if name == "left_phone_grasp_start":
        derivative = event.get("evidence", {}).get("gripper_derivative", 0.0)
        return ("GRIPPER_TRANSITION_TOO_EARLY" if derivative < 0 else "GRIPPER_TRANSITION_TOO_LATE"), detail
    return "OTHER_GLOBAL_PATTERN", detail


def development_taxonomy(development: list[int], split: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for episode_id in development:
        timeline = load_v1_timeline(episode_id)
        for name, event in event_map(timeline).items():
            if name not in REQUIRED_EVENTS or event["confidence_class"] not in ("LOW", "AMBIGUOUS"):
                continue
            category, details = taxonomy_for_event(episode_id, name, event, int(timeline["trajectory_length"]), split)
            rows.append({"episode_id": episode_id, "event_name": name, "category": category, "details": details})
    counts = Counter(row["category"] for row in rows)
    event_counts = Counter(row["event_name"] for row in rows)
    return {
        "scope": "DEVELOPMENT_ONLY",
        "development_episode_ids": development,
        "v1_low_or_ambiguous_event_count": len(rows),
        "category_counts": dict(sorted(counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "episodes": rows,
        "episode_specific_fix_created": False,
    }


def duration_priors(development: list[int]) -> dict[str, Any]:
    values: dict[str, list[tuple[float, float]]] = {f"{left}->{right}": [] for left, right in CORE_DURATION_EDGES}
    values["terminal_prerequisite->task_end"] = []
    for episode_id in development:
        timeline = load_v1_timeline(episode_id)
        events = event_map(timeline)
        duration = float(timeline["time_range_sec"][1] - timeline["time_range_sec"][0])
        for first, second in CORE_DURATION_EDGES:
            delta = float(events[second]["action_time_sec"] - events[first]["action_time_sec"])
            values[f"{first}->{second}"].append((delta, delta / max(duration, 1e-9)))
        prerequisite = max(
            float(events["left_arm_return_near_home"]["action_time_sec"]),
            float(events["right_accessory_release_complete"]["action_time_sec"]),
        )
        delta = float(events["task_end"]["action_time_sec"] - prerequisite)
        values["terminal_prerequisite->task_end"].append((delta, delta / max(duration, 1e-9)))
    result = {}
    for name, rows in values.items():
        seconds = np.asarray([row[0] for row in rows], dtype=np.float64)
        normalized = np.asarray([row[1] for row in rows], dtype=np.float64)
        sec_median = float(np.median(seconds))
        norm_median = float(np.median(normalized))
        result[name] = {
            "source": "DEVELOPMENT_ONLY_V1_SIGNAL_TIMELINES",
            "sample_count": len(rows),
            "median_sec": sec_median,
            "mad_sec": float(np.median(np.abs(seconds - sec_median))),
            "robust_scale_sec": float(max(1.4826 * np.median(np.abs(seconds - sec_median)), 0.35)),
            "q05_q95_sec": np.quantile(seconds, (0.05, 0.95)).tolist(),
            "median_normalized": norm_median,
            "mad_normalized": float(np.median(np.abs(normalized - norm_median))),
            "robust_scale_normalized": float(max(1.4826 * np.median(np.abs(normalized - norm_median)), 0.015)),
            "q05_q95_normalized": np.quantile(normalized, (0.05, 0.95)).tolist(),
        }
    return result


def candidate_configs(priors: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base["v1_base_config"] = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    base["duration_priors"] = priors
    variants: dict[str, dict[str, Any]] = {}
    conservative = copy.deepcopy(base)
    conservative["candidate_name"] = "V2_CONSERVATIVE"
    conservative["decoder_v2"].update({
        "v2_evidence_model_bonus": 0.18,
        "local_evidence_weight": 0.62,
        "duration_prior_weight": 0.14,
        "phase_consistency_weight": 0.24,
        "backward_terminal_weight": 0.28,
    })
    conservative["confidence_v2"].update({"high_threshold": 0.80, "medium_threshold": 0.58, "low_threshold": 0.38})
    variants[conservative["candidate_name"]] = conservative

    balanced = copy.deepcopy(base)
    balanced["candidate_name"] = "V2_BALANCED"
    variants[balanced["candidate_name"]] = balanced

    sequence = copy.deepcopy(base)
    sequence["candidate_name"] = "V2_SEQUENCE_STRONG"
    sequence["decoder_v2"].update({
        "v2_evidence_model_bonus": 0.14,
        "generic_seed_consistency_bonus": 0.03,
        "local_evidence_weight": 0.48,
        "duration_prior_weight": 0.27,
        "phase_consistency_weight": 0.25,
        "backward_terminal_weight": 0.34,
    })
    sequence["confidence_v2"].update({"high_threshold": 0.78, "medium_threshold": 0.55, "low_threshold": 0.35})
    variants[sequence["candidate_name"]] = sequence
    for name, value in variants.items():
        value["protocol"] = {
            "parameter_source": "DEVELOPMENT_ONLY",
            "validation_used": False,
            "heldout_used": False,
            "episode_specific_parameters": False,
            "approved_episode49_reference_used": False,
        }
        value["config_hash"] = canonical_hash({key: row for key, row in value.items() if key != "config_hash"})
    return variants


def taxonomy_markdown(payload: dict[str, Any]) -> str:
    rows = "\n".join(f"| {name} | {count} |" for name, count in payload["category_counts"].items())
    return f"""# Development-only v1 failure taxonomy\n\nOnly frozen DEVELOPMENT episodes were inspected. No episode-specific fix or frame was created.\n\n| Global pattern | Count |\n|---|---:|\n{rows}\n"""


def confidence_diagnostic(results: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for config_name, payload in results.items():
        for episode in payload["episodes"]:
            for event_name in REQUIRED_EVENTS:
                confidence = float(episode[f"{event_name}_confidence"])
                classification = episode[f"{event_name}_class"]
                rows.append({"config": config_name, "event": event_name, "confidence": confidence, "class": classification})
    bins = np.linspace(0.0, 1.0, 11)
    histogram = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = [row for row in rows if lower <= row["confidence"] < upper or (upper == 1.0 and row["confidence"] == 1.0)]
        histogram.append({
            "lower": float(lower), "upper": float(upper), "count": len(selected),
            "high_medium_fraction": float(np.mean([row["class"] in ("HIGH", "MEDIUM") for row in selected])) if selected else None,
        })
    return {
        "calibration_scope": "DEVELOPMENT_ONLY",
        "method": "evidence breadth + dwell stability + local/global alternative separation; no confidence-only relabeling",
        "thresholds_are_part_of_each_predeclared_config": True,
        "development_histogram": histogram,
        "validation_calibration_pending": True,
    }


def main() -> int:
    split = load_split()
    development = [int(value) for value in split["development"]]
    validation = set(int(value) for value in split["validation"])
    heldout = set(int(value) for value in split["heldout_test"])
    ledger = AccessLedger("DEVELOPMENT_TUNING", set(development), validation | heldout)
    taxonomy = development_taxonomy(development, split)
    atomic_json(OUT / "development_failure_taxonomy.json", taxonomy)
    (OUT / "development_failure_taxonomy.md").write_text(taxonomy_markdown(taxonomy), encoding="utf-8")
    priors = duration_priors(development)
    configs = candidate_configs(priors)
    atomic_json(OUT / "detector_v2_config_candidates.json", {
        "status": "PREDECLARED_BEFORE_VALIDATION",
        "created_from": "DEVELOPMENT_ONLY",
        "candidate_count": len(configs),
        "configs": configs,
    })

    manifest = {row["episode_id"]: row for row in read_manifest()}
    inputs = {}
    for episode_id in development:
        inputs[episode_id] = load_episode_inputs(manifest[episode_id], ledger, "v2 development diagnostics")
    results: dict[str, Any] = {}
    for config_name, config in configs.items():
        summaries = []
        for episode_id in development:
            loaded, fk, task_geometry = inputs[episode_id]
            timeline = detect_loaded(loaded, fk, task_geometry, config)
            summaries.append(timeline_summary(episode_id, timeline, manifest[episode_id]))
        results[config_name] = {
            "config_hash": config["config_hash"],
            "scope": "DEVELOPMENT_ONLY",
            "aggregate": aggregate_status(summaries),
            "episodes": summaries,
        }
    atomic_json(OUT / "development_config_results.json", results)
    atomic_json(OUT / "confidence_calibration_v2.json", confidence_diagnostic(results))
    access = ledger.payload()
    access["heldout_access_prohibition_pass"] = not any(value in heldout for value in access["accessed_ids"])
    access["validation_access_prohibition_pass"] = not any(value in validation for value in access["accessed_ids"])
    atomic_json(OUT / "development_access_log.json", access)
    print(json.dumps({
        "development": development,
        "taxonomy": taxonomy["category_counts"],
        "configs": {name: value["aggregate"] for name, value in results.items()},
        "heldout_access_count": 0,
        "validation_access_count": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
