#!/usr/bin/env python3
"""Preserve and reclassify prior Policy-B execution-layer evidence.

This tool is deliberately read-only with respect to every earlier output tree.
It writes a hash-addressed inventory into the new bounded official-async/PAINT
investigation root and authorizes neither Policy A/B hardware execution nor DDS.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/policy_execution_stability_review"
OUTPUT = ROOT / "outputs/policy_execution_stability_official_async_paint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    final_manifest_path = SOURCE / "FINAL_POLICY_EXECUTION_STABILITY_REVIEW_MANIFEST.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    if final_manifest["status"] != "BLOCKED_BY_PERSISTENT_POLICY_JITTER":
        raise RuntimeError("prior stability review does not retain the blocked status")
    prior = {
        "final_manifest": record(final_manifest_path),
        "final_report": record(SOURCE / "final_report.md"),
        "fixed_input_stochasticity": record(
            SOURCE / "fixed_input_stochasticity/fixed_observation_stochasticity.json"
        ),
        "fixed_input_samples": record(
            SOURCE / "fixed_input_stochasticity/fixed_observation_inference_samples.npz"
        ),
        "low_motion_reference": record(
            SOURCE / "low_motion_reference/low_motion_reference.json"
        ),
        "m2_reclassification": record(SOURCE / "m2_reclassification.json"),
        "rtc_audit": record(SOURCE / "rtc_audit/rtc_configuration_audit.json"),
        "ruckig_config": record(
            SOURCE / "jerk_limited_otg/common_g1_28d_ruckig_config.json"
        ),
        "best_prior_candidate": record(
            SOURCE / "best_diagnostic_candidate/common_causal_execution_candidate.json"
        ),
        "analysis": record(SOURCE / "analysis/execution_stability_metrics.json"),
    }
    comparison_videos = {
        path.stem: record(path)
        for path in sorted((SOURCE / "comparison_videos_final").glob("*.mp4"))
    }
    if len(comparison_videos) != 4:
        raise RuntimeError("expected four preserved prior equal-scale comparison videos")

    reclassification = {
        "schema_version": "policy_execution_adapter_reclassification_v1",
        "classification": "DIAGNOSTIC_ONLY_NOT_REAL_ROBOT_APPROVED",
        "applies_to": [
            "M0_NAIVE_H4",
            "M1_PREVIOUS_RTC_H4_D0_G5_EXP",
            "M2_MINIMUM_JERK_CROSSFADE",
            "BOUNDED_COMMITMENT_H4_H8_H12_H16",
            "BOUNDED_OFFICIAL_STYLE_RTC_H8_H10_H12",
            "STATEFUL_RTC_H10_D9_PLUS_RUCKIG",
        ],
        "selected_execution_adapter": None,
        "raw_policy_outputs_preserved": True,
        "prior_outputs_modified": False,
        "real_hardware_safety_readiness": "BLOCKED",
        "real_robot_command_allowed": False,
        "policy_a_or_b_hardware_execution_started": False,
    }
    atomic_json(OUTPUT / "current_adapter_reclassification.json", reclassification)
    manifest = {
        "schema_version": "official_async_paint_preservation_manifest_v1",
        "source_root": str(SOURCE.resolve()),
        "new_output_root": str(OUTPUT.resolve()),
        "prior_evidence": prior,
        "prior_equal_scale_comparison_videos": comparison_videos,
        "reclassification": record(OUTPUT / "current_adapter_reclassification.json"),
        "all_prior_execution_adapters": "DIAGNOSTIC_ONLY_NOT_REAL_ROBOT_APPROVED",
        "prior_outputs_modified": False,
        "dataset_b_modified": False,
        "policy_b_modified_or_retrained": False,
        "camera_modified": False,
        "retargeted_action_labels_modified": False,
        "policy_a_touched": False,
        "real_hardware_safety_readiness": "BLOCKED",
        "real_robot_command_allowed": False,
    }
    atomic_json(OUTPUT / "preservation_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PRESERVED_AND_RECLASSIFIED",
                "output": str(OUTPUT),
                "manifest_sha256": sha256_file(OUTPUT / "preservation_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
