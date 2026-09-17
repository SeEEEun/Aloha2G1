#!/usr/bin/env python3
"""Freeze the human-approved Proposed-B smoke candidate for a B-only batch."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    atomic_json,
    implementation_fingerprint,
    load_json,
    sha256_file,
)


CANDIDATE = REPOSITORY / "outputs/doll_handoff_retargeting/ab_smoke_final_candidate"
BATCH_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21"
FREEZE = BATCH_ROOT / "frozen_approval"
SMOKE_EPISODES = (0, 24, 49)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def copy_with_record(label: str, source: Path) -> dict[str, str]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = FREEZE / label
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    copied_hash = sha256_file(destination)
    if source_hash != copied_hash:
        raise RuntimeError(f"freeze copy checksum mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "source_sha256": source_hash,
        "snapshot": str(destination),
        "snapshot_sha256": copied_hash,
    }


def main() -> int:
    common_path = CANDIDATE / "config/common_config.json"
    proposed_path = CANDIDATE / "config/proposed_config.json"
    common = load_json(common_path)
    proposed = load_json(proposed_path)
    source_manifest = load_json(CANDIDATE / "source_audit/source_manifest.json")
    if int(source_manifest["enumerated_count"]) != 50:
        raise RuntimeError("approved source manifest does not contain exactly 50 episodes")
    if int(source_manifest["invalid_count"]) != 0:
        raise RuntimeError("approved source manifest contains invalid episodes")
    residual = proposed["handoff_cartesian_residual"]
    if residual["status"] != "REMOVED_TASK_SPECIFIC_CARTESIAN_RESIDUAL":
        raise RuntimeError("handoff Cartesian residual removal is not active")
    if float(residual["active_offset_m"]) != 0.0:
        raise RuntimeError("handoff Cartesian residual is non-zero")
    if proposed["resolved"]["interaction_and_ownership_semantics"][
        "episode_specific_parameters"
    ]:
        raise RuntimeError("episode-specific Proposed-B parameters are active")
    if proposed["resolved"]["interaction_and_ownership_semantics"][
        "frame_specific_offsets"
    ]:
        raise RuntimeError("frame-specific Proposed-B offsets are active")
    if common["natural_arm_redundancy"]["scope"] != "COMMON_BASELINE_AND_PROPOSED":
        raise RuntimeError("natural-arm resolver is not marked common")
    if common["natural_arm_redundancy"]["primary_target_mutation_allowed"]:
        raise RuntimeError("natural-arm resolver may mutate primary targets")

    current_implementation, implementation_files = implementation_fingerprint()
    smoke: dict[str, Any] = {}
    for episode in SMOKE_EPISODES:
        stable = f"doll_handoff_20260820_ep{episode:03d}"
        episode_manifest = load_json(
            CANDIDATE / "proposed/metrics" / f"{stable}.manifest.json"
        )
        if episode_manifest["implementation_sha256"] != current_implementation:
            raise RuntimeError(f"implementation drift from approved ep{episode:03d}")
        smoke[f"ep{episode:03d}"] = {
            "status": episode_manifest["status"],
            "cartesian_target_sha256": episode_manifest["cartesian_target_sha256"],
            "trajectory_sha256": episode_manifest["trajectory_sha256"],
            "metrics_sha256": episode_manifest["metrics_sha256"],
            "runtime_common_config_sha256": episode_manifest["common_config_sha256"],
            "runtime_proposed_config_sha256": episode_manifest["config_sha256"],
        }

    protected_common = {
        key: common[key]
        for key in (
            "models",
            "task_registration",
            "source_channels",
            "event_detector",
            "canonical_task_ready_posture",
            "shared_temporal_ik",
            "natural_arm_redundancy",
            "validation",
        )
    }
    files = {
        "config/common_config.json": common_path,
        "config/proposed_config.json": proposed_path,
        "config/task_frame_report.json": CANDIDATE / "config/task_frame_report.json",
        "config/tool_frame_report.json": CANDIDATE / "config/tool_frame_report.json",
        "config/model_unit_audit.json": CANDIDATE / "config/model_unit_audit.json",
        "config/event_detector_config.json": CANDIDATE
        / "event_audit/detector_config.json",
        "config/dex3_whole_hand.sim.json": Path(
            common["models"]["dex3_whole_hand_geometry"]
        ),
        "config/common_natural_arm_solver.json": REPOSITORY
        / "outputs/doll_handoff_retargeting/natural_arm_audit/frozen_common_natural_arm/common_natural_arm_solver.json",
        "scene/scene_layout.json": Path(common["scene_config"]),
        "source/source_manifest.json": CANDIDATE / "source_audit/source_manifest.json",
        "review/smoke_ab_audit.json": CANDIDATE / "comparison/smoke_ab_audit.json",
        "review/final_contact_classification.csv": REPOSITORY
        / "outputs/doll_handoff_retargeting/natural_arm_audit/final_contact_classification.csv",
        "docs/DOLL_HANDOFF_RESEARCH_OVERRIDE_2026-08-21.md": REPOSITORY
        / "docs/DOLL_HANDOFF_RESEARCH_OVERRIDE_2026-08-21.md",
    }
    snapshots = {label: copy_with_record(label, path) for label, path in files.items()}
    manifest = {
        "schema_version": "approved_interaction_centric_proposed_b_batch_freeze_v1",
        "status": "PROPOSED_B_APPROVED_FOR_50_EPISODE_BATCH",
        "approval_basis": "explicit user approval of the current visually reviewed Proposed-B candidate",
        "approved_candidate": str(CANDIDATE),
        "batch_output_root": str(BATCH_ROOT),
        "scope": "PROPOSED_B_ONLY_50_EPISODE_REVIEW",
        "source_count": 50,
        "source_invalid_count": 0,
        "implementation_sha256": current_implementation,
        "implementation_files": implementation_files,
        "approved_resolved_file_sha256": {
            "common_config": sha256_file(common_path),
            "proposed_config": sha256_file(proposed_path),
            "task_frame_report": sha256_file(CANDIDATE / "config/task_frame_report.json"),
            "tool_frame_report": sha256_file(CANDIDATE / "config/tool_frame_report.json"),
            "scene_layout": sha256_file(Path(common["scene_config"])),
        },
        "protected_semantic_hashes": {
            "common_solver_and_registration": canonical_hash(protected_common),
            "proposed_interaction_representation": canonical_hash(proposed),
            "task_frame_report": canonical_hash(
                load_json(CANDIDATE / "config/task_frame_report.json")
            ),
            "tool_frame_report": canonical_hash(
                load_json(CANDIDATE / "config/tool_frame_report.json")
            ),
            "event_detector_config": canonical_hash(
                load_json(CANDIDATE / "event_audit/detector_config.json")
            ),
        },
        "protected_constraints": {
            "scene_changes_allowed": False,
            "task_frame_registration_changes_allowed": False,
            "natural_arm_solver_changes_allowed": False,
            "whole_hand_grasp_frame_changes_allowed": False,
            "dex3_synergy_changes_allowed": False,
            "ownership_semantics_changes_allowed": False,
            "episode_specific_parameters_allowed": False,
            "handoff_cartesian_residual_m": 0.0,
            "baseline_execution_allowed": False,
            "dataset_packaging_allowed": False,
            "policy_training_allowed": False,
        },
        "approved_smoke": smoke,
        "snapshot_files": snapshots,
        "batch_command": (
            "/home/jbnu/miniconda3/envs/isaaclab6/bin/python "
            "tools/retarget_doll_handoff_batch.py --method proposed --episodes all "
            "--output-root outputs/doll_handoff_retargeting/"
            "proposed_b_50_review_2026-08-21"
        ),
    }
    atomic_json(FREEZE / "freeze_manifest.json", manifest)
    manifest_hash = sha256_file(FREEZE / "freeze_manifest.json")
    (FREEZE / "PROPOSED_B_APPROVED_AND_FROZEN").write_text(
        "PROPOSED_B_APPROVED_AND_FROZEN\n"
        f"manifest: {FREEZE / 'freeze_manifest.json'}\n"
        f"manifest_sha256: {manifest_hash}\n"
        f"implementation_sha256: {current_implementation}\n"
        "handoff_cartesian_residual_m: 0.0\n"
        "batch_scope: PROPOSED_B_ONLY\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(FREEZE / "freeze_manifest.json"),
                "manifest_sha256": manifest_hash,
                "implementation_sha256": current_implementation,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
