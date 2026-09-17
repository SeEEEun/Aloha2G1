#!/usr/bin/env python3
"""Freeze the reviewed common natural-arm block without freezing A/B methods."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tools.doll_handoff_retargeting.common import (  # noqa: E402
    COMMON_TEMPLATE,
    atomic_json,
    load_json,
    sha256_file,
)


AUDIT_ROOT = REPOSITORY / "outputs/doll_handoff_retargeting/natural_arm_audit"
REVIEW = AUDIT_ROOT / "final_review/natural_arm_resolver_report.json"
CONTACTS = AUDIT_ROOT / "final_contact_classification.json"
CONTACT_CSV = AUDIT_ROOT / "final_contact_classification.csv"
CONTACT_MD = AUDIT_ROOT / "final_contact_classification.md"
OUTPUT = AUDIT_ROOT / "frozen_common_natural_arm"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    common = load_json(COMMON_TEMPLATE)
    report = load_json(REVIEW)
    contacts = load_json(CONTACTS)
    natural = common["natural_arm_redundancy"]
    if natural != report["method"]["config"]:
        raise RuntimeError("active natural-arm block differs from reviewed candidate")
    required_gates = (
        "cartesian_target_hashes_identical",
        "after_ik_at_least_95_percent",
        "after_joint_limits_zero",
        "after_branch_discontinuities_zero",
        "mean_primary_error_degradation_below_0_1_mm",
        "handoff_order_preserved",
        "release_geometry_preserved_inside_bin",
    )
    failures = [name for name in required_gates if not report["gates"].get(name)]
    if failures:
        raise RuntimeError(f"natural-arm freeze gates failed: {failures}")
    if contacts["invalid_proximal_or_palm_frame_count"] > 1:
        raise RuntimeError("more than one proximal/palm invalid smoke frame remains")
    if contacts["benign_proximity_reclassified_from_penetration_count"] != 0:
        raise RuntimeError("penetrating contact was incorrectly declared benign")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_config = {
        "schema_version": "common_natural_arm_solver_v1",
        "status": "COMMON_NATURAL_ARM_SOLVER_FROZEN",
        "scope": "COMMON_BASELINE_AND_PROPOSED",
        "natural_arm_redundancy": natural,
        "shared_temporal_ik": common["shared_temporal_ik"],
        "validation": {
            key: common["validation"][key]
            for key in (
                "maximum_joint_step_rad",
                "maximum_velocity_rad_s",
                "maximum_acceleration_rad_s2",
                "branch_absolute_step_norm_rad",
                "branch_local_multiplier",
                "collision_penetration_tolerance_m",
            )
        },
        "primary_cartesian_target_mutation_allowed": False,
        "episode_specific_parameters": False,
        "phase_specific_parameters": False,
    }
    config_path = OUTPUT / "common_natural_arm_solver.json"
    atomic_json(config_path, frozen_config)
    manifest = {
        "schema_version": "common_natural_arm_solver_freeze_v1",
        "status": "COMMON_NATURAL_ARM_SOLVER_FROZEN",
        "scope": "COMMON_BASELINE_AND_PROPOSED",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "natural_arm_block_canonical_sha256": canonical_sha256(natural),
        "active_common_template": str(COMMON_TEMPLATE),
        "active_common_template_sha256_at_freeze": sha256_file(COMMON_TEMPLATE),
        "review_report": str(REVIEW),
        "review_report_sha256": sha256_file(REVIEW),
        "contact_classification_csv": str(CONTACT_CSV),
        "contact_classification_csv_sha256": sha256_file(CONTACT_CSV),
        "contact_classification_markdown": str(CONTACT_MD),
        "contact_classification_markdown_sha256": sha256_file(CONTACT_MD),
        "contact_summary": contacts,
        "smoke_metrics": {
            episode: {
                "ik_success_rate": value["after"]["ik_success_rate"],
                "mean_primary_task_error_m": value["after"][
                    "primary_task_position_error_mean_m"
                ],
                "joint_limit_violations": value["after"]["joint_limit_violations"],
                "branch_discontinuities": value["after"]["branch_discontinuities"],
                "cartesian_target_sha256": value["after"]["cartesian_target_sha256"],
            }
            for episode, value in report["episodes"].items()
        },
        "acceptance": {
            "primary_task_targets_unchanged": True,
            "joint_limits_zero": True,
            "branch_discontinuities_zero": True,
            "material_primary_error_increase": False,
            "catastrophic_invalid_self_collision": False,
            "visual_posture_improved": True,
            "residual_distal_contacts_declared_solved": False,
            "residual_one_frame_shoulder_torso_contact_declared_solved": False,
        },
        "note": (
            "This freezes only the common redundant-joint realization. It does not "
            "approve Proposed-B distal handoff contacts, freeze Baseline-A mapping, "
            "or authorize the 50-episode batch."
        ),
    }
    manifest_path = OUTPUT / "freeze_manifest.json"
    atomic_json(manifest_path, manifest)
    sentinel = OUTPUT / "COMMON_NATURAL_ARM_SOLVER_FROZEN"
    sentinel.write_text(
        "COMMON_NATURAL_ARM_SOLVER_FROZEN\n"
        f"config: {config_path}\n"
        f"SHA256: {manifest['config_sha256']}\n"
        f"metrics: {REVIEW}\n"
        f"contact classification: {CONTACT_MD}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
