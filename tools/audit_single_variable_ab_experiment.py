#!/usr/bin/env python3
"""Read-only audit of whether the existing ACT A/B assets isolate representation.

This tool intentionally does not retarget, train, run physics, or rewrite any
scientific artifact.  It compares the actual frozen inputs and emits the reset
reports needed before a replacement experiment can be built.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/single_variable_ab_reset"
COMMON48 = ROOT / "outputs/paper_core_ab/common48_manifest.json"
TRAIN40 = ROOT / "outputs/paper_core_ab/train40_manifest.json"
HELDOUT8 = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
PACKAGING = ROOT / "outputs/paper_core_ab/dataset_packaging_audit.json"
TRAINING_CONTRACT = ROOT / "outputs/paper_core_ab/act_a_b_training_contract.json"
TRAINING_AUDIT = ROOT / "outputs/paper_core_ab/act_a_b_training_audit.json"
A_TRAIN_CONFIG = ROOT / "outputs/paper_core_ab/act_a40/config/train_config.json"
B_TRAIN_CONFIG = ROOT / "outputs/paper_core_ab/act_b40/config/train_config.json"
A_COMMON = ROOT / "outputs/dataset_a_final50_retargeting/config/common_config.json"
B_COMMON = ROOT / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21/config/common_config.json"
A_METHOD = ROOT / "outputs/dataset_a_final50_retargeting/config/baseline_config.json"
B_METHOD = ROOT / "outputs/doll_handoff_retargeting/proposed_b_50_review_2026-08-21/config/proposed_config.json"
REGISTRATION = ROOT / "outputs/final_episode_registered_eval35/00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
CONTACT_MODEL = ROOT / "outputs/final_episode_registered_eval35/01_freeze/FINAL_DOLL_CONTACT_MODEL.json"
ZERO_CONTACT = ROOT / "outputs/final_episode_registered_eval35/00_forensic_audit/dex3_zero_contact_solver80/DEX3_ZERO_CONTACT_ARTICULATION_AUDIT.json"
B03_RESULT = ROOT / "outputs/final_episode_registered_eval35/03_act_b_results/rollouts/eval_02_doll_handoff_20260820_ep023/EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten(child, name))
        return output
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def config_differences(a: dict[str, Any], b: dict[str, Any], ignored: set[str]) -> list[str]:
    fa, fb = flatten(a), flatten(b)
    return sorted(
        key
        for key in set(fa) | set(fb)
        if key not in ignored and fa.get(key) != fb.get(key)
    )


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "maximum": max(values) if values else None,
    }


def scalar_text(value: float | None, scale: float = 1.0, digits: int = 3) -> str:
    return "NA" if value is None else f"{value * scale:.{digits}f}"


def trajectory_pair_audit(entries: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    counts = {
        "same_frame_count": 0,
        "same_timestamps": 0,
        "same_source_frame_index": 0,
        "same_joint_order": 0,
        "same_event_names": 0,
        "same_event_frames": 0,
        "same_source_interaction_positions": 0,
        "same_left_hand_phase_labels": 0,
        "same_right_hand_phase_labels": 0,
        "same_dex3_commands": 0,
        "same_common_config_sha256": 0,
        "same_implementation_sha256": 0,
    }
    total_dex3_different = 0
    maximum_dex3_difference = 0.0
    for row in entries:
        a_path = Path(row["a_trajectory_path"])
        b_path = Path(row["b_trajectory_path"])
        with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
            def both(key: str) -> bool:
                return key in a.files and key in b.files and array_equal(a[key], b[key])

            checks = {
                "same_frame_count": int(a["g1_arm_qpos"].shape[0]) == int(b["g1_arm_qpos"].shape[0]),
                "same_timestamps": both("timestamp"),
                "same_source_frame_index": both("source_frame_index"),
                "same_joint_order": (
                    both("g1_arm_joint_names")
                    and both("left_dex3_joint_names")
                    and both("right_dex3_joint_names")
                    and both("replay_joint_names")
                ),
                "same_event_names": both("event_names"),
                "same_event_frames": both("event_frames"),
                "same_source_interaction_positions": (
                    both("target_left_interaction_frame_position_world")
                    and both("target_right_interaction_frame_position_world")
                ),
                "same_left_hand_phase_labels": both("left_hand_phase"),
                "same_right_hand_phase_labels": both("right_hand_phase"),
                "same_common_config_sha256": str(a["common_config_sha256"].item()) == str(b["common_config_sha256"].item()),
                "same_implementation_sha256": str(a["implementation_sha256"].item()) == str(b["implementation_sha256"].item()),
            }
            a_hands = np.concatenate((a["left_dex3_qpos"], a["right_dex3_qpos"]), axis=1)
            b_hands = np.concatenate((b["left_dex3_qpos"], b["right_dex3_qpos"]), axis=1)
            dex3_diff = np.abs(a_hands.astype(np.float64) - b_hands.astype(np.float64))
            checks["same_dex3_commands"] = bool(np.array_equal(a_hands, b_hands))
            different = int(np.count_nonzero(dex3_diff > 1e-12))
            maximum = float(np.max(dex3_diff))
            total_dex3_different += different
            maximum_dex3_difference = max(maximum_dex3_difference, maximum)
            for key, passed in checks.items():
                counts[key] += int(passed)
            results.append(
                {
                    "stable_episode_id": row["stable_episode_id"],
                    **checks,
                    "dex3_different_scalar_count": different,
                    "maximum_dex3_difference_rad": maximum,
                    "a_common_config_sha256": str(a["common_config_sha256"].item()),
                    "b_common_config_sha256": str(b["common_config_sha256"].item()),
                    "a_implementation_sha256": str(a["implementation_sha256"].item()),
                    "b_implementation_sha256": str(b["implementation_sha256"].item()),
                }
            )
    return {
        "episode_count": len(entries),
        "pass_counts": counts,
        "dex3_different_scalar_count": total_dex3_different,
        "maximum_dex3_difference_rad": maximum_dex3_difference,
        "episodes": results,
    }


def first_existing(archive: Any, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in archive.files:
            return np.asarray(archive[key], dtype=np.float64)
    return None


def reference_audit(registration: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    a_wrist_tracking: list[float] = []
    b_interaction_tracking: list[float] = []
    source_object: list[float] = []
    a_grasp_object: list[float] = []
    b_grasp_object: list[float] = []
    for entry in registration["entries"]:
        object_position = np.asarray(entry["target_object_pose"]["position_xyz_m"], dtype=np.float64)
        record: dict[str, Any] = {
            "eval_number": int(entry["eval_number"]),
            "stable_episode_id": entry["stable_episode_id"],
            "source_recording": entry["source_recording"],
            "object_position_xyz_m": object_position.tolist(),
        }
        for label, method in (("a", "ACT-A40"), ("b", "ACT-B40")):
            info = entry["methods"][method]["conversion_registration"]
            archive_path = Path(info["artifact"])
            frame = int(info["left_grasp_event_frame"])
            with np.load(archive_path, allow_pickle=False) as archive:
                grasp = first_existing(
                    archive,
                    (
                        "achieved_left_physical_grasp_frame_position_world",
                        "achieved_left_static_whole_hand_position_world",
                        "achieved_left_grasp_frame_position_world",
                    ),
                )
                if grasp is None:
                    raise RuntimeError(f"no achieved physical grasp-frame array: {archive_path}")
                wrist = first_existing(archive, ("achieved_left_wrist_position_world",))
                source_frame = first_existing(
                    archive,
                    (
                        "source_left_realization_frame_position_world",
                        "source_left_interaction_frame_position_world",
                        "target_left_interaction_frame_position_world",
                    ),
                )
                achieved_frame = first_existing(
                    archive,
                    (
                        "achieved_left_realization_frame_position_world",
                        "achieved_left_static_whole_hand_position_world",
                    ),
                )
                source_interaction = first_existing(
                    archive, ("target_left_interaction_frame_position_world",)
                )
                frame = min(frame, len(grasp) - 1)
                grasp_distance = float(np.linalg.norm(grasp[frame] - object_position))
                source_distance = (
                    float(np.linalg.norm(source_interaction[frame] - object_position))
                    if source_interaction is not None
                    else None
                )
                tracking = (
                    float(np.linalg.norm(achieved_frame[frame] - source_frame[frame]))
                    if source_frame is not None and achieved_frame is not None
                    else None
                )
                record[f"{label}_reference_grasp_frame_xyz_m"] = grasp[frame].tolist()
                record[f"{label}_reference_grasp_frame_to_object_center_m"] = grasp_distance
                record[f"{label}_reference_wrist_to_object_center_m"] = (
                    float(np.linalg.norm(wrist[frame] - object_position)) if wrist is not None else None
                )
                record[f"{label}_reference_tracking_error_m"] = tracking
                record[f"{label}_source_interaction_to_object_center_m"] = source_distance
                (a_grasp_object if label == "a" else b_grasp_object).append(grasp_distance)
                if source_distance is not None:
                    source_object.append(source_distance)
                if tracking is not None:
                    (a_wrist_tracking if label == "a" else b_interaction_tracking).append(tracking)
        rows.append(record)
    return {
        "episode_count": len(rows),
        "a_wrist_reference_tracking_error_m": stats(a_wrist_tracking),
        "b_interaction_reference_tracking_error_m": stats(b_interaction_tracking),
        "source_interaction_to_registered_object_center_m": stats(source_object),
        "a_physical_grasp_frame_to_registered_object_center_m": stats(a_grasp_object),
        "b_physical_grasp_frame_to_registered_object_center_m": stats(b_grasp_object),
        "a_reference_competent": False,
        "a_reference_competence_reason": (
            "The archived A builder accurately tracks its own wrist-origin target, but it maps "
            "the ALOHA link-6 wrist origin directly to the G1 wrist origin and omits the fixed "
            "source TCP/tool-to-G1-wrist compatibility transform required by the reset contract. "
            "Consequently the realized G1 physical grasp frame is not task-consistent."
        ),
        "b_reference_competence_not_final": True,
        "b_reference_competence_reason": (
            "The archived interaction reference is substantially closer to the registered task, "
            "but it was generated with method-dependent timing/hand realization and is therefore "
            "not evidence from the requested single-variable pipeline."
        ),
        "episodes": rows,
    }


def render_reference_sheet(path: Path, audit: dict[str, Any], contact: dict[str, Any]) -> None:
    collision = np.asarray(contact["collision_dimensions_m"], dtype=np.float64)
    fig, axes = plt.subplots(5, 7, figsize=(14, 10), constrained_layout=True)
    for ax, row in zip(axes.flat, audit["episodes"]):
        obj = np.asarray(row["object_position_xyz_m"], dtype=np.float64)
        a = np.asarray(row["a_reference_grasp_frame_xyz_m"], dtype=np.float64) - obj
        b = np.asarray(row["b_reference_grasp_frame_xyz_m"], dtype=np.float64) - obj
        ax.add_patch(
            Ellipse((0, 0), collision[0] * 1000, collision[1] * 1000,
                    facecolor="#dddddd", edgecolor="#333333", linewidth=0.8)
        )
        ax.scatter(a[0] * 1000, a[1] * 1000, color="#ca3b3b", s=13, label="A" if row["eval_number"] == 1 else None)
        ax.scatter(b[0] * 1000, b[1] * 1000, color="#2468b4", s=13, label="B" if row["eval_number"] == 1 else None)
        ax.plot([0, a[0] * 1000], [0, a[1] * 1000], color="#ca3b3b", alpha=0.45, lw=0.6)
        ax.plot([0, b[0] * 1000], [0, b[1] * 1000], color="#2468b4", alpha=0.45, lw=0.6)
        ax.set_title(f"{row['eval_number']:02d}", fontsize=8)
        ax.set_xlim(-260, 260)
        ax.set_ylim(-260, 260)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Archived reference grasp frames relative to episode-registered doll (top view)\n"
                 "red: WRIST baseline physical grasp frame; blue: INTERACTION reference; gray: collider", fontsize=12)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    common48 = read_json(COMMON48)
    train40 = read_json(TRAIN40)
    heldout8 = read_json(HELDOUT8)
    packaging = read_json(PACKAGING)
    training_contract = read_json(TRAINING_CONTRACT)
    training_audit = read_json(TRAINING_AUDIT)
    registration = read_json(REGISTRATION)
    contact = read_json(CONTACT_MODEL)
    zero_contact = read_json(ZERO_CONTACT)
    b03 = read_json(B03_RESULT)

    pairs = trajectory_pair_audit(common48["entries"])
    a_train = read_json(A_TRAIN_CONFIG)
    b_train = read_json(B_TRAIN_CONFIG)
    allowed_training = {"dataset.repo_id", "dataset.root", "job_name", "output_dir"}
    training_diffs = config_differences(a_train, b_train, allowed_training)
    common_diffs = config_differences(read_json(A_COMMON), read_json(B_COMMON), set())
    reference = reference_audit(registration)
    write_json(out / "REFERENCE_A_B_SANITY_REPORT.json", reference)
    render_reference_sheet(out / "REFERENCE_A_B_SANITY_CONTACT_SHEET.png", reference, contact)

    registration_checks = {
        "entries": registration.get("EVAL35_count"),
        "source_derived": registration.get("source_derived_count"),
        "matched_identical": registration.get("A_B_identical_object_pose_count"),
        "unique_object_poses": registration.get("unique_episode_object_pose_count"),
        "one_global_canonical_pose": registration.get("one_global_canonical_object_pose"),
        "manual_nudges": registration.get("manual_episode_nudges"),
        "policy_output_derived": registration.get("policy_output_derived_object_placement"),
        "maximum_translation_difference_mm": registration.get("maximum_A_B_translation_difference_mm"),
        "maximum_rotation_difference_deg": registration.get("maximum_A_B_rotation_difference_deg"),
    }

    b03_integrity = b03["integrity"]
    b03_runtime = b03_integrity["runtime_summary"]
    articulation = {
        "zero_contact_status": zero_contact.get("status"),
        "zero_contact_mapping_pass": zero_contact.get("summary", {}).get("mapping_pass_count") == 14,
        "zero_contact_sign_pass": zero_contact.get("summary", {}).get("sign_pass_count") == 14,
        "zero_contact_readback_pass": zero_contact.get("summary", {}).get("readback_pass_count") == 14,
        "zero_contact_hard_limits_pass": zero_contact.get("summary", {}).get("runtime_hard_limit_pass_count") == 14,
        "loaded_representative_status": b03.get("status"),
        "loaded_measured_dex3_violation_count": b03_integrity.get("measured_dex3_hard_limit_violation_scalar_count"),
        "loaded_failure_joint": "left_hand_middle_1_joint",
        "loaded_failure_measured_max_rad": 0.0019864142,
        "loaded_failure_authoritative_max_rad": 0.0,
        "loaded_failure_command_rad": -0.2364969326,
        "loaded_failure_doll_contact": "NONE",
        "overall_pass": False,
        "reason": (
            "The isolated sweep passes 14/14, but the representative loaded B03 trace "
            "contains four measured hard-limit violations with a safe negative command and no "
            "doll-digit contact. Loaded articulation validity is therefore not established."
        ),
        "runtime_freeze_sha256": b03_runtime.get("direct_execution_freeze_sha256"),
    }

    confounds = [
        {
            "id": "METHOD_DEPENDENT_DEX3_MAPPING",
            "classification": "UNINTENDED_CONFOUND",
            "evidence": (
                "A archives use binary OPEN/CLOSED hand phases; B archives use semantic "
                "OPEN/PRESHAPE/GRASP/HOLD/RELEASE phases and different Dex3 commands."
            ),
        },
        {
            "id": "DIFFERENT_COMMON_EVENT_CONFIG_REVISIONS",
            "classification": "UNINTENDED_CONFOUND",
            "evidence": (
                f"A and B common config files have different SHA256 values and {len(common_diffs)} "
                "flattened configuration differences, including independently fitted pooled event thresholds."
            ),
        },
        {
            "id": "DIFFERENT_REALIZATION_ADAPTERS_AND_POSTPROCESSING",
            "classification": "UNINTENDED_CONFOUND",
            "evidence": (
                "A was passed through a wrist-frame full-pose repair adapter after its original "
                "conversion; B uses the interaction-frame generic feasibility adapter. The shared "
                "IK backend alone does not make these postprocessing paths identical."
            ),
        },
        {
            "id": "TRAINING_TO_PHYSICS_TASK_REGISTRATION_MISMATCH",
            "classification": "UNINTENDED_CONFOUND",
            "evidence": (
                "The archived training conversions declare a single identity/global task registration "
                "and disallow episode-specific transforms, while the physical evaluator later uses 35 "
                "episode-conditioned object poses."
            ),
        },
        {
            "id": "WRIST_BASELINE_TOOL_FRAME_INCOMPATIBILITY",
            "classification": "UNINTENDED_CONFOUND",
            "evidence": reference["a_reference_competence_reason"],
        },
        {
            "id": "LOADED_DEX3_ARTICULATION_INVALID",
            "classification": "UNINTENDED_CONFOUND",
            "evidence": articulation["reason"],
        },
        {
            "id": "EVAL35_REUSED_DURING_ENGINEERING",
            "classification": "UNINTENDED_CONFOUND",
            "evidence": (
                "EVAL35 outcomes and traces have been repeatedly inspected for registration, contact, "
                "controller, and articulation debugging; it can only be DEV35/DIAGNOSTIC35 now."
            ),
        },
    ]
    intended = [
        {
            "id": "SPATIAL_TARGET_REPRESENTATION",
            "classification": "INTENDED_REPRESENTATION_DIFFERENCE",
            "a": "registered source wrist/TCP SE(3) with fixed coordinate-compatibility transform",
            "b": "registered object/whole-hand/bimanual interaction geometry",
        }
    ]

    audit = {
        "schema_version": "single_variable_ab_dataset_fairness_audit_v1",
        "status": "FAIL_UNINTENDED_CONFOUNDS",
        "read_only": True,
        "physical_rollouts_started": 0,
        "source_pairing": {
            "common48_a_count": common48.get("a_episode_count"),
            "common48_b_count": common48.get("b_episode_count"),
            "train40_count": train40.get("episode_count"),
            "heldout8_count": heldout8.get("episode_count"),
            "packaging_status": packaging.get("status"),
            "same_source_episodes": bool(packaging.get("a_b_train_episode_indices_identical")),
            "same_source_rgb": bool(packaging.get("a_b_source_rgb_exact")),
        },
        "trajectory_pair_audit": pairs,
        "training_parity": {
            "contract_status": training_contract.get("status"),
            "training_audit_status": training_audit.get("status"),
            "config_differences_outside_allowed_paths": training_diffs,
            "architecture_and_procedure_equal": not training_diffs,
            "old_checkpoints_eligible_after_supervision_reset": False,
        },
        "registration": registration_checks,
        "common_config": {
            "a_path": str(A_COMMON),
            "a_sha256": sha256_file(A_COMMON),
            "b_path": str(B_COMMON),
            "b_sha256": sha256_file(B_COMMON),
            "flattened_difference_count": len(common_diffs),
            "difference_paths": common_diffs,
        },
        "intended_differences": intended,
        "unintended_confounds": confounds,
        "unintended_confound_count": len(confounds),
        "reference_report": str(out / "REFERENCE_A_B_SANITY_REPORT.json"),
        "articulation": articulation,
    }
    audit["content_sha256"] = canonical_sha(audit)
    write_json(out / "A_B_DATASET_FAIRNESS_AUDIT.json", audit)
    write_json(out / "DEX3_COMMON_ARTICULATION_AUDIT.json", articulation)

    md = [
        "# A/B Dataset Fairness Audit",
        "",
        "Status: **FAIL — unintended confounds remain**",
        "",
        "This is a read-only audit. No retargeting, training, checkpoint selection, or physical rollout was run.",
        "",
        "## What is already common",
        "",
        f"- Paired COMMON48 sources: {common48.get('a_episode_count')}/{common48.get('b_episode_count')}",
        f"- TRAIN40 and HELDOUT8 membership: {train40.get('episode_count')} / {heldout8.get('episode_count')}",
        f"- Frame counts, timestamps, joint order, source frame indices: {pairs['pass_counts']['same_frame_count']}/48, {pairs['pass_counts']['same_timestamps']}/48, {pairs['pass_counts']['same_joint_order']}/48, {pairs['pass_counts']['same_source_frame_index']}/48",
        f"- Source interaction-position arrays: {pairs['pass_counts']['same_source_interaction_positions']}/48 byte-identical",
        f"- RGB assets and task packaging: {'PASS' if packaging.get('a_b_source_rgb_exact') else 'FAIL'}",
        f"- ACT architecture/training configuration outside allowed dataset/output paths: {'PASS' if not training_diffs else 'FAIL'}",
        "",
        "## Intended scientific difference",
        "",
        "One block only: `representation_mode=WRIST` versus `representation_mode=INTERACTION` in spatial target generation.",
        "",
        "## Unintended confounds",
        "",
    ]
    for index, item in enumerate(confounds, start=1):
        md.append(f"{index}. **{item['id']}** — {item['evidence']}")
    md += [
        "",
        "## Exact archive evidence",
        "",
        f"- Same event frames: {pairs['pass_counts']['same_event_frames']}/48 episodes",
        f"- Same left hand phase labels: {pairs['pass_counts']['same_left_hand_phase_labels']}/48 episodes",
        f"- Same right hand phase labels: {pairs['pass_counts']['same_right_hand_phase_labels']}/48 episodes",
        f"- Same Dex3 commands: {pairs['pass_counts']['same_dex3_commands']}/48 episodes",
        f"- Differing Dex3 scalars: {pairs['dex3_different_scalar_count']:,}",
        f"- Maximum A/B Dex3 command difference: {pairs['maximum_dex3_difference_rad']:.6f} rad",
        f"- Same common-config hash embedded in A/B archives: {pairs['pass_counts']['same_common_config_sha256']}/48",
        f"- Same implementation hash embedded in A/B archives: {pairs['pass_counts']['same_implementation_sha256']}/48",
        "",
        "## Consequence",
        "",
        "Both datasets must be regenerated through one builder after the WRIST reference and common execution are corrected. Because the supervision changes, both ACT policies must then be retrained with the already-parity-checked training procedure. Existing checkpoints cannot be reused for the reset experiment.",
    ]
    (out / "A_B_DATASET_FAIRNESS_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    r = reference
    rmd = f"""# Reference A/B Sanity Report

Status: **A WRIST reference implementation invalid for the requested reset**

The audit uses the 35 registered development episodes and the archived converted
reference trajectories—not ACT predictions and not physical outcomes.

| Diagnostic | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| A wrist target tracking error (mm) | {scalar_text(r['a_wrist_reference_tracking_error_m']['mean'], 1000)} | {scalar_text(r['a_wrist_reference_tracking_error_m']['median'], 1000)} | {scalar_text(r['a_wrist_reference_tracking_error_m']['p95'], 1000)} | {scalar_text(r['a_wrist_reference_tracking_error_m']['maximum'], 1000)} |
| B interaction target tracking error (mm) | {scalar_text(r['b_interaction_reference_tracking_error_m']['mean'], 1000)} | {scalar_text(r['b_interaction_reference_tracking_error_m']['median'], 1000)} | {scalar_text(r['b_interaction_reference_tracking_error_m']['p95'], 1000)} | {scalar_text(r['b_interaction_reference_tracking_error_m']['maximum'], 1000)} |
| Source interaction frame to registered object center (mm) | {scalar_text(r['source_interaction_to_registered_object_center_m']['mean'], 1000)} | {scalar_text(r['source_interaction_to_registered_object_center_m']['median'], 1000)} | {scalar_text(r['source_interaction_to_registered_object_center_m']['p95'], 1000)} | {scalar_text(r['source_interaction_to_registered_object_center_m']['maximum'], 1000)} |
| A realized physical grasp frame to registered object center (mm) | {scalar_text(r['a_physical_grasp_frame_to_registered_object_center_m']['mean'], 1000)} | {scalar_text(r['a_physical_grasp_frame_to_registered_object_center_m']['median'], 1000)} | {scalar_text(r['a_physical_grasp_frame_to_registered_object_center_m']['p95'], 1000)} | {scalar_text(r['a_physical_grasp_frame_to_registered_object_center_m']['maximum'], 1000)} |
| B realized physical grasp frame to registered object center (mm) | {scalar_text(r['b_physical_grasp_frame_to_registered_object_center_m']['mean'], 1000)} | {scalar_text(r['b_physical_grasp_frame_to_registered_object_center_m']['median'], 1000)} | {scalar_text(r['b_physical_grasp_frame_to_registered_object_center_m']['p95'], 1000)} | {scalar_text(r['b_physical_grasp_frame_to_registered_object_center_m']['maximum'], 1000)} |

The archived A implementation tracks the wrist target it constructed, but that
does not establish baseline competence. Its declared mapping is ALOHA link-6
wrist origin directly to G1 wrist origin, followed by pooled morphology scaling.
It does not apply the fixed source TCP/tool-to-G1-wrist compatibility transform
required by the reset. The resulting G1 physical grasp frame is displaced from
the registered task instance. This is a reference-generation bug, not an ACT
generalization result.

The B reference is closer to the task, but it cannot yet be accepted as the
single-variable control because the archived A/B hand timing, Dex3 commands,
config revisions, and realization adapters differ.

See `REFERENCE_A_B_SANITY_CONTACT_SHEET.png` for the deterministic top-view audit.
"""
    (out / "REFERENCE_A_B_SANITY_REPORT.md").write_text(rmd, encoding="utf-8")

    timeline = f"""# Common Temporal Semantics Audit

Status: **FAIL**

- Source frame indices and timestamps match in {pairs['pass_counts']['same_source_frame_index']}/48 and {pairs['pass_counts']['same_timestamps']}/48 archived pairs.
- Event-frame arrays match in {pairs['pass_counts']['same_event_frames']}/48 pairs.
- Left/right hand phase labels match in {pairs['pass_counts']['same_left_hand_phase_labels']}/48 and {pairs['pass_counts']['same_right_hand_phase_labels']}/48 pairs.
- A uses binary OPEN/CLOSED mapping; B uses source-derived semantic states.
- The A and B common configs contain independently fitted event thresholds.

The source chronology can be reused, but the current supervised Dex3 timing is
not common. A replacement builder must extract one timeline before the
representation switch, feed it unchanged to both branches, and generate the
same common Dex3 commands for both.
"""
    (out / "COMMON_TEMPORAL_SEMANTICS_AUDIT.md").write_text(timeline, encoding="utf-8")

    amd = f"""# Common Dex3 Articulation Audit

Status: **FAIL under representative loaded execution**

- Zero-contact mapping: {'PASS' if articulation['zero_contact_mapping_pass'] else 'FAIL'} (14/14)
- Zero-contact sign: {'PASS' if articulation['zero_contact_sign_pass'] else 'FAIL'} (14/14)
- Zero-contact readback: {'PASS' if articulation['zero_contact_readback_pass'] else 'FAIL'} (14/14)
- Zero-contact runtime hard limits: {'PASS' if articulation['zero_contact_hard_limits_pass'] else 'FAIL'} (14/14)
- Loaded B03: `{articulation['loaded_representative_status']}`
- Loaded measured Dex3 limit violations: {articulation['loaded_measured_dex3_violation_count']}
- Joint: `{articulation['loaded_failure_joint']}`
- Safe command: {articulation['loaded_failure_command_rad']:.10f} rad
- Measured maximum: +{articulation['loaded_failure_measured_max_rad']:.10f} rad; authoritative maximum 0 rad
- Doll contact at excursion: NONE

Passing an isolated sweep is insufficient. The common articulation must pass
both zero-contact and representative loaded TRAIN/development tests before a
new freeze. No measured-state clipping is acceptable.
"""
    (out / "DEX3_COMMON_ARTICULATION_AUDIT.md").write_text(amd, encoding="utf-8")

    contract = {
        "schema_version": "single_variable_ab_pipeline_contract_v1",
        "status": "DESIGN_FROZEN_IMPLEMENTATION_NOT_READY",
        "only_switch": {"name": "representation_mode", "values": ["WRIST", "INTERACTION"]},
        "before_switch_common": [
            "source episode loading",
            "episode-conditioned source task registration",
            "source event timeline extraction",
            "source FK and units/frame conventions",
        ],
        "switch_scope": {
            "WRIST": "registered source wrist/TCP SE(3) plus one fixed ALOHA-tool-to-G1-wrist compatibility transform",
            "INTERACTION": "registered object/whole-hand/bimanual interaction targets",
        },
        "after_switch_common": [
            "G1 kinematic model",
            "IK and temporal solver implementation/config",
            "natural-arm regularization",
            "joint limits and hard-limit projector",
            "common source timeline",
            "Dex3 state machine and commands",
            "dataset schema and packaging",
            "ACT architecture/training/selection",
            "physical environment and evaluator",
        ],
        "gates_before_dataset_generation": [
            "A wrist reference competence PASS",
            "B interaction reference competence PASS",
            "common temporal semantics PASS",
            "common Dex3 commands bit-identical A/B",
            "common solver and realization adapter hash-identical",
        ],
        "gates_before_physical_evaluation": [
            "loaded Dex3 articulation PASS",
            "TRAIN-only smoke PASS",
            "fresh untouched FINAL_TEST membership frozen",
        ],
    }
    contract["content_sha256"] = canonical_sha(contract)
    write_json(out / "SINGLE_VARIABLE_PIPELINE_CONTRACT.json", contract)

    print(json.dumps({
        "status": audit["status"],
        "unintended_confounds": len(confounds),
        "a_reference_competent": reference["a_reference_competent"],
        "dex3_articulation_pass": articulation["overall_pass"],
        "output": str(out),
    }, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
