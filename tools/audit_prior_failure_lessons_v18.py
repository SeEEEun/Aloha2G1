#!/usr/bin/env python3
"""Freeze v18 inputs and record evidence-backed lessons before solving."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_full_task_execution_v18"
SR = ROOT / "outputs/scene_registered_retargeting"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
V14 = SR / "current_layout_ep49_root_registered_v14"
V15 = SR / "current_layout_ep49_orientation_dex3_v15"
V16 = SR / "current_layout_ep49_contact_carrier_v16"
V171 = SR / "current_layout_ep49_execution_physics_v17_1"
V171R = SR / "current_layout_ep49_execution_physics_v17_1_renderfix"
V172 = SR / "current_layout_ep49_execution_quality_v17_2"
V172R = SR / "current_layout_ep49_execution_quality_v17_2_renderfix"
PHOTO = SR / "dex3_left_phone_pinch_photo_calibration_v1"
CLOSED = SR / "dex3_left_phone_pinch_closed_contact_v2"
PAD = SR / "dex3_left_phone_pinch_pad_ablation_v1"
FORCE = SR / "dex3_left_phone_retention_force_audit_v1"
WRENCH = SR / "dex3_left_phone_wrench_balanced_pinch_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "v14_fidelity": V14 / "aloha_fidelity_metrics_v14.json",
        "v14_ik": V14 / "ik_metrics_v14.json",
        "v14_collision": V14 / "collision_breakdown_v14.json",
        "v15_numeric_gate": V15 / "numeric_gate_summary.json",
        "v15_collision": V15 / "collision_breakdown.json",
        "v16_numeric_gate": V16 / "numeric_gate_summary.json",
        "v16_fidelity": V16 / "v14_vs_v15_vs_v16_fidelity.json",
        "v17_1_kinematic": V171 / "kinematic_prephysics_result.json",
        "v17_1_phone_grasp": V171 / "phone_grasp_physics_result.json",
        "v17_1_full_task": V171 / "full_task_physics_result.json",
        "v17_1_render_parity": V171R / "repaired_physics_result.json",
        "v17_2_build": V172 / "build_summary.json",
        "v17_2_temporal": V172 / "temporal_smoothness_metrics.json",
        "v17_2_full_physics": V172 / "full_true_physics_diagnostic.json",
        "v17_2_render": V172R / "v17_2_rendered_motion_audit.json",
        "left_photo_identity": PHOTO / "left_dex3_physical_identity.json",
        "left_photo_geometry": PHOTO / "fingertip_geometry_metrics.json",
        "left_photo_primitive": PHOTO / "left_phone_fingertip_pinch_primitive.json",
        "left_closed_contact": CLOSED / "closed_pose_contact_metrics.json",
        "left_pad_ablation": PAD / "candidate_A_vs_B_retention.json",
        "left_force_root_cause": FORCE / "dex3_retention_root_cause.json",
        "left_wrench_ablation": WRENCH / "candidate_A_vs_C_retention_metrics.json",
    }
    missing = [str(path) for path in evidence.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    v15 = load(evidence["v15_numeric_gate"])
    v16 = load(evidence["v16_numeric_gate"])
    v171 = load(evidence["v17_1_kinematic"])
    v171_grasp = load(evidence["v17_1_phone_grasp"])
    v172_temporal = load(evidence["v17_2_temporal"])
    v172_physics = load(evidence["v17_2_full_physics"])
    photo = load(evidence["left_photo_geometry"])
    primitive = load(evidence["left_photo_primitive"])
    closed = load(evidence["left_closed_contact"])
    pad = load(evidence["left_pad_ablation"])
    force = load(evidence["left_force_root_cause"])
    wrench = load(evidence["left_wrench_ablation"])

    with np.load(SOURCE, allow_pickle=False) as archive:
        action = archive["optimized_action"].copy()
        source_shape = list(action.shape)
        source_finite = bool(np.isfinite(action).all())
    with np.load(V14 / "corrected_targets_v14.npz", allow_pickle=False) as archive:
        left = archive["corrected_left_position"].copy()
        right = archive["corrected_right_position"].copy()

    payload = {
        "status": "PRIOR_FAILURE_LESSONS_REVIEWED_BEFORE_V18_SOLVE",
        "reviewed_artifacts": {
            name: {"path": path.resolve(), "sha256": sha256(path)}
            for name, path in evidence.items()
        },
        "source": {
            "path": SOURCE.resolve(), "sha256": sha256(SOURCE),
            "expected_sha256": "a7f5543e07e315d59f52004dab48423a4ee52dfcbafb9b6d5d1a731fcbd3694c",
            "shape": source_shape, "finite": source_finite,
        },
        "v14_cartesian_backbone": {
            "path": (V14 / "corrected_targets_v14.npz").resolve(),
            "file_sha256": sha256(V14 / "corrected_targets_v14.npz"),
            "left_array_sha256": array_sha(left),
            "right_array_sha256": array_sha(right),
            "shape": [list(left.shape), list(right.shape)],
        },
        "lessons": [
            {
                "failure": "V15_V16_CONTACT_CONSTRAINTS_FOUGHT_SOURCE_BEHAVIOR",
                "evidence": {
                    "v15_status": v15.get("status"),
                    "v16_status": v16.get("status"),
                    "v15_orientation_or_gate": v15,
                    "v16_gate_summary": v16,
                },
                "enforced_v18_rule": "V14 left/right Cartesian XYZ arrays are immutable; no grasp-driven translation residual or waypoint is allowed.",
            },
            {
                "failure": "HISTORICAL_RIGHT_C_MIDDLE_HOOK_USED_WRONG_TASK_DIGIT_AND_DROVE_WRIST_TWIST",
                "evidence": "Active-model identity maps right A=index, right B=thumb, right C=middle; v17.1 used right_C as hook.",
                "enforced_v18_rule": "One predefined physical right thumb-index primitive; middle remains non-task. Geometry warning is reported rather than arm-path redesign.",
            },
            {
                "failure": "STALE_FABRIC_RENDER_COULD_SHOW_STATIC_ROBOT_WITH_MOVING_Q",
                "evidence": {
                    "v17_1_renderfix": load(evidence["v17_1_render_parity"]),
                    "v17_2_render_status": load(evidence["v17_2_render"]).get("status"),
                },
                "enforced_v18_rule": "Kinematic and PhysX render paths retain independent command/state/link/mask parity and static-video guards.",
            },
            {
                "failure": "AIR_SUSPENDED_CLOSED_HAND_TEST_DID_NOT_MODEL_TASK_ACQUISITION",
                "evidence": {
                    "closed_pose_contact_status": closed.get("status"),
                    "bilateral_duration_s": closed.get("simultaneous_contact_duration_s"),
                    "static_slip_m": closed.get("retention", {}).get("relative_slip_m"),
                },
                "enforced_v18_rule": "Phone starts at the authored table-supported pose; OPEN-to-PREGRASP-to-PINCH precedes source-derived lift.",
            },
            {
                "failure": "STATIC_PAD_AND_WRENCH_CANDIDATES_DID_NOT_SOLVE_RETENTION",
                "evidence": {
                    "candidate_B_status": pad.get("status"),
                    "candidate_B_recommended": pad.get("recommended_candidate"),
                    "candidate_C_status": wrench.get("status"),
                    "candidate_C_decision": wrench.get("decision"),
                    "force_root_cause": force,
                },
                "enforced_v18_rule": "Candidate A is final; no Candidate D/E/F, friction sweep, effort sweep, or pad model work is performed.",
            },
            {
                "failure": "V17_2_NULLSPACE_SOLVE_INCREASED_TEMPORAL_HIGH_FREQUENCY_MOTION",
                "evidence": {
                    "v17_2_before": v172_temporal.get("before"),
                    "v17_2_after": v172_temporal.get("after"),
                    "v17_2_status": v172_temporal.get("status"),
                },
                "enforced_v18_rule": "Only redundant q and partial-orientation activation are stabilized; source XYZ and source timestamps are untouched.",
            },
            {
                "failure": "V17_1_STAGE_GATE_STOPPED_AFTER_ROTATION_FAILURE",
                "evidence": {
                    "grasp_status": v171_grasp.get("status"),
                    "full_task_status": load(evidence["v17_1_full_task"]).get("status"),
                    "v17_2_full_diagnostic": v172_physics.get("status"),
                },
                "enforced_v18_rule": "Exactly 990 commands execute without stage-gate termination and without recovery.",
            },
        ],
        "approved_left": {
            "status": primitive["status"],
            "q_rad": primitive["selected_static_q_rad"],
            "physical_task_fingers": primitive["physical_task_fingers"],
            "third_role": primitive["third_finger_role"],
            "photo_geometry_status": photo["status"],
            "reoptimization_allowed": False,
        },
        "v17_1_reference_gate": {
            "status": v171.get("status"),
            "position": v171.get("metrics", {}).get("position"),
            "fidelity": v171.get("metrics", {}).get("fidelity"),
            "orientation": v171.get("metrics", {}).get("orientation"),
        },
        "scope_guards": {
            "validation_reads": 0, "heldout_reads": 0, "g1_expert_reads": 0,
            "dds": False, "publisher": False, "real_robot_command": False,
        },
    }
    if payload["source"]["sha256"] != payload["source"]["expected_sha256"]:
        raise RuntimeError("source optimized_action archive SHA-256 mismatch")
    dump(OUT / "prior_failure_lessons_v18.json", payload)
    print(json.dumps({
        "status": payload["status"],
        "source_sha256": payload["source"]["sha256"],
        "left_target_sha256": payload["v14_cartesian_backbone"]["left_array_sha256"],
        "right_target_sha256": payload["v14_cartesian_backbone"]["right_array_sha256"],
        "lesson_count": len(payload["lessons"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
