#!/usr/bin/env python3
"""Freeze the ACT-B frame-0 root cause, common initial contract, and dry-run gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/policy_b_act/isaac_frame0_and_rollout"
LEGACY = OUT / "frame0_legacy_hidden"
COMMON = OUT / "frame0_common_visual"
RESNAP = OUT / "frame0_common_visual_resnap"
PROJECTOR_FREEZE = ROOT / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2/freeze_manifest.json"
FEASIBILITY = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
ERRONEOUS_RUNNER_SHA = "d5717f1f64efa96fc0f17975aa97fd44974f1aed015e932c175a73d34f7a7148"
AUTHORITATIVE_RUNNER_SHA = "ff44c58ff88be8fe86e9c4516ca2923fcce845b97f6d2edec9d249faf02d922f"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def branch_flags(q: np.ndarray, absolute: float, multiplier: float) -> np.ndarray:
    flags = np.zeros(len(q), dtype=bool)
    norms = np.linalg.norm(np.diff(q, axis=0), axis=1)
    for index in range(1, len(q)):
        local = float(np.median(norms[max(0, index - 10) : min(len(norms), index + 9)]))
        flags[index] = norms[index - 1] > max(absolute, multiplier * max(local, 1e-6))
    return flags


def main() -> None:
    legacy = read_json(LEGACY / "frame0_audit_report.json")
    common = read_json(COMMON / "frame0_audit_report.json")
    resnap = read_json(RESNAP / "frame0_audit_report.json")
    limits = read_json(FEASIBILITY)["unchanged_acceptance"]
    projector = read_json(PROJECTOR_FREEZE)
    with np.load(COMMON / "frame0_arrays.npz", allow_pickle=False) as archive:
        names = archive["joint_names"].astype(str).tolist()
        nominal = archive["nominal_common_initial"].astype(np.float64)
        reset = archive["isaac_reset_state"].astype(np.float64)
        settled = archive["isaac_settled_state"].astype(np.float64)
        settled_velocity = archive["isaac_settled_velocity"].astype(np.float64)
        raw = archive["act_e0_raw_chunk"].astype(np.float64)
        hard = archive["act_e0_hard_limit_projected_chunk"].astype(np.float64)
        deployment = archive["act_e0_deployment_projected_chunk"].astype(np.float64)

    if names != projector["joint_names"] or raw.shape != (50, 28):
        raise RuntimeError("frame-0 named interface changed")
    if not np.array_equal(raw[:, :14], deployment[:, :14]):
        raise RuntimeError("common adapter changed an arm output")
    hard_lower = np.asarray([row["hard_lower_rad"] for row in projector["joints"]])
    hard_upper = np.asarray([row["hard_upper_rad"] for row in projector["joints"]])
    hard_violations = (deployment < hard_lower[None]) | (deployment > hard_upper[None])
    command_sequence = np.vstack((settled, deployment))
    qdot = np.diff(command_sequence, axis=0) * 30.0
    qddot = np.vstack(((qdot[0] - settled_velocity) * 30.0, np.diff(qdot, axis=0) * 30.0))
    internal_arm_norms = np.linalg.norm(np.diff(deployment[:, :14], axis=0), axis=1)
    internal_flags = branch_flags(
        deployment[:, :14],
        float(limits["branch_absolute_step_norm_rad"]),
        float(limits["branch_local_multiplier"]),
    )
    first_delta = deployment[0] - settled
    collision = common["preview"]["collision"]
    checks = {
        "native_output_shape_50x28": raw.shape == (50, 28),
        "native_output_finite": bool(np.isfinite(raw).all()),
        "raw_output_preserved_separately": bool(np.array_equal(raw, np.load(COMMON / "frame0_arrays.npz")["act_e0_raw_chunk"])),
        "command_hard_limit_violations_zero_after_frozen_common_adapter": int(np.count_nonzero(hard_violations)) == 0,
        "first_command_max_joint_step": float(np.max(np.abs(first_delta))) <= float(limits["maximum_joint_step_rad"]),
        "velocity_preview": float(np.max(np.abs(qdot))) <= float(limits["maximum_velocity_rad_s"]),
        "acceleration_preview": float(np.max(np.abs(qddot))) <= float(limits["maximum_acceleration_rad_s2"]),
        "branch_gate_on_action_trajectory": int(np.count_nonzero(internal_flags)) == 0,
        "self_collision_zero": collision["invalid_hard_self_collision_incidence"] == 0,
        "arm_torso_collision_zero": collision["arm_torso_collision_incidence"] == 0,
    }
    dry_report = {
        "schema_version": "act_b_frame0_corrected_dry_run_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "commands_executed": 0,
        "controlled_input": "correct common visual-object-free frame-0 RGB plus settled measured native 28D state",
        "checks": checks,
        "raw_act_action_path": str(COMMON / "frame0_arrays.npz:act_e0_raw_chunk"),
        "deployment_projected_action_path": str(COMMON / "frame0_arrays.npz:act_e0_deployment_projected_chunk"),
        "commanded_action": "NONE_DRY_RUN",
        "native_shape": list(raw.shape),
        "raw_finite": bool(np.isfinite(raw).all()),
        "normalization_round_trip_max_abs_rad": common["normalization"]["round_trip_max_abs_rad"],
        "first_command_max_joint_step_rad": float(np.max(np.abs(first_delta))),
        "first_command_arm_l2_rad_diagnostic_only": float(np.linalg.norm(first_delta[:14])),
        "maximum_velocity_preview_rad_s": float(np.max(np.abs(qdot))),
        "maximum_acceleration_preview_rad_s2": float(np.max(np.abs(qddot))),
        "maximum_internal_arm_l2_step_rad": float(np.max(internal_arm_norms)),
        "internal_branch_flag_count": int(np.count_nonzero(internal_flags)),
        "collision": collision,
        "thresholds_unchanged": limits,
        "gate_semantics": {
            "branch": "unchanged branch_flags over adjacent 14D arm configurations inside proposed action trajectory",
            "current_state_to_first_command": "unchanged max-per-joint step, velocity, and acceleration gates",
            "gate_increase": False,
        },
        "real_hardware": "DISABLED",
    }
    atomic_json(OUT / "frame0_corrected_dry_run.json", dry_report)

    contract = {
        "schema_version": "common_g1_policy_initial_state_v1",
        "status": "COMMON_G1_POLICY_INITIAL_STATE_V1",
        "scope": ["ACT-A", "ACT-B", "optional future VLA baseline with identical named 28D interface"],
        "policy_specific": False,
        "command_semantics": "absolute_joint_position_rad",
        "joint_order": names,
        "arm_joint_names": names[:14],
        "dex3_joint_names": names[14:],
        "g1_14_arm_initial_q_rad": nominal[:14],
        "dex3_14_hand_initial_q_rad": nominal[14:],
        "full_28d_initial_q_rad": nominal,
        "value_source": {
            "before_projection": "joint-wise dataset-wide median of the 50 frozen Dataset-B episode frame-0 q_target configurations",
            "after_projection": "exact existing frozen policy-independent simulation-controller-safe projection",
            "source_initial_condition": common["initial_provenance"]["prior_initial_condition"],
            "source_initial_condition_sha256": common["initial_provenance"]["prior_initial_condition_sha256"],
            "projection_freeze_manifest": str(PROJECTOR_FREEZE),
            "projection_freeze_manifest_sha256": sha256_file(PROJECTOR_FREEZE),
        },
        "tolerances": {
            "immediate_reset_max_abs_from_nominal_rad": 1e-6,
            "post_settle_max_abs_from_nominal_rad": 0.01,
            "post_settle_tolerance_provenance": "policy-independent Isaac controller frame-0 characterization; observed maximum 0.009043782949447632 rad, rounded once to 0.01 rad for common A/B use",
            "measured_dex3_hard_limit_excursion_diagnostic_cap_rad": 0.0001,
            "dex3_cap_provenance": "existing precharacterized Isaac-only microscopic PhysX margin audit; not hardware authorization",
        },
        "reset_procedure": [
            "map all 28 joints by exact name; never infer order from articulation array positions",
            "write COMMON_G1_POLICY_INITIAL_STATE_V1 qpos and zero qvel to Isaac",
            "forward simulation once and verify immediate-reset tolerance and named hard limits",
            "hold the identical nominal absolute-position target for 1.0 seconds at the frozen physics timestep",
            "capture current measured native 28D qpos and current diagnostic RGB for policy inference",
            "do not resnap after settle; do not substitute the nominal target for measured proprioception",
        ],
        "settle_duration_seconds": 1.0,
        "observed_verification": {
            "immediate_reset_max_abs_from_nominal_rad": float(np.max(np.abs(reset - nominal))),
            "post_settle_max_abs_from_nominal_rad": float(np.max(np.abs(settled - nominal))),
            "post_settle_arm_l2_from_nominal_rad": float(np.linalg.norm(settled[:14] - nominal[:14])),
            "pass": float(np.max(np.abs(reset - nominal))) <= 1e-6 and float(np.max(np.abs(settled - nominal))) <= 0.01,
        },
        "forbidden_variants": ["policy-specific start pose", "episode-specific start pose", "post-settle resnap workaround", "gate increase"],
        "real_hardware": "NOT_AUTHORIZED",
    }
    contract_path = OUT / "COMMON_G1_POLICY_INITIAL_STATE_V1.json"
    atomic_json(contract_path, contract)

    six = legacy["right_dex3_raw_hard_limit_violation_joints"]
    if len(six) != 6:
        raise RuntimeError(f"legacy six-joint reproduction changed: {six}")
    root_report = {
        "schema_version": "act_b_frame0_root_cause_v1",
        "status": "PASS_ROOT_CAUSE_ISOLATED",
        "right_dex3_violations": six,
        "right_dex3_origin": {
            "classification": ["ACT_RAW_BOUNDARY_OVERSHOOT", "MISSING_USE_OF_ALREADY_FROZEN_GENERIC_DEPLOYMENT_PROJECTION"],
            "isaac_initialization": False,
            "act_raw_prediction": True,
            "act_denormalization": False,
            "joint_order_mismatch": False,
            "missing_common_projection_in_failed_runner": True,
            "normalization_round_trip_max_abs_rad": legacy["normalization"]["round_trip_max_abs_rad"],
            "command_violations_after_exact_frozen_common_projection": 0,
            "act_specific_clamp_created": False,
        },
        "branch_norm_excess": {
            "observed_reported_approximately_rad": 0.0062,
            "classification": "METRIC_IMPLEMENTATION_BUG_WITH_SETTLE_DRIFT_EXPOSING_THE_ERROR",
            "failed_runner_behavior": "applied 0.180-rad arm branch L2 gate to current measured state -> first proposed command",
            "correct_existing_behavior": "apply branch_flags to adjacency inside proposed action trajectory; use max-per-joint step, velocity, acceleration for current state -> first command",
            "settle_drift_arm_l2_rad": common["states"]["settled_arm_l2_from_nominal_rad"],
            "failed_metric_norm_from_settled_rad": common["branch_metric"]["arm_branch_norm_rad"],
            "diagnostic_norm_from_nominal_rad": common["branch_metric"]["arm_norm_vs_nominal_rad"],
            "correct_internal_chunk_branch_flag_count": int(np.count_nonzero(internal_flags)),
            "maximum_internal_arm_l2_step_rad": float(np.max(internal_arm_norms)),
            "gate_rad_unchanged": float(limits["branch_absolute_step_norm_rad"]),
            "gate_increased": False,
            "resnap_test_rejected": {
                "branch_norm_rad": resnap["branch_metric"]["arm_branch_norm_rad"],
                "reason": "changes the controlled image/state pair and creates a genuine first-step limit failure; not used",
            },
        },
        "implementation_evidence": {
            "failed_act_runner_path": str(ROOT / "tools/run_act_b_isaac_diagnostic.py"),
            "failed_act_runner_pre_fix_sha256": ERRONEOUS_RUNNER_SHA,
            "failed_lines": "330-356",
            "authoritative_policy_independent_runner_path": str(ROOT / "tools/run_policy_b_isaac_doll_handoff.py"),
            "authoritative_runner_sha256": AUTHORITATIVE_RUNNER_SHA,
            "authoritative_lines": "1014-1069",
        },
        "mapping": "PASS",
        "normalization": "PASS",
        "initial_state_contract": str(contract_path),
        "initial_state_contract_sha256": sha256_file(contract_path),
        "corrected_frame0_dry_run": str(OUT / "frame0_corrected_dry_run.json"),
        "corrected_frame0_status": dry_report["status"],
        "commands_executed_during_all_frame0_audits": 0,
        "real_hardware": "DISABLED",
    }
    atomic_json(OUT / "frame0_root_cause_report.json", root_report)
    print(json.dumps(root_report, indent=2))


if __name__ == "__main__":
    main()
