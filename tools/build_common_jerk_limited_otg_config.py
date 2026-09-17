#!/usr/bin/env python3
"""Freeze policy-independent 28D Ruckig limits from existing authoritative data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MOTION_REFERENCE = (
    ROOT
    / "outputs/policy_b_causal_execution/dataset_b_motion_reference"
    / "dataset_b_natural_motion_reference.json"
)
PROJECTION_CONFIG = (
    ROOT
    / "outputs/common_g1_deployment_safety/simulation_controller_margin_v2"
    / "freeze_manifest.json"
)
CONTROL_CONTRACT = ROOT / "tools/policy_b_isaac_control_contract.py"
IMPLEMENTATION = ROOT / "tools/common_jerk_limited_otg.py"
OUTPUT = (
    ROOT
    / "outputs/policy_execution_stability_review/jerk_limited_otg"
    / "common_g1_28d_ruckig_config.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    motion = read_json(MOTION_REFERENCE)
    projection = read_json(PROJECTION_CONFIG)
    projection_rows = {row["joint_name"]: row for row in projection["joints"]}
    if motion["fps"] != 30.0 or motion["frames"] != 34478 or motion["episodes"] != 50:
        raise RuntimeError("Dataset-B natural-motion reference is not the frozen 50-episode set")
    joints = []
    for row in motion["per_joint"]:
        name = row["joint"]
        limit = projection_rows[name]
        group = limit["group"]
        # Arm motion is continuously represented, so p95 is a conservative natural
        # envelope. Dex3 labels are sparse primitives with many exact holds; use the
        # observed maximum so every frozen primitive transition remains realizable.
        percentile = "p95" if group == "arm" else "maximum"
        velocity_percentile = "p99" if group == "arm" else "maximum"
        dataset_velocity = float(row["velocity_rad_s"][velocity_percentile])
        # The frozen Isaac actuator contract supplies an authoritative velocity
        # ceiling for every controlled arm and Dex3 joint. Dataset percentiles are
        # used only where an authoritative controller limit is unavailable.
        # The actuator's 12 rad/s ceiling is a hard upper bound, not a useful
        # realization target.  Use the selected Dataset-B natural-motion statistic
        # for both groups and cap it by the authoritative ceiling.  In particular,
        # using 12 rad/s for Dex3 despite an observed maximum of 3.60 rad/s created
        # command steps that violated the existing 4.5 rad/s rollout gate.
        velocity = min(12.0, dataset_velocity)
        acceleration = float(row["acceleration_rad_s2"][percentile])
        jerk = float(row["jerk_rad_s3"][percentile])
        if min(velocity, acceleration, jerk) <= 0.0:
            raise RuntimeError(f"non-positive OTG constraint for {name}")
        joints.append(
            {
                "joint_index": int(row["joint_index"]),
                "joint_name": name,
                "group": group,
                "hard_lower_rad": float(limit["hard_lower_rad"]),
                "hard_upper_rad": float(limit["hard_upper_rad"]),
                "target_safe_lower_rad": float(limit["safe_lower_rad"]),
                "target_safe_upper_rad": float(limit["safe_upper_rad"]),
                "max_velocity_rad_s": velocity,
                "authoritative_velocity_ceiling_rad_s": 12.0,
                "dataset_reference_velocity_rad_s": dataset_velocity,
                "dataset_velocity_percentile": velocity_percentile,
                "max_acceleration_rad_s2": acceleration,
                "max_jerk_rad_s3": jerk,
                "dataset_percentile": percentile,
            }
        )
    joints.sort(key=lambda row: row["joint_index"])
    if [row["joint_index"] for row in joints] != list(range(28)):
        raise RuntimeError("OTG joint order is not canonical 0..27")
    payload = {
        "schema_version": "common_g1_28d_ruckig_otg_v1",
        "status": "ISAAC_DIAGNOSTIC_CANDIDATE_REAL_HARDWARE_NOT_APPROVED",
        "name": "COMMON_G1_28D_JERK_LIMITED_OTG",
        "policy_independent": True,
        "task_phase_episode_object_specific_logic": False,
        "command_semantics": "absolute_joint_position_rad",
        "control_fps": 30.0,
        "control_period_s": 1.0 / 30.0,
        "implementation": {
            "path": str(IMPLEMENTATION),
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "ruckig": {
            "package_version": "0.19.4",
            "interface": "community_python_online_position_otg",
            "target_velocity_rad_s": 0.0,
            "target_acceleration_rad_s2": 0.0,
            "synchronization": "TIME_DEFAULT",
            "current_state_source": "ISAAC_MEASURED_Q_DQ_AND_CAUSAL_COMMITTED_PREFIX_DQ_SLOPE_DDQ_AT_REPLAN",
            "acceleration_estimator": "ordinary least-squares slope over up to execution_horizon past 30-Hz measured dq samples",
            "between_replans": "RUCKIG_OUTPUT_PASS_TO_INPUT",
        },
        "target_selection": {
            "definition": "deployment-safe selected causal-plan row execution_horizon-1",
            "reason": "the endpoint of exactly the prefix committed at the current observation",
            "future_policy_calls_used": False,
            "offline_temporal_consensus_used": False,
        },
        "constraint_derivation": {
            "arms": "per-joint Dataset-B p95 absolute velocity/acceleration/jerk",
            "dex3": "per-joint Dataset-B observed maximum because sparse piecewise-constant primitive transitions make percentiles underrepresent the nonzero transitions",
            "hardware_velocity_ceiling_rad_s": 12.0,
            "velocity_rule": "nominal arms use Dataset-B p99 and Dex3 uses its Dataset-B observed maximum; both are capped at the authoritative 12 rad/s ceiling and are causally elevated only to an already-higher continuous-reference |dq| for braking",
            "acceleration_jerk_rule": "Dataset-B percentile because the controller contract supplies no acceleration or jerk ceiling",
            "task_success_tuning": False,
            "motion_reference": str(MOTION_REFERENCE),
            "motion_reference_sha256": sha256_file(MOTION_REFERENCE),
            "projection_config": str(PROJECTION_CONFIG),
            "projection_config_sha256": sha256_file(PROJECTION_CONFIG),
            "controller_contract": str(CONTROL_CONTRACT),
            "controller_contract_sha256": sha256_file(CONTROL_CONTRACT),
        },
        "input_validation": {
            "finite_required": True,
            "target_within_simulation_safe_interval_required": True,
            "invalid_ruckig_result": "FAIL_CLOSED_NO_COMMAND",
            "current_state_limit_check": False,
            "current_state_limit_check_reason": (
                "measured PhysX state may contain the separately logged microscopic Dex3 hard-bound "
                "excursion; Ruckig starts from the unmodified measurement and brakes inward"
            ),
        },
        "joints": joints,
        "source_integrity": {
            "dataset_b_action_arrays_modified": False,
            "raw_policy_chunks_modified": False,
            "real_robot_command_authorized": False,
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
