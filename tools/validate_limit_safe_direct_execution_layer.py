#!/usr/bin/env python3
"""Pre-rollout proof that every common Dex3 command obeys measured hard limits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_execution_isaac_runtime import (  # noqa: E402
    COMMON_PHYSICAL_CONTROLLER,
    PHYSICAL_ENVIRONMENT,
    PHYSICS_CONFIG,
)
from tools.common_execution_layer import (  # noqa: E402
    ARM_INDICES,
    DEX3_INDICES,
    LEFT_DEX3,
    RIGHT_DEX3,
    WRIST_INDICES,
    Dex3Primitive,
    ExecutionSnapshot,
    _transition,
)
from tools.direct_physical_execution_layer import (  # noqa: E402
    DirectPhysicalDex3ExecutionLayer,
    authoritative_joint_limits,
)


OUT = ROOT / "outputs/final_direct_physical_eval35"
COMMAND_MANIFEST = OUT / "00_preparation/DIRECT_EVAL35_PHYSICAL_COMMAND_MANIFEST.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
RESULT = OUT / "00_preparation/LIMIT_SAFE_DEX3_VALIDATION.json"
REPORT = OUT / "00_preparation/LIMIT_SAFE_DEX3_VALIDATION.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def endpoint_and_transition_checks(
    primitive: Dex3Primitive,
    arm_lower: np.ndarray,
    arm_upper: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dummy = np.zeros((1, 28), dtype=np.float64)
    controller = DirectPhysicalDex3ExecutionLayer(
        primitive,
        ["OPEN_INTENT"],
        dummy,
        dummy.copy(),
        "ACT-A40",
        arm_lower,
        arm_upper,
        lower,
        upper,
    )
    endpoints = {
        "LEFT_OPEN": (controller.left_open, lower[:7], upper[:7]),
        "LEFT_PRESHAPE": (controller.left_preshape, lower[:7], upper[:7]),
        "LEFT_FULL_CLOSE": (controller.left_full_close, lower[:7], upper[:7]),
        "RIGHT_OPEN": (controller.right_open, lower[7:], upper[7:]),
        "RIGHT_PRESHAPE": (controller.right_preshape, lower[7:], upper[7:]),
        "RIGHT_FULL_CLOSE": (controller.right_full_close, lower[7:], upper[7:]),
    }
    endpoint_rows: list[dict[str, Any]] = []
    for name, (value, low, high) in endpoints.items():
        endpoint_rows.append(
            {
                "name": name,
                "q_rad": value.tolist(),
                "hard_limit_violation_scalar_count": int(
                    np.count_nonzero((value < low) | (value > high))
                ),
                "minimum_hard_limit_clearance_rad": float(
                    np.min(np.minimum(value - low, high - value))
                ),
            }
        )
    transitions = (
        ("LEFT_OPEN_TO_PRESHAPE", controller.left_open, controller.left_preshape, primitive.preshape_frames, lower[:7], upper[:7]),
        ("LEFT_PRESHAPE_TO_FULL_CLOSE", controller.left_preshape, controller.left_full_close, primitive.close_frames, lower[:7], upper[:7]),
        ("LEFT_FULL_CLOSE_TO_OPEN", controller.left_full_close, controller.left_open, primitive.release_frames, lower[:7], upper[:7]),
        ("RIGHT_OPEN_TO_PRESHAPE", controller.right_open, controller.right_preshape, primitive.preshape_frames, lower[7:], upper[7:]),
        ("RIGHT_PRESHAPE_TO_FULL_CLOSE", controller.right_preshape, controller.right_full_close, primitive.close_frames, lower[7:], upper[7:]),
        ("RIGHT_FULL_CLOSE_TO_OPEN", controller.right_full_close, controller.right_open, primitive.release_frames, lower[7:], upper[7:]),
    )
    transition_rows: list[dict[str, Any]] = []
    for name, start, stop, frames, low, high in transitions:
        values = np.stack([_transition(start, stop, frame, frames) for frame in range(frames)])
        transition_rows.append(
            {
                "name": name,
                "frames": int(frames),
                "interpolation": "minimum_jerk_convex_interpolation",
                "hard_limit_violation_scalar_count": int(
                    np.count_nonzero((values < low) | (values > high))
                ),
                "minimum_hard_limit_clearance_rad": float(
                    np.min(np.minimum(values - low, high - values))
                ),
                "maximum_adjacent_joint_step_rad": float(
                    np.max(np.abs(np.diff(values, axis=0)), initial=0.0)
                ),
            }
        )
    return endpoint_rows, transition_rows


def synthetic_snapshot(measured: np.ndarray) -> ExecutionSnapshot:
    forces = {digit: 0.02 for digit in ("thumb", "index", "middle")}
    return ExecutionSnapshot(
        measured_q_rad=measured,
        object_world=np.eye(4),
        whole_hand_world={"left": np.eye(4), "right": np.eye(4)},
        digit_force_n={"left": forces.copy(), "right": forces.copy()},
        table_force_n=0.0,
    )


def main() -> int:
    commands = read_json(COMMAND_MANIFEST)
    if commands.get("status") != "FROZEN_BEFORE_PHYSICAL_ROLLOUT" or len(commands.get("records", [])) != 70:
        raise RuntimeError("exact EVAL35 physical commands are unavailable")
    contract = read_json(JOINT_CONTRACT)
    joint_lower, joint_upper, joint_names = authoritative_joint_limits(contract)
    arm_lower, arm_upper = joint_lower[:14], joint_upper[:14]
    lower, upper = joint_lower[14:], joint_upper[14:]
    dex3_names = joint_names[14:]
    primitive = Dex3Primitive.from_frozen_dependencies(
        read_json(PHYSICS_CONFIG),
        read_json(PHYSICAL_ENVIRONMENT),
        read_json(COMMON_PHYSICAL_CONTROLLER),
    )
    endpoint_rows, transition_rows = endpoint_and_transition_checks(
        primitive, arm_lower, arm_upper, lower, upper
    )
    rollout_rows: list[dict[str, Any]] = []
    total_dex3_scalars = total_violations = 0
    total_arm_differences = total_wrist_differences = 0
    total_unexpected_arm_differences = total_remaining_arm_violations = 0
    maximum_arm_hard_limit_correction = 0.0
    maximum_dex3_step = 0.0
    for record in commands["records"]:
        path = Path(record["physical_command"])
        if sha256_file(path) != record["physical_command_sha256"]:
            raise RuntimeError(f"physical command hash drift: {path}")
        with np.load(path, allow_pickle=False) as archive:
            raw = np.asarray(archive["raw_policy_command"], dtype=np.float64)
            safe = np.asarray(archive["policy_safe_command"], dtype=np.float64)
            intent = archive["common_task_intent"].astype(str)
            initial = np.asarray(archive["common_initial_q_rad"], dtype=np.float64)
        controller = DirectPhysicalDex3ExecutionLayer(
            primitive,
            intent,
            raw,
            safe,
            str(record["method"]),
            arm_lower,
            arm_upper,
            lower,
            upper,
        )
        rows = []
        # Deliberately keep the measured state lagged at the initial state.  The
        # common command must remain continuous/limit-safe independently of
        # contact-induced measured-q lag.
        snapshot = synthetic_snapshot(initial)
        for frame in range(len(safe)):
            decision = controller.step(frame, snapshot)
            rows.append(decision.executed_command)
        executed = np.stack(rows)
        dex3 = executed[:, DEX3_INDICES]
        violations = int(np.count_nonzero((dex3 < lower) | (dex3 > upper)))
        expected_arm = np.clip(safe[:, :14], arm_lower, arm_upper)
        arm_differences = int(np.count_nonzero(executed[:, :14] != safe[:, :14]))
        wrist_differences = int(np.count_nonzero(executed[:, WRIST_INDICES] != safe[:, WRIST_INDICES]))
        unexpected_arm_differences = int(
            np.count_nonzero(executed[:, :14] != expected_arm)
        )
        remaining_arm_violations = int(
            np.count_nonzero((executed[:, :14] < arm_lower) | (executed[:, :14] > arm_upper))
        )
        maximum_arm_correction = float(
            np.max(np.abs(executed[:, :14] - safe[:, :14]), initial=0.0)
        )
        adjacent = float(np.max(np.abs(np.diff(dex3, axis=0)), initial=0.0))
        total_dex3_scalars += int(dex3.size)
        total_violations += violations
        total_arm_differences += arm_differences
        total_wrist_differences += wrist_differences
        total_unexpected_arm_differences += unexpected_arm_differences
        total_remaining_arm_violations += remaining_arm_violations
        maximum_arm_hard_limit_correction = max(
            maximum_arm_hard_limit_correction, maximum_arm_correction
        )
        maximum_dex3_step = max(maximum_dex3_step, adjacent)
        rollout_rows.append(
            {
                "method": record["method"],
                "eval_index": int(record["eval_index"]),
                "stable_episode_id": record["stable_episode_id"],
                "frames": int(len(executed)),
                "dex3_command_scalar_count": int(dex3.size),
                "dex3_hard_limit_violation_scalar_count": violations,
                "arm_command_difference_scalar_count": arm_differences,
                "wrist_command_difference_scalar_count": wrist_differences,
                "unexpected_arm_difference_scalar_count": unexpected_arm_differences,
                "remaining_arm_hard_limit_violation_scalar_count": remaining_arm_violations,
                "maximum_arm_hard_limit_correction_rad": maximum_arm_correction,
                "maximum_adjacent_dex3_target_step_rad": adjacent,
                "left_grasp_primitive_triggered": controller.left_trigger is not None,
                "right_handoff_primitive_triggered": controller.right_trigger is not None,
                "left_release_transition_triggered": controller.left_release_frame is not None,
                "right_release_transition_triggered": controller.right_release_frame is not None,
            }
        )
    endpoint_violations = sum(row["hard_limit_violation_scalar_count"] for row in endpoint_rows)
    transition_violations = sum(row["hard_limit_violation_scalar_count"] for row in transition_rows)
    triggers_complete = all(
        row["left_grasp_primitive_triggered"]
        and row["right_handoff_primitive_triggered"]
        and row["left_release_transition_triggered"]
        and row["right_release_transition_triggered"]
        for row in rollout_rows
    )
    passed = bool(
        endpoint_violations == 0
        and transition_violations == 0
        and total_violations == 0
        and total_unexpected_arm_differences == 0
        and total_remaining_arm_violations == 0
        and triggers_complete
    )
    value = {
        "schema_version": "limit_safe_common_dex3_validation_v1",
        "status": "PASS" if passed else "FAIL",
        "scope": "STATIC_ENDPOINTS_FULL_TRANSITIONS_AND_EXACT_ACT_A_B_EVAL35_COMMAND_TIMELINES",
        "authoritative_joint_limit_contract": str(JOINT_CONTRACT.resolve()),
        "authoritative_joint_limit_contract_sha256": sha256_file(JOINT_CONTRACT),
        "dex3_joint_names": list(dex3_names),
        "dex3_lower_rad": lower.tolist(),
        "dex3_upper_rad": upper.tolist(),
        "endpoint_checks": endpoint_rows,
        "transition_checks": transition_rows,
        "rollout_count": len(rollout_rows),
        "commanded_frame_count": int(sum(row["frames"] for row in rollout_rows)),
        "commanded_dex3_scalar_count": total_dex3_scalars,
        "endpoint_hard_limit_violation_scalar_count": endpoint_violations,
        "transition_hard_limit_violation_scalar_count": transition_violations,
        "exact_rollout_hard_limit_violation_scalar_count": total_violations,
        "arm_command_difference_scalar_count": total_arm_differences,
        "wrist_command_difference_scalar_count": total_wrist_differences,
        "unexpected_arm_command_difference_scalar_count": total_unexpected_arm_differences,
        "remaining_arm_hard_limit_violation_scalar_count": total_remaining_arm_violations,
        "maximum_arm_hard_limit_correction_rad": maximum_arm_hard_limit_correction,
        "common_arm_hard_limit_projector": "NEAREST_VALID_VALUE_COMPONENTWISE",
        "maximum_adjacent_dex3_target_step_rad": maximum_dex3_step,
        "both_hands_all_primitive_transitions_triggered": triggers_complete,
        "common_for_a_b": True,
        "method_specific_parameters": False,
        "episode_specific_parameters": False,
        "rollouts": rollout_rows,
    }
    atomic_json(RESULT, value)
    REPORT.write_text(
        "# Limit-safe common Dex3 validation\n\n"
        f"Status: **{value['status']}**\n\n"
        f"- Endpoint violations: **{endpoint_violations}**\n"
        f"- Full-transition violations: **{transition_violations}**\n"
        f"- Exact A/B EVAL35 commanded Dex3 scalars checked: **{total_dex3_scalars}**\n"
        f"- Exact-timeline Dex3 hard-limit violations: **{total_violations}**\n"
        f"- Required arm hard-limit projections: **{total_arm_differences}**\n"
        f"- Required wrist-subset projections: **{total_wrist_differences}**\n"
        f"- Unexpected arm changes: **{total_unexpected_arm_differences}**\n"
        f"- Remaining arm hard-limit violations: **{total_remaining_arm_violations}**\n"
        f"- Maximum arm hard-limit correction: **{maximum_arm_hard_limit_correction:.9f} rad**\n"
        f"- Both-hand grasp/handoff/release transitions covered: **{'YES' if triggers_complete else 'NO'}**\n"
        f"- Maximum adjacent Dex3 target step: **{maximum_dex3_step:.9f} rad**\n\n"
        "The validation uses only the authoritative measured Dex3 hard-limit contract. "
        "No A/B outcome is used to set a target or bound.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value[key] for key in (
        "status",
        "rollout_count",
        "commanded_frame_count",
        "commanded_dex3_scalar_count",
        "endpoint_hard_limit_violation_scalar_count",
        "transition_hard_limit_violation_scalar_count",
        "exact_rollout_hard_limit_violation_scalar_count",
        "arm_command_difference_scalar_count",
        "wrist_command_difference_scalar_count",
        "unexpected_arm_command_difference_scalar_count",
        "remaining_arm_hard_limit_violation_scalar_count",
        "maximum_arm_hard_limit_correction_rad",
        "maximum_adjacent_dex3_target_step_rad",
    )}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
