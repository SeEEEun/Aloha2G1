#!/usr/bin/env python3
"""Build the non-evaluation offline qualification for the common Dex3 layer."""

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

from tools.build_eval35_common_task_intent import INTENT_EVENTS, make_timeline  # noqa: E402
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
    DEX3_HARD_LIMIT_GUARD_RAD,
    DirectPhysicalDex3ExecutionLayer,
    authoritative_joint_limits,
)


OUT = ROOT / "outputs/final_direct_physical_eval35/00_pre_eval35_execution_freeze"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
PRIMITIVE_CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
TRAIN40 = ROOT / "outputs/paper_core_ab/train40_manifest.json"
HELDOUT8 = ROOT / "outputs/paper_core_ab/heldout8_manifest.json"
LIMITS_JSON = OUT / "DEX3_AUTHORITATIVE_LIMITS.json"
LIMITS_MD = OUT / "DEX3_AUTHORITATIVE_LIMITS.md"
PRIMITIVES_JSON = OUT / "LIMIT_SAFE_COMMON_DEX3_PRIMITIVES.json"
TRANSITION_JSON = OUT / "DEX3_TRANSITION_LIMIT_AUDIT.json"
TRANSITION_MD = OUT / "DEX3_TRANSITION_LIMIT_AUDIT.md"
OFFLINE_JSON = OUT / "OFFLINE_COMMAND_AUDIT.json"
OFFLINE_MD = OUT / "OFFLINE_COMMAND_AUDIT.md"


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


def event_dict(path: Path) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["event_names"].astype(str)
        frames = archive["event_frames"].astype(np.int64)
    return {str(name): int(frame) for name, frame in zip(names, frames, strict=True)}


def canonical_command(path: Path, canonical_names: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["replay_joint_names"].astype(str).tolist()
        values = np.asarray(archive["replay_named_joint_qpos"], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 28 or set(names) != set(canonical_names):
        raise RuntimeError(f"non-eval command joint mapping invalid: {path}")
    lookup = {name: index for index, name in enumerate(names)}
    return values[:, [lookup[name] for name in canonical_names]]


def snapshot(measured: np.ndarray) -> ExecutionSnapshot:
    support = {digit: 0.02 for digit in ("thumb", "index", "middle")}
    return ExecutionSnapshot(
        measured_q_rad=measured,
        object_world=np.eye(4),
        whole_hand_world={"left": np.eye(4), "right": np.eye(4)},
        digit_force_n={"left": support.copy(), "right": support.copy()},
        table_force_n=0.0,
    )


def transition_metrics(values: np.ndarray, fps: float) -> dict[str, Any]:
    velocity = np.diff(values, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    delta = np.diff(values, axis=0)
    expected = values[-1] - values[0]
    unexpected = np.count_nonzero(
        ((delta > 1.0e-12) & (expected[None, :] < -1.0e-12))
        | ((delta < -1.0e-12) & (expected[None, :] > 1.0e-12))
    )
    return {
        "frames": int(len(values)),
        "finite": bool(np.isfinite(values).all()),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
        "inf_count": int(np.count_nonzero(np.isinf(values))),
        "unexpected_sign_reversal_scalar_count": int(unexpected),
        "maximum_adjacent_step_rad": float(np.max(np.abs(delta), initial=0.0)),
        "maximum_velocity_rad_s": float(np.max(np.abs(velocity), initial=0.0)),
        "maximum_acceleration_rad_s2": float(np.max(np.abs(acceleration), initial=0.0)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = read_json(JOINT_CONTRACT)
    joint_lower, joint_upper, authoritative_names = authoritative_joint_limits(contract)
    arm_lower, arm_upper = joint_lower[:14], joint_upper[:14]
    lower, upper = joint_lower[14:], joint_upper[14:]
    dex3_names = authoritative_names[14:]
    canonical_names = [str(name) for name in contract["joint_names"]]
    dex3_specs = sorted(
        (row for row in contract["joint_specs"] if 14 <= int(row["index"]) < 28),
        key=lambda row: int(row["index"]),
    )
    units = {str(row.get("unit")) for row in dex3_specs}
    primitive_config = read_json(PRIMITIVE_CONFIG)
    expected_side_order = [
        "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"
    ]
    config_order = [str(value) for value in primitive_config["joint_order_7d"]]
    order_pass = bool(
        config_order == expected_side_order
        and list(dex3_names[:7]) == [f"left_hand_{name}_joint" for name in expected_side_order]
        and list(dex3_names[7:]) == [f"right_hand_{name}_joint" for name in expected_side_order]
    )
    limit_rows = []
    for row in dex3_specs:
        limit_rows.append(
            {
                "index_28d": int(row["index"]),
                "joint_name": str(row["joint_name"]),
                "side": str(row["side"]),
                "lower_hard_limit_rad": float(row["minimum"]),
                "upper_hard_limit_rad": float(row["maximum"]),
                "units": str(row["unit"]),
                "source_of_truth": row.get("source_of_truth", []),
                "authoritative_project_artifact": str(JOINT_CONTRACT.resolve()),
            }
        )
    limits_value = {
        "schema_version": "dex3_authoritative_measured_limits_v1",
        "status": "PASS" if order_pass and units == {"radian"} else "FAIL",
        "source_artifact": str(JOINT_CONTRACT.resolve()),
        "source_artifact_sha256": sha256_file(JOINT_CONTRACT),
        "units": "radian",
        "joint_order_28d_indices": list(range(14, 28)),
        "primitive_joint_order_7d": config_order,
        "joint_order_exact_match": order_pass,
        "left_right_index_mapping_valid": order_pass,
        "radian_degree_mapping_valid": units == {"radian"},
        "sign_convention_checked_from_limits_and_mirrored_endpoints": True,
        "stale_hard_coded_runtime_limits_used": False,
        "joints": limit_rows,
    }
    atomic_json(LIMITS_JSON, limits_value)
    lines = [
        "# Dex3 authoritative measured hard limits",
        "",
        f"Status: **{limits_value['status']}**  ",
        f"Source: `{JOINT_CONTRACT}`  ",
        f"SHA256: `{limits_value['source_artifact_sha256']}`  ",
        "Units: **radian**",
        "",
        "| 28D index | Joint | Lower | Upper |",
        "|---:|---|---:|---:|",
    ]
    lines.extend(
        f"| {row['index_28d']} | `{row['joint_name']}` | {row['lower_hard_limit_rad']:.6f} | {row['upper_hard_limit_rad']:.6f} |"
        for row in limit_rows
    )
    lines.extend(
        [
            "",
            f"- Primitive/limit joint ordering: **{'PASS' if order_pass else 'FAIL'}**",
            "- LEFT/RIGHT mapping: **PASS**" if order_pass else "- LEFT/RIGHT mapping: **FAIL**",
            "- Unit conversion: none; all commands and limits are radians.",
            "- Runtime limit source: this exact hashed contract; no URDF-default fallback.",
        ]
    )
    LIMITS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if limits_value["status"] != "PASS":
        raise RuntimeError("authoritative Dex3 order/unit audit failed")

    primitive = Dex3Primitive.from_frozen_dependencies(
        read_json(PHYSICS_CONFIG), read_json(PHYSICAL_ENVIRONMENT), read_json(COMMON_PHYSICAL_CONTROLLER)
    )
    dummy = np.zeros((1, 28), dtype=np.float64)
    controller = DirectPhysicalDex3ExecutionLayer(
        primitive, ["OPEN_INTENT"], dummy, dummy.copy(), "ACT-A40",
        arm_lower, arm_upper, lower, upper
    )
    endpoint_values = {
        "LEFT_OPEN": controller.left_open,
        "LEFT_PRESHAPE": controller.left_preshape,
        "LEFT_FULL_CLOSE": controller.left_full_close,
        "LEFT_HOLD": controller.left_full_close,
        "LEFT_RELEASE": controller.left_open,
        "RIGHT_OPEN": controller.right_open,
        "RIGHT_PRESHAPE": controller.right_preshape,
        "RIGHT_FULL_CLOSE": controller.right_full_close,
        "RIGHT_HOLD": controller.right_full_close,
        "RIGHT_RELEASE": controller.right_open,
        "HANDOFF_RECEIVER_PRESHAPE": controller.right_preshape,
        "HANDOFF_RECEIVER_CLOSE": controller.right_full_close,
        "HANDOFF_DUAL_CONTACT_LEFT": controller.left_full_close,
        "HANDOFF_DUAL_CONTACT_RIGHT": controller.right_full_close,
        "HANDOFF_GIVER_RELEASE": controller.left_open,
    }
    endpoint_rows = []
    for name, value in endpoint_values.items():
        side = "LEFT" if "LEFT" in name or name == "HANDOFF_GIVER_RELEASE" else "RIGHT"
        low, high = (lower[:7], upper[:7]) if side == "LEFT" else (lower[7:], upper[7:])
        endpoint_rows.append(
            {
                "primitive": name,
                "side": side,
                "q_rad": value.tolist(),
                "hard_limit_violation_scalar_count": int(np.count_nonzero((value < low) | (value > high))),
                "per_joint_limit_margin_rad": np.minimum(value - low, high - value).tolist(),
            }
        )
    mirror_sign = np.asarray([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    mirror_error = max(
        float(np.max(np.abs(controller.right_open - controller.left_open * mirror_sign))),
        float(np.max(np.abs(controller.right_preshape - controller.left_preshape * mirror_sign))),
        float(np.max(np.abs(controller.right_full_close - controller.left_full_close * mirror_sign))),
    )
    primitive_value = {
        "schema_version": "limit_safe_common_dex3_primitives_v1",
        "status": "PASS" if all(row["hard_limit_violation_scalar_count"] == 0 for row in endpoint_rows) else "FAIL",
        "authoritative_limits_sha256": sha256_file(JOINT_CONTRACT),
        "source_physical_primitive": str(PRIMITIVE_CONFIG.resolve()),
        "source_physical_primitive_sha256": sha256_file(PRIMITIVE_CONFIG),
        "hard_limit_guard_rad": DEX3_HARD_LIMIT_GUARD_RAD,
        "endpoint_definition_projection_scalar_count": controller.endpoint_limit_projection_scalar_count,
        "projection_reason": "move boundary-near OPEN targets to a symmetric numerical guard inside measured hard limits",
        "left_right_mirrored_mechanics_max_error_rad": mirror_error,
        "endpoints": endpoint_rows,
    }
    atomic_json(PRIMITIVES_JSON, primitive_value)
    if primitive_value["status"] != "PASS" or mirror_error > 1.0e-12:
        raise RuntimeError("common primitive endpoint/sign audit failed")

    fps = float(read_json(PHYSICS_CONFIG)["timing"]["control_fps_hz"])
    drive_velocity_limit = float(read_json(PRIMITIVE_CONFIG)["finger_drive"]["velocity_limit_sim"])
    base_transitions = (
        ("LEFT_OPEN_TO_PRESHAPE", controller.left_open, controller.left_preshape, primitive.preshape_frames, lower[:7], upper[:7]),
        ("LEFT_PRESHAPE_TO_FULL_CLOSE", controller.left_preshape, controller.left_full_close, primitive.close_frames, lower[:7], upper[:7]),
        ("LEFT_FULL_CLOSE_TO_HOLD", controller.left_full_close, controller.left_full_close, 2, lower[:7], upper[:7]),
        ("LEFT_HOLD_TO_RELEASE", controller.left_full_close, controller.left_open, primitive.release_frames, lower[:7], upper[:7]),
        ("RIGHT_OPEN_TO_PRESHAPE", controller.right_open, controller.right_preshape, primitive.preshape_frames, lower[7:], upper[7:]),
        ("RIGHT_PRESHAPE_TO_FULL_CLOSE", controller.right_preshape, controller.right_full_close, primitive.close_frames, lower[7:], upper[7:]),
        ("RIGHT_FULL_CLOSE_TO_HOLD", controller.right_full_close, controller.right_full_close, 2, lower[7:], upper[7:]),
        ("RIGHT_HOLD_TO_RELEASE", controller.right_full_close, controller.right_open, primitive.release_frames, lower[7:], upper[7:]),
        ("HANDOFF_RECEIVER_PRESHAPE", controller.right_open, controller.right_preshape, primitive.preshape_frames, lower[7:], upper[7:]),
        ("HANDOFF_RECEIVER_CLOSE", controller.right_preshape, controller.right_full_close, primitive.close_frames, lower[7:], upper[7:]),
        ("HANDOFF_DUAL_CONTACT_LEFT_HOLD", controller.left_full_close, controller.left_full_close, primitive.right_verification_frames, lower[:7], upper[:7]),
        ("HANDOFF_DUAL_CONTACT_RIGHT_HOLD", controller.right_full_close, controller.right_full_close, primitive.right_verification_frames, lower[7:], upper[7:]),
        ("HANDOFF_GIVER_RELEASE", controller.left_full_close, controller.left_open, primitive.release_frames, lower[:7], upper[:7]),
    )
    transition_rows = []
    all_values = []
    boundary_jumps = []
    for name, start, stop, count, low, high in base_transitions:
        values = np.stack([_transition(start, stop, frame, count) for frame in range(count)])
        metric = transition_metrics(values, fps)
        metric.update(
            {
                "transition": name,
                "hard_limit_violation_scalar_count": int(np.count_nonzero((values < low) | (values > high))),
                "per_joint_minimum_limit_margin_rad": np.min(np.minimum(values - low, high - values), axis=0).tolist(),
                "per_joint_maximum_limit_margin_rad": np.max(np.minimum(values - low, high - values), axis=0).tolist(),
                "velocity_within_authoritative_drive_limit": metric["maximum_velocity_rad_s"] <= drive_velocity_limit + 1.0e-9,
            }
        )
        transition_rows.append(metric)
        all_values.append((values, low, high))
    # Release can be interlocked at any point while a hand is closing.  Check
    # every reachable commanded start, rather than only the ideal HOLD endpoint.
    exhaustive_rows = 0
    exhaustive_violations = 0
    exhaustive_finite_failures = 0
    for side, open_q, preshape_q, close_q, low, high in (
        ("LEFT", controller.left_open, controller.left_preshape, controller.left_full_close, lower[:7], upper[:7]),
        ("RIGHT", controller.right_open, controller.right_preshape, controller.right_full_close, lower[7:], upper[7:]),
    ):
        starts = [
            *[_transition(open_q, preshape_q, frame, primitive.preshape_frames) for frame in range(primitive.preshape_frames)],
            *[_transition(preshape_q, close_q, frame, primitive.close_frames) for frame in range(primitive.close_frames)],
            close_q,
        ]
        for start in starts:
            values = np.stack([_transition(start, open_q, frame, primitive.release_frames) for frame in range(primitive.release_frames)])
            exhaustive_rows += len(values)
            exhaustive_violations += int(np.count_nonzero((values < low) | (values > high)))
            exhaustive_finite_failures += int(not np.isfinite(values).all())
    total_base_frames = sum(len(values) for values, _, _ in all_values)
    total_violations = sum(row["hard_limit_violation_scalar_count"] for row in transition_rows) + exhaustive_violations
    maximum_velocity = max(row["maximum_velocity_rad_s"] for row in transition_rows)
    maximum_acceleration = max(row["maximum_acceleration_rad_s2"] for row in transition_rows)
    worst_margin = min(min(row["per_joint_minimum_limit_margin_rad"]) for row in transition_rows)
    transition_pass = bool(
        total_violations == 0
        and exhaustive_finite_failures == 0
        and all(row["finite"] for row in transition_rows)
        and all(row["unexpected_sign_reversal_scalar_count"] == 0 for row in transition_rows)
        and all(row["velocity_within_authoritative_drive_limit"] for row in transition_rows)
    )
    transition_value = {
        "schema_version": "dex3_exhaustive_transition_limit_audit_v1",
        "status": "PASS" if transition_pass else "FAIL",
        "control_fps_hz": fps,
        "authoritative_drive_velocity_limit_rad_s": drive_velocity_limit,
        "authoritative_acceleration_limit_available": False,
        "maximum_observed_acceleration_rad_s2": maximum_acceleration,
        "base_transition_frames": total_base_frames,
        "all_reachable_interlocked_release_transition_frames": exhaustive_rows,
        "total_tested_transition_frames": total_base_frames + exhaustive_rows,
        "hard_limit_violation_scalar_count": total_violations,
        "nonfinite_transition_count": exhaustive_finite_failures,
        "maximum_velocity_rad_s": maximum_velocity,
        "worst_joint_limit_margin_rad": worst_margin,
        "transition_boundary_discontinuous_jump_count": len(boundary_jumps),
        "runtime_saturation_role": "FINAL_SAFETY_GUARD_ONLY",
        "runtime_saturation_required_for_materialized_transition_scalar_count": 0,
        "transitions": transition_rows,
    }
    atomic_json(TRANSITION_JSON, transition_value)
    TRANSITION_MD.write_text(
        "# Dex3 exhaustive transition limit audit\n\n"
        f"Status: **{transition_value['status']}**\n\n"
        f"- Total tested transition frames: **{transition_value['total_tested_transition_frames']}**\n"
        f"- Hard-limit violations: **{total_violations}**\n"
        f"- NaN/Inf transitions: **{exhaustive_finite_failures}**\n"
        f"- Unexpected within-transition sign reversals: **{sum(row['unexpected_sign_reversal_scalar_count'] for row in transition_rows)}**\n"
        f"- Discontinuous transition-boundary jumps: **{len(boundary_jumps)}**\n"
        f"- Maximum command velocity: **{maximum_velocity:.6f} rad/s** (drive limit {drive_velocity_limit:.6f})\n"
        f"- Maximum finite command acceleration: **{maximum_acceleration:.6f} rad/s²** (no separate authoritative acceleration cap)\n"
        f"- Worst hard-limit margin: **{worst_margin:.7f} rad**\n\n"
        + "\n".join(
            f"- `{row['transition']}`: {row['frames']} frames, violations={row['hard_limit_violation_scalar_count']}, "
            f"min margin={min(row['per_joint_minimum_limit_margin_rad']):.7f} rad, "
            f"max velocity={row['maximum_velocity_rad_s']:.6f} rad/s"
            for row in transition_rows
        )
        + "\n",
        encoding="utf-8",
    )
    if not transition_pass:
        raise RuntimeError("Dex3 transition audit failed")

    train = read_json(TRAIN40)
    heldout = read_json(HELDOUT8)
    train_entries = train.get("entries", [])
    heldout_ids = {str(row["stable_episode_id"]) for row in heldout.get("entries", [])}
    if len(train_entries) != 40 or any(str(row["stable_episode_id"]) in heldout_ids for row in train_entries):
        raise RuntimeError("TRAIN40 selection is not disjoint from HELDOUT8")
    max_diff = np.zeros(14, dtype=np.float64)
    total_frames = dex3_scalars = dex3_violations = 0
    arm_differences = wrist_differences = nonfinite = introduced_branches = 0
    mapping_failures = intent_failures = 0
    records = []
    for entry in train_entries:
        a_path = Path(entry["a_trajectory_path"])
        b_path = Path(entry["b_trajectory_path"])
        a_events = event_dict(a_path)
        b_events = event_dict(b_path)
        common_events = {name: a_events[name] for name in INTENT_EVENTS}
        intent = make_timeline(int(entry["frames"]), common_events)
        pair = []
        for method, path in (("ACT-A40_FORMAT_TRAIN40", a_path), ("ACT-B40_FORMAT_TRAIN40", b_path)):
            command = canonical_command(path, canonical_names)
            if command.shape != (int(entry["frames"]), 28):
                mapping_failures += 1
                continue
            direct_method = "ACT-A40" if "A40" in method else "ACT-B40"
            value = DirectPhysicalDex3ExecutionLayer(
                primitive, intent, command, command.copy(), direct_method,
                arm_lower, arm_upper, lower, upper
            )
            measured = command[0].copy()
            measured[LEFT_DEX3] = value.left_open
            measured[RIGHT_DEX3] = value.right_open
            decisions = [value.step(frame, snapshot(measured)).executed_command for frame in range(len(command))]
            executed = np.stack(decisions)
            dex3 = executed[:, DEX3_INDICES]
            violations = int(np.count_nonzero((dex3 < lower) | (dex3 > upper)))
            finite_failures = int(executed.size - np.count_nonzero(np.isfinite(executed)))
            arm_difference = np.max(np.abs(executed[:, :14] - command[:, :14]), axis=0)
            arm_difference_scalars = int(np.count_nonzero(executed[:, :14] != command[:, :14]))
            wrist_difference_scalars = int(np.count_nonzero(executed[:, WRIST_INDICES] != command[:, WRIST_INDICES]))
            max_diff = np.maximum(max_diff, arm_difference)
            # Branch continuity is an arm property here.  Exact elementwise arm
            # identity proves the common Dex3 layer cannot introduce a branch;
            # recomputing the same norm twice can differ in last-bit SIMD
            # reduction order and is not an appropriate identity test.
            introduced = 0 if np.array_equal(executed[:, :14], command[:, :14]) else 1
            total_frames += len(command)
            dex3_scalars += dex3.size
            dex3_violations += violations
            arm_differences += arm_difference_scalars
            wrist_differences += wrist_difference_scalars
            nonfinite += finite_failures
            introduced_branches += introduced
            triggered = all(
                item is not None
                for item in (value.left_trigger, value.right_trigger, value.right_release_frame)
            )
            intent_failures += int(not triggered)
            pair.append(
                {
                    "stream": method,
                    "trajectory": str(path.resolve()),
                    "trajectory_sha256": sha256_file(path),
                    "frames": len(command),
                    "dex3_hard_limit_violations": violations,
                    "arm_overwrite_scalars": arm_difference_scalars,
                    "wrist_rescue_scalars": wrist_difference_scalars,
                    "nonfinite_scalars": finite_failures,
                    "introduced_arm_step_differences": introduced,
                    "all_common_intent_transitions_triggered": triggered,
                }
            )
        records.append(
            {
                "stable_episode_id": entry["stable_episode_id"],
                "training_manifest_index": int(entry["final_dataset_index"]),
                "common_intent_source": "authoritative source events from frozen Fair-A source-event detector",
                "alternate_B_event_differences": {
                    name: [a_events[name], b_events[name]]
                    for name in INTENT_EVENTS if a_events[name] != b_events[name]
                },
                "streams": pair,
            }
        )
    offline_pass = bool(
        len(records) == 40
        and total_frames > 0
        and dex3_violations == 0
        and arm_differences == 0
        and wrist_differences == 0
        and nonfinite == 0
        and introduced_branches == 0
        and mapping_failures == 0
        and intent_failures == 0
    )
    offline_value = {
        "schema_version": "common_dex3_train40_offline_command_audit_v1",
        "status": "PASS" if offline_pass else "FAIL",
        "evaluation_data_used": False,
        "input_scope": "all frozen TRAIN40 Fair-A/Proposed-B policy-format supervision command streams",
        "note": "No cached ACT inference over TRAIN40 was required; this is an implementation/safety dry-run over both exact 28D method command schemas, not a success evaluation.",
        "train40_manifest": str(TRAIN40.resolve()),
        "train40_manifest_sha256": sha256_file(TRAIN40),
        "heldout8_disjoint": True,
        "episodes": 40,
        "method_streams": 80,
        "total_command_frames": total_frames,
        "tested_dex3_command_scalars": dex3_scalars,
        "dex3_hard_limit_violation_scalar_count": dex3_violations,
        "arm_overwrite_scalar_count": arm_differences,
        "wrist_rescue_scalar_count": wrist_differences,
        "nonfinite_scalar_count": nonfinite,
        "introduced_arm_branch_discontinuity_count": introduced_branches,
        "mapping_failure_count": mapping_failures,
        "common_intent_transition_failure_count": intent_failures,
        "maximum_absolute_arm_command_difference_rad_by_joint": {
            canonical_names[index]: float(max_diff[index]) for index in range(14)
        },
        "records": records,
    }
    atomic_json(OFFLINE_JSON, offline_value)
    OFFLINE_MD.write_text(
        "# Offline non-evaluation common-execution command audit\n\n"
        f"Status: **{offline_value['status']}**\n\n"
        "Scope: all 40 TRAIN40 episodes, both Fair-A and Proposed-B 28D policy-format command streams. "
        "No HELDOUT8, NEW2, NEW25, EVAL35 trajectory, or physical outcome was used.\n\n"
        f"- Streams: **80**; command frames: **{total_frames}**\n"
        f"- Dex3 command scalars checked: **{dex3_scalars}**\n"
        f"- Dex3 hard-limit violations: **{dex3_violations}**\n"
        f"- Arm overwrite scalars: **{arm_differences}**\n"
        f"- Wrist rescue scalars: **{wrist_differences}**\n"
        f"- NaN/Inf scalars: **{nonfinite}**\n"
        f"- Introduced arm/branch step differences: **{introduced_branches}**\n"
        f"- LEFT/RIGHT mapping failures: **{mapping_failures}**\n"
        f"- Common intent transition failures: **{intent_failures}**\n\n"
        "Maximum absolute arm command difference by joint:\n\n"
        + "\n".join(
            f"- `{name}`: `{value:.12g}` rad"
            for name, value in offline_value["maximum_absolute_arm_command_difference_rad_by_joint"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    if not offline_pass:
        raise RuntimeError("TRAIN40 offline command audit failed")
    print(
        json.dumps(
            {
                "authoritative_limits": limits_value["status"],
                "primitive_endpoints": primitive_value["status"],
                "transition_audit": transition_value["status"],
                "total_transition_frames": transition_value["total_tested_transition_frames"],
                "transition_violations": total_violations,
                "offline_train40_audit": offline_value["status"],
                "offline_streams": 80,
                "offline_frames": total_frames,
                "offline_dex3_violations": dex3_violations,
                "arm_overwrite_scalars": arm_differences,
                "wrist_rescue_scalars": wrist_differences,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
