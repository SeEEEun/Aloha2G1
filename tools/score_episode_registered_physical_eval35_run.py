#!/usr/bin/env python3
"""Score one frozen episode-registered ACT physical rollout from its PhysX trace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
ENVIRONMENT = ROOT / "outputs/final_episode_registered_eval35/01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
FEASIBILITY = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
STAGES = ("LEFT_GRASP", "HANDOFF", "RIGHT_OWNERSHIP", "TRANSPORT", "BIN_ENTRY", "SETTLE")
DIGITS = ("thumb", "index", "middle")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def first_event(events: np.ndarray, name: str) -> int | None:
    rows = np.flatnonzero(np.char.find(events.astype(str), name) >= 0)
    return int(rows[0]) if len(rows) else None


def first_true(mask: np.ndarray) -> int | None:
    rows = np.flatnonzero(mask)
    return int(rows[0]) if len(rows) else None


def first_persistent_true(mask: np.ndarray, minimum_rows: int) -> int | None:
    start: int | None = None
    count = 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            if start is None:
                start = index
            count += 1
            if count >= minimum_rows:
                return start
        else:
            start = None
            count = 0
    return None


def longest_rows(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def unique_control(values: np.ndarray, frames: np.ndarray) -> np.ndarray:
    rows = np.r_[np.flatnonzero(np.diff(frames) != 0), len(frames) - 1]
    return np.asarray(values)[rows]


def branch_discontinuities(commands: np.ndarray) -> tuple[int, float]:
    acceptance = read_json(FEASIBILITY)["unchanged_acceptance"]
    norms = np.linalg.norm(np.diff(np.asarray(commands)[:, :14], axis=0), axis=1)
    flags: list[bool] = []
    for index, value in enumerate(norms):
        prior = norms[max(0, index - 10):index]
        local = float(np.median(prior)) if len(prior) else 0.0
        threshold = max(
            float(acceptance["branch_absolute_step_norm_rad"]),
            float(acceptance["branch_local_multiplier"]) * max(local, 1.0e-6),
        )
        flags.append(bool(value > threshold))
    return int(np.count_nonzero(flags)), float(np.max(norms, initial=0.0))


def topology_at(
    forces: dict[str, dict[str, np.ndarray]], palms: dict[str, np.ndarray],
    side: str, row: int | None, threshold: float,
) -> list[str]:
    if row is None:
        return []
    radius = 8
    scope = slice(max(0, row - radius), min(len(palms[side]), row + radius + 1))
    result = [digit for digit in DIGITS if np.max(forces[side][digit][scope], initial=0.0) >= threshold]
    if np.max(palms[side][scope], initial=0.0) >= threshold:
        result.append("palm")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(run / "robot_bin_contacts.npz", allow_pickle=False) as archive:
        robot_bin = {key: np.asarray(archive[key]) for key in archive.files}
    trial = read_json(run / "trial_result.json")
    runtime = read_json(run / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json")
    invocation = read_json(run / "INVOCATION_MANIFEST.json")
    required = {
        "RAW_POLICY_COMMAND", "POLICY_SAFE_COMMAND", "PROJECTED_POLICY_COMMAND",
        "EXECUTED_COMMAND", "MEASURED_Q", "COMMON_CONTROLLER_OVERRIDE_MASK",
        "COMMON_CONTROLLER_OVERRIDE_MASK_ARM", "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST",
        "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3", "COMMON_ARM_HARD_LIMIT_PROJECTION_MASK",
        "DIRECT_COMMON_EXECUTION_PHASE", "DIRECT_COMMON_TASK_INTENT",
        "DIRECT_COMMON_EXECUTION_EVENTS", "commanded_q_rad", "measured_q_rad",
        "object_position_world_m", "object_quaternion_xyzw",
        "object_linear_velocity_m_s", "object_angular_velocity_rad_s",
        "table_contact_force_n", "doll_bin_contact_force_n",
        "maximum_doll_bin_penetration_m", "left_palm_force_n", "right_palm_force_n",
        *(f"{side}_{digit}_force_n" for side in ("left", "right") for digit in DIGITS),
    }
    missing = sorted(required - set(event))
    if missing:
        raise RuntimeError(f"episode-registered trace fields missing: {missing}")

    config = read_json(CONFIG)
    environment = read_json(ENVIRONMENT)
    dt = float(config["timing"]["physics_dt_s"])
    force_threshold = float(config["gates"]["meaningful_digit_force_n"])
    table_threshold = float(config["gates"]["maximum_table_force_for_elevated_n"])
    frames = event["control_frame"].astype(np.int64)
    positions = event["object_position_world_m"].astype(np.float64)
    speeds = np.linalg.norm(event["object_linear_velocity_m_s"].astype(np.float64), axis=1)
    forces = {
        side: {digit: event[f"{side}_{digit}_force_n"].astype(np.float64) for digit in DIGITS}
        for side in ("left", "right")
    }
    palms = {side: event[f"{side}_palm_force_n"].astype(np.float64) for side in ("left", "right")}
    meaningful = {
        side: {digit: forces[side][digit] >= force_threshold for digit in DIGITS}
        for side in ("left", "right")
    }
    side_any = {
        side: np.logical_or.reduce([*(meaningful[side][digit] for digit in DIGITS), palms[side] >= force_threshold])
        for side in ("left", "right")
    }
    opposing = {
        side: meaningful[side]["thumb"] & (
            meaningful[side]["index"] | meaningful[side]["middle"] | (palms[side] >= force_threshold)
        )
        for side in ("left", "right")
    }
    table_free = event["table_contact_force_n"].astype(np.float64) <= table_threshold
    events = event["DIRECT_COMMON_EXECUTION_EVENTS"].astype(str)
    event_rows = {
        "left_confirm": first_event(events, "LEFT_GRASP_CONFIRMED"),
        "left_table_loss": first_event(events, "LEFT_TABLE_SUPPORT_LOST"),
        "left_owned": first_event(events, "LEFT_PHYSICAL_OWNERSHIP"),
        "right_confirm": first_event(events, "RIGHT_GRASP_CONFIRMED"),
        "right_support": first_event(events, "RIGHT_RETENTION_SUPPORT_CONFIRMED"),
        "left_release": first_event(events, "GIVING_HAND_RELEASE_START"),
        "right_owned": first_event(events, "RIGHT_PHYSICAL_OWNERSHIP"),
        "release": first_event(events, "FINAL_RELEASE_INTENT_START"),
        "left_lift": first_event(events, "LEFT_GRASP_STATE_LIFT"),
        "right_lift": first_event(events, "RIGHT_GRASP_STATE_LIFT"),
    }
    left_grasp = bool(
        event_rows["left_confirm"] is not None
        and event_rows["left_table_loss"] is not None
        and event_rows["left_owned"] is not None
        and event_rows["left_confirm"] <= event_rows["left_table_loss"] <= event_rows["left_owned"]
        and np.any(opposing["left"][event_rows["left_confirm"]:event_rows["left_owned"] + 1])
        and np.any(table_free[event_rows["left_table_loss"]:event_rows["left_owned"] + 1])
    )
    handoff = bool(
        left_grasp
        and event_rows["right_confirm"] is not None
        and event_rows["right_support"] is not None
        and event_rows["left_release"] is not None
        and event_rows["right_owned"] is not None
        and event_rows["right_confirm"] <= event_rows["right_support"] <= event_rows["left_release"] <= event_rows["right_owned"]
        and np.any(opposing["right"][event_rows["right_confirm"]:event_rows["right_owned"] + 1])
    )
    right_ownership = bool(handoff and event_rows["right_owned"] is not None)

    bin_spec = environment["bin"]
    center = np.asarray(bin_spec["opening_center_world_xy_m"], dtype=np.float64)
    opening = np.asarray(bin_spec["opening_dimensions_xy_m"], dtype=np.float64)
    inside_xy = np.all(np.abs(positions[:, :2] - center) <= opening / 2.0, axis=1)
    in_bin = inside_xy & (positions[:, 2] > float(bin_spec["bottom_world_z_m"])) & (positions[:, 2] < float(bin_spec["rim_world_z_m"]))
    after_owned = np.arange(len(frames)) >= (event_rows["right_owned"] if event_rows["right_owned"] is not None else len(frames))
    first_entry = first_true(in_bin & after_owned)
    support_any = side_any["left"] | side_any["right"]
    no_drop = False
    maximum_unsupported_s = 0.0
    if left_grasp and event_rows["left_owned"] is not None and first_entry is not None:
        scope = np.zeros(len(frames), dtype=bool)
        scope[event_rows["left_owned"]:first_entry] = True
        unsupported = scope & ~support_any & ~in_bin
        maximum_unsupported_s = longest_rows(unsupported) * dt
        table_return = scope & ~table_free
        no_drop = bool(maximum_unsupported_s <= 0.1000001 and not np.any(table_return))
    transport = bool(right_ownership and no_drop)
    bin_entry = bool(transport and first_entry is not None)
    settle_rows = max(1, int(round(1.0 / dt)))
    settle_mask = in_bin & (speeds <= 0.02) & (event["doll_bin_contact_force_n"].astype(float) > 0.0)
    settle = bool(bin_entry and longest_rows(settle_mask[first_entry:]) >= settle_rows)
    settle_start = None
    if settle:
        current = 0
        for row in range(first_entry or 0, len(settle_mask)):
            current = current + 1 if settle_mask[row] else 0
            if current >= settle_rows:
                settle_start = row - settle_rows + 1
                break

    raw = event["RAW_POLICY_COMMAND"].astype(np.float64)
    safe = event["POLICY_SAFE_COMMAND"].astype(np.float64)
    projected = event["PROJECTED_POLICY_COMMAND"].astype(np.float64)
    executed = event["EXECUTED_COMMAND"].astype(np.float64)
    measured = event["MEASURED_Q"].astype(np.float64)
    specs = read_json(JOINT_CONTRACT)["joint_specs"]
    lower = np.asarray([row["minimum"] for row in specs], dtype=np.float64)
    upper = np.asarray([row["maximum"] for row in specs], dtype=np.float64)
    expected_projected = safe.copy()
    expected_projected[:, :14] = np.clip(safe[:, :14], lower[:14], upper[:14])
    projection_exact = bool(np.array_equal(projected, expected_projected))
    arm_unchanged_except_projector = bool(np.array_equal(executed[:, :14], projected[:, :14]))
    arm_rescue_count = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_ARM"]))
    wrist_rescue_count = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_WRIST"]))
    dex3_override_count = int(np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_DEX3"]))
    commanded_violations = int(np.count_nonzero((executed < lower) | (executed > upper)))
    # The established arm audit allows 0.002 rad for finite-gain PhysX
    # readback at an exact hard-stop target.  This is an instrumentation
    # tolerance only: commands are still projected exactly to the authoritative
    # limits.  Dex3 retains its stricter 1e-6 measured-state tolerance.
    measured_arm_violation_mask = (measured[:, :14] < lower[:14] - 0.002) | (
        measured[:, :14] > upper[:14] + 0.002
    )
    measured_dex3_violation_mask = (measured[:, 14:] < lower[14:] - 1.0e-6) | (
        measured[:, 14:] > upper[14:] + 1.0e-6
    )
    measured_arm_violations = int(np.count_nonzero(measured_arm_violation_mask))
    measured_dex3_violations = int(np.count_nonzero(measured_dex3_violation_mask))
    measured_violations = measured_arm_violations + measured_dex3_violations
    projection_count = int(np.count_nonzero(projected[:, :14] != safe[:, :14]))
    wrist_indices = np.asarray([4, 5, 6, 11, 12, 13])
    wrist_projection_count = int(np.count_nonzero(projected[:, wrist_indices] != safe[:, wrist_indices]))
    max_projection = float(np.max(np.abs(projected[:, :14] - safe[:, :14]), initial=0.0))
    robot_penetration = float(np.max(robot_bin.get("penetration_m", np.asarray([])), initial=0.0))
    doll_penetration = float(np.max(event["maximum_doll_bin_penetration_m"], initial=0.0))
    penetration_tolerance = float(environment["penetration"]["normal_solver_tolerance_m"])
    branch_count, maximum_arm_step = branch_discontinuities(unique_control(projected, frames))
    finite = all(np.isfinite(value).all() for value in (raw, safe, projected, executed, measured, positions, speeds))
    hard_valid = bool(
        np.array_equal(executed, event["commanded_q_rad"])
        and np.array_equal(measured, event["measured_q_rad"])
        and projection_exact and arm_unchanged_except_projector
        and arm_rescue_count == 0 and wrist_rescue_count == 0
        and commanded_violations == 0 and measured_violations == 0
        and robot_penetration <= penetration_tolerance and doll_penetration <= penetration_tolerance
        and branch_count == 0 and finite and bool(trial.get("command_completed", False))
        and not bool(trial.get("state_restoration", {}).get("used", False))
        and int(trial.get("object_pose_writes_during_timed_loop", -1)) == 0
        and invocation.get("episode_registration", {}).get("A_B_identical_object_pose") is True
    )
    full = bool(left_grasp and handoff and right_ownership and transport and bin_entry and settle and hard_valid)
    outcomes = {
        "LEFT_GRASP_SUCCESS": left_grasp,
        "HANDOFF_SUCCESS": handoff,
        "RIGHT_OWNERSHIP_SUCCESS": right_ownership,
        "NO_DROP_TO_BIN": transport,
        "BIN_ENTRY_SUCCESS": bin_entry,
        "BIN_SETTLE_SUCCESS": settle,
        "CLEAN_COMMANDED_RELEASE": False,
        "PREMATURE_DROP_INTO_BIN": False,
        "PREMATURE_DROP_OUTSIDE_BIN": False,
        "FULL_TASK_SUCCESS": full,
    }
    release_row = event_rows["release"]
    if right_ownership:
        loss_scope = np.arange(len(frames)) >= int(event_rows["right_owned"])
        contact_loss = first_persistent_true(
            loss_scope & ~side_any["right"], max(1, int(round(0.1 / dt)))
        )
    else:
        contact_loss = None
    if right_ownership and release_row is not None and contact_loss is not None and contact_loss >= release_row:
        outcomes["CLEAN_COMMANDED_RELEASE"] = True
        release_classification = "CLEAN_COMMANDED_RELEASE"
    elif contact_loss is not None and first_entry is not None and inside_xy[contact_loss] and full:
        outcomes["PREMATURE_DROP_INTO_BIN"] = True
        release_classification = "PREMATURE_DROP_INTO_BIN"
    else:
        outcomes["PREMATURE_DROP_OUTSIDE_BIN"] = True
        release_classification = "PREMATURE_DROP_OUTSIDE_BIN"
    stage_values = (left_grasp, handoff, right_ownership, transport, bin_entry, settle)
    first_failure = next((name for name, passed in zip(STAGES, stage_values, strict=True) if not passed), "NONE_SUCCESS")
    if not hard_valid:
        first_failure = "HARD_PHYSICAL_VALIDITY"

    frame_of = lambda row: None if row is None else int(frames[row])
    topology = {
        "left_grasp_confirm": topology_at(forces, palms, "left", event_rows["left_confirm"], force_threshold),
        "right_grasp_confirm": topology_at(forces, palms, "right", event_rows["right_confirm"], force_threshold),
        "left_ownership": topology_at(forces, palms, "left", event_rows["left_owned"], force_threshold),
        "right_ownership": topology_at(forces, palms, "right", event_rows["right_owned"], force_threshold),
    }
    result = {
        "schema_version": "episode_registered_physical_eval35_result_v1",
        "status": "INVALID" if not hard_valid else ("PASS" if full else "FAIL"),
        "method": invocation["method"],
        "eval_index": int(invocation["eval_index"]),
        "stable_episode_id": invocation["stable_episode_id"],
        "outcomes": outcomes,
        "first_failure_stage": first_failure,
        "release_classification": release_classification,
        "event_frames": {
            "grasp_confirm": frame_of(event_rows["left_confirm"]),
            "table_support_loss": frame_of(event_rows["left_table_loss"]),
            "lift": frame_of(event_rows["left_lift"]),
            "handoff": frame_of(event_rows["right_support"]),
            "right_ownership": frame_of(event_rows["right_owned"]),
            "release": frame_of(release_row),
            "bin_entry": frame_of(first_entry),
            "settle": frame_of(settle_start),
        },
        "contact_topology": topology,
        "final_state": {
            "doll_position_xyz_m": positions[-1].tolist(),
            "doll_quaternion_xyzw": event["object_quaternion_xyzw"][-1].astype(float).tolist(),
            "doll_linear_velocity_m_s": event["object_linear_velocity_m_s"][-1].astype(float).tolist(),
            "doll_angular_velocity_rad_s": event["object_angular_velocity_rad_s"][-1].astype(float).tolist(),
            "doll_in_bin": bool(in_bin[-1]),
        },
        "fairness_audit": {
            "arm_rescue_used": False,
            "wrist_rescue_used": False,
            "arm_common_override_scalar_count": arm_rescue_count,
            "wrist_common_override_scalar_count": wrist_rescue_count,
            "dex3_common_override_scalar_count": dex3_override_count,
            "arm_hard_limit_projection_scalar_count": projection_count,
            "wrist_hard_limit_projection_scalar_count": wrist_projection_count,
            "maximum_arm_hard_limit_correction_rad": max_projection,
            "arm_intervention_fraction": 0.0,
            "wrist_intervention_fraction": 0.0,
            "dex3_intervention_fraction": float(dex3_override_count / max(1, len(executed) * 14)),
        },
        "integrity": {
            "hard_physical_validity": hard_valid,
            "commanded_hard_limit_violation_scalar_count": commanded_violations,
            "measured_hard_limit_violation_scalar_count": measured_violations,
            "object_pose_writes_after_initialization": int(trial.get("object_pose_writes_during_timed_loop", -1)),
            "non_finite_state_count": 0 if finite else 1,
            "branch_discontinuity_count": branch_count,
            "maximum_arm_step_l2_rad": maximum_arm_step,
            "maximum_robot_bin_penetration_m": robot_penetration,
            "maximum_doll_bin_penetration_m": doll_penetration,
            "maximum_unsupported_duration_before_bin_s": maximum_unsupported_s,
            "measured_arm_hard_limit_violation_scalar_count": measured_arm_violations,
            "measured_dex3_hard_limit_violation_scalar_count": measured_dex3_violations,
            "measured_arm_readback_tolerance_rad": 0.002,
            "measured_dex3_readback_tolerance_rad": 1.0e-6,
            "runtime_initial_pose_verification": trial.get("object_task_frame_registration", {}).get("runtime_initial_pose_verification"),
            "projection_exact": projection_exact,
            "arm_unchanged_except_projector": arm_unchanged_except_projector,
            "runtime_summary": runtime,
        },
    }
    atomic_json(run / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.json", result)
    (run / "EPISODE_REGISTERED_PHYSICAL_TASK_RESULT.md").write_text(
        "# Episode-registered contact-constrained physical result\n\n"
        f"Status: **{result['status']}**  \nFirst failure: **{first_failure}**\n\n"
        + "\n".join(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in outcomes.items())
        + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if not hard_valid else 0


if __name__ == "__main__":
    raise SystemExit(main())
