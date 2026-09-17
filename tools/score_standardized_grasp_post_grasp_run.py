#!/usr/bin/env python3
"""Score one standardized-grasp post-grasp rollout from actual PhysX state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
ENVIRONMENT = ROOT / "outputs/final_episode_registered_eval35/01_freeze/FINAL_PHYSICAL_ENVIRONMENT.json"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
FEASIBILITY = ROOT / "configs/doll_handoff_g1_feasibility_resolver.json"
DIGITS = ("thumb", "index", "middle")
STAGES = ("LIFT", "LEFT_RETENTION", "HANDOFF", "RIGHT_OWNERSHIP", "RIGHT_TRANSPORT", "BIN_ENTRY", "SETTLE")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def first_event(events: np.ndarray, name: str) -> int | None:
    indices = np.flatnonzero(np.char.find(events.astype(str), name) >= 0)
    return int(indices[0]) if len(indices) else None


def first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def first_persistent(mask: np.ndarray, minimum: int) -> int | None:
    start, count = None, 0
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value:
            start = index if start is None else start
            count += 1
            if count >= minimum:
                return int(start)
        else:
            start, count = None, 0
    return None


def longest(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def unique_control(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    rows = np.r_[np.flatnonzero(np.diff(control)), len(control) - 1]
    return np.asarray(values)[rows]


def branch_count(commands: np.ndarray) -> tuple[int, float]:
    settings = read(FEASIBILITY)["unchanged_acceptance"]
    norms = np.linalg.norm(np.diff(commands[:, :14], axis=0), axis=1)
    flags = []
    for index, value in enumerate(norms):
        prior = norms[max(0, index - 10):index]
        local = float(np.median(prior)) if len(prior) else 0.0
        threshold = max(float(settings["branch_absolute_step_norm_rad"]), float(settings["branch_local_multiplier"]) * max(local, 1e-6))
        flags.append(value > threshold)
    return int(np.count_nonzero(flags)), float(np.max(norms, initial=0.0))


def topology(forces: dict[str, dict[str, np.ndarray]], palms: dict[str, np.ndarray], side: str, row: int | None, threshold: float) -> list[str]:
    if row is None:
        return []
    scope = slice(max(0, row - 8), min(len(palms[side]), row + 9))
    answer = [digit for digit in DIGITS if np.max(forces[side][digit][scope], initial=0.0) >= threshold]
    if np.max(palms[side][scope], initial=0.0) >= threshold:
        answer.append("palm")
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    with np.load(run / "event_log.npz", allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(run / "robot_bin_contacts.npz", allow_pickle=False) as archive:
        robot_bin = {key: np.asarray(archive[key]) for key in archive.files}
    trial, runtime, invocation = read(run / "trial_result.json"), read(run / "DIRECT_EXECUTION_RUNTIME_SUMMARY.json"), read(run / "INVOCATION_MANIFEST.json")
    config, environment = read(CONFIG), read(ENVIRONMENT)
    required = {
        "RAW_POLICY_COMMAND", "POLICY_SAFE_COMMAND", "PROJECTED_POLICY_COMMAND", "EXECUTED_COMMAND", "MEASURED_Q",
        "COMMON_CONTROLLER_OVERRIDE_MASK_ARM", "COMMON_CONTROLLER_OVERRIDE_MASK_WRIST", "COMMON_CONTROLLER_OVERRIDE_MASK_DEX3",
        "DIRECT_COMMON_EXECUTION_EVENTS", "DIRECT_COMMON_TASK_INTENT", "commanded_q_rad", "measured_q_rad",
        "object_position_world_m", "object_quaternion_xyzw", "object_linear_velocity_m_s", "object_angular_velocity_rad_s",
        "table_contact_force_n", "doll_bin_contact_force_n", "maximum_doll_bin_penetration_m", "left_palm_force_n", "right_palm_force_n",
        *(f"{side}_{digit}_force_n" for side in ("left", "right") for digit in DIGITS),
    }
    missing = sorted(required - set(event))
    if missing:
        raise RuntimeError(f"standardized-grasp physical trace fields missing: {missing}")

    dt = float(config["timing"]["physics_dt_s"])
    threshold = float(config["gates"]["meaningful_digit_force_n"])
    table_threshold = float(config["gates"]["maximum_table_force_for_elevated_n"])
    control = event["control_frame"].astype(np.int64)
    positions = event["object_position_world_m"].astype(np.float64)
    speeds = np.linalg.norm(event["object_linear_velocity_m_s"].astype(np.float64), axis=1)
    forces = {side: {digit: event[f"{side}_{digit}_force_n"].astype(np.float64) for digit in DIGITS} for side in ("left", "right")}
    palms = {side: event[f"{side}_palm_force_n"].astype(np.float64) for side in ("left", "right")}
    meaningful = {side: {digit: forces[side][digit] >= threshold for digit in DIGITS} for side in ("left", "right")}
    opposing = {side: meaningful[side]["thumb"] & (meaningful[side]["index"] | meaningful[side]["middle"] | (palms[side] >= threshold)) for side in ("left", "right")}
    support = {side: np.logical_or.reduce([*(meaningful[side][digit] for digit in DIGITS), palms[side] >= threshold]) for side in ("left", "right")}
    table_free = event["table_contact_force_n"].astype(np.float64) <= table_threshold
    events = event["DIRECT_COMMON_EXECUTION_EVENTS"].astype(str)
    rows = {
        "right_close": first_event(events, "RIGHT_HANDOFF_CLOSE_INTENT_START"),
        "right_confirm": first_event(events, "RIGHT_GRASP_CONFIRMED"),
        "right_support": first_event(events, "RIGHT_RETENTION_SUPPORT_CONFIRMED"),
        "left_release": first_event(events, "GIVING_HAND_RELEASE_START"),
        "right_owned": first_event(events, "RIGHT_PHYSICAL_OWNERSHIP"),
        "release": first_event(events, "FINAL_RELEASE_INTENT_START"),
    }
    initial_pose = invocation["standardized_initial_state"]["object_pose"]
    initial_z = float(initial_pose["position_xyz_m"][2])
    lift_rows = (positions[:, 2] - initial_z >= 0.020) & opposing["left"] & table_free
    lift_row = first_persistent(lift_rows, max(1, int(round(0.10 / dt))))
    lift = lift_row is not None

    right_close = rows["right_close"] if rows["right_close"] is not None else len(control)
    left_loss = first_persistent((np.arange(len(control)) < right_close) & (~opposing["left"] | ~table_free), max(1, int(round(0.10 / dt))))
    left_retention = bool(lift and left_loss is None)
    handoff = bool(
        left_retention and rows["right_confirm"] is not None and rows["right_support"] is not None
        and rows["left_release"] is not None and rows["right_owned"] is not None
        and rows["right_confirm"] <= rows["right_support"] <= rows["left_release"] <= rows["right_owned"]
        and np.any(opposing["right"][rows["right_confirm"]:rows["right_owned"] + 1])
    )
    ownership = bool(handoff and rows["right_owned"] is not None)

    bin_spec = environment["bin"]
    center = np.asarray(bin_spec["opening_center_world_xy_m"], dtype=np.float64)
    opening = np.asarray(bin_spec["opening_dimensions_xy_m"], dtype=np.float64)
    inside_xy = np.all(np.abs(positions[:, :2] - center) <= opening / 2.0, axis=1)
    in_bin = inside_xy & (positions[:, 2] > float(bin_spec["bottom_world_z_m"])) & (positions[:, 2] < float(bin_spec["rim_world_z_m"]))
    after_owned = np.arange(len(control)) >= (rows["right_owned"] if rows["right_owned"] is not None else len(control))
    entry_row = first_true(in_bin & after_owned)
    right_loss = first_persistent(after_owned & ~support["right"] & ~in_bin, max(1, int(round(0.10 / dt))))
    right_transport = bool(ownership and entry_row is not None and (right_loss is None or right_loss >= entry_row))
    bin_entry = bool(right_transport and entry_row is not None)
    settle_rows = max(1, int(round(1.0 / dt)))
    settle_mask = in_bin & (speeds <= 0.02) & (event["doll_bin_contact_force_n"].astype(np.float64) > 0.0)
    settle = bool(bin_entry and longest(settle_mask[entry_row:]) >= settle_rows)

    raw, safe = event["RAW_POLICY_COMMAND"].astype(np.float64), event["POLICY_SAFE_COMMAND"].astype(np.float64)
    projected, executed, measured = event["PROJECTED_POLICY_COMMAND"].astype(np.float64), event["EXECUTED_COMMAND"].astype(np.float64), event["MEASURED_Q"].astype(np.float64)
    specs = sorted(read(JOINT_CONTRACT)["joint_specs"], key=lambda row: int(row["index"]))
    lower = np.asarray([row["minimum"] for row in specs]); upper = np.asarray([row["maximum"] for row in specs])
    expected = safe.copy(); expected[:, :14] = np.clip(safe[:, :14], lower[:14], upper[:14])
    commanded_violations = int(np.count_nonzero((executed < lower) | (executed > upper)))
    measured_arm = int(np.count_nonzero((measured[:, :14] < lower[:14] - .002) | (measured[:, :14] > upper[:14] + .002)))
    measured_dex3 = int(np.count_nonzero((measured[:, 14:] < lower[14:] - 1e-6) | (measured[:, 14:] > upper[14:] + 1e-6)))
    branches, max_step = branch_count(unique_control(projected, control))
    penetration_tolerance = float(environment["penetration"]["normal_solver_tolerance_m"])
    robot_penetration = float(np.max(robot_bin.get("penetration_m", np.asarray([])), initial=0.0))
    doll_penetration = float(np.max(event["maximum_doll_bin_penetration_m"], initial=0.0))
    finite = all(np.isfinite(value).all() for value in (raw, safe, projected, executed, measured, positions, speeds))
    registration = trial.get("object_task_frame_registration", {}).get("runtime_initial_pose_verification", {})
    hard_valid = bool(
        np.array_equal(executed, event["commanded_q_rad"]) and np.array_equal(measured, event["measured_q_rad"])
        and np.array_equal(projected, expected) and np.array_equal(executed[:, :14], projected[:, :14])
        and not np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_ARM"])
        and not np.count_nonzero(event["COMMON_CONTROLLER_OVERRIDE_MASK_WRIST"])
        and commanded_violations == 0 and measured_arm == 0 and measured_dex3 == 0 and branches == 0 and finite
        and robot_penetration <= penetration_tolerance and doll_penetration <= penetration_tolerance
        and bool(trial.get("command_completed")) and not bool(trial.get("state_restoration", {}).get("used"))
        and int(trial.get("object_pose_writes_during_timed_loop", -1)) == 0
        and registration.get("within_0_1_mm_and_0_1_deg") is True
        and runtime.get("standardized_initial_grasp") is True
    )

    ik_failure = int(invocation["common_ik_first_infeasible_frame"])
    intent_control = unique_control(event["DIRECT_COMMON_TASK_INTENT"], control).astype(str)
    ik_failure_stage = None
    if ik_failure >= 0:
        value = intent_control[min(ik_failure, len(intent_control) - 1)]
        ik_failure_stage = {"LEFT_HOLD_INTENT": "LIFT", "HANDOFF_INTENT": "HANDOFF", "RIGHT_HOLD_INTENT": "RIGHT_TRANSPORT", "FINAL_RELEASE_INTENT": "BIN_ENTRY"}[value]
    cumulative = [lift, left_retention, handoff, ownership, right_transport, bin_entry, settle]
    if ik_failure_stage is not None:
        boundary = STAGES.index(ik_failure_stage)
        cumulative[boundary:] = [False] * (len(cumulative) - boundary)
    lift, left_retention, handoff, ownership, right_transport, bin_entry, settle = cumulative
    full = bool(all(cumulative) and hard_valid)

    release_row = rows["release"]
    contact_loss = first_persistent(after_owned & ~support["right"], max(1, int(round(.10 / dt)))) if ownership else None
    clean = bool(ownership and release_row is not None and contact_loss is not None and contact_loss >= release_row)
    premature_into = bool(contact_loss is not None and entry_row is not None and inside_xy[contact_loss] and settle and not clean)
    premature_outside = bool(contact_loss is not None and not premature_into and contact_loss < (entry_row if entry_row is not None else len(control)))
    outcomes = {
        "STANDARDIZED_INITIAL_GRASP": True,
        "LIFT_SUCCESS": lift,
        "LEFT_RETENTION_SUCCESS": left_retention,
        "HANDOFF_SUCCESS": handoff,
        "RIGHT_OWNERSHIP_SUCCESS": ownership,
        "RIGHT_TRANSPORT_RETENTION": right_transport,
        "BIN_ENTRY_SUCCESS": bin_entry,
        "BIN_SETTLE_SUCCESS": settle,
        "POST_GRASP_FULL_TASK_SUCCESS": full,
        "CLEAN_COMMANDED_RELEASE": clean,
        "PREMATURE_DROP_INTO_BIN": premature_into,
        "PREMATURE_DROP_OUTSIDE_BIN": premature_outside,
    }
    first_failure = next((stage for stage, passed in zip(STAGES, cumulative, strict=True) if not passed), "SUCCESS")
    if not hard_valid:
        first_failure = "INVALID_PHYSICS"
    result = {
        "schema_version": "standardized_grasp_post_grasp_physical_result_v1",
        "status": "INVALID" if not hard_valid else ("PASS" if full else "FAIL"),
        "evaluation_label": "DEV35 STANDARDIZED-GRASP PHYSICAL EVALUATION",
        "not_end_to_end_grasp_acquisition": True,
        "method": invocation["method"], "eval_index": invocation["eval_index"], "stable_episode_id": invocation["stable_episode_id"],
        "outcomes": outcomes, "first_failure_stage": first_failure, "common_ik_failure_stage": ik_failure_stage,
        "event_frames": {key: None if value is None else int(control[value]) for key, value in {**rows, "lift": lift_row, "bin_entry": entry_row}.items()},
        "contact_topology": {"standardized_left": topology(forces, palms, "left", 0, threshold), "right_confirm": topology(forces, palms, "right", rows["right_confirm"], threshold), "right_ownership": topology(forces, palms, "right", rows["right_owned"], threshold)},
        "integrity": {
            "hard_physical_validity": hard_valid, "commanded_hard_limit_violations": commanded_violations,
            "measured_arm_hard_limit_violations": measured_arm, "measured_dex3_hard_limit_violations": measured_dex3,
            "branch_discontinuities": branches, "maximum_arm_step_l2_rad": max_step,
            "maximum_robot_bin_penetration_m": robot_penetration, "maximum_doll_bin_penetration_m": doll_penetration,
            "object_pose_writes_after_initialization": int(trial.get("object_pose_writes_during_timed_loop", -1)),
            "arm_rescue": False, "wrist_rescue": False, "finite": finite,
            "runtime_initial_pose_verification": registration,
        },
        "diagnostics": {
            "maximum_doll_com_lift_mm": 1000.0 * float(np.max(positions[:, 2] - initial_z)),
            "initial_object_z_m": initial_z,
            "common_ik_first_infeasible_control_frame": ik_failure,
            "release_classification": "CLEAN_COMMANDED_RELEASE" if clean else ("PREMATURE_DROP_INTO_BIN" if premature_into else ("PREMATURE_DROP_OUTSIDE_BIN" if premature_outside else "NO_RIGHT_OWNERSHIP_OR_NO_RELEASE")),
        },
    }
    atomic_json(run / "STANDARDIZED_GRASP_POST_GRASP_RESULT.json", result)
    (run / "STANDARDIZED_GRASP_POST_GRASP_RESULT.md").write_text(
        "# Standardized-grasp post-grasp physical result\n\n"
        f"- Status: **{result['status']}**\n- First failure: **{first_failure}**\n"
        "- Initial LEFT grasp: **STANDARDIZED CONTROL (not a method success metric)**\n\n"
        + "\n".join(f"- {key}: `{'PASS' if value else 'FAIL'}`" for key, value in outcomes.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if not hard_valid else 0


if __name__ == "__main__":
    raise SystemExit(main())
