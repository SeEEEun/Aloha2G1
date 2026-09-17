#!/usr/bin/env python3
"""Audit bounded handoff-to-transport physics candidates uniformly.

This tool is deliberately read-only with respect to physics inputs.  It reads
offline reports and Isaac event logs, then writes a compact audit under the
final-task output root.  It does not generate commands, simulate, tune, or
select a candidate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CANDIDATE_ROOT = (
    ROOT / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
)
OUTPUT_ROOT = ROOT / "outputs/final_task_completion_v1/02_handoff_to_transport_grasp"
FORCE_THRESHOLD_N = 0.015
TABLE_THRESHOLD_N = 0.015
MAXIMUM_HELD_OBJECT_SPEED_M_S = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def stage_summary(event: dict[str, np.ndarray], name: str) -> dict[str, Any] | None:
    mask = event["stage"].astype(str) == name
    if not np.any(mask):
        return None
    forces = np.column_stack(
        [
            event["thumb_force_n"],
            event["index_force_n"],
            event["middle_force_n"],
        ]
    )[mask]
    table = event["table_contact_force_n"][mask]
    position = event["object_position_world_m"][mask]
    velocity = event["object_linear_velocity_m_s"][mask]
    frames = event["control_frame"][mask]
    all_three = np.all(forces >= FORCE_THRESHOLD_N, axis=1)
    any_digit = np.any(forces >= FORCE_THRESHOLD_N, axis=1)
    return {
        "control_frame_start": int(np.min(frames)),
        "control_frame_end": int(np.max(frames)),
        "all_three_contact_fraction": float(np.mean(all_three)),
        "any_digit_contact_fraction": float(np.mean(any_digit)),
        "table_free_fraction": float(np.mean(table < TABLE_THRESHOLD_N)),
        "mean_force_n_thumb_index_middle": forces.mean(axis=0).tolist(),
        "minimum_object_z_m": float(np.min(position[:, 2])),
        "maximum_object_z_m": float(np.max(position[:, 2])),
        "maximum_object_speed_m_s": float(
            np.max(np.linalg.norm(velocity, axis=1), initial=0.0)
        ),
    }


def first_event(
    event: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, Any] | None:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    index = int(indices[0])
    return {
        "physics_step_index": index,
        "control_frame": int(event["control_frame"][index]),
        "stage": str(event["stage"][index]),
        "time_s": float(event["timestamp_s"][index]),
        "object_position_world_m": event["object_position_world_m"][index].tolist(),
    }


def audit_candidate(directory: Path) -> dict[str, Any]:
    offline_path = directory / "offline_report.json"
    event_path = directory / "physics_full/event_log.npz"
    trial_path = directory / "physics_full/trial_result.json"
    result: dict[str, Any] = {
        "candidate": directory.name,
        "directory": str(directory),
        "offline_report": str(offline_path) if offline_path.exists() else None,
        "physics_run": event_path.exists(),
    }
    if offline_path.exists():
        offline = json.loads(offline_path.read_text(encoding="utf-8"))
        result["offline_status"] = offline.get("status")
        result["command"] = offline.get("command")
        result["command_sha256"] = offline.get("command_sha256")
        result["offline_collision_counts"] = offline.get("collision_counts")
        result["offline_joint_limit_violation_count"] = offline.get(
            "joint_limit_violation_count"
        )
    if not event_path.exists():
        result["first_failed_stage"] = (
            "OFFLINE_GATE" if result.get("offline_status") == "OFFLINE_FAIL" else "NOT_RUN"
        )
        return result

    with np.load(event_path, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    result["event_log"] = str(event_path)
    result["event_log_sha256"] = sha256_file(event_path)
    trial = (
        json.loads(trial_path.read_text(encoding="utf-8"))
        if trial_path.exists()
        else {}
    )
    runtime_gate = trial.get("runtime_right_three_digit_gate", {})
    result["pre_release_three_digit_gate"] = runtime_gate.get("status", "UNKNOWN")
    result["state_restoration_used"] = bool(
        trial.get("state_restoration", {}).get("used", False)
    )
    result["prohibited_attachment_used"] = bool(
        trial.get("prohibited_attachment_used", False)
    )

    for stage in (
        "RIGHT_THREE_DIGIT_VERIFICATION",
        "LEFT_THUMB_RELEASE",
        "RIGHT_POST_RELEASE_RETENTION",
        "LEFT_OBJECT_RADIAL_CLEARANCE",
        "RIGHT_TRANSPORT_GRIP_VERIFICATION",
        "RIGHT_TRANSPORT_VERTICAL_CLEARANCE",
        "RIGHT_VERTICAL_STABILIZATION",
        "RIGHT_TRANSPORT_TO_BIN",
        "RIGHT_HOLD_OVER_BIN",
        "RIGHT_CONTROLLED_BIN_DESCENT",
        "RIGHT_RELEASE",
        "BIN_SETTLE",
    ):
        summary = stage_summary(event, stage)
        if summary is not None:
            result.setdefault("stages", {})[stage] = summary

    forces = np.column_stack(
        [
            event["thumb_force_n"],
            event["index_force_n"],
            event["middle_force_n"],
        ]
    )
    post_gate = event["control_frame"] >= int(
        runtime_gate.get("evaluated_before_control_frame", 0)
    )
    result["first_all_digit_loss_after_gate"] = first_event(
        event, post_gate & ~np.any(forces >= FORCE_THRESHOLD_N, axis=1)
    )
    result["first_table_contact_after_gate"] = first_event(
        event,
        post_gate & (event["table_contact_force_n"] >= TABLE_THRESHOLD_N),
    )
    result["terminal_object_com_world_m"] = event["object_position_world_m"][-1].tolist()

    # The authoritative simple-proxy configuration caps object speed at
    # 1.0 m/s.  A gravity-driven speed during intentional release is not a
    # grasp/transport instability, but exceeding this gate while the hand is
    # commanded to retain the doll is a hard failure.
    intentional_free_body_stages = np.isin(
        event["stage"].astype(str), ["RIGHT_RELEASE", "BIN_SETTLE"]
    )
    held_speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    held_indices = np.flatnonzero(~intentional_free_body_stages)
    maximum_held_index = int(held_indices[np.argmax(held_speed[held_indices])])
    result["maximum_held_object_speed_m_s"] = float(held_speed[maximum_held_index])
    result["maximum_held_object_speed_event"] = {
        "control_frame": int(event["control_frame"][maximum_held_index]),
        "stage": str(event["stage"][maximum_held_index]),
        "time_s": float(event["timestamp_s"][maximum_held_index]),
    }
    result["held_motion_speed_gate_pass"] = bool(
        result["maximum_held_object_speed_m_s"] <= MAXIMUM_HELD_OBJECT_SPEED_M_S
    )

    stages = result.get("stages", {})
    post = stages.get("RIGHT_POST_RELEASE_RETENTION")
    vertical = stages.get("RIGHT_TRANSPORT_VERTICAL_CLEARANCE")
    horizontal = stages.get("RIGHT_TRANSPORT_TO_BIN")
    settle = stages.get("BIN_SETTLE")
    if result["pre_release_three_digit_gate"] != "PASS":
        failure = "PRE_RELEASE_THREE_DIGIT_GATE"
    elif post is None or post["any_digit_contact_fraction"] < 0.99 or post["table_free_fraction"] < 0.99:
        failure = "OWNERSHIP_TRANSFER"
    elif vertical is None or vertical["any_digit_contact_fraction"] < 0.99 or vertical["table_free_fraction"] < 0.99:
        failure = "VERTICAL_TRANSPORT"
    elif horizontal is None or horizontal["any_digit_contact_fraction"] < 0.99 or horizontal["table_free_fraction"] < 0.99:
        failure = "HORIZONTAL_TRANSPORT"
    elif not result["held_motion_speed_gate_pass"]:
        failure = "HELD_MOTION_DYNAMICS_GATE"
    elif settle is None or settle["maximum_object_speed_m_s"] > 0.05:
        failure = "BIN_RELEASE_OR_SETTLE"
    else:
        failure = "NONE"
    result["first_failed_stage"] = failure
    result["continuous_physical_pass"] = failure == "NONE"
    return result


def main() -> int:
    candidate_names = [
        "R18_HANDOFF_WRIST_R14_HAND_T5_FULL",
        "R19_HANDOFF_R14_ROLL_M5",
        "R20_HANDOFF_R14_ROLL_P5",
        "R21_HANDOFF_R14_PITCH_M5",
        "R22_HANDOFF_R14_PITCH_P5",
        "R23_HANDOFF_R14_YAW_M5",
        "R24_HANDOFF_R14_YAW_P5",
        "R25_HANDOFF_R14_INDEX0_P005",
        "R26_HANDOFF_R14_INDEX0_P010",
        "R27_HANDOFF_R14_INDEX1_P008",
    ]
    rows = [audit_candidate(CANDIDATE_ROOT / name) for name in candidate_names]
    payload = {
        "schema_version": "final_handoff_transport_candidate_audit_v1",
        "force_threshold_n": FORCE_THRESHOLD_N,
        "table_threshold_n": TABLE_THRESHOLD_N,
        "maximum_held_object_speed_m_s": MAXIMUM_HELD_OBJECT_SPEED_M_S,
        "candidate_count": len(rows),
        "physics_run_count": sum(bool(row["physics_run"]) for row in rows),
        "continuous_pass_count": sum(
            bool(row.get("continuous_physical_pass", False)) for row in rows
        ),
        "candidates": rows,
    }
    json_path = OUTPUT_ROOT / "CANDIDATE_PHYSICS_AUDIT.json"
    csv_path = OUTPUT_ROOT / "CANDIDATE_PHYSICS_AUDIT.csv"
    md_path = OUTPUT_ROOT / "CANDIDATE_PHYSICS_AUDIT.md"
    atomic_json(json_path, payload)

    csv_fields = [
        "candidate",
        "offline_status",
        "physics_run",
        "pre_release_three_digit_gate",
        "first_failed_stage",
        "continuous_physical_pass",
        "command_sha256",
        "event_log_sha256",
    ]
    temporary_csv = csv_path.with_suffix(".csv.incomplete")
    with temporary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in csv_fields})
    os.replace(temporary_csv, csv_path)

    lines = [
        "# Handoff-to-transport candidate physics audit",
        "",
        f"Force threshold: `{FORCE_THRESHOLD_N:.3f} N`; table threshold: `{TABLE_THRESHOLD_N:.3f} N`.",
        "",
        "| Candidate | Offline | Physics | Pre-release gate | First failed stage | Full pass |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {offline} | {physics} | {gate} | {failure} | {passed} |".format(
                candidate=row["candidate"],
                offline=row.get("offline_status", "missing"),
                physics="YES" if row["physics_run"] else "NO",
                gate=row.get("pre_release_three_digit_gate", "N/A"),
                failure=row.get("first_failed_stage", "N/A"),
                passed="YES" if row.get("continuous_physical_pass", False) else "NO",
            )
        )
    lines.extend(
        [
            "",
            "No object physics, wrist path, controller gains, scientific A/B artifacts,",
            "or simulation state are modified by this audit.",
        ]
    )
    atomic_text(md_path, "\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
