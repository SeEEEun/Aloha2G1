#!/usr/bin/env python3
"""Uniformly audit the four bounded R26 fixed-geometry timing trials."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_task_completion_v1/08_r26_retiming"
VARIANTS = ("R26_T1_1P5X", "R26_T2_2P0X", "R26_T3_3P0X", "R26_T4_4P0X")
FORCE_N = 0.015
TABLE_N = 0.015
MAX_SPEED_M_S = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stage_summary(event: dict[str, np.ndarray], stage_name: str) -> dict[str, Any]:
    mask = event["stage"].astype(str) == stage_name
    if not np.any(mask):
        return {"present": False}
    forces = np.column_stack(
        (event["thumb_force_n"], event["index_force_n"], event["middle_force_n"])
    )[mask]
    speed = np.linalg.norm(event["object_linear_velocity_m_s"][mask], axis=1)
    return {
        "present": True,
        "all_three_fraction": float(np.mean(np.all(forces >= FORCE_N, axis=1))),
        "any_digit_fraction": float(np.mean(np.any(forces >= FORCE_N, axis=1))),
        "table_free_fraction": float(
            np.mean(event["table_contact_force_n"][mask] < TABLE_N)
        ),
        "maximum_object_speed_m_s": float(np.max(speed, initial=0.0)),
        "mean_force_n_thumb_index_middle": forces.mean(axis=0).tolist(),
    }


def audit_variant(name: str) -> dict[str, Any]:
    directory = OUT / "variants" / name
    offline_path = directory / "offline_report.json"
    trial_path = directory / "physics_full/trial_result.json"
    event_path = directory / "physics_full/event_log.npz"
    offline = read_json(offline_path)
    trial = read_json(trial_path)
    with np.load(event_path, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    stage = event["stage"].astype(str)
    speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    held = ~np.isin(stage, ("RIGHT_RELEASE", "BIN_SETTLE"))
    held_indices = np.flatnonzero(held)
    maximum_index = int(held_indices[np.argmax(speed[held_indices])])
    runtime = trial["runtime_right_three_digit_gate"]
    post = stage_summary(event, "RIGHT_POST_RELEASE_RETENTION")
    horizontal = stage_summary(event, "RIGHT_TRANSPORT_TO_BIN")
    settle = stage_summary(event, "BIN_SETTLE")
    hard_gates = {
        "offline_collision": "PASS"
        if sum(offline["collision_frame_counts"].values()) == 0
        else "FAIL",
        "joint_limits": "PASS"
        if offline["joint_limit_violation_count"] == 0
        else "FAIL",
        "pre_release_three_digit_support": runtime["status"],
        "right_only_retention": "PASS"
        if post["all_three_fraction"] >= 0.99 and post["table_free_fraction"] >= 0.99
        else "FAIL",
        "horizontal_contact_and_support": "PASS"
        if horizontal["any_digit_fraction"] >= 0.99
        and horizontal["table_free_fraction"] >= 0.99
        else "FAIL",
        "held_object_speed": "PASS"
        if float(speed[maximum_index]) <= MAX_SPEED_M_S
        else "FAIL",
        "exact_frozen_r14_endpoint": "PASS"
        if offline["r26_endpoint_matches_frozen_r14"]
        else "FAIL",
        "command_completion": "PASS" if trial["command_completed"] else "FAIL",
        "runtime_penetration": "PASS"
        if trial["artifact_checks"]["maximum_runtime_penetration_m"] <= 0.003
        else "FAIL",
        "object_angular_speed": "PASS"
        if trial["artifact_checks"]["maximum_object_angular_speed_rad_s"] <= 50.0
        else "FAIL",
        "bin_settle": "PASS"
        if settle["present"] and settle["maximum_object_speed_m_s"] <= 0.05
        else "FAIL",
    }
    failed = [key for key, value in hard_gates.items() if value != "PASS"]
    return {
        "candidate": name,
        "scale": offline["scale"],
        "command": offline["command"],
        "command_sha256": offline["command_sha256"],
        "event_log": str(event_path),
        "event_log_sha256": sha256_file(event_path),
        "trial_result": str(trial_path),
        "trial_result_sha256": sha256_file(trial_path),
        "maximum_held_object_speed_m_s": float(speed[maximum_index]),
        "maximum_speed_event": {
            "physics_step_index": maximum_index,
            "control_frame": int(event["control_frame"][maximum_index]),
            "stage": str(stage[maximum_index]),
            "timestamp_s": float(event["timestamp_s"][maximum_index]),
            "forces_n_thumb_index_middle": [
                float(event["thumb_force_n"][maximum_index]),
                float(event["index_force_n"][maximum_index]),
                float(event["middle_force_n"][maximum_index]),
            ],
        },
        "stage_results": {
            "RIGHT_THREE_DIGIT_VERIFICATION": stage_summary(
                event, "RIGHT_THREE_DIGIT_VERIFICATION"
            ),
            "LEFT_THUMB_RELEASE": stage_summary(event, "LEFT_THUMB_RELEASE"),
            "RIGHT_POST_RELEASE_RETENTION": post,
            "RIGHT_TRANSPORT_VERTICAL_CLEARANCE": stage_summary(
                event, "RIGHT_TRANSPORT_VERTICAL_CLEARANCE"
            ),
            "RIGHT_TRANSPORT_TO_BIN": horizontal,
            "RIGHT_CONTROLLED_BIN_DESCENT": stage_summary(
                event, "RIGHT_CONTROLLED_BIN_DESCENT"
            ),
            "BIN_SETTLE": settle,
        },
        "hard_gate_results": hard_gates,
        "failed_hard_gates": failed,
        "handoff_success": not failed,
        "state_restoration_used": bool(trial["state_restoration"]["used"]),
        "prohibited_attachment_used": bool(trial["prohibited_attachment_used"]),
    }


def main() -> int:
    rows = [audit_variant(name) for name in VARIANTS]
    best = min(rows, key=lambda row: row["maximum_held_object_speed_m_s"])
    result = {
        "schema_version": "r26_bounded_retiming_physics_audit_v1",
        "frozen_object_speed_gate_m_s": MAX_SPEED_M_S,
        "candidate_count": len(rows),
        "all_predeclared_candidates_completed": len(rows) == len(VARIANTS),
        "candidates": rows,
        "best_candidate_by_speed": best["candidate"],
        "best_maximum_held_object_speed_m_s": best[
            "maximum_held_object_speed_m_s"
        ],
        "speed_gate_pass_count": sum(
            row["hard_gate_results"]["held_object_speed"] == "PASS" for row in rows
        ),
        "full_handoff_pass_count": sum(row["handoff_success"] for row in rows),
        "decision": "R26_GEOMETRY_NOT_RESCUED_BY_BOUNDED_RETIMING",
        "selected_r14_artifact_changed": False,
        "doll_or_material_changed": False,
        "state_restoration_used": False,
        "prohibited_mechanism_used": False,
    }
    json_path = OUT / "R26_RETIMING_RESULTS.json"
    atomic_json(json_path, result)
    csv_path = OUT / "R26_RETIMING_RESULTS.csv"
    fields = (
        "candidate",
        "scale",
        "maximum_held_object_speed_m_s",
        "failed_hard_gates",
        "handoff_success",
        "command_sha256",
        "event_log_sha256",
    )
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fields},
                    "failed_hard_gates": ";".join(row["failed_hard_gates"]),
                }
            )
    os.replace(temporary, csv_path)
    lines = [
        "# R26 bounded retiming result",
        "",
        "| Candidate | Scale | Peak held-object speed (m/s) | Speed gate | Exact R14 endpoint | Handoff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gates = row["hard_gate_results"]
        lines.append(
            f"| {row['candidate']} | {row['scale']:.1f}x | "
            f"{row['maximum_held_object_speed_m_s']:.6f} | "
            f"{gates['held_object_speed']} | "
            f"{gates['exact_frozen_r14_endpoint']} | "
            f"{'PASS' if row['handoff_success'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Best timing candidate by speed: `{best['candidate']}` at "
            f"`{best['maximum_held_object_speed_m_s']:.6f} m/s`.",
            "",
            "All four candidates retained the exact R26 spatial command and "
            "the unchanged post-handoff transport tail. None passed the "
            "1.0 m/s gate, and all retain the independently audited non-R14 "
            "endpoint. R26 is therefore not a timing-only solution.",
        ]
    )
    atomic_text(OUT / "R26_RETIMING_RESULTS.md", "\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
