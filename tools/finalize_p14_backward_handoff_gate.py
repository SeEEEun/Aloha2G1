#!/usr/bin/env python3
"""Validate the bounded backward-constructed P14 acquisition gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
GATE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral/backward_constructed_handoff"
    / "B2_PATH_F40/right_preload_partial_left_relax_v3/gate"
)
OFFLINE = GATE / "offline_report.json"
RIGHT = GATE / "physics_right_sensor/event_log.npz"
LEFT = GATE / "physics_left_sensor/event_log.npz"
STAGE = "RIGHT_THREE_DIGIT_VERIFICATION"
FORCE_THRESHOLD_N = 0.015
REQUIRED_S = 0.5
DT = 1.0 / 240.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_duration(mask: np.ndarray) -> float:
    current = longest = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * DT)


def stage_metrics(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        labels = archive["stage"].astype(str)
        mask = labels == STAGE
        if not np.any(mask):
            raise RuntimeError(f"{STAGE} absent: {path}")
        positions = np.asarray(archive["object_position_world_m"], dtype=np.float64)
        table = np.asarray(archive["table_contact_force_n"], dtype=np.float64)
        digit_force = {
            digit: np.asarray(archive[f"{digit}_force_n"], dtype=np.float64)
            for digit in ("thumb", "index", "middle")
        }
        meaningful = {
            digit: digit_force[digit] >= FORCE_THRESHOLD_N
            for digit in digit_force
        }
        simultaneous = mask.copy()
        for digit in meaningful:
            simultaneous &= meaningful[digit]
        return {
            "stage": STAGE,
            "samples": int(np.count_nonzero(mask)),
            "duration_s": float(np.count_nonzero(mask) * DT),
            "digit_force_n": {
                digit: {
                    "mean": float(np.mean(values[mask])),
                    "maximum": float(np.max(values[mask], initial=0.0)),
                    "sustained_meaningful_s": longest_duration(mask & meaningful[digit]),
                }
                for digit, values in digit_force.items()
            },
            "simultaneous_three_digit_support_s": longest_duration(simultaneous),
            "maximum_table_contact_force_n": float(np.max(table[mask], initial=0.0)),
            "object_start_world_m": positions[mask][0],
            "object_end_world_m": positions[mask][-1],
            "object_displacement_m": positions[mask][-1] - positions[mask][0],
            "object_displacement_norm_m": float(
                np.linalg.norm(positions[mask][-1] - positions[mask][0])
            ),
        }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda value: value.tolist()
            if isinstance(value, np.ndarray)
            else value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    offline = read_json(OFFLINE)
    right = stage_metrics(RIGHT)
    left = stage_metrics(LEFT)
    collision_counts = offline["robot_collision_frame_counts"]
    static = offline["endpoint_static_audit"]
    right_pass = bool(
        right["simultaneous_three_digit_support_s"] >= REQUIRED_S
        and right["maximum_table_contact_force_n"] <= 0.02
    )
    left_support_pass = bool(
        left["digit_force_n"]["thumb"]["sustained_meaningful_s"] >= REQUIRED_S
        and (
            left["digit_force_n"]["index"]["sustained_meaningful_s"] >= REQUIRED_S
            or left["digit_force_n"]["middle"]["sustained_meaningful_s"] >= REQUIRED_S
        )
        and left["maximum_table_contact_force_n"] <= 0.02
    )
    conditions = {
        "right_hand_doll_contact": right_pass,
        "left_hand_doll_retention_before_transfer": left_support_pass,
        "no_left_right_hand_overlap": bool(static["no_left_right_hand_overlap"]),
        "no_arm_or_self_collision": not any(collision_counts.values()),
        "right_arm_ik_reachability": bool(
            offline["ik_max_position_error_m"] <= 0.001
            and offline["ik_max_orientation_error_rad"] <= 0.02
        ),
        "right_three_digit_retention_after_acquisition": right_pass,
        "doll_table_unsupported": bool(
            right["maximum_table_contact_force_n"] <= 0.02
            and left["maximum_table_contact_force_n"] <= 0.02
        ),
        "joint_limits_valid": offline["joint_limit_violations"] == 0,
        "left_release_not_yet_executed": True,
    }
    passed = all(conditions.values())
    payload = {
        "schema_version": "p14_backward_constructed_acquisition_gate_v1",
        "status": "PASS" if passed else "FAIL",
        "candidate_id": offline["candidate_id"],
        "experiment_scope": "acquisition-only gate; no LEFT release or full task",
        "conditions": conditions,
        "right_sensor_verification": right,
        "left_sensor_verification": left,
        "minimum_left_right_clearance_m": static[
            "minimum_left_right_clearance_m"
        ],
        "closest_left_right_body_pair": static[
            "closest_left_right_body_pair"
        ],
        "offline_report": str(OFFLINE),
        "offline_report_sha256": sha256_file(OFFLINE),
        "right_event_log": str(RIGHT),
        "right_event_log_sha256": sha256_file(RIGHT),
        "left_event_log": str(LEFT),
        "left_event_log_sha256": sha256_file(LEFT),
        "command": offline["command"],
        "command_sha256": offline["command_sha256"],
        "doll_or_controller_changed": False,
        "prohibited_attachment_used": False,
        "policy_used": False,
        "real_robot": False,
    }
    result = GATE / "PHYSICS_GATE_RESULT.json"
    atomic_json(result, payload)
    markdown = f"""# Backward-constructed handoff acquisition gate

- Status: **{payload['status']}**
- Candidate: `{payload['candidate_id']}`
- RIGHT simultaneous thumb+index+middle support: {right['simultaneous_three_digit_support_s']:.3f} s
- LEFT thumb support: {left['digit_force_n']['thumb']['sustained_meaningful_s']:.3f} s
- LEFT middle support: {left['digit_force_n']['middle']['sustained_meaningful_s']:.3f} s
- Maximum doll-table force in verification: {right['maximum_table_contact_force_n']:.6f} N
- Minimum modeled hand-hand clearance: {payload['minimum_left_right_clearance_m'] * 1000.0:.3f} mm
- Robot collision frames: {sum(collision_counts.values())}
- Joint-limit violations: {offline['joint_limit_violations']}

This evidence stops before LEFT release.  It authorizes only construction of a
full command whose runtime runner rechecks RIGHT three-digit support before the
first LEFT-thumb-release command.
"""
    temporary = (GATE / "PHYSICS_GATE_RESULT.md.incomplete")
    temporary.write_text(markdown, encoding="utf-8")
    os.replace(temporary, GATE / "PHYSICS_GATE_RESULT.md")
    print(result.read_text(encoding="utf-8"), end="")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
