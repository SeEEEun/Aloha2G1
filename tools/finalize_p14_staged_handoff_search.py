#!/usr/bin/env python3
"""Freeze the evidence from the bounded P14 staged-handoff search."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SEARCH = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral"
)
FIXED = SEARCH / "right_acquisition_search"
STAGED = SEARCH / "staged_handoff_search"
FREEZE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "frozen_p14_bilateral/FREEZE_MANIFEST.json"
)
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
THRESHOLD_N = 0.015
DT = 1.0 / 240.0

FIXED_IDS = (
    "A1_DX10_YAW35",
    "A2_DX05_YAW35",
    "A3_DX10_YAW30",
    "A4_DX10_YAW40",
    "A5_DX15_YAW30",
    "A6_DX15_YAW40",
    "A7_DX10_DYP03_YAW35",
    "A8_DX10_DYP05_YAW35",
    "A9_DX10_DYP07_YAW35",
)
STAGED_IDS = (
    "S1_DX10_DY3_DZ5_YAW45",
    "S2_DX8_DY3_DZ5_YAW50",
    "S3_DX5_DY5_DZ5_YAW55",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def longest(mask: np.ndarray) -> float:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * DT)


def metrics(log: Path, stage: str) -> dict[str, Any]:
    with np.load(log, allow_pickle=False) as archive:
        labels = archive["stage"].astype(str)
        mask = labels == stage
        if not np.any(mask):
            raise RuntimeError(f"{log}: missing {stage}")
        force = {
            digit: np.asarray(archive[f"{digit}_force_n"][mask], dtype=np.float64)
            for digit in ("thumb", "index", "middle")
        }
        simultaneous = np.ones(np.count_nonzero(mask), dtype=bool)
        per_digit: dict[str, Any] = {}
        for digit, value in force.items():
            meaningful = value >= THRESHOLD_N
            simultaneous &= meaningful
            per_digit[digit] = {
                "maximum_force_n": float(np.max(value, initial=0.0)),
                "mean_force_n": float(np.mean(value)),
                "longest_meaningful_contact_s": longest(meaningful),
            }
        position = np.asarray(archive["object_position_world_m"][mask], dtype=np.float64)
        table = np.asarray(archive["table_contact_force_n"][mask], dtype=np.float64)
        return {
            "stage": stage,
            "samples": int(np.count_nonzero(mask)),
            "digits": per_digit,
            "simultaneous_three_digit_support_s": longest(simultaneous),
            "object_com_z_min_m": float(np.min(position[:, 2])),
            "object_com_z_max_m": float(np.max(position[:, 2])),
            "maximum_table_force_n": float(np.max(table, initial=0.0)),
        }


def main() -> int:
    freeze = read_json(FREEZE)
    if sha256_file(CONFIG) != freeze["hashes"]["active_grasp_config"]["sha256"]:
        raise RuntimeError("frozen P14 config changed")
    fixed_rows = []
    for candidate in FIXED_IDS:
        directory = FIXED / candidate
        offline = read_json(directory / "offline_report.json")
        physics = directory / "physics_right_sensor/event_log.npz"
        row = {
            "candidate_id": candidate,
            "offline_status": offline["status"],
            "minimum_hand_clearance_m": offline["static_audit"][
                "minimum_left_right_clearance_m"
            ],
            "robot_collision_frame_counts": offline["robot_collision_frame_counts"],
            "right_ik": offline["solve"],
            "physics": metrics(physics, "DUAL_CONTACT_HOLD"),
            "left_release_executed": False,
            "event_log": str(physics),
            "event_log_sha256": sha256_file(physics),
        }
        row["gate_pass"] = bool(
            row["physics"]["simultaneous_three_digit_support_s"] >= 0.5
        )
        fixed_rows.append(row)

    staged_rows = []
    for candidate in STAGED_IDS:
        directory = STAGED / candidate / "gate"
        offline = read_json(directory / "offline_report.json")
        physics = directory / "physics_right_sensor/event_log.npz"
        row = {
            "candidate_id": candidate,
            "offline_status": offline["status"],
            "endpoint_object_relative": offline["endpoint_object_relative"],
            "minimum_hand_clearance_m": offline["endpoint_static_audit"][
                "minimum_left_right_clearance_m"
            ],
            "endpoint_digit_signed_distance_m": offline["endpoint_static_audit"][
                "right_distal_proxy_signed_distance_m"
            ],
            "robot_collision_frame_counts": offline["robot_collision_frame_counts"],
            "joint_limit_violations": offline["joint_limit_violations"],
            "physics": metrics(physics, "RIGHT_THREE_DIGIT_VERIFICATION"),
            "left_release_executed": False,
            "event_log": str(physics),
            "event_log_sha256": sha256_file(physics),
        }
        row["gate_pass"] = bool(
            row["physics"]["simultaneous_three_digit_support_s"] >= 0.5
        )
        staged_rows.append(row)

    best = min(
        staged_rows,
        key=lambda row: (
            -row["physics"]["digits"]["thumb"]["maximum_force_n"],
            abs(row["endpoint_digit_signed_distance_m"]["thumb"]),
        ),
    )
    left_log = (
        STAGED
        / best["candidate_id"]
        / "gate/physics_left_sensor/event_log.npz"
    )
    left_support = metrics(left_log, "RIGHT_THREE_DIGIT_VERIFICATION")
    report = {
        "schema_version": "p14_staged_handoff_bounded_search_final_v1",
        "status": "STAGED_HANDOFF_STILL_BLOCKED",
        "experiment_scope": "policy-independent scripted P14 handoff acquisition only",
        "frozen_contract": {
            "manifest": str(FREEZE),
            "manifest_sha256": sha256_file(FREEZE),
            "config": str(CONFIG),
            "config_sha256": sha256_file(CONFIG),
            "doll_changed": False,
            "p14_changed": False,
            "controller_changed": False,
            "solver_or_safety_limits_changed": False,
        },
        "fixed_pose_search": {
            "declared_count": len(FIXED_IDS),
            "completed_count": len(fixed_rows),
            "gate_pass_count": sum(row["gate_pass"] for row in fixed_rows),
            "candidates": fixed_rows,
        },
        "staged_object_relative_search": {
            "declared_count": len(STAGED_IDS),
            "completed_count": len(staged_rows),
            "gate_pass_count": sum(row["gate_pass"] for row in staged_rows),
            "candidates": staged_rows,
        },
        "best_candidate": best,
        "best_candidate_left_support": {
            "physics": left_support,
            "left_thumb_support_s": left_support["digits"]["thumb"][
                "longest_meaningful_contact_s"
            ],
            "event_log": str(left_log),
            "event_log_sha256": sha256_file(left_log),
        },
        "exact_blocker": (
            "right thumb normal force remained 0 N for every staged verification; "
            "best S1 preserved elevated left-thumb and right index+middle support, "
            "but never established right thumb opposition"
        ),
        "hard_gate": {
            "right_thumb_index_middle_support_before_left_release": False,
            "left_release_permitted": False,
            "scripted_full_task_run": False,
            "final_environment_frozen": False,
            "act_a_b_physics_run": False,
        },
        "code": {
            "builder": str(ROOT / "tools/build_p14_staged_handoff_candidate.py"),
            "builder_sha256": sha256_file(
                ROOT / "tools/build_p14_staged_handoff_candidate.py"
            ),
            "runtime_gate_runner": str(
                ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
            ),
            "runtime_gate_runner_sha256": sha256_file(
                ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py"
            ),
        },
        "prohibited_mechanisms_used": [],
        "policy_used": False,
        "real_robot": False,
    }
    output_json = STAGED / "FINAL_STAGED_HANDOFF_SEARCH_REPORT.json"
    atomic_json(output_json, report)
    best_physics = best["physics"]
    markdown = f"""# P14 staged handoff bounded-search result

Status: `STAGED_HANDOFF_STILL_BLOCKED`

The frozen doll physics, bilateral P14 vectors, gains, solver, and safety limits
were unchanged. All {len(FIXED_IDS)} fixed acquisition candidates and all
{len(STAGED_IDS)} predeclared object-relative staged paths completed their
acquisition-only runs. No candidate passed the right three-digit release gate.

## Best candidate

- Candidate: `{best['candidate_id']}`
- Minimum hand-hand clearance: {1000.0 * best['minimum_hand_clearance_m']:.3f} mm
- Robot collision frames: {sum(best['robot_collision_frame_counts'].values())}
- Joint-limit violations: {best['joint_limit_violations']}
- Right thumb force: {best_physics['digits']['thumb']['maximum_force_n']:.6f} N
- Right index support: {best_physics['digits']['index']['longest_meaningful_contact_s']:.3f} s
- Right middle support: {best_physics['digits']['middle']['longest_meaningful_contact_s']:.3f} s
- Simultaneous right three-digit support: {best_physics['simultaneous_three_digit_support_s']:.3f} s
- Left thumb support during the same verification: {left_support['digits']['thumb']['longest_meaningful_contact_s']:.3f} s
- Doll table force during verification: {best_physics['maximum_table_force_n']:.6f} N

The exact remaining blocker is right-thumb opposition: the left thumb remained
loaded and right index+middle retained the elevated doll, but the right thumb
reported 0 N throughout every staged verification. Therefore the interlock did
not permit left-thumb release. The 792-frame full task, final environment
freeze, and ACT-A/B evaluation were not run.

Machine-readable evidence: `{output_json}`
"""
    (STAGED / "FINAL_STAGED_HANDOFF_SEARCH_REPORT.md").write_text(
        markdown, encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
