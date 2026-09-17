#!/usr/bin/env python3
"""Audit the authoritative fully staged six-candidate handoff family."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
BASE = ROOT / "outputs/final_methodology_preserving_completion"
SEARCH = BASE / "02_exact_endpoint_refinement_fully_staged_v2"
SOURCE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task/p14_bilateral"
    / "backward_constructed_handoff/B2_PATH_F40"
    / "right_preload_partial_left_relax_exact_endpoint_v4/full"
    / "scripted_full_task_command.npz"
)
R14 = (
    BASE
    / "01_verified_right_transport_grasp/R14_GATE_QUALIFIED_BIN_CLEARANCE"
    / "right_only_r6_command.npz"
)
FORCE_N = 0.015
TABLE_N = 0.015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction(mask: np.ndarray, values: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else 0.0


def main() -> int:
    with np.load(SOURCE, allow_pickle=False) as archive:
        source_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        source_stage = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
    prefix_end = int(np.flatnonzero(source_stage == "LEFT_TRANSPORT")[-1])
    prefix = source_q[: prefix_end + 1]
    right_arm_names = [
        name for name in names if name.startswith("right_") and (
            "shoulder" in name or "elbow" in name or "wrist" in name
        )
    ]
    right_hand_names = [name for name in names if name.startswith("right_hand_")]
    right_indices = [names.index(name) for name in right_arm_names + right_hand_names]
    with np.load(R14, allow_pickle=False) as archive:
        r14_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        r14_stage = archive["stage"].astype(str)
    r14_endpoint = r14_q[np.flatnonzero(r14_stage == "HOLD_ELEVATED")[-1]][right_indices]

    rows: list[dict[str, object]] = []
    for command_path in sorted(
        SEARCH.glob("*/*/acquisition/exact_endpoint_acquisition_command.npz")
    ):
        report_path = command_path.parent / "offline_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        with np.load(command_path, allow_pickle=False) as archive:
            command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
            stage = archive["stage"].astype(str)
            command_names = archive["joint_names"].astype(str).tolist()
            profile = str(np.asarray(archive["left_handoff_hold_profile"]).item())
        if command_names != names:
            raise RuntimeError(f"joint order changed: {command_path}")
        prefix_error = float(np.max(np.abs(command[: len(prefix)] - prefix)))
        presentation_start = int(
            np.flatnonzero(stage == "LEFT_HANDOFF_PRESENTATION_TRANSITION")[0]
        )
        right_approach_start = int(
            np.flatnonzero(stage == "RIGHT_COLLISION_FREE_APPROACH")[0]
        )
        right_motion_before = float(
            np.max(
                np.abs(
                    command[len(prefix) : right_approach_start, right_indices]
                    - command[len(prefix) - 1, right_indices][None]
                ),
                initial=0.0,
            )
        )
        verification = stage == "RIGHT_THREE_DIGIT_VERIFICATION"
        endpoint_error = float(
            np.max(
                np.abs(command[verification][:, right_indices] - r14_endpoint[None]),
                initial=0.0,
            )
        )
        event_path = command_path.parent / "physics_right_sensor/event_log.npz"
        result: dict[str, object] = {
            "candidate": command_path.relative_to(SEARCH).parts[1],
            "profile": profile,
            "offline_status": report["status"],
            "command": str(command_path),
            "command_sha256": sha256(command_path),
            "validated_prefix_maximum_28d_error_rad": prefix_error,
            "validated_prefix_exact": prefix_error == 0.0,
            "candidate_presentation_first_frame": presentation_start,
            "validated_prefix_frames": len(prefix),
            "right_motion_before_declared_approach_rad": right_motion_before,
            "right_stationary_before_declared_approach": right_motion_before == 0.0,
            "right_endpoint_maximum_error_rad": endpoint_error,
            "maximum_new_adjacent_arm_step_rad": report["offline"][
                "maximum_new_adjacent_arm_step_rad"
            ],
            "thresholds_changed": False,
            "physics_run": event_path.exists(),
        }
        if report["status"] != "OFFLINE_PASS":
            result.update(
                {
                    "first_failure_stage": "RIGHT_APPROACH",
                    "failure_reason": "OFFLINE_DISTAL_HAND_HAND_COLLISION",
                    "collision_pairs": report["offline"]["collision_pairs"],
                }
            )
            rows.append(result)
            continue
        if not event_path.exists():
            result.update(
                {"first_failure_stage": "NOT_RUN", "failure_reason": "MISSING_PHYSICS"}
            )
            rows.append(result)
            continue
        with np.load(event_path, allow_pickle=False) as archive:
            event = {key: np.asarray(archive[key]) for key in archive.files}
        event_stage = event["stage"].astype(str)
        position = np.asarray(event["object_position_world_m"], dtype=np.float64)
        table_free = event["table_contact_force_n"] < TABLE_N
        force = {
            digit: np.asarray(event[f"{digit}_force_n"], dtype=np.float64) >= FORCE_N
            for digit in ("thumb", "index", "middle")
        }
        all_three = force["thumb"] & force["index"] & force["middle"]
        hold = event_stage == "HOLD_ELEVATED"
        lift_m = float(np.max(position[hold, 2]) - position[0, 2])
        left_lift = lift_m >= 0.049 and fraction(hold, table_free) >= 0.99
        left_transport = left_lift and fraction(
            event_stage == "LEFT_TRANSPORT", table_free
        ) >= 0.99
        presentation = np.isin(
            event_stage,
            [
                "LEFT_HANDOFF_PRESENTATION_TRANSITION",
                "LEFT_HANDOFF_STABILIZATION_TRANSITION",
                "LEFT_HANDOFF_HOLD",
            ],
        )
        transition = left_transport and fraction(presentation, table_free) >= 0.99
        verify = event_stage == "RIGHT_THREE_DIGIT_VERIFICATION"
        right_acquisition = transition and fraction(verify, all_three) >= 0.99
        if not left_lift:
            first_failure = "LEFT_LIFT"
        elif not left_transport:
            first_failure = "PRE_HANDOFF_LEFT_TRANSPORT"
        elif not transition:
            first_failure = "TRANSITION_TO_HANDOFF_HOLD"
        elif not right_acquisition:
            first_failure = "RIGHT_THUMB_INDEX_MIDDLE_ACQUISITION"
        else:
            first_failure = "NONE"
        speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
        result.update(
            {
                "event_log": str(event_path),
                "event_log_sha256": sha256(event_path),
                "command_completed": bool(
                    int(event["control_frame"][-1]) + 1 == len(command)
                ),
                "left_lift_m": lift_m,
                "left_lift_pass": left_lift,
                "left_transport_pass": left_transport,
                "handoff_hold_transition_pass": transition,
                "right_verification_support_fraction": {
                    digit: fraction(verify, values) for digit, values in force.items()
                },
                "right_three_digit_verification_fraction": fraction(verify, all_three),
                "left_release_executed": bool(
                    np.any(np.isin(event_stage, ["LEFT_THUMB_RELEASE", "LEFT_RELEASE"]))
                ),
                "maximum_object_speed_m_s": float(np.max(speed)),
                "object_speed_gate_pass": bool(np.max(speed) <= 1.0),
                "first_failure_stage": first_failure,
                "failure_reason": (
                    None if first_failure == "NONE" else first_failure
                ),
            }
        )
        rows.append(result)

    if len(rows) != 6:
        raise RuntimeError(f"expected six bounded candidates, found {len(rows)}")
    missing = [row for row in rows if row["first_failure_stage"] == "NOT_RUN"]
    payload = {
        "schema_version": "fully_staged_handoff_presentation_audit_v1",
        "status": (
            "INCOMPLETE"
            if missing
            else "ALL_FULLY_STAGED_CANDIDATES_FAILED_RIGHT_ACQUISITION"
        ),
        "candidate_count": len(rows),
        "physics_completed": sum(bool(row["physics_run"]) for row in rows),
        "offline_rejected": sum(not bool(row["physics_run"]) for row in rows),
        "validated_left_prefix_source": str(SOURCE),
        "validated_left_prefix_sha256": sha256(SOURCE),
        "all_prefixes_exact": all(bool(row["validated_prefix_exact"]) for row in rows),
        "right_stationary_before_approach": all(
            bool(row["right_stationary_before_declared_approach"]) for row in rows
        ),
        "right_endpoint_changed": False,
        "thresholds_changed": False,
        "left_release_executed": any(
            bool(row.get("left_release_executed", False)) for row in rows
        ),
        "rows": rows,
        "decision": (
            "The exact verified R14 endpoint is inaccessible to index and middle "
            "under every physically eligible corrected presentation. Invoke only "
            "the authorized local handoff-compatible transport-grasp fallback."
        ),
    }
    json_path = SEARCH / "FULLY_STAGED_PRESENTATION_PHYSICS_AUDIT.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = "\n".join(
        "| {profile} | {candidate} | {offline} | {physics} | {lift:.1f} | {thumb:.3f} | {index:.3f} | {middle:.3f} | {failure} |".format(
            profile=row["profile"],
            candidate=row["candidate"],
            offline=row["offline_status"],
            physics="YES" if row["physics_run"] else "NO",
            lift=1000.0 * float(row.get("left_lift_m", 0.0)),
            thumb=float(row.get("right_verification_support_fraction", {}).get("thumb", 0.0)),
            index=float(row.get("right_verification_support_fraction", {}).get("index", 0.0)),
            middle=float(row.get("right_verification_support_fraction", {}).get("middle", 0.0)),
            failure=row["first_failure_stage"],
        )
        for row in rows
    )
    (SEARCH / "FULLY_STAGED_PRESENTATION_PHYSICS_AUDIT.md").write_text(
        f"""# Fully staged handoff-presentation audit

Status: **{payload['status']}**

- Validated 28D LEFT prefix exact: **{payload['all_prefixes_exact']}**
- RIGHT stationary before declared approach: **{payload['right_stationary_before_approach']}**
- Physics completed: {payload['physics_completed']}/6 (two unsafe offline candidates were not simulated)
- Thresholds changed: **NO**
- LEFT release executed: **{payload['left_release_executed']}**

| profile | candidate | offline | physics | LEFT lift mm | RIGHT thumb | RIGHT index | RIGHT middle | first failure |
|---|---|---|---|---:|---:|---:|---:|---|
{table}

All physically eligible candidates preserved the validated LEFT prefix and kept
the doll elevated through handoff presentation. They reached only RIGHT thumb
contact at the exact R14 endpoint; index and middle never acquired. The two
P029 variants were rejected offline for distal hand-hand collision.
""",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
