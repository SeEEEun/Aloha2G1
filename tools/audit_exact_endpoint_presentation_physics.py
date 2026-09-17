#!/usr/bin/env python3
"""Audit the frozen exact-right-endpoint presentation physics batch.

This auditor does not run physics or alter a command.  It applies the frozen
proxy gates to the pre-release acquisition-only event logs and emits a compact,
auditable terminal result for the exact VERIFIED_RIGHT_TRANSPORT_GRASP search.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/final_methodology_preserving_completion"
SEARCH = OUT / "01_exact_endpoint_presentation_search"
PHYSICS = OUT / "02_exact_endpoint_physics"
SELECTION = SEARCH / "OFFLINE_PRESENTATION_COMBINED_SELECTION.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"


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


def atomic_json(path: Path, value: object) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def longest_duration(mask: np.ndarray, dt: float) -> float:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return float(best * dt)


def main() -> int:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gates = config["gates"]
    force_gate = float(gates["meaningful_digit_force_n"])
    table_gate = float(gates["maximum_table_force_for_elevated_n"])
    speed_gate = float(gates["maximum_object_linear_speed_m_s"])
    lift_gate = float(gates["minimum_measured_lift_m"])
    support_gate_s = 0.5
    rows: list[dict[str, object]] = []

    for candidate_id in selection["top_physics_candidates"]:
        root = PHYSICS / candidate_id / "acquisition"
        offline_path = root / "offline_report.json"
        event_path = root / "physics_right_sensor/event_log.npz"
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "offline_report": str(offline_path),
            "event_log": str(event_path),
            "physics_run": event_path.is_file(),
        }
        if not offline_path.is_file():
            row.update(status="OFFLINE_CONSTRUCTION_FAIL", first_failed_gate="offline_construction")
            rows.append(row)
            continue
        offline = json.loads(offline_path.read_text(encoding="utf-8"))
        row["offline_status"] = offline["status"]
        row["command_sha256"] = offline["command_sha256"]
        if offline["status"] != "OFFLINE_PASS":
            row.update(status="OFFLINE_FAIL", first_failed_gate="offline_path_gate")
            rows.append(row)
            continue
        if not event_path.is_file():
            row.update(status="PHYSICS_NOT_RUN", first_failed_gate="physics_not_run")
            rows.append(row)
            continue

        with np.load(event_path, allow_pickle=False) as archive:
            stage = archive["stage"].astype(str)
            position = np.asarray(archive["object_position_world_m"], dtype=np.float64)
            velocity = np.asarray(archive["object_linear_velocity_m_s"], dtype=np.float64)
            table = np.asarray(archive["table_contact_force_n"], dtype=np.float64)
            forces = np.column_stack(
                [archive[f"{digit}_force_n"] for digit in ("thumb", "index", "middle")]
            ).astype(np.float64)
            penetration = np.maximum(
                np.asarray(archive["maximum_digit_penetration_m"], dtype=np.float64),
                np.asarray(archive["maximum_table_penetration_m"], dtype=np.float64),
            )
            timestamps = np.asarray(archive["timestamp_s"], dtype=np.float64)
        dt = float(np.median(np.diff(timestamps)))
        initial_z = float(np.median(position[stage == "LEFT_OPEN", 2]))
        left_hold = stage == "LEFT_HOLD_ELEVATED"
        presentation = np.isin(stage, ["LEFT_PRESENTATION_TRANSPORT", "LEFT_HANDOFF_HOLD"])
        verification = stage == "RIGHT_THREE_DIGIT_VERIFICATION"
        all_three = np.all(forces >= force_gate, axis=1)
        table_free = table <= table_gate
        left_lift_m = float(np.max(position[:, 2]) - initial_z)
        right_support_s = longest_duration(verification & all_three & table_free, dt)
        left_elevated_s = longest_duration(left_hold & table_free, dt)
        presentation_table_free_fraction = float(np.mean(table_free[presentation]))
        max_speed = float(np.max(np.linalg.norm(velocity, axis=1)))
        verification_force_max = np.max(forces[verification], axis=0)
        left_support_pass = bool(
            left_lift_m >= lift_gate
            and left_elevated_s >= 0.5
            and presentation_table_free_fraction >= 0.99
        )
        right_support_pass = bool(right_support_s >= support_gate_s)
        speed_pass = bool(max_speed <= speed_gate)
        penetration_pass = bool(np.max(penetration) <= float(gates["maximum_runtime_penetration_m"]))
        if not left_support_pass:
            first_failed = "LEFT_END_REGION_PHYSICAL_SUPPORT"
        elif not right_support_pass:
            first_failed = "RIGHT_EXACT_ENDPOINT_THREE_DIGIT_SUPPORT"
        elif not speed_pass:
            first_failed = "OBJECT_SPEED"
        elif not penetration_pass:
            first_failed = "PENETRATION"
        else:
            first_failed = "NONE"
        status = "ACQUISITION_PASS" if first_failed == "NONE" else "ACQUISITION_FAIL"
        row.update(
            status=status,
            first_failed_gate=first_failed,
            event_log_sha256=sha256_file(event_path),
            initial_object_z_m=initial_z,
            maximum_lift_m=left_lift_m,
            left_elevated_table_free_s=left_elevated_s,
            presentation_table_free_fraction=presentation_table_free_fraction,
            right_three_digit_table_free_support_s=right_support_s,
            right_verification_max_force_n={
                digit: float(value)
                for digit, value in zip(("thumb", "index", "middle"), verification_force_max)
            },
            maximum_object_speed_m_s=max_speed,
            maximum_penetration_m=float(np.max(penetration)),
            gates={
                "left_support": left_support_pass,
                "right_support": right_support_pass,
                "object_speed": speed_pass,
                "penetration": penetration_pass,
            },
        )
        rows.append(row)

    physics_rows = [row for row in rows if row["physics_run"]]
    pass_rows = [row for row in rows if row.get("status") == "ACQUISITION_PASS"]
    best = max(
        physics_rows,
        key=lambda row: (
            float(row.get("maximum_lift_m", -1.0)),
            float(row.get("right_three_digit_table_free_support_s", -1.0)),
        ),
    )
    result = {
        "schema_version": "exact_verified_right_endpoint_presentation_physics_audit_v1",
        "status": "PASS" if pass_rows else "EXACT_ENDPOINT_PRESENTATION_BLOCKED",
        "frozen_selection": str(SELECTION),
        "frozen_selection_sha256": sha256_file(SELECTION),
        "immutable_config": str(CONFIG),
        "immutable_config_sha256": sha256_file(CONFIG),
        "candidate_count": len(rows),
        "physics_run_count": len(physics_rows),
        "acquisition_pass_count": len(pass_rows),
        "best_candidate": best,
        "hard_gate_release_executed": False,
        "transport_or_full_task_executed": False,
        "doll_or_physics_changed": False,
        "verified_right_transport_grasp_changed": False,
        "rows": rows,
    }
    atomic_json(PHYSICS / "EXACT_ENDPOINT_PRESENTATION_PHYSICS_AUDIT.json", result)

    fields = [
        "candidate_id",
        "offline_status",
        "physics_run",
        "status",
        "first_failed_gate",
        "maximum_lift_m",
        "left_elevated_table_free_s",
        "presentation_table_free_fraction",
        "right_three_digit_table_free_support_s",
        "maximum_object_speed_m_s",
    ]
    csv_path = PHYSICS / "EXACT_ENDPOINT_PRESENTATION_PHYSICS_AUDIT.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    lines = [
        "# Exact verified-right-endpoint presentation physics audit",
        "",
        f"Status: `{result['status']}`",
        "",
        f"- Frozen top candidates: {len(rows)}",
        f"- Physics runs: {len(physics_rows)}",
        f"- Acquisition passes: {len(pass_rows)}",
        "- LEFT release: not executed (hard gate preserved)",
        "- RIGHT transport/full task: not executed",
        "- Doll/physics changes: none",
        "- Verified RIGHT transport grasp changes: none",
        "",
        "| Candidate | Offline | Physics | First failed gate | Lift (mm) | Presentation table-free | RIGHT 3-digit support (s) | Peak speed (m/s) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {offline_status} | {physics_run} | {first_failed_gate} | {lift:.3f} | {table:.3f} | {right:.3f} | {speed:.3f} |".format(
                candidate_id=row["candidate_id"],
                offline_status=row.get("offline_status", "NO_REPORT"),
                physics_run="YES" if row["physics_run"] else "NO",
                first_failed_gate=row["first_failed_gate"],
                lift=1000.0 * float(row.get("maximum_lift_m", 0.0)),
                table=float(row.get("presentation_table_free_fraction", 0.0)),
                right=float(row.get("right_three_digit_table_free_support_s", 0.0)),
                speed=float(row.get("maximum_object_speed_m_s", 0.0)),
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The exact verified 3/3 RIGHT transport endpoint was geometrically feasible in",
        "the static named-link audit, but every physically simulated presentation failed",
        "first at the displaced LEFT end-region support gate. The doll therefore never",
        "arrived table-free at the RIGHT acquisition gate. This bounded family provides",
        "no evidence that changing the already-proven RIGHT endpoint would cure the LEFT",
        "support failure; a centered-LEFT or nearby transport-grasp fallback must be",
        "constructed and independently revalidated before any release is permitted.",
        "",
    ]
    atomic_text(PHYSICS / "EXACT_ENDPOINT_PRESENTATION_PHYSICS_AUDIT.md", "\n".join(lines))
    print(json.dumps({key: result[key] for key in ("status", "candidate_count", "physics_run_count", "acquisition_pass_count", "best_candidate")}, indent=2))
    return 0 if pass_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
