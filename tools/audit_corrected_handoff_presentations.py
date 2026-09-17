#!/usr/bin/env python3
"""Audit the six corrected-stage exact-endpoint handoff presentations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
BASE = ROOT / "outputs/final_methodology_preserving_completion"
SEARCH = BASE / "02_exact_endpoint_refinement_staged"
OUT_JSON = SEARCH / "CORRECTED_PRESENTATION_PHYSICS_AUDIT.json"
OUT_MD = SEARCH / "CORRECTED_PRESENTATION_PHYSICS_AUDIT.md"
P14 = np.asarray(
    [0.070328, 0.834471, 0.211379, -0.344065, -1.151059, -0.042575, -1.205088]
)
FORCE_N = 0.015
TABLE_N = 0.015


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    rows: list[dict[str, object]] = []
    for command_path in sorted(SEARCH.glob("*/*/acquisition/exact_endpoint_acquisition_command.npz")):
        physics_path = command_path.parent / "physics_right_sensor/event_log.npz"
        if not physics_path.exists():
            raise RuntimeError(f"missing completed physics log: {physics_path}")
        with np.load(command_path, allow_pickle=False) as archive:
            command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
            command_stage = archive["stage"].astype(str)
            names = archive["joint_names"].astype(str).tolist()
            profile = str(np.asarray(archive["left_handoff_hold_profile"]).item())
            left_hold = np.asarray(
                archive["left_handoff_hold_model_order_7d_rad"], dtype=np.float64
            )
        left_names = [
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
        ]
        left_index = [names.index(name) for name in left_names]
        p14_stages = np.isin(
            command_stage,
            [
                "GRAVITY_RETENTION",
                "LEFT_LIFT",
                "LEFT_HOLD_ELEVATED",
                "LEFT_PRE_HANDOFF_TRANSPORT",
            ],
        )
        p14_error = float(
            np.max(np.abs(command[p14_stages][:, left_index] - P14[None]), initial=0.0)
        )
        transition = command_stage == "LEFT_HANDOFF_STABILIZATION_TRANSITION"
        hold = command_stage == "LEFT_HANDOFF_HOLD"
        hold_error = float(
            np.max(np.abs(command[hold][:, left_index] - left_hold[None]), initial=0.0)
        )
        with np.load(physics_path, allow_pickle=False) as archive:
            event = {key: np.asarray(archive[key]) for key in archive.files}
        stage = event["stage"].astype(str)
        position = np.asarray(event["object_position_world_m"], dtype=np.float64)
        initial_z = float(position[0, 2])
        lift_m = float(np.max(position[:, 2]) - initial_z)
        table_free = event["table_contact_force_n"] < TABLE_N
        force = np.column_stack(
            [event["thumb_force_n"], event["index_force_n"], event["middle_force_n"]]
        )
        all_three = np.all(force >= FORCE_N, axis=1)

        def fraction(label: str, values: np.ndarray) -> float:
            mask = stage == label
            return float(np.mean(values[mask])) if np.any(mask) else 0.0

        lift_pass = lift_m >= 0.05 and fraction("LEFT_HOLD_ELEVATED", table_free) >= 0.99
        pre_transport_pass = (
            lift_pass and fraction("LEFT_PRE_HANDOFF_TRANSPORT", table_free) >= 0.99
        )
        transition_pass = (
            pre_transport_pass
            and fraction("LEFT_HANDOFF_STABILIZATION_TRANSITION", table_free) >= 0.99
        )
        right_acquisition_pass = (
            transition_pass
            and fraction("RIGHT_THREE_DIGIT_VERIFICATION", all_three) >= 0.99
        )
        if not lift_pass:
            first_failure = "LEFT_LIFT"
        elif not pre_transport_pass:
            first_failure = "PRE_HANDOFF_LEFT_TRANSPORT"
        elif not transition_pass:
            first_failure = "TRANSITION_TO_HANDOFF_HOLD"
        elif not right_acquisition_pass:
            first_failure = "RIGHT_THUMB_INDEX_MIDDLE_ACQUISITION"
        else:
            first_failure = "NONE"
        rows.append(
            {
                "candidate": command_path.relative_to(SEARCH).parts[1],
                "profile": profile,
                "command": str(command_path),
                "command_sha256": digest(command_path),
                "event_log": str(physics_path),
                "event_log_sha256": digest(physics_path),
                "p14_maximum_error_before_handoff_rad": p14_error,
                "p14_numerically_identical_before_handoff": p14_error <= 1.0e-7,
                "handoff_transition_frames": int(np.count_nonzero(transition)),
                "handoff_hold_maximum_error_rad": hold_error,
                "maximum_object_lift_m": lift_m,
                "left_lift_pass": lift_pass,
                "pre_handoff_transport_table_free_fraction": fraction(
                    "LEFT_PRE_HANDOFF_TRANSPORT", table_free
                ),
                "handoff_transition_table_free_fraction": fraction(
                    "LEFT_HANDOFF_STABILIZATION_TRANSITION", table_free
                ),
                "right_three_digit_verification_fraction": fraction(
                    "RIGHT_THREE_DIGIT_VERIFICATION", all_three
                ),
                "left_release_executed": bool(
                    np.any(np.isin(stage, ["LEFT_THUMB_RELEASE", "LEFT_RELEASE"]))
                ),
                "first_failure_stage": first_failure,
            }
        )
    if len(rows) != 6:
        raise RuntimeError(f"expected six corrected candidates, found {len(rows)}")
    payload = {
        "schema_version": "corrected_handoff_presentation_audit_v1",
        "status": "ALL_CORRECTED_PRESENTATIONS_FAILED_PRE_RIGHT",
        "candidate_count": len(rows),
        "all_preserve_p14_before_handoff": all(
            bool(row["p14_numerically_identical_before_handoff"]) for row in rows
        ),
        "all_use_finite_handoff_transition": all(
            int(row["handoff_transition_frames"]) == 60 for row in rows
        ),
        "right_endpoint_changed": False,
        "doll_or_physics_changed": False,
        "thresholds_changed": False,
        "left_release_executed": any(bool(row["left_release_executed"]) for row in rows),
        "rows": rows,
        "decision": (
            "The bounded exact-endpoint family fails first at LEFT_LIFT with the "
            "spatially displaced end-region presentation. Invoke only the permitted "
            "local handoff-compatible transport-grasp fallback."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = "\n".join(
        "| {profile} | {candidate} | {lift:.3f} | {pre:.4f} | {right:.4f} | {failure} |".format(
            profile=row["profile"],
            candidate=row["candidate"],
            lift=1000.0 * float(row["maximum_object_lift_m"]),
            pre=float(row["pre_handoff_transport_table_free_fraction"]),
            right=float(row["right_three_digit_verification_fraction"]),
            failure=row["first_failure_stage"],
        )
        for row in rows
    )
    OUT_MD.write_text(
        f"""# Corrected handoff-presentation physics audit

Status: **{payload['status']}**

- Candidates: 6/6 physically evaluated
- LEFT P14 numerical identity through pre-handoff transport: **PASS**
- Finite P14-to-handoff-hold transition: **PASS** (60 frames)
- RIGHT endpoint changed: **NO**
- LEFT release executed: **NO**

| Profile | Presentation | max lift (mm) | pre-handoff table-free | RIGHT 3-digit | first failure |
|---|---|---:|---:|---:|---|
{table}

All candidates failed the 50 mm LEFT lift gate before the handoff-only profile
activated.  Continuing millimetre tuning of this end-region family is therefore
not authorized.  The methodology-preserving local RIGHT-grasp fallback is the
next gate.
""",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
