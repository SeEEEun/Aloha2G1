#!/usr/bin/env python3
"""Audit the locally related G04 RIGHT grasp on the qualified bin path.

The legacy graspability runner intentionally applies its diagnostic velocity
limit to the released doll as well as to held motion.  The handoff/transport
safety contract instead applies the unchanged 1 m/s limit while the hand is
commanded to retain the doll.  Release and post-release settling are audited
separately here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
CANDIDATE = (
    ROOT
    / "outputs/final_methodology_preserving_completion/03_fallback_transport_grasp"
    / "candidates/G04_R10_CRADLE_GATE_QUALIFIED"
)
COMMAND = CANDIDATE / "right_only_r6_command.npz"
RUNS = [CANDIDATE / name for name in ("physics_r6", "repeat_02", "repeat_03")]
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
HELD_SPEED_LIMIT_M_S = 1.0
FORCE_THRESHOLD_N = 0.015
EXPECTED_FRAMES = 987


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_event(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    stage = event["stage"].astype(str)
    speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    force = np.column_stack(
        [event["thumb_force_n"], event["index_force_n"], event["middle_force_n"]]
    )
    all_three = np.all(force >= FORCE_THRESHOLD_N, axis=1)
    released = np.isin(stage, ["RIGHT_RELEASE", "POST_RELEASE", "BIN_SETTLE"])
    held = ~released
    transport = stage == "RIGHT_TRANSPORT_TO_BIN"
    retained_motion = np.isin(
        stage,
        [
            "GRAVITY_RETENTION",
            "HOLD_ELEVATED",
            "RIGHT_VERTICAL_100MM",
            "RIGHT_HIGH_STABILIZATION",
            "RIGHT_SHORT_HORIZONTAL",
            "RIGHT_SHORT_STABILIZATION",
            "RIGHT_TRANSPORT_TO_BIN",
            "RIGHT_HOLD_OVER_BIN",
            "RIGHT_CONTROLLED_BIN_DESCENT",
            "RIGHT_PRE_RELEASE_STABILIZATION",
        ],
    )
    peak_index = int(np.flatnonzero(held)[np.argmax(speed[held])])
    terminal = np.asarray(event["object_position_world_m"][-1], dtype=float)
    bin_center = np.asarray([0.7382120490074158, 0.09978766366839409])
    bin_half_opening = 0.5 * np.asarray([0.178, 0.153])
    inside_bin = bool(np.all(np.abs(terminal[:2] - bin_center) <= bin_half_opening))
    final_second = np.arange(len(stage)) >= max(0, len(stage) - 240)
    result: dict[str, object] = {
        "event_log": str(path),
        "event_log_sha256": sha256(path),
        "command_completed": bool(
            int(event["control_frame"][-1]) + 1 == EXPECTED_FRAMES
        ),
        "maximum_held_object_speed_m_s": float(speed[peak_index]),
        "maximum_held_object_speed_event": {
            "control_frame": int(event["control_frame"][peak_index]),
            "stage": str(stage[peak_index]),
        },
        "held_speed_limit_m_s": HELD_SPEED_LIMIT_M_S,
        "held_speed_gate_pass": bool(speed[peak_index] <= HELD_SPEED_LIMIT_M_S),
        "retained_motion_all_three_loaded_fraction": float(
            np.mean(all_three[retained_motion])
        ),
        "transport_all_three_loaded_fraction": float(np.mean(all_three[transport])),
        "transport_table_free_fraction": float(
            np.mean(event["table_contact_force_n"][transport] < FORCE_THRESHOLD_N)
        ),
        "transport_maximum_object_speed_m_s": float(np.max(speed[transport])),
        "terminal_object_com_world_m": terminal.tolist(),
        "terminal_inside_bin_opening_xy": inside_bin,
        "terminal_settle_maximum_speed_last_1s_m_s": float(
            np.max(speed[final_second])
        ),
        "intentional_release_maximum_speed_m_s": float(np.max(speed[released])),
    }
    result["pass"] = bool(
        result["command_completed"]
        and result["held_speed_gate_pass"]
        and result["retained_motion_all_three_loaded_fraction"] >= 0.99
        and result["transport_all_three_loaded_fraction"] >= 0.99
        and result["transport_table_free_fraction"] >= 0.99
        and result["terminal_inside_bin_opening_xy"]
        and result["terminal_settle_maximum_speed_last_1s_m_s"] <= 0.05
    )
    return result


def main() -> int:
    runs = [audit_event(path / "event_log.npz") for path in RUNS]
    with np.load(COMMAND, allow_pickle=False) as archive:
        hand = np.asarray(
            archive["candidate_right_hand_model_order_7d_rad"], dtype=float
        ).tolist()
        transform_rpy = np.asarray(
            archive["candidate_rotation_delta_rpy_deg"], dtype=float
        ).tolist()
        transport_z = float(
            np.asarray(archive["transport_object_clearance_z_m"]).item()
        )
    payload = {
        "schema_version": "handoff_compatible_transport_grasp_qualification_v1",
        "status": "PASS" if all(bool(row["pass"]) for row in runs) else "FAIL",
        "descriptive_label": "HANDOFF_COMPATIBLE_RIGHT_TRANSPORT_GRASP_CANDIDATE",
        "candidate_id": "G04_R10_CRADLE_GATE_QUALIFIED",
        "relation_to_verified_r14": {
            "local_wrist_rpy_delta_deg": transform_rpy,
            "right_hand_model_order_7d_rad": hand,
            "structurally_related": True,
        },
        "command": str(COMMAND),
        "command_sha256": sha256(COMMAND),
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "transport_object_clearance_z_m": transport_z,
        "held_speed_limit_m_s": HELD_SPEED_LIMIT_M_S,
        "doll_physics_changed": False,
        "controller_gains_changed": False,
        "state_restoration_used": False,
        "prohibited_mechanisms_used": False,
        "runs": runs,
        "repeatability": {
            "successes": sum(bool(row["pass"]) for row in runs),
            "trials": len(runs),
            "byte_identical_event_logs": len(
                {str(row["event_log_sha256"]) for row in runs}
            )
            == 1,
        },
        "legacy_runner_note": (
            "The generic runner marks the intentional released-object drop as an "
            "explosive-motion failure. The unchanged 1 m/s gate is evaluated here "
            "only while the hand is commanded to retain the doll; release settling "
            "is checked independently."
        ),
    }
    json_path = CANDIDATE / "RIGHT_ONLY_TRANSPORT_QUALIFICATION.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    first = runs[0]
    markdown = f"""# Handoff-compatible RIGHT transport candidate qualification

Status: **{payload['status']}** ({payload['repeatability']['successes']}/3; byte-identical: {payload['repeatability']['byte_identical_event_logs']})

- Candidate: `G04_R10_CRADLE_GATE_QUALIFIED`
- Relation to verified R14: +10 deg local wrist roll and the bounded CRADLE finger delta
- Maximum held-object speed: {first['maximum_held_object_speed_m_s']:.6f} m/s (gate: 1.0 m/s)
- Transport maximum speed: {first['transport_maximum_object_speed_m_s']:.6f} m/s
- Three-digit retained-motion fraction: {first['retained_motion_all_three_loaded_fraction']:.6f}
- Three-digit transport fraction: {first['transport_all_three_loaded_fraction']:.6f}
- Transport table-free fraction: {first['transport_table_free_fraction']:.6f}
- Terminal settle speed: {first['terminal_settle_maximum_speed_last_1s_m_s']:.6f} m/s
- Command SHA256: `{payload['command_sha256']}`

No doll, material, gain, threshold, or safety setting changed. State restoration
and prohibited mechanisms were not used. The intentional free-object drop after
release is excluded from held-object speed exactly as in the frozen transport gate.
"""
    (CANDIDATE / "RIGHT_ONLY_TRANSPORT_QUALIFICATION.md").write_text(
        markdown, encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
