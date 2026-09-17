#!/usr/bin/env python3
"""Extract the first verified RIGHT-only one-second post-handoff state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
SOURCE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral/backward_constructed_handoff"
    / "B2_PATH_F40/right_preload_partial_left_relax_exact_endpoint_v4/full"
)
EVENT = SOURCE / "physics_right_sensor/event_log.npz"
TRIAL = SOURCE / "physics_right_sensor/trial_result.json"
COMMAND = SOURCE / "scripted_full_task_command.npz"
OFFLINE = SOURCE / "offline_report.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
OUT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "success_first_common_execution"
    / "post_handoff_keyframe"
)
EXPECTED = {
    EVENT: "9508e6d8d9e91a9d8955e12d6697145896eb0fb4e7a2457bde828b4b858ab016",
    COMMAND: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def first_sustained(mask: np.ndarray, count: int) -> int:
    values = np.asarray(mask, dtype=np.int8)
    totals = np.convolve(values, np.ones(count, dtype=np.int64), mode="valid")
    rows = np.flatnonzero(totals == count)
    if not len(rows):
        raise RuntimeError("no sustained post-handoff interval exists")
    return int(rows[0])


def main() -> int:
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"authoritative source hash changed: {path}")
    config = read_json(CONFIG)
    trial = read_json(TRIAL)
    offline = read_json(OFFLINE)
    if (
        trial["runtime_right_three_digit_gate"]["status"] != "PASS"
        or not trial["command_completed"]
        or offline["status"] != "OFFLINE_PASS"
        or offline["joint_limit_violations"] != 0
        or sum(offline["robot_collision_frame_counts"].values()) != 0
    ):
        raise RuntimeError("source run no longer satisfies provenance gates")

    with np.load(EVENT, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    names = arrays["joint_names"].astype(str).tolist()
    lookup = {name: index for index, name in enumerate(names)}
    left_names = [
        f"left_hand_{joint}_joint"
        for joint in config["joint_order_7d"]
    ]
    left_indices = np.asarray([lookup[name] for name in left_names])
    left_open = np.asarray(config["hand_states"]["left"]["OPEN"], dtype=np.float64)
    left_command_open = np.all(
        np.isclose(
            arrays["commanded_q_rad"][:, left_indices],
            left_open[None],
            atol=1.0e-7,
            rtol=0.0,
        ),
        axis=1,
    )
    forces = np.column_stack(
        [arrays[f"{digit}_force_n"] for digit in ("thumb", "index", "middle")]
    )
    force_threshold = float(config["gates"]["meaningful_digit_force_n"])
    table_threshold = float(
        config["gates"]["maximum_table_force_for_elevated_n"]
    )
    elevated_threshold = float(arrays["object_position_world_m"][0, 2] + 0.03)
    eligible = (
        (arrays["stage"].astype(str) == "RIGHT_POST_RELEASE_RETENTION")
        & left_command_open
        & np.all(forces >= force_threshold, axis=1)
        & (arrays["table_contact_force_n"] <= table_threshold)
        & (arrays["object_position_world_m"][:, 2] >= elevated_threshold)
    )
    physics_hz = round(1.0 / float(np.median(np.diff(arrays["timestamp_s"]))))
    row = first_sustained(eligible, physics_hz)
    interval = slice(row, row + physics_hz)
    if int(arrays["physics_step"][row]) != 4080:
        raise RuntimeError("authoritative first qualifying keyframe moved")

    keyframe = OUT / "POST_HANDOFF_KEYFRAME.npz"
    atomic_npz(
        keyframe,
        schema_version=np.asarray("post_handoff_keyframe_v1"),
        joint_names=arrays["joint_names"],
        measured_q_rad=arrays["measured_q_rad"][row],
        measured_qd_rad_s=arrays["measured_qd_rad_s"][row],
        commanded_q_rad=arrays["commanded_q_rad"][row],
        object_position_world_m=arrays["object_position_world_m"][row],
        object_quaternion_xyzw=arrays["object_quaternion_xyzw"][row],
        object_linear_velocity_m_s=arrays["object_linear_velocity_m_s"][row],
        object_angular_velocity_rad_s=arrays["object_angular_velocity_rad_s"][row],
        source_physics_step=arrays["physics_step"][row],
        source_control_frame=arrays["control_frame"][row],
        source_timestamp_s=arrays["timestamp_s"][row],
        source_stage=arrays["stage"][row],
        state_restoration_debug_only=np.asarray(True),
        permitted_in_final_continuous_validation=np.asarray(False),
    )
    manifest = {
        "schema_version": "post_handoff_keyframe_manifest_v1",
        "status": "POST_HANDOFF_KEYFRAME_VERIFIED",
        "keyframe": str(keyframe),
        "keyframe_sha256": sha256_file(keyframe),
        "source": {
            "run": str(SOURCE),
            "event_log": str(EVENT),
            "event_log_sha256": sha256_file(EVENT),
            "command": str(COMMAND),
            "command_sha256": sha256_file(COMMAND),
            "trial_result": str(TRIAL),
            "trial_result_sha256": sha256_file(TRIAL),
            "offline_report": str(OFFLINE),
            "offline_report_sha256": sha256_file(OFFLINE),
        },
        "first_qualifying_state": {
            "event_row": row,
            "physics_step": int(arrays["physics_step"][row]),
            "control_frame": int(arrays["control_frame"][row]),
            "timestamp_s": float(arrays["timestamp_s"][row]),
            "stage": str(arrays["stage"][row]),
            "left_command_exactly_open": bool(left_command_open[row]),
            "object_com_world_m": arrays["object_position_world_m"][row].tolist(),
            "object_linear_velocity_m_s": arrays["object_linear_velocity_m_s"][row].tolist(),
            "object_angular_velocity_rad_s": arrays["object_angular_velocity_rad_s"][row].tolist(),
            "right_force_n_thumb_index_middle": forces[row].tolist(),
            "table_force_n": float(arrays["table_contact_force_n"][row]),
        },
        "following_one_second_verification": {
            "physics_steps": physics_hz,
            "minimum_right_force_n_thumb_index_middle": np.min(
                forces[interval], axis=0
            ).tolist(),
            "mean_right_force_n_thumb_index_middle": np.mean(
                forces[interval], axis=0
            ).tolist(),
            "maximum_table_force_n": float(
                np.max(arrays["table_contact_force_n"][interval])
            ),
            "minimum_object_com_z_m": float(
                np.min(arrays["object_position_world_m"][interval, 2])
            ),
            "all_three_sustained": True,
            "doll_elevated": True,
        },
        "robot_safety": {
            "joint_limit_violations": int(offline["joint_limit_violations"]),
            "collision_frame_counts": offline["robot_collision_frame_counts"],
            "ik_max_position_error_m": float(offline["ik_max_position_error_m"]),
            "ik_max_orientation_error_rad": float(
                offline["ik_max_orientation_error_rad"]
            ),
        },
        "state_restoration_scope": "TAIL_DEBUGGING_ONLY",
        "final_continuous_validation_may_restore_state": False,
        "prohibited_mechanisms_used": False,
    }
    atomic_json(OUT / "POST_HANDOFF_KEYFRAME_MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
