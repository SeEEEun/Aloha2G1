#!/usr/bin/env python3
"""Audit exact R26 and build a bounded fixed-geometry retiming family.

This tool never simulates and never modifies the frozen doll, R14 artifact, or
source R26 command.  It records the complete original gate evidence, exposes
the historical named-joint refinement mismatch, and time-scales only the
recorded R26 handoff stages.  The post-handoff T5/R14 transport sequence is
copied byte-for-value from the original command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import model_parts  # noqa: E402
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import (  # noqa: E402
    AUTHORITATIVE_REFERENCES,
    authoritative_joint_ranges,
)


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
SELECTED = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp"
    / "SELECTED_RIGHT_TRANSPORT_GRASP.json"
)
R26 = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
    / "R26_HANDOFF_R14_INDEX0_P010"
)
COMMAND = R26 / "continuous_r14_hand_t5_full_command.npz"
OFFLINE = R26 / "offline_report.json"
TRIAL = R26 / "physics_full/trial_result.json"
EVENT = R26 / "physics_full/event_log.npz"
DEFAULT_OUTPUT = ROOT / "outputs/final_task_completion_v1/08_r26_retiming"

EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    SELECTED: "10d406038795f2f4dfc38ecdfd408de176d63297f05dd4814fc1a4b29392b8c9",
    COMMAND: "2c525c492baaf88a47270679bc112d35a15f721933f0d62365b54625f0f4805e",
    EVENT: "d96a362c341f6bd551ed607a908ae8986908fe9b55164ec531ff73199577d956",
}
FORCE_N = 0.015
TABLE_N = 0.015
MAX_SPEED_M_S = 1.0
TIMING_SCALES = (1.5, 2.0, 3.0, 4.0)
HANDOFF_STAGES = (
    "RIGHT_COLLISION_FREE_APPROACH",
    "RIGHT_PRESHAPE",
    "RIGHT_PRESHAPE_HOLD",
    "BACKWARD_ACQUISITION_PATH",
    "RIGHT_THREE_DIGIT_PRELOAD",
    "LEFT_INDEX_MIDDLE_PARTIAL_RELAX_AFTER_RIGHT_PRELOAD",
    "RIGHT_T4_PRETRANSFER_VERIFICATION",
    "RIGHT_SUPPORTED_R14_HAND_ENCLOSURE",
    "RIGHT_THREE_DIGIT_VERIFICATION",
    "LEFT_THUMB_RELEASE",
    "RIGHT_POST_RELEASE_RETENTION",
)
KEY_POSE_STAGES = (
    "LEFT_HANDOFF_HOLD",
    "RIGHT_COLLISION_FREE_APPROACH",
    "BACKWARD_ACQUISITION_PATH",
    "RIGHT_T4_PRETRANSFER_VERIFICATION",
    "RIGHT_SUPPORTED_R14_HAND_ENCLOSURE",
    "RIGHT_THREE_DIGIT_VERIFICATION",
    "LEFT_THUMB_RELEASE",
    "RIGHT_POST_RELEASE_RETENTION",
)


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
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def longest_duration(mask: np.ndarray, dt: float) -> float:
    current = longest = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt)


def stage_summary(event: dict[str, np.ndarray], stage_name: str) -> dict[str, Any]:
    mask = event["stage"].astype(str) == stage_name
    if not np.any(mask):
        return {"present": False}
    forces = np.column_stack(
        (event["thumb_force_n"], event["index_force_n"], event["middle_force_n"])
    )[mask]
    speed = np.linalg.norm(event["object_linear_velocity_m_s"][mask], axis=1)
    table = event["table_contact_force_n"][mask]
    position = event["object_position_world_m"][mask]
    return {
        "present": True,
        "control_frame_start": int(np.min(event["control_frame"][mask])),
        "control_frame_end": int(np.max(event["control_frame"][mask])),
        "duration_s": float(np.count_nonzero(mask) / 8.0 / 30.0),
        "all_three_fraction": float(np.mean(np.all(forces >= FORCE_N, axis=1))),
        "any_digit_fraction": float(np.mean(np.any(forces >= FORCE_N, axis=1))),
        "table_free_fraction": float(np.mean(table < TABLE_N)),
        "mean_force_n_thumb_index_middle": forces.mean(axis=0).tolist(),
        "maximum_speed_m_s": float(np.max(speed, initial=0.0)),
        "minimum_object_z_m": float(np.min(position[:, 2])),
        "maximum_object_z_m": float(np.max(position[:, 2])),
    }


def wrist_pose_record(
    g1: G1Kinematics,
    names: list[str],
    row: np.ndarray,
) -> dict[str, Any]:
    arm, left, right = model_parts(names, g1, row)
    g1.assign(arm, left, right)
    pose = np.asarray(g1.wrist_pose("right"), dtype=np.float64)
    return {
        "arm_q_model_order_14d_rad": arm.tolist(),
        "right_hand_model_order_7d_rad": right.tolist(),
        "right_wrist_pose_model_4x4": pose.tolist(),
        "right_wrist_position_world_m": g1.model_to_world_position(pose[:3, 3]).tolist(),
    }


def resample_recorded_stage(
    previous: np.ndarray,
    original_rows: np.ndarray,
    new_count: int,
) -> np.ndarray:
    """Time-scale the recorded joint-space curve without changing its polyline."""
    source = np.vstack((previous[None], original_rows))
    old_axis = np.arange(len(source), dtype=np.float64)
    new_axis = np.linspace(0.0, float(len(source) - 1), new_count + 1)[1:]
    out = np.column_stack(
        [np.interp(new_axis, old_axis, source[:, joint]) for joint in range(source.shape[1])]
    )
    out[-1] = original_rows[-1]
    return out


def build_variant(
    scale: float,
    output: Path,
    command: dict[str, np.ndarray],
    g1: G1Kinematics,
) -> dict[str, Any]:
    q = np.asarray(command["commanded_q_rad"], dtype=np.float64)
    stages = command["stage"].astype(str)
    names = command["joint_names"].astype(str).tolist()
    first = int(np.flatnonzero(stages == HANDOFF_STAGES[0])[0])
    final = int(np.flatnonzero(stages == HANDOFF_STAGES[-1])[-1])
    rows = [row.copy() for row in q[:first]]
    labels = stages[:first].tolist()
    scaled_counts: dict[str, dict[str, int]] = {}
    cursor = first
    for stage_name in HANDOFF_STAGES:
        indices = np.flatnonzero(stages == stage_name)
        if not len(indices) or int(indices[0]) != cursor:
            raise RuntimeError(f"R26 handoff stage is missing or noncontiguous: {stage_name}")
        original_rows = q[indices]
        new_count = int(round(len(original_rows) * scale))
        new_rows = resample_recorded_stage(rows[-1], original_rows, new_count)
        rows.extend(new_rows)
        labels.extend([stage_name] * new_count)
        scaled_counts[stage_name] = {
            "original_frames": int(len(original_rows)),
            "retimed_frames": int(new_count),
        }
        cursor = int(indices[-1]) + 1
    if cursor != final + 1:
        raise RuntimeError("R26 handoff retiming did not consume the expected interval")
    rows.extend(q[final + 1 :])
    labels.extend(stages[final + 1 :].tolist())
    retimed = np.asarray(rows, dtype=np.float64)
    retimed_stages = np.asarray(labels)

    # Every stage endpoint and every post-handoff transport command is exact.
    endpoint_error = 0.0
    for stage_name in HANDOFF_STAGES:
        original_end = q[np.flatnonzero(stages == stage_name)[-1]]
        retimed_end = retimed[np.flatnonzero(retimed_stages == stage_name)[-1]]
        endpoint_error = max(endpoint_error, float(np.max(np.abs(original_end - retimed_end))))
    original_tail = q[final + 1 :]
    retimed_tail = retimed[np.flatnonzero(retimed_stages == stages[final + 1])[0] :]
    tail_exact = bool(np.array_equal(original_tail, retimed_tail))

    contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (retimed < lower[None] - 1.0e-9) | (retimed > upper[None] + 1.0e-9)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    geometry = g1.trajectory_geometry(
        retimed[:, arm_indices], retimed[:, left_indices], retimed[:, right_indices], 1.0e-5
    )
    collision_counts = {
        key: int(np.count_nonzero(value))
        for key, value in geometry["collision_flags"].items()
    }
    fps = float(np.asarray(command["control_fps_hz"]).item())
    dq = np.diff(retimed, axis=0) * fps
    ddq = np.diff(dq, axis=0) * fps
    dddq = np.diff(ddq, axis=0) * fps
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and endpoint_error <= 1.0e-12
        and tail_exact
    )

    variant_id = f"R26_T{TIMING_SCALES.index(scale) + 1}_{str(scale).replace('.', 'P')}X"
    variant_dir = output / "variants" / variant_id
    command_path = variant_dir / "retimed_r26_full_command.npz"
    arrays = {key: np.asarray(value) for key, value in command.items()}
    arrays.update(
        {
            "commanded_q_rad": retimed.astype(np.float32),
            "stage": retimed_stages,
            "r26_retiming_scale": np.asarray(scale),
            "r26_geometry_unchanged": np.asarray(True),
            "r26_transport_tail_exact": np.asarray(tail_exact),
            "maximum_object_speed_gate_m_s": np.asarray(MAX_SPEED_M_S),
        }
    )
    save_npz(command_path, arrays)
    report = {
        "schema_version": "r26_fixed_geometry_retiming_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "candidate_id": variant_id,
        "scale": scale,
        "frames": int(len(retimed)),
        "duration_s": float(len(retimed) / fps),
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "source_r26_command": str(COMMAND),
        "source_r26_command_sha256": EXPECTED[COMMAND],
        "spatial_geometry_unchanged": True,
        "handoff_stage_endpoint_max_error_rad": endpoint_error,
        "post_handoff_transport_tail_value_exact": tail_exact,
        "scaled_stage_counts": scaled_counts,
        "joint_limit_violation_count": int(np.count_nonzero(violations)),
        "collision_frame_counts": collision_counts,
        "maximum_adjacent_arm_step_rad": float(
            np.max(np.abs(np.diff(retimed[:, arm_indices], axis=0)), initial=0.0)
        ),
        "kinematics": {
            "maximum_joint_velocity_rad_s": float(np.max(np.abs(dq), initial=0.0)),
            "maximum_joint_acceleration_rad_s2": float(np.max(np.abs(ddq), initial=0.0)),
            "maximum_joint_jerk_rad_s3": float(np.max(np.abs(dddq), initial=0.0)),
        },
        "frozen_object_speed_gate_m_s": MAX_SPEED_M_S,
        "selected_r14_artifact_modified": False,
        "r26_endpoint_matches_frozen_r14": False,
        "state_restoration_used": False,
        "prohibited_mechanism_used": False,
        "policy_used": False,
        "real_robot_used": False,
    }
    atomic_json(variant_dir / "offline_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    for path, digest in EXPECTED.items():
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"authoritative input changed: {path}: {actual} != {digest}")

    selected = read_json(SELECTED)
    offline = read_json(OFFLINE)
    trial = read_json(TRIAL)
    with np.load(COMMAND, allow_pickle=False) as archive:
        command = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(EVENT, allow_pickle=False) as archive:
        event = {key: np.asarray(archive[key]) for key in archive.files}
    q = np.asarray(command["commanded_q_rad"], dtype=np.float64)
    stages = command["stage"].astype(str)
    names = command["joint_names"].astype(str).tolist()
    fps = float(np.asarray(command["control_fps_hz"]).item())
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or not np.isclose(fps, 30.0):
        raise RuntimeError("R26 authoritative 28D contract changed")

    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    lookup = {name: index for index, name in enumerate(names)}
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    selected_model = np.asarray(
        selected["right_transport_hold_model_order_7d_rad"], dtype=np.float64
    )
    verification = stages == "RIGHT_THREE_DIGIT_VERIFICATION"
    r26_endpoint = np.median(q[verification][:, right_indices], axis=0)
    named_delta = {
        name: float(value)
        for name, value in zip(g1.hand_joint_names["right"], r26_endpoint - selected_model, strict=True)
        if abs(float(value)) > 1.0e-7
    }

    forces = np.column_stack(
        (event["thumb_force_n"], event["index_force_n"], event["middle_force_n"])
    )
    held = ~np.isin(event["stage"].astype(str), ("RIGHT_RELEASE", "BIN_SETTLE"))
    speed = np.linalg.norm(event["object_linear_velocity_m_s"], axis=1)
    held_indices = np.flatnonzero(held)
    max_index = int(held_indices[np.argmax(speed[held_indices])])
    runtime_gate = trial["runtime_right_three_digit_gate"]
    post_gate = event["control_frame"] >= int(runtime_gate["evaluated_before_control_frame"])
    stage_names = (
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
    )
    stage_results = {name: stage_summary(event, name) for name in stage_names}
    hard_gates = {
        "pre_release_three_digit_support": runtime_gate["status"],
        "right_only_retention_1s": "PASS"
        if stage_results["RIGHT_POST_RELEASE_RETENTION"]["all_three_fraction"] >= 0.99
        and stage_results["RIGHT_POST_RELEASE_RETENTION"]["table_free_fraction"] >= 0.99
        else "FAIL",
        "continuous_table_unsupported_after_gate": "PASS"
        if bool(np.all(event["table_contact_force_n"][post_gate] < TABLE_N))
        else "FAIL",
        "offline_collision": "PASS"
        if sum(offline["offline"]["collision_frame_counts"].values()) == 0
        else "FAIL",
        "joint_limits": "PASS"
        if offline["offline"]["joint_limit_violation_count"] == 0
        else "FAIL",
        "numerical_completion": "PASS"
        if trial["command_completed"] and np.isfinite(event["object_position_world_m"]).all()
        else "FAIL",
        "runtime_penetration": "PASS"
        if trial["artifact_checks"]["maximum_runtime_penetration_m"] <= 0.003
        else "FAIL",
        "object_angular_speed": "PASS"
        if trial["artifact_checks"]["maximum_object_angular_speed_rad_s"] <= 50.0
        else "FAIL",
        "object_com_step": "PASS"
        if trial["artifact_checks"]["maximum_object_com_step_m"] <= 0.03
        else "FAIL",
        "held_object_linear_speed": "PASS"
        if float(speed[max_index]) <= MAX_SPEED_M_S
        else "FAIL",
        "exact_frozen_r14_endpoint": "PASS" if not named_delta else "FAIL",
    }
    failed = [name for name, status in hard_gates.items() if status != "PASS"]
    key_poses: dict[str, Any] = {}
    for stage_name in KEY_POSE_STAGES:
        index = int(np.flatnonzero(stages == stage_name)[-1])
        key_poses[stage_name] = {
            "control_frame": index,
            **wrist_pose_record(g1, names, q[index]),
        }

    audit = {
        "schema_version": "r26_exact_original_audit_v1",
        "candidate": "R26_HANDOFF_R14_INDEX0_P010",
        "command": str(COMMAND),
        "command_sha256": EXPECTED[COMMAND],
        "event_log": str(EVENT),
        "event_log_sha256": EXPECTED[EVENT],
        "frames": int(len(q)),
        "duration_s": float(len(q) / fps),
        "stage_schedule": [
            {
                "stage": name,
                "control_frame_start": int(np.flatnonzero(stages == name)[0]),
                "control_frame_end": int(np.flatnonzero(stages == name)[-1]),
                "frames": int(np.count_nonzero(stages == name)),
            }
            for name in dict.fromkeys(stages.tolist())
        ],
        "key_wrist_and_hand_poses": key_poses,
        "selected_frozen_r14_model_order_7d_rad": selected_model.tolist(),
        "actual_r26_transport_endpoint_model_order_7d_rad": r26_endpoint.tolist(),
        "actual_named_delta_from_frozen_r14_rad": named_delta,
        "historical_metadata_claimed_joint": str(
            np.asarray(command["right_transport_hand_joint_refinement"]).item()
        ),
        "historical_metadata_claimed_delta_rad": float(
            np.asarray(command["right_transport_hand_delta_rad"]).item()
        ),
        "joint_indexing_finding": (
            "Historical builder labeled index0 but model-order slot 5 is "
            "right_hand_middle_0_joint; the exact artifact therefore holds "
            "middle_0 +0.10 rad through transport."
        ),
        "hard_gate_results": hard_gates,
        "exact_failed_hard_gates": failed,
        "maximum_held_speed_event": {
            "physics_step_index": max_index,
            "control_frame": int(event["control_frame"][max_index]),
            "stage": str(event["stage"][max_index]),
            "timestamp_s": float(event["timestamp_s"][max_index]),
            "speed_m_s": float(speed[max_index]),
            "object_position_world_m": event["object_position_world_m"][max_index].tolist(),
            "forces_n_thumb_index_middle": forces[max_index].tolist(),
            "table_force_n": float(event["table_contact_force_n"][max_index]),
        },
        "artifact_checks": trial["artifact_checks"],
        "stage_contact_and_motion_results": stage_results,
        "left_release_observability": (
            "LEFT hand was commanded fully open and RIGHT-only support persisted; "
            "the historical event log instrumented RIGHT contacts only, so zero "
            "residual LEFT-object contact was not directly sensor-verified."
        ),
        "generic_single_hand_trial_flags_not_used": {
            "three_meaningful_digit_contacts": trial["three_meaningful_digit_contacts"],
            "retention_status": trial["retention"]["status"],
            "lift_status": trial["lift"]["status"],
            "reason": (
                "These generic end-of-run single-hand summaries do not evaluate "
                "the scripted handoff stages; stage-resolved gates above are authoritative."
            ),
        },
        "doll_or_material_changed": False,
        "selected_r14_artifact_changed": False,
        "state_restoration_used": False,
        "prohibited_mechanism_used": False,
        "policy_used": False,
        "real_robot_used": False,
    }
    original_dir = output / "00_original_audit"
    atomic_json(original_dir / "R26_EXACT_AUDIT.json", audit)
    speed_trace = {
        "timestamp_s": event["timestamp_s"],
        "control_frame": event["control_frame"],
        "stage": event["stage"],
        "object_speed_m_s": speed,
        "object_position_world_m": event["object_position_world_m"],
        "thumb_force_n": event["thumb_force_n"],
        "index_force_n": event["index_force_n"],
        "middle_force_n": event["middle_force_n"],
        "table_contact_force_n": event["table_contact_force_n"],
    }
    speed_path = original_dir / "R26_SPEED_TRACE.npz"
    save_npz(speed_path, speed_trace)
    md = f"""# Exact R26 prerequisite audit

R26 is **not** a speed-only near-pass under the current frozen-R14 contract.

- Original command SHA256: `{EXPECTED[COMMAND]}`
- Original event SHA256: `{EXPECTED[EVENT]}`
- Maximum held-object speed: `{float(speed[max_index]):.9f} m/s` (gate `{MAX_SPEED_M_S:.1f} m/s`)
- Pre-release three-digit gate: `{runtime_gate['status']}`
- RIGHT-only post-release retention: `{hard_gates['right_only_retention_1s']}`
- Collision gate: `{hard_gates['offline_collision']}`
- Joint-limit gate: `{hard_gates['joint_limits']}`
- Exact frozen-R14 endpoint: `{hard_gates['exact_frozen_r14_endpoint']}`
- Actual named endpoint delta: `{named_delta}`

The historical builder labeled its refinement `index0 +0.10 rad`, but model
slot 5 is `right_hand_middle_0_joint`.  The exact R26 artifact therefore held
`middle_0 +0.10 rad` during retention, transport, and descent.  The frozen R14
file itself remains unchanged.

The bounded retiming variants preserve this exact historical geometry for
diagnosis.  They cannot be accepted as a frozen-R14 handoff unless a subsequent
exact-R14 endpoint construction passes every gate.
"""
    atomic_text(original_dir / "R26_EXACT_AUDIT.md", md)

    variant_reports = [build_variant(scale, output, command, g1) for scale in TIMING_SCALES]
    summary = {
        "schema_version": "r26_bounded_retiming_family_v1",
        "source_audit": str(original_dir / "R26_EXACT_AUDIT.json"),
        "source_audit_sha256": sha256_file(original_dir / "R26_EXACT_AUDIT.json"),
        "speed_trace": str(speed_path),
        "speed_trace_sha256": sha256_file(speed_path),
        "timing_scales": list(TIMING_SCALES),
        "variants": variant_reports,
        "bounded": True,
        "doll_or_material_changed": False,
        "selected_r14_artifact_changed": False,
    }
    atomic_json(output / "R26_RETIMING_FAMILY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(row["status"] == "OFFLINE_PASS" for row in variant_reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
