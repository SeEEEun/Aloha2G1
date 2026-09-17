#!/usr/bin/env python3
"""Bounded transport-grasp alternative: R14 hand at the valid handoff wrist.

The frame-zero handoff prefix, T5 Cartesian transport path, doll, gains, and
physics remain exact.  The only candidate variable is replacing the known
one-sided T4 digit vector with the independently validated R14 three-digit
transport vector while LEFT still supports the doll.  This is a new grasp
geometry candidate, not another speed/waypoint modification to T4.
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

from tools.build_doll_handoff_proxy_v2_handoff_gate import minimum_jerk  # noqa: E402
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import (  # noqa: E402
    AUTHORITATIVE_REFERENCES,
    authoritative_joint_ranges,
)


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
SOURCE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task"
    / "p14_bilateral/backward_constructed_handoff/B2_PATH_F40"
    / "right_preload_partial_left_relax_exact_endpoint_v4/full"
    / "scripted_full_task_command.npz"
)
T5 = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/success_first_common_execution"
    / "tail_debug/t5_t4_grip_robust_final_transport_v1/tail_command.npz"
)
SELECTED = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp"
    / "SELECTED_RIGHT_TRANSPORT_GRASP.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
    / "R18_HANDOFF_WRIST_R14_HAND_T5_FULL"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wrist-joint", choices=("none", "roll", "pitch", "yaw"), default="none"
    )
    parser.add_argument("--wrist-delta-deg", type=float, default=0.0)
    parser.add_argument(
        "--hand-joint",
        choices=("none", "index0", "index1"),
        default="none",
    )
    parser.add_argument("--hand-delta-rad", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.wrist_joint == "none" and not np.isclose(args.wrist_delta_deg, 0.0):
        raise ValueError("nonzero wrist delta requires a named wrist joint")
    if args.wrist_joint != "none" and not 0.0 < abs(args.wrist_delta_deg) <= 7.5:
        raise ValueError("bounded wrist-grasp refinement is limited to 7.5 degrees")
    if args.hand_joint == "none" and not np.isclose(args.hand_delta_rad, 0.0):
        raise ValueError("nonzero hand delta requires a named hand joint")
    if args.hand_joint != "none" and not 0.0 < args.hand_delta_rad <= 0.10:
        raise ValueError("bounded index closure is limited to +0.10 rad")
    expected = {
        CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
        SOURCE: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
        T5: "7399ed969e78c8542d0aaf12ed372f4ae47fe949f3941efcf5d1b572df4e73f5",
        SELECTED: "10d406038795f2f4dfc38ecdfd408de176d63297f05dd4814fc1a4b29392b8c9",
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"verified dependency changed: {path}")

    with np.load(SOURCE, allow_pickle=False) as archive:
        source_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        source_stage = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
    with np.load(T5, allow_pickle=False) as archive:
        tail_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        tail_stage = archive["stage"].astype(str)
        if archive["joint_names"].astype(str).tolist() != names:
            raise RuntimeError("T5 named joint order changed")
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or not np.isclose(fps, 30.0):
        raise RuntimeError("authoritative 28D interface changed")
    if not np.array_equal(source_q[539], tail_q[0]):
        raise RuntimeError("T5 frame-zero keyframe command is no longer exact")

    common = load_common_config()
    g1 = G1Kinematics(common, load_scene(common))
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_hand = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_hand = [lookup[name] for name in g1.hand_joint_names["right"]]
    wrist_index = None
    wrist_delta = 0.0
    if args.wrist_joint != "none":
        wrist_index = lookup[f"right_wrist_{args.wrist_joint}_joint"]
        wrist_delta = float(np.deg2rad(args.wrist_delta_deg))
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    r14 = np.asarray(
        selected["right_transport_hold_model_order_7d_rad"], dtype=np.float64
    )
    if args.hand_joint != "none":
        r14[{"index0": 5, "index1": 6}[args.hand_joint]] += args.hand_delta_rad
    t4 = source_q[480, right_hand].copy()
    left_support = source_q[480, left_hand].copy()
    left_open = source_q[539, left_hand].copy()
    right_open = tail_q[-1, right_hand].copy()

    rows = [row.copy() for row in source_q[:481]]
    labels = [
        "RIGHT_T4_PRETRANSFER_VERIFICATION"
        if label == "RIGHT_THREE_DIGIT_VERIFICATION"
        else label
        for label in source_stage[:481]
    ]
    base = rows[-1].copy()
    for hand in minimum_jerk(t4, r14, 91)[1:]:
        row = base.copy()
        row[right_hand] = hand
        rows.append(row)
        labels.append("RIGHT_SUPPORTED_R14_HAND_ENCLOSURE")
    base = rows[-1].copy()
    if wrist_index is not None:
        for value in minimum_jerk(
            np.asarray([base[wrist_index]]),
            np.asarray([base[wrist_index] + wrist_delta]),
            61,
        )[1:, 0]:
            row = base.copy()
            row[wrist_index] = value
            rows.append(row)
            labels.append("RIGHT_SUPPORTED_TRANSPORT_WRIST_REFINEMENT")
        base = rows[-1].copy()
    for _ in range(30):
        rows.append(base.copy())
        labels.append("RIGHT_THREE_DIGIT_VERIFICATION")
    for hand in minimum_jerk(left_support, left_open, 31)[1:]:
        row = base.copy()
        row[left_hand] = hand
        rows.append(row)
        labels.append("LEFT_THUMB_RELEASE")
    base = rows[-1].copy()
    for _ in range(30):
        rows.append(base.copy())
        labels.append("RIGHT_POST_RELEASE_RETENTION")

    release_rows = np.flatnonzero(tail_stage == "RIGHT_RELEASE")
    release_values = minimum_jerk(r14, right_open, len(release_rows) + 1)[1:]
    release_lookup = {int(index): release_values[i] for i, index in enumerate(release_rows)}
    for index in range(1, len(tail_q)):
        row = tail_q[index].copy()
        if wrist_index is not None:
            row[wrist_index] += wrist_delta
        if index < int(release_rows[0]):
            row[right_hand] = r14
        elif index in release_lookup:
            row[right_hand] = release_lookup[index]
        else:
            row[right_hand] = right_open
        rows.append(row)
        labels.append(str(tail_stage[index]))

    commands = np.asarray(rows, dtype=np.float64)
    stages = np.asarray(labels)
    contract = json.loads(
        AUTHORITATIVE_REFERENCES["joint_ranges"].read_text(encoding="utf-8")
    )
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (commands < lower[None] - 1.0e-9) | (commands > upper[None] + 1.0e-9)
    geometry = g1.trajectory_geometry(
        commands[:, arm_indices], commands[:, left_hand], commands[:, right_hand], 1.0e-5
    )
    collision_counts = {
        key: int(np.count_nonzero(value))
        for key, value in geometry["collision_flags"].items()
    }
    steps = np.abs(np.diff(commands[:, arm_indices], axis=0))
    maximum_step = float(np.max(steps, initial=0.0))
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and maximum_step <= 0.04
    )

    output.mkdir(parents=True, exist_ok=True)
    command_path = output / "continuous_r14_hand_t5_full_command.npz"
    save_npz(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": stages,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(fps),
            "policy_independent": np.asarray(True),
            "scripted_full_task": np.asarray(True),
            "runtime_right_three_digit_gate_required": np.asarray(True),
            "right_three_digit_gate_minimum_s": np.asarray(0.5),
            "right_transport_candidate": np.asarray("HANDOFF_WRIST_R14_HAND"),
            "right_transport_wrist_joint": np.asarray(args.wrist_joint),
            "right_transport_wrist_delta_deg": np.asarray(args.wrist_delta_deg),
            "selected_r14_hand_model_order_7d_rad": r14.astype(np.float32),
            "right_transport_hand_joint_refinement": np.asarray(args.hand_joint),
            "right_transport_hand_delta_rad": np.asarray(args.hand_delta_rad),
            "state_restoration_used": np.asarray(False),
            "object_pose_writes": np.asarray(0),
            "attachment_used": np.asarray(False),
            "object_follow_used": np.asarray(False),
        },
    )
    report = {
        "schema_version": "final_handoff_pose_r14_hand_transport_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "candidate": "HANDOFF_WRIST_R14_HAND",
        "wrist_joint": args.wrist_joint,
        "wrist_delta_deg": args.wrist_delta_deg,
        "hand_joint": args.hand_joint,
        "hand_delta_rad": args.hand_delta_rad,
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_adjacent_arm_step_rad": maximum_step,
        },
        "dependencies": {str(path): digest for path, digest in expected.items()},
        "doll_or_material_changed": False,
        "transport_path_changed": False,
        "r14_hand_vector_changed": args.hand_joint != "none",
        "state_restoration_used": False,
        "runtime_object_feedback_used": False,
        "prohibited_mechanisms_used": False,
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(output / "offline_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
