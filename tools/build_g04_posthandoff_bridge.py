#!/usr/bin/env python3
"""Replace only H10's failed R14 finger conversion with qualified G04.

The complete frame-zero handoff, RIGHT-only retention, LEFT clearance, and
collision-pruned RIGHT wrist bridge are copied from H10.  Only the final 7D
RIGHT hand interpolation is changed to the independently 3/3-qualified G04
CRADLE vector.  Optional bounded time scaling resamples the same arm curve.
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
from tools.evaluation.contracts import AUTHORITATIVE_REFERENCES, authoritative_joint_ranges  # noqa: E402


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
H10 = (
    ROOT
    / "outputs/final_task_completion_v1/02_handoff_to_transport_grasp"
    / "H10_OBJECT_RELATIVE_R14_REGULARIZED/integrated_handoff_transport_gate_command.npz"
)
G04 = (
    ROOT
    / "outputs/final_methodology_preserving_completion/03_fallback_transport_grasp"
    / "candidates/G04_R10_CRADLE_GATE_QUALIFIED/right_only_r6_command.npz"
)
G04_QUALIFICATION = G04.parent / "RIGHT_ONLY_TRANSPORT_QUALIFICATION.json"
EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    H10: "6198c993e8ebbfbea645d4a48a09b8ddd9b64ba654600aee7218dd816aa17ee0",
    G04: "fab93d5440140451c1ec24e625a7c2e1c2d2c41a799baf15bf8d74f947202acd",
    G04_QUALIFICATION: "fcc71333256e319397474ac2660eceb9acba73163896270bc82b3d85c713d755",
}


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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
    parser.add_argument("--duration-scale", type=float, choices=(1.0, 2.0, 4.0), default=1.0)
    parser.add_argument(
        "--conversion-order",
        choices=(
            "simultaneous",
            "wrist_then_fingers",
            "fingers_then_wrist",
            "late_overlap_25",
            "late_overlap_50",
        ),
        default="simultaneous",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable dependency changed: {path}: {actual}")
    qualification = json.loads(G04_QUALIFICATION.read_text(encoding="utf-8"))
    if qualification.get("status") != "PASS" or qualification.get("repeatability", {}).get("successes") != 3:
        raise RuntimeError("G04 no longer has 3/3 qualification")

    with np.load(H10, allow_pickle=False) as archive:
        source_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        source_stage = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
    with np.load(G04, allow_pickle=False) as archive:
        g04_hand = np.asarray(
            archive["candidate_right_hand_model_order_7d_rad"], dtype=np.float64
        )
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or not np.isclose(fps, 30.0):
        raise RuntimeError("authoritative 28D interface changed")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]

    conversion = np.flatnonzero(source_stage == "RIGHT_CONVERGE_TO_TRANSPORT_GRASP")
    if len(conversion) != 120 or not np.array_equal(conversion, np.arange(conversion[0], conversion[-1] + 1)):
        raise RuntimeError("H10 conversion segment changed")
    boundary = int(conversion[0] - 1)
    end = int(conversion[-1])
    source_with_boundary = np.vstack((source_q[boundary], source_q[conversion]))
    count = int(round(len(conversion) * args.duration_scale))
    source_axis = np.arange(len(source_with_boundary), dtype=np.float64)
    target_axis = np.linspace(0.0, float(len(source_with_boundary) - 1), count + 1)[1:]
    retimed = np.column_stack(
        [
            np.interp(target_axis, source_axis, source_with_boundary[:, joint])
            for joint in range(source_q.shape[1])
        ]
    )
    retimed[-1] = source_q[end]
    start_hand = source_q[boundary, right_indices].copy()

    post = source_q[end + 1 :].copy()
    post[:, right_indices] = g04_hand[None]
    if args.conversion_order == "simultaneous":
        retimed[:, right_indices] = minimum_jerk(start_hand, g04_hand, count + 1)[1:]
        middle = retimed
        middle_labels = np.full(count, "RIGHT_CONVERGE_TO_G04_TRANSPORT_GRASP")
    elif args.conversion_order == "wrist_then_fingers":
        retimed[:, right_indices] = start_hand[None]
        dwell = np.repeat(retimed[-1][None], 30, axis=0)
        finger_count = int(round(90 * args.duration_scale))
        fingers = np.repeat(retimed[-1][None], finger_count, axis=0)
        fingers[:, right_indices] = minimum_jerk(start_hand, g04_hand, finger_count + 1)[1:]
        middle = np.vstack((retimed, dwell, fingers))
        middle_labels = np.concatenate(
            (
                np.full(count, "RIGHT_G04_WRIST_ALIGNMENT"),
                np.full(30, "RIGHT_G04_WRIST_STABILIZATION"),
                np.full(finger_count, "RIGHT_G04_FINGER_ENCLOSURE"),
            )
        )
    elif args.conversion_order == "fingers_then_wrist":
        finger_count = int(round(90 * args.duration_scale))
        fingers = np.repeat(source_q[boundary][None], finger_count, axis=0)
        fingers[:, right_indices] = minimum_jerk(start_hand, g04_hand, finger_count + 1)[1:]
        dwell = np.repeat(fingers[-1][None], 30, axis=0)
        retimed[:, right_indices] = g04_hand[None]
        middle = np.vstack((fingers, dwell, retimed))
        middle_labels = np.concatenate(
            (
                np.full(finger_count, "RIGHT_G04_FINGER_ENCLOSURE"),
                np.full(30, "RIGHT_G04_FINGER_STABILIZATION"),
                np.full(count, "RIGHT_G04_WRIST_ALIGNMENT"),
            )
        )
    else:
        overlap_fraction = 0.25 if args.conversion_order == "late_overlap_25" else 0.50
        overlap_count = max(2, int(round(count * overlap_fraction)))
        overlap_start = count - overlap_count
        retimed[:, right_indices] = start_hand[None]
        retimed[overlap_start:, right_indices] = minimum_jerk(
            start_hand, g04_hand, overlap_count + 1
        )[1:]
        middle = retimed
        middle_labels = np.full(count, "RIGHT_G04_LATE_OVERLAP_CONVERSION")
    commands = np.vstack((source_q[: boundary + 1], middle, post))
    labels = np.concatenate((source_stage[: boundary + 1], middle_labels, source_stage[end + 1 :]))
    labels[labels == "RIGHT_TRANSPORT_GRASP_VERIFICATION"] = "RIGHT_G04_TRANSPORT_GRASP_VERIFICATION"

    contract = json.loads(AUTHORITATIVE_REFERENCES["joint_ranges"].read_text(encoding="utf-8"))
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (commands < lower[None] - 1.0e-9) | (commands > upper[None] + 1.0e-9)
    geometry = g1.trajectory_geometry(
        commands[:, arm_indices], commands[:, left_indices], commands[:, right_indices], 1.0e-5
    )
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()
    }
    maximum_step = float(np.max(np.abs(np.diff(commands[:, arm_indices], axis=0)), initial=0.0))
    maximum_new_step = float(
        np.max(np.abs(np.diff(commands[boundary:], axis=0)[:, arm_indices]), initial=0.0)
    )
    prefix_error = float(np.max(np.abs(commands[: boundary + 1] - source_q[: boundary + 1]), initial=0.0))
    endpoint_error = float(np.max(np.abs(commands[-1, right_indices] - g04_hand), initial=0.0))
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and maximum_new_step <= 0.03
        and prefix_error == 0.0
        and endpoint_error <= 1.0e-7
    )

    command_path = output / "g04_posthandoff_bridge_command.npz"
    save_npz(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": labels,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(fps),
            "runtime_right_three_digit_gate_required": np.asarray(True),
            "right_three_digit_gate_minimum_s": np.asarray(0.5),
            "right_endpoint_family": np.asarray("FALLBACK_G04"),
            "transport_grasp_conversion_order": np.asarray(args.conversion_order),
            "selected_right_transport_hand_model_order_7d_rad": g04_hand.astype(np.float32),
            "policy_independent": np.asarray(True),
            "state_restoration_used": np.asarray(False),
            "object_pose_writes": np.asarray(0),
            "attachment_used": np.asarray(False),
            "object_follow_used": np.asarray(False),
        },
    )
    report = {
        "schema_version": "g04_posthandoff_bridge_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "duration_scale": args.duration_scale,
        "conversion_order": args.conversion_order,
        "construction": "H10 exact through verified handoff, RIGHT ownership, LEFT retreat, and wrist bridge; only failed R14 7D conversion replaced by 3/3-qualified G04 CRADLE",
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "exact_h10_prefix_last_frame": boundary,
        "exact_h10_prefix_maximum_error_rad": prefix_error,
        "g04_endpoint_maximum_hand_error_rad": endpoint_error,
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_adjacent_arm_step_rad": maximum_step,
            "maximum_new_adjacent_arm_step_rad": maximum_new_step,
            "maximum_new_adjacent_arm_step_gate_rad": 0.03,
        },
        "dependencies": {str(path): expected for path, expected in EXPECTED.items()},
        "doll_or_physics_changed": False,
        "controller_gains_changed": False,
        "thresholds_changed": False,
        "state_restoration_used": False,
        "prohibited_mechanisms_used": False,
        "policy_used": False,
        "real_robot_used": False,
    }
    atomic_json(output / "offline_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
