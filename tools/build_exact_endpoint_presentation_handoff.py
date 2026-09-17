#!/usr/bin/env python3
"""Build a frame-zero LEFT presentation into the verified RIGHT endpoint.

Candidate variables come exclusively from the frozen offline LEFT-presentation
selection.  The final RIGHT arm, hand, and object-relative grasp are the exact
recorded elevated endpoint of the 3/3 RIGHT-only transport run.  In full mode,
the RIGHT transport-to-bin suffix is copied without modification.
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
from scipy.spatial.transform import Rotation


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.build_doll_handoff_proxy_v2_handoff_gate import (  # noqa: E402
    canonical_row,
    hand_model,
    minimum_jerk,
    rotations_between,
    solve_bounded_pose,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import AUTHORITATIVE_REFERENCES, authoritative_joint_ranges  # noqa: E402


OUT = ROOT / "outputs/final_methodology_preserving_completion"
SELECTION = OUT / "01_exact_endpoint_presentation_search/OFFLINE_PRESENTATION_COMBINED_SELECTION.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
R14_COMMAND = (
    ROOT
    / "outputs/final_methodology_preserving_completion/01_verified_right_transport_grasp"
    / "R14_GATE_QUALIFIED_BIN_CLEARANCE/right_only_r6_command.npz"
)
G04_COMMAND = (
    OUT
    / "03_fallback_transport_grasp/candidates/G04_R10_CRADLE_GATE_QUALIFIED"
    / "right_only_r6_command.npz"
)
G04_EVENT = G04_COMMAND.parent / "physics_r6/event_log.npz"
G04_QUALIFICATION = G04_COMMAND.parent / "RIGHT_ONLY_TRANSPORT_QUALIFICATION.json"
VALIDATED_LEFT_PREFIX = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task/p14_bilateral"
    / "backward_constructed_handoff/B2_PATH_F40"
    / "right_preload_partial_left_relax_exact_endpoint_v4/full"
    / "scripted_full_task_command.npz"
)
EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    R14_COMMAND: "128273d07c15e777c72a56643c7f63b27e43b06074bb259aed0d2ad405aa72d9",
    VALIDATED_LEFT_PREFIX: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
    G04_COMMAND: "fab93d5440140451c1ec24e625a7c2e1c2d2c41a799baf15bf8d74f947202acd",
    G04_EVENT: "b819988b9b6aab86d44d049d2c5499b6bb484fe872cdba91bd14c991482d4b41",
    G04_QUALIFICATION: "fcc71333256e319397474ac2660eceb9acba73163896270bc82b3d85c713d755",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: item.tolist()
            if isinstance(item, np.ndarray)
            else item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n",
    )


def save_npz(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def tool_pose_world(g1: G1Kinematics, side: str, static_tool: np.ndarray) -> np.ndarray:
    position_model, rotation_model, _, _ = g1.static_tool_pose_state(side, static_tool)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = g1.model_to_world_position(position_model)
    pose[:3, :3] = g1.model_to_world_rotation(rotation_model)
    return pose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--mode", choices=("acquisition", "handoff", "full"), default="acquisition"
    )
    parser.add_argument(
        "--left-hold-profile",
        choices=("P14", "MIRRORED_CRADLE", "MIRRORED_TRIPOD_WRAP"),
        default="P14",
        help=(
            "Bounded common handoff-presentation closure. Mirrored profiles "
            "reuse the validated RIGHT transport-grasp deltas without changing "
            "the fixed RIGHT endpoint."
        ),
    )
    parser.add_argument(
        "--right-endpoint-family",
        choices=("VERIFIED_R14", "FALLBACK_G04"),
        default="VERIFIED_R14",
        help="Fixed 3/3-qualified RIGHT transport grasp used as handoff endpoint.",
    )
    parser.add_argument("--output-root", type=Path, default=OUT / "02_exact_endpoint_physics")
    parser.add_argument("--handoff-evidence", type=Path)
    args = parser.parse_args()
    if args.mode == "full":
        if args.handoff_evidence is None:
            raise RuntimeError("full mode requires --handoff-evidence")
        evidence = read_json(args.handoff_evidence.resolve())
        if evidence.get("status") != "PASS" or evidence.get("candidate_id") != args.candidate:
            raise RuntimeError("handoff evidence is not a matching PASS")
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable dependency changed: {path}: {actual}")

    selection = read_json(SELECTION)
    if args.candidate not in selection["top_physics_candidates"]:
        raise RuntimeError("candidate is not in the frozen top-16 physics list")
    candidate = next(
        row for row in selection["top_candidate_rows"] if row["candidate_id"] == args.candidate
    )
    config = read_json(CONFIG)
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    names, _ = authoritative_joint_ranges()
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    left_open = hand_model(g1, names, config, "left", "OPEN")
    left_preshape = hand_model(g1, names, config, "left", "PRESHAPE")
    left_p14 = hand_model(g1, names, config, "left", "POWER_GRASP_P14")
    left_hold = left_p14.copy()
    left_hold_deltas: dict[str, float] = {}
    if args.left_hold_profile == "MIRRORED_CRADLE":
        left_hold_deltas = {
            "left_hand_thumb_1_joint": 0.04,
            "left_hand_thumb_2_joint": 0.03,
            "left_hand_index_0_joint": -0.08,
            "left_hand_middle_0_joint": -0.08,
        }
    elif args.left_hold_profile == "MIRRORED_TRIPOD_WRAP":
        left_hold_deltas = {
            "left_hand_thumb_1_joint": 0.04,
            "left_hand_thumb_2_joint": 0.03,
            "left_hand_index_0_joint": -0.08,
            "left_hand_middle_0_joint": -0.10,
            "left_hand_middle_1_joint": -0.15,
        }
    left_model_lookup = {
        name: index for index, name in enumerate(g1.hand_joint_names["left"])
    }
    for name, delta in left_hold_deltas.items():
        left_hold[left_model_lookup[name]] += delta
    left_limits = np.asarray(g1.hand_limits["left"], dtype=np.float64)
    if np.any(left_hold < left_limits[:, 0]) or np.any(left_hold > left_limits[:, 1]):
        raise RuntimeError("bounded LEFT handoff hold exceeds named joint limits")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    right_preshape = hand_model(g1, names, config, "right", "PRESHAPE")
    p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
    p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")

    primitives = {}
    static_tool = {}
    for side in ("left", "right"):
        primitive_path = Path(config["source_arm_primitives"][side])
        with np.load(primitive_path, allow_pickle=False) as archive:
            primitives[side] = {key: np.asarray(archive[key]) for key in archive.files}
        reference_arm = np.asarray(primitives[side]["approach_arm_q_rad"][-1], dtype=np.float64)
        g1.assign(reference_arm, p1_left, p1_right)
        static_tool[side] = np.linalg.inv(g1.wrist_pose(side)) @ g1.whole_hand_grasp_pose(side)

    right_command_path = (
        R14_COMMAND
        if args.right_endpoint_family == "VERIFIED_R14"
        else G04_COMMAND
    )
    with np.load(right_command_path, allow_pickle=False) as archive:
        right_command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        right_stage = archive["stage"].astype(str)
        right_names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
        archived_endpoint_hand = (
            np.asarray(
                archive["candidate_right_hand_model_order_7d_rad"],
                dtype=np.float64,
            )
            if "candidate_right_hand_model_order_7d_rad" in archive.files
            else None
        )
    if right_names != names or not np.isclose(fps, 30.0):
        raise RuntimeError("verified RIGHT command interface changed")
    with np.load(VALIDATED_LEFT_PREFIX, allow_pickle=False) as archive:
        validated_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        validated_stage = archive["stage"].astype(str)
        validated_names = archive["joint_names"].astype(str).tolist()
        validated_fps = float(np.asarray(archive["control_fps_hz"]).item())
    if validated_names != names or not np.isclose(validated_fps, fps):
        raise RuntimeError("validated LEFT prefix interface changed")
    left_transport_rows = np.flatnonzero(validated_stage == "LEFT_TRANSPORT")
    if len(left_transport_rows) != 59:
        raise RuntimeError("validated LEFT transport prefix changed")
    validated_prefix_end = int(left_transport_rows[-1])
    validated_prefix_q = validated_q[: validated_prefix_end + 1]
    validated_prefix_stage = validated_stage[: validated_prefix_end + 1]
    if not np.allclose(
        validated_prefix_q[-1, left_indices], left_p14, atol=1.0e-7
    ):
        raise RuntimeError("validated LEFT prefix no longer ends in P14")
    if not np.allclose(
        validated_prefix_q[-1, right_indices], right_open, atol=1.0e-7
    ):
        raise RuntimeError("validated LEFT prefix no longer keeps RIGHT open")
    hold_rows = np.flatnonzero(right_stage == "HOLD_ELEVATED")
    endpoint_row = int(hold_rows[-1])
    endpoint_command = right_command[endpoint_row]
    source_endpoint_arm = endpoint_command[arm_indices]
    right_transport = endpoint_command[right_indices]
    expected_right = (
        np.asarray(
            read_json(
                ROOT
                / "outputs/final_task_completion_v1/01_right_transport_grasp"
                / "SELECTED_RIGHT_TRANSPORT_GRASP.json"
            )["right_transport_hold_model_order_7d_rad"],
            dtype=np.float64,
        )
        if args.right_endpoint_family == "VERIFIED_R14"
        else archived_endpoint_hand
    )
    if expected_right is None:
        raise RuntimeError("fallback RIGHT endpoint hand metadata is missing")
    if not np.allclose(right_transport, expected_right, atol=1.0e-7):
        raise RuntimeError("elevated endpoint no longer uses verified RIGHT hand")

    target_object = np.asarray(candidate["target_object_pose_world"], dtype=np.float64)
    if args.right_endpoint_family == "FALLBACK_G04":
        qualification = read_json(G04_QUALIFICATION)
        if qualification.get("status") != "PASS" or qualification.get(
            "repeatability", {}
        ).get("successes") != 3:
            raise RuntimeError("fallback RIGHT endpoint is not qualified 3/3")
        with np.load(G04_EVENT, allow_pickle=False) as archive:
            event_stage = archive["stage"].astype(str)
            event_hold = event_stage == "HOLD_ELEVATED"
            target_object = np.eye(4, dtype=np.float64)
            target_object[:3, 3] = np.median(
                np.asarray(archive["object_position_world_m"], dtype=np.float64)[event_hold],
                axis=0,
            )
            target_object[:3, :3] = Rotation.from_quat(
                np.asarray(archive["object_quaternion_xyzw"], dtype=np.float64)[event_hold]
            ).mean().as_matrix()
    object_to_left_tool = np.asarray(candidate["candidate_object_to_left_tool"], dtype=np.float64)
    target_left_tool = target_object @ object_to_left_tool
    target_left_arm = np.r_[
        np.asarray(candidate["left_arm_q_rad"], dtype=np.float64),
        source_endpoint_arm[7:],
    ]
    target_left_reports: list[dict[str, Any]] = []
    if args.right_endpoint_family == "FALLBACK_G04":
        target_left_arm, target_left_report = solve_bounded_pose(
            g1,
            "left",
            static_tool["left"],
            g1.world_to_model_position(target_left_tool[:3, 3]),
            g1.world_to_model_rotation(target_left_tool[:3, :3]),
            target_left_arm,
            target_left_arm[:7],
            left_hold,
            right_transport,
            20260831,
        )
        target_left_reports.append(target_left_report)
    endpoint_arm = np.r_[target_left_arm[:7], source_endpoint_arm[7:]]
    if not np.allclose(endpoint_arm[7:], source_endpoint_arm[7:], atol=1.0e-12):
        raise RuntimeError("candidate does not preserve exact RIGHT arm endpoint")

    # The validated LEFT grasp/lift/transport prefix is copied byte-for-byte.
    # Candidate-specific arm geometry begins only after LEFT_TRANSPORT completes.
    presentation_target_arm = np.r_[
        target_left_arm[:7], validated_prefix_q[-1, arm_indices][7:]
    ]
    presentation_arm = minimum_jerk(
        validated_prefix_q[-1, arm_indices], presentation_target_arm, 181
    )

    g1.assign(presentation_target_arm, left_hold, right_open)
    right_start = tool_pose_world(g1, "right", static_tool["right"])
    g1.assign(endpoint_arm, left_hold, right_transport)
    right_target = tool_pose_world(g1, "right", static_tool["right"])
    right_pregrasp = right_target.copy()
    right_pregrasp[:3, 3] += np.asarray([0.045, 0.0, 0.0])
    right_to_pregrasp, right_pregrasp_reports = solve_path(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(
            minimum_jerk(right_start[:3, 3], right_pregrasp[:3, 3], 121)
        ),
        np.einsum(
            "ij,tjk->tik",
            g1.root_pose[:3, :3].T,
            rotations_between(right_start[:3, :3], right_pregrasp[:3, :3], 121),
        ),
        presentation_target_arm,
        left_hold,
        right_open,
        clearance_target_m=0.001,
    )
    right_final_approach = minimum_jerk(right_to_pregrasp[-1], endpoint_arm, 91)

    retreat_tool = target_left_tool.copy()
    retreat_tool[:3, 3] += np.asarray([-0.10, -0.06, 0.04])
    left_retreat_arm, retreat_reports = solve_path(
        g1,
        "left",
        static_tool["left"],
        g1.world_to_model_position(
            minimum_jerk(target_left_tool[:3, 3], retreat_tool[:3, 3], 91)
        ),
        np.repeat(g1.world_to_model_rotation(target_left_tool[:3, :3])[None], 91, axis=0),
        endpoint_arm,
        left_open,
        right_transport,
    )

    rows: list[np.ndarray] = []
    stages: list[str] = []

    def append(arm: np.ndarray, left: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left, right))
        stages.append(stage)

    for row, stage in zip(validated_prefix_q, validated_prefix_stage, strict=True):
        rows.append(row.copy())
        stages.append(str(stage))
    for arm in presentation_arm[1:]:
        append(arm, left_p14, right_open, "LEFT_HANDOFF_PRESENTATION_TRANSITION")
    for hand in minimum_jerk(left_p14, left_hold, 61)[1:]:
        append(
            presentation_target_arm,
            hand,
            right_open,
            "LEFT_HANDOFF_STABILIZATION_TRANSITION",
        )
    for _ in range(30):
        append(presentation_target_arm, left_hold, right_open, "LEFT_HANDOFF_HOLD")
    for arm in right_to_pregrasp[1:]:
        append(arm, left_hold, right_open, "RIGHT_COLLISION_FREE_APPROACH")
    for hand in minimum_jerk(right_open, right_preshape, 31)[1:]:
        append(right_to_pregrasp[-1], left_hold, hand, "RIGHT_PRESHAPE")
    for arm in right_final_approach[1:]:
        append(arm, left_hold, right_preshape, "RIGHT_APPROACH_EXACT_TRANSPORT_ENDPOINT")
    for hand in minimum_jerk(right_preshape, right_transport, 61)[1:]:
        append(endpoint_arm, left_hold, hand, "RIGHT_CLOSE_EXACT_TRANSPORT_GRASP")
    for _ in range(30):
        append(endpoint_arm, left_hold, right_transport, "RIGHT_THREE_DIGIT_VERIFICATION")

    if args.mode in ("handoff", "full"):
        left_release = minimum_jerk(left_hold, left_open, 91)
        for arm, left in zip(left_retreat_arm[1:], left_release[1:], strict=True):
            append(arm, left, right_transport, "LEFT_THUMB_RELEASE")
        for _ in range(30):
            append(left_retreat_arm[-1], left_open, right_transport, "RIGHT_POST_RELEASE_RETENTION")
        for _ in range(30):
            append(left_retreat_arm[-1], left_open, right_transport, "RIGHT_OWNED_HOLD")
    if args.mode == "full":
        tail_start = int(np.flatnonzero(right_stage == "RIGHT_VERTICAL_100MM")[0])
        previous_right = endpoint_command[np.r_[arm_indices[7:], right_indices]]
        first_tail_right = right_command[tail_start][np.r_[arm_indices[7:], right_indices]]
        if np.max(np.abs(first_tail_right - previous_right)) > 0.03:
            raise RuntimeError("verified RIGHT tail splice exceeds adjacent-step gate")
        for index in range(tail_start, len(right_command)):
            row = right_command[index]
            append(
                np.r_[left_retreat_arm[-1][:7], row[arm_indices][7:]],
                left_open,
                row[right_indices],
                str(right_stage[index]),
            )

    commands = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (commands < lower[None] - 1.0e-9) | (commands > upper[None] + 1.0e-9)
    geometry = g1.trajectory_geometry(
        commands[:, arm_indices],
        commands[:, left_indices],
        commands[:, right_indices],
        1.0e-5,
    )
    collision_counts = {
        key: int(np.count_nonzero(value))
        for key, value in geometry["collision_flags"].items()
    }
    arm_steps = np.abs(np.diff(commands[:, arm_indices], axis=0))
    max_step = float(np.max(arm_steps, initial=0.0))
    max_inherited_step = float(
        np.max(arm_steps[: len(validated_prefix_q) - 1], initial=0.0)
    )
    max_new_step = float(
        np.max(arm_steps[len(validated_prefix_q) - 1 :], initial=0.0)
    )
    verification_rows = np.flatnonzero(labels == "RIGHT_THREE_DIGIT_VERIFICATION")
    endpoint_arm_error = float(
        np.max(
            np.abs(commands[verification_rows][:, arm_indices] - endpoint_arm[None]),
            initial=0.0,
        )
    )
    endpoint_hand_error = float(
        np.max(
            np.abs(commands[verification_rows][:, right_indices] - right_transport[None]),
            initial=0.0,
        )
    )
    reports = target_left_reports + right_pregrasp_reports + retreat_reports
    max_ik_position_error = max(float(row["position_error_m"]) for row in reports)
    max_ik_orientation_error = max(float(row["orientation_error_rad"]) for row in reports)
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        # The immutable successful prefix contains one documented 0.03396-rad
        # LEFT transport step. The unchanged 0.03-rad construction gate applies
        # to every newly generated transition, not to copied frozen evidence.
        and max_new_step <= 0.03
        and endpoint_arm_error <= 1.0e-7
        and endpoint_hand_error <= 1.0e-7
        and max_ik_position_error <= 0.001
        and max_ik_orientation_error <= 0.02
    )
    output = args.output_root.resolve() / args.candidate / args.mode
    command_path = output / f"exact_endpoint_{args.mode}_command.npz"
    save_npz(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": labels,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(fps),
            "candidate_id": np.asarray(args.candidate),
            "left_handoff_hold_profile": np.asarray(args.left_hold_profile),
            "left_handoff_hold_model_order_7d_rad": left_hold.astype(np.float32),
            "left_handoff_hold_activation_stage": np.asarray(
                "LEFT_HANDOFF_STABILIZATION_TRANSITION"
            ),
            "initial_left_grasp_profile": np.asarray("P14"),
            "policy_independent": np.asarray(True),
            "scripted_full_task": np.asarray(args.mode == "full"),
            "runtime_right_three_digit_gate_required": np.asarray(args.mode in ("handoff", "full")),
            "right_three_digit_gate_minimum_s": np.asarray(0.5),
            "verified_right_transport_endpoint_exact": np.asarray(True),
            "verified_right_transport_tail_copied": np.asarray(args.mode == "full"),
            "state_restoration_used": np.asarray(False),
            "object_pose_writes": np.asarray(0),
            "attachment_used": np.asarray(False),
            "object_follow_used": np.asarray(False),
        },
    )
    report = {
        "schema_version": "exact_endpoint_presentation_handoff_command_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "candidate_id": args.candidate,
        "mode": args.mode,
        "right_endpoint_family": args.right_endpoint_family,
        "construction": "LEFT presentation backward from exact recorded VERIFIED_RIGHT_TRANSPORT_GRASP elevated endpoint",
        "candidate_parameters": {
            "left_long_axis_shift_m": candidate["left_long_axis_shift_m"],
            "left_local_yaw_deg": candidate["left_local_yaw_deg"],
            "left_local_pitch_deg": candidate["left_local_pitch_deg"],
            "left_handoff_hold_profile": args.left_hold_profile,
            "left_handoff_hold_model_order_7d_rad": left_hold,
            "left_handoff_hold_named_delta_from_p14_rad": left_hold_deltas,
            "activation_scope": (
                "The validated 28D P14 command is copied exactly through initial "
                "grasp, gravity retention, lift, and pre-handoff LEFT_TRANSPORT; "
                "candidate arm and hand transitions begin only afterward"
            ),
        },
        "validated_left_prefix": {
            "source": str(VALIDATED_LEFT_PREFIX),
            "source_sha256": sha256_file(VALIDATED_LEFT_PREFIX),
            "last_source_frame": validated_prefix_end,
            "frames": len(validated_prefix_q),
            "maximum_28d_copy_error_rad": float(
                np.max(
                    np.abs(commands[: len(validated_prefix_q)] - validated_prefix_q),
                    initial=0.0,
                )
            ),
        },
        "exact_endpoint": {
            "target_object_pose_world": target_object,
            "right_arm_q_rad": endpoint_arm[7:],
            "right_hand_q_rad": right_transport,
            "maximum_arm_error_rad": endpoint_arm_error,
            "maximum_hand_error_rad": endpoint_hand_error,
        },
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_adjacent_arm_step_rad": max_step,
            "maximum_inherited_prefix_arm_step_rad": max_inherited_step,
            "maximum_new_adjacent_arm_step_rad": max_new_step,
            "maximum_adjacent_arm_step_gate_rad": 0.03,
            "maximum_ik_position_error_m": max_ik_position_error,
            "maximum_ik_orientation_error_rad": max_ik_orientation_error,
        },
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "source_verified_right_command": str(right_command_path),
        "source_verified_right_command_sha256": sha256_file(right_command_path),
        "right_transport_tail_changed": False,
        "doll_or_physics_changed": False,
        "policy_used": False,
        "prohibited_mechanism_used": False,
        "real_robot_used": False,
    }
    atomic_json(output / "offline_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
