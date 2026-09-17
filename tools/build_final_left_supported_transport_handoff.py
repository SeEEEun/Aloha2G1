#!/usr/bin/env python3
"""Connect the verified handoff to R14 while LEFT still supports the doll.

This builder is deliberately a bounded sequencing experiment.  It reuses the
exact frame-zero physical handoff prefix, the collision-free H10 wrist path,
the immutable selected R14 transport grasp, and the verified LEFT retreat.
Unlike H10--H12, LEFT does not release before the R14 conversion.  No object
state is restored, written, followed, or used at runtime.
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

from tools.build_doll_handoff_proxy_v2_handoff_gate import (  # noqa: E402
    hand_model,
    minimum_jerk,
    solve_path,
)
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
H10 = (
    ROOT
    / "outputs/final_task_completion_v1/02_handoff_to_transport_grasp"
    / "H10_OBJECT_RELATIVE_R14_REGULARIZED"
    / "integrated_handoff_transport_gate_command.npz"
)
SELECTED = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp"
    / "SELECTED_RIGHT_TRANSPORT_GRASP.json"
)
G04_COMMAND = (
    ROOT
    / "outputs/final_methodology_preserving_completion/03_fallback_transport_grasp"
    / "candidates/G04_R10_CRADLE_GATE_QUALIFIED/right_only_r6_command.npz"
)
G04_QUALIFICATION = G04_COMMAND.parent / "RIGHT_ONLY_TRANSPORT_QUALIFICATION.json"


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


def stage_rows(commands: np.ndarray, stages: np.ndarray, label: str) -> np.ndarray:
    rows = commands[stages == label]
    if not len(rows):
        raise RuntimeError(f"required verified stage missing: {label}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversion-order",
        choices=("simultaneous", "wrist_then_fingers", "fingers_then_wrist"),
        default="simultaneous",
    )
    parser.add_argument(
        "--right-endpoint-family",
        choices=("VERIFIED_R14", "FALLBACK_G04"),
        default="VERIFIED_R14",
        help=(
            "Qualified RIGHT transport hand target. FALLBACK_G04 preserves the "
            "same +10-degree wrist family and uses the independently qualified "
            "3/3 CRADLE hand vector."
        ),
    )
    parser.add_argument(
        "--left-preclear-index",
        type=int,
        default=-1,
        help=(
            "Last zero-based sample of the already verified 44-frame LEFT radial "
            "clearance path to execute before R14 convergence; -1 disables it."
        ),
    )
    parser.add_argument(
        "--left-cartesian-post-r14-retreat-x-m",
        type=float,
        default=-0.060,
        help=(
            "Additional world-X LEFT tool retreat after R14 support verification "
            "and before opening; used only with Cartesian preclear."
        ),
    )
    parser.add_argument(
        "--left-cartesian-preclear-x-m",
        type=float,
        default=0.0,
        help=(
            "Bounded world-X LEFT tool offset executed with the existing support "
            "command.  This is mutually exclusive with --left-preclear-index."
        ),
    )
    parser.add_argument(
        "--release-clearance-final-index",
        type=int,
        default=15,
        help=(
            "Last zero-based sample of LEFT_FINAL_SAFE_RETREAT executed with the "
            "existing LEFT support command after R14 verification and before "
            "the finger-release transition."
        ),
    )
    parser.add_argument(
        "--release-before-left-retreat",
        action="store_true",
        help=(
            "After verified R14 support, open LEFT at its fixed collision-free "
            "handoff arm pose, then execute the already validated source LEFT "
            "retreat with the hand open. This avoids moving a closed LEFT thumb "
            "through the RIGHT wrist."
        ),
    )
    parser.add_argument(
        "--coordinated-preclear-end-frame",
        type=int,
        default=-1,
        help=(
            "Zero-based RIGHT bridge frame by which the requested inherited "
            "LEFT preclear endpoint is reached.  This coordinates support "
            "withdrawal with RIGHT approach instead of preclearing first."
        ),
    )
    parser.add_argument(
        "--conversion-duration-scale",
        type=float,
        choices=(1.0, 2.0, 3.0, 4.0),
        default=1.0,
        help=(
            "Bounded time scaling of only the fixed spatial R14 wrist/finger "
            "conversion. Endpoints and the recorded collision-pruned path are unchanged."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.left_cartesian_preclear_x_m and args.left_preclear_index != -1:
        raise ValueError("Cartesian and inherited radial preclear are mutually exclusive")
    if args.coordinated_preclear_end_frame >= 0 and args.left_preclear_index < 0:
        raise ValueError("coordinated preclear requires --left-preclear-index")
    if args.coordinated_preclear_end_frame >= 0 and args.left_cartesian_preclear_x_m:
        raise ValueError("coordinated and Cartesian preclear are mutually exclusive")

    expected = {
        CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
        SOURCE: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
        H10: "6198c993e8ebbfbea645d4a48a09b8ddd9b64ba654600aee7218dd816aa17ee0",
        SELECTED: "10d406038795f2f4dfc38ecdfd408de176d63297f05dd4814fc1a4b29392b8c9",
    }
    if args.right_endpoint_family == "FALLBACK_G04":
        expected.update(
            {
                G04_COMMAND: "fab93d5440140451c1ec24e625a7c2e1c2d2c41a799baf15bf8d74f947202acd",
                G04_QUALIFICATION: "fcc71333256e319397474ac2660eceb9acba73163896270bc82b3d85c713d755",
            }
        )
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"verified dependency changed: {path}: {actual}")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with np.load(SOURCE, allow_pickle=False) as archive:
        source_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        source_stage = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
    with np.load(H10, allow_pickle=False) as archive:
        h10_q = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        h10_stage = archive["stage"].astype(str)
        if archive["joint_names"].astype(str).tolist() != names:
            raise RuntimeError("H10 named joint order changed")
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or not np.isclose(fps, 30.0):
        raise RuntimeError("authoritative 28D interface changed")

    lookup = {name: index for index, name in enumerate(names)}
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    left_arm = [lookup[name] for name in g1.arm_joint_names[:7]]
    right_arm = [lookup[name] for name in g1.arm_joint_names[7:]]
    left_hand = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_hand = [lookup[name] for name in g1.hand_joint_names["right"]]
    arm_indices = left_arm + right_arm

    # Exact continuous prefix through the verified T4 three-digit stage, but
    # before the first LEFT-thumb-release command.
    rows = [row.copy() for row in source_q[:481]]
    labels = source_stage[:481].tolist()
    labels = [
        "RIGHT_T4_PRETRANSFER_VERIFICATION"
        if label == "RIGHT_THREE_DIGIT_VERIFICATION"
        else label
        for label in labels
    ]
    left_support = rows[-1][left_hand].copy()
    right_t4 = rows[-1][right_hand].copy()
    left_open = stage_rows(h10_q, h10_stage, "LEFT_FINAL_SAFE_RETREAT")[-1][left_hand]
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    if args.right_endpoint_family == "VERIFIED_R14":
        right_r14 = np.asarray(
            selected["right_transport_hold_model_order_7d_rad"], dtype=np.float64
        )
    else:
        qualification = json.loads(G04_QUALIFICATION.read_text(encoding="utf-8"))
        if qualification.get("status") != "PASS" or qualification.get(
            "repeatability", {}
        ).get("successes") != 3:
            raise RuntimeError("G04 endpoint is not independently qualified 3/3")
        with np.load(G04_COMMAND, allow_pickle=False) as archive:
            right_r14 = np.asarray(
                archive["candidate_right_hand_model_order_7d_rad"], dtype=np.float64
            )

    def append_from(
        guide: np.ndarray,
        stage: str,
        *,
        left_q: np.ndarray | None = None,
        right_q: np.ndarray | None = None,
        left_arm_q: np.ndarray | None = None,
        right_arm_q: np.ndarray | None = None,
    ) -> None:
        row = rows[-1].copy()
        # LEFT stays at its currently supported pose unless a later, explicit
        # retreat sample is supplied.  The H10 guide carries a fully-retreated
        # LEFT arm and must never leak into the supported conversion.
        if left_arm_q is not None:
            row[left_arm] = left_arm_q
        row[right_arm] = guide[right_arm] if right_arm_q is None else right_arm_q
        row[left_hand] = left_support if left_q is None else left_q
        row[right_hand] = right_t4 if right_q is None else right_q
        rows.append(row)
        labels.append(stage)

    radial_clearance = stage_rows(h10_q, h10_stage, "LEFT_OBJECT_RADIAL_CLEARANCE")
    if not -1 <= args.left_preclear_index < len(radial_clearance):
        raise ValueError("left-preclear-index is outside the verified radial path")
    if args.coordinated_preclear_end_frame < 0:
        for guide in radial_clearance[: args.left_preclear_index + 1]:
            append_from(
                rows[-1],
                "LEFT_SUPPORTED_PRECLEAR",
                left_arm_q=guide[left_arm],
                right_arm_q=rows[-1][right_arm],
            )
    if args.left_cartesian_preclear_x_m:
        primitive_path = Path(config["source_arm_primitives"]["left"])
        with np.load(primitive_path, allow_pickle=False) as archive:
            reference_arm = np.asarray(
                archive["approach_arm_q_rad"][-1], dtype=np.float64
            )
        p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
        p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")
        g1.assign(reference_arm, p1_left, p1_right)
        static_tool_left = (
            np.linalg.inv(g1.wrist_pose("left"))
            @ g1.whole_hand_grasp_pose("left")
        )
        g1.assign(rows[-1][arm_indices], left_support, right_t4)
        start_model, start_rotation, _, _ = g1.static_tool_pose_state(
            "left", static_tool_left
        )
        start_world = g1.model_to_world_position(start_model)
        target_world = start_world + np.asarray(
            [args.left_cartesian_preclear_x_m, 0.0, 0.0]
        )
        cartesian_preclear, _ = solve_path(
            g1,
            "left",
            static_tool_left,
            g1.world_to_model_position(minimum_jerk(start_world, target_world, 61)),
            np.repeat(start_rotation[None], 61, axis=0),
            rows[-1][arm_indices],
            left_support,
            right_t4,
        )
        for arm in cartesian_preclear[1:]:
            append_from(
                rows[-1],
                "LEFT_OBJECT_TANGENTIAL_PRECLEAR",
                left_arm_q=arm[:7],
                right_arm_q=rows[-1][right_arm],
            )

    # Move only RIGHT along the already collision-pruned bridge path while the
    # verified partial LEFT enclosure remains loaded.
    bridge = stage_rows(h10_q, h10_stage, "RIGHT_POSTHANDOFF_APPROACH_P14")
    coordinated_left_arm = None
    if args.coordinated_preclear_end_frame >= 0:
        if args.coordinated_preclear_end_frame >= len(bridge):
            raise ValueError("coordinated preclear endpoint exceeds RIGHT bridge")
        coordinated_left_arm = minimum_jerk(
            rows[-1][left_arm],
            radial_clearance[args.left_preclear_index][left_arm],
            args.coordinated_preclear_end_frame + 2,
        )[1:]
    for bridge_index, guide in enumerate(bridge):
        left_arm_q = None
        stage_name = "RIGHT_SUPPORTED_APPROACH_TO_R14_BRIDGE"
        if coordinated_left_arm is not None:
            left_arm_q = coordinated_left_arm[
                min(bridge_index, len(coordinated_left_arm) - 1)
            ]
            stage_name = "LEFT_RIGHT_COORDINATED_CLEARANCE"
        append_from(
            guide,
            stage_name,
            left_arm_q=left_arm_q,
        )
    for _ in range(30):
        append_from(bridge[-1], "RIGHT_SUPPORTED_BRIDGE_STABILIZATION")

    source_wrist_path = stage_rows(
        h10_q, h10_stage, "RIGHT_CONVERGE_TO_TRANSPORT_GRASP"
    )
    conversion_frames = int(
        round(len(source_wrist_path) * args.conversion_duration_scale)
    )
    # Interpolate along the already collision-pruned recorded curve.  This
    # changes only time sampling: the start, endpoint, and spatial polyline are
    # unchanged.  Including the current boundary row gives a zero-jump join.
    source_with_boundary = np.vstack((rows[-1], source_wrist_path))
    source_axis = np.arange(len(source_with_boundary), dtype=np.float64)
    retimed_axis = np.linspace(
        0.0, float(len(source_with_boundary) - 1), conversion_frames + 1
    )[1:]
    wrist_path = np.column_stack(
        [
            np.interp(retimed_axis, source_axis, source_with_boundary[:, joint])
            for joint in range(source_with_boundary.shape[1])
        ]
    )
    wrist_path[-1] = source_wrist_path[-1]
    finger_path = minimum_jerk(
        right_t4,
        right_r14,
        int(round(90 * args.conversion_duration_scale)) + 1,
    )[1:]
    if args.conversion_order == "simultaneous":
        # H10 contains the exact regularized wrist path and its simultaneous
        # hand interpolation; LEFT support is the only sequencing difference.
        for guide in wrist_path:
            append_from(
                guide,
                "RIGHT_SUPPORTED_CONVERGE_TO_R14",
                right_q=guide[right_hand],
            )
    elif args.conversion_order == "wrist_then_fingers":
        for guide in wrist_path:
            append_from(
                guide, "RIGHT_SUPPORTED_R14_WRIST_ALIGNMENT", right_q=right_t4
            )
        for _ in range(30):
            append_from(
                wrist_path[-1],
                "RIGHT_SUPPORTED_R14_WRIST_STABILIZATION",
                right_q=right_t4,
            )
        for hand in finger_path:
            append_from(
                wrist_path[-1], "RIGHT_SUPPORTED_R14_FINGER_ENCLOSURE", right_q=hand
            )
    else:
        for hand in finger_path:
            append_from(
                bridge[-1], "RIGHT_SUPPORTED_R14_FINGER_ENCLOSURE", right_q=hand
            )
        for _ in range(30):
            append_from(
                bridge[-1],
                "RIGHT_SUPPORTED_R14_FINGER_STABILIZATION",
                right_q=right_r14,
            )
        for guide in wrist_path:
            append_from(
                guide, "RIGHT_SUPPORTED_R14_WRIST_ALIGNMENT", right_q=right_r14
            )

    for _ in range(30):
        append_from(
            wrist_path[-1], "RIGHT_THREE_DIGIT_VERIFICATION", right_q=right_r14
        )
    final_retreat = stage_rows(h10_q, h10_stage, "LEFT_FINAL_SAFE_RETREAT")
    if not -1 <= args.release_clearance_final_index < len(final_retreat):
        raise ValueError("release-clearance-final-index is outside the verified retreat")
    # Once R14 is verified, finish the radial exit and the minimum additional
    # LEFT clearance while its existing partial support command is unchanged.
    # This avoids opening a digit chain into the RIGHT wrist.  It is still a
    # physical load transfer: RIGHT supports the doll while LEFT withdraws.
    if args.release_before_left_retreat:
        # The frame-level H13 audit showed that the supported RIGHT conversion
        # is collision-free.  Every collision occurred afterward while the
        # still-closed LEFT thumb was translated through the RIGHT wrist.
        # Release at the fixed handoff pose first; retreat only after the LEFT
        # colliders have opened and RIGHT support has passed the runtime gate.
        pass
    elif args.left_cartesian_preclear_x_m:
        g1.assign(rows[-1][arm_indices], left_support, right_r14)
        retreat_start_model, retreat_rotation, _, _ = g1.static_tool_pose_state(
            "left", static_tool_left
        )
        retreat_start_world = g1.model_to_world_position(retreat_start_model)
        retreat_target_world = retreat_start_world + np.asarray(
            [args.left_cartesian_post_r14_retreat_x_m, 0.0, 0.0]
        )
        cartesian_post_retreat, _ = solve_path(
            g1,
            "left",
            static_tool_left,
            g1.world_to_model_position(
                minimum_jerk(retreat_start_world, retreat_target_world, 91)
            ),
            np.repeat(retreat_rotation[None], 91, axis=0),
            rows[-1][arm_indices],
            left_support,
            right_r14,
        )
        for arm in cartesian_post_retreat[1:]:
            append_from(
                wrist_path[-1],
                "LEFT_POST_R14_TANGENTIAL_RETREAT",
                right_q=right_r14,
                left_arm_q=arm[:7],
                right_arm_q=wrist_path[-1][right_arm],
            )
    else:
        for guide in radial_clearance[args.left_preclear_index + 1 :]:
            append_from(
                wrist_path[-1],
                "LEFT_POST_R14_RADIAL_RELEASE_CLEARANCE",
                right_q=right_r14,
                left_arm_q=guide[left_arm],
                right_arm_q=wrist_path[-1][right_arm],
            )
        for guide in final_retreat[: args.release_clearance_final_index + 1]:
            append_from(
                wrist_path[-1],
                "LEFT_POST_R14_FINAL_RELEASE_CLEARANCE",
                right_q=right_r14,
                left_arm_q=guide[left_arm],
                right_arm_q=wrist_path[-1][right_arm],
            )
    for hand in minimum_jerk(left_support, left_open, 31)[1:]:
        append_from(
            wrist_path[-1], "LEFT_THUMB_RELEASE", left_q=hand, right_q=right_r14
        )
    for _ in range(30):
        append_from(
            wrist_path[-1],
            "RIGHT_POST_RELEASE_RETENTION",
            left_q=left_open,
            right_q=right_r14,
        )

    if args.release_before_left_retreat:
        # SOURCE LEFT_RETREAT is the exact normal frame-zero retreat that
        # follows its physical release.  Only its named LEFT arm samples are
        # used; RIGHT remains fixed at the selected transport endpoint.
        source_left_retreat = stage_rows(source_q, source_stage, "LEFT_RETREAT")
        for guide in source_left_retreat:
            append_from(
                wrist_path[-1],
                "LEFT_OPEN_SAFE_RETREAT",
                left_q=left_open,
                right_q=right_r14,
                left_arm_q=guide[left_arm],
                right_arm_q=wrist_path[-1][right_arm],
            )
    else:
        # Reuse the verified LEFT-clearance geometry, but keep RIGHT fixed at
        # the selected R14 endpoint rather than T4.
        retreat_parts = (
            ()
            if args.left_cartesian_preclear_x_m
            else (
                (
                    "LEFT_FINAL_SAFE_RETREAT",
                    final_retreat[args.release_clearance_final_index + 1 :],
                ),
            )
        )
        for source_label, retreat_rows in retreat_parts:
            for guide in retreat_rows:
                append_from(
                    wrist_path[-1],
                    source_label,
                    left_q=left_open,
                    right_q=right_r14,
                    left_arm_q=guide[left_arm],
                    right_arm_q=wrist_path[-1][right_arm],
                )
    for _ in range(30):
        append_from(
            rows[-1],
            "RIGHT_POST_RETREAT_RETENTION",
            left_q=left_open,
            right_q=right_r14,
        )

    # The H10 vertical gate is already the selected R14 object-relative lift.
    vertical = stage_rows(h10_q, h10_stage, "RIGHT_VERTICAL_CLEARANCE")
    for guide in vertical:
        append_from(
            guide,
            "RIGHT_VERTICAL_CLEARANCE",
            left_q=left_open,
            right_q=right_r14,
            left_arm_q=rows[-1][left_arm],
        )
    for _ in range(30):
        append_from(
            vertical[-1],
            "RIGHT_HIGH_STABILIZATION",
            left_q=left_open,
            right_q=right_r14,
            left_arm_q=rows[-1][left_arm],
        )

    commands = np.asarray(rows, dtype=np.float64)
    stages = np.asarray(labels)
    contract = json.loads(
        AUTHORITATIVE_REFERENCES["joint_ranges"].read_text(encoding="utf-8")
    )
    lower = np.asarray(
        [float(row["minimum"]) for row in contract["joint_specs"]], dtype=np.float64
    )
    upper = np.asarray(
        [float(row["maximum"]) for row in contract["joint_specs"]], dtype=np.float64
    )
    violations = (commands < lower[None] - 1.0e-9) | (commands > upper[None] + 1.0e-9)
    geometry = g1.trajectory_geometry(
        commands[:, arm_indices], commands[:, left_hand], commands[:, right_hand], 1.0e-5
    )
    collision_counts = {
        key: int(np.count_nonzero(value))
        for key, value in geometry["collision_flags"].items()
    }
    new_start = 481
    steps = np.abs(np.diff(commands[:, arm_indices], axis=0))
    maximum_new_step = float(np.max(steps[new_start - 1 :], initial=0.0))
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and maximum_new_step <= 0.03
    )

    command_path = output / "left_supported_handoff_transport_gate_command.npz"
    save_npz(
        command_path,
        {
            "commanded_q_rad": commands.astype(np.float32),
            "stage": stages,
            "joint_names": np.asarray(names),
            "control_fps_hz": np.asarray(fps),
            "policy_independent": np.asarray(True),
            "scripted_full_task": np.asarray(False),
            "runtime_right_three_digit_gate_required": np.asarray(True),
            "right_three_digit_gate_minimum_s": np.asarray(0.5),
            "transport_grasp_conversion_order": np.asarray(args.conversion_order),
            "attachment_used": np.asarray(False),
            "object_follow_used": np.asarray(False),
            "object_pose_writes": np.asarray(0),
            "state_restoration_used": np.asarray(False),
            "selected_right_transport_hand_model_order_7d_rad": right_r14.astype(np.float32),
            "right_endpoint_family": np.asarray(args.right_endpoint_family),
            "r14_conversion_duration_scale": np.asarray(
                args.conversion_duration_scale
            ),
        },
    )
    report = {
        "schema_version": "final_left_supported_transport_handoff_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "right_endpoint_family": args.right_endpoint_family,
        "conversion_order": args.conversion_order,
        "left_preclear_index": args.left_preclear_index,
        "left_cartesian_preclear_x_m": args.left_cartesian_preclear_x_m,
        "left_cartesian_post_r14_retreat_x_m": args.left_cartesian_post_r14_retreat_x_m,
        "release_clearance_final_index": args.release_clearance_final_index,
        "release_before_left_retreat": args.release_before_left_retreat,
        "coordinated_preclear_end_frame": args.coordinated_preclear_end_frame,
        "conversion_duration_scale": args.conversion_duration_scale,
        "conversion_original_frames": len(source_wrist_path),
        "conversion_retimed_frames": len(wrist_path),
        "construction": (
            "exact frame-zero prefix; LEFT support retained through selected R14 "
            "wrist/finger conversion; gated LEFT release; verified retreat and vertical lift"
        ),
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "dependencies": {str(path): digest for path, digest in expected.items()},
        "offline": {
            "joint_limit_violation_count": int(np.count_nonzero(violations)),
            "collision_frame_counts": collision_counts,
            "collision_pairs": geometry["collision_pairs"],
            "maximum_new_adjacent_arm_step_rad": maximum_new_step,
            "maximum_new_adjacent_arm_step_gate_rad": 0.03,
        },
        "doll_or_material_changed": False,
        "controller_gains_changed": False,
        "state_restoration_used": False,
        "runtime_object_feedback_used": False,
        "prohibited_mechanisms_used": False,
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(output / "offline_gate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
