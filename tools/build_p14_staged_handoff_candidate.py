#!/usr/bin/env python3
"""Build a bounded staged P14 Doll-Handoff trajectory.

The physics, controller, and bilateral P14 endpoints are hash-frozen.  The
only search variable is a short object-relative right-wrist acquisition path.
Left index and middle relax during that path while the frozen left P14 thumb
continues supporting the doll.  Gate-mode commands stop before left-thumb
release.  Full-mode commands contain an explicit runtime three-digit gate.
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
    preliminary_candidate_audit,
    rotations_between,
    solve_bounded_pose,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import (  # noqa: E402
    AUTHORITATIVE_REFERENCES,
    authoritative_joint_ranges,
)


CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
FREEZE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "frozen_p14_bilateral/FREEZE_MANIFEST.json"
)
BASE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral"
)
DEFAULT_OUTPUT = BASE / "staged_handoff_search"

# Endpoints are expressed relative to the independently validated right-hand
# P14 load frame at the handoff object.  Entry is dx=15 mm, dy=0, dz=5 mm,
# yaw=35 deg.  This is one three-member path family, not a static-pose grid.
PATHS: dict[str, dict[str, float]] = {
    "S1_DX10_DY3_DZ5_YAW45": {
        "dx_m": 0.010,
        "dy_m": 0.003,
        "dz_m": 0.005,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 45.0,
    },
    "S2_DX8_DY3_DZ5_YAW50": {
        "dx_m": 0.008,
        "dy_m": 0.003,
        "dz_m": 0.005,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 50.0,
    },
    "S3_DX5_DY5_DZ5_YAW55": {
        "dx_m": 0.005,
        "dy_m": 0.005,
        "dz_m": 0.005,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 55.0,
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def verify_freeze(freeze: dict[str, Any], config: dict[str, Any]) -> None:
    if freeze.get("status") != "FROZEN_BILATERAL_FULL_GRASP_PASS":
        raise RuntimeError("P14 bilateral freeze is not active")
    for row in freeze["hashes"].values():
        path = (FREEZE.parent / row["path"]).resolve()
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen dependency changed: {path}")
    for side in ("left", "right"):
        if config["hand_states"][side]["POWER_GRASP_P14"] != freeze["p14"][f"{side}_7d_rad"]:
            raise RuntimeError(f"frozen {side} P14 endpoint changed")


def save_command(
    path: Path,
    commands: np.ndarray,
    stages: np.ndarray,
    names: list[str],
    fps: float,
    candidate: str,
    handoff_object: np.ndarray,
    bin_object: np.ndarray,
    full: bool,
) -> None:
    temporary = path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=commands.astype(np.float32),
            stage=stages,
            joint_names=np.asarray(names),
            control_fps_hz=np.asarray(fps),
            candidate_id=np.asarray(candidate),
            handoff_object_center_world_m=handoff_object.astype(np.float32),
            bin_object_target_world_m=bin_object.astype(np.float32),
            policy_independent=np.asarray(True),
            scripted_full_task=np.asarray(full),
            staged_sequential_handoff=np.asarray(True),
            runtime_right_three_digit_gate_required=np.asarray(full),
            right_three_digit_gate_minimum_s=np.asarray(0.5),
            left_release_permitted=np.asarray(full),
        )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(PATHS), required=True)
    parser.add_argument("--mode", choices=("gate", "full"), default="gate")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    freeze = read_json(FREEZE)
    config = read_json(CONFIG)
    verify_freeze(freeze, config)
    full = args.mode == "full"
    candidate_dir = args.output_root.resolve() / args.candidate / args.mode
    candidate_dir.mkdir(parents=True, exist_ok=True)

    with np.load(BASE / "scripted_full_task_command.npz", allow_pickle=False) as archive:
        base_commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        base_stages = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
        handoff_object = np.asarray(archive["handoff_object_center_world_m"], dtype=np.float64)
        bin_object = np.asarray(archive["bin_object_target_world_m"], dtype=np.float64)
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or base_commands.shape[1] != 28 or not np.isclose(fps, 30.0):
        raise RuntimeError("frozen base command interface changed")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    left_open = hand_model(g1, names, config, "left", "OPEN")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    left_power = hand_model(g1, names, config, "left", "POWER_GRASP_P14")
    right_power = hand_model(g1, names, config, "right", "POWER_GRASP_P14")
    p1 = {side: hand_model(g1, names, config, side, "POWER_GRASP_P1") for side in ("left", "right")}
    left_thumb_support = left_power.copy()
    left_thumb_support[3:] = left_open[3:]

    primitives: dict[str, dict[str, np.ndarray]] = {}
    static_tool: dict[str, np.ndarray] = {}
    reference_rotation_world: dict[str, np.ndarray] = {}
    reference_arm: dict[str, np.ndarray] = {}
    origin = g1.model_to_world_position(np.zeros(3))
    world_from_model = np.column_stack(
        [g1.model_to_world_position(np.eye(3)[axis]) - origin for axis in range(3)]
    )
    for side in ("left", "right"):
        path = Path(config["source_arm_primitives"][side])
        with np.load(path, allow_pickle=False) as archive:
            primitives[side] = {key: np.asarray(archive[key]) for key in archive.files}
        reference_arm[side] = np.asarray(primitives[side]["lift_arm_q_rad"][-1], dtype=np.float64)
        g1.assign(reference_arm[side], p1["left"], p1["right"])
        static_tool[side] = np.linalg.inv(g1.wrist_pose(side)) @ g1.whole_hand_grasp_pose(side)
        g1.assign(reference_arm[side], left_power, right_power)
        _, rotation_model, _, _ = g1.static_tool_pose_state(side, static_tool[side])
        reference_rotation_world[side] = world_from_model @ rotation_model

    tool_offsets: dict[str, np.ndarray] = {}
    trial_root = (
        ROOT
        / "outputs/dex3_simple_graspable_doll_proxy_v1"
        / "hand_calibration_v2/p14_three_digit_preload/trials"
    )
    for side in ("left", "right"):
        result = read_json(trial_root / side / "trial_result.json")
        if result["status"] != "PASS":
            raise RuntimeError(f"frozen {side} P14 result is not PASS")
        with np.load(result["event_log"], allow_pickle=False) as archive:
            mask = archive["stage"].astype(str) == "HOLD_ELEVATED"
            object_center = np.median(archive["object_position_world_m"][mask][-240:], axis=0)
        tool_position = np.asarray(
            primitives[side]["target_whole_hand_position_world_m"][-1], dtype=np.float64
        )
        tool_offsets[side] = tool_position - object_center

    acquire_rows = np.flatnonzero(base_stages == "RIGHT_ACQUIRE")
    dual_rows = np.flatnonzero(base_stages == "DUAL_CONTACT_HOLD")
    if not len(acquire_rows) or not len(dual_rows):
        raise RuntimeError("frozen base acquisition stages are missing")
    prefix_end = int(acquire_rows[-1] + 1)
    base_arm = np.asarray(base_commands[dual_rows[-1], arm_indices], dtype=np.float64)
    g1.assign(base_arm, left_power, right_power)
    entry_position_model, entry_rotation_model, _, _ = g1.static_tool_pose_state(
        "right", static_tool["right"]
    )
    entry_position_world = g1.model_to_world_position(entry_position_model)
    entry_rotation_world = world_from_model @ entry_rotation_model

    spec = PATHS[args.candidate]
    validated_position_world = handoff_object + tool_offsets["right"]
    target_position_world = validated_position_world + np.asarray(
        [spec["dx_m"], spec["dy_m"], spec["dz_m"]]
    )
    target_rotation_world = (
        Rotation.from_euler(
            "xyz",
            [spec["roll_deg"], spec["pitch_deg"], spec["yaw_deg"]],
            degrees=True,
        ).as_matrix()
        @ reference_rotation_world["right"]
    )
    path_count = 30
    positions_world = minimum_jerk(entry_position_world, target_position_world, path_count)
    rotations_world = rotations_between(entry_rotation_world, target_rotation_world, path_count)
    staged_arm, staged_reports = solve_path(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(positions_world),
        np.einsum("ij,tjk->tik", world_from_model.T, rotations_world),
        base_arm,
        left_thumb_support,
        right_power,
        clearance_target_m=0.001,
    )

    rows: list[np.ndarray] = [row.copy() for row in base_commands[:prefix_end]]
    stages: list[str] = base_stages[:prefix_end].tolist()

    def append(arm: np.ndarray, left: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left, right))
        stages.append(stage)

    for _ in range(8):
        append(base_arm, left_power, right_power, "RIGHT_INDEX_MIDDLE_ACQUISITION_HOLD")
    left_relax_path = minimum_jerk(left_power, left_thumb_support, path_count)
    for arm, left in zip(staged_arm[1:], left_relax_path[1:], strict=True):
        append(arm, left, right_power, "STAGED_LEFT_IM_RELAX_RIGHT_WRIST_PATH")
    verification_frames = 18 if full else 30
    for _ in range(verification_frames):
        append(staged_arm[-1], left_thumb_support, right_power, "RIGHT_THREE_DIGIT_VERIFICATION")

    tail_reports: list[dict[str, Any]] = []
    right_hold_report: dict[str, Any] | None = None
    if full:
        # The runtime runner checks the preceding 0.5 s simultaneous right
        # contact gate before it applies the first command in this stage.
        for left in minimum_jerk(left_thumb_support, left_open, 14)[1:]:
            append(staged_arm[-1], left, right_power, "LEFT_THUMB_RELEASE")
        for _ in range(30):
            append(staged_arm[-1], left_open, right_power, "RIGHT_RETENTION_GATE")

        left_handoff_tool = handoff_object + tool_offsets["left"]
        left_retreat_world = left_handoff_tool + np.asarray([-0.10, -0.06, 0.04])
        left_rotation_model = world_from_model.T @ reference_rotation_world["left"]
        left_retreat, left_reports = solve_path(
            g1,
            "left",
            static_tool["left"],
            g1.world_to_model_position(
                minimum_jerk(left_handoff_tool, left_retreat_world, round(1.5 * fps))
            ),
            np.repeat(left_rotation_model[None], round(1.5 * fps), axis=0),
            staged_arm[-1],
            left_open,
            right_power,
        )
        left_parked = left_retreat[-1]
        right_hold_world = handoff_object + tool_offsets["right"]
        right_rotation_model = world_from_model.T @ reference_rotation_world["right"]
        right_hold_arm, right_hold_report = solve_bounded_pose(
            g1,
            "right",
            static_tool["right"],
            g1.world_to_model_position(right_hold_world),
            right_rotation_model,
            left_parked,
            staged_arm[-1][7:14],
            left_open,
            right_power,
            2026083101,
        )
        bin_tool = bin_object + tool_offsets["right"]
        right_transport, right_reports = solve_path(
            g1,
            "right",
            static_tool["right"],
            g1.world_to_model_position(
                minimum_jerk(right_hold_world, bin_tool, round(2.5 * fps))
            ),
            np.repeat(right_rotation_model[None], round(2.5 * fps), axis=0),
            right_hold_arm,
            left_open,
            right_power,
        )
        tail_reports = left_reports + right_reports
        for arm in left_retreat[1:]:
            append(arm, left_open, right_power, "LEFT_RETREAT")
        for arm in minimum_jerk(left_parked, right_hold_arm, round(1.0 * fps))[1:]:
            append(arm, left_open, right_power, "RIGHT_SETTLE_TO_POWER_GRASP")
        for _ in range(round(1.0 * fps)):
            append(right_hold_arm, left_open, right_power, "RIGHT_OWNED_HOLD")
        for arm in right_transport[1:]:
            append(arm, left_open, right_power, "RIGHT_TRANSPORT_TO_BIN")
        for _ in range(round(0.5 * fps)):
            append(right_transport[-1], left_open, right_power, "RIGHT_HOLD_OVER_BIN")
        for right in minimum_jerk(right_power, right_open, round(1.0 * fps))[1:]:
            append(right_transport[-1], left_open, right, "RIGHT_RELEASE")
        for _ in range(round(2.0 * fps)):
            append(right_transport[-1], left_open, right_open, "POST_RELEASE")

    commands = np.asarray(rows, dtype=np.float64)
    labels = np.asarray(stages)
    expected_frames = 792 if full else prefix_end + 8 + (path_count - 1) + verification_frames
    if len(commands) != expected_frames:
        raise RuntimeError(f"unexpected trajectory length {len(commands)} != {expected_frames}")

    contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray([float(row["minimum"]) for row in contract["joint_specs"]])
    upper = np.asarray([float(row["maximum"]) for row in contract["joint_specs"]])
    violations = (commands < lower[None] - 1e-9) | (commands > upper[None] + 1e-9)
    arm = commands[:, arm_indices]
    left = commands[:, left_indices]
    right = commands[:, right_indices]
    geometry = g1.trajectory_geometry(arm, left, right, 1e-5)
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()
    }
    endpoint_audit = preliminary_candidate_audit(
        g1,
        staged_arm[-1],
        left_thumb_support,
        right_power,
        handoff_object,
        np.asarray(config["geometry_candidates"][0]["dimensions_m"], dtype=np.float64),
    )
    offline_pass = bool(
        not np.any(violations)
        and not sum(collision_counts.values())
        and endpoint_audit["no_left_right_hand_overlap"]
        and endpoint_audit["right_hand_doll_contact"]
        and max(row["position_error_m"] for row in staged_reports + tail_reports)
        <= 0.001
        and max(row["orientation_error_rad"] for row in staged_reports + tail_reports)
        <= 0.02
    )

    command_path = candidate_dir / (
        "scripted_full_task_command.npz" if full else "staged_acquisition_gate_command.npz"
    )
    save_command(
        command_path,
        commands,
        labels,
        names,
        fps,
        args.candidate,
        handoff_object,
        bin_object,
        full,
    )
    report = {
        "schema_version": "p14_staged_sequential_handoff_v1",
        "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
        "mode": args.mode,
        "candidate_id": args.candidate,
        "declared_path_count": len(PATHS),
        "declared_object_relative_paths": PATHS,
        "entry_object_relative": {
            "dx_m": 0.015,
            "dy_m": 0.0,
            "dz_m": 0.005,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 35.0,
        },
        "endpoint_object_relative": spec,
        "varied": "right acquisition wrist path only",
        "left_p14_endpoint_unchanged": True,
        "right_p14_endpoint_unchanged": True,
        "left_index_middle_relaxation": "P14 to OPEN while P14 thumb remains commanded",
        "left_thumb_release_in_command": full,
        "runtime_three_digit_gate_required": full,
        "runtime_three_digit_gate_minimum_s": 0.5,
        "doll_changed": False,
        "controller_changed": False,
        "endpoint_static_audit": endpoint_audit,
        "robot_collision_frame_counts": collision_counts,
        "robot_collision_pairs": geometry["collision_pairs"],
        "joint_limit_violations": int(np.count_nonzero(violations)),
        "ik_max_position_error_m": max(
            row["position_error_m"] for row in staged_reports + tail_reports
        ),
        "ik_max_orientation_error_rad": max(
            row["orientation_error_rad"] for row in staged_reports + tail_reports
        ),
        "right_hold_pose_report": right_hold_report,
        "frames": len(commands),
        "duration_s": len(commands) / fps,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "freeze_manifest": str(FREEZE),
        "freeze_manifest_sha256": sha256_file(FREEZE),
        "config_sha256": sha256_file(CONFIG),
        "policy_used": False,
        "real_robot": False,
    }
    atomic_json(candidate_dir / "offline_report.json", report)
    print((candidate_dir / "offline_report.json").read_text(encoding="utf-8"), end="")
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
