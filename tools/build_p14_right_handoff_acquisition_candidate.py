#!/usr/bin/env python3
"""Build one member of the frozen P14 right-acquisition pose search.

Only the right wrist/palm acquisition transform varies.  Doll physics, both
P14 7D full-close vectors, left trajectory, timing, and controller are loaded
from hash-frozen artifacts.
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
    hand_model,
    minimum_jerk,
    preliminary_candidate_audit,
    solve_path,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import authoritative_joint_ranges  # noqa: E402


FREEZE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "frozen_p14_bilateral/FREEZE_MANIFEST.json"
)
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
BASE = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1"
    / "scripted_full_task/p14_bilateral"
)

# The first six candidates test inward X and yaw.  Their measured PhysX result
# was a zero-load thumb touch with excessive index/middle preload.  A final,
# bounded three-point opposition subgrid therefore translates the entire palm
# +Y by 3/5/7 mm; no further search dimensions or candidates are permitted.
# All offsets are relative to the existing H1 acquisition frame:
# dx=15 mm, yaw=35 deg, dy=0, dz=5 mm.
CANDIDATES = {
    "A1_DX10_YAW35": {"dx_m": 0.010, "dy_m": 0.0, "yaw_deg": 35.0},
    "A2_DX05_YAW35": {"dx_m": 0.005, "dy_m": 0.0, "yaw_deg": 35.0},
    "A3_DX10_YAW30": {"dx_m": 0.010, "dy_m": 0.0, "yaw_deg": 30.0},
    "A4_DX10_YAW40": {"dx_m": 0.010, "dy_m": 0.0, "yaw_deg": 40.0},
    "A5_DX15_YAW30": {"dx_m": 0.015, "dy_m": 0.0, "yaw_deg": 30.0},
    "A6_DX15_YAW40": {"dx_m": 0.015, "dy_m": 0.0, "yaw_deg": 40.0},
    "A7_DX10_DYP03_YAW35": {"dx_m": 0.010, "dy_m": 0.003, "yaw_deg": 35.0},
    "A8_DX10_DYP05_YAW35": {"dx_m": 0.010, "dy_m": 0.005, "yaw_deg": 35.0},
    "A9_DX10_DYP07_YAW35": {"dx_m": 0.010, "dy_m": 0.007, "yaw_deg": 35.0},
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


def verify_freeze(freeze: dict[str, Any]) -> None:
    if freeze.get("status") != "FROZEN_BILATERAL_FULL_GRASP_PASS":
        raise RuntimeError("P14 freeze is not active")
    for row in freeze["hashes"].values():
        path = (FREEZE.parent / row["path"]).resolve()
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen dependency changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--output-root", type=Path, default=BASE / "right_acquisition_search")
    args = parser.parse_args()
    freeze = read_json(FREEZE)
    verify_freeze(freeze)
    config = read_json(CONFIG)
    if config["hand_states"]["left"]["POWER_GRASP_P14"] != freeze["p14"]["left_7d_rad"]:
        raise RuntimeError("frozen left P14 changed")
    if config["hand_states"]["right"]["POWER_GRASP_P14"] != freeze["p14"]["right_7d_rad"]:
        raise RuntimeError("frozen right P14 changed")

    with np.load(BASE / "scripted_full_task_command.npz", allow_pickle=False) as archive:
        commands = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        stages = archive["stage"].astype(str)
        names = archive["joint_names"].astype(str).tolist()
        fps = float(np.asarray(archive["control_fps_hz"]).item())
        handoff_object = np.asarray(archive["handoff_object_center_world_m"], dtype=np.float64)
    authoritative_names, _ = authoritative_joint_ranges()
    if names != authoritative_names or commands.shape[1] != 28:
        raise RuntimeError("base command interface changed")
    if not np.isclose(fps, 30.0):
        raise RuntimeError("base command is not 30 Hz")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    left_power = hand_model(g1, names, config, "left", "POWER_GRASP_P14")
    right_power = hand_model(g1, names, config, "right", "POWER_GRASP_P14")
    p1 = {side: hand_model(g1, names, config, side, "POWER_GRASP_P1") for side in ("left", "right")}
    lookup = {name: index for index, name in enumerate(names)}
    model_arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    acquire_rows = np.flatnonzero(stages == "DUAL_CONTACT_HOLD")
    approach_rows = np.flatnonzero(stages == "RIGHT_APPROACH")
    if not len(acquire_rows) or not len(approach_rows):
        raise RuntimeError("base acquisition stages missing")
    base_row = commands[acquire_rows[-1]]
    base_arm = base_row[model_arm_indices]

    # Reconstruct the frozen wrist-attached whole-hand tool definition.
    primitive_path = Path(config["source_arm_primitives"]["right"])
    with np.load(primitive_path, allow_pickle=False) as primitive:
        reference_arm = np.asarray(primitive["lift_arm_q_rad"][-1], dtype=np.float64)
    g1.assign(reference_arm, p1["left"], p1["right"])
    static_tool = np.linalg.inv(g1.wrist_pose("right")) @ g1.whole_hand_grasp_pose("right")
    g1.assign(base_arm, left_power, right_power)
    base_position_model, base_rotation_model, _, _ = g1.static_tool_pose_state("right", static_tool)
    origin = g1.model_to_world_position(np.zeros(3))
    world_from_model = np.column_stack(
        [g1.model_to_world_position(np.eye(3)[axis]) - origin for axis in range(3)]
    )
    base_position_world = g1.model_to_world_position(base_position_model)
    base_rotation_world = world_from_model @ base_rotation_model
    spec = CANDIDATES[args.candidate]
    delta_x = float(spec["dx_m"] - 0.015)
    delta_y = float(spec["dy_m"])
    delta_yaw = float(spec["yaw_deg"] - 35.0)
    target_position_world = base_position_world + np.asarray([delta_x, delta_y, 0.0])
    target_rotation_world = Rotation.from_euler("z", delta_yaw, degrees=True).as_matrix() @ base_rotation_world
    solved, solve_reports = solve_path(
        g1,
        "right",
        static_tool,
        g1.world_to_model_position(target_position_world[None]),
        (world_from_model.T @ target_rotation_world)[None],
        base_arm,
        left_power,
        right_power,
        clearance_target_m=0.001,
    )
    candidate_arm = solved[-1]
    audit = preliminary_candidate_audit(
        g1,
        candidate_arm,
        left_power,
        right_power,
        handoff_object,
        np.asarray(config["geometry_candidates"][0]["dimensions_m"], dtype=np.float64),
    )

    # Preserve the base prefix and all hand commands, replacing only the right
    # arm's approach/acquisition endpoint.  End before any left release.
    end = acquire_rows[-1] + 1
    candidate_commands = commands[:end].copy()
    candidate_stages = stages[:end].copy()
    approach = minimum_jerk(
        commands[approach_rows[0] - 1][model_arm_indices],
        candidate_arm,
        len(approach_rows) + 1,
    )[1:]
    candidate_commands[approach_rows[:, None], np.asarray(model_arm_indices)[None, :]] = approach
    held_rows = np.flatnonzero(
        np.isin(candidate_stages, ["RIGHT_PRESHAPE", "RIGHT_ACQUIRE", "DUAL_CONTACT_HOLD"])
    )
    candidate_commands[held_rows[:, None], np.asarray(model_arm_indices)[None, :]] = candidate_arm
    # Guarantee exactly 1.0 s of pre-release verification at the endpoint.
    dual = np.flatnonzero(candidate_stages == "DUAL_CONTACT_HOLD")
    if len(dual) < 30:
        add = 30 - len(dual)
        candidate_commands = np.concatenate(
            [candidate_commands, np.repeat(candidate_commands[-1][None], add, axis=0)], axis=0
        )
        candidate_stages = np.concatenate(
            [candidate_stages, np.asarray(["DUAL_CONTACT_HOLD"] * add)]
        )

    arm = candidate_commands[:, model_arm_indices]
    left = candidate_commands[:, left_indices]
    right = candidate_commands[:, right_indices]
    geometry = g1.trajectory_geometry(arm, left, right, 1e-5)
    collision_counts = {key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()}
    candidate_dir = args.output_root.resolve() / args.candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    command_path = candidate_dir / "acquisition_gate_command.npz"
    temporary = command_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=candidate_commands.astype(np.float32),
            stage=candidate_stages,
            joint_names=np.asarray(names),
            control_fps_hz=np.asarray(fps),
            candidate_id=np.asarray(args.candidate),
            policy_independent=np.asarray(True),
            left_release_permitted=np.asarray(False),
        )
    os.replace(temporary, command_path)
    report = {
        "schema_version": "p14_right_handoff_acquisition_candidate_v1",
        "status": "OFFLINE_PASS"
        if not sum(collision_counts.values())
        and audit["no_left_right_hand_overlap"]
        and audit["right_hand_doll_contact"]
        else "OFFLINE_FAIL",
        "candidate_id": args.candidate,
        "declared_search_size": len(CANDIDATES),
        "candidate_contract": CANDIDATES,
        "varied": "right acquisition pose/orientation only",
        "dx_m": spec["dx_m"],
        "dy_m": spec["dy_m"],
        "yaw_deg": spec["yaw_deg"],
        "target_position_world_m": target_position_world,
        "delta_from_previous_m": [delta_x, delta_y, 0.0],
        "delta_yaw_deg": delta_yaw,
        "solve": solve_reports[-1],
        "static_audit": audit,
        "robot_collision_frame_counts": collision_counts,
        "robot_collision_pairs": geometry["collision_pairs"],
        "freeze_manifest": str(FREEZE),
        "freeze_manifest_sha256": sha256_file(FREEZE),
        "frozen_config_sha256": sha256_file(CONFIG),
        "left_p14_unchanged": True,
        "right_p14_unchanged": True,
        "doll_changed": False,
        "command": str(command_path),
        "command_sha256": sha256_file(command_path),
        "command_frames": len(candidate_commands),
        "left_release_in_command": False,
    }
    atomic_json(candidate_dir / "offline_report.json", report)
    print((candidate_dir / "offline_report.json").read_text(encoding="utf-8"), end="")
    return 0 if report["status"] == "OFFLINE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
