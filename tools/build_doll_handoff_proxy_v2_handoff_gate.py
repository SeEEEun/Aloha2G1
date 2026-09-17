#!/usr/bin/env python3
"""Build one bounded, policy-independent Dex3 handoff-gate trajectory.

This is an offline MuJoCo/IK preparation step.  It verifies the already-frozen
proxy-v2 dependencies, uses the frozen P8 hand vectors unchanged, and emits a
trajectory that stops after right-hand retention following left release.  It
does not run the scripted full task and has no policy/checkpoint interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import find_g1_dex3_static_phone_grasp as old  # noqa: E402
import refine_g1_dex3_static_phone_contact as contact  # noqa: E402
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import (  # noqa: E402
    AUTHORITATIVE_REFERENCES,
    authoritative_joint_ranges,
)


CONFIG = ROOT / "configs/doll_handoff_graspable_proxy_v2.json"
FREEZE = ROOT / "outputs/doll_handoff_graspable_proxy_v2/FREEZE_MANIFEST.json"
OUTPUT = ROOT / "outputs/doll_handoff_graspable_proxy_v2/handoff_gate/H1_yaw60_dx30_dz15"


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


def minimum_jerk(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, max(2, count), dtype=np.float64)
    blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return start[None] + blend[:, None] * (end - start)[None]


def hand_model(
    g1: G1Kinematics,
    names: list[str],
    config: dict[str, Any],
    side: str,
    state: str,
) -> np.ndarray:
    block = slice(14, 21) if side == "left" else slice(21, 28)
    by_name = dict(zip(names[block], config["hand_states"][side][state], strict=True))
    return np.asarray([by_name[name] for name in g1.hand_joint_names[side]])


def canonical_row(
    names: list[str],
    g1: G1Kinematics,
    arm: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    model_names = list(g1.arm_joint_names) + list(g1.hand_joint_names["left"]) + list(
        g1.hand_joint_names["right"]
    )
    values = np.r_[arm, left, right]
    lookup = dict(zip(model_names, values, strict=True))
    return np.asarray([lookup[name] for name in names], dtype=np.float64)


def model_parts(
    names: list[str], g1: G1Kinematics, canonical: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = dict(zip(names, canonical, strict=True))
    return (
        np.asarray([lookup[name] for name in g1.arm_joint_names]),
        np.asarray([lookup[name] for name in g1.hand_joint_names["left"]]),
        np.asarray([lookup[name] for name in g1.hand_joint_names["right"]]),
    )


def rotations_between(first: np.ndarray, second: np.ndarray, count: int) -> np.ndarray:
    times = np.linspace(0.0, 1.0, max(2, count))
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack((first, second))))(times).as_matrix()


def solve_path(
    g1: G1Kinematics,
    side: str,
    static_tool: np.ndarray,
    positions_model: np.ndarray,
    rotations_model: np.ndarray,
    seed_full_arm: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    clearance_target_m: float = 0.001,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    block = slice(0, 7) if side == "left" else slice(7, 14)
    lower = g1.arm_limits[block, 0] + 1.0e-7
    upper = g1.arm_limits[block, 1] - 1.0e-7
    full = np.asarray(seed_full_arm, dtype=np.float64).copy()
    previous = full[block].copy()
    rows: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for target_position, target_rotation in zip(
        positions_model, rotations_model, strict=True
    ):
        fixed = full.copy()

        def residual(active: np.ndarray) -> np.ndarray:
            candidate = fixed.copy()
            candidate[block] = active
            g1.assign(candidate, left_hand, right_hand)
            position, rotation, _, _ = g1.static_tool_pose_state(side, static_tool)
            clearance = g1.posture_clearance_state(candidate)
            return np.r_[
                100.0 * (position - target_position),
                8.0
                * Rotation.from_matrix(rotation.T @ target_rotation).as_rotvec(),
                250.0
                * max(
                    0.0,
                    clearance_target_m
                    - float(clearance["TORSO"]["minimum_distance_m"]),
                ),
                150.0
                * max(
                    0.0,
                    clearance_target_m
                    - float(clearance["CROSS_ARM"]["minimum_distance_m"]),
                ),
                0.01 * (active - previous),
            ]

        solution = least_squares(
            residual,
            np.clip(previous, lower, upper),
            bounds=(lower, upper),
            max_nfev=400,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        full[block] = solution.x
        previous = solution.x.copy()
        value = residual(solution.x)
        clearance = g1.posture_clearance_state(full)
        rows.append(full.copy())
        reports.append(
            {
                "success": bool(solution.success),
                "position_error_m": float(np.linalg.norm(value[:3]) / 100.0),
                "orientation_error_rad": float(np.linalg.norm(value[3:6]) / 8.0),
                "torso_clearance_m": float(
                    clearance["TORSO"]["minimum_distance_m"]
                ),
                "cross_arm_clearance_m": float(
                    clearance["CROSS_ARM"]["minimum_distance_m"]
                ),
            }
        )
    return np.asarray(rows), reports


def solve_bounded_pose(
    g1: G1Kinematics,
    side: str,
    static_tool: np.ndarray,
    position_model: np.ndarray,
    rotation_model: np.ndarray,
    base_full_arm: np.ndarray,
    reference_active: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use a fixed 10-seed redundancy search; reject inexact/colliding branches."""

    block = slice(0, 7) if side == "left" else slice(7, 14)
    lower = g1.arm_limits[block, 0] + 1.0e-7
    upper = g1.arm_limits[block, 1] - 1.0e-7
    nominal = np.asarray(base_full_arm, dtype=np.float64)
    rng = np.random.default_rng(seed)
    seeds = [
        np.asarray(reference_active, dtype=np.float64),
        nominal[block].copy(),
        *[
            np.clip(nominal[block] + rng.normal(0.0, 0.55, 7), lower, upper)
            for _ in range(8)
        ],
    ]
    results: list[dict[str, Any]] = []
    for seed_index, active_seed in enumerate(seeds):
        fixed = nominal.copy()

        def residual(active: np.ndarray) -> np.ndarray:
            candidate = fixed.copy()
            candidate[block] = active
            g1.assign(candidate, left_hand, right_hand)
            position, rotation, _, _ = g1.static_tool_pose_state(side, static_tool)
            clearance = g1.posture_clearance_state(candidate)
            return np.r_[
                100.0 * (position - position_model),
                8.0
                * Rotation.from_matrix(rotation.T @ rotation_model).as_rotvec(),
                250.0
                * max(
                    0.0,
                    0.001 - float(clearance["TORSO"]["minimum_distance_m"]),
                ),
                150.0
                * max(
                    0.0,
                    0.001 - float(clearance["CROSS_ARM"]["minimum_distance_m"]),
                ),
                0.0005 * (active - nominal[block]),
            ]

        solution = least_squares(
            residual,
            np.clip(active_seed, lower, upper),
            bounds=(lower, upper),
            max_nfev=500,
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        full = fixed.copy()
        full[block] = solution.x
        g1.assign(full, left_hand, right_hand)
        position, rotation, _, _ = g1.static_tool_pose_state(side, static_tool)
        clearance = g1.posture_clearance_state(full)
        results.append(
            {
                "full_arm": full,
                "seed_index": seed_index,
                "position_error_m": float(np.linalg.norm(position - position_model)),
                "orientation_error_rad": float(
                    np.linalg.norm(
                        Rotation.from_matrix(rotation.T @ rotation_model).as_rotvec()
                    )
                ),
                "torso_clearance_m": float(
                    clearance["TORSO"]["minimum_distance_m"]
                ),
                "cross_arm_clearance_m": float(
                    clearance["CROSS_ARM"]["minimum_distance_m"]
                ),
            }
        )
    valid = [
        row
        for row in results
        if row["position_error_m"] <= 0.001
        and row["orientation_error_rad"] <= 0.02
        and row["torso_clearance_m"] >= 0.0
        and row["cross_arm_clearance_m"] >= 0.0
    ]
    if not valid:
        best = min(
            results,
            key=lambda row: max(0.0, -row["torso_clearance_m"])
            + max(0.0, -row["cross_arm_clearance_m"])
            + row["position_error_m"]
            + 0.03 * row["orientation_error_rad"],
        )
        raise RuntimeError(f"bounded collision-free pose solve failed: {best}")
    chosen = max(
        valid,
        key=lambda row: min(row["torso_clearance_m"], row["cross_arm_clearance_m"]),
    )
    report = {key: value for key, value in chosen.items() if key != "full_arm"}
    report["bounded_seed_count"] = len(seeds)
    return np.asarray(chosen["full_arm"]), report


def preliminary_candidate_audit(
    g1: G1Kinematics,
    arm: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    object_center_world: np.ndarray,
    dimensions: np.ndarray,
) -> dict[str, Any]:
    base = mujoco.MjModel.from_xml_path(str(g1.path))
    directory = Path(tempfile.mkdtemp(prefix="proxy_v2_handoff_gate_"))
    xml_path = directory / "model.xml"
    mujoco.mj_saveLastXML(str(xml_path), base)
    text = xml_path.read_text(encoding="utf-8").replace(
        'meshdir="assets/"', f'meshdir="{g1.path.parent / "assets"}/"'
    )
    center = g1.world_to_model_position(object_center_world)
    radii = dimensions / 2.0
    text = text.replace(
        "<worldbody>",
        "<worldbody>"
        f'<body name="proxy_v2_handoff" pos="{center[0]} {center[1]} {center[2]}">'
        f'<geom name="proxy_v2_handoff_geom" type="ellipsoid" '
        f'size="{radii[0]} {radii[1]} {radii[2]}" contype="1" conaffinity="1"/>'
        "</body>",
        1,
    )
    xml_path.write_text(text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    arm_ids = [
        model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in g1.arm_joint_names
    ]
    hand_ids = {
        side: [
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in g1.hand_joint_names[side]
        ]
        for side in ("left", "right")
    }
    data.qpos[:] = model.key_qpos[0]
    data.qpos[arm_ids] = arm
    data.qpos[hand_ids["left"]] = left
    data.qpos[hand_ids["right"]] = right
    mujoco.mj_forward(model, data)
    proxy = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "proxy_v2_handoff_geom"
    )
    geoms: dict[str, list[tuple[int, str]]] = {"left": [], "right": []}
    for geom_id in range(model.ngeom):
        body = old.body_name(model, geom_id)
        for side in geoms:
            if (
                body.startswith(f"{side}_hand") or body.startswith(f"{side}_wrist")
            ) and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
                geoms[side].append((geom_id, body))
    cross = []
    for left_geom, left_body in geoms["left"]:
        for right_geom, right_body in geoms["right"]:
            distance, _ = contact.distance(model, data, left_geom, right_geom)
            cross.append((distance, left_body, right_body))
    closest = min(cross, key=lambda row: row[0])
    distances: dict[str, float] = {}
    for digit in ("thumb", "index", "middle"):
        suffix = "2_link" if digit == "thumb" else "1_link"
        geom = contact.collision_geoms(model, f"right_hand_{digit}_{suffix}")[-1]
        distances[digit] = float(contact.distance(model, data, geom, proxy)[0])
    return {
        "minimum_left_right_clearance_m": float(closest[0]),
        "closest_left_right_body_pair": [closest[1], closest[2]],
        "right_distal_proxy_signed_distance_m": distances,
        "right_contact_count_at_1mm_tolerance": int(
            sum(value <= 0.001 for value in distances.values())
        ),
        "right_hand_doll_contact": bool(min(distances.values()) <= 0.001),
        "no_left_right_hand_overlap": bool(closest[0] >= 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG)
    freeze = read_json(FREEZE)
    expected = freeze["hashes"]
    dependencies = {
        "config": CONFIG,
        "source_scene": Path(config["source_scene"]),
        "left_arm_primitive": Path(config["source_arm_primitives"]["left"]),
        "right_arm_primitive": Path(config["source_arm_primitives"]["right"]),
        "calibration_runner": ROOT / "tools/run_doll_handoff_graspable_proxy_v2_isaac.py",
    }
    for key, path in dependencies.items():
        if sha256_file(path) != expected[key]["sha256"]:
            raise RuntimeError(f"frozen dependency changed: {key}")
    if freeze.get("status") != "FROZEN_BILATERAL_GRASPABILITY_PASS":
        raise RuntimeError("bilateral proxy graspability is not frozen PASS")

    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    nominal_arm, _ = g1.derive_task_ready_nominal(common, scene)
    names, _ = authoritative_joint_ranges()
    left_open = hand_model(g1, names, config, "left", "OPEN")
    right_open = hand_model(g1, names, config, "right", "OPEN")
    left_preshape = hand_model(g1, names, config, "left", "PRESHAPE")
    right_preshape = hand_model(g1, names, config, "right", "PRESHAPE")
    left_power = hand_model(g1, names, config, "left", "POWER_GRASP_P8")
    right_power = hand_model(g1, names, config, "right", "POWER_GRASP_P8")
    right_acquire_hand = right_preshape.copy()
    for joint_index, joint_name in enumerate(g1.hand_joint_names["right"]):
        if "thumb" in joint_name or "middle" in joint_name:
            right_acquire_hand[joint_index] = right_power[joint_index]
    left_middle_support_hand = left_power.copy()
    for joint_index, joint_name in enumerate(g1.hand_joint_names["left"]):
        if "thumb" in joint_name or "index" in joint_name:
            left_middle_support_hand[joint_index] = left_open[joint_index]
    p1 = {
        side: hand_model(g1, names, config, side, "POWER_GRASP_P1")
        for side in ("left", "right")
    }
    primitives: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        with np.load(dependencies[f"{side}_arm_primitive"], allow_pickle=False) as archive:
            primitives[side] = {key: np.asarray(archive[key]) for key in archive.files}
    reference_arm = {
        side: np.asarray(primitives[side]["lift_arm_q_rad"][-1], dtype=np.float64)
        for side in ("left", "right")
    }
    origin = g1.model_to_world_position(np.zeros(3))
    world_from_model = np.column_stack(
        [
            g1.model_to_world_position(np.eye(3)[axis]) - origin
            for axis in range(3)
        ]
    )
    static_tool: dict[str, np.ndarray] = {}
    reference_rotation_world: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        g1.assign(reference_arm[side], p1["left"], p1["right"])
        static_tool[side] = (
            np.linalg.inv(g1.wrist_pose(side)) @ g1.whole_hand_grasp_pose(side)
        )
        g1.assign(reference_arm[side], left_power, right_power)
        _, rotation, _, _ = g1.static_tool_pose_state(side, static_tool[side])
        reference_rotation_world[side] = world_from_model @ rotation

    # These offsets are derived from the final, hash-frozen bilateral trials.
    object_centers: dict[str, np.ndarray] = {}
    validation_results = {
        side: read_json(ROOT / freeze["bilateral_validation"][side]["result"])
        for side in ("left", "right")
    }
    for side in ("left", "right"):
        log_path = Path(validation_results[side]["event_log"])
        with np.load(log_path, allow_pickle=False) as archive:
            labels = archive["stage"].astype(str)
            positions = np.asarray(archive["object_position_world_m"], dtype=np.float64)
            hold = positions[labels == "HOLD_ELEVATED"]
            object_centers[side] = np.median(hold[-30:], axis=0)
    declared_tool = np.asarray(
        primitives["left"]["target_whole_hand_position_world_m"][-1], dtype=np.float64
    )
    tool_offset = {
        side: declared_tool - object_centers[side] for side in ("left", "right")
    }

    candidate = {
        "candidate_id": "H1_YAW60_DX30_DZ15",
        "search_scope": {
            "coarse_candidates": 1452,
            "refined_candidates": 336,
            "refined_yaw_deg": [45.0, 60.0, 75.0],
            "refined_dx_m": [0.02, 0.025, 0.03, 0.035],
            "refined_dy_m": [0.0, 0.005, 0.01, 0.015],
            "refined_dz_m": [-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015],
            "bounded": True,
        },
        "handoff_object_center_world_m": [0.4175, 0.21, float(object_centers["left"][2])],
        "right_world_z_yaw_from_frozen_grasp_deg": 60.0,
        "right_tool_translation_from_frozen_offset_m": [0.03, 0.0, 0.015],
        "selection_rule": "smallest absolute preliminary proxy penetration among refined candidates with >=1 mm hand-hand clearance, exact IK, and actual right distal contact",
    }
    object_center = np.asarray(candidate["handoff_object_center_world_m"])
    left_start_world = declared_tool.copy()
    left_target_world = object_center + tool_offset["left"]
    right_candidate_world = (
        object_center
        + tool_offset["right"]
        + np.asarray(candidate["right_tool_translation_from_frozen_offset_m"])
    )
    right_final_world = object_center + tool_offset["right"]
    right_candidate_rotation_world = (
        Rotation.from_euler(
            "z", candidate["right_world_z_yaw_from_frozen_grasp_deg"], degrees=True
        ).as_matrix()
        @ reference_rotation_world["right"]
    )

    fps = float(config["timing"]["control_fps_hz"])
    left_table_world = np.asarray(
        primitives["left"]["target_whole_hand_position_world_m"][0],
        dtype=np.float64,
    )
    left_rotation_model = world_from_model.T @ reference_rotation_world["left"]
    right_rotation_model = world_from_model.T @ reference_rotation_world["right"]
    left_table_arm, table_pose_report = solve_bounded_pose(
        g1,
        "left",
        static_tool["left"],
        g1.world_to_model_position(left_table_world),
        left_rotation_model,
        nominal_arm,
        reference_arm["left"][:7],
        left_power,
        right_open,
        2026082701,
    )
    left_lift_count = int(round(float(config["timing"]["lift_transition_s"]) * fps))
    left_lift_positions = minimum_jerk(
        left_table_world, left_start_world, left_lift_count
    )
    left_lift, lift_reports = solve_path(
        g1,
        "left",
        static_tool["left"],
        g1.world_to_model_position(left_lift_positions),
        np.repeat(left_rotation_model[None], left_lift_count, axis=0),
        left_table_arm,
        left_power,
        right_open,
    )
    left_transport_count = int(round(2.0 * fps))
    positions = minimum_jerk(left_start_world, left_target_world, left_transport_count)
    rotations = np.repeat(
        left_rotation_model[None],
        left_transport_count,
        axis=0,
    )
    left_transport, left_reports = solve_path(
        g1,
        "left",
        static_tool["left"],
        g1.world_to_model_position(positions),
        rotations,
        left_lift[-1],
        left_power,
        right_open,
    )
    handoff_arm = left_transport[-1]
    right_candidate_arm, right_candidate_pose_report = solve_bounded_pose(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(right_candidate_world),
        world_from_model.T @ right_candidate_rotation_world,
        handoff_arm,
        reference_arm["right"][7:14],
        left_power,
        right_acquire_hand,
        2026082702,
    )
    left_parked = right_candidate_arm.copy()
    left_parked[:7] = nominal_arm[:7]
    right_final_arm, right_final_pose_report = solve_bounded_pose(
        g1,
        "right",
        static_tool["right"],
        g1.world_to_model_position(right_final_world),
        right_rotation_model,
        left_parked,
        reference_arm["right"][7:14],
        left_open,
        right_power,
        2026082703,
    )
    left_retreat_arm = right_final_arm.copy()
    left_retreat_arm[:7] = nominal_arm[:7]

    rows: list[np.ndarray] = []
    stages: list[str] = []

    def append(arm: np.ndarray, left: np.ndarray, right: np.ndarray, stage: str) -> None:
        rows.append(canonical_row(names, g1, arm, left, right))
        stages.append(stage)

    for _ in range(max(1, round(float(config["timing"]["open_hold_s"]) * fps))):
        append(left_table_arm, left_open, right_open, "LEFT_SETUP_OPEN")
    for left in minimum_jerk(
        left_open,
        left_preshape,
        int(round(float(config["timing"]["preshape_transition_s"]) * fps)),
    )[1:]:
        append(left_table_arm, left, right_open, "LEFT_SETUP_PRESHAPE")
    for left in minimum_jerk(
        left_preshape,
        left_power,
        int(round(float(config["timing"]["power_close_s"]) * fps)),
    )[1:]:
        append(left_table_arm, left, right_open, "LEFT_SETUP_POWER_GRASP")
    for _ in range(
        max(1, round(float(config["timing"]["gravity_retention_s"]) * fps))
    ):
        append(left_table_arm, left_power, right_open, "LEFT_SETUP_GRAVITY_RETENTION")
    for arm in left_lift[1:]:
        append(arm, left_power, right_open, "LEFT_SETUP_LIFT_5CM")
    for _ in range(
        max(1, round(float(config["timing"]["elevated_hold_s"]) * fps))
    ):
        append(left_lift[-1], left_power, right_open, "LEFT_SETUP_HOLD_ELEVATED")
    for arm in left_transport[1:]:
        append(arm, left_power, right_open, "LEFT_TRANSPORT_TO_HANDOFF")
    for _ in range(int(round(1.0 * fps))):
        append(handoff_arm, left_power, right_open, "LEFT_PRETRANSFER_RETENTION")
    approach_count = int(round(2.0 * fps))
    for arm in minimum_jerk(handoff_arm, right_candidate_arm, approach_count)[1:]:
        append(arm, left_power, right_open, "RIGHT_APPROACH_OPEN")
    for right in minimum_jerk(right_open, right_preshape, int(round(1.0 * fps)))[1:]:
        append(right_candidate_arm, left_power, right, "RIGHT_PRESHAPE")
    for right in minimum_jerk(
        right_preshape, right_acquire_hand, int(round(1.5 * fps))
    )[1:]:
        append(right_candidate_arm, left_power, right, "RIGHT_ACQUIRE")
    for _ in range(int(round(0.5 * fps))):
        append(
            right_candidate_arm,
            left_power,
            right_acquire_hand,
            "DUAL_CONTACT_VERIFY",
        )
    transfer_count = int(round(2.0 * fps))
    right_arms = minimum_jerk(right_candidate_arm, right_final_arm, transfer_count)
    left_arms = minimum_jerk(handoff_arm, left_retreat_arm, transfer_count)
    u = np.linspace(0.0, 1.0, transfer_count)

    def blend(value: np.ndarray) -> np.ndarray:
        return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5

    right_enclose = blend(np.clip((u - 0.35) / 0.65, 0.0, 1.0))
    left_hands = minimum_jerk(left_power, left_open, transfer_count)
    right_hands = right_acquire_hand[None] + right_enclose[:, None] * (
        right_power - right_acquire_hand
    )[None]
    for frame in range(1, transfer_count):
        arm = right_arms[frame].copy()
        arm[:7] = left_arms[frame, :7]
        append(
            arm,
            left_hands[frame],
            right_hands[frame],
            "SYNCHRONIZED_CONTACT_SAFE_TRANSFER",
        )
    for _ in range(int(round(1.0 * fps))):
        append(left_retreat_arm, left_open, right_power, "RIGHT_POST_RELEASE_RETENTION")

    commands = np.asarray(rows)
    labels = np.asarray(stages)
    arm_model = np.empty((len(commands), 14), dtype=np.float64)
    left_model = np.empty((len(commands), 7), dtype=np.float64)
    right_model = np.empty((len(commands), 7), dtype=np.float64)
    for frame, command in enumerate(commands):
        arm_model[frame], left_model[frame], right_model[frame] = model_parts(
            names, g1, command
        )
    geometry = g1.trajectory_geometry(arm_model, left_model, right_model, 1.0e-5)
    collision_counts = {
        key: int(np.count_nonzero(value)) for key, value in geometry["collision_flags"].items()
    }
    candidate_audit = preliminary_candidate_audit(
        g1,
        right_candidate_arm,
        left_power,
        right_acquire_hand,
        object_center,
        np.asarray([0.065, 0.060, 0.060]),
    )
    joint_contract = read_json(AUTHORITATIVE_REFERENCES["joint_ranges"])
    lower = np.asarray(
        [float(value["minimum"]) for value in joint_contract["joint_specs"]]
    )
    upper = np.asarray(
        [float(value["maximum"]) for value in joint_contract["joint_specs"]]
    )
    violations = (commands < lower[None] - 1.0e-9) | (
        commands > upper[None] + 1.0e-9
    )
    ik_reports = lift_reports + left_reports
    arm_reachable = bool(
        all(row["success"] for row in ik_reports)
        and max(row["position_error_m"] for row in ik_reports) <= 0.005
        and max(row["orientation_error_rad"] for row in ik_reports) <= 0.1
        and all(
            row["position_error_m"] <= 0.001
            and row["orientation_error_rad"] <= 0.02
            for row in (
                table_pose_report,
                right_candidate_pose_report,
                right_final_pose_report,
            )
        )
    )
    hard_collision_count = sum(collision_counts.values())
    report = {
        "schema_version": "doll_handoff_proxy_v2_handoff_candidate_offline_v1",
        "status": (
            "OFFLINE_GATE_PASS"
            if candidate_audit["right_hand_doll_contact"]
            and candidate_audit["no_left_right_hand_overlap"]
            and arm_reachable
            and hard_collision_count == 0
            and not np.any(violations)
            else "OFFLINE_GATE_FAIL"
        ),
        "candidate": candidate,
        "frozen_proxy_config_sha256": sha256_file(CONFIG),
        "frozen_manifest_sha256": sha256_file(FREEZE),
        "frozen_grasp_profile": "P8",
        "grasp_vectors_modified": False,
        "object_or_material_modified": False,
        "preliminary_candidate_audit": candidate_audit,
        "arm_reachability": {
            "status": "PASS" if arm_reachable else "FAIL",
            "maximum_ik_position_error_m": max(
                row["position_error_m"] for row in ik_reports
            ),
            "maximum_ik_orientation_error_rad": max(
                row["orientation_error_rad"] for row in ik_reports
            ),
            "bounded_pose_reports": {
                "left_table": table_pose_report,
                "right_acquisition": right_candidate_pose_report,
                "right_post_release": right_final_pose_report,
            },
        },
        "trajectory_collision_audit": {
            "status": "PASS" if hard_collision_count == 0 else "FAIL",
            "penetration_tolerance_m": 1.0e-5,
            "frame_counts": collision_counts,
            "pairs": geometry["collision_pairs"],
            "records": geometry["collision_records"],
        },
        "joint_hard_limits": {
            "status": "PASS" if not np.any(violations) else "FAIL",
            "violation_count": int(np.count_nonzero(violations)),
        },
        "command_frames": int(len(commands)),
        "command_duration_s": float(len(commands) / fps),
        "policy_used": False,
        "scripted_full_task_run": False,
        "environment_refrozen": False,
        "act_a_b_run": False,
    }
    command_path = output / "handoff_gate_command.npz"
    temporary = command_path.with_suffix(".npz.incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            commanded_q_rad=commands.astype(np.float32),
            stage=labels,
            joint_names=np.asarray(names),
            control_fps_hz=np.asarray(fps),
            object_handoff_center_world_m=object_center.astype(np.float32),
            policy_independent=np.asarray(True),
            scripted_full_task=np.asarray(False),
            candidate_id=np.asarray(candidate["candidate_id"]),
        )
    os.replace(temporary, command_path)
    report["command"] = str(command_path)
    report["command_sha256"] = sha256_file(command_path)
    atomic_json(output / "offline_candidate_report.json", report)
    print((output / "offline_candidate_report.json").read_text(encoding="utf-8"), end="")
    return 0 if report["status"] == "OFFLINE_GATE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
