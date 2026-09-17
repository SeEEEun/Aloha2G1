#!/usr/bin/env python3
"""Offline accessibility search around the exact verified RIGHT endpoint.

The doll, RIGHT grasp, RIGHT arm endpoint, policies, and safety contract are
immutable.  The only candidate variables are the LEFT grasp region/orientation
relative to the doll.  The target object pose is the recorded elevated pose of
the 3/3 verified RIGHT transport run, allowing its untouched transport tail to
be used after ownership transfer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mujoco
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
    solve_bounded_pose,
)
from tools.doll_handoff_retargeting.common import load_common_config, load_scene  # noqa: E402
from tools.doll_handoff_retargeting.models import G1Kinematics  # noqa: E402
from tools.evaluation.contracts import authoritative_joint_ranges  # noqa: E402


OUT = ROOT / "outputs/final_methodology_preserving_completion"
SEARCH = OUT / "01_exact_endpoint_presentation_search"
EVIDENCE = OUT / "00_evidence/RECOVERED_POSITIVE_EVIDENCE.json"
CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"
R14_COMMAND = (
    ROOT
    / "outputs/final_task_completion_v1/01_right_transport_grasp/candidates"
    / "R14_RELEASE_D1_CENTERED_DEEP_FAST/right_only_r6_command.npz"
)
R14_EVENT = R14_COMMAND.parent / "physics_r6/event_log.npz"
LEFT_EVENT = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/hand_calibration_v2"
    / "p14_three_digit_preload/trials/left/event_log.npz"
)
B2_COMMAND = (
    ROOT
    / "outputs/dex3_simple_graspable_doll_proxy_v1/scripted_full_task/p14_bilateral"
    / "backward_constructed_handoff/B2_PATH_F40"
    / "right_preload_partial_left_relax_exact_endpoint_v4/full"
    / "scripted_full_task_command.npz"
)

EXPECTED = {
    CONFIG: "07f4c1ab715022d63915b4a480ab5af7374a7d10e5867fea6f2910ffe9946b3e",
    R14_COMMAND: "5f5db710e407f60d39f8e0138729f820fb79e3a85941caccb596574ef3ed50bf",
    R14_EVENT: "75c4c4d9da333c23f78ebd7c5ee6a3c63e35c786fbe245788c1a92fc80efe02b",
    LEFT_EVENT: "191bb175dde285a6ba0f1cc6ab9e6b9a3393546fd9ad043316954973c541418f",
    B2_COMMAND: "2a5b1a7a118e3c0fcac33ee2aeab903b0ccf713a11878acf802ea615865728ab",
}

# Exactly 8 × 4 × 3 = 96 predeclared candidates.
LONG_AXIS_SHIFTS_M = (-0.050, -0.040, -0.030, -0.020, 0.020, 0.030, 0.040, 0.050)
LOCAL_YAWS_DEG = (-30.0, -10.0, 10.0, 30.0)
LOCAL_PITCHES_DEG = (-10.0, 0.0, 10.0)
TOP_COUNT = 16
REFINEMENT_LONG_AXIS_SHIFTS_M = (-0.018, -0.014, -0.010, -0.006)
REFINEMENT_LOCAL_YAWS_DEG = (-20.0, -10.0, 0.0, 10.0)
REFINEMENT_LOCAL_PITCHES_DEG = (-5.0, 0.0, 5.0)


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
    def finite(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: finite(row) for key, row in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(row) for row in item]
        if isinstance(item, np.ndarray):
            return finite(item.tolist())
        if isinstance(item, np.generic):
            return finite(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    atomic_text(
        path,
        json.dumps(
            finite(value),
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


def mean_pose(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.median(position, axis=0)
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).mean().as_matrix()
    return pose


def tool_pose_world(
    g1: G1Kinematics, side: str, static_tool: np.ndarray
) -> np.ndarray:
    position_model, rotation_model, _, _ = g1.static_tool_pose_state(side, static_tool)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = g1.model_to_world_position(position_model)
    pose[:3, :3] = g1.model_to_world_rotation(rotation_model)
    return pose


def signed_ellipsoid_distance(point: np.ndarray, radii: np.ndarray) -> float:
    """Stable first-order signed surface distance for ranking only."""

    scaled = np.asarray(point, dtype=np.float64) / radii
    level = float(np.linalg.norm(scaled))
    gradient = 2.0 * np.asarray(point, dtype=np.float64) / (radii * radii)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= 1.0e-12:
        return -float(np.min(radii))
    return float(2.0 * (level - 1.0) / gradient_norm)


def active_geom_rows(g1: G1Kinematics, side: str) -> list[tuple[int, str]]:
    rows = []
    for geom in range(g1.model.ngeom):
        body = int(g1.model.geom_bodyid[geom])
        body_name = mujoco.mj_id2name(g1.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        if not body_name.startswith((f"{side}_hand", f"{side}_wrist")):
            continue
        if not (g1.model.geom_contype[geom] or g1.model.geom_conaffinity[geom]):
            continue
        rows.append((geom, body_name))
    return rows


def region(body: str) -> str:
    for token in ("thumb", "index", "middle", "palm", "wrist"):
        if token in body:
            return token
    return "other"


def clearance_audit(g1: G1Kinematics, table_z: float) -> dict[str, Any]:
    left = active_geom_rows(g1, "left")
    right = active_geom_rows(g1, "right")
    minima = {key: float("inf") for key in ("thumb", "index", "middle", "palm", "wrist", "other")}
    pairs: dict[str, list[str] | None] = {key: None for key in minima}
    overall = (float("inf"), "", "")
    for left_geom, left_body in left:
        key = region(left_body)
        for right_geom, right_body in right:
            distance = float(
                mujoco.mj_geomDistance(
                    g1.model, g1.data, left_geom, right_geom, 1.0, None
                )
            )
            if distance < minima[key]:
                minima[key] = distance
                pairs[key] = [left_body, right_body]
            if distance < overall[0]:
                overall = (distance, left_body, right_body)
    table_clearance = float("inf")
    table_body = ""
    for geom, body in right:
        conservative = float(g1.data.geom_xpos[geom][2] - g1.model.geom_rbound[geom] - table_z)
        if conservative < table_clearance:
            table_clearance = conservative
            table_body = body
    posture = g1.posture_clearance_state(
        g1.data.qpos[g1.arm_qpos_ids]
    )
    return {
        "minimum_left_right_clearance_m": overall[0],
        "closest_left_right_pair": [overall[1], overall[2]],
        "minimum_clearance_by_left_region_m": minima,
        "closest_pair_by_left_region": pairs,
        "conservative_right_hand_table_clearance_m": table_clearance,
        "right_hand_table_limiting_body": table_body,
        "torso_clearance_m": float(posture["TORSO"]["minimum_distance_m"]),
        "cross_arm_clearance_m": float(posture["CROSS_ARM"]["minimum_distance_m"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round",
        choices=("coarse", "refinement"),
        default="coarse",
        dest="search_round",
    )
    args = parser.parse_args()
    if args.search_round == "coarse":
        long_axis_shifts = LONG_AXIS_SHIFTS_M
        local_yaws = LOCAL_YAWS_DEG
        local_pitches = LOCAL_PITCHES_DEG
        output_stem = "OFFLINE_PRESENTATION_SEARCH"
        candidate_prefix = "P"
    else:
        coarse = read_json(SEARCH / "OFFLINE_PRESENTATION_SEARCH.json")
        if len(coarse["candidates"]) != 96:
            raise RuntimeError("coarse 96-candidate audit is not complete")
        long_axis_shifts = REFINEMENT_LONG_AXIS_SHIFTS_M
        local_yaws = REFINEMENT_LOCAL_YAWS_DEG
        local_pitches = REFINEMENT_LOCAL_PITCHES_DEG
        output_stem = "OFFLINE_PRESENTATION_REFINEMENT"
        candidate_prefix = "F"
    expected_count = len(long_axis_shifts) * len(local_yaws) * len(local_pitches)
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"immutable dependency changed: {path}: {actual}")
    evidence = read_json(EVIDENCE)
    config = read_json(CONFIG)
    common = load_common_config()
    scene = load_scene(common)
    g1 = G1Kinematics(common, scene)
    names, _ = authoritative_joint_ranges()
    lookup = {name: index for index, name in enumerate(names)}
    arm_indices = [lookup[name] for name in g1.arm_joint_names]
    left_indices = [lookup[name] for name in g1.hand_joint_names["left"]]
    right_indices = [lookup[name] for name in g1.hand_joint_names["right"]]
    left_p14 = hand_model(g1, names, config, "left", "POWER_GRASP_P14")
    right_transport = np.asarray(
        evidence["verified_right_transport_grasp"]["right_hand_7d_rad"],
        dtype=np.float64,
    )
    p1_left = hand_model(g1, names, config, "left", "POWER_GRASP_P1")
    p1_right = hand_model(g1, names, config, "right", "POWER_GRASP_P1")

    static_tool = {}
    for side in ("left", "right"):
        primitive = Path(config["source_arm_primitives"][side])
        with np.load(primitive, allow_pickle=False) as archive:
            reference_arm = np.asarray(archive["approach_arm_q_rad"][-1], dtype=np.float64)
        g1.assign(reference_arm, p1_left, p1_right)
        static_tool[side] = np.linalg.inv(g1.wrist_pose(side)) @ g1.whole_hand_grasp_pose(side)

    with np.load(R14_EVENT, allow_pickle=False) as archive:
        r14 = {key: np.asarray(archive[key]) for key in archive.files}
    r14_stage = r14["stage"].astype(str)
    endpoint_rows = np.flatnonzero(r14_stage == "HOLD_ELEVATED")[-240:]
    endpoint_object = mean_pose(
        r14["object_position_world_m"][endpoint_rows],
        r14["object_quaternion_xyzw"][endpoint_rows],
    )
    endpoint_command = np.asarray(r14["commanded_q_rad"][endpoint_rows[-1]], dtype=np.float64)
    right_endpoint_arm = endpoint_command[arm_indices].copy()
    if not np.allclose(endpoint_command[right_indices], right_transport, atol=1.0e-7):
        raise RuntimeError("recorded elevated endpoint is not exact verified RIGHT grasp")

    with np.load(LEFT_EVENT, allow_pickle=False) as archive:
        left_event = {key: np.asarray(archive[key]) for key in archive.files}
    left_stage = left_event["stage"].astype(str)
    left_rows = np.flatnonzero(left_stage == "HOLD_ELEVATED")[-240:]
    left_object = mean_pose(
        left_event["object_position_world_m"][left_rows],
        left_event["object_quaternion_xyzw"][left_rows],
    )
    left_command = np.asarray(left_event["commanded_q_rad"][left_rows[-1]], dtype=np.float64)
    g1.assign(
        left_command[arm_indices],
        left_command[left_indices],
        left_command[right_indices],
    )
    baseline_left_tool = tool_pose_world(g1, "left", static_tool["left"])
    baseline_object_to_left_tool = np.linalg.inv(left_object) @ baseline_left_tool

    with np.load(B2_COMMAND, allow_pickle=False) as archive:
        b2_command = np.asarray(archive["commanded_q_rad"], dtype=np.float64)
        b2_stage = archive["stage"].astype(str)
    b2_left_hold = np.flatnonzero(b2_stage == "LEFT_HANDOFF_HOLD")[-1]
    left_seed = b2_command[b2_left_hold, arm_indices][:7]
    base_full_arm = right_endpoint_arm.copy()
    base_full_arm[:7] = left_seed

    radii = np.asarray(config["geometry_candidates"][0]["dimensions_m"], dtype=np.float64) / 2.0
    table_z = float(config["object"]["table_surface_world_z_m"])
    rows: list[dict[str, Any]] = []
    candidate_number = 0
    for shift in long_axis_shifts:
        for yaw in local_yaws:
            for pitch in local_pitches:
                candidate_number += 1
                candidate_id = f"{candidate_prefix}{candidate_number:03d}_X{int(round(1000*shift)):+d}_YAW{int(yaw):+d}_PITCH{int(pitch):+d}"
                candidate_object_to_left_tool = baseline_object_to_left_tool.copy()
                candidate_object_to_left_tool[:3, 3] += np.asarray([shift, 0.0, 0.0])
                candidate_object_to_left_tool[:3, :3] = (
                    Rotation.from_euler("zy", [yaw, pitch], degrees=True).as_matrix()
                    @ baseline_object_to_left_tool[:3, :3]
                )
                target_tool = endpoint_object @ candidate_object_to_left_tool
                report: dict[str, Any] = {
                    "candidate_id": candidate_id,
                    "left_long_axis_shift_m": shift,
                    "left_local_yaw_deg": yaw,
                    "left_local_pitch_deg": pitch,
                    "candidate_object_to_left_tool": candidate_object_to_left_tool,
                    "target_left_tool_world": target_tool,
                    "target_object_pose_world": endpoint_object,
                }
                try:
                    solution, ik = solve_bounded_pose(
                        g1,
                        "left",
                        static_tool["left"],
                        g1.world_to_model_position(target_tool[:3, 3]),
                        g1.world_to_model_rotation(target_tool[:3, :3]),
                        base_full_arm,
                        left_seed,
                        left_p14,
                        right_transport,
                        20260831 + candidate_number,
                    )
                    # Enforce the exact recorded RIGHT endpoint.
                    solution[7:] = right_endpoint_arm[7:]
                    g1.assign(solution, left_p14, right_transport)
                    geometry = g1.trajectory_geometry(
                        solution[None],
                        left_p14[None],
                        right_transport[None],
                        1.0e-5,
                    )
                    collision_counts = {
                        key: int(np.count_nonzero(value))
                        for key, value in geometry["collision_flags"].items()
                    }
                    clearance = clearance_audit(g1, table_z)
                    pad_signed = {}
                    pad_local = {}
                    for digit in ("thumb", "index", "middle"):
                        pad_model, _ = g1.contact_pose("left", digit)
                        pad_world = g1.model_to_world_position(pad_model)
                        local = endpoint_object[:3, :3].T @ (
                            pad_world - endpoint_object[:3, 3]
                        )
                        pad_local[digit] = local
                        pad_signed[digit] = signed_ellipsoid_distance(local, radii)
                    max_surface_error = max(abs(value) for value in pad_signed.values())
                    exact_right_error = float(
                        np.max(np.abs(solution[7:] - right_endpoint_arm[7:]), initial=0.0)
                    )
                    offline_pass = bool(
                        not sum(collision_counts.values())
                        and clearance["minimum_left_right_clearance_m"] >= 0.0
                        and clearance["conservative_right_hand_table_clearance_m"] >= 0.0
                        and clearance["torso_clearance_m"] >= 0.0
                        and clearance["cross_arm_clearance_m"] >= 0.0
                        and max_surface_error <= 0.015
                        and exact_right_error <= 1.0e-12
                    )
                    score = float(
                        4.0 * clearance["minimum_left_right_clearance_m"]
                        - 1.5 * max_surface_error
                        - 0.001 * np.linalg.norm(solution[:7] - left_seed)
                    )
                    report.update(
                        {
                            "status": "OFFLINE_PASS" if offline_pass else "OFFLINE_FAIL",
                            "left_arm_q_rad": solution[:7],
                            "right_arm_q_rad": solution[7:],
                            "ik": ik,
                            "collision_frame_counts": collision_counts,
                            "collision_pairs": geometry["collision_pairs"],
                            "accessibility": clearance,
                            "left_pad_center_object_m": pad_local,
                            "left_pad_ellipsoid_signed_distance_m": pad_signed,
                            "maximum_left_pad_surface_error_m": max_surface_error,
                            "exact_right_endpoint_max_error_rad": exact_right_error,
                            "ranking_score": score,
                        }
                    )
                except Exception as error:
                    report.update(
                        {
                            "status": "OFFLINE_FAIL",
                            "error": f"{type(error).__name__}: {error}",
                            "ranking_score": -1.0e9,
                        }
                    )
                rows.append(report)
                if candidate_number % 8 == 0:
                    atomic_json(
                        SEARCH / "OFFLINE_PRESENTATION_SEARCH.partial.json",
                        {
                            "schema_version": "exact_verified_endpoint_presentation_search_partial_v1",
                            "completed_candidates": candidate_number,
                            "candidate_count": expected_count,
                            "search_round": args.search_round,
                            "candidates": rows,
                        },
                    )
                    print(
                        f"OFFLINE_PRESENTATION_PROGRESS {candidate_number}/{expected_count}",
                        flush=True,
                    )

    if len(rows) != expected_count:
        raise RuntimeError(f"unexpected candidate count: {len(rows)}")
    passing = sorted(
        (row for row in rows if row["status"] == "OFFLINE_PASS"),
        key=lambda row: row["ranking_score"],
        reverse=True,
    )
    selected_top = [row["candidate_id"] for row in passing[:TOP_COUNT]]
    result = {
        "schema_version": "exact_verified_endpoint_presentation_search_v1",
        "construction": "backward from exact recorded elevated VERIFIED_RIGHT_TRANSPORT_GRASP endpoint",
        "search_round": args.search_round,
        "candidate_count": len(rows),
        "offline_pass_count": len(passing),
        "top_physics_candidate_count": len(selected_top),
        "top_physics_candidates": selected_top,
        "candidate_space": {
            "left_long_axis_shifts_m": long_axis_shifts,
            "left_local_yaws_deg": local_yaws,
            "left_local_pitches_deg": local_pitches,
        },
        "exact_right_endpoint": {
            "object_pose_world": endpoint_object,
            "right_arm_q_rad": right_endpoint_arm[7:],
            "right_hand_q_rad": right_transport,
            "source_command": str(R14_COMMAND),
            "source_command_sha256": sha256_file(R14_COMMAND),
        },
        "baseline_left_object_to_tool": baseline_object_to_left_tool,
        "candidates": rows,
        "doll_or_physics_changed": False,
        "right_grasp_changed": False,
        "policy_used": False,
        "real_robot_used": False,
    }
    SEARCH.mkdir(parents=True, exist_ok=True)
    atomic_json(SEARCH / f"{output_stem}.json", result)
    partial = SEARCH / "OFFLINE_PRESENTATION_SEARCH.partial.json"
    if partial.exists():
        partial.unlink()
    fields = [
        "candidate_id",
        "status",
        "left_long_axis_shift_m",
        "left_local_yaw_deg",
        "left_local_pitch_deg",
        "ranking_score",
        "maximum_left_pad_surface_error_m",
    ]
    csv_path = SEARCH / f"{output_stem}.csv"
    temporary = csv_path.with_suffix(".csv.incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    top_lines = []
    for rank, row in enumerate(passing[:TOP_COUNT], start=1):
        top_lines.append(
            f"| {rank} | {row['candidate_id']} | {1000*row['accessibility']['minimum_left_right_clearance_m']:.3f} | "
            f"{1000*row['maximum_left_pad_surface_error_m']:.3f} | {row['ranking_score']:.6f} |"
        )
    markdown = """# Exact verified-endpoint LEFT presentation search

The RIGHT arm endpoint, 7D Dex3 vector, doll, physics, and safety settings were
held exact. The bounded candidates vary only the LEFT object-relative grasp region
and presentation orientation. The target object pose is the recorded elevated
pose from the 3/3 RIGHT-only transport run.

| Rank | Candidate | Hand-hand clearance (mm) | LEFT pad surface error (mm) | Score |
|---:|---|---:|---:|---:|
""" + "\n".join(top_lines) + f"""

Offline PASS: {len(passing)}/{expected_count}. Top candidates selected for path construction:
`{selected_top}`.
"""
    atomic_text(SEARCH / f"{output_stem}.md", markdown)
    atomic_text(
        OUT / "CURRENT_STATUS.md",
        "# Methodology-preserving completion status\n\n"
        "Current gate: STAGE_A_PATH_CONSTRUCTION\n\n"
        f"- Exact verified-endpoint {args.search_round} candidates: {len(passing)}/{expected_count} PASS.\n"
        f"- Top physics candidates predeclared: {len(selected_top)}.\n"
        "- Doll, RIGHT transport grasp, A/B inputs: unchanged.\n"
        "- Full task/freeze/ACT evaluation: not yet eligible.\n",
    )
    print(json.dumps({
        "candidate_count": len(rows),
        "offline_pass_count": len(passing),
        "top_physics_candidates": selected_top,
    }, indent=2))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
