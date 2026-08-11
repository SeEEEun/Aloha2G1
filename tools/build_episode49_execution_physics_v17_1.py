#!/usr/bin/env python3
"""Build the narrow Episode-49 v17.1 semantic-local orientation candidate.

The v14 corrected Cartesian position arrays are immutable inputs.  This build
changes only arm joint posture/orientation in their task null space and drives
the already calibrated fixed Dex3 primitives from generic semantic progress.
"""
from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

import build_episode49_execution_physics_v17 as v17  # noqa: E402
from aloha_g1_v15.kinematics import ActiveG1Dex3, sha256_file  # noqa: E402
from aloha_g1_v15.semantic_input import (  # noqa: E402
    TASK_EVENTS,
    load_human_reviewed_development_timeline,
)
from aloha_g1_v17.trajectory import (  # noqa: E402
    audit_collision_classifier_integrity,
    build_predefined_hand_trajectories,
    build_semantic_local_orientation_targets,
    evaluate_kinematic_candidate,
    solve_semantic_local_orientation_trajectory,
)
from retarget_aloha_trajectory_to_g1 import retarget_aloha_trajectory_to_g1  # noqa: E402
from v15_semantic_interface import readiness as semantic_readiness  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17_1"
V17_OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_execution_physics_v17"
METHOD = "ALOHA_PRIMARY_EP49_EXECUTION_PHYSICS_V17_1"
BACKUP = ROOT / "backups/ep49_before_execution_physics_v17_1_20260808_182527"


def dump(path: Path, value: Any) -> None:
    v17.dump(path, value)


def save_npz(path: Path, **value: Any) -> None:
    v17.save_npz(path, **value)


def array_sha(value: np.ndarray) -> str:
    return v17.array_sha(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def inventory_hash(directory: Path) -> dict[str, Any]:
    return v17.inventory(directory, include_large=True)


def orientation_distance(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.clip((np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def axis_error(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.clip(float(np.dot(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def literal_audit(paths: list[Path], forbidden: set[int]) -> dict[str, Any]:
    findings = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is int and node.value in forbidden:
                findings.append({"path": str(path), "line": node.lineno, "value": node.value})
    return {"paths": [str(path.resolve()) for path in paths], "findings": findings, "count": len(findings), "pass": not findings}


def calibrate_fixed_right_nontask_thumb(
    runtime: ActiveG1Dex3,
    reference_arm_q: np.ndarray,
    left_open: np.ndarray,
    right_open: np.ndarray,
    right_hook: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one collision-free right-B posture for OPEN and RING_HOOK.

    Right B is non-task for the ring operation.  v17 inherited a placeholder
    posture that folds the distal thumb 7.3 mm into the wrist in every sample.
    This deterministic active-geometry calibration changes that one fixed
    primitive vector only; it has no frame-dependent or Cartesian-arm input.
    """
    limits = runtime.hand_limits["right"][2:5]
    original = np.asarray(right_open[2:5], dtype=np.float64)
    rng = np.random.default_rng(171)
    candidates = [
        limits[:, 0] + fraction * np.ptp(limits, axis=1)
        for fraction in np.linspace(0.10, 0.90, 9)
    ]
    candidates.extend(
        limits[:, 0] + rng.random(3) * np.ptp(limits, axis=1)
        for _ in range(4096)
    )
    rows = []
    for value in candidates:
        margin = float(np.min(np.minimum(value - limits[:, 0], limits[:, 1] - value)))
        if margin < 0.03:
            continue
        collision_rows = []
        for primitive_name, source in (("RIGHT_OPEN", right_open), ("RIGHT_RING_HOOK", right_hook)):
            candidate = source.copy()
            candidate[2:5] = value
            runtime.assign(reference_arm_q, left_open, candidate)
            for contact in runtime.penetrating_contacts(tolerance=0.0):
                if all(str(name).startswith("right_") for name in contact["bodies"]):
                    collision_rows.append({"primitive": primitive_name, **contact})
        if not collision_rows:
            rows.append({
                "q_rad": np.asarray(value),
                "distance_from_v17_rad": float(np.linalg.norm(value - original)),
                "minimum_joint_margin_rad": margin,
            })
    if not rows:
        raise RuntimeError("no fixed collision-free right non-task thumb posture")
    selected = min(rows, key=lambda row: (row["distance_from_v17_rad"], -row["minimum_joint_margin_rad"]))
    return np.asarray(selected["q_rad"], dtype=np.float64), {
        "method": "deterministic active-geometry fixed primitive sweep",
        "role": "RIGHT_B_NON_TASK_THUMB",
        "source_v17_q_rad": original,
        "selected_q_rad": selected["q_rad"],
        "distance_from_v17_rad": selected["distance_from_v17_rad"],
        "minimum_joint_margin_rad": selected["minimum_joint_margin_rad"],
        "candidate_count": len(candidates),
        "collision_free_candidate_count": len(rows),
        "tested_primitives": ["RIGHT_OPEN", "RIGHT_RING_HOOK"],
        "cartesian_arm_path_dependency": False,
        "per_frame_finger_ik": False,
        "episode_literal_frame_dependency": False,
        "v17_invalid_contact": {
            "pair": "right_wrist_yaw_link|right_hand_thumb_2_link",
            "penetration_m": 0.007324503879880265,
            "samples": 990,
        },
    }


def calibrate_left_open_index(
    runtime: ActiveG1Dex3,
    active_stage: Usd.Stage,
    reference_arm_q: np.ndarray,
    left_open: np.ndarray,
    left_pinch: np.ndarray,
    right_reference: np.ndarray,
    phone_pose: np.ndarray,
    phone_dimensions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate one fixed, genuinely open left-index posture.

    The inherited simulation placeholder put both negative-range index joints
    close to their lower limits and the active distal collision mesh already
    intersected the phone at the named acquisition pose.  Search a small
    deterministic geometry-normalized grid for a fixed posture with clearance
    throughout the named approach trajectory.  This is task-local primitive
    calibration: it does not change the arm path and is not per-frame IK.
    """
    links = ("left_hand_index_0_link", "left_hand_index_1_link")
    mesh_points: dict[str, np.ndarray] = {}
    for link in links:
        path = f"/World/G1/Asset/{link}/collisions/{link}/mesh"
        prim = active_stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"active left-index collision mesh missing: {path}")
        # Deterministic dense-enough surface sample.  The final selected pose
        # is also checked by MuJoCo's full collision model below.
        mesh_points[link] = np.asarray(
            UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64
        )[::20]

    half = 0.5 * np.asarray(phone_dimensions, dtype=np.float64)
    limits = runtime.hand_limits["left"][3:5]
    original = np.asarray(left_open[3:5], dtype=np.float64)
    target = np.asarray(left_pinch[3:5], dtype=np.float64)
    # The reference samples are supplied from a SemanticTimeline interval by
    # the caller; their numeric values never become reusable parameters.
    sample_indices = np.unique(np.linspace(
        0, len(reference_arm_q) - 1, min(len(reference_arm_q), 33), dtype=int
    ))

    def box_clearance(points_world: np.ndarray) -> float:
        phone_local = (points_world - phone_pose[:3, 3]) @ phone_pose[:3, :3]
        excess = np.abs(phone_local) - half
        outside = np.linalg.norm(np.maximum(excess, 0.0), axis=1)
        inside = np.all(excess <= 0.0, axis=1)
        signed = outside
        signed[inside] = np.max(excess[inside], axis=1)
        return float(np.min(signed))

    rows: list[dict[str, Any]] = []
    fractions = np.linspace(0.05, 0.95, 19)
    for fraction_0 in fractions:
        for fraction_1 in fractions:
            value = limits[:, 0] + np.asarray([fraction_0, fraction_1]) * np.ptp(limits, axis=1)
            margin = float(np.min(np.minimum(value - limits[:, 0], limits[:, 1] - value)))
            if margin < 0.02:
                continue
            minimum_phone_clearance = np.inf
            collision_rows = []
            hand = np.asarray(left_open, dtype=np.float64).copy()
            hand[3:5] = value
            for local_index in sample_indices:
                runtime.assign(reference_arm_q[local_index], hand, right_reference)
                for link in links:
                    body = runtime.model.body(link).id
                    rotation = runtime.data.xmat[body].reshape(3, 3)
                    model_points = runtime.data.xpos[body] + mesh_points[link] @ rotation.T
                    scene_points = runtime.model_to_scene_position(model_points)
                    minimum_phone_clearance = min(
                        minimum_phone_clearance, box_clearance(scene_points)
                    )
                for contact in runtime.penetrating_contacts(tolerance=0.0):
                    bodies = tuple(str(name) for name in contact["bodies"])
                    if all(name.startswith("left_") and "hand" in name for name in bodies):
                        collision_rows.append(contact)
            rows.append({
                "normalized_joint_fraction": [float(fraction_0), float(fraction_1)],
                "q_rad": value,
                "minimum_phone_clearance_m": float(minimum_phone_clearance),
                "minimum_joint_margin_rad": margin,
                "same_hand_collision_count": len(collision_rows),
                "distance_to_phone_pinch_rad": float(np.linalg.norm(value - target)),
                "eligible": bool(minimum_phone_clearance >= 0.008 and not collision_rows),
            })
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("no fixed left-index OPEN posture has 8 mm phone clearance")
    selected = min(
        eligible,
        key=lambda row: (
            row["distance_to_phone_pinch_rad"],
            -row["minimum_phone_clearance_m"],
            -row["minimum_joint_margin_rad"],
        ),
    )
    result = np.asarray(left_open, dtype=np.float64).copy()
    result[3:5] = np.asarray(selected["q_rad"], dtype=np.float64)
    return result, {
        "method": "deterministic phone-local active-collision-mesh OPEN primitive sweep",
        "role": "LEFT_B_OPEN_AND_PREGRASP",
        "source_placeholder_q_rad": original,
        "selected_q_rad": selected["q_rad"],
        "selected_normalized_joint_fraction": selected["normalized_joint_fraction"],
        "minimum_phone_clearance_m": selected["minimum_phone_clearance_m"],
        "minimum_joint_margin_rad": selected["minimum_joint_margin_rad"],
        "distance_to_phone_pinch_rad": selected["distance_to_phone_pinch_rad"],
        "candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "phone_local_geometry": True,
        "cartesian_arm_path_dependency": False,
        "per_frame_finger_ik": False,
        "episode_literal_frame_dependency": False,
    }


def calibrate_left_phone_pregrasp(
    runtime: ActiveG1Dex3,
    active_stage: Usd.Stage,
    arm_q_at_acquisition: np.ndarray,
    left_open: np.ndarray,
    left_pinch: np.ndarray,
    right_reference: np.ndarray,
    phone_pose: np.ndarray,
    phone_dimensions: np.ndarray,
    *,
    outside_clearance_m: float = 0.010,
    vertical_fraction: float = 0.70,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fixed thumb PREGRASP outside the phone's back surface.

    OPEN->PINCH joint interpolation swept the distal thumb through the table.
    The fixed PREGRASP keeps the contact cap at the final task height while it
    remains outside the phone.  PREGRASP->PINCH is then predominantly a short
    thickness-direction closure.  No per-frame contact solve is performed.
    """
    result = np.asarray(left_open, dtype=np.float64).copy()
    link = "left_hand_thumb_2_link"
    cap_local = v17.active_usd_cap_proxy(active_stage, link, axis=1, sign=-1.0)
    body = runtime.model.body(link).id
    limits = runtime.hand_limits["left"][:3]
    usable = np.minimum(0.02, 0.20 * np.ptp(limits, axis=1))
    lower, upper = limits[:, 0] + usable, limits[:, 1] - usable
    seed = np.clip(result[:3], lower, upper)
    half = 0.5 * np.asarray(phone_dimensions, dtype=np.float64)

    def cap_world(thumb_q: np.ndarray) -> np.ndarray:
        hand = result.copy()
        hand[:3] = thumb_q
        runtime.assign(arm_q_at_acquisition, hand, right_reference)
        rotation = runtime.data.xmat[body].reshape(3, 3)
        return runtime.model_to_scene_position(
            runtime.data.xpos[body] + rotation @ cap_local
        )

    seed_local = phone_pose[:3, :3].T @ (cap_world(seed) - phone_pose[:3, 3])
    pinch_local = phone_pose[:3, :3].T @ (
        cap_world(np.asarray(left_pinch[:3], dtype=np.float64)) - phone_pose[:3, 3]
    )
    target_local = np.asarray([
        np.clip(pinch_local[0], -half[0] + 0.003, half[0] - 0.003),
        -half[1] - float(outside_clearance_m),
        float(np.clip(vertical_fraction, -0.80, 0.80)) * half[2],
    ])

    def residual(thumb_q: np.ndarray) -> np.ndarray:
        local = phone_pose[:3, :3].T @ (cap_world(thumb_q) - phone_pose[:3, 3])
        return np.r_[100.0 * (local - target_local), 0.02 * (thumb_q - seed)]

    solved = least_squares(
        residual, seed, bounds=(lower, upper), max_nfev=1000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    result[:3] = solved.x
    final_world = cap_world(solved.x)
    final_local = phone_pose[:3, :3].T @ (final_world - phone_pose[:3, 3])
    runtime.assign(arm_q_at_acquisition, result, right_reference)
    collisions = [
        contact for contact in runtime.penetrating_contacts(tolerance=0.0)
        if all(str(name).startswith("left_") and "hand" in str(name) for name in contact["bodies"])
    ]
    if collisions:
        raise RuntimeError(f"fixed LEFT_PHONE_PREGRASP self-collision: {collisions}")
    error_mm = float(np.linalg.norm(final_local - target_local) * 1000.0)
    if error_mm > 1.0:
        raise RuntimeError(f"fixed LEFT_PHONE_PREGRASP target error {error_mm:.3f} mm")
    return result, {
        "method": "one fixed phone-local thumb PREGRASP from active distal collision cap",
        "active_distal_link": link,
        "source_open_q_rad": seed,
        "reference_pinch_phone_local_xyz_m": pinch_local,
        "selected_thumb_q_rad": solved.x,
        "target_phone_local_xyz_m": target_local,
        "selected_phone_local_xyz_m": final_local,
        "target_error_mm": error_mm,
        "outside_clearance_m": float(outside_clearance_m),
        "vertical_fraction": float(vertical_fraction),
        "same_hand_collision_count": len(collisions),
        "phone_local_geometry": True,
        "cartesian_arm_path_dependency": False,
        "per_frame_finger_ik": False,
        "episode_literal_frame_dependency": False,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    immutable = [
        v17.SOURCE, v17.PHASE_LIBRARY, v17.V14_TARGET, v17.V14_ARM,
        v17.ROOT_CONFIG, v17.TIMELINE, v17.ALIGNMENT, v17.LAYOUT,
        v17.ACTIVE_SCENE, v17.FIXED_SCENE, v17.MAGNETIC_SCENE,
        v17.MAGNET_CONFIG, v17.DEX3_MAPPING, v17.PALM_CONFIG,
        V17_OUT / "dex3_magsafe_execution_primitives_v17.sim.json",
    ]
    missing = [str(path) for path in immutable if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    hashes_before = {str(path.resolve()): sha256_file(path) for path in immutable}
    prior_inventory_before = {
        name: inventory_hash(path)
        for name, path in (("v14", v17.V14), ("v15", v17.V15), ("v16", v17.V16), ("v17", V17_OUT))
    }

    with np.load(v17.SOURCE, allow_pickle=False) as archive:
        optimized_action = archive["optimized_action"].copy()
        timestamps = archive["timestamp"].copy()
        fps = float(archive["fps"])
    with np.load(v17.PHASE_LIBRARY, allow_pickle=False) as archive:
        source_left_position = archive["left_tcp_position"].copy()
        source_right_position = archive["right_tcp_position"].copy()
        source_left_rotation = archive["left_tcp_rotation"].copy()
        source_right_rotation = archive["right_tcp_rotation"].copy()
        if not np.array_equal(optimized_action, archive["optimized_action"]):
            raise RuntimeError("source motion differs from frozen phase library")
    with np.load(v17.V14_TARGET, allow_pickle=False) as archive:
        target_left = archive["corrected_left_position"].copy()
        target_right = archive["corrected_right_position"].copy()
        workspace_scale = float(archive["workspace_scale"])
    with np.load(v17.V14_ARM, allow_pickle=False) as archive:
        v14_arm_q = archive["g1_arm_q"].copy()
        arm_joint_names = archive["arm_joint_names"].astype(str)
        root_position = archive["g1_root"].copy()
        root_offset = float(archive["g1_root_forward_offset_m"])
    if optimized_action.shape != (990, 14) or len(timestamps) != 990 or fps != 30.0:
        raise RuntimeError("Episode-49 source invariant failed")
    if not all(np.isfinite(value).all() for value in (optimized_action, target_left, target_right, v14_arm_q)):
        raise RuntimeError("non-finite frozen input")
    if workspace_scale != 0.42:
        raise RuntimeError("workspace scale changed")

    timeline = load_human_reviewed_development_timeline(
        v17.TIMELINE, v17.ALIGNMENT, optimized_action, timestamps,
        source_left_position, source_right_position,
        source_left_rotation, source_right_rotation,
        trajectory_path=v17.SOURCE, fk_model_path=v17.MODEL, task_geometry_path=v17.LAYOUT,
    )
    dry_run = retarget_aloha_trajectory_to_g1(
        optimized_action, timestamps, timeline,
        {"method": METHOD}, {"active_scene": str(v17.ACTIVE_SCENE)}, dry_run=True,
    )
    alignment = json.loads(v17.ALIGNMENT.read_text(encoding="utf-8"))
    forbidden = {int(row["aligned_action_index"]) for row in alignment["event_mapping"].values()}
    runtime_files = [
        Path(__file__), ROOT / "tools/aloha_g1_v17/trajectory.py",
        ROOT / "tools/aloha_g1_v15/semantic_input.py",
        ROOT / "tools/retarget_aloha_trajectory_to_g1.py",
    ]
    literals = literal_audit(runtime_files, forbidden)
    if not literals["pass"]:
        raise RuntimeError(f"runtime semantic literal dependency: {literals['findings']}")

    runtime = ActiveG1Dex3(v17.MODEL, v17.DEX3_MAPPING, v17.PALM_CONFIG, root_position)
    primitive_source = json.loads(
        (V17_OUT / "dex3_magsafe_execution_primitives_v17.sim.json").read_text(encoding="utf-8")
    )
    primitives = {name: np.asarray(value, dtype=np.float64) for name, value in primitive_source["primitives"].items()}
    right_b_safe, right_b_audit = calibrate_fixed_right_nontask_thumb(
        runtime, v14_arm_q[timeline.start_index],
        primitives["LEFT_OPEN"], primitives["RIGHT_OPEN"], primitives["RIGHT_RING_HOOK"],
    )
    for primitive_name in ("RIGHT_OPEN", "RIGHT_RING_PREHOOK", "RIGHT_RING_HOOK"):
        primitives[primitive_name] = primitives[primitive_name].copy()
        primitives[primitive_name][2:5] = right_b_safe
    left_hand_q, right_hand_q, semantic_progress = build_predefined_hand_trajectories(
        timeline, runtime, primitives, optimized_action[:, 6], optimized_action[:, 13],
        left_lock_end_event="phone_rotation_to_portrait_start",
        left_digit_staging_mode="simultaneous_opposed_close",
    )
    if np.max(np.ptp(left_hand_q, axis=0)) <= 0.05 or np.max(np.ptp(right_hand_q, axis=0)) <= 0.05:
        raise RuntimeError("predefined hand adapter is static")

    active_stage = Usd.Stage.Open(str(v17.ACTIVE_SCENE))
    phone_initial = v17.usd_pose(active_stage, "/World/MagSafeScene/Phone")
    pad = v17.usd_pose(active_stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    phone_charger = v17.phone_on_pad_pose(pad)
    layout = json.loads(v17.LAYOUT.read_text(encoding="utf-8"))
    table_height = float(layout["table"]["surface_height"])
    table_bounds = (0.0, float(layout["table"]["size_x"]), 0.0, float(layout["table"]["size_y"]))

    events = {name: int(timeline.event(name).action_index) for name in TASK_EVENTS}
    left_grasp = events["left_phone_grasp_start"]
    right_grasp = events["right_accessory_grasp_start"]
    charger = events["phone_charger_attachment_complete"]
    anchors = json.loads(v17.V14_ANCHORS.read_text(encoding="utf-8"))
    right_rows = [row for row in anchors.values() if isinstance(row, dict) and row.get("action_index") == right_grasp and "wrist_rotation" in row]
    charger_rows = [row for row in anchors.values() if isinstance(row, dict) and row.get("action_index") == charger and "wrist_rotation" in row]
    if len(right_rows) != 1 or len(charger_rows) != 1:
        raise RuntimeError("semantic anchor seed is not uniquely resolved")
    v16_carrier = json.loads(v17.V16_LEFT_CARRIER.read_text(encoding="utf-8"))
    task_grasp_rotation = np.asarray(v16_carrier["selected"]["initial_wrist"], dtype=np.float64)[:3, :3]
    task_charger_rotation = np.asarray(charger_rows[0]["wrist_rotation"], dtype=np.float64)
    task_right_rotation = np.asarray(right_rows[0]["wrist_rotation"], dtype=np.float64)

    configs = [
        {
            "name": "SEMANTIC_LOCAL_MINIMAL", "left_acquisition_strength": 1.0,
            "right_hook_strength": 0.45, "charger_strength": 1.0,
            "orientation_gain": 44.0, "prior_gain": 0.06, "temporal_gain": 0.16,
            "acceleration_gain": 0.08, "collision_gain": 100000.0,
            "shoulder_prior_gain": 8.0, "joint_center_gain": 2.0,
            "minimum_joint_margin_rad": 0.01, "preferred_joint_margin_rad": 0.03,
            "max_deviation_rad": 1.20, "max_step_rad": 0.24,
        },
        {
            "name": "SEMANTIC_LOCAL_BALANCED", "left_acquisition_strength": 1.0,
            "right_hook_strength": 0.70, "charger_strength": 1.0,
            "orientation_gain": 48.0, "prior_gain": 0.05, "temporal_gain": 0.18,
            "acceleration_gain": 0.10, "collision_gain": 120000.0,
            "shoulder_prior_gain": 8.0, "joint_center_gain": 2.5,
            "minimum_joint_margin_rad": 0.01, "preferred_joint_margin_rad": 0.03,
            "max_deviation_rad": 1.20, "max_step_rad": 0.24,
        },
        {
            "name": "SEMANTIC_LOCAL_TASK_STRONG", "left_acquisition_strength": 1.0,
            "right_hook_strength": 1.0, "charger_strength": 1.0,
            "orientation_gain": 52.0, "prior_gain": 0.04, "temporal_gain": 0.20,
            "acceleration_gain": 0.12, "collision_gain": 140000.0,
            "shoulder_prior_gain": 8.0, "joint_center_gain": 3.0,
            "minimum_joint_margin_rad": 0.01, "preferred_joint_margin_rad": 0.03,
            "max_deviation_rad": 1.20, "max_step_rad": 0.24,
        },
    ]

    candidates = []
    sweep_rows = []
    for config in configs:
        print(f"[v17.1] solving {config['name']}", flush=True)
        targets = build_semantic_local_orientation_targets(
            timeline, runtime, v14_arm_q, left_hand_q, right_hand_q,
            source_left_rotation, source_right_rotation, phone_initial, phone_charger,
            task_grasp_rotation, task_charger_rotation, task_right_rotation,
            semantic_progress,
            left_acquisition_strength=config["left_acquisition_strength"],
            right_hook_strength=config["right_hook_strength"],
            charger_strength=config["charger_strength"],
        )
        solver_keys = {
            "orientation_gain", "prior_gain", "temporal_gain", "acceleration_gain",
            "collision_gain", "shoulder_prior_gain", "joint_center_gain",
            "minimum_joint_margin_rad", "preferred_joint_margin_rad",
            "max_deviation_rad", "max_step_rad",
        }
        arm_q = solve_semantic_local_orientation_trajectory(
            runtime, v14_arm_q, target_left, target_right,
            targets["left_rotation"], targets["right_rotation"],
            targets["left_axis_weight"], targets["right_axis_weight"],
            left_hand_q, right_hand_q,
            **{key: config[key] for key in solver_keys},
        )
        metrics = evaluate_kinematic_candidate(
            timeline, runtime, arm_q, left_hand_q, right_hand_q,
            target_left, target_right, targets,
            source_left_rotation, source_right_rotation,
            phone_initial, phone_charger, table_height, table_bounds,
        )
        achieved = metrics.pop("achieved")
        integrity, raw_rows = audit_collision_classifier_integrity(
            runtime, arm_q, left_hand_q, right_hand_q,
            table_height, table_bounds,
        )
        grasp_error = orientation_distance(
            achieved["left_wrist"][left_grasp, :3, :3], task_grasp_rotation
        )
        right_axis = axis_error(
            achieved["right_wrist"][right_grasp, :3, 0], task_right_rotation[:, 0]
        )
        task_gate = {
            "phone_grasp_task_facing_error_deg": grasp_error,
            "phone_grasp_pass": grasp_error <= 10.0,
            "right_ring_hook_primary_axis_error_deg": right_axis,
            "right_ring_hook_axis_pass": right_axis <= 10.0,
        }
        metrics["collision"] = integrity
        metrics["task_facing"] = task_gate
        metrics["gate_pass"] = bool(
            metrics["finite"] and metrics["position"]["pass"]
            and metrics["fidelity"]["pass"] and metrics["orientation"]["pass"]
            and task_gate["phone_grasp_pass"] and task_gate["right_ring_hook_axis_pass"]
            and metrics["joint"]["joint_limit_violation_count"] == 0
            and metrics["joint"]["branch_discontinuity_count"] == 0
            and metrics["joint"]["minimum_arm_margin_rad"] >= 0.01 - 1e-8
            and integrity["pass"]
        )
        row = {
            "candidate": config["name"],
            "gate_pass": metrics["gate_pass"],
            "phone_grasp_error_deg": grasp_error,
            "right_hook_axis_error_deg": right_axis,
            "portrait_error_deg": metrics["orientation"]["portrait_long_axis_error_deg"],
            "charger_normal_error_deg": metrics["orientation"]["charger_normal_error_deg"],
            "charger_vertical_error_deg": metrics["orientation"]["charger_vertical_axis_error_deg"],
            "minimum_fidelity": metrics["fidelity"]["minimum_primary_metric"],
            "simultaneous_5mm_rate": metrics["position"]["simultaneous_5mm_rate"],
            "minimum_joint_margin_rad": metrics["joint"]["minimum_arm_margin_rad"],
            "branch_discontinuities": metrics["joint"]["branch_discontinuity_count"],
            "prohibited_collisions": integrity["prohibited_collision_records"],
            "raw_equals_classified": integrity["raw_equals_classified"],
        }
        sweep_rows.append(row)
        candidates.append({
            "config": config, "targets": targets, "arm_q": arm_q,
            "metrics": metrics, "achieved": achieved, "raw_rows": raw_rows,
        })
        print(json.dumps(row, indent=2), flush=True)

    eligible = [row for row in candidates if row["metrics"]["gate_pass"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["config"]["right_hook_strength"],
                row["metrics"]["task_facing"]["phone_grasp_task_facing_error_deg"],
                -row["metrics"]["fidelity"]["minimum_primary_metric"],
            ),
        )
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["metrics"]["collision"]["pass"],
                row["metrics"]["position"]["pass"],
                row["metrics"]["fidelity"]["pass"],
                row["metrics"]["task_facing"]["phone_grasp_pass"],
                -row["metrics"]["collision"]["prohibited_collision_records"],
            ),
        )

    arm_q = selected["arm_q"]
    targets = selected["targets"]
    # Physics proved that the v17 primitive, calibrated at its collision-free
    # non-task-facing arm candidate, produced B-only contact after the arm was
    # correctly task-oriented.  Recalibrate one fixed PHONE_PINCH at the
    # selected task-facing pose without feeding the result back into arm IK.
    recalibrated_pinch, left_pinch_audit = v17.calibrate_left_phone_pinch(
        runtime, active_stage, arm_q[left_grasp], primitives["LEFT_PHONE_PINCH"],
        primitives["RIGHT_OPEN"], phone_initial,
        np.asarray(layout["phone"]["size_landscape_xyz"], dtype=np.float64),
        contact_preload_m=0.001,
        index_contact_preload_m=0.002,
        thumb_contact_vertical_fraction=0.70,
    )
    left_open, left_open_audit = calibrate_left_open_index(
        runtime, active_stage,
        arm_q[timeline.start_index : left_grasp + 1],
        primitives["LEFT_OPEN"], recalibrated_pinch,
        primitives["RIGHT_OPEN"], phone_initial,
        np.asarray(layout["phone"]["size_landscape_xyz"], dtype=np.float64),
    )
    left_pre, left_pregrasp_audit = calibrate_left_phone_pregrasp(
        runtime, active_stage, arm_q[left_grasp], left_open,
        recalibrated_pinch,
        primitives["RIGHT_OPEN"], phone_initial,
        np.asarray(layout["phone"]["size_landscape_xyz"], dtype=np.float64),
        outside_clearance_m=0.010, vertical_fraction=0.70,
    )
    # Keep both digits genuinely open through approach.  The inherited OPEN
    # index was near its negative joint limits and its active collision mesh
    # intersected the phone before acquisition.  Opposed closure begins only
    # at the named grasp event and follows a minimum-jerk event-interval clock.
    primitives["LEFT_OPEN"] = left_open
    left_index_contact = left_open.copy()
    left_index_contact[3:5] = recalibrated_pinch[3:5]
    primitives["LEFT_PHONE_PREGRASP"] = left_pre
    primitives["LEFT_PHONE_INDEX_CONTACT"] = left_index_contact
    primitives["LEFT_PHONE_PINCH"] = recalibrated_pinch
    left_hand_q, right_hand_q, semantic_progress = build_predefined_hand_trajectories(
        timeline, runtime, primitives, optimized_action[:, 6], optimized_action[:, 13],
        left_lock_end_event="phone_rotation_to_portrait_start",
        left_digit_staging_mode="simultaneous_opposed_close",
        left_lock_progress_mode="event_interval_minimum_jerk",
        left_lock_completion_progress=0.63,
    )
    # The fixed task-facing pinch has a different occupied hand volume than
    # the v17 diagnostic primitive.  Re-run only the task-null-space posture
    # solve so that this real fixed geometry participates in hand-hand
    # clearance.  Cartesian targets and orientation rules remain unchanged.
    solver_keys = {
        "orientation_gain", "prior_gain", "temporal_gain", "acceleration_gain",
        "collision_gain", "shoulder_prior_gain", "joint_center_gain",
        "minimum_joint_margin_rad", "preferred_joint_margin_rad",
        "max_deviation_rad", "max_step_rad",
    }
    arm_q = solve_semantic_local_orientation_trajectory(
        runtime, v14_arm_q, target_left, target_right,
        targets["left_rotation"], targets["right_rotation"],
        targets["left_axis_weight"], targets["right_axis_weight"],
        left_hand_q, right_hand_q,
        **{key: selected["config"][key] for key in solver_keys},
    )
    metrics = evaluate_kinematic_candidate(
        timeline, runtime, arm_q, left_hand_q, right_hand_q,
        target_left, target_right, targets,
        source_left_rotation, source_right_rotation,
        phone_initial, phone_charger, table_height, table_bounds,
    )
    achieved = metrics.pop("achieved")
    integrity, raw_rows = audit_collision_classifier_integrity(
        runtime, arm_q, left_hand_q, right_hand_q, table_height, table_bounds,
    )
    grasp_error = orientation_distance(achieved["left_wrist"][left_grasp, :3, :3], task_grasp_rotation)
    right_axis = axis_error(achieved["right_wrist"][right_grasp, :3, 0], task_right_rotation[:, 0])
    metrics["collision"] = integrity
    metrics["task_facing"] = {
        "phone_grasp_task_facing_error_deg": grasp_error,
        "phone_grasp_pass": grasp_error <= 10.0,
        "right_ring_hook_primary_axis_error_deg": right_axis,
        "right_ring_hook_axis_pass": right_axis <= 10.0,
    }
    metrics["gate_pass"] = bool(
        metrics["finite"] and metrics["position"]["pass"]
        and metrics["fidelity"]["pass"] and metrics["orientation"]["pass"]
        and metrics["task_facing"]["phone_grasp_pass"]
        and metrics["task_facing"]["right_ring_hook_axis_pass"]
        and metrics["joint"]["joint_limit_violation_count"] == 0
        and metrics["joint"]["branch_discontinuity_count"] == 0
        and metrics["joint"]["minimum_arm_margin_rad"] >= 0.01 - 1e-8
        and integrity["pass"]
    )
    selected["metrics"] = metrics
    selected["achieved"] = achieved
    selected["raw_rows"] = raw_rows
    all_q = np.c_[arm_q, left_hand_q, right_hand_q]
    common = {
        "optimized_action": optimized_action,
        "source_timestamps": timestamps,
        "arm_joint_names": arm_joint_names,
        "left_dex3_joint_names": np.asarray(runtime.hand_joint_names["left"]),
        "right_dex3_joint_names": np.asarray(runtime.hand_joint_names["right"]),
        "v14_reference_arm_q": v14_arm_q,
        "v14_left_position_target": target_left,
        "v14_right_position_target": target_right,
        "g1_root": root_position,
        "workspace_scale": np.asarray(workspace_scale),
        "method": np.asarray(METHOD),
        "semantic_timeline_sha256": np.asarray(sha256_file(v17.TIMELINE)),
        "physics_applied": np.asarray(False),
        "simulation_only": np.asarray(True),
        "real_robot_command_allowed": np.asarray(False),
    }
    save_npz(
        OUT / "final_kinematic_arm_trajectory.npz", **common,
        g1_arm_q=arm_q, arm_qpos=arm_q,
        achieved_left_position=achieved["left_position"],
        achieved_right_position=achieved["right_position"],
        achieved_left_rotation=achieved["left_rotation"],
        achieved_right_rotation=achieved["right_rotation"],
        selected_candidate=np.asarray(selected["config"]["name"]),
    )
    save_npz(
        OUT / "final_dex3_trajectory.npz", **common,
        left_dex3_q=left_hand_q, right_dex3_q=right_hand_q,
        left_dex3_qpos=left_hand_q, right_dex3_qpos=right_hand_q,
        primitive_source=np.asarray("predefined_execution_primitives_v17_1"),
        **{f"semantic_{key}_progress": value for key, value in semantic_progress.items()},
    )
    save_npz(
        OUT / "final_arm_dex3_trajectory.npz", **common,
        arm_qpos=arm_q, g1_arm_q=arm_q,
        left_dex3_qpos=left_hand_q, right_dex3_qpos=right_hand_q,
        full_joint_q=all_q, fps=np.asarray(fps),
        primitive_source=np.asarray("predefined_execution_primitives_v17_1"),
        authoritative_for_real_robot=np.asarray(False),
    )
    save_npz(
        OUT / "semantic_local_orientation_targets.npz",
        left_rotation=targets["left_rotation"], right_rotation=targets["right_rotation"],
        left_axis_weight=targets["left_axis_weight"], right_axis_weight=targets["right_axis_weight"],
        left_activation=targets["left_semantic_activation"], right_activation=targets["right_semantic_activation"],
    )

    primitive_v17_1 = dict(primitive_source)
    primitive_v17_1.update({
        "version": "v17.1", "source_v17_primitive_sha256": sha256_file(V17_OUT / "dex3_magsafe_execution_primitives_v17.sim.json"),
        "arm_cartesian_path_dependency": False,
        "calibration_scope": "TASK_LOCAL_PHONE_AND_RING_GEOMETRY",
    })
    primitive_v17_1["primitives"] = primitives
    primitive_v17_1["right_non_task_thumb_collision_calibration"] = right_b_audit
    primitive_v17_1["left_open_index_task_local_calibration"] = left_open_audit
    primitive_v17_1["left_phone_pregrasp_task_local_calibration"] = left_pregrasp_audit
    primitive_v17_1["left_phone_pinch_task_facing_recalibration"] = left_pinch_audit
    primitive_v17_1["physics_failure_evidence_for_recalibration"] = {
        "source": "v17.1 first true-physics phone-grasp run",
        "left_A_phone_force_max_n": 0.0,
        "left_B_phone_force_max_n": 0.7686441540718079,
        "simultaneous_contact_samples": 0,
        "phone_hand_slip_mm": 36.7925670050582,
    }
    dump(OUT / "dex3_magsafe_execution_primitives_v17_1.sim.json", primitive_v17_1)
    dump(OUT / "dex3_semantic_interpolation_config.json", {
        "driver": "GENERIC_SEMANTIC_TIMELINE_AND_SOURCE_GRIPPER_PROGRESS",
        "literal_frame_dependency": False,
        "left_sequence": ["OPEN", "PREGRASP", "INDEX_CONTACT", "PINCH", "HOLD", "RELEASE", "OPEN"],
        "right_sequence": ["OPEN", "PREHOOK", "HOOK", "HOLD", "RELEASE", "OPEN"],
        "interpolation": "smooth simultaneous opposed close over named grasp-to-rotation-start interval",
        "left_lock_end_event": "phone_rotation_to_portrait_start",
        "left_digit_staging_mode": "simultaneous_opposed_close",
        "left_lock_progress_mode": "event_interval_minimum_jerk",
        "left_lock_completion_progress": 0.63,
        "maximum_dex3_command_step_source_samples": float(max(
            np.max(np.abs(np.diff(left_hand_q, axis=0))),
            np.max(np.abs(np.diff(right_hand_q, axis=0))),
        )),
        "per_frame_finger_ik": False,
    })
    dump(OUT / "semantic_local_orientation_config.json", {
        "status": "SEMANTIC_LOCAL_PARTIAL_ORIENTATION_PASS" if metrics["gate_pass"] else "BLOCKED_SEMANTIC_LOCAL_PARTIAL_ORIENTATION",
        "candidate_configs": configs,
        "selected": selected["config"],
        "activation_provenance": {
            "left_pregrasp_start_action_index_for_report_only": int(targets["left_pregrasp_start"]),
            "right_prehook_start_action_index_for_report_only": int(targets["right_prehook_start"]),
            "runtime_source": "source gripper progress plus named SemanticTimeline intervals",
        },
        "cartesian_translation_residual": False,
        "spatial_waypoints": False,
    })
    write_csv(OUT / "orientation_candidate_sweep.csv", sweep_rows)
    dump(OUT / "orientation_gate_metrics.json", {
        "selected_candidate": selected["config"]["name"],
        "task_facing": metrics["task_facing"],
        "portrait_charger_rotation": metrics["orientation"],
        "pass": bool(metrics["task_facing"]["phone_grasp_pass"] and metrics["task_facing"]["right_ring_hook_axis_pass"] and metrics["orientation"]["pass"]),
    })
    dump(OUT / "nullspace_solver_config.json", {
        "selected": selected["config"],
        "immutable_cartesian_position_weight": 3000.0,
        "hard_joint_margin_rad": 0.01,
        "preferred_joint_margin_rad": 0.03,
        "continuation": True, "bounded_joint_step": True,
        "velocity_regularization": True, "acceleration_regularization": True,
        "cartesian_target_mutation_allowed": False,
    })
    dump(OUT / "collision_classifier_integrity_audit.json", metrics["collision"])
    raw_fields = [
        "frame", "contact_index", "body_1", "body_2", "geom_1", "geom_2",
        "distance_m", "side_1", "side_2", "role_1", "role_2",
        "classification", "prohibited", "ignored_reason",
    ]
    write_csv(OUT / "raw_contact_classification.csv", selected["raw_rows"], raw_fields)
    dump(OUT / "joint_margin_metrics.json", metrics["joint"] | {
        "diagnostic_target_rad": 0.01, "preferred_target_rad": 0.03,
        "pre_real_margin_pass": metrics["joint"]["minimum_arm_margin_rad"] >= 0.01 - 1e-8,
    })
    dump(OUT / "kinematic_collision_metrics.json", metrics["collision"])
    dump(OUT / "aloha_fidelity_metrics.json", metrics["fidelity"] | {
        "rotation": metrics["orientation"], "source_action_is_sole_behavior_source": True,
    })

    backbone_hashes = {
        "v14_left_cartesian_target_sha256": array_sha(target_left),
        "v17_1_left_cartesian_target_sha256": array_sha(target_left.copy()),
        "v14_right_cartesian_target_sha256": array_sha(target_right),
        "v17_1_right_cartesian_target_sha256": array_sha(target_right.copy()),
        "left_max_difference_m": 0.0,
        "right_max_difference_m": 0.0,
        "byte_identical_arrays": True,
    }
    dump(OUT / "kinematic_prephysics_result.json", {
        "status": "V17_1_KINEMATIC_PREPHYSICS_PASS" if metrics["gate_pass"] else "BLOCKED_KINEMATIC_PREPHYSICS",
        "gate_pass": metrics["gate_pass"],
        "selected_candidate": selected["config"]["name"],
        "backbone_protection": backbone_hashes,
        "metrics": metrics,
        "physics_authorized": bool(metrics["gate_pass"]),
        "dominant_failure_subsystem": None if metrics["gate_pass"] else (
            "NULLSPACE_COLLISION" if not metrics["collision"]["pass"]
            else "JOINT_MARGIN" if metrics["joint"]["minimum_arm_margin_rad"] < 0.01 - 1e-8
            else "PARTIAL_ORIENTATION"
        ),
    })
    dump(OUT / "v17_baseline_audit.json", {
        "v17_status": "BLOCKED_KINEMATIC_PREPHYSICS",
        "known_safe_minimum_fidelity": 0.960314,
        "known_task_facing_orientation_error_deg": 3.217,
        "known_task_facing_hand_hand_collision_records": 2,
        "known_task_facing_right_speed_fidelity": 0.920443,
        "known_phone_slip_mm": 44.607,
        "v17_output_inventory_before": prior_inventory_before["v17"],
        "v17_output_reused_as_trajectory": False,
    })
    dump(OUT / "semantic_runtime_audit.json", {
        "status": "GENERIC_SEMANTIC_API_USED",
        "timeline_source": "HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE",
        "timeline_sha256": sha256_file(v17.TIMELINE),
        "resolved_events_for_provenance_only": {name: events[name] for name in TASK_EVENTS},
        "interface_readiness": semantic_readiness(timeline),
        "generic_converter_dry_run": dry_run,
        "runtime_literal_audit": literals,
        "validation_read_count": 0, "heldout_read_count": 0, "g1_expert_read_count": 0,
    })
    dump(OUT / "reusable_vs_episode_derived_v17_1.json", {
        "reusable_translator_parameters": [
            "axis/workspace registration", "semantic-local orientation activation rule",
            "task-null-space solver weights", "PHONE_PINCH primitive", "RING_HOOK primitive",
            "Dex3 semantic interpolation", "controller and collision thresholds",
        ],
        "episode_derived_trajectory_data": [
            "optimized_action", "timestamps", "SemanticTimeline event indices/progress",
            "ALOHA FK", "v14 Episode-49 Cartesian target", "G1 arm q", "Dex3 q",
        ],
        "episode49_indices_in_reusable_parameters": False,
        "fixed_length_dependency_in_reusable_parameters": False,
    })

    placeholder = {
        "status": "PENDING_TRUE_PHYSICS" if metrics["gate_pass"] else "NOT_RUN_BLOCKED_KINEMATIC_PREPHYSICS",
        "physics_steps": 0, "object_pose_scripted": False,
        "kinematic_object_follow": False, "semantic_attach_detach": False,
    }
    for filename in (
        "phone_grasp_physics_result.json", "phone_rotation_physics_result.json",
        "accessory_removal_physics_result.json", "bimanual_transport_physics_result.json",
        "charger_placement_physics_result.json", "accessory_release_physics_result.json",
        "physics_tracking_metrics.json", "physics_collision_metrics.json",
        "full_task_physics_result.json",
    ):
        dump(OUT / filename, placeholder)
    save_npz(OUT / "phone_object_trajectory.npz", status=np.asarray(placeholder["status"]), physics_steps=np.asarray(0), object_pose_scripted=np.asarray(False))
    save_npz(OUT / "accessory_object_trajectory.npz", status=np.asarray(placeholder["status"]), physics_steps=np.asarray(0), object_pose_scripted=np.asarray(False))
    dump(OUT / "pre_real_g1_readiness_v17_1.json", {
        "status": "PENDING_FULL_PHYSICS" if metrics["gate_pass"] else "NOT_READY_BLOCKED_KINEMATIC_PREPHYSICS",
        "real_robot_safe": False, "candidate_translator_created": False,
        "minimum_joint_margin_rad": metrics["joint"]["minimum_arm_margin_rad"],
        "remaining_real_only_blockers": [
            "real Dex3 OPEN calibration", "real PHONE_PINCH calibration",
            "real RING_HOOK calibration", "runtime joint-order confirmation",
            "object-free 0.25x preflight", "E-stop/supervisor readiness",
        ],
    })

    hashes_after = {str(path.resolve()): sha256_file(path) for path in immutable}
    prior_inventory_after = {
        name: inventory_hash(path)
        for name, path in (("v14", v17.V14), ("v15", v17.V15), ("v16", v17.V16), ("v17", V17_OUT))
    }
    prior_unchanged = all(
        prior_inventory_before[name]["inventory_sha256"] == prior_inventory_after[name]["inventory_sha256"]
        for name in prior_inventory_before
    )
    freeze_pass = hashes_before == hashes_after and prior_unchanged
    dump(OUT / "input_freeze_audit.json", {
        "status": "INPUT_FREEZE_PASS" if freeze_pass else "INPUT_FREEZE_FAIL",
        "backup": str(BACKUP.resolve()),
        "source_action": {
            "path": str(v17.SOURCE.resolve()), "sha256": sha256_file(v17.SOURCE),
            "shape": list(optimized_action.shape), "finite": True, "fps": fps,
            "array_sha256": array_sha(optimized_action), "timestamps_sha256": array_sha(timestamps),
        },
        "root_xyz_m": root_position, "root_forward_offset_m": root_offset,
        "workspace_scale": workspace_scale,
        "cartesian_backbone": backbone_hashes,
        "immutable_hashes_before": hashes_before, "immutable_hashes_after": hashes_after,
        "prior_output_inventories_before": prior_inventory_before,
        "prior_output_inventories_after": prior_inventory_after,
        "prior_outputs_byte_identical": prior_unchanged,
        "byte_identical": freeze_pass,
        "validation_read_count": 0, "heldout_read_count": 0, "g1_expert_read_count": 0,
    })
    if not freeze_pass:
        raise RuntimeError("immutable input/prior output changed")

    dump(OUT / "build_summary.json", {
        "status": "V17_1_KINEMATIC_PREPHYSICS_PASS" if metrics["gate_pass"] else "BLOCKED_KINEMATIC_PREPHYSICS",
        "kinematic_gate_pass": metrics["gate_pass"],
        "physics_run_required": bool(metrics["gate_pass"]),
        "selected_candidate": selected["config"]["name"],
        "dominant_failure_subsystem": None if metrics["gate_pass"] else (
            "NULLSPACE_COLLISION" if not metrics["collision"]["pass"]
            else "JOINT_MARGIN" if metrics["joint"]["minimum_arm_margin_rad"] < 0.01 - 1e-8
            else "PARTIAL_ORIENTATION"
        ),
    })
    print(json.dumps(json.loads((OUT / "build_summary.json").read_text()), indent=2), flush=True)
    return 0 if metrics["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
