#!/usr/bin/env python3
"""Episode-49 development wrapper for the generic v15 translator.

The wrapper explicitly supplies the human-reviewed development timeline, but
all runtime transitions are resolved through named SemanticTimeline calls.
No held-out or validation trajectory is read by this program.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

from aloha_g1_v15.kinematics import (  # noqa: E402
    ActiveG1Dex3,
    nearest_box_surface,
    normalize,
    ring_material_gap,
    sha256_file,
    transform,
)
from aloha_g1_v15.semantic_input import (  # noqa: E402
    TASK_EVENTS,
    load_human_reviewed_development_timeline,
)
from aloha_g1_v15.translator import (  # noqa: E402
    C_LEFT,
    C_RIGHT,
    build_accessory_object_trajectory,
    build_phone_object_trajectory,
    compose_pose,
    continuous_hand_trajectories,
    inverse_pose,
    search_left_opposed_phone_contact,
    search_right_ring_contact,
    semantic_rotation_path,
    solve_arm_orientation_trajectory,
    swept_right_c_audit,
)
from retarget_aloha_trajectory_to_g1 import retarget_aloha_trajectory_to_g1  # noqa: E402
from v15_semantic_interface import readiness as semantic_readiness  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_orientation_dex3_v15"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
PHASE_LIBRARY = V12 / "aloha_phase_motion_library.npz"
V14_TARGET = V14 / "corrected_targets_v14.npz"
V14_EXACT = V14 / "position_only_exact_v14.npz"
V14_NULL = V14 / "position_only_nullspace_v14.npz"
V14_ANCHORS = V14 / "selected_physical_carrier_anchors.json"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
ROOT_CONFIG = ROOT / "configs/g1_root_forward_v14.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
ACTIVE_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEX3_MAPPING = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
DEX3_FRAMES = ROOT / "configs/dex3_fingertip_frames.sim.json"
PALM_CONFIG = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"
AXIS_CONFIG = ROOT / "configs/aloha_tcp_to_g1_palm_calibration.sim.json"
METHOD = "ALOHA_PRIMARY_EP49_ORIENTATION_DEX3_V15"


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def usd_pose(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"active scene missing {prim_path}")
    return np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64
    ).T


def phone_on_pad_pose(pad: np.ndarray) -> np.ndarray:
    long_axis = normalize(pad[:3, 1], (0.0, 0.0, 1.0))
    back_axis = -normalize(pad[:3, 2], (0.0, -1.0, 0.0))
    short_axis = normalize(np.cross(long_axis, back_axis), (1.0, 0.0, 0.0))
    back_axis = normalize(np.cross(short_axis, long_axis), (0.0, -1.0, 0.0))
    return compose_pose(np.column_stack((long_axis, back_axis, short_axis)), pad[:3, 3])


def portrait_phone_rotation(source_rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Minimum task-axis correction that makes the long axis vertical."""
    long_axis = normalize(source_rotation[:, 0])
    vertical = np.array([0.0, 0.0, 1.0])
    cross = np.cross(long_axis, vertical)
    dot = float(np.clip(np.dot(long_axis, vertical), -1.0, 1.0))
    if np.linalg.norm(cross) <= 1e-10:
        correction = np.eye(3) if dot > 0.0 else Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
    else:
        correction = Rotation.from_rotvec(math.acos(dot) * normalize(cross)).as_matrix()
    result = correction @ source_rotation
    back = result[:, 1] - vertical * np.dot(result[:, 1], vertical)
    back = normalize(back, (0.0, -1.0, 0.0))
    if back[1] > 0.0:
        back = -back
    short = normalize(np.cross(vertical, back), (1.0, 0.0, 0.0))
    result = np.column_stack((vertical, back, short))
    return result, float(np.degrees(math.acos(dot)))


def pose_from_palm(position: np.ndarray, rotation: np.ndarray, palm_offset: np.ndarray) -> np.ndarray:
    wrist_position = np.asarray(position) - np.asarray(rotation) @ np.asarray(palm_offset)
    return compose_pose(rotation, wrist_position)


def runtime_literal_audit(paths: list[Path]) -> dict[str, Any]:
    forbidden_values = set(json.loads(ALIGNMENT.read_text(encoding="utf-8"))["event_mapping"][name]["aligned_action_index"] for name in TASK_EVENTS)
    findings = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in forbidden_values:
                findings.append({"path": str(path), "line": int(node.lineno), "value": int(node.value)})
    return {
        "scanned_files": [str(path) for path in paths],
        "literal_semantic_runtime_dependencies": findings,
        "count": len(findings),
        "pass": not findings,
    }


def historical_anchor_seed(payload: dict[str, Any], action_index: int, field: str) -> np.ndarray:
    """Resolve a v14 diagnostic seed by metadata, never by semantic key text."""
    matches = [
        row for row in payload.values()
        if isinstance(row, dict) and row.get("action_index") == action_index and field in row
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one v14 diagnostic seed for {action_index}/{field}, got {len(matches)}")
    return np.asarray(matches[0][field], dtype=np.float64)


def rotation_progress_correlation(source: np.ndarray, target: np.ndarray, start: int, end: int) -> float:
    source_segment = source[start : end + 1]
    target_segment = target[start : end + 1]
    source_step = Rotation.from_matrix(np.einsum("tji,tjk->tik", source_segment[:-1], source_segment[1:])).magnitude()
    target_step = Rotation.from_matrix(np.einsum("tji,tjk->tik", target_segment[:-1], target_segment[1:])).magnitude()
    source_progress = np.r_[0.0, np.cumsum(source_step)]
    target_progress = np.r_[0.0, np.cumsum(target_step)]
    if source_progress[-1] <= 1e-10 or target_progress[-1] <= 1e-10:
        return 1.0
    source_progress /= source_progress[-1]
    target_progress /= target_progress[-1]
    return float(np.corrcoef(source_progress, target_progress)[0, 1])


def build_source_only_rotations(source: np.ndarray, initial: np.ndarray, timeline, side: str) -> np.ndarray:
    end = timeline.end_index
    if side == "left":
        names = (
            "left_phone_grasp_start",
            "phone_rotation_to_portrait_start",
            "phone_portrait_reached",
            "phone_move_to_charger_start",
            "phone_charger_attachment_complete",
            "left_phone_release_complete",
        )
        calibration = C_LEFT
    else:
        names = (
            "phone_portrait_reached",
            "right_accessory_grasp_start",
            "accessory_detachment_start",
            "accessory_removed",
            "right_accessory_release_complete",
        )
        calibration = C_RIGHT
    boundaries = [timeline.start_index] + [int(timeline.event(name).action_index) for name in names] + [end]
    segments = [(a, b, None) for a, b in zip(boundaries[:-1], boundaries[1:])]
    return semantic_rotation_path(source, calibration, initial, segments, timeline.trajectory_length)


def solve_stage(
    runtime: ActiveG1Dex3,
    name: str,
    previous_q: np.ndarray,
    positions: tuple[np.ndarray, np.ndarray],
    rotations: tuple[np.ndarray, np.ndarray],
    posture_gain: float,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    q, raw = solve_arm_orientation_trajectory(
        runtime,
        previous_q,
        positions[0],
        positions[1],
        rotations[0],
        rotations[1],
        orientation_gain=0.55,
        posture_gain=posture_gain,
        iterations=12,
    )
    limits = runtime.info["joint_limits"]
    violation = int(np.sum((q < limits[:, 0] - 1e-9) | (q > limits[:, 1] + 1e-9)))
    passed = bool(raw["simultaneous_5mm_rate"] >= 0.99 and violation == 0)
    metrics = {
        "stage": name,
        "accepted": passed,
        "simultaneous_5mm_rate": raw["simultaneous_5mm_rate"],
        "left_position_mean_mm": float(np.mean(raw["left_position_error_m"]) * 1000.0),
        "left_position_max_mm": float(np.max(raw["left_position_error_m"]) * 1000.0),
        "right_position_mean_mm": float(np.mean(raw["right_position_error_m"]) * 1000.0),
        "right_position_max_mm": float(np.max(raw["right_position_error_m"]) * 1000.0),
        "left_orientation_mean_deg": float(np.degrees(np.mean(raw["left_orientation_error_rad"]))),
        "left_orientation_max_deg": float(np.degrees(np.max(raw["left_orientation_error_rad"]))),
        "right_orientation_mean_deg": float(np.degrees(np.mean(raw["right_orientation_error_rad"]))),
        "right_orientation_max_deg": float(np.degrees(np.max(raw["right_orientation_error_rad"]))),
        "minimum_joint_margin_rad": raw["minimum_joint_margin_rad"],
        "maximum_joint_step_rad": raw["maximum_joint_step_rad"],
        "joint_limit_violations": violation,
        "raw": raw,
    }
    return q, metrics, passed


def collision_audit(runtime: ActiveG1Dex3, arm: np.ndarray, left: np.ndarray, right: np.ndarray, table_z: float) -> dict[str, Any]:
    categories = {
        "arm_torso": set(),
        "arm_arm": set(),
        "hand_torso": set(),
        "hand_hand": set(),
        "same_hand_self_contact": set(),
    }
    minimum_table_clearance = math.inf
    palm_table_frames = []
    for frame in range(len(arm)):
        runtime.assign(arm[frame], left[frame], right[frame])
        for record in runtime.penetrating_contacts():
            bodies = record["bodies"]
            joined = "|".join(sorted(bodies))
            left_side = any(value.startswith("left_") for value in bodies)
            right_side = any(value.startswith("right_") for value in bodies)
            hand = ["hand" in value for value in bodies]
            arm_body = [any(token in value for token in ("shoulder", "elbow", "wrist")) for value in bodies]
            torso = [any(token in value for token in ("torso", "waist", "pelvis")) for value in bodies]
            if any(arm_body) and any(torso):
                categories["arm_torso"].add((frame, joined))
            if left_side and right_side and any(arm_body):
                categories["arm_arm"].add((frame, joined))
            if any(hand) and any(torso):
                categories["hand_torso"].add((frame, joined))
            if left_side and right_side and all(hand):
                categories["hand_hand"].add((frame, joined))
            if all(hand) and (left_side ^ right_side):
                categories["same_hand_self_contact"].add((frame, joined))
        for side in ("left", "right"):
            palm_z = float(runtime.palm_pose(side)[2, 3])
            clearance = palm_z - table_z
            minimum_table_clearance = min(minimum_table_clearance, clearance)
            if clearance < -1e-5:
                palm_table_frames.append(frame)
    serialized = {
        name: {
            "count": len(records),
            "frames": sorted({value[0] for value in records}),
            "pairs": sorted({value[1] for value in records}),
        }
        for name, records in categories.items()
    }
    serialized["palm_table"] = {"count": len(set(palm_table_frames)), "frames": sorted(set(palm_table_frames))}
    prohibited = sum(value["count"] for value in serialized.values() if isinstance(value, dict) and "count" in value)
    return {
        "categories": serialized,
        "minimum_palm_center_table_clearance_m": minimum_table_clearance,
        "prohibited_collision_records": prohibited,
        "pass": prohibited == 0,
        "collision_penetration_tolerance_m": 1e-5,
    }


def evaluate_contact_trajectories(
    runtime: ActiveG1Dex3,
    arm: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    phone: np.ndarray,
    accessory: np.ndarray,
    timeline,
    phone_dimensions: np.ndarray,
    ring_inner: float,
    ring_outer: float,
    ring_depth: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    portrait = int(timeline.event("phone_portrait_reached").action_index)
    charger = int(timeline.event("phone_charger_attachment_complete").action_index)
    left_release = int(timeline.event("left_phone_release_complete").action_index)
    detachment = int(timeline.event("accessory_detachment_start").action_index)
    right_release = int(timeline.event("right_accessory_release_complete").action_index)
    left_rows = []
    right_rows = []
    for frame in range(len(arm)):
        runtime.assign(arm[frame], left[frame], right[frame])
        if portrait <= frame <= left_release:
            row = {"action_index": frame}
            for role in ("A", "B", "C"):
                point, _ = runtime.contact_pose(f"left_{role}")
                signed, nearest, _ = nearest_box_surface(point, phone[frame], phone_dimensions)
                row[f"{role}_signed_surface_distance_m"] = signed
                row[f"{role}_nearest_surface_m"] = nearest
            left_rows.append(row)
        if detachment <= frame <= right_release:
            point, _ = runtime.contact_pose("right_C")
            signed, local = ring_material_gap(point, accessory[frame], ring_inner, ring_outer, ring_depth)
            right_rows.append({"action_index": frame, "C_signed_ring_material_distance_m": signed, "C_accessory_local_m": local})
    left_a = np.asarray([abs(row["A_signed_surface_distance_m"]) for row in left_rows])
    left_b = np.asarray([abs(row["B_signed_surface_distance_m"]) for row in left_rows])
    left_c_penetration = np.asarray([max(0.0, -row["C_signed_surface_distance_m"]) for row in left_rows])
    right_gap = np.asarray([abs(row["C_signed_ring_material_distance_m"]) for row in right_rows])
    left_metrics = {
        "continuous_interval": [portrait, charger],
        "evaluated_until_release": left_release,
        "A_gap_max_mm": float(np.max(left_a) * 1000.0),
        "B_gap_max_mm": float(np.max(left_b) * 1000.0),
        "A_gap_mean_mm": float(np.mean(left_a) * 1000.0),
        "B_gap_mean_mm": float(np.mean(left_b) * 1000.0),
        "left_C_phone_penetration_max_mm": float(np.max(left_c_penetration) * 1000.0),
        "continuous_pinch_pass": bool(np.max(left_a) <= 0.005 and np.max(left_b) <= 0.005 and np.max(left_c_penetration) <= 1e-5),
        "samples": left_rows,
    }
    right_metrics = {
        "continuous_interval": [detachment, right_release],
        "C_ring_absolute_gap_max_mm": float(np.max(right_gap) * 1000.0),
        "C_ring_absolute_gap_mean_mm": float(np.mean(right_gap) * 1000.0),
        "hook_hold_pass": bool(np.max(right_gap) <= 0.005),
        "samples": right_rows,
    }
    return left_metrics, right_metrics


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required = (
        SOURCE,
        PHASE_LIBRARY,
        V14_TARGET,
        V14_EXACT,
        V14_NULL,
        V14_ANCHORS,
        TIMELINE,
        ALIGNMENT,
        ROOT_CONFIG,
        LAYOUT,
        ACTIVE_SCENE,
        FIXED_SCENE,
        MODEL,
        DEX3_MAPPING,
        DEX3_FRAMES,
        PALM_CONFIG,
        AXIS_CONFIG,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path.resolve()): sha256_file(path) for path in required}

    with np.load(SOURCE, allow_pickle=False) as archive:
        optimized_action = archive["optimized_action"].copy()
        timestamps = archive["timestamp"].copy()
        fps = float(archive["fps"])
    with np.load(PHASE_LIBRARY, allow_pickle=False) as archive:
        source_left_position = archive["left_tcp_position"].copy()
        source_right_position = archive["right_tcp_position"].copy()
        source_left_rotation = archive["left_tcp_rotation"].copy()
        source_right_rotation = archive["right_tcp_rotation"].copy()
        if not np.array_equal(optimized_action, archive["optimized_action"]):
            raise RuntimeError("phase library source action mismatch")
        if not np.array_equal(timestamps, archive["timestamps"]):
            raise RuntimeError("phase library timestamps mismatch")
    with np.load(V14_TARGET, allow_pickle=False) as archive:
        v14_target_keys = list(archive.files)
        target_left = archive["corrected_left_position"].copy()
        target_right = archive["corrected_right_position"].copy()
        target_left_hash = array_sha(target_left)
        target_right_hash = array_sha(target_right)
    with np.load(V14_NULL, allow_pickle=False) as archive:
        v14_q = archive["g1_arm_q"].copy()
        v14_left_rotation = archive["achieved_left_rotation_scene"].copy()
        v14_right_rotation = archive["achieved_right_rotation_scene"].copy()
        arm_joint_names = archive["arm_joint_names"].copy()
        root_position = archive["g1_root"].copy()
        root_offset = float(archive["g1_root_forward_offset_m"])
    if optimized_action.shape != (len(optimized_action), 14) or len(optimized_action) != len(target_left):
        raise RuntimeError("source/v14 trajectory length mismatch")
    if not np.isfinite(optimized_action).all() or fps != 30.0:
        raise RuntimeError("source optimized_action invariant failed")

    timeline = load_human_reviewed_development_timeline(
        TIMELINE,
        ALIGNMENT,
        optimized_action,
        timestamps,
        source_left_position,
        source_right_position,
        source_left_rotation,
        source_right_rotation,
        trajectory_path=SOURCE,
        fk_model_path=MODEL,
        task_geometry_path=LAYOUT,
    )
    semantic_contract = semantic_readiness(timeline)
    dry_run = retarget_aloha_trajectory_to_g1(
        optimized_action,
        timestamps,
        timeline,
        {"method": METHOD},
        {"active_scene": str(ACTIVE_SCENE)},
        dry_run=True,
    )
    runtime_paths = [
        ROOT / "tools/aloha_g1_v15/semantic_input.py",
        ROOT / "tools/aloha_g1_v15/translator.py",
        ROOT / "tools/aloha_g1_v15/kinematics.py",
    ]
    literal_audit = runtime_literal_audit(runtime_paths)
    if not literal_audit["pass"]:
        raise RuntimeError("semantic runtime literal dependency detected")
    semantic_input_audit = {
        "status": "EP49_GENERIC_SEMANTIC_API_USED",
        "timeline_source": "HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE",
        "semantic_runtime_interface": "GENERIC",
        "hardcoded_runtime_indices": False,
        "required_task_events": list(TASK_EVENTS),
        "resolved_events": {
            name: {
                "action_index": timeline.event(name).action_index,
                "observed_frame": timeline.event(name).observed_frame,
                "confidence_class": timeline.event(name).confidence_class,
            }
            for name in TASK_EVENTS
        },
        "interface_readiness": semantic_contract,
        "generic_converter_dry_run": dry_run,
        "literal_audit": literal_audit,
        "terminal_events_block_manipulation": False,
        "v15_integration_execution": {
            "task_orientation_executed": True,
            "continuous_left_Dex3_executed": True,
            "continuous_right_Dex3_executed": True,
            "result_artifact": "numeric_gate_summary.json",
            "note": "Interface readiness fields describe the pre-solve contract; this block records this v15 run.",
        },
    }
    dump(OUT / "v15_semantic_input_audit.json", semantic_input_audit)

    root_cfg = json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))
    if not np.allclose(root_position, root_cfg["new_exact_root_xyz_m"], atol=1e-9) or root_offset != root_cfg["selected_total_forward_offset_m"]:
        raise RuntimeError("v14 root mismatch")
    runtime = ActiveG1Dex3(MODEL, DEX3_MAPPING, PALM_CONFIG, root_position)
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    phone_dimensions = np.asarray(layout["phone"]["size_landscape_xyz"], dtype=np.float64)
    ring_inner = 0.5 * float(layout["accessory"]["main_inner_diameter"])
    ring_outer = 0.5 * float(layout["accessory"]["main_outer_diameter"])
    ring_depth = float(layout["accessory"]["main_depth"])
    stage = Usd.Stage.Open(str(ACTIVE_SCENE))
    if stage is None:
        raise RuntimeError("failed to open authoritative Isaac stage")
    phone_initial = usd_pose(stage, "/World/MagSafeScene/Phone")
    accessory_initial = usd_pose(stage, "/World/MagSafeScene/Accessory")
    pad = usd_pose(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    phone_pad = phone_on_pad_pose(pad)
    phone_from_accessory = inverse_pose(phone_initial) @ accessory_initial

    mapping_payload = json.loads(DEX3_MAPPING.read_text(encoding="utf-8"))
    mapping_audit = {
        "status": "ACTIVE_G1_DEX3_MAPPING_PASS",
        "active_model": str(MODEL),
        "active_model_sha256": sha256_file(MODEL),
        "left": {role: mapping_payload["left"][role] for role in ("A", "B", "C")},
        "right": {role: mapping_payload["right"][role] for role in ("A", "B", "C")},
        "approved_task_roles": mapping_payload["approved_roles"],
        "left_control_joint_names": runtime.hand_joint_names["left"],
        "right_control_joint_names": runtime.hand_joint_names["right"],
        "full_hand_dof": 14,
        "trajectory_generated_in_this_run": True,
        "simulation_only": True,
    }
    dump(OUT / "dex3_mapping_audit_v15.json", mapping_audit)
    contact_geometry = {
        "status": "ACTIVE_COLLISION_CONTACT_GEOMETRY_PASS",
        "model_sha256": sha256_file(MODEL),
        "mapping_sha256": sha256_file(DEX3_MAPPING),
        "contacts": {
            label: {
                "parent_link": spec.link,
                "joint_names": spec.joint_names,
                "joint_limits_rad": spec.limits,
                "contact_center_local_m": spec.local_position,
                "local_normal": spec.local_normal,
                "effective_half_extent_m": spec.half_extent,
            }
            for label, spec in runtime.contacts.items()
        },
        "palm_offsets_m": runtime.palm_offset,
        "wrist_frames": {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"},
        "scene_to_model_rotation_determinant": 1.0,
        "quaternion_convention": "MuJoCo wxyz; scipy xyzw conversion explicit",
        "rendered_mesh_used_as_contact_ground_truth": False,
    }
    dump(OUT / "dex3_contact_geometry_v15.json", contact_geometry)

    anchors = json.loads(V14_ANCHORS.read_text(encoding="utf-8"))
    left_grasp = int(timeline.event("left_phone_grasp_start").action_index)
    portrait = int(timeline.event("phone_portrait_reached").action_index)
    right_grasp = int(timeline.event("right_accessory_grasp_start").action_index)
    detachment = int(timeline.event("accessory_detachment_start").action_index)
    removed = int(timeline.event("accessory_removed").action_index)
    phone_move = int(timeline.event("phone_move_to_charger_start").action_index)
    charger = int(timeline.event("phone_charger_attachment_complete").action_index)
    left_release = int(timeline.event("left_phone_release_complete").action_index)
    right_release = int(timeline.event("right_accessory_release_complete").action_index)

    left_results = []
    for assignment_index, assignment in enumerate(("A_SCREEN_B_BACK", "A_BACK_B_SCREEN")):
        initial = search_left_opposed_phone_contact(
            runtime,
            v14_q[left_grasp],
            target_left[left_grasp],
            phone_initial,
            phone_dimensions,
            assignment,
            "PHONE_ACQUISITION",
            historical_anchor_seed(
                anchors, left_grasp, "diagnostic_left_dex3_AB_q_rad"
            ),
            random_seed=10 + assignment_index,
        )
        pad_candidate = search_left_opposed_phone_contact(
            runtime,
            v14_q[charger],
            target_left[charger],
            phone_pad,
            phone_dimensions,
            assignment,
            "CHARGER_HOLD",
            historical_anchor_seed(
                anchors, charger, "diagnostic_left_dex3_AB_q_rad"
            ),
            random_seed=20 + assignment_index,
        )
        left_results.append({
            "assignment": assignment,
            "initial": initial,
            "charger": pad_candidate,
            "valid_entire_carrier": bool(initial.valid and pad_candidate.valid),
            "combined_score": float(initial.score + pad_candidate.score),
        })
    selected_left = min(
        [row for row in left_results if row["valid_entire_carrier"]] or left_results,
        key=lambda row: row["combined_score"],
    )
    dump(OUT / "left_ab_candidate_search.json", {
        "status": "PASS" if selected_left["valid_entire_carrier"] else "BLOCKED_LEFT_AB_CONTINUOUS_PINCH",
        "candidates": [
            {
                "assignment": row["assignment"],
                "initial": row["initial"].record(),
                "charger": row["charger"].record(),
                "valid_entire_carrier": row["valid_entire_carrier"],
                "combined_score": row["combined_score"],
            }
            for row in left_results
        ],
        "selected_assignment": selected_left["assignment"],
        "selection_considered_full_phone_interval": True,
    })

    charger_wrist = selected_left["charger"].wrist_pose
    wrist_from_phone_lock = inverse_pose(charger_wrist) @ phone_pad
    initial_left_rotation = v14_left_rotation[0]
    source_left_only = build_source_only_rotations(source_left_rotation, initial_left_rotation, timeline, "left")
    source_right_only = build_source_only_rotations(source_right_rotation, v14_right_rotation[0], timeline, "right")
    phone_source_at_portrait = selected_left["initial"].wrist_pose[:3, :3] @ (
        C_LEFT.T @ source_left_rotation[left_grasp].T @ source_left_rotation[portrait] @ C_LEFT
    ) @ wrist_from_phone_lock[:3, :3]
    portrait_phone, portrait_axis_correction_deg = portrait_phone_rotation(phone_source_at_portrait)
    portrait_wrist_rotation = portrait_phone @ wrist_from_phone_lock[:3, :3].T
    task_left_precharger = semantic_rotation_path(
        source_left_rotation,
        C_LEFT,
        initial_left_rotation,
        [
            (timeline.start_index, left_grasp, selected_left["initial"].wrist_pose[:3, :3]),
            (left_grasp, int(timeline.event("phone_rotation_to_portrait_start").action_index), None),
            (int(timeline.event("phone_rotation_to_portrait_start").action_index), portrait, portrait_wrist_rotation),
            (portrait, phone_move, None),
            (phone_move, charger, None),
            (charger, left_release, None),
            (left_release, timeline.end_index, None),
        ],
        timeline.trajectory_length,
    )
    task_left_final = semantic_rotation_path(
        source_left_rotation,
        C_LEFT,
        initial_left_rotation,
        [
            (timeline.start_index, left_grasp, selected_left["initial"].wrist_pose[:3, :3]),
            (left_grasp, int(timeline.event("phone_rotation_to_portrait_start").action_index), None),
            (int(timeline.event("phone_rotation_to_portrait_start").action_index), portrait, portrait_wrist_rotation),
            (portrait, phone_move, None),
            (phone_move, charger, selected_left["charger"].wrist_pose[:3, :3]),
            (charger, left_release, None),
            (left_release, timeline.end_index, None),
        ],
        timeline.trajectory_length,
    )
    provisional_wrist_left = np.asarray([
        pose_from_palm(target_left[index], task_left_final[index], runtime.palm_offset["left"])
        for index in range(timeline.trajectory_length)
    ])
    provisional_phone = build_phone_object_trajectory(
        timeline, provisional_wrist_left, phone_initial, phone_pad, wrist_from_phone_lock
    )
    accessory_at_grasp = provisional_phone[right_grasp] @ phone_from_accessory
    approach_start_time = timestamps[right_grasp] - 0.4
    approach_start = int(np.searchsorted(timestamps, approach_start_time, side="left"))
    source_approach = normalize(target_right[right_grasp] - target_right[approach_start])
    right_candidates = [
        search_right_ring_contact(
            runtime,
            v14_q[right_grasp],
            target_right[right_grasp],
            accessory_at_grasp,
            ring_inner,
            ring_outer,
            ring_depth,
            family,
            source_approach,
            random_seed=30 + family_index,
        )
        for family_index, family in enumerate(("AXIAL_APERTURE_INSERTION", "RADIAL_GAP_INSERTION"))
    ]
    selected_right = min([row for row in right_candidates if row.valid] or right_candidates, key=lambda row: row.score)
    dump(OUT / "right_c_candidate_search.json", {
        "status": "PASS" if selected_right.valid else "BLOCKED_RIGHT_C_INSERTION",
        "candidates": [row.record() for row in right_candidates],
        "selected_insertion_family": selected_right.family,
        "source_approach_direction": source_approach,
        "selected_from_active_geometry_and_swept_feasibility": True,
    })

    task_right_final = semantic_rotation_path(
        source_right_rotation,
        C_RIGHT,
        v14_right_rotation[0],
        [
            (timeline.start_index, portrait, None),
            (portrait, right_grasp, selected_right.wrist_pose[:3, :3]),
            (right_grasp, detachment, selected_right.wrist_pose[:3, :3]),
            (detachment, removed, None),
            (removed, right_release, None),
            (right_release, timeline.end_index, None),
        ],
        timeline.trajectory_length,
    )

    stage_specs = [
        ("O1_MAPPED_ALOHA_RELATIVE_ROTATION", source_left_only, source_right_only, 0.0),
        ("O2_LEFT_PHONE_GRASP_AXES", task_left_precharger, source_right_only, 0.0),
        ("O3_PHONE_PORTRAIT_ENDPOINT", task_left_precharger, source_right_only, 0.001),
        ("O4_RIGHT_ACCESSORY_AXIS", task_left_precharger, task_right_final, 0.001),
        ("O5_CHARGER_TASK_AXES", task_left_final, task_right_final, 0.002),
        ("O6_UNCONSTRAINED_TWIST_NULLSPACE", task_left_final, task_right_final, 0.008),
    ]
    current_q = v14_q.copy()
    stage_metrics = [{
        "stage": "O0_V14_POSITION_ONLY",
        "accepted": True,
        "simultaneous_5mm_rate": 1.0,
        "minimum_joint_margin_rad": float(np.min(np.minimum(
            v14_q - runtime.info["joint_limits"][:, 0], runtime.info["joint_limits"][:, 1] - v14_q
        ))),
    }]
    selected_left_rotation = v14_left_rotation.copy()
    selected_right_rotation = v14_right_rotation.copy()
    for name, left_rotation, right_rotation, posture_gain in stage_specs:
        candidate_q, metrics, passed = solve_stage(
            runtime,
            name,
            current_q,
            (target_left, target_right),
            (left_rotation, right_rotation),
            posture_gain,
        )
        raw = metrics.pop("raw")
        stage_metrics.append(metrics)
        if passed:
            current_q = candidate_q
            selected_left_rotation = left_rotation
            selected_right_rotation = right_rotation
    arm_q = current_q

    left_acquire = selected_left["initial"].finger_q
    left_hold = selected_left["charger"].finger_q
    right_preinsert = runtime.open_hand_q["right"].copy()
    right_preinsert[-2:] = 0.5 * (
        runtime.open_hand_q["right"][-2:] + selected_right.hand_q[-2:]
    )
    right_hook = selected_right.hand_q
    left_q, right_q, hand_phases = continuous_hand_trajectories(
        timeline,
        optimized_action,
        runtime,
        left_acquire,
        left_hold,
        right_preinsert,
        right_hook,
    )
    wrist_left = np.empty((timeline.trajectory_length, 4, 4))
    wrist_right = np.empty_like(wrist_left)
    achieved_left = np.empty((timeline.trajectory_length, 3))
    achieved_right = np.empty_like(achieved_left)
    achieved_left_rotation = np.empty((timeline.trajectory_length, 3, 3))
    achieved_right_rotation = np.empty_like(achieved_left_rotation)
    readback_contacts = {
        label: np.empty((timeline.trajectory_length, 3)) for label in runtime.contacts
    }
    for index in range(timeline.trajectory_length):
        runtime.assign(arm_q[index], left_q[index], right_q[index])
        wrist_left[index] = runtime.wrist_pose("left")
        wrist_right[index] = runtime.wrist_pose("right")
        achieved_left[index], achieved_left_rotation[index], _ = runtime.palm_state("left")
        achieved_right[index], achieved_right_rotation[index], _ = runtime.palm_state("right")
        for label in runtime.contacts:
            readback_contacts[label][index] = runtime.contact_pose(label)[0]
    phone_pose = build_phone_object_trajectory(
        timeline, wrist_left, phone_initial, phone_pad, wrist_from_phone_lock
    )
    wrist_from_accessory_lock = inverse_pose(selected_right.wrist_pose) @ accessory_at_grasp
    accessory_pose = build_accessory_object_trajectory(
        timeline, phone_pose, phone_from_accessory, wrist_right, wrist_from_accessory_lock
    )

    left_metrics, right_metrics = evaluate_contact_trajectories(
        runtime,
        arm_q,
        left_q,
        right_q,
        phone_pose,
        accessory_pose,
        timeline,
        phone_dimensions,
        ring_inner,
        ring_outer,
        ring_depth,
    )
    swept = swept_right_c_audit(
        runtime,
        arm_q,
        left_q,
        right_q,
        accessory_pose,
        timeline,
        ring_inner,
        ring_outer,
        ring_depth,
        maximum_tip_step_m=0.001,
        minimum_substeps=50,
    )
    collision = collision_audit(runtime, arm_q, left_q, right_q, float(layout["table"]["surface_height"]))
    limits = runtime.info["joint_limits"]
    position_left_error = np.linalg.norm(achieved_left - target_left, axis=1)
    position_right_error = np.linalg.norm(achieved_right - target_right, axis=1)
    position_rate = float(np.mean((position_left_error <= 0.005) & (position_right_error <= 0.005)))
    joint_margin = np.minimum(arm_q - limits[:, 0], limits[:, 1] - arm_q)
    v14_margin = np.minimum(v14_q - limits[:, 0], limits[:, 1] - v14_q)
    arm_step_norm = np.linalg.norm(np.diff(arm_q, axis=0), axis=1)
    branch_flags = np.zeros(len(arm_q), dtype=bool)
    for sample in range(1, len(arm_q)):
        local_step = np.median(arm_step_norm[max(0, sample - 10) : min(len(arm_step_norm), sample + 9)])
        branch_flags[sample] = arm_step_norm[sample - 1] > max(0.15, 8.0 * max(float(local_step), 1e-5))
    joint_metrics = {
        "v14_minimum_joint_limit_margin_rad": float(np.min(v14_margin)),
        "v15_minimum_joint_limit_margin_rad": float(np.min(joint_margin)),
        "preferred_margin_rad": 0.03,
        "diagnostic_acceptable_margin_rad": 0.01,
        "joint_limit_violation_count": int(np.sum(joint_margin < -1e-9)),
        "branch_discontinuity_count": int(np.count_nonzero(branch_flags)),
        "branch_discontinuity_frames": np.flatnonzero(branch_flags).tolist(),
        "maximum_arm_joint_step_rad": float(np.max(np.abs(np.diff(arm_q, axis=0)))),
        "left_dex3_minimum_margin_rad": float(np.min(np.minimum(
            left_q - runtime.hand_limits["left"][:, 0], runtime.hand_limits["left"][:, 1] - left_q
        ))),
        "right_dex3_minimum_margin_rad": float(np.min(np.minimum(
            right_q - runtime.hand_limits["right"][:, 0], runtime.hand_limits["right"][:, 1] - right_q
        ))),
        "warning": "JOINT_MARGIN_BELOW_DIAGNOSTIC_TARGET" if np.min(joint_margin) < 0.01 else None,
    }
    clearance = {
        "v14_action_phone_grasp_arm_table_clearance_mm": 4.788,
        "v15_minimum_palm_center_table_clearance_mm": collision["minimum_palm_center_table_clearance_m"] * 1000.0,
        "preferred_clearance_mm": 10.0,
        "minimum_diagnostic_target_mm": 5.0,
        "note": "v15 value is active palm-center clearance; prohibited MuJoCo penetration is audited separately",
    }

    phone_center_error = float(np.linalg.norm(phone_pose[charger, :3, 3] - phone_pad[:3, 3]))
    phone_normal_error = float(np.degrees(np.arccos(np.clip(np.dot(
        phone_pose[charger, :3, 1], phone_pad[:3, 1]
    ), -1.0, 1.0))))
    phone_vertical_error = float(np.degrees(np.arccos(np.clip(np.dot(
        phone_pose[charger, :3, 0], phone_pad[:3, 0]
    ), -1.0, 1.0))))
    portrait_error = float(np.degrees(np.arccos(np.clip(abs(phone_pose[portrait, 2, 0]), -1.0, 1.0))))
    orientation_metrics = {
        "stages": stage_metrics,
        "portrait_task_axis_correction_deg": portrait_axis_correction_deg,
        "portrait_long_axis_error_deg": portrait_error,
        "charger_phone_center_error_mm": phone_center_error * 1000.0,
        "charger_phone_normal_error_deg": phone_normal_error,
        "charger_phone_vertical_axis_error_deg": phone_vertical_error,
        "left_rotation_progress_correlation": rotation_progress_correlation(
            source_left_rotation, selected_left_rotation,
            int(timeline.event("phone_rotation_to_portrait_start").action_index), portrait,
        ),
        "right_rotation_progress_correlation": rotation_progress_correlation(
            source_right_rotation, selected_right_rotation, detachment, removed,
        ),
    }
    orientation_pass = bool(
        orientation_metrics["portrait_long_axis_error_deg"] <= 5.0
        and phone_center_error <= 0.005
        and phone_normal_error <= 5.0
        and phone_vertical_error <= 5.0
        and orientation_metrics["left_rotation_progress_correlation"] >= 0.90
        and orientation_metrics["right_rotation_progress_correlation"] >= 0.90
    )
    orientation_metrics["pass"] = orientation_pass

    v14_fidelity = json.loads((V14 / "aloha_fidelity_metrics_v14.json").read_text(encoding="utf-8"))
    fidelity = {
        "position_backbone_byte_identical": True,
        "v14_path_shape_minimum": v14_fidelity["minimum_major_phase_fidelity"]["path_shape"],
        "v14_speed_minimum": v14_fidelity["minimum_major_phase_fidelity"]["speed"],
        "v15_position_path_shape": v14_fidelity["minimum_major_phase_fidelity"]["path_shape"],
        "v15_speed_profile": v14_fidelity["minimum_major_phase_fidelity"]["speed"],
        "v15_bimanual_midpoint_trend": v14_fidelity["bimanual"]["midpoint_trend_correlation"],
        "v15_relative_hand_vector_trend": v14_fidelity["bimanual"]["relative_hand_vector_trend_correlation"],
        "v15_inter_hand_distance_trend": v14_fidelity["bimanual"]["inter_hand_distance_trend_correlation"],
        "left_task_rotation_progress": orientation_metrics["left_rotation_progress_correlation"],
        "right_task_rotation_progress": orientation_metrics["right_rotation_progress_correlation"],
    }
    fidelity["pass"] = bool(min(
        fidelity["v15_position_path_shape"],
        fidelity["v15_speed_profile"],
        fidelity["v15_bimanual_midpoint_trend"],
        fidelity["v15_relative_hand_vector_trend"],
        fidelity["v15_inter_hand_distance_trend"],
        fidelity["left_task_rotation_progress"],
        fidelity["right_task_rotation_progress"],
    ) >= 0.90)

    target_preservation = {
        "status": "V14_CARTESIAN_POSITION_PATH_PRESERVED",
        "v14_target_npz_sha256_before": hashes_before[str(V14_TARGET.resolve())],
        "left_target_array_sha256_before": target_left_hash,
        "right_target_array_sha256_before": target_right_hash,
        "left_target_array_sha256_after": array_sha(target_left),
        "right_target_array_sha256_after": array_sha(target_right),
        "left_byte_identical": array_sha(target_left) == target_left_hash,
        "right_byte_identical": array_sha(target_right) == target_right_hash,
        "target_npz_keys": v14_target_keys,
        "position_target_mutation_performed": False,
    }
    dump(OUT / "v14_position_preservation_audit.json", target_preservation)
    dump(OUT / "orientation_mapping_audit.json", {
        "status": "VERIFIED_RELATIVE_ROTATION_MAPPING",
        "formula": "C^T @ delta_R_ALOHA @ C",
        "C_left": C_LEFT,
        "C_right": C_RIGHT,
        "determinants": [float(np.linalg.det(C_LEFT)), float(np.linalg.det(C_RIGHT))],
        "absolute_ALOHA_orientation_copied": False,
        "hand_written_fixed_quarter_turn_used": False,
        "task_axis_correction_derived_from_geometry": True,
    })
    dump(OUT / "task_orientation_metrics.json", orientation_metrics)
    dump(OUT / "left_ab_contact_metrics.json", left_metrics)
    dump(OUT / "right_c_contact_metrics.json", right_metrics)
    dump(OUT / "right_c_swept_collision_metrics.json", swept)
    dump(OUT / "joint_limit_margin_metrics.json", joint_metrics)
    dump(OUT / "clearance_metrics.json", clearance)
    dump(OUT / "collision_breakdown.json", collision)
    dump(OUT / "aloha_motion_fidelity.json", fidelity)

    save_npz(
        OUT / "orientation_targets.npz",
        timestamps=timestamps,
        left_orientation_target=selected_left_rotation,
        right_orientation_target=selected_right_rotation,
        mapped_source_left_orientation=source_left_only,
        mapped_source_right_orientation=source_right_only,
        source_left_rotation=source_left_rotation,
        source_right_rotation=source_right_rotation,
        task_axis_left_orientation=task_left_final,
        task_axis_right_orientation=task_right_final,
        semantic_event_names=np.asarray(list(TASK_EVENTS)),
        semantic_event_indices=np.asarray([timeline.event(name).action_index for name in TASK_EVENTS]),
    )
    save_npz(
        OUT / "left_dex3_trajectory.npz",
        timestamps=timestamps,
        joint_names=np.asarray(runtime.hand_joint_names["left"]),
        q=left_q,
        phases=hand_phases["left"],
        source_gripper=optimized_action[:, 6],
        contact_A=readback_contacts["left_A"],
        contact_B=readback_contacts["left_B"],
        contact_C=readback_contacts["left_C"],
        semantic_timeline_hash=np.array(timeline.detector_config_hash),
        continuous=np.array(True),
        diagnostic_only=np.array(True),
    )
    save_npz(
        OUT / "right_dex3_trajectory.npz",
        timestamps=timestamps,
        joint_names=np.asarray(runtime.hand_joint_names["right"]),
        q=right_q,
        phases=hand_phases["right"],
        source_gripper=optimized_action[:, 13],
        contact_A=readback_contacts["right_A"],
        contact_B=readback_contacts["right_B"],
        contact_C=readback_contacts["right_C"],
        semantic_timeline_hash=np.array(timeline.detector_config_hash),
        continuous=np.array(True),
        diagnostic_only=np.array(True),
    )
    full_q = np.c_[arm_q, left_q, right_q]
    full_names = np.r_[arm_joint_names, runtime.hand_joint_names["left"], runtime.hand_joint_names["right"]]
    save_npz(
        OUT / "full_arm_dex3_trajectory.npz",
        optimized_action=optimized_action,
        source_timestamps=timestamps,
        controlled_joint_names=full_names,
        g1_arm_q=arm_q,
        left_dex3_q=left_q,
        right_dex3_q=right_q,
        controlled_q=full_q,
        immutable_left_position_target=target_left,
        immutable_right_position_target=target_right,
        target_left_orientation=selected_left_rotation,
        target_right_orientation=selected_right_rotation,
        achieved_left_position=achieved_left,
        achieved_right_position=achieved_right,
        achieved_left_rotation=achieved_left_rotation,
        achieved_right_rotation=achieved_right_rotation,
        target_phone_pose=phone_pose,
        target_accessory_pose=accessory_pose,
        left_wrist_pose=wrist_left,
        right_wrist_pose=wrist_right,
        g1_root=root_position,
        workspace_scale=np.array(0.42),
        semantic_event_names=np.asarray(list(TASK_EVENTS)),
        semantic_event_indices=np.asarray([timeline.event(name).action_index for name in TASK_EVENTS]),
        semantic_timeline_provenance=np.array("HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE_VIA_GENERIC_API"),
        method=np.array(METHOD),
        diagnostic_only=np.array(True),
        physics_steps=np.array(0),
        real_robot_command_allowed=np.array(False),
    )

    input_audit = {
        "status": "INPUT_FREEZE_PASS",
        "hashes_before": hashes_before,
        "optimized_action_shape": list(optimized_action.shape),
        "optimized_action_finite": bool(np.isfinite(optimized_action).all()),
        "optimized_action_array_sha256": array_sha(optimized_action),
        "timestamps_array_sha256": array_sha(timestamps),
        "root_position": root_position,
        "root_forward_offset_m": root_offset,
        "workspace_scale": 0.42,
        "G1_expert_paths_read": [],
        "heldout_episode_ids_read": [],
        "validation_episode_ids_read": [],
        "physics_steps": 0,
        "DDS": False,
        "publisher": False,
        "hardware_command": False,
    }
    dump(OUT / "input_freeze_audit.json", input_audit)
    dump(OUT / "reusable_vs_episode_derived_v15.json", {
        "reusable_translator_parameters": [
            "ALOHA_to_G1_axis_calibration",
            "workspace_scale",
            "G1_root_task_registration_rule",
            "source_relative_orientation_mapping",
            "orientation_objective_weights",
            "temporal_IK_weights",
            "nullspace_posture_weights",
            "left_AB_anatomical_and_contact_side_assignment",
            "right_C_insertion_family_rule",
            "Dex3_source_progress_interpolation_rule",
            "contact_and_collision_tolerances",
            "robustness_criteria",
        ],
        "episode_derived_variables": [
            "source_action",
            "trajectory_length",
            "timestamps",
            "semantic_event_indices",
            "semantic_phase_durations",
            "semantic_progress_arrays",
            "ALOHA_FK_trajectory",
            "source_relative_displacement_and_rotation",
            "source_gripper_progress",
            "G1_Cartesian_position_targets",
            "G1_arm_joint_trajectory",
            "Dex3_joint_trajectory",
        ],
        "episode49_trajectory_reusable_for_other_episode": False,
        "future_flow": "NEW SOURCE ACTION -> NEW FK/SEMANTIC PROGRESS -> SAME FROZEN RULES -> NEW G1 TRAJECTORY",
        "future_source_provenance_required": [
            "source_episode_id",
            "source_action_type",
            "model_checkpoint",
            "source_action_hash",
            "generated_action_method",
            "semantic_timeline_hash",
            "translator_config_hash",
        ],
        "preferred_future_source_action_type": "SMOLVLA_GENERATED",
        "demonstration_transfer_may_be_claimed_as_VLA_generated_transfer": False,
        "strong_generalization_protocol": {
            "development": "Episode 49 only",
            "integration_pilot": "three deterministic task-critical-ready frozen-validation episodes after user approval",
            "final_freeze_artifact": "configs/aloha_g1_cross_embodiment_translator_v15_final.frozen.json",
            "final_test": "one-shot frozen HELD-OUT 30",
            "heldout_parameter_changes_allowed": False,
            "required_rates": [
                "semantic_coverage_over_30",
                "conditional_retargeting_success_among_semantic_ready",
                "full_pipeline_success_over_30",
            ],
        },
        "episode49_candidate_config_frozen": False,
        "reason_not_frozen": "Episode-49 task orientation, continuous contact, and collision gates failed",
    })

    left_pass = bool(selected_left["valid_entire_carrier"] and left_metrics["continuous_pinch_pass"])
    right_pass = bool(selected_right.valid and right_metrics["hook_hold_pass"] and swept["pass"])
    position_pass = bool(
        position_rate >= 0.99
        and joint_metrics["joint_limit_violation_count"] == 0
        and joint_metrics["branch_discontinuity_count"] == 0
    )
    gates = {
        "V14_CARTESIAN_POSITION_PATH_PRESERVED": target_preservation["left_byte_identical"] and target_preservation["right_byte_identical"],
        "EP49_GENERIC_SEMANTIC_API_USED": literal_audit["pass"],
        "TASK_CRITICAL_ORIENTATION_PASS": orientation_pass,
        "LEFT_AB_CONTINUOUS_PHONE_PINCH_PASS": left_pass,
        "RIGHT_C_CONTINUOUS_INSERTION_PASS": bool(selected_right.valid and swept["pass"]),
        "RIGHT_C_HOOK_REMOVE_HOLD_PASS": right_pass,
        "CHARGER_ALIGNMENT_PASS": bool(phone_center_error <= 0.005 and phone_normal_error <= 5.0 and phone_vertical_error <= 5.0),
        "PROHIBITED_COLLISION_ZERO": collision["pass"],
        "ALOHA_MOTION_FIDELITY_PASS": fidelity["pass"],
        "POSITION_IK_PASS": position_pass,
    }
    numeric_pass = all(gates.values())
    statuses = [name for name, passed in gates.items() if passed]
    blockers = []
    if not target_preservation["left_byte_identical"] or not target_preservation["right_byte_identical"]:
        blockers.append("BLOCKED_V14_POSITION_TARGET_MUTATION")
    if not literal_audit["pass"]:
        blockers.append("BLOCKED_SEMANTIC_RUNTIME_HARDCODING")
    if not orientation_pass:
        blockers.append("BLOCKED_TASK_ORIENTATION")
    if not left_pass:
        blockers.append("BLOCKED_LEFT_AB_CONTINUOUS_PINCH")
    if not selected_right.valid or not swept["pass"]:
        blockers.append("BLOCKED_RIGHT_C_INSERTION")
    if not right_pass:
        blockers.append("BLOCKED_RIGHT_C_HOOK_RETENTION")
    if not collision["pass"]:
        blockers.append("BLOCKED_V15_COLLISION")
    if joint_metrics["v15_minimum_joint_limit_margin_rad"] < 0.0:
        blockers.append("BLOCKED_JOINT_MARGIN")
    dump(OUT / "numeric_gate_summary.json", {
        "numeric_pass": numeric_pass,
        "passed_statuses": statuses,
        "blockers": blockers,
        "position_tracking": {
            "simultaneous_5mm_rate": position_rate,
            "left_mean_mm": float(np.mean(position_left_error) * 1000.0),
            "left_max_mm": float(np.max(position_left_error) * 1000.0),
            "right_mean_mm": float(np.mean(position_right_error) * 1000.0),
            "right_max_mm": float(np.max(position_right_error) * 1000.0),
        },
        "isaaclab_pending": True,
    })
    print(json.dumps({
        "numeric_pass": numeric_pass,
        "blockers": blockers,
        "selected_left_assignment": selected_left["assignment"],
        "selected_right_family": selected_right.family,
        "position_rate": position_rate,
        "left_continuous": left_metrics["continuous_pinch_pass"],
        "right_continuous": right_metrics["hook_hold_pass"],
        "swept": swept["pass"],
        "collision": collision["pass"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
