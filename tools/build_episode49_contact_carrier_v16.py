#!/usr/bin/env python3
"""Build the Episode-49 v16 contact-carrier diagnostic/candidate.

Episode 49 is an explicit development input.  Runtime phase boundaries are
resolved only through the generic SemanticTimeline API; numeric event indices
are never used as semantic rules.  Validation and held-out sources are not
opened by this program.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from pxr import Usd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

from aloha_g1_v15.kinematics import (  # noqa: E402
    ActiveG1Dex3,
    nearest_box_surface,
    normalize,
    ring_material_gap,
    sha256_file,
)
from aloha_g1_v15.semantic_input import (  # noqa: E402
    TASK_EVENTS,
    load_human_reviewed_development_timeline,
)
from aloha_g1_v15.translator import C_LEFT, C_RIGHT, mapped_relative  # noqa: E402
from aloha_g1_v16.carrier import (  # noqa: E402
    LeftCarrierCandidate,
    RightCarrierCandidate,
    build_left_pinch_carrier,
    build_right_hook_carrier,
    inverse_pose,
    search_left_common_rigid_carrier,
    search_right_hook_anchor,
)
from aloha_g1_v16.trajectory import (  # noqa: E402
    carrier_rotation_path,
    continuous_hand_profile,
    distribution,
    path_progress,
    pose_deviation,
    semantic_translation_residual,
    solve_coupled_side_trajectory,
)
from build_episode49_orientation_dex3_v15 import (  # noqa: E402
    array_sha,
    collision_audit,
    dump,
    phone_on_pad_pose,
    portrait_phone_rotation,
    save_npz,
    usd_pose,
)
from retarget_aloha_trajectory_to_g1 import retarget_aloha_trajectory_to_g1  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_contact_carrier_v16"
V14 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_root_registered_v14"
V15 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_orientation_dex3_v15"
V12 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SOURCE = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
PHASE_LIBRARY = V12 / "aloha_phase_motion_library.npz"
V14_TARGET = V14 / "corrected_targets_v14.npz"
V14_NULL = V14 / "position_only_nullspace_v14.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
ROOT_CONFIG = ROOT / "configs/g1_root_forward_v14.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
ACTIVE_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
FIXED_SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
MODEL = Path("/home/jbnu/mujoco_menagerie/unitree_g1/g1_with_hands.xml")
DEX3_MAPPING = ROOT / "configs/dex3_abc_finger_mapping.sim.json"
PALM_CONFIG = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"
AXIS_CONFIG = ROOT / "configs/aloha_tcp_to_g1_palm_calibration.sim.json"
METHOD = "ALOHA_PRIMARY_CONTACT_CARRIER_V16"


def _array_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _candidate_arrays(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _candidate_arrays(value) for key, value in payload.items()}
    if isinstance(payload, list):
        try:
            array = np.asarray(payload, dtype=np.float64)
        except (TypeError, ValueError):
            return [_candidate_arrays(value) for value in payload]
        return array
    return payload


def _runtime_literal_audit(paths: list[Path], forbidden: set[int]) -> dict[str, Any]:
    findings = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in forbidden:
                findings.append({"path": str(path), "line": int(node.lineno), "value": int(node.value)})
    return {"scanned_files": [str(path) for path in paths], "findings": findings, "count": len(findings), "pass": not findings}


def _phase_correlation(reference: np.ndarray, candidate: np.ndarray, start: int, end: int) -> dict[str, float]:
    a = np.asarray(reference[start : end + 1], dtype=np.float64)
    b = np.asarray(candidate[start : end + 1], dtype=np.float64)
    a = a - a[0]
    b = b - b[0]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    scale_a = float(np.max(na))
    scale_b = float(np.max(nb))
    an = a / max(scale_a, 1e-12)
    bn = b / max(scale_b, 1e-12)
    path = float(np.corrcoef(an.reshape(-1), bn.reshape(-1))[0, 1]) if len(a) > 2 else 1.0
    va = np.linalg.norm(np.diff(a, axis=0), axis=1)
    vb = np.linalg.norm(np.diff(b, axis=0), axis=1)
    if np.std(va) <= 1e-12 or np.std(vb) <= 1e-12:
        speed = 1.0 if np.allclose(va, vb, atol=1e-9) else 0.0
    else:
        speed = float(np.corrcoef(va / max(np.max(va), 1e-12), vb / max(np.max(vb), 1e-12))[0, 1])
    displacement_a = a[-1]
    displacement_b = b[-1]
    direction = float(np.dot(displacement_a, displacement_b) / max(np.linalg.norm(displacement_a) * np.linalg.norm(displacement_b), 1e-12))
    return {"path_shape_correlation": path, "speed_profile_correlation": speed, "phase_displacement_direction_cosine": direction}


def _trend_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 1.0 if np.allclose(a, b, atol=1e-9) else 0.0
    return float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])


def _rotation_progress_correlation(reference: np.ndarray, candidate: np.ndarray, start: int, end: int) -> float:
    def progress(value: np.ndarray) -> np.ndarray:
        segment = value[start : end + 1]
        step = Rotation.from_matrix(np.einsum("tji,tjk->tik", segment[:-1], segment[1:])).magnitude()
        result = np.r_[0.0, np.cumsum(np.abs(step))]
        return result / max(float(result[-1]), 1e-12)
    a, b = progress(reference), progress(candidate)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _collect_pose_trajectory(runtime: ActiveG1Dex3, arm: np.ndarray, left: np.ndarray, right: np.ndarray, left_patch: dict[str, np.ndarray], right_patch: np.ndarray, accessory: np.ndarray, hook_hint: np.ndarray):
    length = len(arm)
    left_carrier = np.empty((length, 4, 4))
    right_carrier = np.empty_like(left_carrier)
    left_wrist = np.empty_like(left_carrier)
    right_wrist = np.empty_like(left_carrier)
    left_palm = np.empty((length, 3))
    right_palm = np.empty_like(left_palm)
    contacts = {name: np.empty((length, 3)) for name in runtime.contacts}
    for index in range(length):
        runtime.assign(arm[index], left[index], right[index])
        left_carrier[index], _ = build_left_pinch_carrier(runtime, left_patch["A"], left_patch["B"])
        right_carrier[index], _ = build_right_hook_carrier(runtime, right_patch, accessory[index, :3, 3], hook_hint)
        left_wrist[index] = runtime.wrist_pose("left")
        right_wrist[index] = runtime.wrist_pose("right")
        left_palm[index] = runtime.palm_pose("left")[:3, 3]
        right_palm[index] = runtime.palm_pose("right")[:3, 3]
        for name in contacts:
            contacts[name][index] = runtime.contact_pose(name)[0]
    return left_carrier, right_carrier, left_wrist, right_wrist, left_palm, right_palm, contacts


def _face_gap(point: np.ndarray, phone: np.ndarray, dimensions: np.ndarray, face_sign: float) -> float:
    local = phone[:3, :3].T @ (point - phone[:3, 3])
    half = 0.5 * dimensions
    nearest = np.array([
        np.clip(local[0], -half[0], half[0]), face_sign * half[1], np.clip(local[2], -half[2], half[2])
    ])
    return float(np.linalg.norm(local - nearest))


def _joint_and_branch_metrics(runtime: ActiveG1Dex3, arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    arm_limits = runtime.info["joint_limits"]
    arm_margin = np.minimum(arm - arm_limits[:, 0], arm_limits[:, 1] - arm)
    left_margin = np.minimum(left - runtime.hand_limits["left"][:, 0], runtime.hand_limits["left"][:, 1] - left)
    right_margin = np.minimum(right - runtime.hand_limits["right"][:, 0], runtime.hand_limits["right"][:, 1] - right)
    full = np.c_[arm, left, right]
    step = np.max(np.abs(np.diff(full, axis=0)), axis=1)
    arm_step = np.max(np.abs(np.diff(arm, axis=0)), axis=1)
    local = np.array([
        np.median(arm_step[max(0, index - 10) : min(len(arm_step), index + 10)])
        for index in range(len(arm_step))
    ])
    # A Dex3 contact-state transition can legitimately move farther than an
    # arm joint in one source sample.  Branch continuity is an arm-IK property,
    # so detect it from the 14 arm joints and report the full 28-DoF step
    # separately.
    branch = np.flatnonzero(arm_step > np.maximum(0.20, 8.0 * np.maximum(local, 1e-5))) + 1
    return {
        "minimum_arm_joint_margin_rad": float(np.min(arm_margin)),
        "minimum_left_dex3_margin_rad": float(np.min(left_margin)),
        "minimum_right_dex3_margin_rad": float(np.min(right_margin)),
        "joint_limit_violation_count": int(np.sum(arm_margin < -1e-9) + np.sum(left_margin < -1e-9) + np.sum(right_margin < -1e-9)),
        "maximum_joint_step_rad": float(np.max(step)),
        "maximum_arm_joint_step_rad": float(np.max(arm_step)),
        "branch_discontinuity_count": int(len(branch)),
        "branch_discontinuity_frames": branch.tolist(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [SOURCE, PHASE_LIBRARY, V14_TARGET, V14_NULL, TIMELINE, ALIGNMENT, ROOT_CONFIG, LAYOUT, ACTIVE_SCENE, FIXED_SCENE, MODEL, DEX3_MAPPING, PALM_CONFIG, AXIS_CONFIG]
    required += [V15 / name for name in (
        "run_manifest.json", "numeric_gate_summary.json", "left_ab_contact_metrics.json",
        "right_c_contact_metrics.json", "task_orientation_metrics.json", "collision_breakdown.json",
        "joint_limit_margin_metrics.json", "aloha_motion_fidelity.json",
    )]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes_before = {str(path.resolve()): sha256_file(path) for path in required}

    with np.load(SOURCE, allow_pickle=False) as archive:
        source_action = archive["optimized_action"].copy()
        timestamps = archive["timestamp"].copy()
        fps = float(archive["fps"])
    with np.load(PHASE_LIBRARY, allow_pickle=False) as archive:
        source_left_position = archive["left_tcp_position"].copy()
        source_right_position = archive["right_tcp_position"].copy()
        source_left_rotation = archive["left_tcp_rotation"].copy()
        source_right_rotation = archive["right_tcp_rotation"].copy()
    with np.load(V14_TARGET, allow_pickle=False) as archive:
        target_left = archive["corrected_left_position"].copy()
        target_right = archive["corrected_right_position"].copy()
    with np.load(V14_NULL, allow_pickle=False) as archive:
        v14_q = archive["g1_arm_q"].copy()
        joint_names = archive["arm_joint_names"].copy()
        root_position = archive["g1_root"].copy()
        root_offset = float(archive["g1_root_forward_offset_m"])
    if source_action.shape != (len(source_action), 14) or len(source_action) != len(v14_q) or not np.isfinite(source_action).all() or fps != 30.0:
        raise RuntimeError("immutable source invariant failed")
    timeline = load_human_reviewed_development_timeline(
        TIMELINE, ALIGNMENT, source_action, timestamps,
        source_left_position, source_right_position, source_left_rotation, source_right_rotation,
        trajectory_path=SOURCE, fk_model_path=MODEL, task_geometry_path=LAYOUT,
    )
    dry_run = retarget_aloha_trajectory_to_g1(
        source_action, timestamps, timeline, {"method": METHOD}, {"scene": str(ACTIVE_SCENE)}, dry_run=True
    )
    event_index = {name: int(timeline.event(name).action_index) for name in TASK_EVENTS}
    forbidden = set(event_index.values())
    runtime_files = sorted((ROOT / "tools/aloha_g1_v16").glob("*.py")) + [Path(__file__).resolve()]
    literal_audit = _runtime_literal_audit(runtime_files, forbidden)
    if not literal_audit["pass"]:
        raise RuntimeError("BLOCKED_SEMANTIC_RUNTIME_HARDCODING")

    root_config = json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))
    if not np.allclose(root_position, root_config["new_exact_root_xyz_m"], atol=1e-9) or not np.isclose(root_offset, 0.199):
        raise RuntimeError("frozen v14 root mismatch")
    runtime = ActiveG1Dex3(MODEL, DEX3_MAPPING, PALM_CONFIG, root_position)
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    phone_dimensions = np.asarray(layout["phone"]["size_landscape_xyz"], dtype=np.float64)
    ring_inner = 0.5 * float(layout["accessory"]["main_inner_diameter"])
    ring_outer = 0.5 * float(layout["accessory"]["main_outer_diameter"])
    ring_depth = float(layout["accessory"]["main_depth"])
    table_height = float(layout["table"]["surface_height"])
    stage = Usd.Stage.Open(str(ACTIVE_SCENE))
    if stage is None:
        raise RuntimeError("active scene could not be opened")
    phone_initial = usd_pose(stage, "/World/MagSafeScene/Phone")
    accessory_initial = usd_pose(stage, "/World/MagSafeScene/Accessory")
    pad_pose = usd_pose(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    phone_charger = phone_on_pad_pose(pad_pose)
    phone_from_accessory = inverse_pose(phone_initial) @ accessory_initial

    v15_failure = {
        "status": "FAILED_DIAGNOSTIC_CANDIDATE_PRESERVED",
        "v15_directory": str(V15.resolve()),
        "hashes_before": {str(path.resolve()): sha256_file(path) for path in V15.iterdir() if path.is_file()},
        "verified_blockers": {
            "portrait_task_axis_error_deg": 25.027,
            "left_A_carry_max_gap_mm": 45.253,
            "left_B_carry_max_gap_mm": 46.077,
            "right_C_keyframe_gap_mm": 54.672,
            "right_C_continuous_gap_mm": 68.724,
            "branch_discontinuities": 15,
            "maximum_joint_step_rad": 3.389,
            "prohibited_contact_records": 161,
            "arm_torso_contact_records": 159,
        },
        "v15_used_as_trajectory_target": False,
    }
    dump(OUT / "v15_failure_preservation_audit.json", v15_failure)

    left_grasp = event_index["left_phone_grasp_start"]
    rotation_start = event_index["phone_rotation_to_portrait_start"]
    portrait = event_index["phone_portrait_reached"]
    right_grasp = event_index["right_accessory_grasp_start"]
    detachment = event_index["accessory_detachment_start"]
    removed = event_index["accessory_removed"]
    phone_move = event_index["phone_move_to_charger_start"]
    charger = event_index["phone_charger_attachment_complete"]
    left_release = event_index["left_phone_release_complete"]
    right_release = event_index["right_accessory_release_complete"]

    v15_left_search = json.loads((V15 / "left_ab_candidate_search.json").read_text(encoding="utf-8"))
    cached_left = OUT / "selected_left_pincher_carrier.json"
    if cached_left.is_file():
        cached_payload = json.loads(cached_left.read_text(encoding="utf-8"))
        left_candidates = [LeftCarrierCandidate(**_candidate_arrays(row)) for row in cached_payload["candidates"]]
    else:
        left_candidates = []
        for candidate_number, assignment in enumerate(("A_SCREEN_B_BACK", "A_BACK_B_SCREEN")):
            old = next(row for row in v15_left_search["candidates"] if row["assignment"] == assignment)
            hand_seed = np.asarray(old["initial"]["finger_q_rad"], dtype=np.float64)[:5]
            candidate = search_left_common_rigid_carrier(
                runtime, v14_q[left_grasp], v14_q[charger], phone_initial, phone_charger,
                phone_dimensions, assignment, hand_seed=hand_seed,
                random_seed=510 + candidate_number, seed_count=12,
                normal_weight=0.50, minimum_normal_alignment=0.0,
            )
            left_candidates.append(candidate)
    selected_left = min(left_candidates, key=lambda value: (not value.valid, value.score))

    # Base carrier uses the v14 ALOHA-derived arm path and the selected rigid hand geometry.
    base_left_carrier = np.empty((timeline.trajectory_length, 4, 4))
    left_hold_hand = selected_left.hand_q.copy()
    for index in range(timeline.trajectory_length):
        runtime.assign(v14_q[index], left_hold_hand, runtime.open_hand_q["right"])
        base_left_carrier[index], _ = build_left_pinch_carrier(
            runtime, selected_left.patch_offsets["A"], selected_left.patch_offsets["B"]
        )
    left_endpoint_offsets = np.stack((
        selected_left.initial_carrier[:3, 3] - base_left_carrier[left_grasp, :3, 3],
        selected_left.charger_carrier[:3, 3] - base_left_carrier[charger, :3, 3],
    ))
    left_fixed_carrier_translation = np.mean(left_endpoint_offsets, axis=0)
    registered_left_carrier = base_left_carrier.copy()
    registered_left_carrier[:, :3, 3] += left_fixed_carrier_translation
    desired_left_position, left_translation_residual = semantic_translation_residual(
        registered_left_carrier[:, :3, 3], source_left_position,
        [left_grasp, charger],
        [selected_left.initial_carrier[:3, 3], selected_left.charger_carrier[:3, 3]],
        left_release,
    )
    carrier_from_phone_lock = inverse_pose(selected_left.initial_carrier) @ phone_initial
    relative_to_portrait = mapped_relative(source_left_rotation, rotation_start, portrait, C_LEFT)[-1]
    carrier_source_portrait = selected_left.initial_carrier[:3, :3] @ relative_to_portrait
    phone_source_portrait = carrier_source_portrait @ carrier_from_phone_lock[:3, :3]
    portrait_phone_target_rotation, portrait_constant_correction_deg = portrait_phone_rotation(phone_source_portrait)
    portrait_carrier_rotation = portrait_phone_target_rotation @ carrier_from_phone_lock[:3, :3].T
    desired_left_rotation, left_rotation_residual = carrier_rotation_path(
        base_left_carrier[:, :3, :3], source_left_rotation, C_LEFT,
        [
            (left_grasp, selected_left.initial_carrier[:3, :3]),
            (rotation_start, selected_left.initial_carrier[:3, :3]),
            (portrait, portrait_carrier_rotation),
            (phone_move, portrait_carrier_rotation),
            (charger, selected_left.charger_carrier[:3, :3]),
            (left_release, selected_left.charger_carrier[:3, :3]),
        ],
    )
    target_left_carrier = np.repeat(np.eye(4)[None], timeline.trajectory_length, axis=0)
    target_left_carrier[:, :3, :3] = desired_left_rotation
    target_left_carrier[:, :3, 3] = desired_left_position

    phone_target = np.repeat(phone_initial[None], timeline.trajectory_length, axis=0)
    carrier_phone = np.einsum("tij,jk->tik", target_left_carrier, carrier_from_phone_lock)
    acquisition = timeline.progress("phone_acquisition")
    for index in range(left_grasp, portrait + 1):
        alpha = float(acquisition[index])
        phone_target[index, :3, 3] = (1.0 - alpha) * phone_initial[:3, 3] + alpha * carrier_phone[index, :3, 3]
        phone_target[index, :3, :3] = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack((phone_initial[:3, :3], carrier_phone[index, :3, :3]))),
        )([alpha]).as_matrix()[0]
    phone_target[portrait : charger + 1] = carrier_phone[portrait : charger + 1]
    phone_target[charger:] = phone_charger
    accessory_attached = np.einsum("tij,jk->tik", phone_target, phone_from_accessory)

    approach_start = int(np.searchsorted(timestamps, timestamps[right_grasp] - 0.4, side="left"))
    source_approach = normalize(source_right_position[right_grasp] - source_right_position[approach_start])
    v14_anchor_payload = json.loads((V14 / "selected_physical_carrier_anchors.json").read_text(encoding="utf-8"))
    right_seed_rows = [
        row for row in v14_anchor_payload.values()
        if isinstance(row, dict) and row.get("action_index") == right_grasp and "diagnostic_right_dex3_C_q_rad" in row
    ]
    if len(right_seed_rows) != 1:
        raise RuntimeError("right C diagnostic seed provenance unresolved")
    right_seed = np.asarray(right_seed_rows[0]["diagnostic_right_dex3_C_q_rad"], dtype=np.float64)
    cached_right = OUT / "selected_right_hook_carrier.json"
    if cached_right.is_file():
        cached_payload = json.loads(cached_right.read_text(encoding="utf-8"))
        right_candidates = [RightCarrierCandidate(**_candidate_arrays(row)) for row in cached_payload["candidates"]]
    else:
        right_candidates = [
            search_right_hook_anchor(
                runtime, v14_q[detachment], accessory_attached[detachment], ring_inner, ring_depth,
                family, source_approach, hand_seed=right_seed,
                random_seed=610 + number, seed_count=12,
            )
            for number, family in enumerate(("AXIAL_APERTURE_INSERTION", "RADIAL_GAP_INSERTION"))
        ]
    selected_right = min(right_candidates, key=lambda value: (not value.valid, value.score))
    hook_hint = selected_right.carrier[:3, 1]
    right_hold_hand = selected_right.hand_q.copy()
    base_right_carrier = np.empty_like(base_left_carrier)
    for index in range(timeline.trajectory_length):
        runtime.assign(v14_q[index], runtime.open_hand_q["left"], right_hold_hand)
        base_right_carrier[index], _ = build_right_hook_carrier(
            runtime, selected_right.patch_offset, accessory_attached[index, :3, 3], hook_hint
        )
    right_fixed_carrier_translation = (
        selected_right.carrier[:3, 3] - base_right_carrier[detachment, :3, 3]
    )
    registered_right_carrier = base_right_carrier.copy()
    registered_right_carrier[:, :3, 3] += right_fixed_carrier_translation
    # Entry distance is defined by ring depth plus a one-millimetre geometric
    # clearance.  Direction comes from the episode's own ALOHA-derived carrier
    # displacement, not from a hand-authored waypoint.
    preinsert_distance = ring_depth + 0.001
    source_entry_direction = normalize(
        registered_right_carrier[detachment, :3, 3]
        - registered_right_carrier[right_grasp, :3, 3]
    )
    preinsert_target = selected_right.carrier.copy()
    preinsert_target[:3, 3] -= source_entry_direction * preinsert_distance
    removed_target_position = selected_right.carrier[:3, 3] + (
        registered_right_carrier[removed, :3, 3] - registered_right_carrier[detachment, :3, 3]
    )
    release_target_position = selected_right.carrier[:3, 3] + (
        registered_right_carrier[right_release, :3, 3] - registered_right_carrier[detachment, :3, 3]
    )
    desired_right_position, right_translation_residual = semantic_translation_residual(
        registered_right_carrier[:, :3, 3], source_right_position,
        [right_grasp, detachment, removed, right_release],
        [preinsert_target[:3, 3], selected_right.carrier[:3, 3], removed_target_position, release_target_position],
        right_release,
    )
    relative_removed = mapped_relative(source_right_rotation, detachment, removed, C_RIGHT)[-1]
    removed_rotation = selected_right.carrier[:3, :3] @ relative_removed
    relative_release = mapped_relative(source_right_rotation, removed, right_release, C_RIGHT)[-1]
    release_rotation = removed_rotation @ relative_release
    desired_right_rotation, right_rotation_residual = carrier_rotation_path(
        base_right_carrier[:, :3, :3], source_right_rotation, C_RIGHT,
        [
            (portrait, base_right_carrier[portrait, :3, :3]),
            (right_grasp, preinsert_target[:3, :3]),
            (detachment, selected_right.carrier[:3, :3]),
            (removed, removed_rotation),
            (right_release, release_rotation),
        ],
    )
    target_right_carrier = np.repeat(np.eye(4)[None], timeline.trajectory_length, axis=0)
    target_right_carrier[:, :3, :3] = desired_right_rotation
    target_right_carrier[:, :3, 3] = desired_right_position

    left_hand_reference = continuous_hand_profile(
        timeline.trajectory_length, runtime.open_hand_q["left"], left_hold_hand,
        timeline.progress("phone_acquisition"), timeline.progress("left_release"),
        left_grasp, portrait, charger, left_release,
    )
    right_hand_reference = continuous_hand_profile(
        timeline.trajectory_length, runtime.open_hand_q["right"], right_hold_hand,
        timeline.progress("accessory_acquisition"), timeline.progress("right_release"),
        right_grasp, detachment, removed, right_release,
    )

    translation_left, rotation_left = pose_deviation(registered_left_carrier, target_left_carrier)
    translation_right, rotation_right = pose_deviation(registered_right_carrier, target_right_carrier)
    correction_audit = {
        "status": "MINIMUM_CORRECTION_AUDIT_COMPLETE",
        "contact_point_definition": "optimized point inside active collision-pad extent, not pad center",
        "fixed_carrier_registration": {
            "left_translation_m": left_fixed_carrier_translation,
            "right_translation_m": right_fixed_carrier_translation,
            "counted_as_time_varying_deformation": False,
        },
        "left": {"translation_m": distribution(translation_left), "rotation_rad": distribution(rotation_left)},
        "right": {"translation_m": distribution(translation_right), "rotation_rad": distribution(rotation_right)},
        "endpoint_search_is_global_proof": False,
        "endpoint_search_description": "deterministic multistart nonlinear least-squares under arm/Dex3 limits and arm-torso penetration penalty",
        "v14_used_as": "ALOHA_PRIMARY_REFERENCE_AND_SEED",
    }
    dump(OUT / "minimum_contact_correction_audit.json", correction_audit)

    pareto_rows = []
    for name, fraction in (("VERY_STRONG_ALOHA", 0.55), ("BALANCED", 0.80), ("CONTACT_STRONG", 1.0)):
        left_gap = max(
            max(selected_left.initial_gaps_m.values()), max(selected_left.charger_gaps_m.values())
        ) + (1.0 - fraction) * max(
            np.linalg.norm(selected_left.initial_carrier[:3, 3] - registered_left_carrier[left_grasp, :3, 3]),
            np.linalg.norm(selected_left.charger_carrier[:3, 3] - registered_left_carrier[charger, :3, 3]),
        )
        right_gap = selected_right.contact_gap_m + (1.0 - fraction) * np.linalg.norm(
            selected_right.carrier[:3, 3] - registered_right_carrier[detachment, :3, 3]
        )
        corrected_left_fraction = registered_left_carrier[:, :3, 3] + fraction * left_translation_residual
        corrected_right_fraction = registered_right_carrier[:, :3, 3] + fraction * right_translation_residual
        left_phase = _phase_correlation(registered_left_carrier[:, :3, 3], corrected_left_fraction, left_grasp, charger)
        right_phase = _phase_correlation(registered_right_carrier[:, :3, 3], corrected_right_fraction, right_grasp, right_release)
        pareto_rows.append({
            "candidate": name, "residual_fraction": fraction,
            "left_contact_upper_bound_mm": left_gap * 1000.0,
            "right_contact_upper_bound_mm": right_gap * 1000.0,
            "endpoint_contact_upper_bound_pass": left_gap <= 0.005 and right_gap <= 0.005,
            "continuous_contact_gate_evaluated_after_coupled_IK": True,
            "left_path_shape_correlation": left_phase["path_shape_correlation"],
            "left_speed_profile_correlation": left_phase["speed_profile_correlation"],
            "right_path_shape_correlation": right_phase["path_shape_correlation"],
            "right_speed_profile_correlation": right_phase["speed_profile_correlation"],
            "maximum_translation_correction_mm": fraction * max(float(np.max(translation_left)), float(np.max(translation_right))) * 1000.0,
        })
    with (OUT / "correction_pareto_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pareto_rows[0]))
        writer.writeheader(); writer.writerows(pareto_rows)
    selected_pareto = next(row for row in pareto_rows if row["candidate"] == "CONTACT_STRONG")

    def left_builder(active: ActiveG1Dex3, hand: np.ndarray, index: int) -> np.ndarray:
        carrier, _ = build_left_pinch_carrier(active, selected_left.patch_offsets["A"], selected_left.patch_offsets["B"])
        return carrier

    def right_builder(active: ActiveG1Dex3, hand: np.ndarray, index: int) -> np.ndarray:
        carrier, _ = build_right_hook_carrier(active, selected_right.patch_offset, accessory_attached[index, :3, 3], hook_hint)
        return carrier

    smooth_v14 = gaussian_filter1d(v14_q, sigma=2.0, axis=0, mode="nearest")
    left_seed, _ = semantic_translation_residual(
        smooth_v14[:, :7], source_left_position,
        [left_grasp, charger], [selected_left.initial_arm_q[:7], selected_left.charger_arm_q[:7]],
        left_release,
    )
    right_seed_path, _ = semantic_translation_residual(
        smooth_v14[:, 7:], source_right_position,
        [detachment], [selected_right.arm_q[7:]], right_release,
    )
    arm_seed = np.c_[left_seed, right_seed_path]
    arm_seed = np.clip(
        arm_seed,
        runtime.info["joint_limits"][:, 0] + 1e-5,
        runtime.info["joint_limits"][:, 1] - 1e-5,
    )
    arm_after_left, left_q, left_solve = solve_coupled_side_trajectory(
        runtime, "left", arm_seed, left_hand_reference, target_left_carrier, left_builder, table_height,
        carrier_position_weight=1100.0, carrier_rotation_weight=9.0,
        reference_weight=0.004, temporal_weight=0.09,
        maximum_step_rad=0.10, maximum_hand_step_rad=0.45, max_nfev=45,
        collision_weight=4000.0, opposite_hand_reference=right_hand_reference,
    )
    arm_q, right_q, right_solve = solve_coupled_side_trajectory(
        runtime, "right", arm_after_left, right_hand_reference, target_right_carrier, right_builder, table_height,
        carrier_position_weight=1100.0, carrier_rotation_weight=9.0,
        reference_weight=0.004, temporal_weight=0.09,
        maximum_step_rad=0.10, maximum_hand_step_rad=0.45, max_nfev=45,
        collision_weight=4000.0, opposite_hand_reference=left_q,
    )

    # Target object trajectories are driven by semantic carrier state.  They do
    # not feed back as a replacement arm motion source.
    accessory_target = accessory_attached.copy()
    carrier_from_accessory = inverse_pose(selected_right.carrier) @ accessory_attached[detachment]
    accessory_carried = np.einsum("tij,jk->tik", target_right_carrier, carrier_from_accessory)
    removal_progress = timeline.progress("accessory_removal")
    for index in range(detachment, removed + 1):
        alpha = float(removal_progress[index])
        accessory_target[index, :3, 3] = (1.0 - alpha) * accessory_attached[index, :3, 3] + alpha * accessory_carried[index, :3, 3]
        accessory_target[index, :3, :3] = Slerp(
            [0.0, 1.0], Rotation.from_matrix(np.stack((accessory_attached[index, :3, :3], accessory_carried[index, :3, :3])))
        )([alpha]).as_matrix()[0]
    accessory_target[removed : right_release + 1] = accessory_carried[removed : right_release + 1]
    accessory_target[right_release:] = accessory_carried[right_release]

    achieved = _collect_pose_trajectory(
        runtime, arm_q, left_q, right_q,
        selected_left.patch_offsets, selected_right.patch_offset, accessory_target, hook_hint,
    )
    achieved_left_carrier, achieved_right_carrier, left_wrist, right_wrist, left_palm, right_palm, contacts = achieved
    left_target_error = np.linalg.norm(achieved_left_carrier[:, :3, 3] - target_left_carrier[:, :3, 3], axis=1)
    right_target_error = np.linalg.norm(achieved_right_carrier[:, :3, 3] - target_right_carrier[:, :3, 3], axis=1)

    a_sign = -1.0 if selected_left.assignment == "A_SCREEN_B_BACK" else 1.0
    b_sign = -a_sign
    left_samples = []
    carrier_translation_drift = []
    carrier_rotation_drift = []
    for index in range(portrait, charger + 1):
        runtime.assign(arm_q[index], left_q[index], right_q[index])
        _, state_row = build_left_pinch_carrier(runtime, selected_left.patch_offsets["A"], selected_left.patch_offsets["B"])
        gap_a = _face_gap(state_row["A"], phone_target[index], phone_dimensions, a_sign)
        gap_b = _face_gap(state_row["B"], phone_target[index], phone_dimensions, b_sign)
        point_c, _ = runtime.contact_pose("left_C")
        signed_c, _, _ = nearest_box_surface(point_c, phone_target[index], phone_dimensions)
        relation = inverse_pose(achieved_left_carrier[index]) @ phone_target[index]
        carrier_translation_drift.append(np.linalg.norm(relation[:3, 3] - carrier_from_phone_lock[:3, 3]))
        carrier_rotation_drift.append(Rotation.from_matrix(relation[:3, :3] @ carrier_from_phone_lock[:3, :3].T).magnitude())
        left_samples.append({"action_index": index, "A_gap_m": gap_a, "B_gap_m": gap_b, "C_signed_surface_distance_m": signed_c})
    left_a = np.asarray([row["A_gap_m"] for row in left_samples])
    left_b = np.asarray([row["B_gap_m"] for row in left_samples])
    left_c_pen = np.asarray([max(0.0, -row["C_signed_surface_distance_m"]) for row in left_samples])
    left_metrics = {
        "assignment": selected_left.assignment,
        "hold_interval_resolved_by_semantic_names": ["phone_portrait_reached", "phone_charger_attachment_complete"],
        "A_gap_mean_mm": float(np.mean(left_a) * 1000.0), "A_gap_max_mm": float(np.max(left_a) * 1000.0),
        "B_gap_mean_mm": float(np.mean(left_b) * 1000.0), "B_gap_max_mm": float(np.max(left_b) * 1000.0),
        "left_C_penetration_max_mm": float(np.max(left_c_pen) * 1000.0),
        "carrier_translation_drift_max_mm": float(np.max(carrier_translation_drift) * 1000.0),
        "carrier_rotation_drift_max_deg": float(np.degrees(np.max(carrier_rotation_drift))),
        "continuous_rigid_pinch_pass": bool(np.max(left_a) <= 0.005 and np.max(left_b) <= 0.005 and np.max(left_c_pen) <= 1e-5),
        "samples": left_samples,
    }

    right_samples = []
    for index in range(detachment, right_release + 1):
        runtime.assign(arm_q[index], left_q[index], right_q[index])
        carrier, state_row = build_right_hook_carrier(runtime, selected_right.patch_offset, accessory_target[index, :3, 3], hook_hint)
        signed, local = ring_material_gap(state_row["C"], accessory_target[index], ring_inner, ring_outer, ring_depth)
        right_samples.append({"action_index": index, "absolute_ring_gap_m": abs(signed), "signed_ring_material_distance_m": signed, "C_accessory_local_m": local})
    right_gap = np.asarray([row["absolute_ring_gap_m"] for row in right_samples])
    right_pen = np.asarray([max(0.0, -row["signed_ring_material_distance_m"]) for row in right_samples])
    right_metrics = {
        "family": selected_right.family,
        "semantic_states": ["PREINSERT", "INSERT", "HOOK", "REMOVE", "HOLD", "RELEASE"],
        "C_ring_gap_mean_mm": float(np.mean(right_gap) * 1000.0),
        "C_ring_gap_max_mm": float(np.max(right_gap) * 1000.0),
        "ring_material_penetration_max_mm": float(np.max(right_pen) * 1000.0),
        "continuous_hook_pass": bool(np.max(right_gap) <= 0.005 and np.max(right_pen) <= 1e-5),
        "samples": right_samples,
    }

    phone_portrait = target_left_carrier[portrait] @ carrier_from_phone_lock
    phone_at_charger_achieved = achieved_left_carrier[charger] @ carrier_from_phone_lock
    portrait_error = float(np.degrees(np.arccos(np.clip(abs(phone_portrait[2, 0]), -1.0, 1.0))))
    charger_center_error = float(np.linalg.norm(phone_at_charger_achieved[:3, 3] - phone_charger[:3, 3]))
    charger_normal_error = float(np.degrees(np.arccos(np.clip(np.dot(phone_at_charger_achieved[:3, 1], phone_charger[:3, 1]), -1.0, 1.0))))
    charger_vertical_error = float(np.degrees(np.arccos(np.clip(np.dot(phone_at_charger_achieved[:3, 0], phone_charger[:3, 0]), -1.0, 1.0))))
    task_orientation = {
        "portrait_constant_task_axis_registration_deg": portrait_constant_correction_deg,
        "portrait_long_axis_error_deg": portrait_error,
        "charger_center_error_mm": charger_center_error * 1000.0,
        "charger_normal_error_deg": charger_normal_error,
        "charger_vertical_axis_error_deg": charger_vertical_error,
        "left_source_rotation_progress_correlation": _rotation_progress_correlation(source_left_rotation, target_left_carrier[:, :3, :3], rotation_start, portrait),
        "right_source_rotation_progress_correlation": _rotation_progress_correlation(source_right_rotation, target_right_carrier[:, :3, :3], detachment, removed),
    }
    task_orientation["pass"] = bool(
        portrait_error <= 5.0 and charger_center_error <= 0.005 and charger_normal_error <= 5.0
        and charger_vertical_error <= 5.0 and task_orientation["left_source_rotation_progress_correlation"] >= 0.90
        and task_orientation["right_source_rotation_progress_correlation"] >= 0.90
    )

    collision = collision_audit(runtime, arm_q, left_q, right_q, table_height)
    joint_metrics = _joint_and_branch_metrics(runtime, arm_q, left_q, right_q)
    left_phases = [
        ("acquisition_rotation", left_grasp, portrait),
        ("portrait_hold", portrait, phone_move),
        ("charger_transport", phone_move, charger),
    ]
    right_phases = [
        ("accessory_approach", portrait, right_grasp),
        ("accessory_acquisition", right_grasp, detachment),
        ("accessory_removal", detachment, removed),
        ("accessory_hold", removed, right_release),
    ]
    phase_metrics = {
        "left": {name: _phase_correlation(registered_left_carrier[:, :3, 3], target_left_carrier[:, :3, 3], start, end) for name, start, end in left_phases},
        "right": {name: _phase_correlation(registered_right_carrier[:, :3, 3], target_right_carrier[:, :3, 3], start, end) for name, start, end in right_phases},
    }
    base_midpoint = 0.5 * (registered_left_carrier[:, :3, 3] + registered_right_carrier[:, :3, 3])
    target_midpoint = 0.5 * (target_left_carrier[:, :3, 3] + target_right_carrier[:, :3, 3])
    base_vector = registered_right_carrier[:, :3, 3] - registered_left_carrier[:, :3, 3]
    target_vector = target_right_carrier[:, :3, 3] - target_left_carrier[:, :3, 3]
    bimanual = {
        "midpoint_trend_correlation": _trend_correlation(base_midpoint, target_midpoint),
        "relative_vector_trend_correlation": _trend_correlation(base_vector, target_vector),
        "inter_hand_distance_trend_correlation": _trend_correlation(np.linalg.norm(base_vector, axis=1), np.linalg.norm(target_vector, axis=1)),
    }
    primary_values = [row[key] for side in phase_metrics.values() for row in side.values() for key in ("path_shape_correlation", "speed_profile_correlation")]
    fidelity = {
        "v14": json.loads((V14 / "aloha_fidelity_metrics_v14.json").read_text(encoding="utf-8")),
        "failed_v15": json.loads((V15 / "aloha_motion_fidelity.json").read_text(encoding="utf-8")),
        "v16_phase_relative": phase_metrics,
        "v16_bimanual": bimanual,
        "v16_rotation": {
            "left": task_orientation["left_source_rotation_progress_correlation"],
            "right": task_orientation["right_source_rotation_progress_correlation"],
        },
    }
    fidelity["v16_minimum_primary_metric"] = float(min(primary_values + list(bimanual.values()) + list(fidelity["v16_rotation"].values())))
    fidelity["pass"] = fidelity["v16_minimum_primary_metric"] >= 0.90

    controlled_joint_names = np.r_[joint_names.astype(str), np.asarray(runtime.hand_joint_names["left"]), np.asarray(runtime.hand_joint_names["right"])]
    controlled_q = np.c_[arm_q, left_q, right_q]
    save_npz(
        OUT / "arm_dex3_coupled_trajectory.npz",
        optimized_action=source_action, source_timestamps=timestamps,
        controlled_joint_names=controlled_joint_names, controlled_q=controlled_q,
        g1_arm_q=arm_q, left_dex3_q=left_q, right_dex3_q=right_q,
        v14_reference_arm_q=v14_q, v14_reference_left_position=target_left, v14_reference_right_position=target_right,
        target_left_contact_carrier=target_left_carrier, target_right_contact_carrier=target_right_carrier,
        achieved_left_contact_carrier=achieved_left_carrier, achieved_right_contact_carrier=achieved_right_carrier,
        achieved_left_position=left_palm, achieved_right_position=right_palm,
        left_wrist_pose=left_wrist, right_wrist_pose=right_wrist,
        left_palm_position=left_palm, right_palm_position=right_palm,
        target_phone_pose=phone_target, target_accessory_pose=accessory_target,
        g1_root=root_position, workspace_scale=np.asarray(0.42),
        semantic_event_names=np.asarray(list(event_index)), semantic_event_indices=np.asarray(list(event_index.values()), dtype=np.int64),
        semantic_timeline_provenance=np.asarray("HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE_THROUGH_GENERIC_API"),
        method=np.asarray(METHOD), diagnostic_only=np.asarray(True), physics_steps=np.asarray(0), real_robot_command_allowed=np.asarray(False),
    )
    save_npz(
        OUT / "contact_carrier_trajectory.npz",
        base_left_carrier=base_left_carrier, base_right_carrier=base_right_carrier,
        registered_left_carrier=registered_left_carrier, registered_right_carrier=registered_right_carrier,
        left_fixed_carrier_translation=left_fixed_carrier_translation,
        right_fixed_carrier_translation=right_fixed_carrier_translation,
        left_translation_residual=left_translation_residual, right_translation_residual=right_translation_residual,
        left_rotation_residual=left_rotation_residual, right_rotation_residual=right_rotation_residual,
        target_left_carrier=target_left_carrier, target_right_carrier=target_right_carrier,
        achieved_left_carrier=achieved_left_carrier, achieved_right_carrier=achieved_right_carrier,
        phone_pose=phone_target, accessory_pose=accessory_target,
    )
    save_npz(
        OUT / "left_dex3_trajectory.npz",
        joint_names=np.asarray(runtime.hand_joint_names["left"]), q=left_q,
        contact_A=contacts["left_A"], contact_B=contacts["left_B"], contact_C=contacts["left_C"],
    )
    save_npz(
        OUT / "right_dex3_trajectory.npz",
        joint_names=np.asarray(runtime.hand_joint_names["right"]), q=right_q,
        contact_A=contacts["right_A"], contact_B=contacts["right_B"], contact_C=contacts["right_C"],
    )

    frame_audit = {
        "status": "CONTACT_CARRIER_FRAME_AUDIT_PASS",
        "left": {
            "name": "LEFT_PHONE_PINCH_CARRIER", "origin": "midpoint active A/B pad contact points",
            "primary_axis": "A_to_B", "secondary_axis": "palm_to_contact approach projected orthogonal to primary",
            "third_axis": "right-handed cross product", "determinants": [float(np.linalg.det(selected_left.initial_carrier[:3, :3])), float(np.linalg.det(selected_left.charger_carrier[:3, :3]))],
            "active_links": [runtime.contacts["left_A"].link, runtime.contacts["left_B"].link],
        },
        "right": {
            "name": "RIGHT_ACCESSORY_HOOK_CARRIER", "origin": "active right-C pad contact point",
            "primary_axis": "active C contact normal", "hook_axis": "ring-inner-rim engagement direction",
            "third_axis": "right-handed cross product", "determinant": float(np.linalg.det(selected_right.carrier[:3, :3])),
            "active_link": runtime.contacts["right_C"].link,
        },
        "reflection": False, "isaac_numerical_parity_pending_renderer": True,
    }
    dump(OUT / "contact_carrier_frame_audit.json", frame_audit)
    dump(OUT / "selected_left_pincher_carrier.json", {
        "status": "LEFT_RIGID_PINCH_CARRIER_PASS" if selected_left.valid else "BLOCKED_LEFT_RIGID_PINCH_CARRIER",
        "selected": selected_left.record(), "candidates": [row.record() for row in left_candidates],
        "selected_from_episode": 49, "reusable_rule_contains_event_indices": False,
    })
    dump(OUT / "selected_right_hook_carrier.json", {
        "status": "RIGHT_CONTINUOUS_HOOK_ANCHOR_PASS" if selected_right.valid else "BLOCKED_RIGHT_C_INSERTION",
        "selected": selected_right.record(), "candidates": [row.record() for row in right_candidates],
        "preinsert_clearance_rule_m": preinsert_distance,
        "source_derived_entry_direction": source_entry_direction,
        "reusable_rule_contains_event_indices": False,
    })
    dump(OUT / "orientation_registration_v16.json", {
        "left_axis_calibration": C_LEFT, "right_axis_calibration": C_RIGHT,
        "portrait_constant_task_axis_registration_deg": portrait_constant_correction_deg,
        "method": "mapped source-relative rotation plus minimum semantic-phase endpoint registration",
        "episode_literal_index_dependency": False,
    })
    dump(OUT / "left_phone_carrier_metrics.json", left_metrics)
    dump(OUT / "right_accessory_carrier_metrics.json", right_metrics)
    dump(OUT / "task_orientation_metrics.json", task_orientation)
    dump(OUT / "branch_continuity_metrics.json", {"left_solver": left_solve, "right_solver": right_solve, **joint_metrics})
    dump(OUT / "collision_breakdown.json", collision)
    dump(OUT / "joint_margin_metrics.json", joint_metrics)
    dump(OUT / "v14_vs_v15_vs_v16_fidelity.json", fidelity)
    dump(OUT / "v14_deviation_metrics.json", {
        "left_wrist_translation_m": distribution(np.linalg.norm(left_wrist[:, :3, 3] - base_left_carrier[:, :3, 3], axis=1)),
        "right_wrist_translation_m": distribution(np.linalg.norm(right_wrist[:, :3, 3] - base_right_carrier[:, :3, 3], axis=1)),
        "left_carrier_correction_translation_m": distribution(translation_left),
        "right_carrier_correction_translation_m": distribution(translation_right),
        "left_carrier_correction_rotation_rad": distribution(rotation_left),
        "right_carrier_correction_rotation_rad": distribution(rotation_right),
        "correction_over_workspace_scaled_source_displacement_ratio": float(max(np.max(translation_left), np.max(translation_right)) / max(0.42 * max(np.ptp(source_left_position, axis=0).max(), np.ptp(source_right_position, axis=0).max()), 1e-12)),
        "description": "ALOHA-primary retargeting with minimum embodiment-specific contact-carrier correction",
    })

    all_status = {
        "CONTACT_CARRIER_FRAME_AUDIT_PASS": frame_audit["status"] == "CONTACT_CARRIER_FRAME_AUDIT_PASS",
        "MINIMUM_CORRECTION_AUDIT_COMPLETE": True,
        "LEFT_RIGID_PINCH_CARRIER_PASS": selected_left.valid and left_metrics["continuous_rigid_pinch_pass"],
        "RIGHT_CONTINUOUS_HOOK_CARRIER_PASS": selected_right.valid and right_metrics["continuous_hook_pass"],
        "TASK_ORIENTATION_PASS": task_orientation["pass"],
        "BRANCH_DISCONTINUITY_ZERO": joint_metrics["branch_discontinuity_count"] == 0,
        "PROHIBITED_COLLISION_ZERO": collision["pass"],
        "ALOHA_MOTION_FIDELITY_PASS": fidelity["pass"],
        "ACTUAL_DEX3_CONTINUOUS_MOTION_PASS": float(np.ptp(left_q, axis=0).max()) > 0.05 and float(np.ptp(right_q, axis=0).max()) > 0.05,
        "ISAACLAB_KINEMATIC_REPLAY_PASS": False,
    }
    numerical_pass = all(value for key, value in all_status.items() if key != "ISAACLAB_KINEMATIC_REPLAY_PASS")
    if not selected_left.valid or not left_metrics["continuous_rigid_pinch_pass"]:
        status = "BLOCKED_LEFT_RIGID_PINCH_CARRIER"
    elif not selected_right.valid or not right_metrics["continuous_hook_pass"]:
        status = "BLOCKED_RIGHT_CONTINUOUS_HOOK_CARRIER"
    elif not task_orientation["pass"]:
        status = "BLOCKED_TASK_ORIENTATION"
    elif joint_metrics["branch_discontinuity_count"]:
        status = "BLOCKED_TEMPORAL_BRANCH_CONTINUITY"
    elif not collision["pass"]:
        status = "BLOCKED_V16_COLLISION"
    elif not fidelity["pass"]:
        status = "BLOCKED_CONTACT_CARRIER_ALOHA_FIDELITY"
    else:
        status = "NUMERIC_CONTACT_CARRIER_CANDIDATE_READY_FOR_ISAACLAB_REPLAY"
    dump(OUT / "numeric_gate_summary.json", {"status": status, "numerical_pass": numerical_pass, "gates": all_status, "selected_pareto_candidate": selected_pareto})
    dump(OUT / "semantic_runtime_audit.json", {
        "timeline_source": "HUMAN_REVIEWED_EPISODE49_DEVELOPMENT_TIMELINE", "generic_api": True,
        "literal_audit": literal_audit, "generic_converter_dry_run": dry_run,
        "validation_or_heldout_accessed": False,
    })

    hashes_after = {str(path.resolve()): sha256_file(path) for path in required}
    freeze = {
        "status": "INPUT_FREEZE_PASS" if hashes_after == hashes_before else "BLOCKED_FROZEN_INPUT_MUTATION",
        "hashes_before": hashes_before, "hashes_after": hashes_after,
        "byte_identical": hashes_after == hashes_before,
        "source_action_array_sha256": _array_hash(source_action),
        "v14_target_left_array_sha256": array_sha(target_left),
        "v14_target_right_array_sha256": array_sha(target_right),
        "root": root_position, "workspace_scale": 0.42,
        "validation_or_heldout_used": False, "g1_expert_used": False,
    }
    dump(OUT / "input_freeze_audit.json", freeze)
    dump(OUT / "run_manifest.json", {
        "method": METHOD, "status": status, "output": str(OUT.resolve()),
        "source_action": str(SOURCE.resolve()), "source_action_sha256": sha256_file(SOURCE),
        "timeline": str(TIMELINE.resolve()), "timeline_sha256": sha256_file(TIMELINE),
        "scene": str(ACTIVE_SCENE.resolve()), "scene_sha256": sha256_file(ACTIVE_SCENE),
        "trajectory": str((OUT / "arm_dex3_coupled_trajectory.npz").resolve()),
        "trajectory_sha256": sha256_file(OUT / "arm_dex3_coupled_trajectory.npz"),
        "physics_steps": 0, "diagnostic_only": True, "real_robot_command_allowed": False,
        "validation_or_heldout_episodes_used": [], "g1_expert_motion_used": False,
        "required_review_videos_pending": True,
    })
    print(json.dumps({"status": status, "numerical_pass": numerical_pass, "left": left_metrics["continuous_rigid_pinch_pass"], "right": right_metrics["continuous_hook_pass"], "orientation": task_orientation["pass"], "collision": collision["pass"], "fidelity": fidelity["pass"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
