#!/usr/bin/env python3
"""Prepare the controlled standardized-grasp DEV35 A/B physical experiment.

The only method-dependent operation is ``RepresentationBuilder.build``.  Both
outputs are cut at the same source-derived stable LEFT hold boundary, expressed
as SE(3) increments from that boundary, and left-multiplied by one persisted
physically qualified G1/Dex3 grasp state.  Everything after that representation
switch is common and fail-closed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/standardized_grasp_ab_dev35"
RESET = ROOT / "outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4"
REGISTRATION = ROOT / "outputs/final_episode_registered_eval35/00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
INTENT = ROOT / "outputs/final_direct_physical_eval35/00_preparation/COMMON_SOURCE_TASK_INTENT_EVAL35.json"
QUALIFICATION = ROOT / "outputs/final_episode_registered_eval35/00_qualification/FINAL_COMMON_DEX3_GRASP_QUALIFICATION.json"
QUAL_TRACE = ROOT / "outputs/final_episode_registered_eval35/00_qualification/solver80_requalification/scripted_full_01/event_log.npz"
JOINT_CONTRACT = ROOT / "outputs/doll_handoff_dataset_b_final/retargeted_actions/freeze_manifest.json"
PHYSICS_CONFIG = ROOT / "configs/dex3_simple_graspable_doll_grasp_v2.json"

from tools.common_execution_layer import Dex3Primitive, read_json
from tools.common_execution_isaac_runtime import COMMON_PHYSICAL_CONTROLLER, PHYSICAL_ENVIRONMENT, PHYSICS_CONFIG as COMMON_PHYSICS_CONFIG
from tools.common_g1_morphology_adapter import CommonG1MorphologyAdapter
from tools.direct_physical_execution_layer import authoritative_joint_limits
from tools.doll_handoff_retargeting.common import branch_flags, load_common_config, load_scene
from tools.doll_handoff_retargeting.events import EpisodeEvents
from tools.doll_handoff_retargeting.models import ALOHAKinematics, G1Kinematics
from tools.doll_handoff_retargeting.retarget import RepresentationBuilder, SharedTemporalIK
from tools.doll_handoff_retargeting.source import fixed_list_numpy


CANONICAL_JOINT_NAMES = tuple(read_json(JOINT_CONTRACT)["joint_names"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def quaternion_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64) / np.linalg.norm(first)
    b = np.asarray(second, dtype=np.float64) / np.linalg.norm(second)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def qualified_initial_state(g1: G1Kinematics) -> dict[str, Any]:
    qualification = read_json(QUALIFICATION)
    if qualification.get("status") not in {
        "READY_TO_FREEZE", "PHYSICAL_CONTACT_QUALIFICATION_READY_TO_FREEZE"
    }:
        raise RuntimeError("authoritative physical grasp qualification is not complete")
    aggregate = qualification["aggregate"]
    for key in ("left_standalone_passes", "right_standalone_passes", "natural_release_passes", "scripted_full_task_passes"):
        if int(aggregate[key]) != 3:
            raise RuntimeError(f"qualification does not prove 3/3: {key}")
    if int(aggregate["commanded_hard_limit_violations"]) or int(aggregate["measured_hard_limit_violations"]):
        raise RuntimeError("qualified grasp has a hard-limit violation")

    with np.load(QUAL_TRACE, allow_pickle=False) as trace:
        names = tuple(trace["joint_names"].astype(str).tolist())
        if names != CANONICAL_JOINT_NAMES:
            raise RuntimeError("qualification trace joint ordering changed")
        control = np.asarray(trace["control_frame"], dtype=np.int64)
        stage = trace["stage"].astype(str)
        table = np.asarray(trace["table_contact_force_n"], dtype=np.float64)
        forces = np.column_stack(
            [
                trace["left_thumb_force_n"],
                trace["left_index_force_n"],
                trace["left_middle_force_n"],
            ]
        ).astype(np.float64)
        eligible: list[int] = []
        for frame in np.unique(control):
            indices = np.flatnonzero(control == frame)
            valid = (
                (stage[indices] == "HOLD_ELEVATED")
                & (table[indices] <= 0.05)
                & (np.sum(forces[indices] >= 0.05, axis=1) >= 2)
            )
            if int(np.count_nonzero(valid)) >= 6:
                eligible.append(int(frame))
        runs: list[list[int]] = []
        for frame in eligible:
            if not runs or frame != runs[-1][-1] + 1:
                runs.append([frame])
            else:
                runs[-1].append(frame)
        stable = next((run for run in runs if len(run) >= 30), None)
        if stable is None:
            raise RuntimeError("no persisted one-second table-free qualified LEFT hold")
        selected_control = stable[15]
        selected_rows = np.flatnonzero(control == selected_control)
        selected = int(selected_rows[-1])
        q = np.asarray(trace["measured_q_rad"][selected], dtype=np.float64)
        hold = np.asarray(trace["EXECUTED_COMMAND"][selected, 14:21], dtype=np.float64)
        object_position = np.asarray(trace["object_position_world_m"][selected], dtype=np.float64)
        object_quaternion = np.asarray(trace["object_quaternion_xyzw"][selected], dtype=np.float64)
        selected_forces = forces[selected]

    lower, upper, _ = authoritative_joint_limits(read_json(JOINT_CONTRACT))
    if np.any(q < lower - 1e-6) or np.any(q > upper + 1e-6):
        raise RuntimeError("persisted qualification state is outside authoritative limits")
    g1.assign(q[:14], canonical_to_g1_hand(g1, q[14:21], "left"), canonical_to_g1_hand(g1, q[21:28], "right"))
    shared_wrist = {side: g1.wrist_pose(side).copy() for side in ("left", "right")}
    return {
        "schema_version": "standardized_physical_grasp_initial_state_v1",
        "qualification": str(QUALIFICATION),
        "qualification_sha256": sha256(QUALIFICATION),
        "source_trace": str(QUAL_TRACE),
        "source_trace_sha256": sha256(QUAL_TRACE),
        "selection_rule": "six-of-eight valid physics samples per control frame; first >=30-frame HOLD_ELEVATED run; control frame at run_start+15; last physics substep",
        "selected_control_frame": selected_control,
        "selected_physics_row": selected,
        "joint_names": list(CANONICAL_JOINT_NAMES),
        "measured_q_rad": q,
        "left_hold_target_q_rad": hold,
        "object_position_world_m": object_position,
        "object_quaternion_xyzw": object_quaternion,
        "object_linear_velocity_m_s": [0.0, 0.0, 0.0],
        "object_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "left_contact_forces_n": {
            "thumb": selected_forces[0], "index": selected_forces[1], "middle": selected_forces[2]
        },
        "table_contact_force_n": 0.0,
        "table_free": True,
        "mechanically_retained": True,
        "attachment_used": False,
        "object_pose_writes_after_initialization": 0,
        "shared_wrist_pose_model": shared_wrist,
    }


def canonical_to_g1_hand(g1: G1Kinematics, values: np.ndarray, side: str) -> np.ndarray:
    names = CANONICAL_JOINT_NAMES[14:21] if side == "left" else CANONICAL_JOINT_NAMES[21:28]
    by_name = dict(zip(names, np.asarray(values, dtype=np.float64)))
    return np.asarray([by_name[name] for name in g1.hand_joint_names[side]], dtype=np.float64)


def event_stub(index: int, source: str, count: int, frames: dict[str, int]) -> EpisodeEvents:
    ownership = np.full(count, "NO_OWNER", dtype="U20")
    hold = int(frames["LEFT_STABLE_HOLD"])
    receive = int(frames["RIGHT_CLOSE_ONSET"])
    release = int(frames["LEFT_RELEASE"])
    final = int(frames["RIGHT_FINAL_RELEASE"])
    ownership[hold:receive] = "LEFT_OWNED"
    ownership[receive:release] = "DUAL_CONTACT"
    ownership[release:final] = "RIGHT_OWNED"
    ownership[final:] = "RELEASED"
    zeros = np.zeros(count, dtype=np.float64)
    return EpisodeEvents(
        episode_index=index,
        source_name=source,
        frames={}, transitions={},
        smoothed_gripper={"left": zeros, "right": zeros},
        binary_labels={"left": np.zeros(count, dtype="U8"), "right": np.zeros(count, dtype="U8")},
        semantic_labels={"left": np.zeros(count, dtype="U8"), "right": np.zeros(count, dtype="U8")},
        ownership_labels=ownership,
        inter_hand_distance_m=zeros,
        handoff_window=(receive, release), anomalies=(), source_semantic_valid=True,
    )


def rebase_targets(raw: dict[str, Any], t0: int, shared: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        position = np.asarray(raw[f"{side}_wrist_position"], dtype=np.float64)[t0:]
        rotation = np.asarray(raw[f"{side}_wrist_rotation"], dtype=np.float64)[t0:]
        p0, r0 = position[0], rotation[0]
        delta_position = (position - p0) @ r0
        delta_rotation = np.einsum("ij,tjk->tik", r0.T, rotation)
        shared_pose = np.asarray(shared[side], dtype=np.float64)
        result[f"{side}_wrist_position"] = delta_position @ shared_pose[:3, :3].T + shared_pose[:3, 3]
        result[f"{side}_wrist_rotation"] = np.einsum("ij,tjk->tik", shared_pose[:3, :3], delta_rotation)
        result[f"raw_{side}_wrist_position"] = position
        result[f"raw_{side}_wrist_rotation"] = rotation
    return result


def shape_error(raw: dict[str, np.ndarray], rebased: dict[str, np.ndarray]) -> dict[str, float]:
    translation, angle = [], []
    for side in ("left", "right"):
        rp = raw[f"raw_{side}_wrist_position"]
        rr = raw[f"raw_{side}_wrist_rotation"]
        ep = rebased[f"{side}_wrist_position"]
        er = rebased[f"{side}_wrist_rotation"]
        raw_steps = np.einsum("tij,tj->ti", rr[:-1].transpose(0, 2, 1), np.diff(rp, axis=0))
        eval_steps = np.einsum("tij,tj->ti", er[:-1].transpose(0, 2, 1), np.diff(ep, axis=0))
        translation.extend(np.linalg.norm(raw_steps - eval_steps, axis=1))
        raw_delta = np.einsum("tji,tjk->tik", rr[:-1], rr[1:])
        eval_delta = np.einsum("tji,tjk->tik", er[:-1], er[1:])
        angle.extend(Rotation.from_matrix(np.einsum("tij,tkj->tik", raw_delta, eval_delta)).magnitude())
    return {
        "maximum_relative_translation_shape_error_mm": 1000.0 * float(np.max(translation, initial=0.0)),
        "maximum_relative_rotation_shape_error_deg": float(np.degrees(np.max(angle, initial=0.0))),
    }


def common_intent(count: int, source_frames: dict[str, int], t0: int) -> tuple[np.ndarray, dict[str, int]]:
    relative = {
        "GRASP_CONFIRMED": 0,
        "RIGHT_ACQUIRE_BEGIN": max(0, int(source_frames["RIGHT_CLOSE_ONSET"]) - t0),
        "LEFT_RELEASE_BOUNDARY": max(0, int(source_frames["LEFT_RELEASE"]) - t0),
        "FINAL_RELEASE": max(0, int(source_frames["RIGHT_FINAL_RELEASE"]) - t0),
    }
    if not 0 < relative["RIGHT_ACQUIRE_BEGIN"] <= relative["LEFT_RELEASE_BOUNDARY"] <= relative["FINAL_RELEASE"] < count:
        raise RuntimeError(f"invalid post-grasp event order: {relative} count={count}")
    values = np.full(count, "LEFT_HOLD_INTENT", dtype="U24")
    values[relative["RIGHT_ACQUIRE_BEGIN"]:relative["LEFT_RELEASE_BOUNDARY"]] = "HANDOFF_INTENT"
    values[relative["LEFT_RELEASE_BOUNDARY"]:relative["FINAL_RELEASE"]] = "RIGHT_HOLD_INTENT"
    values[relative["FINAL_RELEASE"]:] = "FINAL_RELEASE_INTENT"
    return values, relative


def hand_sequences(g1: G1Kinematics, initial: dict[str, Any], intent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    primitive = Dex3Primitive.from_frozen_dependencies(
        read_json(COMMON_PHYSICS_CONFIG), read_json(PHYSICAL_ENVIRONMENT), read_json(COMMON_PHYSICAL_CONTROLLER)
    )
    left_hold = canonical_to_g1_hand(g1, np.asarray(initial["measured_q_rad"])[14:21], "left")
    right_open = canonical_to_g1_hand(g1, primitive.right_open, "right")
    right_close = canonical_to_g1_hand(g1, primitive.right_full_close, "right")
    left_open = canonical_to_g1_hand(g1, primitive.left_open, "left")
    left = np.repeat(left_hold[None], len(intent), axis=0)
    right = np.repeat(right_open[None], len(intent), axis=0)
    handoff = int(np.flatnonzero(intent == "HANDOFF_INTENT")[0])
    close_length = max(1, primitive.preshape_frames + primitive.close_frames)
    for offset in range(close_length):
        frame = handoff + offset
        if frame >= len(intent): break
        alpha = min(1.0, (offset + 1) / close_length)
        right[frame] = (1.0 - alpha) * right_open + alpha * right_close
    right[handoff + close_length:] = right_close
    release_frames = np.flatnonzero(intent == "RIGHT_HOLD_INTENT")
    if len(release_frames):
        start = int(release_frames[0])
        for offset in range(primitive.release_frames):
            frame = start + offset
            if frame >= len(intent): break
            alpha = min(1.0, (offset + 1) / max(1, primitive.release_frames))
            left[frame] = (1.0 - alpha) * left_hold + alpha * left_open
        left[start + primitive.release_frames:] = left_open
    final_frames = np.flatnonzero(intent == "FINAL_RELEASE_INTENT")
    if len(final_frames):
        start = int(final_frames[0])
        for offset in range(primitive.release_frames):
            frame = start + offset
            if frame >= len(intent): break
            alpha = min(1.0, (offset + 1) / max(1, primitive.release_frames))
            right[frame] = (1.0 - alpha) * right_close + alpha * right_open
        right[start + primitive.release_frames:] = right_open
    return left, right


def solve_common(
    common: dict[str, Any], g1: G1Kinematics, targets: dict[str, np.ndarray],
    initial: dict[str, Any], intent: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    shared_q = np.asarray(initial["measured_q_rad"], dtype=np.float64)[:14]
    solver = SharedTemporalIK(common, g1, shared_q, natural_arm_enabled=True)
    count = len(intent)
    q = np.repeat(shared_q[None], count, axis=0)
    accepted = np.zeros(count, dtype=bool)
    accepted[0] = True
    reports: list[dict[str, Any]] = [{"frame": 0, "accepted": True, "shared_initial_state": True}]
    first_rejected: int | None = None
    first_persistent_infeasible: int | None = None
    rejected_run_start: int | None = None
    rejection_debounce_frames = 15  # 0.5 s at the authoritative 30 Hz cadence.
    previous = shared_q.copy()
    previous2 = shared_q.copy()
    for frame in range(1, count):
        iterations = int(common["shared_temporal_ik"]["max_iterations_per_frame"])
        step = float(common["shared_temporal_ik"]["max_frame_joint_step_rad"])
        predicted = np.clip(2.0 * previous - previous2, g1.arm_limits[:, 0], g1.arm_limits[:, 1])
        candidates = [
            solver._solve_seed(targets, frame, previous, previous, previous2, iterations, step)
        ]
        if not bool(candidates[0][1]["accepted"]):
            candidates.extend(
                solver._solve_seed(targets, frame, seed, previous, previous2, iterations, step)
                for seed in (predicted, shared_q)
            )
        selected = min(
            candidates,
            key=lambda item: (
                0 if item[1]["accepted"] else 1,
                item[1]["position_error_max_m"] / float(common["shared_temporal_ik"]["position_tolerance_m"])
                + item[1]["orientation_error_max_rad"] / float(common["shared_temporal_ik"]["orientation_tolerance_rad"]),
                float(np.linalg.norm(item[0] - previous)),
            ),
        )
        reports.append({"frame": frame, **selected[1]})
        q[frame] = selected[0]
        accepted[frame] = bool(selected[1]["accepted"])
        if not accepted[frame] and first_rejected is None:
            first_rejected = frame
        if accepted[frame]:
            rejected_run_start = None
        elif rejected_run_start is None:
            rejected_run_start = frame
        previous2, previous = previous, q[frame].copy()
        if (
            rejected_run_start is not None
            and frame - rejected_run_start + 1 >= rejection_debounce_frames
        ):
            first_persistent_infeasible = rejected_run_start
            q[first_persistent_infeasible:] = q[max(0, first_persistent_infeasible - 1)]
            accepted[first_persistent_infeasible:] = False
            break

    left_hand, right_hand = hand_sequences(g1, initial, intent)
    adapter = CommonG1MorphologyAdapter(common, g1, shared_q)
    before = adapter._collision_metrics(q, left_hand, right_hand, 30.0)
    collision_metadata: dict[str, Any] = {
        "status": "FAIL_CLOSED_AT_FIRST_PROHIBITED_SELF_COLLISION"
        if int(before["hard_collision_frame_count"])
        else "NOT_REQUIRED",
        "target_position_modified": False,
        "target_orientation_modified": False,
    }

    branches = branch_flags(
        q,
        float(common["validation"]["branch_absolute_step_norm_rad"]),
        float(common["validation"]["branch_local_multiplier"]),
    )
    first_branch = int(np.flatnonzero(branches)[0] + 1) if np.any(branches) else None
    after = adapter._collision_metrics(q, left_hand, right_hand, 30.0)
    hard_frames = list(map(int, after["hard_frames"]))
    safety_failure_candidates = [value for value in (first_branch, hard_frames[0] if hard_frames else None) if value is not None]
    safety_failure = min(safety_failure_candidates) if safety_failure_candidates else None
    if safety_failure is not None:
        q[safety_failure:] = q[max(0, safety_failure - 1)]
        accepted[safety_failure:] = False
        branches = branch_flags(
            q,
            float(common["validation"]["branch_absolute_step_norm_rad"]),
            float(common["validation"]["branch_local_multiplier"]),
        )
        after = adapter._collision_metrics(q, left_hand, right_hand, 30.0)

    position, rotation = adapter._pose_arrays(q)
    position_error = np.maximum(
        np.linalg.norm(position["left"] - targets["left_wrist_position"], axis=1),
        np.linalg.norm(position["right"] - targets["right_wrist_position"], axis=1),
    )
    orientation_error = np.maximum(
        Rotation.from_matrix(np.einsum("tij,tkj->tik", targets["left_wrist_rotation"], rotation["left"])).magnitude(),
        Rotation.from_matrix(np.einsum("tij,tkj->tik", targets["right_wrist_rotation"], rotation["right"])).magnitude(),
    )
    lower, upper, _ = authoritative_joint_limits(read_json(JOINT_CONTRACT))
    hard = int(np.count_nonzero((q < lower[:14] - 1e-9) | (q > upper[:14] + 1e-9)))
    active_count = first_persistent_infeasible if first_persistent_infeasible is not None else count
    acceptance_rate = float(np.mean(accepted[:active_count])) if active_count else 0.0
    required_rate = float(common["shared_temporal_ik"]["required_success_rate"])
    first_infeasible = first_persistent_infeasible
    metadata = {
        "schema_version": "standardized_grasp_common_sequential_ik_v1",
        "method_blind": True,
        "method_or_representation_identity_consumed": False,
        "common_shared_initial_q": shared_q,
        "first_rejected_frame": first_rejected,
        "persistent_rejection_debounce_frames": rejection_debounce_frames,
        "target_acceptance_rate": acceptance_rate,
        "required_target_acceptance_rate": required_rate,
        "first_infeasible_frame": first_infeasible,
        "safety_fail_closed_frame": safety_failure,
        "safety_failure_is_fail_closed_hold_not_spatial_rescue": True,
        "accepted_frame_count": int(np.count_nonzero(accepted)),
        "frame_count": count,
        "position_residual_m": {"mean": float(np.mean(position_error)), "p95": float(np.quantile(position_error, .95)), "max": float(np.max(position_error))},
        "orientation_residual_rad": {"mean": float(np.mean(orientation_error)), "p95": float(np.quantile(orientation_error, .95)), "max": float(np.max(orientation_error))},
        "hard_limit_violations": hard,
        "branch_discontinuities": int(np.count_nonzero(branches)),
        "hard_self_collision_frames": int(after["hard_collision_frame_count"]),
        "finite": bool(np.isfinite(q).all()),
        "collision_solver": collision_metadata,
        "frame_reports_until_failure": reports,
    }
    return q, metadata


def build(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    common = load_common_config(RESET / "config/common_config.json")
    scene = load_scene(common)
    aloha = ALOHAKinematics(common, scene)
    g1 = G1Kinematics(common, scene)
    alignment = read_json(RESET / "config/tool_frame_report.json")["source_to_target_axis_alignment"]
    proposed = read_json(RESET / "config/proposed_config.json")
    baseline = read_json(RESET / "config/baseline_workspace_mapping_report.json")
    representation = RepresentationBuilder(common, scene, g1, alignment, proposed, baseline)
    initial = qualified_initial_state(g1)
    initial_path = OUT / "00_control/QUALIFIED_STANDARDIZED_INITIAL_GRASP.json"
    atomic_json(initial_path, initial)

    object_registration = {
        "schema_version": "common_task_frame_registration_v1",
        "status": "STANDARDIZED_GRASP_CONTROL_CONDITION",
        "method_independent": True,
        "episode_independent": True,
        "changes_object_physics": False,
        "base_physics_config": str(PHYSICS_CONFIG),
        "base_physics_config_sha256": sha256(PHYSICS_CONFIG),
        "registered_doll_center_world_xy_m": initial["object_position_world_m"][:2],
        "registered_doll_center_world_z_m": initial["object_position_world_m"][2],
        "registered_doll_orientation_quaternion_xyzw": initial["object_quaternion_xyzw"],
        "source": str(QUAL_TRACE),
        "source_sha256": sha256(QUAL_TRACE),
        "object_pose_writes_after_initialization": 0,
        "task_bin_rule": "reuse frozen canonical 150 mm bin pose; the DEV35 registration manifest declares bin fixed for all 35 episodes",
    }
    object_path = OUT / "00_control/STANDARDIZED_OBJECT_REGISTRATION.json"
    atomic_json(object_path, object_registration)

    registration = read_json(REGISTRATION)
    event_manifest = read_json(INTENT)
    entries = registration["entries"]
    records = event_manifest["records"]
    if len(entries) != 35 or len(records) != 35:
        raise RuntimeError("DEV35 membership is not exactly 35")
    event_by_stable = {row["stable_episode_id"]: row for row in records}
    methods = (("A", "WRIST"), ("B", "INTERACTION"))
    selected_indices = set(args.indices if args.indices else range(35))
    manifest_rows: list[dict[str, Any]] = []

    for entry in entries:
        index = int(entry["eval_index"])
        if index not in selected_indices:
            continue
        source = str(entry["source_recording"])
        stable = str(entry["stable_episode_id"])
        event_row = event_by_stable[stable]
        raw_root = ROOT / "raw_recordings" / source
        parquet_files = sorted((raw_root / "data").rglob("*.parquet"))
        if len(parquet_files) != 1:
            raise RuntimeError(f"source {source} does not resolve to one parquet")
        table = pq.read_table(parquet_files[0])
        state = fixed_list_numpy(table["observation.state"], 14).astype(np.float64)
        if len(state) != int(event_row["frames"]):
            raise RuntimeError(f"source/event frame mismatch: {source}")
        fk = aloha.fk(state)
        source_frames = {key: int(value) for key, value in event_row["event_frames"].items()}
        t0 = source_frames["LEFT_STABLE_HOLD"]
        event = event_stub(index, source, len(state), source_frames)
        for label, mode in methods:
            raw = representation.build(mode, fk, event if mode == "INTERACTION" else None)
            targets = rebase_targets(raw, t0, initial["shared_wrist_pose_model"])
            intent, relative_events = common_intent(len(state) - t0, source_frames, t0)
            q, ik = solve_common(common, g1, targets, initial, intent)
            shape = shape_error(targets, targets)
            if shape["maximum_relative_translation_shape_error_mm"] > 1e-8 or shape["maximum_relative_rotation_shape_error_deg"] > 1e-7:
                raise RuntimeError("SE(3) rebase changed relative trajectory shape")
            command = np.empty((len(q), 28), dtype=np.float64)
            command[:, :14] = q
            command[:, 14:21] = np.asarray(initial["measured_q_rad"])[14:21]
            command[:, 21:28] = np.asarray(initial["measured_q_rad"])[21:28]
            stage = np.asarray([{"LEFT_HOLD_INTENT":"LEFT_TRANSPORT", "HANDOFF_INTENT":"HANDOFF", "RIGHT_HOLD_INTENT":"RIGHT_TRANSPORT", "FINAL_RELEASE_INTENT":"FINAL_RELEASE"}[value] for value in intent], dtype="U24")
            output = OUT / "01_prepared_commands" / label / f"eval_{index:02d}_{stable}.npz"
            atomic_npz(
                output,
                raw_policy_command=command,
                commanded_q_rad=command,
                joint_names=np.asarray(CANONICAL_JOINT_NAMES),
                control_fps_hz=np.asarray(30.0),
                runtime_right_three_digit_gate_required=np.asarray(False),
                common_task_intent=intent,
                stage=stage,
                method=np.asarray(label.lower()),
                eval_index=np.asarray(index, dtype=np.int64),
                provenance=np.asarray("DEV35_STANDARDIZED_GRASP"),
                pregrasp_classifier_used=np.asarray(False),
                wrist_distance_gate_used=np.asarray(False),
                arm_rescue_allowed=np.asarray(False),
                wrist_rescue_allowed=np.asarray(False),
                stable_episode_id=np.asarray(stable),
                source_recording=np.asarray(source),
                common_initial_q_rad=np.asarray(initial["measured_q_rad"], dtype=np.float64),
                standardized_initial_grasp=np.asarray(True),
                standardized_left_hold_target_q_rad=np.asarray(initial["left_hold_target_q_rad"], dtype=np.float64),
                standardized_initial_state_sha256=np.asarray(sha256(initial_path)),
                source_grasp_confirmed_frame=np.asarray(t0, dtype=np.int64),
                common_relative_event_frames_json=np.asarray(json.dumps(relative_events, sort_keys=True)),
                raw_left_wrist_position=targets["raw_left_wrist_position"],
                raw_left_wrist_rotation=targets["raw_left_wrist_rotation"],
                raw_right_wrist_position=targets["raw_right_wrist_position"],
                raw_right_wrist_rotation=targets["raw_right_wrist_rotation"],
                rebased_left_wrist_position=targets["left_wrist_position"],
                rebased_left_wrist_rotation=targets["left_wrist_rotation"],
                rebased_right_wrist_position=targets["right_wrist_position"],
                rebased_right_wrist_rotation=targets["right_wrist_rotation"],
                common_ik_first_infeasible_frame=np.asarray(-1 if ik["first_infeasible_frame"] is None else ik["first_infeasible_frame"], dtype=np.int64),
            )
            row = {
                "eval_index": index,
                "method": label,
                "representation_mode": mode,
                "stable_episode_id": stable,
                "source_recording": source,
                "source_parquet": str(parquet_files[0]),
                "source_parquet_sha256": sha256(parquet_files[0]),
                "source_grasp_confirmed_boundary": t0,
                "relative_event_frames": relative_events,
                "command": str(output),
                "command_sha256": sha256(output),
                "frame_count": len(q),
                "standardized_initial_state_sha256": sha256(initial_path),
                "object_registration_sha256": sha256(object_path),
                "relative_trajectory_shape_preservation": shape,
                "ik": ik,
                "initial_joint_discontinuity_rad": float(np.max(np.abs(q[0] - np.asarray(initial["measured_q_rad"])[:14]))),
                "arm_rescue": False,
                "wrist_rescue": False,
            }
            atomic_json(output.with_suffix(".json"), row)
            manifest_rows.append(row)
            print(f"{label}{index:02d} frames={len(q)} IK_prefix={ik['accepted_frame_count']} failure={ik['first_infeasible_frame']} collision={ik['hard_self_collision_frames']}", flush=True)

    manifest = {
        "schema_version": "standardized_grasp_ab_dev35_preparation_v1",
        "status": "PREPARED" if len(manifest_rows) == 70 else "PARTIAL_PREPARATION",
        "evaluation_label": "DEV35 STANDARDIZED-GRASP PHYSICAL EVALUATION",
        "not_end_to_end_grasp_acquisition": True,
        "previous_results_status": "PRE_STANDARDIZED_GRASP_DIAGNOSTIC_ONLY",
        "single_variable": {"A": "WRIST POST-GRASP TARGET", "B": "INTERACTION POST-GRASP TARGET"},
        "standardized_initial_state": str(initial_path),
        "standardized_initial_state_sha256": sha256(initial_path),
        "object_registration": str(object_path),
        "object_registration_sha256": sha256(object_path),
        "task_bin_registration": "frozen canonical bin fixed for all DEV35 episodes",
        "common_source_timeline": str(INTENT),
        "common_source_timeline_sha256": sha256(INTENT),
        "registration_provenance": str(REGISTRATION),
        "registration_provenance_sha256": sha256(REGISTRATION),
        "A_B_same_initial_state": True,
        "A_B_same_Dex3_controller": True,
        "A_B_same_IK": True,
        "A_B_same_physics": True,
        "records": manifest_rows,
    }
    manifest_path = OUT / "01_prepared_commands/STANDARDIZED_GRASP_AB_COMMAND_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="*")
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
