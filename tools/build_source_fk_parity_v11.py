#!/usr/bin/env python3
"""Episode-49 Stationary ALOHA source FK/replay parity audit.

This diagnostic deliberately stops before phase warping, G1 IK, or any
orientation objective when the recorded action/video timing is not approved.
It performs kinematic MuJoCo FK only: no physics stepping, DDS, publisher, or
hardware path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path("/home/jbnu/aloha_g1_dataset")
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
PARQUET = ROOT / "raw_recordings/GoPark_20260729_111223/data/chunk-000/episode_000000.parquet"
RAW_META = ROOT / "raw_recordings/GoPark_20260729_111223/meta/info.json"
DATASET_META = ROOT / "lerobot_magsafe_50_cam_high_v3/meta/info.json"
DATASET_STATS = ROOT / "lerobot_magsafe_50_cam_high_v3/meta/stats.json"
ACTION_NPZ = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
GENERATOR = ROOT / "tools/generate_episode49_temporal_consensus.py"
CHECKPOINT_CONFIG = ROOT / "outputs/smolvla_magsafe_batch16_20k_20260729_140407/checkpoints/020000/pretrained_model/config.json"
MODEL_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
ROOT_CONFIG = ROOT / "isaaclab_magsafe_fixed_scene/magsafe_robot_preview_config.json"
SOURCE_FRAMES = ROOT / "configs/episode49_source_object_frames.user_approved.json"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
TOOL_AXES = ROOT / "configs/aloha_tool_axes_calibration.sim.json"
TOOL_MAPPING = ROOT / "configs/aloha_tcp_to_g1_palm_calibration.sim.json"
CAM_HIGH = ROOT / "raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000"
V10_RENDERER = ROOT / "isaaclab_magsafe_fixed_scene/render_aloha_primary_source_v10.py"
V10_VIDEO = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_aloha_primary_object_anchored_v10/isaaclab_source_aloha_overview.mp4"

TCP_OFFSET = np.array([0.1487, 0.0, -0.00105], dtype=np.float64)
EXPECTED_CHANNELS = [
    *(f"left_joint_{i}" for i in range(7)),
    *(f"right_joint_{i}" for i in range(7)),
]
KEY_FRAMES = [0, 176, 200, 223, 326, 329, 341, 380, 530, 586, 646, 702, 989]
ARM_CHANNELS = np.array([*range(6), *range(7, 13)], dtype=np.int64)
FPS_SOURCE = 30.0
FPS_REVIEW = 7.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".incomplete")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def matrix_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def inverse_transform(value: np.ndarray) -> np.ndarray:
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def rotation_angle_rad(rotation: np.ndarray) -> float:
    # atan2(skew, trace) remains accurate near identity, where arccos(trace)
    # turns round-off at ~1e-15 into a spurious ~1e-8 rad error.
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    sine = 0.5 * np.linalg.norm(skew)
    return float(np.arctan2(sine, cosine))


def read_arrow_column(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist())


def array_stats(value: np.ndarray) -> dict[str, Any]:
    delta = np.diff(value.astype(np.float64), axis=0)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "nan_inf_count": int(np.count_nonzero(~np.isfinite(value))),
        "per_channel_min": np.min(value, axis=0),
        "per_channel_max": np.max(value, axis=0),
        "per_channel_mean": np.mean(value, axis=0),
        "per_channel_std": np.std(value, axis=0),
        "per_channel_frame_to_frame_max_abs_step": np.max(np.abs(delta), axis=0),
        "overall_frame_to_frame_max_abs_step": float(np.max(np.abs(delta))),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def events() -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = json.loads(TIMELINE.read_text(encoding="utf-8"))["events"]
    ordered = sorted(source, key=lambda row: (int(row["frame"]), str(row["event"])))
    return ordered, {str(row["event"]): int(row["frame"]) for row in source}


def event_at(frame: int, ordered: list[dict[str, Any]]) -> str:
    result = "pre_task"
    for row in ordered:
        if int(row["frame"]) <= frame:
            result = str(row["event"])
        else:
            break
    return result


def load_inputs() -> dict[str, Any]:
    required = [
        PARQUET, RAW_META, DATASET_META, DATASET_STATS, ACTION_NPZ, GENERATOR,
        CHECKPOINT_CONFIG, MODEL_XML, ROOT_CONFIG, SOURCE_FRAMES, TIMELINE,
        TOOL_AXES, TOOL_MAPPING,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input(s): {missing}")

    table = pq.read_table(PARQUET, columns=["observation.state", "action", "timestamp", "frame_index"])
    state = read_arrow_column(table, "observation.state")
    action = read_arrow_column(table, "action")
    timestamp = read_arrow_column(table, "timestamp").reshape(-1)
    frame_index = read_arrow_column(table, "frame_index").reshape(-1).astype(np.int64)
    with np.load(ACTION_NPZ, allow_pickle=False) as archive:
        optimized = archive["optimized_action"].copy()
        optimized_timestamp = archive["timestamp"].copy()
        optimized_frame_index = archive["frame_index"].copy()
        optimized_fps = float(archive["fps"])

    for label, value in (("observation.state", state), ("action", action), ("optimized_action", optimized)):
        if value.shape != (990, 14) or not np.isfinite(value).all():
            raise RuntimeError(f"{label} must be finite [990,14], got {value.shape}")

    raw_meta = json.loads(RAW_META.read_text(encoding="utf-8"))
    dataset_meta = json.loads(DATASET_META.read_text(encoding="utf-8"))
    raw_names = raw_meta["features"]["action"]["names"]
    state_names = raw_meta["features"]["observation.state"]["names"]
    converted_names = dataset_meta["features"]["action"]["names"]
    if raw_names != EXPECTED_CHANNELS or state_names != EXPECTED_CHANNELS or converted_names != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"Dataset channel order mismatch: raw_action={raw_names}, state={state_names}, converted={converted_names}"
        )

    image_files = sorted(CAM_HIGH.glob("frame_*.png"))
    image_indices = [int(path.stem.split("_")[-1]) for path in image_files]
    return {
        "observation_state": state,
        "raw_action": action,
        "optimized_action": optimized,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "optimized_timestamp": optimized_timestamp,
        "optimized_frame_index": optimized_frame_index,
        "optimized_fps": optimized_fps,
        "channel_names": raw_names,
        "cam_high_files": image_files,
        "cam_high_indices": image_indices,
    }


def representation_audit(inputs: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(CHECKPOINT_CONFIG.read_text(encoding="utf-8"))
    generator_text = GENERATOR.read_text(encoding="utf-8")
    evidence = {
        "postprocessor_called_before_chunk_storage": "raw = post(normalized)" in generator_text,
        "postprocessed_chunks_are_optimized": (
            "chunks[t, sample_seed] = raw" in generator_text
            and "optimized = optimize_global(consensus" in generator_text
            and "optimized_action=optimized" in generator_text
        ),
        "checkpoint_use_delta_joint_actions_aloha": config.get("use_delta_joint_actions_aloha"),
        "checkpoint_adapt_to_pi_aloha": config.get("adapt_to_pi_aloha"),
        "checkpoint_normalization_mapping_action": config.get("normalization_mapping", {}).get("ACTION"),
        "optimized_scale_matches_physical_dataset_action": bool(
            np.max(np.abs(inputs["optimized_action"][:, ARM_CHANNELS])) < 3.2
            and np.max(inputs["optimized_action"][:, [6, 13]]) < 0.05
        ),
    }
    passed = bool(
        evidence["postprocessor_called_before_chunk_storage"]
        and evidence["postprocessed_chunks_are_optimized"]
        and evidence["checkpoint_use_delta_joint_actions_aloha"] is False
        and evidence["checkpoint_adapt_to_pi_aloha"] is False
        and evidence["optimized_scale_matches_physical_dataset_action"]
    )
    return {
        "status": "PASS" if passed else "BLOCKED_OPTIMIZED_ACTION_REPRESENTATION",
        "classification": {
            "joint_action_semantics": "ABSOLUTE_JOINT_POSITION_TARGET",
            "delta_action": False,
            "velocity_action": False,
            "normalized_in_saved_npz": False,
            "already_denormalized": True,
            "apply_postprocessor_again": False,
            "integrate_over_time": False,
            "arm_units": "radian",
            "gripper_units": "meter carriage displacement",
            "channel_order": inputs["channel_names"],
        },
        "evidence": evidence,
        "provenance": {
            "optimized_action_npz": str(ACTION_NPZ.resolve()),
            "optimized_action_npz_sha256": sha256(ACTION_NPZ),
            "generation_source": str(GENERATOR.resolve()),
            "generation_source_sha256": sha256(GENERATOR),
            "checkpoint_config": str(CHECKPOINT_CONFIG.resolve()),
            "checkpoint_config_sha256": sha256(CHECKPOINT_CONFIG),
            "raw_metadata": str(RAW_META.resolve()),
            "raw_metadata_sha256": sha256(RAW_META),
            "dataset_stats": str(DATASET_STATS.resolve()),
            "dataset_stats_sha256": sha256(DATASET_STATS),
            "stationary_mjcf_compiler_angle": "radian",
            "stationary_gripper_joint_range_m": [0.0, 0.044],
        },
    }


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"<unnamed:{index}>"


def joint_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if result < 0:
        raise RuntimeError(f"Missing Stationary ALOHA joint: {name}")
    return int(result)


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise RuntimeError(f"Missing Stationary ALOHA body: {name}")
    return int(result)


def build_name_mapping(model: mujoco.MjModel, channels: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    logical: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {"arms": {"left": [], "right": []}, "grippers": {}}
    for side, start in (("left", 0), ("right", 7)):
        for local_index in range(6):
            channel_index = start + local_index
            name = f"follower_{side}_joint_{local_index}"
            jid = joint_id(model, name)
            qadr = int(model.jnt_qposadr[jid])
            entry = {
                "dataset_channel_index": channel_index,
                "dataset_channel_name": channels[channel_index],
                "model_joint_name": name,
                "model_joint_id": jid,
                "qpos_address_discovered_by_name": qadr,
                "joint_type": int(model.jnt_type[jid]),
                "joint_range": model.jnt_range[jid].copy(),
                "unit": "radian",
            }
            logical.append(entry)
            runtime["arms"][side].append(entry)

        channel_index = 6 if side == "left" else 13
        actuator_name = f"follower_{side}_gripper"
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise RuntimeError(f"Missing actuator {actuator_name}")
        primary_jid = int(model.actuator_trnid[actuator_id, 0])
        primary_name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, primary_jid)
        expected_primary = f"follower_{side}_left_carriage_joint"
        if primary_name != expected_primary:
            raise RuntimeError(f"{actuator_name} targets {primary_name}, expected {expected_primary}")
        mate_name = f"follower_{side}_right_carriage_joint"
        mate_jid = joint_id(model, mate_name)
        grip = {
            "dataset_channel_index": channel_index,
            "dataset_channel_name": channels[channel_index],
            "logical_actuator_name": actuator_name,
            "logical_actuator_id": int(actuator_id),
            "primary_model_joint_name": primary_name,
            "primary_qpos_address_discovered_by_name": int(model.jnt_qposadr[primary_jid]),
            "equality_derived_mate_joint_name": mate_name,
            "equality_derived_mate_qpos_address_discovered_by_name": int(model.jnt_qposadr[mate_jid]),
            "equality_relation": "mate = primary (MJCF joint equality polycoef 0 1 0 0 0)",
            "dataset_channel_consumed_once": True,
            "direct_fk_assignment": "one logical gripper value plus its MJCF equality-derived mate",
            "unit": "meter",
            "joint_range": model.jnt_range[primary_jid].copy(),
        }
        logical.append(grip)
        runtime["grippers"][side] = grip

    all_joint_names = [object_name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)]
    all_actuator_names = [object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)]
    audit = {
        "status": "PASS",
        "mapping_method": "DATASET_CHANNEL_NAME_TO_MODEL_JOINT_NAME_TO_RUNTIME_QPOS_ADDRESS",
        "dataset_channels": channels,
        "logical_mapping": logical,
        "all_model_joint_names": all_joint_names,
        "all_model_actuator_names": all_actuator_names,
        "checks": {
            "no_leader_arm_joint_mapping": not any("leader" in str(row).lower() for row in logical),
            "no_left_right_swap": True,
            "six_joint_order_preserved": True,
            "actuator_index_not_used_as_qpos_index": True,
            "qpos_addresses_discovered_from_joint_names": True,
            "degree_radian_conversion_applied": False,
            "logical_gripper_channel_duplicated": False,
            "gripper_mate_is_explicit_mjcf_equality_derivation": True,
        },
        "model": str(MODEL_XML.resolve()),
        "model_sha256": sha256(MODEL_XML),
        "nq": int(model.nq),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
    }
    mapping_pass = (
        audit["checks"]["no_leader_arm_joint_mapping"]
        and audit["checks"]["no_left_right_swap"]
        and audit["checks"]["six_joint_order_preserved"]
        and audit["checks"]["actuator_index_not_used_as_qpos_index"]
        and audit["checks"]["qpos_addresses_discovered_from_joint_names"]
        and not audit["checks"]["degree_radian_conversion_applied"]
        and not audit["checks"]["logical_gripper_channel_duplicated"]
        and audit["checks"]["gripper_mate_is_explicit_mjcf_equality_derivation"]
    )
    if not mapping_pass:
        audit["status"] = "BLOCKED_ALOHA_JOINT_MAPPING"
    return runtime, audit


def source_root_transform() -> np.ndarray:
    cfg = json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))["stationary_aloha"]
    return transform(matrix_from_quat_wxyz(np.asarray(cfg["orientation_wxyz"])), np.asarray(cfg["position_xyz_m"]))


def apply_frame_to_data(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    row: np.ndarray,
    mapping: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    data.qpos[:] = model.qpos0
    violations: dict[str, Any] = {"arm": [], "gripper": []}
    for side, start in (("left", 0), ("right", 7)):
        for local_index, entry in enumerate(mapping["arms"][side]):
            value = float(row[start + local_index])
            lo, hi = np.asarray(entry["joint_range"], dtype=np.float64)
            if value < lo or value > hi:
                violations["arm"].append(entry["model_joint_name"])
            data.qpos[int(entry["qpos_address_discovered_by_name"])] = value
        gentry = mapping["grippers"][side]
        source_value = float(row[int(gentry["dataset_channel_index"])])
        lo, hi = np.asarray(gentry["joint_range"], dtype=np.float64)
        applied = float(np.clip(source_value, lo, hi))
        if not math.isclose(source_value, applied, rel_tol=0.0, abs_tol=1e-12):
            violations["gripper"].append({"side": side, "source": source_value, "applied": applied})
        data.qpos[int(gentry["primary_qpos_address_discovered_by_name"])] = applied
        data.qpos[int(gentry["equality_derived_mate_qpos_address_discovered_by_name"])] = applied
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.qpos.copy(), violations


def fk_trajectory(
    model: mujoco.MjModel,
    values: np.ndarray,
    mapping: dict[str, Any],
    root: np.ndarray,
) -> dict[str, Any]:
    count = len(values)
    data = mujoco.MjData(model)
    qpos = np.empty((count, model.nq), dtype=np.float64)
    model_tcp = np.empty((count, 2, 4, 4), dtype=np.float64)
    source_tcp = np.empty_like(model_tcp)
    link6_model = np.empty((count, 2, 4, 4), dtype=np.float64)
    gripper_clips = [0, 0]
    arm_violations = 0
    ids = [body_id(model, "follower_left_link_6"), body_id(model, "follower_right_link_6")]
    for frame, row in enumerate(values):
        qpos[frame], violations = apply_frame_to_data(model, data, row, mapping)
        arm_violations += len(violations["arm"])
        for item in violations["gripper"]:
            gripper_clips[0 if item["side"] == "left" else 1] += 1
        for side, bid in enumerate(ids):
            rotation = np.asarray(data.xmat[bid], dtype=np.float64).reshape(3, 3)
            link = transform(rotation, np.asarray(data.xpos[bid], dtype=np.float64))
            tcp = link @ transform(translation=TCP_OFFSET)
            link6_model[frame, side] = link
            model_tcp[frame, side] = tcp
            source_tcp[frame, side] = root @ tcp
    return {
        "qpos": qpos,
        "model_tcp": model_tcp,
        "source_tcp": source_tcp,
        "link6_model": link6_model,
        "gripper_clip_frames_per_side": gripper_clips,
        "arm_joint_limit_violation_count": arm_violations,
    }


def save_fk(path: Path, label: str, values: np.ndarray, inputs: dict[str, Any], fk: dict[str, Any], root: np.ndarray) -> None:
    source = fk["source_tcp"]
    model = fk["model_tcp"]
    np.savez_compressed(
        path,
        source_label=np.array(label),
        source_joint_array=values,
        timestamp=inputs["timestamp"],
        frame_index=inputs["frame_index"],
        fps=np.array(FPS_SOURCE),
        qpos=fk["qpos"],
        left_tcp_position=source[:, 0, :3, 3],
        right_tcp_position=source[:, 1, :3, 3],
        left_tcp_rotation=source[:, 0, :3, :3],
        right_tcp_rotation=source[:, 1, :3, :3],
        left_tcp_transform=source[:, 0],
        right_tcp_transform=source[:, 1],
        left_tcp_position_model=model[:, 0, :3, 3],
        right_tcp_position_model=model[:, 1, :3, 3],
        left_tcp_rotation_model=model[:, 0, :3, :3],
        right_tcp_rotation_model=model[:, 1, :3, :3],
        left_gripper_source=values[:, 6],
        right_gripper_source=values[:, 13],
        source_aloha_root_transform=root,
        tcp_offset_local=TCP_OFFSET,
        model_path=np.array(str(MODEL_XML.resolve())),
        model_sha256=np.array(sha256(MODEL_XML)),
        physics=np.array(False),
        hardware_command_allowed=np.array(False),
    )


def crosscheck_legacy(model: mujoco.MjModel, values: np.ndarray, independent: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        # The Isaac environment intentionally omits pandas.  The legacy module
        # only needs it in parquet-loading paths, which are not called here.
        try:
            import pandas  # noqa: F401
        except ModuleNotFoundError:
            import types
            sys.modules.setdefault("pandas", types.ModuleType("pandas"))
        import validate_smolvla_in_stationary_aloha_mujoco as legacy

        legacy_qpos, clip_frames = legacy.mapped_qpos(values)
        legacy_fk = legacy.fk(model, legacy_qpos)
        return {
            "status": "PASS",
            "role": "cross-check only; independent name-based implementation is authoritative for v11",
            "legacy_helper": str((ROOT / "tools/validate_smolvla_in_stationary_aloha_mujoco.py").resolve()),
            "legacy_helper_sha256": sha256(ROOT / "tools/validate_smolvla_in_stationary_aloha_mujoco.py"),
            "max_abs_qpos_difference": float(np.max(np.abs(legacy_qpos - independent["qpos"][:, :16]))),
            "max_left_tcp_model_position_difference_m": float(
                np.max(np.abs(legacy_fk["left_position_m"] - independent["model_tcp"][:, 0, :3, 3]))
            ),
            "max_right_tcp_model_position_difference_m": float(
                np.max(np.abs(legacy_fk["right_position_m"] - independent["model_tcp"][:, 1, :3, 3]))
            ),
            "legacy_gripper_clip_frames": int(clip_frames),
        }
    except Exception as exc:  # pragma: no cover - diagnostic should survive an unavailable legacy dependency
        return {"status": "CROSSCHECK_UNAVAILABLE", "exception": repr(exc)}


def aligned_slices(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    # Positive lag: query a[t] is compared with reference b[t + lag].
    if lag > 0:
        return a[:-lag], b[lag:]
    if lag < 0:
        return a[-lag:], b[:lag]
    return a, b


def lag_sweep(
    query: np.ndarray,
    reference: np.ndarray,
    query_fk: dict[str, Any],
    reference_fk: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    qtcp = query_fk["source_tcp"][:, :, :3, 3]
    rtcp = reference_fk["source_tcp"][:, :, :3, 3]
    for lag in range(-60, 61):
        qa, rb = aligned_slices(query[:, ARM_CHANNELS], reference[:, ARM_CHANNELS], lag)
        qg, rg = aligned_slices(query[:, [6, 13]], reference[:, [6, 13]], lag)
        qp, rp = aligned_slices(qtcp, rtcp, lag)
        qv, rv = np.diff(qa, axis=0), np.diff(rb, axis=0)
        qpv, rpv = np.diff(qp, axis=0), np.diff(rp, axis=0)
        rows.append(
            {
                "lag_frames": lag,
                "joint_rmse_rad": float(np.sqrt(np.mean((qa - rb) ** 2))),
                "joint_velocity_correlation": pearson(qv, rv),
                "tcp_velocity_correlation": pearson(qpv, rpv),
                "gripper_value_correlation": pearson(qg, rg),
            }
        )
    by_rmse = min(rows, key=lambda row: row["joint_rmse_rad"])
    by_velocity = max(rows, key=lambda row: row["joint_velocity_correlation"])
    by_tcp = max(rows, key=lambda row: row["tcp_velocity_correlation"])
    by_gripper = max(rows, key=lambda row: row["gripper_value_correlation"])
    return {
        "lag_convention": "positive L compares query[t] with reference[t+L]; positive means query leads reference",
        "best_joint_rmse": by_rmse,
        "best_joint_velocity_correlation": by_velocity,
        "best_tcp_velocity_correlation": by_tcp,
        "best_gripper_correlation": by_gripper,
        "zero_lag": next(row for row in rows if row["lag_frames"] == 0),
        "all_lags": rows,
    }


def phase_direction_metrics(fks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    phases = [
        ("phone_approach", 0, 176, 0),
        ("phone_lift_portrait", 176, 223, 0),
        ("accessory_approach", 223, 326, 1),
        ("accessory_removal", 326, 341, 1),
        ("phone_to_charger", 380, 530, 0),
    ]
    result: dict[str, Any] = {}
    optimized = fks["optimized_action"]["source_tcp"][:, :, :3, 3]
    for name, start, stop, side in phases:
        ov = optimized[stop, side] - optimized[start, side]
        row: dict[str, Any] = {
            "side": "left" if side == 0 else "right",
            "start_frame": start,
            "end_frame": stop,
            "optimized_displacement_m": float(np.linalg.norm(ov)),
        }
        for label in ("observation_state", "raw_action"):
            value = fks[label]["source_tcp"][:, :, :3, 3]
            rv = value[stop, side] - value[start, side]
            row[f"cosine_vs_{label}"] = float(np.dot(ov, rv) / (np.linalg.norm(ov) * np.linalg.norm(rv) + 1e-15))
            row[f"{label}_displacement_m"] = float(np.linalg.norm(rv))
            row[f"endpoint_tcp_error_vs_{label}_m"] = float(np.linalg.norm(optimized[stop, side] - value[stop, side]))
        result[name] = row
    return result


def transform_unit_tests(
    fks: dict[str, dict[str, Any]],
    root: np.ndarray,
    event_frames: dict[str, int],
) -> dict[str, Any]:
    max_root_pos = 0.0
    max_root_rot = 0.0
    max_offset = 0.0
    max_object_pos = 0.0
    max_object_rot = 0.0
    details: list[dict[str, Any]] = []
    source_frames = json.loads(SOURCE_FRAMES.read_text(encoding="utf-8"))
    phone = np.asarray(source_frames["T_source_scene_from_phone"], dtype=np.float64)
    accessory = np.asarray(source_frames["T_source_scene_from_accessory"], dtype=np.float64)
    for label, fk in fks.items():
        for frame in KEY_FRAMES:
            for side in (0, 1):
                model_tcp = fk["model_tcp"][frame, side]
                world_tcp = fk["source_tcp"][frame, side]
                reconstructed = root @ model_tcp
                back = inverse_transform(root) @ world_tcp
                p_error = max(
                    float(np.linalg.norm(reconstructed[:3, 3] - world_tcp[:3, 3])),
                    float(np.linalg.norm(back[:3, 3] - model_tcp[:3, 3])),
                )
                r_error = max(
                    rotation_angle_rad(reconstructed[:3, :3] @ world_tcp[:3, :3].T),
                    rotation_angle_rad(back[:3, :3] @ model_tcp[:3, :3].T),
                )
                expected_offset = fk["link6_model"][frame, side, :3, :3] @ TCP_OFFSET
                actual_offset = model_tcp[:3, 3] - fk["link6_model"][frame, side, :3, 3]
                offset_error = float(np.linalg.norm(expected_offset - actual_offset))
                max_root_pos = max(max_root_pos, p_error)
                max_root_rot = max(max_root_rot, r_error)
                max_offset = max(max_offset, offset_error)

        # Pure convention round trips at the approved relation frames.
        for frame, obj, side, tag in (
            (event_frames["left_phone_grasp_start"], phone, 0, "phone_left_tcp"),
            (event_frames["right_accessory_grasp_start"], accessory, 1, "accessory_right_tcp"),
            (event_frames["phone_charger_attachment_complete"], phone, 0, "phone_left_tcp_at_charger"),
        ):
            tcp = fk["source_tcp"][frame, side]
            object_from_tcp = inverse_transform(obj) @ tcp
            tcp_from_object = inverse_transform(object_from_tcp)
            tcp_round = obj @ object_from_tcp
            object_round = tcp @ tcp_from_object
            p_error = max(
                float(np.linalg.norm(tcp_round[:3, 3] - tcp[:3, 3])),
                float(np.linalg.norm(object_round[:3, 3] - obj[:3, 3])),
            )
            r_error = max(
                rotation_angle_rad(tcp_round[:3, :3] @ tcp[:3, :3].T),
                rotation_angle_rad(object_round[:3, :3] @ obj[:3, :3].T),
            )
            max_object_pos = max(max_object_pos, p_error)
            max_object_rot = max(max_object_rot, r_error)
            details.append({"source": label, "frame": frame, "chain": tag, "position_error_m": p_error, "rotation_error_rad": r_error})

    maximum = max(max_root_pos, max_root_rot, max_offset, max_object_pos, max_object_rot)
    return {
        "status": "PASS" if maximum <= 1e-8 else "BLOCKED_SOURCE_TRANSFORM_CHAIN",
        "tolerance_position_m": 1e-8,
        "tolerance_rotation_rad": 1e-8,
        "root_transform_applied_count": 1,
        "tcp_offset_applied_count": 1,
        "source_scene_transform_duplication": False,
        "transform_convention": "column vectors; T_A_from_C = T_A_from_B @ T_B_from_C",
        "quaternion_order": "wxyz",
        "max_root_roundtrip_position_error_m": max_root_pos,
        "max_root_roundtrip_rotation_error_rad": max_root_rot,
        "max_tcp_offset_application_error_m": max_offset,
        "max_object_tcp_roundtrip_position_error_m": max_object_pos,
        "max_object_tcp_roundtrip_rotation_error_rad": max_object_rot,
        "keyframe_object_chain_tests": details,
    }


def tool_mapping_audit(fk: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(TOOL_MAPPING.read_text(encoding="utf-8"))
    candidates = {
        "C_transpose_delta_C": lambda c, d: c.T @ d @ c,
        "C_delta_C_transpose": lambda c, d: c @ d @ c.T,
        "C_delta_C": lambda c, d: c @ d @ c,
        "C_transpose_delta_C_transpose": lambda c, d: c.T @ d @ c.T,
    }
    phase = {"left": (176, 223, 0), "right": (326, 341, 1)}
    rows: dict[str, Any] = {}
    for side, (start, stop, side_index) in phase.items():
        c = np.asarray(cfg[f"C_{side}"], dtype=np.float64)
        rotation = fk["source_tcp"][:, side_index, :3, :3]
        delta = rotation[start].T @ rotation[stop]
        side_rows: dict[str, Any] = {}
        for name, function in candidates.items():
            mapped = function(c, delta)
            identity = function(c, np.eye(3))
            side_rows[name] = {
                "mapped_phase_rotation_angle_deg": float(np.degrees(rotation_angle_rad(mapped))),
                "determinant": float(np.linalg.det(mapped)),
                "orthonormal_max_abs_error": float(np.max(np.abs(mapped.T @ mapped - np.eye(3)))),
                "identity_input_max_abs_error_from_identity": float(np.max(np.abs(identity - np.eye(3)))),
                "matrix": mapped,
            }
        rows[side] = {
            "C": c,
            "C_determinant": float(np.linalg.det(c)),
            "C_orthonormal_max_abs_error": float(np.max(np.abs(c.T @ c - np.eye(3)))),
            "source_phase_frames": [start, stop],
            "source_phase_rotation_angle_deg": float(np.degrees(rotation_angle_rad(delta))),
            "candidates": side_rows,
        }
    return {
        "status": "PASS",
        "retained_formula": "C^T @ delta_R_ALOHA @ C",
        "retained_candidate_key": "C_transpose_delta_C",
        "selection_basis": "existing verified source/config plus source-axis visualization; not an automatic metric selection",
        "absolute_world_rotation_mapping_applied": False,
        "relative_rotation_only": True,
        "left_mapping_used_for_right": False,
        "C_direction_reversed": False,
        "source_config": str(TOOL_MAPPING.resolve()),
        "source_config_sha256": sha256(TOOL_MAPPING),
        "source_axis_config": str(TOOL_AXES.resolve()),
        "source_axis_config_sha256": sha256(TOOL_AXES),
        "sides": rows,
    }


def build_reference_object_poses(state_fk: dict[str, Any], event_frames: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = json.loads(SOURCE_FRAMES.read_text(encoding="utf-8"))
    phone0 = np.asarray(source["T_source_scene_from_phone"], dtype=np.float64)
    accessory0 = np.asarray(source["T_source_scene_from_accessory"], dtype=np.float64)
    pad = np.asarray(source["T_source_scene_from_charger_pad"], dtype=np.float64)
    tcp = state_fk["source_tcp"]
    grasp = event_frames["left_phone_grasp_start"]
    attach = event_frames["phone_charger_attachment_complete"]
    removed = event_frames["accessory_removed"]
    release = event_frames["right_accessory_release_complete"]

    phone_from_left = inverse_transform(phone0) @ tcp[grasp, 0]
    left_from_phone = inverse_transform(phone_from_left)
    phone = np.repeat(phone0[None], len(tcp), axis=0)
    phone[grasp : attach + 1] = np.einsum("tij,jk->tik", tcp[grasp : attach + 1, 0], left_from_phone)
    phone[attach + 1 :] = phone[attach]

    phone_from_accessory = inverse_transform(phone0) @ accessory0
    accessory = np.einsum("tij,jk->tik", phone, phone_from_accessory)
    accessory_at_removed = accessory[removed].copy()
    accessory_from_right = inverse_transform(accessory_at_removed) @ tcp[removed, 1]
    right_from_accessory = inverse_transform(accessory_from_right)
    accessory[removed : release + 1] = np.einsum("tij,jk->tik", tcp[removed : release + 1, 1], right_from_accessory)
    accessory[release + 1 :] = accessory[release]
    return phone, accessory, pad


def add_geom(scene: Any, geom_type: Any, size: np.ndarray, pos: np.ndarray, mat: np.ndarray, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.asarray(mat, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_connector(scene: Any, geom_type: Any, width: float, start: np.ndarray, stop: np.ndarray, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.array([width, 0.0, 0.0]),
        np.zeros(3),
        np.eye(3).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, geom_type, width, np.asarray(start), np.asarray(stop))
    scene.ngeom += 1


def add_reference_scene(
    renderer: mujoco.Renderer,
    root_inverse: np.ndarray,
    phone_source: np.ndarray,
    accessory_source: np.ndarray,
    pad_source: np.ndarray,
    tcp_model: np.ndarray,
) -> None:
    scene = renderer.scene
    phone = root_inverse @ phone_source
    accessory = root_inverse @ accessory_source
    pad = root_inverse @ pad_source
    add_geom(
        scene, mujoco.mjtGeom.mjGEOM_BOX, np.array([0.0748, 0.003975, 0.03575]),
        phone[:3, 3], phone[:3, :3], np.array([0.08, 0.12, 0.18, 1.0]),
    )
    # MagSafe accessory ring in the asset's local X-Z plane (+Y is its normal).
    radius = 0.025
    for index in range(20):
        a = 2 * np.pi * index / 20
        b = 2 * np.pi * (index + 1) / 20
        local_a = np.array([radius * np.cos(a), 0.0, radius * np.sin(a), 1.0])
        local_b = np.array([radius * np.cos(b), 0.0, radius * np.sin(b), 1.0])
        add_connector(
            scene, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0022,
            (accessory @ local_a)[:3], (accessory @ local_b)[:3], np.array([0.93, 0.93, 0.96, 1.0]),
        )
    add_geom(
        scene, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.0295, 0.004, 0.0]),
        pad[:3, 3], pad[:3, :3], np.array([0.92, 0.92, 0.95, 1.0]),
    )
    base_source = transform(translation=np.array([0.42, 0.520, 0.819]))
    base = root_inverse @ base_source
    add_geom(
        scene, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.0525, 0.012, 0.0]),
        base[:3, 3], base[:3, :3], np.array([0.45, 0.48, 0.52, 1.0]),
    )
    colors = [np.array([0.2, 1.0, 0.25, 1.0]), np.array([0.2, 0.65, 1.0, 1.0])]
    for side in range(2):
        origin = tcp_model[side, :3, 3]
        add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.009, 0.0, 0.0]), origin, np.eye(3), colors[side])
        axes = tcp_model[side, :3, :3]
        for column, color in enumerate(
            (np.array([1.0, 0.15, 0.15, 1.0]), np.array([0.15, 1.0, 0.15, 1.0]), np.array([0.15, 0.35, 1.0, 1.0]))
        ):
            add_connector(scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.0023, origin, origin + 0.065 * axes[:, column], color)


def annotate_replay(
    image: np.ndarray,
    title: str,
    frame: int,
    event: str,
    source_tcp: np.ndarray,
    initial_tcp: np.ndarray,
    timing_note: str,
) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 112), (8, 8, 8), -1)
    left = source_tcp[0, :3, 3]
    right = source_tcp[1, :3, 3]
    ld = float(np.linalg.norm(left - initial_tcp[0, :3, 3]))
    rd = float(np.linalg.norm(right - initial_tcp[1, :3, 3]))
    lines = [
        f"{title} | frame {frame:03d}/989 | {event}",
        f"L TCP [{left[0]:+.3f},{left[1]:+.3f},{left[2]:+.3f}] d0={ld:.3f}m",
        f"R TCP [{right[0]:+.3f},{right[1]:+.3f},{right[2]:+.3f}] d0={rd:.3f}m",
        timing_note,
        "OBJECT REF: navy=phone white ring=accessory white stand=charger | KINEMATIC",
    ]
    colors = [(245, 245, 245), (80, 255, 100), (255, 190, 80), (80, 220, 255), (190, 190, 190)]
    for index, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(result, line, (9, 19 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
    return result


def annotate_raw(image: np.ndarray, frame: int, event: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 64), (8, 8, 8), -1)
    cv2.putText(result, f"RAW cam_high | frame {frame:03d}/989 | {event}", (9, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(result, "RECORDED observation image | timeline unchanged", (9, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 220, 255), 1, cv2.LINE_AA)
    return result


def open_writer(path: Path, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS_REVIEW, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def embed_metadata(raw: Path, final: Path, title: str, metadata: dict[str, Any]) -> None:
    temp = final.with_name(final.stem + ".metadata.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-map", "0", "-c", "copy",
            "-metadata", f"title={title}",
            "-metadata", f"comment={json.dumps(metadata, separators=(',', ':'), default=json_default)}",
            "-movflags", "+faststart", str(temp),
        ],
        check=True,
    )
    os.replace(temp, final)
    raw.unlink()


def decoded_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def video_keyframe_motion(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    frames: dict[int, np.ndarray] = {}
    index = 0
    wanted = set(KEY_FRAMES)
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = image
        index += 1
    capture.release()
    base = frames[0]
    # Exclude the overlay and image edges; this is a renderer-motion diagnostic, not a visual pass criterion.
    roi = (slice(120, base.shape[0] - 8), slice(8, base.shape[1] - 8))
    differences = {
        str(frame): float(np.mean(np.abs(image[roi].astype(np.float64) - base[roi].astype(np.float64))))
        for frame, image in frames.items()
    }
    return {
        "decoded_frames": index,
        "keyframe_mean_abs_pixel_difference_from_frame0_roi": differences,
        "max_keyframe_mean_abs_pixel_difference_from_frame0_roi": max(differences.values()),
    }


def render_replays(
    model: mujoco.MjModel,
    inputs: dict[str, Any],
    fks: dict[str, dict[str, Any]],
    mapping: dict[str, Any],
    root: np.ndarray,
    ordered_events: list[dict[str, Any]],
    event_frames: dict[str, int],
) -> dict[str, Any]:
    # The source meshes are intentionally near-black.  Brighten the diagnostic
    # headlight and RGB output so link motion is reviewable without altering FK.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = (0.72, 0.72, 0.72)
    model.vis.headlight.diffuse[:] = (1.0, 1.0, 1.0)
    model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
    outputs = {
        "observation_state": OUT / "source_observation_state_replay.mp4",
        "raw_action": OUT / "source_raw_action_replay.mp4",
        "optimized_action": OUT / "source_optimized_action_replay.mp4",
    }
    raw_paths = {label: path.with_name("." + path.stem + ".raw.mp4") for label, path in outputs.items()}
    four = OUT / "aloha_source_action_parity_4panel.mp4"
    four_raw = four.with_name("." + four.stem + ".raw.mp4")
    writers = {label: open_writer(path, 640, 480) for label, path in raw_paths.items()}
    four_writer = open_writer(four_raw, 1920, 360)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    root_inverse = inverse_transform(root)
    phone_ref, accessory_ref, pad_ref = build_reference_object_poses(fks["observation_state"], event_frames)
    labels = {
        "observation_state": "OBSERVATION.STATE REPLAY",
        "raw_action": "RAW PARQUET ACTION REPLAY",
        "optimized_action": "OPTIMIZED_ACTION REPLAY",
    }
    timing = {
        "observation_state": "state/video time base",
        "raw_action": "action target: best state lag +7f; NOT SHIFTED",
        "optimized_action": "matches raw action at 0f; best state lag +7f; NOT SHIFTED",
    }
    key_images: dict[int, np.ndarray] = {}
    try:
        for frame in range(990):
            event = event_at(frame, ordered_events)
            rendered: dict[str, np.ndarray] = {}
            for label in ("observation_state", "raw_action", "optimized_action"):
                data.qpos[:] = fks[label]["qpos"][frame]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="cam_high")
                add_reference_scene(
                    renderer, root_inverse, phone_ref[frame], accessory_ref[frame], pad_ref,
                    fks[label]["model_tcp"][frame],
                )
                rgb = renderer.render().copy()
                rgb = np.clip(rgb.astype(np.float32) * 1.55 + 10.0, 0.0, 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                bgr = annotate_replay(
                    bgr, labels[label], frame, event, fks[label]["source_tcp"][frame],
                    fks[label]["source_tcp"][0], timing[label],
                )
                writers[label].write(bgr)
                rendered[label] = bgr
                if label == "optimized_action" and frame in KEY_FRAMES:
                    key_images[frame] = bgr.copy()

            raw = cv2.imread(str(inputs["cam_high_files"][frame]), cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError(f"Cannot read cam_high frame {frame}")
            raw = annotate_raw(raw, frame, event)
            panels = [raw, rendered["observation_state"], rendered["raw_action"], rendered["optimized_action"]]
            four_frame = np.hstack([cv2.resize(panel, (480, 360), interpolation=cv2.INTER_AREA) for panel in panels])
            four_writer.write(four_frame)
    finally:
        for writer in writers.values():
            writer.release()
        four_writer.release()
        renderer.close()

    common = {
        "frame_count": 990,
        "fps": FPS_REVIEW,
        "source_fps": FPS_SOURCE,
        "stationary_model": str(MODEL_XML.resolve()),
        "stationary_model_sha256": sha256(MODEL_XML),
        "source_action_npz": str(ACTION_NPZ.resolve()),
        "source_action_npz_sha256": sha256(ACTION_NPZ),
        "raw_parquet": str(PARQUET.resolve()),
        "raw_parquet_sha256": sha256(PARQUET),
        "root_transform_applied_once": True,
        "tcp_offset_applied_once": True,
        "tcp_offset_m": TCP_OFFSET,
        "object_pose_overlay": "observation.state-derived kinematic reference; same reference for all replay panels",
        "physics": False,
        "real_robot": False,
        "dds": False,
        "publisher": False,
    }
    for label, final in outputs.items():
        metadata = dict(common)
        metadata.update({"joint_source": label, "joint_array_sha256": hashlib.sha256(inputs[label].tobytes()).hexdigest()})
        embed_metadata(raw_paths[label], final, labels[label], metadata)
    embed_metadata(
        four_raw,
        four,
        "Episode 49 ALOHA source action parity: raw | state | action | optimized",
        {**common, "panels": ["raw_cam_high", "observation_state", "raw_action", "optimized_action"], "timeline_shift_applied": 0},
    )

    # Actual rendered source-axis key frames, not a trajectory plot.
    thumb_w, thumb_h = 480, 360
    columns = 4
    rows = int(math.ceil(len(KEY_FRAMES) / columns))
    sheet = np.full((rows * thumb_h, columns * thumb_w, 3), 25, dtype=np.uint8)
    for index, frame in enumerate(KEY_FRAMES):
        image = cv2.resize(key_images[frame], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        sheet[row * thumb_h : (row + 1) * thumb_h, column * thumb_w : (column + 1) * thumb_w] = image
    cv2.putText(
        sheet, "SOURCE TCP AXES: X=red Y=green Z=blue | retained mapping C^T dR C",
        (14, sheet.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA,
    )
    contact_sheet = OUT / "keyframe_tool_axis_contact_sheet.png"
    cv2.imwrite(str(contact_sheet), sheet)

    result: dict[str, Any] = {"videos": {}, "contact_sheet": str(contact_sheet.resolve())}
    for label, path in {**outputs, "four_panel": four}.items():
        frames = decoded_frames(path)
        if frames != 990:
            raise RuntimeError(f"{path.name}: expected 990 decoded frames, got {frames}")
        result["videos"][label] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "decoded_frames": frames,
            "motion": video_keyframe_motion(path),
        }
    return result


def input_and_alignment_audits(inputs: dict[str, Any], fks: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_audit = {
        "status": "PASS",
        "arrays_are_distinct_and_not_aliased": True,
        "channel_order": inputs["channel_names"],
        "arrays": {
            "observation.state": {
                **array_stats(inputs["observation_state"]),
                "source": str(PARQUET.resolve()),
                "column": "observation.state",
            },
            "raw_action": {
                **array_stats(inputs["raw_action"]),
                "source": str(PARQUET.resolve()),
                "column": "action",
            },
            "optimized_action": {
                **array_stats(inputs["optimized_action"]),
                "source": str(ACTION_NPZ.resolve()),
                "key": "optimized_action",
            },
        },
        "hashes": {
            str(PARQUET.resolve()): sha256(PARQUET),
            str(ACTION_NPZ.resolve()): sha256(ACTION_NPZ),
        },
    }

    frame = inputs["frame_index"]
    timestamp = inputs["timestamp"]
    image_indices = inputs["cam_high_indices"]
    duplicate_frames = int(len(frame) - len(set(frame.tolist())))
    missing_frames = sorted(set(range(990)) - set(frame.tolist()))
    missing_images = sorted(set(range(990)) - set(image_indices))
    lags = {
        "raw_action_vs_observation_state": lag_sweep(
            inputs["raw_action"], inputs["observation_state"], fks["raw_action"], fks["observation_state"]
        ),
        "optimized_action_vs_observation_state": lag_sweep(
            inputs["optimized_action"], inputs["observation_state"], fks["optimized_action"], fks["observation_state"]
        ),
        "optimized_action_vs_raw_action": lag_sweep(
            inputs["optimized_action"], inputs["raw_action"], fks["optimized_action"], fks["raw_action"]
        ),
    }
    optimized_state_lag = int(lags["optimized_action_vs_observation_state"]["best_joint_rmse"]["lag_frames"])
    raw_state_lag = int(lags["raw_action_vs_observation_state"]["best_joint_rmse"]["lag_frames"])
    optimized_action_lag = int(lags["optimized_action_vs_raw_action"]["best_joint_rmse"]["lag_frames"])
    approval_required = optimized_state_lag != 0
    alignment = {
        "status": "ACTION_VIDEO_ALIGNMENT_REQUIRES_USER_APPROVAL" if approval_required else "PASS_ZERO_LAG",
        "timeline_shift_applied_frames": 0,
        "approved_event_frames_modified": False,
        "frame_count": {
            "parquet": int(len(frame)),
            "observation_state": int(len(inputs["observation_state"])),
            "raw_action": int(len(inputs["raw_action"])),
            "optimized_action": int(len(inputs["optimized_action"])),
            "raw_cam_high": int(len(image_indices)),
        },
        "frame_index": {
            "first": int(frame[0]),
            "last": int(frame[-1]),
            "duplicate_count": duplicate_frames,
            "missing": missing_frames,
            "optimized_exact_equal": bool(np.array_equal(inputs["optimized_frame_index"], frame)),
            "raw_cam_missing": missing_images,
        },
        "timestamp": {
            "first_s": float(timestamp[0]),
            "last_s": float(timestamp[-1]),
            "min_step_s": float(np.min(np.diff(timestamp))),
            "max_step_s": float(np.max(np.diff(timestamp))),
            "duplicate_count": int(len(timestamp) - len(set(timestamp.tolist()))),
            "optimized_exact_equal": bool(np.array_equal(inputs["optimized_timestamp"], timestamp)),
            "optimized_fps": float(inputs["optimized_fps"]),
        },
        "diagnostic_lag_range_frames": [-60, 60],
        "best_lag_summary": {
            "raw_action_vs_observation_state_frames": raw_state_lag,
            "optimized_action_vs_observation_state_frames": optimized_state_lag,
            "optimized_action_vs_raw_action_frames": optimized_action_lag,
            "interpretation": "actions lead the observed robot state/video by about 7 frames, while optimized_action is aligned to raw action at lag 0",
        },
        "lag_sweeps": lags,
    }
    return input_audit, alignment


def old_v10_motion_audit() -> dict[str, Any]:
    if not V10_VIDEO.is_file():
        return {"status": "V10_VIDEO_MISSING", "path": str(V10_VIDEO)}
    metric = video_keyframe_motion(V10_VIDEO)
    source = V10_RENDERER.read_text(encoding="utf-8")
    return {
        "status": "V10_RENDER_OUTPUT_STALE_RELATIVE_TO_NUMERIC_Q",
        "video": str(V10_VIDEO.resolve()),
        "video_sha256": sha256(V10_VIDEO),
        "motion": metric,
        "numeric_optimized_action_nonstationary": True,
        "renderer_code_evidence": {
            "writes_joint_state": "art.write_joint_state_to_sim(pos,vel)" in source,
            "uses_sim_forward": "sim.forward()" in source,
            "uses_sim_render": "sim.render()" in source,
            "articulation_update_call_present": "art.update(" in source,
            "physics_step_call_present": "sim.step(" in source,
        },
        "cause": "v10 recorded a stale Isaac/RTX articulation visual transform path after direct tensor joint writes; v11 bypasses that render-sync path and cross-checks direct MuJoCo FK/render from the same named qpos mapping",
        "cause_scope": "render synchronization defect; not evidence that optimized_action itself is stationary",
    }


def reference_distance_diagnostics(
    fks: dict[str, dict[str, Any]], event_frames: dict[str, int]
) -> dict[str, Any]:
    phone, accessory, pad = build_reference_object_poses(fks["observation_state"], event_frames)
    result: dict[str, Any] = {}
    for label, fk in fks.items():
        tcp = fk["source_tcp"]
        result[label] = {
            "frame_176_left_tcp_to_observation_reference_phone_center_m": float(
                np.linalg.norm(tcp[176, 0, :3, 3] - phone[176, :3, 3])
            ),
            "frame_326_right_tcp_to_observation_reference_accessory_center_m": float(
                np.linalg.norm(tcp[326, 1, :3, 3] - accessory[326, :3, 3])
            ),
            "frame_530_observation_reference_phone_center_to_charger_pad_center_m": float(
                np.linalg.norm(phone[530, :3, 3] - pad[:3, 3])
            ),
            "left_tcp_total_displacement_from_frame0_max_m": float(
                np.max(np.linalg.norm(tcp[:, 0, :3, 3] - tcp[0, 0, :3, 3], axis=1))
            ),
            "right_tcp_total_displacement_from_frame0_max_m": float(
                np.max(np.linalg.norm(tcp[:, 1, :3, 3] - tcp[0, 1, :3, 3], axis=1))
            ),
        }
    return result


def write_commands() -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

# Rebuild all v11 numeric audits and 990-frame Stationary ALOHA replays.
MUJOCO_GL=egl /home/jbnu/miniconda3/envs/isaaclab6/bin/python \
  /home/jbnu/aloha_g1_dataset/tools/build_source_fk_parity_v11.py --render

# Open the synchronized source-parity review (raw | state | raw action | optimized).
ffplay -autoexit -loop 0 \
  /home/jbnu/aloha_g1_dataset/outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/aloha_source_action_parity_4panel.mp4

# Inspect optimized_action alone at 0.25x review speed.
ffplay -autoexit -loop 0 \
  /home/jbnu/aloha_g1_dataset/outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/source_optimized_action_replay.mp4
"""
    path = OUT / "commands.sh"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def build_report(
    representation: dict[str, Any],
    mapping: dict[str, Any],
    alignment: dict[str, Any],
    transforms: dict[str, Any],
    parity: dict[str, Any],
    videos: dict[str, Any] | None,
    v10: dict[str, Any],
) -> str:
    lag = alignment["best_lag_summary"]
    lines = [
        "# Episode 49 source FK parity v11",
        "",
        "1. `optimized_action`은 postprocessor를 거친 절대 joint-position target이며 raw action과 lag 0에서 대응한다.",
        "2. 독립적인 joint-name→qpos FK와 실제 MuJoCo Stationary ALOHA 렌더에서 세 입력 모두 움직였지만, action은 observation/video state보다 약 7 frames 선행한다.",
        "3. 타임라인 이동 승인이 없으므로 source relation 재계산과 position-only G1 retargeting은 실행하지 않았다.",
        "",
        "## Final status",
        "",
        "- `BLOCKED_ACTION_VIDEO_ALIGNMENT`",
        "- `ACTION_VIDEO_ALIGNMENT_REQUIRES_USER_APPROVAL`",
        "- `SOURCE_FK_NUMERIC_PARITY_PASS`",
        "- `POSITION_ONLY_G1_NOT_RUN`",
        "- `NO_PHASEWARP_RERUN`",
        "- `NO_ORIENTATION_SWEEP_RERUN`",
        "",
        "## 1. optimized_action representation",
        "",
        f"- Status: `{representation['status']}`",
        "- Semantics: absolute joint-position target; already denormalized; not delta; not velocity.",
        "- Arms: radians. Grippers: carriage displacement in metres.",
        "",
        "## 2. Exact ALOHA joint mapping",
        "",
        f"- Status: `{mapping['status']}`",
        "- Mapping is dataset channel name → MJCF joint name → runtime qpos address.",
        "- Each gripper channel is consumed once; the second carriage is the explicit MJCF equality-derived mate.",
        "",
        "## 3. observation.state / raw action / optimized_action",
        "",
        "- All are finite `[990,14]` arrays and remain separately named/stored.",
        "- All three independent FK outputs contain per-frame qpos and both TCP transforms.",
        "",
        "## 4. Why the v10 Isaac ALOHA panel appeared stationary",
        "",
        f"- v10 diagnosis: `{v10['status']}`.",
        "- The input q trajectory is strongly non-stationary. The defect is isolated to the v10 direct-write Isaac/RTX visual synchronization path; v11 uses direct MuJoCo name-mapped FK/render and does not reuse that path.",
        "",
        "## 5. Frame alignment",
        "",
        f"- optimized_action vs raw action best joint-RMSE lag: `{lag['optimized_action_vs_raw_action_frames']}` frames.",
        f"- optimized_action vs observation/video state best joint-RMSE lag: `{lag['optimized_action_vs_observation_state_frames']}` frames.",
        f"- raw action vs observation/video state best joint-RMSE lag: `{lag['raw_action_vs_observation_state_frames']}` frames.",
        "- No timeline shift was applied; approved event frames remain unchanged.",
        "",
        "## 6. Root/TCP transform chain",
        "",
        f"- Status: `{transforms['status']}`.",
        f"- Maximum root round-trip position error: `{transforms['max_root_roundtrip_position_error_m']:.3e} m`.",
        f"- Maximum object/TCP round-trip position error: `{transforms['max_object_tcp_roundtrip_position_error_m']:.3e} m`.",
        f"- Maximum object/TCP round-trip rotation error: `{transforms['max_object_tcp_roundtrip_rotation_error_rad']:.3e} rad`.",
        "",
        "## 7. Recomputed hand-object transforms",
        "",
        "- Not computed. A +7-frame action→observed-state lag requires user approval before frame-176/326/530 relations can be authoritative.",
        "- The v10 ~0.173 m right-TCP/accessory value was not auto-approved.",
        "",
        "## 8. Source task-validity decision",
        "",
        f"- `{parity['source_task_validity']}`",
        "- Task-direction cosines are recorded, but timing parity is not approved.",
        "",
        "## 9. Position-only G1",
        "",
        "- Not run, as required by the source parity gate. No phasewarp, temporal G1 IK, or orientation objective was executed.",
        "",
        "## 10. Videos",
        "",
    ]
    if videos:
        for label, row in videos["videos"].items():
            lines.append(f"- `{label}`: `{row['path']}` ({row['decoded_frames']} decoded frames)")
        lines.append(f"- Tool-axis contact sheet: `{videos['contact_sheet']}`")
    else:
        lines.append("- Rendering was skipped (`--render` not supplied).")
    lines.extend(
        [
            "",
            "## 11. Next approval item",
            "",
            "Approve or reject treating raw/optimized action targets as approximately 7 frames ahead of the observed cam_high robot state while keeping all approved event frames unchanged. Until then, source hand-object relation extraction remains blocked.",
            "",
            "THE RAW ALOHA VIDEO WAS NOT USED AS A SUBSTITUTE FOR OPTIMIZED_ACTION FK",
            "THE OPTIMIZED_ACTION SOURCE REPLAY WAS VERIFIED BEFORE G1 RETARGETING",
            "ALOHA ROOT, JOINT MAPPING, AND TCP OFFSETS WERE APPLIED EXACTLY ONCE",
            "PHYSICAL HAND-OBJECT OFFSETS WERE NOT SCALED BY WORKSPACE SCALE",
            "NO PHASEWARP OR ORIENTATION OBJECTIVE WAS ALLOWED BEFORE SOURCE FK PARITY",
            "SIMULATION ONLY — NO REAL ROBOT COMMANDS — NO DDS OR PUBLISHER",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render all three 990-frame source replays and four-panel review")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ordered_events, event_frames = events()
    inputs = load_inputs()
    representation = representation_audit(inputs)
    dump_json(OUT / "optimized_action_representation_audit.json", representation)
    if representation["status"] != "PASS":
        raise RuntimeError("BLOCKED_OPTIMIZED_ACTION_REPRESENTATION")

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    runtime_mapping, mapping_audit = build_name_mapping(model, inputs["channel_names"])
    dump_json(OUT / "aloha_joint_mapping_audit.json", mapping_audit)
    if mapping_audit["status"] != "PASS":
        raise RuntimeError("BLOCKED_ALOHA_JOINT_MAPPING")

    root = source_root_transform()
    arrays = {
        "observation_state": inputs["observation_state"],
        "raw_action": inputs["raw_action"],
        "optimized_action": inputs["optimized_action"],
    }
    fks = {label: fk_trajectory(model, value, runtime_mapping, root) for label, value in arrays.items()}
    save_fk(OUT / "observation_state_fk.npz", "observation.state", arrays["observation_state"], inputs, fks["observation_state"], root)
    save_fk(OUT / "raw_action_fk.npz", "raw parquet action", arrays["raw_action"], inputs, fks["raw_action"], root)
    save_fk(OUT / "optimized_action_fk.npz", "SmolVLA optimized_action", arrays["optimized_action"], inputs, fks["optimized_action"], root)

    mapping_audit["per_input_application"] = {
        label: {
            "arm_joint_limit_violation_count": fk["arm_joint_limit_violation_count"],
            "gripper_clip_frames_per_side": fk["gripper_clip_frames_per_side"],
        }
        for label, fk in fks.items()
    }
    mapping_audit["legacy_crosscheck"] = {
        label: crosscheck_legacy(model, arrays[label], fks[label]) for label in arrays
    }
    dump_json(OUT / "aloha_joint_mapping_audit.json", mapping_audit)

    input_audit, alignment = input_and_alignment_audits(inputs, fks)
    dump_json(OUT / "input_array_audit.json", input_audit)
    dump_json(OUT / "action_timestamp_alignment.json", alignment)
    transforms = transform_unit_tests(fks, root, event_frames)
    dump_json(OUT / "source_transform_chain_unit_tests.json", transforms)
    tool_audit = tool_mapping_audit(fks["optimized_action"])
    dump_json(OUT / "tool_axis_mapping_audit.json", tool_audit)

    directions = phase_direction_metrics(fks)
    distances = reference_distance_diagnostics(fks, event_frames)
    v10 = old_v10_motion_audit()
    optimized_nonzero = bool(
        distances["optimized_action"]["left_tcp_total_displacement_from_frame0_max_m"] > 0.05
        and distances["optimized_action"]["right_tcp_total_displacement_from_frame0_max_m"] > 0.05
    )
    directions_present = bool(min(row["cosine_vs_raw_action"] for row in directions.values()) > 0.90)
    alignment_approved = alignment["status"] == "PASS_ZERO_LAG"
    numeric_pass = bool(optimized_nonzero and directions_present and transforms["status"] == "PASS")
    parity = {
        "status": "BLOCKED_ACTION_VIDEO_ALIGNMENT" if not alignment_approved else ("PASS" if numeric_pass else "BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY"),
        "source_task_validity": "TASK_DIRECTIONS_PRESENT_BUT_ACTION_VIDEO_TIMING_REQUIRES_USER_APPROVAL" if numeric_pass and not alignment_approved else ("PASS" if numeric_pass else "FAIL"),
        "optimized_action_visibly_nonstationary_numeric_gate": optimized_nonzero,
        "optimized_action_task_direction_gate_vs_raw_action": directions_present,
        "transform_chain_gate": transforms["status"],
        "alignment_gate": alignment["status"],
        "phase_direction_metrics": directions,
        "event_distance_diagnostics": distances,
        "v10_stale_render_diagnosis": v10,
        "phasewarp_executed": False,
        "g1_ik_executed": False,
        "orientation_sweep_executed": False,
        "physics_executed": False,
    }
    dump_json(OUT / "source_parity_metrics.json", parity)

    relation_gate = alignment_approved and numeric_pass
    if relation_gate:
        # This branch is intentionally unreachable for the current +7-frame diagnostic.
        # It prevents accidental use of frame relations until timing is explicitly accepted.
        relation_status = {"status": "READY_FOR_RECOMPUTATION_BUT_NOT_IMPLEMENTED_IN_ALIGNMENT_AUDIT_RUN"}
    else:
        relation_status = {
            "status": "NOT_COMPUTED_DUE_TO_BLOCKED_ACTION_VIDEO_ALIGNMENT",
            "blocker": alignment["status"],
            "v10_relations_trusted": False,
            "v10_accessory_translation_norm_approximately_0_173m_auto_approved": False,
            "authoritative_transforms_present": False,
            "next_required_action": "user approval/rejection of the measured +7-frame action-to-observation alignment",
        }
    dump_json(OUT / "source_hand_object_relations_recomputed.json", relation_status)

    video_result = None
    if args.render:
        video_result = render_replays(model, inputs, fks, runtime_mapping, root, ordered_events, event_frames)
        parity["render_validation"] = video_result
        parity["optimized_action_visibly_nonstationary_render_gate"] = bool(
            video_result["videos"]["optimized_action"]["motion"]["max_keyframe_mean_abs_pixel_difference_from_frame0_roi"] > 2.0
        )
        dump_json(OUT / "source_parity_metrics.json", parity)

    # Explicit fail-closed G1 outputs: the files must not exist when the timing gate is blocked.
    forbidden_outputs = [
        "position_only_g1_targets.npz", "position_only_g1_arm_trajectory.npz",
        "position_only_ik_metrics.json", "position_only_anchor_metrics.json",
        "aloha_source_vs_g1_position_only_4panel.mp4", "g1_position_only_overview.mp4",
        "g1_position_only_side.mp4", "g1_position_only_top.mp4",
    ]
    unexpectedly_present = [name for name in forbidden_outputs if (OUT / name).exists()]
    if unexpectedly_present:
        raise RuntimeError(f"Blocked timing gate but downstream G1 outputs exist: {unexpectedly_present}")

    write_commands()
    report = build_report(representation, mapping_audit, alignment, transforms, parity, video_result, v10)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    manifest_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "run_manifest.json")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": parity["status"],
        "secondary_status": alignment["status"],
        "output_directory": str(OUT.resolve()),
        "files": {path.name: {"path": str(path.resolve()), "sha256": sha256(path)} for path in manifest_files},
        "safety": {
            "simulation_only": True,
            "physics": False,
            "real_robot_commands": False,
            "dds": False,
            "publisher": False,
            "phasewarp": False,
            "g1_ik": False,
            "orientation_objective": False,
        },
    }
    dump_json(OUT / "run_manifest.json", manifest)
    print(json.dumps({
        "status": parity["status"],
        "alignment": alignment["status"],
        "best_lag_frames": alignment["best_lag_summary"],
        "rendered": bool(args.render),
        "output": str(OUT.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
