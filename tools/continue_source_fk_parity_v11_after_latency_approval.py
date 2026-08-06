#!/usr/bin/env python3
"""Apply the user-approved Episode-49 +7-frame command latency convention.

Approved observation/video frames and timestamps are immutable.  This script
creates a read-only lookup view where observed frame f uses optimized_action
sample max(f-7, 0), retains action samples 983..989 as terminal command records,
and recomputes the authoritative source hand/object relations.  It performs FK
only: no G1 IK, phasewarp, orientation objective, physics, DDS, or hardware I/O.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import build_source_fk_parity_v11 as base  # noqa: E402


OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
OPT_FK = OUT / "optimized_action_fk.npz"
STATE_FK = OUT / "observation_state_fk.npz"
RAW_FK = OUT / "raw_action_fk.npz"
SOURCE_FRAMES = ROOT / "configs/episode49_source_object_frames.user_approved.json"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ACTION_NPZ = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
MODEL_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")

LAG = 7
FPS = 30.0
LATENCY_SECONDS = LAG / FPS
EVENT_TO_SAMPLE = {
    "left_phone_grasp_start": (176, 169),
    "right_accessory_grasp_start": (326, 319),
    "accessory_removed": (341, 334),
    "phone_charger_attachment_complete": (530, 523),
    "left_phone_release_complete": (586, 579),
    "right_accessory_release_complete": (646, 639),
    "task_end": (702, 695),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".incomplete")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n", encoding="utf-8")
    os.replace(temp, path)


def transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = rotation
    if translation is not None:
        value[:3, 3] = translation
    return value


def inverse(value: np.ndarray) -> np.ndarray:
    rotation = value[:3, :3]
    return transform(rotation.T, -rotation.T @ value[:3, 3])


def rotation_angle(value: np.ndarray) -> float:
    cosine = np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.array(
        [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]],
        dtype=np.float64,
    )
    return float(np.arctan2(0.5 * np.linalg.norm(skew), cosine))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def closest_point_box_distance(point: np.ndarray, center: np.ndarray, rotation: np.ndarray, half_size: np.ndarray) -> float:
    local = rotation.T @ (point - center)
    closest = np.minimum(np.maximum(local, -half_size), half_size)
    return float(np.linalg.norm(local - closest))


def nearest_gripper_geometry(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    source_point: np.ndarray,
    source_root: np.ndarray,
    side: str,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    point_model = (inverse(source_root) @ np.r_[source_point, 1.0])[:3]
    candidates: list[dict[str, Any]] = []
    prefix = f"follower_{side}_gripper_"
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith(prefix):
            continue
        geom_type = int(model.geom_type[geom_id])
        if geom_type != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        distance = closest_point_box_distance(point_model, center, rotation, np.asarray(model.geom_size[geom_id]))
        candidates.append({"geom_id": geom_id, "geom_name": name, "distance_to_box_surface_m": distance})
    if not candidates:
        raise RuntimeError(f"No named {side} gripper box geometries")
    return min(candidates, key=lambda row: row["distance_to_box_surface_m"])


def object_surface_samples(kind: str) -> np.ndarray:
    """Dense local-frame surface samples for a diagnostic geometry-distance gate."""
    if kind == "phone":
        half = np.array([0.1496, 0.00795, 0.0715], dtype=np.float64) / 2.0
        values = [np.linspace(-extent, extent, 7) for extent in half]
        points = []
        for x in values[0]:
            for y in values[1]:
                for z in values[2]:
                    if (
                        np.isclose(abs(x), half[0])
                        or np.isclose(abs(y), half[1])
                        or np.isclose(abs(z), half[2])
                    ):
                        points.append([x, y, z])
        return np.asarray(points, dtype=np.float64)
    if kind == "accessory":
        layout = json.loads((ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json").read_text(encoding="utf-8"))
        acc = layout["accessory"]
        points: list[list[float]] = []
        for radius in (
            float(acc["main_inner_diameter"]) / 2.0,
            float(acc["main_outer_diameter"]) / 2.0,
        ):
            for angle in np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False):
                for depth in (-float(acc["main_depth"]) / 2.0, float(acc["main_depth"]) / 2.0):
                    points.append([radius * np.cos(angle), depth, radius * np.sin(angle)])
        support = np.asarray(acc["support_center_offset_from_main_xyz"], dtype=np.float64)
        for radius in (
            float(acc["support_inner_diameter"]) / 2.0,
            float(acc["support_outer_diameter"]) / 2.0,
        ):
            for angle in np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False):
                for depth in (-float(acc["support_depth"]) / 2.0, float(acc["support_depth"]) / 2.0):
                    points.append((support + np.array([radius * np.cos(angle), depth, radius * np.sin(angle)])).tolist())
        return np.asarray(points, dtype=np.float64)
    raise ValueError(kind)


def nearest_object_surface_to_gripper_geometry(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    object_pose_source: np.ndarray,
    source_root: np.ndarray,
    side: str,
    object_kind: str,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    object_model = inverse(source_root) @ object_pose_source
    local_samples = object_surface_samples(object_kind)
    samples_model = (object_model[:3, :3] @ local_samples.T).T + object_model[:3, 3]
    best: dict[str, Any] | None = None
    prefix = f"follower_{side}_gripper_"
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith(prefix) or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        half_size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        local = (rotation.T @ (samples_model - center).T).T
        closest = np.minimum(np.maximum(local, -half_size), half_size)
        distances = np.linalg.norm(local - closest, axis=1)
        sample_index = int(np.argmin(distances))
        row = {
            "object_kind": object_kind,
            "sample_count": int(len(local_samples)),
            "geom_id": geom_id,
            "geom_name": name,
            "sampled_object_surface_to_gripper_box_distance_m": float(distances[sample_index]),
            "nearest_object_surface_point_source_scene_m": (
                object_pose_source[:3, :3] @ local_samples[sample_index] + object_pose_source[:3, 3]
            ),
        }
        if best is None or row["sampled_object_surface_to_gripper_box_distance_m"] < best["sampled_object_surface_to_gripper_box_distance_m"]:
            best = row
    if best is None:
        raise RuntimeError(f"No named {side} gripper boxes for {object_kind}")
    return best


def relation_record(
    name: str,
    observed_frame: int,
    action_sample: int,
    object_pose: np.ndarray,
    tcp_pose: np.ndarray,
    object_from_tcp: np.ndarray,
    nearest: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "name": name,
        "observed_frame": observed_frame,
        "video_timestamp_s": observed_frame / FPS,
        "optimized_action_sample_index": action_sample,
        "optimized_action_sample_timestamp_s": action_sample / FPS,
        "action_to_observation_lag_frames": LAG,
        "T_source_object_from_ALOHA_TCP": object_from_tcp,
        "T_source_ALOHA_TCP_from_object": inverse(object_from_tcp),
        "translation_norm_m": float(np.linalg.norm(object_from_tcp[:3, 3])),
        "rotation_angle_deg": float(np.degrees(rotation_angle(object_from_tcp[:3, :3]))),
        "object_center_source_scene_m": object_pose[:3, 3],
        "tcp_position_source_scene_m": tcp_pose[:3, 3],
        "tcp_to_object_center_distance_m": float(np.linalg.norm(tcp_pose[:3, 3] - object_pose[:3, 3])),
    }
    if nearest is not None:
        row["nearest_gripper_geometry"] = nearest
    return row


def phase_direction_metrics(aligned: np.ndarray, state: np.ndarray, raw_aligned: np.ndarray) -> dict[str, Any]:
    phases = [
        ("phone_approach", 0, 176, 0),
        ("phone_lift_portrait", 176, 223, 0),
        ("accessory_approach", 223, 326, 1),
        ("accessory_removal", 326, 341, 1),
        ("phone_to_charger", 380, 530, 0),
    ]
    result: dict[str, Any] = {}
    for name, start, stop, side in phases:
        query = aligned[stop, side, :3, 3] - aligned[start, side, :3, 3]
        row: dict[str, Any] = {
            "side": "left" if side == 0 else "right",
            "frames": [start, stop],
            "optimized_aligned_displacement_m": float(np.linalg.norm(query)),
        }
        for label, reference in (("observation_state", state), ("raw_action_aligned", raw_aligned)):
            ref = reference[stop, side, :3, 3] - reference[start, side, :3, 3]
            row[f"direction_cosine_vs_{label}"] = float(
                np.dot(query, ref) / (np.linalg.norm(query) * np.linalg.norm(ref) + 1e-15)
            )
            row[f"endpoint_error_vs_{label}_m"] = float(
                np.linalg.norm(aligned[stop, side, :3, 3] - reference[stop, side, :3, 3])
            )
        result[name] = row
    return result


def main() -> int:
    for path in (OPT_FK, STATE_FK, RAW_FK, SOURCE_FRAMES, TIMELINE, ACTION_NPZ, MODEL_XML):
        if not path.is_file():
            raise FileNotFoundError(path)
    timeline_hash_before = sha256(TIMELINE)
    timeline_payload = json.loads(TIMELINE.read_text(encoding="utf-8"))
    timeline_events = {str(row["event"]): int(row["frame"]) for row in timeline_payload["events"]}
    for event, (observed, sample) in EVENT_TO_SAMPLE.items():
        if timeline_events[event] != observed or sample != observed - LAG:
            raise RuntimeError(f"Approved event/sample mismatch for {event}")

    optimized_fk = load_npz(OPT_FK)
    state_fk = load_npz(STATE_FK)
    raw_fk = load_npz(RAW_FK)
    with np.load(ACTION_NPZ, allow_pickle=False) as source:
        optimized_action = source["optimized_action"].copy()
        action_timestamp = source["timestamp"].copy()
        action_frame_index = source["frame_index"].copy()
    if not np.array_equal(optimized_action, optimized_fk["source_joint_array"]):
        raise RuntimeError("optimized_action changed between the source NPZ and v11 FK")

    observed_frames = np.arange(990, dtype=np.int64)
    action_lookup = np.maximum(observed_frames - LAG, 0)
    precommand_hold = observed_frames < LAG
    terminal_indices = np.arange(983, 990, dtype=np.int64)
    if not np.array_equal(action_lookup[:7], np.zeros(7, dtype=np.int64)):
        raise RuntimeError("pre-command hold mapping is not exactly action sample 0")
    if int(action_lookup[-1]) != 982 or not np.array_equal(terminal_indices, np.arange(action_lookup[-1] + 1, 990)):
        raise RuntimeError("terminal command sample retention mismatch")

    optimized_tcp_original = np.stack(
        (optimized_fk["left_tcp_transform"], optimized_fk["right_tcp_transform"]), axis=1
    )
    state_tcp = np.stack((state_fk["left_tcp_transform"], state_fk["right_tcp_transform"]), axis=1)
    raw_tcp_original = np.stack((raw_fk["left_tcp_transform"], raw_fk["right_tcp_transform"]), axis=1)
    optimized_tcp_aligned = optimized_tcp_original[action_lookup]
    raw_tcp_aligned = raw_tcp_original[action_lookup]

    approval = {
        "status": "USER_APPROVED_ACTION_TO_OBSERVATION_LATENCY",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "scope": "EPISODE_49_SOURCE_FK_AND_RELATION_LOOKUP_ONLY",
        "action_to_observation_lag_frames": LAG,
        "action_sample_for_observed_frame": "observed_frame - 7",
        "fps": FPS,
        "latency_seconds": LATENCY_SECONDS,
        "approved_event_frames_unchanged": True,
        "video_observation_timestamps_unchanged": True,
        "optimized_action_values_unchanged": True,
        "optimized_action_order_unchanged": True,
        "phase_durations_unchanged": True,
        "left_right_roles_unchanged": True,
        "task_order_unchanged": True,
        "raw_video_shifted": False,
        "timeline_json_rewritten": False,
        "frames_0_6": "diagnostic pre-command hold using optimized_action[0]",
        "negative_indexing": False,
        "wrap_or_extrapolation": False,
        "terminal_samples_983_989": "retained as post-command terminal samples; not discarded",
        "event_to_action_sample": {
            event: {"observed_frame": observed, "optimized_action_sample": sample}
            for event, (observed, sample) in EVENT_TO_SAMPLE.items()
        },
        "provenance": {
            "optimized_action_npz": str(ACTION_NPZ.resolve()),
            "optimized_action_npz_sha256": sha256(ACTION_NPZ),
            "approved_timeline": str(TIMELINE.resolve()),
            "approved_timeline_sha256_before": timeline_hash_before,
        },
    }
    dump(OUT / "action_to_observation_latency.approved.json", approval)
    np.savez_compressed(
        OUT / "source_action_latency_alignment.npz",
        optimized_action=optimized_action,
        optimized_action_original_frame_index=action_frame_index,
        optimized_action_original_timestamp=action_timestamp,
        observed_frame_index=observed_frames,
        observed_timestamp=state_fk["timestamp"],
        action_sample_index_for_observed_frame=action_lookup,
        diagnostic_precommand_hold_mask=precommand_hold,
        observation_aligned_optimized_action=optimized_action[action_lookup],
        observation_aligned_left_tcp_transform=optimized_tcp_aligned[:, 0],
        observation_aligned_right_tcp_transform=optimized_tcp_aligned[:, 1],
        post_command_terminal_sample_indices=terminal_indices,
        post_command_terminal_action=optimized_action[terminal_indices],
        post_command_terminal_timestamp=action_timestamp[terminal_indices],
        action_to_observation_lag_frames=np.array(LAG),
        fps=np.array(FPS),
        latency_seconds=np.array(LATENCY_SECONDS),
        timeline_modified=np.array(False),
    )

    frames = json.loads(SOURCE_FRAMES.read_text(encoding="utf-8"))
    phone_initial = np.asarray(frames["T_source_scene_from_phone"], dtype=np.float64)
    accessory_initial = np.asarray(frames["T_source_scene_from_accessory"], dtype=np.float64)
    charger_pad = np.asarray(frames["T_source_scene_from_charger_pad"], dtype=np.float64)
    phone_from_accessory = inverse(phone_initial) @ accessory_initial

    grasp_frame, grasp_sample = EVENT_TO_SAMPLE["left_phone_grasp_start"]
    accessory_frame, accessory_sample = EVENT_TO_SAMPLE["right_accessory_grasp_start"]
    removed_frame, removed_sample = EVENT_TO_SAMPLE["accessory_removed"]
    attach_frame, attach_sample = EVENT_TO_SAMPLE["phone_charger_attachment_complete"]
    _, left_release_sample = EVENT_TO_SAMPLE["left_phone_release_complete"]
    right_release_frame, right_release_sample = EVENT_TO_SAMPLE["right_accessory_release_complete"]

    phone_from_left_tcp = inverse(phone_initial) @ optimized_tcp_aligned[grasp_frame, 0]
    left_tcp_from_phone = inverse(phone_from_left_tcp)
    phone_pose = np.repeat(phone_initial[None], 990, axis=0)
    phone_pose[grasp_frame : attach_frame + 1] = np.einsum(
        "tij,jk->tik", optimized_tcp_aligned[grasp_frame : attach_frame + 1, 0], left_tcp_from_phone
    )
    phone_pose[attach_frame + 1 :] = phone_pose[attach_frame]

    accessory_at_grasp = phone_pose[accessory_frame] @ phone_from_accessory
    accessory_from_right_tcp = inverse(accessory_at_grasp) @ optimized_tcp_aligned[accessory_frame, 1]
    right_tcp_from_accessory = inverse(accessory_from_right_tcp)
    accessory_pose = np.einsum("tij,jk->tik", phone_pose, phone_from_accessory)
    accessory_pose[accessory_frame : right_release_frame + 1] = np.einsum(
        "tij,jk->tik", optimized_tcp_aligned[accessory_frame : right_release_frame + 1, 1], right_tcp_from_accessory
    )
    accessory_pose[right_release_frame + 1 :] = accessory_pose[right_release_frame]

    phone_at_charger = phone_pose[attach_frame]
    charger_pad_from_phone = inverse(charger_pad) @ phone_at_charger
    phone_from_charger_pad = inverse(charger_pad_from_phone)
    accessory_removal_transform = inverse(accessory_at_grasp) @ accessory_pose[removed_frame]

    np.savez_compressed(
        OUT / "source_phone_pose_trajectory_latency_aligned.npz",
        observed_frame_index=observed_frames,
        observed_timestamp=state_fk["timestamp"],
        action_sample_index_for_observed_frame=action_lookup,
        T_source_scene_from_phone=phone_pose,
        T_source_phone_from_left_ALOHA_TCP=phone_from_left_tcp,
        T_source_left_ALOHA_TCP_from_phone=left_tcp_from_phone,
        grasp_observed_frame=np.array(grasp_frame),
        grasp_action_sample=np.array(grasp_sample),
        charger_attachment_observed_frame=np.array(attach_frame),
        charger_attachment_action_sample=np.array(attach_sample),
        action_to_observation_lag_frames=np.array(LAG),
    )
    np.savez_compressed(
        OUT / "source_accessory_pose_trajectory_latency_aligned.npz",
        observed_frame_index=observed_frames,
        observed_timestamp=state_fk["timestamp"],
        action_sample_index_for_observed_frame=action_lookup,
        T_source_scene_from_accessory=accessory_pose,
        T_source_accessory_from_right_ALOHA_TCP=accessory_from_right_tcp,
        T_source_right_ALOHA_TCP_from_accessory=right_tcp_from_accessory,
        T_accessory_grasp_from_removed=accessory_removal_transform,
        grasp_observed_frame=np.array(accessory_frame),
        grasp_action_sample=np.array(accessory_sample),
        removed_observed_frame=np.array(removed_frame),
        removed_action_sample=np.array(removed_sample),
        release_observed_frame=np.array(right_release_frame),
        release_action_sample=np.array(right_release_sample),
        action_to_observation_lag_frames=np.array(LAG),
    )

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    root = np.asarray(optimized_fk["source_aloha_root_transform"], dtype=np.float64)
    left_nearest = nearest_gripper_geometry(
        model, optimized_fk["qpos"][grasp_sample], phone_initial[:3, 3], root, "left"
    )
    left_nearest["object_surface"] = nearest_object_surface_to_gripper_geometry(
        model, optimized_fk["qpos"][grasp_sample], phone_initial, root, "left", "phone"
    )
    right_nearest = nearest_gripper_geometry(
        model, optimized_fk["qpos"][accessory_sample], accessory_at_grasp[:3, 3], root, "right"
    )
    right_nearest["object_surface"] = nearest_object_surface_to_gripper_geometry(
        model, optimized_fk["qpos"][accessory_sample], accessory_at_grasp, root, "right", "accessory"
    )
    right_removed_nearest = nearest_gripper_geometry(
        model, optimized_fk["qpos"][removed_sample], accessory_pose[removed_frame, :3, 3], root, "right"
    )
    right_removed_nearest["object_surface"] = nearest_object_surface_to_gripper_geometry(
        model, optimized_fk["qpos"][removed_sample], accessory_pose[removed_frame], root, "right", "accessory"
    )
    left_charger_nearest = nearest_gripper_geometry(
        model, optimized_fk["qpos"][attach_sample], phone_at_charger[:3, 3], root, "left"
    )
    left_charger_nearest["object_surface"] = nearest_object_surface_to_gripper_geometry(
        model, optimized_fk["qpos"][attach_sample], phone_at_charger, root, "left", "phone"
    )

    records = {
        "left_phone_grasp": relation_record(
            "left_phone_grasp", grasp_frame, grasp_sample, phone_initial,
            optimized_tcp_aligned[grasp_frame, 0], phone_from_left_tcp, left_nearest,
        ),
        "right_accessory_grasp": relation_record(
            "right_accessory_grasp", accessory_frame, accessory_sample, accessory_at_grasp,
            optimized_tcp_aligned[accessory_frame, 1], accessory_from_right_tcp, right_nearest,
        ),
        "right_accessory_removed": {
            **relation_record(
                "right_accessory_removed", removed_frame, removed_sample, accessory_pose[removed_frame],
                optimized_tcp_aligned[removed_frame, 1],
                inverse(accessory_pose[removed_frame]) @ optimized_tcp_aligned[removed_frame, 1],
                right_removed_nearest,
            ),
            "T_source_accessory_grasp_from_removed_accessory": accessory_removal_transform,
            "removal_translation_source_accessory_frame_m": accessory_removal_transform[:3, 3],
            "removal_translation_norm_m": float(np.linalg.norm(accessory_removal_transform[:3, 3])),
            "removal_rotation_deg": float(np.degrees(rotation_angle(accessory_removal_transform[:3, :3]))),
        },
        "phone_charger_attachment": {
            **relation_record(
                "phone_charger_attachment", attach_frame, attach_sample, phone_at_charger,
                optimized_tcp_aligned[attach_frame, 0], inverse(phone_at_charger) @ optimized_tcp_aligned[attach_frame, 0],
                left_charger_nearest,
            ),
            "T_source_charger_pad_from_phone": charger_pad_from_phone,
            "T_source_phone_from_charger_pad": phone_from_charger_pad,
            "phone_center_to_charger_pad_face_center_m": float(
                np.linalg.norm(phone_at_charger[:3, 3] - charger_pad[:3, 3])
            ),
        },
    }

    # Exact bidirectional chain tests for every authoritative relation.
    chain_tests: dict[str, Any] = {}
    maximum_position = 0.0
    maximum_rotation = 0.0
    for key, object_pose, tcp_pose, object_from_tcp in (
        ("left_phone_grasp", phone_initial, optimized_tcp_aligned[grasp_frame, 0], phone_from_left_tcp),
        ("right_accessory_grasp", accessory_at_grasp, optimized_tcp_aligned[accessory_frame, 1], accessory_from_right_tcp),
        ("right_accessory_removed", accessory_pose[removed_frame], optimized_tcp_aligned[removed_frame, 1], inverse(accessory_pose[removed_frame]) @ optimized_tcp_aligned[removed_frame, 1]),
        ("phone_charger_attachment", phone_at_charger, optimized_tcp_aligned[attach_frame, 0], inverse(phone_at_charger) @ optimized_tcp_aligned[attach_frame, 0]),
    ):
        reconstructed_tcp = object_pose @ object_from_tcp
        reconstructed_object = tcp_pose @ inverse(object_from_tcp)
        position_error = max(
            float(np.linalg.norm(reconstructed_tcp[:3, 3] - tcp_pose[:3, 3])),
            float(np.linalg.norm(reconstructed_object[:3, 3] - object_pose[:3, 3])),
        )
        rotation_error = max(
            rotation_angle(reconstructed_tcp[:3, :3] @ tcp_pose[:3, :3].T),
            rotation_angle(reconstructed_object[:3, :3] @ object_pose[:3, :3].T),
        )
        chain_tests[key] = {"position_error_m": position_error, "rotation_error_rad": rotation_error}
        maximum_position = max(maximum_position, position_error)
        maximum_rotation = max(maximum_rotation, rotation_error)

    semantic_grasp_gate_m = 0.020
    surface_distances = {
        "left_phone_grasp_m": float(left_nearest["object_surface"]["sampled_object_surface_to_gripper_box_distance_m"]),
        "right_accessory_grasp_m": float(right_nearest["object_surface"]["sampled_object_surface_to_gripper_box_distance_m"]),
        "right_accessory_removed_m": float(right_removed_nearest["object_surface"]["sampled_object_surface_to_gripper_box_distance_m"]),
        "left_phone_at_charger_m": float(left_charger_nearest["object_surface"]["sampled_object_surface_to_gripper_box_distance_m"]),
    }
    semantic_relation_pass = bool(
        surface_distances["left_phone_grasp_m"] <= semantic_grasp_gate_m
        and surface_distances["right_accessory_grasp_m"] <= semantic_grasp_gate_m
        and surface_distances["right_accessory_removed_m"] <= semantic_grasp_gate_m
    )
    source_relations = {
        "status": (
            "SOURCE_HAND_OBJECT_RELATIONS_RECOMPUTED_WITH_USER_APPROVED_LATENCY"
            if semantic_relation_pass
            else "BLOCKED_SOURCE_HAND_OBJECT_RELATION"
        ),
        "authoritative": semantic_relation_pass,
        "v11_failure_classification": (
            None if semantic_relation_pass else "BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY"
        ),
        "action_to_observation_lag_frames": LAG,
        "action_sample_for_observed_frame": "observed_frame - 7",
        "fps": FPS,
        "latency_seconds": LATENCY_SECONDS,
        "approved_event_frames_unchanged": True,
        "video_timestamps_unchanged": True,
        "optimized_action_values_and_order_unchanged": True,
        "workspace_scale_applied_to_physical_hand_object_offsets": False,
        "source_absolute_world_coordinates_for_target_generation": False,
        "phone_from_accessory_attachment_transform": phone_from_accessory,
        "relations": records,
        "release_samples": {
            "left": {"observed_frame": 586, "action_sample": left_release_sample},
            "right": {"observed_frame": 646, "action_sample": right_release_sample},
            "task_end": {"observed_frame": 702, "action_sample": EVENT_TO_SAMPLE["task_end"][1]},
        },
        "round_trip_tests": {
            "position_tolerance_m": 1e-8,
            "rotation_tolerance_rad": 1e-8,
            "maximum_position_error_m": maximum_position,
            "maximum_rotation_error_rad": maximum_rotation,
            "pass": bool(maximum_position <= 1e-8 and maximum_rotation <= 1e-8),
            "details": chain_tests,
        },
        "semantic_grasp_proximity_gate": {
            "threshold_m": semantic_grasp_gate_m,
            "method": "dense source-object surface samples to named Stationary ALOHA gripper pad/tip OBBs",
            "distances": surface_distances,
            "pass": semantic_relation_pass,
            "exact_blocker": (
                None
                if semantic_relation_pass
                else "At observed frame 326 / optimized_action[319], the reconstructed moving accessory is not within grasp proximity of the right gripper under the approved frame-176 rigid phone carrier relation."
            ),
        },
        "provenance": {
            "latency_approval": str((OUT / "action_to_observation_latency.approved.json").resolve()),
            "optimized_action_fk": str(OPT_FK.resolve()),
            "optimized_action_fk_sha256": sha256(OPT_FK),
            "source_object_frames": str(SOURCE_FRAMES.resolve()),
            "source_object_frames_sha256": sha256(SOURCE_FRAMES),
            "stationary_model": str(MODEL_XML.resolve()),
            "stationary_model_sha256": sha256(MODEL_XML),
        },
    }
    if not source_relations["round_trip_tests"]["pass"]:
        source_relations["status"] = "BLOCKED_SOURCE_TRANSFORM_CHAIN"
        source_relations["authoritative"] = False
    dump(OUT / "source_hand_object_relations_recomputed.json", source_relations)

    direction = phase_direction_metrics(optimized_tcp_aligned, state_tcp, raw_tcp_aligned)
    minimum_direction = min(
        row["direction_cosine_vs_raw_action_aligned"] for row in direction.values()
    )
    event_errors = {
        "frame_176_left_tcp_vs_observation_state_m": float(
            np.linalg.norm(optimized_tcp_aligned[176, 0, :3, 3] - state_tcp[176, 0, :3, 3])
        ),
        "frame_326_right_tcp_vs_observation_state_m": float(
            np.linalg.norm(optimized_tcp_aligned[326, 1, :3, 3] - state_tcp[326, 1, :3, 3])
        ),
        "frame_341_right_tcp_vs_observation_state_m": float(
            np.linalg.norm(optimized_tcp_aligned[341, 1, :3, 3] - state_tcp[341, 1, :3, 3])
        ),
        "frame_530_left_tcp_vs_observation_state_m": float(
            np.linalg.norm(optimized_tcp_aligned[530, 0, :3, 3] - state_tcp[530, 0, :3, 3])
        ),
    }
    parity = json.loads((OUT / "source_parity_metrics.json").read_text(encoding="utf-8"))
    relation_gate_pass = bool(source_relations["round_trip_tests"]["pass"] and semantic_relation_pass)
    parity.update(
        {
            "status": (
                "PASS_USER_APPROVED_ACTION_TO_OBSERVATION_LATENCY"
                if relation_gate_pass
                else "BLOCKED_OPTIMIZED_ACTION_TASK_VALIDITY"
            ),
            "source_task_validity": (
                "PASS_LATENCY_ALIGNED_SOURCE_TASK_DIRECTIONS"
                if relation_gate_pass
                else "ACTION_FK_PARITY_PASS_BUT_OBJECT_RELATION_SEMANTICS_FAIL"
            ),
            "alignment_gate": "PASS_USER_APPROVED_PLUS_7_FRAME_COMMAND_LATENCY",
            "action_to_observation_lag_frames": LAG,
            "action_sample_for_observed_frame": "observed_frame - 7",
            "fps": FPS,
            "latency_seconds": LATENCY_SECONDS,
            "timeline_shift_applied": False,
            "latency_aligned_phase_direction_metrics": direction,
            "latency_aligned_minimum_direction_cosine_vs_raw_action": minimum_direction,
            "latency_aligned_event_tcp_errors_vs_observation_state": event_errors,
            "source_relation_gate_passed": relation_gate_pass,
            "phasewarp_executed": False,
            "g1_ik_executed": False,
            "orientation_sweep_executed": False,
        }
    )
    dump(OUT / "source_parity_metrics.json", parity)

    alignment = json.loads((OUT / "action_timestamp_alignment.json").read_text(encoding="utf-8"))
    alignment.update(
        {
            "status": "PASS_USER_APPROVED_ACTION_TO_OBSERVATION_LATENCY",
            "user_approval_record": str((OUT / "action_to_observation_latency.approved.json").resolve()),
            "action_to_observation_lag_frames": LAG,
            "action_sample_for_observed_frame": "observed_frame - 7",
            "fps": FPS,
            "latency_seconds": LATENCY_SECONDS,
            "timeline_shift_applied_frames": 0,
            "raw_video_shifted": False,
            "approved_event_frames_modified": False,
            "diagnostic_precommand_hold_frames": list(range(7)),
            "post_command_terminal_sample_indices_retained": terminal_indices,
        }
    )
    dump(OUT / "action_timestamp_alignment.json", alignment)

    timeline_hash_after = sha256(TIMELINE)
    approval["provenance"]["approved_timeline_sha256_after"] = timeline_hash_after
    approval["provenance"]["timeline_byte_identical_before_after"] = timeline_hash_before == timeline_hash_after
    dump(OUT / "action_to_observation_latency.approved.json", approval)
    if timeline_hash_before != timeline_hash_after:
        raise RuntimeError("Approved timeline changed")

    summary = {
        "status": source_relations["status"],
        "parity_status": parity["status"],
        "action_to_observation_lag_frames": LAG,
        "latency_seconds": LATENCY_SECONDS,
        "event_action_samples": approval["event_to_action_sample"],
        "minimum_phase_direction_cosine_vs_raw_action_aligned": minimum_direction,
        "event_tcp_errors_vs_observation_state_m": event_errors,
        "relation_translation_norms_m": {
            key: float(value["translation_norm_m"]) for key, value in records.items()
        },
        "phone_center_to_charger_pad_face_center_m": records["phone_charger_attachment"]["phone_center_to_charger_pad_face_center_m"],
        "semantic_grasp_surface_distances_m": surface_distances,
        "semantic_relation_gate_pass": semantic_relation_pass,
        "output": str(OUT.resolve()),
    }
    print(json.dumps(summary, indent=2, default=default))
    return 0 if relation_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
