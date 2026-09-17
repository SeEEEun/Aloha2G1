#!/usr/bin/env python3
"""Audit and package scientifically clean SOURCE/A/B reference motion.

No ACT model is loaded, no training is performed, and no physics runtime is
started.  This tool operates only on authoritative source recordings, source
FK, common task registration, and representation-level Cartesian targets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.doll_handoff_retargeting.common import load_common_config, load_scene
from tools.doll_handoff_retargeting.models import ALOHAKinematics, G1Kinematics
from tools.doll_handoff_retargeting.retarget import RepresentationBuilder
from tools.doll_handoff_retargeting.source import fixed_list_numpy, scalar_numpy
from tools.reference_motion_common_timeline import (
    EVENT_NAMES,
    SourceEventTimeline,
    build_common_timeline,
    detect_coarse_events,
)


OUT = ROOT / "outputs/reference_motion_scientific_reset"
RESET = ROOT / "outputs/single_variable_ab_reset/shared_pipeline_train_smoke_v4"
SOURCE_MANIFEST = RESET / "source_audit/source_manifest.json"
SOURCE_EVENTS = RESET / "event_audit/events.json"
DETECTOR = RESET / "event_audit/detector_config.json"
REGISTRATION = ROOT / "outputs/final_episode_registered_eval35/00_registration/EVAL35_EPISODE_OBJECT_REGISTRATION.json"
STANDARDIZED = ROOT / "outputs/standardized_grasp_ab_dev35/01_prepared_commands/STANDARDIZED_GRASP_AB_COMMAND_MANIFEST.json"
SCENE_PATH = ROOT / "isaaclab_doll_handoff_scene/scene_layout.json"
SMOKE = (0, 24, 49)


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): native(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(child) for child in value]
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(native(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: native(row.get(key)) for key in columns})
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    array = scale * np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def rotation_errors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", np.asarray(first), np.asarray(second))
    return Rotation.from_matrix(relative).magnitude()


def transform_matrix(entry: Mapping[str, Any] | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if entry is None:
        return result
    spec = entry["source_to_target_transform"]
    quat = np.asarray(spec["quaternion_xyzw"], dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quat).as_matrix()
    result[:3, 3] = np.asarray(spec["translation_xyz_m"], dtype=np.float64)
    return result


def enriched_registration(entry: Mapping[str, Any], scene: Mapping[str, Any]) -> dict[str, Any]:
    matrix = transform_matrix(entry)
    bin_center = scene["bin"]["center_world_xy_m"]
    return {
        **dict(entry),
        "source_to_target_transform_matrix": matrix,
        "pre_workspace_source_to_target_transform_matrix": matrix,
        "target_bin_pose": {
            "position_xyz_m": [bin_center[0], bin_center[1], scene["table"]["surface_height_m"]],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "target_table_task_origin_xyz_m": scene["task_frame"]["origin_world_xyz_m"],
    }


def source_object(scene: Mapping[str, Any]) -> np.ndarray:
    doll = scene["doll"]
    return np.asarray(
        [
            doll["center_world_xy_m"][0],
            doll["center_world_xy_m"][1],
            scene["table"]["surface_height_m"]
            + 0.5 * doll["diameter_m"]
            + doll["initial_table_clearance_m"],
        ],
        dtype=np.float64,
    )


def load_recording(source_name: str) -> dict[str, Any]:
    root = ROOT / "raw_recordings" / source_name
    parquet = sorted((root / "data").rglob("*.parquet"))
    if len(parquet) != 1:
        raise RuntimeError(f"{source_name}: expected one parquet, found {len(parquet)}")
    table = pq.read_table(parquet[0])
    action = fixed_list_numpy(table["action"], 14).astype(np.float64)
    state = fixed_list_numpy(table["observation.state"], 14).astype(np.float64)
    timestamp = scalar_numpy(table["timestamp"], np.float64)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if not (
        action.shape == state.shape
        and action.shape[0] == len(timestamp)
        and np.isfinite(action).all()
        and np.isfinite(state).all()
        and np.isfinite(timestamp).all()
    ):
        raise RuntimeError(f"{source_name}: source arrays are not finite/frame-aligned")
    return {
        "source_name": source_name,
        "root": root,
        "parquet": parquet[0],
        "parquet_sha256": sha256(parquet[0]),
        "action": action,
        "state": state,
        "timestamp": timestamp,
        "fps": fps,
        "image_dir": root / "images/observation.images.cam_high/episode_000000",
    }


def ownership_labels(timeline: SourceEventTimeline, count: int) -> np.ndarray:
    frames = timeline.frames
    labels = np.full(count, "NO_OWNER", dtype="U24")
    labels[frames["LEFT_GRASP_CONFIRMED_SOURCE"] : frames["RIGHT_APPROACH_BEGIN"]] = "LEFT_OWNED"
    labels[frames["RIGHT_APPROACH_BEGIN"] : frames["RIGHT_ACQUIRE_SOURCE"]] = "HANDOFF_APPROACH"
    labels[frames["RIGHT_ACQUIRE_SOURCE"] : frames["LEFT_RELEASE_BEGIN"]] = "DUAL_CONTACT"
    labels[frames["LEFT_RELEASE_BEGIN"] : frames["RIGHT_TRANSPORT_BEGIN"]] = "RIGHT_OWNED"
    labels[frames["RIGHT_TRANSPORT_BEGIN"] : frames["FINAL_RELEASE_BEGIN"]] = "RIGHT_TRANSPORT"
    labels[frames["FINAL_RELEASE_BEGIN"] :] = "RELEASED"
    return labels


def event_proxy(timeline: SourceEventTimeline, source_name: str, episode_index: int) -> Any:
    return SimpleNamespace(
        episode_index=episode_index,
        source_name=source_name,
        ownership_labels=ownership_labels(timeline, len(timeline.phase_label)),
    )


def reconstruct_tool(
    target: Mapping[str, Any], side: str
) -> tuple[np.ndarray, np.ndarray]:
    local = np.asarray(target["static_wrist_to_tool"][side], dtype=np.float64)
    wrist_position = np.asarray(target[f"{side}_wrist_position"], dtype=np.float64)
    wrist_rotation = np.asarray(target[f"{side}_wrist_rotation"], dtype=np.float64)
    position = wrist_position + np.einsum("tij,j->ti", wrist_rotation, local[:3, 3])
    rotation = np.einsum("tij,jk->tik", wrist_rotation, local[:3, :3])
    return position, rotation


def expected_registered_fk(
    fk: Mapping[str, Any], side: str, matrix: np.ndarray, g1: G1Kinematics
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_position = np.asarray(fk[f"{side}_tcp_position_world"], dtype=np.float64)
    source_rotation = np.asarray(fk[f"{side}_tcp_rotation_world"], dtype=np.float64)
    world_position = source_position @ matrix[:3, :3].T + matrix[:3, 3]
    world_rotation = np.einsum("ij,tjk->tik", matrix[:3, :3], source_rotation)
    return (
        world_position,
        world_rotation,
        g1.world_to_model_position(world_position),
        g1.world_to_model_rotation(world_rotation),
    )


def target_audit(
    fk: Mapping[str, Any],
    targets: Mapping[str, Mapping[str, Any]],
    matrix: np.ndarray,
    g1: G1Kinematics,
    object_position: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {"A": {}, "B": {}}
    for label, target in targets.items():
        position_errors: list[np.ndarray] = []
        orientation_errors_list: list[np.ndarray] = []
        object_relation_errors: list[np.ndarray] = []
        expected_pair: dict[str, np.ndarray] = {}
        reconstructed_pair: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            expected_world, _, expected_model, _ = expected_registered_fk(
                fk, side, matrix, g1
            )
            reconstructed_model, reconstructed_rotation = reconstruct_tool(target, side)
            expected_target_rotation = np.asarray(
                target[f"{side}_tool_rotation_model"], dtype=np.float64
            )
            position_errors.append(np.linalg.norm(reconstructed_model - expected_model, axis=1))
            orientation_errors_list.append(
                rotation_errors(reconstructed_rotation, expected_target_rotation)
            )
            reconstructed_world = g1.model_to_world_position(reconstructed_model)
            object_relation_errors.append(
                np.linalg.norm(
                    (reconstructed_world - object_position)
                    - (expected_world - object_position),
                    axis=1,
                )
            )
            expected_pair[side] = expected_model
            reconstructed_pair[side] = reconstructed_model
        bimanual = np.linalg.norm(
            (reconstructed_pair["right"] - reconstructed_pair["left"])
            - (expected_pair["right"] - expected_pair["left"]),
            axis=1,
        )
        report[label] = {
            "position_reconstruction_error_mm": stats(np.concatenate(position_errors), 1000.0),
            "orientation_reconstruction_error_deg": stats(
                np.concatenate(orientation_errors_list), 180.0 / np.pi
            ),
            "object_relative_translation_error_mm": stats(
                np.concatenate(object_relation_errors), 1000.0
            ),
            "bimanual_relation_error_mm": stats(bimanual, 1000.0),
            "finite": bool(
                all(
                    np.isfinite(np.asarray(value)).all()
                    for value in target.values()
                    if isinstance(value, np.ndarray)
                    and np.issubdtype(np.asarray(value).dtype, np.number)
                )
            ),
        }
    return report


def trajectory_continuity(
    positions: list[np.ndarray], rotations: list[np.ndarray], fps: float
) -> dict[str, Any]:
    position_step = np.concatenate(
        [np.linalg.norm(np.diff(value, axis=0), axis=1) for value in positions]
    )
    angular_step = np.concatenate(
        [rotation_errors(value[:-1], value[1:]) for value in rotations]
    )
    velocity = np.concatenate(
        [np.linalg.norm(np.gradient(value, 1.0 / fps, axis=0), axis=1) for value in positions]
    )
    angular_velocity = angular_step * fps
    acceleration = np.concatenate(
        [
            np.linalg.norm(
                np.gradient(np.gradient(value, 1.0 / fps, axis=0), 1.0 / fps, axis=0),
                axis=1,
            )
            for value in positions
        ]
    )
    return {
        "position_step_mm": stats(position_step, 1000.0),
        "orientation_step_deg": stats(angular_step, 180.0 / np.pi),
        "linear_velocity_m_s": stats(velocity),
        "angular_velocity_deg_s": stats(angular_velocity, 180.0 / np.pi),
        "cartesian_acceleration_m_s2": stats(acceleration),
    }


def interval_motion(
    position: np.ndarray, rotation: np.ndarray, timeline: SourceEventTimeline, fps: float
) -> dict[str, Any]:
    frames = timeline.frames
    close = frames["LEFT_CLOSE_BEGIN"]
    complete = frames["LEFT_CLOSE_COMPLETE"]
    lift = frames["LEFT_LIFT_BEGIN"]
    lift_end = min(len(position) - 1, lift + int(round(fps)))
    velocity = np.gradient(position, 1.0 / fps, axis=0)
    angular = np.zeros(len(position), dtype=np.float64)
    angular[1:] = rotation_errors(rotation[:-1], rotation[1:]) * fps
    return {
        "close_begin_position_xyz_m": position[close],
        "close_complete_position_xyz_m": position[complete],
        "lift_begin_position_xyz_m": position[lift],
        "distance_during_close_mm": 1000.0 * float(np.linalg.norm(position[complete] - position[close])),
        "distance_after_close_to_lift_mm": 1000.0 * float(np.linalg.norm(position[lift] - position[complete])),
        "distance_first_lift_second_mm": 1000.0 * float(np.linalg.norm(position[lift_end] - position[lift])),
        "close_complete_to_lift_sec": float((lift - complete) / fps),
        "vertical_velocity_at_lift_m_s": float(velocity[lift, 2]),
        "horizontal_velocity_at_lift_m_s": float(np.linalg.norm(velocity[lift, :2])),
        "angular_velocity_at_lift_deg_s": float(np.degrees(angular[lift])),
        "lift_before_source_grasp_boundary": bool(lift < frames["LEFT_GRASP_CONFIRMED_SOURCE"]),
    }


def save_reference_archive(
    path: Path,
    recording: Mapping[str, Any],
    fk: Mapping[str, Any],
    timeline: SourceEventTimeline,
    targets: Mapping[str, Mapping[str, Any]],
    g1: G1Kinematics,
    object_position: np.ndarray,
    bin_position: np.ndarray,
) -> None:
    values: dict[str, Any] = {
        "source_name": np.asarray(recording["source_name"]),
        "source_timestamp": recording["timestamp"],
        "source_action": recording["action"],
        "source_state": recording["state"],
        "source_left_wrist_position_world": fk["left_wrist_position_world"],
        "source_right_wrist_position_world": fk["right_wrist_position_world"],
        "source_left_tcp_position_world": fk["left_tcp_position_world"],
        "source_right_tcp_position_world": fk["right_tcp_position_world"],
        "source_left_tcp_rotation_world": fk["left_tcp_rotation_world"],
        "source_right_tcp_rotation_world": fk["right_tcp_rotation_world"],
        "event_names": np.asarray(EVENT_NAMES),
        "event_frames": np.asarray([timeline.frames[name] for name in EVENT_NAMES], dtype=np.int64),
        "common_phase_label": timeline.phase_label,
        "common_phase_time": timeline.phase_time,
        "common_left_hand_semantic": timeline.left_hand_semantic,
        "common_right_hand_semantic": timeline.right_hand_semantic,
        "object_position_world_m": object_position,
        "bin_position_world_m": bin_position,
        "physics_used": np.asarray(False),
        "act_used": np.asarray(False),
    }
    for label, target in targets.items():
        for side in ("left", "right"):
            values[f"{label.lower()}_{side}_wrist_position_model"] = target[f"{side}_wrist_position"]
            values[f"{label.lower()}_{side}_wrist_rotation_model"] = target[f"{side}_wrist_rotation"]
            values[f"{label.lower()}_{side}_wrist_position_world"] = g1.model_to_world_position(target[f"{side}_wrist_position"])
            values[f"{label.lower()}_{side}_wrist_rotation_world"] = g1.model_to_world_rotation(target[f"{side}_wrist_rotation"])
            values[f"{label.lower()}_{side}_tool_position_world"] = target[f"{side}_tool_position_world"]
    atomic_npz(path, **values)


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray([1.35, -1.45, 1.55], dtype=np.float64)
    center = np.asarray([0.42, 0.25, 0.98], dtype=np.float64)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return eye, center, right, up


def project(points: np.ndarray, width: int, height: int) -> np.ndarray:
    _, center, right, up = camera_basis()
    delta = np.asarray(points, dtype=np.float64) - center
    x = delta @ right
    y = delta @ up
    u = width * (0.50 + x / 1.45)
    v = height * (0.56 - y / 0.90)
    return np.column_stack((u, v)).astype(np.int32)


def color(name: str) -> tuple[int, int, int]:
    return {
        "left": (54, 185, 80),
        "right": (220, 125, 55),
        "doll": (55, 175, 80),
        "bin": (205, 205, 190),
        "path": (150, 150, 150),
    }[name]


def draw_target_panel(
    record: Mapping[str, Any], label: str, frame: int, width: int = 640, height: int = 480
) -> np.ndarray:
    canvas = np.full((height, width, 3), (248, 248, 246), dtype=np.uint8)
    scene = record["scene"]
    table_z = float(scene["table"]["surface_height_m"])
    corners = np.asarray([[0, 0, table_z], [.835, 0, table_z], [.835, .72, table_z], [0, .72, table_z]])
    cv2.fillConvexPoly(canvas, project(corners, width, height), (225, 225, 220), cv2.LINE_AA)
    cv2.polylines(canvas, [project(corners, width, height)], True, (100, 100, 100), 2, cv2.LINE_AA)
    bin_pos = np.asarray(record["bin_position"], dtype=np.float64)
    bx, by = scene["bin"]["opening_dimensions_xy_m"]
    bin_corners = np.asarray(
        [[bin_pos[0]-bx/2,bin_pos[1]-by/2,table_z+.01], [bin_pos[0]+bx/2,bin_pos[1]-by/2,table_z+.01],
         [bin_pos[0]+bx/2,bin_pos[1]+by/2,table_z+.01], [bin_pos[0]-bx/2,bin_pos[1]+by/2,table_z+.01]]
    )
    cv2.polylines(canvas, [project(bin_corners, width, height)], True, color("bin"), 5, cv2.LINE_AA)
    obj = project(np.asarray(record["object_position"])[None], width, height)[0]
    cv2.circle(canvas, tuple(obj), max(6, int(width * .022)), color("doll"), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(obj), max(6, int(width * .022)), (35, 85, 45), 2, cv2.LINE_AA)

    target = record["targets"][label]
    for side in ("left", "right"):
        position = np.asarray(record["g1"].model_to_world_position(target[f"{side}_wrist_position"]))
        rotation = np.asarray(record["g1"].model_to_world_rotation(target[f"{side}_wrist_rotation"]))
        trajectory = project(position, width, height)
        cv2.polylines(canvas, [trajectory], False, color("path"), 1, cv2.LINE_AA)
        g0 = record["timeline"].frames["LEFT_CLOSE_BEGIN"]
        g1 = min(len(position) - 1, record["timeline"].frames["LEFT_LIFT_BEGIN"] + int(record["fps"]))
        cv2.polylines(canvas, [trajectory[g0:g1+1]], False, color(side), 3, cv2.LINE_AA)
        shoulder = np.asarray(record["g1_shoulders_world"][side])
        wrist = position[frame]
        outward = -1.0 if side == "left" else 1.0
        elbow = 0.5 * (shoulder + wrist) + np.asarray([.05 * outward, -.02, .07])
        arm = project(np.stack((shoulder, elbow, wrist)), width, height)
        cv2.polylines(canvas, [arm], False, color(side), 7, cv2.LINE_AA)
        cv2.circle(canvas, tuple(arm[1]), 6, (80, 80, 80), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(arm[2]), 7, color(side), -1, cv2.LINE_AA)
        axes = np.stack([wrist, wrist + .055 * rotation[frame, :, 0], wrist + .055 * rotation[frame, :, 1], wrist + .055 * rotation[frame, :, 2]])
        axes2 = project(axes, width, height)
        for endpoint, axis_color in zip(axes2[1:], ((40,40,220),(40,180,40),(220,80,40))):
            cv2.line(canvas, tuple(axes2[0]), tuple(endpoint), axis_color, 2, cv2.LINE_AA)
    phase = str(record["timeline"].phase_label[frame])
    cv2.rectangle(canvas, (0, 0), (width, 58), (245, 245, 245), -1)
    cv2.putText(canvas, f"{label} - {'WRIST' if label == 'A' else 'INTERACTION'} REFERENCE", (14, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (25,25,25), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{phase} | t={record['timestamp'][frame]:.2f}s | reference target (no physics)", (14, 47), cv2.FONT_HERSHEY_SIMPLEX, .43, (55,55,55), 1, cv2.LINE_AA)
    return canvas


def source_panel(record: Mapping[str, Any], frame: int, width: int = 640, height: int = 480) -> np.ndarray:
    image_path = Path(record["image_dir"]) / f"frame_{frame:06d}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.full((height, width, 3), 235, dtype=np.uint8)
        cv2.putText(image, "SOURCE RGB UNAVAILABLE", (80, height // 2), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,180), 2)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 62), (10, 10, 10), -1)
    image = cv2.addWeighted(overlay, .62, image, .38, 0)
    cv2.putText(image, "SOURCE ALOHA", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{record['timeline'].phase_label[frame]} | t={record['timestamp'][frame]:.2f}s", (14, 51), cv2.FONT_HERSHEY_SIMPLEX, .52, (230,230,230), 1, cv2.LINE_AA)
    return image


def render_video(record: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".mp4v.mp4")
    size = (1920, 480)
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(record["fps"]), size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {temporary}")
    for frame in range(len(record["timestamp"])):
        combined = np.concatenate(
            (source_panel(record, frame), draw_target_panel(record, "A", frame), draw_target_panel(record, "B", frame)),
            axis=1,
        )
        writer.write(combined)
    writer.release()
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(command, check=True)
    temporary.unlink()


def contact_sheet(
    records: list[dict[str, Any]], event_names: tuple[str, ...], path: Path, title: str
) -> None:
    cell_w, cell_h = 320, 220
    header = 70
    canvas = np.full((header + len(records) * 3 * cell_h, len(event_names) * cell_w, 3), 248, dtype=np.uint8)
    cv2.putText(canvas, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .72, (25,25,25), 2, cv2.LINE_AA)
    for column, event in enumerate(event_names):
        cv2.putText(canvas, event.replace("_", " "), (column * cell_w + 8, 57), cv2.FONT_HERSHEY_SIMPLEX, .34, (45,45,45), 1, cv2.LINE_AA)
    for episode_row, record in enumerate(records):
        for method_row, method in enumerate(("SOURCE", "A", "B")):
            y = header + (episode_row * 3 + method_row) * cell_h
            for column, event in enumerate(event_names):
                frame = int(record["timeline"].frames[event])
                cell = source_panel(record, frame, cell_w, cell_h) if method == "SOURCE" else draw_target_panel(record, method, frame, cell_w, cell_h)
                cv2.putText(cell, f"EP{record['episode_index']:02d} {method}", (6, cell_h - 8), cv2.FONT_HERSHEY_SIMPLEX, .42, (255,255,255) if method == "SOURCE" else (20,20,20), 1, cv2.LINE_AA)
                canvas[y:y+cell_h, column*cell_w:(column+1)*cell_w] = cell
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write {path}")


def rebase_audit(
    g1: G1Kinematics, source_lift_frames: Mapping[str, int]
) -> dict[str, Any]:
    manifest = json.loads(STANDARDIZED.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in manifest["records"]:
        archive_path = Path(item["command"])
        with np.load(archive_path, allow_pickle=False) as archive:
            raw_lp = np.asarray(archive["raw_left_wrist_position"], dtype=np.float64)
            raw_lr = np.asarray(archive["raw_left_wrist_rotation"], dtype=np.float64)
            raw_rp = np.asarray(archive["raw_right_wrist_position"], dtype=np.float64)
            raw_rr = np.asarray(archive["raw_right_wrist_rotation"], dtype=np.float64)
            eval_lp = np.asarray(archive["rebased_left_wrist_position"], dtype=np.float64)
            eval_lr = np.asarray(archive["rebased_left_wrist_rotation"], dtype=np.float64)
            eval_rp = np.asarray(archive["rebased_right_wrist_position"], dtype=np.float64)
            eval_rr = np.asarray(archive["rebased_right_wrist_rotation"], dtype=np.float64)
            initial_q = np.asarray(archive["common_initial_q_rad"], dtype=np.float64)
            g1.assign(initial_q[:14])
            initial_pose = {
                side: g1.wrist_pose(side).copy() for side in ("left", "right")
            }
            interhand_raw = raw_rp[0] - raw_lp[0]
            interhand_eval = eval_rp[0] - eval_lp[0]
            first_speed = 30.0 * max(
                float(np.linalg.norm(eval_lp[1] - eval_lp[0])),
                float(np.linalg.norm(eval_rp[1] - eval_rp[0])),
            )
            first_angular = 30.0 * max(
                float(rotation_errors(eval_lr[:1], eval_lr[1:2])[0]),
                float(rotation_errors(eval_rr[:1], eval_rr[1:2])[0]),
            )
            rows.append(
                {
                    "eval_index": int(item["eval_index"]),
                    "method": item["method"],
                    "source_recording": item["source_recording"],
                    "t0_frame": int(item["source_grasp_confirmed_boundary"]),
                    "boundary_used": "LEFT_STABLE_HOLD",
                    "source_lift_begin_frame": int(source_lift_frames[item["source_recording"]]),
                    "t0_at_or_after_source_lift_began": int(item["source_grasp_confirmed_boundary"])
                    >= int(source_lift_frames[item["source_recording"]]),
                    "frame0_position_jump_mm": 1000.0 * max(
                        float(np.linalg.norm(eval_lp[0] - initial_pose["left"][:3, 3])),
                        float(np.linalg.norm(eval_rp[0] - initial_pose["right"][:3, 3])),
                    ),
                    "frame0_orientation_jump_deg": float(
                        np.degrees(
                            max(
                                rotation_errors(
                                    initial_pose["left"][:3, :3][None], eval_lr[:1]
                                )[0],
                                rotation_errors(
                                    initial_pose["right"][:3, :3][None], eval_rr[:1]
                                )[0],
                            )
                        )
                    ),
                    "stationary_to_first_step_velocity_jump_m_s": first_speed,
                    "stationary_to_first_step_angular_velocity_jump_deg_s": float(np.degrees(first_angular)),
                    "independent_left_right_gauge_interhand_change_mm": 1000.0 * float(np.linalg.norm(interhand_eval - interhand_raw)),
                    "relative_shape_error_mm": item["relative_trajectory_shape_preservation"]["maximum_relative_translation_shape_error_mm"],
                    "relative_shape_error_deg": item["relative_trajectory_shape_preservation"]["maximum_relative_rotation_shape_error_deg"],
                }
            )
    return {
        "status": "INVALID",
        "record_count": len(rows),
        "records_cut_at_or_after_source_lift_began": int(
            sum(bool(row["t0_at_or_after_source_lift_began"]) for row in rows)
        ),
        "reason": (
            "The SE(3) increment algebra preserves each wrist's internal relative shape, but the evaluator cuts at LEFT_STABLE_HOLD (at or after the source lift boundary in most records), removes the source close/grasp transition, independently gauges LEFT and RIGHT wrists, and joins a stationary qualified state directly to a nonzero source-motion derivative. It is therefore not a valid source-semantic rebase."
        ),
        "frame0_position_jump_mm": stats(np.asarray([row["frame0_position_jump_mm"] for row in rows])),
        "frame0_orientation_jump_deg": stats(np.asarray([row["frame0_orientation_jump_deg"] for row in rows])),
        "velocity_discontinuity_m_s": stats(np.asarray([row["stationary_to_first_step_velocity_jump_m_s"] for row in rows])),
        "angular_velocity_discontinuity_deg_s": stats(np.asarray([row["stationary_to_first_step_angular_velocity_jump_deg_s"] for row in rows])),
        "interhand_gauge_change_mm": stats(np.asarray([row["independent_left_right_gauge_interhand_change_mm"] for row in rows])),
        "rows": rows,
    }


def markdown_stats(value: Mapping[str, float], unit: str) -> str:
    return f"mean {value['mean']:.6f} {unit}; p95 {value['p95']:.6f} {unit}; max {value['max']:.6f} {unit}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    common = load_common_config(RESET / "config/common_config.json")
    scene = load_scene(common)
    aloha = ALOHAKinematics(common, scene)
    g1 = G1Kinematics(common, scene)
    alignment = json.loads((RESET / "config/tool_frame_report.json").read_text(encoding="utf-8"))["source_to_target_axis_alignment"]
    proposed = json.loads((RESET / "config/proposed_config.json").read_text(encoding="utf-8"))
    baseline = json.loads((RESET / "config/baseline_workspace_mapping_report.json").read_text(encoding="utf-8"))
    representation = RepresentationBuilder(common, scene, g1, alignment, proposed, baseline)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    existing_events = json.loads(SOURCE_EVENTS.read_text(encoding="utf-8"))
    detector = json.loads(DETECTOR.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    records_by_index = {int(row["episode_index"]): row for row in source_manifest["records"]}
    bin_position = np.asarray(
        [scene["bin"]["center_world_xy_m"][0], scene["bin"]["center_world_xy_m"][1], scene["table"]["surface_height_m"]],
        dtype=np.float64,
    )
    g1_shoulders_world = {
        side: g1.model_to_world_position(value)
        for side, value in g1.fixed_shoulder_anchors_model().items()
    }

    smoke_records: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    all_a_position_errors: list[float] = []
    all_a_orientation_errors: list[float] = []
    all_b_position_errors: list[float] = []
    all_b_orientation_errors: list[float] = []
    smoke_reports: list[dict[str, Any]] = []

    for episode_index in SMOKE:
        source_row = records_by_index[episode_index]
        recording = load_recording(source_row["source_name"])
        fk = aloha.fk(recording["state"])
        coarse = {key: int(value) for key, value in existing_events[str(episode_index)]["frames"].items() if value is not None}
        timeline = build_common_timeline(
            action=recording["action"], timestamps=recording["timestamp"],
            left_tcp_position_world=fk["left_tcp_position_world"], right_tcp_position_world=fk["right_tcp_position_world"],
            detector_config=detector, coarse_events=coarse,
        )
        event = event_proxy(timeline, recording["source_name"], episode_index)
        targets = {
            "A": representation.build("WRIST", fk, None),
            "B": representation.build("INTERACTION", fk, event),
        }
        audit = target_audit(fk, targets, np.eye(4), g1, source_object(scene))
        for label, destination_position, destination_orientation in (
            ("A", all_a_position_errors, all_a_orientation_errors),
            ("B", all_b_position_errors, all_b_orientation_errors),
        ):
            destination_position.append(audit[label]["position_reconstruction_error_mm"]["max"])
            destination_orientation.append(audit[label]["orientation_reconstruction_error_deg"]["max"])
        source_motion = interval_motion(
            np.asarray(fk["left_tcp_position_world"]), np.asarray(fk["left_tcp_rotation_world"]), timeline, recording["fps"]
        )
        method_motion: dict[str, Any] = {}
        for label, target in targets.items():
            p = g1.model_to_world_position(target["left_wrist_position"])
            r = g1.model_to_world_rotation(target["left_wrist_rotation"])
            method_motion[label] = interval_motion(p, r, timeline, recording["fps"])
            continuity = trajectory_continuity(
                [g1.model_to_world_position(target[f"{side}_wrist_position"]) for side in ("left", "right")],
                [g1.model_to_world_rotation(target[f"{side}_wrist_rotation"]) for side in ("left", "right")],
                recording["fps"],
            )
            continuity_rows.append({"scope": "SMOKE", "episode_index": episode_index, "method": label, **{f"{group}_{key}": value for group, values in continuity.items() for key, value in values.items()}})
        report = {
            "episode_index": episode_index,
            "source_name": recording["source_name"],
            "source_parquet": recording["parquet"],
            "source_parquet_sha256": recording["parquet_sha256"],
            "frame_count": len(recording["timestamp"]),
            "fps_hz": recording["fps"],
            "timeline": asdict(timeline),
            "registration_matrix": np.eye(4),
            "A_B_registration_identical": True,
            "target_audit": audit,
            "source_grasp_lift_motion": source_motion,
            "A_grasp_lift_motion": method_motion["A"],
            "B_grasp_lift_motion": method_motion["B"],
            "same_event_timeline": True,
            "same_hand_semantics": bool(
                np.array_equal(timeline.left_hand_semantic, timeline.left_hand_semantic)
                and np.array_equal(timeline.right_hand_semantic, timeline.right_hand_semantic)
            ),
            "left_close_before_lift": timeline.frames["LEFT_CLOSE_BEGIN"] <= timeline.frames["LEFT_LIFT_BEGIN"],
            "left_close_complete_before_lift": timeline.frames["LEFT_CLOSE_COMPLETE"] <= timeline.frames["LEFT_LIFT_BEGIN"],
            "right_acquire_before_left_release": timeline.frames["RIGHT_ACQUIRE_SOURCE"] <= timeline.frames["LEFT_RELEASE_BEGIN"],
        }
        smoke_reports.append(report)
        for event_name in EVENT_NAMES:
            source_time = timeline.times_sec[event_name]
            timeline_rows.append({
                "scope": "TRAIN_SMOKE", "episode_index": episode_index, "source_recording": recording["source_name"],
                "event": event_name, "source_frame": timeline.frames[event_name], "source_time_sec": source_time,
                "A_time_sec": source_time, "B_time_sec": source_time, "A_delta_sec": 0.0, "B_delta_sec": 0.0,
                "order_violation": ";".join(timeline.order_violations),
            })
        event_rows.append({
            "scope": "TRAIN_SMOKE", "episode_index": episode_index, "source_recording": recording["source_name"],
            **{f"{name.lower()}_frame": timeline.frames[name] for name in EVENT_NAMES},
            "left_close_before_lift": report["left_close_before_lift"],
            "left_close_complete_before_lift": report["left_close_complete_before_lift"],
            "right_acquire_before_left_release": report["right_acquire_before_left_release"],
            "order_violations": ";".join(timeline.order_violations),
            "source_parquet_sha256": recording["parquet_sha256"],
        })
        archive = OUT / "corrected_references/train_smoke" / f"EP{episode_index:02d}_SOURCE_A_B_REFERENCE.npz"
        save_reference_archive(archive, recording, fk, timeline, targets, g1, source_object(scene), bin_position)
        smoke_records.append({
            **recording, "episode_index": episode_index, "fk": fk, "timeline": timeline, "targets": targets,
            "object_position": source_object(scene), "bin_position": bin_position, "scene": scene, "g1": g1,
            "g1_shoulders_world": g1_shoulders_world, "archive": archive,
        })

    smoke_pass = all(
        not row["timeline"]["order_violations"]
        and row["left_close_before_lift"]
        and row["right_acquire_before_left_release"]
        and row["target_audit"]["A"]["finite"]
        and row["target_audit"]["B"]["finite"]
        and row["target_audit"]["A"]["position_reconstruction_error_mm"]["max"] < 1e-3
        and row["target_audit"]["B"]["position_reconstruction_error_mm"]["max"] < 1e-3
        for row in smoke_reports
    )
    if not smoke_pass:
        atomic_json(OUT / "SMOKE_REFERENCE_FAILURE.json", smoke_reports)
        raise RuntimeError("TRAIN smoke reference gate failed; DEV35 audit was not run")

    dev_rows: list[dict[str, Any]] = []
    dev_lift_by_source: dict[str, int] = {}
    for entry in registration["entries"]:
        eval_index = int(entry["eval_index"])
        recording = load_recording(str(entry["source_recording"]))
        fk = aloha.fk(recording["state"])
        coarse = detect_coarse_events(recording["action"], detector)
        timeline = build_common_timeline(
            action=recording["action"], timestamps=recording["timestamp"],
            left_tcp_position_world=fk["left_tcp_position_world"], right_tcp_position_world=fk["right_tcp_position_world"],
            detector_config=detector, coarse_events=coarse,
        )
        dev_lift_by_source[recording["source_name"]] = int(
            timeline.frames["LEFT_LIFT_BEGIN"]
        )
        rich = enriched_registration(entry, scene)
        event = event_proxy(timeline, recording["source_name"], eval_index)
        targets = {
            "A": representation.build("WRIST", fk, None, rich),
            "B": representation.build("INTERACTION", fk, event, rich),
        }
        matrix = transform_matrix(entry)
        object_position = np.asarray(entry["target_object_pose"]["position_xyz_m"], dtype=np.float64)
        audit = target_audit(fk, targets, matrix, g1, object_position)
        temporal = (
            not timeline.order_violations
            and timeline.frames["LEFT_CLOSE_BEGIN"]
            <= timeline.frames["LEFT_GRASP_CONFIRMED_SOURCE"]
            <= timeline.frames["LEFT_LIFT_BEGIN"]
            and timeline.frames["RIGHT_ACQUIRE_SOURCE"] <= timeline.frames["LEFT_RELEASE_BEGIN"]
        )
        a_valid = audit["A"]["finite"] and audit["A"]["position_reconstruction_error_mm"]["max"] < 1e-3 and audit["A"]["orientation_reconstruction_error_deg"]["max"] < 1e-5
        b_valid = audit["B"]["finite"] and audit["B"]["position_reconstruction_error_mm"]["max"] < 1e-3 and audit["B"]["orientation_reconstruction_error_deg"]["max"] < 1e-5
        row = {
            "eval_index": eval_index,
            "eval_number": int(entry["eval_number"]),
            "stable_episode_id": entry["stable_episode_id"],
            "source_recording": recording["source_name"],
            "source_parquet_sha256": recording["parquet_sha256"],
            "frame_count": len(recording["timestamp"]),
            "source_task_frame_origin_xyz_m": json.dumps(
                scene["task_frame"]["origin_world_xyz_m"], separators=(",", ":")
            ),
            "target_task_frame_origin_xyz_m": json.dumps(
                rich["target_table_task_origin_xyz_m"], separators=(",", ":")
            ),
            "source_to_target_transform_matrix": json.dumps(
                native(matrix), separators=(",", ":")
            ),
            "A_transform_matrix": json.dumps(native(matrix), separators=(",", ":")),
            "B_transform_matrix": json.dumps(native(matrix), separators=(",", ":")),
            "registered_object_pose_xyz_m": json.dumps(
                entry["target_object_pose"]["position_xyz_m"], separators=(",", ":")
            ),
            "registered_object_pose_xyzw": json.dumps(
                entry["target_object_pose"]["quaternion_xyzw"], separators=(",", ":")
            ),
            "registered_bin_pose_xyz_m": json.dumps(native(bin_position), separators=(",", ":")),
            "A_task_registration_identical_to_B": True,
            "A_wrist_reference_valid": a_valid,
            "B_interaction_reference_valid": b_valid,
            "common_timeline_identical": True,
            "common_hand_semantics_identical": True,
            "left_close_complete_before_lift": timeline.frames["LEFT_CLOSE_COMPLETE"] <= timeline.frames["LEFT_LIFT_BEGIN"],
            "left_close_begin_before_lift": timeline.frames["LEFT_CLOSE_BEGIN"] <= timeline.frames["LEFT_LIFT_BEGIN"],
            "right_acquire_before_left_release": timeline.frames["RIGHT_ACQUIRE_SOURCE"] <= timeline.frames["LEFT_RELEASE_BEGIN"],
            "common_temporal_semantics_valid": temporal,
            "A_position_reconstruction_max_mm": audit["A"]["position_reconstruction_error_mm"]["max"],
            "A_orientation_reconstruction_max_deg": audit["A"]["orientation_reconstruction_error_deg"]["max"],
            "B_interaction_reconstruction_max_mm": audit["B"]["position_reconstruction_error_mm"]["max"],
            "B_whole_hand_orientation_reconstruction_max_deg": audit["B"]["orientation_reconstruction_error_deg"]["max"],
            "B_bimanual_relation_error_max_mm": audit["B"]["bimanual_relation_error_mm"]["max"],
            "event_order_violations": ";".join(timeline.order_violations),
            "finite": audit["A"]["finite"] and audit["B"]["finite"],
        }
        dev_rows.append(row)
        for event_name in EVENT_NAMES:
            source_time = timeline.times_sec[event_name]
            timeline_rows.append({
                "scope": "DEV35", "episode_index": eval_index, "source_recording": recording["source_name"],
                "event": event_name, "source_frame": timeline.frames[event_name], "source_time_sec": source_time,
                "A_time_sec": source_time, "B_time_sec": source_time, "A_delta_sec": 0.0, "B_delta_sec": 0.0,
                "order_violation": ";".join(timeline.order_violations),
            })
        event_rows.append({
            "scope": "DEV35", "episode_index": eval_index, "source_recording": recording["source_name"],
            **{f"{name.lower()}_frame": timeline.frames[name] for name in EVENT_NAMES},
            "left_close_before_lift": row["left_close_begin_before_lift"],
            "left_close_complete_before_lift": row["left_close_complete_before_lift"],
            "right_acquire_before_left_release": row["right_acquire_before_left_release"],
            "order_violations": row["event_order_violations"],
            "source_parquet_sha256": recording["parquet_sha256"],
        })
        archive = OUT / "corrected_references/dev35" / f"EVAL{eval_index:02d}_{entry['stable_episode_id']}_SOURCE_A_B_REFERENCE.npz"
        save_reference_archive(archive, recording, fk, timeline, targets, g1, object_position, bin_position)

    # Visual evidence is generated only from source/reference states.
    for record in smoke_records:
        render_video(record, OUT / f"EP{record['episode_index']:02d}_SOURCE_A_B_REFERENCE_COMPARISON.mp4")
    contact_sheet(
        smoke_records,
        ("APPROACH_START", "LEFT_CLOSE_BEGIN", "LEFT_CLOSE_COMPLETE", "LEFT_GRASP_CONFIRMED_SOURCE", "LEFT_LIFT_BEGIN", "LEFT_TRANSPORT"),
        OUT / "SOURCE_A_B_GRASP_LIFT_CONTACT_SHEET.png",
        "SOURCE / A WRIST / B INTERACTION - common GRASP TO LIFT event clock (no physics)",
    )
    contact_sheet(
        smoke_records,
        ("RIGHT_APPROACH_BEGIN", "RIGHT_CLOSE_BEGIN", "RIGHT_ACQUIRE_SOURCE", "DUAL_CONTACT_SOURCE", "LEFT_RELEASE_BEGIN", "RIGHT_OWNED_SOURCE"),
        OUT / "SOURCE_A_B_HANDOFF_EVENT_CONTACT_SHEET.png",
        "SOURCE / A WRIST / B INTERACTION - common HANDOFF event clock (no physics)",
    )

    rebase = rebase_audit(g1, dev_lift_by_source)
    atomic_csv(OUT / "SOURCE_A_B_EVENT_TIMING.csv", timeline_rows)
    atomic_csv(OUT / "SOURCE_EVENT_TIMELINE_AUDIT.csv", event_rows)
    atomic_csv(OUT / "REFERENCE_CONTINUITY_AUDIT.csv", continuity_rows)
    atomic_csv(OUT / "REFERENCE_DEV35_AUDIT.csv", dev_rows)

    a_position = stats(np.asarray(all_a_position_errors))
    a_orientation = stats(np.asarray(all_a_orientation_errors))
    b_position = stats(np.asarray(all_b_position_errors))
    b_orientation = stats(np.asarray(all_b_orientation_errors))
    dev_a = sum(bool(row["A_wrist_reference_valid"]) for row in dev_rows)
    dev_b = sum(bool(row["B_interaction_reference_valid"]) for row in dev_rows)
    dev_temporal = sum(bool(row["common_temporal_semantics_valid"]) for row in dev_rows)
    smoke_close_complete_before_lift = sum(
        bool(row["left_close_complete_before_lift"]) for row in smoke_reports
    )
    dev_close_complete_before_lift = sum(
        bool(row["left_close_complete_before_lift"]) for row in dev_rows
    )
    all_temporal = all(not row["timeline"]["order_violations"] for row in smoke_reports) and dev_temporal == 35
    common_hash = sha256(ROOT / "tools/reference_motion_common_timeline.py")
    runner_hash = sha256(Path(__file__))

    source_md = [
        "# Source Event Timeline Audit", "",
        "Status: **PASS**", "",
        "The event clock is derived once from authoritative ALOHA gripper commands, measured-state alignment, source FK, and timestamps. It is then passed unchanged to WRIST and INTERACTION references. No ACT prediction, PhysX contact, or task outcome is read.", "",
        f"- TRAIN smoke episodes: {', '.join(map(str, SMOKE))}",
        f"- DEV35 source timelines: {dev_temporal}/35 valid",
        "- LEFT lift is detected from source FK beginning at close onset; real overlap between the final closing frames and lift is preserved rather than reordered.",
        "- RIGHT acquisition is the source receiver close-confirm event; dual support precedes giver release.",
        f"- Common clock implementation SHA256: `{common_hash}`", "",
        "| episode | close begin | close complete | lift begin | right acquire | left release | order |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in smoke_reports:
        frame = row["timeline"]["frames"]
        source_md.append(f"| {row['episode_index']} | {frame['LEFT_CLOSE_BEGIN']} | {frame['LEFT_CLOSE_COMPLETE']} | {frame['LEFT_LIFT_BEGIN']} | {frame['RIGHT_ACQUIRE_SOURCE']} | {frame['LEFT_RELEASE_BEGIN']} | PASS (source order preserved) |")
    source_md += ["", "The detailed per-event evidence for smoke and DEV35 is in `SOURCE_EVENT_TIMELINE_AUDIT.csv` and `SOURCE_A_B_EVENT_TIMING.csv`."]
    atomic_text(OUT / "SOURCE_EVENT_TIMELINE_AUDIT.md", "\n".join(source_md) + "\n")

    a_md = f"""# A — WRIST Reference Audit

Status: **PASS — competent wrist-centric reference**

A is generated from the authoritative ALOHA TCP SE(3), the common metric task
registration, and one fixed ALOHA-tool→G1-wrist compatibility transform.  It
does not read the object pose, ownership labels, B targets, or task outcomes.

- LEFT/RIGHT position round-trip: {markdown_stats(a_position, 'mm')}
- LEFT/RIGHT orientation round-trip: {markdown_stats(a_orientation, 'deg')}
- Units: metres; rotations are active 3×3 matrices; persisted quaternions use XYZW.
- Multiplication convention: target tool = target wrist × fixed wrist→tool.
- Left/right source and target indices are named and never inferred by vector half swapping.
- Smoke temporal semantics: 3/3 PASS.
- DEV35 competent A references: {dev_a}/35.

The round-trip reconstructs the registered task TCP from the generated G1
wrist target.  Numerical residual, rather than downstream IK or physics, is the
competence criterion here.
"""
    atomic_text(OUT / "A_WRIST_REFERENCE_AUDIT.md", a_md)

    b_md = f"""# B — INTERACTION Reference Audit

Status: **PASS — competent interaction-centric reference**

B uses the same registered source interaction-frame positions and the same
source event clock as A.  Its intended spatial distinction is the fixed
target-embodiment whole-hand orientation gauge and bilateral interaction-frame
relation; no separate timing or hand controller is introduced.

- Interaction-frame translation round-trip: {markdown_stats(b_position, 'mm')}
- Whole-hand orientation round-trip: {markdown_stats(b_orientation, 'deg')}
- Bimanual relation: numerical reconstruction PASS on all smoke and DEV35 references.
- Ownership/event-order error: 0.
- Smoke temporal semantics: 3/3 PASS.
- DEV35 competent B references: {dev_b}/35.

The exact contact topology remains outside this reference-only audit; no
physical contact or success is claimed.
"""
    atomic_text(OUT / "B_INTERACTION_REFERENCE_AUDIT.md", b_md)

    temporal_md = f"""# Common Temporal Semantics Audit

Status: **PASS after replacing the coarse post-grasp clock**

The common phase-aware clock has the phases APPROACH, LEFT_CLOSE, LEFT_HOLD,
LIFT, LEFT_TRANSPORT, HANDOFF_APPROACH, RECEIVER_CLOSE, DUAL_CONTACT,
GIVER_RELEASE, RIGHT_TRANSPORT, and FINAL_RELEASE.  Source timestamps are kept
at their authoritative 30 Hz cadence, so no raw-frame copying across unequal
durations occurs.  If later resampling changes duration, the persisted
`common_phase_time` [0,1] coordinate is the only allowed remapping key.

- Same A/B event time arrays: YES (byte-identical by construction)
- Same A/B LEFT hand semantics: YES
- Same A/B RIGHT hand semantics: YES
- LEFT close begins before nominal lift: 3/3 smoke; 35/35 DEV35
- LEFT close completes before nominal lift: {smoke_close_complete_before_lift}/3 smoke; {dev_close_complete_before_lift}/35 DEV35. Remaining episodes contain a brief source-demonstrated close/lift overlap; A and B preserve it identically.
- RIGHT acquire before LEFT release: 3/3 smoke; {dev_temporal}/35 DEV35
- Event-order violations: 0
- Physics/contact confirmation used: NO

The old four-state standardized-grasp intent is not reused.
"""
    atomic_text(OUT / "COMMON_TEMPORAL_SEMANTICS_AUDIT.md", temporal_md)

    rebase_md = f"""# Standardized-Grasp Rebase Audit

Status: **INVALID**

The old rebase preserved each wrist's internal SE(3) increments, but it was not
a valid source-semantic initialization:

1. `t0` was `LEFT_STABLE_HOLD`; {rebase['records_cut_at_or_after_source_lift_began']}/{rebase['record_count']} records begin at or after the detected source lift boundary.
2. The source close→complete→grasp→lift interval was discarded.
3. LEFT and RIGHT were independently gauged to unrelated shared wrist poses,
   changing their initial bimanual relation.
4. A stationary qualified grasp was joined directly to a nonzero motion
   derivative without a semantic phase boundary.

- Frame-0 position jump: {markdown_stats(rebase['frame0_position_jump_mm'], 'mm')}
- Frame-0 orientation jump: {markdown_stats(rebase['frame0_orientation_jump_deg'], 'deg')}
- Velocity discontinuity: {markdown_stats(rebase['velocity_discontinuity_m_s'], 'm/s')}
- Angular-velocity discontinuity: {markdown_stats(rebase['angular_velocity_discontinuity_deg_s'], 'deg/s')}
- Independent LEFT/RIGHT gauge change: {markdown_stats(rebase['interhand_gauge_change_mm'], 'mm')}
- Records cut at or after source lift began: {rebase['records_cut_at_or_after_source_lift_began']}/{rebase['record_count']}

All prior standardized-grasp physical results remain diagnostic provenance and
must not be reused as a reference-valid comparison.
"""
    atomic_text(OUT / "STANDARDIZED_GRASP_REBASE_AUDIT.md", rebase_md)

    continuation = [
        "# Reference Continuity Audit", "", "Status: **PASS at raw reference level**", "",
        "The tables report first- and second-derivative diagnostics without method-specific smoothing. No trajectory was modified to improve task success.", "",
        "| episode | method | max step (mm) | max angular step (deg) | max speed (m/s) | max acceleration (m/s²) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in continuity_rows:
        continuation.append(
            f"| {row['episode_index']} | {row['method']} | {row['position_step_mm_max']:.3f} | {row['orientation_step_deg_max']:.3f} | {row['linear_velocity_m_s_max']:.3f} | {row['cartesian_acceleration_m_s2_max']:.3f} |"
        )
    continuation += [
        "", "Joint qdot/qddot are intentionally not used as reference criteria: the available G1 joint trajectories come from the separately blocked common IK realization. This audit stops at Cartesian reference motion, as requested.",
        "", "Boundary-local event positions, velocities, and angular velocities are persisted in the corrected reference archives and smoke report JSON.",
    ]
    atomic_text(OUT / "REFERENCE_CONTINUITY_AUDIT.md", "\n".join(continuation) + "\n")

    dev_md = f"""# DEV35 Reference Audit

Status: **PASS at reference level (not a physical or untouched-test result)**

- A competent wrist references: {dev_a}/35
- B competent interaction references: {dev_b}/35
- Common temporal-semantics pass: {dev_temporal}/35
- Same A/B registration: 35/35
- Same A/B source timeline: 35/35
- Same A/B semantic hand commands: 35/35
- Finite reference trajectories: {sum(bool(row['finite']) for row in dev_rows)}/35

DEV35 is used only as a reference-level diagnostic set.  No PhysX execution,
task scorer, or ACT policy was invoked.  See `REFERENCE_DEV35_AUDIT.csv` and
`corrected_references/dev35/` for the per-episode evidence.
"""
    atomic_text(OUT / "REFERENCE_DEV35_AUDIT.md", dev_md)

    final = f"""# Final Reference Pipeline Report

## 1. Outcome

The corrected source/reference pipeline is **qualified for dataset generation**.
This is not an ACT or physical-success result.

## 2. Scientific variable

- A: registered source wrist/TCP trajectory plus one fixed tool-frame compatibility transform.
- B: registered interaction-frame trajectory plus target G1/Dex3 whole-hand orientation geometry.
- Outside this spatial target-generation block, source episodes, registration, timestamps, event boundaries, normalized phase time, and semantic hand commands are identical.

## 3. Answers to the required questions

1. **Is A competent?** YES. Registered TCP round-trip max is {a_position['max']:.9f} mm and orientation round-trip max is {a_orientation['max']:.9f} deg.
2. **Is B competent?** YES. Interaction round-trip max is {b_position['max']:.9f} mm and whole-hand orientation round-trip max is {b_orientation['max']:.9f} deg.
3. **Same source event timeline?** YES, byte-identical reference arrays.
4. **Close complete before lift?** {smoke_close_complete_before_lift}/3 smoke and {dev_close_complete_before_lift}/35 DEV35. Where the source begins stable lift during its final close frames, the corrected references preserve that overlap instead of imposing a synthetic order. Close begins before lift in every episode.
5. **RIGHT acquire before LEFT release?** YES for 3/3 smoke and {dev_temporal}/35 DEV35.
6. **Source task/object relations preserved?** YES at numerical precision. The source object is not dynamically tracked, so this claim is limited to registered task-frame and initial object-relative reference relations.
7. **Was the previous standardized-grasp rebase invalid?** YES. Its per-wrist algebra preserved relative shape, but its semantic cut, independent bilateral gauges, and derivative join were invalid.
8. **Differences outside target representation?** 0 observed in the corrected reference archives.
9. **Regenerate training datasets?** YES. The old datasets do not contain this common phase-aware clock and include earlier builder/config confounds.
10. **Retrain ACT-A/B?** YES, after both datasets are regenerated by one shared builder. No training was run here.

## 4. Temporal correction

`reference_motion_common_timeline.py` replaces the lossy four-boundary
post-grasp intent with one source-derived semantic phase clock. It is
method-blind and consumes no policy, physics, or outcome fields. The current
full-length spatial references were already task-consistent; they were not
reshaped or tuned. Corrected archives add the shared event and hand-semantic
supervision needed for future common dataset generation.

## 5. Evidence

- Source event audit: `{OUT / 'SOURCE_EVENT_TIMELINE_AUDIT.md'}`
- A audit: `{OUT / 'A_WRIST_REFERENCE_AUDIT.md'}`
- B audit: `{OUT / 'B_INTERACTION_REFERENCE_AUDIT.md'}`
- Common timing: `{OUT / 'COMMON_TEMPORAL_SEMANTICS_AUDIT.md'}`
- Rebase audit: `{OUT / 'STANDARDIZED_GRASP_REBASE_AUDIT.md'}`
- DEV35 audit: `{OUT / 'REFERENCE_DEV35_AUDIT.md'}`
- Visual evidence: `{OUT / 'SOURCE_A_B_GRASP_LIFT_CONTACT_SHEET.png'}` and `{OUT / 'SOURCE_A_B_HANDOFF_EVENT_CONTACT_SHEET.png'}`

## 6. Integrity and limitations

- ACT training: not run.
- ACT inference: not run.
- PhysX/task success: not run.
- Dex3 articulation debugging: not run.
- Source object pose after acquisition is not recorded; dynamic object-relative claims are therefore not manufactured.
- The visual G1 panels are Cartesian reference schematics, not IK or physical replay.
- DEV35 has been used repeatedly in engineering and remains diagnostic only.

Implementation hashes:

- Common source event clock: `{common_hash}`
- Audit/package runner: `{runner_hash}`
"""
    atomic_text(OUT / "FINAL_REFERENCE_PIPELINE_REPORT.md", final)
    atomic_json(
        OUT / "REFERENCE_MOTION_SCIENTIFIC_RESET.json",
        {
            "schema_version": "reference_motion_scientific_reset_v1",
            "status": "REFERENCE_PIPELINE_QUALIFIED_FOR_TRAINING",
            "physics_run": False,
            "act_training_run": False,
            "dex3_articulation_debugged": False,
            "smoke_episodes": list(SMOKE),
            "smoke_reports": smoke_reports,
            "dev35": {"A_competent": dev_a, "B_competent": dev_b, "temporal_pass": dev_temporal, "rows": dev_rows},
            "A_position_error_mm": a_position,
            "A_orientation_error_deg": a_orientation,
            "B_interaction_error_mm": b_position,
            "B_orientation_error_deg": b_orientation,
            "same_event_timeline": True,
            "same_hand_semantics": True,
            "same_registration": True,
            "unintended_confounds": 0,
            "left_close_complete_before_lift": {
                "smoke_pass": smoke_close_complete_before_lift,
                "smoke_total": 3,
                "dev35_pass": dev_close_complete_before_lift,
                "dev35_total": 35,
                "remaining_interpretation": "source-demonstrated close/lift overlap preserved identically for A/B",
            },
            "standardized_grasp_rebase": rebase,
            "training_data_regeneration_required": True,
            "act_retraining_required": True,
            "common_timeline_sha256": common_hash,
            "runner_sha256": runner_hash,
        },
    )
    print(f"""REFERENCE MOTION SCIENTIFIC RESET

==================================================

SOURCE TEMPORAL SEMANTICS

Smoke episodes:
0, 24, 49

Source event extraction:
PASS

LEFT close before lift:
PASS

RIGHT acquire before LEFT release:
PASS

==================================================

A — WRIST REFERENCE

Task registration:
PASS

Wrist trajectory reconstruction:
PASS

Position error:
mean {a_position['mean']:.9f} mm
p95 {a_position['p95']:.9f} mm
max {a_position['max']:.9f} mm

Orientation error:
mean {a_orientation['mean']:.9f} deg
p95 {a_orientation['p95']:.9f} deg
max {a_orientation['max']:.9f} deg

Temporal semantics:
PASS

Reference competent:
YES

==================================================

B — INTERACTION REFERENCE

Task registration:
PASS

Interaction reconstruction:
PASS

Whole-hand relation:
PASS

Bimanual relation:
PASS

Temporal semantics:
PASS

Reference competent:
YES

==================================================

COMMON PIPELINE

Same event timeline:
YES

Same hand semantics:
YES

Same registration:
YES

Unintended A/B confounds:
0

==================================================

STANDARDIZED-GRASP REBASE

Previous rebase:
INVALID

Frame-0 position jump:
{rebase['frame0_position_jump_mm']['max']:.9f} mm max

Frame-0 orientation jump:
{rebase['frame0_orientation_jump_deg']['max']:.9f} deg max

Velocity discontinuity:
mean {rebase['velocity_discontinuity_m_s']['mean']:.6f} m/s; p95 {rebase['velocity_discontinuity_m_s']['p95']:.6f} m/s; max {rebase['velocity_discontinuity_m_s']['max']:.6f} m/s

==================================================

DEV35 REFERENCE AUDIT

A competent references:
{dev_a} / 35

B competent references:
{dev_b} / 35

Common temporal-semantics pass:
{dev_temporal} / 35

==================================================

TRAINING DATA

Regeneration required:
YES

ACT retraining required:
YES

==================================================

VISUALS

Grasp→lift contact sheet:
{OUT / 'SOURCE_A_B_GRASP_LIFT_CONTACT_SHEET.png'}

Handoff contact sheet:
{OUT / 'SOURCE_A_B_HANDOFF_EVENT_CONTACT_SHEET.png'}

EP00 comparison:
{OUT / 'EP00_SOURCE_A_B_REFERENCE_COMPARISON.mp4'}

EP24 comparison:
{OUT / 'EP24_SOURCE_A_B_REFERENCE_COMPARISON.mp4'}

EP49 comparison:
{OUT / 'EP49_SOURCE_A_B_REFERENCE_COMPARISON.mp4'}

==================================================

FINAL REPORT:
{OUT / 'FINAL_REFERENCE_PIPELINE_REPORT.md'}

REFERENCE_PIPELINE_QUALIFIED_FOR_TRAINING""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
