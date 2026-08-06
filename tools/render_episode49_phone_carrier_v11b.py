#!/usr/bin/env python3
"""Render the Episode-49 v11b phone-carrier audit with actual ALOHA geometry.

All robot panels replay the unchanged latency-aligned optimized_action.  Only
diagnostic phone/accessory state differs between carrier candidates.  No
physics, G1, IK, phasewarp, orientation retargeting, Dex3, DDS, or hardware.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np


ROOT = Path("/home/jbnu/aloha_g1_dataset")
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import build_source_fk_parity_v11 as base  # noqa: E402


V11 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_phone_carrier_audit_v11b"
MODEL_XML = Path("/home/jbnu/trossen_arm_mujoco/trossen_arm_mujoco/assets/stationary_ai/stationary_ai.xml")
ACTION = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
SOURCE_FRAMES = ROOT / "configs/episode49_source_object_frames.user_approved.json"
CAM_HIGH = ROOT / "raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000"

PHONE_NPZ = OUT / "reconstructed_phone_trajectories.npz"
ACCESSORY_NPZ = OUT / "reconstructed_accessory_trajectories.npz"
CANDIDATES_JSON = OUT / "phone_carrier_candidates.json"
COMPARISON_JSON = OUT / "phone_carrier_transform_comparison.json"
STABLE_JSON = OUT / "stable_carrier_interval.json"
ACCESSORY_JSON = OUT / "accessory_surface_gap_metrics.json"
PAD_JSON = OUT / "phone_pad_metrics.json"
VIABILITY_JSON = OUT / "single_rigid_carrier_viability.json"
OPT_FK = V11 / "optimized_action_fk.npz"

FPS_REVIEW = 7.5
FRAME_COUNT = 990
KEY_FRAMES = [176, 200, 223, 300, 319, 326, 329, 334, 341, 380, 523, 530, 586]
COLORS = {
    "phone": np.array([0.05, 0.12, 0.24, 1.0]),
    "phone_hypothesis": np.array([0.0, 0.75, 1.0, 0.30]),
    "phone_known": np.array([0.95, 0.58, 0.08, 0.35]),
    "accessory": np.array([0.95, 0.95, 0.98, 1.0]),
    "accessory_hypothesis": np.array([1.0, 0.25, 0.2, 0.45]),
    "pad": np.array([0.90, 0.92, 0.95, 1.0]),
    "left": np.array([0.2, 1.0, 0.25, 1.0]),
    "right": np.array([0.2, 0.65, 1.0, 1.0]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def inverse(value: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second), -1.0, 1.0))))


def add_box(renderer: mujoco.Renderer, pose_model: np.ndarray, rgba: np.ndarray) -> None:
    base.add_geom(
        renderer.scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array([0.0748, 0.003975, 0.03575]),
        pose_model[:3, 3],
        pose_model[:3, :3],
        rgba,
    )


def add_ring(renderer: mujoco.Renderer, pose_model: np.ndarray, rgba: np.ndarray, width: float = 0.0022) -> None:
    radius = 0.025
    for index in range(28):
        first = 2.0 * np.pi * index / 28
        second = 2.0 * np.pi * (index + 1) / 28
        local_first = np.array([radius * np.cos(first), 0.0, radius * np.sin(first), 1.0])
        local_second = np.array([radius * np.cos(second), 0.0, radius * np.sin(second), 1.0])
        base.add_connector(
            renderer.scene,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            width,
            (pose_model @ local_first)[:3],
            (pose_model @ local_second)[:3],
            rgba,
        )


def add_axes(renderer: mujoco.Renderer, pose_model: np.ndarray, scale: float = 0.052) -> None:
    origin = pose_model[:3, 3]
    for column, color in enumerate(
        (np.array([1.0, 0.12, 0.12, 1.0]), np.array([0.12, 1.0, 0.12, 1.0]), np.array([0.12, 0.35, 1.0, 1.0]))
    ):
        base.add_connector(
            renderer.scene,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.002,
            origin,
            origin + scale * pose_model[:3, column],
            color,
        )


def add_pad(renderer: mujoco.Renderer, root_inverse: np.ndarray, pad_source: np.ndarray) -> None:
    pad = root_inverse @ pad_source
    base.add_geom(
        renderer.scene,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.0295, 0.004, 0.0]),
        pad[:3, 3],
        pad[:3, :3],
        COLORS["pad"],
    )
    base_source = np.eye(4)
    base_source[:3, 3] = np.array([0.42, 0.520, 0.819])
    stand_base = root_inverse @ base_source
    base.add_geom(
        renderer.scene,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([0.0525, 0.012, 0.0]),
        stand_base[:3, 3],
        stand_base[:3, :3],
        np.array([0.45, 0.48, 0.52, 1.0]),
    )


def add_tcp_markers(renderer: mujoco.Renderer, tcp_model: np.ndarray) -> None:
    for side, color in enumerate((COLORS["left"], COLORS["right"])):
        pose = tcp_model[side]
        base.add_geom(
            renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.008, 0.0, 0.0]),
            pose[:3, 3], np.eye(3), color,
        )
        add_axes(renderer, pose, scale=0.045)


def add_gap_connector(
    renderer: mujoco.Renderer,
    root_inverse: np.ndarray,
    object_point_source: np.ndarray,
    gripper_point_source: np.ndarray,
) -> None:
    object_point = (root_inverse @ np.r_[object_point_source, 1.0])[:3]
    gripper_point = (root_inverse @ np.r_[gripper_point_source, 1.0])[:3]
    base.add_geom(
        renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.005, 0.0, 0.0]),
        object_point, np.eye(3), np.array([1.0, 0.1, 0.1, 1.0]),
    )
    base.add_geom(
        renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.005, 0.0, 0.0]),
        gripper_point, np.eye(3), np.array([1.0, 1.0, 0.1, 1.0]),
    )
    base.add_connector(
        renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.003,
        gripper_point, object_point, np.array([1.0, 0.05, 0.05, 1.0]),
    )


def event_at(frame: int, ordered: list[dict[str, Any]]) -> str:
    result = "pre_task"
    for row in ordered:
        if int(row["frame"]) <= frame:
            result = str(row["event"])
        else:
            break
    return result


def overlay_lines(image: np.ndarray, lines: list[str], colors: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    result = image.copy()
    height = 19 + 21 * len(lines)
    cv2.rectangle(result, (0, 0), (result.shape[1], height), (6, 6, 6), -1)
    if colors is None:
        colors = [(245, 245, 245)] * len(lines)
    for index, line in enumerate(lines):
        cv2.putText(
            result, line, (8, 19 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX,
            0.40, colors[min(index, len(colors) - 1)], 1, cv2.LINE_AA,
        )
    return result


def candidate_state_text(
    candidate_id: str,
    candidate_index: int,
    frame: int,
    action_index: int,
    event: str,
    phone_pose: np.ndarray,
    phone_valid: bool,
    accessory_pose: np.ndarray,
    accessory_valid: bool,
    tcp_source: np.ndarray,
    desired_phone: np.ndarray,
    gap_value: float | None,
) -> list[str]:
    if phone_valid:
        phone_text = f"phone [{phone_pose[0,3]:+.3f},{phone_pose[1,3]:+.3f},{phone_pose[2,3]:+.3f}]"
    else:
        phone_text = "phone OBJECT STATE UNRESOLVED DURING GRASP ACQUISITION"
    if accessory_valid:
        accessory_text = f"accessory [{accessory_pose[0,3]:+.3f},{accessory_pose[1,3]:+.3f},{accessory_pose[2,3]:+.3f}]"
    elif frame > 341:
        accessory_text = "accessory state unresolved after removal; no right carrier fabricated"
    else:
        accessory_text = "accessory object state unresolved"
    center_error = float(np.linalg.norm(phone_pose[:3, 3] - desired_phone[:3, 3])) if np.isfinite(phone_pose).all() else float("nan")
    normal_error = angle_between_deg(phone_pose[:3, 1], desired_phone[:3, 1]) if np.isfinite(phone_pose).all() else float("nan")
    gap_text = "right ring gap n/a" if gap_value is None else f"right ring-surface gap={gap_value*1000:.1f}mm"
    prefix = "PIECEWISE OBJECT CARRIER DIAGNOSTIC | NOT YET APPROVED" if candidate_index in (1, 2) else "CONTACT-START RIGID HYPOTHESIS | DIAGNOSTIC ONLY"
    return [
        f"{candidate_id} | observed {frame:03d}/989 -> action[{action_index:03d}] | {event}",
        phone_text,
        accessory_text,
        f"L TCP [{tcp_source[0,0,3]:+.3f},{tcp_source[0,1,3]:+.3f},{tcp_source[0,2,3]:+.3f}] R TCP [{tcp_source[1,0,3]:+.3f},{tcp_source[1,1,3]:+.3f},{tcp_source[1,2,3]:+.3f}]",
        f"phone-pad center={center_error*1000:.1f}mm normal={normal_error:.1f}deg | {gap_text}",
        prefix + " | NO G1 / NO PHASEWARP / KINEMATIC ONLY",
    ]


def raw_panel(path: Path, frame: int, action_index: int, event: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    return overlay_lines(
        image,
        [
            f"RAW cam_high | observed {frame:03d}/989 | {event}",
            f"approved alignment reference: action[{action_index:03d}] | video/timeline NOT shifted",
            "recorded image only; not substituted for FK or object reconstruction",
            "phone carrier audit | source diagnostic | NO G1 / NO PHYSICS",
        ],
        [(245, 245, 245), (70, 220, 255), (170, 220, 170), (100, 100, 255)],
    )


def add_candidate_objects(
    renderer: mujoco.Renderer,
    root_inverse: np.ndarray,
    candidate_index: int,
    frame: int,
    phone_pose: np.ndarray,
    phone_valid: bool,
    rigid_phone_pose: np.ndarray,
    accessory_pose: np.ndarray,
    accessory_valid: bool,
    attached_accessory_pose: np.ndarray,
    initial_phone: np.ndarray,
    desired_phone: np.ndarray,
    pad: np.ndarray,
    tcp_model: np.ndarray,
    gap_points: tuple[np.ndarray, np.ndarray] | None,
    force_hypothesis_objects: bool = False,
) -> None:
    add_pad(renderer, root_inverse, pad)
    add_tcp_markers(renderer, tcp_model)
    if phone_valid:
        phone_model = root_inverse @ phone_pose
        add_box(renderer, phone_model, COLORS["phone"])
        add_axes(renderer, phone_model)
    else:
        # Two transparent hypotheses expose the inconsistency without claiming
        # an object trajectory during acquisition.
        add_box(renderer, root_inverse @ initial_phone, COLORS["phone_known"])
        add_box(renderer, root_inverse @ rigid_phone_pose, COLORS["phone_hypothesis"])
        add_axes(renderer, root_inverse @ rigid_phone_pose)
    if accessory_valid:
        add_ring(renderer, root_inverse @ accessory_pose, COLORS["accessory"])
    elif force_hypothesis_objects or frame == 341:
        add_ring(renderer, root_inverse @ attached_accessory_pose, COLORS["accessory_hypothesis"])
    if frame == 530:
        add_box(renderer, root_inverse @ desired_phone, np.array([0.1, 1.0, 0.2, 0.22]))
        add_axes(renderer, root_inverse @ desired_phone)
    if gap_points is not None:
        add_gap_connector(renderer, root_inverse, gap_points[0], gap_points[1])


def render_candidate(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str | mujoco.MjvCamera,
    qpos: np.ndarray,
    candidate_id: str,
    candidate_index: int,
    frame: int,
    action_index: int,
    event: str,
    phone_pose: np.ndarray,
    phone_valid: bool,
    rigid_phone_pose: np.ndarray,
    accessory_pose: np.ndarray,
    accessory_valid: bool,
    attached_accessory_pose: np.ndarray,
    initial_phone: np.ndarray,
    desired_phone: np.ndarray,
    pad: np.ndarray,
    tcp_model: np.ndarray,
    tcp_source: np.ndarray,
    root_inverse: np.ndarray,
    gap_value: float | None,
    gap_points: tuple[np.ndarray, np.ndarray] | None,
    force_hypothesis_objects: bool = False,
    compact: bool = False,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    add_candidate_objects(
        renderer, root_inverse, candidate_index, frame, phone_pose, phone_valid,
        rigid_phone_pose, accessory_pose, accessory_valid, attached_accessory_pose,
        initial_phone, desired_phone, pad, tcp_model, gap_points, force_hypothesis_objects,
    )
    image = renderer.render().copy()
    image = np.clip(image.astype(np.float32) * 1.52 + 9.0, 0.0, 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lines = candidate_state_text(
        candidate_id, candidate_index, frame, action_index, event, phone_pose,
        phone_valid, accessory_pose, accessory_valid, tcp_source, desired_phone, gap_value,
    )
    if compact:
        lines = [lines[0], lines[4], lines[5]]
    return overlay_lines(
        image,
        lines,
        [(245, 245, 245), (80, 255, 120), (245, 225, 150), (120, 210, 255), (80, 100, 255), (100, 100, 255)],
    )


def free_camera(focus_model: np.ndarray, kind: str) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = focus_model
    camera.distance = 0.42 if kind == "phone" else 0.32
    camera.azimuth = 115.0
    camera.elevation = -28.0
    return camera


def save_sheet(path: Path, rows: list[list[np.ndarray]], cell_size: tuple[int, int]) -> None:
    width, height = cell_size
    normalized = [
        [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in row]
        for row in rows
    ]
    sheet = np.vstack([np.hstack(row) for row in normalized])
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Could not write {path}")


def draw_comparison_chart(
    path: Path,
    comparison: dict[str, Any],
    pad_metrics: dict[str, Any],
    accessory_metrics: dict[str, Any],
    viability: dict[str, Any],
) -> None:
    canvas = np.full((1160, 1900, 3), (18, 18, 20), dtype=np.uint8)
    cv2.putText(canvas, "EP49 PHONE CARRIER TRANSFORM COMPARISON", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, viability["status"], (45, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.83, (80, 120, 255), 2, cv2.LINE_AA)
    headers = ["PAIR", "TRANSLATION", "ROTATION", "SAME RIGID CARRIER"]
    xs = [55, 790, 1110, 1460]
    for x, value in zip(xs, headers):
        cv2.putText(canvas, value, (x, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (100, 230, 255), 1, cv2.LINE_AA)
    for row_index, row in enumerate(comparison["pairwise"]):
        y = 235 + row_index * 90
        name = f"{row['first']}  vs  {row['second']}"
        values = [name, f"{row['translation_difference_m']*1000:.3f} mm", f"{row['rotation_difference_deg']:.3f} deg", str(row["same_rigid_carrier"]).upper()]
        for x, value in zip(xs, values):
            color = (100, 255, 130) if value == "TRUE" else ((80, 90, 255) if value == "FALSE" else (235, 235, 235))
            cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 1, cv2.LINE_AA)
        # Translation and rotation bars make the >10mm/>10deg threshold obvious.
        cv2.rectangle(canvas, (790, y + 18), (790 + int(min(row["translation_difference_m"] * 1000 / 100 * 260, 260)), y + 32), (90, 100, 255), -1)
        cv2.rectangle(canvas, (1110, y + 18), (1110 + int(min(row["rotation_difference_deg"] / 120 * 260, 260)), y + 32), (90, 100, 255), -1)

    y0 = 545
    headers = ["CANDIDATE", "F223 PORTRAIT", "F326 RING GAP", "F341 ATTACHED GAP", "F530 PAD CENTER", "F530 NORMAL"]
    xs = [55, 650, 930, 1170, 1450, 1690]
    for x, value in zip(xs, headers):
        cv2.putText(canvas, value, (x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 230, 255), 1, cv2.LINE_AA)
    for row_index, candidate_id in enumerate(("CONTACT_START_176", "CHARGER_ANCHORED_530", "OBSERVED_GEOMETRY_530")):
        y = y0 + 68 + row_index * 80
        portrait = pad_metrics["candidate_frame_223_portrait_metrics"][candidate_id]["long_axis_to_nearest_world_vertical_deg"]
        events = accessory_metrics["candidate_metrics"][candidate_id]["events"]
        pad_row = pad_metrics["candidate_frame_530_metrics"][candidate_id]
        values = [
            candidate_id,
            f"{portrait:.2f} deg",
            f"{events['326']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm",
            f"{events['341']['right_gripper_to_ring_surface_gap_m']*1000:.1f} mm",
            f"{pad_row['center_to_pad_face_m']*1000:.2f} mm",
            f"{pad_row['back_normal_to_desired_deg']:.2f} deg",
        ]
        for x, value in zip(xs, values):
            cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (238, 238, 238), 1, cv2.LINE_AA)

    notes = [
        "Rigid-carrier equivalence threshold: translation <=10 mm AND rotation <=10 deg.",
        "Accessory hypothesis uses only the verified phone-back attachment [0, 0.006425, 0] m.",
        "No right-hand carrier is fitted after a failed acquisition contact; frame-341 value is the phone-attached boundary hypothesis.",
        "CHARGER_ANCHORED_530 satisfies the pad pose by construction but does not validate frame-176 contact or accessory semantics.",
        "PIECEWISE OBJECT CARRIER DIAGNOSTIC - NOT YET APPROVED - NO G1 TARGET GENERATED.",
    ]
    for index, line in enumerate(notes):
        cv2.putText(canvas, line, (55, 900 + 48 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (170, 210, 240) if index < 4 else (80, 100, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(path)


def main() -> int:
    required = [
        PHONE_NPZ, ACCESSORY_NPZ, CANDIDATES_JSON, COMPARISON_JSON, STABLE_JSON,
        ACCESSORY_JSON, PAD_JSON, VIABILITY_JSON, OPT_FK, MODEL_XML, ACTION, TIMELINE, SOURCE_FRAMES,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    phone = load_npz(PHONE_NPZ)
    accessory = load_npz(ACCESSORY_NPZ)
    optimized_fk = load_npz(OPT_FK)
    source_frames = load_json(SOURCE_FRAMES)
    comparison = load_json(COMPARISON_JSON)
    accessory_metrics = load_json(ACCESSORY_JSON)
    pad_metrics = load_json(PAD_JSON)
    viability = load_json(VIABILITY_JSON)

    candidate_ids = [str(value) for value in phone["candidate_ids"]]
    if candidate_ids != ["CONTACT_START_176", "CHARGER_ANCHORED_530", "OBSERVED_GEOMETRY_530"]:
        raise RuntimeError(candidate_ids)
    lookup = phone["aligned_action_index"].astype(np.int64)
    phone_rigid = phone["T_source_scene_from_phone_rigid_reconstruction"]
    phone_visual = phone["T_source_scene_from_phone_visual_diagnostic"]
    phone_valid = phone["phone_visual_valid_mask"]
    accessory_attached = accessory["T_source_scene_from_accessory_phone_attached_hypothesis"]
    accessory_visual = accessory["T_source_scene_from_accessory_visual_diagnostic"]
    accessory_valid = accessory["accessory_visual_valid_mask"]
    gaps = accessory["right_gripper_to_ring_surface_gap_m"]
    gap_object_points = accessory["nearest_ring_surface_point_source_scene_m"]
    gap_gripper_points = accessory["nearest_right_gripper_surface_point_source_scene_m"]
    initial_phone = phone["T_source_scene_from_initial_phone"]
    desired_phone = phone["T_source_scene_from_desired_phone_on_charger"]
    pad = np.asarray(source_frames["T_source_scene_from_charger_pad"], dtype=np.float64)

    raw_files = sorted(CAM_HIGH.glob("frame_*.png"))
    if len(raw_files) != FRAME_COUNT:
        raise RuntimeError(f"Expected 990 cam_high frames, got {len(raw_files)}")
    ordered, _ = base.events()

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = (0.72, 0.72, 0.72)
    model.vis.headlight.diffuse[:] = (1.0, 1.0, 1.0)
    model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    source_root = optimized_fk["source_aloha_root_transform"]
    root_inverse = inverse(source_root)
    qpos = optimized_fk["qpos"]
    model_tcp = np.stack((
        np.concatenate((optimized_fk["left_tcp_rotation_model"], optimized_fk["left_tcp_position_model"][:, :, None]), axis=2),
        np.concatenate((optimized_fk["right_tcp_rotation_model"], optimized_fk["right_tcp_position_model"][:, :, None]), axis=2),
    ), axis=1)
    # Complete the homogeneous row discarded by the compact reconstruction above.
    homogeneous = np.zeros((FRAME_COUNT, 2, 4, 4), dtype=np.float64)
    homogeneous[:, :, :3, :] = model_tcp
    homogeneous[:, :, 3, 3] = 1.0
    model_tcp = homogeneous
    source_tcp = np.stack((optimized_fk["left_tcp_transform"], optimized_fk["right_tcp_transform"]), axis=1)

    video_raw = OUT / ".phone_carrier_candidate_4panel.raw.mp4"
    video = OUT / "phone_carrier_candidate_4panel.mp4"
    writer = base.open_writer(video_raw, 1920, 360)
    key_overview: dict[int, list[np.ndarray]] = {}
    try:
        for frame in range(FRAME_COUNT):
            action_index = int(lookup[frame])
            event = event_at(frame, ordered)
            raw = raw_panel(raw_files[frame], frame, action_index, event)
            row = [cv2.resize(raw, (480, 360), interpolation=cv2.INTER_AREA)]
            for candidate_index, candidate_id in enumerate(candidate_ids):
                if 300 <= frame <= 350:
                    sweep_index = frame - 300
                    gap_value = float(gaps[candidate_index, sweep_index])
                    gap_points = (
                        gap_object_points[candidate_index, sweep_index],
                        gap_gripper_points[candidate_index, sweep_index],
                    )
                else:
                    gap_value = None
                    gap_points = None
                pose_for_text = (
                    phone_visual[candidate_index, frame]
                    if bool(phone_valid[candidate_index, frame])
                    else phone_rigid[candidate_index, frame]
                )
                accessory_for_text = (
                    accessory_visual[candidate_index, frame]
                    if bool(accessory_valid[candidate_index, frame])
                    else accessory_attached[candidate_index, frame]
                )
                panel = render_candidate(
                    renderer, model, data, "cam_high", qpos[action_index], candidate_id,
                    candidate_index, frame, action_index, event, pose_for_text,
                    bool(phone_valid[candidate_index, frame]), phone_rigid[candidate_index, frame],
                    accessory_for_text, bool(accessory_valid[candidate_index, frame]),
                    accessory_attached[candidate_index, frame], initial_phone, desired_phone,
                    pad, model_tcp[action_index], source_tcp[action_index], root_inverse,
                    gap_value, gap_points,
                )
                row.append(cv2.resize(panel, (480, 360), interpolation=cv2.INTER_AREA))
            frame_image = np.hstack(row)
            writer.write(frame_image)
            if frame in KEY_FRAMES:
                key_overview[frame] = [value.copy() for value in row]
    finally:
        writer.release()

    metadata = {
        "status": viability["status"],
        "secondary_status": viability["secondary_status"],
        "panels": ["raw_cam_high", *candidate_ids],
        "frame_count": 990,
        "output_fps": 7.5,
        "source_fps": 30.0,
        "action_to_observation_lag_frames": 7,
        "action_sample_for_observed_frame": "observed_frame - 7",
        "optimized_action_path": str(ACTION.resolve()),
        "optimized_action_sha256": sha256(ACTION),
        "approved_timeline_path": str(TIMELINE.resolve()),
        "approved_timeline_sha256": sha256(TIMELINE),
        "stationary_aloha_model": str(MODEL_XML.resolve()),
        "stationary_aloha_model_sha256": sha256(MODEL_XML),
        "phone_trajectory_npz": str(PHONE_NPZ.resolve()),
        "phone_trajectory_npz_sha256": sha256(PHONE_NPZ),
        "accessory_trajectory_npz": str(ACCESSORY_NPZ.resolve()),
        "accessory_trajectory_npz_sha256": sha256(ACCESSORY_NPZ),
        "optimized_action_modified": False,
        "event_frames_modified": False,
        "observation_state_used_as_motion_source": False,
        "g1": False,
        "phasewarp": False,
        "orientation_target": False,
        "dex3": False,
        "physics": False,
        "dds": False,
        "publisher": False,
        "hardware": False,
    }
    base.embed_metadata(video_raw, video, "Episode49 phone carrier candidate audit v11b", metadata)
    if base.decoded_frames(video) != FRAME_COUNT:
        raise RuntimeError("4-panel video does not contain 990 decoded frames")

    overview_rows = [key_overview[frame] for frame in KEY_FRAMES]
    overview_path = OUT / "phone_carrier_contact_sheet_overview.png"
    save_sheet(overview_path, overview_rows, (480, 360))

    phone_rows: list[list[np.ndarray]] = []
    accessory_rows: list[list[np.ndarray]] = []
    for frame in KEY_FRAMES:
        action_index = int(lookup[frame])
        event = event_at(frame, ordered)
        phone_row: list[np.ndarray] = []
        accessory_row: list[np.ndarray] = []
        for candidate_index, candidate_id in enumerate(candidate_ids):
            if 300 <= frame <= 350:
                sweep_index = frame - 300
                gap_value = float(gaps[candidate_index, sweep_index])
                gap_points = (
                    gap_object_points[candidate_index, sweep_index],
                    gap_gripper_points[candidate_index, sweep_index],
                )
            else:
                gap_value = None
                gap_points = None
            phone_pose = (
                phone_visual[candidate_index, frame]
                if bool(phone_valid[candidate_index, frame])
                else phone_rigid[candidate_index, frame]
            )
            accessory_pose = (
                accessory_visual[candidate_index, frame]
                if bool(accessory_valid[candidate_index, frame])
                else accessory_attached[candidate_index, frame]
            )
            phone_focus_model = (root_inverse @ phone_rigid[candidate_index, frame])[:3, 3]
            accessory_focus_model = (root_inverse @ accessory_attached[candidate_index, frame])[:3, 3]
            phone_image = render_candidate(
                renderer, model, data, free_camera(phone_focus_model, "phone"), qpos[action_index],
                candidate_id, candidate_index, frame, action_index, event, phone_pose,
                bool(phone_valid[candidate_index, frame]), phone_rigid[candidate_index, frame],
                accessory_pose, bool(accessory_valid[candidate_index, frame]), accessory_attached[candidate_index, frame],
                initial_phone, desired_phone, pad, model_tcp[action_index], source_tcp[action_index], root_inverse,
                gap_value, gap_points, force_hypothesis_objects=False, compact=True,
            )
            accessory_image = render_candidate(
                renderer, model, data, free_camera(accessory_focus_model, "accessory"), qpos[action_index],
                candidate_id, candidate_index, frame, action_index, event, phone_pose,
                bool(phone_valid[candidate_index, frame]), phone_rigid[candidate_index, frame],
                accessory_pose, bool(accessory_valid[candidate_index, frame]), accessory_attached[candidate_index, frame],
                initial_phone, desired_phone, pad, model_tcp[action_index], source_tcp[action_index], root_inverse,
                gap_value, gap_points, force_hypothesis_objects=True, compact=True,
            )
            phone_row.append(phone_image)
            accessory_row.append(accessory_image)
        phone_rows.append(phone_row)
        accessory_rows.append(accessory_row)
    phone_closeup = OUT / "phone_carrier_contact_sheet_closeup.png"
    accessory_closeup = OUT / "accessory_contact_sheet_closeup.png"
    save_sheet(phone_closeup, phone_rows, (480, 360))
    save_sheet(accessory_closeup, accessory_rows, (480, 360))

    chart_path = OUT / "carrier_transform_comparison.png"
    draw_comparison_chart(chart_path, comparison, pad_metrics, accessory_metrics, viability)

    visual = {
        "status": "PHONE_CARRIER_AUDIT_VISUALS_READY",
        "primary_status": viability["status"],
        "secondary_status": viability["secondary_status"],
        "video": {
            "path": str(video.resolve()),
            "sha256": sha256(video),
            "decoded_frames": base.decoded_frames(video),
            "fps": FPS_REVIEW,
        },
        "contact_sheets": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (overview_path, phone_closeup, accessory_closeup, chart_path)
        },
        "metadata": metadata,
    }
    base.dump_json(OUT / "visual_audit.json", visual)
    renderer.close()
    print(json.dumps(visual, indent=2, default=base.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
