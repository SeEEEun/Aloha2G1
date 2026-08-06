#!/usr/bin/env python3
"""Render source-only visual evidence for the Episode-49 accessory audit.

Every robot panel replays the unchanged latency-aligned optimized_action on the
actual Stationary ALOHA MuJoCo model.  Object geometry is kinematic diagnostic
overlay only.  No G1, IK, phasewarp, orientation retargeting, Dex3 target
motion, physics, DDS, publisher, or hardware path is present.
"""
from __future__ import annotations

import hashlib
import json
import math
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
import audit_episode49_accessory_semantics_v11c as audit  # noqa: E402


V11 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11"
V11B = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_phone_carrier_audit_v11b"
OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_accessory_semantics_audit_v11c"
CAM_HIGH = ROOT / "raw_recordings/GoPark_20260729_111223/images/observation.images.cam_high/episode_000000"
MODEL_XML = audit.MODEL_XML
ACTION = audit.ACTION
TIMELINE = audit.TIMELINE
LAYOUT = audit.LAYOUT
ACCESSORY_USD = audit.ACCESSORY_USD
SCENE_USD = audit.SCENE_USD
OPT_FK = audit.OPT_FK
PHONE_NPZ = audit.PHONE_NPZ
DISTANCE_NPZ = OUT / "accessory_distance_timeseries.npz"
ASSET_JSON = OUT / "accessory_asset_frame_audit.json"
LOCAL_JSON = OUT / "accessory_local_frame_attachment_audit.json"
GAP_JSON = OUT / "ring_center_gap_orientation_audit.json"
CONTACT_JSON = OUT / "right_contact_proxy_audit.json"
SEMANTICS_JSON = OUT / "frame_326_341_semantics_audit.json"
DECISION_JSON = OUT / "five_cause_decision.json"
INVARIANTS_JSON = OUT / "constraint_invariants.json"

FRAME_COUNT = 990
FPS_REVIEW = 7.5
VIDEO_KEY_FRAMES = [176, 200, 223, 280, 300, 310, 319, 326, 329, 334, 341, 350, 380, 530, 586]
SHEET_FRAMES = [223, 280, 300, 310, 319, 326, 329, 334, 341, 350]

COLORS = {
    "phone": np.array([0.04, 0.16, 0.30, 1.0]),
    "phone_unresolved": np.array([1.0, 0.62, 0.05, 0.25]),
    "main": np.array([0.96, 0.96, 0.98, 1.0]),
    "support": np.array([0.95, 0.20, 0.90, 1.0]),
    "hinge": np.array([1.0, 0.56, 0.10, 1.0]),
    "counterfactual": np.array([1.0, 0.12, 0.10, 0.38]),
    "left": np.array([0.18, 1.0, 0.28, 1.0]),
    "right": np.array([0.20, 0.68, 1.0, 1.0]),
    "proxy": np.array([0.95, 0.78, 0.12, 0.38]),
    "nearest": np.array([1.0, 0.15, 0.08, 0.70]),
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
        return {name: archive[name].copy() for name in archive.files}


def overlay(image: np.ndarray, lines: list[str], colors: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    result = image.copy()
    height = min(result.shape[0], 13 + 20 * len(lines))
    cv2.rectangle(result, (0, 0), (result.shape[1], height), (5, 5, 6), -1)
    if colors is None:
        colors = [(240, 240, 240)] * len(lines)
    for index, line in enumerate(lines):
        cv2.putText(
            result, line, (7, 17 + index * 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, colors[min(index, len(colors) - 1)], 1, cv2.LINE_AA,
        )
    return result


def add_axes(renderer: mujoco.Renderer, pose_model: np.ndarray, scale: float = 0.045) -> None:
    origin = pose_model[:3, 3]
    colors = (
        np.array([1.0, 0.1, 0.1, 1.0]),
        np.array([0.1, 1.0, 0.1, 1.0]),
        np.array([0.1, 0.35, 1.0, 1.0]),
    )
    for axis, color in enumerate(colors):
        base.add_connector(
            renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.0018,
            origin, origin + scale * pose_model[:3, axis], color,
        )


def add_phone(renderer: mujoco.Renderer, pose_model: np.ndarray, rgba: np.ndarray) -> None:
    base.add_geom(
        renderer.scene, mujoco.mjtGeom.mjGEOM_BOX,
        np.array([0.0748, 0.003975, 0.03575]), pose_model[:3, 3], pose_model[:3, :3], rgba,
    )


def add_circle(
    renderer: mujoco.Renderer,
    pose_model: np.ndarray,
    center_local: np.ndarray,
    radius: float,
    axis: str,
    rgba: np.ndarray,
    *,
    start_degrees: float = 0.0,
    end_degrees: float = 360.0,
    segments: int = 56,
    width: float = 0.0025,
) -> None:
    angles = np.linspace(math.radians(start_degrees), math.radians(end_degrees), segments + 1)
    points: list[np.ndarray] = []
    for angle in angles:
        if axis.upper() == "Y":
            local = center_local + np.array([radius * math.cos(angle), 0.0, radius * math.sin(angle)])
        elif axis.upper() == "Z":
            local = center_local + np.array([radius * math.cos(angle), radius * math.sin(angle), 0.0])
        else:
            raise ValueError(axis)
        points.append((pose_model @ np.r_[local, 1.0])[:3])
    for first, second in zip(points[:-1], points[1:]):
        base.add_connector(renderer.scene, mujoco.mjtGeom.mjGEOM_CAPSULE, width, first, second, rgba)


def rotate_about_x_pose(angle_degrees: float, pivot: np.ndarray) -> np.ndarray:
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]], dtype=np.float64)
    return audit.transform(None, pivot) @ audit.transform(rotation, None) @ audit.transform(None, -pivot)


def add_accessory(
    renderer: mujoco.Renderer,
    pose_model: np.ndarray,
    asset: dict[str, Any],
    *,
    counterfactual: bool = False,
    support_hinge_angle_degrees: float = 0.0,
    ghost_support: bool = False,
) -> None:
    main = asset["main_ring"]
    support = asset["support_ring"]
    hinge = asset["hinge"]
    alpha_color = COLORS["counterfactual"] if counterfactual else COLORS["main"]
    add_circle(
        renderer, pose_model, np.asarray(main["center_local_m"]),
        0.5 * (float(main["outer_radius_m"]) + float(main["inner_radius_m"])),
        "Y", alpha_color, start_degrees=-72.0, end_degrees=252.0,
    )
    support_pose = pose_model @ rotate_about_x_pose(
        support_hinge_angle_degrees, np.asarray(hinge["center_local_m"], dtype=np.float64)
    )
    support_color = np.array([0.95, 0.2, 0.9, 0.27]) if ghost_support else COLORS["support"]
    if counterfactual:
        support_color = np.array([1.0, 0.12, 0.10, 0.30])
    add_circle(
        renderer, support_pose, np.asarray(support["center_local_m"]),
        0.5 * (float(support["outer_radius_m"]) + float(support["inner_radius_m"])),
        "Z", support_color,
    )
    hinge_local = np.asarray(hinge["center_local_m"], dtype=np.float64)
    hinge_first = (pose_model @ np.r_[hinge_local + np.array([-0.0055, 0.0, 0.0]), 1.0])[:3]
    hinge_second = (pose_model @ np.r_[hinge_local + np.array([0.0055, 0.0, 0.0]), 1.0])[:3]
    base.add_connector(
        renderer.scene, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0032,
        hinge_first, hinge_second, COLORS["counterfactual"] if counterfactual else COLORS["hinge"],
    )
    # Main center, support center, and opening direction make the local frame
    # and gap convention visible rather than implied.
    main_center = pose_model[:3, 3]
    support_center = (support_pose @ np.r_[np.asarray(support["center_local_m"]), 1.0])[:3]
    base.add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0.0, 0.0]), main_center, np.eye(3), np.array([0.2, 1.0, 1.0, 1.0]))
    base.add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0.0, 0.0]), support_center, np.eye(3), support_color)
    opening = -pose_model[:3, 2]
    base.add_connector(renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.0025, main_center, main_center + 0.042 * opening, np.array([1.0, 0.15, 0.95, 1.0]))


def add_tcp(renderer: mujoco.Renderer, tcp_model: np.ndarray) -> None:
    base.add_geom(
        renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0065, 0.0, 0.0]),
        tcp_model[:3, 3], np.eye(3), COLORS["right"],
    )
    add_axes(renderer, tcp_model, scale=0.035)


def add_contact_boxes(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nearest_name: str | None,
) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("follower_right_gripper_"):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        color = COLORS["nearest"] if name == nearest_name else COLORS["proxy"]
        base.add_geom(
            renderer.scene, mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(model.geom_size[geom_id], dtype=np.float64),
            np.asarray(data.geom_xpos[geom_id], dtype=np.float64),
            np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3),
            color,
        )


def add_gap_line(
    renderer: mujoco.Renderer,
    root_inverse: np.ndarray,
    object_point_source: np.ndarray,
    gripper_point_source: np.ndarray,
    *,
    counterfactual: bool,
) -> None:
    object_point = (root_inverse @ np.r_[object_point_source, 1.0])[:3]
    gripper_point = (root_inverse @ np.r_[gripper_point_source, 1.0])[:3]
    color = np.array([1.0, 0.1, 0.1, 1.0]) if not counterfactual else np.array([1.0, 0.2, 0.8, 1.0])
    base.add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0048, 0.0, 0.0]), object_point, np.eye(3), color)
    base.add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0048, 0.0, 0.0]), gripper_point, np.eye(3), np.array([1.0, 1.0, 0.1, 1.0]))
    base.add_connector(renderer.scene, mujoco.mjtGeom.mjGEOM_ARROW, 0.0028, gripper_point, object_point, color)


def free_camera(focus_model: np.ndarray, azimuth: float, elevation: float, distance: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = focus_model
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    return camera


def raw_panel(path: Path, frame: int, action_index: int, event: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(path)
    lines = [
        f"RAW cam_high | observed {frame:03d}/989 | {event}",
        f"approved lookup: optimized_action[{action_index:03d}] -> observed frame {frame:03d}",
        "raw RGB is visual evidence only; NOT used as calibrated 3-D geometry",
        "timeline/video/action unchanged | SOURCE AUDIT ONLY | NO G1 / NO PHYSICS",
    ]
    return overlay(image, lines, [(245, 245, 245), (80, 220, 255), (170, 220, 170), (80, 100, 255)])


def object_policy(frame: int) -> tuple[str, bool, bool]:
    if 176 <= frame <= 222:
        return "OBJECT STATE UNRESOLVED DURING GRASP ACQUISITION", False, False
    if frame < 176:
        return "known initial source object state", True, True
    if 223 <= frame < 341:
        return "CHARGER_ANCHORED_530 diagnostic carrier; phone-attached accessory", True, True
    if frame == 341:
        return "ACCESSORY_REMOVED boundary; attached pose shown counterfactually", True, True
    return "accessory state unresolved after removal; no right carrier fabricated", True, False


def render_scene_panel(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str | mujoco.MjvCamera,
    qpos: np.ndarray,
    frame: int,
    action_index: int,
    event: str,
    phone_model: np.ndarray,
    accessory_model: np.ndarray,
    initial_phone_model: np.ndarray,
    initial_accessory_model: np.ndarray,
    tcp_model: np.ndarray,
    root_inverse: np.ndarray,
    asset: dict[str, Any],
    gap_m: float | None,
    object_point: np.ndarray | None,
    gripper_point: np.ndarray | None,
    nearest_name: str | None,
    *,
    closeup: bool,
    best_hinge_angle: float,
    best_hinge_gap_m: float,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    scene_option = None
    if closeup:
        # X-ray contact-proxy view: the overview already shows the complete
        # actual robot.  Hide model groups here, then add only the six actual
        # gripper collision OBBs plus audited object geometry and markers so
        # arm/table occlusion cannot conceal the measured gap.
        scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_option)
        scene_option.geomgroup[:] = 0
        scene_option.sitegroup[:] = 0
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    policy, phone_valid, accessory_valid = object_policy(frame)
    if frame < 176:
        add_phone(renderer, initial_phone_model, COLORS["phone"])
        add_accessory(renderer, initial_accessory_model, asset)
    elif not phone_valid:
        add_phone(renderer, initial_phone_model, COLORS["phone_unresolved"])
    else:
        add_phone(renderer, phone_model, COLORS["phone"])
        add_axes(renderer, phone_model, scale=0.04)
        if accessory_valid:
            add_accessory(renderer, accessory_model, asset, counterfactual=(frame >= 341))
            if closeup and 280 <= frame <= 341:
                add_accessory(
                    renderer, accessory_model, asset, support_hinge_angle_degrees=best_hinge_angle,
                    ghost_support=True,
                )
    add_tcp(renderer, tcp_model)
    add_contact_boxes(renderer, model, data, nearest_name if 280 <= frame <= 341 else None)
    if object_point is not None and gripper_point is not None and 280 <= frame <= 341:
        add_gap_line(renderer, root_inverse, object_point, gripper_point, counterfactual=(frame >= 341))
    image = renderer.render().copy()
    image = np.clip(image.astype(np.float32) * 1.48 + 8.0, 0.0, 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gap_text = "gap n/a" if gap_m is None else f"actual ALOHA OBB gap={gap_m*1000:.2f} mm"
    panel_name = "ACTUAL ASSET + ALOHA COLLISION BOXES" if not closeup else "CONTACT-PROXY X-RAY (ACTUAL 6 OBBs)"
    lines = [
        f"{panel_name} | observed {frame:03d} -> action[{action_index:03d}] | {event}",
        policy,
        f"{gap_text} | nearest={nearest_name or 'n/a'}",
        "main=white support=magenta hinge=orange gap-arrow=magenta",
    ]
    if closeup:
        lines.append(f"best UNADOPTED hinge sweep={best_hinge_angle:+.0f}deg still {best_hinge_gap_m*1000:.1f}mm")
    lines.append("DIAGNOSTIC OBJECT OVERLAY | NO G1 / NO DEX3 MOTION / NO PHYSICS")
    return overlay(
        image, lines,
        [(245, 245, 245), (80, 220, 255), (80, 100, 255), (220, 200, 255), (220, 180, 255), (80, 100, 255)],
    )


def plot_dashboard(
    frame: int,
    frames: np.ndarray,
    optimized_gap: np.ndarray,
    observed_gap: np.ndarray,
    tcp_gap: np.ndarray,
    full_gap: np.ndarray,
    semantics: dict[str, Any],
    decision: dict[str, Any],
) -> np.ndarray:
    canvas = np.full((480, 640, 3), (16, 16, 18), dtype=np.uint8)
    cv2.putText(canvas, "FIVE-CAUSE NUMERIC DASHBOARD", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (245, 245, 245), 1, cv2.LINE_AA)
    plot_left, plot_top, plot_right, plot_bottom = 50, 52, 620, 245
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (75, 75, 78), 1)
    max_gap = 0.18

    def px(frame_value: int) -> int:
        return int(plot_left + (frame_value - int(frames[0])) / (int(frames[-1]) - int(frames[0])) * (plot_right - plot_left))

    def py(value: float) -> int:
        return int(plot_bottom - np.clip(value / max_gap, 0.0, 1.0) * (plot_bottom - plot_top))

    for threshold, label in ((0.01, "10mm gate"), (0.05, "50mm"), (0.10, "100mm"), (0.15, "150mm")):
        y = py(threshold)
        cv2.line(canvas, (plot_left, y), (plot_right, y), (55, 55, 58), 1)
        cv2.putText(canvas, label, (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (150, 150, 155), 1, cv2.LINE_AA)
    for event_frame, label in ((326, "326"), (329, "329"), (341, "341")):
        x = px(event_frame)
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), (80, 120, 255), 1)
        cv2.putText(canvas, label, (x - 10, plot_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 180, 255), 1, cv2.LINE_AA)

    def line(values: np.ndarray, color: tuple[int, int, int]) -> None:
        points = np.array([[px(int(f)), py(float(v))] for f, v in zip(frames, values)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)

    line(optimized_gap, (60, 80, 255))
    line(observed_gap, (80, 230, 120))
    line(tcp_gap, (255, 180, 50))
    line(full_gap, (220, 120, 220))
    if int(frames[0]) <= frame <= int(frames[-1]):
        cv2.line(canvas, (px(frame), plot_top), (px(frame), plot_bottom), (255, 255, 255), 1)
    cv2.putText(canvas, "red optimized OBB | green observation OBB | blue TCP | purple full ring", (55, 267), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (205, 205, 210), 1, cv2.LINE_AA)

    causes = decision["five_possibilities"]
    labels = [
        ("1 frame/attachment", "authored PASS; axis best 40.42mm; hinge best 63.82mm", (80, 180, 255)),
        ("2 center/gap", "full-ring lower bound 80.36mm; gap angle rejected", (80, 180, 255)),
        ("3 contact proxy", "actual 6 ALOHA OBBs used; best f326 80.36mm", (80, 180, 255)),
        ("4 event semantics", "f341 attached test INVALID; f326 is visual acquisition boundary", (80, 220, 255)),
        ("5 optimized reach", "NOT isolated: aligned right-arm direction matches observation", (90, 230, 130)),
    ]
    y = 302
    for name, detail, color in labels:
        cv2.putText(canvas, name, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, detail, (155, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (225, 225, 228), 1, cv2.LINE_AA)
        y += 31
    cv2.putText(canvas, decision["status"], (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 100, 255), 1, cv2.LINE_AA)
    return canvas


def save_sheet(path: Path, rows: list[list[np.ndarray]], cell_size: tuple[int, int]) -> None:
    width, height = cell_size
    normalized = [[cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in row] for row in rows]
    sheet = np.vstack([np.hstack(row) for row in normalized])
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(path)


def draw_gap_plot(path: Path, frames: np.ndarray, gaps: dict[str, np.ndarray], semantics: dict[str, Any]) -> None:
    canvas = np.full((900, 1600, 3), (17, 17, 19), dtype=np.uint8)
    cv2.putText(canvas, "EP49 ACCESSORY DISTANCE AUDIT (CHARGER_ANCHORED_530 DIAGNOSTIC)", (45, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (245, 245, 245), 2, cv2.LINE_AA)
    left, top, right, bottom = 100, 110, 1540, 660
    cv2.rectangle(canvas, (left, top), (right, bottom), (100, 100, 105), 1)
    max_gap = 0.20

    def px(value: int) -> int:
        return int(left + (value - int(frames[0])) / (int(frames[-1]) - int(frames[0])) * (right - left))

    def py(value: float) -> int:
        return int(bottom - np.clip(value / max_gap, 0.0, 1.0) * (bottom - top))

    for mm in (10, 25, 50, 75, 100, 125, 150, 175, 200):
        y = py(mm / 1000.0)
        cv2.line(canvas, (left, y), (right, y), (48, 48, 52), 1)
        cv2.putText(canvas, f"{mm} mm", (25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 165, 170), 1, cv2.LINE_AA)
    for frame, name in ((326, "grasp_start"), (329, "detach_start"), (341, "removed")):
        x = px(frame)
        cv2.line(canvas, (x, top), (x, bottom), (80, 120, 255), 2)
        cv2.putText(canvas, f"{frame} {name}", (x - 55, top - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (100, 180, 255), 1, cv2.LINE_AA)
    palette = {
        "optimized actual OBB": (60, 80, 255),
        "observation actual OBB": (80, 230, 120),
        "optimized TCP": (255, 180, 50),
        "complete-ring lower bound": (220, 120, 220),
        "main C-ring": (230, 230, 230),
        "support ring": (220, 80, 220),
        "hinge": (60, 180, 255),
    }
    for label, values in gaps.items():
        points = np.array([[px(int(frame)), py(float(value))] for frame, value in zip(frames, values)], dtype=np.int32)
        cv2.polylines(canvas, [points], False, palette[label], 2, cv2.LINE_AA)
    x, y = 90, 715
    for label, color in palette.items():
        cv2.line(canvas, (x, y), (x + 34, y), color, 4)
        cv2.putText(canvas, label, (x + 44, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 228), 1, cv2.LINE_AA)
        x += 330
        if x > 1350:
            x, y = 90, y + 55
    cv2.putText(canvas, "Frame 341 attached-pose values are counterfactual because the approved event is accessory_removed.", (90, 845), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 120, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(path)


def draw_gap_sweep(path: Path, gap_audit: dict[str, Any], local_audit: dict[str, Any]) -> None:
    canvas = np.full((920, 1600, 3), (17, 17, 19), dtype=np.uint8)
    cv2.putText(canvas, "RING GAP / AXIS / SUPPORT-HINGE DIAGNOSTIC SWEEPS", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
    gap_rows = gap_audit["sweep_5deg"]
    angles = np.array([row["gap_center_degrees"] for row in gap_rows], dtype=np.float64)
    values = np.array([row["events"]["326"]["gap_m"] * 1000.0 for row in gap_rows])
    hinge_rows = local_audit["support_ring_hinge_articulation_diagnostic"]["all_candidates"]
    hinge_angles = np.array([row["support_ring_rotation_about_hinge_x_degrees"] for row in hinge_rows], dtype=np.float64)
    hinge_values = np.array([row["events"]["326"]["gap_m"] * 1000.0 for row in hinge_rows])

    def chart(left: int, top: int, right: int, bottom: int, x_values: np.ndarray, y_values: np.ndarray, title: str, color: tuple[int, int, int]) -> None:
        cv2.rectangle(canvas, (left, top), (right, bottom), (85, 85, 90), 1)
        cv2.putText(canvas, title, (left, top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 230, 235), 1, cv2.LINE_AA)
        ymin, ymax = 0.0, max(100.0, float(np.max(y_values)) * 1.05)
        points = []
        for xx, yy in zip(x_values, y_values):
            px = int(left + (xx - x_values.min()) / (x_values.max() - x_values.min()) * (right - left))
            py = int(bottom - (yy - ymin) / (ymax - ymin) * (bottom - top))
            points.append((px, py))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        gate_y = int(bottom - 10.0 / ymax * (bottom - top))
        cv2.line(canvas, (left, gate_y), (right, gate_y), (80, 100, 255), 2)
        minimum = int(np.argmin(y_values))
        cv2.circle(canvas, points[minimum], 7, (80, 255, 120), -1)
        cv2.putText(canvas, f"best {x_values[minimum]:+.0f} deg = {y_values[minimum]:.2f} mm", (left + 20, bottom + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 120), 1, cv2.LINE_AA)

    chart(70, 135, 760, 565, angles, values, "36-deg main-ring gap center sweep", (220, 120, 220))
    chart(840, 135, 1530, 565, hinge_angles, hinge_values, "support-ring hinge +X sweep (UNADOPTED)", (220, 80, 220))
    best_axis = local_audit["best_frame_326_axis_candidate"]
    notes = [
        f"Authoritative -90 deg opening: {gap_audit['authoritative_frame_326_gap_m']*1000:.3f} mm",
        f"Complete main-ring lower bound: {gap_audit['complete_main_ring_lower_bound_frame_326_m']*1000:.3f} mm",
        f"Best proper signed-axis permutation: {best_axis['candidate_id']} = {best_axis['events']['326']['gap_m']*1000:.3f} mm",
        f"All are above the 10 mm contact gate. No diagnostic rotation was adopted.",
        "Main ring plane X-Z, normal +Y; opening centered along local -Z. Support ring plane X-Y, normal +Z.",
    ]
    for index, line in enumerate(notes):
        cv2.putText(canvas, line, (75, 690 + 43 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 225) if index < 3 else (80, 120, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(path)


def draw_cause_matrix(path: Path, decision: dict[str, Any]) -> None:
    canvas = np.full((1060, 1800, 3), (17, 17, 19), dtype=np.uint8)
    cv2.putText(canvas, "EP49 ACCESSORY GAP: FIVE-POSSIBILITY DECISION MATRIX", (45, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, decision["status"], (45, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 110, 255), 2, cv2.LINE_AA)
    rows = [
        ("1", "accessory local frame / attachment", "AUTHORED FRAME PASS; tested origins/axes/hinge remain >10mm"),
        ("2", "ring center / gap orientation", "REJECTED AS LARGE-GAP CAUSE; full-ring lower bound = 80.36mm"),
        ("3", "right-hand contact proxy", "TCP REJECTED; all six actual ALOHA pad/tip OBBs tested"),
        ("4", "frame 326 / 341 semantics", "PARTIAL CAUSE: f341 still-attached test invalid; f326 remains unresolved"),
        ("5", "optimized_action non-reach", "NOT ESTABLISHED; aligned optimized right arm tracks observation"),
    ]
    y = 190
    for number, title, result in rows:
        cv2.rectangle(canvas, (45, y - 38), (1755, y + 105), (42, 42, 46), -1)
        cv2.putText(canvas, number, (70, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.18, (100, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, title, (145, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 242), 1, cv2.LINE_AA)
        cv2.putText(canvas, result, (145, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 235, 140) if number == "5" else (110, 190, 255), 1, cv2.LINE_AA)
        y += 165
    notes = [
        "Exact blocker: the rigid CHARGER_ANCHORED_530 phone/accessory object-state hypothesis does not co-locate the modeled ring with the observed right hand at f326.",
        "This audit does not authorize a new carrier, accessory offset, event shift, G1 target, IK, phasewarp, orientation target, Dex3 motion, or physics.",
    ]
    for index, line in enumerate(notes):
        cv2.putText(canvas, line, (55, 985 + index * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 120, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(path)


def main() -> int:
    required = [
        MODEL_XML, ACTION, TIMELINE, LAYOUT, ACCESSORY_USD, SCENE_USD, OPT_FK,
        PHONE_NPZ, DISTANCE_NPZ, ASSET_JSON, LOCAL_JSON, GAP_JSON, CONTACT_JSON,
        SEMANTICS_JSON, DECISION_JSON, INVARIANTS_JSON,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    raw_files = sorted(CAM_HIGH.glob("frame_*.png"))
    if len(raw_files) != FRAME_COUNT:
        raise RuntimeError(f"Expected 990 raw cam_high frames, got {len(raw_files)}")

    optimized_fk = load_npz(OPT_FK)
    phone = load_npz(PHONE_NPZ)
    distances = load_npz(DISTANCE_NPZ)
    asset = load_json(ASSET_JSON)
    local_audit = load_json(LOCAL_JSON)
    gap_audit = load_json(GAP_JSON)
    contact_audit = load_json(CONTACT_JSON)
    semantics = load_json(SEMANTICS_JSON)
    decision = load_json(DECISION_JSON)
    invariants = load_json(INVARIANTS_JSON)
    if invariants["status"] != "PASS_NO_FORBIDDEN_MUTATION_OR_DOWNSTREAM_GENERATION":
        raise RuntimeError("Frozen-source invariant audit is not PASS")

    candidate_ids = [str(value) for value in phone["candidate_ids"]]
    charger_index = candidate_ids.index("CHARGER_ANCHORED_530")
    phone_source = phone["T_source_scene_from_phone_rigid_reconstruction"][charger_index]
    attachment = distances["attachment_transform"]
    accessory_source = np.einsum("tij,jk->tik", phone_source, attachment)
    lookup = phone["aligned_action_index"].astype(np.int64)
    source_root = optimized_fk["source_aloha_root_transform"]
    root_inverse = audit.inverse(source_root)
    phone_model = np.einsum("ij,tjk->tik", root_inverse, phone_source)
    accessory_model = np.einsum("ij,tjk->tik", root_inverse, accessory_source)
    initial_phone_model = root_inverse @ phone["T_source_scene_from_initial_phone"]
    initial_accessory_model = initial_phone_model @ attachment
    qpos = optimized_fk["qpos"]
    tcp_model = np.zeros((FRAME_COUNT, 4, 4), dtype=np.float64)
    tcp_model[:, :3, :3] = optimized_fk["right_tcp_rotation_model"]
    tcp_model[:, :3, 3] = optimized_fk["right_tcp_position_model"]
    tcp_model[:, 3, 3] = 1.0

    audit_frames = distances["observed_frames"].astype(np.int64)
    source_names = [str(value) for value in distances["source_names"]]
    component_names = [str(value) for value in distances["component_names"]]
    source_index = {name: index for index, name in enumerate(source_names)}
    component_index = {name: index for index, name in enumerate(component_names)}
    gap_array = distances["component_surface_gap_m"]
    optimized_gap = gap_array[source_index["optimized_action_aligned"], :, component_index["authoritative_all"]]
    observed_gap = gap_array[source_index["observation_state"], :, component_index["authoritative_all"]]
    tcp_gap = distances["right_TCP_to_authoritative_surface_m"][source_index["optimized_action_aligned"]]
    full_gap = gap_array[source_index["optimized_action_aligned"], :, component_index["complete_main_all"]]
    main_gap = gap_array[source_index["optimized_action_aligned"], :, component_index["main_c_ring"]]
    support_gap = gap_array[source_index["optimized_action_aligned"], :, component_index["support_ring"]]
    hinge_gap = gap_array[source_index["optimized_action_aligned"], :, component_index["hinge"]]
    nearest_object = distances["nearest_accessory_surface_point_source_scene_m"][source_index["optimized_action_aligned"]]
    nearest_gripper = distances["nearest_ALOHA_contact_box_point_source_scene_m"][source_index["optimized_action_aligned"]]
    nearest_names = distances["nearest_ALOHA_contact_box_name"][source_index["optimized_action_aligned"]]
    frame_to_local = {int(frame): index for index, frame in enumerate(audit_frames)}
    ordered = audit.event_rows()
    best_hinge = local_audit["support_ring_hinge_articulation_diagnostic"]["best_frame_326"]
    best_hinge_angle = float(best_hinge["support_ring_rotation_about_hinge_x_degrees"])
    best_hinge_gap_m = float(best_hinge["events"]["326"]["gap_m"])

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = (0.72, 0.72, 0.72)
    model.vis.headlight.diffuse[:] = (1.0, 1.0, 1.0)
    model.vis.headlight.specular[:] = (0.18, 0.18, 0.18)
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)

    raw_video = OUT / ".accessory_source_audit_4panel.raw.mp4"
    video = OUT / "accessory_source_audit_4panel.mp4"
    writer = base.open_writer(raw_video, 1920, 360)
    key_rows: dict[int, list[np.ndarray]] = {}
    try:
        for frame in range(FRAME_COUNT):
            action_index = int(lookup[frame])
            event = audit.event_at(frame, ordered)
            raw = raw_panel(raw_files[frame], frame, action_index, event)
            local = frame_to_local.get(frame)
            if local is None:
                gap_m = None
                object_point = None
                gripper_point = None
                nearest_name = None
            else:
                gap_m = float(optimized_gap[local])
                object_point = nearest_object[local]
                gripper_point = nearest_gripper[local]
                nearest_name = str(nearest_names[local])
            overview = render_scene_panel(
                renderer, model, data, "cam_high", qpos[action_index], frame, action_index, event,
                phone_model[frame], accessory_model[frame], initial_phone_model, initial_accessory_model,
                tcp_model[action_index], root_inverse, asset, gap_m, object_point, gripper_point,
                nearest_name, closeup=False, best_hinge_angle=best_hinge_angle, best_hinge_gap_m=best_hinge_gap_m,
            )
            if accessory_model[frame, :3, 3].all() and frame >= 223:
                focus = 0.5 * (accessory_model[frame, :3, 3] + tcp_model[action_index, :3, 3])
            else:
                focus = tcp_model[action_index, :3, 3]
            camera = free_camera(focus, 118.0, -24.0, 0.30)
            closeup = render_scene_panel(
                renderer, model, data, camera, qpos[action_index], frame, action_index, event,
                phone_model[frame], accessory_model[frame], initial_phone_model, initial_accessory_model,
                tcp_model[action_index], root_inverse, asset, gap_m, object_point, gripper_point,
                nearest_name, closeup=True, best_hinge_angle=best_hinge_angle, best_hinge_gap_m=best_hinge_gap_m,
            )
            dashboard = plot_dashboard(frame, audit_frames, optimized_gap, observed_gap, tcp_gap, full_gap, semantics, decision)
            row = [
                cv2.resize(raw, (480, 360), interpolation=cv2.INTER_AREA),
                cv2.resize(overview, (480, 360), interpolation=cv2.INTER_AREA),
                cv2.resize(closeup, (480, 360), interpolation=cv2.INTER_AREA),
                cv2.resize(dashboard, (480, 360), interpolation=cv2.INTER_AREA),
            ]
            writer.write(np.hstack(row))
            if frame in VIDEO_KEY_FRAMES:
                key_rows[frame] = [value.copy() for value in row]
    finally:
        writer.release()

    metadata = {
        "status": decision["status"],
        "panels": ["raw_cam_high", "authoritative_asset_actual_ALOHA_boxes", "contact_proxy_closeup", "five_cause_dashboard"],
        "frame_count": FRAME_COUNT,
        "source_fps": 30.0,
        "output_fps": FPS_REVIEW,
        "action_to_observation_lag_frames": 7,
        "action_sample_for_observed_frame": "max(observed_frame - 7, 0); frames 0-6 are diagnostic pre-command hold",
        "optimized_action_path": str(ACTION.resolve()),
        "optimized_action_sha256": sha256(ACTION),
        "approved_timeline_path": str(TIMELINE.resolve()),
        "approved_timeline_sha256": sha256(TIMELINE),
        "phone_carrier_input": str(PHONE_NPZ.resolve()),
        "phone_carrier_input_sha256": sha256(PHONE_NPZ),
        "phone_carrier_status": "CHARGER_ANCHORED_530 DIAGNOSTIC ONLY FROM OBSERVED FRAME 223; NOT APPROVED FOR G1",
        "stationary_ALOHA_model": str(MODEL_XML.resolve()),
        "stationary_ALOHA_model_sha256": sha256(MODEL_XML),
        "source_accessory_USD": str(ACCESSORY_USD.resolve()),
        "source_accessory_USD_sha256": sha256(ACCESSORY_USD),
        "source_scene_USD": str(SCENE_USD.resolve()),
        "source_scene_USD_sha256": sha256(SCENE_USD),
        "optimized_action_modified": False,
        "approved_event_frames_modified": False,
        "timestamps_modified": False,
        "phone_carrier_modified": False,
        "source_ALOHA_motion_modified": False,
        "G1": False,
        "IK": False,
        "phasewarp": False,
        "orientation_retargeting": False,
        "Dex3_target_motion": False,
        "physics": False,
        "DDS": False,
        "publisher": False,
        "hardware": False,
    }
    base.embed_metadata(raw_video, video, "Episode49 source accessory geometry/contact audit v11c", metadata)
    decoded = base.decoded_frames(video)
    if decoded != FRAME_COUNT:
        raise RuntimeError(f"Expected {FRAME_COUNT} decoded frames, got {decoded}")

    contact_sheet = OUT / "accessory_event_contact_sheet.png"
    save_sheet(contact_sheet, [key_rows[frame] for frame in SHEET_FRAMES], (480, 360))

    # Raw close crops alongside the calibrated-free 3-D diagnostic panels.
    crop_rows: list[list[np.ndarray]] = []
    for frame in SHEET_FRAMES:
        raw = cv2.imread(str(raw_files[frame]), cv2.IMREAD_COLOR)
        crop = raw[145:455, 180:510]
        crop = overlay(crop, [f"RAW crop f{frame} | {audit.event_at(frame, ordered)}", "uncalibrated visual evidence"], [(245, 245, 245), (170, 220, 170)])
        crop_rows.append([crop, key_rows[frame][2], key_rows[frame][3]])
    raw_crop_sheet = OUT / "accessory_raw_vs_proxy_contact_sheet.png"
    save_sheet(raw_crop_sheet, crop_rows, (520, 390))

    gap_plot = OUT / "accessory_gap_timeseries.png"
    draw_gap_plot(
        gap_plot, audit_frames,
        {
            "optimized actual OBB": optimized_gap,
            "observation actual OBB": observed_gap,
            "optimized TCP": tcp_gap,
            "complete-ring lower bound": full_gap,
            "main C-ring": main_gap,
            "support ring": support_gap,
            "hinge": hinge_gap,
        },
        semantics,
    )
    sweep_plot = OUT / "ring_gap_axis_hinge_sweeps.png"
    draw_gap_sweep(sweep_plot, gap_audit, local_audit)
    cause_matrix = OUT / "accessory_five_cause_matrix.png"
    draw_cause_matrix(cause_matrix, decision)

    visual = {
        "status": "SOURCE_ACCESSORY_AUDIT_VISUAL_EVIDENCE_READY",
        "decision_status": decision["status"],
        "video": {
            "path": str(video.resolve()),
            "sha256": sha256(video),
            "decoded_frames": decoded,
            "fps": FPS_REVIEW,
            "metadata": metadata,
        },
        "images": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (contact_sheet, raw_crop_sheet, gap_plot, sweep_plot, cause_matrix)
        },
    }
    base.dump_json(OUT / "visual_evidence_audit.json", visual)
    renderer.close()
    print(json.dumps(visual, indent=2, default=base.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
