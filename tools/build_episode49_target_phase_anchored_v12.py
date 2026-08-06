#!/usr/bin/env python3
"""Build target-side, phase-anchored Episode-49 targets without source object poses.

This program is deliberately limited to immutable ALOHA FK, one common global
task registration, target-scene geometric anchors, and C2 phase-boundary
residuals.  It does not solve G1 IK, load a G1 expert trajectory, fit Dex3, or
step object physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from pxr import Usd, UsdGeom
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

ROOT = Path("/home/jbnu/aloha_g1_dataset")
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab_magsafe_fixed_scene")]

OUT = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_target_phase_anchored_v12"
SRC = ROOT / "evaluation/smolvla_episode49_temporal_consensus/episode_000049_temporal_consensus.npz"
TIMELINE = ROOT / "configs/episode49_task_timeline.approved.json"
ALIGNMENT = ROOT / "configs/episode49_action_observation_alignment.approved.json"
LAYOUT = ROOT / "isaaclab_magsafe_fixed_scene/scene_layout.json"
SCENE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_fixed_scene.usda"
ACTIVE_STAGE = ROOT / "isaaclab_magsafe_fixed_scene/generated/magsafe_g1_model_preview.usda"
REGISTRATION = ROOT / "configs/magsafe_task_frame_registration.sim.json"
PALM_CAL = ROOT / "configs/g1_dex3_palm_frame_calibration.sim.json"
TOOL_CAL = ROOT / "configs/aloha_tool_axes_calibration.sim.json"
V8 = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_restore_original_v8/restored_exact_arm_trajectory.npz"
V11_FK = ROOT / "outputs/scene_registered_retargeting/current_layout_ep49_source_fk_parity_v11/optimized_action_fk.npz"

METHOD = "ALOHA_PRIMARY_TARGET_SIDE_PHASE_ANCHORED_RETARGETING"
FPS = 30.0
SCALE = 0.42
LAG = 7
KNOTS = np.array([0, 169, 193, 216, 319, 322, 334, 373, 523, 579, 639, 695, 989], dtype=int)
R_SCENE_FROM_MODEL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
MODEL_ROOT = np.array([0.0, 0.0, 0.7922728583])
G1_ROOT = np.array([0.44514890950197095, -0.35257022755443246, 0.7922728583])
C_LEFT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
C_RIGHT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

EVENT_PHASES = {
    "left": {
        "L0_start_to_phone_grasp": (0, 169),
        "L1_grasp_to_portrait": (169, 216),
        "L2_portrait_hold": (216, 373),
        "L3_phone_transport_to_charger": (373, 523),
        "L4_release_and_return": (523, 695),
        "L5_terminal_command_hold": (695, 989),
    },
    "right": {
        "R0_start_to_portrait": (0, 216),
        "R1_accessory_approach": (216, 319),
        "R2_accessory_grasp_removal": (319, 334),
        "R3_accessory_hold_transport": (334, 639),
        "R4_terminal_motion": (639, 695),
        "R5_terminal_command_hold": (695, 989),
    },
}

REJECTED = [
    "outputs/right_c_ring_insertion",
    "outputs/scene_registered_retargeting/current_layout_ep49_fingertip_semantic_v3",
    "outputs/scene_registered_retargeting/current_layout_ep49_left_hold_right_c_v4",
    "outputs/scene_registered_retargeting/current_layout_ep49_left_ab_contactframe_v5",
    "outputs/scene_registered_retargeting/current_layout_ep49_left_ab_humanlike_v6",
    "outputs/scene_registered_retargeting/current_layout_ep49_left_ab_reachability_v7",
    "outputs/left_ab_grasp_visual_diagnosis",
]
DIAGNOSTIC_ONLY = [
    "outputs/scene_registered_retargeting/current_layout_ep49_phone_carrier_audit_v11b",
    "outputs/scene_registered_retargeting/current_layout_ep49_accessory_semantics_audit_v11c",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument(
        "--global-yaw-override-deg",
        type=float,
        default=None,
        help="Diagnostic-only selection within the recorded global yaw sweep.",
    )
    return p.parse_args()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    tmp.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def save_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".incomplete")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def make_transform(rotation=np.eye(3), position=np.zeros(3)) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(rotation, float)
    out[:3, 3] = np.asarray(position, float)
    return out


def inv_transform(value: np.ndarray) -> np.ndarray:
    r = value[:3, :3]
    return make_transform(r.T, -r.T @ value[:3, 3])


def quat_rotation_wxyz(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    return Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()


def active_transform(stage: Usd.Stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"missing authoritative USD prim: {path}")
    return np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float
    ).T


def normalize(v: np.ndarray, fallback=None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        if fallback is None:
            raise ValueError("zero vector")
        return normalize(np.asarray(fallback, float))
    return np.asarray(v, float) / n


def minimal_vector_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a, b = normalize(source), normalize(target)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1e-12:
        if dot > 0:
            return np.eye(3)
        axis = normalize(np.cross(a, [1.0, 0.0, 0.0]), np.cross(a, [0.0, 1.0, 0.0]))
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    axis = normalize(cross)
    return Rotation.from_rotvec(math.acos(dot) * axis).as_matrix()


def relative_rotations(rotations: np.ndarray, start: int, calibration: np.ndarray) -> np.ndarray:
    delta = np.einsum("ji,tjk->tik", rotations[start], rotations)
    return np.einsum("ij,tjk,kl->til", calibration.T, delta, calibration)


def phase_progress(position: np.ndarray, rotation: np.ndarray, start: int, end: int) -> np.ndarray:
    p = position[start : end + 1]
    r = rotation[start : end + 1]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    angular = Rotation.from_matrix(np.einsum("tji,tjk->tik", r[:-1], r[1:])).magnitude()
    step = step + 0.04 * angular
    cumulative = np.r_[0.0, np.cumsum(step)]
    if cumulative[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(cumulative))
    return cumulative / cumulative[-1]


def normalized_signal_candidates(signal: np.ndarray, start: int, end: int, closing_low=True):
    segment = np.asarray(signal[start : end + 1], float)
    direction = segment[0] - segment[-1] if closing_low else segment[-1] - segment[0]
    if abs(direction) < 1e-9:
        raw = np.linspace(0.0, 1.0, len(segment))
    else:
        raw = (segment[0] - segment) / direction if closing_low else (segment - segment[0]) / direction
    raw = np.clip(raw, 0.0, 1.0)
    raw[0], raw[-1] = 0.0, 1.0
    kernel = np.ones(7) / 7.0
    padded = np.pad(raw, (3, 3), mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")
    smooth = np.clip((smooth - smooth[0]) / max(smooth[-1] - smooth[0], 1e-9), 0.0, 1.0)
    monotonic = np.maximum.accumulate(smooth)
    monotonic = (monotonic - monotonic[0]) / max(monotonic[-1] - monotonic[0], 1e-9)
    candidates = {
        "signal_normalized": raw,
        "signal_smoothed": smooth,
        "monotonic_projection": monotonic,
    }
    scores = {}
    for name, curve in candidates.items():
        accel = float(np.sum(np.diff(curve, n=2) ** 2))
        timing = float(np.mean((curve - raw) ** 2))
        monotonic_violation = float(np.sum(np.minimum(np.diff(curve), 0.0) ** 2))
        scores[name] = {
            "object_acceleration_energy": accel,
            "source_gripper_timing_mse": timing,
            "monotonic_violation_energy": monotonic_violation,
            "selection_score": accel + 4.0 * timing + 1e4 * monotonic_violation,
        }
    selected = min(scores, key=lambda x: scores[x]["selection_score"])
    return candidates, scores, selected


def smooth5(u: np.ndarray) -> np.ndarray:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def combined_source_progress(left_p, right_p, left_r, right_r) -> np.ndarray:
    step = np.linalg.norm(np.diff(left_p, axis=0), axis=1)
    step += np.linalg.norm(np.diff(right_p, axis=0), axis=1)
    step += 0.04 * Rotation.from_matrix(np.einsum("tji,tjk->tik", left_r[:-1], left_r[1:])).magnitude()
    step += 0.04 * Rotation.from_matrix(np.einsum("tji,tjk->tik", right_r[:-1], right_r[1:])).magnitude()
    return np.r_[0.0, np.cumsum(step)]


def blend_knot_values(values: np.ndarray, source_progress: np.ndarray) -> np.ndarray:
    out = np.empty((990, values.shape[1]), dtype=float)
    for i, (a, b) in enumerate(zip(KNOTS[:-1], KNOTS[1:])):
        u = source_progress[a : b + 1].copy()
        span = u[-1] - u[0]
        if span <= 1e-12:
            u = np.linspace(0.0, 1.0, b - a + 1)
        else:
            u = (u - u[0]) / span
        w = smooth5(u)[:, None]
        out[a : b + 1] = (1.0 - w) * values[i] + w * values[i + 1]
    return out


def solve_residual_knots(anchor_l, anchor_r, weights):
    n = len(KNOTS)
    d1 = np.diff(np.eye(n), axis=0)
    d2 = np.diff(np.eye(n), n=2, axis=0)
    eye = np.eye(n)
    h = weights["magnitude"] * block_diag(eye, eye)
    h += weights["velocity"] * block_diag(d1.T @ d1, d1.T @ d1)
    h += weights["acceleration"] * block_diag(d2.T @ d2, d2.T @ d2)
    common_difference = np.hstack((eye, -eye))
    h += weights["bimanual"] * common_difference.T @ common_difference
    h += 1e-10 * np.eye(2 * n)
    solved = []
    for dim in range(3):
        rows, values = [], []
        for anchors, offset in ((anchor_l, 0), (anchor_r, n)):
            for frame, value in anchors.items():
                idx = np.flatnonzero(KNOTS == int(frame))
                if len(idx) != 1:
                    raise RuntimeError(f"anchor is not an approved knot: {frame}")
                row = np.zeros(2 * n)
                row[offset + int(idx[0])] = 1.0
                rows.append(row)
                values.append(float(value[dim]))
        # Preserve the approved terminal command tail without discarding 983--989.
        for offset in (0, n):
            row = np.zeros(2 * n)
            row[offset + n - 1] = 1.0
            row[offset + n - 2] = -1.0
            rows.append(row)
            values.append(0.0)
        a = np.asarray(rows)
        b = np.asarray(values)
        kkt = np.block([[h, a.T], [a, np.zeros((len(a), len(a)))]] )
        solution = np.linalg.solve(kkt, np.r_[np.zeros(2 * n), b])[: 2 * n]
        solved.append(solution)
    result = np.asarray(solved).T
    return result[:n], result[n:]


def apply_rotvec(base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return np.asarray([Rotation.from_rotvec(rv).as_matrix() @ r for rv, r in zip(residual, base)])


def rotation_residual(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target @ base.T).as_rotvec()


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b, atol=1e-12) else 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def phase_fidelity(base_l, base_r, corrected_l, corrected_r, base_lr, base_rr, corrected_lr, corrected_rr):
    phases = {}
    minimum = {"path_shape": 1.0, "speed": 1.0, "rotation_progress": 1.0}
    major = set()
    for side, definitions in EVENT_PHASES.items():
        for name, (a, b) in definitions.items():
            if b - a >= 15 and "terminal" not in name:
                major.add((side, name))
            x, y = (base_l, corrected_l) if side == "left" else (base_r, corrected_r)
            r0, r1 = (base_lr, corrected_lr) if side == "left" else (base_rr, corrected_rr)
            xb, yb = x[a : b + 1] - x[a], y[a : b + 1] - y[a]
            dx, dy = np.diff(xb, axis=0), np.diff(yb, axis=0)
            speed_x, speed_y = np.linalg.norm(dx, axis=1), np.linalg.norm(dy, axis=1)
            acceleration_x, acceleration_y = np.diff(speed_x), np.diff(speed_y)
            curvature_x = np.linalg.norm(np.diff(dx, axis=0), axis=1)
            curvature_y = np.linalg.norm(np.diff(dy, axis=0), axis=1)
            tangent = np.sum(dx * dy, axis=1) / (np.linalg.norm(dx, axis=1) * np.linalg.norm(dy, axis=1) + 1e-12)
            p0 = Rotation.from_matrix(np.einsum("ji,tjk->tik", r0[a], r0[a : b + 1])).magnitude()
            p1 = Rotation.from_matrix(np.einsum("ji,tjk->tik", r1[a], r1[a : b + 1])).magnitude()
            source_disp = float(np.linalg.norm(x[b] - x[a]))
            residual_disp = float(np.linalg.norm((y[b] - x[b]) - (y[a] - x[a])))
            source_direction = normalize(x[b] - x[a], [1, 0, 0])
            target_direction = normalize(y[b] - y[a], source_direction)
            row = {
                "start_action_index": a,
                "end_action_index": b,
                "normalized_path_shape_correlation": correlation(xb, yb),
                "tangent_direction_cosine": float(np.mean(tangent)) if len(tangent) else 1.0,
                "normalized_speed_profile_correlation": correlation(speed_x, speed_y),
                "acceleration_profile_correlation": correlation(acceleration_x, acceleration_y),
                "curvature_correlation": correlation(curvature_x, curvature_y),
                "relative_rotation_progress_correlation": correlation(p0, p1),
                "phase_displacement_direction_cosine": float(np.dot(source_direction, target_direction)),
                "source_phase_displacement_m": source_disp,
                "residual_displacement_m": residual_disp,
                "residual_source_phase_displacement_ratio": residual_disp / max(source_disp, 1e-9),
                "event_timing_difference_frames": 0,
            }
            phases[name] = row
            if (side, name) in major:
                minimum["path_shape"] = min(minimum["path_shape"], row["normalized_path_shape_correlation"])
                minimum["speed"] = min(minimum["speed"], row["normalized_speed_profile_correlation"])
                if p0[-1] > math.radians(2.0):
                    minimum["rotation_progress"] = min(minimum["rotation_progress"], row["relative_rotation_progress_correlation"])
    bmid, cmid = 0.5 * (base_l + base_r), 0.5 * (corrected_l + corrected_r)
    brel, crel = base_r - base_l, corrected_r - corrected_l
    bimanual = {
        "midpoint_trend_correlation": correlation(np.diff(bmid, axis=0), np.diff(cmid, axis=0)),
        "midpoint_rmse_m": float(np.sqrt(np.mean(((cmid - cmid[0]) - (bmid - bmid[0])) ** 2))),
        "relative_hand_vector_trend_correlation": correlation(np.diff(brel, axis=0), np.diff(crel, axis=0)),
        "relative_hand_vector_rmse_m": float(np.sqrt(np.mean(((crel - crel[0]) - (brel - brel[0])) ** 2))),
        "inter_hand_distance_trend_correlation": correlation(np.linalg.norm(brel, axis=1), np.linalg.norm(crel, axis=1)),
    }
    return phases, bimanual, minimum


def pregrasp_right_fidelity(base_r: np.ndarray, corrected_r: np.ndarray) -> dict[str, float]:
    """Fast fidelity objective for the only free pre-grasp knot coefficients."""
    values = {}
    for label, (a, b) in {"R0": (0, 216), "R1": (216, 319)}.items():
        x = base_r[a : b + 1] - base_r[a]
        y = corrected_r[a : b + 1] - corrected_r[a]
        sx = np.linalg.norm(np.diff(x, axis=0), axis=1)
        sy = np.linalg.norm(np.diff(y, axis=0), axis=1)
        values[f"{label}_path"] = correlation(x, y)
        values[f"{label}_speed"] = correlation(sx, sy)
    values["minimum"] = min(values.values())
    return values


def target_frame_from_axes(position, x_axis, y_axis) -> np.ndarray:
    x = normalize(x_axis)
    y = normalize(y_axis - x * np.dot(x, y_axis))
    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))
    return make_transform(np.column_stack((x, y, z)), position)


def neutral_future_finger_proxy_local(info: dict, data: mujoco.MjData, side: str, finger: str) -> np.ndarray:
    """Read a neutral distal-mesh reach vector from the active G1 model.

    This is a static arm-anchor diagnostic only.  It neither commands nor fits
    a Dex3 joint and is stored separately from the palm target.
    """
    model = info["model"]
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    suffix = "2_link" if finger == "thumb" else "1_link"
    distal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_{finger}_{suffix}")
    wrist_r = data.xmat[wrist_id].reshape(3, 3)
    palm = data.xpos[wrist_id] + wrist_r @ np.array([0.0415, 0.003 if side == "left" else -0.003, 0.0])
    candidates = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != distal_id:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start : start + count]
        geom_r = data.geom_xmat[geom_id].reshape(3, 3)
        world = data.geom_xpos[geom_id] + vertices @ geom_r.T
        local = (world - palm) @ wrist_r
        candidates.append(local[np.argmax(local[:, 0])])
    if not candidates:
        raise RuntimeError(f"active G1 {side}-{finger} distal mesh proxy was not found")
    return max(candidates, key=lambda x: x[0]).copy()


def neutral_future_right_c_proxy_local(info: dict, data: mujoco.MjData) -> np.ndarray:
    return neutral_future_finger_proxy_local(info, data, "right", "middle")


def evaluate_target_phone(base_rot_l, corrected_l, corrected_rot_l, phone_initial, phone_portrait, phone_desired, lock_palm_to_phone, alpha_l):
    phone = np.empty((990, 4, 4))
    for k in range(990):
        if k < 169:
            phone[k] = phone_initial
        elif k <= 216:
            carrier = make_transform(corrected_rot_l[k], corrected_l[k]) @ lock_palm_to_phone
            a = float(alpha_l[k - 169])
            # Translation blending is target-object state only; arm motion is untouched.
            position = (1.0 - a) * phone_initial[:3, 3] + a * carrier[:3, 3]
            rel_source = corrected_rot_l[k] @ corrected_rot_l[169].T
            primary_rotation = rel_source @ phone_initial[:3, :3]
            endpoint_correction = phone_portrait[:3, :3] @ (corrected_rot_l[216] @ corrected_rot_l[169].T @ phone_initial[:3, :3]).T
            correction = Rotation.from_rotvec(a * Rotation.from_matrix(endpoint_correction).as_rotvec()).as_matrix()
            phone[k] = make_transform(correction @ primary_rotation, position)
        elif k < 523:
            phone[k] = make_transform(corrected_rot_l[k], corrected_l[k]) @ lock_palm_to_phone
        else:
            phone[k] = phone_desired
    return phone


def evaluate_target_accessory(phone, corrected_r, corrected_rot_r, phone_to_accessory, alpha_r):
    attached = np.einsum("tij,jk->tik", phone, phone_to_accessory)
    palm319 = make_transform(corrected_rot_r[319], corrected_r[319])
    palm_to_accessory = inv_transform(palm319) @ attached[319]
    accessory = np.empty_like(attached)
    release_pose = None
    for k in range(990):
        if k < 319:
            accessory[k] = attached[k]
        elif k <= 334:
            carrier = make_transform(corrected_rot_r[k], corrected_r[k]) @ palm_to_accessory
            a = float(alpha_r[k - 319])
            p = (1 - a) * attached[k, :3, 3] + a * carrier[:3, 3]
            relative = carrier[:3, :3] @ attached[k, :3, :3].T
            r = Rotation.from_rotvec(a * Rotation.from_matrix(relative).as_rotvec()).as_matrix() @ attached[k, :3, :3]
            accessory[k] = make_transform(r, p)
        elif k < 639:
            accessory[k] = make_transform(corrected_rot_r[k], corrected_r[k]) @ palm_to_accessory
        elif k == 639:
            release_pose = make_transform(corrected_rot_r[k], corrected_r[k]) @ palm_to_accessory
            accessory[k] = release_pose
        else:
            accessory[k] = release_pose
    return accessory, attached, palm_to_accessory


def create_plots(out, base_l, base_r, corrected_l, corrected_r, base_lr, corrected_lr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = [
        ("base_vs_corrected_paths.png", [(base_l[:, 0], base_l[:, 2], "base L"), (corrected_l[:, 0], corrected_l[:, 2], "corrected L"), (base_r[:, 0], base_r[:, 2], "base R"), (corrected_r[:, 0], corrected_r[:, 2], "corrected R")], "scene X", "scene Z"),
        ("speed_profiles.png", [(np.arange(989), np.linalg.norm(np.diff(base_l, axis=0), axis=1), "base L"), (np.arange(989), np.linalg.norm(np.diff(corrected_l, axis=0), axis=1), "corrected L"), (np.arange(989), np.linalg.norm(np.diff(base_r, axis=0), axis=1), "base R"), (np.arange(989), np.linalg.norm(np.diff(corrected_r, axis=0), axis=1), "corrected R")], "action index", "m/sample"),
        ("bimanual_relation.png", [(np.arange(990), np.linalg.norm(base_r - base_l, axis=1), "base distance"), (np.arange(990), np.linalg.norm(corrected_r - corrected_l, axis=1), "corrected distance")], "action index", "inter-hand distance m"),
    ]
    for filename, series, xlabel, ylabel in plots:
        fig, ax = plt.subplots(figsize=(12, 5))
        for x, y, label in series:
            ax.plot(x, y, label=label)
        for knot in KNOTS:
            ax.axvline(knot, color="gray", alpha=0.15)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(np.linalg.norm(np.diff(base_l, n=2, axis=0), axis=1), label="base left")
    ax.plot(np.linalg.norm(np.diff(corrected_l, n=2, axis=0), axis=1), label="corrected left")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "curvature_profiles.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    p0 = Rotation.from_matrix(np.einsum("ji,tjk->tik", base_lr[169], base_lr)).magnitude()
    p1 = Rotation.from_matrix(np.einsum("ji,tjk->tik", corrected_lr[169], corrected_lr)).magnitude()
    ax.plot(p0, label="base left rotation progress")
    ax.plot(p1, label="corrected left rotation progress")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "rotation_progress.png", dpi=180); plt.close(fig)


def main() -> int:
    args = parse_args()
    required = [SRC, TIMELINE, ALIGNMENT, LAYOUT, SCENE, ACTIVE_STAGE, REGISTRATION, PALM_CAL, TOOL_CAL, V8, V11_FK]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)
    backup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = OUT / "backups" / backup_stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    prior = [p for p in OUT.iterdir() if p.name != "backups"]
    backed_up = []
    for path in prior:
        target = backup_dir / path.name
        if path.is_file():
            shutil.copy2(path, target); backed_up.append(str(target))
    dump(backup_dir / "backup_manifest.json", {
        "created_utc": backup_stamp,
        "preexisting_v12_files": [str(p) for p in prior],
        "backed_up_files": backed_up,
        "note": "New-output run; only pre-existing regular files are copied before replacement.",
    })

    scene_hash_before = {str(p.resolve()): sha(p) for p in (LAYOUT, SCENE, ACTIVE_STAGE, REGISTRATION)}
    source_hash = sha(SRC)
    timeline_hash_before = sha(TIMELINE)
    alignment_hash = sha(ALIGNMENT)
    with np.load(SRC, allow_pickle=False) as z:
        action = z["optimized_action"].copy()
        timestamps = z["timestamp"].copy()
        source_fps = float(z["fps"])
    if action.shape != (990, 14) or not np.isfinite(action).all():
        raise RuntimeError("optimized_action is not finite [990,14]")
    if timestamps.shape != (990,) or source_fps != FPS:
        raise RuntimeError("source timestamp/fps invariant failed")

    timeline = json.loads(TIMELINE.read_text())
    alignment = json.loads(ALIGNMENT.read_text())
    observed_events = {e["event"]: int(e["frame"]) for e in timeline["events"]}
    event_mapping = {}
    for event, frame in observed_events.items():
        action_index = frame - LAG
        if alignment["event_mapping"][event]["aligned_action_index"] != action_index:
            raise RuntimeError(f"alignment mismatch for {event}")
        event_mapping[event] = {"observed_event_frame": frame, "aligned_action_index": action_index}
    expected_knots = sorted(set([0, 989] + [x["aligned_action_index"] for x in event_mapping.values()]))
    if expected_knots != KNOTS.tolist():
        raise RuntimeError(f"approved knot mismatch: {expected_knots} != {KNOTS.tolist()}")

    import retarget_episode49_optimized_action_to_g1 as core
    aloha_model, _ = core.aloha.load_validated_model(core.ALOHA_XML)
    aloha_qpos, clipped = core.aloha.mapped_qpos(action)
    fk = core.aloha.fk(aloha_model, aloha_qpos)
    left_source_p = fk["left_position_m"]
    right_source_p = fk["right_position_m"]
    left_source_r = Rotation.from_quat(fk["left_quaternion_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
    right_source_r = Rotation.from_quat(fk["right_quaternion_wxyz"][:, [1, 2, 3, 0]]).as_matrix()
    with np.load(V11_FK, allow_pickle=False) as z:
        if not np.array_equal(z["source_joint_array"], action):
            raise RuntimeError("v11 verified FK action mismatch")
        v11_left = z["left_tcp_position_model"].copy()
        v11_right = z["right_tcp_position_model"].copy()
    fk_crosscheck = max(float(np.max(np.abs(v11_left - left_source_p))), float(np.max(np.abs(v11_right - right_source_p))))
    if fk_crosscheck > 1e-12:
        raise RuntimeError(f"verified ALOHA FK cross-check failed: {fk_crosscheck}")

    left_velocity = np.gradient(left_source_p, 1.0 / FPS, axis=0)
    right_velocity = np.gradient(right_source_p, 1.0 / FPS, axis=0)
    left_angular_velocity = np.zeros((990, 3)); right_angular_velocity = np.zeros((990, 3))
    left_angular_velocity[1:] = Rotation.from_matrix(np.einsum("tji,tjk->tik", left_source_r[:-1], left_source_r[1:])).as_rotvec() * FPS
    right_angular_velocity[1:] = Rotation.from_matrix(np.einsum("tji,tjk->tik", right_source_r[:-1], right_source_r[1:])).as_rotvec() * FPS
    save_npz(OUT / "aloha_fk_source.npz", optimized_action=action, timestamps=timestamps, fps=np.array(FPS), action_index=np.arange(990), left_tcp_position=left_source_p, right_tcp_position=right_source_p, left_tcp_rotation=left_source_r, right_tcp_rotation=right_source_r, left_linear_velocity=left_velocity, right_linear_velocity=right_velocity, left_angular_velocity=left_angular_velocity, right_angular_velocity=right_angular_velocity, left_gripper=action[:, 6], right_gripper=action[:, 13], tcp_offset_local_m=np.array([0.1487, 0.0, -0.00105]), action_to_observation_lag_frames=np.array(LAG), method=np.array(METHOD), real_robot_command_allowed=np.array(False))

    library_npz = {
        "optimized_action": action,
        "timestamps": timestamps,
        "action_index": np.arange(990),
        "left_tcp_position": left_source_p,
        "right_tcp_position": right_source_p,
        "left_tcp_rotation": left_source_r,
        "right_tcp_rotation": right_source_r,
        "left_linear_velocity": left_velocity,
        "right_linear_velocity": right_velocity,
        "left_angular_velocity": left_angular_velocity,
        "right_angular_velocity": right_angular_velocity,
        "left_gripper": action[:, 6],
        "right_gripper": action[:, 13],
    }
    library_json = {"method": METHOD, "source": str(SRC.resolve()), "source_sha256": source_hash, "workspace_scale": SCALE, "phases": {}}
    for side, phases in EVENT_PHASES.items():
        p = left_source_p if side == "left" else right_source_p
        r = left_source_r if side == "left" else right_source_r
        for name, (start, end) in phases.items():
            rel_p = p[start : end + 1] - p[start]
            rel_r = np.einsum("ji,tjk->tik", r[start], r[start : end + 1])
            progress = phase_progress(p, r, start, end)
            key = name.lower()
            library_npz[f"{key}_relative_translation"] = rel_p
            library_npz[f"{key}_relative_rotation"] = rel_r
            library_npz[f"{key}_normalized_progress"] = progress
            displacement = p[end] - p[start]
            library_json["phases"][name] = {
                "side": side, "start_action_index": start, "end_action_index": end,
                "sample_count": end - start + 1,
                "phase_displacement_m": float(np.linalg.norm(displacement)),
                "phase_displacement_vector_m": displacement.tolist(),
                "arc_length_m": float(np.sum(np.linalg.norm(np.diff(p[start : end + 1], axis=0), axis=1))),
                "rotation_progress_rad": float(Rotation.from_matrix(rel_r[-1]).magnitude()),
            }
    midpoint = 0.5 * (left_source_p + right_source_p)
    relative_vector = right_source_p - left_source_p
    library_npz["bimanual_midpoint"] = midpoint
    library_npz["right_minus_left_relative_vector"] = relative_vector
    save_npz(OUT / "aloha_phase_motion_library.npz", **library_npz)
    dump(OUT / "aloha_phase_motion_library.json", library_json)

    with np.load(V8, allow_pickle=False) as z:
        if not np.array_equal(z["optimized_action"], action):
            raise RuntimeError("v8 base uses a different action")
        if not np.array_equal(z["source_timestamp"], timestamps):
            raise RuntimeError("v8 base timestamps differ")
        base_model_l = z["base_target_left_position"].copy()
        base_model_r = z["base_target_right_position"].copy()
        warm_q = z["g1_arm_joint_trajectory"].copy()
        arm_joint_names = z["arm_joint_names"].copy()

    info = core.ik.validate_model(core.G1_XML)
    data = mujoco.MjData(info["model"])
    initial_state = core.frame_state(info, data, warm_q[0])
    future_right_c_proxy_local = neutral_future_right_c_proxy_local(info, data)
    future_right_c_reach_m = float(np.linalg.norm(future_right_c_proxy_local))
    future_left_a_proxy_local = neutral_future_finger_proxy_local(info, data, "left", "thumb")
    future_left_b_proxy_local = neutral_future_finger_proxy_local(info, data, "left", "index")
    future_left_ab_midpoint_local = 0.5 * (future_left_a_proxy_local + future_left_b_proxy_local)
    initial_l_rot = quat_rotation_wxyz(initial_state["left_quat"])
    initial_r_rot = quat_rotation_wxyz(initial_state["right_quat"])
    mapped_l = relative_rotations(left_source_r, 0, C_LEFT)
    mapped_r = relative_rotations(right_source_r, 0, C_RIGHT)
    base_model_lr = np.einsum("ij,tjk->tik", initial_l_rot, mapped_l)
    base_model_rr = np.einsum("ij,tjk->tik", initial_r_rot, mapped_r)

    stage = Usd.Stage.Open(str(ACTIVE_STAGE))
    if stage is None:
        raise RuntimeError(f"cannot open active stage {ACTIVE_STAGE}")
    phone_initial = active_transform(stage, "/World/MagSafeScene/Phone")
    accessory_initial = active_transform(stage, "/World/MagSafeScene/Accessory")
    charger_root = active_transform(stage, "/World/MagSafeScene/Charger")
    pad_face_asset = active_transform(stage, "/World/MagSafeScene/Charger/Visuals/PadFace")
    active_g1 = active_transform(stage, "/World/G1")
    pad_center = pad_face_asset[:3, 3]
    pad_tangent_vertical = normalize(pad_face_asset[:3, 1])
    pad_outward_normal = normalize(pad_face_asset[:3, 2])
    desired_phone_rotation = np.column_stack((pad_tangent_vertical, -pad_outward_normal, np.cross(pad_tangent_vertical, -pad_outward_normal)))
    if np.linalg.det(desired_phone_rotation) < 0:
        desired_phone_rotation[:, 2] *= -1
    desired_phone = make_transform(desired_phone_rotation, pad_center)
    layout = json.loads(LAYOUT.read_text())
    phone_dims = np.asarray(layout["phone"]["size_landscape_xyz"], float)
    phone_left_surface = phone_initial[:3, 3] + phone_initial[:3, :3] @ np.array([-0.5 * phone_dims[0], 0.0, 0.0])

    # The position-only palm landmark must be an embodiment-reachable carrier
    # pose, not the result of rotating a neutral Dex3 vertex by an unconstrained
    # ALOHA wrist orientation.  The latter placed the palm on the far side of
    # the phone and outside the G1 arm workspace.  Use the active-model neutral
    # A/B reach-envelope length and place the palm on the line from the phone's
    # authoritative left surface toward the active G1 left shoulder.  This is
    # arm/contact-proxy registration only: no finger joint is moved or fitted.
    left_shoulder_id = mujoco.mj_name2id(
        info["model"], mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link"
    )
    right_shoulder_id = mujoco.mj_name2id(
        info["model"], mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link"
    )
    core.ik.assign_arm_qpos(data, info["stand_qpos"], info["arm_qpos_ids"], info["stand_arm_q"])
    mujoco.mj_forward(info["model"], data)
    left_shoulder_model = data.xpos[left_shoulder_id].copy()
    right_shoulder_model = data.xpos[right_shoulder_id].copy()
    left_shoulder_scene = R_SCENE_FROM_MODEL @ (left_shoulder_model - MODEL_ROOT) + G1_ROOT
    right_shoulder_scene = R_SCENE_FROM_MODEL @ (right_shoulder_model - MODEL_ROOT) + G1_ROOT
    left_elbow_id = mujoco.mj_name2id(info["model"], mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    left_wrist_id = mujoco.mj_name2id(info["model"], mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    active_arm_chain_length_m = (
        float(np.linalg.norm(data.xpos[left_elbow_id] - data.xpos[left_shoulder_id]))
        + float(np.linalg.norm(data.xpos[left_wrist_id] - data.xpos[left_elbow_id]))
        + float(np.linalg.norm(np.array([0.0415, 0.003, 0.0])))
    )
    active_arm_reach_gate_m = active_arm_chain_length_m - 0.010
    future_left_ab_reach_m = float(np.linalg.norm(future_left_ab_midpoint_local))
    phone_surface_to_shoulder = normalize(left_shoulder_scene - phone_left_surface)
    phone_palm_anchor = phone_left_surface + future_left_ab_reach_m * phone_surface_to_shoulder

    mapped_phone_rotation = C_LEFT.T @ (left_source_r[169].T @ left_source_r[216]) @ C_LEFT
    raw_portrait_rotation = phone_initial[:3, :3] @ mapped_phone_rotation
    portrait_options = []
    for sign in (1.0, -1.0):
        correction = minimal_vector_alignment(raw_portrait_rotation[:, 0], np.array([0.0, 0.0, sign]))
        final = correction @ raw_portrait_rotation
        angle = float(Rotation.from_matrix(correction).magnitude())
        accessible = float(np.dot(final[:, 1], np.array([0.0, 1.0, 0.0])))
        score = angle + (0.0 if accessible >= 0.0 else math.pi)
        portrait_options.append((score, correction, final, sign, accessible))
    _, portrait_axis_correction, portrait_rotation, portrait_sign, portrait_accessibility = min(portrait_options, key=lambda x: x[0])

    # Fit the single allowed global task rotation before any phase residual.
    # Translation always pins the left action-169 palm landmark exactly.  A
    # bounded yaw sweep then minimizes the two remaining target-side endpoint
    # corrections (left charger and moving right accessory) while retaining
    # the same rotation for both hands and every sample.  This removes common
    # layout mismatch; it is explicitly excluded from deformation metrics.
    phone_half_diagonal_m = float(np.linalg.norm(0.5 * phone_dims))
    phone_carrier_center_max_m = phone_half_diagonal_m + future_left_ab_reach_m
    global_registration_rows = []
    ring_radius = 0.5 * float(layout["accessory"]["main_outer_diameter"])

    def evaluate_global_registration(roll_deg, pitch_deg, yaw_deg, sweep_stage):
        task_rotation = Rotation.from_euler(
            "xyz", [float(roll_deg), float(pitch_deg), float(yaw_deg)], degrees=True
        ).as_matrix()
        candidate_rotation = task_rotation @ R_SCENE_FROM_MODEL
        candidate_translation = phone_palm_anchor - candidate_rotation @ base_model_l[169]
        candidate_l = (candidate_rotation @ base_model_l.T).T + candidate_translation
        candidate_r = (candidate_rotation @ base_model_r.T).T + candidate_translation
        candidate_lr = np.einsum("ij,tjk->tik", candidate_rotation, base_model_lr)
        candidate_rr = np.einsum("ij,tjk->tik", candidate_rotation, base_model_rr)
        candidate_palm216 = make_transform(candidate_lr[216], candidate_l[216])
        candidate_lock_rotation = candidate_lr[216].T @ portrait_rotation
        candidate_palm523_rotation = desired_phone_rotation @ candidate_lock_rotation.T
        # Pick the closest feasible action-523 carrier point to the immutable
        # globally registered ALOHA base.  Feasibility is the intersection of
        # the active arm reach sphere and the physical phone-box plus neutral
        # A/B reach envelope.  Alternating convex projections are deterministic
        # and add no waypoint or per-frame target.
        candidate_palm523_position = candidate_l[523].copy()
        for _ in range(32):
            shoulder_delta = candidate_palm523_position - left_shoulder_scene
            shoulder_norm = float(np.linalg.norm(shoulder_delta))
            if shoulder_norm > active_arm_reach_gate_m:
                candidate_palm523_position = left_shoulder_scene + shoulder_delta * (active_arm_reach_gate_m / shoulder_norm)
            phone_delta = candidate_palm523_position - pad_center
            phone_norm = float(np.linalg.norm(phone_delta))
            if phone_norm > phone_carrier_center_max_m:
                candidate_palm523_position = pad_center + phone_delta * (phone_carrier_center_max_m / phone_norm)
        candidate_lock_translation = candidate_palm523_rotation.T @ (pad_center - candidate_palm523_position)
        candidate_lock = make_transform(candidate_lock_rotation, candidate_lock_translation)
        candidate_phone319 = make_transform(candidate_lr[319], candidate_l[319]) @ candidate_lock
        candidate_accessory319 = candidate_phone319 @ make_transform(np.eye(3), np.array([0.0, 0.006425, 0.0]))
        center319 = candidate_accessory319[:3, 3]
        approach319 = normalize(candidate_r[319] - candidate_r[216], [0, 1, 0])
        normal319 = normalize(candidate_accessory319[:3, 1])
        plane319 = approach319 - normal319 * np.dot(approach319, normal319)
        if np.linalg.norm(plane319) < 1e-9:
            plane319 = candidate_accessory319[:3, 0]
        plane319 = normalize(plane319)
        surface319 = center319 - ring_radius * plane319
        direction319 = normalize(surface319 - candidate_r[319], plane319)
        candidate_right_anchor = surface319 - future_right_c_reach_m * direction319
        candidate_right_contact_residual = candidate_right_anchor - candidate_r[319]
        # R3 keeps the same phase-boundary residual while preserving the
        # immutable ALOHA hold/transport curve.  Audit that whole phase, not
        # only its grasp endpoint, against the active right-arm reach sphere.
        candidate_right_hold = candidate_r[334:640] + candidate_right_contact_residual
        right_hold_distance = np.linalg.norm(candidate_right_hold - right_shoulder_scene, axis=1)
        right_hold_max_reach = float(np.max(right_hold_distance))
        right_hold_max_action = int(334 + np.argmax(right_hold_distance))
        left_residual_norm = float(np.linalg.norm(candidate_palm523_position - candidate_l[523]))
        right_residual_norm = float(np.linalg.norm(candidate_right_anchor - candidate_r[319]))
        left_reach = float(np.linalg.norm(candidate_palm523_position - left_shoulder_scene))
        right_reach = float(np.linalg.norm(candidate_right_anchor - right_shoulder_scene))
        # The active neutral shoulder-to-palm length is about 0.40 m.  Reach
        # excess is a feasibility penalty, not a change to either trajectory.
        reach_excess = (
            max(0.0, left_reach - active_arm_reach_gate_m) ** 2
            + max(0.0, right_reach - active_arm_reach_gate_m) ** 2
        )
        hold_reach_excess = max(0.0, right_hold_max_reach - active_arm_reach_gate_m) ** 2
        objective = (
            (left_residual_norm / max(np.linalg.norm(candidate_l[523] - candidate_l[373]), 1e-6)) ** 2
            + (right_residual_norm / max(np.linalg.norm(candidate_r[319] - candidate_r[216]), 1e-6)) ** 2
            + 250.0 * reach_excess
            + 10000.0 * hold_reach_excess
            + 1e-4 * (
                math.radians(float(roll_deg)) ** 2
                + math.radians(float(pitch_deg)) ** 2
                + math.radians(float(yaw_deg)) ** 2
            )
        )
        row = {
            "sweep_stage": sweep_stage,
            "roll_deg": float(roll_deg),
            "pitch_deg": float(pitch_deg),
            "yaw_deg": float(yaw_deg),
            "objective": float(objective),
            "left_charger_residual_m": left_residual_norm,
            "right_accessory_residual_m": right_residual_norm,
            "left_anchor_shoulder_distance_m": left_reach,
            "right_anchor_shoulder_distance_m": right_reach,
            "right_hold_max_shoulder_distance_m": right_hold_max_reach,
            "right_hold_max_action_index": right_hold_max_action,
            "left_carrier_phone_center_distance_m": float(np.linalg.norm(candidate_palm523_position - pad_center)),
        }
        candidate = (
            objective, candidate_rotation, candidate_translation,
            candidate_l, candidate_r, candidate_lr, candidate_rr,
            candidate_lock, candidate_palm523_rotation,
            candidate_palm523_position, roll_deg, pitch_deg, yaw_deg,
        )
        return row, candidate

    yaw_values = (
        [float(args.global_yaw_override_deg)]
        if args.global_yaw_override_deg is not None
        else np.arange(-75.0, 75.0001, 2.5)
    )
    candidates = []
    for roll_deg in np.arange(-25.0, 25.0001, 5.0):
        for pitch_deg in np.arange(-25.0, 25.0001, 5.0):
            for yaw_deg in yaw_values:
                row, candidate = evaluate_global_registration(roll_deg, pitch_deg, yaw_deg, "coarse")
                global_registration_rows.append(row)
                candidates.append(candidate)
    coarse_best = min(candidates, key=lambda value: value[0])
    coarse_roll, coarse_pitch, coarse_yaw = map(float, coarse_best[-3:])
    refine_yaws = [coarse_yaw] if args.global_yaw_override_deg is not None else np.arange(coarse_yaw - 4.0, coarse_yaw + 4.0001, 1.0)
    for roll_deg in np.arange(coarse_roll - 4.0, coarse_roll + 4.0001, 1.0):
        for pitch_deg in np.arange(coarse_pitch - 4.0, coarse_pitch + 4.0001, 1.0):
            for yaw_deg in refine_yaws:
                row, candidate = evaluate_global_registration(roll_deg, pitch_deg, yaw_deg, "refine")
                global_registration_rows.append(row)
                candidates.append(candidate)
    best_registration = min(candidates, key=lambda value: value[0])
    (
        _, global_rotation, global_translation, base_l, base_r, base_lr,
        base_rr, lock_palm_to_phone, palm523_rotation, palm523_position,
        selected_global_roll_deg, selected_global_pitch_deg, selected_global_yaw_deg,
    ) = best_registration
    if np.linalg.norm(base_l[169] - phone_palm_anchor) > 1e-12:
        raise RuntimeError("global landmark registration failed")

    # The 169--216 interval is explicitly grasp acquisition, not a verified
    # rigid carrier.  Preserve its ALOHA-derived palm displacement exactly and
    # define the embodiment-specific lock only at the endpoint.  The fixed
    # lock is chosen from the action-523 target phone-on-pad pose and the active
    # G1 reach envelope; it is not a recovered source carrier transform.
    left_portrait_contact = phone_palm_anchor + (base_l[216] - base_l[169])
    palm216 = make_transform(base_lr[216], left_portrait_contact)
    phone_portrait = palm216 @ lock_palm_to_phone
    portrait_phone_center = phone_portrait[:3, 3]
    carrier_center_offset_m = float(np.linalg.norm(pad_center - palm523_position))
    palm523 = make_transform(palm523_rotation, palm523_position)
    if np.linalg.norm((palm523 @ lock_palm_to_phone)[:3, 3] - pad_center) > 1e-12:
        raise RuntimeError("target-side charger carrier construction failed")

    # Target gripper acquisition curves affect object state only.
    left_curves, left_curve_scores, left_curve_name = normalized_signal_candidates(action[:, 6], 169, 216)
    right_curves, right_curve_scores, right_curve_name = normalized_signal_candidates(action[:, 13], 319, 334)
    alpha_l = left_curves[left_curve_name]
    alpha_r = right_curves[right_curve_name]

    weights_grid = {
        "VERY_STRONG_ALOHA": {"magnitude": 14.0, "velocity": 220.0, "acceleration": 1800.0, "bimanual": 55.0},
        "STRONG_ALOHA": {"magnitude": 6.0, "velocity": 120.0, "acceleration": 850.0, "bimanual": 28.0},
        "BALANCED": {"magnitude": 2.0, "velocity": 55.0, "acceleration": 320.0, "bimanual": 10.0},
    }
    source_progress = combined_source_progress(left_source_p, right_source_p, left_source_r, right_source_r)
    left_charger_residual = palm523[:3, 3] - base_l[523]
    anchor_l_position = {
        0: np.zeros(3),
        169: phone_palm_anchor - base_l[169],
        216: left_portrait_contact - base_l[216],
        319: left_portrait_contact - base_l[216],
        322: left_portrait_contact - base_l[216],
        334: left_portrait_contact - base_l[216],
        373: left_portrait_contact - base_l[216],
        523: left_charger_residual,
        # Constant residual through release/return and the terminal command
        # tail preserves the immutable phase-relative curve exactly.
        695: left_charger_residual,
    }
    left_charger_rotation_residual = rotation_residual(palm523[:3, :3], base_lr[523])
    anchor_l_rotation = {
        0: np.zeros(3), 169: np.zeros(3), 193: np.zeros(3),
        216: np.zeros(3), 319: np.zeros(3), 322: np.zeros(3),
        334: np.zeros(3), 373: np.zeros(3),
        523: left_charger_rotation_residual,
        579: left_charger_rotation_residual,
        639: left_charger_rotation_residual,
        695: left_charger_rotation_residual,
    }

    # Coupled target-side phone/accessory anchor registration.  STRONG_ALOHA is
    # used only to converge the shared anchor, then every candidate receives the
    # exact same frozen anchor values.
    right_anchor319 = base_r[319].copy()
    right_anchor334 = right_anchor319 + (base_r[334] - base_r[319])
    coupled_iterations = []
    canonical = weights_grid["STRONG_ALOHA"]
    for iteration in range(10):
        right_contact_residual = right_anchor319 - base_r[319]
        anchor_r_position = {
            0: np.zeros(3),
            319: right_contact_residual,
            322: right_contact_residual,
            334: right_anchor334 - base_r[334],
            639: right_contact_residual,
            695: right_contact_residual,
        }
        kl, kr = solve_residual_knots(anchor_l_position, anchor_r_position, canonical)
        res_l, res_r = blend_knot_values(kl, source_progress), blend_knot_values(kr, source_progress)
        current_l, current_r = base_l + res_l, base_r + res_r
        rkl, rkr = solve_residual_knots(anchor_l_rotation, {}, canonical)
        current_lr = apply_rotvec(base_lr, blend_knot_values(rkl, source_progress))
        current_rr = base_rr.copy()
        canonical_phone = evaluate_target_phone(base_lr, current_l, current_lr, phone_initial, phone_portrait, desired_phone, lock_palm_to_phone, alpha_l)
        accessory319 = canonical_phone[319] @ make_transform(np.eye(3), np.array([0.0, 0.006425, 0.0]))
        center = accessory319[:3, 3]
        approach = normalize(base_r[319] - base_r[216], [0, 1, 0])
        ring_normal = normalize(accessory319[:3, 1])
        approach_plane = approach - ring_normal * np.dot(approach, ring_normal)
        if np.linalg.norm(approach_plane) < 1e-9:
            approach_plane = accessory319[:3, 0]
        approach_plane = normalize(approach_plane)
        ring_surface319 = center - 0.5 * float(layout["accessory"]["main_outer_diameter"]) * approach_plane
        # Position-only anchors the palm at the nearest source-preserving pose
        # from which the *static neutral reach envelope* of future right-C can
        # meet the ring.  No Dex3 joints are moved or optimized here.
        direction_to_ring = normalize(ring_surface319 - base_r[319], approach_plane)
        new319 = ring_surface319 - future_right_c_reach_m * direction_to_ring
        new334 = new319 + (base_r[334] - base_r[319])
        update = max(float(np.linalg.norm(new319 - right_anchor319)), float(np.linalg.norm(new334 - right_anchor334)))
        coupled_iterations.append({"iteration": iteration + 1, "right_anchor_update_m": update, "phone_at_319_m": canonical_phone[319, :3, 3].tolist(), "accessory_center_at_319_m": center.tolist()})
        right_anchor319, right_anchor334 = new319, new334
        if update < 0.0005:
            break
    coupled_converged = coupled_iterations[-1]["right_anchor_update_m"] < 0.0005
    right_contact_residual = right_anchor319 - base_r[319]
    anchor_r_position = {
        0: np.zeros(3),
        319: right_contact_residual,
        322: right_contact_residual,
        334: right_anchor334 - base_r[334],
        639: right_contact_residual,
        695: right_contact_residual,
    }

    # Task-axis right orientation uses target ring normal and source approach;
    # it remains separate from position-only IK.
    approach_x = normalize(ring_surface319 - right_anchor319, [0, 1, 0])
    ring_normal = normalize(canonical_phone[319, :3, :3] @ np.array([0.0, 1.0, 0.0]))
    y_axis = normalize(np.cross(ring_normal, approach_x), [1, 0, 0])
    approach_x = normalize(np.cross(y_axis, ring_normal))
    right319_rotation = np.column_stack((approach_x, y_axis, ring_normal))
    mapped_right_removal = C_RIGHT.T @ (right_source_r[319].T @ right_source_r[334]) @ C_RIGHT
    right334_rotation = right319_rotation @ mapped_right_removal
    anchor_r_rotation = {319: rotation_residual(right319_rotation, base_rr[319]), 334: rotation_residual(right334_rotation, base_rr[334])}

    results, arrays = {}, {}
    for name, weights in weights_grid.items():
        kl, kr = solve_residual_knots(anchor_l_position, anchor_r_position, weights)
        rkl, rkr = solve_residual_knots(anchor_l_rotation, anchor_r_rotation, weights)
        residual_l = blend_knot_values(kl, source_progress)
        # Optimize only free, approved-boundary coefficients.  The search is
        # monotone along the minimum endpoint-correction direction and ranks
        # phase-relative path/speed fidelity before correction energy.  It is
        # not a waypoint trajectory and introduces no intermediate knot.
        endpoint = anchor_r_position[319]
        free_indices = [int(np.flatnonzero(KNOTS == k)[0]) for k in (169, 193, 216)]
        search_rows = []
        grid_values = np.linspace(0.0, 1.0, 21)
        for f169 in grid_values:
            for f193 in grid_values[grid_values >= f169 - 1e-12]:
                for f216 in grid_values[grid_values >= f193 - 1e-12]:
                    trial_knots = kr.copy()
                    trial_knots[free_indices] = np.asarray([f169, f193, f216])[:, None] * endpoint
                    trial_residual = blend_knot_values(trial_knots, source_progress)
                    trial_corrected = base_r + trial_residual
                    fidelity = pregrasp_right_fidelity(base_r, trial_corrected)
                    magnitude = float(np.sum(trial_residual**2))
                    velocity = float(np.sum(np.diff(trial_residual, axis=0)**2))
                    acceleration = float(np.sum(np.diff(trial_residual, n=2, axis=0)**2))
                    weighted_energy = weights["magnitude"] * magnitude + weights["velocity"] * velocity + weights["acceleration"] * acceleration
                    search_rows.append((fidelity["minimum"], -weighted_energy, (float(f169), float(f193), float(f216)), trial_knots, trial_residual, fidelity))
        valid_refinements = [row for row in search_rows if row[0] >= 0.90]
        refinement_pool = valid_refinements if valid_refinements else search_rows
        refinement = max(refinement_pool, key=lambda row: (row[0], row[1]))
        _, _, free_fractions, kr, residual_r, pregrasp_fidelity = refinement
        rotation_res_l, rotation_res_r = blend_knot_values(rkl, source_progress), blend_knot_values(rkr, source_progress)
        corrected_l, corrected_r = base_l + residual_l, base_r + residual_r
        task_axis_lr, task_axis_rr = apply_rotvec(base_lr, rotation_res_l), apply_rotvec(base_rr, rotation_res_r)
        # Position-only/O1 retains the mapped ALOHA relative rotation exactly.
        # Task-axis rotations are staged separately and are not silently folded
        # into the primary candidate before position gates pass.
        corrected_lr, corrected_rr = base_lr.copy(), base_rr.copy()
        phases, bimanual, minimum = phase_fidelity(base_l, base_r, corrected_l, corrected_r, base_lr, base_rr, corrected_lr, corrected_rr)
        anchor_errors = {
            "left_action_169_m": float(np.linalg.norm(corrected_l[169] - phone_palm_anchor)),
            "left_action_216_m": float(np.linalg.norm(corrected_l[216] - left_portrait_contact)),
            "right_action_319_m": float(np.linalg.norm(corrected_r[319] - right_anchor319)),
            "right_action_334_m": float(np.linalg.norm(corrected_r[334] - right_anchor334)),
            "left_action_523_m": float(np.linalg.norm(corrected_l[523] - palm523[:3, 3])),
        }
        energy = {
            "magnitude": float(np.sum(residual_l**2) + np.sum(residual_r**2)),
            "velocity": float(np.sum(np.diff(residual_l, axis=0)**2) + np.sum(np.diff(residual_r, axis=0)**2)),
            "acceleration": float(np.sum(np.diff(residual_l, n=2, axis=0)**2) + np.sum(np.diff(residual_r, n=2, axis=0)**2)),
        }
        anchor_valid = max(anchor_errors.values()) <= 1e-8
        fidelity_valid = minimum["path_shape"] >= 0.90 and minimum["speed"] >= 0.90 and minimum["rotation_progress"] >= 0.90
        results[name] = {"anchor_valid": anchor_valid, "fidelity_valid": fidelity_valid, "anchor_errors": anchor_errors, "minimum_major_phase_fidelity": minimum, "bimanual": bimanual, "residual_energy": energy, "total_residual_energy": sum(energy.values()), "free_pregrasp_knot_optimization": {"knot_action_indices": [169, 193, 216], "fractions_of_required_action319_residual": free_fractions, "monotone": True, "search_candidates": len(search_rows), "threshold_valid_candidates": len(valid_refinements), "pregrasp_fidelity": pregrasp_fidelity}, "phases": phases}
        arrays[name] = (corrected_l, corrected_r, corrected_lr, corrected_rr, task_axis_lr, task_axis_rr, residual_l, residual_r, rotation_res_l, rotation_res_r, kl, kr, rkl, rkr)

    anchor_valid_names = [name for name, row in results.items() if row["anchor_valid"]]
    if not anchor_valid_names:
        raise RuntimeError("BLOCKED_PHASE_RESIDUAL: no anchor-valid candidate")
    fidelity_valid_names = [name for name in anchor_valid_names if results[name]["fidelity_valid"]]
    pool = fidelity_valid_names if fidelity_valid_names else anchor_valid_names
    selected = max(pool, key=lambda name: (results[name]["minimum_major_phase_fidelity"]["path_shape"], results[name]["minimum_major_phase_fidelity"]["speed"], results[name]["minimum_major_phase_fidelity"]["rotation_progress"], -results[name]["total_residual_energy"]))
    corrected_l, corrected_r, corrected_lr, corrected_rr, task_axis_lr, task_axis_rr, residual_l, residual_r, rotation_res_l, rotation_res_r, kl, kr, rkl, rkr = arrays[selected]

    phone_trajectory = evaluate_target_phone(base_lr, corrected_l, task_axis_lr, phone_initial, phone_portrait, desired_phone, lock_palm_to_phone, alpha_l)
    phone_to_accessory = make_transform(np.eye(3), np.array([0.0, 0.006425, 0.0]))
    accessory_trajectory, accessory_attached, palm_to_accessory = evaluate_target_accessory(phone_trajectory, corrected_r, task_axis_rr, phone_to_accessory, alpha_r)

    common_translation = 0.5 * (residual_l + residual_r)
    left_specific = residual_l - common_translation
    right_specific = residual_r - common_translation
    common_rotation = 0.5 * (rotation_res_l + rotation_res_r)
    save_npz(OUT / "globally_registered_base_targets.npz", optimized_action=action, timestamps=timestamps, base_aloha_derived_left_position_model=base_model_l, base_aloha_derived_right_position_model=base_model_r, global_registration_rotation=global_rotation, global_registration_translation=global_translation, global_workspace_scale=np.array(SCALE), globally_registered_left_position=base_l, globally_registered_right_position=base_r, globally_registered_left_rotation=base_lr, globally_registered_right_rotation=base_rr, v8_exact_warm_start_only=warm_q, arm_joint_names=arm_joint_names, method=np.array(METHOD), real_robot_command_allowed=np.array(False))
    save_npz(OUT / "phase_residual_components.npz", knot_action_indices=KNOTS, left_translation_knots=kl, right_translation_knots=kr, left_rotation_vector_knots=rkl, right_rotation_vector_knots=rkr, common_translation_residual=common_translation, left_specific_translation_residual=left_specific, right_specific_translation_residual=right_specific, common_rotation_vector_residual=common_rotation, left_rotation_vector_residual=rotation_res_l, right_rotation_vector_residual=rotation_res_r, total_left_translation_residual=residual_l, total_right_translation_residual=residual_r, source_progress=source_progress, selected_candidate=np.array(selected))
    save_npz(OUT / "corrected_aloha_targets.npz", optimized_action=action, timestamps=timestamps, action_indices=np.arange(990), observed_frame_for_action=np.where(np.arange(990) <= 982, np.arange(990) + LAG, -1), base_left_position=base_l, base_right_position=base_r, base_left_rotation=base_lr, base_right_rotation=base_rr, global_registration_rotation=global_rotation, global_registration_translation=global_translation, common_translation_residual=common_translation, left_specific_translation_residual=left_specific, right_specific_translation_residual=right_specific, left_translation_residual=residual_l, right_translation_residual=residual_r, left_rotation_vector_residual=rotation_res_l, right_rotation_vector_residual=rotation_res_r, corrected_left_position=corrected_l, corrected_right_position=corrected_r, corrected_left_rotation=corrected_lr, corrected_right_rotation=corrected_rr, task_axis_left_rotation=task_axis_lr, task_axis_right_rotation=task_axis_rr, residual_knots=KNOTS, selected_candidate=np.array(selected), method=np.array(METHOD), diagnostic_only=np.array(True), real_robot_command_allowed=np.array(False))
    save_npz(OUT / "target_phone_pose_trajectory.npz", action_index=np.arange(990), pose=phone_trajectory, position=phone_trajectory[:, :3, 3], rotation=phone_trajectory[:, :3, :3], alpha_left_acquisition=np.r_[np.zeros(169), alpha_l, np.ones(990 - 217)], phone_initial_pose=phone_initial, phone_portrait_pose=phone_portrait, phone_on_charger_pose=desired_phone, palm_to_phone_lock=lock_palm_to_phone, model_scope=np.array("TARGET_SIDE_EMBODIMENT_SPECIFIC_DIAGNOSTIC_OBJECT_STATE"), physics=np.array(False))
    save_npz(OUT / "target_accessory_pose_trajectory.npz", action_index=np.arange(990), pose=accessory_trajectory, position=accessory_trajectory[:, :3, 3], rotation=accessory_trajectory[:, :3, :3], attached_to_phone_pose=accessory_attached, alpha_right_acquisition=np.r_[np.zeros(319), alpha_r, np.ones(990 - 335)], phone_to_accessory_attachment=phone_to_accessory, palm_to_accessory_lock=palm_to_accessory, model_scope=np.array("TARGET_SIDE_EMBODIMENT_SPECIFIC_DIAGNOSTIC_OBJECT_STATE"), physics=np.array(False))

    global_standard_translation = G1_ROOT - R_SCENE_FROM_MODEL @ MODEL_ROOT
    dump(OUT / "global_task_registration.json", {
        "method": "one common fixed similarity transform after validated scale in base target",
        "workspace_scale": SCALE,
        "scale_automatically_changed": False,
        "rotation_scene_from_g1_model": global_rotation,
        "selected_additional_task_rpy_deg": [float(selected_global_roll_deg), float(selected_global_pitch_deg), float(selected_global_yaw_deg)],
        "global_rotation_coarse_search_rpy_deg": {"roll": [-25.0, 25.0, 5.0], "pitch": [-25.0, 25.0, 5.0], "yaw": [-75.0, 75.0, 2.5]},
        "global_rotation_refinement_step_deg": 1.0,
        "global_rotation_selection": "minimum common-registration endpoint correction with active-arm reach penalty",
        "diagnostic_yaw_override_deg": args.global_yaw_override_deg,
        "global_rotation_candidate_results": global_registration_rows,
        "translation_scene_m": global_translation,
        "primary_landmark": {"event": "left_phone_grasp_start", "observed_frame": 176, "action_index": 169, "target_palm_anchor_m": phone_palm_anchor, "authoritative_phone_left_surface_m": phone_left_surface, "active_left_shoulder_m": left_shoulder_scene, "future_A_B_static_proxy_midpoint_reach_m": future_left_ab_reach_m, "future_A_B_static_proxy_midpoint_local_m": future_left_ab_midpoint_local, "alignment_error_m": float(np.linalg.norm(base_l[169] - phone_palm_anchor))},
        "same_transform_both_hands_all_samples": True,
        "common_offset_removed_vs_standard_v8_scene_registration_m": global_translation - global_standard_translation,
        "fixed_global_registration_excluded_from_deformation_metrics": True,
    })
    dump(OUT / "target_phone_carrier_model.json", {
        "status": "TARGET_SIDE_EMBODIMENT_SPECIFIC_TASK_REGISTRATION_MODEL",
        "not_source_ground_truth": True,
        "source_single_carrier_assumed": False,
        "initial_fixed_action_range": [0, 168],
        "acquisition_action_range": [169, 216],
        "rigid_target_carrier_action_range": [216, 522],
        "charger_fixed_action_range": [523, 989],
        "acquisition_candidates": {k: v.tolist() for k, v in left_curves.items()},
        "candidate_scores": left_curve_scores,
        "selected_acquisition_curve": left_curve_name,
        "mapped_source_rotation_action_169_to_216": mapped_phone_rotation,
        "task_axis_correction": portrait_axis_correction,
        "task_axis_correction_angle_deg": float(np.degrees(Rotation.from_matrix(portrait_axis_correction).magnitude())),
        "portrait_long_axis_sign": portrait_sign,
        "portrait_back_accessibility_dot_plus_y": portrait_accessibility,
        "target_phone_from_left_palm_lock": inv_transform(lock_palm_to_phone),
        "target_left_palm_to_phone_lock": lock_palm_to_phone,
        "carrier_phone_center_offset_m": carrier_center_offset_m,
        "carrier_phone_center_max_from_asset_and_neutral_proxy_m": phone_carrier_center_max_m,
        "active_arm_chain_length_m": active_arm_chain_length_m,
        "active_arm_reach_gate_m": active_arm_reach_gate_m,
        "charger_palm_anchor_m": palm523_position,
        "active_left_shoulder_m": left_shoulder_scene,
        "future_left_A_static_neutral_proxy_local_from_palm_m": future_left_a_proxy_local,
        "future_left_B_static_neutral_proxy_local_from_palm_m": future_left_b_proxy_local,
        "future_left_A_B_static_midpoint_local_from_palm_m": future_left_ab_midpoint_local,
        "dex3_joint_motion_generated": False,
    })
    dump(OUT / "target_accessory_carrier_model.json", {
        "status": "TARGET_SIDE_EMBODIMENT_SPECIFIC_TASK_REGISTRATION_MODEL",
        "not_source_ground_truth": True,
        "phone_to_accessory_attachment_translation_m": [0.0, 0.006425, 0.0],
        "attached_to_dynamic_target_phone_before_action": 319,
        "acquisition_action_range": [319, 334],
        "rigid_target_carrier_action_range": [334, 638],
        "release_fixed_action_range": [639, 989],
        "acquisition_candidates": {k: v.tolist() for k, v in right_curves.items()},
        "candidate_scores": right_curve_scores,
        "selected_acquisition_curve": right_curve_name,
        "target_right_palm_to_accessory_lock": palm_to_accessory,
        "future_right_C_static_neutral_proxy_local_from_palm_m": future_right_c_proxy_local,
        "future_right_C_static_neutral_reach_m": future_right_c_reach_m,
        "dex3_joint_motion_generated": False,
        "coupled_anchor_iterations": coupled_iterations,
        "coupled_converged": coupled_converged,
    })
    dump(OUT / "target_left_phase_anchors.json", {
        "anchor_frame_domain": "optimized_action_index",
        "anchors": {
            "phone_grasp": {"observed_event_frame": 176, "aligned_action_index": 169, "position_m": phone_palm_anchor, "future_left_A_B_proxy_midpoint_target_m": phone_left_surface, "future_left_A_B_static_midpoint_local_m": future_left_ab_midpoint_local, "basis": "active-model palm carrier point one neutral A/B reach-envelope length from the authoritative phone -X surface toward the active left shoulder; ALOHA approach tangent retained by the globally registered base path; no Dex3 fitting"},
            "portrait": {"observed_event_frame": 223, "aligned_action_index": 216, "position_m": left_portrait_contact, "phone_pose": phone_portrait, "basis": "mapped ALOHA rotation plus minimum vertical-long-axis correction"},
            "charger": {"observed_event_frame": 530, "aligned_action_index": 523, "position_m": palm523[:3, 3], "phone_pose": desired_phone, "basis": "pad face and target-side phone/palm lock; wrist not placed at charger root"},
        },
    })
    dump(OUT / "target_right_phase_anchors.json", {
        "anchor_frame_domain": "optimized_action_index",
        "anchors": {
            "accessory_grasp": {"observed_event_frame": 326, "aligned_action_index": 319, "position_m": right_anchor319, "future_right_C_proxy_target_m": ring_surface319, "future_right_C_static_neutral_proxy_local_m": future_right_c_proxy_local, "basis": "nearest ALOHA-preserving palm pose whose static neutral future-C reach envelope meets the dynamic target accessory annulus; no Dex3 fitting"},
            "accessory_removed": {"observed_event_frame": 341, "aligned_action_index": 334, "position_m": right_anchor334, "basis": "action-319 target anchor plus immutable scaled ALOHA phase-relative removal displacement"},
        },
    })
    dump(OUT / "phase_residual_candidate_grid.json", {"method": METHOD, "knots_action_indices": KNOTS, "same_anchors_all_candidates": True, "source_progress_parameterization": True, "C2_continuity": True, "weight_profiles": weights_grid})
    dump(OUT / "phase_residual_candidate_results.json", results)
    dump(OUT / "selected_phase_residual.json", {"selected": selected, "selection_rule": "anchor-valid; prefer candidates meeting all 0.90 fidelity thresholds; lexicographically maximize path, speed, rotation fidelity, then minimize energy", "fidelity_threshold_candidate_pool": fidelity_valid_names, "result": results[selected], "coupled_target_anchor_converged": coupled_converged})
    fidelity_status = "PASS" if results[selected]["fidelity_valid"] else "BLOCKED_ALOHA_FIDELITY"
    dump(OUT / "aloha_fidelity_metrics.json", {"status": fidelity_status, "selected_candidate": selected, "thresholds": {"path_shape": 0.90, "speed": 0.90, "rotation_progress": 0.90}, "minimum_major_phase_fidelity": results[selected]["minimum_major_phase_fidelity"], "per_phase": results[selected]["phases"], "bimanual": results[selected]["bimanual"], "global_similarity_registration_excluded": True, "hard_invariants": {"optimized_action_samples_exact": 990, "optimized_action_array_equal": True, "timestamps_array_equal": True, "observed_event_frames_exact": True, "aligned_action_indices_exact": True, "phase_durations_exact": True, "left_right_roles_exact": True}})
    dump(OUT / "residual_energy_metrics.json", {"selected_candidate": selected, **results[selected]["residual_energy"], "per_phase_residual_displacement": {k: v["residual_displacement_m"] for k, v in results[selected]["phases"].items()}})
    dump(OUT / "anchor_metrics.json", {"stage": "constructed_cartesian_targets_before_IK", "gate_m": 0.005, "errors": results[selected]["anchor_errors"], "all_constructed_anchors_exact": max(results[selected]["anchor_errors"].values()) <= 1e-8})

    dump(OUT / "method_contract.json", {
        "method": METHOD,
        "exact_source_object_state_transfer_claimed": False,
        "source_object_world_pose_required": False,
        "source_motion_used": ["phase-relative translation and path", "relative rotation progress", "speed timing", "approach/removal progression", "bimanual timing and roles"],
        "stored_components": ["immutable ALOHA phase motion", "global task registration", "target-side object anchors", "phase-conditioned residual"],
        "no_hand_written_waypoints": True, "no_g1_expert": True, "no_dex3": True, "no_physics": True, "no_hardware": True,
    })
    dump(OUT / "rejected_branch_audit.json", {"rejected_paths": REJECTED, "diagnostic_source_object_audits": DIAGNOSTIC_ONLY, "loaded_as_target_or_seed": [], "v8_loaded_fields": ["base_target_left_position", "base_target_right_position", "g1_arm_joint_trajectory warm start", "arm_joint_names", "optimized_action/timestamp invariant checks"], "source_carrier_v11b_loaded": False, "accessory_audit_v11c_loaded": False})
    dump(OUT / "timeline_alignment_audit.json", {"approved_timeline": str(TIMELINE.resolve()), "timeline_sha256_before": timeline_hash_before, "alignment_config": str(ALIGNMENT.resolve()), "alignment_sha256": alignment_hash, "fps": FPS, "action_to_observation_lag_frames": LAG, "latency_seconds": LAG / FPS, "mapping_rule": "observed_frame - 7", "events": event_mapping, "action_samples_0_989_retained": True, "terminal_samples_983_989": "POST-OBSERVATION TERMINAL COMMAND SAMPLE", "timeline_rewritten": False})
    dump(OUT / "input_audit.json", {"status": "PASS", "sole_primary_motion_source": str(SRC.resolve()), "source_sha256": source_hash, "source_key": "optimized_action", "shape": [990, 14], "fps": FPS, "finite": True, "optimized_action_unchanged": True, "timestamps_unchanged": True, "verified_fk_crosscheck_max_abs_m": fk_crosscheck, "aloha_mapping_clipped_frames": int(clipped), "workspace_scale": SCALE, "no_source_object_pose_loaded": True, "no_dex3": True, "no_physics": True, "no_dds_publisher_hardware": True})
    dump(OUT / "environment_audit.json", {"authoritative_stage": str(ACTIVE_STAGE.resolve()), "active_transforms": {"G1": active_g1, "phone": phone_initial, "accessory": accessory_initial, "charger_root": charger_root, "charger_pad_face_asset": pad_face_asset, "desired_phone_on_pad": desired_phone}, "g1_root_expected": G1_ROOT, "g1_root_error_m": float(np.linalg.norm(active_g1[:3, 3] - G1_ROOT)), "root_forward_offset_m": 0.15, "phone_y_m": float(phone_initial[1, 3]), "charger_y_m": float(charger_root[1, 3]), "scene_hashes_before": scene_hash_before})

    if not args.skip_plots:
        create_plots(OUT, base_l, base_r, corrected_l, corrected_r, base_lr, corrected_lr)

    scene_hash_after = {str(p.resolve()): sha(p) for p in (LAYOUT, SCENE, ACTIVE_STAGE, REGISTRATION)}
    timeline_hash_after = sha(TIMELINE)
    environment = json.loads((OUT / "environment_audit.json").read_text())
    environment["scene_hashes_after"] = scene_hash_after
    environment["target_scene_byte_identical_before_after"] = scene_hash_before == scene_hash_after
    environment["target_layout_unchanged"] = bool(scene_hash_before == scene_hash_after and abs(phone_initial[1, 3] - 0.07) < 1e-12 and abs(charger_root[1, 3] - 0.21) < 1e-12 and np.linalg.norm(active_g1[:3, 3] - G1_ROOT) < 1e-12)
    dump(OUT / "environment_audit.json", environment)
    timeline_audit = json.loads((OUT / "timeline_alignment_audit.json").read_text())
    timeline_audit["timeline_sha256_after"] = timeline_hash_after
    timeline_audit["timeline_byte_identical_before_after"] = timeline_hash_before == timeline_hash_after
    dump(OUT / "timeline_alignment_audit.json", timeline_audit)
    status = "TARGETS_READY_FOR_POSITION_ONLY_IK" if environment["target_layout_unchanged"] and coupled_converged else "BLOCKED_TARGET_ACCESSORY_MODEL"
    dump(OUT / "offline_build_status.json", {"status": status, "fidelity_status": fidelity_status, "selected_candidate": selected, "coupled_target_anchor_converged": coupled_converged, "target_layout_unchanged": environment["target_layout_unchanged"]})
    print(json.dumps(json.loads((OUT / "offline_build_status.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
